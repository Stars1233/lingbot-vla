from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
import utils3d
from huggingface_hub import hf_hub_download

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, angle_diff_vec3
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
# from .modules_dinov2 import DINOv2Encoder, MLP, ConvStack
from .modules_dinov2_rgbd import DINOv2Encoder, MLP, ConvStack

    
class MoRGBDModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        depth_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        remap_depth_in: Literal['linear', 'log'] = 'log',
        num_tokens_range: List[int] = [1200, 3600],
        **deprecated_kwargs
    ):
        super(MoRGBDModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        self.remap_depth_in = remap_depth_in
        
        self.encoder = DINOv2Encoder(**encoder) 
        self.neck = ConvStack(**neck)
        if depth_head is not None:
            self.depth_head = ConvStack(**depth_head) 
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, oss_manager=None, **hf_kwargs) -> 'MoRGBDModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if pretrained_model_name_or_path.startswith('oss://'):
            assert oss_manager is not None, "oss_manager must be provided if pretrained_model_name_or_path starts with 'oss://'"
            with oss_manager.open(pretrained_model_name_or_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=True)
        else:
            if Path(pretrained_model_name_or_path).exists():
                checkpoint_path = pretrained_model_name_or_path
            else:
                checkpoint_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    repo_type="model",
                    filename="model.pt",
                    **hf_kwargs
                )
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'], strict=False)
        
        return model
    
    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()
    
    def forward(self, image: torch.Tensor, num_tokens: Union[int, torch.LongTensor], depth: Union[None, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype
        
        # Debug mode
        # depth = torch.ones(batch_size, 1, img_h, img_w).to(device=device, dtype=dtype)
        assert depth is not None  # in this version, depth is required
        if depth.dim() == 3:
            depth = depth.unsqueeze(1) # from (B, H, W) to (B, 1, H, W)

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token, _, _ = self.encoder(image, depth, base_h, base_w, return_class_token=True, remap_depth_in=self.remap_depth_in)

        features = features + cls_token[..., None, None]
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        depth_reg, normal, mask = (getattr(self, head)(features)[-1] if hasattr(self, head) else None for head in ['depth_head', 'normal_head', 'mask_head'])
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None
        
        # Resize
        depth_reg, normal, mask = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [depth_reg, normal, mask])
        
        # Remap output
        if depth_reg is not None:
            depth_reg = depth_reg.exp().squeeze(1)
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask_prob = mask.squeeze(1).sigmoid()
            # mask_logits = mask.squeeze(1)
        else:
            mask_prob = None
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'depth_reg': depth_reg,
            'normal': normal,
            'mask': mask_prob,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    def forward_feat(self, image: torch.Tensor, num_tokens: Union[int, torch.LongTensor], depth: Union[None, torch.Tensor]=None):
        batch_size, _, img_h, img_w = image.shape        
        assert depth is not None  # in this version, depth is required
        if depth.dim() == 3:
            depth = depth.unsqueeze(1) # from (B, H, W) to (B, 1, H, W)

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)
    
        # Backbones encoding
        features, cls_token, _, _ = self.encoder(image, depth, base_h, base_w, return_class_token=True, remap_depth_in=self.remap_depth_in)

        return features, cls_token
    
    def forward_depth_from_feat(self, features: torch.Tensor, cls_token: torch.Tensor, num_tokens: Union[int, torch.LongTensor], img_h: int, img_w: int):
        batch_size = features.shape[0]   
        device, dtype = features.device, features.dtype 
        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)
        
        features = features + cls_token[..., None, None]
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        depth_reg, normal, mask = (getattr(self, head)(features)[-1] if hasattr(self, head) else None for head in ['depth_head', 'normal_head', 'mask_head'])
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None
        
        # Resize
        depth_reg, normal, mask = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [depth_reg, normal, mask])
        
        # Remap output
        if depth_reg is not None:
            depth_reg = depth_reg.exp().squeeze(1)
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask_prob = mask.squeeze(1).sigmoid()
            # mask_logits = mask.squeeze(1)
        else:
            mask_prob = None
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'depth_reg': depth_reg,
            'normal': normal,
            'mask': mask_prob,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict
    
    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        depth_in: torch.Tensor = None,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        depth_down_scale: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500. 
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`. 
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        if (depth_in is not None) and (depth_in.dim() == 2):
            depth_in = depth_in.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height
        
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_fp16 and self.dtype != torch.bfloat16):
            output = self.forward(image, num_tokens=num_tokens, depth=depth_in)
        depth_reg, normal, mask, metric_scale = (output.get(k, None) for k in ['depth_reg', 'normal', 'mask', 'metric_scale'])

        # Always process the output in fp32 precision
        depth_reg, normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [depth_reg, normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None
                
            depth = depth_reg
            # points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
            if intrinsics is not None:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
            else:
                points = None

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'depth_reg': depth,
            'mask': mask_binary,
            'normal': normal
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
    
    @torch.inference_mode()
    def infer_feat(
        self, 
        image: torch.Tensor, 
        depth_in: torch.Tensor = None,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        depth_down_scale: int = 1,
    ):
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500. 
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`. 
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        if (depth_in is not None) and (depth_in.dim() == 2):
            depth_in = depth_in.unsqueeze(0).to(dtype=self.dtype, device=self.device)
        
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_fp16 and self.dtype != torch.bfloat16):
            features, cls_token = self.forward_feat(image, num_tokens=num_tokens, depth=depth_in)
  
        return features, cls_token
    
    @torch.inference_mode()
    def dec_depth(
        self, 
        features: torch.Tensor, 
        cls_token: torch.Tensor,
        num_tokens: int = None,
        img_h: int = None,
        img_w: int = None,
        resolution_level: int = 9,
        use_fp16: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        intrinsics: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        features = features.to(dtype=self.dtype, device=self.device)
        if features.dim() == 3:
            features = features.unsqueeze(0)
            omit_batch_dim = True
        else:
            omit_batch_dim = False
        
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_fp16 and self.dtype != torch.bfloat16):
            output = self.forward_depth_from_feat(features, cls_token, num_tokens, img_h, img_w)
        depth_reg, normal, mask, metric_scale = (output.get(k, None) for k in ['depth_reg', 'normal', 'mask', 'metric_scale'])

        # Always process the output in fp32 precision
        depth_reg, normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [depth_reg, normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None
                
            depth = depth_reg
            # points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
            if intrinsics is not None:
                points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
            else:
                points = None

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'depth_reg': depth,
            'mask': mask_binary,
            'normal': normal
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict

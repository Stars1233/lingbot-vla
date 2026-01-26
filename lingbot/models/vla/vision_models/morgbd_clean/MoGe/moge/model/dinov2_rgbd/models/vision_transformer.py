# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional, List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from ..layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from ..layers import PatchEmbedMLP

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        in_depth_mask_on=False,
        depth_mask_emb_mode='conv',
        embed_layer_depth=PatchEmbedMLP,
        mask_ratio=0.6,
        img_mask_ratio=0.0
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.in_depth_mask_on = in_depth_mask_on
        self.depth_mask_emb_mode = depth_mask_emb_mode
        self.mask_ratio = mask_ratio
        self.img_mask_ratio = img_mask_ratio
        if self.in_depth_mask_on:  # xxx-xxxxx-xxxx example conv_depth_mask_2c-add-fuse
            # self.depth_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            if 'conv_depth_mask_2c' in self.depth_mask_emb_mode:
                self.depth_mask_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim)
            elif 'mlp_depth_mask_2c' in self.depth_mask_emb_mode:
                self.depth_mask_patch_embed = embed_layer_depth(img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim)
            elif 'conv_depth_1c' in self.depth_mask_emb_mode:
                self.depth_mask_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
            else:
                raise

            if 'fuse' in self.depth_mask_emb_mode:
                if 'catFeat' in self.depth_mask_emb_mode:
                    self.fuse = Mlp(in_features=embed_dim*2, hidden_features=1024, out_features=embed_dim)
                else:
                    self.fuse = Mlp(in_features=embed_dim, hidden_features=1024, out_features=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        batch_size = x.shape[0]
        N = self.pos_embed.shape[1] - 1
        if not self.onnx_compatible_mode and npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if not self.onnx_compatible_mode and self.interpolate_offset > 0:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (h0, w0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos_embed[:, None, :].expand(patch_pos_embed.shape[0], -1, -1), patch_pos_embed), dim=1).to(previous_dtype)

    def interpolate_pos_encoding_without_cls(self, x, h, w, input_pos_embed):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        batch_size = x.shape[0]
        N = input_pos_embed.shape[1]
        if not self.onnx_compatible_mode and npatch == N and w == h:
            return input_pos_embed
        patch_pos_embed = input_pos_embed.float()
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if not self.onnx_compatible_mode and self.interpolate_offset > 0:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (h0, w0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return patch_pos_embed.to(previous_dtype)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, D, H, W)
        x: (N, L, patch_size**2 *D)
        """
        dim = imgs.shape[1]
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], dim, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * dim))
        return x
    
    def unpatchify(self, x, h_in, w_in, dim):
        """
        x: (N, L, patch_size**2 *D)
        imgs: (N, D, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1]**.5)
        # assert h * w == x.shape[1]

        h = h_in // p
        w = w_in // p
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], dim, h * p, w * p))
        return imgs

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, h, w = x.shape
        mae_mask, mae_ids_restore = None, None

        if self.in_depth_mask_on:
            assert nc == 5
            img = x[:, :3]
            depth_mask = x[:, 3:]
            depth = x[:, 3:4]
            x_img = self.patch_embed(img)
            
            if 'dymr' in self.depth_mask_emb_mode:
                depth_14 = self.patchify(depth)
                depth_14_count = (depth_14 > 1e-10).sum(dim=-1)
                depth_14_valid_mask = depth_14_count > depth_14.shape[-1] * 0.9
                depth_14 = depth_14 * (depth_14_valid_mask[..., None]).float()
                if self.training:
                    rand_thres = float(torch.randint(2, 6, (1,)).item()) / 10.
                    depth_14_valid_mask2 = (torch.rand_like(depth_14[..., 0]) > rand_thres)
                    depth_14 = depth_14 * (depth_14_valid_mask2[..., None]).float()
                depth = self.unpatchify(depth_14, h_in=h, w_in=w, dim=1)

            # import pdb; pdb.set_trace()

            if '2c' in self.depth_mask_emb_mode:
                x_depth_mask = self.depth_mask_patch_embed(depth_mask)
            elif '1c' in self.depth_mask_emb_mode:
                # x_depth_mask = self.depth_mask_patch_embed(depth_mask[:, 0:1])
                x_depth_mask = self.depth_mask_patch_embed(depth)
            else:
                raise NotImplementedError
            
            if 'catFeat' in self.depth_mask_emb_mode:
                x = torch.cat([x_img, x_depth_mask], dim=-1)
                if 'fuse' in self.depth_mask_emb_mode:
                    x = self.fuse(x)
                if masks is not None:
                    raise NotImplementedError
                    # x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + self.interpolate_pos_encoding(x, h, w)
            elif 'catToken' in self.depth_mask_emb_mode or 'cat' in self.depth_mask_emb_mode:
                assert masks is None
                N, L, _ = x_depth_mask.shape  # batch, length, dim
                assert N == B
                ## process depth tokens
                depth_pose_enc = self.interpolate_pos_encoding_without_cls(x_depth_mask, h, w, self.pos_embed[:, 1:]).repeat(N, 1, 1)
                assert depth_pose_enc.shape[0] == N
                mask_ratio = self.mask_ratio
                if mask_ratio == 0:
                    x_depth_mask_masked = x_depth_mask
                    depth_pose_enc_masked = depth_pose_enc
                else:
                    if N == 1 and not self.training:
                        x_depth_mask_masked = x_depth_mask
                        depth_pose_enc_masked = depth_pose_enc
                    else:
                        if 'dymr' in self.depth_mask_emb_mode:  # dynamic mask ratio
                            mask_ratio = float(torch.randint(1, 11, (1,)).item()) * 0.1 * mask_ratio
                            max_mask_ratio = self.mask_ratio
                            mask_ratio = float(torch.randint(30, int(max_mask_ratio*100), (1,)).item()) / 100.0
                            # pass
                        len_keep = int(L * (1 - mask_ratio))
                        noise = torch.rand(N, L, device=x_img.device)  # noise in [0, 1]
                        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                        ids_keep = ids_shuffle[:, :len_keep]

                        x_depth_mask_masked = torch.gather(x_depth_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_depth_mask.shape[-1]))
                        depth_pose_enc_masked = torch.gather(depth_pose_enc, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, depth_pose_enc.shape[-1]))

                        # depth_14_masked = torch.gather(depth_14, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, depth_14.shape[-1]))
                        # import pdb; pdb.set_trace()
                        
                ## process image tokens
                img_pose_enc = self.interpolate_pos_encoding_without_cls(x_img, h, w, self.pos_embed[:, 1:]).repeat(N, 1, 1)
                assert img_pose_enc.shape[0] == N
                img_mask_ratio = self.img_mask_ratio
                if img_mask_ratio == 0:
                    x_img_masked = x_img
                    img_pose_enc_masked = img_pose_enc
                else:
                    if N == 1 and not self.training:
                        x_img_masked = x_img
                        img_pose_enc_masked = img_pose_enc
                    else:
                        img_len_keep = int(L * (1 - img_mask_ratio))
                        noise = torch.rand(N, L, device=x_img.device)  # noise in [0, 1]
                        img_ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                        img_ids_restore = torch.argsort(img_ids_shuffle, dim=1)
                        img_ids_keep = img_ids_shuffle[:, :img_len_keep]    

                        x_img_masked = torch.gather(x_img, dim=1, index=img_ids_keep.unsqueeze(-1).repeat(1, 1, x_img.shape[-1]))
                        img_pose_enc_masked = torch.gather(img_pose_enc, dim=1, index=img_ids_keep.unsqueeze(-1).repeat(1, 1, img_pose_enc.shape[-1]))
                        
                        # generate the binary mask: 0 is keep, 1 is remove
                        img_tk_mask = torch.ones([N, L], device=x.device)
                        img_tk_mask[:, :img_len_keep] = 0
                        # unshuffle to get the binary mask
                        img_tk_mask = torch.gather(img_tk_mask, dim=1, index=img_ids_restore)

                        mae_mask, mae_ids_restore = img_tk_mask, img_ids_restore

                # cat img and depth tokens
                x = torch.cat([x_img_masked, x_depth_mask_masked], dim=1)
                if 'fuse' in self.depth_mask_emb_mode:
                    x = self.fuse(x)
                # cat cls token
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

                # get cls pose enc
                cls_pos_enc = self.pos_embed[:, :1].repeat(B, 1, 1)

                # cat all pose enc
                cls_img_depth_pos_enc = torch.cat([cls_pos_enc, img_pose_enc_masked, depth_pose_enc_masked], dim=1)

                # get type pose enc
                num_img_patches = x_img_masked.shape[1]
                type_enc = torch.ones(x.shape[0], x.shape[1], 1).to(device=x.device, dtype=x.dtype)
                type_enc[:, 0] = 0 # cls token
                type_enc[:,1+num_img_patches:] = 2 # depth token
                x = x + cls_img_depth_pos_enc + type_enc
            else:  # add
                x = x_img + x_depth_mask
                if 'fuse' in self.depth_mask_emb_mode:
                    x = self.fuse(x)
                if masks is not None:
                    x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = self.patch_embed(x)
            if masks is not None:
                x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, h, w)

        # import pdb; pdb.set_trace()

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x, mae_mask, mae_ids_restore
    
    def prepare_tokens_with_masks_downsample_infer(self, x, depth, masks=None, depth_sample_step=4):
        B, nc, h, w = x.shape
        mae_mask, mae_ids_restore = None, None
        _, _, hd, wd = depth.shape

        assert nc == 3
        img = x[:, :3]
        # depth = x[:, 3:4]

        # import pdb; pdb.set_trace()
        x_img = self.patch_embed(img)
        x_depth_mask = self.depth_mask_patch_embed(depth)

        assert masks is None
        N, L, _ = x_depth_mask.shape  # batch, length, dim
        assert N == B
        ## process depth tokens
        depth_pose_enc = self.interpolate_pos_encoding_without_cls(x_depth_mask, hd, wd, self.pos_embed[:, 1:]).repeat(N, 1, 1)
        assert depth_pose_enc.shape[0] == N
        x_depth_mask_masked = x_depth_mask[:, ::depth_sample_step]
        depth_pose_enc_masked = depth_pose_enc[:, ::depth_sample_step]
        # import pdb; pdb.set_trace()
 
        ## process image tokens
        img_pose_enc = self.interpolate_pos_encoding_without_cls(x_img, h, w, self.pos_embed[:, 1:]).repeat(N, 1, 1)
        assert img_pose_enc.shape[0] == N
        x_img_masked = x_img
        img_pose_enc_masked = img_pose_enc
        
        # cat img and depth tokens
        x = torch.cat([x_img_masked, x_depth_mask_masked], dim=1)
        if 'fuse' in self.depth_mask_emb_mode:
            x = self.fuse(x)
        # cat cls token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # get cls pose enc
        cls_pos_enc = self.pos_embed[:, :1].repeat(B, 1, 1)

        # cat all pose enc
        cls_img_depth_pos_enc = torch.cat([cls_pos_enc, img_pose_enc_masked, depth_pose_enc_masked], dim=1)

        # get type pose enc
        num_img_patches = x_img_masked.shape[1]
        type_enc = torch.ones(x.shape[0], x.shape[1], 1).to(device=x.device, dtype=x.dtype)
        type_enc[:, 0] = 0 # cls token
        type_enc[:,1+num_img_patches:] = 2 # depth token
        x = x + cls_img_depth_pos_enc + type_enc

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x, mae_mask, mae_ids_restore
    
    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks, ar in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1, return_mae_aux=False, depth=None):
        if depth is None:
            x, mae_mask, mae_ids_restore = self.prepare_tokens_with_masks(x)
        else:
            x, mae_mask, mae_ids_restore = self.prepare_tokens_with_masks_downsample_infer(x, depth)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        if return_mae_aux:
            return output, mae_mask, mae_ids_restore
        else:
            return output

    def _get_intermediate_layers_chunked(self, x, n=1, return_mae_aux=False, depth=None):
        if depth is None:
            x, mae_mask, mae_ids_restore = self.prepare_tokens_with_masks(x)
        else:
            x, mae_mask, mae_ids_restore = self.prepare_tokens_with_masks_downsample_infer(x, depth)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        if return_mae_aux:
            return output, mae_mask, mae_ids_restore
        else:
            return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        return_mae_aux: bool = False,
        depth=None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n, depth=depth)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n, depth=depth)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
    
    def get_intermediate_layers_mae(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        depth=None,
    ):
        if self.chunked_blocks:
            outputs, mae_mask, mae_ids_restore = self._get_intermediate_layers_chunked(x, n, return_mae_aux=True, depth=depth)
        else:
            outputs, mae_mask, mae_ids_restore = self._get_intermediate_layers_not_chunked(x, n, return_mae_aux=True, depth=depth)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens)), mae_mask, mae_ids_restore
        return tuple(outputs), mae_mask, mae_ids_restore

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

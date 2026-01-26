import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib
import einops
from PIL import Image, ImageDraw

from lingbot.models.vla.vision_models.depth_anything_v2 import DepthAnythingV2_Backbone, DepthAnythingV2_Head

try:
    from morgbd.model.v2 import MoRGBDModel as v2_morgbd
    from moge.model.v2 import MoGeModel as v2
    from moge.utils.vis import colorize_depth
except:
    print('Load MoGe Module Failed!!')

def make_grid(images, pil_images):
    # Assuming each image is the same size
    
    new_images = []
    new_captions = []
    for image, pil_image in zip(images, pil_images):
        new_images.append(image)
        pil_image = pil_image.resize((image.size[0], image.size[1]))
        new_images.append(pil_image)
        new_captions.append("Predicted")
        new_captions.append("GT")
    
    images = new_images
    captions = new_captions

    width, height = images[0].size
    font_size = 14
    caption_height = font_size + 10

    # Calculate the size of the final image
    images_per_row = min(len(images), 16)  # Round up for odd number of images
    row_count = (len(images) + 1) // images_per_row
    total_width = width * images_per_row
    total_height = (height + caption_height) * row_count

    # Create a new blank image
    new_image = Image.new("RGB", (total_width, total_height), "white")

    draw = ImageDraw.Draw(new_image)

    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * width
        y_offset = row * (height + caption_height)
        
        new_image.paste(image, (x_offset, y_offset))
        text_position = (x_offset + 10, y_offset + height)
        draw.text(text_position, caption, fill="red", font_size=font_size)
    
    return new_image

def build_depth_model(config):

    model_type = config['depth']['model_type']

    if model_type == 'DepthAnythingV2':
        dav2_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        dav2_backbone = DepthAnythingV2_Backbone(**dav2_cfg)
        dav2_backbone.load_state_dict(torch.load(config['depth']['expert_path'], map_location='cpu'))
        for p in dav2_backbone.parameters():
            p.requires_grad = False
        dav2_backbone.cuda().to(dtype=torch.bfloat16)
        dav2_backbone.eval()

        dav2_head = DepthAnythingV2_Head()
        dav2_head.load_state_dict(torch.load(config['depth']['expert_path']), strict=False)
        for p in dav2_head.parameters():
            p.requires_grad = False
        dav2_head.cuda().to(dtype=torch.bfloat16)
        dav2_head.eval()
        
        return dav2_backbone, dav2_head
    elif model_type == 'MoRGBD':
        moge_model = v2.from_pretrained(config['depth']['moge_path'])
        for p in moge_model.parameters():
            p.requires_grad = False
        moge_model.cuda()
        moge_model.eval()

        morgbd_model = v2_morgbd.from_pretrained(config['depth']['morgbd_path'])
        for p in morgbd_model.parameters():
            p.requires_grad = False
        morgbd_model.cuda()
        morgbd_model.eval()
        return moge_model, morgbd_model

def get_depth_target(model_type, depth_model, pil_images):
    device = pil_images.device
    B, _, C, H, W = pil_images.shape
    images = einops.rearrange(pil_images, 'b n c h w -> (b n) c h w', n=3).contiguous().float()

    if model_type == 'DepthAnythingV2':
        dav2_backbone = depth_model
        feat = dav2_backbone.infer_image(images, device=device)
        depth_target = (feat[0][0] + feat[1][0] + feat[2][0] + feat[3][0]) / 4
        cls_token = None
    
    elif model_type == 'MoRGBD':
        input_images = images / 255.0
        moge_model, morgbd_model = depth_model
        output_moge = moge_model.infer(input_images, resolution_level=3, num_tokens=256)
        depth_pred = output_moge['depth'].squeeze().detach().clone() # moge2
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
        depth_pred *= 1
        depth_down_scale = 1
        depth_target, cls_token = morgbd_model.infer_feat(input_images, depth_pred, 
                                                depth_down_scale=depth_down_scale,
                                                resolution_level=3,
                                                num_tokens=256)
        depth_target = depth_target.permute(0, 2, 3, 1)
        depth_target = depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])

    return depth_target.to(dtype=torch.bfloat16), cls_token

def log_depth(vis_head, depth_pred_feats, depth_target_feats=None, steps=0, config=None, cls_token=None):
    model_type = config['depth']['model_type']
    llm_image_token_size = config['llm']['image_token_size']
    depth_token_size = config['depth']['token_size']
    visual_dir = config['visual_dir']

    if config['mode'] == "direct":
        depth_pred_feats = depth_pred_feats.view(depth_pred_feats.shape[0], llm_image_token_size, llm_image_token_size, depth_pred_feats.shape[-1])
        depth_pred_feats = depth_pred_feats.permute(0, 3, 1, 2)
        depth_pred_feats = F.interpolate(depth_pred_feats, size=(depth_token_size, depth_token_size), mode="bilinear", align_corners=False)
    elif config['mode'] == "query":
        depth_pred_feats = depth_pred_feats.view(depth_pred_feats.shape[0], depth_token_size, depth_token_size, depth_pred_feats.shape[-1])
        depth_pred_feats = depth_pred_feats.permute(0, 3, 1, 2)

    if model_type  == 'DepthAnythingV2':
        dav2_head = vis_head

        depth_pred_feats  = depth_pred_feats.permute(0, 2, 3, 1)
        depth_pred_feats  = depth_pred_feats.view(depth_pred_feats.shape[0], -1, depth_pred_feats.shape[-1])

        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        def _get_depth_from_feat(depth_feat):
            depth = dav2_head([(depth_feat, None)] * 4)
            min_val = depth.amin(dim=(1, 2), keepdim=True)
            max_val = depth.amax(dim=(1, 2), keepdim=True)
            depth = (depth - min_val) / (max_val - min_val)
            return depth

        depth_preds = []
        depth_targets = []
        for depth_pred_feat, depth_target_feat in zip(depth_pred_feats, depth_target_feats):
            depth_pred = _get_depth_from_feat(depth_pred_feat.unsqueeze(0))
            depth_target = _get_depth_from_feat(depth_target_feat.unsqueeze(0))

            depth_preds.append(depth_pred.squeeze(0).float().detach())
            depth_targets.append(depth_target.squeeze(0).float().detach())

        def _visualize_depth(depth):
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_depth)

        pred_depths, gt_depths = [], []

        for pred, target in zip(depth_preds, depth_targets):
            gt = _visualize_depth(target.float())
            gt_depths.append(gt)

            pred = _visualize_depth(pred)
            pred_depths.append(pred)
        
        n = len(pred_depths)
        c = min(n, 16)
        r = n // c
        pred_depths = pred_depths[:c*r]
        gt_depths = gt_depths[:c*r]
        #masks_grid = make_grid(pred_depths, gt_depths)

        for idx, (pred_img, gt_img) in enumerate(zip(pred_depths, gt_depths)):
            dst_path = os.path.join(visual_dir, f"depth_dav2_{steps}_{idx}.png")
            dst = Image.new("RGB", (gt_img.width + pred_img.width, gt_img.height),)
            dst.paste(gt_img, (0, 0))
            dst.paste(pred_img, (gt_img.width, 0))
            dst.save(dst_path)
    elif model_type == 'MoRGBD':
        import cv2
        morgbd_model = vis_head
        depth_target_feats = depth_target_feats.view(depth_target_feats.shape[0], depth_token_size, depth_token_size, depth_target_feats.shape[-1])
        depth_target_feats = depth_target_feats.permute(0, 3, 1, 2)
        
        output_morgbd_preds = morgbd_model.dec_depth(depth_pred_feats, cls_token, num_tokens=256, resolution_level=3, img_h=224, img_w=224)
        output_morgbd_targets = morgbd_model.dec_depth(depth_target_feats, cls_token, num_tokens=256, resolution_level=3, img_h=224, img_w=224)

        output_morgbd_preds = output_morgbd_preds['depth_reg'].squeeze().cpu().numpy()
        output_morgbd_targets = output_morgbd_targets['depth_reg'].squeeze().cpu().numpy()

        for idx, (output_morgbd_target, output_morgbd_pred) in enumerate(zip(output_morgbd_targets, output_morgbd_preds)):

            depth_list = [output_morgbd_target, output_morgbd_pred]
            depth_color_list = [cv2.cvtColor(colorize_depth(depth_raw), cv2.COLOR_RGB2BGR) for depth_raw in depth_list]

            depth_concat = np.concatenate(depth_color_list, axis=1)

            dst_path = os.path.join(visual_dir, f"depth_morgbd_{steps}_{idx}.png")
            cv2.imwrite(dst_path,depth_concat)



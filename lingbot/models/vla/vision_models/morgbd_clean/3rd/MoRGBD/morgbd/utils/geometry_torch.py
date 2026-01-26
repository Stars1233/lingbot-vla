from typing import *
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
import utils3d

from .tools import timeit
from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def gaussian_blur_2d(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.exp(-(torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=input.dtype, device=input.device) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).reshape(1, 1, kernel_size, kernel_size)
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    input = F.conv2d(input, kernel, groups=input.shape[1])
    return input


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

def intrinsics_to_fov(intrinsics: torch.Tensor):
    """
    Returns field of view in radians from normalized intrinsics matrix.
    ### Parameters:
    - intrinsics: torch.Tensor of shape (..., 3, 3)

    ### Returns:
    - fov_x: torch.Tensor of shape (...)
    - fov_y: torch.Tensor of shape (...)
    """
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


def point_map_to_depth_legacy(points: torch.Tensor):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    # Solve least squares problem
    b = (uv * points[..., 2:]).flatten(-3, -1)                        # (..., H * W * 2)
    A = torch.stack([points[..., :2], -uv.expand_as(points[..., :2])], dim=-1).flatten(-4, -2)   # (..., H * W * 2, 2)

    M = A.transpose(-2, -1) @ A 
    solution = (torch.inverse(M + 1e-6 * torch.eye(2).to(A)) @ (A.transpose(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution.unbind(-1)

    depth = points[..., 2] + shift[..., None, None]
    fov_x = torch.atan(width / diagonal / focal) * 2
    fov_y = torch.atan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def view_plane_uv_to_focal(uv: torch.Tensor):
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], device=uv.device, dtype=uv.dtype)
    focal = (uv * normed_uv).sum() / uv.square().sum().add(1e-12)
    return focal


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    if height * width > downsample_size[0] * downsample_size[1]:
        points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
        uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
        mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    else:
        points_lr, uv_lr, mask_lr = points, uv, mask
        
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def theshold_depth_change(depth: torch.Tensor, mask: torch.Tensor, pooler: Literal['min', 'max'], rtol: float = 0.2, kernel_size: int = 3):
    *batch_shape, height, width = depth.shape
    depth = depth.reshape(-1, 1, height, width)
    mask = mask.reshape(-1, 1, height, width)
    if pooler =='max':
        pooled_depth = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth > depth * (1 + rtol)
    elif pooler =='min':
        pooled_depth = -F.max_pool2d(-torch.where(mask, depth, torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask =  pooled_depth < depth * (1 - rtol)
    else:
        raise ValueError(f'Unsupported pooler: {pooler}')
    output_mask = output_mask.reshape(*batch_shape, height, width)
    return output_mask


def dilate_with_mask(input: torch.Tensor, mask: torch.BoolTensor, filter: Literal['min', 'max', 'mean', 'median'] = 'mean', iterations: int = 1) -> torch.Tensor:
    kernel = torch.tensor([[False, True, False], [True, True, True], [False, True, False]], device=input.device, dtype=torch.bool)
    for _ in range(iterations):
        input_window = utils3d.pt.sliding_window(F.pad(input, (1, 1, 1, 1), mode='constant', value=0), window_size=3, stride=1, dim=(-2, -1))
        mask_window = kernel & utils3d.pt.sliding_window(F.pad(mask, (1, 1, 1, 1), mode='constant', value=False), window_size=3, stride=1, dim=(-2, -1))    
        if filter =='min':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.inf).min(dim=(-2, -1)).values)
        elif filter =='max':
            input = torch.where(mask, input, torch.where(mask_window, input_window, -torch.inf).max(dim=(-2, -1)).values)
        elif filter == 'mean':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).nanmean(dim=(-2, -1)))
        elif filter =='median':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).flatten(-2).nanmedian(dim=-1).values)
        mask = mask_window.any(dim=(-2, -1))
    return input, mask


def refine_depth_with_normal(depth: torch.Tensor, normal: torch.Tensor, intrinsics: torch.Tensor, iterations: int = 10, damp: float = 1e-3, eps: float = 1e-12, kernel_size: int = 5) -> torch.Tensor:
    device, dtype = depth.device, depth.dtype
    height, width = depth.shape[-2:]
    radius = kernel_size // 2

    duv = torch.stack(torch.meshgrid(torch.linspace(-radius / width, radius / width, kernel_size, device=device, dtype=dtype), torch.linspace(-radius / height, radius / height, kernel_size, device=device, dtype=dtype), indexing='xy'), dim=-1).to(dtype=dtype, device=device)

    log_depth = depth.clamp_min_(eps).log()
    log_depth_diff = utils3d.pt.sliding_window(log_depth, window_size=kernel_size, stride=1, dim=(-2, -1)) - log_depth[..., radius:-radius, radius:-radius, None, None] 
    
    weight = torch.exp(-(log_depth_diff / duv.norm(dim=-1).clamp_min_(eps) / 10).square())
    tot_weight = weight.sum(dim=(-2, -1)).clamp_min_(eps)

    uv = utils3d.pt.uv_map((height, width), device=device, dtype=dtype)
    K_inv = torch.inverse(intrinsics)

    grad = -(normal[..., None, :2] @ K_inv[..., None, None, :2, :2]).squeeze(-2) \
            / (normal[..., None, 2:] + normal[..., None, :2] @ (K_inv[..., None, None, :2, :2] @ uv[..., :, None] + K_inv[..., None, None, :2, 2:])).squeeze(-2)
    laplacian = (weight * ((utils3d.pt.sliding_window(grad, window_size=kernel_size, stride=1, dim=(-3, -2)) + grad[..., radius:-radius, radius:-radius, :, None, None]) * (duv.permute(2, 0, 1) / 2)).sum(dim=-3)).sum(dim=(-2, -1))
    
    laplacian = laplacian.clamp(-0.1, 0.1)
    log_depth_refine = log_depth.clone()

    for _ in range(iterations):
        log_depth_refine[..., radius:-radius, radius:-radius] = 0.1 * log_depth_refine[..., radius:-radius, radius:-radius] + 0.9 * (damp * log_depth[..., radius:-radius, radius:-radius] - laplacian + (weight * utils3d.pt.sliding_window_2d(log_depth_refine, window_size=kernel_size, stride=1, dim=(-2, -1))).sum(dim=(-2, -1))) / (tot_weight + damp) 

    depth_refine = log_depth_refine.exp()

    return depth_refine

def quat_to_rot(q):
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4
    if q.dim() == 1:
        q = q.unsqueeze(0)  # 1, 4
    elif q.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError
    
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    # R = torch.zeros((batch_size, 3, 3), device='cuda')
    qr = q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.zeros((batch_size, 4), dtype=R.dtype, device=R.device)

    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = R00 + R11 + R22 
    q0 = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) 
    q[:, 0] = q0

    q1 = (R21 - R12) / (4 * q0 + 1e-8)
    q2 = (R02 - R20) / (4 * q0 + 1e-8)
    q3 = (R10 - R01) / (4 * q0 + 1e-8)

    q[:, 1] = q1
    q[:, 2] = q2
    q[:, 3] = q3

    q = q / torch.norm(q, dim=1, keepdim=True)

    return q

def rast_fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def rast_focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_rast_ProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class ViewInfo(nn.Module):
    def __init__(self, cam_info, gt_info):
        super().__init__()
        self.intrinsic = cam_info['intrinsic'].cuda()
        self.pose = cam_info['pose'].cuda()
        self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
        self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
        self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
        self.raster_cam_center = cam_info['raster_cam_center'].cuda()
        self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
        self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
        self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
        self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
        self.raster_img_center = cam_info['raster_img_center'].cuda()

        if 'rgb' in gt_info and gt_info['rgb'] is not None:
            self.rgb = gt_info['rgb'].cuda()
        else:
            self.rgb = None
        self.image_path = gt_info['image_path']

        if 'mono_depth' in gt_info:
            self.mono_depth = gt_info['mono_depth'].cuda()
        if 'mono_normal_local' in gt_info:
            self.mono_normal_local = gt_info['mono_normal_local'].cuda()
        if 'index' in gt_info:
            self.index = gt_info['index']
        if 'patch_depth_high' in gt_info:
            self.patch_depth_high = gt_info['patch_depth_high'].cuda()
        if 'patch_normal_high' in gt_info:
            self.patch_normal_high = gt_info['patch_normal_high'].cuda()
        if 'patch_depth_low' in gt_info:
            self.patch_depth_low = gt_info['patch_depth_low'].cuda()
        if 'patch_normal_low' in gt_info:
            self.patch_normal_low = gt_info['patch_normal_low'].cuda()
        
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None

def get_raster_cameras_simple(intrinsics, poses, height, width, zfar=10., znear=0.01):
    focal_length_x = intrinsics[0,0]
    focal_length_y = intrinsics[1,1]
    FovY = rast_focal2fov(focal_length_y, height)
    FovX = rast_focal2fov(focal_length_x, width)

    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    c2w = poses  # 4, 4 
    w2c = c2w.inverse()  # 4, 4
    w2c_right = w2c.T

    world_view_transform = w2c_right.clone().float()
    projection_matrix = get_rast_ProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda().float()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    cam_info = {
                "intrinsic": intrinsics.clone(),
                "pose": c2w.clone(),  # camera to world
                "raster_cam_w2c": world_view_transform.clone(),
                "raster_cam_proj": projection_matrix.clone(),
                "raster_cam_fullproj": full_proj_transform.clone(),
                "raster_cam_center": camera_center.clone(),
                "raster_cam_FovX": torch.tensor([FovX]).cuda().clone(),
                "raster_cam_FovY": torch.tensor([FovY]).cuda().clone(),
                "raster_img_center": torch.tensor([cx, cy]).cuda().clone(),
            }
    gt_info = {
        "image_path": ''
    }
    return ViewInfo(cam_info, gt_info)
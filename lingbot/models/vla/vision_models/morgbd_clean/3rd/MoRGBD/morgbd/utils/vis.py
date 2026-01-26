from typing import *

import numpy as np
import matplotlib
import trimesh
from ..utils.geometry_torch import quat_to_rot
import random
import torch
import torch.nn.functional as F
import os

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)
    
    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_segmentation(segmentation: np.ndarray, cmap: str = 'Set1') -> np.ndarray:
    colored = matplotlib.colormaps[cmap]((segmentation % 20) / 20)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: Tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map

class random_color(object):
    def __init__(self, color_num=20000):
        num_of_colors=color_num
        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(num_of_colors)]

    def __call__(self, ret_n = 10):
        assert len(self.colors) > ret_n
        ret_color = np.zeros([ret_n, 3])
        for i in range(ret_n):
            hex_color = self.colors[i][1:]
            ret_color[i] = np.array([int(hex_color[j:j + 2], 16) for j in (0, 2, 4)])
        ret_color[0] *= 0
        return ret_color

def plot_rectangle_planes_trimesh(plane_centers, plane_normals, plane_radii, rot_q, suffix='', out_path=None, plane_id=None, color_type='', flip_3d=False):
    plane_normals_standard = torch.zeros_like(plane_normals)
    plane_normals_standard[..., -1] = 1
    if plane_radii.shape[-1] == 2:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 0]  # n
        radii_y_n = plane_radii[..., 1]  # n
    elif plane_radii.shape[-1] == 4:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 2]  # n
        radii_y_n = plane_radii[..., 3]  # n
    else:
        raise NotImplementedError
    
    zero_tmp = torch.zeros_like(radii_x_p)  # n
    v1 = torch.stack([radii_x_p, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v2 = torch.stack([-radii_x_n, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v3 = torch.stack([-radii_x_n, -radii_y_n, zero_tmp], dim=-1)  # n, 3
    v4 = torch.stack([radii_x_p, -radii_y_n, zero_tmp], dim=-1)  # n, 3
    
    vertices_standard = torch.stack([v1, v2, v3, v4], dim=1).cuda()  # n, 4, 3
    rot_q = F.normalize(rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    vertices_transformed = torch.bmm(rot_matrix, vertices_standard.permute(0, 2, 1)).permute(0, 2, 1) + plane_centers[:, None]  # n, 4, 3
    vertices_all = vertices_transformed.reshape(-1, 3).detach().cpu().numpy()  # 4n, 3
    
    N = vertices_all.shape[0] // 4
    if flip_3d:
        vertices_all[:, 1] *= -1
        vertices_all[:, 2] *= -1

    if color_type == 'normal':
        normal_color = (plane_normals + 1.) / 2.
        normal_color = normal_color.reshape(-1, 1, 3).repeat(1, 4, 1)
        colors = normal_color.detach().cpu().numpy().reshape(-1, 3)
        suffix = suffix + '_colorNormal'
    elif color_type == 'prim':
        if plane_id is None:
            plane_id = torch.arange(plane_centers.shape[0])
        plane_id = plane_id.reshape(-1, 1).repeat(1, 4).int().cuda()
        color_vis = random_color(plane_id.unique().max().item() + 10)
        colorMap_vis = color_vis(plane_id.unique().max().item() + 1)
        colors = colorMap_vis[plane_id.detach().cpu().numpy().reshape(-1)]
        colors = colors.astype(np.float64) / 255.
        suffix = suffix + '_colorPrim'
    else:
        pass

    idx_v1 = np.arange(N).reshape(-1, 1) * 4
    idx_v2 = idx_v1 + 1
    idx_v3 = idx_v1 + 2
    idx_v4 = idx_v1 + 3

    triangle_vIdx1 = np.concatenate([idx_v1, idx_v2, idx_v3], axis=-1)  # 2n, 3
    triangle_vIdx2 = np.concatenate([idx_v3, idx_v4, idx_v1], axis=-1)  # 2n, 3
    triangle_vIdx = np.concatenate([triangle_vIdx1, triangle_vIdx2], axis=0)  # 4n, 3

    triangle_mesh = trimesh.Trimesh(
        vertices=vertices_all,           # (V, 3), 
        faces=triangle_vIdx,                 # (F, 3), 
        vertex_colors=(colors*255).astype(np.uint8) # (V, 3) or (V, 4)，
    )

    if out_path is not None and os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        triangle_mesh.export(os.path.join(out_path, "prim_%s.ply" % (suffix)))
    return triangle_mesh

def draw_prims(plane_normal, plane_center, plane_radii,  plane_rot_q, suffix='rawPrims', epoch=-1, to_unscaled_coord=False, plane_id=None, plot_dir ="outputs_ply", flip_3d=False):
    if isinstance(plane_normal, np.ndarray):
        plane_normal = torch.from_numpy(plane_normal).cuda()
    if isinstance(plane_center, np.ndarray):
        plane_center = torch.from_numpy(plane_center).cuda()
    if isinstance(plane_radii, np.ndarray):
        plane_radii = torch.from_numpy(plane_radii).cuda()
    if isinstance(plane_rot_q, np.ndarray):
        plane_rot_q = torch.from_numpy(plane_rot_q).cuda()

    # normal_mesh = plot_rectangle_planes_trimesh(
    #     plane_center, plane_normal, plane_radii, plane_rot_q, 
    #     suffix='%s'%(suffix), 
    #     out_path=plot_dir,
    #     plane_id=plane_id, 
    #     color_type='normal')

    prim_mesh = plot_rectangle_planes_trimesh(plane_center, plane_normal, plane_radii, plane_rot_q, suffix='%s'%(suffix), out_path=plot_dir,plane_id=plane_id, color_type='prim',flip_3d=flip_3d)
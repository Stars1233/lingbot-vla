import os
from pathlib import Path
import json
import time
import random
from typing import *
import traceback
import itertools
from numbers import Number
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.v2.functional as TF
import utils3d
import pipeline
from tqdm import tqdm

from ..utils.io import *
from ..utils.geometry_numpy import harmonic_mean_numpy, norm3d, depth_occlusion_edge_numpy
from ..utils.data_augmentation import sample_perspective, warp_perspective, image_color_augmentation


class TrainDataLoaderPipeline:
    def __init__(self, config: dict, batch_size: int, num_load_workers: int = 4, num_process_workers: int = 8, buffer_size: int = 8):
        self.config = config

        self.batch_size = batch_size
        self.clamp_max_depth = config['clamp_max_depth']
        self.fov_range_absolute = config.get('fov_range_absolute', 0.0)
        self.fov_range_relative = config.get('fov_range_relative', 0.0)
        self.center_augmentation = config.get('center_augmentation', 0.0)
        self.image_augmentation = config.get('image_augmentation', [])
        self.depth_interpolation = config.get('depth_interpolation', 'bilinear')

        self.mask_depth_in_on = config.get('mask_depth_in_on', False)
        self.mask_max_ratio = config.get('mask_max_ratio', 0.5)
        self.mask_max_num = config.get('mask_max_num', 15)
        self.mask_min_num = config.get('mask_min_num', 5)
        self.min_mask_size_lb = config.get('min_mask_size_lb', 10)
        self.max_mask_size_lb = config.get('max_mask_size_lb', 40)

        # self.noise_depth_in_on = config.get('noise_depth_in_on', False)

        if 'image_sizes' in config:
            self.image_size_strategy = 'fixed'
            self.image_sizes = config['image_sizes']
        elif 'aspect_ratio_range' in config and 'area_range' in config:
            self.image_size_strategy = 'aspect_area'
            self.aspect_ratio_range = config['aspect_ratio_range']
            self.area_range = config['area_range']
        else:
            raise ValueError('Invalid image size configuration')

        # Load datasets
        self.datasets = {}
        for dataset in tqdm(config['datasets'], desc='Loading datasets'):
            name = dataset['name']
            content = Path(dataset['path'], dataset.get('index', '.index.txt')).joinpath().read_text()
            filenames = content.splitlines()
            self.datasets[name] = {
                **dataset,
                'path': dataset['path'],
                'filenames': filenames,
            }
            if 'intrinsics' in dataset:
                intrinsics = np.loadtxt(dataset['intrinsics'])
                self.datasets[name]['intrinsics'] = intrinsics.astype(np.float32)
        self.dataset_names = [dataset['name'] for dataset in config['datasets']]
        self.dataset_weights = [dataset['weight'] for dataset in config['datasets']]

        # Build pipeline
        self.pipeline = pipeline.Sequential([
            self._sample_batch,
            pipeline.Unbatch(),
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Batch(self.batch_size),
            self._collate_batch,
            pipeline.Buffer(buffer_size),
        ])

        self.invalid_instance = {
            'intrinsics': np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            'image': np.zeros((256, 256, 3), dtype=np.uint8),
            'depth': np.ones((256, 256), dtype=np.float32),  # gt depth
            'depth_in': np.ones((256, 256), dtype=np.float32),  # input depth
            'depth_mask': np.ones((256, 256), dtype=bool),
            'depth_mask_inf': np.zeros((256, 256), dtype=bool),
            'label_type': 'invalid',
        }

    def _sample_batch(self):
        batch_id = 0
        last_area = None
        while True:
            # Depending on the sample strategy, choose a dataset and a filename
            batch_id += 1
            batch = []
            
            # Sample instances
            for _ in range(self.batch_size):
                dataset_name = random.choices(self.dataset_names, weights=self.dataset_weights)[0]
                filename = random.choice(self.datasets[dataset_name]['filenames'])

                sub_files = filename.split(' ')
                if len(sub_files) == 3:
                    if self.datasets[dataset_name]['path'] in sub_files[0]:
                        rgb_path = sub_files[0]
                        depth_gt_path = sub_files[1]
                        depth_raw_path = sub_files[2]
                    else:
                        rgb_path = Path(self.datasets[dataset_name]['path'], sub_files[0])
                        depth_gt_path = Path(self.datasets[dataset_name]['path'], sub_files[1])
                        depth_raw_path = Path(self.datasets[dataset_name]['path'], sub_files[2])
                    instance = {
                        'batch_id': batch_id,
                        'seed': random.randint(0, 2 ** 32 - 1),
                        'dataset': dataset_name,
                        'filename': filename,
                        'path_rgb': rgb_path,
                        'path_depth_gt': depth_gt_path,
                        'path_depth_raw': depth_raw_path,
                        'label_type': self.datasets[dataset_name]['label_type'],
                    }
                else:
                    path = Path(self.datasets[dataset_name]['path'], filename)

                    instance = {
                        'batch_id': batch_id,
                        'seed': random.randint(0, 2 ** 32 - 1),
                        'dataset': dataset_name,
                        'filename': filename,
                        'path': path,
                        'label_type': self.datasets[dataset_name]['label_type'],
                    }
                batch.append(instance)

            # Decide the image size for this batch
            if self.image_size_strategy == 'fixed':
                width, height = random.choice(self.config['image_sizes'])
            elif self.image_size_strategy == 'aspect_area':
                area = random.uniform(*self.area_range)
                aspect_ratio_ranges = [self.datasets[instance['dataset']].get('aspect_ratio_range', self.aspect_ratio_range) for instance in batch]
                aspect_ratio_range = (min(r[0] for r in aspect_ratio_ranges), max(r[1] for r in aspect_ratio_ranges))
                aspect_ratio = random.uniform(*aspect_ratio_range)
                width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            else:
                raise ValueError('Invalid image size strategy')
            
            for instance in batch:
                instance['width'], instance['height'] = width, height
            
            yield batch

    def _load_instance(self, instance: dict):
        try:
            if 'path' in instance:
                image = read_image(Path(instance['path'], 'image.jpg'))
                depth = read_depth(Path(instance['path'], self.datasets[instance['dataset']].get('depth', 'depth.png')))
                depth_in = read_depth(Path(instance['path'], self.datasets[instance['dataset']].get('depth_in', 'depth.png')))

                meta = read_json(Path(instance['path'], 'meta.json'))
                intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
            else:
                image = read_image(Path(instance['path_rgb']))

                depth = cv2.imread(instance['path_depth_gt'], cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth_in = cv2.imread(instance['path_depth_raw'], cv2.IMREAD_UNCHANGED).astype(np.float32)

                depth[depth == 0.] = np.nan
                depth_in[depth_in == 0.] = np.nan
                # depth = read_depth(Path(instance['path_depth_gt']))
                # depth_in = read_depth(Path(instance['path_depth_raw']))

                intrinsics = self.datasets[instance['dataset']]['intrinsics']
                
            data = {
                'image': image,
                'depth': depth,
                'depth_in': depth_in,
                'intrinsics': intrinsics
            }
            instance.update({
                **data,
            })
        except Exception as e:
            print(f"Failed to load instance {instance['dataset']}/{instance['filename']} because of exception:", e)
            instance.update(self.invalid_instance)
        return instance

    def _process_instance(self, instance: Dict[str, Union[np.ndarray, str, float, bool]]):
        raw_image, raw_depth, raw_intrinsics, label_type = instance['image'], instance['depth'], instance['intrinsics'], instance['label_type']
        raw_depth_in = instance['depth_in']
        raw_normal, raw_normal_mask = utils3d.np.depth_map_to_normal_map(raw_depth, intrinsics=raw_intrinsics, mask=np.isfinite(raw_depth), edge_threshold=88)
        raw_normal = np.where(raw_normal_mask[..., None], raw_normal, np.nan)
        depth_unit = self.datasets[instance['dataset']].get('depth_unit', None)

        raw_height, raw_width = raw_image.shape[:2]
        raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(raw_intrinsics)
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height
        
        rng = np.random.default_rng(instance['seed'])

        noise_depth_in_on = self.datasets[instance['dataset']].get('noise_depth_in_on', True)
        far = self.datasets[instance['dataset']].get('far', self.clamp_max_depth)

        # Sample perspective transformation
        tgt_intrinsics, R = sample_perspective(
            raw_intrinsics, 
            tgt_aspect=tgt_aspect,
            center_augmentation=self.datasets[instance['dataset']].get('center_augmentation', self.center_augmentation),
            fov_range_absolute=self.datasets[instance['dataset']].get('fov_range_absolute', self.fov_range_absolute),
            fov_range_relative=self.datasets[instance['dataset']].get('fov_range_relative', self.fov_range_relative),
            rng=rng
        )

        # Warp
        transform = tgt_intrinsics @ R @ np.linalg.inv(raw_intrinsics)
        # - Warp image
        tgt_image = warp_perspective(raw_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
        # - Warp depth
        depth_edge_mask = utils3d.np.depth_map_edge(raw_depth, mask=np.isfinite(raw_depth), kernel_size=5, ltol=0.01)
        depth_bilinear_mask = np.isfinite(raw_depth) & ~depth_edge_mask
        warped_depth_bilinear_mask = warp_perspective(depth_bilinear_mask.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
        warped_depth_nearest = warp_perspective(raw_depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth))
        warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
        warped_depth = np.where(warped_depth_bilinear_mask == 1., warped_depth_bilinear, warped_depth_nearest)
        tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
        tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
        # - Warp depth_in
        depth_edge_mask_ = utils3d.np.depth_map_edge(raw_depth_in, mask=np.isfinite(raw_depth_in), kernel_size=5, ltol=0.01)
        depth_bilinear_mask_ = np.isfinite(raw_depth_in) & ~depth_edge_mask_
        warped_depth_bilinear_mask_ = warp_perspective(depth_bilinear_mask_.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
        warped_depth_nearest_ = warp_perspective(raw_depth_in, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth_in))
        warped_depth_bilinear_ = 1 / warp_perspective(1 / raw_depth_in, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
        warped_depth_ = np.where(warped_depth_bilinear_mask_ == 1., warped_depth_bilinear_, warped_depth_nearest_)
        tgt_depth_in = warped_depth_ / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
        # - Warp normal
        warped_normal = warp_perspective(raw_normal, transform, (tgt_height, tgt_width), interpolation='bilinear')
        tgt_normal = warped_normal @ R.T

        # import pdb; pdb.set_trace()

        # always make sure that mask is not empty
        if np.isfinite(tgt_depth).sum() / tgt_depth.size < 0.001:
            tgt_depth = np.ones_like(tgt_depth)
            instance['label_type'] = 'invalid'
        
        if np.isfinite(tgt_depth_in).sum() / tgt_depth_in.size < 0.001:
            tgt_depth_in = np.ones_like(tgt_depth_in)
            instance['label_type'] = 'invalid'
        
        # Flip augmentation
        if rng.choice([True, False]):
            tgt_image = np.flip(tgt_image, axis=1).copy()
            tgt_depth = np.flip(tgt_depth, axis=1).copy()
            tgt_depth_in = np.flip(tgt_depth_in, axis=1).copy()
            tgt_normal = np.flip(tgt_normal, axis=1).copy() * [-1, 1, 1]
            # NOTE: if cx != 0.5, flip intrinsics accordingly. 
        
        # Color augmentation
        image_augmentation = self.datasets[instance['dataset']].get('image_augmentation', self.image_augmentation)
        tgt_image = image_color_augmentation(
            tgt_image, 
            augmentations=image_augmentation, 
            rng=rng, 
            depth=tgt_depth,
        )

        # Set metric flag if depth is in metric unit
        if depth_unit is not None:
            tgt_depth *= depth_unit
            tgt_depth_in *= depth_unit
            instance['is_metric'] = True
        else:
            instance['is_metric'] = False
        
        # Clip maximum depth
        max_depth = np.nanquantile(np.where(np.isfinite(tgt_depth), tgt_depth, np.nan), 0.01) * self.clamp_max_depth
        max_depth = min(max_depth, far)
        tgt_depth = np.where(np.isfinite(tgt_depth), np.clip(tgt_depth, 0, max_depth), tgt_depth)

        max_depth_ = np.nanquantile(np.where(np.isfinite(tgt_depth_in), tgt_depth_in, np.nan), 0.01) * self.clamp_max_depth
        max_depth_ = min(max_depth_, far)
        tgt_depth_in = np.where(np.isfinite(tgt_depth_in), np.clip(tgt_depth_in, 0, max_depth_), tgt_depth_in)
        
        tgt_depth_in = np.nan_to_num(tgt_depth_in, nan=0.0, posinf=0.0, neginf=0.0)
        if noise_depth_in_on:
            depth_noise = np.random.uniform(-0.015, 0.015, size=tgt_depth_in.shape)
            depth_noise2 = np.random.normal(loc=0.0, scale=0.005, size=tgt_depth_in.shape)
            final_noise = np.clip(depth_noise + depth_noise2, -0.1, 0.1)

            tgt_depth_in = tgt_depth_in + final_noise
            tgt_depth_in = np.clip(tgt_depth_in, 0.0, max_depth_)

        # set all invalid depth to zero
        tgt_depth_in = np.nan_to_num(tgt_depth_in, nan=0.0, posinf=0.0, neginf=0.0)
        if self.mask_depth_in_on:
            num_blocks = np.random.randint(self.mask_min_num, self.mask_max_num)
            dh, dw = tgt_depth_in.shape[0], tgt_depth_in.shape[1]
            block_mask = np.ones((dh, dw), dtype=tgt_depth_in.dtype)
            for _ in range(num_blocks):
                max_block_size_h = min(max(int(dh * self.mask_max_ratio), self.max_mask_size_lb), dh-10)
                max_block_size_w = min(max(int(dw * self.mask_max_ratio), self.max_mask_size_lb), dw-10)
                min_block_size_h = int(min(max(self.min_mask_size_lb, max_block_size_h*0.2), max_block_size_h-1))
                min_block_size_w = int(min(max(self.min_mask_size_lb, max_block_size_w*0.2), max_block_size_w-1))
                block_size_h = np.random.randint(min_block_size_h, max_block_size_h)
                block_size_w = np.random.randint(min_block_size_w, max_block_size_w)
                bh, bw = block_size_h, block_size_w
                top = np.random.randint(0, dh - bh + 1)
                left = np.random.randint(0, dw - bw + 1)
                block_mask[top:top+bh, left:left+bw] = 0
            tgt_depth_in = tgt_depth_in * block_mask

        # tgt_depth_in = np.nan_to_num(tgt_depth_in, nan=0.0, posinf=0.0, neginf=0.0)
        # import pdb; pdb.set_trace()
        tgt_depth_mask_inf = np.isinf(tgt_depth)
        if self.datasets[instance['dataset']].get('finite_depth_mask', None) == "only_known":
            tgt_depth_mask_fin = np.isfinite(tgt_depth)
        else:
            tgt_depth_mask_fin = ~tgt_depth_mask_inf

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_in': torch.from_numpy(tgt_depth_in).float(),
            'depth_mask_fin': torch.from_numpy(tgt_depth_mask_fin).bool(),
            'depth_mask_inf': torch.from_numpy(tgt_depth_mask_inf).bool(),
            "normal": torch.from_numpy(tgt_normal).float(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
        })
        return instance

    def _collate_batch(self, instances: List[Dict[str, Any]]):
        # batch = {k: torch.stack([instance[k] for instance in instances], dim=0) for k in ['image', 'depth', 'depth_mask_fin', 'depth_mask_inf', 'normal', 'intrinsics']}
        batch = {k: torch.stack([instance[k] for instance in instances], dim=0) for k in ['image', 'depth', 'depth_in', 'depth_mask_fin', 'depth_mask_inf', 'normal', 'intrinsics']}
        
        batch = {
            'label_type': [instance['label_type'] for instance in instances],
            'is_metric': [instance['is_metric'] for instance in instances],
            'info': [{'dataset': instance['dataset'], 'filename': instance['filename']} for instance in instances],
            **batch,
        }
        return batch
    
    def get(self) -> Dict[str, Union[torch.Tensor, str]]:
        return self.pipeline.get()

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.stop()
        return False



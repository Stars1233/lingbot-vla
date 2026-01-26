import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click
import accelerate
import tempfile
from concurrent.futures import ThreadPoolExecutor
import trimesh

@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'oss_output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, oracle_mode: bool, oss_output_path: Union[str, Path], dump_pred: bool, dump_gt: bool):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from moge.test.baseline import MGEBaselineInterface
    from moge.test.dataloader import EvalDataLoaderPipeline
    from moge.test.metrics import compute_metrics
    from moge.utils.geometry_torch import intrinsics_to_fov
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    from train_utils import lazy_oss
    import datetime

    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)
    
    assert oss_output_path.startswith('oss://')

    oss_ckpt_path = ctx.args[ctx.args.index('--pretrained')+1]
    _temp_str = oss_ckpt_path[6:].split('/')
    bucket = _temp_str[0]
    ckpt_path = '/'.join(_temp_str[1:])
    oss_output_path = '/'.join(oss_output_path[6:].split('/')[1:])

    oss_manager = lazy_oss(bucket)

    output_root = tempfile.mkdtemp()

    output_path = os.path.join(output_root,'metrics.json')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    oss_output_path = os.path.join(oss_output_path,timestamp)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {'checkpoint': oss_ckpt_path}
    # Iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []

        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar,
            ThreadPoolExecutor(max_workers=1) as submit,
        ):  
            # Iterate over the samples in the dataset
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                image = sample['image']
                gt_intrinsics = sample['intrinsics']

                # Inference
                torch.cuda.synchronize()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image)
                    torch.cuda.synchronize()

                # Compute metrics
                metrics, misc = compute_metrics(pred, sample, vis=dump_pred or dump_gt)
                metrics['inference_time'] = timer.time
                metrics_list.append(metrics)

                # Dump results
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                
                dump_path_oss = Path(oss_output_path) / benchmark_name / sample['filename'].replace('.zip', '')
                
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    _rgb = cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'image.jpg'), str(dump_path_oss / 'pred' / 'image.jpg'))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)
                    submit.submit(oss_manager.upload_from_local, str(Path(dump_path, 'pred', 'metrics.json')), str(dump_path_oss / 'pred' / 'metrics.json'))

                    if 'pred_points' in misc:
                        points = misc['pred_points'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'points.exr'), str(dump_path_oss / 'pred' / 'points.exr'))
                        pcd = trimesh.points.PointCloud(misc['pred_points'].cpu().numpy().reshape(-1,3),colors = _rgb.reshape(-1,3))
                        _ = pcd.export(str(dump_path / 'pred' / 'points.ply'))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'points.ply'), str(dump_path_oss / 'pred' / 'points.ply'))

                    
                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if 'mask' in pred:
                            mask = pred['mask'].cpu().numpy()
                            depth = np.where(mask, depth, np.inf)
                        cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'depth.png'), str(dump_path_oss / 'pred' / 'depth.png'))

                    if 'mask' in pred:
                        mask = pred['mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'mask.png'), str(dump_path_oss / 'pred' / 'mask.png'))

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'normal.png'), str(dump_path_oss / 'pred' / 'normal.png'))

                    if 'intrinsics' in pred:
                        intrinsics = pred['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'pred' / 'fov.json'), str(dump_path_oss / 'pred' / 'fov.json'))
                
                if dump_gt:
                    _rgb = cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'image.jpg'), str(dump_path_oss / 'gt' / 'image.jpg'))

                    if 'points' in sample:
                        points = sample['points']
                        cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'points.exr'), str(dump_path_oss / 'gt' / 'points.exr'))
                        pcd = trimesh.points.PointCloud(points.cpu().numpy().reshape(-1,3),colors = _rgb.reshape(-1,3))
                        _ = pcd.export(str(dump_path / 'gt' / 'points.ply'))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'points.ply'), str(dump_path_oss / 'gt' / 'points.ply'))

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'depth.png'), str(dump_path_oss / 'gt' / 'depth.png'))

                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'normal.png'), str(dump_path_oss / 'gt' / 'normal.png'))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' /'mask.png'), str(dump_path_oss / 'gt' / 'mask.png'))

                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'gt' / 'info.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)
                        submit.submit(oss_manager.upload_from_local, str(dump_path / 'gt' / 'info.json'), str(dump_path_oss / 'gt' / 'info.json'))

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    Path(output_path).write_text(
                        json.dumps({
                            **all_metrics, 
                            benchmark_name: key_average(metrics_list)
                        }, indent=4)
                    )
                    oss_manager.upload_from_local(output_path, os.path.join(oss_output_path,'metrics.json'))
                pbar.update(1)

            all_metrics[benchmark_name] = key_average(metrics_list)

    # Save final results
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == '__main__':
    main()

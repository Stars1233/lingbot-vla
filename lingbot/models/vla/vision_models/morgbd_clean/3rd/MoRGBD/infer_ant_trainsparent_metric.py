import cv2
import torch
from morgbd.model.v1 import MoRGBDModel as v1
from morgbd.model.v2 import MoRGBDModel as v2
from morgbd.utils.vis import colorize_depth, colorize_normal
from pathlib import Path
import numpy as np
import os
import trimesh
import argparse
from tqdm import tqdm
import utils3d
from datetime import datetime
# from train_utils.oss_manager import create_oss_manager, lazy_oss
from train_utils.oss_manager2 import create_oss_manager, lazy_oss
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='MoGe模型推理脚本')
    parser.add_argument('--ckpt_path', '-c', type=str, default='', help='模型检查点路径')
    parser.add_argument('--save_dir', '-s', type=str, default='', help='结果保存目录')
    parser.add_argument('--sample_step', type=int, default=1, help='采样步长')
    parser.add_argument('--k_on', action='store_true', help='是否使用真实相机内参')
    parser.add_argument('--model_type', '-m', type=str, default='v1', help='模型类型=[v1, v2]')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    parser.add_argument('--workspace', type=str, default='workspace_MoRGBD/res_ant_rgbd', help='')

    parser.add_argument('--data_root', type=str, default='path/to/data_root', help='')
    # parser.add_argument('--data_root', type=str, default='/robby/share/3D/datasets/ant_trainsparent_v2/ant_trainsparent_v2', help='')
    # parser.add_argument('--data_root', type=str, default='/robby/share/3D/qinxiage/training_data/real_200w', help='')


    parser.add_argument('--val_list_file', type=str, default='path/to/val.txt', help='')
    # parser.add_argument('--val_list_file', type=str, default='/robby/share/3D/datasets/ant_trainsparent_v2/ant_trainsparent_v2/val.txt', help='')
    # parser.add_argument('--val_list_file', type=str, default='/robby/share/3D/qinxiage/training_data/real_200w/val_transparent.txt', help='')
    # parser.add_argument('--val_list_file', type=str, default='/robby/share/3D/qinxiage/training_data/real_200w/val_room.txt', help='')
    # parser.add_argument('--val_list_file', type=str, default='/robby/share/datasets/vilab/RobbyStereoSim/val_2k.txt', help='')

    parser.add_argument('--val_include_data_root', action='store_true')
    parser.add_argument('--oss', type=str, default=None, help='')
    return parser.parse_args()

def load_validation_list(val_list_file_path, sample_step):
    """
    加载验证集列表
    
    Args:
        val_list_file_path (str): 验证集列表文件路径
        sample_step (int): 采样步长
    
    Returns:
        list: 验证集列表
    """
    val_list = []
    with open(val_list_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            items = line.strip().split()
            val_list.append(items)
    # 按采样步长进行采样
    val_list = val_list[::sample_step]
    return val_list

def setup_camera_intrinsics(use_gt_intrinsics):
    """
    设置相机内参
    
    Args:
        use_gt_intrinsics (bool): 是否使用真实相机内参
    
    Returns:
        tuple: (fov_x, intrinsics) 视野角和相机内参矩阵
    """
    # 相机内参
    focal_x = 460.139587
    focal_y = 460.199005
    center_x = 319.656128
    center_y = 237.396271
    
    if use_gt_intrinsics:
        # 归一化相机内参
        intrinsics = np.array([
            [focal_x/640, 0, center_x/640], 
            [0, focal_y/480, center_y/480], 
            [0, 0, 1]
        ])
        fov_x, _ = utils3d.torch.intrinsics_to_fov(torch.from_numpy(intrinsics))
        fov_x = torch.rad2deg(fov_x)
    else:
        fov_x = None
        intrinsics = None
        
    return fov_x, intrinsics

def load_depth_map(depth_path):
    """
    加载深度图并转换为米单位
    
    Args:
        depth_path (str): 深度图路径
    
    Returns:
        np.ndarray: 深度图（米单位）
    """
    # 读取深度图并转换为米单位（原始单位为毫米）
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=10.0, neginf=0.0)
    return depth_map

def preprocess_input_image(image_path, device):
    """
    预处理输入图像
    
    Args:
        image_path (str): 图像路径
        device (torch.device): 设备
    
    Returns:
        tuple: (numpy_image, tensor_image) numpy格式和tensor格式的图像
    """
    # 读取图像并转换为RGB格式
    image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # 转换为tensor并归一化到[0, 1]范围
    image_tensor = torch.tensor(image_np / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    return image_np, image_tensor

def apply_mask_to_depth(depth_map, mask_level):
    """
    对深度图应用遮罩
    
    Args:
        depth_map (np.ndarray): 原始深度图
        mask_level (str): 遮罩级别 ('null', 'small', 'large')
    
    Returns:
        np.ndarray: 应用遮罩后的深度图
    """
    masked_depth = depth_map.copy()
    
    if mask_level == 'small':
        # 应用小范围遮罩
        masked_depth[110:230, 150:250] = 0.0
        masked_depth[160:280, 290:380] = 0.0
        masked_depth[380:460, 100:290] = 0.0
        masked_depth[392:461, 430:517] = 0.0
        masked_depth[80:200, 430:517] = 0.0
    elif mask_level == 'large':
        # 应用大范围遮罩
        masked_depth[30:-30, 30:-30] = 0.0
        
    return masked_depth

def calculate_depth_error(depth_gt, depth_pred):
    """
    计算深度预测误差
    
    Args:
        depth_gt (np.ndarray): 真实深度图
        depth_pred (np.ndarray): 预测深度图
    
    Returns:
        float: 平均绝对误差
    """
    # 检查预测值中的NaN和Inf
    nan_inf_mask = np.isnan(depth_pred) | np.isinf(depth_pred) | np.isnan(depth_gt) | np.isinf(depth_gt)
    valid_gt_pred_mask = ~nan_inf_mask
    
    # 只计算有效区域的误差
    valid_mask = (depth_gt > 0) & valid_gt_pred_mask & (depth_pred > 0)
    error = np.mean(np.abs((depth_gt[valid_mask] - depth_pred[valid_mask])))
    return error

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置设备
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")
    
    # 设置工作目录和数据路径
    workspace = args.workspace
    data_root = args.data_root
    val_list_file = args.val_list_file

    oss = args.oss
    if oss is not None:
        _split = oss[6:].split('/')
        oss_vis_root = '/'.join(_split[1:])
    else:
        oss_vis_root = None
    
    # 设置保存目录
    if len(args.save_dir) > 0:
        save_dir = Path(workspace).joinpath(f'{args.model_type}_{args.save_dir}')
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    # 加载验证集列表
    val_list = load_validation_list(val_list_file, args.sample_step)

    # 遮罩级别列表
    mask_level_list = ['null', 'small', 'large']
    # mask_level_list = ['null']
    
    # 为每个遮罩级别创建误差存储列表
    error_dict = {mask_level: [] for mask_level in mask_level_list}
    error_dict.update({'raw': []})
    
    # 设置相机内参
    fov_x, intrinsics = setup_camera_intrinsics(True)
    intrinsics = torch.from_numpy(intrinsics).to(device=DEVICE, dtype=torch.float32).squeeze()[None]
    
    if args.ckpt_path.startswith('oss://'):
        _split = args.ckpt_path[6:].split('/')
        bucket_name = _split[0]
        oss_root = '/'.join(_split[1:])
        oss_manager = create_oss_manager(
            os.environ.get('OSS_AK',None),
            os.environ.get('OSS_SK',None),
            bucket_name,
            endpoint=os.environ.get('OSS_DOMAIN', None),
            verbose=False,
            )
    else:
        oss_manager = None
        oss_root = None
    
    # 根据model_type加载对应的模型
    if args.model_type == 'v1':
        model = v1.from_pretrained(args.ckpt_path, oss_manager=oss_manager).to(DEVICE)
    elif args.model_type == 'v2':
        model = v2.from_pretrained(args.ckpt_path, oss_manager=oss_manager).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    model.eval()
    
    depth_key = 'null'
    if args.model_type == 'v1':
        depth_key = 'depth'
    elif args.model_type == 'v2':
        depth_key = 'depth'
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
   
    with (
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):
        # 遍历验证集进行推理
        # for val_item in tqdm(val_list, total=len(val_list)):
        for item_idx, val_item in enumerate(tqdm(val_list, total=len(val_list))):
            large_error_vis_on = False
            error_threshold = 0.1

            assert len(val_item) == 3
            if args.val_include_data_root:
                img_path = val_item[0]
                depth_gt_path = val_item[1]
                depth_raw_path = val_item[2]
            else:
                img_path = os.path.join(data_root, val_item[0])
                depth_gt_path = os.path.join(data_root, val_item[1])
                depth_raw_path = os.path.join(data_root, val_item[2])
            
            # 加载真实深度图和原始深度图
            depth_gt_np_in = load_depth_map(depth_gt_path)
            depth_raw_np_in = load_depth_map(depth_raw_path)
            
            depth_gt_np_in[depth_gt_np_in>10.0] = 0.0
            depth_raw_np_in[depth_raw_np_in>10.0] = 0.0
            error_raw = calculate_depth_error(depth_gt_np_in.copy(), depth_raw_np_in.copy())
            error_dict['raw'].append(error_raw)
            print(f"raw - 误差: {error_raw:.6f}")
            
            # 预处理输入图像
            input_image_np, input_image = preprocess_input_image(img_path, DEVICE)
            
            # 用于存储不同mask level的深度图
            depth_pred_list = []
            depth_raw_list = []
            
            # 对不同遮罩级别进行推理
            for mask_level in mask_level_list:
                depth_gt_np = depth_gt_np_in.copy()
                depth_raw_np = depth_raw_np_in.copy()
                
                # 应用遮罩
                if mask_level != 'null':
                    depth_raw_np = apply_mask_to_depth(depth_raw_np, mask_level)
                
                # 转换为tensor
                depth_raw = torch.from_numpy(depth_raw_np).to(device=DEVICE, dtype=torch.float32)
                
                # 模型推理            
                if args.model_type == 'v1' or args.model_type == 'v2':
                    output = model.infer(input_image, depth_in=depth_raw, apply_mask=False, fov_x=fov_x.clone() if fov_x is not None else fov_x, intrinsics=intrinsics)
                else:
                    raise ValueError(f"不支持的模型类型: {args.model_type}")
                
                # 获取预测深度图
                depth_pred = output[depth_key].squeeze().cpu().numpy()
                
                # 计算误差
                error = calculate_depth_error(depth_gt_np, depth_pred)
                # if error > error_threshold:
                    # large_error_vis_on = True
                
                # 将误差添加到对应遮罩级别的列表中
                error_dict[mask_level].append(error)
                print(f"{mask_level:<10} - 误差: {error:.6f}")

                # print(img_path)
                # import pdb; pdb.set_trace()     
                
                # 存储深度图用于后续拼接
                depth_pred_list.append(depth_pred)
                depth_raw_list.append(depth_raw_np)
                
                # 可视化结果
                if (save_dir is not None and item_idx % 50 == 0) or large_error_vis_on:
                    # 创建保存目录（同一个输入图像的不同mask level结果保存到同一个目录下）
                    data_subpath = os.path.join(*(val_item[0].split('/')[:-1]))
                    data_subpath = data_subpath.replace('/', '_')
                    vis_save_dir = save_dir.joinpath(data_subpath)
                    vis_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 获取预测掩码
                    mask_pred = output['mask'].squeeze().cpu().numpy()
                    mask_gt = np.ones_like(depth_pred) > 0
                    
                    # 保存输入图像 和 真实点云（只保存一次）
                    if mask_level == 'null':
                        # cv2.imwrite(str(vis_save_dir.joinpath(f'image.jpg')), cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))
                        # 保存真实点云
                        points_gt = utils3d.pt.depth_map_to_point_map(torch.from_numpy(depth_gt_np).to(device=intrinsics.device)[None], intrinsics=intrinsics).squeeze().cpu().numpy()
                        verts = points_gt[mask_gt][::2]
                        verts_color = input_image_np[mask_gt][::2]
                        point_cloud = trimesh.PointCloud(verts, verts_color)
                        point_cloud.export(str(vis_save_dir.joinpath(f'points_gt.ply')))

                        # 保存原始点云
                        points_raw = utils3d.pt.depth_map_to_point_map(torch.from_numpy(depth_raw_np_in).to(device=intrinsics.device)[None], intrinsics=intrinsics).squeeze().cpu().numpy()
                        verts = points_raw[mask_gt][::2]
                        verts_color = input_image_np[mask_gt][::2]
                        point_cloud = trimesh.PointCloud(verts, verts_color)
                        point_cloud.export(str(vis_save_dir.joinpath(f'points_raw.ply')))

                    # 保存点云数据（通过在文件名后加mask level的后缀来区分）
                    try:
                        points_pred = output['points'].squeeze().cpu().numpy()
                    except:
                        points_pred = utils3d.pt.depth_map_to_point_map(torch.from_numpy(depth_pred).to(device=intrinsics.device)[None], intrinsics=intrinsics).squeeze().cpu().numpy()
                        
                    
                    verts = points_pred[mask_pred][::2]
                    verts_color = input_image_np[mask_pred][::2]
                    point_cloud = trimesh.PointCloud(verts, verts_color)
                    point_cloud.export(str(vis_save_dir.joinpath(f'points_pred_{mask_level}.ply')))
            
            # 拼接并保存深度图
            if ((save_dir is not None and item_idx % 50 == 0) or large_error_vis_on) and len(depth_pred_list) > 0:
                # 创建保存目录（同一个输入图像的不同mask level结果保存到同一个目录下）
                data_subpath = os.path.join(*(val_item[0].split('/')[:-1]))
                data_subpath = data_subpath.replace('/', '_')
                vis_save_dir = save_dir.joinpath(data_subpath)
                vis_save_dir.mkdir(parents=True, exist_ok=True)
                
                # 获取预测掩码（使用null级别的掩码）
                # mask_pred = output['mask'].squeeze().cpu().numpy()
                mask_pred = None
                mask_gt = np.ones_like(depth_pred_list[0]) > 0

                img_out = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
                
                # 获取真实深度图
                depth_gt_color = cv2.cvtColor(colorize_depth(depth_gt_np_in, mask_gt), cv2.COLOR_RGB2BGR)
                # depth_placeholder = np.ones_like(depth_gt_color) * 255

                # 获取原始深度图
                depth_raw_color_list = [cv2.cvtColor(colorize_depth(depth_raw, mask_pred), cv2.COLOR_RGB2BGR) for depth_raw in depth_raw_list]
                depth_raw_concat = np.concatenate([depth_gt_color] + depth_raw_color_list, axis=1)

                # 拼接预测深度图
                depth_pred_color_list = [cv2.cvtColor(colorize_depth(depth_pred, mask_pred), cv2.COLOR_RGB2BGR) for depth_pred in depth_pred_list]
                depth_pred_concat = np.concatenate([img_out] + depth_pred_color_list, axis=1)
                
                depth_out = np.concatenate([depth_raw_concat, depth_pred_concat], axis=0)
                cv2.imwrite(str(vis_save_dir.joinpath(f'depth_compare_vis.png')), depth_out)

                if oss_manager and oss_vis_root:
                    save_checkpoint_executor.submit(
                        oss_manager.upload_from_local,
                        str(vis_save_dir),
                        str(Path(oss_vis_root)/vis_save_dir),
                    )
                
                if large_error_vis_on:
                    import pdb; pdb.set_trace()
                print(data_subpath)
                # import pdb; pdb.set_trace()
 
    # 计算并输出每个遮罩级别的误差均值和中位数
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"\n误差统计结果 ({depth_key} || {current_date}):")
    print(f'model path: {args.ckpt_path}')
    for mask_level in (mask_level_list+['raw']):
        errors = error_dict[mask_level]
        if errors:
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            print(f"{mask_level:<10} - 均值: {mean_error:.6f}, 中位数: {median_error:.6f}")
        else:
            print(f"{mask_level:<10} - 无数据")
    print("=" * 50)
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()

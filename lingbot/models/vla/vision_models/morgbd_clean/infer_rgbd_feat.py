import cv2
import torch
import torch.nn.functional as F
from moge.model.v3_dr import MoGeModel as v3_dr
from moge.model.v2 import MoGeModel as v2
from moge.utils.vis import colorize_depth, colorize_normal
from depth_anything_3.api import DepthAnything3
from morgbd.model.v2 import MoRGBDModel as v2_morgbd


from pathlib import Path
import numpy as np
import os
import trimesh
import argparse
from tqdm import tqdm
import utils3d
from datetime import datetime
import time

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='MoGe模型推理脚本')
    # img path
    parser.add_argument('--img_path', '-i', type=str, default='example_img2.jpg', help='图像路径')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    parser.add_argument('--depth_model', type=str, default='moge2-large', help='depth model')
    parser.add_argument('--rgbd_model', type=str, default='v2_dr', help='depth model')
    parser.add_argument('--depth_down_scale', type=int, default=1, help='depth down scale')
    return parser.parse_args()


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


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置设备
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {DEVICE}")
    
    # 设置保存目录
    save_dir = Path(f'res')
    save_dir.mkdir(parents=True, exist_ok=True)

    # load rgbd model
    if args.rgbd_model == 'v3_dr':
        morgbd_ckpt_path = 'ckpts/morgbd_v3dr.pt'
        morgbd_model = v3_dr.from_pretrained(morgbd_ckpt_path).to(DEVICE)
        morgbd_model.eval()
    elif args.rgbd_model == 'v2_morgbd':
        morgbd_ckpt_path = 'ckpts/morgbd_v2_mixdata.pt'
        morgbd_model = v2_morgbd.from_pretrained(morgbd_ckpt_path).to(DEVICE)
        morgbd_model.eval()
    else:
        raise ValueError(f'Unknown rgbd model: {args.rgbd_model}')

    # get img path
    img_path = args.img_path

    # load img
    input_image_np, input_image = preprocess_input_image(img_path, DEVICE)
    # resize to 224*224
    input_image = F.interpolate(input_image, size=(224, 224), mode='bilinear', align_corners=False)
    # resie input_image_np to 224*224
    input_image_np = cv2.resize(input_image_np, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    repeat_num = 1
    if args.depth_model == 'moge2-large':
        moge_ckpt_path = 'ckpts/moge2-vitl-normal.pt'
        # load moge2 model
        moge_model = v2.from_pretrained(moge_ckpt_path).to(DEVICE)
        moge_model.eval()
        # forward moge_model
        output_moge = moge_model.infer(input_image, resolution_level=3, num_tokens=1600)
        # output_moge = moge_model.infer(input_image)
        start_time = time.time()
        for i in range(repeat_num):
            output_moge = moge_model.infer(input_image, resolution_level=3, num_tokens=1600)
            # output_moge = moge_model.infer(input_image)
        end_time = time.time()
        print(f"depth[{args.depth_model}] 推理时间: {(end_time - start_time)/repeat_num:.2f}秒")
        # get depth moge2
        depth_pred = output_moge['depth'].squeeze().detach().clone() # moge2
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
        depth_pred *= 1
    elif args.depth_model == 'moge2-base':
        moge_ckpt_path = 'ckpts/moge2-vitb-normal.pt'
        # load moge2 model
        moge_model = v2.from_pretrained(moge_ckpt_path).to(DEVICE)
        moge_model.eval()
        # forward moge_model
        output_moge = moge_model.infer(input_image, resolution_level=3, num_tokens=1600)
        start_time = time.time()
        for i in range(repeat_num):
            output_moge = moge_model.infer(input_image, resolution_level=3, num_tokens=1600)
        end_time = time.time()
        print(f"depth[{args.depth_model}] 推理时间: {(end_time - start_time)/repeat_num:.2f}秒")
        # get depth moge2
        depth_pred = output_moge['depth'].squeeze().detach().clone() # moge2
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
        depth_pred *= 1
    elif args.depth_model == 'da3-metric-large':
        # load da3
        # Load model from Hugging Face Hub
        da3_model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
        da3_model = da3_model.to(device=DEVICE)
        da3_model.eval()
        # forward da3_model
        prediction = da3_model.inference([input_image_np], process_res=224)
        start_time = time.time()
        for i in range(repeat_num):
            # prediction = da3_model.inference([img_path])
            prediction = da3_model.inference([input_image_np], process_res=224)
        end_time = time.time()
        print(f"depth[{args.depth_model}] 推理时间: {(end_time - start_time)/repeat_num:.2f}秒")
        # print(prediction.depth.shape) 
        # get depth da3
        depth_pred = torch.from_numpy(prediction.depth).squeeze().to(DEVICE) # da3
        depth_pred = F.interpolate(depth_pred[None, None], size=input_image.shape[-2:], mode='bilinear', align_corners=False).squeeze()
        depth_pred *= 10
    else:
        raise ValueError(f'Unknown depth model: {args.depth_model}')
    
    depth_pred_np = depth_pred.cpu().numpy()
    depth_down_scale = args.depth_down_scale
    # forward morgbd_model
    feat, cls_token = morgbd_model.infer_feat(input_image, depth_pred, 
                                                depth_down_scale=depth_down_scale,
                                                resolution_level=3,
                                                num_tokens=1200)
    # ---------------------------------------- test time morgbd start
    repeat_num = 1
    start_time = time.time()
    for i in range(repeat_num):
        feat, cls_token = morgbd_model.infer_feat(input_image, depth_pred, 
                                                depth_down_scale=depth_down_scale,
                                                resolution_level=3,
                                                num_tokens=1200)
    end_time = time.time()
    print(f"morgbd (dds={depth_down_scale}) 推理时间: {(end_time - start_time)/repeat_num:.2f}秒")
    # ---------------------------------------- test time morgbd end

    # ---------------------------------------- dec depth from features
    img_h, img_w = input_image.shape[-2:]
    output_morgbd_from_feat = morgbd_model.dec_depth(feat, cls_token, num_tokens=1200, resolution_level=3, img_h=img_h, img_w=img_w)
    depth_pred_from_feat_np = output_morgbd_from_feat['depth_reg'].squeeze().cpu().numpy()

    # ---------------------------------below: test depth reg---------------------------------
    # forward morgbd_model
    output_morgbd = morgbd_model.infer(input_image, depth_pred, 
                                        depth_down_scale=depth_down_scale,
                                        resolution_level=3,
                                        num_tokens=1200)
    # get depth
    depth_pred2_np = output_morgbd['depth_reg'].squeeze().cpu().numpy()
    depth_list = [depth_pred_np, depth_pred2_np, depth_pred_from_feat_np]
    depth_color_list = [cv2.cvtColor(colorize_depth(depth_raw), cv2.COLOR_RGB2BGR) for depth_raw in depth_list]
    depth_concat = np.concatenate(depth_color_list, axis=1)
    cv2.imwrite(str(save_dir.joinpath(f'depth_compare_vis_{args.depth_model}_dds-{depth_down_scale}.png')), depth_concat)

    # depth_diff = np.abs(depth_pred_np - depth_pred2_np)
    depth_diff = np.abs(depth_pred2_np - depth_pred_from_feat_np)
    print(f'depth diff: {np.mean(depth_diff)}')
    # import pdb; pdb.set_trace()
                        
if __name__ == "__main__":
    main()

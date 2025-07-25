# Script that renders underwater image given RGB+D and backscatter / attenuation parameters
import math
import os
from os import makedirs
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render, render_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from deepseecolor.models import (
    AttenuateNetV3,
    BackscatterNetV2,
    ImprovedBackscatterNetV2,
    ImprovedAttenuateNetV3,
    RGBGuidedAttenuateNet,
    MultiscaleBackscatterNet
)

# 场景特定的水下参数 - 根据不同场景特性进行参数调整
SCENE_PARAMS = {
    'Curasao': {  # 加勒比海库拉索岛，水质清澈
        'beta_d': [2.0, 1.8, 1.4],  # 衰减系数更小，水体透明度很高
        'beta_b': [1.3, 1.2, 1.0],  # 散射系数更小
        'b_inf': [0.04, 0.14, 0.38]  # 远距离颜色更蓝
    },
    'IUI3-RedSea': {  # 红海，水质清澈
        'beta_d': [2.3, 2.1, 1.6],
        'beta_b': [1.6, 1.5, 1.2],
        'b_inf': [0.06, 0.18, 0.37]  # 红海水色偏蓝绿
    },
     'JapaneseGradens-RedSea': {  # 红海日本花园，水质中等
         'beta_d': [2.6, 2.4, 1.8],
         'beta_b': [1.9, 1.7, 1.4],
         'b_inf': [0.07, 0.2, 0.39]
    },
    'Panama': {  # 巴拿马，水质浑浊
        'beta_d': [3.0, 2.8, 2.1],  # 衰减系数较大，因为水质浑浊
        'beta_b': [2.2, 2.0, 1.7],  # 散射系数较大
        'b_inf': [0.09, 0.25, 0.35]  # 远距离颜色偏绿
    },
    'Saltpond': {  # 盐水池环境，水质中等偏浑浊
        'beta_d': [2.8, 2.6, 2.0],  # 衰减系数介于巴拿马和红海之间
        'beta_b': [2.0, 1.9, 1.6],  # 散射系数适中
        'b_inf': [0.08, 0.22, 0.32]  # 远距离颜色偏蓝绿但略浑
    }
}

# 默认参数（用于未指定场景的情况）
WATER_BETA_D = [2.6, 2.4, 1.8]
WATER_BETA_B = [1.9, 1.7, 1.4]
WATER_B_INF = [0.07, 0.2, 0.39]
# FOG_BETA_B= 1.2
FOG_BETA_B= 2.4

def get_scene_name_from_path(path):
    """从路径中提取场景名称"""
    path = str(path).lower()
    if 'curasao' in path:
        return 'Curasao'
    elif 'iui3-redsea' in path:
        return 'IUI3-RedSea'
    elif 'japanesegradens-redsea' in path:
        return 'JapaneseGradens-RedSea'
    elif 'panama' in path:
        return 'Panama'
    elif 'saltpond' in path:
        return 'Saltpond'
    return None

def detect_scene_from_training_images(path):
    """
    通过分析训练图像特征自动检测场景类型
    使用色彩分布、亮度等特性推断水体类型
    
    Args:
        path: 包含训练图像的路径
    
    Returns:
        检测到的场景类型或None
    """
    # 首先尝试从路径名称判断
    scene_name = get_scene_name_from_path(path)
    if scene_name:
        return scene_name
        
    # 路径名称无法确定时，尝试加载和分析图像特性
    try:
        import glob
        import numpy as np
        from PIL import Image
        
        # 查找训练图像
        image_formats = ['*.jpg', '*.JPG', '*.png', '*.PNG']
        images = []
        for fmt in image_formats:
            images.extend(glob.glob(str(Path(path) / 'images' / fmt)))
            images.extend(glob.glob(str(Path(path) / fmt)))
        
        if not images:
            return None
            
        # 只分析前10张图像
        sample_images = images[:min(10, len(images))]
        
        # 计算平均颜色和亮度特征
        avg_r, avg_g, avg_b = 0, 0, 0
        avg_brightness = 0
        color_std = 0
        
        for img_path in sample_images:
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            avg_r += np.mean(img[:,:,0])
            avg_g += np.mean(img[:,:,1])
            avg_b += np.mean(img[:,:,2])
            avg_brightness += np.mean(img)
            color_std += np.std(img)
        
        count = len(sample_images)
        avg_r /= count
        avg_g /= count
        avg_b /= count
        avg_brightness /= count
        color_std /= count
        
        # 基于颜色特性推断场景
        if avg_b > 0.5 and avg_brightness > 0.4 and color_std < 0.25:
            # 蓝色调明亮 - 可能是加勒比海
            return 'Curasao'
        elif avg_g > avg_r and avg_g > avg_b and avg_brightness < 0.3:
            # 绿色调浑浊 - 可能是巴拿马
            return 'Panama'
        elif avg_b > avg_r and avg_brightness > 0.3 and avg_brightness < 0.45:
            # 适中蓝色调和亮度 - 可能是红海
            return 'IUI3-RedSea'
        elif avg_brightness < 0.35 and color_std > 0.2:
            # 较暗且有较强对比度 - 可能是红海日本花园
            return 'JapaneseGradens-RedSea'
        elif avg_g > 0.25 and avg_g > avg_b and avg_brightness > 0.25 and avg_brightness < 0.4:
            # 绿色调适中，亮度中等 - 可能是盐水池
            return 'Saltpond'
    except Exception as e:
        print(f"自动检测场景类型失败: {e}")
    
    return None

def render_uw(rgb, d, scene_name=None):
    """根据场景名称渲染水下图像"""
    J = rgb

    # 根据场景名称选择参数
    if scene_name in SCENE_PARAMS:
        params = SCENE_PARAMS[scene_name]
        beta_d = params['beta_d']
        beta_b = params['beta_b']
        b_inf = params['b_inf']
    else:
        beta_d = WATER_BETA_D
        beta_b = WATER_BETA_B
        b_inf = WATER_B_INF
    
    # 计算衰减项
    tmp = torch.from_numpy(np.array(beta_d)).view((3, 1, 1, 1)).float().cuda()
    at_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp)
    attenuation = torch.exp(at_exp_coeff)

    # 计算散射项
    tmp2 = torch.from_numpy(np.array(beta_b)).view((3, 1, 1, 1)).float().cuda()
    bs_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp2)
    backscatter = 1 - torch.exp(bs_exp_coeff)

    # 应用无限远处的背景颜色
    tmp3 = torch.from_numpy(np.array(b_inf)).view(3, 1, 1).float().cuda()
    I = J * attenuation + tmp3 * backscatter
    return I

def render_fog(rgb, d):
    J = rgb
    transmittance = torch.exp(-1.0 * FOG_BETA_B * d)

    atmospheric = estimate_atmospheric_light(rgb).view(3, 1, 1)

    I = J * transmittance + atmospheric * (1 - transmittance).expand((3, *list(transmittance.shape[1:])))
    return I

def dark_channel_estimate(rgb):
    patch_size = 41
    padding =  patch_size // 2
    rgb_max = torch.nn.MaxPool3d(
        kernel_size=(3, patch_size, patch_size),
        stride=1,
        padding=(0, padding, padding)
    )
    if len(rgb.size()) == 3:
        dcp = torch.abs(rgb_max(-rgb.unsqueeze(0)))
    else:
        dcp = torch.abs(rgb_max(-rgb))
    return dcp.squeeze()

def estimate_atmospheric_light(rgb):
    dcp = dark_channel_estimate(rgb)

    flat_dcp = torch.flatten(dcp)
    flat_r = torch.flatten(rgb[0])
    flat_g = torch.flatten(rgb[1])
    flat_b = torch.flatten(rgb[2])

    k = math.ceil(0.001 * len(flat_dcp))
    vals, idxs = torch.topk(flat_dcp, k)

    median_val = torch.median(vals)
    median_idxs = torch.where(vals == median_val)
    color_idxs = idxs[median_idxs]

    atmospheric_light = torch.stack([torch.mean(flat_r[color_idxs]), torch.mean(flat_g[color_idxs]), torch.mean(flat_b[color_idxs])])

    return atmospheric_light

def render_uw_with_model(rgb, d, bs_model, at_model, scene_name=None):
    """使用深度学习模型进行水下图像渲染，支持场景特定参数"""
    # 确保输入是批次形式
    if len(rgb.shape) == 3:
        rgb = rgb.unsqueeze(0)
    if len(d.shape) == 3:
        d = d.unsqueeze(0)
    
    # 如果提供了场景名称，根据场景设置模型参数
    if scene_name in SCENE_PARAMS and hasattr(bs_model, 'B_inf') and hasattr(at_model, 'attenuation_conv_params'):
        params = SCENE_PARAMS[scene_name]
        # 设置背景散射模型参数
        try:
            bs_model.backscatter_conv_params.data = torch.tensor(params['beta_b']).reshape(3, 1, 1, 1).to(bs_model.backscatter_conv_params.device)
            bs_model.B_inf.data = torch.tensor(params['b_inf']).reshape(3, 1, 1).to(bs_model.B_inf.device)
            print(f"应用{scene_name}的散射参数")
        except Exception as e:
            print(f"设置散射模型参数失败: {e}")
        
        # 设置衰减模型参数
        try:
            at_model.attenuation_conv_params.data = torch.tensor(params['beta_d']).reshape(3, 1, 1, 1).to(at_model.attenuation_conv_params.device)
            print(f"应用{scene_name}的衰减参数")
        except Exception as e:
            print(f"设置衰减模型参数失败: {e}")
    
    # 使用模型进行衰减和散射估计
    attenuation_map = at_model(d, rgb)
    backscatter = bs_model(d, rgb)
    
    # 计算水下图像 I = J*A + B
    direct = rgb * attenuation_map
    underwater_image = torch.clamp(direct + backscatter, 0.0, 1.0)
    
    return underwater_image

def render_set(model_path, name, iteration, views, gaussians, pipeline, render_background,
               do_seathru: bool = False,
               add_water: bool = False,
               add_fog: bool = False,
               learned_bg = None,
               bs_model = None,
               at_model = None,
               save_as_jpeg: bool = False):
    # 从路径中提取场景名称
    scene_name = get_scene_name_from_path(model_path)
    
    if do_seathru:
        assert bs_model is not None
        assert at_model is not None
        no_water_dir = model_path / name / "no_water"
        backscatter_dir = model_path / name / "backscatter"
        attenuation_dir = model_path / name / "attenuation"
        with_water_dir = model_path / name / "with_water"
        makedirs(no_water_dir, exist_ok=True)
        makedirs(backscatter_dir, exist_ok=True)
        makedirs(attenuation_dir, exist_ok=True)
        makedirs(with_water_dir, exist_ok=True)

    if learned_bg is not None:
        bg_dir = model_path / name / "bg"
        makedirs(bg_dir, exist_ok=True)

    if add_water:
        water_dir = model_path / name / "add_water"
        makedirs(water_dir, exist_ok=True)

    if add_fog:
        fog_dir = model_path / name / "add_fog"
        makedirs(fog_dir, exist_ok=True)

    render_dir = model_path / name / "render"
    depth_dir = model_path / name / "depth"
    makedirs(render_dir, exist_ok=True)
    makedirs(depth_dir, exist_ok=True)

    timings_3dgs = []
    timings_seathru = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        render_pkg = render(view, gaussians, pipeline, render_background)
        rendered_image, image_alpha = render_pkg["render"], render_pkg["alpha"]
        render_depth_pkg = render_depth(view, gaussians, pipeline, render_background)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)
        depth_image = depth_image / image_alpha
        end_3dgs = time.time()
        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
            # print(f"nans/infs in depth image")
            valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
            if len(valid_depth_vals) == 0:
                print(f"[training] everything is nan {view.image_name}")
                not_nan_max = 100.0
            else:
                not_nan_max = torch.max(valid_depth_vals).item()
            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
        if depth_image.min() != depth_image.max():
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        else:
            depth_image = depth_image = depth_image / depth_image.max()

        # normalized_depth_image = depth_image / depth_image.max()

        if learned_bg is not None:
            bg_image = torch.sigmoid(learned_bg).reshape(3, 1, 1) * (1 - image_alpha)
            image = rendered_image + bg_image
            torchvision.utils.save_image(bg_image, bg_dir / f"{view.image_name}.png")
        else:
            image = rendered_image

        if add_water:
            # 使用物理模型的水下渲染
            added_uw_image = render_uw(view.original_image, depth_image, scene_name)
            torchvision.utils.save_image(added_uw_image, water_dir / f"{view.image_name}.JPG")
            
            # 如果已加载深度学习模型，也使用它渲染
            if bs_model is not None and at_model is not None:
                model_uw_image = render_uw_with_model(view.original_image, depth_image, bs_model, at_model, scene_name)
                model_water_dir = model_path / name / "model_water"
                makedirs(model_water_dir, exist_ok=True)
                torchvision.utils.save_image(model_uw_image.squeeze(), model_water_dir / f"{view.image_name}.JPG")

        if add_fog:
            added_fog_image = render_fog(view.original_image, depth_image)
            torchvision.utils.save_image(added_fog_image, fog_dir / f"{view.image_name}.JPG")

        if do_seathru:
            image_batch = torch.unsqueeze(image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)

            # 传递RGB信息给模型以增强边缘感知和特征提取
            attenuation_map_batch = at_model(depth_image_batch, image_batch)
            #attenuation_map_batch = at_model(depth_image_batch)

            backscatter_batch = bs_model(depth_image_batch, image_batch)
            #backscatter_batch = bs_model(depth_image_batch)

            direct_batch = image_batch * attenuation_map_batch
            underwater_image_batch = torch.clamp(direct_batch + backscatter_batch, 0.0, 1.0)
            end_seathru = time.time()
            timings_seathru.append(end_seathru - start_time)

            torchvision.utils.save_image(backscatter_batch.squeeze(), backscatter_dir / f"{view.image_name}.png")
            torchvision.utils.save_image(attenuation_map_batch.squeeze(), attenuation_dir / f"{view.image_name}.png")
            if save_as_jpeg:
                torchvision.utils.save_image(underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.JPG")
            else:
                torchvision.utils.save_image(underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.png")

        if save_as_jpeg:
            torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.JPG")
        else:
            torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.png")
        plt.imsave(depth_dir / f"{view.image_name}.png", depth_image.detach().cpu().numpy().squeeze())
        # torchvision.utils.save_image(normalized_depth_image, depth_dir / f"{view.image_name}.png")

        timings_3dgs.append(end_3dgs - start_time)

    if do_seathru:
        print(f"[Seathru] Average time for {len(views)} images: {np.mean(timings_seathru)}")
    print(f"[3DGS] Average time for {len(views)} images: {np.mean(timings_3dgs)}")

def render_sets(
        model_params : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
        do_seathru: bool,
        add_water: bool,
        add_fog: bool,
        use_improved_model: bool = True
):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # learned background
        learned_bg = None
        learned_bg_path =  f"{model_params.model_path}/bg_{scene.loaded_iter}.pth"
        # if os.path.exists(learned_bg_path):
        #     print(f"Loading background {learned_bg_path}")
        #     learned_bg = torch.load(learned_bg_path)

        # backscatter attenuation models
        attenuation_model_path =  f"{model_params.model_path}/attenuate_{scene.loaded_iter}.pth"
        backscatter_model_path =  f"{model_params.model_path}/backscatter_{scene.loaded_iter}.pth"
        bs_model = None
        at_model = None
        if do_seathru:
            print(f"Loading backscatter model {backscatter_model_path}")
            print(f"Loading attenuation model {attenuation_model_path}")
            assert os.path.exists(attenuation_model_path)
            assert os.path.exists(backscatter_model_path)
            
            # 检测场景类型以应用适当的初始参数
            scene_name = get_scene_name_from_path(model_params.model_path)
            print(f"检测到场景类型: {scene_name}")
            
            if use_improved_model:
                # 尝试加载为新模型，如果失败则回退到旧模型
                try:
                    at_model = RGBGuidedAttenuateNet(scale=5.0)
                    at_model.load_state_dict(torch.load(attenuation_model_path))
                    print("使用RGB引导的衰减模型")
                except:
                    try:
                        at_model = ImprovedAttenuateNetV3(scale=5.0)
                        at_model.load_state_dict(torch.load(attenuation_model_path))
                        print("使用改进的衰减模型v3")
                    except:
                        at_model = AttenuateNetV3(scale=5.0)
                        at_model.load_state_dict(torch.load(attenuation_model_path))
                        print("使用标准衰减模型v3")
                
                try:
                    bs_model = MultiscaleBackscatterNet(use_residual=False, scale=5.0)
                    bs_model.load_state_dict(torch.load(backscatter_model_path))
                    print("使用多尺度散射模型")
                except:
                    try:
                        bs_model = ImprovedBackscatterNetV2(use_residual=False, scale=5.0)
                        bs_model.load_state_dict(torch.load(backscatter_model_path))
                        print("使用改进的散射模型v2")
                    except:
                        bs_model = BackscatterNetV2(use_residual=False, scale=5.0)
                        bs_model.load_state_dict(torch.load(backscatter_model_path))
                        print("使用标准散射模型v2")
            else:
                # 使用传统模型
                at_model = AttenuateNetV3(scale=5.0)
                bs_model = BackscatterNetV2(use_residual=False, scale=5.0)
                at_model.load_state_dict(torch.load(attenuation_model_path))
                bs_model.load_state_dict(torch.load(backscatter_model_path))
                print("使用标准水下成像模型")
            
            # 如果有场景特定参数，初始化模型参数
            if scene_name in SCENE_PARAMS:
                params = SCENE_PARAMS[scene_name]
                # 设置背景散射模型参数
                if hasattr(bs_model, 'B_inf'):
                    bs_model.B_inf.data = torch.tensor(params['b_inf']).reshape(3, 1, 1).to(bs_model.B_inf.device)
                    bs_model.backscatter_conv_params.data = torch.tensor(params['beta_b']).reshape(3, 1, 1, 1).to(bs_model.backscatter_conv_params.device)
                    print(f"初始化{scene_name}的散射参数")
                
                # 设置衰减模型参数
                if hasattr(at_model, 'attenuation_conv_params'):
                    at_model.attenuation_conv_params.data = torch.tensor(params['beta_d']).reshape(3, 1, 1, 1).to(at_model.attenuation_conv_params.device)
                    print(f"初始化{scene_name}的衰减参数")
            
            at_model.cuda()
            bs_model.cuda()
            at_model.eval()
            bs_model.eval()

        if not skip_train:
             render_set(Path(model_params.model_path), "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, do_seathru, add_water, add_fog, learned_bg, bs_model, at_model)

        if not skip_test:
             render_set(Path(model_params.model_path), "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, do_seathru, add_water, add_fog, learned_bg, bs_model, at_model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser) # need to specify -s or -m and maybe --eval
    pipeline = PipelineParams(parser) # safe to ignore
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seathru", action="store_true")
    parser.add_argument("--add_water", action="store_true")
    parser.add_argument("--add_fog", action="store_true")
    parser.add_argument("--use_improved_model", action="store_true", help="使用改进的RGB引导衰减和多尺度散射模型")
    parser.add_argument("--scene_name", type=str, default=None, help="场景名称，用于选择特定的水下参数")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=0)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.seathru, args.add_water, args.add_fog, args.use_improved_model)

import math

from kornia.color import rgb_to_lab
import torch
import torch.nn as nn

class AttenuateLoss(nn.Module):
    def __init__(self, override=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5
        self.override=override

    def forward(self, direct, J):
        if self.override is not None:
            return self.override

        # dxy: this doesn't seem to do much (just pushes to be between 0-1)
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return intensity_loss + spatial_variation_loss + saturation_loss

class BackscatterLoss(nn.Module):
    def __init__(self, override=None, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.override=override

    def forward(self, direct, depth=None):
        if self.override is not None:
            return self.override

        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        if (neg > 0):
            print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss

class DarkChannelPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 41
        self.padding = self.patch_size // 2
        self.rgb_max = nn.MaxPool3d(
            kernel_size=(3, self.patch_size, self.patch_size),
            stride=1,
            padding=(0, self.padding, self.padding))
        self.l1 = nn.L1Loss()

    def forward(self, rgb, d=None):
        '''
        Following the procedure from:
        (1) Single image haze removal using dark channel prior (2009)

        * Run a 15x15 kernel across the whole image and take the minimum
        between all color channels

        Assumes rgb is in BCHW
        '''
        dcp = torch.abs(self.rgb_max(-rgb))
        loss = self.l1(dcp, torch.zeros_like(dcp))
        return loss, dcp

class DarkChannelPriorLossV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.num_depth_bins = 10
        self.pct = 0.01

    def rgb2gray(self, rgb):
        # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        # let's just use opencv standard values
        bs, _, h, w = rgb.size()
        gray = torch.zeros((bs, 1, h, w)).to(rgb.device)
        gray[:] = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        return gray

    def forward(self, rgb, d):
        '''
        Following the procedure from:
        (1) Single image haze removal using dark channel prior (2009)
        (2) A Method for Removing Water From Underwater Images (2019)

        * Bin up depth
        * Estimate darkest 1% color intensity value in each bin
        * Push these pixels to be black

        Assumes rgb is in BCHW
        '''
        d_min, d_max = d.min(), d.max()
        d_range = d_max - d_min

        loss = 0.0

        gray = self.rgb2gray(rgb)
        dcp = torch.zeros_like(gray)

        for i in range(self.num_depth_bins):
            # yes i'm creating a copy of the gray image every time
            gray = self.rgb2gray(rgb)

            lo = d_min + i * d_range / self.num_depth_bins
            hi = d_min + (i + 1) * d_range / self.num_depth_bins

            d_mask = torch.logical_and(d >= lo, d < hi)
            num_vals = torch.sum(d_mask)

            if num_vals == 0:
                print(f"empty depth bin")
                continue

            gray_bin = gray[d_mask]
            sorted_gray_bin, _ = torch.sort(gray_bin)

            try:
                gray_threshold = sorted_gray_bin[math.ceil(num_vals * self.pct) - 1]
            except:
                import pdb; pdb.set_trace()

            lt_gray_thresh = gray <= gray_threshold

            dcp_mask = torch.logical_and(d_mask, lt_gray_thresh)
            dcp[dcp_mask] = gray[dcp_mask]

        loss = self.l1(dcp, torch.zeros_like(dcp))

        return loss, dcp

class DarkChannelPriorLossV3(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        # if (neg > 0):
        #     print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss, torch.zeros_like(direct)

class DeattenuateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self, direct, J):
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return saturation_loss + intensity_loss + spatial_variation_loss

class GrayWorldPriorLoss(nn.Module):
    def __init__(self, target_intensity=0.5):
        super().__init__()
        self.target_intensity = target_intensity

    def forward(self, J):
        '''
        J: bchw or bc(flat)
        '''
        if len(J.size()) == 4:
            channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 3:
            channel_intensities = torch.mean(J, dim=[-1])
        else:
            assert False
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        return intensity_loss

class RgbSpatialVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, J, direct):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return spatial_variation_loss

class RgbSaturationLoss(nn.Module):
    '''
    useful for keeping tensor values corresponding to an image within a range
    e.g. to penalize for being outside of [0, 1] set saturation val to 1.0
    '''
    def __init__(self, saturation_val: float):
        super().__init__()
        self.relu = nn.ReLU()
        self.saturation_val = saturation_val

    def forward(self, rgb):
        saturation_loss = (self.relu(-rgb) + self.relu(rgb - self.saturation_val)).square().mean()
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        return saturation_loss

class AlphaBackgroundLoss(nn.Module):
    '''
    penalize alpha mask for values close (by L2 norm RGB) to the background color

    should probably detach rgb and background when they come into this
    '''
    def __init__(self, use_kornia: bool = False):
        super().__init__()
        self.use_kornia = use_kornia
        if use_kornia:
            self.range = math.sqrt(100 * 100 + 255 * 255 + 255 * 255)
            self.threshold = 50
        else:
            self.range = math.sqrt(3)
            self.threshold = 0.2 * math.sqrt(3)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, rgb, background, alpha):
        if self.use_kornia:
            lab_background = rgb_to_lab(background.reshape(3, 1, 1))
            lab_image = rgb_to_lab(rgb)
            diff = lab_image - lab_background
            dist = torch.linalg.vector_norm(diff, dim=0)
        else:
            if len(rgb.size()) == 2:
                diff = rgb - background
                dist = torch.linalg.vector_norm(diff, dim=1)
            else:
                diff = rgb - background.reshape(3, 1, 1)
                dist = torch.linalg.vector_norm(diff, dim=0)

        # other approach
        new_approach = False
        if new_approach:
            clamped_diff = torch.max(dist - self.threshold, torch.Tensor([0.0]).cuda())
            if self.use_kornia:
                mask = torch.exp(-clamped_diff / 10)
            else:
                mask = torch.exp(-clamped_diff / 0.05)
            masked_alpha = alpha * mask
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.mse(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb; pdb.set_trace()
        else:
            mask = dist < self.threshold
            if len(alpha.size()) == 1:
                masked_alpha = alpha[mask]
            else:
                masked_alpha = alpha[:, mask]
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.l1(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb; pdb.set_trace()
        return loss


def mixture_of_laplacians_loss(x):
    lp1 = torch.exp(-torch.abs(x)/0.1)
    lp2 = torch.exp(-torch.abs(1-x)/0.1)
    return -torch.mean(torch.log(lp1 + lp2))

class EdgeAwareBackscatterLoss(nn.Module):
    """
    边缘感知散射损失函数
    在深度边缘处对散射进行更严格的约束，期望边缘处的散射强度更低。
    """
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.edge_weight = edge_weight
        
    def forward(self, backscatter, depth):
        # 计算深度边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(depth.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(depth.device)
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        
        # 深度边缘处应有较低的散射
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_mask = edge_magnitude > edge_magnitude.mean()
        
        # 边缘处散射应较低，非边缘处正常
        edge_loss = self.l1(backscatter * edge_mask, torch.zeros_like(backscatter * edge_mask))
        normal_loss = self.smooth_l1(backscatter, torch.clamp(backscatter, 0, 1))
        
        return normal_loss + self.edge_weight * edge_loss

class MultiscaleFeatureLoss(nn.Module):
    """
    多尺度特征损失函数
    在不同尺度下比较输出与目标，捕捉全局和局部信息。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, target, depth=None):
        # 基础损失
        base_loss = self.l1(output, target)
        
        # 多尺度处理
        scales = [1, 2, 4]
        ms_loss = 0.0
        
        for scale in scales:
            if scale > 1:
                size = (output.shape[2] // scale, output.shape[3] // scale)
                output_down = torch.nn.functional.interpolate(output, size=size, mode='bilinear', align_corners=False)
                target_down = torch.nn.functional.interpolate(target, size=size, mode='bilinear', align_corners=False)
                ms_loss += self.l1(output_down, target_down)
        
        return base_loss + 0.5 * ms_loss / len(scales)

class PhysicalPriorLoss(nn.Module):
    """
    物理先验约束损失
    应用水下物理规律：不同波长光在水中的衰减速率不同，
    通常红光衰减最快，蓝光衰减最慢。
    """
    def __init__(self, channel_weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        # 波长约束权重：红色衰减最强，蓝色最弱
        if channel_weights is None:
            self.channel_weights = torch.tensor([1.0, 0.8, 0.6])  # RGB
        else:
            self.channel_weights = channel_weights
            
    def forward(self, attenuation):
        # 获取每个通道的平均衰减系数
        mean_attenuation = torch.mean(attenuation, dim=[2, 3])  # [B,3]
        
        batch_size = attenuation.shape[0]
        channel_weights = self.channel_weights.to(attenuation.device).expand(batch_size, -1)
        
        # 约束不同通道衰减系数的相对大小关系
        # 红色通道应有最低值(最强衰减)，蓝色通道最高值(最弱衰减)
        loss = 0.0
        
        # 确保红色衰减 < 绿色衰减 < 蓝色衰减
        red_green_constraint = torch.relu(mean_attenuation[:, 0] - mean_attenuation[:, 1])
        green_blue_constraint = torch.relu(mean_attenuation[:, 1] - mean_attenuation[:, 2])
        
        loss = red_green_constraint.mean() + green_blue_constraint.mean()
        
        return loss

class WaterTypeAdaptiveLoss(nn.Module):
    """
    水体类型自适应损失
    根据估计的水体清澈度自动调整散射和衰减损失的权重。
    清澈水体更看重衰减，浑浊水体更看重散射。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, rgb, backscatter, attenuation):
        # 估计水体类型
        avg_color = torch.mean(rgb, dim=[2, 3])  # [B,3]
        
        # 蓝绿比例作为水体清澈度指标
        water_clarity = avg_color[:, 2] / (avg_color[:, 1] + 1e-6)
        
        # 根据水体类型计算权重系数（维持为标量）
        bs_weight = torch.sigmoid(5.0 - 10.0 * water_clarity).mean()
        
        # 清澈水体：更看重衰减损失；浑浊水体：更看重散射损失
        backscatter_constraint = torch.mean(backscatter)
        attenuation_constraint = torch.mean(1.0 - attenuation)
        
        # 使用标量权重确保形状兼容
        loss = bs_weight * backscatter_constraint + (1.0 - bs_weight) * attenuation_constraint
        
        return loss

class BackscatterAttenuationConsistencyLoss(nn.Module):
    """
    散射-衰减一致性损失
    强制散射与衰减表现出相互一致的关系，
    同时考虑它们与深度的物理关系。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, backscatter, attenuation, depth):
        # 散射应随深度增加而增加
        # 衰减应随深度增加而减小
        
        # 计算散射与深度的相关性约束
        bs_depth_correlation = torch.mean(backscatter * depth)
        
        # 计算衰减与深度的负相关性约束
        at_depth_correlation = -torch.mean(attenuation * depth)
        
        # 散射与衰减的互补约束
        bs_at_consistency = self.mse(backscatter + attenuation, torch.ones_like(backscatter))
        
        return bs_depth_correlation + at_depth_correlation + 0.1 * bs_at_consistency

class ImprovedEdgeSmoothLoss(nn.Module):
    """
    改进的边缘平滑度损失
    根据RGB图像的边缘信息和深度边缘，
    对结果进行自适应的平滑约束。
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, depth, rgb=None):
        # 计算深度梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(output.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(output.device)
        
        # 提取RGB边缘（如果有）
        if rgb is not None:
            # 转为灰度图
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            
            pad = nn.ReplicationPad2d(1)
            gray_padded = pad(gray)
            gray_grad_x = torch.abs(torch.nn.functional.conv2d(gray_padded, sobel_x))
            gray_grad_y = torch.abs(torch.nn.functional.conv2d(gray_padded, sobel_y))
            rgb_edge = torch.sqrt(gray_grad_x**2 + gray_grad_y**2)
            
            # RGB边缘权重
            rgb_weight = torch.exp(-10.0 * rgb_edge)
        else:
            rgb_weight = 1.0
            
        # 将输出转换为灰度图
        output_gray = 0.299 * output[:, 0:1] + 0.587 * output[:, 1:2] + 0.114 * output[:, 2:3]
        
        # 计算输出梯度
        pad = nn.ReplicationPad2d(1)
        output_gray_padded = pad(output_gray)
        output_grad_x = torch.nn.functional.conv2d(output_gray_padded, sobel_x)
        output_grad_y = torch.nn.functional.conv2d(output_gray_padded, sobel_y)
        
        # 只在非RGB边缘处强制平滑
        smooth_loss = (rgb_weight * (torch.abs(output_grad_x) + torch.abs(output_grad_y))).mean()
        
        return smooth_loss


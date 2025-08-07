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
    Edge-aware backscatter loss function
    Applies stricter constraints on scattering at depth edges, expecting lower scattering intensity at edges.
    """
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.edge_weight = edge_weight
        
    def forward(self, backscatter, depth):
        # calculate depth edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(depth.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(depth.device)
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        
        # depth edges should have lower scattering
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_mask = edge_magnitude > edge_magnitude.mean()
        
        # edges should have lower scattering, non-edges remain normal
        edge_loss = self.l1(backscatter * edge_mask, torch.zeros_like(backscatter * edge_mask))
        normal_loss = self.smooth_l1(backscatter, torch.clamp(backscatter, 0, 1))
        
        return normal_loss + self.edge_weight * edge_loss

class MultiscaleFeatureLoss(nn.Module):
    """
    Multiscale feature loss function
    Compares output with target at different scales, capturing global and local information.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, target, depth=None):
        # base loss
        base_loss = self.l1(output, target)
        
        # multiscale processing
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
    Physical prior constraint loss
    Applies underwater physics laws: different wavelengths attenuate at different rates in water,
    typically red light attenuates fastest, blue light attenuates slowest.
    """
    def __init__(self, channel_weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        # wavelength constraint weights: red attenuates strongest, blue weakest
        if channel_weights is None:
            self.channel_weights = torch.tensor([1.0, 0.8, 0.6])  # RGB
        else:
            self.channel_weights = channel_weights
            
    def forward(self, attenuation):
        # get average attenuation coefficient for each channel
        mean_attenuation = torch.mean(attenuation, dim=[2, 3])  # [B,3]
        
        batch_size = attenuation.shape[0]
        channel_weights = self.channel_weights.to(attenuation.device).expand(batch_size, -1)
        
        # constrain relative magnitude relationships between channel attenuation coefficients
        # red channel should have lowest value (strongest attenuation), blue channel highest (weakest)
        loss = 0.0
        
        # ensure red attenuation < green attenuation < blue attenuation
        red_green_constraint = torch.relu(mean_attenuation[:, 0] - mean_attenuation[:, 1])
        green_blue_constraint = torch.relu(mean_attenuation[:, 1] - mean_attenuation[:, 2])
        
        loss = red_green_constraint.mean() + green_blue_constraint.mean()
        
        return loss

class WaterTypeAdaptiveLoss(nn.Module):
    """
    Water type adaptive loss
    Automatically adjusts weights of scattering and attenuation losses based on estimated water clarity.
    Clear water focuses more on attenuation, turbid water focuses more on scattering.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, rgb, backscatter, attenuation):
        # estimate water type
        avg_color = torch.mean(rgb, dim=[2, 3])  # [B,3]
        
        # blue-green ratio as water clarity indicator
        water_clarity = avg_color[:, 2] / (avg_color[:, 1] + 1e-6)
        
        # calculate weight coefficient based on water type (keep as scalar)
        bs_weight = torch.sigmoid(5.0 - 10.0 * water_clarity).mean()
        
        # clear water: focus more on attenuation loss; turbid water: focus more on scattering loss
        backscatter_constraint = torch.mean(backscatter)
        attenuation_constraint = torch.mean(1.0 - attenuation)
        
        # use scalar weights to ensure shape compatibility
        loss = bs_weight * backscatter_constraint + (1.0 - bs_weight) * attenuation_constraint
        
        return loss

class BackscatterAttenuationConsistencyLoss(nn.Module):
    """
    Backscatter-attenuation consistency loss
    Enforces mutually consistent relationship between scattering and attenuation,
    while considering their physical relationship with depth.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, backscatter, attenuation, depth):
        # scattering should increase with depth
        # attenuation should decrease with depth
        
        # calculate scattering-depth correlation constraint
        bs_depth_correlation = torch.mean(backscatter * depth)
        
        # calculate attenuation-depth negative correlation constraint
        at_depth_correlation = -torch.mean(attenuation * depth)
        
        # complementary constraint between scattering and attenuation
        bs_at_consistency = self.mse(backscatter + attenuation, torch.ones_like(backscatter))
        
        return bs_depth_correlation + at_depth_correlation + 0.1 * bs_at_consistency

class ImprovedEdgeSmoothLoss(nn.Module):
    """
    Improved edge smoothness loss
    Applies adaptive smoothing constraints on results based on 
    edge information from RGB images and depth edges.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, depth, rgb=None):
        # calculate depth gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(output.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(output.device)
        
        # extract RGB edges (if available)
        if rgb is not None:
            # convert to grayscale
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            
            pad = nn.ReplicationPad2d(1)
            gray_padded = pad(gray)
            gray_grad_x = torch.abs(torch.nn.functional.conv2d(gray_padded, sobel_x))
            gray_grad_y = torch.abs(torch.nn.functional.conv2d(gray_padded, sobel_y))
            rgb_edge = torch.sqrt(gray_grad_x**2 + gray_grad_y**2)
            
            # RGB edge weights
            rgb_weight = torch.exp(-10.0 * rgb_edge)
        else:
            rgb_weight = 1.0
            
        # convert output to grayscale
        output_gray = 0.299 * output[:, 0:1] + 0.587 * output[:, 1:2] + 0.114 * output[:, 2:3]
        
        # calculate output gradients
        pad = nn.ReplicationPad2d(1)
        output_gray_padded = pad(output_gray)
        output_grad_x = torch.nn.functional.conv2d(output_gray_padded, sobel_x)
        output_grad_y = torch.nn.functional.conv2d(output_gray_padded, sobel_y)
        
        # enforce smoothness only at non-RGB edges
        smooth_loss = (rgb_weight * (torch.abs(output_grad_x) + torch.abs(output_grad_y))).mean()
        
        return smooth_loss


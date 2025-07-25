import torch
import torch.nn as nn

class BackscatterNet(nn.Module):
    def __init__(self, use_residual: bool = True):
        super().__init__()

        self.backscatter_conv = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.uniform_(self.backscatter_conv.weight, 0, 5)

        self.use_residual = use_residual
        if use_residual:
            self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)
            nn.init.uniform_(self.residual_conv.weight, 0, 5)
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, depth, uw_image=None):
        beta_b_conv = self.relu(self.backscatter_conv(depth))
        Bc = self.B_inf * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            beta_d_conv = self.relu(self.residual_conv(depth))
            residual_term = self.J_prime * torch.exp(-beta_d_conv)
            Bc = Bc + residual_term  # 避免原地操作
        backscatter = self.sigmoid(Bc)

        # if depth is zero'd out (i.e. bad estimate), do not use it for backscatter either
        backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)

        # backwards compat with og code
        if uw_image is not None:
            direct = uw_image - backscatter_masked
            return direct, backscatter_masked
        else:
            return backscatter_masked

class BackscatterNetV2(nn.Module):
    '''
    backscatter = B_inf * (1 - exp(- a * z)) + J_prime * exp(- b * z)

    main difference with bsv1 is B_inf and J_prime go through a sigmoid
    which might make them more easily learnable (and keep them constrained to [0, 1])
    '''
    def __init__(self, use_residual: bool = False, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()

        self.scale = scale

        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

        self.use_residual = use_residual
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.B_inf = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1))

        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()

        self.do_sigmoid = do_sigmoid

        print(f"Using backscatterv2 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        if self.do_sigmoid:
            beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params)))
        else:
            # beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.backscatter_conv_params))
            beta_b_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.backscatter_conv_params), 0.0)

        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params)))
            else:
                # beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.residual_conv_params))
                beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.residual_conv_params), 0.0)
            residual_term = torch.sigmoid(self.J_prime) * torch.exp(-beta_d_conv)
            Bc = Bc + residual_term  # 避免原地操作
        backscatter = Bc

        # if depth is zero'd out (i.e. bad estimate), do not use it for backscatter either
        # backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)

        # backwards compat with og code
        return backscatter

    def forward_rgb(self, rgb):
        from render_uw import estimate_atmospheric_light

        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))

        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)

        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())

class AttenuateNet(nn.Module):
    '''
    beta_d(z) = a  * exp(-b * z) + c * exp(-d * z)
    a, c: (0, inf)
    b, d: (0, inf)

    attenuation_map = exp(-beta_d * z)
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False):
        super().__init__()
        self.attenuation_conv_params = nn.Parameter(torch.rand(6, 1, 1, 1)) # b, d from SeaThru
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1)) # a, c from SeaThru

        self.relu = nn.ReLU()

        self.scale = scale
        self.do_sigmoid = do_sigmoid
        print(f"Using attenuatenetv1 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        # true_color: J
        # generates attenuation coefficients, a_c(z) (DSC eqn 12)
        if self.do_sigmoid:
            attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))))
            beta_d = torch.stack(tuple(
                torch.sum(attn_conv[:, i:i + 2, :, :] * torch.sigmoid(self.attenuation_coef[i:i + 2]), dim=1) for i in
                range(0, 6, 2)), dim=1)
        else:
            # attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.attenuation_conv_params)))
            attn_conv = torch.exp(-torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0))
            beta_d = torch.stack(tuple(
                torch.sum(attn_conv[:, i:i + 2, :, :] * torch.clamp(self.attenuation_coef[i:i + 2]), dim=1) for i in
                range(0, 6, 2)), dim=1)

        # generate attenuation map A_c(z) = GED(z * a_c(z)) (DSC eqn 13)
        # attenuation_map = torch.exp(-1.0 * torch.relu(beta_d) * depth)
        attenuation_map = torch.exp(-1.0 * torch.clamp(beta_d, 0.0) * depth)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map

class AttenuateNetV2(nn.Module):
    '''
    beta_d(z) = a  * exp(-b * z) + c * exp(-d * z)
    a, c: (0, inf)
    b, d: (0, inf)

    attenuation_map = exp(-beta_d * z)

    this version drops c, d terms
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False):
        super().__init__()
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        self.attenuation_coef = nn.Parameter(torch.rand(3, 1, 1))
        self.relu = nn.ReLU()

        self.scale = scale
        self.do_sigmoid = do_sigmoid
        print(f"Using attenuatenetv2 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        # true_color: J

        # generates attenuation coefficients, a_c(z) (DSC eqn 12)
        if self.do_sigmoid:
            attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))))
            beta_d = torch.concatenate(tuple(
                torch.sum(attn_conv[:, i:i+1, :, :] * torch.sigmoid(self.attenuation_coef[i]), dim=1, keepdim=True) for i in
                range(3)), dim=1)
        else:
            # attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.attenuation_conv_params)))
            attn_conv = torch.exp(-torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0))
            beta_d = torch.concatenate(tuple(
                torch.sum(attn_conv[:, i:i+1, :, :] * torch.clamp(self.attenuation_coef[i], 0.0), dim=1, keepdim=True) for i in
                range(3)), dim=1)

        # generate attenuation map A_c(z) = GED(z * a_c(z)) (DSC eqn 13)
        # attenuation_map = torch.exp(-1.0 * torch.relu(beta_d) * depth)
        attenuation_map = torch.exp(-1.0 * torch.clamp(beta_d, 0.0) * depth)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class AttenuateNetV3(nn.Module):
    '''
    attenuation_map = exp(-beta_d * z)

    this one
    * does not try to scale the parameters (so here, they lie between 0 and 1 i.e. sigmoid output)
    * does not have any max attenuation
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()

        # self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.3, 1.2, 0.1]).reshape(3, 1, 1, 1)) #nn.Parameter(torch.rand(3, 1, 1, 1))
        # self.attenuation_conv_params = nn.Parameter(inverse_sigmoid(torch.Tensor([0.8, 0.8, 0.2])).reshape(3, 1, 1, 1))
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
        self.attenuation_coef = None
        self.scale = scale
        self.do_sigmoid = do_sigmoid

        self.relu = nn.ReLU()
        print(f"Using attenuatenetv3 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        if self.do_sigmoid:
            beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params)))
        else:
            beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0)

        attenuation_map = torch.exp(-beta_d_conv)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map
    
class ImprovedBackscatterNetV2(nn.Module):
    def __init__(self, use_residual: bool = True, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.use_residual = use_residual
        
        # 基本参数
        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            
        # 边缘感知处理
        self.edge_process = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.B_inf = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1))
        
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        
        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()
        
    def forward(self, depth, rgb=None):
        # 使用Sobel滤波器替代diff()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # 生成边缘感知因子
        edge_aware_factor = self.edge_process(combined_gradient)
        # 调整大小以匹配原始深度图
        edge_aware_factor = torch.nn.functional.interpolate(
            edge_aware_factor, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False
        )
        
        # 基础散射系数计算
        if self.do_sigmoid:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params))
        else:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.backscatter_conv_params)
            
        # 应用边缘感知因子调制散射系数 - 在边缘处减少散射
        beta_b_conv = beta_b_conv * (1.0 - 0.1 * edge_aware_factor)
        beta_b_conv = torch.clamp(beta_b_conv, 0.01, None)  # 设置最小值0.01
        
        # 计算散射
        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        
        # 残差项处理
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params))
            else:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.residual_conv_params)
            
            # 应用置信度和边缘信息 - 避免原地操作
            modified_beta_d = beta_d_conv * (1.0 - 0.1 * edge_aware_factor)
            beta_d_clamped = torch.clamp(modified_beta_d, 0.0)
            
            # 应用残差 - 避免原地操作
            residual_term = torch.sigmoid(self.J_prime) * torch.exp(-beta_d_clamped)
            Bc_with_residual = Bc + residual_term
            Bc_final = torch.clamp(Bc_with_residual, 0.01, 1.0)
        else:
            # 没有残差项，直接裁剪
            Bc_final = torch.clamp(Bc, 0.01, 1.0)
        
        return Bc_final
    
    def forward_rgb(self, rgb):
        from render_uw import estimate_atmospheric_light
        
        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))
            
        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)
        
        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())

class ImprovedAttenuateNetV3(nn.Module):
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        
        # 基本衰减参数
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
            
        # 波长物理先验
        self.channel_weights = nn.Parameter(torch.Tensor([1.0, 0.8, 0.6]).reshape(3, 1, 1))
        
        # 边缘感知处理
        self.edge_modulation = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),  # 输出3通道对应RGB
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, depth, rgb=None):
        # 使用Sobel滤波器替代diff()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # 生成边缘感知调制因子
        edge_factor = self.edge_modulation(combined_gradient)  # [B,3,H,W-1]
        # 调整尺寸以匹配原始深度图
        edge_factor = torch.nn.functional.interpolate(
            edge_factor, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False
        )
        
        # 基本衰减系数计算
        if self.do_sigmoid:
            beta_base = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))
        else:
            beta_base = torch.nn.functional.conv2d(depth, self.attenuation_conv_params)
        
        if rgb is not None:
            # 应用水体类型特定参数 - 避免原地操作
            modified_beta = torch.zeros_like(beta_base)
            for i in range(3):
                modified_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * type_specific_params[:,i:i+1,:,:]
            beta_base = modified_beta
        
        # 应用通道特定权重（波长物理先验）
        weighted_beta = torch.zeros_like(beta_base)
        for i in range(3):
            weighted_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * self.channel_weights[i:i+1]
        
        # 应用边缘感知调制 - 在边缘处调整衰减强度
        # 边缘处增强通透性（减少衰减）
        modulated_beta = weighted_beta * (1.0 - 0.2 * edge_factor)
        beta_d_conv = torch.clamp(modulated_beta, 0.0)
        
        # 计算衰减图
        attenuation_map = torch.exp(-beta_d_conv)
        
        return attenuation_map

class RGBGuidedAttenuateNet(nn.Module):
    """
    RGB引导的衰减优化模型
    - 利用RGB边缘信息增强深度边缘感知
    - 实现通道特定的衰减系数学习
    - 设计波长相关的光谱衰减模型
    """
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        
        # 基本衰减参数 - 针对RGB三个通道
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
            
        # 波长物理先验 - 根据物理规律不同波长的吸收率不同
        self.channel_weights = nn.Parameter(torch.Tensor([1.0, 0.8, 0.6]).reshape(3, 1, 1))
        
        # RGB特征提取网络
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 深度特征提取网络
        self.depth_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 特征融合网络 - 处理RGB+深度情况下的特征
        self.fusion_network = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合网络 - 仅处理深度情况下的特征 (combined_gradient[2] + depth_features[8] = 10通道)
        self.depth_only_fusion_network = nn.Sequential(
            nn.Conv2d(10, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 边缘感知处理 - Sobel滤波器
        self.edge_detection = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # RGB + 深度 = 4通道
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),  # 3通道对应RGB
            nn.Sigmoid()
        )
        
        # 水体类型分类器 - 用于选择适当的衰减参数
        self.water_type_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 3),  # 3种水体类型
            nn.Softmax(dim=1)
        )
        
        # 水体类型衰减参数 - 3种水体类型，每种有3个RGB通道参数
        self.water_type_params = nn.Parameter(torch.Tensor([
            [1.2, 1.0, 0.9],  # 清澈水体
            [1.5, 1.3, 1.1],  # 浑浊水体
            [1.8, 1.6, 1.4]   # 高浑浊水体
        ]).reshape(3, 3))
        
        self.relu = nn.ReLU()
        
        print(f"初始化RGB引导衰减网络，scale: {self.scale}, sigmoid: {self.do_sigmoid}")
        
    def forward(self, depth, rgb=None):
        if rgb is None:
            # 如果没有提供RGB信息，退化为ImprovedAttenuateNetV3
            # 使用Sobel滤波器提取深度边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
            pad = nn.ReplicationPad2d(1)
            depth_padded = pad(depth)
            grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
            grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
            combined_gradient = torch.cat([grad_x, grad_y], dim=1)
            
            # 提取深度特征
            depth_features = self.depth_feature_extractor(depth)
            
            # 生成边缘感知调制因子（没有RGB时简化版本）
            edge_factor = torch.cat([combined_gradient, depth_features], dim=1)
            edge_factor = self.depth_only_fusion_network(edge_factor)
        else:
            # 提取RGB特征
            rgb_features = self.rgb_feature_extractor(rgb)
            
            # 提取深度特征
            depth_features = self.depth_feature_extractor(depth)
            
            # 融合RGB和深度特征
            fused_features = torch.cat([rgb_features, depth_features], dim=1)
            
            # 生成边缘感知调制因子
            combined_input = torch.cat([rgb, depth], dim=1)  # [B,4,H,W]
            edge_factor = self.edge_detection(combined_input)
            
            # 水体类型分类
            avg_rgb = torch.mean(rgb, dim=[2, 3])  # [B,3]
            water_type_weights = self.water_type_classifier(rgb)  # [B,3]
            
            # 根据水体类型选择衰减参数
            # [B,3,1,1] = 矩阵乘法([B,3] @ [3,3])并重新形状
            type_specific_params = torch.matmul(water_type_weights, self.water_type_params).unsqueeze(-1).unsqueeze(-1)
        
        # 基本衰减系数计算
        if self.do_sigmoid:
            beta_base = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))
        else:
            beta_base = torch.nn.functional.conv2d(depth, self.attenuation_conv_params)
            
        # 如果有RGB信息，应用水体类型特定参数
        if rgb is not None:
            # 应用水体类型特定参数 - 避免原地操作
            modified_beta = torch.zeros_like(beta_base)
            for i in range(3):
                modified_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * type_specific_params[:,i:i+1,:,:]
            beta_base = modified_beta
        
        # 应用通道特定权重（波长物理先验）
        weighted_beta = torch.zeros_like(beta_base)
        for i in range(3):
            weighted_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * self.channel_weights[i:i+1]
        
        # 应用边缘感知调制 - 在边缘处调整衰减强度
        # 边缘处增强通透性（减少衰减）
        modulated_beta = weighted_beta * (1.0 - 0.2 * edge_factor)
        beta_d_conv = torch.clamp(modulated_beta, 0.0)
        
        # 计算衰减图
        attenuation_map = torch.exp(-beta_d_conv)
        
        return attenuation_map

class MultiscaleBackscatterNet(nn.Module):
    """
    多尺度深度感知散射模型
    - 引入特征金字塔网络处理不同尺度的深度信息
    - 设计注意力模块突出关键深度特征
    - 加入深度置信度评估机制
    """
    def __init__(self, use_residual: bool = True, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.use_residual = use_residual
        
        # 基本参数
        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        
        # B_inf参数 - 无限远处的背景颜色
        self.B_inf = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1))
        
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        
        # 多尺度深度特征提取 - 特征金字塔网络
        # 下采样路径
        self.down_sample1 = nn.MaxPool2d(kernel_size=2)
        self.down_sample2 = nn.MaxPool2d(kernel_size=2)
        
        # 不同尺度的特征提取
        self.conv_scale1 = nn.Sequential(  # 原始尺度
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv_scale2 = nn.Sequential(  # 1/2尺度
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv_scale3 = nn.Sequential(  # 1/4尺度
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 上采样路径
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # 多尺度特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),  # 8+8+8=24通道
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 注意力模块 - 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 注意力模块 - 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 深度置信度评估
        self.confidence_estimation = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 边缘感知处理
        self.edge_process = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()
        
        print(f"初始化多尺度散射网络，scale: {self.scale}, sigmoid: {self.do_sigmoid}, residual: {self.use_residual}")
        
    def forward(self, depth, rgb=None):
        batch_size = depth.size(0)
        
        # 深度置信度评估
        confidence_map = self.confidence_estimation(depth)
        
        # 多尺度特征提取
        # 原始尺度
        features_scale1 = self.conv_scale1(depth)
        
        # 1/2尺度
        depth_down1 = self.down_sample1(depth)
        features_scale2 = self.conv_scale2(depth_down1)
        features_scale2_up = self.up_sample1(features_scale2)
        
        # 1/4尺度
        depth_down2 = self.down_sample2(depth_down1)
        features_scale3 = self.conv_scale3(depth_down2)
        features_scale3_up = self.up_sample2(features_scale3)
        
        # 特征融合 - 确保所有特征尺寸一致
        if features_scale1.size() != features_scale2_up.size():
            features_scale2_up = torch.nn.functional.interpolate(
                features_scale2_up, size=(features_scale1.size(2), features_scale1.size(3)), 
                mode='bilinear', align_corners=False
            )
        
        if features_scale1.size() != features_scale3_up.size():
            features_scale3_up = torch.nn.functional.interpolate(
                features_scale3_up, size=(features_scale1.size(2), features_scale1.size(3)), 
                mode='bilinear', align_corners=False
            )
        
        # 合并多尺度特征
        fused_features = torch.cat([features_scale1, features_scale2_up, features_scale3_up], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 通道注意力机制
        channel_weights = self.channel_attention(fused_features)
        # 修改：避免原地操作
        attended_features = fused_features * channel_weights
        
        # 空间注意力机制
        spatial_weights = self.spatial_attention(attended_features)
        # 修改：避免原地操作
        attended_features_spatial = attended_features * spatial_weights
        
        # 使用Sobel滤波器计算深度边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # 边缘感知处理
        edge_aware_factor = self.edge_process(combined_gradient)
        
        # 基础散射系数计算 - 引入多尺度特征
        if self.do_sigmoid:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params))
        else:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.backscatter_conv_params)
            
        # 应用深度置信度 - 低置信度区域减少依赖
        beta_with_conf = beta_b_conv * confidence_map
        
        # 应用边缘感知因子 - 边缘处减少散射
        beta_with_edge = beta_with_conf * (1.0 - 0.2 * edge_aware_factor)
        
        # 应用多尺度特征增强
        # 转换特征形状以适应操作
        feature_weight = torch.mean(attended_features_spatial, dim=1, keepdim=True)
        feature_weight_resized = torch.nn.functional.interpolate(
            feature_weight, size=(beta_with_edge.size(2), beta_with_edge.size(3)), 
            mode='bilinear', align_corners=False
        )
        
        # 特征权重归一化 - 避免原地操作
        feature_min = feature_weight_resized.min()
        feature_max = feature_weight_resized.max()
        normalized_feature_weight = (feature_weight_resized - feature_min) / (feature_max - feature_min + 1e-8)
        
        # 应用特征增强 - 特征丰富区域增强散射效果 - 避免原地操作
        enhanced_beta = beta_with_edge * (1.0 + 0.1 * normalized_feature_weight)
        beta_b_conv_clamped = torch.clamp(enhanced_beta, 0.001, None)  # 设置最小值
        
        # 计算散射
        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv_clamped))
        
        # 残差项处理
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params))
            else:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.residual_conv_params)
            
            # 应用置信度和边缘信息 - 避免原地操作
            modified_beta_d = beta_d_conv * confidence_map * (1.0 - 0.1 * edge_aware_factor)
            beta_d_clamped = torch.clamp(modified_beta_d, 0.0)
            
            # 应用残差 - 避免原地操作
            residual_term = torch.sigmoid(self.J_prime) * torch.exp(-beta_d_clamped)
            Bc_with_residual = Bc + residual_term
            Bc_final = torch.clamp(Bc_with_residual, 0.01, 1.0)
        else:
            # 没有残差项，直接裁剪
            Bc_final = torch.clamp(Bc, 0.01, 1.0)
        
        return Bc_final
    
    def forward_rgb(self, rgb):
        """从RGB图像估计大气光B_inf"""
        from render_uw import estimate_atmospheric_light
        
        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))
            
        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)
        
        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())


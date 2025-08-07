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
            Bc = Bc + residual_term  # avoid in-place operation
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
            Bc = Bc + residual_term  # avoid in-place operation
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
        
        # basic parameters
        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            
        # edge-aware processing
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
        # use Sobel filters instead of diff()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # generate edge-aware factor
        edge_aware_factor = self.edge_process(combined_gradient)
        # resize to match original depth map
        edge_aware_factor = torch.nn.functional.interpolate(
            edge_aware_factor, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False
        )
        
        # basic scattering coefficient calculation
        if self.do_sigmoid:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params))
        else:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.backscatter_conv_params)
            
        # apply edge-aware factor to modulate scattering coefficients - reduce scattering at edges
        beta_b_conv = beta_b_conv * (1.0 - 0.1 * edge_aware_factor)
        beta_b_conv = torch.clamp(beta_b_conv, 0.01, None)  # set minimum value 0.01
        
        # calculate scattering
        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        
        # residual term processing
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params))
            else:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.residual_conv_params)
            
            # apply confidence and edge info - avoid in-place operation
            modified_beta_d = beta_d_conv * (1.0 - 0.1 * edge_aware_factor)
            beta_d_clamped = torch.clamp(modified_beta_d, 0.0)
            
            # apply residual - avoid in-place operation
            residual_term = torch.sigmoid(self.J_prime) * torch.exp(-beta_d_clamped)
            Bc_with_residual = Bc + residual_term
            Bc_final = torch.clamp(Bc_with_residual, 0.01, 1.0)
        else:
            # no residual term, clip directly
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
        
        # basic attenuation parameters
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
            
        # wavelength physical prior
        self.channel_weights = nn.Parameter(torch.Tensor([1.0, 0.8, 0.6]).reshape(3, 1, 1))
        
        # edge-aware processing
        self.edge_modulation = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),  # output 3 channels for RGB
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, depth, rgb=None):
        # use Sobel filter instead of diff()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # generate edge-aware modulation factor
        edge_factor = self.edge_modulation(combined_gradient)  # [B,3,H,W-1]
        # resize to match original depth map
        edge_factor = torch.nn.functional.interpolate(
            edge_factor, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False
        )
        
        # basic attenuation coefficient calculation
        if self.do_sigmoid:
            beta_base = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))
        else:
            beta_base = torch.nn.functional.conv2d(depth, self.attenuation_conv_params)
        
        if rgb is not None:
            # apply water type specific parameters - avoid in-place operation
            modified_beta = torch.zeros_like(beta_base)
            for i in range(3):
                modified_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * type_specific_params[:,i:i+1,:,:]
            beta_base = modified_beta
        
        # apply channel specific weights (wavelength physical prior)
        weighted_beta = torch.zeros_like(beta_base)
        for i in range(3):
            weighted_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * self.channel_weights[i:i+1]
        
        # apply edge-aware modulation - adjust attenuation intensity at edges
        # enhance transparency at edges (reduce attenuation)
        modulated_beta = weighted_beta * (1.0 - 0.2 * edge_factor)
        beta_d_conv = torch.clamp(modulated_beta, 0.0)
        
        # calculate attenuation map
        attenuation_map = torch.exp(-beta_d_conv)
        
        return attenuation_map

class RGBGuidedAttenuateNet(nn.Module):
    
    # RGB-guided attenuation optimization model
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        
        # basic attenuation parameters for RGB three channels
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
            
        # wavelength physical prior - different absorption rates for different wavelengths
        self.channel_weights = nn.Parameter(torch.Tensor([1.0, 0.8, 0.6]).reshape(3, 1, 1))
        
        # RGB feature extraction network
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # depth feature extraction network
        self.depth_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # feature fusion network - handles RGB+depth features
        self.fusion_network = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # feature fusion network - depth-only features (combined_gradient[2] + depth_features[8] = 10 channels)
        self.depth_only_fusion_network = nn.Sequential(
            nn.Conv2d(10, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # edge-aware processing - Sobel filters
        self.edge_detection = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # RGB + depth = 4 channels
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),  # 3 channels for RGB
            nn.Sigmoid()
        )
        
        # water type classifier - selects appropriate attenuation parameters
        self.water_type_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 3),  # 3 water types
            nn.Softmax(dim=1)
        )
        
        # water type attenuation parameters - 3 water types, each with 3 RGB channel parameters
        self.water_type_params = nn.Parameter(torch.Tensor([
            [1.2, 1.0, 0.9],  # clear water
            [1.5, 1.3, 1.1],  # turbid water
            [1.8, 1.6, 1.4]   # highly turbid water
        ]).reshape(3, 3))
        
        self.relu = nn.ReLU()
        
        print(f"Initialized RGB-guided attenuation network, scale: {self.scale}, sigmoid: {self.do_sigmoid}")
        
    def forward(self, depth, rgb=None):
        if rgb is None:
            # if no RGB info provided, fallback to ImprovedAttenuateNetV3
            # use Sobel filter to extract depth edges
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
            pad = nn.ReplicationPad2d(1)
            depth_padded = pad(depth)
            grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
            grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
            combined_gradient = torch.cat([grad_x, grad_y], dim=1)
            
            # extract depth features
            depth_features = self.depth_feature_extractor(depth)
            
            # generate edge-aware modulation factor (simplified version without RGB)
            edge_factor = torch.cat([combined_gradient, depth_features], dim=1)
            edge_factor = self.depth_only_fusion_network(edge_factor)
        else:
            # extract RGB features
            rgb_features = self.rgb_feature_extractor(rgb)
            
            # extract depth features
            depth_features = self.depth_feature_extractor(depth)
            
            # fuse RGB and depth features
            fused_features = torch.cat([rgb_features, depth_features], dim=1)
            
            # generate edge-aware modulation factor
            combined_input = torch.cat([rgb, depth], dim=1)  # [B,4,H,W]
            edge_factor = self.edge_detection(combined_input)
            
            # water type classification
            avg_rgb = torch.mean(rgb, dim=[2, 3])  # [B,3]
            water_type_weights = self.water_type_classifier(rgb)  # [B,3]
            
            # select attenuation parameters based on water type
            # [B,3,1,1] = matrix multiplication ([B,3] @ [3,3]) and reshape
            type_specific_params = torch.matmul(water_type_weights, self.water_type_params).unsqueeze(-1).unsqueeze(-1)
        
        # basic attenuation coefficient calculation
        if self.do_sigmoid:
            beta_base = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))
        else:
            beta_base = torch.nn.functional.conv2d(depth, self.attenuation_conv_params)
            
        # if RGB info available, apply water type specific parameters
        if rgb is not None:
            # apply water type specific parameters - avoid in-place operation
            modified_beta = torch.zeros_like(beta_base)
            for i in range(3):
                modified_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * type_specific_params[:,i:i+1,:,:]
            beta_base = modified_beta
        
        # apply channel specific weights (wavelength physical prior)
        weighted_beta = torch.zeros_like(beta_base)
        for i in range(3):
            weighted_beta[:,i:i+1,:,:] = beta_base[:,i:i+1,:,:] * self.channel_weights[i:i+1]
        
        # apply edge-aware modulation - adjust attenuation intensity at edges
        # enhance transparency at edges (reduce attenuation)
        modulated_beta = weighted_beta * (1.0 - 0.2 * edge_factor)
        beta_d_conv = torch.clamp(modulated_beta, 0.0)
        
        # calculate attenuation map
        attenuation_map = torch.exp(-beta_d_conv)
        
        return attenuation_map

class MultiscaleBackscatterNet(nn.Module):
    
    # Multiscale depth-aware backscatter model
    def __init__(self, use_residual: bool = True, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()
        
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.use_residual = use_residual
        
        # basic parameters
        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        
        # B_inf parameters - background color at infinite distance
        self.B_inf = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1))
        
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        
        # multiscale depth feature extraction - feature pyramid network
        # downsampling path
        self.down_sample1 = nn.MaxPool2d(kernel_size=2)
        self.down_sample2 = nn.MaxPool2d(kernel_size=2)
        
        # feature extraction at different scales
        self.conv_scale1 = nn.Sequential(  # original scale
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv_scale2 = nn.Sequential(  # 1/2 scale
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv_scale3 = nn.Sequential(  # 1/4 scale
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # upsampling path
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # multiscale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),  # 8+8+8=24 channels
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # attention module - channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # attention module - spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # depth confidence estimation
        self.confidence_estimation = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # edge-aware processing
        self.edge_process = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()
        
        print(f"Initialized multiscale backscatter network, scale: {self.scale}, sigmoid: {self.do_sigmoid}, residual: {self.use_residual}")
        
    def forward(self, depth, rgb=None):
        batch_size = depth.size(0)
        
        # depth confidence estimation
        confidence_map = self.confidence_estimation(depth)
        
        # multiscale feature extraction
        # original scale
        features_scale1 = self.conv_scale1(depth)
        
        # 1/2 scale
        depth_down1 = self.down_sample1(depth)
        features_scale2 = self.conv_scale2(depth_down1)
        features_scale2_up = self.up_sample1(features_scale2)
        
        # 1/4 scale
        depth_down2 = self.down_sample2(depth_down1)
        features_scale3 = self.conv_scale3(depth_down2)
        features_scale3_up = self.up_sample2(features_scale3)
        
        # feature fusion - ensure all features have consistent dimensions
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
        
        # merge multiscale features
        fused_features = torch.cat([features_scale1, features_scale2_up, features_scale3_up], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # channel attention mechanism
        channel_weights = self.channel_attention(fused_features)
        # modification: avoid in-place operation
        attended_features = fused_features * channel_weights
        
        # spatial attention mechanism
        spatial_weights = self.spatial_attention(attended_features)
        # modification: avoid in-place operation
        attended_features_spatial = attended_features * spatial_weights
        
        # use Sobel filter to calculate depth edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).cuda()
        pad = nn.ReplicationPad2d(1)
        depth_padded = pad(depth)
        grad_x = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_x))
        grad_y = torch.abs(torch.nn.functional.conv2d(depth_padded, sobel_y))
        combined_gradient = torch.cat([grad_x, grad_y], dim=1)
        
        # edge-aware processing
        edge_aware_factor = self.edge_process(combined_gradient)
        
        # basic scattering coefficient calculation - introduce multiscale features
        if self.do_sigmoid:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params))
        else:
            beta_b_conv = torch.nn.functional.conv2d(depth, self.backscatter_conv_params)
            
        # apply depth confidence - reduce dependency in low confidence regions
        beta_with_conf = beta_b_conv * confidence_map
        
        # apply edge-aware factor - reduce scattering at edges
        beta_with_edge = beta_with_conf * (1.0 - 0.2 * edge_aware_factor)
        
        # apply multiscale feature enhancement
        # convert feature shape for operation
        feature_weight = torch.mean(attended_features_spatial, dim=1, keepdim=True)
        feature_weight_resized = torch.nn.functional.interpolate(
            feature_weight, size=(beta_with_edge.size(2), beta_with_edge.size(3)), 
            mode='bilinear', align_corners=False
        )
        
        # feature weight normalization - avoid in-place operation
        feature_min = feature_weight_resized.min()
        feature_max = feature_weight_resized.max()
        normalized_feature_weight = (feature_weight_resized - feature_min) / (feature_max - feature_min + 1e-8)
        
        # apply feature enhancement - enhance scattering in feature-rich regions - avoid in-place operation
        enhanced_beta = beta_with_edge * (1.0 + 0.1 * normalized_feature_weight)
        beta_b_conv_clamped = torch.clamp(enhanced_beta, 0.001, None)  # set minimum value
        
        # calculate scattering
        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv_clamped))
        
        # residual term processing
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params))
            else:
                beta_d_conv = torch.nn.functional.conv2d(depth, self.residual_conv_params)
            
            # apply confidence and edge info - avoid in-place operation
            modified_beta_d = beta_d_conv * confidence_map * (1.0 - 0.1 * edge_aware_factor)
            beta_d_clamped = torch.clamp(modified_beta_d, 0.0)
            
            # apply residual - avoid in-place operation
            residual_term = torch.sigmoid(self.J_prime) * torch.exp(-beta_d_clamped)
            Bc_with_residual = Bc + residual_term
            Bc_final = torch.clamp(Bc_with_residual, 0.01, 1.0)
        else:
            # no residual term, clip directly
            Bc_final = torch.clamp(Bc, 0.01, 1.0)
        
        return Bc_final
    
    def forward_rgb(self, rgb):
        """Estimate atmospheric light B_inf from RGB image"""
        from render_uw import estimate_atmospheric_light
        
        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))
            
        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)
        
        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())


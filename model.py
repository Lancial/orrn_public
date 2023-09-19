import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torchdiffeq import odeint
from monai.networks.blocks import Warp
from utils import *


'''
Some codes are adopted from RRN: https://github.com/Novestars/Recursive_Refinement_Network.git
'''

class ODENet(nn.Module):
    
    def __init__(self, ndims=3, mode='4d', backward=False, scaling=True, global_context=True, recursive=False, step_size=0.2):
        super().__init__()    
        self.ndims=ndims
        self.scaling=scaling
        if mode == '4d':
            self.ts = [i for i in range(0, 6)]
            self.step_size = step_size
        elif mode == '3d':
            self.ts = [0, 1]
            self.step_size=step_size
        self.mode=mode
        self.backward = backward
        self.global_context = global_context
        self.encoder = GRUEncoder_DET(ndims=ndims, num_levels=2, encode_t=False, global_context=global_context)
        self.velocities = []

        self.odefunc = IterRefineODE(feature_channels=32,
            step_size=self.step_size,
            scaling=self.scaling,
            global_context=global_context,
            recursive=recursive)

    def set_step_size(self, s):
        self.step_size = s

    def forward(self, x, ts=None, trajectory=False):
        if self.mode == '3d':
            assert(x.shape[1]==2)
        features = []
        h = None
        for i in range(x.shape[1]):
            img_t = x[:, i:i+1]
            (_, x2), h = self.encoder(h, img_t)
            if ts != None and i not in ts:
                features.append(None)
            else:
                features.append(x2)
        B, _, H, W, D = x2.shape
        flows= [torch.zeros(B, self.ndims, H, W, D).cuda()]

        if ts is None:
            ts = self.ts
        for idx_t, t in enumerate(ts[1:]):
            if not self.backward:
                feature_x = features[0]
                feature_y = features[t]
            else:
                feature_x = features[t]
                feature_y = features[0]
            if self.global_context:
                ode_in =  torch.cat([flows[-1], feature_x, feature_y, h, torch.ones(B, 1, H, W, D).cuda()*t], 1)
            else:
                ode_in =  torch.cat([flows[-1], feature_x, feature_y, torch.ones(B, 1, H, W, D).cuda()*t], 1)

            if trajectory == False:
  
                ode_out_t = odeint(self.odefunc, 
                                    ode_in, 
                                    torch.tensor([ts[idx_t], t]).float().cuda(), 
                                    method='euler', 
                                    options={'step_size': self.step_size})[-1]


                flows.append(ode_out_t[:, :3])
            else:
                # torch.from_numpy(np.linspace(ts[idx_t], t, 11)).float()
                ode_out_t = odeint(self.odefunc, 
                                    ode_in, 
                                    torch.from_numpy(np.linspace(ts[idx_t], t, 11)).float().cuda(), 
                                    method='euler', 
                                    options={'step_size': self.step_size})
                
                for i in range(1, ode_out_t.shape[0]):
                    flows.append(ode_out_t[i, :, :3])
        return flows
    
    def predict(self, x):
        flows = self.forward(x)
        velocities_forward = self.odefunc.vel_record
        velocities_backward = [-v for v in velocities_forward[::-1]]
        
        phi_backward= torch.zeros_like(flows[-1]).cuda()
        for v in velocities_backward:
            v_dt = v * self.step_size
            v_dt = self.odefunc.vecint(v_dt)
            phi_backward = self.odefunc.stn(phi_backward, v_dt) + v_dt
            # v_dt = self.odefunc.stn(phi_backward, v) + v - phi_backward
            # phi_backward += v_dt * self.step_size 
        return flows, phi_backward, velocities_forward


class IterRefineODE(nn.Module):

    def __init__(self, feature_channels, ndims=3, step_size=0.2, scaling=True, global_context=True, recursive=False):
        super().__init__()
        self.ndims = ndims
        # TODO: this is hardcoded, if input dimension is different, then modify
        self.stn = STN(size=(48, 40, 48))
        self.vel_record = []
        self.step_size=step_size
        self.scaling = scaling
        self.global_context = global_context
        self.flow_layer = self._build_flow_layer(feature_channels)
        self.recursive = recursive

    def forward(self, t, x, vel=True):
        B, C, H, W, D = x.shape

        if self.global_context:
            disp, feature_x, feature_y, ctx, t_end = x[:, :3], x[:, 3:35], x[:, 35:67], x[:, 67:99], x[:, -1:]
        else:
            disp, feature_x, feature_y, t_end = x[:, :3], x[:, 3:35], x[:, 35:67], x[:, -1:]

        t_end = t_end.mean(dim=[0, 2, 3, 4])

        target_feature = feature_y

        warped_x = self.stn(feature_x, disp)
        warped_feature_x_normalized, feature_target_normalized = normalize_features(
            [warped_x, target_feature],
            normalize=True,
            center=True,
            moments_across_channels=True,
            moments_across_images=True)

        cost_volume = compute_cost_volume(feature_target_normalized, warped_feature_x_normalized, max_displacement=2)
        cost_volume = F.leaky_relu(cost_volume, negative_slope=0.05)

        if self.global_context:
            x_in = torch.cat([cost_volume, target_feature, warped_x, ctx, disp], dim=1)
        else:
            x_in = torch.cat([cost_volume, target_feature, warped_x, disp], dim=1)

        for layer in self.flow_layer[:-1]:
            x_out = layer(x_in)
            x_in = torch.cat([x_in, x_out], dim=1)          
        context = x_out
        velocity = self.flow_layer[-1](context)

        velocity = self.stn(disp, velocity) + velocity - disp
        
        if self.scaling:
            velocity = velocity / (t_end - t)
        if self.recursive:   
            velocity /= self.step_size
        return torch.cat([velocity, torch.zeros(B, C-3, H, W, D).cuda()], 1)
    

    def _build_flow_layer(self, feature_channels):
        block_layers = [64, 48, 32, 16]
        layers = nn.ModuleList()
        if self.global_context:
            last_in_channels = (125+feature_channels *3 +3)
        else:
            last_in_channels = (125+feature_channels *2 +3)
        for c in block_layers:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=last_in_channels,
                        out_channels=c,
                        kernel_size=(3, 3, 3),
                        padding=1),
                    nn.LeakyReLU(
                        negative_slope=0.02)
                ))
            last_in_channels += c
        layers.append(
            nn.Conv3d(
                in_channels=block_layers[-1],
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding=1))
        return layers



class IterRefineODEMS(nn.Module):

    def __init__(self, feature_channels, ndims=3, step_size=0.2, scaling=True, global_context=True):
        super().__init__()
        self.ndims = ndims
        self.step_size=step_size
        self.scaling = scaling
        self.global_context = global_context
        self.flow_layer = self._build_flow_layer(feature_channels)
        self.flow_layer2 = self._build_flow_layer2(feature_channels//2)

        # TODO: this is hardcoded, if input dimension is different, then modify
        self.stn = STN(size=(48, 40, 48)) 
        self.stn2 = STN(size=(96, 80, 96))
        

    def forward(self, t, x):

        disp_up, feature_x, feature_y, feature_x_up, feature_y_up, ctx = x

        disp = ResizeTransform(disp_up, factor=0.5)
        target_feature = feature_y
        warped_x = self.stn(feature_x, disp)
        warped_feature_x_normalized, feature_target_normalized = normalize_features(
            [warped_x, target_feature],
            normalize=True,
            center=True,
            moments_across_channels=True,
            moments_across_images=True)

        cost_volume = compute_cost_volume(feature_target_normalized, warped_feature_x_normalized, max_displacement=2)
        cost_volume = F.leaky_relu(cost_volume, negative_slope=0.05)
        x_in = torch.cat([cost_volume, target_feature, warped_x, ctx, disp], dim=1)
        for layer in self.flow_layer[:-1]:
            x_out = layer(x_in)
            x_in = torch.cat([x_in, x_out], dim=1)          
        context = x_out
        velocity = self.flow_layer[-1](context)


        up_context = ResizeImage(context, factor=2)
        velocity_up = ResizeTransform(velocity, factor=2)
        disp_up2 = self.stn2(disp_up, velocity_up) + velocity_up
        warped_x_up = self.stn2(feature_x_up, disp_up2)
        target_feature_up = feature_y_up
        warped_feature_x_up_normalized, feature_target_up_normalized = normalize_features(
            [warped_x_up, target_feature_up],
            normalize=True,
            center=True,
            moments_across_channels=True,
            moments_across_images=True)
        cost_volume = compute_cost_volume(feature_target_up_normalized, warped_feature_x_up_normalized, max_displacement=1)
        cost_volume = F.leaky_relu(cost_volume, negative_slope=0.05)
        x_in = torch.cat([cost_volume, target_feature_up, warped_x_up, up_context, disp_up2], dim=1)
        for layer in self.flow_layer2[:-1]:
            x_out = layer(x_in)
            x_in = torch.cat([x_in, x_out], dim=1)          
        context = x_out
        velocity_up2 = self.flow_layer2[-1](context)

        velocity_final = self.stn2(disp_up2, velocity_up2) + velocity_up2 - disp_up
        if self.scaling:
            velocity_final = velocity_final / (1 - t)
        
        return (velocity_final, torch.zeros_like(feature_x).cuda(), torch.zeros_like(feature_y).cuda(),
               torch.zeros_like(feature_x_up).cuda(), torch.zeros_like(feature_y_up).cuda(), torch.zeros_like(ctx).cuda())
    
    def _build_flow_layer2(self, feature_channels):
        block_layers = [32, 16]
        layers = nn.ModuleList()
        last_in_channels = (27 + feature_channels * 3 + 3)

        for c in block_layers:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=last_in_channels,
                        out_channels=c,
                        kernel_size=(3, 3, 3),
                        padding=1),
                    nn.LeakyReLU(
                        negative_slope=0.02)
                ))
            last_in_channels += c
        layers.append(
            nn.Conv3d(
                in_channels=block_layers[-1],
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding=1))
        return layers

    def _build_flow_layer(self, feature_channels):
        block_layers = [64, 48, 32, 16]
        layers = nn.ModuleList()
        if self.global_context:
            last_in_channels = (125+feature_channels *3 +3)
        else:
            last_in_channels = (125+feature_channels *2 +3)
        for c in block_layers:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=last_in_channels,
                        out_channels=c,
                        kernel_size=(3, 3, 3),
                        padding=1),
                    nn.LeakyReLU(
                        negative_slope=0.02)
                ))
            last_in_channels += c
        layers.append(
            nn.Conv3d(
                in_channels=block_layers[-1],
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding=1))
        return layers


class ConvGRU_DET(nn.Module):
    def __init__(self, latent_dim, input_dim, ndims=3):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.latent_dim=latent_dim
        self.input_dim=input_dim
        self.update_gate = nn.Sequential(Conv(input_dim + latent_dim , latent_dim, kernel_size=3, padding=1),
                                         nn.Sigmoid())
        
        
        self.reset_gate = nn.Sequential(
               Conv(input_dim + latent_dim, latent_dim, kernel_size=3, padding=1),
               nn.Sigmoid())
        
        self.new_state_net = nn.Sequential(Conv(latent_dim+ input_dim, latent_dim, kernel_size=3, padding=1), 
                                           nn.Tanh())
        init_network_weights(net=self.update_gate, module=Conv)
        init_network_weights(net=self.reset_gate, module=Conv)
        init_network_weights(net=self.new_state_net, module=Conv)


    def forward(self, h, x):
        B = x.shape[0]
        if h is None:
            h = torch.zeros(B, self.latent_dim, *x.shape[2:]).cuda()

        y_concat = torch.cat([h, x], 1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        concat = torch.cat([reset_gate*h, x], 1)
        new_h = self.new_state_net(concat)
        new_h = (1-update_gate) * new_h + update_gate * new_h
 
        return new_h
    
class CostVRefinementBlock(nn.Module):
    
    def __init__(self, feature_channels, ndims=3):
        super().__init__()
        self.ndims = ndims
        self.flow_layer = self._build_flow_layer(feature_channels)
        self.stn = Warp()
        
    def forward(self, moving, fixed, flow, context):
        shape = moving.shape[-3:]
        up_context = ResizeImage(context, shape, self.ndims)
        warped_moving = self.stn(moving, flow)
        
        warped_moving_normalized, fixed_normalized = normalize_features(
            [warped_moving, fixed],
            normalize=True,
            center=True,
            moments_across_channels=True,
            moments_across_images=True)
        
        cost_volume = compute_cost_volume(fixed_normalized, warped_moving_normalized, max_displacement=2)
        cost_volume = F.leaky_relu(cost_volume, negative_slope=0.05)
        
        x_in = torch.cat([up_context, flow, cost_volume, fixed], dim=1)
        
        for layer in self.flow_layer[:-1]:
            x_out = layer(x_in)
            x_in = torch.cat([x_in, x_out], dim=1)          
        context = x_out
        refinement = self.flow_layer[-1](context)
        return context, flow + refinement
        
    def _build_flow_layer(self, feature_channels):
        block_layers = [64, 48, 32, 16]
        layers = nn.ModuleList()
        last_in_channels = (125+feature_channels)
        last_in_channels += 3 + 32

        for c in block_layers:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=last_in_channels,
                        out_channels=c,
                        kernel_size=(3, 3, 3),
                        padding=1),
                    nn.LeakyReLU(
                        negative_slope=0.02)
                ))
            last_in_channels += c
        layers.append(
            nn.Conv3d(
                in_channels=block_layers[-1],
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding=1))
        return layers
        

class FeaturePyramid(nn.Module):
    def __init__(self,leaky_relu_alpha=0.1,
               filters=None,
               original_layer_sizes=True,
               num_levels=3,
               channel_multiplier=1.,
               pyramid_resolution='half',
               num_channels=1):
        super().__init__()

        self._channel_multiplier = channel_multiplier
        if filters is None:
            filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                           (3, 196))[:num_levels]

        assert filters
        assert all(len(t) == 2 for t in filters)
        assert all(t[0] > 0 for t in filters)

        self._leaky_relu_alpha = leaky_relu_alpha
        self._convs = nn.ModuleList()

        c = num_channels

        for level, (num_layers, num_filters) in enumerate(filters):
            group = nn.ModuleList()
            for i in range(num_layers):
                stride = 1
                if i == 0 or (i == 1 and level == 0 and pyramid_resolution == 'quarter'):
                    stride = 2
                conv = nn.Conv3d(
                    in_channels=c,
                    out_channels=int(num_filters * self._channel_multiplier),
                    kernel_size=3,
                    stride=stride,
                    padding=0)
                group.append(conv)
                c = int(num_filters * self._channel_multiplier)
            self._convs.append(group)

    def forward(self, x):
        x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
        features = []
        for level, conv_tuple in enumerate(self._convs):
            for i, conv in enumerate(conv_tuple):
                if level > 0 or i < len(conv_tuple):
                    x = func.pad(x, pad=[1, 1, 1, 1,1,1], mode='constant', value=0)
                x = conv(x)
                x = func.leaky_relu(x, negative_slope=self._leaky_relu_alpha)
            features.append(x)

        return features

    def forward2(self, x):
        x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
        features = []
        for level, conv_tuple in enumerate(self._convs):
            for i, conv in enumerate(conv_tuple):
                if level > 0 or i < len(conv_tuple):
                    x = func.pad(x, pad=[1, 1, 1, 1,1,1], mode='constant', value=0)
                x = conv(x)
                x = func.leaky_relu(x, negative_slope=self._leaky_relu_alpha)
            features.append(x)
            if level == 1:
                x = x.detach.clone()
        
        return features

    
   
class GRUEncoder_DET(nn.Module):
    def __init__(self, num_features=[16, 32, 64], input_dim=1, num_levels=2,ndims=3, encode_t=True, global_context=True):
        super().__init__()
        layers = []
        self.encode_t=encode_t
        self.latent_dim = 32
        self.global_context = global_context
        self._pyramid = FeaturePyramid(num_levels=num_levels)
        if self.global_context:
            if encode_t:
                self.conv4 = ConvGRU_DET(latent_dim=self.latent_dim, input_dim=num_features[num_levels-1]+1, ndims=ndims)
            else:
                self.conv4 = ConvGRU_DET(latent_dim=self.latent_dim, input_dim=num_features[num_levels-1], ndims=ndims)
          
    def forward(self, h, x):
        xs = self._pyramid(x)
        if self.global_context:
            h = self.conv4(h, xs[-1])
        return xs, h

class Feature_pyramid_wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self._pyramid = FeaturePyramid(num_levels=2)

    def forward(self, x):
        x1, x2 = self._pyramid(x)
        return x1, x2



def state_dict_descend(state_dict, prefix):
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

def load_matched_state_dict(model, state_dicts, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        match = False
        for state_dict in state_dicts:
            if key == state_dict and curr_state_dict[key].shape == state_dicts[state_dict].shape:
                curr_state_dict[key] = state_dicts[key]
                num_matched += 1
                match=True
        if not match:
            print(key, 'unmatched')

    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')
        

def init_network_weights(net, module, std=0.1):
    for m in net.modules():
        if isinstance(m, module):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)
            
def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def compute_cost_volume(features1, features2, max_displacement):
    # Set maximum displacement and compute the number of image shifts.
    _, _, height, width, depth = features1.shape
    if max_displacement <= 0 or max_displacement >= height:
        raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed to compute the
    # cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = torch.nn.functional.pad(
        input=features2,
        pad=[max_disp, max_disp, max_disp, max_disp, max_disp, max_disp],
        mode='constant')
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            for k in range(num_shifts):
                prod = features1 * features2_padded[:, :, i:(height + i), j:(width + j),k:(depth+k)]
                corr = torch.mean(prod, dim=1, keepdim=True)
                cost_list.append(corr)
    cost_volume = torch.cat(cost_list, dim=1)
    return cost_volume


def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
    # Compute feature statistics.

    dim = [1, 2, 3, 4] if moments_across_channels else [2, 3, 4]

    means = []
    stds = []

    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=dim, keepdim=True)
        std = torch.std(feature_image, dim=dim, keepdim=True)
        means.append(mean)
        stds.append(std)

    if moments_across_images:
        means = [torch.mean(torch.stack(means), dim=0, keepdim=False)] * len(means)
        stds = [torch.mean(torch.stack(stds), dim=0, keepdim=False)] * len(stds)

    # Center and normalize features.
    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, means)
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, stds)]

    return feature_list
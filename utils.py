import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.cm as cm
from torchvision.utils import make_grid
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def visualize_3d_scalar_image(log_writer, step, name, data):
    """
        data: list of tensor [(1, 1, H, W, D) ...]
    """
    data = torch.cat(data, 0).cpu().detach()
    T, C, H, W, D = data.shape
    log_writer.add_image(name+'_axis1', make_grid(data[:, :, H//4, :, :]), step)
    log_writer.add_image(name+'_axis2', make_grid(data[:, :, :, W//2, :]), step)
    log_writer.add_image(name+'_axis3', make_grid(data[:, :, :, :, D//2]), step)

def visualize_3d_ground_truth(log_writer, step, imgs, ref_indices):
    """
        gt: tensor (1, 10, H, W, D)
    """
    T, C, H, W, D = imgs.shape
    gt = imgs.permute(1, 0, 2, 3, 4).cpu().detach()
    to_vis = gt[:, :, H//4, :, :]
    to_vis[ref_indices, :, :, :] += 0.25
    to_vis = torch.clamp(to_vis, min=0, max=1)
    log_writer.add_image('gt_axis1', make_grid(to_vis), step)
    to_vis = gt[:, :, :, W//2, :]
    to_vis[ref_indices, :, :, :] += 0.25
    to_vis = torch.clamp(to_vis, min=0, max=1)
    log_writer.add_image('gt_axis2', make_grid(to_vis), step)
    to_vis = gt[:, :, :, :, D//2]
    to_vis[ref_indices, :, :, :] += 0.25
    to_vis = torch.clamp(to_vis, min=0, max=1)
    log_writer.add_image('gt_axis3', make_grid(to_vis), step)

def plot_flow(slices_in,           # the 2D slices
         titles=None,         # list of titles
         cmaps=None,          # list of colormaps
         width=15,            # width in in
         indexing='ij',       # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,   # whether to match the image view, i.e. flip y axis
         grid=False,          # option to plot the images in a grid or a single row
         show=False,           # option to actually show the plot (plt.show())
         quiver_width=None,
         plot_block=True,  # option to plt.show()
         scale=1):            # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    slices_in = np.copy(slices_in)  # Since img_indexing, indexing may modify slices_in in memory

    if indexing == 'ij':
        for si, slc in enumerate(slices_in):
            # Make y values negative so y-axis will point down in plot
            slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(v, u,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show(block=plot_block)

    return (fig, axs)


def visualize_flow_with_arrows(flow, save=False, dir_path='.'):
    B, C, W, H = flow.shape
    flows = []
    for i in range(B):
        fig, _ = plot_flow([flow[i].numpy().transpose(1, 2, 0)[::6,::6]], width=4, show=False, scale=2)
        if save:
            fig.savefig(os.path.join(dir_path, '0' * (3-len(str(i))) + str(i) + '.png'))
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        flows.append(torch.from_numpy(image_from_plot).float().permute(2, 0, 1) / 255)
    return torch.stack(flows)


def visualize_3d_flow(log_writer, step, name, data):
    """
        data: list of tensor [(1, 3, H, W, D) ...]
    """
    data = torch.cat(data, 0).cpu().detach()
    T, C, H, W, D = data.shape
    log_writer.add_image(name+'_axis1', make_grid(visualize_flow_with_arrows(data[:, [1, 2], H//4, :, :])), step)
    log_writer.add_image(name+'_axis2', make_grid(visualize_flow_with_arrows(data[:, [0, 2], :, W//2, :])), step)
    log_writer.add_image(name+'_axis3', make_grid(visualize_flow_with_arrows(data[:, [0, 1],:, :, D//2])), step)


def compute_trajectories_irregularity(trajectories, step_size):
    velocities = (trajectories[1:] - trajectories[:-1]) / step_size
    acceleration = (velocities[1:] - velocities[:-1]) / step_size
    return torch.sqrt((acceleration * acceleration).sum(2)).mean()


def unit_flow_to_monai_flow(flow):
    _, _, x, y, z = flow.shape
    flow[:, 0, :, :, :] = flow[:, 0, :, :, :] * (z-1)/2
    flow[:, 1, :, :, :] = flow[:, 1, :, :, :] * (y-1)/2
    flow[:, 2, :, :, :] = flow[:, 2, :, :, :] * (x-1)/2
    return flow[:, [2, 1, 0]]



# from LapIRN
def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

# from LapIRN
def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

# from LapIRN
def neg_Jdet_percentage(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = (F.relu(neg_Jdet) > 0).float()

    return torch.mean(selected_neg_Jdet)

# flow must be formulated the same as in LapIRN
def compute_neg_Jdet(flows, grid):
    return [neg_Jdet_percentage(flow, grid).item() for flow in flows]


def ResizeImage(img, target_shape=None,factor=None,  ndims=3):
    if factor is not None and target_shape is None:
        H, W, D = img.shape[-3:] 
        target_shape = (H*factor, W* factor, D*factor)
    if ndims==3:
        img_rs = torch.nn.functional.interpolate(img, size=(target_shape[0], target_shape[1], target_shape[2]),
                                                mode='trilinear',
                                                align_corners=True)
    elif ndims==2:
        img_rs = torch.nn.functional.interpolate(img, size=(target_shape[0], target_shape[1]),
                                                mode='bilinear',
                                                align_corners=True)
        
    return img_rs

def ResizeTransform(flow, target_shape=None, factor=None, ndims=3):
    device = flow.device
    if factor is not None and target_shape is None:
        H, W, D = flow.shape[-3:] 
        target_shape = (int(H*factor), int(W* factor), int(D*factor))
    if ndims==3:
        _, c, h, w, d = flow.shape
        ratio = torch.FloatTensor([target_shape[0] / h, target_shape[1] / w, target_shape[2] / d])
        flow_hr = torch.nn.functional.interpolate(flow, size=(target_shape[0], target_shape[1], target_shape[2]),
                                                mode='trilinear',
                                                align_corners=True) * ratio.to(device).view(1, -1, 1, 1, 1)
    elif ndims==2:
        _, c, h, w = flow.shape
        ratio = torch.FloatTensor([target_shape[0] / h, target_shape[1] / w])
        flow_hr = torch.nn.functional.interpolate(flow, size=(target_shape[0], target_shape[1]),
                                                mode='bilinear',
                                                align_corners=True) * ratio.to(device).view(1, -1, 1, 1)    
        
    return flow_hr


def trilinear_sample(flow, xyz):
    xyz0 = torch.ceil(xyz + 0.001).int()
    xyz1 = torch.floor(xyz + 0.001).int()
    xyzd = (xyz - xyz0) / (xyz1 - xyz0)
    x0, y0, z0 = xyz0[0], xyz0[1], xyz0[2]
    x1, y1, z1 = xyz1[0], xyz1[1], xyz1[2]
    xd, yd, zd = xyzd[0], xyzd[1], xyzd[2]
    C000 = flow[0, :, x0, y0, z0]
    C100 = flow[0, :, x1, y0, z0]
    C001 = flow[0, :, x0, y0, z1]
    C101 = flow[0, :, x1, y0, z1]
    C010 = flow[0, :, x0, y1, z0]
    C110 = flow[0, :, x1, y1, z0]
    C011 = flow[0, :, x0, y1, z1]
    C111 = flow[0, :, x1, y1, z1]

    C00 = C000 * (1-xd) + C100 * xd
    C01 = C001 * (1-xd) + C101 * xd
    C10 = C010 * (1-xd) + C110 * xd
    C11 = C011 * (1-xd) + C111 * xd

    C0 = C00 * (1-yd) + C10 * yd
    C1 = C01 * (1-yd) + C11 * yd

    return C0 * (1-zd) + C1 * zd



def smooth_loss(flow):
    dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


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


def upsample(img, is_flow, scale_factor=2.0, align_corners=True):
    img_resized = F.interpolate(img, scale_factor=scale_factor, mode='trilinear',
                                            align_corners=align_corners)

    if is_flow:
        img_resized *= scale_factor

    return img_resized



class STN(nn.Module):

    def __init__(self, mode='bilinear', padding_mode='border', size=(128, 128, 128)):
        super().__init__()
        self._interp_mode = mode
        self._padding_mode = padding_mode
        self.size = size
        self.grid = self.get_reference_grid(size).cuda()

    def get_reference_grid(self, size):
        mesh_points = [torch.arange(0, dim) for dim in size]
        grid = torch.stack(self.meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
        grid = torch.stack([grid] * 1, dim=0)  # (batch, spatial_dims, ...)x
        return grid
    
    def meshgrid_ij(self, *tensors):
        if torch.meshgrid.__kwdefaults__ is not None and "indexing" in torch.meshgrid.__kwdefaults__:
            return torch.meshgrid(*tensors, indexing="ij")  # new api pytorch after 1.10
        return torch.meshgrid(*tensors)

    def forward(self, image: torch.Tensor, ddf: torch.Tensor):
        spatial_dims = len(image.shape) - 2
        
        if spatial_dims not in (2, 3):
            raise NotImplementedError(f"got unsupported spatial_dims={spatial_dims}, currently support 2 or 3.")
        ddf_shape = (image.shape[0], spatial_dims) + tuple(image.shape[2:])
        if ddf.shape != ddf_shape:
            raise ValueError(
                f"Given input {spatial_dims}-d image shape {image.shape}, " f"the input DDF shape must be {ddf_shape}."
            )
        grid = self.grid + ddf
        grid = grid.permute([0] + list(range(2, 2 + spatial_dims)) + [1])  # (batch, ..., spatial_dims)

        for i, dim in enumerate(grid.shape[1:-1]):
            grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
            
        index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
        
        grid = grid[..., index_ordering]  # z, y, x -> x, y, z

        return F.grid_sample(
            image, grid, mode=self._interp_mode, padding_mode=f"{self._padding_mode}", align_corners=True
        )
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def normalize_dino(img, return_min_max=False, vlim=None):
    if img.numel() == 0:
        return img[..., :3]

    _img = img[..., :3]
    _ndims = len(img.shape) - 1
    _dims = [1] * _ndims + [3]

    if vlim is None:
        vmin = _img.reshape(-1, 3).min(dim=0)[0].view(*_dims)
        vmax = _img.reshape(-1, 3).max(dim=0)[0].view(*_dims)
    else:
        vmin, vmax = vlim

    if return_min_max:
        return (_img - vmin) / (vmax - vmin), (vmin, vmax)
    else:
        return (_img - vmin) / (vmax - vmin)
    
def apply_cmap_to_torch_tensor(x, cmap, vlim):
    if vlim is not None:
        _x = x.clip(*vlim)
    else:
        _x = x

    _x = (_x - _x.min()) / (_x.max() - _x.min())

    if cmap == 'step':
        _low_mask = _x < 0.1
        _high_mask = _x > 0.9
        _x = torch.stack([
            torch.ones_like(_x),
            1-_x,
            torch.zeros_like(_x)
        ], dim=-1)
        _x[_high_mask] = 0.
        _x[_low_mask] = 0.8
    else:
        _cmap = plt.colormaps[cmap]
        #remove alpha
        _x = torch.tensor(_cmap(_x.cpu().numpy()), device=_x.device)[..., :3]

    return _x

def traj_to_o3d(traj, color=[0., 0., 0.]):
    if isinstance(traj, torch.Tensor):
        return traj_to_o3d(traj.detach().cpu().numpy())

    adj = np.stack([
        np.arange(traj.shape[0]-1),
        np.arange(1, traj.shape[0])
    ], axis=-1)

    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(traj[:, :3])
    out.lines = o3d.utility.Vector2iVector(adj)

    out.paint_uniform_color(color)

    return out

def get_atv_mesh(fp='/home/tartandriver/tartandriver_ws/src/core/tartandriver_utils/atv_mesh/textured.obj'):
    """
    Load the ATV mesh into open3d for viz. Note that the ATV will be transformed such that
        (0,0,0) lines up (roughly) with the center of the rear axle (and in FLU)
    TODO figure out better pathing
    """
    mesh = o3d.io.read_triangle_mesh(fp, enable_post_processing=True)
    H = np.array([
        [0., 0., 1., 1.37],
        [1., 0., 0., 0.1],
        [0., 1., 0., 0.75],
        [0., 0., 0., 1.]
    ])
    return mesh.transform(H)

def make_bev_mesh(metadata, height, mask, colors):
    xy_coords = metadata.get_coords()
    coords = torch.cat([xy_coords, height.unsqueeze(-1)], dim=-1)

    ## simplest approach - every tile is 2 flat triangles ##
    dxs = torch.tensor([
        [0., 0., 0.],
        [metadata.resolution[0], 0., 0.],
        [0., metadata.resolution[1], 0.],
        [metadata.resolution[0], metadata.resolution[1], 0.]
    ], device=height.device)

    vertices = coords.view(metadata.N[0], metadata.N[1], 1, 3) + dxs.view(1, 1, 4, 3) #[WxHx4x3]

    heights_pad = torch.nn.functional.pad(height.unsqueeze(0), pad=(0,1,0,1), mode='replicate')[0]
    neighbor_heights = torch.stack([
        heights_pad[:-1, :-1],
        heights_pad[1:, :-1],
        heights_pad[:-1, 1:],
        heights_pad[1:, 1:]
    ], dim=-1)

    mask_pad = torch.nn.functional.pad(mask.unsqueeze(0).float(), pad=(0,1,0,1), mode='replicate')[0] > 1e-4

    mask = mask_pad[:-1, :-1] & mask_pad[1:, :-1] & mask_pad[:-1, 1:] & mask_pad[1:, 1:]

    vertices[..., -1] = neighbor_heights
    vertices = vertices[mask] #[Px4x3]
    coords = vertices[:, 0]
    colors = colors[mask]
    
    #triangles are one-sided so copy each
    adj_dxs = torch.tensor([
        [0,1,2],
        [1,2,3],
        [2,1,0],
        [3,2,1],
    ])

    base_dxs = torch.arange(coords.shape[0]) * 4
    base_dxs = base_dxs.unsqueeze(-1).tile(1, adj_dxs.shape[0]) #[Px2]
    adjs = base_dxs.view(-1, adj_dxs.shape[0], 1) + adj_dxs.view(1,-1,3) #[Px3]

    colors = colors.view(-1, 1, 3).tile(1,4,1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy().reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.cpu().numpy().reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(adjs.cpu().numpy().reshape(-1, 3))

    # mesh.compute_vertex_normals()

    return mesh
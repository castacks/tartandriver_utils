import torch
import numpy as np
import open3d as o3d

def normalize_dino(img, return_min_max=False):
    if img.numel() == 0:
        return img[..., :3]

    _img = img[..., :3]
    _ndims = len(img.shape) - 1
    _dims = [1] * _ndims + [3]
    vmin = _img.reshape(-1, 3).min(dim=0)[0].view(*_dims)
    vmax = _img.reshape(-1, 3).max(dim=0)[0].view(*_dims)
    if return_min_max:
        return (_img - vmin) / (vmax - vmin), (vmin, vmax)
    else:
        return (_img - vmin) / (vmax - vmin)

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
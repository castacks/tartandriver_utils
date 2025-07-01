import torch
import numpy as np
import open3d as o3d

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
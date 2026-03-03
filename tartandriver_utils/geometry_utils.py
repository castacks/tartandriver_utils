#collection of trajectory/rotation math

import torch
import numpy as np
import scipy.interpolate, scipy.spatial

ODOM_MASK = np.asarray([0,0,0,1,1,1,1,0,0,0,0,0,0], dtype=bool)
IMU_MASK = np.asarray([1,1,1,1,0,0,0,0,0,0], dtype=bool)
POSE_MASK = np.asarray([0,0,0,1,1,1,1], dtype=bool)

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    return np.arctan2(2 * (quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2))

def yaw_to_quat(yaw):
    return np.array([
        np.cos(yaw/2.),
        0.,
        0.,
        np.sin(yaw/2.)
    ])

def pose_to_htm(pose):
    """convert a pose (position + quaternion) to a homogeneous transform matrix
    """
    if isinstance(pose, torch.Tensor):
        device = pose.device
        res = pose_to_htm(pose.cpu().numpy())
        return torch.tensor(res, dtype=pose.dtype, device=device)

    if len(pose.shape) == 1:
        return pose_to_htm(np.expand_dims(pose, 0))[0]

    n = pose.shape[0]

    p = pose[:, :3]
    q = pose[:, 3:7]

    R = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()

    htm = np.tile(np.eye(4), [n,1,1])
    htm[:, :3, :3] = R
    htm[:, :3, -1] = p

    return htm

def htm_to_pose(htm):
    """convert a htm (4x4) matrix to a pose (position + quaternion)
    (quaternion will be [qx, qy, qz, qw])
    """
    if isinstance(htm, torch.Tensor):
        device = htm.device
        res = htm_to_pose(htm.cpu().numpy())
        return torch.tensor(res, device=device)

    if len(htm.shape) == 2:
        return htm_to_pose(np.expand_dims(htm, 0))[0]

    n = htm.shape[0]

    R = htm[:, :3, :3]
    p = htm[:, :3, -1]
    q = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
    pose = np.concatenate([p, q], axis=-1)

    return pose

def transform_points(points, htm):
    """transform a set of points using a homogeneous transform matrix
    Args:
        points: [P x N] Tensor of points (assuming first three channels are [x,y,z])
        htm: [4 x 4] transform patrix

    Returns:
        points: [P x N] Tensor of points where the first three channels have been transformed according to htm (note that this modifies points)
    """
    pt_pos = points[:, :3]
    pt_pos = torch.cat([pt_pos, torch.ones_like(pt_pos[:, [0]])], dim=-1)
    pt_tf_pos = htm.view(1, 4, 4) @ pt_pos.view(-1, 4, 1)
    points[:, :3] = pt_tf_pos[:, :3, 0]
    return points

class MultiDimensionalInterpolator:
    """
    Helper class for interpolating generic timeseries
    Expects either a N-element trajectory as: [T x N], where N is an arbitrary
    dimension timeseries. This could be an odometry trajectory, command list, etc.

    Funtionally, this works identically to the scipy interpolation object.
    This interpolator can handle torch or numpy inputs and returns the same
    type as the input.
    """
    def __init__(self,
                 times: np.ndarray | torch.Tensor,
                 traj: np.ndarray | torch.Tensor,
                 rot_mask: np.ndarray | torch.Tensor=None,
                 tol: float=1e-1,
                 interp_kwargs={}):
        """
        Args:
            traj: the traj to interpolate (of shape [T x N])
            times: the times corresponding to the traj
            rot_mask: Mask for rotation interpolation instead of generic 1D
                      interpolations. 1 indicates rotation index. For now, we
                      assume all rot elements are next to each other and of
                      order: qx, qy, qz, qw
            tol: the amount of allowable extrapolation
        """
        # Format into np. Assuming if traj is torch, then output should be torch.
        if isinstance(times, torch.Tensor):
            times = times.cpu().numpy()
        if isinstance(rot_mask, torch.Tensor):
            rot_mask = rot_mask.cpu().numpy()
        if isinstance(traj, torch.Tensor):
            self._torch_types = True
            self._device = traj.device
            traj = traj.cpu().numpy()
        else:
            self._torch_types = False

        # Set defauls and reshape traj if only 1D
        if len(traj.shape) == 1:
            traj = traj.reshape(traj.shape[0], 1)
        if rot_mask is None:
            rot_mask = np.zeros(traj.shape[-1], dtype=bool)

        assert len(traj.shape) == 2, 'Expected traj of shape [T x N], got {}'.format(traj.shape)
        assert traj.shape[-1] == rot_mask.shape[0], 'Rotation mask dimension must match traj N={}'.format(traj.shape[-1])
        assert times.shape[0] == traj.shape[0], 'Got {} times, but {} steps in traj'.format(times.shape[0], traj.shape[0])
        assert (~rot_mask).all() or rot_mask.sum() == 4, 'Requires either no rotation or mask with 4 elements for a quaternion, not {}'.format(rot_mask.sum())

        self._tol = tol
        self._N = traj.shape[-1]
        self._traj_interp = np.zeros(self._N, dtype=traj.dtype)
        self._rot_mask = rot_mask
        self._has_rot = np.any(self._rot_mask)

        #add tol
        times = np.concatenate([np.array([times[0]-self._tol]), times, np.array([times[-1] + self._tol])])
        traj = np.concatenate([traj[[0]], traj, traj[[-1]]], axis=0)

        #ensure that times are in strictly increasing order
        idxs = np.argsort(times)

        times_sorted  = times[idxs]
        traj_sorted = traj[idxs]

        tdiffs = times_sorted[1:] - times_sorted[:-1]
        valid_mask = np.ones(times_sorted.shape[0], dtype=bool)
        valid_mask[1:] = (tdiffs > 1e-16)

        times_to_interp = times_sorted[valid_mask]
        traj_to_interp = traj_sorted[valid_mask]

        # Create interpolators only for non-rotation dimensions
        self._interps = [
            scipy.interpolate.interp1d(times_to_interp, traj_to_interp[:, i], **interp_kwargs) 
            for i in np.where(~self._rot_mask)[0]
        ]

        # Create interpolators for rotation dimensions
        if self._has_rot:
            rots = scipy.spatial.transform.Rotation.from_quat(traj_to_interp[:, self._rot_mask])
            self._rot_interp = scipy.spatial.transform.Slerp(times_to_interp, rots, **interp_kwargs)

    def __call__(self, qtimes):
        """
        Interpolate the traj according to qtimes.
        Args:
            qtimes: the set of times to query
        """
        return self[qtimes]

    def __getitem__(self, qtimes):
        """
        Interpolate the traj according to qtimes.
        Args:
            qtimes: the set of times to query
        """
        res = np.copy(self._traj_interp)
        if isinstance(qtimes, torch.Tensor):
            qtimes = qtimes.cpu().numpy()

        interp = np.stack([itrp(qtimes) for itrp in self._interps])
        res[~self._rot_mask] = interp

        if self._has_rot:
            rot_interp = self._rot_interp(qtimes).as_quat()
            res[self._rot_mask] = rot_interp

        if self._torch_types:
            return torch.from_numpy(res)
        else:
            return res


class TrajectoryInterpolator(MultiDimensionalInterpolator):
    """
    Helper class for interpolating generic timeseries
    Expects either a 7 or 13-element trajectory as: [T x {7,13}], for
    interpolating odometry trajectories of [x y z qx qy qz qw vx vy vz wx wy wz]
    or pose trajectores of [x y z qx qy qz qw]

    funtionally, works identically to the scipy interpolation object
    """
    def __init__(self,
                 times: np.ndarray | torch.Tensor,
                 traj: np.ndarray | torch.Tensor,
                 tol: float=1e-1,
                 interp_kwargs={}):
        
        assert len(traj.shape) == 2, 'Expected traj of shape [T x {7,13}], got {}'.format(traj.shape)
        if traj.shape[-1] == 7:
            mask = POSE_MASK
        elif traj.shape[-1] == 13:
            mask = ODOM_MASK
        super().__init__(times, traj, mask, tol, interp_kwargs)
    
    def __call__(self, qtimes):
        """
        Interpolate the traj according to qtimes.
        Args:
            qtimes: the set of times to query
        """
        return super().__call__(qtimes)

    def __getitem__(self, qtimes):
        """
        Interpolate the traj according to qtimes.
        Args:
            qtimes: the set of times to query
        """
        return super().__getitem__(qtimes)


def make_footprint(length, width, nl, nw, length_offset, width_offset, device='cpu'):
    xs = torch.linspace(-length/2., length/2., nl, device=device) + length_offset
    ys = torch.linspace(-width/2., width/2., nw, device=device) + width_offset
    footprint = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).view(-1, 2)
    return footprint

def apply_footprint(traj, footprint):
    """
    Given a B x K x T x 3 tensor of states (last dim is [x, y, th]),
    return a B x K x T x F x 2 tensor of positions (F is each footprint sample)
    """
    tdims = traj.shape[:-1]
    nf = footprint.shape[0]

    pos = traj[..., :2]
    th = traj[..., 2]

    R = torch.stack([
        torch.stack([th.cos(), -th.sin()], dim=-1),
        torch.stack([th.sin(), th.cos()], dim=-1),
    ], dim=-2) #[B x K x T x 2 x 2]

    R_expand = R.view(*tdims, 1, 2, 2) #[B x K x T x F x 2 x 2]
    footprint_expand = footprint.view(1, 1, 1, nf, 2, 1) #[B x K x T x F x 2 x 1]

    footprint_rot = (R_expand @ footprint_expand).view(*tdims, nf, 2) #[B x K x T X F x 2]
    footprint_traj = pos.view(*tdims, 1, 2) + footprint_rot

    return footprint_traj

if __name__ == '__main__':
    #test trajectory interpolator by reading a rosbag
    import rosbag
    import matplotlib.pyplot as plt

    bag_fp = '/home/striest/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_train/20220630/2022-06-30-16-13-36_0.bag'
    odom_topic = '/odometry/filtered_odom'

    bag = rosbag.Bag(bag_fp, 'r')
    traj = []
    timestamps = []

    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        traj.append(np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]))
        timestamps.append(msg.header.stamp.to_sec())

    #zero x, y, z.
    traj = np.stack(traj, axis=0)
    traj[:, :3] -= traj[[0], :3]
    timestamps = np.array(timestamps)

    skip = 20
    subtraj = traj[::skip]
    subtimes = timestamps[::skip]

    tinterp = TrajectoryInterpolator(subtimes, subtraj)
    qtimes = timestamps[:-2*skip]

    itraj = tinterp(qtimes)

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    axs[0].scatter(subtraj[:, 0], subtraj[:, 1], c='b', marker='.', label='knot pts')
    axs[0].plot(itraj[:, 0], itraj[:, 1], c='r', label='interp')
    axs[0].legend()

    colors = 'rgbcmyk'
    for i, label in enumerate(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']):
        color = colors[i % len(colors)]
        axs[1].scatter(subtimes, subtraj[:, i], c=color, label=label, s=1.)
        axs[1].plot(qtimes, itraj[:, i], c=color, alpha=0.5)
    axs[1].legend()
        
    plt.show()
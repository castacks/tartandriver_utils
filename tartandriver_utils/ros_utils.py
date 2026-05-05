import numpy as np
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose

from core_interfaces.msg import Mission, Waypoint

def stamp_to_time(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def time_to_stamp(t):
    sec = int(t)
    nsec = int((t-sec)*1e9)

    return Time(sec=sec, nanosec=nsec)

def quat_to_yaw(orientation: Quaternion):
    """
        Retrieve yaw from quaternion

        :param orientation: Quaternion

        :returns yaw:
    """
    qw = orientation.w
    qx = orientation.x
    qy = orientation.y
    qz = orientation.z
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    return yaw

def waypoint_dict_to_msg(wpt):
    """
    Helper to convert waypoint from dictionary format (for YAMLs) to Waypoint()
    """
    waypoint = Waypoint()
    waypoint.header.frame_id = wpt['frame_id']
    waypoint.radius = wpt['radius']
    waypoint.position.x = wpt['pose']['x']
    waypoint.position.y = wpt['pose']['y']
    waypoint.position.z = wpt['pose']['z']
    return waypoint

def waypoint_pose_to_msg(wpt: Pose, frame, radius):
    """
    Helper to convert waypoint from Pose to Waypoint()
    """
    waypoint = Waypoint()
    waypoint.header.frame_id = frame
    waypoint.radius = radius
    waypoint.position.x = wpt.position.x
    waypoint.position.y = wpt.position.y
    waypoint.position.z = wpt.position.z
    return waypoint

def odom_to_pose(msg: Odometry):
    """Return (translation np.ndarray(3,), rotation-matrix np.ndarray(3,3))."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    t = np.array([p.x, p.y, p.z])
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return t, R

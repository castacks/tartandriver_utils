import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion

from core_interfaces.msg import Waypoint

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

def waypoint_to_msg(wpt):
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

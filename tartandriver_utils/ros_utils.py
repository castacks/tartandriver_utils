import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, Pose

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

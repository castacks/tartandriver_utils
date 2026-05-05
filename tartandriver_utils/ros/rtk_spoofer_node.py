import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
# import ros2_numpy
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch
import torch
from collections import deque
from scipy.spatial.transform import Rotation

from tartandriver_utils.ros_utils import odom_to_pose


class RTKSpooferNode(Node):
    def __init__(self):
        super().__init__('rtk_spoofer_node')

        self.declare_parameter('rtk_topic', '/rtk_gps/ekf/odometry_earth')
        self.declare_parameter('undistorted_cloud_topic', '/superodometry/feature_info')
        self.declare_parameter('publish_topic_odom', '/odometry/spoofed')
        self.declare_parameter('publish_topic_cloud', '/superodometry/velodyne_cloud_registered')
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('frame_id', 'sensor_init')

        rtk_topic = self.get_parameter('rtk_topic').value
        cloud_topic = self.get_parameter('undistorted_cloud_topic').value
        publish_topic_odom = self.get_parameter('publish_topic_odom').value
        publish_topic_cloud = self.get_parameter('publish_topic_cloud').value
        publish_rate = self.get_parameter('publish_rate').value
        self.frame_id  = self.get_parameter('frame_id').value

        self.init_pose: Odometry = None   # first message received

        # SE3 origin stored as (translation, rotation-matrix)
        self.origin_t: np.ndarray = None
        self.origin_R: np.ndarray = None

        # Rolling 2-second buffer of (stamp_sec, t, R) for closest-pose lookup
        self.pose_buffer = deque()
        self.buffer_seconds: float = 2.0

        self.rtk_sub = self.create_subscription(Odometry, rtk_topic, self._rtk_callback, 10)
        self.cloud_sub = self.create_subscription(PointCloud2, cloud_topic, self._cloud_callback, 10 )

        self.odom_pub = self.create_publisher(Odometry, publish_topic_odom, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, publish_topic_cloud, 10)

        self.create_timer(1.0 / publish_rate, self._publish_callback)

        self.get_logger().info(
            f"RTKSpooferNode ready. Listening on '{rtk_topic}', "
            f"publishing to '{publish_topic_odom}' at {publish_rate} Hz."
        )

    def _rtk_callback(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t, R = odom_to_pose(msg)

        self.pose_buffer.append((stamp, t, R))
        cutoff = stamp - self.buffer_seconds
        while self.pose_buffer and self.pose_buffer[0][0] < cutoff:
            self.pose_buffer.popleft()

        if self.init_pose is None:
            self.init_pose = msg
            self.origin_t = t
            self.origin_R = R
            self.get_logger().info(
                f"Origin initialised at  t={t.tolist()}  "
                f"(yaw={np.degrees(Rotation.from_matrix(R).as_euler('xyz')[2]):.2f} deg)"
            )

    def _closest_pose(self, query_sec: float):
        """Return (t, R) from the buffer entry closest to query_sec."""
        if not self.pose_buffer:
            return None, None
        _, t, R = min(self.pose_buffer, key=lambda e: abs(e[0] - query_sec))
        return t, R

    def _cloud_callback(self, msg: PointCloud2) -> None:
        """Transform the undistorted cloud into the origin frame and republish."""
        if self.origin_t is None:
            return

        cloud_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cur_t, cur_R = self._closest_pose(cloud_stamp)
        if cur_t is None:
            return

        pc = PointCloudTorch.from_rosmsg(msg)
        xyz = pc.pts.numpy()  # shape (N, 3)
        if xyz.shape[0] == 0:
            return

        xyz_world = (cur_R @ xyz.T).T + cur_t
        xyz_rel = (self.origin_R.T @ (xyz_world - self.origin_t).T).T

        pc.pts = torch.tensor(xyz_rel, dtype=torch.float32)
        pc.frame_id = self.frame_id
        self.cloud_pub.publish(pc.to_rosmsg())

    def _publish_callback(self) -> None:
        """Publish the relative odometry at the configured rate."""
        if self.origin_t is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        cur_t, cur_R = self._closest_pose(now)
        if cur_t is None:
            return

        rel_R = self.origin_R.T @ cur_R
        rel_t = self.origin_R.T @ (cur_t - self.origin_t)
        rel_q = Rotation.from_matrix(rel_R).as_quat()  # [x, y, z, w]

        odom = Odometry()
        sec = int(now)
        odom.header.stamp.sec = sec
        odom.header.stamp.nanosec = int((now - sec) * 1e9)
        odom.header.frame_id = self.frame_id
        odom.pose.pose.position.x = float(rel_t[0])
        odom.pose.pose.position.y = float(rel_t[1])
        odom.pose.pose.position.z = float(rel_t[2])
        odom.pose.pose.orientation.x = float(rel_q[0])
        odom.pose.pose.orientation.y = float(rel_q[1])
        odom.pose.pose.orientation.z = float(rel_q[2])
        odom.pose.pose.orientation.w = float(rel_q[3])
        self.odom_pub.publish(odom)

def main():
    rclpy.init()
    rtk_spoofer_node = RTKSpooferNode()
    rclpy.spin(rtk_spoofer_node)
    rtk_spoofer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

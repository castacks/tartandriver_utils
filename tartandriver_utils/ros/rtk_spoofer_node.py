import rclpy
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from super_odometry_msgs.msg import LaserFeature

from tartandriver_utils.ros_utils import odom_to_pose
from tartandriver_utils.geometry_utils import transform_points

from ros_torch_converter.datatypes.pointcloud import PointCloudTorch
from ros_torch_converter.datatypes.transform import TransformTorch


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

        # SE3 origin stored as (translation, rotation-matrix)
        self.origin_t: np.ndarray = None
        self.origin_R: np.ndarray = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Put each callback in its own group since MultiThreadedExecutor. Reentrant
        # vs Mutually exclusive doesn't matter since just one callback per group
        rtk_group = ReentrantCallbackGroup()
        cloud_group = ReentrantCallbackGroup()
        self.rtk_sub = self.create_subscription(Odometry, rtk_topic, self._rtk_callback, 10, callback_group=rtk_group)
        # self.cloud_sub = self.create_subscription(LaserFeature, cloud_topic, self._cloud_callback, 10, callback_group=cloud_group)
        self.cloud_sub = self.create_subscription(PointCloud2, cloud_topic, self._cloud_callback, 10, callback_group=cloud_group)

        self.odom_pub = self.create_publisher(Odometry, publish_topic_odom, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, publish_topic_cloud, 10)

        self.get_logger().info(
            f"RTKSpooferNode ready. Listening on '{rtk_topic}', "
            f"publishing to '{publish_topic_odom}' at {publish_rate} Hz."
        )

    def _rtk_callback(self, msg: Odometry) -> None:
        t, R = odom_to_pose(msg)
        if self.origin_t is None:
            self.origin_t = t
            self.origin_R = R
            self.get_logger().info(
                f"Origin initialized at  t={t.tolist()}  "
                f"(yaw={np.degrees(Rotation.from_matrix(R).as_euler('xyz')[2]):.2f} deg)"
            )

        rel_R = self.origin_R.T @ R
        rel_t = self.origin_R.T @ (t - self.origin_t)
        rel_q = Rotation.from_matrix(rel_R).as_quat()

        # publish odom
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = 'vehicle_rtk'
        odom.pose.pose.position.x = float(rel_t[0])
        odom.pose.pose.position.y = float(rel_t[1])
        odom.pose.pose.position.z = float(rel_t[2])
        odom.pose.pose.orientation.x = float(rel_q[0])
        odom.pose.pose.orientation.y = float(rel_q[1])
        odom.pose.pose.orientation.z = float(rel_q[2])
        odom.pose.pose.orientation.w = float(rel_q[3])
        J = np.block([[self.origin_R.T, np.zeros((3, 3))],
                      [np.zeros((3, 3)), self.origin_R.T]])
        cov = np.array(msg.pose.covariance).reshape(6, 6)
        odom.pose.covariance = (J @ cov @ J.T).flatten().tolist()
        odom.twist = msg.twist  # twist twist and covariance is in body frame, no rotation needed
        self.odom_pub.publish(odom)
        
        # publish TF
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = 'vehicle_rtk'

        t.transform.translation.x = float(rel_t[0])
        t.transform.translation.y = float(rel_t[1])
        t.transform.translation.z = float(rel_t[2])
        t.transform.rotation.x = float(rel_q[0])
        t.transform.rotation.y = float(rel_q[1])
        t.transform.rotation.z = float(rel_q[2])
        t.transform.rotation.w = float(rel_q[3])

        self.tf_broadcaster.sendTransform(t)

    def _cloud_callback(self, msg: PointCloud2) -> None:
        """Transform the undistorted cloud into the origin frame and republish."""
        stamp = msg.header.stamp
        src_frame = msg.header.frame_id
        dst_frame = self.frame_id

        try:
            tf_msg = self.tf_buffer.lookup_transform(dst_frame, src_frame, stamp, Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"cant tf from {dst_frame} to {src_frame}: {e}")
            return
        htm = TransformTorch.from_rosmsg(tf_msg, device='cpu').transform

        pc = PointCloudTorch.from_rosmsg(msg)
        xyz_src = pc.pts
        if xyz_src.shape[0] == 0:
            return

        xyz_dst = transform_points(xyz_src, htm)

        pc.pts = torch.tensor(xyz_dst, dtype=torch.float32)
        pc.frame_id = self.frame_id
        self.cloud_pub.publish(pc.to_rosmsg())

    # use undistorted cloud from super odometry
    # def _cloud_callback(self, msg: LaserFeature) -> None:
    #     """Transform the undistorted cloud into the origin frame and republish."""
    #     pc = PointCloudTorch.from_rosmsg(msg.cloud_nodistortion)
    #     xyz_src = pc.pts
    #     if xyz_src.shape[0] == 0:
    #         return
    #     pc.frame_id = 'vehicle'
    #     self.cloud_pub.publish(pc.to_rosmsg())


def main():
    rclpy.init()
    rtk_spoofer_node = RTKSpooferNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(rtk_spoofer_node)
    executor.spin()
    rtk_spoofer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

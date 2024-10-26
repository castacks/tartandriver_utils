import numpy as np
from dataclasses import dataclass, field
import string

from tartandriver_utils.ros_utils import waypoint_to_msg

from rclpy.node import Node
from rclpy.time import Duration

from std_msgs.msg import Bool, ColorRGBA
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker, MarkerArray

from core_interfaces.msg import Waypoint, Mission

@dataclass
class MarkerConfig():
    """
    'Struct' for Marker configuration, including default Marker
    fields as well as custom fields for additional functionality

    Useful for setting configuraiton for Marker without creating Marker
    object.
    """
    ## Default Marker fields
    frame_id: string = ''
    namespace: string = ''
    type: int = Marker.SPHERE
    scale: list = field(default_factory=lambda: [1., 1., 1.])
    rgb: list = field(default_factory=lambda: [0., 1., 1.])
    alpha: float = 1.
    lifetime: float = 0.
    frame_locked: bool = True
    label: string = ''

    ## Custom fields
    # Waypoint fade
    points_in_view: int = 10
    fade_alpha: bool = True
    fade_fxn: int = 'quadratic'
    fade_past: bool = True

    # Text
    z_offset: float = 0.

    # Point aesthetic
    core_mode: bool = True

class MissionVisualizer():
    """
    Utility to convert waypoint data to visualized marker messages of
    configured format

    :param waypoints: waypoint data (in form of Waypoint, Mission, or
                      Dictionary/YAML content)
    :param node: rclpy.node to be used for clock
    """
    def __init__(self, waypoints, node: Node): #, config: MarkerConfig):
        self._waypoints = self._convert_waypoints(waypoints)
        if self._waypoints is None:
             print("Invalid entry")
             exit(1)
        self._node = node

        self._default_mission_point_cfg = MarkerConfig(
            namespace = 'mission_point',
            type = Marker.SPHERE,
            rgb = [1., 0., 0.],
            alpha = 1.,
            lifetime = 0.,
            frame_locked = True,
        )

        self._default_mission_text_cfg = MarkerConfig(
            namespace = 'mission_text',
            type = Marker.TEXT_VIEW_FACING,
            rgb = [1., 0., 0.],
            alpha = 1.,
            lifetime = 0.,
            frame_locked = True,
            label = 'waypoint'
        )

    def to_viz_msg(self, config) -> Marker:
        """
        Convert waypoints to marker message type for visualization

        :param config: pseudo 'struct' for Marker configuration
        """
        markers = MarkerArray()

        viz_waypoints = self._waypoints
        total_waypoints, total_places = self._get_totals(viz_waypoints)

        # Trimming points in view
        viz_numbers = np.arange(total_waypoints).tolist()
        if config.points_in_view < total_waypoints:
            if config.fade_past:
                viz_waypoints = viz_waypoints[:config.points_in_view]
                viz_numbers = viz_numbers[:config.points_in_view]
            else:
                viz_waypoints = viz_waypoints[-config.points_in_view:]
                viz_numbers = viz_numbers[-config.points_in_view:]
                
        
        total_waypoints, total_places = self._get_totals(viz_waypoints)

        # Fading waypoints to reduce clutter
        if config.fade_alpha:
            if config.fade_past:
                alphas = (np.linspace(config.alpha, 0, total_waypoints+1)**2).tolist()
            else:
                alphas = (np.linspace(0, config.alpha, total_waypoints+1)**2).tolist()
                alphas = alphas[1:]
        else:
            alphas = [config.alpha] * total_waypoints

        for wpt, i, alpha in zip(viz_waypoints, viz_numbers, alphas):
            marker = Marker()
            marker.header.stamp = self._node.get_clock().now().to_msg()
            marker.header.frame_id = config.frame_id
            marker.ns = config.namespace # Must have unique namespaces, otherwise overwriting
            marker.id = i
            marker.type = config.type
            marker.action = Marker.ADD

            wpt_pose = Pose()
            wpt_pose.position = wpt.position
            wpt_pose.position.z += config.z_offset
            marker.pose = wpt_pose

            wpt_scale = Vector3()
            wpt_scale.x=config.scale[0]
            wpt_scale.y=config.scale[1]
            wpt_scale.z=config.scale[2]
            if config.core_mode:
                # Turn marker into more opaque core
                wpt_scale.x=config.scale[0]/4
                wpt_scale.y=config.scale[1]/4
                wpt_scale.z=config.scale[2]/4
            marker.scale = wpt_scale

            wpt_color = ColorRGBA()
            wpt_color.r=config.rgb[0]
            wpt_color.g=config.rgb[1]
            wpt_color.b=config.rgb[2]
            wpt_color.a=alpha
            marker.color = wpt_color

            marker.lifetime = Duration(nanoseconds=config.lifetime).to_msg()
            marker.frame_locked = config.frame_locked

            if config.label and marker.type == Marker.TEXT_VIEW_FACING:
                marker.text = f"{config.label}_{i:0{total_places}d}"

            markers.markers.append(marker)
            if config.core_mode and not marker.type == Marker.TEXT_VIEW_FACING:
                shell_marker = Marker()
                shell_marker.header.stamp = self._node.get_clock().now().to_msg()
                shell_marker.header.frame_id = config.frame_id
                shell_marker.ns = config.namespace+"_shell" # Must have unique namespaces, otherwise overwriting
                shell_marker.id = i+total_waypoints
                shell_marker.type = config.type
                shell_marker.action = Marker.ADD

                wpt_pose = Pose()
                wpt_pose.position = wpt.position
                wpt_pose.position.z += config.z_offset
                shell_marker.pose = wpt_pose

                wpt_scale = Vector3()
                    # Turn shell_marker into more transparent shell
                wpt_scale.x=config.scale[0]
                wpt_scale.y=config.scale[1]
                wpt_scale.z=config.scale[2]
                shell_marker.scale = wpt_scale

                wpt_color = ColorRGBA()
                wpt_color.r=config.rgb[0]
                wpt_color.g=config.rgb[1]
                wpt_color.b=config.rgb[2]
                wpt_color.a=alpha/4
                shell_marker.color = wpt_color

                shell_marker.lifetime = Duration(nanoseconds=config.lifetime).to_msg()
                shell_marker.frame_locked = config.frame_locked

                markers.markers.append(shell_marker)
        return markers
    
    @property
    def waypoints(self):
        return self._waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints):
        if new_waypoints is not None:
            self._waypoints = self._convert_waypoints(new_waypoints)
            self._get_totals(self._waypoints)

    @property
    def default_mission_point_viz(self):
        return self._default_mission_point_cfg

    @default_mission_point_viz.setter
    def default_mission_point_viz(self, new_config):
        self._default_mission_point_cfg = new_config

    @property
    def default_mission_text_viz(self):
        return self._default_mission_text_cfg

    @default_mission_text_viz.setter
    def default_mission_text_viz(self, new_config):
        self._default_mission_text_cfg = new_config

    def _convert_waypoints(self, unfmt_waypoints):
        """
        Convert user input to Waypoints() object
        
        :param waypoints: Waypoints to convert (Waypoint, Mission, list of dicts)

        :return: List of Waypoints()
        """
        if isinstance(unfmt_waypoints, Waypoint):
            return [unfmt_waypoints]
        elif isinstance(unfmt_waypoints, Mission):
            return unfmt_waypoints.waypoints
        elif isinstance(unfmt_waypoints, list):
            waypoints = []
            for wpt in unfmt_waypoints:
                assert isinstance(wpt, dict)
                assert self._matching_keys(wpt, ['frame_id', 'pose', 'radius'])
                assert self._matching_keys(wpt['pose'], ['x','y','z','yaw'])
                waypoints.append(waypoint_to_msg(wpt))
            return waypoints

    def _matching_keys(self, dictionary: dict, keys: list):
        """
        Helper function to determine if list of keys matches keys
        in dictionary
        """
        return set(dictionary) == set(keys)

    def _get_totals(self, waypoints):
        """
        Helper function to update total number of waypoints and total whole
        number places (ex: 001), for numerical annotation in text
        """

        total_waypoints = len(waypoints)
        total_places = len(str(np.max([0, total_waypoints-1])))
        return total_waypoints, total_places

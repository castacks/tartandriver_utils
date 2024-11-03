import numpy as np
from dataclasses import dataclass, field
import string
from typing import Union
import copy

from tartandriver_utils.ros_utils import waypoint_dict_to_msg, waypoint_pose_to_msg

from rclpy.node import Node
from rclpy.time import Duration

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

from core_interfaces.msg import Waypoint, Mission


@dataclass
class MarkerConfig:
    """
    'Struct' for Marker configuration, including default Marker
    fields as well as custom fields for additional functionality

    Useful for setting configuraiton for Marker without creating Marker
    object.
    """

    ## Default Marker fields
    # all fields are from Marker() Message. See if you are unfamiliar
    frame_id: string = ""
    namespace: string = ""
    type: int = Marker.SPHERE
    scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rgb: list = field(default_factory=lambda: [0.0, 1.0, 1.0])
    alpha: float = 1.0
    lifetime: float = 1.0
    frame_locked: bool = True
    label: string = ""

    ## Custom fields
    # Numbering
    selection_mode: bool = False  # if vizing a selected mission vs generating mission

    # Waypoint fade
    points_in_view: int = 10  # how many points to display
    fade_alpha: bool = True  # whether or not to fade the points away
    fade_fxn: int = "quadratic"  # fade away function
    fade_past: bool = True  # whether to fade past points or fade upcoming points

    # Text
    z_offset: float = 0.0  # z offset for text

    # Point aesthetic
    core_mode: bool = True  # display waypoint sphere with transparent outer shell for
    # showing wayoint radius


@dataclass
class WaypointData:
    _waypoints: list[Waypoint]
    total: int = field(init=False)
    # mission_total: int = field(init=False)
    places: int = field(init=False)
    numbers: np.ndarray = field(init=False)

    def __post_init__(self):
        self._waypoints = self._convert_waypoints(self._waypoints)
        self._update_totals()

    def update(self, new_waypoints):
        if new_waypoints is not None:
            self._waypoints = self._convert_waypoints(new_waypoints)
            self._update_totals()

    def _convert_waypoints(
        self, unfmt_waypoints: Union[Waypoint, Mission, PoseArray, list]
    ) -> list[Waypoint]:
        """
        Convert user input to Waypoints() object

        :param waypoints: Waypoints to convert (Waypoint, Mission, list of dicts)

        :return: List of Waypoints()
        """
        if isinstance(unfmt_waypoints, Waypoint):
            return [unfmt_waypoints]
        elif isinstance(unfmt_waypoints, Mission):
            return unfmt_waypoints.waypoints
        elif isinstance(unfmt_waypoints, PoseArray):
            waypoints = []
            for wpt in unfmt_waypoints.poses:
                waypoints.append(waypoint_pose_to_msg(wpt, "map", 4.0))
            return waypoints
        elif isinstance(unfmt_waypoints, list):
            waypoints = []
            for wpt in unfmt_waypoints:
                if isinstance(wpt, Waypoint):
                    waypoints.append(wpt)
                if isinstance(wpt, dict):
                    assert self._matching_keys(wpt, ["frame_id", "pose", "radius"])
                    assert self._matching_keys(wpt["pose"], ["x", "y", "z", "yaw"])
                    waypoints.append(waypoint_dict_to_msg(wpt))
            return waypoints

    def _matching_keys(self, dictionary: dict, keys: list):
        """
        Helper function to determine if list of keys matches keys
        in dictionary
        """
        return set(dictionary) == set(keys)

    def _update_totals(self):
        """
        Helper function to update total number of waypoints and total whole
        number places (ex: 001), for numerical annotation in text
        """

        self.total = len(self._waypoints)
        self.places = len(str(np.max([0, self.total - 1])))

    @property
    def waypoints(self) -> list[Waypoint]:
        return self._waypoints


class MissionVisualizer:
    """
    Utility to convert waypoint data to visualized marker messages of
    configured format

    :param waypoints: waypoint data (in form of Waypoint, Mission, or
                      Dictionary/YAML content)
    :param node: rclpy.node to be used for clock
    """

    def __init__(self, waypoints, node: Node):
        self._node = node
        self._mission_waypoints = WaypointData(waypoints)
        self._waypoints = WaypointData(waypoints)
        if self._waypoints is None:
            print("Invalid entry")
            exit(1)
        self._viz_waypoints = None

        self._default_mission_point_cfg = MarkerConfig(
            namespace="mission_point",
            type=Marker.SPHERE,
            rgb=[1.0, 0.0, 0.0],
            alpha=1.0,
            lifetime=1.0,
            frame_locked=True,
        )

        self._default_mission_text_cfg = MarkerConfig(
            namespace="mission_text",
            type=Marker.TEXT_VIEW_FACING,
            rgb=[1.0, 0.0, 0.0],
            alpha=1.0,
            lifetime=1.0,
            frame_locked=True,
            label="waypoint",
        )
        self._total = 0

    def to_viz_msg(self, config: MarkerConfig) -> Marker:
        """
        Convert waypoints to marker message type for visualization

        :param config: pseudo 'struct' for Marker configuration
        """
        markers = MarkerArray()

        self._viz_waypoints = copy.deepcopy(self._waypoints)

        # Numbering
        viz_numbers = self._number_waypoints(self._viz_waypoints, config.selection_mode)

        # Trim
        viz_numbers = self._trim_points_in_view(
            self._viz_waypoints, viz_numbers, config
        )

        # Fading waypoints
        alphas = self._set_alphas(
            self._viz_waypoints, config.alpha, config.fade_alpha, config.fade_past
        )

        for wpt, i, alpha in zip(self._viz_waypoints.waypoints, viz_numbers, alphas):
            marker = Marker()
            marker.header.stamp = self._node.get_clock().now().to_msg()
            marker.header.frame_id = config.frame_id
            marker.ns = (
                config.namespace
            )  # Must have unique namespaces, otherwise overwriting
            if config.type == Marker.TEXT_VIEW_FACING:
                marker.id = i + 2 * self._viz_waypoints.total
            else:
                marker.id = i
            marker.type = config.type
            marker.action = Marker.ADD

            wpt_pose = Pose()
            wpt_pose.position.x = wpt.position.x
            wpt_pose.position.y = wpt.position.y
            wpt_pose.position.z = wpt.position.z + config.z_offset
            marker.pose = wpt_pose

            wpt_scale = Vector3()
            wpt_scale.x = config.scale[0]
            wpt_scale.y = config.scale[1]
            wpt_scale.z = config.scale[2]
            if config.core_mode:
                # Turn marker into more opaque core
                wpt_scale.x = config.scale[0] / 4
                wpt_scale.y = config.scale[1] / 4
                wpt_scale.z = config.scale[2] / 4
            marker.scale = wpt_scale

            wpt_color = ColorRGBA()
            wpt_color.r = config.rgb[0]
            wpt_color.g = config.rgb[1]
            wpt_color.b = config.rgb[2]
            wpt_color.a = alpha
            marker.color = wpt_color

            marker.lifetime = Duration(nanoseconds=config.lifetime).to_msg()
            marker.frame_locked = config.frame_locked

            if config.label and marker.type == Marker.TEXT_VIEW_FACING:
                marker.text = f"{config.label}_{i:0{self._viz_waypoints.places}d}"

            markers.markers.append(marker)
            shell_marker = Marker()
            if config.core_mode and not marker.type == Marker.TEXT_VIEW_FACING:
                shell_marker.header.stamp = self._node.get_clock().now().to_msg()
                shell_marker.header.frame_id = config.frame_id
                shell_marker.ns = (
                    config.namespace + "_shell"
                )  # Must have unique namespaces, otherwise overwriting
                shell_marker.id = i + self._viz_waypoints.total
                shell_marker.type = config.type
                shell_marker.action = Marker.ADD

                wpt_pose = Pose()
                wpt_pose.position.x = wpt.position.x
                wpt_pose.position.y = wpt.position.y
                wpt_pose.position.z = wpt.position.z + config.z_offset
                shell_marker.pose = wpt_pose

                wpt_scale = Vector3()
                # Turn shell_marker into more transparent shell
                wpt_scale.x = config.scale[0]
                wpt_scale.y = config.scale[1]
                wpt_scale.z = config.scale[2]
                shell_marker.scale = wpt_scale

                wpt_color = ColorRGBA()
                wpt_color.r = config.rgb[0]
                wpt_color.g = config.rgb[1]
                wpt_color.b = config.rgb[2]
                wpt_color.a = alpha / 4
                shell_marker.color = wpt_color

                shell_marker.lifetime = Duration(nanoseconds=config.lifetime).to_msg()
                shell_marker.frame_locked = config.frame_locked

                markers.markers.append(shell_marker)
        return markers

    def load_mission(self, new_waypoints):
        """
        Load new mission. Acts like waypoints property, but it instead resets
        internal total waypoints for visualizing numbers
        """
        self._mission_waypoints.total = 0
        if new_waypoints is not None:
            self._mission_waypoints.update(new_waypoints)
            self._node.get_logger().info(f"{self._mission_waypoints.total}")

    @property
    def waypoints(self):
        return self._waypoints.waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints):
        self._waypoints.update(new_waypoints)

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

    def _trim_points_in_view(
        self, var_wpts: WaypointData, viz_nums: np.ndarray, config: MarkerConfig
    ) -> np.ndarray:
        """
        Trim waypoints to only show the ones in view
        """
        if config.points_in_view < self._mission_waypoints.total:
            if config.fade_past:
                var_wpts.update(var_wpts.waypoints[: config.points_in_view])
                viz_nums = viz_nums[: config.points_in_view]
            else:
                var_wpts.update(var_wpts.waypoints[-config.points_in_view :])
                viz_nums = viz_nums[-config.points_in_view :]

        self._node.get_logger().info(f"var_wpts len: {len(var_wpts.waypoints)}")
        return viz_nums

    def _set_alphas(
        self, var_wpts: WaypointData, alpha: float, fade_alpha: bool, fade_past: bool
    ) -> np.ndarray:
        if fade_alpha:
            if fade_past:
                alphas = (np.linspace(alpha, 0, var_wpts.total + 1) ** 2).tolist()
            else:
                alphas = (np.linspace(0, alpha, var_wpts.total + 1) ** 2).tolist()
                alphas = alphas[1:]
        else:
            alphas = [alpha] * var_wpts.total
        return np.array(alphas)

    def _number_waypoints(
        self, var_wpts: WaypointData, selection_mode: bool
    ) -> np.ndarray:
        viz_numbers = np.arange(var_wpts.total).tolist()
        if selection_mode:
            passed_points = self._mission_waypoints.total - var_wpts.total
            viz_numbers = [int(v + passed_points) for v in viz_numbers]
        return viz_numbers

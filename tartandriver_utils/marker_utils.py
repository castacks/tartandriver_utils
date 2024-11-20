import numpy as np
from dataclasses import dataclass, field
import string
from typing import Union
import copy
import yaml
import os

from tartandriver_utils.ros_utils import waypoint_dict_to_msg, waypoint_pose_to_msg

from rclpy.node import Node
from rclpy.time import Time, Duration

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

    :param {marker_fields}: Default marker fields found in Marker() Message
    :param points_in_view: How many marker points to display (for trajectories)
    :param fade_alpha: Whether or not to fade points away
    :param fade_past: Whether to fade past points or fade upcoming points
    :param z_offset: The z-offset for text
    :param lag_numbers_mode: Display number text as robot lags waypoints
    :param core_mode: Display marker sphere (core) with transparent outer shell for showing waypoint radius
    :param blink_mode: Blink marker at twice frequency from presently set lifetime
    """
    # ID
    config_name: string = ""

    ## Default Marker fields
    # all fields are from Marker() Message. See if you are unfamiliar
    frame_id: string = ""
    namespace: string = ""
    type: int = Marker.SPHERE
    scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rgb: list = field(default_factory=lambda: [0.0, 1.0, 1.0])
    alpha: float = 1.0
    lifetime: float = -1.0
    frame_locked: bool = True
    label: string = ""

    ## Custom fields
    # Marker fade
    points_in_view: int = 10
    fade_alpha: bool = True
    fade_past: bool = True

    # Text
    z_offset: float = 0.0

    # Numbering
    lag_numbers_mode: bool = False

    # Point aesthetic
    core_mode: bool = True
    blink_mode: bool = False

    # Interal storage
    _node: Node = None
    _initializing: bool = False

    def _update_field(self, new_config, key: str):
        """
        Helper function for updating config through dictionary/struct fields
        """
        # if data is private, do not override (_node, _initizializing)
        if key in {'_node', '_initializing'}:
            return
        
        # struct data updated from config
        if isinstance(new_config, dict):
            new_data = new_config.get(key, getattr(self, key))
        elif isinstance(new_config, MarkerConfig):
            new_data = getattr(new_config, key)

        setattr(self, key, new_data)

    def __setattr__(self, key, value):
        """
        Overloaded attribute setter with conversion utility and logger utility
        """
        # If no node, can do no logging. Set attribute first.
        if not self._node:
            super().__setattr__(key, value)
            return
        
        # Set default update format
        fmt=None

        # Key specific checks
        if key in {"config_name", "_initializing", "_node"}:
            # Skip special logging for these attributes
            super().__setattr__(key, value)
            return
        elif key == "lifetime":
            value = self._convert_lifetime(value)
            fmt='.2e'

        # Apply logging for all other keys
        if hasattr(self, "_initializing") and self._initializing:
            # Log "Set" messages during initialization
            if fmt:
                self._node.get_logger().info(f"Set {self.config_name} {key}={value:{fmt}}")
            else:
                self._node.get_logger().info(f"Set {self.config_name} {key}={value}")
        else:
            # Log "Updated" messages for post-initialization changes
            if hasattr(self, key) and getattr(self, key) != value:
                if fmt:
                    self._node.get_logger().info(f"Updated {self.config_name} {key}={value:{fmt}}")
                else:
                    self._node.get_logger().info(f"Updated {self.config_name} {key}={value}")
        
        super().__setattr__(key, value)

    def _convert_lifetime(self, time: Union[Time, Duration, float]):
        """
        Helper function to accept multiple types of lifetime inputs and convert to nanoseconds
        """
        converted_time = None
        if isinstance(time, Time) or isinstance(time, Duration):
            converted_time = time.nanoseconds
        else:
            converted_time = time

        if self.blink_mode:
            converted_time /= 2

        return converted_time

@dataclass
class WaypointData:
    _waypoints: list[Waypoint]
    total: int = field(init=False)
    places: int = field(init=False)
    numbers: list = field(init=False)
    radius: int = field(init=False)

    def __post_init__(self):
        self._waypoints = self._convert_waypoints(self._waypoints)
        self._update_totals()
        self.numbers = np.arange(self.total).tolist()
        self.radius = self._waypoints[0].radius if self._waypoints else 4.

    def update(self, new_waypoints):
        if new_waypoints:
            self._waypoints = self._convert_waypoints(new_waypoints)
            self._update_totals()
            self.radius = self._waypoints[0].radius

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


class MarkerVisualizer:
    """
    General utility to visualize all marker data with convenient modularity,
    easy-to-update configurations, and many visualization helper functions
    
    Cannot be used as standalone, but is a base class for all other visualizers
    """
    def __init__(self, node: Node, configs):
        self._node = node
        self._config = None # placeholder init for when setter called
        self.config = configs # call setter
    
    def update_all_configs(self, key, value):
        """
        Update all configs with the same data.
        Useful for frames, lifetime, etc.
        """
        if key in {"config_name", "namespace", "_node", "_initializing"}:
            self._node.get_logger().warn(f"Cannot change MarkerConfig key:{key}")
            return
        new_cfg = {key: value}
        for cfg in self._config:
            cfg._update_field(new_cfg, key)

    def _parse_config(self, new_config):
        configs = []
        init = False
        if isinstance(new_config, dict):
            config = self._config
            if not config:
                init = True
                c = []
                for cfg_name, _ in new_config.items():
                    c.append(MarkerConfig(config_name=cfg_name, _node=self._node, _initializing=init))
                config = c
            for cfg, [_, new_cfg] in zip(config, new_config.items()):
                for field in vars(cfg).keys():
                    cfg._update_field(new_cfg, field)
                if init:
                    cfg._initializing = not init
                configs.append(cfg) 
        elif isinstance(new_config, list):
            config = self._config
            if not config:
                init = True
                c = []
                for _ in range(len(new_config)):
                    c.append(MarkerConfig(_node=self._node, _initializing=init))
                config = c    
            for cfg, new_cfg in zip(config, new_config):
                for field in vars(cfg).keys():
                    cfg._update_field(new_cfg, field)
                if init:
                    cfg._initializing = not init
                configs.append(cfg)    
        else:
            raise TypeError("Unsupported config type. Expected dict or list[MarkerConfig].")

        return configs

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_configs: Union[str, dict, MarkerConfig, list]):
        if self._config is None:
            self._config = []

        if isinstance(new_configs, str):
            self._node.get_logger().info("Loading YAML path config...")
            assert os.path.exists(new_configs)
            with open(new_configs, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self._config = self._parse_config(config)

        elif isinstance(new_configs, dict):
            self._node.get_logger().info("Loading dictionary config...")
            self._config = self._parse_config(new_configs)

        elif isinstance(new_configs, MarkerConfig):
            self._node.get_logger().info("Loading MarkerConfig...")
            self._config = self._parse_config([new_configs])
        
        elif isinstance(new_configs, list):
            self._node.get_logger().info(f"Loading {len(new_configs):d} MarkerConfigs...")
            combined_config = [
                cfg for cfg in new_configs if isinstance(cfg, MarkerConfig)
            ]
            self._config = self._parse_config(combined_config)
        
        else:
            raise TypeError("configs must be a MarkerConfig, list of MarkerConfig, string, or dictionary")


class MissionVisualizer(MarkerVisualizer):
    """
    Utility to convert waypoint data to visualized marker messages of
    configured format

    :param waypoints: waypoint data (in form of Waypoint, Mission, or
                      Dictionary/YAML content)
    :param node: rclpy.node to be used for clock
    :param configs: configurations for marker visualization. Passed as MarkerConfig, list[MarkerConfig], YAML path, Dictionary
    """

    def __init__(
        self, waypoints, node: Node, configs
    ):
        self._node = node
        self._mission_waypoints = WaypointData(waypoints)
        self._waypoints = WaypointData(waypoints)
        if self._waypoints is None:
            print("Invalid entry")
            exit(1)
        
        super().__init__(self._node, configs)

    def to_viz_msg(self) -> Marker:
        """
        Convert waypoints to marker message type for visualization

        :param config: pseudo 'struct' for Marker configuration
        """
        markers = MarkerArray()

        if self._config is not None:
            for config in self._config:
                viz_waypoints = copy.deepcopy(self._waypoints)

                # Numbering
                self._number_waypoints(viz_waypoints, config.lag_numbers_mode)

                # Trim
                self._trim_points_in_view(
                    viz_waypoints, config.points_in_view, config.fade_past
                )

                # Fading waypoints
                alphas = self._set_alphas(
                    viz_waypoints, config.alpha, config.fade_alpha, config.fade_past
                )

                # Create marker
                for wpt, i, alpha in zip(
                    viz_waypoints.waypoints, viz_waypoints.numbers, alphas
                ):
                    marker = Marker()
                    marker.header.stamp = self._node.get_clock().now().to_msg()
                    marker.header.frame_id = config.frame_id
                    marker.ns = (
                        config.namespace
                    )  # Must have unique namespaces, otherwise overwriting
                    if config.type == Marker.TEXT_VIEW_FACING:
                        marker.id = i + 2 * viz_waypoints.total
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
                        marker.text = f"{config.label}_{i:0{viz_waypoints.places}d}"

                    markers.markers.append(marker)
                    shell_marker = Marker()
                    if config.core_mode and not marker.type == Marker.TEXT_VIEW_FACING:
                        shell_marker.header.stamp = (
                            self._node.get_clock().now().to_msg()
                        )
                        shell_marker.header.frame_id = config.frame_id
                        shell_marker.ns = (
                            config.namespace + "_shell"
                        )  # Must have unique namespaces, otherwise overwriting
                        shell_marker.id = i + viz_waypoints.total
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

                        shell_marker.lifetime = Duration(
                            nanoseconds=config.lifetime
                        ).to_msg()
                        shell_marker.frame_locked = config.frame_locked

                        markers.markers.append(shell_marker)
        return markers

    def load_mission(self, new_waypoints):
        """
        Load new mission. Acts like waypoints property, but it instead resets
        internal total waypoints for visualizing numbers. Also, update scale to
        reflect new radius.
        """
        self._mission_waypoints.total = 0
        if new_waypoints is not None:
            self._mission_waypoints.update(new_waypoints)
            self.update_all_configs("scale", [self._mission_waypoints.radius] * 3)
            self._node.get_logger().info(f"New mission loaded with {self._mission_waypoints.total} points")

    def _trim_points_in_view(
        self, var_wpts: WaypointData, points_in_view: int, fade_past: bool
    ) -> np.ndarray:
        """
        Trim waypoints to only show the ones in view
        """
        # Set reference total based on presence of mission_waypoints.
        # If no mission_waypoints, then this is a generator, and we just use
        # regular waypoints instead.
        if self._mission_waypoints.waypoints:
            waypoints_total = self._mission_waypoints.total
        else:
            waypoints_total = self._waypoints.total

        # Trim
        if points_in_view < waypoints_total:
            if fade_past:
                var_wpts.update(var_wpts.waypoints[:points_in_view])
                var_wpts.numbers = var_wpts.numbers[:points_in_view]
            else:
                var_wpts.update(var_wpts.waypoints[-points_in_view:])
                var_wpts.numbers = var_wpts.numbers[-points_in_view:]

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
        self, var_wpts: WaypointData, lag_numbers_mode: bool
    ) -> np.ndarray:
        viz_numbers = np.arange(var_wpts.total).tolist()
        if lag_numbers_mode:
            passed_points = self._mission_waypoints.total - var_wpts.total
            var_wpts.numbers = [int(v + passed_points) for v in viz_numbers]
        else:
            var_wpts.numbers = viz_numbers

    @property
    def waypoints(self):
        return self._waypoints.waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints):
        self._waypoints.update(new_waypoints)

    @property
    def waypoint_data(self):
        return self._waypoints

import numpy as np
from dataclasses import dataclass, field
import string
from typing import Union
import copy
import yaml
import os

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
    # Numbering
    selection_mode: bool = False  # if vizing a selected mission vs generating mission

    # Waypoint fade
    points_in_view: int = 10  # how many points to display
    fade_alpha: bool = True  # whether or not to fade the points away
    fade_past: bool = True  # whether to fade past points or fade upcoming points

    # Text
    z_offset: float = 0.0  # z offset for text

    # Point aesthetic
    core_mode: bool = True  # display waypoint sphere with transparent outer shell for
    # showing wayoint radius

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
        if not self._node:
            pass
        elif key in {"config_name", "_initializing", "_node"}:
            pass
        elif hasattr(self, "_initializing") and self._initializing:
            # Only show "Set" messages during initialization
            self._node.get_logger().info(f"Set {self.config_name} {key}={value}")
        else:
            # For post-initialization updates, show "Updated" messages
            if hasattr(self, key) and getattr(self, key) != value:
                self._node.get_logger().info(f"Updated {self.config_name} {key}={value}")
        super().__setattr__(key, value)


@dataclass
class WaypointData:
    _waypoints: list[Waypoint]
    total: int = field(init=False)
    places: int = field(init=False)
    numbers: list = field(init=False)

    def __post_init__(self):
        self._waypoints = self._convert_waypoints(self._waypoints)
        self._update_totals()
        self.numbers = np.arange(self.total).tolist()

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
    :param configs: configurations for marker visualization. Passed as MarkerConfig, list[MarkerConfig], YAML path, Dictionary

    TODO: Can be generalized further for non-waypoint use-cases i.e. MPPI, planning, etc.
    """

    def __init__(
        self, waypoints, node: Node, configs
    ):
        self._node = node
        self._mission_waypoints = WaypointData(waypoints)
        self._waypoints = WaypointData(waypoints)
        self._config = None  # placeholder init for when setter called
        self.config = configs
        if self._waypoints is None:
            print("Invalid entry")
            exit(1)

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
                self._number_waypoints(viz_waypoints, config.selection_mode)

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
        internal total waypoints for visualizing numbers
        """
        self._mission_waypoints.total = 0
        if new_waypoints is not None:
            self._mission_waypoints.update(new_waypoints)
            self._node.get_logger().info(f"New mission loaded with {self._mission_waypoints.total} points")

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

    def _trim_points_in_view(
        self, var_wpts: WaypointData, points_in_view: int, fade_past: bool
    ) -> np.ndarray:
        """
        Trim waypoints to only show the ones in view
        """
        if points_in_view < self._mission_waypoints.total:
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
        self, var_wpts: WaypointData, selection_mode: bool
    ) -> np.ndarray:
        viz_numbers = np.arange(var_wpts.total).tolist()
        if selection_mode:
            passed_points = self._mission_waypoints.total - var_wpts.total
            var_wpts.numbers = [int(v + passed_points) for v in viz_numbers]
        else:
            var_wpts.numbers = viz_numbers

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_configs):
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

    @property
    def config_names(self):
        return [cfg.config_name for cfg in self._config]

    @property
    def waypoints(self):
        return self._waypoints.waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints):
        self._waypoints.update(new_waypoints)

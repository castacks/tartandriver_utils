import numpy as np
from dataclasses import dataclass, field
import string
from typing import Union
import copy
import yaml
import os
import matplotlib as mpl

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
    :param rgb_start: Start RGB color for color gradient
    :param rgb_end: End RGB color for color gradient
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
    colors: list = field(default_factory=lambda: [])
    alpha: float = 1.0
    lifetime: float = -1.0
    frame_locked: bool = True
    label: string = ""

    ## Custom fields
    # Color gradient
    rgb_start: list = field(default_factory=lambda: [])
    rgb_end: list = field(default_factory=lambda: [])

    # Marker fade
    points_in_view: int = 10
    fade_alpha: bool = True
    fade_past: bool = True

    # Text
    z_offset: float = 0.0

    # Numbering
    lag_numbers_mode: bool = False

    # Point aesthetic
    core_mode: bool = False
    blink_mode: bool = False

    # Interal storage
    _node: Node = None
    _initializing: bool = False
    _quiet: bool = False

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
        
        # Key specific checks with custom logging
        if key in {"config_name", "_initializing", "_node", "_quiet"}:
            # Skip special logging for these attributes
            super().__setattr__(key, value)
        elif key == "colors":
            self._update_logger(key, value, show_value=False)
            super().__setattr__(key, value)
        elif key == "lifetime":
            value = self._convert_lifetime(value)
            self._update_logger(key, value, fmt='.2e')
            super().__setattr__(key, value)
        else:
            # Apply logging for all other keys
            self._update_logger(key, value)
            super().__setattr__(key, value)

    def _update_logger(self, key, value=None, fmt=None, show_value=True,):
        """
        Helper function for config parameter updates and sets
        """
        log_msg = f"{self.config_name} {key}"
        if not self._quiet:
            # State what key being set/updated to
            if value is not None and show_value:
                if fmt:
                    log_msg = log_msg + f"={value:{fmt}}"
                else:
                    log_msg = log_msg + f"={value}"
            
            if hasattr(self, "_initializing") and self._initializing:
                # Log "Set" messages during initialization
                log_msg = "Set " + log_msg
                self._node.get_logger().info(log_msg)
            elif hasattr(self, key) and getattr(self, key) != value:
                # Log "Updated" messages for post-initialization changes
                log_msg = "Updated " + log_msg
                self._node.get_logger().info(log_msg)

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

    def _make_color_gradient(self, size: int, c0=None, c1=None):
        """
        Update colors to be a color gradient
        """
        # Redundancy check
        if self.colors and c0 == self.colors[0] and c1 == self.colors[-1]:
            return
        # No start or end but requesting gradient
        if not self.rgb_start and not self.rgb_end:
            self._quiet = True # janky hack mate
            self.rgb_start = self.rgb
            self.rgb_end = self.rgb
            self._quiet = False
        # Same start and end
        if self.rgb_start == self.rgb_end:
            self._quiet = True
            self.colors = [self.rgb] * (size)
            self._quiet = False
            return
        # Override
        if c0 and c1:
            self.rgb_start = c0
            self.rgb_end = c1
        
        colors = []
        for c in range(size):
            colors.append([(1 - c / (size-1)) * self.rgb_start[i] + (c / (size-1)) * self.rgb_end[i] for i in range(3)])
        self.colors = colors
        return

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

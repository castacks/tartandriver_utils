import numpy as np
import copy

from rclpy.node import Node
from rclpy.time import Duration

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Pose
from visualization_msgs.msg import Marker, MarkerArray

from tartandriver_utils.marker_utils import WaypointData, MarkerVisualizer, MarkerConfig

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

                # Colors
                colors = self._set_colors(config, viz_waypoints.total)

                # Create marker
                for wpt, i, color, alpha in zip(
                    viz_waypoints.waypoints, viz_waypoints.numbers, colors, alphas
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
                    wpt_color.r = color[0]
                    wpt_color.g = color[1]
                    wpt_color.b = color[2]
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
                        wpt_color.r = color[0]
                        wpt_color.g = color[1]
                        wpt_color.b = color[2]
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

    def _set_colors(self, cfg: MarkerConfig, size: int):
        cfg._make_color_gradient(size)
        return cfg.colors

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

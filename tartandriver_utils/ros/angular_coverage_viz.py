#!/usr/bin/env python3
"""
Standalone live viewer for the AngularCoverage node's output, overlaid on a tif.

RViz can't overlay on a GeoTIFF, so this ROS node subscribes to the coverage GridMap
topic and renders the wedge-ring glyphs over the satellite image with matplotlib,
updating as new maps arrive.

Follows the mission_manager/mission_selector_node.py pattern: a single-threaded
rclpy.spin(node) drives both the subscription and a ROS timer; the timer callback
services the matplotlib GUI on the main thread via plt.show(block=False) + plt.pause().
(A background spin thread + plt.show() tends to silently drop callbacks / not redraw.)

Run:
    python3 angular_coverage_viz.py --ros-args \
        -p tif:=/path/to/gascola.tif -p topic:=/angular_coverage -p n_bins:=8

If nothing shows, check the backend logged on startup: a non-interactive backend
(e.g. 'agg') won't display -- force one with  MPLBACKEND=TkAgg  ...
"""

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry

from pyproj import Transformer

from torch_coordinator.nodes.angular_coverage import CoverageViz

import matplotlib
matplotlib.rcParams["figure.raise_window"] = False
import matplotlib.pyplot as plt

import numpy as np
import argparse


class AngularCoverageViz(Node):
    def __init__(self, tif_path, coverage_topic, n_bins, gps_topic, gps_crs, window_m):
        super().__init__("angular_coverage_viz")

        self.tif_path = tif_path
        self.topic = coverage_topic
        self.n_bins = n_bins
        self.gps_topic = gps_topic
        rate = self.declare_parameter("rate", 5.0).get_parameter_value().double_value
        # assert self.tif_path, "set the 'tif' parameter (-p tif:=/path/to.tif)"

        self.viz = CoverageViz(self.tif_path, interactive=True, window_m=window_m)
        self.viz.setup()

        # GPS position (ECEF, etc.) -> tif CRS, mirroring the coverage node's conversion,
        # so the vehicle marker / zoom window land in the same pixels as the glyphs.
        tif_crs = self.viz.tif.crs.to_string()
        self.to_dest = Transformer.from_crs(gps_crs, tif_crs, always_xy=True)

        self.latest = None
        self.veh_en = None          # latest vehicle (easting, northing) in the tif CRS
        self.count = 0
        self.dirty = False

        self.create_subscription(GridMap, self.topic, self.grid_callback, 10)
        self.create_subscription(Odometry, self.gps_topic, self.gps_callback, 10)
        self.create_timer(1.0 / rate, self.timer_callback)
        self.get_logger().info(
            f"subscribed to {self.topic} (+ gps {self.gps_topic})  |  "
            f"matplotlib backend = {matplotlib.get_backend()}"
        )

    def _decode_layer(self, msg, key, nx, ny):
        """Pull one GridMap layer back to a canonical (nx, ny) array.

        BEVGridTorch.from_rosmsg returns data as (n_layers, ny, nx); for the coarse,
        non-square coverage grid that permutation can't be undone with a transpose, so
        we invert BEVGridTorch.to_rosmsg's serialization here directly instead. That
        serialization is  flip(axis=(0,1)) -> transpose(1,0) -> C-order flatten, whose
        exact inverse is  reshape(ny, nx) -> transpose -> flip(axis=(0,1)).
        """
        flat = np.asarray(msg.data[msg.layers.index(key)].data, dtype=np.float32)
        return flat.reshape(ny, nx).T[::-1, ::-1].copy()

    def grid_callback(self, msg):
        for key in ("angle_bits", "known"):
            if key not in msg.layers:
                self.get_logger().warn(f"GridMap missing expected layer '{key}'")
                return

        res = msg.info.resolution
        lx, ly = msg.info.length_x, msg.info.length_y
        nx, ny = int(round(lx / res)), int(round(ly / res))
        origin = np.array([msg.info.pose.position.x - 0.5 * lx,
                           msg.info.pose.position.y - 0.5 * ly])
        resolution = np.array([res, res])

        bits = self._decode_layer(msg, "angle_bits", nx, ny)
        known = self._decode_layer(msg, "known", nx, ny) > 0
        self.latest = (bits, known, origin, resolution)
        self.count += 1
        self.dirty = True
        self.get_logger().info(
            f"msg #{self.count}: {int(known.sum())} observed cells", throttle_duration_sec=1.0
        )

    def gps_callback(self, msg):
        p = msg.pose.pose.position
        # ECEF (x, y, z) -> tif CRS; take (easting, northing), dropping any z passthrough
        e, n = self.to_dest.transform(p.x, p.y, p.z)[:2]
        self.veh_en = np.array([e, n])
        self.dirty = True

    def timer_callback(self):
        plt.show(block=False)  # start / keep the GUI up (non-blocking)
        if self.dirty and self.latest is not None:
            bits, known, origin, resolution = self.latest
            self.viz.render(bits, known, origin, resolution, self.n_bins, vehicle_en=self.veh_en)
            self.viz.ax.set_title(f"angular coverage - {int(known.sum())} cells, {self.count} msgs")
            self.dirty = False
        plt.pause(1e-2)        # service the matplotlib event loop on the main thread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=False, help="satellite GeoTIFF (same CRS as the grid)")
    ap.add_argument("--topic", default="/angular_coverage", help="coverage GridMap topic")
    ap.add_argument("--n-bins", type=int, default=8, help="number of angle bins")
    ap.add_argument("--gps-topic", default="/rtk_gps/ekf/odometry_earth",
                    help="Odometry topic with the vehicle position (centers the zoom window)")
    ap.add_argument("--gps-crs", default="EPSG:4978",
                    help="CRS of the GPS positions (EPSG:4978 = ECEF); converted to the tif CRS")
    ap.add_argument("--window", type=float, default=150.0,
                    help="side length (m) of the vehicle-centered zoom window")
    args, _ = ap.parse_known_args()
    rclpy.init()
    # node = AngularCoverageViz(args.tif, args.topic, args.n_bins,
    #                           args.gps_topic, args.gps_crs, args.window)
    tif_path = '/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/gascola.tif'

    #TODO
    node = AngularCoverageViz(tif_path, args.topic, args.n_bins,
                              args.gps_topic, args.gps_crs, args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

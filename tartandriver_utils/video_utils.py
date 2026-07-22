import glob
import math
import multiprocessing as mp
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import tqdm
import yaml
from PIL import Image, ImageDraw, ImageFont

GRAVITY_MPS2 = 9.80665

def load_video_config(path):
    """Load + normalize a video render config describing *what* to render.
    """
    cfg = {}
    if path:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

    hud = None
    hud_cfg = cfg.get("hud")
    if hud_cfg and hud_cfg.get("enabled", True):
        orient_cfg = hud_cfg.get("orient") or {}
        hud = {
            "odom_kitti_dir": hud_cfg.get("odom_kitti_dir", "sensors/novatel_gps_odom"),
            "imu_kitti_dir": hud_cfg.get("imu_kitti_dir", "sensors/novatel_imu"),
            "gmeter_max_g": hud_cfg.get("gmeter_max_g", 1.0),
            "gmeter_trail": hud_cfg.get("gmeter_trail", 15),
            "orient": {"enabled": orient_cfg.get("enabled", True)},
            "map_tif": hud_cfg.get("map_tif"),
            "map_source": hud_cfg.get("map_source", "auto"),
            "allow_network_tiles": hud_cfg.get("allow_network_tiles", False),
            "workers": hud_cfg.get("workers"),
        }

    return {"hud": hud, "pip": cfg.get("pip", []) or []}


def _load_interp_rbstate(dataset_dir, sub_dir):
    """Load an INTERP-type RBState's `interp_data.txt`/`interp_timestamps.txt`
    (13 cols: x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz). None if the dir is missing.
    """
    base_dir = os.path.join(dataset_dir, sub_dir)
    data_fp = os.path.join(base_dir, "interp_data.txt")
    ts_fp = os.path.join(base_dir, "interp_timestamps.txt")
    if not (os.path.exists(data_fp) and os.path.exists(ts_fp)):
        return None
    data = np.loadtxt(data_fp).reshape(-1, 13)
    times = np.atleast_1d(np.loadtxt(ts_fp)).astype(np.float64)
    return data, times


def _is_frozen_pose(data, tol=1e-3):
    """True if position+orientation never change -- e.g. GPS never got a fix
    (common under dense canopy for this off-road stack), as opposed to a
    genuinely stationary vehicle, which still has sensor-noise-scale jitter."""
    if len(data) < 2:
        return True
    return bool(np.ptp(data[:, :7], axis=0).max() < tol)


def collect_video_hud_odom(dataset_dir, odom_dir="sensors/novatel_gps_odom",
                           fallback_odom_dir="super_odometry/odometry"):
    """
    Collect meter-frame odometry and speed for HUD overlays.
    """
    empty = {
        "gps_xy": np.zeros((0, 2), dtype=np.float64),
        "gps_times": np.zeros((0,), dtype=np.float64),
        "speed_mps": np.zeros((0,), dtype=np.float64),
        "speed_times": np.zeros((0,), dtype=np.float64),
        "quat_xyzw": np.zeros((0, 4), dtype=np.float64),
        "quat_times": np.zeros((0,), dtype=np.float64),
        "bag_start_time": np.nan,
    }

    loaded = _load_interp_rbstate(dataset_dir, odom_dir)
    if loaded is None:
        print(f"  [hud] odom dir not found: {os.path.join(dataset_dir, odom_dir)}")
    elif _is_frozen_pose(loaded[0]):
        print(f"  [hud] {odom_dir} position+orientation never change (GPS likely never got a "
              f"fix); falling back to {fallback_odom_dir}")
        loaded = None

    if loaded is None and fallback_odom_dir:
        loaded = _load_interp_rbstate(dataset_dir, fallback_odom_dir)
        if loaded is None:
            print(f"  [hud] fallback odom dir not found: {os.path.join(dataset_dir, fallback_odom_dir)}; "
                  "rendering timestamp-only HUD")

    if loaded is None:
        return empty

    data, times = loaded
    return {
        "gps_xy": data[:, :2],
        "gps_times": times,
        "speed_mps": np.linalg.norm(data[:, 7:10], axis=1),
        "speed_times": times,
        "quat_xyzw": data[:, 3:7],  # (x, y, z, w) vehicle orientation
        "quat_times": times,
        "bag_start_time": np.nan,
    }


def collect_video_hud_imu(dataset_dir, imu_dir="sensors/novatel_imu"):
    """
    Collect body-frame linear acceleration (x, y) for the HUD g-meter

    Imu is also TimeSpec.INTERP -- see the note in collect_video_hud_odom for why
    this must read interp_data.txt/interp_timestamps.txt rather than data.txt/
    timestamps.txt.
    """
    base_dir = os.path.join(dataset_dir, imu_dir)
    data_fp = os.path.join(base_dir, "interp_data.txt")
    ts_fp = os.path.join(base_dir, "interp_timestamps.txt")

    if not (os.path.exists(data_fp) and os.path.exists(ts_fp)):
        print(f"  [hud] imu dir not found: {base_dir}; g-meter HUD disabled")
        return {
            "accel_xy": np.zeros((0, 2), dtype=np.float64),
            "times": np.zeros((0,), dtype=np.float64),
        }

    data = np.loadtxt(data_fp).reshape(-1, 10)
    times = np.atleast_1d(np.loadtxt(ts_fp)).astype(np.float64)

    return {
        "accel_xy": data[:, 7:9],
        "times": times,
    }


def _as_map_size(map_size):
    if isinstance(map_size, int):
        return map_size, map_size
    return int(map_size[0]), int(map_size[1])


def _project_route(gps_xy, map_size=(165, 165), padding=12):
    """
    Project xy meter-frame coordinates into minimap pixel coordinates.
    """
    map_w, map_h = _as_map_size(map_size)
    gps_xy = np.asarray(gps_xy, dtype=np.float64)
    if gps_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64), 1.0, np.array([0.0, 0.0])

    mins = np.nanmin(gps_xy, axis=0)
    maxs = np.nanmax(gps_xy, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    usable_w = max(map_w - 2 * padding, 1)
    usable_h = max(map_h - 2 * padding, 1)
    scale = min(usable_w / span[0], usable_h / span[1])

    route_w = span[0] * scale
    route_h = span[1] * scale
    offset = np.array([
        (map_w - route_w) * 0.5 - mins[0] * scale,
        (map_h + route_h) * 0.5 + mins[1] * scale,
    ])
    pixels = _apply_route_projection(gps_xy, scale, offset)
    return pixels, scale, offset


def _apply_route_projection(xy, scale, offset):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pixels = np.empty((xy.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = xy[:, 0] * scale + offset[0]
    pixels[:, 1] = -xy[:, 1] * scale + offset[1]
    return pixels


def _load_default_font(size=20, bold=False):
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ])
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _resolve_map_tif(map_tif):
    if map_tif is None:
        return None

    path = Path(map_tif).expanduser()
    if path.exists():
        return str(path)

    if not path.is_absolute():
        candidates = []
        try:
            candidates.append(Path(__file__).resolve().parents[4] / path)
        except IndexError:
            pass
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / path)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    print(f"  [overlay] GeoTIFF path not found: {map_tif}")
    return str(path)


def _project_route_on_tif(gps_xy, frame_xy, map_tif, map_size):
    import rasterio
    from rasterio.transform import rowcol

    map_w, map_h = _as_map_size(map_size)
    with rasterio.open(map_tif) as tif:
        rgb_map = tif.read([1, 2, 3])
        rgb_map = np.transpose(rgb_map, [1, 2, 0])

        rows, cols = rowcol(tif.transform, -gps_xy[:, 1], gps_xy[:, 0])
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)

        valid = (
            np.isfinite(rows)
            & np.isfinite(cols)
            & (rows >= 0)
            & (cols >= 0)
            & (rows < rgb_map.shape[0])
            & (cols < rgb_map.shape[1])
        )
        if valid.sum() < 2:
            raise ValueError(
                "not enough route points inside GeoTIFF bounds "
                f"({valid.sum()}/{len(rows)} inside)"
            )

        rows_valid = rows[valid]
        cols_valid = cols[valid]
        center_r = 0.5 * (rows_valid.min() + rows_valid.max())
        center_c = 0.5 * (cols_valid.min() + cols_valid.max())
        side = max(rows_valid.max() - rows_valid.min(), cols_valid.max() - cols_valid.min()) + 100.0
        side = max(side, 1.0)

        r0 = int(math.floor(center_r - side * 0.5))
        r1 = int(math.ceil(center_r + side * 0.5))
        c0 = int(math.floor(center_c - side * 0.5))
        c1 = int(math.ceil(center_c + side * 0.5))

        pad_top = max(0, -r0)
        pad_left = max(0, -c0)
        pad_bottom = max(0, r1 - rgb_map.shape[0])
        pad_right = max(0, c1 - rgb_map.shape[1])

        r0_clip = max(0, r0)
        r1_clip = min(rgb_map.shape[0], r1)
        c0_clip = max(0, c0)
        c1_clip = min(rgb_map.shape[1], c1)
        crop = rgb_map[r0_clip:r1_clip, c0_clip:c1_clip]
        if crop.size == 0:
            raise ValueError("GeoTIFF crop is empty")

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            crop = np.pad(
                crop,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

        frame_rows = None
        frame_cols = None
        if frame_xy is not None and len(frame_xy):
            frame_rows, frame_cols = rowcol(tif.transform, -frame_xy[:, 1], frame_xy[:, 0])

    bg = Image.fromarray(crop.astype(np.uint8), mode="RGB")
    crop_w, crop_h = bg.size
    bg = bg.resize((map_w, map_h), Image.Resampling.LANCZOS).convert("RGBA")

    scale_x = map_w / max(crop_w, 1)
    scale_y = map_h / max(crop_h, 1)
    route_px = np.column_stack([
        (cols - c0) * scale_x,
        (rows - r0) * scale_y,
    ])

    frame_px = None
    if frame_rows is not None and frame_cols is not None:
        frame_px = np.column_stack([
            (np.asarray(frame_cols, dtype=np.float64) - c0) * scale_x,
            (np.asarray(frame_rows, dtype=np.float64) - r0) * scale_y,
        ])

    # The rowcol() call above maps odom +x -> geographic north (odom +y -> west),
    return bg, route_px, frame_px, True


def _load_minimap_background(
    gps_xy,
    map_tif=None,
    map_source="auto",
    map_size=(165, 165),
    cache_dir=None,
    allow_network_tiles=False,
    frame_xy=None,
):
    """
    Return a static minimap background plus route/current-frame pixel coordinates.
    """
    map_w, map_h = _as_map_size(map_size)
    gps_xy = np.asarray(gps_xy, dtype=np.float64)
    frame_xy = None if frame_xy is None else np.asarray(frame_xy, dtype=np.float64)
    map_source = (map_source or "auto").lower()

    if map_source in ("auto", "tif") and map_tif:
        map_tif = _resolve_map_tif(map_tif)
        try:
            print(f"  [overlay] using GeoTIFF minimap: {map_tif}")
            return _project_route_on_tif(gps_xy, frame_xy, map_tif, (map_w, map_h))
        except Exception as exc:
            print(f"  [overlay] GeoTIFF minimap unavailable ({exc}); falling back to track view.")
    elif map_source in ("auto", "tif"):
        print("  [overlay] no GeoTIFF map passed; using track view minimap.")

    if map_source == "osm":
        if not allow_network_tiles:
            print("  [overlay] OSM requested but --allow_network_tiles is false; falling back to track view.")
        else:
            print("  [overlay] OSM tiles need lat/lon data; falling back to meter-frame track view.")

    bg = Image.new("RGBA", (map_w, map_h), (26, 26, 46, 235))
    route_px, scale, offset = _project_route(gps_xy, (map_w, map_h), padding=14)
    frame_px = None
    if frame_xy is not None and len(frame_xy):
        frame_px = _apply_route_projection(frame_xy, scale, offset)
    return bg, route_px, frame_px, False


def _draw_gmeter(draw, cx, cy, radius, trail_ax_g, trail_ay_g, max_g, font_label):
    """Circular accelerometer g-meter with a fixed scale and a fading trail.

    trail_ax_g/trail_ay_g are sequences of recent longitudinal/lateral g samples
    (oldest -> newest; the last element is the current frame), REP-103 body frame
    (+ax forward, +ay left). The dial has fixed rings at each integer g up to
    max_g -- the only numbers shown are those fixed limits (no dynamic readouts).
    The current accel vector (ay, ax) is a solid dot; older samples fade out.
    """
    minor_ring = (255, 255, 255, 30)
    major_ring = (255, 255, 255, 78)
    spoke_col = (255, 255, 255, 26)
    axis_col = (255, 255, 255, 70)
    diag = 0.70710678  # cos/sin 45deg

    # Radial spokes (8 directions) + brighter primary cross, for angular reference.
    for dx, dy in [(diag, diag), (diag, -diag)]:
        draw.line([cx - dx * radius, cy - dy * radius, cx + dx * radius, cy + dy * radius],
                  fill=spoke_col, width=1)
    draw.line([cx - radius, cy, cx + radius, cy], fill=axis_col, width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=axis_col, width=1)

    # Concentric rings every 0.5g: minor faint (inner circles), integer g major.
    g = 0.5
    while g < max_g - 1e-6:
        rr = radius * (g / max_g)
        is_major = abs(g - round(g)) < 1e-6
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     outline=major_ring if is_major else minor_ring, width=1)
        if is_major:  # label sits just outside its ring on the up-right diagonal
            lr = rr + 8
            draw.text((cx + lr * diag, cy - lr * diag), f"{g:g}g", font=font_label,
                      fill=(195, 195, 195, 210), anchor="mm")
        g += 0.5

    # Outer full-scale ring + label just outside it.
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=major_ring, width=1)
    lr = radius + 9
    draw.text((cx + lr * diag, cy - lr * diag), f"{max_g:g}g", font=font_label,
              fill=(215, 215, 215, 225), anchor="mm")

    # Tick marks around the outer ring every 30deg.
    for k in range(12):
        a = math.radians(30 * k)
        ca, sa = math.cos(a), math.sin(a)
        draw.line([cx + radius * ca, cy - radius * sa,
                   cx + (radius - 4) * ca, cy - (radius - 4) * sa],
                  fill=axis_col, width=1)

    draw.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=(255, 255, 255, 120))

    trail_ax_g = np.atleast_1d(np.asarray(trail_ax_g, dtype=np.float64))
    trail_ay_g = np.atleast_1d(np.asarray(trail_ay_g, dtype=np.float64))
    n = int(min(len(trail_ax_g), len(trail_ay_g)))
    for i in range(n):
        ax_g = trail_ax_g[i]
        ay_g = trail_ay_g[i]
        if not (np.isfinite(ax_g) and np.isfinite(ay_g)):
            continue
        mag = math.hypot(ax_g, ay_g)
        scale = radius / max_g if mag <= max_g else radius / max(mag, 1e-6)
        dot_x = cx - ay_g * scale
        dot_y = cy - ax_g * scale
        frac = (i + 1) / n  # oldest -> newest
        if i == n - 1:
            r = 5
            draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r],
                         fill=(255, 90, 90, 235), outline=(0, 0, 0, 180), width=1)
        else:
            r = 1.5 + 2.5 * frac
            alpha = int(30 + 150 * frac)
            draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r],
                         fill=(255, 140, 120, alpha))


_VEHICLE_MESH_CACHE = {}


def _make_box(x0, x1, y0, y1, z0, z1):
    """8-vertex box with 12 outward-wound triangles (forward=+x, left=+y, up=+z)."""
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    f = np.array([
        [0, 3, 2], [0, 2, 1],   # bottom (-z)
        [4, 5, 6], [4, 6, 7],   # top (+z)
        [0, 1, 5], [0, 5, 4],   # -y
        [3, 6, 2], [3, 7, 6],   # +y  (fixed winding)
        [1, 2, 6], [1, 6, 5],   # +x
        [0, 4, 7], [0, 7, 3],   # -x
    ], dtype=np.int64)
    return v, f


def _make_wheel(cx, cy, cz, r, hw, nseg=10):
    """Octagon-ish prism (axle along y) for a wheel."""
    ang = np.linspace(0.0, 2 * np.pi, nseg, endpoint=False)
    ring = np.column_stack([np.cos(ang) * r, np.zeros(nseg), np.sin(ang) * r])
    left = ring + np.array([cx, cy + hw, cz])
    right = ring + np.array([cx, cy - hw, cz])
    v = np.vstack([left, right])
    f = []
    for i in range(nseg):
        j = (i + 1) % nseg
        f.append([i, j, nseg + j])
        f.append([i, nseg + j, nseg + i])
    for i in range(1, nseg - 1):
        f.append([0, i, i + 1])                    # left cap
        f.append([nseg, nseg + i + 1, nseg + i])   # right cap
    return v, np.array(f, dtype=np.int64)


def _stylized_vehicle_mesh():
    """A clean low-poly car (chassis + cabin + 4 wheels), forward=+x, with
    per-face colors. Reads far better than a decimated photogrammetry scan."""
    parts_v, parts_f, parts_c = [], [], []
    offset = 0

    def add(v, f, color):
        nonlocal offset
        parts_v.append(v)
        parts_f.append(f + offset)
        parts_c.append(np.tile(np.asarray(color, dtype=np.float64), (len(f), 1)))
        offset += len(v)

    body_c = (64, 122, 192)
    cabin_c = (150, 186, 226)
    wheel_c = (38, 38, 46)

    add(*_make_box(-1.0, 1.0, -0.5, 0.5, 0.18, 0.55), color=body_c)      # chassis
    add(*_make_box(-0.55, 0.35, -0.42, 0.42, 0.55, 0.92), color=cabin_c)  # cabin (rear-biased)
    add(*_make_box(0.72, 1.02, -0.46, 0.46, 0.22, 0.42), color=body_c)    # hood/nose (front stub)
    for wx, wy in [(0.62, 0.52), (0.62, -0.52), (-0.62, 0.52), (-0.62, -0.52)]:
        add(*_make_wheel(wx, wy, 0.16, 0.34, 0.14), color=wheel_c)

    verts = np.vstack(parts_v)
    faces = np.vstack(parts_f)
    face_colors = np.vstack(parts_c)
    return verts, faces, face_colors


def _load_vehicle_mesh():
    """Return the built-in low-poly car mesh (verts + faces + per-face colors),
    normalized to a unit half-extent in the REP-103 body frame (+x fwd, +y left,
    +z up). Cached per process."""
    if "mesh" in _VEHICLE_MESH_CACHE:
        return _VEHICLE_MESH_CACHE["mesh"]

    verts, faces_arr, face_colors = _stylized_vehicle_mesh()
    verts = verts - verts.mean(axis=0)
    scale = float(np.max(np.abs(verts)))
    if scale > 1e-9:
        verts = verts / scale

    mesh = {"verts": verts, "faces": faces_arr, "face_colors": face_colors}
    _VEHICLE_MESH_CACHE["mesh"] = mesh
    return mesh


def _camera_basis(elev_deg=22.0, azim_deg=-55.0):
    """Fixed 3rd-person camera basis in world (REP-103) coords. Returns
    (right, up, toward_viewer) unit vectors for an orthographic projection."""
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    cam_z = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-9)
    right = np.cross(np.array([0.0, 0.0, 1.0]), cam_z)
    rn = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if rn < 1e-6 else right / rn
    cam_up = np.cross(cam_z, right)
    return right, cam_up, cam_z


def _draw_vehicle_orientation(draw, cx, cy, size, mesh, R_body,
                              elev_deg=24.0, azim_deg=128.0, ref_label="start"):
    """Draw a small flat-shaded 3rd-person vehicle at (cx, cy) within a box of
    half-width `size`, oriented by world_from_body rotation R_body, resting on a
    fixed ground plane, with colored body axes (X=red, Y=green, Z=blue).

    The camera sits behind-and-above the vehicle (chase view), so vehicle-forward
    (+x, red axis) points *into* the screen -- a forward-driving car heads away
    from the viewer, matching the forward-facing camera feed the HUD overlays.

    A fixed arrow marks world +x on the ground: "N" (true north) when the caller
    has a georeferenced minimap to confirm odom +x is north, else "start" (the
    trajectory's initial heading, with R_body pre-referenced to it)."""
    right, cam_up, cam_z = _camera_basis(elev_deg, azim_deg)
    pix = size * 0.44

    def project(pts_world):
        pts = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
        sx = pts @ right
        sy = pts @ cam_up
        depth = pts @ cam_z
        px = cx + sx * pix
        py = cy - sy * pix
        return px, py, depth

    verts = mesh["verts"]
    faces = mesh["faces"]
    verts_w = verts @ np.asarray(R_body, dtype=np.float64).T

    # --- Ground plane at the vehicle's lowest point (world-horizontal) ---
    ground_z = float(verts_w[:, 2].min())
    ext = 1.7
    n_grid = 4
    plane_fill = (70, 90, 120, 90)
    grid_col = (150, 170, 200, 110)
    corners = np.array([
        [-ext, -ext, ground_z], [ext, -ext, ground_z],
        [ext, ext, ground_z], [-ext, ext, ground_z],
    ])
    cpx, cpy, _ = project(corners)
    draw.polygon(list(zip(cpx, cpy)), fill=plane_fill)
    for i in range(n_grid + 1):
        t = -ext + 2 * ext * i / n_grid
        ax0, ay0, _ = project([[t, -ext, ground_z]])
        ax1, ay1, _ = project([[t, ext, ground_z]])
        draw.line([ax0[0], ay0[0], ax1[0], ay1[0]], fill=grid_col, width=1)
        bx0, by0, _ = project([[-ext, t, ground_z]])
        bx1, by1, _ = project([[ext, t, ground_z]])
        draw.line([bx0[0], by0[0], bx1[0], by1[0]], fill=grid_col, width=1)

    # Fixed reference arrow on the ground at world +x (true north if ref_label is
    # "N", else the trajectory's start heading). The plane/arrow don't move, so
    # the car visibly turns against them.
    ref_col = (255, 210, 90, 200) if ref_label == "N" else (110, 195, 255, 165)
    rx0, ry0, _ = project([[0.15, 0.0, ground_z]])
    rx1, ry1, _ = project([[1.55, 0.0, ground_z]])
    draw.line([rx0[0], ry0[0], rx1[0], ry1[0]], fill=ref_col, width=2)
    hxl, hyl, _ = project([[1.2, 0.2, ground_z]])
    hxr, hyr, _ = project([[1.2, -0.2, ground_z]])
    draw.polygon([(rx1[0], ry1[0]), (hxl[0], hyl[0]), (hxr[0], hyr[0])], fill=ref_col)
    lx, ly, _ = project([[1.55, 0.0, ground_z]])
    draw.text((lx[0] - 16, ly[0]), ref_label, fill=ref_col, anchor="mm")

    # --- Vehicle: depth-sorted painter's fill with per-face color + Lambert shading ---
    px, py, depth = project(verts_w)
    light = np.array([-0.35, -0.45, 0.82])
    light = light / np.linalg.norm(light)
    v0 = verts_w[faces[:, 0]]
    v1 = verts_w[faces[:, 1]]
    v2 = verts_w[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nnorm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(nnorm, 1e-9, None)
    inten = np.abs(normals @ light)
    ambient = 0.45
    shade = ambient + (1.0 - ambient) * inten
    face_colors = mesh.get("face_colors")
    base = np.array([200.0, 206.0, 214.0])
    edge = (24, 26, 32, 200)
    face_depth = depth[faces].mean(axis=1)
    order = np.argsort(face_depth)  # far (smaller depth) first
    for fi in order:
        tri = faces[fi]
        c = base if face_colors is None else face_colors[fi]
        col = tuple(int(x) for x in np.clip(c * shade[fi], 0, 255)) + (240,)
        draw.polygon([(px[tri[0]], py[tri[0]]),
                      (px[tri[1]], py[tri[1]]),
                      (px[tri[2]], py[tri[2]])], fill=col, outline=edge)

    # --- Colored body axes from the vehicle origin ---
    ox, oy, _ = project([[0.0, 0.0, 0.0]])
    axis_len = 1.35
    axes = [
        (np.array([axis_len, 0.0, 0.0]), (255, 80, 80, 255)),   # X forward - red
        (np.array([0.0, axis_len, 0.0]), (90, 230, 110, 255)),  # Y left - green
        (np.array([0.0, 0.0, axis_len]), (90, 160, 255, 255)),  # Z up - blue
    ]
    for vec, color in axes:
        ex, ey, _ = project([R_body @ vec])
        draw.line([ox[0], oy[0], ex[0], ey[0]], fill=color, width=2)
        draw.ellipse([ex[0] - 2.5, ey[0] - 2.5, ex[0] + 2.5, ey[0] + 2.5], fill=color)


def _draw_polyline(draw, points, fill, width=2):
    points = np.asarray(points)
    if len(points) < 2:
        return
    finite = np.isfinite(points).all(axis=1)
    if finite.sum() < 2:
        return
    draw.line([tuple(p) for p in points[finite]], fill=fill, width=width, joint="curve")


def render_overlay_frame(args_tuple):
    (
        frame_idx,
        speed_mps,
        wall_time_str,
        trail_ax_g,
        trail_ay_g,
        gmeter_max_g,
        R_body,
        vehicle_mesh,
        ref_label,
        map_background,
        route_px,
        frame_px,
        canvas_size,
        map_size,
        out_path,
    ) = args_tuple

    canvas_w, canvas_h = canvas_size
    map_w, map_h = _as_map_size(map_size)
    margin = 16
    panel_pad = 10

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_time = _load_default_font(16)
    font_speed = _load_default_font(26, bold=True)
    font_label = _load_default_font(13)

    # --- Bottom-left: Minimap ---
    map_panel_w = map_w + 2 * panel_pad
    map_panel_h = map_h + 2 * panel_pad
    map_panel_x = margin
    map_panel_y = canvas_h - map_panel_h - margin
    map_x = map_panel_x + panel_pad
    map_y = map_panel_y + panel_pad

    draw.rounded_rectangle(
        [map_panel_x, map_panel_y, map_panel_x + map_panel_w, map_panel_y + map_panel_h],
        radius=8, fill=(0, 0, 0, 120), outline=(255, 255, 255, 45), width=1,
    )

    minimap = map_background.copy()
    map_draw = ImageDraw.Draw(minimap)
    map_draw.rectangle([0, 0, map_w - 1, map_h - 1], outline=(255, 255, 255, 70), width=1)
    _draw_polyline(map_draw, route_px, fill=(255, 255, 255, 170), width=2)
    if frame_px is not None and len(frame_px):
        upto = min(frame_idx + 1, len(frame_px))
        _draw_polyline(map_draw, frame_px[:upto], fill=(84, 210, 255, 235), width=3)
        pos = frame_px[min(frame_idx, len(frame_px) - 1)]
        if np.isfinite(pos).all():
            radius = 6
            map_draw.ellipse(
                [pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius],
                fill=(255, 218, 66, 255), outline=(0, 0, 0, 180), width=2,
            )
            prev_idx = max(0, min(frame_idx, len(frame_px) - 1) - 3)
            delta = pos - frame_px[prev_idx]
            if np.linalg.norm(delta) > 1e-3 and np.isfinite(delta).all():
                direction = delta / np.linalg.norm(delta)
                end = pos + direction * 18
                map_draw.line([tuple(pos), tuple(end)], fill=(255, 218, 66, 255), width=3)

    img.alpha_composite(minimap, dest=(map_x, map_y))

    # --- Bottom-right: Info panel (time + speed + g-meter + orientation) ---
    gm_radius = 38
    gm_diameter = 2 * gm_radius
    gm_label_w_side = 36
    gm_label_h_vert = 14

    gmeter_block_w = gm_diameter + 2 * gm_label_w_side
    gmeter_block_h = gm_diameter + 2 * gm_label_h_vert

    trail_ax_g = np.atleast_1d(np.asarray(trail_ax_g, dtype=np.float64))
    trail_ay_g = np.atleast_1d(np.asarray(trail_ay_g, dtype=np.float64))
    has_imu = (
        len(trail_ax_g) and len(trail_ay_g)
        and np.isfinite(trail_ax_g[-1]) and np.isfinite(trail_ay_g[-1])
    )
    has_orient = R_body is not None and vehicle_mesh is not None
    orient_size = 46
    orient_block_h = 2 * orient_size + 6

    text_block_w = 200
    inner_h = 20 + 4 + 32  # time + gap + speed
    if has_imu:
        inner_h += 8 + gmeter_block_h
    if has_orient:
        inner_h += 8 + orient_block_h

    info_panel_w = max(text_block_w, gmeter_block_w, 2 * orient_size) + 2 * panel_pad
    info_panel_h = inner_h + 2 * panel_pad
    info_panel_x = canvas_w - info_panel_w - margin
    info_panel_y = canvas_h - info_panel_h - margin

    draw.rounded_rectangle(
        [info_panel_x, info_panel_y, info_panel_x + info_panel_w, info_panel_y + info_panel_h],
        radius=8, fill=(0, 0, 0, 150), outline=(255, 255, 255, 45), width=1,
    )

    cx = info_panel_x + panel_pad
    cy = info_panel_y + panel_pad

    draw.text((cx, cy), wall_time_str, font=font_time, fill=(200, 200, 200, 220))
    cy += 24

    speed_text = f"{speed_mps:.2f} m/s" if np.isfinite(speed_mps) else "--.- m/s"
    draw.text((cx, cy), speed_text, font=font_speed, fill=(255, 255, 255, 245))
    cy += 36

    if has_imu:
        cy += 8
        gm_cx = info_panel_x + info_panel_w / 2
        gm_cy = cy + gm_label_h_vert + gm_radius
        _draw_gmeter(draw, gm_cx, gm_cy, gm_radius, trail_ax_g, trail_ay_g, gmeter_max_g, font_label)
        cy += gmeter_block_h

    if has_orient:
        cy += 8
        ov_cx = info_panel_x + info_panel_w / 2
        ov_cy = cy + orient_size
        _draw_vehicle_orientation(draw, ov_cx, ov_cy, orient_size, vehicle_mesh, R_body,
                                  ref_label=ref_label)

    img.save(out_path)
    return out_path


def render_overlay_frames_parallel(
    frame_ts,
    gps_xy,
    gps_times,
    speed_mps,
    speed_times,
    bag_start_time,
    overlay_dir,
    canvas_size,
    map_size=(165, 165),
    n_workers=None,
    map_tif=None,
    map_source="auto",
    allow_network_tiles=False,
    accel_xy=None,
    accel_times=None,
    gmeter_max_g=1.0,
    gmeter_trail=15,
    quat_xyzw=None,
    quat_times=None,
    orient_cfg=None,
):
    frame_ts = np.asarray(frame_ts, dtype=np.float64)
    gps_xy = np.asarray(gps_xy, dtype=np.float64)
    gps_times = np.asarray(gps_times, dtype=np.float64)
    speed_mps = np.asarray(speed_mps, dtype=np.float64)
    speed_times = np.asarray(speed_times, dtype=np.float64)

    if len(frame_ts) == 0:
        return None

    os.makedirs(overlay_dir, exist_ok=True)
    if len(gps_xy) >= 2 and len(gps_times) >= 2:
        frame_x = np.interp(frame_ts, gps_times, gps_xy[:, 0])
        frame_y = np.interp(frame_ts, gps_times, gps_xy[:, 1])
        frame_xy = np.column_stack([frame_x, frame_y])
        bg, route_px, frame_px, north_available = _load_minimap_background(
            gps_xy,
            map_tif=map_tif,
            map_source=map_source,
            map_size=map_size,
            cache_dir=os.path.join(os.path.dirname(overlay_dir), "map_cache"),
            allow_network_tiles=allow_network_tiles,
            frame_xy=frame_xy,
        )
    else:
        map_w, map_h = _as_map_size(map_size)
        bg = Image.new("RGBA", (map_w, map_h), (26, 26, 46, 235))
        route_px = np.zeros((0, 2), dtype=np.float64)
        frame_px = None
        north_available = False

    if len(speed_mps) >= 1 and len(speed_times) >= 1:
        frame_speed = np.interp(frame_ts, speed_times, speed_mps)
    else:
        frame_speed = np.full(len(frame_ts), np.nan)

    has_imu = (
        accel_xy is not None and len(accel_xy) >= 2
        and accel_times is not None and len(accel_times) >= 2
    )
    if has_imu:
        accel_xy = np.asarray(accel_xy, dtype=np.float64)
        accel_times = np.asarray(accel_times, dtype=np.float64)
        frame_ax_g = np.interp(frame_ts, accel_times, accel_xy[:, 0]) / GRAVITY_MPS2
        frame_ay_g = np.interp(frame_ts, accel_times, accel_xy[:, 1]) / GRAVITY_MPS2
    else:
        frame_ax_g = np.full(len(frame_ts), np.nan)
        frame_ay_g = np.full(len(frame_ts), np.nan)

    # Per-frame vehicle orientation (world_from_body rotation matrices) + mesh.
    orient_cfg = orient_cfg or {}
    orient_enabled = orient_cfg.get("enabled", True)
    has_orient = (
        orient_enabled
        and quat_xyzw is not None and len(quat_xyzw) >= 1
        and quat_times is not None and len(quat_times) >= 1
    )
    frame_R = None
    vehicle_mesh = None
    ref_label = "start"
    if has_orient:
        from scipy.spatial.transform import Rotation as R
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
        quat_times = np.asarray(quat_times, dtype=np.float64)
        # nlerp: component-wise interpolate then renormalize (fine for small steps).
        fq = np.column_stack([
            np.interp(frame_ts, quat_times, quat_xyzw[:, k]) for k in range(4)
        ])
        norms = np.linalg.norm(fq, axis=1, keepdims=True)
        fq = fq / np.clip(norms, 1e-9, None)
        frame_R = R.from_quat(fq).as_matrix()  # (N, 3, 3)
        if north_available:
            # GeoTIFF minimap resolved -> gps_xy
            ref_label = "N"
        else:
            # No north reference: treat the trajectory's starting heading as forward.
            yaw0 = math.atan2(frame_R[0][1, 0], frame_R[0][0, 0])
            c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
            Rz0 = np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]])
            frame_R = Rz0 @ frame_R
            ref_label = "start"
        vehicle_mesh = _load_vehicle_mesh()

    if not np.isfinite(bag_start_time):
        bag_start_time = frame_ts[0]
    wall_times = bag_start_time + (frame_ts - frame_ts[0])
    wall_time_strs = [
        datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        for t in wall_times
    ]

    trail = max(1, int(gmeter_trail))
    jobs = []
    for idx in range(len(frame_ts)):
        lo = max(0, idx - trail + 1)
        jobs.append((
            idx,
            float(frame_speed[idx]),
            wall_time_strs[idx],
            frame_ax_g[lo:idx + 1].copy(),
            frame_ay_g[lo:idx + 1].copy(),
            gmeter_max_g,
            frame_R[idx] if frame_R is not None else None,
            vehicle_mesh,
            ref_label,
            bg,
            route_px,
            frame_px,
            tuple(canvas_size),
            map_size,
            os.path.join(overlay_dir, f"{idx:08d}.png"),
        ))

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), len(jobs)))
    if n_workers == 1:
        for job in tqdm.tqdm(jobs, desc="overlays", unit="frame"):
            render_overlay_frame(job)
    else:
        with mp.Pool(n_workers) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(render_overlay_frame, jobs), total=len(jobs), desc="overlays", unit="frame"):
                pass
    return overlay_dir


def render_kitti_video_with_hud(
    frames_dir,
    output_path=None,
    fps=None,
    gps_xy=None,
    gps_times=None,
    speed_mps=None,
    speed_times=None,
    bag_start_time=np.nan,
    overlay_root=None,
    map_tif=None,
    map_source="auto",
    allow_network_tiles=False,
    n_workers=None,
    cleanup_overlay_dir=True,
    accel_xy=None,
    accel_times=None,
    gmeter_max_g=1.0,
    gmeter_trail=15,
    quat_xyzw=None,
    quat_times=None,
    orient_cfg=None,
    pip=None,
):
    """
    Render an existing KITTI image directory with timestamp/speed/minimap HUD,
    a g-meter, a 3D vehicle-orientation widget, and optional picture-in-picture
    insets (see `render_kitti_video`).
    """
    def _render_without_hud():
        return render_kitti_video(frames_dir, output_path=output_path, fps=fps, pip=pip)

    pngs = sorted(glob.glob(os.path.join(frames_dir, "????????.png")))
    if not pngs:
        return _render_without_hud()

    timestamps_fp = os.path.join(frames_dir, "timestamps.txt")
    if os.path.exists(timestamps_fp):
        frame_ts = np.loadtxt(timestamps_fp)
        frame_ts = np.atleast_1d(frame_ts).astype(np.float64)
    else:
        if fps is None or fps <= 0:
            fps = 10.0
        frame_ts = np.arange(len(pngs), dtype=np.float64) / fps

    if len(frame_ts) != len(pngs):
        print(
            f"  [overlay] timestamp count mismatch for {frames_dir} "
            f"({len(frame_ts)} timestamps for {len(pngs)} frames); rendering without HUD."
        )
        return _render_without_hud()

    with Image.open(pngs[0]) as img:
        canvas_size = img.size

    if output_path is None:
        output_path = frames_dir.rstrip(os.sep) + ".mp4"

    if overlay_root is None:
        overlay_root = os.path.join(os.path.dirname(output_path), "overlays")
    overlay_name = os.path.splitext(os.path.basename(output_path))[0]
    overlay_dir = os.path.join(overlay_root, overlay_name)

    gps_xy = np.zeros((0, 2), dtype=np.float64) if gps_xy is None else np.asarray(gps_xy, dtype=np.float64)
    gps_times = np.zeros((0,), dtype=np.float64) if gps_times is None else np.asarray(gps_times, dtype=np.float64)
    speed_mps = np.zeros((0,), dtype=np.float64) if speed_mps is None else np.asarray(speed_mps, dtype=np.float64)
    speed_times = np.zeros((0,), dtype=np.float64) if speed_times is None else np.asarray(speed_times, dtype=np.float64)

    render_overlay_frames_parallel(
        frame_ts,
        gps_xy,
        gps_times,
        speed_mps,
        speed_times,
        bag_start_time,
        overlay_dir,
        canvas_size,
        n_workers=n_workers,
        map_tif=map_tif,
        map_source=map_source,
        allow_network_tiles=allow_network_tiles,
        accel_xy=accel_xy,
        accel_times=accel_times,
        gmeter_max_g=gmeter_max_g,
        gmeter_trail=gmeter_trail,
        quat_xyzw=quat_xyzw,
        quat_times=quat_times,
        orient_cfg=orient_cfg,
    )
    rendered = render_kitti_video(frames_dir, output_path=output_path, fps=fps, overlay_dir=overlay_dir, pip=pip)
    if rendered and cleanup_overlay_dir and os.path.isdir(overlay_dir):
        shutil.rmtree(overlay_dir)
        try:
            os.rmdir(overlay_root)
        except OSError:
            pass
        print(f"  [overlay] deleted temporary overlays: {overlay_dir}")
    return rendered


def _pip_overlay_xy(pos, margin, cum):
    """ffmpeg overlay x:y expressions for a PiP inset at the given corner.

    `cum` is the horizontal pixel offset already consumed by previously placed
    insets sharing this corner, so multiple insets stack along the edge."""
    m = margin
    if pos == "bottom-left":
        return f"{m + cum}", f"main_h-overlay_h-{m}"
    if pos == "top-right":
        return f"main_w-overlay_w-{m + cum}", f"{m}"
    if pos == "top-left":
        return f"{m + cum}", f"{m}"
    if pos == "bottom-center":
        return f"(main_w-overlay_w)/2+{cum}", f"main_h-overlay_h-{m}"
    # default: bottom-right
    return f"main_w-overlay_w-{m + cum}", f"main_h-overlay_h-{m}"


def _frame_index(path):
    """Parse the zero-padded numeric index out of a `NNNNNNNN.png` path."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _load_frame_timestamps(dir_path, files):
    """Return per-file timestamps for `files`, looked up by each file's own numeric
    index into `dir_path/timestamps.txt` (NOT by position in `files`) — the array in
    that file is padded with -1 for any index never written, so a hole earlier in
    the sequence must not shift every later file's timestamp. None if unavailable."""
    fp = os.path.join(dir_path, "timestamps.txt")
    if not os.path.exists(fp):
        return None
    raw = np.atleast_1d(np.loadtxt(fp)).astype(np.float64)
    idxs = np.array([_frame_index(f) for f in files])
    if idxs.max(initial=-1) >= len(raw):
        return None
    return raw[idxs]


def _nearest_by_timestamp(src_ts, query_ts):
    """For each time in `query_ts`, return the index into `src_ts` of its nearest neighbor."""
    order = np.argsort(src_ts)
    src_sorted = src_ts[order]
    right = np.clip(np.searchsorted(src_sorted, query_ts), 0, len(src_sorted) - 1)
    left = np.clip(right - 1, 0, len(src_sorted) - 1)
    use_left = np.abs(src_sorted[left] - query_ts) < np.abs(src_sorted[right] - query_ts)
    return order[np.where(use_left, left, right)]


def _timestamp_align_dir(inset_files, inset_ts, main_ts, tmp_root, name):
    """Symlink inset frames into `tmp_root/name/%08d.png` so index i is the inset capture nearest in real time to main frame i (not just index i)."""
    aligned_dir = os.path.join(tmp_root, name)
    os.makedirs(aligned_dir, exist_ok=True)
    nearest = _nearest_by_timestamp(inset_ts, main_ts)
    for out_i, src_i in enumerate(nearest):
        dst = os.path.join(aligned_dir, f"{out_i:08d}.png")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(inset_files[src_i]), dst)
    return aligned_dir


def _normalize_pip_specs(pip, n_frames, main_size, main_frame_ts=None, tmp_root=None):
    """Validate + normalize PiP inset specs, dropping insets that don't exist in
    the dataset, and re-indexing each inset's frames to align with the main
    video's frame timestamps (nearest capture time) rather than assuming file
    index N in the inset represents the same instant as file index N in main.
    """
    if not pip:
        return []
    W, _H = main_size
    valid = []
    for k, spec in enumerate(pip):
        d = spec.get("dir")
        if not d or not os.path.isdir(d):
            print(f"  [pip] inset dir missing, skipping: {d}")
            continue
        insets = sorted(glob.glob(os.path.join(d, "????????.png")))
        if not insets:
            print(f"  [pip] no frames in inset dir, skipping: {d}")
            continue

        inset_dir = d
        inset_ts = _load_frame_timestamps(d, insets)
        if main_frame_ts is not None and inset_ts is not None:
            inset_dir = _timestamp_align_dir(insets, inset_ts, main_frame_ts, tmp_root, f"pip{k}")
        elif len(insets) != n_frames:
            print(
                f"  [pip] inset frame count mismatch ({len(insets)} vs {n_frames}) and no "
                f"timestamps available to align; skipping inset: {d}"
            )
            continue
        else:
            print(f"  [pip] no timestamps available for {d}; falling back to raw index alignment (not time-synced).")

        pw = max(2, int(round(W * float(spec.get("scale", 0.25)))))
        if pw % 2:
            pw += 1
        valid.append({
            "dir": inset_dir,
            "pw": pw,
            "pos": spec.get("pos", "bottom-right"),
            "margin": int(spec.get("margin", 16)),
            "border": int(spec.get("border", 3)),
            "border_color": spec.get("border_color", "white"),
        })
    return valid


def _build_pip_filter(specs, input_offset, base_label):
    """Build the PiP portion of an ffmpeg -filter_complex graph.
    """
    segments = []
    cur = base_label
    cum = {}  # cumulative horizontal offset (px) per corner
    for k, s in enumerate(specs):
        idx = input_offset + k
        b = s["border"]
        if b > 0:
            prep = (
                f"[{idx}:v]scale={s['pw']}:-2,"
                f"pad=iw+{2 * b}:ih+{2 * b}:{b}:{b}:color={s['border_color']}[pip{k}]"
            )
        else:
            prep = f"[{idx}:v]scale={s['pw']}:-2[pip{k}]"
        segments.append(prep)

        c = cum.get(s["pos"], 0)
        x, y = _pip_overlay_xy(s["pos"], s["margin"], c)
        out = f"pipout{k}"
        segments.append(f"{cur}[pip{k}]overlay={x}:{y}[{out}]")
        cur = f"[{out}]"
        cum[s["pos"]] = c + s["pw"] + 2 * b + s["margin"]
    return ";".join(segments), cur


def _pip_variant_path(output_path):
    """`foo.mp4` -> `foo_pip.mp4`."""
    root, ext = os.path.splitext(output_path)
    return f"{root}_pip{ext}"


def render_kitti_video(frames_dir, output_path=None, fps=None, overlay_dir=None, pip=None):
    """
    Render a directory of PNG frames into an MP4 with an optional RGBA HUD
    overlay.
    """
    rendered = _render_kitti_video_once(frames_dir, output_path=output_path, fps=fps,
                                        overlay_dir=overlay_dir, pip=None)
    if pip:
        _render_kitti_video_once(frames_dir, output_path=_pip_variant_path(output_path or frames_dir.rstrip(os.sep) + ".mp4"),
                                 fps=fps, overlay_dir=overlay_dir, pip=pip)
    return rendered


def _render_kitti_video_once(frames_dir, output_path=None, fps=None, overlay_dir=None, pip=None):
    """
    Render a directory of PNG frames into a single MP4 with an optional RGBA
    HUD overlay and optional picture-in-picture insets baked in.
    """

    pngs = sorted(glob.glob(os.path.join(frames_dir, "????????.png")))
    if not pngs:
        return None

    if output_path is None:
        output_path = frames_dir.rstrip(os.sep) + ".mp4"

    n_frames = len(pngs)
    main_frame_ts = _load_frame_timestamps(frames_dir, pngs)

    if fps is None and main_frame_ts is not None and len(main_frame_ts) > 1:
        fps = float(1.0 / np.median(np.diff(main_frame_ts)))

    if fps is None or fps <= 0:
        fps = 10.0

    # %{eif\:n+1\:d} is ffmpeg's evaluate-integer-format for 1-indexed frame number
    drawtext = (
        f"drawtext=text='%{{eif\\:n+1\\:d}}/{n_frames}'"
        ":fontcolor=white:fontsize=24:x=10:y=10"
        ":box=1:boxcolor=black@0.5:boxborderw=5"
    )

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", os.path.join(frames_dir, "%08d.png"),
    ]

    if overlay_dir is not None:
        overlays = sorted(glob.glob(os.path.join(overlay_dir, "????????.png")))
        if len(overlays) != n_frames:
            print(
                f"  [video] overlay count mismatch ({len(overlays)} overlays for {n_frames} frames); "
                "rendering without HUD overlay."
            )
            overlay_dir = None

    has_overlay = overlay_dir is not None

    pip_specs = []
    pip_tmp_root = None
    if pip:
        with Image.open(pngs[0]) as _img:
            main_size = _img.size
        pip_tmp_root = os.path.join(os.path.dirname(output_path) or ".",
                                    ".pip_tmp_" + os.path.splitext(os.path.basename(output_path))[0])
        pip_specs = _normalize_pip_specs(pip, n_frames, main_size,
                                         main_frame_ts=main_frame_ts, tmp_root=pip_tmp_root)

    if has_overlay or pip_specs:
        input_offset = 1
        if has_overlay:
            cmd.extend([
                "-framerate", str(fps),
                "-start_number", "0",
                "-i", os.path.join(overlay_dir, "%08d.png"),
            ])
            input_offset = 2
        for s in pip_specs:
            cmd.extend([
                "-framerate", str(fps),
                "-start_number", "0",
                "-i", os.path.join(s["dir"], "%08d.png"),
            ])

        segments = []
        if has_overlay:
            segments.append("[0:v][1:v]overlay=0:0[hud]")
            base_label = "[hud]"
        else:
            base_label = "[0:v]"

        if pip_specs:
            pip_seg, final_label = _build_pip_filter(pip_specs, input_offset, base_label)
            segments.append(pip_seg)
        else:
            final_label = base_label

        segments.append(f"{final_label}{drawtext}")
        cmd.extend(["-filter_complex", ";".join(segments)])
    else:
        cmd.extend(["-vf", drawtext])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-threads", "0",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ])

    label = os.path.basename(output_path) if output_path else os.path.basename(frames_dir)
    stderr_lines = []
    try:
        with tqdm.tqdm(total=n_frames, desc=label, unit="frame") as pbar:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            current = 0
            for line in proc.stdout:
                if line.startswith("frame="):
                    try:
                        frame = int(line.split("=", 1)[1].strip())
                        pbar.update(frame - current)
                        current = frame
                    except ValueError:
                        pass
            _, stderr_out = proc.communicate()
            stderr_lines = stderr_out

        if proc.returncode != 0:
            print(f"  [video] ERROR rendering {frames_dir}:\n{stderr_lines[-500:]}")
            return None

        print(f"  [video] rendered: {output_path}")
        return output_path
    finally:
        if pip_tmp_root and os.path.isdir(pip_tmp_root):
            shutil.rmtree(pip_tmp_root, ignore_errors=True)


def discover_kitti_image_dirs(dataset_dir):
    modality_dirs = {}
    if not os.path.isdir(dataset_dir):
        return modality_dirs

    for group in sorted(os.listdir(dataset_dir)):
        group_dir = os.path.join(dataset_dir, group)
        if not os.path.isdir(group_dir):
            continue
        for name in sorted(os.listdir(group_dir)):
            name_dir = os.path.join(group_dir, name)
            if os.path.isdir(name_dir) and glob.glob(os.path.join(name_dir, "????????.png")):
                modality_dirs[f"{group}/{name}"] = name_dir

    return modality_dirs


def render_dataset_videos(modality_dirs, video_config, viz_dir, hud_data=None, imu_data=None):
    os.makedirs(viz_dir, exist_ok=True)
    hud = video_config["hud"]

    pip_by_main = {}
    for spec in video_config["pip"]:
        inset_key = spec["inset"]
        if inset_key not in modality_dirs:
            print(f"  [pip] inset modality '{inset_key}' not in dataset; skipping")
            continue
        pip_by_main.setdefault(spec["main"], []).append({
            "dir": modality_dirs[inset_key],
            "scale": spec.get("scale", 0.25),
            "pos": spec.get("pos", "bottom-right"),
            "margin": spec.get("margin", 16),
            "border": spec.get("border", 3),
            "border_color": spec.get("border_color", "white"),
        })

    rendered = {}
    for key, frames_dir in modality_dirs.items():
        group, name = key.split("/", 1)
        output_path = os.path.join(viz_dir, f"{group}_{name}.mp4")
        pip = pip_by_main.get(key)

        if hud is not None and hud_data is not None:
            v = render_kitti_video_with_hud(
                frames_dir,
                output_path=output_path,
                gps_xy=hud_data["gps_xy"],
                gps_times=hud_data["gps_times"],
                speed_mps=hud_data["speed_mps"],
                speed_times=hud_data["speed_times"],
                bag_start_time=hud_data["bag_start_time"],
                overlay_root=os.path.join(viz_dir, "overlays"),
                map_tif=hud["map_tif"],
                map_source=hud["map_source"],
                allow_network_tiles=hud["allow_network_tiles"],
                n_workers=hud["workers"],
                accel_xy=imu_data["accel_xy"] if imu_data is not None else None,
                accel_times=imu_data["times"] if imu_data is not None else None,
                gmeter_max_g=hud["gmeter_max_g"],
                gmeter_trail=hud["gmeter_trail"],
                quat_xyzw=hud_data.get("quat_xyzw"),
                quat_times=hud_data.get("quat_times"),
                orient_cfg=hud["orient"],
                pip=pip,
            )
        else:
            v = render_kitti_video(frames_dir, output_path=output_path, pip=pip)

        if v:
            rendered[key] = v

    return rendered

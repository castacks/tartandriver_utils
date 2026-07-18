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
        hud = {
            "odom_topic": hud_cfg.get("odom_topic", "/odometry/filtered_odom"),
            "imu_topic": hud_cfg.get("imu_topic", "/novatel/imu/data"),
            "gmeter_max_g": hud_cfg.get("gmeter_max_g", 1.0),
            "map_tif": hud_cfg.get("map_tif"),
            "map_source": hud_cfg.get("map_source", "auto"),
            "allow_network_tiles": hud_cfg.get("allow_network_tiles", False),
            "workers": hud_cfg.get("workers"),
        }

    return {"hud": hud, "pip": cfg.get("pip", []) or []}


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

    return bg, route_px, frame_px


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
    return bg, route_px, frame_px


def _draw_gmeter(draw, cx, cy, radius, ax_g, ay_g, max_g, font_label):
    """Circular accelerometer g-meter, racing-dashboard style.

    ax_g is longitudinal g (+forward/accel, -rearward/brake), ay_g is lateral g
    (+left, -right, REP-103 body frame). The dot's position is the (ay, ax)
    vector clipped to the dial; readouts at 12/3/6/9 o'clock show the current
    accel/right/brake/left magnitude (each >= 0, 0 outside that phase).
    """
    ring_color = (255, 255, 255, 60)
    axis_color = (255, 255, 255, 90)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline=ring_color, width=1)
    draw.ellipse([cx - radius / 2, cy - radius / 2, cx + radius / 2, cy + radius / 2],
                 outline=ring_color, width=1)
    draw.line([cx - radius, cy, cx + radius, cy], fill=axis_color, width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=axis_color, width=1)

    if np.isfinite(ax_g) and np.isfinite(ay_g):
        mag = math.hypot(ax_g, ay_g)
        scale = radius / max_g if mag <= max_g else radius / max(mag, 1e-6)
        dot_x = cx - ay_g * scale
        dot_y = cy - ax_g * scale
        r = 5
        draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r],
                     fill=(255, 90, 90, 235), outline=(0, 0, 0, 180), width=1)
    else:
        ax_g = ay_g = 0.0

    accel_g = max(ax_g, 0.0)
    brake_g = max(-ax_g, 0.0)
    right_g = max(-ay_g, 0.0)
    left_g = max(ay_g, 0.0)

    label_pad = 6
    labels = [
        (f"{accel_g:.2f}g", (cx, cy - radius - label_pad), "ms", (160, 255, 100, 230)),
        (f"{brake_g:.2f}g", (cx, cy + radius + label_pad), "mt", (255, 110, 90, 230)),
        (f"{right_g:.2f}g", (cx + radius + label_pad, cy), "lm", (120, 200, 255, 230)),
        (f"{left_g:.2f}g", (cx - radius - label_pad, cy), "rm", (120, 200, 255, 230)),
    ]
    for text, pos, anchor, color in labels:
        draw.text(pos, text, font=font_label, fill=color, anchor=anchor)


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
        frame_ax_g,
        frame_ay_g,
        gmeter_max_g,
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

    # --- Bottom-right: Info panel (time + speed + g-meter) ---
    gm_radius = 38
    gm_diameter = 2 * gm_radius
    gm_label_w_side = 36
    gm_label_h_vert = 14

    gmeter_block_w = gm_diameter + 2 * gm_label_w_side
    gmeter_block_h = gm_diameter + 2 * gm_label_h_vert

    has_imu = np.isfinite(frame_ax_g) and np.isfinite(frame_ay_g)
    text_block_w = 200
    inner_h = 20 + 4 + 32  # time + gap + speed
    if has_imu:
        inner_h += 8 + gmeter_block_h

    info_panel_w = max(text_block_w, gmeter_block_w) + 2 * panel_pad
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
        _draw_gmeter(draw, gm_cx, gm_cy, gm_radius, frame_ax_g, frame_ay_g, gmeter_max_g, font_label)

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
        bg, route_px, frame_px = _load_minimap_background(
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

    if not np.isfinite(bag_start_time):
        bag_start_time = frame_ts[0]
    wall_times = bag_start_time + (frame_ts - frame_ts[0])
    wall_time_strs = [
        datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        for t in wall_times
    ]

    jobs = []
    for idx in range(len(frame_ts)):
        jobs.append((
            idx,
            float(frame_speed[idx]),
            wall_time_strs[idx],
            float(frame_ax_g[idx]),
            float(frame_ay_g[idx]),
            gmeter_max_g,
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
    pip=None,
):
    """
    Render an existing KITTI image directory with timestamp/speed/minimap HUD
    and optional picture-in-picture insets (see `render_kitti_video`).
    """
    pngs = sorted(glob.glob(os.path.join(frames_dir, "????????.png")))
    if not pngs:
        return render_kitti_video(frames_dir, output_path=output_path, fps=fps, pip=pip)

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
        return render_kitti_video(frames_dir, output_path=output_path, fps=fps, pip=pip)

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


def render_kitti_video(frames_dir, output_path=None, fps=None, overlay_dir=None, pip=None):
    """
    Render a directory of PNG frames into an MP4 with an optional RGBA HUD
    overlay and optional picture-in-picture insets.
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

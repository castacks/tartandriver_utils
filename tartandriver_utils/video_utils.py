import glob
import os
import subprocess

import numpy as np
import tqdm


def render_kitti_video(frames_dir, output_path=None, fps=None):
    """
    Render a directory of PNG frames into an MP4 with a frame counter overlay.
    """

    pngs = sorted(glob.glob(os.path.join(frames_dir, "????????.png")))
    if not pngs:
        return None

    if output_path is None:
        output_path = frames_dir.rstrip(os.sep) + ".mp4"

    if os.path.exists(output_path):
        print(f"  [video] skipping (already exists): {output_path}")
        return output_path

    if fps is None:
        timestamps_fp = os.path.join(frames_dir, "timestamps.txt")
        if os.path.exists(timestamps_fp):
            ts = np.loadtxt(timestamps_fp)
            if ts.ndim == 1 and len(ts) > 1:
                fps = float(1.0 / np.median(np.diff(ts)))

    if fps is None or fps <= 0:
        fps = 10.0

    n_frames = len(pngs)

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
        "-vf", drawtext,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]

    label = os.path.basename(output_path) if output_path else os.path.basename(frames_dir)
    stderr_lines = []
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

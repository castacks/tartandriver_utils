"""Lazy per-path staging against an rclone remote.
"""
import json
import subprocess

DEFAULT_REMOTE = "airlab_storage"
DEFAULT_FLAGS = ["--multi-thread-streams=0", "--transfers=10", "--stats=15s", "--stats-one-line"]


class RcloneStager:
    def __init__(self, remote=DEFAULT_REMOTE, flags=None):
        self.remote = remote
        self.flags = flags if flags is not None else DEFAULT_FLAGS

    def _remote_path(self, relpath):
        return f"{self.remote}:{relpath}"

    def copy_in(self, remote_relpath, local_dir):
        subprocess.run(
            ["rclone", "copy", self._remote_path(remote_relpath), local_dir] + self.flags,
            check=True,
        )

    def copy_out(self, local_dir, remote_relpath):
        subprocess.run(
            ["rclone", "copy", local_dir, self._remote_path(remote_relpath)] + self.flags,
            check=True,
        )

    def list(self, remote_root):
        """rclone lsjson --recursive; returns the parsed entry list (IsDir/Path/...)."""
        result = subprocess.run(
            ["rclone", "lsjson", "--recursive", self._remote_path(remote_root)],
            stdout=subprocess.PIPE, text=True, check=True,
        )
        return json.loads(result.stdout)

    def exists(self, remote_relpath, filename="target_timestamps.txt"):
        """Whether remote_relpath already holds `filename` (mirrors os_utils.is_kitti_dir)."""
        result = subprocess.run(
            ["rclone", "lsf", self._remote_path(remote_relpath)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        if result.returncode != 0:
            return False
        return filename in result.stdout.split()

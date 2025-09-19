import os
import numpy as np

def is_rosbag_dir(fp):
    """
    Determine if a dir is a valid rosbag (check for mcaps and metadata.yaml)
    """
    dir_files = os.listdir(fp)

    has_metadata = "metadata.yaml" in dir_files
    has_mcaps = any([df[-5:] == ".mcap" for df in dir_files])

    return has_metadata and has_mcaps

def is_kitti_dir(fp):
    """
    Determine if a dir is a valid rosbag (check for target_timestamps.txt)
    """
    dir_files = os.listdir(fp)

    has_timestamps = "target_timestamps.txt" in dir_files

    return has_timestamps

def kitti_n_frames(dir):
    """
    Get the number of frames in a KITTI dataset
    """
    return np.loadtxt(os.path.join(dir, 'target_timestamps.txt')).shape[0]
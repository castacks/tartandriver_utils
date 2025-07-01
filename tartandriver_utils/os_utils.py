import os

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

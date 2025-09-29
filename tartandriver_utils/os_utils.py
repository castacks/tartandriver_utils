import os
import numpy as np
import yaml
class YamlLoader(yaml.SafeLoader):
    """
    YAML Loader to loading YAMLs with nested YAMLs

    Example usage:
    Consider two separate yamls:

    `foo.yaml`:
    ```
    a: 1
    b: ['hello', 'there']
    c: !include bar.yaml
    ```
    
    `bar.yaml`:
    ```
    x: 42
    y:
    - 'General'
    - 'Kenobi'
    ```
    
    We automatically loaded nested YAMLs like so:
    ```
    from tartandriver_utils.os_utils import YamlLoader
    with open('conf1.yaml', 'r') as f:
        config = yaml.load(f, YamlLoader)
    ```
    Resulting config
    ```
    print(config)
    { 'a': 1,
      'b': ['hello', 'there'],
      'c': {'x': 42, 'y': ['General', 'Kenobi']}
    }
    ```
    """

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(YamlLoader, self).__init__(stream)
        super(YamlLoader, self).add_constructor('!include', YamlLoader.include)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, YamlLoader)


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
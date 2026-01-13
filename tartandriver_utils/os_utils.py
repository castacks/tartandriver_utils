import io
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

        with open(filename, "r") as f:
            content = os.path.expandvars(f.read())
        stream = io.StringIO(content)
        stream.name = filename
        return yaml.load(stream, YamlLoader)
        
def load_yaml(fp):
    # identify and expand any env vars
    with open(fp,"r") as f:
       content = os.path.expandvars(f.read())
    # turn string content into stream for loader
    stream = io.StringIO(content)
    stream.name = fp
    return yaml.load(stream, YamlLoader)

def save_yaml(config, fp):
    yaml.dump(config, open(fp, 'w'), default_flow_style=False)

def is_rosbag_dir(fp):
    """
    Determine if a dir is a valid rosbag (check for mcaps and metadata.yaml)
    """
    if not os.path.isdir(fp):
        return False
    
    dir_files = os.listdir(fp)

    has_metadata = "metadata.yaml" in dir_files
    has_mcaps = any([df[-5:] == ".mcap" for df in dir_files])

    return has_metadata and has_mcaps

def is_kitti_dir(fp):
    """
    Determine if a dir is a valid rosbag (check for target_timestamps.txt)
    """
    if not os.path.isdir(fp):
        return False

    dir_files = os.listdir(fp)

    has_timestamps = "target_timestamps.txt" in dir_files

    return has_timestamps

def kitti_n_frames(dir):
    """
    Get the number of frames in a KITTI dataset
    """
    if os.path.exists(os.path.join(dir, 'target_timestamps.txt')):
        return np.loadtxt(os.path.join(dir, 'target_timestamps.txt')).shape[0]
    
    elif os.path.exists(os.path.join(dir, 'timestamps.txt')):
        return np.loadtxt(os.path.join(dir, 'timestamps.txt')).shape[0]
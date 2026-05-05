import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def generate_launch_description():
    # Config and directory paths provided in launch for relative pathing
    spoof_config_path = get_share_file(
        "tartandriver_utils", "config/rtk_spoofer_config.yaml")
    super_odometry_config_path = get_share_file(
        "super_odometry", "config/learningphysics/yamaha.yaml")
    super_odometry_calib_path = get_share_file(
        "super_odometry", "config/learningphysics/yamaha_calibration.yaml")
    

    # Arguments
    spoof_config_path_arg = DeclareLaunchArgument(
        "spoof_config",
        default_value=spoof_config_path,
        description="Path to spoof_config file for mission_generator node"
    )
    super_odometry_config_path_arg = DeclareLaunchArgument(
        "super_odometry_config_file",
        default_value=super_odometry_config_path,
        description="Path to config file for superodometry, used for feature extraction"
    )
    super_odometry_calib_path_arg = DeclareLaunchArgument(
        "super_odometry_calibration_file",
        default_value=super_odometry_calib_path,
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true"
    )

    # Nodes
    rtk_spoofer_node = Node(
        package="tartandriver_utils",
        name="rtk_spoofer_node",
        executable="rtk_spoofer_node", 
        parameters=[
            LaunchConfiguration("spoof_config"),
            {"use_sim_time": LaunchConfiguration("use_sim_time"),}
        ]
    )
    # can't get feature extraction for dewarping/undistortion to yield good result
    # feature_extraction_node = Node(
    #     package="super_odometry",
    #     executable="feature_extraction_node",
    #     output={
    #         "stdout": "screen",
    #         "stderr": "screen",
    #     },
    #     parameters=[LaunchConfiguration("super_odometry_config_file"),
    #         { "calibration_file": LaunchConfiguration("super_odometry_calibration_file"),
    #           "use_sim_time": LaunchConfiguration("use_sim_time"),
    #     }],
    # )
    # laser_mapping_node = Node(
    #     package="super_odometry",
    #     executable="laser_mapping_node",
    #     output={
    #         "stdout": "screen",
    #         "stderr": "screen",
    #     },
    #     parameters=[LaunchConfiguration("super_odometry_config_file"),
    #         { "calibration_file": LaunchConfiguration("super_odometry_calibration_file"),
    #           "use_sim_time": LaunchConfiguration("use_sim_time"),
    #     }],
    # )

    return LaunchDescription(
        [
            spoof_config_path_arg,
            super_odometry_config_path_arg,
            super_odometry_calib_path_arg,
            use_sim_time_arg,
            rtk_spoofer_node,
            # feature_extraction_node,
            # laser_mapping_node,
        ]
    )

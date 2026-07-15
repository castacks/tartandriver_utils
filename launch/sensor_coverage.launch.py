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
    # spoof_config_path = get_share_file(
        # "tartandriver_utils", "config/rtk_spoofer_config.yaml")
    # super_odometry_config_path = get_share_file(
        # "super_odometry", "config/learningphysics/yamaha.yaml")
    # super_odometry_calib_path = get_share_file(
        # "super_odometry", "config/learningphysics/yamaha_calibration.yaml")
    

    # Arguments
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true"
    )

    # Nodes
    coverage_viz = Node(
        package="tartandriver_utils",
        name="angular_coverage_viz_node",
        executable="angular_coverage_viz_node", 
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time"),}
        ]
    )
    # uncomment feature_extraction if want to register dewarped cloud
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
            use_sim_time_arg,
            coverage_viz,

        ]
    )

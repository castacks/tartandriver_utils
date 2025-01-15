import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_file_name = 'atv.urdf.xml'
    urdf = os.path.join(
        get_package_share_directory('tartandriver_utils'),
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf]),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_vehicle_to_vehicle_viz",
            output="screen",
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                "--x", "-1.0",
                "--y", ".1",
                "--z", "-.75",
                "--qx", ".5",
                "--qy", ".5",
                "--qz", ".5",
                "--qw", ".5",
                "--frame-id", "vehicle",
                "--child-frame-id", "vehicle_viz"
            ]
        )
    ])
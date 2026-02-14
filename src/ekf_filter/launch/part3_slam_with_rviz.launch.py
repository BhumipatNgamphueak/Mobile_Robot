#!/usr/bin/env python3
"""
Part 3 Launch File: Full SLAM with slam_toolbox + RViz Visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get package directory and derive workspace results path
    pkg_dir = get_package_share_directory('ekf_filter')
    config_file = os.path.join(pkg_dir, 'config', 'mapper_params_online_async_B.yaml')
    results_dir = os.path.normpath(os.path.join(pkg_dir, '..', '..', '..', '..', 'results'))

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Scan Republisher Node (republishes scan with base_link frame and current timestamps)
    scan_republisher_node = Node(
        package='ekf_filter',
        executable='scan_republisher_node.py',
        name='scan_republisher_node',
        output='screen',
    )

    # Read Data Node
    read_data_node = Node(
        package='differential_drive_model',
        executable='read_data_node.py',
        name='read_data_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # EKF Odometry Node
    ekf_odom_node = Node(
        package='ekf_filter',
        executable='ekf_odometry_node.py',
        name='ekf_odometry_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # Path Publisher Node
    path_publisher_node = Node(
        package='ekf_filter',
        executable='path_publisher_node.py',
        name='path_publisher_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'odom_topic': '/ekf/odometry',
            'path_topic': '/robot_path',
            'sample_rate': 5,
        }],
    )

    # SLAM Toolbox Node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[]
    )

    # Auto Map Saver Node
    auto_map_saver_node = Node(
        package='ekf_filter',
        executable='auto_map_saver_node.py',
        name='auto_map_saver_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'output_dir': results_dir,
            'map_name': 'slam_map',
        }],
    )

    # SLAM Trajectory Saver Node
    slam_trajectory_saver_node = Node(
        package='ekf_filter',
        executable='slam_trajectory_saver_node.py',
        name='slam_trajectory_saver_node',
        output='screen',
        parameters=[{
            'output_dir': results_dir,
            'output_file': 'slam_trajectory.csv',
        }],
    )

    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'slam_view.rviz')],
    )

    return LaunchDescription([
        use_sim_time_arg,
        scan_republisher_node,
        read_data_node,
        ekf_odom_node,
        path_publisher_node,
        slam_toolbox_node,
        auto_map_saver_node,
        slam_trajectory_saver_node,
        rviz_node,
    ])

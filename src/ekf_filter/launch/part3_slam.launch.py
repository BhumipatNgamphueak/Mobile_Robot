#!/usr/bin/env python3
"""
Part 3 Launch File: Full SLAM with slam_toolbox
Launches SLAM with EKF odometry
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get package directory
    pkg_dir = get_package_share_directory('ekf_filter')
    config_file = os.path.join(pkg_dir, 'config', 'mapper_params_online_async.yaml')

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',  # Set to false - we handle timing internally
        description='Use simulation time'
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

    # Synchronized Republisher Node (ensures scan and odom have EXACTLY matching timestamps)
    synchronized_republisher_node = Node(
        package='differential_drive_model',
        executable='synchronized_republisher_node.py',
        name='synchronized_republisher_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # EKF Odometry Node (provides odometry for SLAM)
    ekf_odom_node = Node(
        package='ekf_filter',
        executable='ekf_odometry_node.py',
        name='ekf_odometry_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # Path Publisher Node (visualize robot trajectory)
    path_publisher_node = Node(
        package='ekf_filter',
        executable='path_publisher_node.py',
        name='path_publisher_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'odom_topic': '/odom_republished',
            'path_topic': '/robot_path',
            'sample_rate': 5,  # Add pose every 5 odometry messages
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

    return LaunchDescription([
        use_sim_time_arg,
        read_data_node,
        synchronized_republisher_node,
        ekf_odom_node,
        path_publisher_node,
        slam_toolbox_node,
    ])

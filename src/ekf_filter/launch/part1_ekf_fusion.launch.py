#!/usr/bin/env python3
"""
Part 1 Launch File: EKF Odometry Fusion
Launches ONLY nodes needed for Part 1 (wheel odometry and EKF comparison)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # Read Data Node
    read_data_node = Node(
        package='differential_drive_model',
        executable='read_data_node.py',
        name='read_data_node',
        output='screen',
    )

    # Wheel Odometry Node (Baseline)
    wheel_odom_node = Node(
        package='differential_drive_model',
        executable='wheel_odometry_node.py',
        name='wheel_odometry_node',
        output='screen',
    )

    # EKF Odometry Node
    ekf_odom_node = Node(
        package='ekf_filter',
        executable='ekf_odometry_node.py',
        name='ekf_odometry_node',
        output='screen',
    )

    return LaunchDescription([
        read_data_node,
        wheel_odom_node,
        ekf_odom_node,
    ])

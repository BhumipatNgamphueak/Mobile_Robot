#!/usr/bin/env python3
"""
Part 2 Launch File: ICP Odometry Refinement with RViz
Launches all nodes including ICP scan matching, map building, and visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Derive workspace paths from installed package share directory
    pkg_dir = get_package_share_directory('ekf_filter')
    results_dir = os.path.normpath(os.path.join(pkg_dir, '..', '..', '..', '..', 'results'))
    rviz_config = os.path.join(pkg_dir, 'rviz', 'icp_view.rviz')

    # Read Data Node
    read_data_node = Node(
        package='differential_drive_model',
        executable='read_data_node.py',
        name='read_data_node',
        output='screen',
    )

    # Wheel Odometry Node
    wheel_odom_node = Node(
        package='differential_drive_model',
        executable='wheel_odometry_node.py',
        name='wheel_odometry_node',
        output='screen',
    )

    # EKF Odometry Node (provides initial guess for ICP)
    ekf_odom_node = Node(
        package='ekf_filter',
        executable='ekf_odometry_node.py',
        name='ekf_odometry_node',
        output='screen',
    )

    # ICP Odometry Node (pure ICP output, EKF as initial guess)
    icp_odom_node = Node(
        package='ekf_filter',
        executable='icp_odometry_node.py',
        name='icp_odometry_node',
        output='screen',
    )

    # ICP Map Builder Node
    icp_map_builder_node = Node(
        package='ekf_filter',
        executable='icp_map_builder_node.py',
        name='icp_map_builder_node',
        output='screen',
        parameters=[{
            'output_dir': results_dir,
            'map_name': 'icp_map',
            'resolution': 0.05,
            'map_size_x': 50.0,
            'map_size_y': 50.0,
        }],
    )

    # Path Publisher for Wheel Odometry
    wheel_path_pub = Node(
        package='ekf_filter',
        executable='path_publisher_node.py',
        name='wheel_path_publisher',
        parameters=[
            {'odom_topic': '/wheel/odometry'},
            {'path_topic': '/wheel/path'},
            {'max_path_length': 10000},
            {'sample_rate': 1}
        ],
        output='screen',
    )

    # Path Publisher for EKF Odometry
    ekf_path_pub = Node(
        package='ekf_filter',
        executable='path_publisher_node.py',
        name='ekf_path_publisher',
        parameters=[
            {'odom_topic': '/ekf/odometry'},
            {'path_topic': '/ekf/path'},
            {'max_path_length': 10000},
            {'sample_rate': 1}
        ],
        output='screen',
    )

    # Path Publisher for ICP Odometry
    icp_path_pub = Node(
        package='ekf_filter',
        executable='path_publisher_node.py',
        name='icp_path_publisher',
        parameters=[
            {'odom_topic': '/icp/odometry'},
            {'path_topic': '/icp/path'},
            {'max_path_length': 10000},
            {'sample_rate': 1}
        ],
        output='screen',
    )

    # Static TF for base_link_wheel to base_scan
    static_tf_pub = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_scan_publisher',
        arguments=['0.032', '0', '0.172', '0', '0', '0', 'base_link_wheel', 'base_scan'],
        output='screen',
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        read_data_node,
        wheel_odom_node,
        ekf_odom_node,
        icp_odom_node,
        icp_map_builder_node,
        wheel_path_pub,
        ekf_path_pub,
        icp_path_pub,
        static_tf_pub,
        rviz_node,
    ])

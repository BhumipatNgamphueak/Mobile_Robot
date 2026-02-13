#!/usr/bin/env python3
"""
Part 2 Launch File with RViz: ICP Odometry Refinement
Launches all nodes including ICP scan matching and RViz visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # RViz config path
    rviz_config = '/home/prime/mobile_lab1/rviz_configs/part2_icp.rviz'

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

    # EKF Odometry Node
    ekf_odom_node = Node(
        package='ekf_filter',
        executable='ekf_odometry_node.py',
        name='ekf_odometry_node',
        output='screen',
    )

    # ICP Odometry Node
    icp_odom_node = Node(
        package='ekf_filter',
        executable='icp_odometry_node.py',
        name='icp_odometry_node',
        output='screen',
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

    # Static TF publisher for base_link_wheel to base_scan transform
    # TurtleBot3 Burger: laser is mounted at the front, 0.032m forward, 0.172m up
    # Attach scan to wheel odometry frame for visualization
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

    # Optional: Laser scan assembler for accumulated map visualization
    # Uncomment to enable:
    # laser_assembler = Node(
    #     package='laser_assembler',
    #     executable='laser_scan_assembler',
    #     name='laser_assembler',
    #     output='screen',
    # )

    return LaunchDescription([
        read_data_node,
        wheel_odom_node,
        ekf_odom_node,
        icp_odom_node,
        wheel_path_pub,
        ekf_path_pub,
        icp_path_pub,
        static_tf_pub,
        rviz_node,
    ])

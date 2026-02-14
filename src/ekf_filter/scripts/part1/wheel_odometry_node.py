#!/usr/bin/python3
"""
Part 1: Wheel Odometry Node (Baseline)
Computes pure wheel odometry without sensor fusion for comparison
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import csv
import os
from ament_index_python.packages import get_package_share_directory


def _ws_results():
    share = get_package_share_directory('ekf_filter')
    ws = os.path.normpath(os.path.join(share, '..', '..', '..', '..'))
    return os.path.join(ws, 'results')


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__('wheel_odometry_node')

        # Subscriber
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/read_data/joint_states',
            self.joint_states_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odometry', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Robot parameters (TurtleBot3 Burger)
        self.wheel_radius = 0.033  # meters
        self.wheel_separation = 0.160  # meters

        # State: [x, y, theta]
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

        # Previous state
        self.prev_joint_positions = None
        self.prev_time = None

        # Data logging
        self.trajectory_file = os.path.join(_ws_results(), 'wheel_odometry.csv')
        self.init_csv_file()

        self.get_logger().info('Wheel Odometry Node initialized')
        self.get_logger().info(f'Saving trajectory to: {self.trajectory_file}')

    def init_csv_file(self):
        """Initialize CSV file for trajectory logging"""
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)
        with open(self.trajectory_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'theta', 'v', 'omega'])

    def joint_states_callback(self, msg):
        """Compute wheel odometry from joint states"""

        if len(msg.position) < 2:
            return

        left_pos = msg.position[0]
        right_pos = msg.position[1]

        current_time = self.get_clock().now()

        if self.prev_joint_positions is not None and self.prev_time is not None:
            dt = (current_time - self.prev_time).nanoseconds / 1e9

            if dt > 0:
                # Calculate wheel displacements
                d_left = (left_pos - self.prev_joint_positions[0]) * self.wheel_radius
                d_right = (right_pos - self.prev_joint_positions[1]) * self.wheel_radius

                # Calculate velocities
                self.v = (d_right + d_left) / (2.0 * dt)
                self.omega = (d_right - d_left) / (self.wheel_separation * dt)

                # Update pose using differential drive kinematics
                if abs(self.omega) < 1e-6:
                    # Straight line motion
                    self.x += self.v * math.cos(self.theta) * dt
                    self.y += self.v * math.sin(self.theta) * dt
                else:
                    # Curved motion
                    self.x += (self.v / self.omega) * (math.sin(self.theta + self.omega * dt) - math.sin(self.theta))
                    self.y += (self.v / self.omega) * (-math.cos(self.theta + self.omega * dt) + math.cos(self.theta))
                    self.theta += self.omega * dt

                # Normalize theta
                self.theta = self.normalize_angle(self.theta)

                # Log data
                self.log_trajectory(current_time)

                # Publish odometry
                self.publish_odometry(current_time)

        # Update previous values
        self.prev_joint_positions = [left_pos, right_pos]
        self.prev_time = current_time

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def log_trajectory(self, timestamp):
        """Log trajectory data to CSV"""
        with open(self.trajectory_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.nanoseconds,
                self.x,
                self.y,
                self.theta,
                self.v,
                self.omega
            ])

    def publish_odometry(self, timestamp):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link_wheel'

        # Position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        # Orientation
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        # Velocity
        odom_msg.twist.twist.linear.x = self.v
        odom_msg.twist.twist.angular.z = self.omega

        self.odom_pub.publish(odom_msg)

        # Broadcast TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link_wheel'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

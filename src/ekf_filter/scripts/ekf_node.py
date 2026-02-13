#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # Subscribers to read_data_node topics
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/read_data/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/read_data/imu',
            self.imu_callback,
            10
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            '/read_data/joint_states',
            self.joint_states_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/ekf/odometry', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Robot parameters (TurtleBot3 Burger)
        self.wheel_radius = 0.033  # meters
        self.wheel_separation = 0.160  # meters (distance between wheels)

        # EKF state: [x, y, theta, v, omega]
        # x, y: position in meters
        # theta: orientation in radians
        # v: linear velocity in m/s
        # omega: angular velocity in rad/s
        self.state = np.zeros(5)  # [x, y, theta, v, omega]
        self.covariance = np.eye(5) * 0.1  # Initial covariance

        # Process noise covariance (Q)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])

        # Measurement noise covariance (R)
        self.R_imu = np.diag([0.01])  # For angular velocity from IMU
        self.R_odom = np.diag([0.05, 0.05])  # For linear and angular velocity from wheel odometry

        # Previous joint states for odometry calculation
        self.prev_joint_positions = None
        self.prev_time = None

        # Message counters
        self.scan_count = 0
        self.imu_count = 0
        self.joint_states_count = 0

        self.get_logger().info('EKF Node initialized and subscribing to topics:')
        self.get_logger().info('  - /read_data/scan')
        self.get_logger().info('  - /read_data/imu')
        self.get_logger().info('  - /read_data/joint_states')

    def scan_callback(self, msg):
        """Callback for LaserScan messages"""
        self.scan_count += 1
        if self.scan_count % 10 == 0:
            self.get_logger().info(f'Received {self.scan_count} scan messages')

    def imu_callback(self, msg):
        """Callback for IMU messages - uses angular velocity for EKF update"""
        self.imu_count += 1

        # Extract angular velocity around z-axis (yaw rate)
        omega_z = msg.angular_velocity.z

        # EKF Measurement Update with IMU data
        # Measurement: z = omega (angular velocity)
        z = np.array([omega_z])

        # Measurement matrix H: we're measuring the 5th state (omega)
        H = np.zeros((1, 5))
        H[0, 4] = 1.0  # Measure omega

        # Innovation
        y = z - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_imu

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.covariance = (np.eye(5) - K @ H) @ self.covariance

        if self.imu_count % 50 == 0:
            self.get_logger().info(f'IMU: omega_z={omega_z:.3f} rad/s | State: x={self.state[0]:.2f}, y={self.state[1]:.2f}, theta={self.state[2]:.2f}')

    def joint_states_callback(self, msg):
        """Callback for JointState messages - compute wheel odometry"""
        self.joint_states_count += 1

        # Extract wheel positions
        # Assuming order: ['wheel_left_joint', 'wheel_right_joint']
        if len(msg.position) < 2:
            return

        left_pos = msg.position[0]
        right_pos = msg.position[1]

        current_time = self.get_clock().now()

        if self.prev_joint_positions is not None and self.prev_time is not None:
            # Calculate time difference
            dt = (current_time - self.prev_time).nanoseconds / 1e9

            if dt > 0:
                # Calculate wheel displacements
                d_left = (left_pos - self.prev_joint_positions[0]) * self.wheel_radius
                d_right = (right_pos - self.prev_joint_positions[1]) * self.wheel_radius

                # Calculate linear and angular velocities from wheel odometry
                v_odom = (d_right + d_left) / (2.0 * dt)
                omega_odom = (d_right - d_left) / (self.wheel_separation * dt)

                # EKF Prediction Step
                self.ekf_predict(dt)

                # EKF Measurement Update with wheel odometry
                z = np.array([v_odom, omega_odom])

                # Measurement matrix H: measuring v and omega (states 4 and 5)
                H = np.zeros((2, 5))
                H[0, 3] = 1.0  # Measure v
                H[1, 4] = 1.0  # Measure omega

                # Innovation
                y = z - H @ self.state

                # Innovation covariance
                S = H @ self.covariance @ H.T + self.R_odom

                # Kalman gain
                K = self.covariance @ H.T @ np.linalg.inv(S)

                # Update state
                self.state = self.state + K @ y

                # Update covariance
                self.covariance = (np.eye(5) - K @ H) @ self.covariance

                # Normalize theta to [-pi, pi]
                self.state[2] = self.normalize_angle(self.state[2])

                # Publish odometry
                self.publish_odometry(current_time)

                if self.joint_states_count % 50 == 0:
                    self.get_logger().info(
                        f'Joint States: v={v_odom:.3f} m/s, omega={omega_odom:.3f} rad/s | '
                        f'EKF State: x={self.state[0]:.2f} m, y={self.state[1]:.2f} m, theta={math.degrees(self.state[2]):.1f}°'
                    )

        # Update previous values
        self.prev_joint_positions = [left_pos, right_pos]
        self.prev_time = current_time

    def ekf_predict(self, dt):
        """EKF Prediction step using motion model"""

        # Extract state variables
        x, y, theta, v, omega = self.state

        # Motion model (differential drive)
        if abs(omega) < 1e-6:
            # Straight line motion
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            # Curved motion
            x_new = x + (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            y_new = y + (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
            theta_new = theta + omega * dt

        # Update state
        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = self.normalize_angle(theta_new)
        # v and omega remain the same in prediction

        # Jacobian of motion model with respect to state
        F = np.eye(5)
        if abs(omega) < 1e-6:
            F[0, 2] = -v * math.sin(theta) * dt
            F[1, 2] = v * math.cos(theta) * dt
            F[0, 3] = math.cos(theta) * dt
            F[1, 3] = math.sin(theta) * dt
        else:
            F[0, 2] = (v / omega) * (math.cos(theta + omega * dt) - math.cos(theta))
            F[1, 2] = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            F[0, 3] = (1 / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            F[1, 3] = (1 / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
            F[0, 4] = -(v / (omega**2)) * (math.sin(theta + omega * dt) - math.sin(theta)) + (v / omega) * math.cos(theta + omega * dt) * dt
            F[1, 4] = -(v / (omega**2)) * (-math.cos(theta + omega * dt) + math.cos(theta)) + (v / omega) * math.sin(theta + omega * dt) * dt
            F[2, 4] = dt

        # Update covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def publish_odometry(self, timestamp):
        """Publish odometry message"""

        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Position
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = 0.0

        # Orientation (convert theta to quaternion)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.state[2] / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.state[2] / 2.0)

        # Velocity
        odom_msg.twist.twist.linear.x = self.state[3]
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = self.state[4]

        # Covariance (6x6 for pose, 6x6 for twist)
        # Fill pose covariance
        for i in range(36):
            odom_msg.pose.covariance[i] = 0.0
        odom_msg.pose.covariance[0] = self.covariance[0, 0]  # x
        odom_msg.pose.covariance[7] = self.covariance[1, 1]  # y
        odom_msg.pose.covariance[35] = self.covariance[2, 2]  # theta

        self.odom_pub.publish(odom_msg)

        # Broadcast TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = self.state[0]
        tf_msg.transform.translation.y = self.state[1]
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

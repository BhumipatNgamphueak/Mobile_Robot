#!/usr/bin/python3
"""
Part 1: EKF Odometry Node with Data Logging
Extended Kalman Filter fusing wheel odometry and IMU (gyroscope + accelerometer)

State vector: [x, y, theta, v, omega]
    x     - position in x (meters)
    y     - position in y (meters)
    theta - heading angle (radians)
    v     - linear velocity (m/s)
    omega - angular velocity (rad/s)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import csv
import os


class EKFOdometryNode(Node):
    def __init__(self):
        super().__init__('ekf_odometry_node')

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/read_data/imu', self.imu_callback, 10)
        self.joint_states_sub = self.create_subscription(JointState, '/read_data/joint_states', self.joint_states_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/ekf/odometry', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Robot parameters (TurtleBot3 Burger)
        self.wheel_radius = 0.033        # meters
        self.wheel_separation = 0.160    # meters

        # EKF state: [x, y, theta, v, omega]
        self.state = np.zeros(5)
        self.covariance = np.eye(5) * 0.1

        self.Q = np.diag([
            0.01,   # Q_x: position x process noise (m²)
            0.01,   # Q_y: position y process noise (m²)
            0.01,   # Q_theta: heading process noise (rad²)
            0.010,   # Q_v: linear velocity process noise (m/s)²
            0.01    # Q_omega: angular velocity process noise (rad/s)²
        ])

        # ============================================================
        # R MATRICES - MEASUREMENT NOISE COVARIANCE
        # ============================================================
        # R represents sensor noise/uncertainty
        #
        # Higher R = less trust in that sensor
        # Lower R  = more trust in that sensor
        # ============================================================

        # IMU Gyroscope: measures angular velocity (omega_z)
        self.R_imu_gyro = np.diag([0.0030])    # (rad/s)² - gyro noise

        # IMU Accelerometer: measures linear acceleration (accel_x)
        self.R_imu_accel = np.diag([0.1907])   # (m/s²)² - accelerometer noise

        # Wheel Odometry: measures [v, omega]
        self.R_odom = np.diag([
            0.0005,    # linear velocity noise (m/s)²
            0.0031     # angular velocity noise (rad/s)²
        ])

        # ============================================================
        # OUTLIER REJECTION (Mahalanobis Distance Gating)
        # ============================================================
        # Chi-squared thresholds for outlier rejection
        # For 1 DOF: 3.84 (95%), 6.63 (99%), 10.83 (99.9%)
        # For 2 DOF: 5.99 (95%), 9.21 (99%), 13.82 (99.9%)
        self.gate_threshold_1dof = 2000000   # 99% confidence for 1D measurements
        self.gate_threshold_2dof = 2000000   # 99% confidence for 2D measurements
        self.outlier_count = 0
        self.total_measurements = 0

        # ============================================================
        # IMU BIAS VALUES (from calibration)
        # ============================================================
        self.prev_imu_time = None
        self.gyro_bias_z = 0.00  # Set from calibration (rad/s)
        self.accel_bias_x = 0.185
        self.accel_bias_y = 0.136965
        self.accel_samples_x = []
        self.accel_samples_y = []
        self.bias_samples_needed = 75  # Collect 50 samples for bias

        # Previous state
        self.prev_joint_positions = None
        self.prev_time = None

        # Data logging
        self.trajectory_file = '/home/prime/mobile_lab1/results/ekf_odometry.csv'
        self.init_csv_file()

        # Timer to publish TF at high frequency (fixes RViz extrapolation errors)
        self.tf_timer = self.create_timer(0.02, self.publish_tf)  # 50Hz

        self.get_logger().info('EKF Odometry Node initialized')
        self.get_logger().info('Sensors: Wheel Encoders + IMU Gyroscope + IMU Accelerometer')
        self.get_logger().info(f'Outlier rejection enabled: gate_1dof={self.gate_threshold_1dof}, gate_2dof={self.gate_threshold_2dof}')
        self.get_logger().info(f'Saving trajectory to: {self.trajectory_file}')

    def init_csv_file(self):
        """Initialize CSV file"""
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)
        with open(self.trajectory_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'theta', 'v', 'omega', 'cov_xx', 'cov_yy', 'cov_tt'])

    def imu_callback(self, msg):
        """
        IMU measurement update
        Uses: gyroscope (omega_z) and accelerometer (accel_x, accel_y)
        """
        current_time = self.get_clock().now()

        # Get IMU measurements (with bias removal)
        omega_z = msg.angular_velocity.z - self.gyro_bias_z  # Gyroscope: angular velocity (bias corrected)
        accel_x = msg.linear_acceleration.x   # Accelerometer: forward accel (body frame)
        accel_y = msg.linear_acceleration.y   # Accelerometer: lateral accel (body frame)

        # ============================================================
        # BIAS ESTIMATION (first 50 samples when robot is stationary)
        # ============================================================
        if len(self.accel_samples_x) < self.bias_samples_needed:
            self.accel_samples_x.append(accel_x)
            self.accel_samples_y.append(accel_y)
            if len(self.accel_samples_x) == self.bias_samples_needed:
                self.accel_bias_x = np.mean(self.accel_samples_x)
                self.accel_bias_y = np.mean(self.accel_samples_y)
                self.get_logger().info(f'Accel bias estimated: x={self.accel_bias_x:.4f}, y={self.accel_bias_y:.4f} m/s²')
            accel_x_corrected = 0.0
            accel_y_corrected = 0.0
        else:
            # Remove bias
            accel_x_corrected = accel_x - self.accel_bias_x
            accel_y_corrected = accel_y - self.accel_bias_y

        # ============================================================
        # UPDATE 1: GYROSCOPE (measures omega directly)
        # ============================================================
        # Measurement: z = omega_z
        # Measurement model: z = H * state, where H selects omega from state
        z_gyro = np.array([omega_z])
        H_gyro = np.zeros((1, 5))
        H_gyro[0, 4] = 1.0  # omega_z maps to state[4] (omega)

        # Compute innovation and check for outliers
        y_gyro = z_gyro - H_gyro @ self.state           # Innovation
        S_gyro = H_gyro @ self.covariance @ H_gyro.T + self.R_imu_gyro  # Innovation covariance

        # Only update if measurement passes Mahalanobis gate
        if self.mahalanobis_gate(y_gyro, S_gyro, self.gate_threshold_1dof):
            K_gyro = self.covariance @ H_gyro.T @ np.linalg.inv(S_gyro)     # Kalman gain
            self.state = self.state + K_gyro @ y_gyro
            self.covariance = (np.eye(5) - K_gyro @ H_gyro) @ self.covariance

        # ============================================================
        # UPDATE 2: ACCELEROMETER (measures dv/dt)
        # ============================================================
        # accel_x = dv/dt (forward acceleration in body frame)
        # We integrate to get velocity: v_new = v_old + accel_x * dt
        if len(self.accel_samples_x) >= self.bias_samples_needed:
            if self.prev_imu_time is not None:
                dt = (current_time - self.prev_imu_time).nanoseconds / 1e9
                if 0 < dt < 0.5:  # Valid dt
                    # Only use if acceleration is significant (filter noise)
                    if abs(accel_x_corrected) > 0.05:
                        # Integrate acceleration to get velocity estimate
                        v_from_accel = self.state[3] + accel_x_corrected * dt

                        # Measurement update for velocity
                        z_accel = np.array([v_from_accel])
                        H_accel = np.zeros((1, 5))
                        H_accel[0, 3] = 1.0  # Maps to state[3] (v)

                        y_accel = z_accel - H_accel @ self.state
                        S_accel = H_accel @ self.covariance @ H_accel.T + self.R_imu_accel

                        # Only update if measurement passes Mahalanobis gate
                        if self.mahalanobis_gate(y_accel, S_accel, self.gate_threshold_1dof):
                            K_accel = self.covariance @ H_accel.T @ np.linalg.inv(S_accel)
                            self.state = self.state + K_accel @ y_accel
                            self.covariance = (np.eye(5) - K_accel @ H_accel) @ self.covariance

                    # ============================================================
                    # UPDATE 3: CENTRIPETAL ACCELERATION (accel_y = v * omega)
                    # ============================================================
                    # During turning, lateral acceleration = centripetal acceleration
                    # accel_y = v * omega => omega = accel_y / v
                    v_current = self.state[3]
                    if abs(v_current) > 0.05 and abs(accel_y_corrected) > 0.02:
                        omega_from_centripetal = accel_y_corrected / v_current

                        z_omega = np.array([omega_from_centripetal])
                        H_omega = np.zeros((1, 5))
                        H_omega[0, 4] = 1.0

                        # Higher noise for indirect measurement
                        R_centripetal = np.diag([2.0])

                        y_omega = z_omega - H_omega @ self.state
                        S_omega = H_omega @ self.covariance @ H_omega.T + R_centripetal

                        # Only update if measurement passes Mahalanobis gate
                        if self.mahalanobis_gate(y_omega, S_omega, self.gate_threshold_1dof):
                            K_omega = self.covariance @ H_omega.T @ np.linalg.inv(S_omega)
                            self.state = self.state + K_omega @ y_omega
                            self.covariance = (np.eye(5) - K_omega @ H_omega) @ self.covariance

        self.prev_imu_time = current_time

    def joint_states_callback(self, msg):
        """
        Wheel encoder measurement update
        Computes v and omega from differential drive kinematics
        """
        if len(msg.position) < 2:
            return

        left_pos = msg.position[0]
        right_pos = msg.position[1]
        current_time = self.get_clock().now()

        if self.prev_joint_positions is not None and self.prev_time is not None:
            dt = (current_time - self.prev_time).nanoseconds / 1e9

            if dt > 0:
                # ============================================================
                # WHEEL ODOMETRY CALCULATION
                # ============================================================
                # Differential drive kinematics:
                #   v = (v_right + v_left) / 2
                #   omega = (v_right - v_left) / wheel_separation
                d_left = (left_pos - self.prev_joint_positions[0]) * self.wheel_radius
                d_right = (right_pos - self.prev_joint_positions[1]) * self.wheel_radius

                v_odom = (d_right + d_left) / (2.0 * dt)
                omega_odom = (d_right - d_left) / (self.wheel_separation * dt)

                # ============================================================
                # EKF PREDICTION STEP
                # ============================================================
                self.ekf_predict(dt)

                # ============================================================
                # EKF UPDATE: WHEEL ODOMETRY
                # ============================================================
                # Measurement: z = [v, omega] from wheel encoders
                z = np.array([v_odom, omega_odom])
                H = np.zeros((2, 5))
                H[0, 3] = 1.0  # v_odom maps to state[3] (v)
                H[1, 4] = 1.0  # omega_odom maps to state[4] (omega)

                y = z - H @ self.state
                S = H @ self.covariance @ H.T + self.R_odom

                # Only update if measurement passes Mahalanobis gate (2 DOF)
                if self.mahalanobis_gate(y, S, self.gate_threshold_2dof):
                    K = self.covariance @ H.T @ np.linalg.inv(S)
                    self.state = self.state + K @ y
                    self.covariance = (np.eye(5) - K @ H) @ self.covariance
                    self.state[2] = self.normalize_angle(self.state[2])

                # Log and publish
                self.log_trajectory(current_time)
                self.publish_odometry(current_time)

        self.prev_joint_positions = [left_pos, right_pos]
        self.prev_time = current_time

    def ekf_predict(self, dt):
        """
        EKF Prediction Step
        Uses differential drive motion model to predict next state
        """
        x, y, theta, v, omega = self.state

        # ============================================================
        # MOTION MODEL (Differential Drive)
        # ============================================================
        if abs(omega) < 1e-6:
            # Straight line motion (omega ≈ 0)
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            # Curved motion (arc)
            x_new = x + (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            y_new = y + (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
            theta_new = theta + omega * dt

        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = self.normalize_angle(theta_new)

        # ============================================================
        # JACOBIAN F = df/dx (linearization of motion model)
        # ============================================================
        F = np.eye(5)
        if abs(omega) < 1e-6:
            F[0, 2] = -v * math.sin(theta) * dt      # dx/dtheta
            F[1, 2] = v * math.cos(theta) * dt       # dy/dtheta
            F[0, 3] = math.cos(theta) * dt           # dx/dv
            F[1, 3] = math.sin(theta) * dt           # dy/dv
        else:
            F[0, 2] = (v / omega) * (math.cos(theta + omega * dt) - math.cos(theta))
            F[1, 2] = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            F[0, 3] = (1 / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            F[1, 3] = (1 / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
            F[0, 4] = -(v / (omega**2)) * (math.sin(theta + omega * dt) - math.sin(theta)) + (v / omega) * math.cos(theta + omega * dt) * dt
            F[1, 4] = -(v / (omega**2)) * (-math.cos(theta + omega * dt) + math.cos(theta)) + (v / omega) * math.sin(theta + omega * dt) * dt
            F[2, 4] = dt

        # ============================================================
        # COVARIANCE PREDICTION: P = F * P * F' + Q
        # ============================================================
        self.covariance = F @ self.covariance @ F.T + self.Q

        # Limit covariance to prevent explosion
        max_position_var = 10.0
        max_angle_var = 1.0
        max_vel_var = 5.0

        self.covariance[0, 0] = min(self.covariance[0, 0], max_position_var)
        self.covariance[1, 1] = min(self.covariance[1, 1], max_position_var)
        self.covariance[2, 2] = min(self.covariance[2, 2], max_angle_var)
        self.covariance[3, 3] = min(self.covariance[3, 3], max_vel_var)
        self.covariance[4, 4] = min(self.covariance[4, 4], max_vel_var)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def mahalanobis_gate(self, innovation, S, threshold):
        """
        Check if measurement passes Mahalanobis distance gate (outlier rejection)

        Args:
            innovation: y = z - H*x (measurement - prediction)
            S: Innovation covariance (H*P*H' + R)
            threshold: Chi-squared threshold for gating

        Returns:
            True if measurement is acceptable, False if outlier
        """
        # Mahalanobis distance squared: d² = y' * S^(-1) * y
        try:
            S_inv = np.linalg.inv(S)
            d_squared = innovation.T @ S_inv @ innovation

            self.total_measurements += 1

            if d_squared > threshold:
                self.outlier_count += 1
                # Log occasional outlier warnings (not every one to avoid spam)
                if self.outlier_count % 10 == 1:
                    self.get_logger().warn(
                        f'Outlier rejected: d²={d_squared:.2f} > {threshold:.2f} '
                        f'(total: {self.outlier_count}/{self.total_measurements})'
                    )
                return False
            return True
        except np.linalg.LinAlgError:
            # If S is singular, accept measurement
            return True

    def log_trajectory(self, timestamp):
        """Log trajectory to CSV"""
        with open(self.trajectory_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.nanoseconds,
                self.state[0], self.state[1], self.state[2],
                self.state[3], self.state[4],
                self.covariance[0, 0], self.covariance[1, 1], self.covariance[2, 2]
            ])

    def publish_odometry(self, timestamp):
        """Publish odometry message and TF"""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.z = math.sin(self.state[2] / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.state[2] / 2.0)

        odom_msg.twist.twist.linear.x = self.state[3]
        odom_msg.twist.twist.angular.z = self.state[4]

        self.odom_pub.publish(odom_msg)
        # Note: TF is published by publish_tf() timer at 50Hz for smooth transforms

    def publish_tf(self):
        """Publish TF at high frequency to prevent extrapolation errors"""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = self.state[0]
        tf_msg.transform.translation.y = self.state[1]
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z = math.sin(self.state[2] / 2.0)
        tf_msg.transform.rotation.w = math.cos(self.state[2] / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

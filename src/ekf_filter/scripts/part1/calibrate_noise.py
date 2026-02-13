#!/usr/bin/python3
"""
Sensor Noise Calibration Script
Collects static sensor data to estimate R (measurement noise) matrices.

Instructions:
1. Place robot on flat surface, completely stationary
2. Run this script: ros2 run ekf_filter calibrate_noise.py
3. Wait for data collection to complete (~10 seconds)
4. Use the printed values in your EKF
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
import numpy as np


class NoiseCalibrationNode(Node):
    def __init__(self):
        super().__init__('noise_calibration_node')

        # Data storage
        self.gyro_z_samples = []
        self.accel_x_samples = []
        self.accel_y_samples = []
        self.wheel_left_samples = []
        self.wheel_right_samples = []

        # Wheel velocity calculation
        self.prev_left_pos = None
        self.prev_right_pos = None
        self.prev_time = None

        # Robot parameters
        self.wheel_radius = 0.033
        self.wheel_separation = 0.160

        # Collection settings
        self.samples_needed = 1000
        self.collection_complete = False

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/read_data/imu', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/read_data/joint_states', self.joint_callback, 10)

        self.get_logger().info('=' * 60)
        self.get_logger().info('SENSOR NOISE CALIBRATION')
        self.get_logger().info('=' * 60)
        self.get_logger().info('IMPORTANT: Keep robot COMPLETELY STATIONARY!')
        self.get_logger().info(f'Collecting {self.samples_needed} samples...')
        self.get_logger().info('=' * 60)

    def imu_callback(self, msg):
        if self.collection_complete:
            return

        self.gyro_z_samples.append(msg.angular_velocity.z)
        self.accel_x_samples.append(msg.linear_acceleration.x)
        self.accel_y_samples.append(msg.linear_acceleration.y)

        # Progress update every 100 samples
        if len(self.gyro_z_samples) % 100 == 0:
            self.get_logger().info(f'IMU samples: {len(self.gyro_z_samples)}/{self.samples_needed}')

        self.check_completion()

    def joint_callback(self, msg):
        if self.collection_complete:
            return
        if len(msg.position) < 2:
            return

        current_time = self.get_clock().now()
        left_pos = msg.position[0]
        right_pos = msg.position[1]

        if self.prev_left_pos is not None and self.prev_time is not None:
            dt = (current_time - self.prev_time).nanoseconds / 1e9
            if dt > 0:
                # Calculate velocities
                v_left = (left_pos - self.prev_left_pos) * self.wheel_radius / dt
                v_right = (right_pos - self.prev_right_pos) * self.wheel_radius / dt

                v_linear = (v_right + v_left) / 2.0
                v_angular = (v_right - v_left) / self.wheel_separation

                self.wheel_left_samples.append(v_linear)
                self.wheel_right_samples.append(v_angular)

        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos
        self.prev_time = current_time

        self.check_completion()

    def check_completion(self):
        if (len(self.gyro_z_samples) >= self.samples_needed and
            len(self.wheel_left_samples) >= self.samples_needed - 1):
            self.collection_complete = True
            self.compute_noise_parameters()

    def compute_noise_parameters(self):
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('CALIBRATION RESULTS')
        self.get_logger().info('=' * 60)

        # IMU Gyroscope
        gyro_z = np.array(self.gyro_z_samples)
        gyro_var = np.var(gyro_z)
        gyro_std = np.std(gyro_z)
        gyro_mean = np.mean(gyro_z)

        self.get_logger().info('')
        self.get_logger().info('--- IMU GYROSCOPE (omega_z) ---')
        self.get_logger().info(f'  Mean:     {gyro_mean:.6f} rad/s (bias)')
        self.get_logger().info(f'  Std Dev:  {gyro_std:.6f} rad/s')
        self.get_logger().info(f'  Variance: {gyro_var:.6f} (rad/s)²')
        self.get_logger().info(f'  >> R_imu_gyro = np.diag([{gyro_var:.4f}])')

        # IMU Accelerometer X
        accel_x = np.array(self.accel_x_samples)
        accel_x_var = np.var(accel_x)
        accel_x_std = np.std(accel_x)
        accel_x_mean = np.mean(accel_x)

        self.get_logger().info('')
        self.get_logger().info('--- IMU ACCELEROMETER X ---')
        self.get_logger().info(f'  Mean:     {accel_x_mean:.6f} m/s² (bias)')
        self.get_logger().info(f'  Std Dev:  {accel_x_std:.6f} m/s²')
        self.get_logger().info(f'  Variance: {accel_x_var:.6f} (m/s²)²')
        self.get_logger().info(f'  >> R_imu_accel = np.diag([{accel_x_var:.4f}])')

        # IMU Accelerometer Y
        accel_y = np.array(self.accel_y_samples)
        accel_y_var = np.var(accel_y)
        accel_y_mean = np.mean(accel_y)

        self.get_logger().info('')
        self.get_logger().info('--- IMU ACCELEROMETER Y ---')
        self.get_logger().info(f'  Mean:     {accel_y_mean:.6f} m/s² (bias)')
        self.get_logger().info(f'  Variance: {accel_y_var:.6f} (m/s²)²')

        # Wheel Odometry
        v_linear = np.array(self.wheel_left_samples)
        v_angular = np.array(self.wheel_right_samples)
        v_linear_var = np.var(v_linear)
        v_angular_var = np.var(v_angular)

        self.get_logger().info('')
        self.get_logger().info('--- WHEEL ODOMETRY ---')
        self.get_logger().info(f'  Linear vel variance:  {v_linear_var:.6f} (m/s)²')
        self.get_logger().info(f'  Angular vel variance: {v_angular_var:.6f} (rad/s)²')
        self.get_logger().info(f'  >> R_odom = np.diag([{v_linear_var:.4f}, {v_angular_var:.4f}])')

        # Summary for copy-paste
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('COPY-PASTE VALUES FOR EKF:')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'self.R_imu_gyro = np.diag([{gyro_var:.4f}])')
        self.get_logger().info(f'self.R_imu_accel = np.diag([{accel_x_var:.4f}])')
        self.get_logger().info(f'self.R_odom = np.diag([{v_linear_var:.4f}, {v_angular_var:.4f}])')
        self.get_logger().info('')
        self.get_logger().info('# Bias values for initialization:')
        self.get_logger().info(f'self.accel_bias_x = {accel_x_mean:.6f}')
        self.get_logger().info(f'self.accel_bias_y = {accel_y_mean:.6f}')
        self.get_logger().info(f'self.gyro_bias_z = {gyro_mean:.6f}')
        self.get_logger().info('=' * 60)

        # Shutdown after completion
        self.get_logger().info('')
        self.get_logger().info('Calibration complete! You can now stop the node (Ctrl+C)')


def main(args=None):
    rclpy.init(args=args)
    node = NoiseCalibrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/python3
"""
Part 2: ICP Odometry Node
Uses ICP scan matching with EKF odometry as initial guess
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import csv
import os
from scipy.spatial import KDTree


class ICPOdometryNode(Node):
    def __init__(self):
        super().__init__('icp_odometry_node')

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/read_data/scan', self.scan_callback, 10)
        self.ekf_odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.ekf_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/icp/odometry', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # EKF state for initial guess
        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_theta = 0.0

        # Previous scan
        self.prev_scan_points = None
        self.prev_scan_time = None

        # ICP parameters
        self.max_iterations = 200
        self.tolerance = 1e-5
        self.max_correspondence_dist = 1.0  # TUNED: Allow more correspondences for noisy scans

        # Fusion parameters - BALANCED MODE
        self.ekf_trust_weight = 0.25  # TUNED: 25% EKF, 75% ICP (more balanced)
        self.max_icp_correction_trans = 0.15  # TUNED: Limit large ICP jumps (meters)
        self.max_icp_correction_rot = 0.3  # TUNED: Limit large rotations (radians ≈ 17°)
        self.enable_correction_limits = True  # ENABLED: Prevent ICP divergence

        # Data logging
        self.trajectory_file = '/home/prime/mobile_lab1/results/icp_odometry.csv'
        self.init_csv_file()

        self.scan_count = 0

        self.get_logger().info(f'ICP Odometry Node initialized - Saving to: {self.trajectory_file}')

    def init_csv_file(self):
        """Initialize CSV file"""
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)
        with open(self.trajectory_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'theta'])

    def ekf_callback(self, msg):
        """Store EKF odometry for initial guess"""
        self.ekf_x = msg.pose.pose.position.x
        self.ekf_y = msg.pose.pose.position.y

        # Extract yaw from quaternion (proper formula)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.ekf_theta = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        """Process laser scan with ICP"""
        self.scan_count += 1

        # Convert scan to points
        current_points = self.scan_to_points(msg)

        if current_points.shape[0] < 10:
            self.get_logger().warn(f'Not enough valid scan points: {current_points.shape[0]}')
            return

        current_time = self.get_clock().now()

        if self.prev_scan_points is not None:
            # Use EKF odometry as initial guess for transformation
            # Compute delta in GLOBAL frame
            dx_global = self.ekf_x - self.x
            dy_global = self.ekf_y - self.y
            dtheta = self.normalize_angle(self.ekf_theta - self.theta)

            # Convert global frame delta to LOCAL/ROBOT frame
            # ICP needs relative motion in robot's coordinate system
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            dx = dx_global * cos_theta + dy_global * sin_theta
            dy = -dx_global * sin_theta + dy_global * cos_theta

            # Run ICP
            T, icp_info = self.icp(self.prev_scan_points, current_points,
                        initial_transform=[dx, dy, dtheta])

            # Extract transformation
            dx_icp_raw = T[0]
            dy_icp_raw = T[1]
            dtheta_icp_raw = T[2]

            # Check ICP quality - increase EKF trust if ICP is uncertain
            icp_quality_good = (
                icp_info['converged'] and
                icp_info['num_correspondences'] > 30 and  # TUNED: Require sufficient matches
                icp_info['final_error'] < 0.08  # TUNED: Stricter error threshold
            )

            if not icp_quality_good:
                # Poor ICP quality - trust EKF more
                self.get_logger().warn(
                    f'Poor ICP quality: converged={icp_info["converged"]}, '
                    f'correspondences={icp_info["num_correspondences"]}, '
                    f'error={icp_info["final_error"]:.3f} - Trusting EKF more'
                )
                # Boost EKF trust significantly when ICP is poor
                alpha_effective = min(0.8, self.ekf_trust_weight + 0.4)
            else:
                # Good ICP quality - use base weight
                alpha_effective = self.ekf_trust_weight

            # Apply correction limits (if enabled)
            if self.enable_correction_limits:
                # Limit translation correction
                trans_correction = math.sqrt((dx_icp_raw - dx)**2 + (dy_icp_raw - dy)**2)
                if trans_correction > self.max_icp_correction_trans:
                    self.get_logger().info(
                        f'Translation limit activated: {trans_correction:.3f}m → {self.max_icp_correction_trans:.3f}m'
                    )
                    scale = self.max_icp_correction_trans / trans_correction
                    dx_icp_raw = dx + (dx_icp_raw - dx) * scale
                    dy_icp_raw = dy + (dy_icp_raw - dy) * scale

                # Limit rotation correction
                rot_correction = abs(dtheta_icp_raw - dtheta)
                if rot_correction > self.max_icp_correction_rot:
                    self.get_logger().info(
                        f'Rotation limit activated: {math.degrees(rot_correction):.1f}° → {math.degrees(self.max_icp_correction_rot):.1f}°'
                    )
                    if dtheta_icp_raw > dtheta:
                        dtheta_icp_raw = dtheta + self.max_icp_correction_rot
                    else:
                        dtheta_icp_raw = dtheta - self.max_icp_correction_rot

            # Blend EKF and ICP estimates
            # ekf_trust_weight = 0.0 means full ICP, 1.0 means full EKF
            # Use alpha_effective which adjusts based on ICP quality
            dx_final = alpha_effective * dx + (1 - alpha_effective) * dx_icp_raw
            dy_final = alpha_effective * dy + (1 - alpha_effective) * dy_icp_raw
            dtheta_final = alpha_effective * dtheta + (1 - alpha_effective) * dtheta_icp_raw

            # Update pose
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            self.x += dx_final * cos_theta - dy_final * sin_theta
            self.y += dx_final * sin_theta + dy_final * cos_theta
            self.theta = self.normalize_angle(self.theta + dtheta_final)

            # Log and publish
            self.log_trajectory(current_time)
            self.publish_odometry(current_time)

        # Store current scan
        self.prev_scan_points = current_points
        self.prev_scan_time = current_time

    def scan_to_points(self, scan_msg):
        """Convert LaserScan to 2D points"""
        points = []
        angle = scan_msg.angle_min

        for _, r in enumerate(scan_msg.ranges):
            if scan_msg.range_min < r < scan_msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append([x, y])
            angle += scan_msg.angle_increment

        return np.array(points)

    def icp(self, source, target, initial_transform=[0, 0, 0], max_iter=None, tol=None):
        """
        Iterative Closest Point algorithm
        source: Nx2 array of points
        target: Mx2 array of points
        initial_transform: [dx, dy, dtheta]
        Returns: ([dx, dy, dtheta], convergence_info)
        """
        if max_iter is None:
            max_iter = self.max_iterations
        if tol is None:
            tol = self.tolerance

        # Apply initial transformation to source
        dx, dy, dtheta = initial_transform
        source_transformed = self.transform_points(source, dx, dy, dtheta)

        # Build KD-tree for target
        tree = KDTree(target)

        prev_error = float('inf')
        num_valid_correspondences = 0
        converged = False

        for _ in range(max_iter):
            # Find closest points
            distances, indices = tree.query(source_transformed)

            # Filter correspondences by distance
            valid = distances < self.max_correspondence_dist
            num_valid_correspondences = np.sum(valid)

            if num_valid_correspondences < 10:  # TUNED: Need at least 10 for reliable ICP
                # ICP failed - not enough correspondences
                # This means ICP will return the INITIAL transformation (EKF guess) unchanged!
                converged = False
                prev_error = -1
                break

            source_valid = source_transformed[valid]
            target_valid = target[indices[valid]]

            # Compute transformation using SVD
            dt = self.compute_transformation(source_valid, target_valid)

            # Update transformation (proper 2D composition)
            # Rotation can be added directly
            dtheta_old = dtheta
            dtheta = self.normalize_angle(dtheta + dt[2])

            # Translation must be rotated from old frame
            cos_t = math.cos(dtheta_old)
            sin_t = math.sin(dtheta_old)
            dx += dt[0] * cos_t - dt[1] * sin_t
            dy += dt[0] * sin_t + dt[1] * cos_t

            # Apply transformation
            source_transformed = self.transform_points(source, dx, dy, dtheta)

            # Check convergence
            error = np.mean(distances[valid])
            if abs(prev_error - error) < tol:
                converged = True
                break

            prev_error = error

        # Return transformation and convergence info
        convergence_info = {
            'converged': converged,
            'num_correspondences': num_valid_correspondences,
            'final_error': prev_error if prev_error != float('inf') else -1
        }

        return [dx, dy, self.normalize_angle(dtheta)], convergence_info

    def compute_transformation(self, source, target):
        """
        Compute transformation between source and target using least squares
        Returns [dx, dy, dtheta]
        """
        # Compute centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)

        # Center the points
        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Extract rotation angle
        dtheta = math.atan2(R[1, 0], R[0, 0])

        # Compute translation
        t = target_centroid - R @ source_centroid

        return [t[0], t[1], dtheta]

    def transform_points(self, points, dx, dy, dtheta):
        """Transform points by [dx, dy, dtheta]"""
        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)

        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])
        t = np.array([dx, dy])

        return (R @ points.T).T + t

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def log_trajectory(self, timestamp):
        """Log trajectory"""
        with open(self.trajectory_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp.nanoseconds, self.x, self.y, self.theta])

    def publish_odometry(self, timestamp):
        """Publish odometry"""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link_icp'

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        self.odom_pub.publish(odom_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link_icp'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ICPOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

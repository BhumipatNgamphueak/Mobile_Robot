#!/usr/bin/python3
"""
Part 2: ICP Odometry Node
Pure ICP output, uses EKF odometry as initial guess only
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

        # EKF state for initial guess only
        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_theta = 0.0

        # Previous scan
        self.prev_scan_points = None
        self.prev_scan_time = None

        # ICP parameters
        self.max_iterations = 200
        self.tolerance = 1e-5
        self.max_correspondence_dist = 1.2  # Allow more correspondences (increased for better matching)

        # Adaptive trust based on ICP quality
        self.use_adaptive_trust = True  # Enable adaptive blending
        self.min_ekf_trust = 0.05  # Minimal EKF influence when ICP is confident
        self.max_ekf_trust = 0.35  # Moderate EKF trust when ICP is uncertain

        # Separate trust for rotation (IMU gyro is very accurate)
        self.rotation_ekf_bonus = 0.1  # Slight extra trust for rotation

        # Data logging
        self.trajectory_file = '/home/prime/Mobile_Robot/results/icp_odometry.csv'
        self.init_csv_file()

        self.scan_count = 0

        if self.use_adaptive_trust:
            self.get_logger().info(f'ICP Odometry Node initialized (Adaptive trust: EKF {self.min_ekf_trust*100:.0f}%-{self.max_ekf_trust*100:.0f}%)')
        else:
            avg_trust = (self.min_ekf_trust + self.max_ekf_trust) / 2.0
            self.get_logger().info(f'ICP Odometry Node initialized (EKF trust: {avg_trust*100:.0f}%, ICP: {(1-avg_trust)*100:.0f}%)')

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

        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.ekf_theta = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

    def scan_callback(self, msg):
        """Process laser scan with pure ICP (EKF as initial guess only)"""
        self.scan_count += 1

        # Convert scan to points
        current_points = self.scan_to_points(msg)

        if current_points.shape[0] < 10:
            self.get_logger().warn(f'Not enough valid scan points: {current_points.shape[0]}')
            return

        current_time = self.get_clock().now()

        if self.prev_scan_points is not None:
            # Use EKF delta as initial guess for ICP
            dx_global = self.ekf_x - self.x
            dy_global = self.ekf_y - self.y
            dtheta = self.normalize_angle(self.ekf_theta - self.theta)

            # Convert to local/robot frame for ICP
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            dx = dx_global * cos_theta + dy_global * sin_theta
            dy = -dx_global * sin_theta + dy_global * cos_theta

            # Run ICP
            T, icp_info = self.icp(self.prev_scan_points, current_points,
                        initial_transform=[dx, dy, dtheta])

            # Get ICP result
            dx_icp = T[0]
            dy_icp = T[1]
            dtheta_icp = T[2]

            # Debug: Show ICP vs EKF delta every 10 scans
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f'Motion - EKF: dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f} | '
                    f'ICP: dx={dx_icp:.3f}, dy={dy_icp:.3f}, dtheta={dtheta_icp:.3f}'
                )

            # Adaptive blending based on ICP quality
            if not icp_info['converged'] or icp_info['num_correspondences'] < 20:
                # Poor ICP - trust EKF completely
                ekf_trust = 1.0
                self.get_logger().warn(
                    f'Poor ICP: converged={icp_info["converged"]}, '
                    f'correspondences={icp_info["num_correspondences"]}, '
                    f'error={icp_info["final_error"]:.3f} - Using EKF (trust=100%)'
                )
            else:
                # Good ICP - compute adaptive trust based on match quality
                if self.use_adaptive_trust:
                    # Quality metrics
                    num_corr = icp_info['num_correspondences']
                    final_error = icp_info['final_error']

                    # More correspondences = trust ICP more (very relaxed: 50 correspondences is enough)
                    corr_score = min(num_corr / 50.0, 1.0)  # Normalize to [0,1]

                    # Lower error = trust ICP more (very relaxed: <0.3m error is acceptable)
                    error_score = max(0, 1.0 - (final_error / 0.3))  # <0.3m error is good

                    # Combined quality score
                    quality = (corr_score + error_score) / 2.0

                    # High quality -> low EKF trust (trust ICP)
                    # Low quality -> high EKF trust (trust EKF)
                    ekf_trust = self.max_ekf_trust - quality * (self.max_ekf_trust - self.min_ekf_trust)

                    # Debug logging every 10 scans
                    if self.scan_count % 10 == 0:
                        rotation_trust = min(1.0, ekf_trust + self.rotation_ekf_bonus)
                        self.get_logger().info(
                            f'ICP Quality: corr={num_corr} (score={corr_score:.2f}), '
                            f'error={final_error:.3f}m (score={error_score:.2f}), quality={quality:.2f}\n'
                            f'  → Translation: EKF {ekf_trust:.0%}, ICP {1-ekf_trust:.0%} | '
                            f'Rotation: EKF {rotation_trust:.0%}, ICP {1-rotation_trust:.0%}'
                        )
                else:
                    ekf_trust = (self.min_ekf_trust + self.max_ekf_trust) / 2.0

            # Blend ICP with EKF (separate trust for translation vs rotation)
            # Translation: use computed trust
            dx_icp = ekf_trust * dx + (1 - ekf_trust) * dx_icp
            dy_icp = ekf_trust * dy + (1 - ekf_trust) * dy_icp

            # Rotation: trust EKF more (IMU gyro is very accurate)
            rotation_trust = min(1.0, ekf_trust + self.rotation_ekf_bonus)
            dtheta_icp = rotation_trust * dtheta + (1 - rotation_trust) * dtheta_icp

            # Update pose in global frame (pure ICP output)
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            self.x += dx_icp * cos_theta - dy_icp * sin_theta
            self.y += dx_icp * sin_theta + dy_icp * cos_theta
            self.theta = self.normalize_angle(self.theta + dtheta_icp)

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

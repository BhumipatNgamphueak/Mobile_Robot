#!/usr/bin/env python3
"""
SLAM Trajectory Saver Node
Extracts SLAM poses from TF (map -> base_link) and saves to CSV
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
import csv
import os
import math
from ament_index_python.packages import get_package_share_directory


def _ws_results():
    share = get_package_share_directory('ekf_filter')
    ws = os.path.normpath(os.path.join(share, '..', '..', '..', '..'))
    return os.path.join(ws, 'results')


class SLAMTrajectorySaverNode(Node):
    def __init__(self):
        super().__init__('slam_trajectory_saver_node')

        # Parameters
        self.declare_parameter('output_dir', _ws_results())
        self.declare_parameter('output_file', 'slam_trajectory.csv')

        output_dir = self.get_parameter('output_dir').value
        output_file = self.get_parameter('output_file').value
        self.trajectory_path = os.path.join(output_dir, output_file)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open CSV file directly
        self.f = open(self.trajectory_path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['timestamp', 'x', 'y', 'theta'])
        self.f.flush()
        self.pose_count = 0

        self.get_logger().info(f'CSV file opened: {self.trajectory_path}')

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Track last saved timestamp to skip stale/duplicate transforms
        self.last_saved_timestamp = None

        # Timer to query TF at 10Hz
        self.timer = self.create_timer(0.1, self.save_pose_from_tf)

        self.get_logger().info(f'SLAM Trajectory Saver Node started')

    def save_pose_from_tf(self):
        """Query TF and save pose"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )

            stamp = transform.header.stamp
            timestamp = stamp.sec * 1_000_000_000 + stamp.nanosec

            # Skip if same timestamp as last saved (stale transform)
            if self.last_saved_timestamp is not None and timestamp == self.last_saved_timestamp:
                return

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            quat = transform.transform.rotation
            siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
            cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
            theta = math.atan2(siny_cosp, cosy_cosp)

            self.writer.writerow([timestamp, x, y, theta])
            self.f.flush()
            self.last_saved_timestamp = timestamp
            self.pose_count += 1

            if self.pose_count % 100 == 0:
                self.get_logger().info(f'Saved {self.pose_count} SLAM poses (x={x:.2f}, y={y:.2f})')

        except Exception as e:
            if self.pose_count == 0:
                self.get_logger().warn(
                    f'Waiting for TF map->base_link: {str(e)[:60]}',
                    throttle_duration_sec=5.0
                )


def main(args=None):
    rclpy.init(args=args)
    node = SLAMTrajectorySaverNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if hasattr(node, 'f') and node.f:
            node.f.close()
            node.get_logger().info(f'Saved {node.pose_count} SLAM poses to {node.trajectory_path}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

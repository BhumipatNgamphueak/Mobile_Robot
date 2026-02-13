#!/usr/bin/python3
"""
Scan Republisher Node
Republishes laser scans with updated timestamps and frame_id for SLAM
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanRepublisherNode(Node):
    def __init__(self):
        super().__init__('scan_republisher_node')

        # Subscribe to original scans
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/read_data/scan',
            self.scan_callback,
            10
        )

        # Publisher for republished scans
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        self.scan_count = 0

        self.get_logger().info('Scan Republisher Node initialized')
        self.get_logger().info('Subscribing to: /read_data/scan')
        self.get_logger().info('Publishing to: /scan (with base_link frame and current timestamps)')

    def scan_callback(self, msg):
        """Republish scan with updated frame and current timestamp"""
        # Create new message
        new_msg = LaserScan()

        # Use current timestamp and change frame to base_link
        new_msg.header.stamp = self.get_clock().now().to_msg()  # USE CURRENT TIME
        new_msg.header.frame_id = 'base_link'  # Change from base_scan to base_link

        # Copy all other fields
        new_msg.angle_min = msg.angle_min
        new_msg.angle_max = msg.angle_max
        new_msg.angle_increment = msg.angle_increment
        new_msg.time_increment = msg.time_increment
        new_msg.scan_time = msg.scan_time
        new_msg.range_min = msg.range_min
        new_msg.range_max = msg.range_max
        new_msg.ranges = msg.ranges
        new_msg.intensities = msg.intensities

        # Publish
        self.scan_pub.publish(new_msg)

        # Log progress
        self.scan_count += 1
        if self.scan_count % 20 == 0:
            self.get_logger().info(f'Republished {self.scan_count} scans (frame: base_link)')


def main(args=None):
    rclpy.init(args=args)
    node = ScanRepublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

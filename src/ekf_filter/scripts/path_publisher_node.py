#!/usr/bin/python3
"""
Path Publisher Node
Subscribes to odometry and publishes a path for visualization in RViz
Uses TF2 to properly transform poses from odom frame to map frame
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped


class PathPublisherNode(Node):
    def __init__(self):
        super().__init__('path_publisher_node')

        # Declare parameters
        self.declare_parameter('odom_topic', '/odom_republished')
        self.declare_parameter('path_topic', '/robot_path')
        self.declare_parameter('max_path_length', 10000)
        self.declare_parameter('sample_rate', 10)

        # Get parameters
        odom_topic = self.get_parameter('odom_topic').value
        path_topic = self.get_parameter('path_topic').value
        self.max_path_length = self.get_parameter('max_path_length').value
        self.sample_rate = self.get_parameter('sample_rate').value

        # TF2 buffer and listener for frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # Publisher for path
        self.path_pub = self.create_publisher(Path, path_topic, 10)

        # Path message
        self.path = Path()
        self.path.header.frame_id = 'map'

        # Counter for sampling
        self.odom_count = 0
        self.tf_ready = False

        self.get_logger().info(f'Path Publisher Node initialized')
        self.get_logger().info(f'Subscribing to: {odom_topic}')
        self.get_logger().info(f'Publishing to: {path_topic}')
        self.get_logger().info(f'Using TF2 to transform odom->map')

    def odom_callback(self, msg):
        """Add odometry pose to path and publish"""
        self.odom_count += 1

        # Sample at specified rate
        if self.odom_count % self.sample_rate != 0:
            return

        # Create PoseStamped from Odometry (in odom frame)
        pose_in_odom = PoseStamped()
        pose_in_odom.header = msg.header  # frame_id should be 'odom'
        pose_in_odom.pose = msg.pose.pose

        # Try to transform pose from odom to map frame
        try:
            # Look up transform from odom to map
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,  # usually 'odom'
                rclpy.time.Time(),  # Use latest available transform
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            # Transform the pose to map frame
            pose_in_map = do_transform_pose_stamped(pose_in_odom, transform)
            pose_in_map.header.frame_id = 'map'

            if not self.tf_ready:
                self.tf_ready = True
                self.get_logger().info('TF map->odom available, path now tracking in map frame')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            # If transform not available yet, use odom frame directly
            # This happens before SLAM initializes the map->odom transform
            if self.tf_ready:
                self.get_logger().warn(f'TF lookup failed: {e}')
            pose_in_map = pose_in_odom
            pose_in_map.header.frame_id = 'odom'  # Fall back to odom frame
            self.path.header.frame_id = 'odom'

        # Once we have map->odom, always use map frame
        if self.tf_ready:
            self.path.header.frame_id = 'map'
            pose_in_map.header.frame_id = 'map'

        # Add to path
        self.path.poses.append(pose_in_map)

        # Limit path length
        if len(self.path.poses) > self.max_path_length:
            self.path.poses.pop(0)

        # Update path timestamp
        self.path.header.stamp = self.get_clock().now().to_msg()

        # Publish path
        self.path_pub.publish(self.path)

        # Log progress
        if len(self.path.poses) % 100 == 0:
            self.get_logger().info(f'Path length: {len(self.path.poses)} poses')


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

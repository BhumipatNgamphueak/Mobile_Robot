#!/usr/bin/env python3
"""
ICP Map Builder Node
Builds a TRUE 2D occupancy grid map from ICP odometry + laser scans
Similar to SLAM but using ICP odometry instead of SLAM poses
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import numpy as np
import yaml
import os
from PIL import Image


class ICPMapBuilderNode(Node):
    def __init__(self):
        super().__init__('icp_map_builder_node')

        # Parameters
        self.declare_parameter('output_dir', '/home/prime/Mobile_Robot/results')
        self.declare_parameter('map_name', 'icp_map')
        self.declare_parameter('resolution', 0.05)  # 5cm per pixel (same as SLAM)
        self.declare_parameter('map_size_x', 50.0)  # meters
        self.declare_parameter('map_size_y', 50.0)  # meters

        self.output_dir = self.get_parameter('output_dir').value
        self.map_name = self.get_parameter('map_name').value
        self.resolution = self.get_parameter('resolution').value
        self.map_size_x = self.get_parameter('map_size_x').value
        self.map_size_y = self.get_parameter('map_size_y').value

        # Calculate grid dimensions
        self.width = int(self.map_size_x / self.resolution)
        self.height = int(self.map_size_y / self.resolution)
        self.origin_x = -self.map_size_x / 2.0
        self.origin_y = -self.map_size_y / 2.0

        # Initialize occupancy grid (unknown = -1)
        self.occupancy_grid = np.full((self.height, self.width), -1, dtype=np.int8)

        # Hit/miss counters for probabilistic mapping
        self.hit_count = np.zeros((self.height, self.width), dtype=np.int32)
        self.miss_count = np.zeros((self.height, self.width), dtype=np.int32)

        # Current pose from ICP odometry
        self.current_pose = None
        self.scan_count = 0

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/icp/odometry',  # ICP odometry topic
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/read_data/scan',
            self.scan_callback,
            10
        )

        # Publisher for visualization
        self.map_pub = self.create_publisher(OccupancyGrid, '/icp_map', 10)

        # Timer to publish map periodically
        self.publish_timer = self.create_timer(2.0, self.publish_map)

        # Timer to check for completion
        self.last_scan_time = None
        self.check_timer = self.create_timer(1.0, self.check_and_save)
        self.map_saved = False

        self.get_logger().info('ICP Map Builder Node started')
        self.get_logger().info(f'  Map resolution: {self.resolution} m/pixel')
        self.get_logger().info(f'  Map size: {self.map_size_x}x{self.map_size_y} m')
        self.get_logger().info(f'  Grid size: {self.width}x{self.height} pixels')
        self.get_logger().info('  Waiting for ICP odometry and laser scans...')

    def odom_callback(self, msg):
        """Update current pose from ICP odometry"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan and update map"""
        if self.current_pose is None:
            return

        self.last_scan_time = self.get_clock().now()
        self.scan_count += 1

        # Extract robot pose
        x = self.current_pose.position.x
        y = self.current_pose.position.y

        # Get yaw from quaternion
        quat = self.current_pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Process each laser ray
        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            # Skip invalid ranges
            if r < msg.range_min or r > msg.range_max or np.isnan(r) or np.isinf(r):
                angle += msg.angle_increment
                continue

            # Calculate endpoint in global frame
            beam_angle = yaw + angle
            end_x = x + r * np.cos(beam_angle)
            end_y = y + r * np.sin(beam_angle)

            # Mark endpoint as occupied (hit)
            self.mark_occupied(end_x, end_y)

            # Mark cells along the ray as free (misses)
            self.mark_ray_free(x, y, end_x, end_y)

            angle += msg.angle_increment

        # Log progress
        if self.scan_count % 50 == 0:
            self.get_logger().info(f'Processed {self.scan_count} scans')

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def mark_occupied(self, x, y):
        """Mark a cell as occupied"""
        grid_x, grid_y = self.world_to_grid(x, y)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            self.hit_count[grid_y, grid_x] += 1

    def mark_ray_free(self, x0, y0, x1, y1):
        """Mark cells along a ray as free using Bresenham's line algorithm"""
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)

        # Bresenham's line algorithm
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        x, y = gx0, gy0

        while True:
            # Mark as free (but not the endpoint)
            if (x, y) != (gx1, gy1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.miss_count[y, x] += 1

            if x == gx1 and y == gy1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def update_occupancy_grid(self):
        """Update occupancy grid based on hit/miss counts"""
        # Probabilistic occupancy calculation
        total_count = self.hit_count + self.miss_count

        # Avoid division by zero
        valid_mask = total_count > 0

        # Calculate occupancy probability (0-100)
        occupancy_prob = np.zeros((self.height, self.width))
        occupancy_prob[valid_mask] = (self.hit_count[valid_mask] /
                                      total_count[valid_mask] * 100)

        # Convert to ROS occupancy grid format (-1=unknown, 0-100=probability)
        self.occupancy_grid = np.full((self.height, self.width), -1, dtype=np.int8)

        # Free space (low probability of occupation)
        self.occupancy_grid[occupancy_prob < 25] = 0

        # Occupied space (high probability of occupation)
        self.occupancy_grid[occupancy_prob > 65] = 100

        # Uncertain space (keep as unknown)
        # occupancy_grid already initialized to -1

    def publish_map(self):
        """Publish current map for visualization"""
        if self.scan_count == 0:
            return

        self.update_occupancy_grid()

        # Create OccupancyGrid message
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'odom'

        # Map metadata
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.origin.position.x = self.origin_x
        map_msg.info.origin.position.y = self.origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten grid (row-major order)
        map_msg.data = self.occupancy_grid.flatten().tolist()

        self.map_pub.publish(map_msg)

    def check_and_save(self):
        """Check if mapping is done and save"""
        if self.map_saved:
            return

        if self.last_scan_time is None:
            return

        # Check if no scans received for 5 seconds
        time_since_last = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9

        if time_since_last > 5.0 and self.scan_count > 0:
            self.save_map()

    def save_map(self):
        """Save map to PGM and YAML files (same format as SLAM)"""
        if self.map_saved:
            return

        self.get_logger().info('=' * 70)
        self.get_logger().info('🗺️  AUTO-SAVING ICP MAP!')
        self.get_logger().info('=' * 70)

        # Update final occupancy grid
        self.update_occupancy_grid()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # File paths
        pgm_file = os.path.join(self.output_dir, f'{self.map_name}.pgm')
        yaml_file = os.path.join(self.output_dir, f'{self.map_name}.yaml')

        # Convert occupancy grid to image format (0-255)
        # ROS format: -1=unknown, 0=free, 100=occupied
        # PGM format: 0=black(occupied), 255=white(free), 205=unknown
        image_data = np.zeros((self.height, self.width), dtype=np.uint8)

        # Free space -> white (255)
        image_data[self.occupancy_grid == 0] = 254

        # Occupied space -> black (0)
        image_data[self.occupancy_grid == 100] = 0

        # Unknown space -> gray (205)
        image_data[self.occupancy_grid == -1] = 205

        # Flip vertically for correct orientation
        image_data = np.flipud(image_data)

        # Save PGM file
        img = Image.fromarray(image_data, mode='L')
        img.save(pgm_file)

        # Save YAML metadata
        yaml_data = {
            'image': os.path.basename(pgm_file),
            'mode': 'trinary',
            'resolution': float(self.resolution),
            'origin': [float(self.origin_x), float(self.origin_y), 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.25
        }

        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        self.map_saved = True

        self.get_logger().info(f'✓ ICP Map saved successfully!')
        self.get_logger().info(f'✓ Files created:')
        self.get_logger().info(f'  - {pgm_file} (occupancy grid image)')
        self.get_logger().info(f'  - {yaml_file} (map metadata)')
        self.get_logger().info(f'✓ Total scans processed: {self.scan_count}')
        self.get_logger().info(f'✓ Map size: {self.width}x{self.height} pixels')
        self.get_logger().info('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = ICPMapBuilderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        # Save map before shutdown
        if node.scan_count > 0 and not node.map_saved:
            node.get_logger().info('Shutdown detected - saving map...')
            node.save_map()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

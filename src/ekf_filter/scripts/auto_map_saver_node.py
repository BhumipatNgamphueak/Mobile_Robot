#!/usr/bin/env python3
"""
Automatic Map Saver Node for SLAM
Saves the map periodically and on shutdown directly from /map topic data
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from PIL import Image
import yaml
import os
from ament_index_python.packages import get_package_share_directory


def _ws_results():
    share = get_package_share_directory('ekf_filter')
    ws = os.path.normpath(os.path.join(share, '..', '..', '..', '..'))
    return os.path.join(ws, 'results')


class AutoMapSaverNode(Node):
    def __init__(self):
        super().__init__('auto_map_saver_node')

        # Parameters
        self.declare_parameter('output_dir', _ws_results())
        self.declare_parameter('map_name', 'slam_map')
        self.declare_parameter('save_interval', 30.0)  # Save every 30 seconds

        self.output_dir = self.get_parameter('output_dir').value
        self.map_name = self.get_parameter('map_name').value
        self.save_interval = self.get_parameter('save_interval').value

        # Subscribe to /map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # State
        self.latest_map = None
        self.map_count = 0
        self.last_save_time = None
        self.save_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info(f'Auto Map Saver started (saves every {self.save_interval}s)')

    def map_callback(self, msg):
        """Store latest map and save periodically"""
        self.latest_map = msg
        self.map_count += 1

        now = self.get_clock().now()

        # Save periodically
        if self.last_save_time is None:
            self.last_save_time = now
        else:
            elapsed = (now - self.last_save_time).nanoseconds / 1e9
            if elapsed >= self.save_interval:
                self.save_map()
                self.last_save_time = now

    def save_map(self):
        """Save map directly from stored OccupancyGrid data"""
        if self.latest_map is None:
            return

        msg = self.latest_map
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Convert occupancy data to image
        # ROS: -1=unknown, 0=free, 100=occupied
        # PGM: 205=unknown, 254=free, 0=occupied
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        image_data = np.full((height, width), 205, dtype=np.uint8)
        image_data[data == 0] = 254      # Free -> white
        image_data[data == 100] = 0      # Occupied -> black
        image_data = np.flipud(image_data)  # Flip for correct orientation

        # Save PGM
        pgm_path = os.path.join(self.output_dir, f'{self.map_name}.pgm')
        img = Image.fromarray(image_data, mode='L')
        img.save(pgm_path)

        # Save YAML
        yaml_path = os.path.join(self.output_dir, f'{self.map_name}.yaml')
        yaml_data = {
            'image': f'{self.map_name}.pgm',
            'mode': 'trinary',
            'resolution': float(resolution),
            'origin': [float(origin_x), float(origin_y), 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.25
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        self.save_count += 1
        self.get_logger().info(
            f'Map saved ({self.save_count}x) - {width}x{height} pixels, '
            f'{self.map_count} updates received'
        )


def main(args=None):
    rclpy.init(args=args)
    node = AutoMapSaverNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Save map one last time on shutdown
        if node.latest_map is not None:
            node.get_logger().info('Shutdown - saving final map...')
            node.save_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

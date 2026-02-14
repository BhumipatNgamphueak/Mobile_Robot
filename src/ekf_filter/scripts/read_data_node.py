#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
from sensor_msgs.msg import LaserScan, Imu, JointState
import os
from ament_index_python.packages import get_package_share_directory


def _ws_root():
    share = get_package_share_directory('ekf_filter')
    return os.path.normpath(os.path.join(share, '..', '..', '..', '..'))


class Read_data_Node(Node):
    def __init__(self):
        super().__init__('read_data_node')

        # Path to the bag file – override via: ros2 run ... --ros-args -p bag_path:=/your/path.db3
        _default_bag = os.path.join(_ws_root(), 'src', 'FRA532_LAB1_DATASET',
                                    'fibo_floor3_seq02', 'fibo_floor3_seq02_0.db3')
        self.declare_parameter('bag_path', _default_bag)
        self.bag_path = self.get_parameter('bag_path').get_parameter_value().string_value

        # Data storage
        self.scan_data = []
        self.imu_data = []
        self.joint_states_data = []

        # Publishers
        self.scan_pub = self.create_publisher(LaserScan, '/read_data/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/read_data/imu', 10)
        self.joint_states_pub = self.create_publisher(JointState, '/read_data/joint_states', 10)

        # Read bag file
        self.get_logger().info('Starting to read bag file...')
        self.read_bag_file()
        self.get_logger().info(f'Finished reading bag file:')
        self.get_logger().info(f'  - Scan messages: {len(self.scan_data)}')
        self.get_logger().info(f'  - IMU messages: {len(self.imu_data)}')
        self.get_logger().info(f'  - Joint states messages: {len(self.joint_states_data)}')

        # Publishing parameters
        self.current_index = 0
        self.publish_rate = 20.0  # Hz - publish at 20 Hz to match IMU rate

        # Create timer to publish data
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_data_callback)
        self.get_logger().info(f'Starting to publish data at {self.publish_rate} Hz...')

    def read_bag_file(self):
        """Read raw data from ROS2 bag file (db3 format)"""

        if not os.path.exists(self.bag_path):
            self.get_logger().error(f'Bag file not found: {self.bag_path}')
            return

        # Connect to SQLite database
        conn = sqlite3.connect(self.bag_path)
        cursor = conn.cursor()

        # Get topics information
        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()

        topic_id_map = {}
        topic_type_map = {}

        for topic_id, topic_name, topic_type in topics:
            topic_id_map[topic_name] = topic_id
            topic_type_map[topic_name] = topic_type
            self.get_logger().info(f'Found topic: {topic_name} (type: {topic_type})')

        # Read messages for each topic
        for topic_name, topic_id in topic_id_map.items():
            topic_type = topic_type_map[topic_name]

            # Get message type
            msg_type = get_message(topic_type)

            # Query messages for this topic
            cursor.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,)
            )

            rows = cursor.fetchall()
            self.get_logger().info(f'Reading {len(rows)} messages from {topic_name}...')

            for timestamp, data in rows:
                # Deserialize message
                msg = deserialize_message(data, msg_type)

                # Store based on topic
                if topic_name == '/scan':
                    self.scan_data.append({
                        'timestamp': timestamp,
                        'message': msg
                    })
                elif topic_name == '/imu':
                    self.imu_data.append({
                        'timestamp': timestamp,
                        'message': msg
                    })
                elif topic_name == '/joint_states':
                    self.joint_states_data.append({
                        'timestamp': timestamp,
                        'message': msg
                    })

        conn.close()

    def publish_data_callback(self):
        """Publish sensor data from bag file sequentially"""

        if self.current_index >= len(self.imu_data):
            self.get_logger().info('Finished publishing all data. Looping back to start...')
            self.current_index = 0
            return

        # Publish IMU data (most frequent - 20 Hz)
        if self.current_index < len(self.imu_data):
            imu_msg = self.imu_data[self.current_index]['message']
            self.imu_pub.publish(imu_msg)

        # Publish Joint States data (20 Hz)
        if self.current_index < len(self.joint_states_data):
            js_msg = self.joint_states_data[self.current_index]['message']
            self.joint_states_pub.publish(js_msg)

        # Publish Scan data (5 Hz - publish every 4th callback)
        scan_index = self.current_index // 4
        if self.current_index % 4 == 0 and scan_index < len(self.scan_data):
            scan_msg = self.scan_data[scan_index]['message']
            self.scan_pub.publish(scan_msg)

        self.current_index += 1

        # Log progress every 100 messages
        if self.current_index % 100 == 0:
            progress = (self.current_index / len(self.imu_data)) * 100
            self.get_logger().info(f'Publishing progress: {progress:.1f}%')

    def print_sample_data(self):
        """Print sample data from each topic for verification"""

        if self.scan_data:
            self.get_logger().info('\n=== Sample LaserScan Data ===')
            scan = self.scan_data[0]['message']
            self.get_logger().info(f'Timestamp: {self.scan_data[0]["timestamp"]}')
            self.get_logger().info(f'Angle min: {scan.angle_min}, max: {scan.angle_max}')
            self.get_logger().info(f'Range min: {scan.range_min}, max: {scan.range_max}')
            self.get_logger().info(f'Number of ranges: {len(scan.ranges)}')

        if self.imu_data:
            self.get_logger().info('\n=== Sample IMU Data ===')
            imu = self.imu_data[0]['message']
            self.get_logger().info(f'Timestamp: {self.imu_data[0]["timestamp"]}')
            self.get_logger().info(f'Angular velocity: x={imu.angular_velocity.x}, y={imu.angular_velocity.y}, z={imu.angular_velocity.z}')
            self.get_logger().info(f'Linear acceleration: x={imu.linear_acceleration.x}, y={imu.linear_acceleration.y}, z={imu.linear_acceleration.z}')

        if self.joint_states_data:
            self.get_logger().info('\n=== Sample JointState Data ===')
            js = self.joint_states_data[0]['message']
            self.get_logger().info(f'Timestamp: {self.joint_states_data[0]["timestamp"]}')
            self.get_logger().info(f'Joint names: {js.name}')
            self.get_logger().info(f'Positions: {js.position}')
            self.get_logger().info(f'Velocities: {js.velocity}')


def main(args=None):
    rclpy.init(args=args)
    node = Read_data_Node()

    # Print sample data for verification
    node.print_sample_data()

    # Keep node alive to publish data
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

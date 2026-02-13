#!/usr/bin/env python3
"""
View SLAM map from saved PGM file
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import os
import sys


def view_map(map_path=None):
    # Default path
    if map_path is None:
        map_path = '/home/prime/mobile_lab1/results/slam_map'

    # Handle both with and without extension
    if map_path.endswith('.pgm'):
        pgm_file = map_path
        yaml_file = map_path.replace('.pgm', '.yaml')
    elif map_path.endswith('.yaml'):
        yaml_file = map_path
        pgm_file = map_path.replace('.yaml', '.pgm')
    else:
        pgm_file = map_path + '.pgm'
        yaml_file = map_path + '.yaml'

    # Check if files exist
    if not os.path.exists(pgm_file):
        print(f"Error: Map file not found: {pgm_file}")
        print("\nTo save a map from slam_toolbox, run:")
        print("  ros2 run nav2_map_server map_saver_cli -f ~/mobile_lab1/results/slam_map")
        return

    # Load map image
    img = mpimg.imread(pgm_file)

    # Load metadata if available
    resolution = 0.05  # default
    origin = [0, 0, 0]
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            map_metadata = yaml.safe_load(f)
            resolution = map_metadata.get('resolution', 0.05)
            origin = map_metadata.get('origin', [0, 0, 0])
            print(f"Map resolution: {resolution} m/pixel")
            print(f"Map origin: {origin}")

    # Calculate map dimensions in meters
    height, width = img.shape[:2]
    map_width_m = width * resolution
    map_height_m = height * resolution
    print(f"Map size: {width}x{height} pixels ({map_width_m:.1f}x{map_height_m:.1f} meters)")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display map with proper extent (in meters)
    extent = [
        origin[0],
        origin[0] + map_width_m,
        origin[1],
        origin[1] + map_height_m
    ]

    im = ax.imshow(img, cmap='gray', origin='lower', extent=extent)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'SLAM Map\n{pgm_file}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Occupancy (0=free, 205=unknown, 254=occupied)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        view_map(sys.argv[1])
    else:
        view_map()

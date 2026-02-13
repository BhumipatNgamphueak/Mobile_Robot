#!/usr/bin/python3
"""
Build and visualize 2D occupancy map from ICP odometry + scan data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

def plot_trajectory_with_scans(odom_file, title="Map"):
    """Plot trajectory and scan points"""
    # Read odometry
    df = pd.read_csv(odom_file)

    # Extract poses
    x = df['x'].values
    y = df['y'].values
    theta = df['theta'].values

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Trajectory
    ax1.plot(x, y, 'r-', linewidth=2, label='ICP Path')
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax1.plot(x[-1], y[-1], 'rx', markersize=10, label='End')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title(f'{title} - Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # Plot 2: Direction arrows
    arrow_step = max(1, len(x) // 50)  # Show ~50 arrows
    for i in range(0, len(x), arrow_step):
        dx = 0.3 * np.cos(theta[i])
        dy = 0.3 * np.sin(theta[i])
        ax2.arrow(x[i], y[i], dx, dy, head_width=0.1, head_length=0.05,
                 fc='blue', ec='blue', alpha=0.6)

    ax2.plot(x, y, 'r-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title(f'{title} - Robot Orientations')
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(f'/home/prime/mobile_lab1/results/{title.lower().replace(" ", "_")}_map.png', dpi=150)
    plt.show()

    print(f"Map saved to results/{title.lower().replace(' ', '_')}_map.png")
    print(f"Total distance traveled: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)):.2f} meters")
    print(f"Start pose: ({x[0]:.2f}, {y[0]:.2f}, {np.degrees(theta[0]):.1f}°)")
    print(f"End pose: ({x[-1]:.2f}, {y[-1]:.2f}, {np.degrees(theta[-1]):.1f}°)")

def compare_trajectories():
    """Compare wheel, EKF, and ICP trajectories"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Load all trajectories
    files = {
        'Wheel': '/home/prime/mobile_lab1/results/wheel_odometry.csv',
        'EKF': '/home/prime/mobile_lab1/results/ekf_odometry.csv',
        'ICP': '/home/prime/mobile_lab1/results/icp_odometry.csv'
    }

    colors = {'Wheel': 'green', 'EKF': 'blue', 'ICP': 'red'}

    for name, file in files.items():
        try:
            df = pd.read_csv(file)
            x = df['x'].values
            y = df['y'].values
            ax.plot(x, y, color=colors[name], linewidth=2, label=name, alpha=0.8)
            ax.plot(x[0], y[0], 'o', color=colors[name], markersize=8)
            ax.plot(x[-1], y[-1], 'x', color=colors[name], markersize=10)

            print(f"{name} - Distance: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)):.2f}m, "
                  f"End: ({x[-1]:.2f}, {y[-1]:.2f})")
        except Exception as e:
            print(f"Could not load {name}: {e}")

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Odometry Comparison: Wheel vs EKF vs ICP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig('/home/prime/mobile_lab1/results/trajectory_comparison.png', dpi=150)
    plt.show()

    print("\nComparison saved to results/trajectory_comparison.png")

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Map Builder from CSV Trajectories")
    print("=" * 60)

    # Compare all trajectories
    print("\nGenerating trajectory comparison...")
    compare_trajectories()

    # Individual maps
    print("\nGenerating ICP map...")
    plot_trajectory_with_scans('/home/prime/mobile_lab1/results/icp_odometry.csv', 'ICP Odometry')

    print("\nGenerating EKF map...")
    plot_trajectory_with_scans('/home/prime/mobile_lab1/results/ekf_odometry.csv', 'EKF Odometry')

    print("\nDone! Check results/ folder for images.")

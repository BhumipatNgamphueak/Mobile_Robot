#!/usr/bin/python3
"""
Visualization script to plot and compare trajectories for each lab part
Includes IMU, Wheel Odometry, and EKF data comparisons

Usage:
    python3 plot_trajectories.py           # Plot all parts
    python3 plot_trajectories.py --part1   # Plot Part 1 only
    python3 plot_trajectories.py --part2   # Plot Part 2 only
    python3 plot_trajectories.py --part3   # Plot Part 3 only
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


RESULTS_DIR = '/home/prime/mobile_lab1/results'


def load_trajectory(filename):
    """Load trajectory from CSV file"""
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found")
        return None

    df = pd.read_csv(filename)
    return df


def compute_distance(traj):
    """Compute total distance traveled"""
    dx = np.diff(traj['x'].values)
    dy = np.diff(traj['y'].values)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)


def plot_part1():
    """
    Part 1: Wheel Odometry vs EKF Odometry Comparison
    Including IMU angular velocity comparison
    """
    print("\n" + "="*60)
    print("PART 1: Wheel Odometry vs EKF Odometry (with IMU)")
    print("="*60)

    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))
    imu_data = load_trajectory(os.path.join(RESULTS_DIR, 'imu_data.csv'))

    if wheel_odom is None and ekf_odom is None:
        print("No data available for Part 1. Run part1_ekf_fusion.launch.py first.")
        return

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Part 1: Sensor Fusion Comparison (IMU + Wheel Odom → EKF)', fontsize=14, fontweight='bold')

    # Plot 1: 2D Trajectory Comparison
    ax1 = plt.subplot(2, 3, 1)
    if wheel_odom is not None:
        ax1.plot(wheel_odom['x'].values, wheel_odom['y'].values, 'b-',
                 label='Wheel Odometry', linewidth=2, alpha=0.8)
        ax1.plot(wheel_odom['x'].iloc[0], wheel_odom['y'].iloc[0], 'go', markersize=10, label='Start')
        ax1.plot(wheel_odom['x'].iloc[-1], wheel_odom['y'].iloc[-1], 'bx', markersize=10, label='End (Wheel)')
    if ekf_odom is not None:
        ax1.plot(ekf_odom['x'].values, ekf_odom['y'].values, 'r-',
                 label='EKF Odometry', linewidth=2, alpha=0.8)
        ax1.plot(ekf_odom['x'].iloc[-1], ekf_odom['y'].iloc[-1], 'rx', markersize=10, label='End (EKF)')

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('2D Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Prepare time arrays
    time_wheel = None
    time_ekf = None
    time_imu = None
    if wheel_odom is not None:
        time_wheel = (wheel_odom['timestamp'].values - wheel_odom['timestamp'].iloc[0]) / 1e9
    if ekf_odom is not None:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9
    if imu_data is not None:
        time_imu = (imu_data['timestamp'].values - imu_data['timestamp'].iloc[0]) / 1e9

    # Plot 2: Angular Velocity Comparison (IMU vs Wheel vs EKF)
    ax2 = plt.subplot(2, 3, 2)
    if imu_data is not None:
        ax2.plot(time_imu, imu_data['omega_z'].values, 'g-', label='IMU (raw)', linewidth=1, alpha=0.6)
    if wheel_odom is not None and 'omega' in wheel_odom.columns:
        ax2.plot(time_wheel, wheel_odom['omega'].values, 'b-', label='Wheel Odom', linewidth=1.5, alpha=0.8)
    if ekf_odom is not None and 'omega' in ekf_odom.columns:
        ax2.plot(time_ekf, ekf_odom['omega'].values, 'r-', label='EKF (fused)', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity: IMU vs Wheel vs EKF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Linear Velocity (Wheel vs EKF)
    ax3 = plt.subplot(2, 3, 3)
    if wheel_odom is not None and 'v' in wheel_odom.columns:
        ax3.plot(time_wheel, wheel_odom['v'].values, 'b-', label='Wheel Odom', linewidth=1.5, alpha=0.8)
    if ekf_odom is not None and 'v' in ekf_odom.columns:
        ax3.plot(time_ekf, ekf_odom['v'].values, 'r-', label='EKF (fused)', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Linear Velocity: Wheel vs EKF')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: X Position vs Time
    ax4 = plt.subplot(2, 3, 4)
    if wheel_odom is not None:
        ax4.plot(time_wheel, wheel_odom['x'].values, 'b-', label='Wheel Odometry', linewidth=1.5)
    if ekf_odom is not None:
        ax4.plot(time_ekf, ekf_odom['x'].values, 'r-', label='EKF Odometry', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('X Position (meters)')
    ax4.set_title('X Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Y Position vs Time
    ax5 = plt.subplot(2, 3, 5)
    if wheel_odom is not None:
        ax5.plot(time_wheel, wheel_odom['y'].values, 'b-', label='Wheel Odometry', linewidth=1.5)
    if ekf_odom is not None:
        ax5.plot(time_ekf, ekf_odom['y'].values, 'r-', label='EKF Odometry', linewidth=1.5)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Y Position (meters)')
    ax5.set_title('Y Position vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Orientation vs Time
    ax6 = plt.subplot(2, 3, 6)
    if wheel_odom is not None:
        ax6.plot(time_wheel, np.degrees(wheel_odom['theta'].values), 'b-',
                 label='Wheel Odometry', linewidth=1.5)
    if ekf_odom is not None:
        ax6.plot(time_ekf, np.degrees(ekf_odom['theta'].values), 'r-',
                 label='EKF Odometry', linewidth=1.5)
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Orientation (degrees)')
    ax6.set_title('Orientation vs Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'part1_trajectory.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Part 1 plot saved to: {output_file}")

    # Print metrics
    print("\n--- Part 1 Metrics ---")
    if wheel_odom is not None:
        print(f"Wheel Odometry:")
        print(f"  Total distance: {compute_distance(wheel_odom):.2f} m")
        print(f"  Final position: ({wheel_odom['x'].iloc[-1]:.3f}, {wheel_odom['y'].iloc[-1]:.3f}) m")
        print(f"  Final orientation: {np.degrees(wheel_odom['theta'].iloc[-1]):.2f} deg")

    if ekf_odom is not None:
        print(f"EKF Odometry:")
        print(f"  Total distance: {compute_distance(ekf_odom):.2f} m")
        print(f"  Final position: ({ekf_odom['x'].iloc[-1]:.3f}, {ekf_odom['y'].iloc[-1]:.3f}) m")
        print(f"  Final orientation: {np.degrees(ekf_odom['theta'].iloc[-1]):.2f} deg")

    if imu_data is not None:
        print(f"IMU Data:")
        print(f"  Total samples: {len(imu_data)}")
        print(f"  Omega range: [{imu_data['omega_z'].min():.3f}, {imu_data['omega_z'].max():.3f}] rad/s")

    if wheel_odom is not None and ekf_odom is not None:
        diff = np.sqrt((ekf_odom['x'].iloc[-1] - wheel_odom['x'].iloc[-1])**2 +
                       (ekf_odom['y'].iloc[-1] - wheel_odom['y'].iloc[-1])**2)
        print(f"Final position difference (EKF vs Wheel): {diff:.3f} m")

    plt.show()


def plot_imu_detail():
    """
    Plot detailed IMU data analysis
    """
    print("\n" + "="*60)
    print("IMU Data Analysis")
    print("="*60)

    imu_data = load_trajectory(os.path.join(RESULTS_DIR, 'imu_data.csv'))
    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))

    if imu_data is None:
        print("No IMU data available. Run part1_ekf_fusion.launch.py first.")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('IMU Sensor Data Analysis', fontsize=14, fontweight='bold')

    time_imu = (imu_data['timestamp'].values - imu_data['timestamp'].iloc[0]) / 1e9

    # Plot 1: Angular Velocity (omega_z)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_imu, imu_data['omega_z'].values, 'g-', linewidth=0.8, alpha=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('IMU Angular Velocity (omega_z)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Linear Acceleration X
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time_imu, imu_data['accel_x'].values, 'm-', linewidth=0.8, alpha=0.8)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title('IMU Linear Acceleration X')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Linear Acceleration Y
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_imu, imu_data['accel_y'].values, 'c-', linewidth=0.8, alpha=0.8)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('IMU Linear Acceleration Y')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular Velocity Comparison
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time_imu, imu_data['omega_z'].values, 'g-', label='IMU (raw)', linewidth=0.8, alpha=0.6)
    if wheel_odom is not None and 'omega' in wheel_odom.columns:
        time_wheel = (wheel_odom['timestamp'].values - wheel_odom['timestamp'].iloc[0]) / 1e9
        ax4.plot(time_wheel, wheel_odom['omega'].values, 'b-', label='Wheel', linewidth=1.2, alpha=0.8)
    if ekf_odom is not None and 'omega' in ekf_odom.columns:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9
        ax4.plot(time_ekf, ekf_odom['omega'].values, 'r-', label='EKF', linewidth=1.2, alpha=0.8)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.set_title('Angular Velocity Comparison: IMU vs Wheel vs EKF')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'imu_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"IMU analysis plot saved to: {output_file}")
    plt.show()


def plot_part2():
    """
    Part 2: All Odometry Methods Comparison (Wheel, EKF, ICP)
    """
    print("\n" + "="*60)
    print("PART 2: Wheel vs EKF vs ICP Odometry")
    print("="*60)

    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))
    icp_odom = load_trajectory(os.path.join(RESULTS_DIR, 'icp_odometry.csv'))
    imu_data = load_trajectory(os.path.join(RESULTS_DIR, 'imu_data.csv'))

    if wheel_odom is None and ekf_odom is None and icp_odom is None:
        print("No data available for Part 2. Run part2_icp_refinement.launch.py first.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Part 2: Wheel vs EKF vs ICP Odometry Comparison', fontsize=14, fontweight='bold')

    # Plot 1: 2D Trajectory Comparison
    ax1 = plt.subplot(2, 3, 1)
    if wheel_odom is not None:
        ax1.plot(wheel_odom['x'].values, wheel_odom['y'].values, 'b-',
                 label='Wheel Odometry', linewidth=2, alpha=0.7)
    if ekf_odom is not None:
        ax1.plot(ekf_odom['x'].values, ekf_odom['y'].values, 'r-',
                 label='EKF Odometry', linewidth=2, alpha=0.7)
    if icp_odom is not None:
        ax1.plot(icp_odom['x'].values, icp_odom['y'].values, 'g-',
                 label='ICP Odometry', linewidth=2, alpha=0.7)

    if wheel_odom is not None:
        ax1.plot(wheel_odom['x'].iloc[0], wheel_odom['y'].iloc[0], 'ko', markersize=10, label='Start')

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('2D Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Prepare time arrays
    time_wheel, time_ekf, time_icp, time_imu = None, None, None, None
    if wheel_odom is not None:
        time_wheel = (wheel_odom['timestamp'].values - wheel_odom['timestamp'].iloc[0]) / 1e9
    if ekf_odom is not None:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9
    if icp_odom is not None:
        time_icp = (icp_odom['timestamp'].values - icp_odom['timestamp'].iloc[0]) / 1e9
    if imu_data is not None:
        time_imu = (imu_data['timestamp'].values - imu_data['timestamp'].iloc[0]) / 1e9

    # Plot 2: Angular Velocity (All sources)
    ax2 = plt.subplot(2, 3, 2)
    if imu_data is not None:
        ax2.plot(time_imu, imu_data['omega_z'].values, 'purple', label='IMU (raw)', linewidth=0.8, alpha=0.5)
    if wheel_odom is not None and 'omega' in wheel_odom.columns:
        ax2.plot(time_wheel, wheel_odom['omega'].values, 'b-', label='Wheel', linewidth=1.2, alpha=0.7)
    if ekf_odom is not None and 'omega' in ekf_odom.columns:
        ax2.plot(time_ekf, ekf_odom['omega'].values, 'r-', label='EKF', linewidth=1.2, alpha=0.7)
    if icp_odom is not None and 'omega' in icp_odom.columns:
        ax2.plot(time_icp, icp_odom['omega'].values, 'g-', label='ICP', linewidth=1.2, alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Linear Velocity
    ax3 = plt.subplot(2, 3, 3)
    if wheel_odom is not None and 'v' in wheel_odom.columns:
        ax3.plot(time_wheel, wheel_odom['v'].values, 'b-', label='Wheel', linewidth=1.2, alpha=0.7)
    if ekf_odom is not None and 'v' in ekf_odom.columns:
        ax3.plot(time_ekf, ekf_odom['v'].values, 'r-', label='EKF', linewidth=1.2, alpha=0.7)
    if icp_odom is not None and 'v' in icp_odom.columns:
        ax3.plot(time_icp, icp_odom['v'].values, 'g-', label='ICP', linewidth=1.2, alpha=0.7)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Linear Velocity Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: X Position vs Time
    ax4 = plt.subplot(2, 3, 4)
    if wheel_odom is not None:
        ax4.plot(time_wheel, wheel_odom['x'].values, 'b-', label='Wheel', linewidth=1.5, alpha=0.7)
    if ekf_odom is not None:
        ax4.plot(time_ekf, ekf_odom['x'].values, 'r-', label='EKF', linewidth=1.5, alpha=0.7)
    if icp_odom is not None:
        ax4.plot(time_icp, icp_odom['x'].values, 'g-', label='ICP', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('X Position (meters)')
    ax4.set_title('X Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Y Position vs Time
    ax5 = plt.subplot(2, 3, 5)
    if wheel_odom is not None:
        ax5.plot(time_wheel, wheel_odom['y'].values, 'b-', label='Wheel', linewidth=1.5, alpha=0.7)
    if ekf_odom is not None:
        ax5.plot(time_ekf, ekf_odom['y'].values, 'r-', label='EKF', linewidth=1.5, alpha=0.7)
    if icp_odom is not None:
        ax5.plot(time_icp, icp_odom['y'].values, 'g-', label='ICP', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Y Position (meters)')
    ax5.set_title('Y Position vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Orientation vs Time
    ax6 = plt.subplot(2, 3, 6)
    if wheel_odom is not None:
        ax6.plot(time_wheel, np.degrees(wheel_odom['theta'].values), 'b-', label='Wheel', linewidth=1.5, alpha=0.7)
    if ekf_odom is not None:
        ax6.plot(time_ekf, np.degrees(ekf_odom['theta'].values), 'r-', label='EKF', linewidth=1.5, alpha=0.7)
    if icp_odom is not None:
        ax6.plot(time_icp, np.degrees(icp_odom['theta'].values), 'g-', label='ICP', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Orientation (degrees)')
    ax6.set_title('Orientation vs Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'part2_trajectory.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Part 2 plot saved to: {output_file}")

    # Print metrics
    print("\n--- Part 2 Metrics ---")
    for name, traj in [('Wheel Odometry', wheel_odom),
                       ('EKF Odometry', ekf_odom),
                       ('ICP Odometry', icp_odom)]:
        if traj is not None:
            print(f"{name}:")
            print(f"  Total distance: {compute_distance(traj):.2f} m")
            print(f"  Final position: ({traj['x'].iloc[-1]:.3f}, {traj['y'].iloc[-1]:.3f}) m")
            print(f"  Final orientation: {np.degrees(traj['theta'].iloc[-1]):.2f} deg")

    plt.show()


def plot_part3():
    """
    Part 3: SLAM Trajectory
    """
    print("\n" + "="*60)
    print("PART 3: SLAM Trajectory")
    print("="*60)

    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))
    slam_odom = load_trajectory(os.path.join(RESULTS_DIR, 'slam_trajectory.csv'))

    if ekf_odom is None and slam_odom is None:
        print("No data available for Part 3. Run part3_slam.launch.py first.")
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Part 3: SLAM Trajectory with Loop Closure', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(2, 2, 1)

    if ekf_odom is not None:
        ax1.plot(ekf_odom['x'].values, ekf_odom['y'].values, 'r-',
                 label='EKF Odometry (Input to SLAM)', linewidth=2, alpha=0.7)
        ax1.plot(ekf_odom['x'].iloc[0], ekf_odom['y'].iloc[0], 'go',
                 markersize=12, label='Start', zorder=5)
        ax1.plot(ekf_odom['x'].iloc[-1], ekf_odom['y'].iloc[-1], 'rx',
                 markersize=12, mew=3, label='End (EKF)', zorder=5)

    if slam_odom is not None:
        ax1.plot(slam_odom['x'].values, slam_odom['y'].values, 'b-',
                 label='SLAM Corrected', linewidth=2, alpha=0.8)
        ax1.plot(slam_odom['x'].iloc[-1], slam_odom['y'].iloc[-1], 'bx',
                 markersize=12, mew=3, label='End (SLAM)', zorder=5)

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('2D Trajectory (SLAM)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    ax2 = plt.subplot(2, 2, 2)
    if ekf_odom is not None:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_ekf, ekf_odom['x'].values, 'r-', label='EKF', linewidth=1.5)
    if slam_odom is not None and 'timestamp' in slam_odom.columns:
        time_slam = (slam_odom['timestamp'].values - slam_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_slam, slam_odom['x'].values, 'b-', label='SLAM', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('X Position (meters)')
    ax2.set_title('X Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    if ekf_odom is not None:
        ax3.plot(time_ekf, ekf_odom['y'].values, 'r-', label='EKF', linewidth=1.5)
    if slam_odom is not None and 'timestamp' in slam_odom.columns:
        ax3.plot(time_slam, slam_odom['y'].values, 'b-', label='SLAM', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Y Position (meters)')
    ax3.set_title('Y Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    if ekf_odom is not None:
        ax4.plot(time_ekf, np.degrees(ekf_odom['theta'].values), 'r-', label='EKF', linewidth=1.5)
    if slam_odom is not None and 'theta' in slam_odom.columns:
        ax4.plot(time_slam, np.degrees(slam_odom['theta'].values), 'b-', label='SLAM', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Orientation (degrees)')
    ax4.set_title('Orientation vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'part3_trajectory.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Part 3 plot saved to: {output_file}")

    print("\n--- Part 3 Metrics ---")
    if ekf_odom is not None:
        print(f"EKF Odometry (SLAM input):")
        print(f"  Total distance: {compute_distance(ekf_odom):.2f} m")
        print(f"  Final position: ({ekf_odom['x'].iloc[-1]:.3f}, {ekf_odom['y'].iloc[-1]:.3f}) m")

        start_end_dist = np.sqrt((ekf_odom['x'].iloc[-1] - ekf_odom['x'].iloc[0])**2 +
                                  (ekf_odom['y'].iloc[-1] - ekf_odom['y'].iloc[0])**2)
        print(f"  Start-End distance: {start_end_dist:.3f} m")

    plt.show()


def plot_sensor_measurements():
    """
    Plot all raw sensor measurements separately
    - IMU Gyroscope (omega_z)
    - IMU Accelerometer (accel_x, accel_y, accel_z)
    - Wheel Encoder (v, omega)
    """
    print("\n" + "="*60)
    print("SENSOR MEASUREMENTS")
    print("="*60)

    imu_data = load_trajectory(os.path.join(RESULTS_DIR, 'imu_data.csv'))
    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))

    if imu_data is None and wheel_odom is None:
        print("No sensor data available. Run the launch file first.")
        return

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Raw Sensor Measurements', fontsize=14, fontweight='bold')

    # Prepare time arrays
    time_imu = None
    time_wheel = None
    time_ekf = None
    if imu_data is not None:
        time_imu = (imu_data['timestamp'].values - imu_data['timestamp'].iloc[0]) / 1e9
    if wheel_odom is not None:
        time_wheel = (wheel_odom['timestamp'].values - wheel_odom['timestamp'].iloc[0]) / 1e9
    if ekf_odom is not None:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9

    # ============================================================
    # ROW 1: IMU GYROSCOPE
    # ============================================================
    ax1 = plt.subplot(3, 2, 1)
    if imu_data is not None:
        ax1.plot(time_imu, imu_data['omega_z'].values, 'g-', linewidth=0.8, alpha=0.8, label='IMU Gyro')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('IMU Gyroscope (omega_z)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Statistics
    if imu_data is not None:
        stats_text = f"Mean: {imu_data['omega_z'].mean():.4f}\nStd: {imu_data['omega_z'].std():.4f}\nMin: {imu_data['omega_z'].min():.4f}\nMax: {imu_data['omega_z'].max():.4f}"
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # ROW 1: WHEEL ENCODER OMEGA
    # ============================================================
    ax2 = plt.subplot(3, 2, 2)
    if wheel_odom is not None and 'omega' in wheel_odom.columns:
        ax2.plot(time_wheel, wheel_odom['omega'].values, 'b-', linewidth=0.8, alpha=0.8, label='Wheel Encoder')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Wheel Encoder (omega)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if wheel_odom is not None and 'omega' in wheel_odom.columns:
        stats_text = f"Mean: {wheel_odom['omega'].mean():.4f}\nStd: {wheel_odom['omega'].std():.4f}\nMin: {wheel_odom['omega'].min():.4f}\nMax: {wheel_odom['omega'].max():.4f}"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # ROW 2: IMU ACCELEROMETER X (Forward)
    # ============================================================
    ax3 = plt.subplot(3, 2, 3)
    if imu_data is not None:
        ax3.plot(time_imu, imu_data['accel_x'].values, 'm-', linewidth=0.8, alpha=0.8, label='Accel X')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('IMU Accelerometer X (Forward)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    if imu_data is not None:
        stats_text = f"Mean: {imu_data['accel_x'].mean():.4f}\nStd: {imu_data['accel_x'].std():.4f}\nMin: {imu_data['accel_x'].min():.4f}\nMax: {imu_data['accel_x'].max():.4f}"
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # ROW 2: WHEEL ENCODER VELOCITY
    # ============================================================
    ax4 = plt.subplot(3, 2, 4)
    if wheel_odom is not None and 'v' in wheel_odom.columns:
        ax4.plot(time_wheel, wheel_odom['v'].values, 'b-', linewidth=0.8, alpha=0.8, label='Wheel Encoder')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Linear Velocity (m/s)')
    ax4.set_title('Wheel Encoder (v)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    if wheel_odom is not None and 'v' in wheel_odom.columns:
        stats_text = f"Mean: {wheel_odom['v'].mean():.4f}\nStd: {wheel_odom['v'].std():.4f}\nMin: {wheel_odom['v'].min():.4f}\nMax: {wheel_odom['v'].max():.4f}"
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # ROW 3: IMU ACCELEROMETER Y (Lateral / Centripetal)
    # ============================================================
    ax5 = plt.subplot(3, 2, 5)
    if imu_data is not None:
        ax5.plot(time_imu, imu_data['accel_y'].values, 'c-', linewidth=0.8, alpha=0.8, label='Accel Y')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.set_title('IMU Accelerometer Y (Lateral/Centripetal)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    if imu_data is not None:
        stats_text = f"Mean: {imu_data['accel_y'].mean():.4f}\nStd: {imu_data['accel_y'].std():.4f}\nMin: {imu_data['accel_y'].min():.4f}\nMax: {imu_data['accel_y'].max():.4f}"
        ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # ROW 3: IMU ACCELEROMETER Z (Vertical)
    # ============================================================
    ax6 = plt.subplot(3, 2, 6)
    if imu_data is not None and 'accel_z' in imu_data.columns:
        ax6.plot(time_imu, imu_data['accel_z'].values, 'orange', linewidth=0.8, alpha=0.8, label='Accel Z')
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.set_title('IMU Accelerometer Z (Vertical - includes gravity)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    if imu_data is not None and 'accel_z' in imu_data.columns:
        stats_text = f"Mean: {imu_data['accel_z'].mean():.4f}\nStd: {imu_data['accel_z'].std():.4f}\nMin: {imu_data['accel_z'].min():.4f}\nMax: {imu_data['accel_z'].max():.4f}"
        ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'sensor_measurements.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sensor measurements plot saved to: {output_file}")

    # Print sensor statistics
    print("\n--- Sensor Statistics ---")
    if imu_data is not None:
        print(f"\nIMU Gyroscope (omega_z):")
        print(f"  Samples: {len(imu_data)}")
        print(f"  Mean: {imu_data['omega_z'].mean():.4f} rad/s")
        print(f"  Std:  {imu_data['omega_z'].std():.4f} rad/s")
        print(f"  Range: [{imu_data['omega_z'].min():.4f}, {imu_data['omega_z'].max():.4f}] rad/s")

        print(f"\nIMU Accelerometer X (forward):")
        print(f"  Mean: {imu_data['accel_x'].mean():.4f} m/s²")
        print(f"  Std:  {imu_data['accel_x'].std():.4f} m/s²")
        print(f"  Range: [{imu_data['accel_x'].min():.4f}, {imu_data['accel_x'].max():.4f}] m/s²")

        print(f"\nIMU Accelerometer Y (lateral):")
        print(f"  Mean: {imu_data['accel_y'].mean():.4f} m/s²")
        print(f"  Std:  {imu_data['accel_y'].std():.4f} m/s²")
        print(f"  Range: [{imu_data['accel_y'].min():.4f}, {imu_data['accel_y'].max():.4f}] m/s²")

        if 'accel_z' in imu_data.columns:
            print(f"\nIMU Accelerometer Z (vertical):")
            print(f"  Mean: {imu_data['accel_z'].mean():.4f} m/s² (gravity ≈ 9.81)")
            print(f"  Std:  {imu_data['accel_z'].std():.4f} m/s²")

    if wheel_odom is not None:
        print(f"\nWheel Encoder (v):")
        print(f"  Samples: {len(wheel_odom)}")
        print(f"  Mean: {wheel_odom['v'].mean():.4f} m/s")
        print(f"  Std:  {wheel_odom['v'].std():.4f} m/s")
        print(f"  Range: [{wheel_odom['v'].min():.4f}, {wheel_odom['v'].max():.4f}] m/s")

        print(f"\nWheel Encoder (omega):")
        print(f"  Mean: {wheel_odom['omega'].mean():.4f} rad/s")
        print(f"  Std:  {wheel_odom['omega'].std():.4f} rad/s")
        print(f"  Range: [{wheel_odom['omega'].min():.4f}, {wheel_odom['omega'].max():.4f}] rad/s")

    plt.show()


def plot_ekf_covariance():
    """Plot EKF covariance over time"""
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))

    if ekf_odom is None or 'cov_xx' not in ekf_odom.columns:
        print("EKF covariance data not available")
        return

    time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('EKF Covariance (Uncertainty) Over Time', fontsize=14, fontweight='bold')

    axes[0].plot(time_ekf, np.sqrt(ekf_odom['cov_xx'].values), 'r-', linewidth=1.5)
    axes[0].fill_between(time_ekf, 0, np.sqrt(ekf_odom['cov_xx'].values), alpha=0.3, color='red')
    axes[0].set_ylabel('X Std Dev (m)')
    axes[0].set_title('X Position Uncertainty')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_ekf, np.sqrt(ekf_odom['cov_yy'].values), 'g-', linewidth=1.5)
    axes[1].fill_between(time_ekf, 0, np.sqrt(ekf_odom['cov_yy'].values), alpha=0.3, color='green')
    axes[1].set_ylabel('Y Std Dev (m)')
    axes[1].set_title('Y Position Uncertainty')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_ekf, np.degrees(np.sqrt(ekf_odom['cov_tt'].values)), 'b-', linewidth=1.5)
    axes[2].fill_between(time_ekf, 0, np.degrees(np.sqrt(ekf_odom['cov_tt'].values)), alpha=0.3, color='blue')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Theta Std Dev (degrees)')
    axes[2].set_title('Orientation Uncertainty')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(RESULTS_DIR, 'ekf_covariance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Covariance plot saved to: {output_file}")
    plt.show()


def print_usage():
    """Print usage instructions"""
    print("""
Usage: python3 plot_trajectories.py [OPTIONS]

Options:
    --part1     Plot Part 1 trajectory (Wheel vs EKF with IMU)
    --part2     Plot Part 2 trajectory (Wheel vs EKF vs ICP)
    --part3     Plot Part 3 trajectory (SLAM)
    --sensors   Plot all raw sensor measurements (IMU + Wheel)
    --imu       Plot detailed IMU analysis
    --cov       Plot EKF covariance
    (no args)   Plot all parts

Examples:
    python3 plot_trajectories.py --part1
    python3 plot_trajectories.py --sensors
    python3 plot_trajectories.py --imu
    """)


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    args = sys.argv[1:]

    if '--help' in args or '-h' in args:
        print_usage()
        sys.exit(0)

    if len(args) == 0:
        print("Plotting all parts...")
        plot_part1()
        plot_part2()
        plot_part3()
        plot_sensor_measurements()
        plot_imu_detail()
        plot_ekf_covariance()
    else:
        if '--part1' in args:
            plot_part1()
        if '--part2' in args:
            plot_part2()
        if '--part3' in args:
            plot_part3()
        if '--sensors' in args:
            plot_sensor_measurements()
        if '--imu' in args:
            plot_imu_detail()
        if '--cov' in args:
            plot_ekf_covariance()

    print("\nDone!")

#!/usr/bin/python3
"""
Generate Complete Deliverables for Mobile Robotics Lab
Creates all plots and comparison figures for:
- Wheel Odometry
- EKF Odometry
- ICP Odometry
- SLAM Trajectory (if available)
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

    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def compute_distance(traj):
    """Compute total distance traveled"""
    dx = np.diff(traj['x'].values)
    dy = np.diff(traj['y'].values)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)


def compute_drift(traj):
    """Compute drift (distance between start and end)"""
    dx = traj['x'].iloc[-1] - traj['x'].iloc[0]
    dy = traj['y'].iloc[-1] - traj['y'].iloc[0]
    return np.sqrt(dx**2 + dy**2)


def plot_complete_comparison():
    """
    Create comprehensive comparison plot with all odometry methods
    """
    print("\n" + "="*70)
    print("GENERATING COMPLETE TRAJECTORY COMPARISON")
    print("="*70)

    # Load all data
    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))
    icp_odom = load_trajectory(os.path.join(RESULTS_DIR, 'icp_odometry.csv'))
    slam_odom = load_trajectory(os.path.join(RESULTS_DIR, 'slam_trajectory.csv'))

    if not any([wheel_odom is not None, ekf_odom is not None, icp_odom is not None]):
        print("ERROR: No trajectory data found. Please run the experiments first.")
        return

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Complete Odometry Methods Comparison', fontsize=16, fontweight='bold')

    # ============================================================
    # Plot 1: 2D Trajectory Comparison (Main Plot)
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)

    if wheel_odom is not None:
        ax1.plot(wheel_odom['x'].values, wheel_odom['y'].values, 'b-',
                 label='Wheel Odometry', linewidth=2.5, alpha=0.7)
        ax1.plot(wheel_odom['x'].iloc[0], wheel_odom['y'].iloc[0], 'ko',
                 markersize=12, label='Start', zorder=10)

    if ekf_odom is not None:
        ax1.plot(ekf_odom['x'].values, ekf_odom['y'].values, 'r-',
                 label='EKF Odometry (IMU+Wheel)', linewidth=2.5, alpha=0.7)

    if icp_odom is not None:
        ax1.plot(icp_odom['x'].values, icp_odom['y'].values, 'g-',
                 label='ICP Odometry (Scan Matching)', linewidth=2.5, alpha=0.7)

    if slam_odom is not None:
        ax1.plot(slam_odom['x'].values, slam_odom['y'].values, 'm-',
                 label='SLAM (Loop Closure)', linewidth=2.5, alpha=0.8)

    ax1.set_xlabel('X Position (meters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (meters)', fontsize=11, fontweight='bold')
    ax1.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')

    # ============================================================
    # Plot 2: X Position vs Time
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)

    if wheel_odom is not None:
        time_wheel = (wheel_odom['timestamp'].values - wheel_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_wheel, wheel_odom['x'].values, 'b-', label='Wheel', linewidth=2, alpha=0.7)

    if ekf_odom is not None:
        time_ekf = (ekf_odom['timestamp'].values - ekf_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_ekf, ekf_odom['x'].values, 'r-', label='EKF', linewidth=2, alpha=0.7)

    if icp_odom is not None:
        time_icp = (icp_odom['timestamp'].values - icp_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_icp, icp_odom['x'].values, 'g-', label='ICP', linewidth=2, alpha=0.7)

    if slam_odom is not None and 'timestamp' in slam_odom.columns:
        time_slam = (slam_odom['timestamp'].values - slam_odom['timestamp'].iloc[0]) / 1e9
        ax2.plot(time_slam, slam_odom['x'].values, 'm-', label='SLAM', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('X Position (meters)', fontsize=11, fontweight='bold')
    ax2.set_title('X Position vs Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # Plot 3: Y Position vs Time
    # ============================================================
    ax3 = plt.subplot(2, 3, 3)

    if wheel_odom is not None:
        ax3.plot(time_wheel, wheel_odom['y'].values, 'b-', label='Wheel', linewidth=2, alpha=0.7)

    if ekf_odom is not None:
        ax3.plot(time_ekf, ekf_odom['y'].values, 'r-', label='EKF', linewidth=2, alpha=0.7)

    if icp_odom is not None:
        ax3.plot(time_icp, icp_odom['y'].values, 'g-', label='ICP', linewidth=2, alpha=0.7)

    if slam_odom is not None and 'timestamp' in slam_odom.columns:
        ax3.plot(time_slam, slam_odom['y'].values, 'm-', label='SLAM', linewidth=2, alpha=0.7)

    ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Y Position (meters)', fontsize=11, fontweight='bold')
    ax3.set_title('Y Position vs Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # Plot 4: Orientation vs Time
    # ============================================================
    ax4 = plt.subplot(2, 3, 4)

    if wheel_odom is not None:
        ax4.plot(time_wheel, np.degrees(wheel_odom['theta'].values), 'b-',
                 label='Wheel', linewidth=2, alpha=0.7)

    if ekf_odom is not None:
        ax4.plot(time_ekf, np.degrees(ekf_odom['theta'].values), 'r-',
                 label='EKF', linewidth=2, alpha=0.7)

    if icp_odom is not None:
        ax4.plot(time_icp, np.degrees(icp_odom['theta'].values), 'g-',
                 label='ICP', linewidth=2, alpha=0.7)

    if slam_odom is not None and 'theta' in slam_odom.columns:
        ax4.plot(time_slam, np.degrees(slam_odom['theta'].values), 'm-',
                 label='SLAM', linewidth=2, alpha=0.7)

    ax4.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Orientation (degrees)', fontsize=11, fontweight='bold')
    ax4.set_title('Orientation vs Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # Plot 5: Position Error (Drift) Comparison
    # ============================================================
    ax5 = plt.subplot(2, 3, 5)

    methods = []
    drifts = []
    colors = []

    if wheel_odom is not None:
        methods.append('Wheel')
        drifts.append(compute_drift(wheel_odom))
        colors.append('blue')

    if ekf_odom is not None:
        methods.append('EKF')
        drifts.append(compute_drift(ekf_odom))
        colors.append('red')

    if icp_odom is not None:
        methods.append('ICP')
        drifts.append(compute_drift(icp_odom))
        colors.append('green')

    if slam_odom is not None:
        methods.append('SLAM')
        drifts.append(compute_drift(slam_odom))
        colors.append('magenta')

    bars = ax5.bar(methods, drifts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, drift in zip(bars, drifts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{drift:.3f}m',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax5.set_ylabel('Drift (meters)', fontsize=11, fontweight='bold')
    ax5.set_title('Loop Closure Error (Start-End Distance)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # Plot 6: Total Distance Traveled
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)

    methods2 = []
    distances = []

    if wheel_odom is not None:
        methods2.append('Wheel')
        distances.append(compute_distance(wheel_odom))

    if ekf_odom is not None:
        methods2.append('EKF')
        distances.append(compute_distance(ekf_odom))

    if icp_odom is not None:
        methods2.append('ICP')
        distances.append(compute_distance(icp_odom))

    if slam_odom is not None:
        methods2.append('SLAM')
        distances.append(compute_distance(slam_odom))

    bars2 = ax6.bar(methods2, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, dist in zip(bars2, distances):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.2f}m',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax6.set_ylabel('Total Distance (meters)', fontsize=11, fontweight='bold')
    ax6.set_title('Total Distance Traveled', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(RESULTS_DIR, 'complete_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Complete comparison saved to: {output_file}")

    # Print detailed metrics
    print("\n" + "="*70)
    print("TRAJECTORY METRICS SUMMARY")
    print("="*70)

    for name, traj in [('Wheel Odometry', wheel_odom),
                       ('EKF Odometry', ekf_odom),
                       ('ICP Odometry', icp_odom),
                       ('SLAM Trajectory', slam_odom)]:
        if traj is not None:
            print(f"\n{name}:")
            print(f"  Total distance traveled: {compute_distance(traj):.3f} m")
            print(f"  Loop closure error (drift): {compute_drift(traj):.3f} m")
            print(f"  Final position: ({traj['x'].iloc[-1]:.3f}, {traj['y'].iloc[-1]:.3f}) m")
            print(f"  Final orientation: {np.degrees(traj['theta'].iloc[-1]):.2f}°")
            print(f"  Number of samples: {len(traj)}")

    plt.show()
    return output_file


def generate_analysis_report():
    """Generate detailed analysis and comparison report"""

    report_file = os.path.join(RESULTS_DIR, 'ANALYSIS_REPORT.md')

    # Load data
    wheel_odom = load_trajectory(os.path.join(RESULTS_DIR, 'wheel_odometry.csv'))
    ekf_odom = load_trajectory(os.path.join(RESULTS_DIR, 'ekf_odometry.csv'))
    icp_odom = load_trajectory(os.path.join(RESULTS_DIR, 'icp_odometry.csv'))
    slam_odom = load_trajectory(os.path.join(RESULTS_DIR, 'slam_trajectory.csv'))

    with open(report_file, 'w') as f:
        f.write("# Odometry Methods Comparison: Analysis and Discussion\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report compares four odometry methods for mobile robot localization:\n")
        f.write("1. **Wheel Odometry** (Dead reckoning)\n")
        f.write("2. **EKF Odometry** (Sensor fusion: Wheel + IMU)\n")
        f.write("3. **ICP Odometry** (Laser scan matching)\n")
        f.write("4. **SLAM** (Simultaneous Localization and Mapping with loop closure)\n\n")

        f.write("---\n\n")
        f.write("## 1. Quantitative Comparison\n\n")
        f.write("### Performance Metrics\n\n")
        f.write("| Method | Total Distance (m) | Loop Closure Error (m) | Samples | Drift Rate (%/m) |\n")
        f.write("|--------|-------------------|------------------------|---------|------------------|\n")

        for name, traj in [('Wheel Odometry', wheel_odom),
                           ('EKF Odometry', ekf_odom),
                           ('ICP Odometry', icp_odom),
                           ('SLAM', slam_odom)]:
            if traj is not None:
                dist = compute_distance(traj)
                drift = compute_drift(traj)
                drift_rate = (drift / dist * 100) if dist > 0 else 0
                f.write(f"| {name} | {dist:.3f} | {drift:.3f} | {len(traj)} | {drift_rate:.3f} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Method-by-Method Analysis\n\n")

        # Wheel Odometry
        f.write("### 2.1 Wheel Odometry\n\n")
        f.write("**Principle:** Dead reckoning using wheel encoder measurements\n\n")
        f.write("**Advantages:**\n")
        f.write("- Simple and fast computation\n")
        f.write("- No external sensors required\n")
        f.write("- High update rate\n\n")
        f.write("**Disadvantages:**\n")
        f.write("- Accumulates unbounded error over time\n")
        f.write("- Susceptible to wheel slip\n")
        f.write("- No absolute position reference\n\n")

        if wheel_odom is not None:
            drift = compute_drift(wheel_odom)
            dist = compute_distance(wheel_odom)
            f.write(f"**Observed Performance:**\n")
            f.write(f"- Loop closure error: {drift:.3f} m\n")
            f.write(f"- Total distance: {dist:.3f} m\n")
            f.write(f"- Drift rate: {(drift/dist*100):.2f}%\n\n")

        # EKF Odometry
        f.write("### 2.2 EKF Odometry (Sensor Fusion)\n\n")
        f.write("**Principle:** Extended Kalman Filter fusing wheel encoders + IMU gyroscope\n\n")
        f.write("**Advantages:**\n")
        f.write("- Improved orientation estimate from IMU\n")
        f.write("- Probabilistic state estimation with covariance\n")
        f.write("- Handles sensor noise optimally\n")
        f.write("- Better than wheel-only in turns\n\n")
        f.write("**Disadvantages:**\n")
        f.write("- Still accumulates drift (no absolute reference)\n")
        f.write("- Requires careful noise parameter tuning\n")
        f.write("- IMU bias/drift affects long-term accuracy\n\n")

        if ekf_odom is not None:
            drift = compute_drift(ekf_odom)
            dist = compute_distance(ekf_odom)
            f.write(f"**Observed Performance:**\n")
            f.write(f"- Loop closure error: {drift:.3f} m\n")
            f.write(f"- Total distance: {dist:.3f} m\n")
            f.write(f"- Drift rate: {(drift/dist*100):.2f}%\n")

            if wheel_odom is not None:
                wheel_drift = compute_drift(wheel_odom)
                improvement = ((wheel_drift - drift) / wheel_drift * 100)
                f.write(f"- Improvement over wheel odometry: {improvement:.1f}%\n\n")
            else:
                f.write("\n")

        # ICP Odometry
        f.write("### 2.3 ICP Odometry (Scan Matching)\n\n")
        f.write("**Principle:** Iterative Closest Point algorithm matching consecutive laser scans\n\n")
        f.write("**Advantages:**\n")
        f.write("- Corrects for wheel slip by observing environment\n")
        f.write("- Works in feature-rich environments\n")
        f.write("- Can be more accurate than wheel odometry in short term\n")
        f.write("- Fuses with EKF for robustness\n\n")
        f.write("**Disadvantages:**\n")
        f.write("- Fails in featureless environments (corridors, open spaces)\n")
        f.write("- Computationally expensive\n")
        f.write("- Local minima can cause incorrect matches\n")
        f.write("- Still accumulates drift without loop closure\n\n")

        if icp_odom is not None:
            drift = compute_drift(icp_odom)
            dist = compute_distance(icp_odom)
            f.write(f"**Observed Performance:**\n")
            f.write(f"- Loop closure error: {drift:.3f} m\n")
            f.write(f"- Total distance: {dist:.3f} m\n")
            f.write(f"- Drift rate: {(drift/dist*100):.2f}%\n")

            if ekf_odom is not None:
                ekf_drift = compute_drift(ekf_odom)
                improvement = ((ekf_drift - drift) / ekf_drift * 100) if ekf_drift > 0 else 0
                f.write(f"- Improvement over EKF odometry: {improvement:.1f}%\n\n")
            else:
                f.write("\n")

        # SLAM
        f.write("### 2.4 SLAM (slam_toolbox)\n\n")
        f.write("**Principle:** Graph-based SLAM with scan matching and loop closure detection\n\n")
        f.write("**Advantages:**\n")
        f.write("- Loop closure dramatically reduces drift\n")
        f.write("- Builds consistent map of environment\n")
        f.write("- Can correct for accumulated errors retrospectively\n")
        f.write("- Best long-term accuracy\n\n")
        f.write("**Disadvantages:**\n")
        f.write("- Most computationally expensive\n")
        f.write("- Requires loop closure opportunities\n")
        f.write("- Complex parameter tuning\n")
        f.write("- Can fail in dynamic/changing environments\n\n")

        if slam_odom is not None:
            drift = compute_drift(slam_odom)
            dist = compute_distance(slam_odom)
            f.write(f"**Observed Performance:**\n")
            f.write(f"- Loop closure error: {drift:.3f} m\n")
            f.write(f"- Total distance: {dist:.3f} m\n")
            f.write(f"- Drift rate: {(drift/dist*100):.2f}%\n\n")

        f.write("---\n\n")
        f.write("## 3. Key Findings\n\n")
        f.write("### Accuracy Ranking (Best to Worst)\n")

        methods_drifts = []
        if slam_odom is not None:
            methods_drifts.append(('SLAM', compute_drift(slam_odom)))
        if icp_odom is not None:
            methods_drifts.append(('ICP', compute_drift(icp_odom)))
        if ekf_odom is not None:
            methods_drifts.append(('EKF', compute_drift(ekf_odom)))
        if wheel_odom is not None:
            methods_drifts.append(('Wheel', compute_drift(wheel_odom)))

        methods_drifts.sort(key=lambda x: x[1])

        for i, (name, drift) in enumerate(methods_drifts, 1):
            f.write(f"{i}. **{name}** - {drift:.3f}m drift\n")

        f.write("\n### Robustness Observations\n\n")
        f.write("- **Wheel Odometry**: Consistent but accumulates error linearly\n")
        f.write("- **EKF**: More robust to orientation errors than wheel-only\n")
        f.write("- **ICP**: Quality depends on environment structure\n")
        f.write("- **SLAM**: Best overall, but requires good loop closure\n\n")

        f.write("---\n\n")
        f.write("## 4. Conclusions\n\n")
        f.write("1. **For short trajectories** (<10m): Wheel or EKF odometry sufficient\n")
        f.write("2. **For medium trajectories**: ICP odometry provides good balance\n")
        f.write("3. **For long trajectories with loops**: SLAM is essential\n")
        f.write("4. **Real-world deployment**: Multi-sensor fusion (EKF + ICP) with SLAM backend recommended\n\n")

        f.write("## 5. Recommendations\n\n")
        f.write("**Application-Specific Guidance:**\n\n")
        f.write("- **Warehouse robots**: ICP + SLAM (structured environment, loop closures)\n")
        f.write("- **Outdoor robots**: EKF with GPS (open spaces, poor scan matching)\n")
        f.write("- **Indoor navigation**: SLAM with visual features\n")
        f.write("- **Resource-constrained**: EKF odometry (good accuracy/cost tradeoff)\n\n")

        f.write("---\n\n")
        f.write(f"*Report generated automatically from experimental data*\n")

    print(f"✓ Analysis report saved to: {report_file}")
    return report_file


def main():
    """Main function to generate all deliverables"""
    print("="*70)
    print("MOBILE ROBOTICS LAB - DELIVERABLES GENERATOR")
    print("="*70)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate plots
    print("\n[1/2] Generating trajectory comparison plots...")
    plot_complete_comparison()

    # Generate analysis report
    print("\n[2/2] Generating analysis report...")
    generate_analysis_report()

    print("\n" + "="*70)
    print("✓ ALL DELIVERABLES GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    print("  - complete_comparison.png (Trajectory plots)")
    print("  - ANALYSIS_REPORT.md (Detailed analysis)")
    print("\nNote: Maps are saved separately from SLAM runs")
    print("="*70)


if __name__ == '__main__':
    main()

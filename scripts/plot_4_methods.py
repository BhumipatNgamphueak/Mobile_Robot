#!/usr/bin/env python3
"""Plot trajectories from 4 methods separately and as overlay"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_RESULTS = Path(__file__).resolve().parent.parent / 'results'

def read_csv(path, x_col='x', y_col='y'):
    x, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[x_col]))
            y.append(float(row[y_col]))
    return np.array(x), np.array(y)

# Read data
wheel_x, wheel_y = read_csv(_RESULTS / 'wheel_odometry.csv')
ekf_x, ekf_y = read_csv(_RESULTS / 'ekf_odometry.csv')
icp_x, icp_y = read_csv(_RESULTS / 'icp_odometry.csv')
slam_x, slam_y = read_csv(_RESULTS / 'slam_trajectory.csv')

# --- Individual plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Trajectory Comparison - 4 Methods (Separately)', fontsize=16, fontweight='bold')

methods = [
    (axes[0,0], wheel_x, wheel_y, 'Wheel Odometry', 'gray', len(wheel_x)),
    (axes[0,1], ekf_x, ekf_y, 'EKF Odometry', 'blue', len(ekf_x)),
    (axes[1,0], icp_x, icp_y, 'ICP Odometry', 'red', len(icp_x)),
    (axes[1,1], slam_x, slam_y, 'SLAM', 'green', len(slam_x)),
]

for ax, x, y, title, color, n in methods:
    ax.plot(x, y, color=color, linewidth=1.2, alpha=0.8)
    ax.plot(x[0], y[0], 'ko', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'k^', markersize=8, label='End')
    ax.set_title(f'{title} ({n} points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(_RESULTS / 'four_methods_separate.png', dpi=150, bbox_inches='tight')
print('Saved: four_methods_separate.png')

# --- Overlay plot ---
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
ax2.plot(wheel_x, wheel_y, color='gray', linewidth=1.0, alpha=0.6, label=f'Wheel Odometry ({len(wheel_x)} pts)')
ax2.plot(ekf_x, ekf_y, color='blue', linewidth=1.2, alpha=0.8, label=f'EKF ({len(ekf_x)} pts)')
ax2.plot(icp_x, icp_y, color='red', linewidth=1.2, alpha=0.8, label=f'ICP ({len(icp_x)} pts)')
ax2.plot(slam_x, slam_y, color='green', linewidth=1.5, alpha=0.9, label=f'SLAM ({len(slam_x)} pts)')
ax2.plot(slam_x[0], slam_y[0], 'ko', markersize=10, label='Start')
ax2.set_title('All 4 Methods Overlay', fontsize=15, fontweight='bold')
ax2.set_xlabel('X (m)', fontsize=12)
ax2.set_ylabel('Y (m)', fontsize=12)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(_RESULTS / 'four_methods_overlay.png', dpi=150, bbox_inches='tight')
print('Saved: four_methods_overlay.png')

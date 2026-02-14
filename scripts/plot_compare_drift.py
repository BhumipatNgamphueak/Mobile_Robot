#!/usr/bin/env python3
"""
Plot compare_drift run for sequence_0:
  1. ICP map vs SLAM map – side by side (with real-world coords)
  2. 4-method trajectory overlay
Saves:
  results/sequence_0/compare_drift/compare_drift_maps.png
  results/sequence_0/compare_drift/compare_drift_trajectories.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parent.parent / 'results' / 'sequence_0' / 'compare_drift'


# ─── helpers ────────────────────────────────────────────────────────────────

def load_pgm(pgm_path, yaml_path):
    img = mpimg.imread(str(pgm_path))
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    res = float(meta['resolution'])
    ox, oy = meta['origin'][0], meta['origin'][1]
    h, w = img.shape[:2]
    extent = [ox, ox + w * res, oy, oy + h * res]
    return img, extent


def load_csv(path, xcol='x', ycol='y'):
    df = pd.read_csv(path)
    return df[xcol].values, df[ycol].values


# ─── Figure 1: map comparison ────────────────────────────────────────────────

def plot_maps():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        'Compare-Drift Run – Sequence 0\n'
        'ICP Map vs SLAM Map (EKF-guided)',
        fontsize=14, fontweight='bold'
    )

    # ICP map
    icp_img, icp_ext = load_pgm(BASE / 'icp_map.pgm', BASE / 'icp_map.yaml')
    axes[0].imshow(icp_img, cmap='gray', origin='upper',
                   extent=icp_ext, interpolation='nearest', aspect='equal')
    axes[0].set_title('ICP Map\n(LiDAR scans projected from ICP poses)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].grid(True, alpha=0.2)

    # SLAM map
    slam_img, slam_ext = load_pgm(BASE / 'slam_map.pgm', BASE / 'slam_map.yaml')
    axes[1].imshow(slam_img, cmap='gray', origin='upper',
                   extent=slam_ext, interpolation='nearest', aspect='equal')
    axes[1].set_title('SLAM Map\n(slam_toolbox + EKF odometry + loop closure)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    out = BASE / 'compare_drift_maps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─── Figure 2: 4-method trajectory overlay ───────────────────────────────────

def plot_trajectories():
    files = {
        'Wheel Odometry': ('gray',  BASE / 'wheel_odometry.csv'),
        'EKF Odometry':   ('blue',  BASE / 'ekf_odometry.csv'),
        'ICP Odometry':   ('red',   BASE / 'icp_odometry.csv'),
        'SLAM':           ('green', BASE / 'slam_trajectory.csv'),
    }

    # compute loop-closure error for each method
    def lce(x, y):
        return np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

    def drift(x, y):
        d = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        return lce(x, y) / d * 100 if d > 0 else float('nan')

    # ── subplot layout: overlay + 4 individual ──
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Compare-Drift Run – Sequence 0: 4-Method Trajectory Comparison',
                 fontsize=14, fontweight='bold')

    ax_all = fig.add_subplot(2, 3, (1, 4))   # big overlay on left
    ind_axes = [fig.add_subplot(2, 3, i) for i in [2, 3, 5, 6]]

    colors = ['gray', 'blue', 'red', 'green']
    loaded = {}

    for (label, (color, path)), ax_ind in zip(files.items(), ind_axes):
        if not path.exists():
            for ax in [ax_all, ax_ind]:
                ax.text(0.5, 0.5, f'{label}\nnot found',
                        ha='center', va='center', transform=ax.transAxes)
            continue

        x, y = load_csv(path)
        loaded[label] = (x, y, color)
        err = lce(x, y)
        dr  = drift(x, y)

        # overlay
        ax_all.plot(x, y, color=color, linewidth=1.2, alpha=0.8,
                    label=f'{label} (LCE={err:.2f}m, {dr:.1f}%)')

        # individual
        ax_ind.plot(x, y, color=color, linewidth=1.2)
        ax_ind.plot(x[0], y[0], 'ko', markersize=7, label='Start')
        ax_ind.plot(x[-1], y[-1], 'k^', markersize=7, label='End')
        ax_ind.set_title(f'{label}\nLCE={err:.2f} m  Drift={dr:.1f}%',
                         fontsize=10, fontweight='bold')
        ax_ind.set_xlabel('X (m)', fontsize=9)
        ax_ind.set_ylabel('Y (m)', fontsize=9)
        ax_ind.set_aspect('equal')
        ax_ind.grid(True, alpha=0.3)
        ax_ind.legend(fontsize=8)

    # mark start on overlay
    if 'SLAM' in loaded:
        sx, sy, _ = loaded['SLAM']
        ax_all.plot(sx[0], sy[0], 'ko', markersize=10, zorder=5, label='Start')

    ax_all.set_title('All 4 Methods – Overlay', fontsize=12, fontweight='bold')
    ax_all.set_xlabel('X (m)')
    ax_all.set_ylabel('Y (m)')
    ax_all.set_aspect('equal')
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(fontsize=9)

    plt.tight_layout()
    out = BASE / 'compare_drift_trajectories.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─── Figure 3: ICP map with trajectory overlays ──────────────────────────────

def plot_trajectories_on_map():
    """Overlay all 4 trajectories on top of the ICP occupancy map."""
    files = {
        'Wheel Odometry': ('orange', BASE / 'wheel_odometry.csv'),
        'EKF Odometry':   ('cyan',   BASE / 'ekf_odometry.csv'),
        'ICP Odometry':   ('red',    BASE / 'icp_odometry.csv'),
        'SLAM':           ('lime',   BASE / 'slam_trajectory.csv'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        'Compare-Drift Run – Sequence 0\n'
        'All 4 Trajectories overlaid on ICP Map (left) and SLAM Map (right)',
        fontsize=13, fontweight='bold'
    )

    maps = [
        (BASE / 'icp_map.pgm',  BASE / 'icp_map.yaml',  'ICP Map + Trajectories'),
        (BASE / 'slam_map.pgm', BASE / 'slam_map.yaml',  'SLAM Map + Trajectories'),
    ]

    for ax, (pgm, yml, title) in zip(axes, maps):
        img, ext = load_pgm(pgm, yml)
        ax.imshow(img, cmap='gray', origin='upper',
                  extent=ext, interpolation='nearest', aspect='equal', alpha=0.75)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.2)

        for label, (color, path) in files.items():
            if not path.exists():
                continue
            x, y = load_csv(path)
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9, label=label)
            ax.plot(x[0], y[0], 'wo', markersize=6, markeredgecolor='black',
                    markeredgewidth=0.8)

        ax.legend(fontsize=9, loc='upper right',
                  framealpha=0.85, facecolor='white')

    plt.tight_layout()
    out = BASE / 'compare_drift_traj_on_map.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_maps()
    plot_trajectories()
    plot_trajectories_on_map()
    print('Done.')

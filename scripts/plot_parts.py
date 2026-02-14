#!/usr/bin/env python3
"""
Generate per-part focused comparison plots:
  Part 1 – Wheel Odometry vs EKF Odometry (3 sequences)
  Part 2 – EKF Odometry vs ICP Odometry   (3 sequences)

Saves:
  results/part1_wheel_vs_ekf.png
  results/part2_ekf_vs_icp.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results'

SEQ_INFO = {
    0: {'label': 'Sequence 0\nEmpty Hallway',
        'base': RESULTS / 'sequence_0',
        'config_wheel_ekf': None,        # in seq root
        'config_icp': None},
    1: {'label': 'Sequence 1\nSharp Turns',
        'base': RESULTS / 'sequence_1',
        'config_wheel_ekf': None,
        'config_icp': None},
    2: {'label': 'Sequence 2\nSmooth Motion',
        'base': RESULTS / 'sequence_2',
        'config_wheel_ekf': None,
        'config_icp': None},
}

# ekf_odometry may also live in Config_A or Config_B sub-directories
EKF_FALLBACKS = {
    0: [RESULTS / 'sequence_0' / 'Config_A' / 'ekf_odometry.csv'],
    1: [RESULTS / 'sequence_1' / 'Config_A' / 'ekf_odometry.csv',
        RESULTS / 'sequence_1' / 'Config_B' / 'ekf_odometry.csv'],
    2: [RESULTS / 'sequence_2' / 'Config_A' / 'ekf_odometry.csv',
        RESULTS / 'sequence_2' / 'Config_B' / 'ekf_odometry.csv'],
}


def load(path):
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    return df['x'].values, df['y'].values


def find_ekf(seq_id):
    base = SEQ_INFO[seq_id]['base']
    primary = base / 'ekf_odometry.csv'
    if primary.exists():
        return primary
    for fb in EKF_FALLBACKS[seq_id]:
        if fb.exists():
            return fb
    return None


def lce(x, y):
    return np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)


def drift_pct(x, y):
    d = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    return lce(x, y) / d * 100 if d > 0 else float('nan')


def metric_str(x, y):
    return f'LCE={lce(x,y):.2f} m  Drift={drift_pct(x,y):.1f}%'


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Wheel vs EKF – 3×2 grid (overlay left, individual right per sequence)
# ──────────────────────────────────────────────────────────────────────────────

def plot_part1():
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        'Part 1: EKF Odometry Fusion\n'
        'Wheel Odometry (dead reckoning) vs EKF Odometry (Wheel + IMU)',
        fontsize=15, fontweight='bold'
    )

    col_titles = ['Overlay (Both Methods)', 'Wheel Odometry Only', 'EKF Odometry Only']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=12, fontweight='bold', pad=10)

    for row, seq_id in enumerate(range(3)):
        base = SEQ_INFO[seq_id]['base']
        seq_label = SEQ_INFO[seq_id]['label']

        wheel = load(base / 'wheel_odometry.csv')
        ekf_path = find_ekf(seq_id)
        ekf = load(ekf_path)

        ax_ov  = axes[row][0]   # overlay
        ax_wh  = axes[row][1]   # wheel only
        ax_ekf = axes[row][2]   # ekf only

        # y-axis label with sequence name
        ax_ov.set_ylabel(seq_label, fontsize=11, fontweight='bold', labelpad=8)

        # ── overlay ──
        if wheel:
            wx, wy = wheel
            ax_ov.plot(wx, wy, color='#888888', linewidth=1.2, alpha=0.85,
                       label=f'Wheel  ({metric_str(wx, wy)})')
        if ekf:
            ex, ey = ekf
            ax_ov.plot(ex, ey, color='#1565C0', linewidth=1.6, alpha=0.9,
                       label=f'EKF  ({metric_str(ex, ey)})')
        if wheel:
            ax_ov.plot(wx[0], wy[0], 'ko', markersize=8, zorder=5)
        ax_ov.set_aspect('equal'); ax_ov.grid(True, alpha=0.3)
        ax_ov.set_xlabel('X (m)', fontsize=9); ax_ov.set_ylabel('Y (m)', fontsize=9)
        ax_ov.legend(fontsize=8, loc='best', framealpha=0.9)

        # ── wheel only ──
        if wheel:
            wx, wy = wheel
            ax_wh.plot(wx, wy, color='#888888', linewidth=1.3)
            ax_wh.plot(wx[0], wy[0], 'go', markersize=8, label='Start')
            ax_wh.plot(wx[-1], wy[-1], 'rs', markersize=8, label='End')
            ax_wh.set_title(f'Wheel Odometry\n{metric_str(wx,wy)}',
                            fontsize=9, pad=4)
        else:
            ax_wh.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax_wh.transAxes)
        ax_wh.set_aspect('equal'); ax_wh.grid(True, alpha=0.3)
        ax_wh.set_xlabel('X (m)', fontsize=9); ax_wh.set_ylabel('Y (m)', fontsize=9)
        ax_wh.legend(fontsize=8)

        # ── ekf only ──
        if ekf:
            ex, ey = ekf
            ax_ekf.plot(ex, ey, color='#1565C0', linewidth=1.3)
            ax_ekf.plot(ex[0], ey[0], 'go', markersize=8, label='Start')
            ax_ekf.plot(ex[-1], ey[-1], 'rs', markersize=8, label='End')
            ax_ekf.set_title(f'EKF Odometry\n{metric_str(ex,ey)}',
                             fontsize=9, pad=4)
        else:
            ax_ekf.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_ekf.transAxes)
        ax_ekf.set_aspect('equal'); ax_ekf.grid(True, alpha=0.3)
        ax_ekf.set_xlabel('X (m)', fontsize=9); ax_ekf.set_ylabel('Y (m)', fontsize=9)
        ax_ekf.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS / 'part1_wheel_vs_ekf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: EKF vs ICP – 3×3 grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_part2():
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        'Part 2: ICP Odometry Refinement\n'
        'EKF Odometry (initial guess) vs ICP Odometry (scan-matching result)',
        fontsize=15, fontweight='bold'
    )

    col_titles = ['Overlay (Both Methods)', 'EKF Odometry Only', 'ICP Odometry Only']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=12, fontweight='bold', pad=10)

    for row, seq_id in enumerate(range(3)):
        base = SEQ_INFO[seq_id]['base']
        seq_label = SEQ_INFO[seq_id]['label']

        ekf_path = find_ekf(seq_id)
        ekf = load(ekf_path)
        icp = load(base / 'icp_odometry.csv')

        ax_ov  = axes[row][0]
        ax_ekf = axes[row][1]
        ax_icp = axes[row][2]

        ax_ov.set_ylabel(seq_label, fontsize=11, fontweight='bold', labelpad=8)

        # ── overlay ──
        if ekf:
            ex, ey = ekf
            ax_ov.plot(ex, ey, color='#1565C0', linewidth=1.6, alpha=0.8,
                       label=f'EKF  ({metric_str(ex, ey)})')
        if icp:
            ix, iy = icp
            ax_ov.plot(ix, iy, color='#C62828', linewidth=1.6, alpha=0.9,
                       label=f'ICP  ({metric_str(ix, iy)})')
        if ekf:
            ax_ov.plot(ex[0], ey[0], 'ko', markersize=8, zorder=5)
        ax_ov.set_aspect('equal'); ax_ov.grid(True, alpha=0.3)
        ax_ov.set_xlabel('X (m)', fontsize=9); ax_ov.set_ylabel('Y (m)', fontsize=9)
        ax_ov.legend(fontsize=8, loc='best', framealpha=0.9)

        # ── ekf only ──
        if ekf:
            ex, ey = ekf
            ax_ekf.plot(ex, ey, color='#1565C0', linewidth=1.3)
            ax_ekf.plot(ex[0], ey[0], 'go', markersize=8, label='Start')
            ax_ekf.plot(ex[-1], ey[-1], 'rs', markersize=8, label='End')
            ax_ekf.set_title(f'EKF Odometry\n{metric_str(ex,ey)}', fontsize=9, pad=4)
        else:
            ax_ekf.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_ekf.transAxes)
        ax_ekf.set_aspect('equal'); ax_ekf.grid(True, alpha=0.3)
        ax_ekf.set_xlabel('X (m)', fontsize=9); ax_ekf.set_ylabel('Y (m)', fontsize=9)
        ax_ekf.legend(fontsize=8)

        # ── icp only ──
        if icp:
            ix, iy = icp
            ax_icp.plot(ix, iy, color='#C62828', linewidth=1.3)
            ax_icp.plot(ix[0], iy[0], 'go', markersize=8, label='Start')
            ax_icp.plot(ix[-1], iy[-1], 'rs', markersize=8, label='End')
            ax_icp.set_title(f'ICP Odometry\n{metric_str(ix,iy)}', fontsize=9, pad=4)
        else:
            ax_icp.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_icp.transAxes)
        ax_icp.set_aspect('equal'); ax_icp.grid(True, alpha=0.3)
        ax_icp.set_xlabel('X (m)', fontsize=9); ax_icp.set_ylabel('Y (m)', fontsize=9)
        ax_icp.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS / 'part2_ekf_vs_icp.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_part1()
    plot_part2()
    print('Done.')

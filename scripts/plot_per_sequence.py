#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results'

SEQUENCES = {
    0: 'Sequence 0 – Empty Hallway',
    1: 'Sequence 1 – Sharp Turns',
    2: 'Sequence 2 – Smooth Motion',
}


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if 'x' not in df.columns or 'y' not in df.columns:
            return None
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
        return df if len(df) > 5 else None
    except Exception:
        return None


def lce(df):
    if df is None or len(df) < 2:
        return float('nan')
    s = df[['x', 'y']].iloc[0].values
    e = df[['x', 'y']].iloc[-1].values
    return float(np.linalg.norm(e - s))


def drift_pct(df):
    if df is None or len(df) < 2:
        return float('nan')
    dx, dy = np.diff(df['x'].values), np.diff(df['y'].values)
    d = float(np.sum(np.sqrt(dx**2 + dy**2)))
    le = lce(df)
    return le / d * 100 if d > 0 else float('nan')


def subtitle(df):
    le = lce(df)
    dr = drift_pct(df)
    if np.isnan(le):
        return 'No data'
    return f'LCE={le:.2f} m  Drift={dr:.1f}%'


def draw(ax, df, color, lw=1.4):
    if df is None or len(df) < 2:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#aaaaaa')
        return False
    x, y = df['x'].values, df['y'].values
    ax.plot(x, y, color=color, lw=lw, alpha=0.85)
    ax.plot(x[0],  y[0],  'o', color='#22aa22', ms=7, zorder=5, label='Start')
    ax.plot(x[-1], y[-1], '^', color='#cc2222', ms=7, zorder=5, label='End')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.legend(fontsize=8)
    return True


def plot_sequence(seq_id):
    label = SEQUENCES[seq_id]
    base = RESULTS / f'sequence_{seq_id}'

    wheel_df = load(base / 'wheel_odometry.csv')
    icp_df   = load(base / 'icp_odometry.csv')

    ekf_a = load(base / 'Config_A' / 'ekf_odometry.csv')
    ekf_b = load(base / 'Config_B' / 'ekf_odometry.csv')
    ekf_df = ekf_a if ekf_a is not None else ekf_b

    slam_a = load(base / 'Config_A' / 'slam_trajectory.csv')
    slam_b = load(base / 'Config_B' / 'slam_trajectory.csv')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'All Methods – {label}', fontsize=15, fontweight='bold', y=1.01)

    panels = [
        (axes[0, 0], wheel_df, '#888888', 'Wheel Odometry'),
        (axes[0, 1], ekf_df,   '#1565C0', 'EKF Odometry'),
        (axes[0, 2], icp_df,   '#C62828', 'ICP Odometry'),
        (axes[1, 0], slam_a,   '#E65100', 'SLAM Config A (Relaxed)'),
        (axes[1, 1], slam_b,   '#6A1B9A', 'SLAM Config B (Strict)'),
    ]

    for ax, df, color, title in panels:
        draw(ax, df, color)
        sub = subtitle(df)
        ax.set_title(f'{title}\n{sub}', fontsize=10, fontweight='bold')

    ax_ov = axes[1, 2]
    overlay_data = [
        (wheel_df, '#888888', 'Wheel', 1.0),
        (ekf_df,   '#1565C0', 'EKF',   1.4),
        (icp_df,   '#C62828', 'ICP',   1.4),
        (slam_a,   '#E65100', 'SLAM-A', 1.6),
        (slam_b,   '#6A1B9A', 'SLAM-B', 2.0),
    ]
    any_drawn = False
    for df, color, name, lw in overlay_data:
        if df is not None and len(df) > 1:
            ax_ov.plot(df['x'].values, df['y'].values, color=color, lw=lw,
                       alpha=0.85, label=name)
            any_drawn = True
    if any_drawn:
        ref = next((d for d, *_ in overlay_data if d is not None and len(d) > 1), None)
        if ref is not None:
            ax_ov.plot(ref['x'].iloc[0], ref['y'].iloc[0], 'o',
                       color='#22aa22', ms=9, zorder=5)
        ax_ov.legend(fontsize=9)
        ax_ov.set_aspect('equal')
        ax_ov.grid(True, alpha=0.25)
        ax_ov.set_xlabel('X (m)', fontsize=9)
        ax_ov.set_ylabel('Y (m)', fontsize=9)
    else:
        ax_ov.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax_ov.transAxes, fontsize=12, color='#aaaaaa')
    ax_ov.set_title('All Methods Overlay', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = RESULTS / f'sequence_{seq_id}_all_methods.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    for seq in range(3):
        plot_sequence(seq)
    print('Done.')

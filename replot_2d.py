#!/usr/bin/env python3
"""
Regenerate corrected plots for straight_2d and sine_2d experiments.

Root cause: the data collector records one extra sample after the trajectory
ends, at which point the trajectory generator has reset ref_x back to 0.
This produces a massive spurious error spike in the last row(s).

Fix: detect any row near the end where ref_x drops by > 1 m from the previous
row (trajectory reset), and trim from that point onward before plotting.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA_ROOT = os.path.expanduser('~/Mobile_Robot/data')

EXPERIMENTS = [
    ('straight_2d_20260301_154909', 'Straight 2D (no wind)'),
    ('straight_2d_20260301_160012', 'Straight 2D (wind 4 m/s -y)'),
    ('sine_2d_20260301_155013',     'Sine 2D (no wind)'),
    ('sine_2d_20260301_160049',     'Sine 2D (wind 4 m/s -y)'),
]


def trim_reset_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing rows where trajectory reference resets back to 0."""
    ref_x = df['ref_x'].to_numpy()
    for i in range(len(ref_x) - 1, 0, -1):
        if ref_x[i - 1] - ref_x[i] > 1.0:
            trimmed = df.iloc[:i].copy()
            dropped = len(df) - i
            print(f'  Trimmed {dropped} reset-artifact row(s) '
                  f'(ref_x: {ref_x[i-1]:.3f} -> {ref_x[i]:.3f} '
                  f'at t={df["t_rel"].iloc[i]:.3f} s)')
            return trimmed
    print('  No reset artifact found — using full dataset.')
    return df.copy()


def recompute_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['err_x']      = df['x']  - df['ref_x']
    df['err_y']      = df['y']  - df['ref_y']
    df['err_z']      = df['z']  - df['ref_z']
    df['err_3d']     = np.sqrt(df['err_x']**2 + df['err_y']**2 + df['err_z']**2)
    df['err_vx']     = df['vx'] - df['ref_vx']
    df['err_vy']     = df['vy'] - df['ref_vy']
    df['err_vz']     = df['vz'] - df['ref_vz']
    df['err_vel_3d'] = np.sqrt(df['err_vx']**2 + df['err_vy']**2 + df['err_vz']**2)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    m = {}
    m['rmse_x']    = float(np.sqrt((df['err_x']**2).mean()))
    m['rmse_y']    = float(np.sqrt((df['err_y']**2).mean()))
    m['rmse_z']    = float(np.sqrt((df['err_z']**2).mean()))
    m['rmse_3d']   = float(np.sqrt((df['err_3d']**2).mean()))
    m['max_err_x'] = float(df['err_x'].abs().max())
    m['max_err_y'] = float(df['err_y'].abs().max())
    m['max_err_z'] = float(df['err_z'].abs().max())
    m['max_err_3d'] = float(df['err_3d'].abs().max())
    ss = df.iloc[int(0.8 * len(df)):]
    m['ss_err_x']  = float(ss['err_x'].abs().mean())
    m['ss_err_y']  = float(ss['err_y'].abs().mean())
    m['ss_err_z']  = float(ss['err_z'].abs().mean())
    m['ss_err_3d'] = float(ss['err_3d'].abs().mean())
    return m


# ─── Plotting ─────────────────────────────────────────────────────────────────

def v(df, col):
    """Return numpy array for a column."""
    return df[col].to_numpy()


def plot_3d(df, label, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n = len(df)
    mark_step = max(1, n // 30)
    ax.plot(v(df,'ref_x'), v(df,'ref_y'), v(df,'ref_z'),
            'r--', lw=1.8, alpha=0.9, label='Reference',
            marker='o', markevery=mark_step, markersize=3)
    ax.plot(v(df,'x'), v(df,'y'), v(df,'z'),
            'b-', lw=1.2, alpha=0.85, label='Actual')
    r0 = df.iloc[0]
    r1 = df.iloc[-1]
    ax.scatter(float(r0['x']), float(r0['y']), float(r0['z']),
               c='green', s=80, marker='o', label='Start')
    ax.scatter(float(r1['x']), float(r1['y']), float(r1['z']),
               c='red',   s=80, marker='x', label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(f'{label} — 3-D Trajectory Tracking')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, '3d_trajectory.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved 3d_trajectory.png')


def plot_pos_tracking(df, label, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = v(df, 't_rel')
    for ax, name, col, rcol in zip(axes,
                                    ['X', 'Y', 'Z'],
                                    ['x', 'y', 'z'],
                                    ['ref_x', 'ref_y', 'ref_z']):
        ax.plot(t, v(df, col),  'b-',  lw=1,   label='Actual')
        ax.plot(t, v(df, rcol), 'r--', lw=1,   label='Reference')
        ax.set_ylabel(f'{name} [m]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{label} — Position Tracking', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'position_tracking.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved position_tracking.png')


def plot_pos_error(df, met, label, out):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = v(df, 't_rel')
    items = [
        ('X error',   'err_x',  met['rmse_x']),
        ('Y error',   'err_y',  met['rmse_y']),
        ('Z error',   'err_z',  met['rmse_z']),
        ('3-D error', 'err_3d', met['rmse_3d']),
    ]
    for ax, (name, col, rmse) in zip(axes, items):
        ax.plot(t, v(df, col), 'b-', lw=0.8)
        ax.axhline( rmse, color='orange', ls='--', lw=1, label=f'RMSE = {rmse:.4f} m')
        if col != 'err_3d':
            ax.axhline(-rmse, color='orange', ls='--', lw=1)
        t_ss = t[int(0.8 * len(t))]
        ax.axvspan(t_ss, t[-1], alpha=0.08, color='green', label='Steady-state region')
        ax.set_ylabel(f'{name} [m]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{label} — Position Error', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'position_error.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved position_error.png')


def plot_vel_tracking(df, label, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = v(df, 't_rel')
    for ax, name, col, rcol in zip(axes,
                                    ['Vx', 'Vy', 'Vz'],
                                    ['vx', 'vy', 'vz'],
                                    ['ref_vx', 'ref_vy', 'ref_vz']):
        ax.plot(t, v(df, col),  'b-',  lw=0.9, label='Actual')
        ax.plot(t, v(df, rcol), 'r--', lw=0.9, label='Reference')
        ax.set_ylabel(f'{name} [m/s]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{label} — Velocity Tracking (World Frame)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'velocity_tracking.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved velocity_tracking.png')


# ─── Main ─────────────────────────────────────────────────────────────────────

for folder, label in EXPERIMENTS:
    out = os.path.join(DATA_ROOT, folder)
    state_csv = os.path.join(out, 'state_data.csv')

    print(f'\n=== {label} ===')
    print(f'    Folder: {folder}')

    df_raw = pd.read_csv(state_csv)
    print(f'  Raw rows: {len(df_raw)}')

    df = trim_reset_artifact(df_raw)
    df = recompute_errors(df)
    df.reset_index(drop=True, inplace=True)

    met = compute_metrics(df)
    print(f'  Corrected RMSE  x={met["rmse_x"]:.4f}  y={met["rmse_y"]:.4f}  '
          f'z={met["rmse_z"]:.4f}  3d={met["rmse_3d"]:.4f} m')
    print(f'  Corrected max   x={met["max_err_x"]:.4f}  y={met["max_err_y"]:.4f}  '
          f'z={met["max_err_z"]:.4f}  3d={met["max_err_3d"]:.4f} m')

    plot_3d(df, label, out)
    plot_pos_tracking(df, label, out)
    plot_pos_error(df, met, label, out)
    plot_vel_tracking(df, label, out)

    # Update metrics.csv with corrected RMSE / max / SS values
    mcsv = os.path.join(out, 'metrics.csv')
    mdf = pd.read_csv(mcsv, index_col='metric')
    for key, val in met.items():
        if key in mdf.index:
            mdf.at[key, 'value'] = val
        else:
            mdf.loc[key] = val
    mdf.to_csv(mcsv)
    print(f'  Updated metrics.csv')

print('\nDone.')

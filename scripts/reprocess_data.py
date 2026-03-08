#!/usr/bin/env python3
"""
Post-process collected flight data: trim POST_HOVER samples,
recalculate metrics, and regenerate all plots.

Usage:
  python3 reprocess_data.py                    # process all 20260308 dirs
  python3 reprocess_data.py <dir1> <dir2> ...  # process specific dirs
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------------------------------------------------------------------------
# Constants (must match data_collector.py)
# ---------------------------------------------------------------------------
MASS            = 1.5
GRAVITY         = 9.81
HOVER_THRUST    = MASS * GRAVITY
IXX, IYY, IZZ  = 0.0347563, 0.07, 0.0977
KF              = 8.54858e-06
KM              = 0.06
MAX_ROTOR_SPEED = 1500.0
HOVER_OMEGA     = np.sqrt(HOVER_THRUST / (4.0 * KF))
LIN_LIMIT_DEG   = 15.0
LIN_LIMIT_RAD   = np.radians(LIN_LIMIT_DEG)


def trim_post_hover(sdf):
    """Remove trailing POST_HOVER samples detected by reference discontinuity."""
    ref_pos = sdf[['ref_x', 'ref_y', 'ref_z']].values
    ref_jump = np.linalg.norm(np.diff(ref_pos, axis=0), axis=1)
    # Check the last 5% of samples
    check_from = max(0, len(ref_jump) - max(10, len(ref_jump) // 20))
    tail_jumps = ref_jump[check_from:]
    big = np.where(tail_jumps > 0.3)[0]
    if len(big) > 0:
        cut_idx = check_from + big[0]  # last valid row index
        n_trimmed = len(sdf) - (cut_idx + 1)
        sdf = sdf.iloc[:cut_idx + 1].copy()
        sdf.reset_index(drop=True, inplace=True)
        print(f'  Trimmed {n_trimmed} POST_HOVER samples')
    else:
        print(f'  No POST_HOVER discontinuity detected')
    return sdf


def trim_ctrl(cdf, t_max):
    """Trim control data to match the state data time range."""
    cdf = cdf[cdf['t'] <= t_max + 0.025].copy()
    cdf.reset_index(drop=True, inplace=True)
    return cdf


def recompute_errors(sdf):
    """Recompute tracking errors after trimming."""
    for ax in ('x', 'y', 'z'):
        sdf[f'err_{ax}'] = sdf[ax] - sdf[f'ref_{ax}']
    for ax in ('phi', 'theta', 'psi'):
        err = sdf[ax] - sdf[f'ref_{ax}']
        sdf[f'err_{ax}'] = (err + np.pi) % (2 * np.pi) - np.pi
    for ax in ('vx', 'vy', 'vz'):
        sdf[f'err_{ax}'] = sdf[ax] - sdf[f'ref_{ax}']
    sdf['err_3d'] = np.sqrt(
        sdf['err_x']**2 + sdf['err_y']**2 + sdf['err_z']**2)
    sdf['err_vel_3d'] = np.sqrt(
        sdf['err_vx']**2 + sdf['err_vy']**2 + sdf['err_vz']**2)
    # Recompute t_rel
    t_min = sdf['t'].iloc[0]
    sdf['t_rel'] = sdf['t'] - t_min
    return sdf


def compute_metrics(sdf, cdf, traj, wind):
    """Compute all metrics from trimmed data."""
    m = {}
    dur = sdf['t_rel'].iloc[-1] - sdf['t_rel'].iloc[0]
    m['duration_s'] = dur
    m['n_state_samples'] = len(sdf)
    m['n_ctrl_samples'] = len(cdf)
    m['trajectory_type'] = traj
    m['wind_x'] = wind[0]
    m['wind_y'] = wind[1]
    m['wind_z'] = wind[2]
    m['is_wind'] = any(abs(w) > 0.01 for w in wind)

    # Position RMSE
    for ax in ('x', 'y', 'z'):
        m[f'rmse_{ax}'] = np.sqrt(np.mean(sdf[f'err_{ax}']**2))
        m[f'max_err_{ax}'] = np.max(np.abs(sdf[f'err_{ax}']))
    m['rmse_3d'] = np.sqrt(np.mean(sdf['err_3d']**2))
    m['max_err_3d'] = np.max(sdf['err_3d'])

    # Velocity RMSE
    for ax in ('vx', 'vy', 'vz'):
        m[f'rmse_{ax}'] = np.sqrt(np.mean(sdf[f'err_{ax}']**2))
    m['rmse_vel_3d'] = np.sqrt(np.mean(sdf['err_vel_3d']**2))

    # Settling time
    m['settling_time_s'] = _settling_time(sdf)

    # Steady-state error (last 20%)
    n20 = max(1, int(0.8 * len(sdf)))
    tail = sdf.iloc[n20:]
    for ax in ('x', 'y', 'z'):
        m[f'ss_err_{ax}'] = np.mean(np.abs(tail[f'err_{ax}']))
    m['ss_err_3d'] = np.mean(tail['err_3d'])

    # Thrust
    if len(cdf) > 0:
        m['mean_thrust'] = np.mean(cdf['thrust'])
        m['max_thrust'] = np.max(cdf['thrust'])
        m['min_thrust'] = np.min(cdf['thrust'])
        m['thrust_dev'] = np.mean(np.abs(cdf['thrust'] - HOVER_THRUST))

        for ax in ('tau_phi', 'tau_theta', 'tau_psi'):
            m[f'rms_{ax}'] = np.sqrt(np.mean(cdf[ax]**2))
            m[f'max_{ax}'] = np.max(np.abs(cdf[ax]))

        ws = cdf[['w0', 'w1', 'w2', 'w3']].values
        util = ws / MAX_ROTOR_SPEED * 100.0
        m['mean_motor_util'] = np.mean(util)
        m['max_motor_util'] = np.max(util)
        for i in range(4):
            m[f'mean_util_r{i}'] = np.mean(util[:, i])

        sat_mask = np.any(ws >= 0.99 * MAX_ROTOR_SPEED, axis=1)
        m['sat_events'] = int(np.sum(sat_mask))
        m['sat_pct'] = np.sum(sat_mask) / len(cdf) * 100.0

        t_ctrl = cdf['t_rel'].values
        u_sq = (cdf['thrust']**2 + cdf['tau_phi']**2 +
                cdf['tau_theta']**2 + cdf['tau_psi']**2).values
        m['ctrl_energy'] = float(np.trapz(u_sq, t_ctrl))

    # Linearisation validity
    m['max_phi_deg'] = np.degrees(np.max(np.abs(sdf['phi'])))
    m['max_theta_deg'] = np.degrees(np.max(np.abs(sdf['theta'])))
    within = ((np.abs(sdf['phi']) < LIN_LIMIT_RAD) &
              (np.abs(sdf['theta']) < LIN_LIMIT_RAD))
    m['pct_within_linear'] = np.mean(within) * 100.0
    m['linearization_valid'] = bool(
        m['max_phi_deg'] < LIN_LIMIT_DEG and
        m['max_theta_deg'] < LIN_LIMIT_DEG)

    # Signed steady-state offset
    m['ss_offset_x'] = float(np.mean(tail['err_x']))
    m['ss_offset_y'] = float(np.mean(tail['err_y']))
    m['ss_offset_z'] = float(np.mean(tail['err_z']))
    m['wind_detected'] = (abs(m['ss_offset_x']) > 0.05 or
                          abs(m['ss_offset_y']) > 0.05)

    # MPC debug
    has_debug = 'solve_ok' in cdf.columns
    m['has_mpc_debug'] = has_debug
    if has_debug:
        n_total = len(cdf)
        n_ok = int(cdf['solve_ok'].sum())
        m['mpc_solve_ok'] = n_ok
        m['mpc_solve_fail'] = n_total - n_ok
        m['mpc_solve_pct'] = n_ok / n_total * 100.0

        m['cost_state_mean'] = float(cdf['cost_state'].mean())
        m['cost_state_max'] = float(cdf['cost_state'].max())
        m['cost_input_mean'] = float(cdf['cost_input'].mean())
        m['cost_input_max'] = float(cdf['cost_input'].max())
        m['cost_total_mean'] = float(
            (cdf['cost_state'] + cdf['cost_input']).mean())

        ts = cdf['torque_scale']
        m['torque_scale_mean'] = float(ts.mean())
        m['torque_scale_min'] = float(ts.min())
        n_scaled = int((ts < 0.999).sum())
        m['torque_scaled_pct'] = n_scaled / n_total * 100.0

        m['ud_thrust_rms'] = float(np.sqrt(np.mean(cdf['ud_T']**2)))
        m['ui_thrust_rms'] = float(np.sqrt(np.mean(cdf['ui_T']**2)))
        for ax in ('phi', 'theta', 'psi'):
            m[f'ud_tau_{ax}_rms'] = float(
                np.sqrt(np.mean(cdf[f'ud_{ax}']**2)))
            m[f'ui_tau_{ax}_rms'] = float(
                np.sqrt(np.mean(cdf[f'ui_{ax}']**2)))

        int_mag = np.sqrt(cdf['int_err_x']**2 + cdf['int_err_y']**2 +
                          cdf['int_err_z']**2)
        m['int_err_mag_final'] = float(int_mag.iloc[-1])
        m['int_err_mag_max'] = float(int_mag.max())

    return m


def _settling_time(sdf, target_z=1.0, thr=0.02, window=2.0):
    z = sdf['z'].values
    t = sdf['t_rel'].values
    within = np.abs(z - target_z) < thr
    for i in range(len(t)):
        if not within[i]:
            continue
        end = t[i] + window
        j = i
        ok = True
        while j < len(t) and t[j] <= end:
            if not within[j]:
                ok = False
                break
            j += 1
        if ok and j > i:
            return float(t[i])
    return float('nan')


# ---------------------------------------------------------------------------
# Plotting functions (replicate data_collector.py exactly)
# ---------------------------------------------------------------------------

def plot_3d(sdf, traj, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n = len(sdf)
    mark_step = max(1, n // 30)
    ax.plot(sdf['ref_x'].to_numpy(), sdf['ref_y'].to_numpy(), sdf['ref_z'].to_numpy(),
            'r--', lw=1.8, alpha=0.9, label='Reference',
            marker='o', markevery=mark_step, markersize=3)
    ax.plot(sdf['x'].to_numpy(), sdf['y'].to_numpy(), sdf['z'].to_numpy(),
            'b-', lw=1.2, alpha=0.85, label='Actual')
    ax.scatter(*[sdf.iloc[0][c] for c in ('x', 'y', 'z')],
               c='green', s=80, marker='o', label='Start')
    ax.scatter(*[sdf.iloc[-1][c] for c in ('x', 'y', 'z')],
               c='red', s=80, marker='x', label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'{traj} — 3-D Trajectory Tracking')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, '3d_trajectory.png'), dpi=150)
    plt.close(fig)


def plot_pos_tracking(sdf, traj, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = sdf['t_rel'].to_numpy()
    for ax, lab, c, rc in zip(axes, ['X', 'Y', 'Z'],
                               ['x', 'y', 'z'],
                               ['ref_x', 'ref_y', 'ref_z']):
        ax.plot(t, sdf[c].to_numpy(), 'b-', lw=1, label='Actual')
        ax.plot(t, sdf[rc].to_numpy(), 'r--', lw=1, label='Reference')
        ax.set_ylabel(f'{lab} [m]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Position Tracking', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'position_tracking.png'), dpi=150)
    plt.close(fig)


def plot_vel_tracking(sdf, traj, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = sdf['t_rel'].to_numpy()
    for ax, lab, c, rc in zip(axes, ['Vx', 'Vy', 'Vz'],
                               ['vx', 'vy', 'vz'],
                               ['ref_vx', 'ref_vy', 'ref_vz']):
        ax.plot(t, sdf[c].to_numpy(), 'b-', lw=0.9, label='Actual')
        ax.plot(t, sdf[rc].to_numpy(), 'r--', lw=0.9, label='Reference')
        ax.set_ylabel(f'{lab} [m/s]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Velocity Tracking (World Frame)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'velocity_tracking.png'), dpi=150)
    plt.close(fig)


def plot_pos_error(sdf, met, traj, out):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = sdf['t_rel'].to_numpy()
    labels = ['X error', 'Y error', 'Z error', '3-D error']
    cols = ['err_x', 'err_y', 'err_z', 'err_3d']
    rmses = [met['rmse_x'], met['rmse_y'], met['rmse_z'], met['rmse_3d']]
    for ax, lab, col, rmse in zip(axes, labels, cols, rmses):
        ax.plot(t, sdf[col].to_numpy(), 'b-', lw=0.8)
        ax.axhline(rmse, color='orange', ls='--', lw=1,
                    label=f'RMSE = {rmse:.4f} m')
        if col != 'err_3d':
            ax.axhline(-rmse, color='orange', ls='--', lw=1)
        t_ss = t[int(0.8 * len(t))]
        ax.axvspan(t_ss, t[-1], alpha=0.08, color='green',
                    label='Steady-state region')
        ax.set_ylabel(f'{lab} [m]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Position Error', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'position_error.png'), dpi=150)
    plt.close(fig)


def plot_wrench(cdf, traj, out):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = cdf['t_rel'].to_numpy()
    labels = ['Thrust T [N]', 'τ_φ (roll) [N·m]',
              'τ_θ (pitch) [N·m]', 'τ_ψ (yaw) [N·m]']
    cols = ['thrust', 'tau_phi', 'tau_theta', 'tau_psi']
    for ax, lab, col in zip(axes, labels, cols):
        ax.plot(t, cdf[col].to_numpy(), 'b-', lw=0.8)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[0].axhline(HOVER_THRUST, color='gray', ls=':', lw=1,
                     label=f'Hover mg = {HOVER_THRUST:.2f} N')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Control Wrench (MPC Output)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'control_wrench.png'), dpi=150)
    plt.close(fig)


def plot_motors(cdf, traj, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    t = cdf['t_rel'].to_numpy()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, c in enumerate(colors):
        ax.plot(t, cdf[f'w{i}'].to_numpy(), color=c, lw=0.8, label=f'Rotor {i}')
    ax.axhline(MAX_ROTOR_SPEED, color='red', ls='--', lw=1.2,
                label=f'Max ({MAX_ROTOR_SPEED:.0f} rad/s)')
    ax.axhline(HOVER_OMEGA, color='gray', ls=':', lw=1,
                label=f'Hover ({HOVER_OMEGA:.0f} rad/s)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Rotor Speed [rad/s]')
    ax.set_title(f'{traj} — Motor Speeds')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'motor_speeds.png'), dpi=150)
    plt.close(fig)


def plot_euler(sdf, traj, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = sdf['t_rel'].to_numpy()
    labels = ['φ (roll)', 'θ (pitch)', 'ψ (yaw)']
    cols = ['phi', 'theta', 'psi']
    refs = ['ref_phi', 'ref_theta', 'ref_psi']
    for ax, lab, col, rc in zip(axes, labels, cols, refs):
        ax.plot(t, np.degrees(sdf[col].to_numpy()), 'b-', lw=0.8, label='Actual')
        ax.plot(t, np.degrees(sdf[rc].to_numpy()), 'r--', lw=0.8, label='Reference')
        if col in ('phi', 'theta'):
            ax.axhline(LIN_LIMIT_DEG, color='orange', ls=':', lw=1)
            ax.axhline(-LIN_LIMIT_DEG, color='orange', ls=':', lw=1,
                        label=f'±{LIN_LIMIT_DEG:.0f}° linear bound')
        ax.set_ylabel(f'{lab} [deg]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Euler Angles', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'euler_angles.png'), dpi=150)
    plt.close(fig)


def plot_motor_hist(cdf, met, traj, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ws = cdf[['w0', 'w1', 'w2', 'w3']].values.flatten()
    util = ws / MAX_ROTOR_SPEED * 100.0
    ax.hist(util, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(met['mean_motor_util'], color='orange', ls='--', lw=1.5,
                label=f'Mean = {met["mean_motor_util"]:.1f}%')
    ax.axvline(99.0, color='red', ls='--', lw=1.2,
                label=f'Saturation 99%  (events: {met["sat_events"]})')
    ax.set_xlabel('Motor Utilisation [%]')
    ax.set_ylabel('Count')
    ax.set_title(f'{traj} — Motor Utilisation Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'motor_utilization.png'), dpi=150)
    plt.close(fig)


def plot_mpc_cost(cdf, traj, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = cdf['t_rel'].to_numpy()
    cost_state = cdf['cost_state'].to_numpy()
    cost_input = cdf['cost_input'].to_numpy()
    cost_total = cost_state + cost_input

    axes[0].plot(t, cost_state, 'b-', lw=0.8, label="State cost  e'Qe")
    axes[0].set_ylabel('State Cost')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, cost_input, 'r-', lw=0.8, label="Input cost  u'Ru")
    axes[1].set_ylabel('Input Cost')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(t, 0, cost_state, alpha=0.4,
                          color='tab:blue', label='State cost')
    axes[2].fill_between(t, cost_state, cost_total, alpha=0.4,
                          color='tab:red', label='Input cost')
    axes[2].set_ylabel('Total Cost')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{traj} — MPC Cost Decomposition', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'mpc_cost.png'), dpi=150)
    plt.close(fig)


def plot_wrench_decomp(cdf, traj, out):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = cdf['t_rel'].to_numpy()
    labels = ['Thrust [N]', 'τ_φ [N·m]', 'τ_θ [N·m]', 'τ_ψ [N·m]']
    delta_cols = ['ud_T', 'ud_phi', 'ud_theta', 'ud_psi']
    integ_cols = ['ui_T', 'ui_phi', 'ui_theta', 'ui_psi']
    total_cols = ['ut_T', 'ut_phi', 'ut_theta', 'ut_psi']
    hover_vals = [HOVER_THRUST, 0.0, 0.0, 0.0]

    for ax, lab, dc, ic, tc, hv in zip(
            axes, labels, delta_cols, integ_cols, total_cols, hover_vals):
        ax.plot(t, cdf[tc].to_numpy(), 'k-', lw=1.0, label='Total', alpha=0.9)
        ax.plot(t, cdf[dc].to_numpy(), 'b-', lw=0.7, label='MPC Δ', alpha=0.7)
        ax.plot(t, cdf[ic].to_numpy(), 'g-', lw=0.7, label='Integral', alpha=0.7)
        if hv != 0:
            ax.axhline(hv, color='gray', ls=':', lw=1,
                        label=f'Hover = {hv:.2f}')
        ax.set_ylabel(lab)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{traj} — Wrench Decomposition '
                 f'(Hover + MPC Δ + Integral)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, 'wrench_decomposition.png'), dpi=150)
    plt.close(fig)


def plot_torque_scale(cdf, traj, out):
    fig, ax = plt.subplots(figsize=(12, 4))
    t = cdf['t_rel'].to_numpy()
    ts = cdf['torque_scale'].to_numpy()
    ax.plot(t, ts, 'b-', lw=0.8)
    ax.fill_between(t, ts, 1.0, where=(ts < 0.999),
                     color='red', alpha=0.3, label='Torque scaled')
    ax.axhline(1.0, color='gray', ls=':', lw=1, label='No saturation')
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque Scale Factor')
    n_scaled = (ts < 0.999).sum()
    ax.set_title(f'{traj} — Motor Allocation Torque Scaling  '
                 f'({n_scaled} events, min = {ts.min():.3f})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'torque_scale.png'), dpi=150)
    plt.close(fig)


def plot_integral_err(cdf, traj, out):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = cdf['t_rel'].to_numpy()

    for ax_lab, col, color in [('X', 'int_err_x', 'tab:blue'),
                                ('Y', 'int_err_y', 'tab:orange'),
                                ('Z', 'int_err_z', 'tab:green')]:
        axes[0].plot(t, cdf[col].to_numpy(), color=color, lw=0.8,
                      label=f'{ax_lab}')
    axes[0].set_ylabel('Integral Error [m·s]')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Integral Error Accumulator (per axis)')
    axes[0].grid(True, alpha=0.3)

    int_mag = np.sqrt(cdf['int_err_x'].to_numpy()**2 +
                      cdf['int_err_y'].to_numpy()**2 +
                      cdf['int_err_z'].to_numpy()**2)
    axes[1].plot(t, int_mag, 'k-', lw=0.8)
    axes[1].set_ylabel('|Integral Error| [m·s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Integral Error Magnitude')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'{traj} — Integral Wind Rejection Accumulator',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, 'integral_error.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_dir(data_dir):
    """Process a single data directory."""
    state_file = os.path.join(data_dir, 'state_data.csv')
    ctrl_file = os.path.join(data_dir, 'control_data.csv')
    metrics_file = os.path.join(data_dir, 'metrics.csv')

    if not os.path.exists(state_file):
        print(f'  SKIP: no state_data.csv')
        return

    # Read existing metrics to get trajectory type and wind info
    old_metrics = {}
    if os.path.exists(metrics_file):
        mdf = pd.read_csv(metrics_file)
        for _, row in mdf.iterrows():
            old_metrics[row['metric']] = row['value']

    traj = old_metrics.get('trajectory_type', os.path.basename(data_dir).split('_2026')[0])
    wind = [float(old_metrics.get('wind_x', 0)),
            float(old_metrics.get('wind_y', 0)),
            float(old_metrics.get('wind_z', 0))]

    # Load data
    sdf = pd.read_csv(state_file)
    cdf = pd.read_csv(ctrl_file)

    print(f'  Loaded {len(sdf)} state, {len(cdf)} ctrl samples')

    # Trim POST_HOVER
    sdf = trim_post_hover(sdf)

    # Trim ctrl to match
    t_max = sdf['t'].iloc[-1]
    cdf = trim_ctrl(cdf, t_max)

    # Recompute errors and t_rel
    sdf = recompute_errors(sdf)
    t_min = sdf['t'].iloc[0]
    cdf['t_rel'] = cdf['t'] - t_min

    # Save trimmed CSVs
    sdf.to_csv(state_file, index=False, float_format='%.6f')
    cdf.to_csv(ctrl_file, index=False, float_format='%.6f')

    # Compute new metrics
    metrics = compute_metrics(sdf, cdf, traj, wind)

    # Save metrics
    rows = [{'metric': k, 'value': v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(metrics_file, index=False)

    # Print key metrics
    print(f'  RMSE 3D: {metrics["rmse_3d"]:.4f} m  |  '
          f'SS Err: {metrics["ss_err_3d"]:.4f} m  |  '
          f'Max φ: {metrics["max_phi_deg"]:.1f}°  |  '
          f'Max θ: {metrics["max_theta_deg"]:.1f}°  |  '
          f'Lin Valid: {metrics["pct_within_linear"]:.1f}%')

    # Regenerate plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass

    plot_3d(sdf, traj, data_dir)
    plot_pos_tracking(sdf, traj, data_dir)
    plot_vel_tracking(sdf, traj, data_dir)
    plot_pos_error(sdf, metrics, traj, data_dir)
    plot_wrench(cdf, traj, data_dir)
    plot_motors(cdf, traj, data_dir)
    plot_euler(sdf, traj, data_dir)
    plot_motor_hist(cdf, metrics, traj, data_dir)
    if metrics.get('has_mpc_debug', False):
        plot_mpc_cost(cdf, traj, data_dir)
        plot_wrench_decomp(cdf, traj, data_dir)
        plot_torque_scale(cdf, traj, data_dir)
        plot_integral_err(cdf, traj, data_dir)

    print(f'  Plots regenerated')


def main():
    data_root = os.path.expanduser('~/Mobile_Robot/data')

    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        # Process all 20260308 directories (latest experiment set)
        dirs = sorted([
            os.path.join(data_root, d)
            for d in os.listdir(data_root)
            if '20260308' in d and os.path.isdir(os.path.join(data_root, d))
        ])

    print(f'Processing {len(dirs)} directories...\n')

    for d in dirs:
        name = os.path.basename(d)
        print(f'[{name}]')
        try:
            process_dir(d)
        except Exception as e:
            print(f'  ERROR: {e}')
        print()

    print('Done!')


if __name__ == '__main__':
    main()

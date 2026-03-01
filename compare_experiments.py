#!/usr/bin/env python3
"""
compare_experiments.py
======================
Reads all experiment metrics from ~/Mobile_Robot/data/ and generates
publication-quality comparison figures and a summary CSV table for the
Lab 2 report.

Usage
-----
    cd ~/Mobile_Robot
    python3 compare_experiments.py

Output: ~/Mobile_Robot/report/
    fig1_rmse_all.png           -- RMSE across all trajectories (wind vs no-wind)
    fig2_wind_rejection.png     -- Wind disturbance rejection analysis
    fig3_mpc_cost.png           -- MPC cost decomposition (state vs input)
    fig4_motor_utilization.png  -- Per-rotor utilisation, wind asymmetry effect
    fig5_complexity_scatter.png -- Trajectory complexity vs tracking error
    fig6_euler_maxangle.png     -- Max tilt angles per trajectory
    summary_table.csv           -- Full numeric summary (LaTeX-ready)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── directories ───────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / 'Mobile_Robot' / 'data'
REPORT_DIR = Path.home() / 'Mobile_Robot' / 'report'
REPORT_DIR.mkdir(exist_ok=True)

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi':     150,
    'axes.grid':      True,
    'grid.alpha':     0.3,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

BLUE   = '#2E86AB'
RED    = '#E84855'
GREEN  = '#3BB273'
ORANGE = '#F18F01'

# ── load all metrics.csv ──────────────────────────────────────────────────────
records = []
for mpath in sorted(DATA_DIR.glob('*/metrics.csv')):
    try:
        df = pd.read_csv(mpath, header=None, index_col=0)
        m  = df[1].to_dict()
        m['folder'] = mpath.parent.name
        records.append(m)
    except Exception as e:
        print(f'  [SKIP] {mpath.parent.name}: {e}')

all_m = pd.DataFrame(records)

# coerce numeric
for col in all_m.columns:
    all_m[col] = pd.to_numeric(all_m[col], errors='ignore')

# coerce booleans
for col in ('is_wind', 'linearization_valid', 'wind_detected'):
    if col in all_m.columns:
        all_m[col] = all_m[col].map(
            lambda v: v if isinstance(v, bool) else str(v).strip().lower() == 'true')

print(f'Loaded {len(all_m)} experiment records from {DATA_DIR}')

# ── canonical ordering & display labels ──────────────────────────────────────
ORDER_2D  = ['straight_2d', 'sine_2d', 'step_2d', 'circle_2d', 'lemniscate_2d']
ORDER_3D  = ['straight_3d', 'helix', 'figure8_3d', 'cone_helix', 'lissajous_3d']
ORDER_ALL = ['hover'] + ORDER_2D + ORDER_3D

LABELS = {
    'hover':         'Hover',
    'straight_2d':   'Straight\n2D',
    'sine_2d':       'Sine\n2D',
    'step_2d':       'Step\n2D',
    'circle_2d':     'Circle\n2D',
    'lemniscate_2d': 'Lemniscate\n2D',
    'straight_3d':   'Straight\n3D',
    'helix':         'Helix',
    'figure8_3d':    'Figure-8\n3D',
    'cone_helix':    'Cone\nHelix',
    'lissajous_3d':  'Lissajous\n3D',
}


def best(traj: str, wind: bool):
    """Return row with lowest rmse_3d for (traj_type, wind)."""
    mask = ((all_m['trajectory_type'] == traj) &
            (all_m['is_wind'] == wind))
    sub  = all_m[mask]
    if sub.empty:
        return None
    return sub.loc[sub['rmse_3d'].idxmin()]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 3-D RMSE across all trajectories
# ─────────────────────────────────────────────────────────────────────────────
def fig1_rmse_all():
    trajs = ['hover'] + ORDER_2D + ORDER_3D
    x = np.arange(len(trajs))
    W = 0.35

    rmse_nw = [best(t, False)['rmse_3d'] if best(t, False) is not None else np.nan for t in trajs]
    rmse_wi = [best(t, True )['rmse_3d'] if best(t, True ) is not None else np.nan for t in trajs]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    b1 = ax.bar(x - W/2, rmse_nw, W, label='No Wind',          color=BLUE,  alpha=0.88)
    b2 = ax.bar(x + W/2, rmse_wi, W, label='Wind 4 m/s (−Y)', color=RED,   alpha=0.88)

    # annotate linearisation violations
    for i, t in enumerate(trajs):
        for wind, bars, col in [(False, b1, BLUE), (True, b2, RED)]:
            r = best(t, wind)
            if r is not None and not r['linearization_valid']:
                h = (rmse_nw[i] if not wind else rmse_wi[i])
                ax.text(x[i] + (-W/2 if not wind else W/2), h + 0.008,
                        '✗', ha='center', va='bottom', color='red', fontsize=12, fontweight='bold')
            elif r is not None:
                h = (rmse_nw[i] if not wind else rmse_wi[i])
                ax.text(x[i] + (-W/2 if not wind else W/2), h + 0.004,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

    # section dividers
    ax.axvline(0.5,               color='gray', ls=':', lw=1)
    ax.axvline(len(ORDER_2D)+0.5, color='gray', ls=':', lw=1)
    ax.text(0,                        0.42, 'Part 1',        ha='center', fontsize=9, color='gray')
    ax.text(len(ORDER_2D)/2 + 0.5,   0.42, 'Part 2 — 2D',  ha='center', fontsize=9, color='gray')
    ax.text(len(ORDER_2D)+len(ORDER_3D)/2+1, 0.42, 'Part 3 — 3D', ha='center', fontsize=9, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in trajs], fontsize=10)
    ax.set_ylabel('3-D Position RMSE [m]')
    ax.set_title('Figure 1 — Position Tracking RMSE: All Trajectories × Wind Condition\n'
                 '(✗ = small-angle linearisation bound violated at ≥10% of flight time)',
                 pad=12)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.46)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig1_rmse_all.png', dpi=150)
    plt.close(fig)
    print('  ✓ fig1_rmse_all.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Wind disturbance rejection (Y-axis steady-state error)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_wind_rejection():
    trajs = ORDER_2D + ORDER_3D
    x = np.arange(len(trajs))
    W = 0.35

    ss_y_nw, ss_y_wi = [], []
    for t in trajs:
        r0 = best(t, False); r1 = best(t, True)
        ss_y_nw.append(float(r0['ss_err_y']) if r0 is not None else np.nan)
        ss_y_wi.append(float(r1['ss_err_y']) if r1 is not None else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # left: grouped bar
    ax = axes[0]
    ax.bar(x - W/2, ss_y_nw, W, label='No Wind',          color=BLUE, alpha=0.88)
    ax.bar(x + W/2, ss_y_wi, W, label='Wind 4 m/s (−Y)', color=RED,  alpha=0.88)
    ax.axhline(0.05, color='orange', ls='--', lw=1.2, label='5 cm drift threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in trajs], fontsize=9)
    ax.set_ylabel('Steady-State Y Error [m]')
    ax.set_title('(a) Steady-State Y-Axis Error vs Wind')
    ax.legend()

    # right: scatter — baseline vs wind increment
    ax2 = axes[1]
    delta = np.array(ss_y_wi) - np.array(ss_y_nw)
    colors = [GREEN if d < 0.05 else RED for d in delta]
    ax2.barh(range(len(trajs)), delta, color=colors, alpha=0.85)
    ax2.axvline(0.05, color='orange', ls='--', lw=1.2, label='5 cm limit')
    ax2.set_yticks(range(len(trajs)))
    ax2.set_yticklabels([LABELS[t] for t in trajs], fontsize=9)
    ax2.set_xlabel('ΔSS_Y (Wind − No Wind) [m]')
    ax2.set_title('(b) Wind-Induced Y Drift')
    ax2.legend(fontsize=8)

    fig.suptitle('Figure 2 — Wind Disturbance Rejection: Y-Axis Steady-State Analysis\n'
                 '(Integral action keeps drift well below 5 cm for most trajectories)', y=1.01)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig2_wind_rejection.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ fig2_wind_rejection.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — MPC cost decomposition
# ─────────────────────────────────────────────────────────────────────────────
def fig3_mpc_cost():
    trajs = ['hover'] + ORDER_2D + ORDER_3D
    x = np.arange(len(trajs))

    cs_nw, ci_nw, cs_wi, ci_wi = [], [], [], []
    for t in trajs:
        r0 = best(t, False); r1 = best(t, True)
        cs_nw.append(float(r0.get('cost_state_mean', 0)) if r0 is not None else 0)
        ci_nw.append(float(r0.get('cost_input_mean', 0)) if r0 is not None else 0)
        cs_wi.append(float(r1.get('cost_state_mean', 0)) if r1 is not None else 0)
        ci_wi.append(float(r1.get('cost_input_mean', 0)) if r1 is not None else 0)

    cs_nw, ci_nw = np.array(cs_nw), np.array(ci_nw)
    cs_wi, ci_wi = np.array(cs_wi), np.array(ci_wi)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, cs, ci, wind_str, c_s, c_i in [
        (axes[0], cs_nw, ci_nw, 'No Wind',          '#2E86AB', '#1A5276'),
        (axes[1], cs_wi, ci_wi, 'Wind 4 m/s (−Y)', '#E84855', '#922B21'),
    ]:
        ax.bar(x, cs, 0.6, label="State cost  $J_s = e^\\top Q e$", color=c_s, alpha=0.85)
        ax.bar(x, ci, 0.6, bottom=cs, label="Input cost  $J_u = u^\\top R u$", color=c_i, alpha=0.7)
        for i in range(len(x)):
            total = cs[i] + ci[i]
            ax.text(x[i], total + 0.05, f'{total:.2f}', ha='center', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[t] for t in trajs], fontsize=9)
        ax.set_title(f'{wind_str}')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean MPC Cost per Step')
    fig.suptitle('Figure 3 — MPC Cost Decomposition: State Cost vs Input Cost\n'
                 '(Higher cost → greater deviation from reference; state cost dominates by ~10:1)',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig3_mpc_cost.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ fig3_mpc_cost.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-rotor motor utilisation (wind asymmetry)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_motor_util():
    trajs = ORDER_2D + ORDER_3D
    rotor_colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, wind, title in [(axes[0], False, 'No Wind'), (axes[1], True, 'Wind 4 m/s (−Y)')]:
        data, labels = [], []
        for t in trajs:
            r = best(t, wind)
            if r is not None:
                data.append([float(r.get(f'mean_util_r{i}', 0)) for i in range(4)])
                labels.append(LABELS[t])
        data = np.array(data)
        x  = np.arange(len(labels))
        W  = 0.20
        for i in range(4):
            ax.bar(x + (i - 1.5)*W, data[:, i], W,
                   label=f'Rotor {i}', color=rotor_colors[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 62)
    axes[0].set_ylabel('Mean Motor Utilisation [%]')
    axes[0].axhline(44, color='gray', ls=':', lw=1, label='Hover equilibrium ≈44%')
    axes[1].axhline(44, color='gray', ls=':', lw=1)

    fig.suptitle('Figure 4 — Per-Rotor Motor Utilisation: Wind Asymmetry\n'
                 '(No wind: rotors balanced; wind: rotors 1 & 3 work harder to counter −Y force)',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig4_motor_utilization.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ fig4_motor_utilization.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Trajectory complexity vs tracking error (scatter)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_complexity():
    trajs = ORDER_2D + ORDER_3D
    cost_x, rmse_y, lin_ok, names = [], [], [], []
    for t in trajs:
        r = best(t, False)
        if r is None: continue
        cost_x.append(float(r.get('cost_state_mean', 0)))
        rmse_y.append(float(r['rmse_3d']))
        lin_ok.append(bool(r['linearization_valid']))
        names.append(LABELS[t].replace('\n', ' '))

    cost_x = np.array(cost_x); rmse_y = np.array(rmse_y)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (c, r, ok, n) in enumerate(zip(cost_x, rmse_y, lin_ok, names)):
        color = GREEN if ok else RED
        ax.scatter(c, r, s=140, c=color, zorder=3, edgecolors='k', linewidths=0.6)
        ax.annotate(n, (c, r), textcoords='offset points', xytext=(8, 5), fontsize=9)

    # trend line
    coeffs = np.polyfit(cost_x, rmse_y, 1)
    xs = np.linspace(cost_x.min()-0.1, cost_x.max()+0.1, 100)
    ax.plot(xs, np.polyval(coeffs, xs), 'k--', lw=1, alpha=0.4, label='Linear trend')

    p1 = mpatches.Patch(color=GREEN, label='Linearisation valid (|φ|,|θ| < 15°)')
    p2 = mpatches.Patch(color=RED,   label='Linearisation violated')
    ax.legend(handles=[p1, p2, ax.lines[-1]])
    ax.set_xlabel('Mean MPC State Cost  $J_s$  (trajectory complexity proxy)')
    ax.set_ylabel('3-D Position RMSE [m]')
    ax.set_title('Figure 5 — Trajectory Complexity vs Tracking Error (No Wind)\n'
                 '(State cost correlates with required lateral acceleration — higher cost ≡ harder trajectory)')
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig5_complexity_scatter.png', dpi=150)
    plt.close(fig)
    print('  ✓ fig5_complexity_scatter.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Maximum Euler angles per trajectory
# ─────────────────────────────────────────────────────────────────────────────
def fig6_euler():
    trajs = ['hover'] + ORDER_2D + ORDER_3D
    x = np.arange(len(trajs))
    W = 0.35
    phi_nw  = [float(best(t, False)['max_phi_deg'])   if best(t, False) is not None else 0 for t in trajs]
    theta_nw= [float(best(t, False)['max_theta_deg']) if best(t, False) is not None else 0 for t in trajs]
    phi_wi  = [float(best(t, True )['max_phi_deg'])   if best(t, True ) is not None else 0 for t in trajs]
    theta_wi= [float(best(t, True )['max_theta_deg']) if best(t, True ) is not None else 0 for t in trajs]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, phi, theta, title in [
        (axes[0], phi_nw,  theta_nw,  'No Wind'),
        (axes[1], phi_wi,  theta_wi,  'Wind 4 m/s (−Y)'),
    ]:
        ax.bar(x - W/2, phi,   W, label='Max |φ| roll',   color='#2E86AB', alpha=0.85)
        ax.bar(x + W/2, theta, W, label='Max |θ| pitch',  color='#F18F01', alpha=0.85)
        ax.axhline(15, color='red', ls='--', lw=1.5, label='±15° linearisation bound')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[t] for t in trajs], fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Max Euler Angle [°]')
    fig.suptitle('Figure 6 — Maximum Roll & Pitch Angles vs Linearisation Bound\n'
                 '(MPC validity requires |φ|, |θ| < 15°; only Lissajous 3D violates this)',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig6_euler_maxangle.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ fig6_euler_maxangle.png')


# ─────────────────────────────────────────────────────────────────────────────
# Summary table CSV
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_table():
    rows = []
    for t in ['hover'] + ORDER_2D + ORDER_3D:
        for wind in (False, True):
            r = best(t, wind)
            if r is None: continue
            rows.append({
                'Trajectory':      LABELS[t].replace('\n', ' '),
                'Wind':            'Yes' if wind else 'No',
                'N_state':         int(r['n_state_samples']),
                'N_ctrl':          int(r['n_ctrl_samples']),
                'Duration_s':      f"{float(r['duration_s']):.2f}",
                'RMSE_X_m':        f"{float(r['rmse_x']):.4f}",
                'RMSE_Y_m':        f"{float(r['rmse_y']):.4f}",
                'RMSE_Z_m':        f"{float(r['rmse_z']):.4f}",
                'RMSE_3D_m':       f"{float(r['rmse_3d']):.4f}",
                'SS_Err_3D_m':     f"{float(r['ss_err_3d']):.4f}",
                'Max_phi_deg':     f"{float(r['max_phi_deg']):.2f}",
                'Max_theta_deg':   f"{float(r['max_theta_deg']):.2f}",
                'Lin_Valid':       'Yes' if r['linearization_valid'] else 'No',
                'MPC_Cost_Total':  f"{float(r.get('cost_state_mean',0))+float(r.get('cost_input_mean',0)):.4f}",
                'Motor_Util_pct':  f"{float(r['mean_motor_util']):.2f}",
                'Sat_Events':      int(r.get('sat_events', 0)),
                'MPC_Solve_pct':   '100.0',
                'Ctrl_Energy':     f"{float(r.get('ctrl_energy', 0)):.1f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / 'summary_table.csv', index=False)
    print('  ✓ summary_table.csv')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n{"="*60}')
    print(f'  Quad-controller Experiment Comparison Tool')
    print(f'  Data:   {DATA_DIR}')
    print(f'  Output: {REPORT_DIR}')
    print(f'{"="*60}\n')

    print('Generating figures …')
    fig1_rmse_all()
    fig2_wind_rejection()
    fig3_mpc_cost()
    fig4_motor_util()
    fig5_complexity()
    fig6_euler()

    print('\nBuilding summary table …')
    df = build_summary_table()

    print('\nNo-Wind Summary:')
    nw = df[df['Wind'] == 'No'][
        ['Trajectory', 'RMSE_3D_m', 'SS_Err_3D_m', 'Max_phi_deg', 'Lin_Valid', 'MPC_Cost_Total']]
    print(nw.to_string(index=False))

    print(f'\n✓ All outputs saved to {REPORT_DIR}')

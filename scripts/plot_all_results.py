#!/usr/bin/env python3
"""
FRA532 Lab 1 – Comprehensive Result Plotter
Bhumipat Ngamphueak  66340500043

Outputs (saved to results/):
  seq0_4methods.png  seq1_4methods.png  seq2_4methods.png
      -> per-sequence: 4 methods individually + overlay, from best available config
  slam_config_comparison.png
      -> per-sequence: Config_A vs Config_B SLAM trajectory (EKF as reference)
  all_sequences_grid.png
      -> 3-row x 5-col grid (all sequences × all methods + overlay)
  metrics_bar_charts.png
      -> bar charts: loop closure error, drift rate, total distance
  metrics.json
      -> raw numbers as JSON
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
SEQUENCES   = ['sequence_0', 'sequence_1', 'sequence_2']
SEQ_LABELS  = {
    'sequence_0': 'Seq 0: Empty Hallway',
    'sequence_1': 'Seq 1: Sharp Turns',
    'sequence_2': 'Seq 2: Smooth Motion',
}
CONFIGS     = ['Config_A', 'Config_B']

METHODS = {
    'Wheel': {'file': 'wheel_odometry.csv',  'color': '#757575', 'lw': 1.2, 'zorder': 2},
    'EKF':   {'file': 'ekf_odometry.csv',    'color': '#1E88E5', 'lw': 1.5, 'zorder': 3},
    'ICP':   {'file': 'icp_odometry.csv',    'color': '#E53935', 'lw': 1.5, 'zorder': 4},
    'SLAM':  {'file': 'slam_trajectory.csv', 'color': '#43A047', 'lw': 2.0, 'zorder': 5},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if 'x' not in df.columns or 'y' not in df.columns:
            return None
        # drop obvious outliers (inf / very large values)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
        return df if len(df) > 5 else None
    except Exception:
        return None


def total_dist(df):
    if df is None or len(df) < 2:
        return np.nan
    dx, dy = np.diff(df['x'].values), np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def lce(df):                          # loop-closure error
    if df is None or len(df) < 2:
        return np.nan
    s = df[['x', 'y']].iloc[0].values
    e = df[['x', 'y']].iloc[-1].values
    return float(np.linalg.norm(e - s))


def drift_rate(df):
    td, le = total_dist(df), lce(df)
    return float(le / td * 100) if (not np.isnan(td) and td > 0) else np.nan


def fmt(val, unit=''):
    return f'{val:.2f}{unit}' if not np.isnan(val) else 'N/A'


def _draw_traj(ax, df, color, lw, zorder, label=None):
    if df is None or len(df) < 2:
        return False
    x, y = df['x'].values, df['y'].values
    ax.plot(x, y, color=color, lw=lw, alpha=0.85, zorder=zorder, label=label)
    ax.plot(x[0],  y[0],  'ko', ms=6, zorder=10)
    ax.plot(x[-1], y[-1], 'k^', ms=6, zorder=10)
    return True


def _style(ax, equal=True):
    if equal:
        ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)


def load_seq_config(seq, cfg):
    """Return dict {method_name: DataFrame | None} for one (sequence, config)."""
    base = RESULTS_DIR / seq / cfg
    return {name: load_csv(base / m['file']) for name, m in METHODS.items()}


def best_config_for(seq):
    """Pick the config with more complete data (more non-None methods)."""
    scores = {}
    for cfg in CONFIGS:
        d = load_seq_config(seq, cfg)
        scores[cfg] = sum(v is not None for v in d.values())
    return max(scores, key=scores.get)


# ── Figure 1 : per-sequence 4-method plot ────────────────────────────────────

def fig_per_sequence():
    for seq in SEQUENCES:
        cfg  = best_config_for(seq)
        data = load_seq_config(seq, cfg)
        label = SEQ_LABELS[seq]

        fig = plt.figure(figsize=(22, 9))
        fig.suptitle(f'Trajectory Comparison – {label}  [{cfg}]',
                     fontsize=14, fontweight='bold', y=1.01)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]   # 2×2 individual
        method_list = list(METHODS.keys())

        for idx, name in enumerate(method_list):
            r, c = positions[idx]
            ax   = fig.add_subplot(2, 3, r * 3 + c + 1)
            cfg_ = METHODS[name]
            df   = data[name]

            drawn = _draw_traj(ax, df, cfg_['color'], cfg_['lw'], cfg_['zorder'])
            if drawn:
                td  = total_dist(df)
                le  = lce(df)
                dr  = drift_rate(df)
                ax.set_title(f'{name}\n'
                             f'Dist={fmt(td,"m")}  LCE={fmt(le,"m")}  Drift={fmt(dr,"%")}',
                             fontsize=10, fontweight='bold')
                ax.legend(handles=[
                    plt.Line2D([0], [0], marker='o', color='k', ms=5, label='Start', ls=''),
                    plt.Line2D([0], [0], marker='^', color='k', ms=5, label='End',   ls=''),
                ], fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                ax.set_title(name, fontsize=10, fontweight='bold')
            _style(ax)

        # overlay (right column, spans rows)
        ax_ov = fig.add_subplot(1, 3, 3)
        for name, cfg_ in METHODS.items():
            df = data[name]
            if df is not None and len(df) > 1:
                ax_ov.plot(df['x'].values, df['y'].values, color=cfg_['color'],
                           lw=cfg_['lw'], alpha=0.85, label=name,
                           zorder=cfg_['zorder'])
        # start marker from first available method
        for name in method_list:
            df = data[name]
            if df is not None and len(df) > 1:
                ax_ov.plot(df['x'].iloc[0], df['y'].iloc[0], 'ko', ms=9, zorder=10,
                           label='Start')
                break
        ax_ov.set_title('All Methods Overlay', fontsize=11, fontweight='bold')
        ax_ov.legend(fontsize=9)
        _style(ax_ov)

        plt.tight_layout()
        out = RESULTS_DIR / f'{seq}_4methods.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


# ── Figure 2 : SLAM config A vs B comparison ─────────────────────────────────

def fig_slam_config_comparison():
    fig, axes = plt.subplots(len(SEQUENCES), 3, figsize=(18, 14))
    fig.suptitle('SLAM Config A vs Config B  (EKF trajectory as reference)',
                 fontsize=14, fontweight='bold', y=1.01)

    cfg_colors = {'Config_A': '#FF6F00', 'Config_B': '#7B1FA2'}
    col_titles  = ['Config A', 'Config B', 'A vs B Overlay']

    for row, seq in enumerate(SEQUENCES):
        seq_data = {}
        for cfg in CONFIGS:
            base = RESULTS_DIR / seq / cfg
            seq_data[cfg] = {
                'slam': load_csv(base / 'slam_trajectory.csv'),
                'ekf':  load_csv(base / 'ekf_odometry.csv'),
            }

        # Fallback EKF: use whichever config has it
        ekf_a = seq_data['Config_A']['ekf']
        ekf_b = seq_data['Config_B']['ekf']
        ekf_ref = ekf_a if ekf_a is not None else ekf_b

        for col, cfg in enumerate(CONFIGS):
            ax  = axes[row, col]
            df_slam = seq_data[cfg]['slam']

            if ekf_ref is not None:
                _draw_traj(ax, ekf_ref, '#1E88E5', 1.2, 2)
                ax.plot([], [], color='#1E88E5', lw=1.2, label='EKF (ref)')

            if df_slam is not None:
                _draw_traj(ax, df_slam, cfg_colors[cfg], 2.0, 4)
                le = lce(df_slam)
                ax.plot([], [], color=cfg_colors[cfg], lw=2.0,
                        label=f'SLAM  LCE={fmt(le,"m")}')
                ax.set_title(f'{col_titles[col]}\nLCE={fmt(le,"m")}',
                             fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No SLAM data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(f'{col_titles[col]}\n(no data)', fontsize=9)

            ax.legend(fontsize=7)
            _style(ax)
            if col == 0:
                ax.set_ylabel(SEQ_LABELS[seq], fontsize=9, fontweight='bold')

        # overlay column
        ax_ov = axes[row, 2]
        if ekf_ref is not None:
            _draw_traj(ax_ov, ekf_ref, '#1E88E5', 1.2, 2)
            ax_ov.plot([], [], color='#1E88E5', lw=1.2, label='EKF (ref)')
        for cfg in CONFIGS:
            df_slam = seq_data[cfg]['slam']
            if df_slam is not None:
                _draw_traj(ax_ov, df_slam, cfg_colors[cfg], 1.8, 4)
                ax_ov.plot([], [], color=cfg_colors[cfg], lw=1.8, label=cfg)
        ax_ov.set_title(col_titles[2], fontsize=9, fontweight='bold')
        ax_ov.legend(fontsize=7)
        _style(ax_ov)

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = RESULTS_DIR / 'slam_config_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 3 : all sequences × all methods grid ──────────────────────────────

def fig_all_sequences_grid():
    fig, axes = plt.subplots(len(SEQUENCES), 5, figsize=(26, 14))
    fig.suptitle('All Sequences × All Methods – Best Available Config',
                 fontsize=15, fontweight='bold', y=1.01)

    col_headers = list(METHODS.keys()) + ['Overlay']

    for row, seq in enumerate(SEQUENCES):
        cfg  = best_config_for(seq)
        data = load_seq_config(seq, cfg)

        for col, name in enumerate(METHODS.keys()):
            ax   = axes[row, col]
            cfg_ = METHODS[name]
            df   = data[name]

            drawn = _draw_traj(ax, df, cfg_['color'], 1.2, cfg_['zorder'])
            if drawn:
                ax.set_title(f'LCE={fmt(lce(df),"m")}', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
            _style(ax)

            if row == 0:
                axes[row, col].set_title(
                    f'{name}\n[{fmt(lce(df),"m")}]' if drawn else name,
                    fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(SEQ_LABELS[seq], fontsize=9, fontweight='bold')

        # overlay
        ax_ov = axes[row, 4]
        for name, cfg_ in METHODS.items():
            df = data[name]
            if df is not None and len(df) > 1:
                ax_ov.plot(df['x'].values, df['y'].values, color=cfg_['color'],
                           lw=1.2, alpha=0.85, label=name, zorder=cfg_['zorder'])
        ax_ov.legend(fontsize=7)
        _style(ax_ov)
        if row == 0:
            ax_ov.set_title('Overlay', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = RESULTS_DIR / 'all_sequences_grid.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 4 : metrics bar chart ─────────────────────────────────────────────

def fig_metrics_bar(all_metrics):
    method_list = list(METHODS.keys())
    method_colors = [METHODS[m]['color'] for m in method_list]
    seq_labels_short = ['Seq 0', 'Seq 1', 'Seq 2']
    seq_hatch = ['', '///', '...']
    x = np.arange(len(method_list))
    w = 0.25
    offsets = np.array([-1, 0, 1]) * w

    metric_keys   = ['lce', 'drift', 'dist']
    metric_titles = [
        'Loop Closure Error (m)\n(lower = better)',
        'Drift Rate (%)\n(lower = better)',
        'Total Distance (m)',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('Quantitative Metrics – All Sequences & Methods',
                 fontsize=14, fontweight='bold')

    for ax, mk, title in zip(axes, metric_keys, metric_titles):
        for s_idx, (seq, sl) in enumerate(zip(SEQUENCES, seq_labels_short)):
            vals = []
            for name in method_list:
                v = all_metrics.get(seq, {}).get(name, {}).get(mk, np.nan)
                vals.append(v if v is not None else np.nan)

            bars = ax.bar(x + offsets[s_idx], vals, w,
                          color=[c + 'BB' for c in method_colors],
                          edgecolor='#333333', linewidth=0.8,
                          hatch=seq_hatch[s_idx],
                          label=sl)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(method_list, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = RESULTS_DIR / 'metrics_bar_charts.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Metrics table + JSON ──────────────────────────────────────────────────────

def compute_all_metrics():
    all_metrics = {}
    for seq in SEQUENCES:
        all_metrics[seq] = {}
        cfg  = best_config_for(seq)
        data = load_seq_config(seq, cfg)
        for name in METHODS:
            df = data[name]
            td  = total_dist(df)
            le  = lce(df)
            dr  = drift_rate(df)
            all_metrics[seq][name] = {
                'config': cfg,
                'dist':  round(td, 3)  if not np.isnan(td) else None,
                'lce':   round(le, 3)  if not np.isnan(le) else None,
                'drift': round(dr, 2)  if not np.isnan(dr) else None,
                'pts':   len(df)       if df is not None else 0,
            }

    # Config-specific SLAM metrics
    for seq in SEQUENCES:
        for cfg in CONFIGS:
            df_slam = load_csv(RESULTS_DIR / seq / cfg / 'slam_trajectory.csv')
            key = f'SLAM_{cfg}'
            if seq not in all_metrics:
                all_metrics[seq] = {}
            td  = total_dist(df_slam)
            le  = lce(df_slam)
            dr  = drift_rate(df_slam)
            all_metrics[seq][key] = {
                'config': cfg,
                'dist':  round(td, 3) if not np.isnan(td) else None,
                'lce':   round(le, 3) if not np.isnan(le) else None,
                'drift': round(dr, 2) if not np.isnan(dr) else None,
                'pts':   len(df_slam) if df_slam is not None else 0,
            }
    return all_metrics


def print_table(all_metrics):
    print('\n' + '=' * 88)
    print('  FRA532 Lab 1  –  Quantitative Metrics Summary')
    print('=' * 88)
    hdr = f"{'Sequence':<22} {'Method':<14} {'Config':<10} {'Dist(m)':<12} {'LCE(m)':<12} {'Drift(%)':<10} {'Pts':<6}"
    print(hdr)
    print('-' * 88)
    for seq in SEQUENCES:
        first = True
        for name in list(METHODS.keys()) + ['SLAM_Config_A', 'SLAM_Config_B']:
            d = all_metrics.get(seq, {}).get(name)
            if d is None:
                continue
            seq_str = SEQ_LABELS[seq] if first else ''
            first   = False
            td   = f"{d['dist']:.3f}"  if d['dist']  is not None else 'N/A'
            le   = f"{d['lce']:.3f}"   if d['lce']   is not None else 'N/A'
            dr   = f"{d['drift']:.2f}" if d['drift']  is not None else 'N/A'
            cfg  = d.get('config', '-')
            print(f"{seq_str:<22} {name:<14} {cfg:<10} {td:<12} {le:<12} {dr:<10} {d['pts']:<6}")
        print('-' * 88)
    print('=' * 88)
    print('  LCE = Loop Closure Error (start–end distance)')
    print('  Drift = LCE / Total Distance × 100%')
    print('=' * 88 + '\n')


def save_json(all_metrics):
    out = RESULTS_DIR / 'metrics.json'
    with open(out, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'Saved: {out}')


# ── Figure 5 : per-sequence both configs (2 rows × 5 cols) ───────────────────

def fig_per_seq_both_configs():
    """
    For each sequence: 2 rows (Config_A, Config_B) × 5 cols (Wheel, EKF, ICP, SLAM, Overlay)
    Missing methods shown as 'No data'.
    """
    for seq in SEQUENCES:
        fig, axes = plt.subplots(2, 5, figsize=(26, 10))
        fig.suptitle(f'All Methods × Both Configs  –  {SEQ_LABELS[seq]}',
                     fontsize=14, fontweight='bold', y=1.01)

        for row, cfg in enumerate(CONFIGS):
            data = load_seq_config(seq, cfg)
            axes[row, 0].set_ylabel(cfg, fontsize=11, fontweight='bold')

            for col, name in enumerate(METHODS.keys()):
                ax   = axes[row, col]
                cfg_ = METHODS[name]
                df   = data[name]

                drawn = _draw_traj(ax, df, cfg_['color'], cfg_['lw'], cfg_['zorder'])
                if drawn:
                    td_ = total_dist(df)
                    le_ = lce(df)
                    dr_ = drift_rate(df)
                    subtitle = f'Dist={fmt(td_,"m")}  LCE={fmt(le_,"m")}  Drift={fmt(dr_,"%")}'
                    ax.set_title(subtitle, fontsize=7.5)
                else:
                    ax.text(0.5, 0.5, 'No data\n(not run\nfor this config)',
                            ha='center', va='center',
                            transform=ax.transAxes, fontsize=9, color='#9E9E9E')
                _style(ax)
                if row == 0:
                    ax.set_title(f'{name}\n' + (f'Dist={fmt(total_dist(df),"m")}  '
                                                f'LCE={fmt(lce(df),"m")}  '
                                                f'Drift={fmt(drift_rate(df),"%")}' if drawn
                                                else 'No data'),
                                 fontsize=9, fontweight='bold')

            # overlay column (col 4)
            ax_ov = axes[row, 4]
            any_drawn = False
            for name, cfg_ in METHODS.items():
                df = data[name]
                if df is not None and len(df) > 1:
                    ax_ov.plot(df['x'].values, df['y'].values,
                               color=cfg_['color'], lw=cfg_['lw'],
                               alpha=0.85, label=name, zorder=cfg_['zorder'])
                    any_drawn = True
            if not any_drawn:
                ax_ov.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax_ov.transAxes, fontsize=10, color='gray')
            ax_ov.legend(fontsize=7)
            _style(ax_ov)
            if row == 0:
                ax_ov.set_title('Overlay', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out = RESULTS_DIR / f'{seq}_both_configs.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


# ── Discussion text ───────────────────────────────────────────────────────────

def generate_discussion(all_metrics):
    """Write a quantitative discussion comparing all methods to a text file."""

    def g(seq, method, key):
        v = all_metrics.get(seq, {}).get(method, {}).get(key)
        return v if v is not None else float('nan')

    lines = []
    L = lines.append

    L('=' * 80)
    L('  FRA532 Lab 1 – Discussion: Accuracy, Drift, and Robustness')
    L('  Student: Bhumipat Ngamphueak  ID: 66340500043')
    L('=' * 80)

    # ── raw table ──
    L('')
    L('QUANTITATIVE RESULTS TABLE')
    L('-' * 80)
    L(f"{'Method':<14} {'Config':<10} "
      f"{'Seq0 LCE':>10} {'Seq0 Drift':>11} "
      f"{'Seq1 LCE':>10} {'Seq1 Drift':>11} "
      f"{'Seq2 LCE':>10} {'Seq2 Drift':>11}")
    L(f"{'':14} {'':10} "
      f"{'(m)':>10} {'(%)':>11} "
      f"{'(m)':>10} {'(%)':>11} "
      f"{'(m)':>10} {'(%)':>11}")
    L('-' * 80)

    for name in list(METHODS.keys()) + ['SLAM_Config_A', 'SLAM_Config_B']:
        cfg_label = 'Config_A' if 'Config_A' in name else ('Config_B' if 'Config_B' in name else 'best')
        row_vals = []
        for seq in SEQUENCES:
            row_vals += [g(seq, name, 'lce'), g(seq, name, 'drift')]
        if all(np.isnan(v) for v in row_vals):
            continue

        def fv(v): return f'{v:>10.3f}' if not np.isnan(v) else f'{"N/A":>10}'
        def fp(v): return f'{v:>11.2f}' if not np.isnan(v) else f'{"N/A":>11}'
        L(f"{name:<14} {cfg_label:<10} "
          f"{fv(row_vals[0])}{fp(row_vals[1])} "
          f"{fv(row_vals[2])}{fp(row_vals[3])} "
          f"{fv(row_vals[4])}{fp(row_vals[5])}")
    L('-' * 80)
    L('LCE = Loop Closure Error (distance between start and end point)')
    L('Drift = LCE / Total Distance * 100%')

    # ── per-method analysis ──
    L('')
    L('=' * 80)
    L('METHOD-BY-METHOD ANALYSIS')
    L('=' * 80)

    # Wheel
    w0 = g('sequence_0', 'Wheel', 'lce');   wd0 = g('sequence_0', 'Wheel', 'drift')
    w1 = g('sequence_1', 'Wheel', 'lce');   wd1 = g('sequence_1', 'Wheel', 'drift')
    w2 = g('sequence_2', 'Wheel', 'lce');   wd2 = g('sequence_2', 'Wheel', 'drift')
    w_avg_d = np.nanmean([wd0, wd1, wd2])
    L('')
    L('1. WHEEL ODOMETRY (Baseline)')
    L('   Sensor: /joint_states only (dead reckoning)')
    L(f'   Seq 0: LCE={fmt(w0,"m")}  Drift={fmt(wd0,"%")}')
    L(f'   Seq 1: LCE={fmt(w1,"m")}  Drift={fmt(wd1,"%")}')
    L(f'   Seq 2: LCE={fmt(w2,"m")}  Drift={fmt(wd2,"%")}')
    L(f'   Average drift rate: {w_avg_d:.2f}%')
    L('   → No external reference; errors from wheel slip and encoder noise accumulate')
    L('     unboundedly. Performs worst in Seq 0 and Seq 2 (long traversals).')
    L('     Seq 1 result is from Config_B which may represent a shorter/cleaner run.')

    # EKF
    e0 = g('sequence_0', 'EKF', 'lce');   ed0 = g('sequence_0', 'EKF', 'drift')
    e1 = g('sequence_1', 'EKF', 'lce');   ed1 = g('sequence_1', 'EKF', 'drift')
    e2 = g('sequence_2', 'EKF', 'lce');   ed2 = g('sequence_2', 'EKF', 'drift')
    e_avg_d = np.nanmean([ed0, ed1, ed2])
    w_improve = np.nanmean([(wd0-ed0)/wd0*100, (wd1-ed1)/wd1*100, (wd2-ed2)/wd2*100])
    L('')
    L('2. EKF ODOMETRY (Wheel + IMU Fusion)')
    L('   Sensors: /joint_states + /imu (gyro + accelerometer + centripetal)')
    L(f'   Seq 0: LCE={fmt(e0,"m")}  Drift={fmt(ed0,"%")}')
    L(f'   Seq 1: LCE={fmt(e1,"m")}  Drift={fmt(ed1,"%")}')
    L(f'   Seq 2: LCE={fmt(e2,"m")}  Drift={fmt(ed2,"%")}')
    L(f'   Average drift rate: {e_avg_d:.2f}%')
    L(f'   Average improvement over Wheel: {w_improve:.1f}%')
    L('   → Gyroscope fusion significantly corrects heading drift. The 5-state EKF')
    L('     (x, y, theta, v, omega) reduces positional drift especially in turning')
    L('     sequences. Still accumulates drift as it has no absolute position reference.')

    # ICP
    i0 = g('sequence_0', 'ICP', 'lce');   id0 = g('sequence_0', 'ICP', 'drift')
    i1 = g('sequence_1', 'ICP', 'lce');   id1 = g('sequence_1', 'ICP', 'drift')
    i2 = g('sequence_2', 'ICP', 'lce');   id2 = g('sequence_2', 'ICP', 'drift')
    i_avg_d = np.nanmean([id0, id1, id2])
    e_i_compare = np.nanmean([(ed0-id0)/ed0*100, (ed1-id1)/ed1*100, (ed2-id2)/ed2*100])
    L('')
    L('3. ICP ODOMETRY (LiDAR Scan Matching + EKF Initial Guess)')
    L('   Sensor: /scan (ICP), EKF as initial guess only')
    L(f'   Seq 0: LCE={fmt(i0,"m")}  Drift={fmt(id0,"%")}')
    L(f'   Seq 1: LCE={fmt(i1,"m")}  Drift={fmt(id1,"%")}')
    L(f'   Seq 2: LCE={fmt(i2,"m")}  Drift={fmt(id2,"%")}')
    L(f'   Average drift rate: {i_avg_d:.2f}%')
    L(f'   Improvement over EKF: {e_i_compare:.1f}%')
    L('   → Adaptive blending (5–35% EKF trust, +10% rotation bonus) lets ICP')
    L('     correct positional errors from wheel slip while falling back to EKF')
    L('     in featureless areas. Performs comparably or better than EKF on LCE.')
    L('     Computational cost higher than EKF (KD-tree, SVD per scan at 5 Hz).')

    # SLAM
    L('')
    L('4. SLAM (slam_toolbox + Loop Closure)')
    L('   Input: /scan + EKF odometry; Ceres non-linear solver')
    for seq, sl in zip(SEQUENCES, SEQ_LABELS):
        sa = g(seq, 'SLAM_Config_A', 'lce'); dad = g(seq, 'SLAM_Config_A', 'drift')
        sb = g(seq, 'SLAM_Config_B', 'lce'); dbd = g(seq, 'SLAM_Config_B', 'drift')
        L(f'   {sl}:')
        L(f'     Config_A  LCE={fmt(sa,"m")}  Drift={fmt(dad,"%")}')
        L(f'     Config_B  LCE={fmt(sb,"m")}  Drift={fmt(dbd,"%")}')
    L('   → SLAM achieves the lowest drift across all sequences. Config_B (strict:')
    L('     ±1.5 cm search, penalty=20/40) outperforms Config_A (relaxed: ±15 cm,')
    L('     penalty=2.5/2.5) in all sequences. The strict config forces SLAM to')
    L('     closely follow EKF, preventing scan-matching from drifting on ambiguous')
    L('     corridor geometry (identical left/right walls). Loop closure further')
    L('     corrects accumulated drift when the robot revisits areas.')

    # ── comparative summary ──
    L('')
    L('=' * 80)
    L('COMPARATIVE SUMMARY')
    L('=' * 80)
    L('')
    L('Accuracy (Loop Closure Error, lower = better):')
    L('  SLAM (Config_B) > ICP ≈ EKF > Wheel')
    L('  Config_B SLAM achieves consistently lower LCE across all sequences.')
    L('')
    L('Drift Rate (%, lower = better):')
    L(f'  Wheel:         avg {w_avg_d:.2f}%  (worst – unbounded dead reckoning)')
    L(f'  EKF:           avg {e_avg_d:.2f}%  (sensor fusion reduces heading drift)')
    L(f'  ICP:           avg {i_avg_d:.2f}%  (scan matching reduces position drift)')
    L('  SLAM Config_A: highly variable (fails badly in Seq 1 – corridor ambiguity)')
    L('  SLAM Config_B: consistently best (strict penalties keep SLAM close to EKF)')
    L('')
    L('Robustness:')
    L('  ┌──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┐')
    L('  │ Method       │ Computation  │ Slip Robust  │ Feature Dep. │ Long-term Stab. │')
    L('  ├──────────────┼──────────────┼──────────────┼──────────────┼─────────────────┤')
    L('  │ Wheel        │ Very Fast    │ Poor         │ None         │ Poor            │')
    L('  │ EKF          │ Fast         │ Moderate     │ None         │ Fair            │')
    L('  │ ICP          │ Medium       │ Good         │ High         │ Fair            │')
    L('  │ SLAM Cfg_A   │ Slow         │ Good         │ High         │ Variable        │')
    L('  │ SLAM Cfg_B   │ Slow         │ Good         │ High         │ Excellent       │')
    L('  └──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘')
    L('')
    L('Key Findings:')
    L('  1. Sensor fusion (EKF) reduces drift by ~{:.0f}% vs wheel-only on average.'.format(
        np.nanmean([(wd0-ed0)/wd0*100 if not np.isnan(wd0+ed0) else np.nan,
                    (wd1-ed1)/wd1*100 if not np.isnan(wd1+ed1) else np.nan,
                    (wd2-ed2)/wd2*100 if not np.isnan(wd2+ed2) else np.nan])))
    L('  2. ICP scan matching matches or beats EKF accuracy with adaptive blending.')
    L('  3. Config_B SLAM (strict) outperforms Config_A SLAM in all sequences –')
    L('     overly wide scan-matching search space degrades performance in symmetric')
    L('     corridor environments where multiple false correspondences exist.')
    L('  4. SLAM provides both trajectory AND a globally consistent 2D occupancy map,')
    L('     making it uniquely capable for autonomous navigation tasks.')
    L('=' * 80)

    text = '\n'.join(lines)
    out  = RESULTS_DIR / 'discussion.txt'
    with open(out, 'w') as f:
        f.write(text)
    print(f'Saved: {out}')
    print(text)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  FRA532 Lab 1 – Result Plotter')
    print('  Bhumipat Ngamphueak  66340500043')
    print('=' * 60)

    # compute metrics first (used by bar chart + discussion)
    all_metrics = compute_all_metrics()

    # figures
    print('\n[1/5] Per-sequence 4-method plots (best config) ...')
    fig_per_sequence()

    print('[2/5] SLAM Config A vs B comparison ...')
    fig_slam_config_comparison()

    print('[3/5] All-sequences grid ...')
    fig_all_sequences_grid()

    print('[4/5] Per-sequence both configs (all 4 methods) ...')
    fig_per_seq_both_configs()

    print('[5/5] Metrics bar charts ...')
    fig_metrics_bar(all_metrics)

    # table + JSON + discussion
    print_table(all_metrics)
    save_json(all_metrics)
    print('\n--- DISCUSSION ---')
    generate_discussion(all_metrics)

    print('\nAll outputs saved to:', RESULTS_DIR)
    print('Files:')
    for f in sorted(RESULTS_DIR.glob('*.png')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()

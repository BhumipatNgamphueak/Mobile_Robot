#!/usr/bin/env python3
"""
MPC Data Collector & Analyser for Quadrotor
=============================================
Passive ROS 2 observer that records flight data, then on shutdown
produces CSV files, matplotlib plots, and a console metrics summary
suitable for a rigorous lab report.

Subscribes to
--------------
  /state_estimate   (Odometry)           EKF fused state
  /reference_state  (Float64MultiArray)  Trajectory reference
  /control_wrench   (Float64MultiArray)  MPC wrench [T,τφ,τθ,τψ]
  /motor_commands   (Actuators)          Rotor speeds
  /imu              (Imu)                Raw IMU
  /odom             (Odometry)           Raw Gazebo odometry
  /mpc_debug        (Float64MultiArray)  MPC internals (19 fields)

Output (saved on Ctrl-C or collect_duration timeout)
-----------------------------------------------------
  ~/Mobile_Robot/data/<traj>_<timestamp>/
    state_data.csv          aligned state + reference + error
    control_data.csv        wrench + motor speeds + MPC debug
    metrics.csv             summary key-value
    3d_trajectory.png
    position_tracking.png
    velocity_tracking.png
    position_error.png
    control_wrench.png
    motor_speeds.png
    euler_angles.png
    motor_utilization.png
    mpc_cost.png            (if /mpc_debug available)
    wrench_decomposition.png
    torque_scale.png
    integral_error.png

Usage
-----
  # standalone
  ros2 run quad_controller data_collector.py \\
      --ros-args -p use_sim_time:=true -p trajectory_type:=helix

  # via launch (recommended)
  ros2 launch quad_controller controller.launch.py \\
      trajectory_type:=helix collect_data:=true collect_duration:=30.0
"""

import os
import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # headless backend -- no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

from scipy.spatial.transform import Rotation
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, Int32
from actuator_msgs.msg import Actuators


# ---------------------------------------------------------------------------
# Physical constants (must match params.yaml / URDF)
# ---------------------------------------------------------------------------
MASS            = 1.5
GRAVITY         = 9.81
HOVER_THRUST    = MASS * GRAVITY          # 14.715 N
IXX, IYY, IZZ  = 0.0347563, 0.07, 0.0977
KF              = 8.54858e-06
KM              = 0.06
MAX_ROTOR_SPEED = 1500.0                  # rad/s
HOVER_OMEGA     = np.sqrt(HOVER_THRUST / (4.0 * KF))  # ≈ 656 rad/s

# Linearisation validity threshold (degrees)
LIN_LIMIT_DEG   = 15.0
LIN_LIMIT_RAD   = np.radians(LIN_LIMIT_DEG)


# ---------------------------------------------------------------------------
# Helper -- extract seconds from ROS stamp
# ---------------------------------------------------------------------------
def _stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# ===================================================================
# Main node
# ===================================================================
class DataCollectorNode(Node):

    def __init__(self):
        super().__init__('data_collector')

        # ---- parameters ----
        self.declare_parameter('collect_duration',  0.0)
        self.declare_parameter('output_dir',
                               os.path.expanduser('~/Mobile_Robot/data'))
        self.declare_parameter('trajectory_type',   'unknown')
        self.declare_parameter('wind_x', 0.0)
        self.declare_parameter('wind_y', 0.0)
        self.declare_parameter('wind_z', 0.0)

        self._duration = self.get_parameter('collect_duration').value
        self._out_root = self.get_parameter('output_dir').value
        self._traj     = self.get_parameter('trajectory_type').value
        self._wind     = (
            self.get_parameter('wind_x').value,
            self.get_parameter('wind_y').value,
            self.get_parameter('wind_z').value,
        )

        # ---- data buffers (list-of-dicts, turned into DataFrames later) ----
        self._state_buf  = []
        self._ref_buf    = []
        self._wrench_buf = []
        self._motor_buf  = []
        self._imu_buf    = []
        self._odom_buf   = []
        self._debug_buf  = []

        self._t0       = None      # first timestamp (seconds, reset at FLYING start)
        self._shutdown_done = False
        self._prev_phase = -1

        # Recording gate: for hover-only experiments start immediately;
        # for trajectory experiments wait for FLYING phase signal so we
        # don't pollute the dataset with hover stabilisation data.
        self._recording = (self._traj == 'hover')
        self._duration_timer = None  # created when FLYING phase begins

        # ---- QoS profiles (must match publishers) ----
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # ---- subscriptions ----
        self.create_subscription(
            Odometry, '/state_estimate', self._state_cb, reliable_qos)
        self.create_subscription(
            Float64MultiArray, '/reference_state', self._ref_cb, reliable_qos)
        self.create_subscription(
            Float64MultiArray, '/control_wrench', self._wrench_cb, reliable_qos)
        self.create_subscription(
            Actuators, '/motor_commands', self._motor_cb, reliable_qos)
        self.create_subscription(
            Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, sensor_qos)
        self.create_subscription(
            Float64MultiArray, '/mpc_debug', self._debug_cb, reliable_qos)
        # Phase signal from trajectory_generator — gates when recording starts/stops
        self.create_subscription(
            Int32, '/trajectory_phase', self._phase_cb, reliable_qos)

        # ---- duration timer for hover-only experiments (started at launch) ----
        # For trajectory experiments the timer is started when FLYING begins.
        if self._recording and self._duration > 0:
            self._duration_timer = self.create_timer(self._duration, self._timer_expired)

        self.get_logger().info(
            f'Data collector ready  traj={self._traj}  '
            f'duration={"∞" if self._duration <= 0 else f"{self._duration:.0f}s"}  '
            f'recording={"immediate (hover)" if self._recording else "waiting for FLYING phase"}')

    # ------------------------------------------------------------------ #
    # time helpers
    # ------------------------------------------------------------------ #
    def _rel(self, t_abs):
        if self._t0 is None:
            self._t0 = t_abs
        return t_abs - self._t0

    def _now_sec(self):
        return _stamp_sec(self.get_clock().now().to_msg())

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def _state_cb(self, msg: Odometry):
        if not self._recording:
            return
        t = _stamp_sec(msg.header.stamp)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        q_arr = np.array([q.x, q.y, q.z, q.w])
        norm = np.linalg.norm(q_arr)
        if norm < 1e-6:
            return
        rot = Rotation.from_quat(q_arr / norm)
        yaw, roll, pitch = rot.as_euler('zxy')

        self._state_buf.append({
            't': t, 't_rel': self._rel(t),
            'x': p.x, 'y': p.y, 'z': p.z,
            'phi': roll, 'theta': pitch, 'psi': yaw,
            'vx': v.x, 'vy': v.y, 'vz': v.z,
            'p': w.x, 'q': w.y, 'r': w.z,
        })

    def _ref_cb(self, msg: Float64MultiArray):
        if not self._recording:
            return
        t = self._now_sec()
        d = list(msg.data)
        if len(d) < 12:
            d.extend([0.0] * (12 - len(d)))
        self._ref_buf.append({
            't': t, 't_rel': self._rel(t),
            'ref_x': d[0], 'ref_y': d[1], 'ref_z': d[2],
            'ref_phi': d[3], 'ref_theta': d[4], 'ref_psi': d[5],
            'ref_vx': d[6], 'ref_vy': d[7], 'ref_vz': d[8],
            'ref_p': d[9], 'ref_q': d[10], 'ref_r': d[11],
        })

    def _wrench_cb(self, msg: Float64MultiArray):
        if not self._recording:
            return
        t = self._now_sec()
        d = list(msg.data)
        if len(d) < 4:
            return
        self._wrench_buf.append({
            't': t, 't_rel': self._rel(t),
            'thrust': d[0], 'tau_phi': d[1],
            'tau_theta': d[2], 'tau_psi': d[3],
        })

    def _motor_cb(self, msg: Actuators):
        if not self._recording:
            return
        t = _stamp_sec(msg.header.stamp)
        v = list(msg.velocity)
        if len(v) < 4:
            return
        self._motor_buf.append({
            't': t, 't_rel': self._rel(t),
            'w0': v[0], 'w1': v[1], 'w2': v[2], 'w3': v[3],
        })

    def _imu_cb(self, msg: Imu):
        if not self._recording:
            return
        t = _stamp_sec(msg.header.stamp)
        a = msg.linear_acceleration
        g = msg.angular_velocity
        self._imu_buf.append({
            't': t, 't_rel': self._rel(t),
            'ax': a.x, 'ay': a.y, 'az': a.z,
            'gx': g.x, 'gy': g.y, 'gz': g.z,
        })

    def _odom_cb(self, msg: Odometry):
        if not self._recording:
            return
        t = _stamp_sec(msg.header.stamp)
        p = msg.pose.pose.position
        self._odom_buf.append({
            't': t, 't_rel': self._rel(t),
            'ox': p.x, 'oy': p.y, 'oz': p.z,
        })

    def _debug_cb(self, msg: Float64MultiArray):
        if not self._recording:
            return
        t = self._now_sec()
        d = list(msg.data)
        if len(d) < 19:
            return
        self._debug_buf.append({
            't': t, 't_rel': self._rel(t),
            # MPC perturbation (delta from hover)
            'ud_T': d[0], 'ud_phi': d[1], 'ud_theta': d[2], 'ud_psi': d[3],
            # Integral contribution
            'ui_T': d[4], 'ui_phi': d[5], 'ui_theta': d[6], 'ui_psi': d[7],
            # Total wrench
            'ut_T': d[8], 'ut_phi': d[9], 'ut_theta': d[10], 'ut_psi': d[11],
            # Diagnostics
            'torque_scale': d[12],
            'cost_state': d[13],
            'cost_input': d[14],
            'solve_ok': d[15],
            # Integral error accumulator
            'int_err_x': d[16], 'int_err_y': d[17], 'int_err_z': d[18],
        })

    # ------------------------------------------------------------------ #
    # trajectory phase callback  (0=PRE_HOVER, 1=FLYING, 2=POST_HOVER)
    # ------------------------------------------------------------------ #
    def _phase_cb(self, msg: Int32):
        phase = msg.data
        if phase == self._prev_phase:
            return
        self._prev_phase = phase

        if phase == 1:  # FLYING — start recording fresh
            # Discard any pre-hover data and reset timestamp origin
            self._state_buf.clear()
            self._ref_buf.clear()
            self._wrench_buf.clear()
            self._motor_buf.clear()
            self._imu_buf.clear()
            self._odom_buf.clear()
            self._debug_buf.clear()
            self._t0 = None
            self._recording = True
            self.get_logger().info('Phase FLYING — recording started')
            # Start optional duration guard (fires if traj_duration=0 / forever)
            if self._duration > 0 and self._duration_timer is None:
                self._duration_timer = self.create_timer(
                    self._duration, self._timer_expired)

        elif phase == 2:  # POST_HOVER — trajectory finished, auto-save
            if not self._recording:
                return
            self._recording = False
            self.get_logger().info('Phase POST_HOVER — trajectory done, saving data …')
            self.on_shutdown()

    # ------------------------------------------------------------------ #
    # duration timer
    # ------------------------------------------------------------------ #
    def _timer_expired(self):
        self.get_logger().info('Collection duration reached — processing …')
        self.on_shutdown()
        raise SystemExit

    # ================================================================== #
    # SHUTDOWN — all processing happens here
    # ================================================================== #
    def on_shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        n_state = len(self._state_buf)
        n_ctrl  = len(self._wrench_buf)
        if n_state < 20 or n_ctrl < 20:
            self.get_logger().warn(
                f'Not enough data (state={n_state}, ctrl={n_ctrl}). Skipping.')
            return

        self.get_logger().info(
            f'Processing {n_state} state / {n_ctrl} control samples …')

        # --- build aligned dataframes ---
        state_df, ctrl_df = self._build_dataframes()
        if state_df is None:
            return

        # --- compute metrics ---
        metrics = self._compute_metrics(state_df, ctrl_df)

        # --- output directory ---
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(self._out_root, f'{self._traj}_{ts}')
        os.makedirs(out_dir, exist_ok=True)

        # --- save ---
        self._save_csv(state_df, ctrl_df, metrics, out_dir)
        self._generate_plots(state_df, ctrl_df, metrics, out_dir)
        self._print_summary(metrics, out_dir)

    # ------------------------------------------------------------------ #
    # DataFrame construction & alignment
    # ------------------------------------------------------------------ #
    def _build_dataframes(self):
        state_df = pd.DataFrame(self._state_buf)
        ref_df   = pd.DataFrame(self._ref_buf)
        wrench_df = pd.DataFrame(self._wrench_buf)
        motor_df  = pd.DataFrame(self._motor_buf)

        # Sort by time
        for df in (state_df, ref_df, wrench_df, motor_df):
            df.sort_values('t', inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Align reference onto state timestamps
        state_df = pd.merge_asof(
            state_df, ref_df.drop(columns=['t_rel']),
            on='t', tolerance=0.025, direction='nearest')

        # Fill any NaN refs (early samples before traj generator starts)
        ref_cols = [c for c in state_df.columns if c.startswith('ref_')]
        state_df[ref_cols] = state_df[ref_cols].ffill().bfill().fillna(0.0)

        # Compute errors
        for ax in ('x', 'y', 'z'):
            state_df[f'err_{ax}'] = state_df[ax] - state_df[f'ref_{ax}']
        for ax in ('phi', 'theta', 'psi'):
            err = state_df[ax] - state_df[f'ref_{ax}']
            state_df[f'err_{ax}'] = (err + np.pi) % (2 * np.pi) - np.pi
        for ax in ('vx', 'vy', 'vz'):
            state_df[f'err_{ax}'] = state_df[ax] - state_df[f'ref_{ax}']
        state_df['err_3d'] = np.sqrt(
            state_df['err_x']**2 + state_df['err_y']**2 +
            state_df['err_z']**2)
        state_df['err_vel_3d'] = np.sqrt(
            state_df['err_vx']**2 + state_df['err_vy']**2 +
            state_df['err_vz']**2)

        # Recompute t_rel from earliest sample
        t_min = state_df['t'].iloc[0]
        state_df['t_rel'] = state_df['t'] - t_min

        # Control df — merge wrench + motors + debug
        ctrl_df = pd.merge_asof(
            wrench_df, motor_df.drop(columns=['t_rel']),
            on='t', tolerance=0.025, direction='nearest')
        motor_cols = ['w0', 'w1', 'w2', 'w3']
        ctrl_df[motor_cols] = ctrl_df[motor_cols].ffill().bfill().fillna(0.0)

        # Merge MPC debug if available
        if len(self._debug_buf) > 10:
            dbg_df = pd.DataFrame(self._debug_buf)
            dbg_df.sort_values('t', inplace=True)
            dbg_df.reset_index(drop=True, inplace=True)
            ctrl_df = pd.merge_asof(
                ctrl_df, dbg_df.drop(columns=['t_rel']),
                on='t', tolerance=0.025, direction='nearest')
            dbg_cols = [c for c in ctrl_df.columns
                        if c.startswith(('ud_', 'ui_', 'ut_', 'int_err_',
                                         'torque_scale', 'cost_', 'solve_'))]
            ctrl_df[dbg_cols] = ctrl_df[dbg_cols].ffill().bfill().fillna(0.0)

        ctrl_df['t_rel'] = ctrl_df['t'] - t_min

        return state_df, ctrl_df

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _compute_metrics(self, sdf, cdf):
        m = {}
        dur = sdf['t_rel'].iloc[-1] - sdf['t_rel'].iloc[0]
        m['duration_s'] = dur
        m['n_state_samples'] = len(sdf)
        m['n_ctrl_samples']  = len(cdf)
        m['trajectory_type'] = self._traj
        m['wind_x'] = self._wind[0]
        m['wind_y'] = self._wind[1]
        m['wind_z'] = self._wind[2]
        m['is_wind'] = any(abs(w) > 0.01 for w in self._wind)

        # -- position RMSE --
        for ax in ('x', 'y', 'z'):
            m[f'rmse_{ax}'] = np.sqrt(np.mean(sdf[f'err_{ax}']**2))
            m[f'max_err_{ax}'] = np.max(np.abs(sdf[f'err_{ax}']))
        m['rmse_3d']    = np.sqrt(np.mean(sdf['err_3d']**2))
        m['max_err_3d'] = np.max(sdf['err_3d'])

        # -- velocity RMSE --
        for ax in ('vx', 'vy', 'vz'):
            m[f'rmse_{ax}'] = np.sqrt(np.mean(sdf[f'err_{ax}']**2))
        m['rmse_vel_3d'] = np.sqrt(np.mean(sdf['err_vel_3d']**2))

        # -- settling time (2 % criterion on z, 2 s window) --
        m['settling_time_s'] = self._settling_time(sdf)

        # -- steady-state error (last 20 %) --
        n20 = max(1, int(0.8 * len(sdf)))
        tail = sdf.iloc[n20:]
        for ax in ('x', 'y', 'z'):
            m[f'ss_err_{ax}'] = np.mean(np.abs(tail[f'err_{ax}']))
        m['ss_err_3d'] = np.mean(tail['err_3d'])

        # -- thrust --
        if len(cdf) > 0:
            m['mean_thrust']  = np.mean(cdf['thrust'])
            m['max_thrust']   = np.max(cdf['thrust'])
            m['min_thrust']   = np.min(cdf['thrust'])
            m['thrust_dev']   = np.mean(np.abs(cdf['thrust'] - HOVER_THRUST))

            for ax in ('tau_phi', 'tau_theta', 'tau_psi'):
                m[f'rms_{ax}'] = np.sqrt(np.mean(cdf[ax]**2))
                m[f'max_{ax}'] = np.max(np.abs(cdf[ax]))

            # motor utilisation
            ws = cdf[['w0', 'w1', 'w2', 'w3']].values
            util = ws / MAX_ROTOR_SPEED * 100.0
            m['mean_motor_util'] = np.mean(util)
            m['max_motor_util']  = np.max(util)
            for i in range(4):
                m[f'mean_util_r{i}'] = np.mean(util[:, i])

            sat_mask = np.any(ws >= 0.99 * MAX_ROTOR_SPEED, axis=1)
            m['sat_events'] = int(np.sum(sat_mask))
            m['sat_pct']    = np.sum(sat_mask) / len(cdf) * 100.0

            # control energy  ∫‖u‖² dt
            t_ctrl = cdf['t_rel'].values
            u_sq = (cdf['thrust']**2 + cdf['tau_phi']**2 +
                    cdf['tau_theta']**2 + cdf['tau_psi']**2).values
            m['ctrl_energy'] = float(np.trapz(u_sq, t_ctrl))

        # -- linearisation validity --
        m['max_phi_deg']   = np.degrees(np.max(np.abs(sdf['phi'])))
        m['max_theta_deg'] = np.degrees(np.max(np.abs(sdf['theta'])))
        within = ((np.abs(sdf['phi']) < LIN_LIMIT_RAD) &
                  (np.abs(sdf['theta']) < LIN_LIMIT_RAD))
        m['pct_within_linear'] = np.mean(within) * 100.0
        m['linearization_valid'] = bool(
            m['max_phi_deg'] < LIN_LIMIT_DEG and
            m['max_theta_deg'] < LIN_LIMIT_DEG)

        # Signed steady-state offset (mean, not absolute) — always computed
        # useful for wind analysis: integral action should drive this to ~0
        m['ss_offset_x'] = float(np.mean(tail['err_x']))
        m['ss_offset_y'] = float(np.mean(tail['err_y']))
        m['ss_offset_z'] = float(np.mean(tail['err_z']))
        # heuristic flag: significant residual drift observed
        m['wind_detected'] = (abs(m['ss_offset_x']) > 0.05 or
                              abs(m['ss_offset_y']) > 0.05)

        # -- MPC internals (only if debug data merged) --
        # solve_ok=1 every tick the closed-form gain was applied;
        # solve_ok=0 only when state was NaN/Inf (control skipped entirely).
        has_debug = 'solve_ok' in cdf.columns
        m['has_mpc_debug'] = has_debug
        if has_debug:
            # Solve / skip count
            n_total = len(cdf)
            n_ok = int(cdf['solve_ok'].sum())
            m['mpc_solve_ok']   = n_ok
            m['mpc_solve_fail'] = n_total - n_ok
            m['mpc_solve_pct']  = n_ok / n_total * 100.0
            # Note: this MPC uses a closed-form solution (K_r, K_x precomputed).
            # It cannot fail to converge; solve_fail counts corrupted-state skips.

            # Cost decomposition
            m['cost_state_mean'] = float(cdf['cost_state'].mean())
            m['cost_state_max']  = float(cdf['cost_state'].max())
            m['cost_input_mean'] = float(cdf['cost_input'].mean())
            m['cost_input_max']  = float(cdf['cost_input'].max())
            m['cost_total_mean'] = float(
                (cdf['cost_state'] + cdf['cost_input']).mean())

            # Torque scale (allocation saturation)
            ts = cdf['torque_scale']
            m['torque_scale_mean'] = float(ts.mean())
            m['torque_scale_min']  = float(ts.min())
            n_scaled = int((ts < 0.999).sum())
            m['torque_scaled_pct'] = n_scaled / n_total * 100.0

            # Wrench decomposition: MPC delta vs integral vs hover
            m['ud_thrust_rms'] = float(np.sqrt(np.mean(cdf['ud_T']**2)))
            m['ui_thrust_rms'] = float(np.sqrt(np.mean(cdf['ui_T']**2)))
            for ax in ('phi', 'theta', 'psi'):
                m[f'ud_tau_{ax}_rms'] = float(
                    np.sqrt(np.mean(cdf[f'ud_{ax}']**2)))
                m[f'ui_tau_{ax}_rms'] = float(
                    np.sqrt(np.mean(cdf[f'ui_{ax}']**2)))

            # Integral error accumulator magnitude
            int_mag = np.sqrt(cdf['int_err_x']**2 + cdf['int_err_y']**2 +
                              cdf['int_err_z']**2)
            m['int_err_mag_final'] = float(int_mag.iloc[-1])
            m['int_err_mag_max']   = float(int_mag.max())

        return m

    @staticmethod
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

    # ------------------------------------------------------------------ #
    # CSV output
    # ------------------------------------------------------------------ #
    def _save_csv(self, sdf, cdf, metrics, out):
        sdf.to_csv(os.path.join(out, 'state_data.csv'), index=False,
                    float_format='%.6f')
        cdf.to_csv(os.path.join(out, 'control_data.csv'), index=False,
                    float_format='%.6f')
        rows = [{'metric': k, 'value': v} for k, v in metrics.items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(out, 'metrics.csv'), index=False)
        self.get_logger().info(f'CSV files saved to {out}')

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    def _generate_plots(self, sdf, cdf, met, out):
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass

        traj = met['trajectory_type']
        self._plot_3d(sdf, traj, out)
        self._plot_pos_tracking(sdf, traj, out)
        self._plot_vel_tracking(sdf, traj, out)
        self._plot_pos_error(sdf, met, traj, out)
        self._plot_wrench(cdf, traj, out)
        self._plot_motors(cdf, traj, out)
        self._plot_euler(sdf, traj, out)
        self._plot_motor_hist(cdf, met, traj, out)
        if met.get('has_mpc_debug', False):
            self._plot_mpc_cost(cdf, traj, out)
            self._plot_wrench_decomp(cdf, traj, out)
            self._plot_torque_scale(cdf, traj, out)
            self._plot_integral_err(cdf, traj, out)
        self.get_logger().info(f'Plots saved to {out}')

    # -- 1. 3-D trajectory --
    def _plot_3d(self, sdf, traj, out):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(sdf['x'], sdf['y'], sdf['z'],
                'b-', lw=1.2, label='Actual')
        ax.plot(sdf['ref_x'], sdf['ref_y'], sdf['ref_z'],
                'r--', lw=1.2, label='Reference')
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

    # -- 2. position tracking --
    def _plot_pos_tracking(self, sdf, traj, out):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        t = sdf['t_rel']
        for ax, lab, c, rc in zip(axes,
                                   ['X', 'Y', 'Z'],
                                   ['x', 'y', 'z'],
                                   ['ref_x', 'ref_y', 'ref_z']):
            ax.plot(t, sdf[c], 'b-', lw=1, label='Actual')
            ax.plot(t, sdf[rc], 'r--', lw=1, label='Reference')
            ax.set_ylabel(f'{lab} [m]')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{traj} — Position Tracking', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, 'position_tracking.png'), dpi=150)
        plt.close(fig)

    # -- 3. velocity tracking (world frame) --
    def _plot_vel_tracking(self, sdf, traj, out):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        t = sdf['t_rel']
        for ax, lab, c, rc in zip(axes,
                                   ['Vx', 'Vy', 'Vz'],
                                   ['vx', 'vy', 'vz'],
                                   ['ref_vx', 'ref_vy', 'ref_vz']):
            ax.plot(t, sdf[c], 'b-', lw=0.9, label='Actual')
            ax.plot(t, sdf[rc], 'r--', lw=0.9, label='Reference')
            ax.set_ylabel(f'{lab} [m/s]')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{traj} — Velocity Tracking (World Frame)', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, 'velocity_tracking.png'), dpi=150)
        plt.close(fig)

    # -- 4. position error --
    def _plot_pos_error(self, sdf, met, traj, out):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        t = sdf['t_rel']
        labels = ['X error', 'Y error', 'Z error', '3-D error']
        cols   = ['err_x', 'err_y', 'err_z', 'err_3d']
        rmses  = [met['rmse_x'], met['rmse_y'], met['rmse_z'], met['rmse_3d']]
        for ax, lab, col, rmse in zip(axes, labels, cols, rmses):
            ax.plot(t, sdf[col], 'b-', lw=0.8)
            ax.axhline(rmse, color='orange', ls='--', lw=1,
                        label=f'RMSE = {rmse:.4f} m')
            if col != 'err_3d':
                ax.axhline(-rmse, color='orange', ls='--', lw=1)
            # shade last 20 %
            t_ss = t.iloc[int(0.8 * len(t))]
            ax.axvspan(t_ss, t.iloc[-1], alpha=0.08, color='green',
                        label='Steady-state region')
            ax.set_ylabel(f'{lab} [m]')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{traj} — Position Error', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, 'position_error.png'), dpi=150)
        plt.close(fig)

    # -- 4. control wrench --
    def _plot_wrench(self, cdf, traj, out):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        t = cdf['t_rel']
        labels  = ['Thrust T [N]', 'τ_φ (roll) [N·m]',
                    'τ_θ (pitch) [N·m]', 'τ_ψ (yaw) [N·m]']
        cols    = ['thrust', 'tau_phi', 'tau_theta', 'tau_psi']
        for ax, lab, col in zip(axes, labels, cols):
            ax.plot(t, cdf[col], 'b-', lw=0.8)
            ax.set_ylabel(lab)
            ax.grid(True, alpha=0.3)
        # hover thrust reference on thrust plot
        axes[0].axhline(HOVER_THRUST, color='gray', ls=':', lw=1,
                         label=f'Hover mg = {HOVER_THRUST:.2f} N')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{traj} — Control Wrench (MPC Output)', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, 'control_wrench.png'), dpi=150)
        plt.close(fig)

    # -- 5. motor speeds --
    def _plot_motors(self, cdf, traj, out):
        fig, ax = plt.subplots(figsize=(12, 5))
        t = cdf['t_rel']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i, c in enumerate(colors):
            ax.plot(t, cdf[f'w{i}'], color=c, lw=0.8, label=f'Rotor {i}')
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

    # -- 6. Euler angles --
    def _plot_euler(self, sdf, traj, out):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        t = sdf['t_rel']
        labels = ['φ (roll)', 'θ (pitch)', 'ψ (yaw)']
        cols   = ['phi', 'theta', 'psi']
        refs   = ['ref_phi', 'ref_theta', 'ref_psi']
        for ax, lab, col, rc in zip(axes, labels, cols, refs):
            ax.plot(t, np.degrees(sdf[col]), 'b-', lw=0.8, label='Actual')
            ax.plot(t, np.degrees(sdf[rc]), 'r--', lw=0.8, label='Reference')
            # linearisation band (roll/pitch only)
            if col in ('phi', 'theta'):
                ax.axhline( LIN_LIMIT_DEG, color='orange', ls=':', lw=1)
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

    # -- 7. motor utilisation histogram --
    def _plot_motor_hist(self, cdf, met, traj, out):
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

    # -- 8. MPC cost decomposition --
    def _plot_mpc_cost(self, cdf, traj, out):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        t = cdf['t_rel']
        cost_total = cdf['cost_state'] + cdf['cost_input']

        axes[0].plot(t, cdf['cost_state'], 'b-', lw=0.8, label='State cost  e\'Qe')
        axes[0].set_ylabel('State Cost')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, cdf['cost_input'], 'r-', lw=0.8, label='Input cost  u\'Ru')
        axes[1].set_ylabel('Input Cost')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].fill_between(t, 0, cdf['cost_state'], alpha=0.4,
                              color='tab:blue', label='State cost')
        axes[2].fill_between(t, cdf['cost_state'], cost_total, alpha=0.4,
                              color='tab:red', label='Input cost')
        axes[2].set_ylabel('Total Cost')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend(loc='upper right', fontsize=8)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{traj} — MPC Cost Decomposition', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out, 'mpc_cost.png'), dpi=150)
        plt.close(fig)

    # -- 9. wrench decomposition (hover + MPC delta + integral) --
    def _plot_wrench_decomp(self, cdf, traj, out):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        t = cdf['t_rel']
        labels = ['Thrust [N]', 'τ_φ [N·m]', 'τ_θ [N·m]', 'τ_ψ [N·m]']
        delta_cols  = ['ud_T', 'ud_phi', 'ud_theta', 'ud_psi']
        integ_cols  = ['ui_T', 'ui_phi', 'ui_theta', 'ui_psi']
        total_cols  = ['ut_T', 'ut_phi', 'ut_theta', 'ut_psi']
        hover_vals  = [HOVER_THRUST, 0.0, 0.0, 0.0]

        for ax, lab, dc, ic, tc, hv in zip(
                axes, labels, delta_cols, integ_cols, total_cols, hover_vals):
            ax.plot(t, cdf[tc], 'k-', lw=1.0, label='Total', alpha=0.9)
            ax.plot(t, cdf[dc], 'b-', lw=0.7, label='MPC Δ', alpha=0.7)
            ax.plot(t, cdf[ic], 'g-', lw=0.7, label='Integral', alpha=0.7)
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

    # -- 10. torque scale timeline --
    def _plot_torque_scale(self, cdf, traj, out):
        fig, ax = plt.subplots(figsize=(12, 4))
        t = cdf['t_rel']
        ts = cdf['torque_scale']
        ax.plot(t, ts, 'b-', lw=0.8)
        ax.fill_between(t, ts, 1.0, where=(ts < 0.999),
                         color='red', alpha=0.3, label='Torque scaled')
        ax.axhline(1.0, color='gray', ls=':', lw=1, label='No saturation')
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque Scale Factor')
        n_scaled = (ts < 0.999).sum()
        ax.set_title(f'{traj} — Motor Allocation Torque Scaling  '
                     f'({n_scaled} events, '
                     f'min = {ts.min():.3f})')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'torque_scale.png'), dpi=150)
        plt.close(fig)

    # -- 11. integral error accumulator --
    def _plot_integral_err(self, cdf, traj, out):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = cdf['t_rel']

        for ax_lab, col, color in [('X', 'int_err_x', 'tab:blue'),
                                    ('Y', 'int_err_y', 'tab:orange'),
                                    ('Z', 'int_err_z', 'tab:green')]:
            axes[0].plot(t, cdf[col], color=color, lw=0.8,
                          label=f'{ax_lab}')
        axes[0].set_ylabel('Integral Error [m·s]')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].set_title('Integral Error Accumulator (per axis)')
        axes[0].grid(True, alpha=0.3)

        int_mag = np.sqrt(cdf['int_err_x']**2 + cdf['int_err_y']**2 +
                          cdf['int_err_z']**2)
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

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    def _print_summary(self, m, out):
        sep = '=' * 76
        wx, wy, wz = m['wind_x'], m['wind_y'], m['wind_z']
        wind_str = (f'Wind [{wx:.1f}, {wy:.1f}, {wz:.1f}] m/s'
                    if m['is_wind'] else 'No wind')
        print(f'\n{sep}')
        print(f'{"MPC DATA COLLECTION REPORT":^76}')
        print(f'{"Trajectory: " + m["trajectory_type"] + "   |   " + wind_str:^76}')
        print(f'{"Duration: %.1fs | State samples: %d | Ctrl samples: %d" % (m["duration_s"], m["n_state_samples"], m["n_ctrl_samples"]):^76}')
        print(sep)

        print('\n  TRACKING PERFORMANCE')
        print('  ' + '-' * 40)
        print(f'  Position RMSE:     X = {m["rmse_x"]:.4f} m'
              f'    Y = {m["rmse_y"]:.4f} m'
              f'    Z = {m["rmse_z"]:.4f} m'
              f'    3D = {m["rmse_3d"]:.4f} m')
        print(f'  Max Pos Error:     X = {m["max_err_x"]:.4f} m'
              f'    Y = {m["max_err_y"]:.4f} m'
              f'    Z = {m["max_err_z"]:.4f} m'
              f'    3D = {m["max_err_3d"]:.4f} m')
        print(f'  Velocity RMSE:     X = {m["rmse_vx"]:.4f} m/s'
              f'  Y = {m["rmse_vy"]:.4f} m/s'
              f'  Z = {m["rmse_vz"]:.4f} m/s'
              f'  3D = {m["rmse_vel_3d"]:.4f} m/s')
        st = m['settling_time_s']
        print(f'  Settling Time:     {"%.2f s" % st if not np.isnan(st) else "N/A"}'
              f'  (2% of z=1.0 m, 2 s window)')
        print(f'  Steady-State Err:  X = {m["ss_err_x"]:.4f} m'
              f'    Y = {m["ss_err_y"]:.4f} m'
              f'    Z = {m["ss_err_z"]:.4f} m'
              f'    3D = {m["ss_err_3d"]:.4f} m')

        print('\n  CONTROL EFFORT')
        print('  ' + '-' * 40)
        print(f'  Mean Thrust:       {m.get("mean_thrust", 0):.2f} N'
              f'  (hover = {HOVER_THRUST:.3f} N,'
              f' deviation = {m.get("thrust_dev", 0):.3f} N)')
        print(f'  Thrust Range:      [{m.get("min_thrust", 0):.2f},'
              f' {m.get("max_thrust", 0):.2f}] N')
        print(f'  RMS Torques:       τ_φ = {m.get("rms_tau_phi", 0):.4f}'
              f'   τ_θ = {m.get("rms_tau_theta", 0):.4f}'
              f'   τ_ψ = {m.get("rms_tau_psi", 0):.4f} N·m')
        print(f'  Max Torques:       τ_φ = {m.get("max_tau_phi", 0):.4f}'
              f'   τ_θ = {m.get("max_tau_theta", 0):.4f}'
              f'   τ_ψ = {m.get("max_tau_psi", 0):.4f} N·m')
        print(f'  Motor Utilisation: mean = {m.get("mean_motor_util", 0):.1f}%'
              f'   max = {m.get("max_motor_util", 0):.1f}%')
        print(f'  Saturation Events: {m.get("sat_events", 0)}'
              f'  ({m.get("sat_pct", 0):.1f}%)')
        print(f'  Control Energy:    ∫‖u‖² dt = {m.get("ctrl_energy", 0):.1f}')

        print('\n  MPC LINEARISATION VALIDITY')
        print('  ' + '-' * 40)
        print(f'  Max |φ|:  {m["max_phi_deg"]:.1f}°'
              f'      Max |θ|:  {m["max_theta_deg"]:.1f}°')
        print(f'  Within ±{LIN_LIMIT_DEG:.0f}° bound: '
              f'{m["pct_within_linear"]:.1f}% of flight')
        valid = 'VALID' if m['linearization_valid'] else 'VIOLATED'
        print(f'  Linearisation:     {valid}')

        if m.get('has_mpc_debug', False):
            print('\n  MPC INTERNALS')
            print('  ' + '-' * 40)
            print(f'  Solver type:       Closed-form (K_r·Xref + K_x·x0)')
            print(f'  Solve Success:     {m["mpc_solve_ok"]}/{m["mpc_solve_ok"] + m["mpc_solve_fail"]}'
                  f'  ({m["mpc_solve_pct"]:.1f}%)'
                  f'  [skip=NaN/Inf state]')
            print(f'  Cost (state):      mean = {m["cost_state_mean"]:.4f}'
                  f'   max = {m["cost_state_max"]:.4f}')
            print(f'  Cost (input):      mean = {m["cost_input_mean"]:.4f}'
                  f'   max = {m["cost_input_max"]:.4f}')
            print(f'  Cost (total):      mean = {m["cost_total_mean"]:.4f}')
            print(f'  Torque Scale:      mean = {m["torque_scale_mean"]:.3f}'
                  f'   min = {m["torque_scale_min"]:.3f}'
                  f'   scaled = {m["torque_scaled_pct"]:.1f}%')
            print(f'  Wrench RMS (MPC Δ):  T = {m["ud_thrust_rms"]:.3f} N'
                  f'   τ_φ = {m["ud_tau_phi_rms"]:.4f}'
                  f'   τ_θ = {m["ud_tau_theta_rms"]:.4f}'
                  f'   τ_ψ = {m["ud_tau_psi_rms"]:.4f} N·m')
            print(f'  Wrench RMS (Integ):  T = {m["ui_thrust_rms"]:.3f} N'
                  f'   τ_φ = {m["ui_tau_phi_rms"]:.4f}'
                  f'   τ_θ = {m["ui_tau_theta_rms"]:.4f}'
                  f'   τ_ψ = {m["ui_tau_psi_rms"]:.4f} N·m')
            print(f'  Integral |e|:      final = {m["int_err_mag_final"]:.4f}'
                  f'   max = {m["int_err_mag_max"]:.4f} m·s')

        if m.get('is_wind', False):
            print('\n  WIND REJECTION')
            print('  ' + '-' * 40)
            print(f'  Applied wind:  [{m["wind_x"]:.1f}, {m["wind_y"]:.1f},'
                  f' {m["wind_z"]:.1f}] m/s')
            print(f'  SS mean offset: X = {m["ss_offset_x"]:+.4f} m'
                  f'   Y = {m["ss_offset_y"]:+.4f} m'
                  f'   Z = {m["ss_offset_z"]:+.4f} m')
            rejected = not m['wind_detected']
            print(f'  Drift suppressed: {"YES (integral action effective)" if rejected else "NO (residual drift > 5 cm)"}')

        print(f'\n  OUTPUT  →  {out}/')
        print(sep + '\n')


# ===================================================================
# Entry point
# ===================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.on_shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

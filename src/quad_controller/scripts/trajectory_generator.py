#!/usr/bin/env python3
"""
Trajectory Generator for Quadrotor — Lab 2
============================================
Publishes a 12-state reference to the MPC at a fixed rate.

Flight sequence (per lab spec)
-------------------------------
  Phase 0 — PRE_HOVER   : hold hover for `hover_time` seconds so the drone
                           stabilises before the trajectory begins.
  Phase 1 — FLYING      : execute the configured trajectory for `traj_duration`
                           seconds (0 = fly indefinitely).
  Phase 2 — POST_HOVER  : return to and hold the initial hover position.

If trajectory_type == 'hover' the generator stays in PRE_HOVER forever.

2D trajectories  (traj_plane = 'xz' or 'xy')
----------------------------------------------
  The 'traj_plane' parameter selects the plane for all 2D trajectories:
    'xz'  (default)  forward = +x, lateral = Δz  (altitude change)
    'xy'             forward = +x, lateral = +y   (horizontal turn)

  straight_2d   Constant-speed line with optional trapezoidal acceleration
  sine_2d       Forward at constant speed, lateral oscillates as a sine wave
  step_2d       Staircase: discrete lateral jumps at fixed forward intervals
  lemniscate_2d Figure-8 (lemniscate of Bernoulli) — best MPC showcase
                  Tests both positive/negative motion on coupled axes
  circle_2d     Full circle — constant centripetal acceleration test

Part 3  (3-D)
--------------
  straight_3d   Trapezoidal-velocity line along direction [1, 0.5, 0.3]
  helix         Helical spiral: circles in x-y while climbing in z
  figure8_3d    Horizontal figure-8 (lemniscate) at constant z
  cone_helix    Conical spiral: expanding-radius helix forming a cone
  lissajous_3d  3D Lissajous (freq ratio 1:2:3) — strongest MPC showcase
                  All three axes move at different frequencies simultaneously

hover (default)
  The drone stays at the configured hover position.

Parameters (all set in config/params.yaml or via CLI)
--------------------
  trajectory_type   : string  — see names above          (default: hover)
  traj_plane        : string  — 'xz' or 'xy'             (default: xz)
  publish_rate      : float   — Hz                        (default 50)
  hover_time        : float   — seconds before trajectory starts  (default 5.0)
  hover_z           : float   — hover height [m]          (default 1.0)
  traj_speed        : float   — translational speed [m/s] (default 0.5)
  traj_accel        : float   — accel for trapezoidal ramp [m/s²], 0 = instant
                                (default 0.5)
  traj_amplitude    : float   — amplitude of oscillation [m] (default 0.3)
  traj_frequency    : float   — oscillation / orbit frequency [Hz] (default 0.2)
  traj_radius       : float   — radius for circular/lemniscate [m] (default 0.5)
  traj_climb_rate   : float   — z climb rate for helix [m/s]  (default 0.15)
  traj_yaw_rate     : float   — desired yaw rate [rad/s]      (default 0.0)
  traj_duration     : float   — seconds to fly trajectory, 0 = forever (default 0)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64MultiArray, Int32
from geometry_msgs.msg import PoseStamped


N_STATES = 12   # state dimension [x,y,z,φ,θ,ψ,ẋ,ẏ,ż,p,q,r]

# ---------------------------------------------------------------------------
# Trajectory catalogue
# ---------------------------------------------------------------------------

class TrajectoryBase:
    """Common interface for all trajectory types."""

    def __init__(self, params: dict):
        self.p = params          # dict with all parameters
        self.t0 = None           # time (seconds) when trajectory started

    def start(self, t_now: float):
        self.t0 = t_now

    def reference(self, t_now: float) -> np.ndarray:
        """Return 12-state reference [x,y,z,φ,θ,ψ,ẋ,ẏ,ż,p,q,r] at time t."""
        raise NotImplementedError

    def _elapsed(self, t_now: float) -> float:
        if self.t0 is None:
            self.t0 = t_now
        return t_now - self.t0

    def _plane_ref(self, fwd: float, lat: float,
                   vfwd: float = 0.0, vlat: float = 0.0) -> np.ndarray:
        """
        Build a 12-state reference from forward/lateral coords in the
        configured plane.

          'xz'  (default): fwd→x, lat→Δz (altitude offset),  y=0
          'xy'           : fwd→x, lat→y  (horizontal motion), z=hover_z

        Roll, pitch, yaw and angular rates are all left at zero — the MPC
        controls attitude internally to achieve the demanded position/velocity.
        """
        plane = self.p.get('traj_plane', 'xz')
        z0    = self.p['hover_z']
        ref   = np.zeros(12)
        if plane == 'xy':
            ref[0], ref[1], ref[2] = fwd, lat, z0
            ref[6], ref[7], ref[8] = vfwd, vlat, 0.0
        else:  # xz  (the README default — drone flies in +x while z changes)
            ref[0], ref[1], ref[2] = fwd, 0.0, z0 + lat
            ref[6], ref[7], ref[8] = vfwd, 0.0, vlat
        return ref

    @staticmethod
    def _trapezoid(dt: float, v_max: float, a: float):
        """
        Trapezoidal velocity profile: ramp up at acceleration `a` m/s² to
        `v_max`, then cruise.  Returns (position, velocity) at elapsed time dt.
        If a <= 0 use constant speed (no ramp).
        """
        if a <= 0:
            return v_max * dt, v_max
        t_ramp = v_max / a
        if dt < t_ramp:
            return 0.5 * a * dt ** 2, a * dt
        else:
            x_ramp = 0.5 * a * t_ramp ** 2
            return x_ramp + v_max * (dt - t_ramp), v_max

    @staticmethod
    def _hann_ramp(dt: float, t_ramp: float):
        """
        Hann-window amplitude envelope for oscillating trajectories.

        Returns (ramp, ramp_dot) where:
          ramp(0)      = 0   →  amplitude starts at zero  → v(0) = 0
          ramp(t_ramp) = 1   →  full amplitude reached
          ramp(t>t_ramp) = 1 →  steady oscillation
          ramp_dot = 0 at both endpoints → no jerk discontinuity

        Usage (example for A·sin(ωt)):
          ramp, ramp_dot = self._hann_ramp(dt, 1.0/f)
          lat  = A * ramp * sin(ωt)
          vlat = A * ramp_dot * sin(ωt) + A * ramp * ω * cos(ωt)
        """
        if t_ramp <= 0.0 or dt >= t_ramp:
            return 1.0, 0.0
        s        = math.pi * dt / t_ramp
        ramp     = 0.5 * (1.0 - math.cos(s))
        ramp_dot = 0.5 * math.pi / t_ramp * math.sin(s)
        return ramp, ramp_dot


# ---------------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------------

class HoverTrajectory(TrajectoryBase):
    """Stay at a fixed hover position."""

    def reference(self, t_now):
        ref = np.zeros(12)
        ref[0] = self.p.get('hover_x', 0.0)
        ref[1] = self.p.get('hover_y', 0.0)
        ref[2] = self.p['hover_z']
        return ref


# ---------------------------------------------------------------------------
# Part 2 — 2D trajectories  (plane selected by traj_plane)
# ---------------------------------------------------------------------------

class StraightLine2D(TrajectoryBase):
    """
    Straight line in the configured plane (+x = forward direction).

    Trapezoidal velocity profile (traj_accel > 0):
      - Ramps from 0 to traj_speed at traj_accel m/s²
      - Then cruises at traj_speed
      - Provides smooth acceleration feedforward to the MPC

    If traj_accel = 0: constant speed from start (legacy behaviour).

    MPC showcase value:
      The velocity feedforward allows the MPC to track the exact cruise
      speed without steady-state lag, unlike a pure error-based controller.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        v_max = self.p['traj_speed']
        a     = self.p.get('traj_accel', 0.5)

        pos, vel = self._trapezoid(dt, v_max, a)
        return self._plane_ref(pos, 0.0, vel, 0.0)


class SineWave2D(TrajectoryBase):
    """
    Forward at constant speed; lateral axis oscillates as a sine wave.

    traj_plane='xz' : drone advances in +x while altitude oscillates
    traj_plane='xy' : drone advances in +x while y-position slaloms

    Smooth startup: oscillation amplitude is ramped using a Hann window
    over the first full period (1/f seconds).

      Without ramp: vlat(0) = A·ω ≠ 0  →  velocity step at trajectory start
      With ramp:    vlat(0) = 0         →  smooth entry from hover

    Full velocity feedforward (including the ramp derivative) is provided
    so the MPC can feed-forward both the oscillation and the slow amplitude
    envelope without any lag.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        v     = self.p['traj_speed']
        A     = self.p['traj_amplitude']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)

        lat  = A * ramp * math.sin(omega * dt)
        vlat = (A * ramp_dot * math.sin(omega * dt) +
                A * ramp    * omega * math.cos(omega * dt))

        return self._plane_ref(v * dt, lat, v, vlat)


class StaircaseStep2D(TrajectoryBase):
    """
    Staircase: advance continuously in the forward direction; make discrete
    lateral jumps every 1/traj_frequency seconds.

    traj_plane='xz': altitude staircase (climb traj_amplitude/2 each step)
    traj_plane='xy': y-position staircase (horizontal step changes)

    The discrete steps (zero velocity feedforward on the lateral axis)
    act as step-response inputs — a good test of settling time and overshoot
    for any controller.
    """

    def reference(self, t_now):
        dt = self._elapsed(t_now)
        v  = self.p['traj_speed']
        A  = self.p['traj_amplitude']  # lateral shift per step [m]
        f  = self.p['traj_frequency']  # step frequency [Hz]

        step = math.floor(dt * f)
        lat  = A * step * 0.5
        lat  = lat % (2.0 * A)         # cycle back after 4 steps

        # No velocity feedforward on the lateral axis (it's a step input)
        return self._plane_ref(v * dt, lat, v, 0.0)


class Lemniscate2D(TrajectoryBase):
    """
    Figure-8 (lemniscate of Bernoulli) in the configured plane.

    Parametric form (starts and returns to origin at t=0, t=1/f, …):
      fwd(t) = R · sin(ωt)
      lat(t) = (R/2) · sin(2ωt)

    Why this is the best 2D MPC showcase:
    - Tests both positive AND negative motion on both axes
    - Requires look-ahead at the crossing point (velocity reversal near origin)
    - PID overshoots at the direction reversals; MPC anticipates and brakes early
    - The 2:1 frequency coupling challenges pure error-based controllers
    - Continuous and differentiable → full velocity feedforward available

    Parameters: traj_radius (R), traj_frequency (f), traj_plane
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp

        fwd  = R_eff * math.sin(omega * dt)
        lat  = (R_eff / 2.0) * math.sin(2.0 * omega * dt)
        vfwd = (R * ramp_dot * math.sin(omega * dt) +
                R_eff * omega * math.cos(omega * dt))
        vlat = (R * ramp_dot / 2.0 * math.sin(2.0 * omega * dt) +
                R_eff * omega * math.cos(2.0 * omega * dt))

        return self._plane_ref(fwd, lat, vfwd, vlat)


class Circle2D(TrajectoryBase):
    """
    Full circle in the configured plane.

    Starts at origin, sweeps a radius-R circle:
      fwd(t) = R_eff · sin(ωt)
      lat(t) = R_eff · (1 − cos(ωt))

    Smooth startup: radius ramps from 0 to R via Hann window (one revolution).
    Without ramp vfwd(0) = R·ω ≠ 0; with ramp all initial velocities = 0.

    Why this is a good MPC showcase:
    - Requires sustained centripetal acceleration on both axes simultaneously
    - In xy-plane with wind: shows integral action holding the orbit
    - In xz-plane: tests combined altitude + lateral control at constant load

    Parameters: traj_radius (R), traj_frequency (f), traj_plane
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp

        fwd  = R_eff * math.sin(omega * dt)
        lat  = R_eff * (1.0 - math.cos(omega * dt))
        vfwd = (R * ramp_dot * math.sin(omega * dt) +
                R_eff * omega * math.cos(omega * dt))
        vlat = (R * ramp_dot * (1.0 - math.cos(omega * dt)) +
                R_eff * omega * math.sin(omega * dt))

        return self._plane_ref(fwd, lat, vfwd, vlat)


# ---------------------------------------------------------------------------
# Part 3 — 3D trajectories
# ---------------------------------------------------------------------------

class StraightLine3D(TrajectoryBase):
    """
    Move along direction [1, 0.5, 0.3] (normalised) with trapezoidal speed.

    Uses the same traj_accel ramp as StraightLine2D.
    Demonstrates simultaneous x, y, z motion with smooth acceleration.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        v_max = self.p['traj_speed']
        a     = self.p.get('traj_accel', 0.5)
        z0    = self.p['hover_z']

        d = np.array([1.0, 0.5, 0.3])
        d = d / np.linalg.norm(d)

        dist, speed = self._trapezoid(dt, v_max, a)

        ref = np.zeros(12)
        ref[0:3] = np.array([0.0, 0.0, z0]) + d * dist
        ref[6:9] = d * speed
        return ref


class HelixTrajectory(TrajectoryBase):
    """
    Helix spiral: circle in x-y while climbing in z.

      x(t) = R_eff(t) · sin(ωt)
      y(t) = R_eff(t) · (1 − cos(ωt))
      z(t) = z0 + climb · t

    Smooth startup: orbit radius ramps from 0 to R via Hann window over
    the first full revolution (1/f seconds).

      Without ramp: vx(0) = R·ω ≠ 0  →  hard jerk at trajectory start
      With ramp:    vx(0) = vy(0) = 0 →  smooth entry from hover

    Exact velocity feedforward accounts for the changing radius (chain rule):
      ẋ = Ṙ · sin(ωt) + R_eff · ω · cos(ωt)
      ẏ = Ṙ · (1−cos(ωt)) + R_eff · ω · sin(ωt)
      ż = climb  (constant)

    Yaw is kept at zero — the linearised MPC tracks lateral motion via
    roll/pitch.  Tracking a rotating yaw while commanding roll/pitch would
    couple badly with the small-angle linearisation.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f
        climb = self.p['traj_climb_rate']
        z0    = self.p['hover_z']

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp
        R_dot = R * ramp_dot   # dR_eff/dt

        ref = np.zeros(12)
        ref[0] =  R_eff * math.sin(omega * dt)
        ref[1] =  R_eff * (1.0 - math.cos(omega * dt))
        ref[2] = z0 + climb * dt
        ref[6] =  R_dot * math.sin(omega * dt) + R_eff * omega * math.cos(omega * dt)
        ref[7] =  R_dot * (1.0 - math.cos(omega * dt)) + R_eff * omega * math.sin(omega * dt)
        ref[8] = climb
        return ref


class Figure8_3D(TrajectoryBase):
    """
    Horizontal figure-8 (lemniscate) at constant altitude.

      x(t) = R_eff · sin(ωt)
      y(t) = (R_eff/2) · sin(2ωt)
      z(t) = hover_z

    Smooth startup via Hann window radius ramp (one revolution).
    Without ramp vx(0) = vy(0) = R·ω ≠ 0; with ramp both start at 0.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f
        z0    = self.p['hover_z']

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp

        ref = np.zeros(12)
        ref[0] = R_eff * math.sin(omega * dt)
        ref[1] = (R_eff / 2.0) * math.sin(2.0 * omega * dt)
        ref[2] = z0
        ref[6] = (R * ramp_dot * math.sin(omega * dt) +
                  R_eff * omega * math.cos(omega * dt))
        ref[7] = (R * ramp_dot / 2.0 * math.sin(2.0 * omega * dt) +
                  R_eff * omega * math.cos(2.0 * omega * dt))
        return ref


class ConeHelix(TrajectoryBase):
    """
    Conical spiral: radius expands linearly with time while climbing in z.

    R(t) = Ṙ·t   where  Ṙ = traj_radius · traj_frequency [m/s]
    x(t) = R(t)·sin(ωt),   y(t) = R(t)·(1−cos(ωt)),   z = z0 + climb·t

    Velocities (exact time derivatives):
      ẋ = Ṙ·sin(ωt) + R·ω·cos(ωt)
      ẏ = Ṙ·(1−cos(ωt)) + R·ω·sin(ωt)
      ż = traj_climb_rate
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R_dot = self.p['traj_radius'] * self.p['traj_frequency']
        omega = 2.0 * math.pi * self.p['traj_frequency']
        climb = self.p['traj_climb_rate']
        z0    = self.p['hover_z']

        R = R_dot * dt

        ref = np.zeros(12)
        ref[0] = R * math.sin(omega * dt)
        ref[1] = R * (1.0 - math.cos(omega * dt))
        ref[2] = z0 + climb * dt
        ref[6] = R_dot * math.sin(omega * dt) + R * omega * math.cos(omega * dt)
        ref[7] = R_dot * (1.0 - math.cos(omega * dt)) + R * omega * math.sin(omega * dt)
        ref[8] = climb
        return ref


class Lissajous3D(TrajectoryBase):
    """
    3D Lissajous figure with frequency ratio 1:2:3.

      x(t) = R · sin(  ωt)
      y(t) = R · sin(2·ωt)
      z(t) = z0 + A · sin(3·ωt)

    Why this is the strongest 3D MPC showcase:
    - All three axes move simultaneously at different frequencies
    - The 1:2:3 ratio makes the path quasi-aperiodic → visits the whole volume
    - No axis can be tracked independently — they couple through roll/pitch
    - MPC's prediction horizon lets it anticipate the multi-axis reversals
    - A PID or LQR with no feedforward will show systematic phase lag on every axis

    Key parameters:
      traj_radius    [m]  — amplitude in x and y
      traj_amplitude [m]  — amplitude in z  (keep ≤ hover_z to avoid ground)
      traj_frequency [Hz] — base frequency  (0.15 Hz → x period ≈ 6.7 s)

    Suggested settings for a 10 s experiment:
      traj_radius:=0.4  traj_amplitude:=0.25  traj_frequency:=0.15
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        A     = self.p['traj_amplitude']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f
        z0    = self.p['hover_z']

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp
        A_eff = A * ramp

        ref = np.zeros(12)
        ref[0] = R_eff * math.sin(omega * dt)
        ref[1] = R_eff * math.sin(2.0 * omega * dt)
        ref[2] = z0 + A_eff * math.sin(3.0 * omega * dt)
        ref[6] = (R * ramp_dot * math.sin(omega * dt) +
                  R_eff * omega * math.cos(omega * dt))
        ref[7] = (R * ramp_dot * math.sin(2.0 * omega * dt) +
                  2.0 * R_eff * omega * math.cos(2.0 * omega * dt))
        ref[8] = (A * ramp_dot * math.sin(3.0 * omega * dt) +
                  3.0 * A_eff * omega * math.cos(3.0 * omega * dt))
        return ref


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

TRAJECTORY_CLASSES = {
    # Baseline
    'hover':          HoverTrajectory,
    # Part 2 — 2D (plane: xz or xy)
    'straight_2d':    StraightLine2D,
    'sine_2d':        SineWave2D,
    'step_2d':        StaircaseStep2D,
    'lemniscate_2d':  Lemniscate2D,
    'circle_2d':      Circle2D,
    # Part 3 — 3D
    'straight_3d':    StraightLine3D,
    'helix':          HelixTrajectory,
    'figure8_3d':     Figure8_3D,
    'cone_helix':     ConeHelix,
    'lissajous_3d':   Lissajous3D,
}


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

_PHASE_PRE_HOVER  = 0
_PHASE_FLYING     = 1
_PHASE_POST_HOVER = 2


class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('publish_rate',      50.0)
        self.declare_parameter('trajectory_type',   'hover')
        self.declare_parameter('hover_time',         5.0)
        self.declare_parameter('hover_z',            1.0)
        self.declare_parameter('hover_x',            0.0)
        self.declare_parameter('hover_y',            0.0)
        self.declare_parameter('traj_speed',         0.5)
        self.declare_parameter('traj_accel',         0.5)   # m/s²; 0 = instant
        self.declare_parameter('traj_amplitude',     0.3)
        self.declare_parameter('traj_frequency',     0.2)
        self.declare_parameter('traj_radius',        0.5)
        self.declare_parameter('traj_climb_rate',    0.15)
        self.declare_parameter('traj_yaw_rate',      0.0)
        self.declare_parameter('traj_duration',      0.0)
        self.declare_parameter('traj_plane',         'xz')  # 'xz' or 'xy'
        self.declare_parameter('horizon_steps',      0)     # 0 = no preview
        self.declare_parameter('horizon_dt',         0.02)  # MPC dt

        self._load_params()

        # ------------------------------------------------------------------ #
        # QoS
        # ------------------------------------------------------------------ #
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ------------------------------------------------------------------ #
        # Callback groups
        # ------------------------------------------------------------------ #
        timer_grp = MutuallyExclusiveCallbackGroup()
        sub_grp   = MutuallyExclusiveCallbackGroup()

        # ------------------------------------------------------------------ #
        # Publisher / Subscriber
        # ------------------------------------------------------------------ #
        self.ref_pub = self.create_publisher(
            Float64MultiArray, '/reference_state', reliable_qos)
        self.phase_pub = self.create_publisher(
            Int32, '/trajectory_phase', reliable_qos)
        self._last_published_phase = -1

        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb,
            reliable_qos, callback_group=sub_grp)

        # ------------------------------------------------------------------ #
        # Timer  (wall clock — use_sim_time=False so timers start immediately)
        # ------------------------------------------------------------------ #
        self._dt = 1.0 / self._rate
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(self._dt, self._publish_cb, callback_group=timer_grp)

        plane_str = self._params.get('traj_plane', 'xz').upper()
        phase_desc = (
            f'hover {self._hover_time:.0f}s → '
            f'fly "{self._traj_type}" [{plane_str}]'
            + (f' {self._duration:.0f}s → hover' if self._duration > 0 else ' (forever)')
        ) if self._traj_type != 'hover' else 'hover (steady)'

        self.get_logger().info(
            f'Trajectory generator ready  rate={self._rate}Hz  '
            f'sequence: {phase_desc}')

    # ---------------------------------------------------------------------- #
    # Parameter loading
    # ---------------------------------------------------------------------- #

    def _load_params(self):
        g = self.get_parameter
        self._rate       = g('publish_rate').value
        self._traj_type  = g('trajectory_type').value
        self._hover_time = g('hover_time').value
        self._duration   = g('traj_duration').value

        self._horizon_steps = int(g('horizon_steps').value)
        self._horizon_dt    = float(g('horizon_dt').value)

        self._params = {
            'hover_z':         g('hover_z').value,
            'hover_x':         g('hover_x').value,
            'hover_y':         g('hover_y').value,
            'traj_speed':      g('traj_speed').value,
            'traj_accel':      g('traj_accel').value,
            'traj_amplitude':  g('traj_amplitude').value,
            'traj_frequency':  g('traj_frequency').value,
            'traj_radius':     g('traj_radius').value,
            'traj_climb_rate': g('traj_climb_rate').value,
            'traj_yaw_rate':   g('traj_yaw_rate').value,
            'traj_plane':      g('traj_plane').value,
        }

        cls = TRAJECTORY_CLASSES.get(self._traj_type, HoverTrajectory)
        if cls is HoverTrajectory and self._traj_type != 'hover':
            self.get_logger().warn(
                f'Unknown trajectory_type "{self._traj_type}", falling back to hover')

        self._phase        = _PHASE_PRE_HOVER
        self._phase_t0     = None
        self._hover_traj   = HoverTrajectory(self._params)
        self._active_traj  = cls(self._params)

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def _goal_cb(self, msg: PoseStamped):
        """Accept an ad-hoc hover waypoint from RViz / CLI."""
        z = msg.pose.position.z if msg.pose.position.z > 0.05 \
            else self._params.get('hover_z', 1.0)
        self._params['hover_x'] = msg.pose.position.x
        self._params['hover_y'] = msg.pose.position.y
        self._params['hover_z'] = z
        self._hover_traj.p  = self._params.copy()
        self._active_traj.p = self._params.copy()
        self._hover_traj.t0 = None
        self.get_logger().info(
            f'Goal updated → ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, {z:.2f})')

    def _publish_cb(self):
        t_now = self.get_clock().now().nanoseconds * 1e-9

        if self._phase_t0 is None:
            self._phase_t0 = t_now
            self._hover_traj.start(t_now)

        elapsed = t_now - self._phase_t0

        # ---- Phase transitions ----
        if self._phase == _PHASE_PRE_HOVER:
            if self._traj_type != 'hover' and elapsed >= self._hover_time:
                self._phase    = _PHASE_FLYING
                self._phase_t0 = t_now
                elapsed        = 0.0
                self._active_traj.start(t_now)
                self.get_logger().info(
                    f'Phase → FLYING  trajectory="{self._traj_type}"  '
                    f'plane={self._params.get("traj_plane","xz")}')

        elif self._phase == _PHASE_FLYING:
            if self._duration > 0 and elapsed >= self._duration:
                self._phase    = _PHASE_POST_HOVER
                self._phase_t0 = t_now
                self._hover_traj.t0 = None
                self.get_logger().info('Phase → POST_HOVER  (returning to hover)')

        # ---- Select active reference ----
        traj = (self._active_traj
                if self._phase == _PHASE_FLYING
                else self._hover_traj)
        ref = traj.reference(t_now)

        # Build message: current ref + optional N-step preview
        if self._horizon_steps > 0:
            preview = np.empty(N_STATES + self._horizon_steps * N_STATES)
            preview[:N_STATES] = ref
            for k in range(self._horizon_steps):
                t_k = t_now + (k + 1) * self._horizon_dt
                preview[N_STATES + k * N_STATES:
                        N_STATES + (k + 1) * N_STATES] = traj.reference(t_k)
            msg = Float64MultiArray()
            msg.data = preview.tolist()
        else:
            msg = Float64MultiArray()
            msg.data = ref.tolist()
        self.ref_pub.publish(msg)

        # Publish phase on every change so data_collector can gate recording
        if self._phase != self._last_published_phase:
            phase_msg = Int32()
            phase_msg.data = self._phase
            self.phase_pub.publish(phase_msg)
            self._last_published_phase = self._phase

    # ---------------------------------------------------------------------- #
    # Public method: switch trajectory at runtime
    # ---------------------------------------------------------------------- #

    def set_trajectory(self, name: str):
        if name not in TRAJECTORY_CLASSES:
            self.get_logger().error(f'Unknown trajectory: {name}')
            return
        self._traj_type   = name
        self._active_traj = TRAJECTORY_CLASSES[name](self._params)
        self.get_logger().info(f'Switched to trajectory: {name}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

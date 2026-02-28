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

Part 2  (2-D, x-z plane  —  y = 0 throughout)
  straight_2d   Constant-speed straight line in +x, constant z
  sine_2d       x increases at constant speed, z oscillates as a sine wave
  step_2d       Staircase: climb dz every dx metres in x-z plane

Part 3  (3-D)
  straight_3d   Straight line with simultaneous x, y, z motion
  helix         Helical spiral: circles in x-y while climbing in z
  figure8_3d    Horizontal figure-8 (lemniscate) at constant z
  cone_helix    Conical spiral: expanding-radius helix forming a cone

hover (default)
  The drone stays at the configured hover position — useful as baseline
  to verify the MPC is working before running trajectories.

Parameters (all set in config/params.yaml or via CLI)
--------------------
  trajectory_type   : string  — one of the names above  (default: hover)
  publish_rate      : float   — Hz (default 50)
  hover_time        : float   — seconds to hold hover before trajectory starts
                                (default 5.0)
  hover_z           : float   — hover height [m]  (default 1.0)
  traj_speed        : float   — translational speed [m/s]  (default 0.5)
  traj_amplitude    : float   — amplitude of oscillation [m]  (default 0.3)
  traj_frequency    : float   — oscillation frequency [Hz]  (default 0.2)
  traj_radius       : float   — radius for circular/helix  [m]  (default 1.0)
  traj_climb_rate   : float   — z climb rate for helix  [m/s]  (default 0.2)
  traj_yaw_rate     : float   — desired yaw rate [rad/s]  (default 0.0)
  traj_duration     : float   — seconds to fly trajectory before returning to
                                hover; 0 = fly forever  (default 0)

ROS2 pros used
--------------
  * Declared parameters → change trajectory at runtime via `ros2 param set`
  * MutuallyExclusiveCallbackGroup → timer and goal_pose sub run concurrently
  * MultiThreadedExecutor
  * use_sim_time=False → timers start immediately, no /clock dependency
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped


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


class HoverTrajectory(TrajectoryBase):
    """Stay at a fixed hover position."""

    def reference(self, t_now):
        ref = np.zeros(12)
        ref[0] = self.p.get('hover_x', 0.0)
        ref[1] = self.p.get('hover_y', 0.0)
        ref[2] = self.p['hover_z']
        return ref


class StraightLine2D(TrajectoryBase):
    """Move in +x at constant speed; z constant. (x-z plane trajectory)"""

    def reference(self, t_now):
        dt = self._elapsed(t_now)
        v  = self.p['traj_speed']
        ref = np.zeros(12)
        ref[0] = v * dt           # x
        ref[1] = 0.0              # y = 0  (stay in x-z plane)
        ref[2] = self.p['hover_z']
        ref[6] = v                # ẋ
        return ref


class SineWave2D(TrajectoryBase):
    """
    x-z plane trajectory: x increases at constant speed,
    z oscillates as a sine wave. y = 0 throughout.
    z(t) = hover_z + A * sin(2π f t),  y = 0 (x-z plane)
    """

    def reference(self, t_now):
        dt = self._elapsed(t_now)
        v  = self.p['traj_speed']
        A  = self.p['traj_amplitude']
        f  = self.p['traj_frequency']
        z0 = self.p['hover_z']
        omega = 2.0 * math.pi * f

        ref = np.zeros(12)
        ref[0] = v * dt                              # x  (forward)
        ref[1] = 0.0                                 # y = 0  (x-z plane)
        ref[2] = z0 + A * math.sin(omega * dt)       # z  oscillates
        ref[6] = v                                   # ẋ
        ref[8] = A * omega * math.cos(omega * dt)    # ż
        return ref


class StaircaseStep2D(TrajectoryBase):
    """
    Staircase in x-z: advance dx in x then climb dz in z, repeat.
    Both x and z move in steps; drone stays in x-z plane.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        v     = self.p['traj_speed']
        A     = self.p['traj_amplitude']     # height per step [m]
        f     = self.p['traj_frequency']     # step frequency [Hz]
        z0    = self.p['hover_z']

        # Triangular wave for z gives a smooth staircase
        period = 1.0 / f
        phase  = (dt % period) / period      # 0..1 within period
        step   = math.floor(dt * f)
        z = z0 + A * step * 0.5             # staircase climbs
        # Limit z to hover_z + 2*amplitude range then cycle back
        z_range = 2.0 * A
        z = z0 + (z - z0) % z_range

        ref = np.zeros(12)
        ref[0] = v * dt
        ref[1] = 0.0
        ref[2] = z
        ref[6] = v
        return ref


class StraightLine3D(TrajectoryBase):
    """
    Move along direction [1, 0.5, 0.3] (normalised) at constant speed.
    Demonstrates simultaneous x, y, z motion.
    """

    def reference(self, t_now):
        dt   = self._elapsed(t_now)
        v    = self.p['traj_speed']
        z0   = self.p['hover_z']
        # Direction vector (normalised)
        d = np.array([1.0, 0.5, 0.3])
        d = d / np.linalg.norm(d)

        pos = np.array([0.0, 0.0, z0]) + d * v * dt
        vel = d * v

        ref = np.zeros(12)
        ref[0:3] = pos
        ref[6:9] = vel
        return ref


class HelixTrajectory(TrajectoryBase):
    """
    Helix spiral: circle in x-y while climbing in z.
    Starts from hover position (0, 0) using phase-shifted circle:
      x(t) = R sin(ω t),  y(t) = R (1 - cos(ω t)),  z(t) = z0 + climb * t

    Yaw is kept at zero — the linearised MPC tracks lateral motion via
    roll/pitch far more reliably than when simultaneously rotating yaw.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        climb = self.p['traj_climb_rate']
        z0    = self.p['hover_z']
        omega = 2.0 * math.pi * f

        ref = np.zeros(12)
        ref[0] =  R * math.sin(omega * dt)
        ref[1] =  R * (1.0 - math.cos(omega * dt))
        ref[2] = z0 + climb * dt
        ref[6] =  R * omega * math.cos(omega * dt)    # ẋ
        ref[7] =  R * omega * math.sin(omega * dt)    # ẏ
        ref[8] = climb                                 # ż
        # yaw = 0: the linearised MPC cannot reliably track continuous
        # yaw rotation while also commanding large roll/pitch for
        # lateral motion — the coupling destabilises the controller.
        return ref


class Figure8_3D(TrajectoryBase):
    """
    Horizontal figure-8 (lemniscate of Bernoulli) at constant altitude.
    x(t) = A sin(ωt),   y(t) = A/2 sin(2ωt),   z = hover_z

    Yaw is kept at zero for the same reason as HelixTrajectory.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        A     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        z0    = self.p['hover_z']
        omega = 2.0 * math.pi * f

        ref = np.zeros(12)
        ref[0] = A * math.sin(omega * dt)
        ref[1] = (A / 2.0) * math.sin(2.0 * omega * dt)
        ref[2] = z0
        ref[6] = A * omega * math.cos(omega * dt)              # ẋ
        ref[7] = A * omega * math.cos(2.0 * omega * dt)        # ẏ
        return ref


class ConeHelix(TrajectoryBase):
    """
    Conical spiral: radius expands linearly with time while climbing in z.

    The path traces an outward-expanding cone — each revolution is wider
    than the previous one.  Starts at the origin with zero lateral velocity
    so the drone transitions smoothly from hover.

    Parametrisation
    ---------------
      R(t)  = traj_radius * f * t          radius gained per revolution
      x(t)  = R(t) * sin(ω t)
      y(t)  = R(t) * (1 - cos(ω t))       phase-shift keeps start at (0,0)
      z(t)  = hover_z + traj_climb_rate * t

    Velocities (exact derivatives)
    --------------------------------
      ẋ  = Ṙ * sin(ω t) + R * ω * cos(ω t)
      ẏ  = Ṙ * (1 - cos(ω t)) + R * ω * sin(ω t)
      ż  = traj_climb_rate

    where  Ṙ = traj_radius * f  (constant radius rate)

    Parameters used
    ---------------
      traj_radius     : [m] radius added per revolution   (default 0.5)
      traj_frequency  : [Hz] revolution frequency         (default 0.1)
      traj_climb_rate : [m/s] vertical climb speed        (default 0.15)
      hover_z         : [m] starting altitude

    MPC note
    --------
    Lateral accelerations grow with R, so use short durations (≤ 30 s) or
    small traj_radius to keep roll/pitch within the ±15° linearisation bound.
    """

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R_rate = self.p['traj_radius'] * self.p['traj_frequency']   # Ṙ [m/s]
        f      = self.p['traj_frequency']
        climb  = self.p['traj_climb_rate']
        z0     = self.p['hover_z']
        omega  = 2.0 * math.pi * f

        R = R_rate * dt                         # expanding radius

        ref = np.zeros(12)
        ref[0] = R * math.sin(omega * dt)
        ref[1] = R * (1.0 - math.cos(omega * dt))
        ref[2] = z0 + climb * dt
        ref[6] = (R_rate * math.sin(omega * dt) +
                  R * omega * math.cos(omega * dt))      # ẋ
        ref[7] = (R_rate * (1.0 - math.cos(omega * dt)) +
                  R * omega * math.sin(omega * dt))      # ẏ
        ref[8] = climb                                    # ż
        return ref


# Map parameter string → class
TRAJECTORY_CLASSES = {
    'hover':       HoverTrajectory,
    'straight_2d': StraightLine2D,
    'sine_2d':     SineWave2D,
    'step_2d':     StaircaseStep2D,
    'straight_3d': StraightLine3D,
    'helix':       HelixTrajectory,
    'figure8_3d':  Figure8_3D,
    'cone_helix':  ConeHelix,
}


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

# Phase constants
_PHASE_PRE_HOVER  = 0   # stabilise at hover before trajectory
_PHASE_FLYING     = 1   # execute the active trajectory
_PHASE_POST_HOVER = 2   # return to hover after trajectory ends


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
        self.declare_parameter('traj_amplitude',     0.3)
        self.declare_parameter('traj_frequency',     0.2)
        self.declare_parameter('traj_radius',        0.5)
        self.declare_parameter('traj_climb_rate',    0.15)
        self.declare_parameter('traj_yaw_rate',      0.0)
        self.declare_parameter('traj_duration',      0.0)

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

        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb,
            reliable_qos, callback_group=sub_grp)

        # ------------------------------------------------------------------ #
        # Timer  (use wall clock — use_sim_time=False set in launch file)
        # ------------------------------------------------------------------ #
        self._dt = 1.0 / self._rate
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(self._dt, self._publish_cb, callback_group=timer_grp)

        phase_desc = (
            f'hover {self._hover_time:.0f}s → '
            f'fly "{self._traj_type}"'
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

        self._params = {
            'hover_z':         g('hover_z').value,
            'hover_x':         g('hover_x').value,
            'hover_y':         g('hover_y').value,
            'traj_speed':      g('traj_speed').value,
            'traj_amplitude':  g('traj_amplitude').value,
            'traj_frequency':  g('traj_frequency').value,
            'traj_radius':     g('traj_radius').value,
            'traj_climb_rate': g('traj_climb_rate').value,
            'traj_yaw_rate':   g('traj_yaw_rate').value,
        }

        cls = TRAJECTORY_CLASSES.get(self._traj_type, HoverTrajectory)
        if cls is HoverTrajectory and self._traj_type != 'hover':
            self.get_logger().warn(
                f'Unknown trajectory_type "{self._traj_type}", falling back to hover')

        # Phase state machine
        self._phase        = _PHASE_PRE_HOVER
        self._phase_t0     = None     # set on first publish tick
        # PRE_HOVER uses a plain HoverTrajectory
        self._hover_traj   = HoverTrajectory(self._params)
        # Active trajectory (used during FLYING phase)
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
        self._hover_traj.t0 = None    # reset hover reference
        self.get_logger().info(
            f'Goal updated → ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, {z:.2f})')

    def _publish_cb(self):
        t_now = self.get_clock().now().nanoseconds * 1e-9

        # Initialise phase clock on first tick
        if self._phase_t0 is None:
            self._phase_t0 = t_now
            self._hover_traj.start(t_now)

        elapsed = t_now - self._phase_t0   # seconds since phase started

        # ---- Phase transitions ----
        if self._phase == _PHASE_PRE_HOVER:
            if self._traj_type != 'hover' and elapsed >= self._hover_time:
                self._phase = _PHASE_FLYING
                self._phase_t0 = t_now
                elapsed = 0.0
                self._active_traj.start(t_now)
                self.get_logger().info(
                    f'Phase → FLYING  trajectory="{self._traj_type}"')

        elif self._phase == _PHASE_FLYING:
            if self._duration > 0 and elapsed >= self._duration:
                self._phase = _PHASE_POST_HOVER
                self._phase_t0 = t_now
                self._hover_traj.t0 = None   # re-anchor hover to now
                self.get_logger().info('Phase → POST_HOVER  (returning to hover)')

        # ---- Select active reference ----
        if self._phase == _PHASE_FLYING:
            ref = self._active_traj.reference(t_now)
        else:
            ref = self._hover_traj.reference(t_now)

        msg = Float64MultiArray()
        msg.data = ref.tolist()
        self.ref_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Public method: switch trajectory at runtime
    # ---------------------------------------------------------------------- #

    def set_trajectory(self, name: str):
        if name not in TRAJECTORY_CLASSES:
            self.get_logger().error(f'Unknown trajectory: {name}')
            return
        self._traj_type  = name
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

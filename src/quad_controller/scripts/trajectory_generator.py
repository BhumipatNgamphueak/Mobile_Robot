#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64MultiArray, Int32
from geometry_msgs.msg import PoseStamped


N_STATES = 12


class TrajectoryBase:

    def __init__(self, params: dict):
        self.p = params
        self.t0 = None

    def start(self, t_now: float):
        self.t0 = t_now

    def reference(self, t_now: float) -> np.ndarray:
        raise NotImplementedError

    def _elapsed(self, t_now: float) -> float:
        if self.t0 is None:
            self.t0 = t_now
        return t_now - self.t0

    def _plane_ref(self, fwd: float, lat: float,
                   vfwd: float = 0.0, vlat: float = 0.0) -> np.ndarray:
        plane = self.p.get('traj_plane', 'xz')
        z0    = self.p['hover_z']
        ref   = np.zeros(12)
        if plane == 'xy':
            ref[0], ref[1], ref[2] = fwd, lat, z0
            ref[6], ref[7], ref[8] = vfwd, vlat, 0.0
        else:
            ref[0], ref[1], ref[2] = fwd, 0.0, z0 + lat
            ref[6], ref[7], ref[8] = vfwd, 0.0, vlat
        return ref

    @staticmethod
    def _trapezoid(dt: float, v_max: float, a: float):
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
        if t_ramp <= 0.0 or dt >= t_ramp:
            return 1.0, 0.0
        s        = math.pi * dt / t_ramp
        ramp     = 0.5 * (1.0 - math.cos(s))
        ramp_dot = 0.5 * math.pi / t_ramp * math.sin(s)
        return ramp, ramp_dot


class HoverTrajectory(TrajectoryBase):

    def reference(self, t_now):
        ref = np.zeros(12)
        ref[0] = self.p.get('hover_x', 0.0)
        ref[1] = self.p.get('hover_y', 0.0)
        ref[2] = self.p['hover_z']
        return ref


class StraightLine2D(TrajectoryBase):

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        v_max = self.p['traj_speed']
        a     = self.p.get('traj_accel', 0.5)

        pos, vel = self._trapezoid(dt, v_max, a)
        return self._plane_ref(pos, 0.0, vel, 0.0)


class SineWave2D(TrajectoryBase):

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

    def reference(self, t_now):
        dt = self._elapsed(t_now)
        v  = self.p['traj_speed']
        A  = self.p['traj_amplitude']
        f  = self.p['traj_frequency']

        step = math.floor(dt * f)
        lat  = A * step * 0.5
        lat  = lat % (2.0 * A)

        return self._plane_ref(v * dt, lat, v, 0.0)


class Lemniscate2D(TrajectoryBase):

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


class StraightLine3D(TrajectoryBase):

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

    def reference(self, t_now):
        dt    = self._elapsed(t_now)
        R     = self.p['traj_radius']
        f     = self.p['traj_frequency']
        omega = 2.0 * math.pi * f
        climb = self.p['traj_climb_rate']
        z0    = self.p['hover_z']

        ramp, ramp_dot = self._hann_ramp(dt, 1.0 / f)
        R_eff = R * ramp
        R_dot = R * ramp_dot

        ref = np.zeros(12)
        ref[0] =  R_eff * math.sin(omega * dt)
        ref[1] =  R_eff * (1.0 - math.cos(omega * dt))
        ref[2] = z0 + climb * dt
        ref[6] =  R_dot * math.sin(omega * dt) + R_eff * omega * math.cos(omega * dt)
        ref[7] =  R_dot * (1.0 - math.cos(omega * dt)) + R_eff * omega * math.sin(omega * dt)
        ref[8] = climb
        return ref


class Figure8_3D(TrajectoryBase):

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


TRAJECTORY_CLASSES = {
    'hover':          HoverTrajectory,
    'straight_2d':    StraightLine2D,
    'sine_2d':        SineWave2D,
    'step_2d':        StaircaseStep2D,
    'lemniscate_2d':  Lemniscate2D,
    'circle_2d':      Circle2D,
    'straight_3d':    StraightLine3D,
    'helix':          HelixTrajectory,
    'figure8_3d':     Figure8_3D,
    'cone_helix':     ConeHelix,
    'lissajous_3d':   Lissajous3D,
}


_PHASE_PRE_HOVER  = 0
_PHASE_FLYING     = 1
_PHASE_POST_HOVER = 2


class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        self.declare_parameter('publish_rate',      50.0)
        self.declare_parameter('trajectory_type',   'hover')
        self.declare_parameter('hover_time',         5.0)
        self.declare_parameter('hover_z',            1.0)
        self.declare_parameter('hover_x',            0.0)
        self.declare_parameter('hover_y',            0.0)
        self.declare_parameter('traj_speed',         0.5)
        self.declare_parameter('traj_accel',         0.5)
        self.declare_parameter('traj_amplitude',     0.3)
        self.declare_parameter('traj_frequency',     0.2)
        self.declare_parameter('traj_radius',        0.5)
        self.declare_parameter('traj_climb_rate',    0.15)
        self.declare_parameter('traj_yaw_rate',      0.0)
        self.declare_parameter('traj_duration',      0.0)
        self.declare_parameter('traj_plane',         'xz')
        self.declare_parameter('horizon_steps',      0)
        self.declare_parameter('horizon_dt',         0.02)

        self._load_params()

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        timer_grp = MutuallyExclusiveCallbackGroup()
        sub_grp   = MutuallyExclusiveCallbackGroup()

        self.ref_pub = self.create_publisher(
            Float64MultiArray, '/reference_state', reliable_qos)
        self.phase_pub = self.create_publisher(
            Int32, '/trajectory_phase', reliable_qos)
        self._last_published_phase = -1

        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb,
            reliable_qos, callback_group=sub_grp)

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

    def _goal_cb(self, msg: PoseStamped):
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

        traj = (self._active_traj
                if self._phase == _PHASE_FLYING
                else self._hover_traj)
        ref = traj.reference(t_now)

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

        if self._phase != self._last_published_phase:
            phase_msg = Int32()
            phase_msg.data = self._phase
            self.phase_pub.publish(phase_msg)
            self._last_published_phase = self._phase

    def set_trajectory(self, name: str):
        if name not in TRAJECTORY_CLASSES:
            self.get_logger().error(f'Unknown trajectory: {name}')
            return
        self._traj_type   = name
        self._active_traj = TRAJECTORY_CLASSES[name](self._params)
        self.get_logger().info(f'Switched to trajectory: {name}')


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

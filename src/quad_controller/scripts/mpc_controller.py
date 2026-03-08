#!/usr/bin/env python3

import numpy as np
from scipy.signal import cont2discrete
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy)

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker


N_STATES = 12
N_INPUTS = 4


class MPCController(Node):

    def __init__(self):
        super().__init__('mpc_controller')

        self.declare_parameter('mass',           1.5)
        self.declare_parameter('gravity',        9.81)
        self.declare_parameter('ixx',            0.0347563)
        self.declare_parameter('iyy',            0.07)
        self.declare_parameter('izz',            0.0977)
        self.declare_parameter('kF',             8.54858e-06)
        self.declare_parameter('kM',             0.06)
        self.declare_parameter('dt',             0.02)
        self.declare_parameter('horizon',        20)
        self.declare_parameter('q_pos',          20.0)
        self.declare_parameter('q_ang',          10.0)
        self.declare_parameter('q_vel',           5.0)
        self.declare_parameter('q_angvel',        1.0)
        self.declare_parameter('r_thrust',        0.01)
        self.declare_parameter('r_moment',        0.1)
        self.declare_parameter('max_rotor_speed', 1500.0)
        self.declare_parameter('ki_xy',           2.0)
        self.declare_parameter('ki_z',            3.0)
        self.declare_parameter('int_max',         2.0)
        self.declare_parameter('wind_x',          0.0)
        self.declare_parameter('wind_y',          0.0)
        self.declare_parameter('wind_z',          0.0)

        self._configured = False
        self._active     = False

        self.get_logger().info('MPC controller created')
        self._configure()
        self._activate()

    def _configure(self):
        p = self.get_parameter
        self.m    = p('mass').value
        self.g    = p('gravity').value
        self.Ixx  = p('ixx').value
        self.Iyy  = p('iyy').value
        self.Izz  = p('izz').value
        self.kF   = p('kF').value
        self.kM   = p('kM').value
        self.dt   = p('dt').value
        self.N    = p('horizon').value
        self.wmax = p('max_rotor_speed').value

        self.ki_xy   = p('ki_xy').value
        self.ki_z    = p('ki_z').value
        self.int_max = p('int_max').value
        self._wind   = np.array([p('wind_x').value,
                                 p('wind_y').value,
                                 p('wind_z').value])

        self._build_system()
        self._build_mpc()
        self._build_allocation()

        self.x_state = np.zeros(N_STATES)
        self.x_ref   = np.zeros(N_STATES)
        self.x_ref[2] = 1.0
        self._ref_preview = None

        self._integral_err = np.zeros(3)

        self._actual_path = Path()
        self._actual_path.header.frame_id = 'odom'
        self._ref_path = Path()
        self._ref_path.header.frame_id = 'odom'
        self._PATH_MAX = 2000
        self._path_tick = 0

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(
            Odometry, '/state_estimate', self._state_cb, sensor_qos)
        self.create_subscription(
            Float64MultiArray, '/reference_state', self._ref_cb, reliable_qos)

        self._motor_pub = self.create_publisher(
            Actuators, '/motor_commands', reliable_qos)
        self._wrench_pub = self.create_publisher(
            Float64MultiArray, '/control_wrench', reliable_qos)
        self._actual_path_pub = self.create_publisher(
            Path, '/actual_path', reliable_qos)
        self._ref_path_pub = self.create_publisher(
            Path, '/reference_path', reliable_qos)
        self._wind_pub = self.create_publisher(
            Marker, '/wind_marker', reliable_qos)
        self._debug_pub = self.create_publisher(
            Float64MultiArray, '/mpc_debug', reliable_qos)

        self._configured = True
        self.get_logger().info(
            f'MPC configured  N={self.N}  dt={self.dt}s  '
            f'hover_omega={self.omega_hover.round(1)} rad/s  '
            f'kappa(kF*kM)={self.kF * self.kM:.3e}')

    def _activate(self):
        if not self._configured:
            self.get_logger().error('Cannot activate before configure()')
            return
        self._ctrl_timer = self.create_timer(self.dt, self._control_loop)
        self._active = True
        self.get_logger().info('MPC activated -- control loop running')

    def _deactivate(self):
        if self._active:
            self.destroy_timer(self._ctrl_timer)
            self._active = False
            self._publish_zeros()
            self.get_logger().info('MPC deactivated')

    def _build_system(self):
        n, m = N_STATES, N_INPUTS

        Ac = np.zeros((n, n))
        Ac[0:3, 6:9]  = np.eye(3)
        Ac[3:6, 9:12] = np.eye(3)
        Ac[6, 4]  =  self.g
        Ac[7, 3]  = -self.g

        Bc = np.zeros((n, m))
        Bc[8,  0] = 1.0 / self.m
        Bc[9,  1] = 1.0 / self.Ixx
        Bc[10, 2] = 1.0 / self.Iyy
        Bc[11, 3] = 1.0 / self.Izz

        sys_d = cont2discrete(
            (Ac, Bc, np.eye(n), np.zeros((n, m))), self.dt, method='zoh')
        self.Ad, self.Bd = sys_d[0], sys_d[1]

    def _build_mpc(self):
        n, m, N = N_STATES, N_INPUTS, self.N
        p = self.get_parameter

        Q = np.diag([p('q_pos').value]    * 3 +
                    [p('q_ang').value]    * 3 +
                    [p('q_vel').value]    * 3 +
                    [p('q_angvel').value] * 3)
        R = np.diag([p('r_thrust').value] +
                    [p('r_moment').value] * 3)

        Phi = np.zeros((N * n, n))
        Ad_pow = np.eye(n)
        for i in range(N):
            Ad_pow = Ad_pow @ self.Ad
            Phi[i*n:(i+1)*n] = Ad_pow

        Gamma = np.zeros((N * n, N * m))
        for col in range(N):
            Ad_pow_Bd = self.Bd.copy()
            for row in range(col, N):
                Gamma[row*n:(row+1)*n, col*m:(col+1)*m] = Ad_pow_Bd
                Ad_pow_Bd = self.Ad @ Ad_pow_Bd

        Q_bar = np.kron(np.eye(N), Q)
        R_bar = np.kron(np.eye(N), R)

        GtQ  = Gamma.T @ Q_bar
        H    = GtQ @ Gamma + R_bar
        H_inv = np.linalg.inv(H)

        K_full = H_inv @ GtQ
        self.K_r = K_full[0:m, :]
        self.K_x = -(H_inv @ GtQ @ Phi)[0:m, :]

    def _build_allocation(self):
        kF = self.kF
        kappa = self.kF * self.kM

        rotors = [
            ( 0.13, -0.22, +1),   # rotor_0 front-right ccw
            (-0.13,  0.20, +1),   # rotor_1 rear-left   ccw
            ( 0.13,  0.22, -1),   # rotor_2 front-left  cw
            (-0.13, -0.20, -1),   # rotor_3 rear-right  cw
        ]
        A = np.zeros((4, 4))
        for i, (rx, ry, d) in enumerate(rotors):
            A[0, i] = kF
            A[1, i] = kF *   ry
            A[2, i] = kF * (-rx)
            A[3, i] = -d  * kappa

        self.alloc_fwd = A
        self.alloc_inv = np.linalg.inv(A)

        omega_sq_hov   = self.alloc_inv @ np.array([self.m * self.g, 0, 0, 0])
        self.omega_hover = np.sqrt(np.maximum(omega_sq_hov, 0.0))

        self.get_logger().info(
            f'Allocation matrix condition number: {np.linalg.cond(A):.1f}')

    def _constrained_allocation(self, u):
        omega_sq = self.alloc_inv @ u
        omega_sq = np.maximum(omega_sq, 0.0)
        omega    = np.sqrt(omega_sq)

        if np.all(omega <= self.wmax):
            return omega, 1.0

        u_thrust = np.array([u[0], 0.0, 0.0, 0.0])
        u_torque = u - u_thrust

        osq_t = np.maximum(self.alloc_inv @ u_thrust, 0.0)
        o_t   = np.sqrt(osq_t)

        if np.any(o_t > self.wmax):
            scale_t = (self.wmax * 0.95) / np.max(o_t)
            u_thrust *= scale_t
            omega_sq = np.maximum(self.alloc_inv @ u_thrust, 0.0)
            return np.clip(np.sqrt(omega_sq), 0.0, self.wmax), 0.0

        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) * 0.5
            u_try = u_thrust + mid * u_torque
            osq = np.maximum(self.alloc_inv @ u_try, 0.0)
            if np.all(np.sqrt(osq) <= self.wmax):
                lo = mid
            else:
                hi = mid

        u_final = u_thrust + lo * u_torque
        omega_sq = np.maximum(self.alloc_inv @ u_final, 0.0)
        return np.clip(np.sqrt(omega_sq), 0.0, self.wmax), lo

    def _state_cb(self, msg: Odometry):
        pos  = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        vel  = msg.twist.twist.linear
        rate = msg.twist.twist.angular

        q_arr = np.array([quat.x, quat.y, quat.z, quat.w])
        if np.any(np.isnan(q_arr)) or np.any(np.isinf(q_arr)):
            return
        norm = np.linalg.norm(q_arr)
        if norm < 1e-6:
            return

        rot = Rotation.from_quat(q_arr / norm)
        yaw, roll, pitch = rot.as_euler('zxy')

        self.x_state = np.array([
            pos.x,  pos.y,  pos.z,
            roll,   pitch,  yaw,
            vel.x,  vel.y,  vel.z,
            rate.x, rate.y, rate.z,
        ])

        ps = PoseStamped()
        ps.header.stamp    = msg.header.stamp
        ps.header.frame_id = 'odom'
        ps.pose = msg.pose.pose
        self._actual_path.poses.append(ps)
        if len(self._actual_path.poses) > self._PATH_MAX:
            self._actual_path.poses.pop(0)

    def _ref_cb(self, msg: Float64MultiArray):
        data = np.array(msg.data)
        if len(data) > N_STATES and len(data) % N_STATES == 0:
            self.x_ref = data[:N_STATES].copy()
            self._ref_preview = data[N_STATES:]
        elif len(data) >= N_STATES:
            self.x_ref = data[:N_STATES].copy()
            self._ref_preview = None
        elif len(data) == 3:
            ref      = np.zeros(N_STATES)
            ref[0:3] = data
            self.x_ref = ref
            self._ref_preview = None

        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'odom'
        ps.pose.position.x = self.x_ref[0]
        ps.pose.position.y = self.x_ref[1]
        ps.pose.position.z = self.x_ref[2]
        rot  = Rotation.from_euler('zxy', [self.x_ref[5], self.x_ref[3], self.x_ref[4]])
        q    = rot.as_quat()
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        self._ref_path.poses.append(ps)
        if len(self._ref_path.poses) > self._PATH_MAX:
            self._ref_path.poses.pop(0)

    def _control_loop(self):
        x0    = self.x_state
        x_ref = self.x_ref

        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            self._publish_zeros()
            self._publish_debug_skip()
            return

        N, n, m = self.N, N_STATES, N_INPUTS

        psi = x0[5]
        c, s = np.cos(psi), np.sin(psi)

        x0_h = x0.copy()
        x0_h[0] =  c * x0[0] + s * x0[1]
        x0_h[1] = -s * x0[0] + c * x0[1]
        x0_h[5] = 0.0
        x0_h[6] =  c * x0[6] + s * x0[7]
        x0_h[7] = -s * x0[6] + c * x0[7]

        xr_h = x_ref.copy()
        xr_h[0] =  c * x_ref[0] + s * x_ref[1]
        xr_h[1] = -s * x_ref[0] + c * x_ref[1]
        xr_h[5] = (x_ref[5] - psi + np.pi) % (2 * np.pi) - np.pi
        xr_h[6] =  c * x_ref[6] + s * x_ref[7]
        xr_h[7] = -s * x_ref[6] + c * x_ref[7]

        if (self._ref_preview is not None
                and len(self._ref_preview) >= N * n):
            X_ref = np.empty(N * n)
            preview = self._ref_preview
            for k in range(N):
                rk = preview[k * n:(k + 1) * n].copy()
                rk[0] =  c * preview[k*n + 0] + s * preview[k*n + 1]
                rk[1] = -s * preview[k*n + 0] + c * preview[k*n + 1]
                rk[5] = (preview[k*n + 5] - psi + np.pi) % (2*np.pi) - np.pi
                rk[6] =  c * preview[k*n + 6] + s * preview[k*n + 7]
                rk[7] = -s * preview[k*n + 6] + c * preview[k*n + 7]
                X_ref[k * n:(k + 1) * n] = rk
        else:
            X_ref = np.tile(xr_h, N)

        u_delta = self.K_r @ X_ref + self.K_x @ x0_h

        u_total = u_delta.copy()
        u_total[0] += self.m * self.g

        u_integral = np.zeros(4)
        if x0[2] > 0.15:
            pos_err = x_ref[0:3] - x0[0:3]
            self._integral_err += pos_err * self.dt
            self._integral_err = np.clip(
                self._integral_err, -self.int_max, self.int_max)

            u_integral[0] = self.ki_z * self._integral_err[2]

            ie_x =  c * self._integral_err[0] + s * self._integral_err[1]
            ie_y = -s * self._integral_err[0] + c * self._integral_err[1]
            u_integral[2] =  self.ki_xy * ie_x
            u_integral[1] = -self.ki_xy * ie_y

            u_total += u_integral

        wrench_msg = Float64MultiArray()
        wrench_msg.data = u_total.tolist()
        self._wrench_pub.publish(wrench_msg)

        omega, torque_scale = self._constrained_allocation(u_total)

        e_h = x0_h - xr_h
        p = self.get_parameter
        Q_diag = np.array([p('q_pos').value]  * 3 +
                          [p('q_ang').value]  * 3 +
                          [p('q_vel').value]  * 3 +
                          [p('q_angvel').value] * 3)
        R_diag = np.array([p('r_thrust').value] +
                          [p('r_moment').value] * 3)
        cost_state = float(e_h @ np.diag(Q_diag) @ e_h)
        cost_input = float(u_delta @ np.diag(R_diag) @ u_delta)

        dbg = Float64MultiArray()
        dbg.data = (
            u_delta.tolist() +
            u_integral.tolist() +
            u_total.tolist() +
            [torque_scale, cost_state, cost_input, 1.0,
             self._integral_err[0],
             self._integral_err[1],
             self._integral_err[2]]
        )
        self._debug_pub.publish(dbg)

        cmd = Actuators()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.velocity     = omega.tolist()
        self._motor_pub.publish(cmd)

        self._path_tick += 1
        if self._path_tick >= 5:
            self._path_tick = 0
            now = self.get_clock().now().to_msg()
            self._actual_path.header.stamp = now
            self._ref_path.header.stamp    = now
            self._actual_path_pub.publish(self._actual_path)
            self._ref_path_pub.publish(self._ref_path)
            self._publish_wind_marker(now, x0)

    def _publish_debug_skip(self):
        dbg = Float64MultiArray()
        dbg.data = [0.0] * 16 + [0.0, 0.0, 0.0]
        self._debug_pub.publish(dbg)

    def _publish_wind_marker(self, stamp, x0):
        w = self._wind
        speed = np.linalg.norm(w)
        if speed < 0.01:
            return

        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = 'odom'
        m.ns     = 'wind'
        m.id     = 0
        m.type   = Marker.ARROW
        m.action = Marker.ADD

        start = PoseStamped().pose.position
        start.x = x0[0]
        start.y = x0[1]
        start.z = x0[2] + 1.0
        scale = 0.5
        end = PoseStamped().pose.position
        end.x = start.x + w[0] * scale
        end.y = start.y + w[1] * scale
        end.z = start.z + w[2] * scale
        m.points = [start, end]

        m.scale.x = 0.05
        m.scale.y = 0.10
        m.scale.z = 0.12
        m.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)
        m.lifetime.sec = 0

        self._wind_pub.publish(m)

    def _publish_zeros(self):
        cmd = Actuators()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.velocity     = [0.0, 0.0, 0.0, 0.0]
        self._motor_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._deactivate()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

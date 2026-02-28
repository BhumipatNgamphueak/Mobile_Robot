#!/usr/bin/env python3
"""
EKF State Estimator for Quadrotor
==================================
Fuses IMU (accel + gyro) and Odometry (position + orientation) using an
Extended Kalman Filter to produce a clean 12-DOF state estimate.

State vector:  x = [x, y, z, phi, theta, psi, xdot, ydot, zdot, p, q, r]
Control input: u = [T, tau_phi, tau_theta, tau_psi]  (from MPC feedback)

ROS2 pros used:
  - BEST_EFFORT QoS for sensor data (mirrors DDS sensor profile)
  - RELIABLE QoS for state estimate output
  - MultiThreadedExecutor for concurrent sensor callbacks
  - Declared parameters for all tunable EKF gains
"""

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES = 12   # [x, y, z, phi, theta, psi, xd, yd, zd, p, q, r]
IDX_POS  = slice(0, 3)    # x, y, z
IDX_ANG  = slice(3, 6)    # phi, theta, psi
IDX_VEL  = slice(6, 9)    # xdot, ydot, zdot
IDX_RATE = slice(9, 12)   # p, q, r  (body-frame angular rates)


class EKFNode(Node):
    """Extended Kalman Filter node for quadrotor state estimation."""

    def __init__(self):
        super().__init__('ekf_node')

        # ------------------------------------------------------------------ #
        # Declare ROS2 parameters (all tunable without recompiling)
        # ------------------------------------------------------------------ #
        self.declare_parameter('dt',          0.01)    # prediction period [s]
        self.declare_parameter('mass',        1.5)
        self.declare_parameter('gravity',     9.81)
        self.declare_parameter('ixx',         0.0347563)
        self.declare_parameter('iyy',         0.07)
        self.declare_parameter('izz',         0.0977)
        # Process noise (diagonal of Q)
        self.declare_parameter('q_pos',       0.01)
        self.declare_parameter('q_ang',       0.005)
        self.declare_parameter('q_vel',       0.1)
        self.declare_parameter('q_angvel',    0.05)
        # Measurement noise — odometry
        self.declare_parameter('r_odom_pos',  0.02)
        self.declare_parameter('r_odom_ang',  0.02)
        # Measurement noise — IMU
        self.declare_parameter('r_imu_acc',   0.3)
        self.declare_parameter('r_imu_gyro',  0.005)

        self._load_params()

        # ------------------------------------------------------------------ #
        # EKF state & covariance
        # ------------------------------------------------------------------ #
        self.x = np.zeros(N_STATES)       # spawn is at z=0 (ground)
        self.P = np.eye(N_STATES) * 0.1

        # Last known control input (updated via /control_wrench feedback)
        # Initialised to hover thrust so prediction is sensible from t=0
        self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])

        # Safety bounds for state clamping (prevents numerical blowup)
        self._x_max = np.array([
            50.0, 50.0, 50.0,              # position [m]
            np.pi, np.pi/2, np.pi,         # angles [rad]
            20.0, 20.0, 20.0,              # velocity [m/s]
            30.0, 30.0, 30.0,              # angular rate [rad/s]
        ])
        self._u_max = np.array([
            self.m * self.g * 4.0,          # max thrust = 4x hover
            5.0, 5.0, 2.0,                 # max torques [N·m]
        ])

        # ------------------------------------------------------------------ #
        # QoS profiles
        # ------------------------------------------------------------------ #
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ------------------------------------------------------------------ #
        # Subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(Imu,      '/imu',  self._imu_cb,  sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)
        # Listen to MPC output to use thrust/torque in prediction step
        self.create_subscription(
            Float64MultiArray, '/control_wrench', self._ctrl_cb, reliable_qos)

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self.state_pub = self.create_publisher(
            Odometry, '/state_estimate', reliable_qos)

        # ------------------------------------------------------------------ #
        # Prediction timer
        # ------------------------------------------------------------------ #
        self.create_timer(self.dt, self._predict)

        self.get_logger().info('EKF node started  (dt=%.3f s)' % self.dt)

    # ---------------------------------------------------------------------- #
    # Parameter loading
    # ---------------------------------------------------------------------- #
    def _load_params(self):
        g = self.get_parameter
        self.dt   = g('dt').value
        self.m    = g('mass').value
        self.g    = g('gravity').value
        self.Ixx  = g('ixx').value
        self.Iyy  = g('iyy').value
        self.Izz  = g('izz').value

        qp = g('q_pos').value
        qa = g('q_ang').value
        qv = g('q_vel').value
        qw = g('q_angvel').value
        self.Q = np.diag([qp, qp, qp, qa, qa, qa, qv, qv, qv, qw, qw, qw])

        rp = g('r_odom_pos').value
        ro = g('r_odom_ang').value
        self.R_odom = np.diag([rp, rp, rp, ro, ro, ro])

        ra = g('r_imu_acc').value
        rg = g('r_imu_gyro').value
        self.R_imu = np.diag([ra, ra, ra, rg, rg, rg])

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #
    def _ctrl_cb(self, msg: Float64MultiArray):
        """Store latest [T, tau_x, tau_y, tau_z] for prediction."""
        if len(msg.data) >= 4:
            u = np.array(msg.data[:4])
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                return  # reject corrupt wrench
            self.u[:] = np.clip(u, -self._u_max, self._u_max)
            self.u[0] = max(self.u[0], 0.0)  # thrust is non-negative

    def _imu_cb(self, msg: Imu):
        """EKF measurement update — IMU (accelerometer + gyroscope)."""
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z

        z_meas = np.array([ax, ay, az, gx, gy, gz])

        p, q, r = self.x[IDX_RATE]
        T       = self.u[0]

        # Predicted specific-force in body frame.
        # For a multirotor, the accelerometer measures F_non_gravity/m
        # = thrust/m along body-z, regardless of attitude.
        # ax ≈ 0, ay ≈ 0, az ≈ T/m  (exact for any orientation)
        acc_pred = np.array([0.0, 0.0, T / self.m])
        z_pred   = np.array([acc_pred[0], acc_pred[1], acc_pred[2], p, q, r])

        # Measurement Jacobian H (6 x 12)
        # Accelerometer ax,ay have no state dependence (always ≈ 0)
        # az = T/m depends on control input, not state → H[2,:] = 0
        # Gyroscope directly measures body rates
        H = np.zeros((6, N_STATES))
        H[3, 9]  = 1.0    # gx = p
        H[4, 10] = 1.0    # gy = q
        H[5, 11] = 1.0    # gz = r

        self._update(z_meas, z_pred, H, self.R_imu, angle_idx=[])

    def _odom_cb(self, msg: Odometry):
        """EKF measurement update — odometry (position + orientation)."""
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        rot  = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        # ZXY convention: first rotation yaw(z), then roll(x), then pitch(y)
        yaw, roll, pitch = rot.as_euler('zxy')

        z_meas = np.array([pos.x, pos.y, pos.z, roll, pitch, yaw])
        z_pred = np.array([self.x[0], self.x[1], self.x[2],
                           self.x[3], self.x[4], self.x[5]])

        # Linear measurement: H directly maps state → measurement
        H = np.zeros((6, N_STATES))
        H[0:6, 0:6] = np.eye(6)

        self._update(z_meas, z_pred, H, self.R_odom, angle_idx=[3, 4, 5])

    # ---------------------------------------------------------------------- #
    # EKF core
    # ---------------------------------------------------------------------- #
    def _predict(self):
        """EKF prediction step using nonlinear quadrotor dynamics."""
        x   = self.x.copy()
        phi, theta, psi = x[IDX_ANG]
        xd, yd, zd      = x[IDX_VEL]
        p,  q,  r       = x[IDX_RATE]
        T, tau_x, tau_y, tau_z = self.u

        R_wb = self._rot_matrix(phi, theta, psi)

        # --- Translational acceleration (inertial frame) ---
        thrust_body = np.array([0.0, 0.0, T / self.m])
        acc_inertial = R_wb @ thrust_body + np.array([0.0, 0.0, -self.g])

        # --- Euler-angle rates from body angular rates (ZXY kinematic eq.) ---
        ct, tt = np.cos(theta), np.tan(theta)
        cp, sp = np.cos(phi),   np.sin(phi)
        phi_dot   = p + (q * sp + r * cp) * tt
        theta_dot = q * cp - r * sp
        psi_dot   = (q * sp + r * cp) / (ct + 1e-9)

        # --- Angular acceleration (Euler equations, body frame) ---
        pdot = tau_x / self.Ixx + q * r * (self.Iyy - self.Izz) / self.Ixx
        qdot = tau_y / self.Iyy + p * r * (self.Izz - self.Ixx) / self.Iyy
        rdot = tau_z / self.Izz + p * q * (self.Ixx - self.Iyy) / self.Izz

        # --- Euler integration ---
        dt = self.dt
        x_new = x.copy()
        x_new[0:3]  += np.array([xd, yd, zd]) * dt
        x_new[3]    += phi_dot   * dt
        x_new[4]    += theta_dot * dt
        x_new[5]    += psi_dot   * dt
        x_new[6:9]  += acc_inertial * dt
        x_new[9]    += pdot * dt
        x_new[10]   += qdot * dt
        x_new[11]   += rdot * dt

        # --- Linearised Jacobian for covariance propagation ---
        F  = self._jacobian(x)
        Fd = np.eye(N_STATES) + F * dt

        # Guard: reset to safe state if NaN/inf crept in
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            self.get_logger().warn('EKF state diverged — resetting to zero')
            self.x = np.zeros(N_STATES)
            self.P = np.eye(N_STATES) * 0.1
            self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])
            self._publish()
            return

        # Clamp state to physical bounds
        self.x = np.clip(x_new, -self._x_max, self._x_max)

        P_new = Fd @ self.P @ Fd.T + self.Q
        # Symmetrise and cap covariance to prevent runaway
        P_new = 0.5 * (P_new + P_new.T)
        np.clip(P_new.diagonal(), 0.0, 1e4, out=P_new[np.diag_indices_from(P_new)])
        self.P = P_new

        self._publish()

    def _update(self, z_meas, z_pred, H, R, angle_idx):
        """Standard EKF update step."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        innov = z_meas - z_pred
        # Wrap angle innovations to [-pi, pi]
        for i in angle_idx:
            innov[i] = (innov[i] + np.pi) % (2 * np.pi) - np.pi

        self.x = self.x + K @ innov
        I_KH   = np.eye(N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form (numerically stable)

    def _jacobian(self, x):
        """Linearised dynamics Jacobian F = d(f)/dx at current state."""
        F = np.zeros((N_STATES, N_STATES))
        phi, theta = x[3], x[4]
        ct = np.cos(theta)

        # pos ← vel
        F[0:3, 6:9] = np.eye(3)

        # angle ← rate (simplified diagonal at small angles)
        F[3:6, 9:12] = np.eye(3)

        # vel ← angle  (linearised thrust coupling)
        F[6, 4] =  self.g * ct          # ẍ ≈ g·θ
        F[7, 3] = -self.g * np.cos(phi) # ÿ ≈ -g·φ

        return F

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _rot_matrix(phi, theta, psi):
        """ZXY rotation matrix: body → world."""
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi),   np.sin(psi)
        return np.array([
            [cy*ct - sy*sp*st, -sy*cp, cy*st + ct*sy*sp],
            [sy*ct + cy*sp*st,  cy*cp, sy*st - cy*ct*sp],
            [-cp*st,            sp,    cp*ct           ],
        ])

    def _publish(self):
        """Publish state estimate as nav_msgs/Odometry."""
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        x = self.x
        msg.pose.pose.position.x = x[0]
        msg.pose.pose.position.y = x[1]
        msg.pose.pose.position.z = x[2]

        # ZXY: yaw=x[5], roll=x[3], pitch=x[4]
        rot  = Rotation.from_euler('zxy', [x[5], x[3], x[4]])
        quat = rot.as_quat()  # [qx, qy, qz, qw]
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.twist.twist.linear.x  = x[6]
        msg.twist.twist.linear.y  = x[7]
        msg.twist.twist.linear.z  = x[8]
        msg.twist.twist.angular.x = x[9]
        msg.twist.twist.angular.y = x[10]
        msg.twist.twist.angular.z = x[11]

        self.state_pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
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

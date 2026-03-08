#!/usr/bin/env python3

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


N_STATES = 12
IDX_POS  = slice(0, 3)
IDX_ANG  = slice(3, 6)
IDX_VEL  = slice(6, 9)
IDX_RATE = slice(9, 12)


class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        self.declare_parameter('dt',          0.01)
        self.declare_parameter('mass',        1.5)
        self.declare_parameter('gravity',     9.81)
        self.declare_parameter('ixx',         0.0347563)
        self.declare_parameter('iyy',         0.07)
        self.declare_parameter('izz',         0.0977)
        self.declare_parameter('q_pos',       0.01)
        self.declare_parameter('q_ang',       0.005)
        self.declare_parameter('q_vel',       0.1)
        self.declare_parameter('q_angvel',    0.05)
        self.declare_parameter('r_odom_pos',  0.02)
        self.declare_parameter('r_odom_ang',  0.02)
        self.declare_parameter('r_imu_acc',   0.3)
        self.declare_parameter('r_imu_gyro',  0.005)

        self._load_params()

        self.x = np.zeros(N_STATES)
        self.P = np.eye(N_STATES) * 0.1

        self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])

        self._x_max = np.array([
            50.0, 50.0, 50.0,
            np.pi, np.pi/2, np.pi,
            20.0, 20.0, 20.0,
            30.0, 30.0, 30.0,
        ])
        self._u_max = np.array([
            self.m * self.g * 4.0,
            5.0, 5.0, 2.0,
        ])

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

        self.create_subscription(Imu,      '/imu',  self._imu_cb,  sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)
        self.create_subscription(
            Float64MultiArray, '/control_wrench', self._ctrl_cb, reliable_qos)

        self.state_pub = self.create_publisher(
            Odometry, '/state_estimate', reliable_qos)

        self.create_timer(self.dt, self._predict)

        self.get_logger().info('EKF node started  (dt=%.3f s)' % self.dt)

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

    def _ctrl_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            u = np.array(msg.data[:4])
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                return
            self.u[:] = np.clip(u, -self._u_max, self._u_max)
            self.u[0] = max(self.u[0], 0.0)

    def _imu_cb(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z

        z_meas = np.array([ax, ay, az, gx, gy, gz])

        p, q, r = self.x[IDX_RATE]
        T       = self.u[0]

        acc_pred = np.array([0.0, 0.0, T / self.m])
        z_pred   = np.array([acc_pred[0], acc_pred[1], acc_pred[2], p, q, r])

        H = np.zeros((6, N_STATES))
        H[3, 9]  = 1.0
        H[4, 10] = 1.0
        H[5, 11] = 1.0

        self._update(z_meas, z_pred, H, self.R_imu, angle_idx=[])

    def _odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        rot  = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        yaw, roll, pitch = rot.as_euler('zxy')

        z_meas = np.array([pos.x, pos.y, pos.z, roll, pitch, yaw])
        z_pred = np.array([self.x[0], self.x[1], self.x[2],
                           self.x[3], self.x[4], self.x[5]])

        H = np.zeros((6, N_STATES))
        H[0:6, 0:6] = np.eye(6)

        self._update(z_meas, z_pred, H, self.R_odom, angle_idx=[3, 4, 5])

    def _predict(self):
        x   = self.x.copy()
        phi, theta, psi = x[IDX_ANG]
        xd, yd, zd      = x[IDX_VEL]
        p,  q,  r       = x[IDX_RATE]
        T, tau_x, tau_y, tau_z = self.u

        R_wb = self._rot_matrix(phi, theta, psi)

        thrust_body = np.array([0.0, 0.0, T / self.m])
        acc_inertial = R_wb @ thrust_body + np.array([0.0, 0.0, -self.g])

        ct, tt = np.cos(theta), np.tan(theta)
        cp, sp = np.cos(phi),   np.sin(phi)
        phi_dot   = p + (q * sp + r * cp) * tt
        theta_dot = q * cp - r * sp
        psi_dot   = (q * sp + r * cp) / (ct + 1e-9)

        pdot = tau_x / self.Ixx + q * r * (self.Iyy - self.Izz) / self.Ixx
        qdot = tau_y / self.Iyy + p * r * (self.Izz - self.Ixx) / self.Iyy
        rdot = tau_z / self.Izz + p * q * (self.Ixx - self.Iyy) / self.Izz

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

        F  = self._jacobian(x)
        Fd = np.eye(N_STATES) + F * dt

        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            self.get_logger().warn('EKF state diverged — resetting to zero')
            self.x = np.zeros(N_STATES)
            self.P = np.eye(N_STATES) * 0.1
            self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])
            self._publish()
            return

        self.x = np.clip(x_new, -self._x_max, self._x_max)

        P_new = Fd @ self.P @ Fd.T + self.Q
        P_new = 0.5 * (P_new + P_new.T)
        np.clip(P_new.diagonal(), 0.0, 1e4, out=P_new[np.diag_indices_from(P_new)])
        self.P = P_new

        self._publish()

    def _update(self, z_meas, z_pred, H, R, angle_idx):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        innov = z_meas - z_pred
        for i in angle_idx:
            innov[i] = (innov[i] + np.pi) % (2 * np.pi) - np.pi

        self.x = self.x + K @ innov
        I_KH   = np.eye(N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def _jacobian(self, x):
        F = np.zeros((N_STATES, N_STATES))
        phi, theta = x[3], x[4]

        F[0:3, 6:9] = np.eye(3)
        F[3:6, 9:12] = np.eye(3)
        F[6, 4] =  self.g * np.cos(theta)
        F[7, 3] = -self.g * np.cos(phi)

        return F

    @staticmethod
    def _rot_matrix(phi, theta, psi):
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi),   np.sin(psi)
        return np.array([
            [cy*ct - sy*sp*st, -sy*cp, cy*st + ct*sy*sp],
            [sy*ct + cy*sp*st,  cy*cp, sy*st - cy*ct*sp],
            [-cp*st,            sp,    cp*ct           ],
        ])

    def _publish(self):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        x = self.x
        msg.pose.pose.position.x = x[0]
        msg.pose.pose.position.y = x[1]
        msg.pose.pose.position.z = x[2]

        rot  = Rotation.from_euler('zxy', [x[5], x[3], x[4]])
        quat = rot.as_quat()
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

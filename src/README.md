[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Mnj7rZ_g)

# LAB 2: Quadrotor UAV Control — MPC Hover and Trajectory Tracking

> **Group size:** 2–3 students

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Kinematics and Dynamics](#3-kinematics-and-dynamics)
4. [Controller Design](#4-controller-design)
5. [State Estimation](#5-state-estimation)
6. [Trajectory Design](#6-trajectory-design)
7. [Experimental Setup](#7-experimental-setup)
8. [Results](#8-results)
9. [Discussion and Analysis](#9-discussion-and-analysis)
10. [References](#10-references)

---

## 1. Overview

This laboratory implements a **Model Predictive Controller (MPC)** for a simulated quadrotor UAV to achieve stable hover and trajectory tracking in both disturbance-free and wind-disturbed environments. The simulation is built on **ROS 2** with **Ignition Gazebo Fortress (6.17.0)**, using the `quad_description` package to provide the physical robot model and world definitions.

The controller is designed as a **linearised unconstrained batch-form MPC** operating at 50 Hz. It stabilises the drone at a 1.0 m hover setpoint and subsequently tracks a series of 2D and 3D reference trajectories. An **Extended Kalman Filter (EKF)** runs at 100 Hz to fuse IMU and odometry measurements into a clean 12-state estimate consumed by the MPC.

Wind disturbance rejection is achieved through an **integral action** term appended to the MPC output, which eliminates the steady-state position offset caused by the constant 4 m/s wind load defined in `wind.sdf`.

### Objectives

| Part | Task | Environments |
|------|------|-------------|
| 1 | Stable hover at $z = 1.0$ m | No wind · Wind |
| 2 | 2D trajectory tracking (x-z plane) | No wind · Wind |
| 3 | 3D trajectory tracking | No wind · Wind |

### Software Stack

| Component | Version / Package |
|-----------|------------------|
| ROS 2 | Humble |
| Gazebo | Ignition Fortress 6.17.0 |
| Robot description | `quad_description` |
| Controller & EKF | `quad_controller` (this package) |
| State estimation | Custom EKF node |
| Bridge | `ros_gz_bridge` |

---

## 2. System Architecture

### 2.1 ROS 2 Node Graph

```
                     ┌─────────────────────────────────────┐
                     │           Ignition Gazebo            │
                     │  ┌──────────┐   ┌─────────────────┐ │
                     │  │  World   │   │   Quadrotor      │ │
                     │  │  (SDF)   │   │  (URDF/xacro)   │ │
                     └──┴──────────┴───┴────────┬────────┴─┘
                                                 │
                        /imu  (BEST_EFFORT)       │  /odom  (BEST_EFFORT)
                        ◄────────────────────────┘────────────────────────
                        │                                                  │
                ┌───────▼──────────────────────────────────────────────────▼───┐
                │                      ekf_node  (100 Hz)                      │
                │         Extended Kalman Filter — fuses IMU + odometry         │
                └───────────────────────┬──────────────────────────────────────┘
                                        │  /state_estimate  (RELIABLE)
                ┌───────────────────────▼──────────────────────────────────────┐
                │                   mpc_controller  (50 Hz)                    │
                │    Linearised MPC + Motor Allocation + Integral Action        │
                │    ┌──────────────────────────────────────────────────────┐  │
                │    │  /reference_state ◄── trajectory_generator  (50 Hz) │  │
                │    └──────────────────────────────────────────────────────┘  │
                └──────┬────────────────────────────┬─────────────────────────┘
                       │  /motor_commands            │  /mpc_debug
                       │  /control_wrench            │  /actual_path
                       ▼                             ▼
                   Gazebo Motors              data_collector
```

### 2.2 Topic Summary

| Topic | Message Type | QoS | Description |
|-------|-------------|-----|-------------|
| `/imu` | `sensor_msgs/Imu` | BEST_EFFORT | Raw accelerometer + gyroscope at ~400 Hz |
| `/odom` | `nav_msgs/Odometry` | BEST_EFFORT | Gazebo ground-truth pose at ~100 Hz |
| `/state_estimate` | `nav_msgs/Odometry` | RELIABLE | EKF 12-state output at 100 Hz |
| `/reference_state` | `std_msgs/Float64MultiArray` | RELIABLE | 12-state trajectory reference at 50 Hz |
| `/control_wrench` | `std_msgs/Float64MultiArray` | RELIABLE | Total wrench $[T,\tau_\phi,\tau_\theta,\tau_\psi]$ at 50 Hz |
| `/motor_commands` | `actuator_msgs/Actuators` | RELIABLE | Rotor speeds $[\omega_0,\omega_1,\omega_2,\omega_3]$ at 50 Hz |
| `/mpc_debug` | `std_msgs/Float64MultiArray` | RELIABLE | 19-field MPC internal telemetry at 50 Hz |

### 2.3 Package Structure

```
quad_controller/
├── config/
│   └── params.yaml          # All tunable parameters (gains, physical model)
├── launch/
│   └── controller.launch.py # Launches EKF + MPC + trajectory + data collector
├── scripts/
│   ├── ekf_node.py          # Extended Kalman Filter state estimator
│   ├── mpc_controller.py    # Linearised MPC with motor allocation
│   ├── trajectory_generator.py  # Reference trajectory publisher
│   └── data_collector.py    # Passive observer — CSVs + plots on shutdown
```

---

## 3. Kinematics and Dynamics

### 3.1 Reference Frames

Two frames are used throughout:

- **World frame** $\{W\}$: inertial, East-North-Up (ENU), fixed to the Gazebo origin.
- **Body frame** $\{B\}$: attached to the centre of mass of the quadrotor, $z$-axis pointing upward through the propeller disc plane.

### 3.2 State Vector

The 12-dimensional state vector is defined as:

$$\mathbf{x} = \begin{bmatrix} x & y & z & \phi & \theta & \psi & \dot{x} & \dot{y} & \dot{z} & p & q & r \end{bmatrix}^\top \in \mathbb{R}^{12}$$

where $(x, y, z)$ is the position in $\{W\}$, $(\phi, \theta, \psi)$ are the **ZXY Euler angles** (roll, pitch, yaw), $(\dot{x}, \dot{y}, \dot{z})$ are linear velocities in $\{W\}$, and $(p, q, r)$ are body-frame angular rates.

**ZXY convention:** The rotation matrix from body to world is $R = R_z(\psi)\,R_x(\phi)\,R_y(\theta)$, which yields the parametrisation used by `scipy.spatial.transform.Rotation.as_euler('zxy')`.

### 3.3 Rigid Body Dynamics

Applying Newton-Euler equations to the quadrotor:

**Translational dynamics (world frame):**

$$m\ddot{\mathbf{p}} = R\,\mathbf{f}_B - m g\,\hat{z}_W$$

where $\mathbf{f}_B = [0, 0, T]^\top$ is the total thrust in the body frame and $T = \sum_{i=0}^{3} T_i$.

**Rotational dynamics (body frame):**

$$\mathbf{I}\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega}$$

where $\mathbf{I} = \text{diag}(I_{xx}, I_{yy}, I_{zz})$ is the inertia tensor and $\boldsymbol{\tau} = [\tau_\phi, \tau_\theta, \tau_\psi]^\top$ is the body torque.

### 3.4 Motor Model

Each rotor $i$ produces:

$$T_i = k_F \omega_i^2, \qquad \tau_{d,i} = -d_i \, k_F \, k_M \, \omega_i^2$$

where $k_F = 8.549 \times 10^{-6}\ \text{N/(rad/s)}^2$ is the thrust coefficient, $k_M = 0.06$ is the dimensionless moment-to-thrust ratio (Gazebo `momentConstant`), and $d_i \in \{+1, -1\}$ is the spinning direction (CCW = $+1$, CW = $-1$).

The effective yaw coefficient is $\kappa = k_F k_M = 5.129 \times 10^{-7}\ \text{N}{\cdot}\text{m/(rad/s)}^2$.

### 3.5 Motor Layout and Allocation Matrix

Rotor positions and directions (from `quadrotor_base.xacro`):

| Rotor | Position $(r_x, r_y)$ [m] | Direction $d_i$ |
|-------|--------------------------|----------------|
| 0 (front-right) | $(+0.13,\ -0.22)$ | CCW ($+1$) |
| 1 (rear-left)   | $(-0.13,\ +0.20)$ | CCW ($+1$) |
| 2 (front-left)  | $(+0.13,\ +0.22)$ | CW  ($-1$) |
| 3 (rear-right)  | $(-0.13,\ -0.20)$ | CW  ($-1$) |

The wrench is related to squared rotor speeds by:

$$\begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} = \underbrace{\begin{bmatrix} k_F & k_F & k_F & k_F \\ k_F r_{y,0} & k_F r_{y,1} & k_F r_{y,2} & k_F r_{y,3} \\ -k_F r_{x,0} & -k_F r_{x,1} & -k_F r_{x,2} & -k_F r_{x,3} \\ -d_0\kappa & -d_1\kappa & -d_2\kappa & -d_3\kappa \end{bmatrix}}_{\mathbf{A}} \begin{bmatrix} \omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \end{bmatrix}$$

Motor speeds are recovered by $\boldsymbol{\omega}^2 = \mathbf{A}^{-1} \mathbf{u}$, where $\mathbf{u} = [T, \tau_\phi, \tau_\theta, \tau_\psi]^\top$.

### 3.6 Linearisation around Hover

At the hover equilibrium $\mathbf{x}_0 = \mathbf{0}_{12}$, $\mathbf{u}_0 = [mg, 0, 0, 0]^\top$, and under the small-angle assumption ($|\phi|, |\theta| \ll 1$), the nonlinear dynamics reduce to the continuous LTI system $\dot{\mathbf{x}} = A_c\,\mathbf{x} + B_c\,\mathbf{u}$:

$$A_c = \begin{bmatrix} \mathbf{0}_3 & \mathbf{0}_3 & I_3 & \mathbf{0}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & I_3 \\ \mathbf{0}_{3\times3}^* & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \end{bmatrix}, \quad B_c = \begin{bmatrix} \mathbf{0}_{6\times4} \\ B_{\text{acc}} \end{bmatrix}$$

The non-zero coupling entries in $A_c$ are:

$$\ddot{x} \approx g\,\theta, \qquad \ddot{y} \approx -g\,\phi$$

and $B_{\text{acc}} = \text{diag}(0, 0, 1/m, 1/I_{xx}, 1/I_{yy}, 1/I_{zz})$ (rows 7–12, columns $T, \tau_\phi, \tau_\theta, \tau_\psi$).

The system is discretised at $\Delta t = 0.02$ s using zero-order hold (ZOH):

$$\mathbf{x}_{k+1} = A_d\,\mathbf{x}_k + B_d\,\mathbf{u}_k$$

### 3.7 Assumptions

The following assumptions are made throughout this laboratory:

1. **Rigid body:** The quadrotor frame is perfectly rigid with no structural flexibility.
2. **Small angles:** The linearisation is valid only when $|\phi| < 15°$ and $|\theta| < 15°$. Violations degrade MPC performance.
3. **Quadratic thrust law:** Each rotor produces thrust strictly proportional to $\omega_i^2$ with no motor dynamics or delays.
4. **No body aerodynamic drag:** Translational drag on the airframe is neglected. Only rotor thrust and gravity act on the centre of mass.
5. **Constant wind:** In the wind experiment, the disturbance is a constant body-level force (Gazebo `WindEffects` plugin, 4 m/s in the $-y$ direction). Wind gusts and turbulence are not modelled.
6. **Decoupled yaw:** Yaw dynamics are controlled independently; MPC reference yaw is fixed at zero during all trajectory experiments to keep the linearisation valid.
7. **Inertia tensor is diagonal:** Off-diagonal products of inertia are assumed negligible.

---

## 4. Controller Design

### 4.1 MPC Problem Formulation

The MPC minimises a finite-horizon quadratic cost over a prediction horizon $N = 20$ steps at $\Delta t = 0.02$ s (0.4 s lookahead):

$$\min_{\mathbf{U}} \quad J = \sum_{k=0}^{N-1} \left[ (\mathbf{x}_k - \mathbf{x}_{\text{ref}})^\top Q\,(\mathbf{x}_k - \mathbf{x}_{\text{ref}}) + \mathbf{u}_k^\top R\,\mathbf{u}_k \right]$$

subject to $\mathbf{x}_{k+1} = A_d\,\mathbf{x}_k + B_d\,\mathbf{u}_k$, where $\mathbf{u}_k = [T - mg,\, \tau_\phi,\, \tau_\theta,\, \tau_\psi]^\top$ is the control perturbation from the hover equilibrium $\mathbf{u}_0 = [mg,0,0,0]^\top$.

The diagonal weight matrices penalise tracking error and control effort:

$$Q = \text{diag}(\underbrace{20,20,20}_{\text{position}},\ \underbrace{10,10,10}_{\text{attitude}},\ \underbrace{5,5,5}_{\text{velocity}},\ \underbrace{1,1,1}_{\text{ang. rate}})$$

$$R = \text{diag}(\underbrace{0.01}_{T},\ \underbrace{0.1,0.1,0.1}_{\tau_\phi,\tau_\theta,\tau_\psi})$$

Position is weighted highest to prioritise tracking; attitude and velocity provide damping; the low thrust weight allows large thrust excursions to maintain altitude. Torques are penalised 10× more than thrust because attitude aggressiveness is the primary source of linearisation breakdown.

### 4.2 Batch-Form Prediction Matrices

The predicted state sequence over the full horizon is expressed as a linear function of the current state $\mathbf{x}_0$ and the stacked control sequence $\mathbf{U} = [\mathbf{u}_0^\top, \ldots, \mathbf{u}_{N-1}^\top]^\top \in \mathbb{R}^{Nm}$:

$$\mathbf{X} = \Phi\,\mathbf{x}_0 + \Gamma\,\mathbf{U} \qquad \mathbf{X} = [\mathbf{x}_1^\top,\ldots,\mathbf{x}_N^\top]^\top \in \mathbb{R}^{Nn}$$

The **free-response matrix** $\Phi$ stacks powers of $A_d$ beginning at $A_d^1$ (block row $i$ contains $A_d^{i+1}$, for $i = 0,\ldots,N-1$):

$$\Phi = \begin{bmatrix} A_d \\ A_d^2 \\ \vdots \\ A_d^N \end{bmatrix} \in \mathbb{R}^{Nn \times n}$$

The **forced-response matrix** $\Gamma$ is lower-block-triangular. At zero-indexed block position $(i,j)$ with $i \geq j$:

$$\Gamma_{ij} = A_d^{i-j}\,B_d, \qquad \Gamma = \begin{bmatrix} B_d & 0 & \cdots & 0 \\ A_d B_d & B_d & \cdots & 0 \\ A_d^2 B_d & A_d B_d & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ A_d^{N-1}B_d & A_d^{N-2}B_d & \cdots & B_d \end{bmatrix} \in \mathbb{R}^{Nn \times Nm}$$

The lower-triangular structure captures causality: input at step $j$ only influences states at steps $j, j+1, \ldots, N-1$. Both $\Phi$ and $\Gamma$ are computed once at node startup using the ZOH-discretised matrices from `scipy.signal.cont2discrete`. The block cost matrices are $\bar{Q} = I_N \otimes Q$ and $\bar{R} = I_N \otimes R$.

### 4.3 Closed-Form Optimal Gain Computation

Substituting $\mathbf{X} = \Phi\mathbf{x}_0 + \Gamma\mathbf{U}$ into the quadratic cost and setting $\partial J/\partial \mathbf{U} = 0$:

$$\mathbf{U}^* = \underbrace{(\Gamma^\top\bar{Q}\,\Gamma + \bar{R})^{-1}}_{H^{-1}} \underbrace{\Gamma^\top\bar{Q}}_{G^\top Q} \left(\mathbf{X}_{\text{ref}} - \Phi\,\mathbf{x}_0\right)$$

Expanding the reference and state-dependent terms:

$$\mathbf{U}^* = \underbrace{H^{-1}G^\top Q}_{\text{ref gain}}\,\mathbf{X}_{\text{ref}} \underbrace{- H^{-1}G^\top Q\,\Phi}_{\text{state gain}}\,\mathbf{x}_0$$

By the **receding horizon principle**, only the first $m$ rows (first control input) are applied. The precomputed gains are:

$$K_r = \bigl[H^{-1}G^\top Q\bigr]_{0:m,\,:} \in \mathbb{R}^{m \times Nn} \qquad \text{(reference gain)}$$

$$K_x = -\bigl[H^{-1}G^\top Q\,\Phi\bigr]_{0:m,\,:} \in \mathbb{R}^{m \times n} \qquad \text{(state feedback gain)}$$

The **negative sign in $K_x$** arises from the sign reversal of the state term $-\Phi\mathbf{x}_0$ relative to the reference term $+\mathbf{X}_{\text{ref}}$. Once computed offline, the optimal perturbation control at every 50 Hz tick is:

$$\mathbf{u}_\delta = K_r\,\mathbf{X}_{\text{ref}} + K_x\,\mathbf{x}_0^h$$

where $\mathbf{X}_{\text{ref}} = \mathbf{1}_N \otimes \mathbf{x}_{\text{ref}}^h$ (reference tiled $N$ times) and $\mathbf{x}_0^h$ is the heading-frame state (Section 4.5). The online computation is two matrix-vector products — no QP solver is invoked — guaranteeing a **100% solve rate** and deterministic 50 Hz execution. The matrix inversion $H^{-1}$ is feasible because the problem is unconstrained; hard rotor limits are enforced post-hoc by the thrust-priority allocator.

### 4.4 Online Control Loop

Each 50 Hz control tick executes five sequential steps:

**Step 1 — Yaw frame rotation.**
The hover-point linearisation couples pitch to world $x$-acceleration and roll to world $y$-acceleration, valid only when body and world frames are aligned ($\psi = 0$). Both state and reference are rotated into the drone's heading frame to restore this alignment at arbitrary yaw:

$$\begin{bmatrix} x^h \\ y^h \end{bmatrix} = \underbrace{\begin{bmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{bmatrix}}_{R_z(-\psi)} \begin{bmatrix} x \\ y \end{bmatrix}, \qquad \begin{bmatrix} \dot{x}^h \\ \dot{y}^h \end{bmatrix} = R_z(-\psi) \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix}$$

The yaw state is zeroed ($\psi^h = 0$) and the yaw reference is set to the yaw error $\Delta\psi_{\text{ref}}$ (wrapped to $[-\pi,\pi]$). All other state components (altitude $z$, attitude $\phi,\theta$, angular rates $p,q,r$) are unchanged.

**Step 2 — MPC perturbation.**
The closed-form gain produces the optimal control deviation from the hover equilibrium:

$$\mathbf{u}_\delta = K_r\,\mathbf{X}_{\text{ref}} + K_x\,\mathbf{x}_0^h$$

**Step 3 — Hover feedforward.**
The gravity feedforward restores the true wrench:

$$\mathbf{u}_{\text{total}} = \mathbf{u}_\delta + \begin{bmatrix}mg \\ 0 \\ 0 \\ 0\end{bmatrix}$$

**Step 4 — Integral action** (if $z > 0.15$ m):
The wind-rejection integrator is updated and its correction added (see Section 4.7):

$$\mathbf{u}_{\text{total}} \mathrel{+}= \mathbf{u}_I$$

**Step 5 — Thrust-priority motor allocation.**
$\mathbf{u}_{\text{total}}$ is inverted to rotor speeds; if saturation occurs a binary search scales torques while preserving thrust (see Section 4.6). Rotor speeds $[\omega_0,\omega_1,\omega_2,\omega_3]$ are published to `/motor_commands`.

### 4.5 Yaw Compensation

The hover linearisation derives $\ddot{x} \approx g\theta$ and $\ddot{y} \approx -g\phi$ by expanding $R_{WB}\,[0,0,T/m]^\top$ at $\phi=\theta=\psi=0$. When the drone yaws by $\psi \neq 0$, the body $x$-axis no longer aligns with the world $x$-axis: pitch now accelerates along $(\cos\psi, \sin\psi)$ in the world plane, and roll along $(-\sin\psi, \cos\psi)$. Feeding world-frame position error directly into $K_r, K_x$ would therefore generate incorrect roll/pitch commands with heading-dependent cross-coupling.

By rotating both $\mathbf{x}_0$ and $\mathbf{x}_{\text{ref}}$ into the heading frame first, the MPC always operates in a virtual frame where the drone's nose points along $+x$. The output torques $[\tau_\phi,\tau_\theta]$ are body-frame quantities in that virtual frame, which coincides with the actual body frame — so no additional back-rotation is needed before commanding motors.

### 4.6 Motor Allocation with Thrust Priority

The allocation matrix $\mathbf{A}$ maps squared rotor speeds to wrench:

$$\boldsymbol{\omega}^2 = \mathbf{A}^{-1}\mathbf{u}_{\text{total}}, \qquad \omega_i = \sqrt{\max(\omega_i^2, 0)}$$

If all $\omega_i \leq \omega_{\max} = 1500$ rad/s the allocation is accepted. Otherwise, **thrust-priority allocation** proceeds:

1. Decompose $\mathbf{u}_T = [T, 0, 0, 0]^\top$, $\mathbf{u}_\tau = [0, \tau_\phi, \tau_\theta, \tau_\psi]^\top$.
2. If thrust alone saturates a rotor, scale $T$ to $0.95\,\omega_{\max}^2 k_F^{-1}$ and drop all torques ($\lambda = 0$).
3. Otherwise, **binary search** over $\lambda \in [0,1]$ for 20 iterations (accuracy $\approx 10^{-6}$):
   $$\lambda^* = \max\{\lambda : \boldsymbol{\omega}(\mathbf{u}_T + \lambda\,\mathbf{u}_\tau) \leq \omega_{\max}\}$$
   Apply $\mathbf{u}_{\text{final}} = \mathbf{u}_T + \lambda^*\,\mathbf{u}_\tau$.

The **torque scale** $\lambda^*$ (field `[12]` of `/mpc_debug`) indicates saturation severity: $\lambda^* = 1.0$ = full torque authority, $\lambda^* < 1.0$ = torques sacrificed to preserve altitude. Thrust is always protected because position control (altitude) takes priority over attitude tracking.

### 4.7 Integral Action for Wind Rejection

The proportional MPC has no internal model of constant disturbances: a persistent aerodynamic offset shifts the equilibrium point and manifests as a steady-state position error that the MPC cannot cancel. An **integral action** is therefore appended to accumulate the persistent error and generate a corrective bias.

**World-frame accumulation** (while airborne, $z > 0.15$ m):

$$\mathbf{e}_I[k+1] = \text{clip}\!\left(\mathbf{e}_I[k] + (\mathbf{p}_{\text{ref}} - \mathbf{p})\,\Delta t,\ -e_{\max},\ +e_{\max}\right), \qquad e_{\max} = 2.0\ \text{m}{\cdot}\text{s}$$

The anti-windup clamp prevents runaway accumulation during large setpoint transitions.

**Heading-frame rotation for $xy$:** The integral correction acts through roll and pitch, which are body-relative. The accumulated world-frame integral is first rotated into the heading frame:

$$e_{I,x}^h = \cos\psi\,e_{I,x} + \sin\psi\,e_{I,y}, \qquad e_{I,y}^h = -\sin\psi\,e_{I,x} + \cos\psi\,e_{I,y}$$

**Correction wrench:**

$$u_I^T = k_{i,z}\,e_{I,z}, \qquad u_I^{\tau_\theta} = k_{i,xy}\,e_{I,x}^h \quad(\text{pitch for }x), \qquad u_I^{\tau_\phi} = -k_{i,xy}\,e_{I,y}^h \quad(\text{roll for }y)$$

The negative sign for roll follows from the hover linearisation coupling $\ddot{y} \approx -g\phi$: a positive $y$-error (drone behind reference) requires a negative roll torque to tilt the drone toward $+y$. The complete applied wrench is:

$$\mathbf{u}_{\text{total}} = \underbrace{[mg,\,0,\,0,\,0]^\top}_{\text{hover feedforward}} + \underbrace{\mathbf{u}_\delta}_{\text{MPC perturbation}} + \underbrace{\mathbf{u}_I}_{\text{integral correction}}$$

### 4.8 Controller Parameters Summary

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Control period | $\Delta t$ | 0.02 s (50 Hz) |
| Prediction horizon | $N$ | 20 steps (0.4 s lookahead) |
| Position weight | $q_{\text{pos}}$ | 20.0 |
| Attitude weight | $q_{\text{ang}}$ | 10.0 |
| Velocity weight | $q_{\text{vel}}$ | 5.0 |
| Angular rate weight | $q_{\dot\omega}$ | 1.0 |
| Thrust input weight | $r_T$ | 0.01 |
| Torque input weight | $r_\tau$ | 0.1 |
| Integral gain (xy) | $k_{i,xy}$ | 2.0 |
| Integral gain (z) | $k_{i,z}$ | 3.0 |
| Integral clamp | $e_{\max}$ | 2.0 m·s |
| Max rotor speed | $\omega_{\max}$ | 1500 rad/s |
| Hover thrust | $mg$ | 14.715 N |

---

## 5. State Estimation

An **Extended Kalman Filter (EKF)** running at 100 Hz fuses IMU and odometry data into a clean 12-state estimate consumed by the MPC. The EKF uses **nonlinear dynamics for state propagation** but a **simplified linearised Jacobian for covariance propagation** — capturing the dominant couplings efficiently without computing the full nonlinear Jacobian.

### 5.1 Nonlinear State Prediction

At each 100 Hz prediction tick, the EKF propagates the full nonlinear equations of motion using the most recently received wrench $[T,\tau_\phi,\tau_\theta,\tau_\psi]$ from `/control_wrench`.

**Translational dynamics (world frame):**

$$\dot{\mathbf{p}} = \mathbf{v}$$

$$\dot{\mathbf{v}} = R_{WB}(\phi,\theta,\psi)\begin{bmatrix}0\\0\\T/m\end{bmatrix} + \begin{bmatrix}0\\0\\-g\end{bmatrix}$$

where the **ZXY rotation matrix** $R_{WB}$ transforms body-frame vectors to world frame (sequence: yaw $\psi$ first, then roll $\phi$, then pitch $\theta$):

$$R_{WB} = \begin{bmatrix} c_\psi c_\theta - s_\psi s_\phi s_\theta & -s_\psi c_\phi & c_\psi s_\theta + c_\theta s_\psi s_\phi \\ s_\psi c_\theta + c_\psi s_\phi s_\theta & \phantom{+}c_\psi c_\phi & s_\psi s_\theta - c_\psi c_\theta s_\phi \\ -c_\phi s_\theta & \phantom{+}s_\phi & c_\phi c_\theta \end{bmatrix}$$

using $c_\phi = \cos\phi$, $s_\phi = \sin\phi$, etc.

**Euler-angle kinematics (ZXY):**

$$\dot{\phi} = p + (q\sin\phi + r\cos\phi)\tan\theta$$
$$\dot{\theta} = q\cos\phi - r\sin\phi$$
$$\dot{\psi} = \frac{q\sin\phi + r\cos\phi}{\cos\theta}$$

These are the exact inverse kinematic equations for ZXY Euler parametrisation. A small-$\epsilon$ guard is applied to $\cos\theta$ to prevent singularity near $\theta = \pm90°$.

**Angular acceleration — Euler equations with gyroscopic coupling:**

$$\dot{p} = \frac{\tau_\phi}{I_{xx}} + \frac{(I_{yy}-I_{zz})}{I_{xx}}\,qr$$

$$\dot{q} = \frac{\tau_\theta}{I_{yy}} + \frac{(I_{zz}-I_{xx})}{I_{yy}}\,pr$$

$$\dot{r} = \frac{\tau_\psi}{I_{zz}} + \frac{(I_{xx}-I_{yy})}{I_{zz}}\,pq$$

The gyroscopic cross-product terms $(qr,\, pr,\, pq)$ couple the three angular rates. All equations are integrated with **forward Euler** at $\Delta t = 0.01$ s.

### 5.2 Linearised Jacobian for Covariance Propagation

Rather than computing the full nonlinear Jacobian $\partial f/\partial\mathbf{x}$ (which involves partial derivatives of $R_{WB}$ and gyroscopic terms), the EKF uses a **simplified Jacobian** retaining only the dominant linear couplings:

$$F = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\text{simplified}} = \begin{bmatrix} 0_3 & 0_3 & I_3 & 0_3 \\ 0_3 & 0_3 & 0_3 & I_3 \\ 0_3 & F_g & 0_3 & 0_3 \\ 0_3 & 0_3 & 0_3 & 0_3 \end{bmatrix}$$

where $F_g$ captures the linearised gravity-attitude coupling evaluated at the current angles:

$$F_g[6,4] = g\cos\theta \quad\Leftarrow\quad \ddot{x} \approx g\sin\theta \approx g\theta$$

$$F_g[7,3] = -g\cos\phi \quad\Leftarrow\quad \ddot{y} \approx -g\sin\phi \approx -g\phi$$

Note that the $I_3$ blocks for angle-rate coupling use the small-angle approximation $\dot{\phi}\approx p$, $\dot{\theta}\approx q$, $\dot{\psi}\approx r$, and the gyroscopic terms are omitted from $F$ (they are second-order in angular rate and negligible near hover). The discrete-time Jacobian is $F_d = I + F\Delta t$ and the covariance is propagated as:

$$P_{k|k-1} = F_d\,P_{k-1}\,F_d^\top + Q_{\text{proc}}$$

### 5.3 Measurement Update

Two sensors trigger independent asynchronous EKF updates:

**Odometry** (`/odom`, Gazebo ground-truth pose):

$$\mathbf{z}_{\text{odom}} = [x,\, y,\, z,\, \phi,\, \theta,\, \psi]^\top, \qquad H_{\text{odom}} = \begin{bmatrix} I_6 & 0_{6\times 6} \end{bmatrix}$$

This is a linear measurement model that directly observes all 6 pose states. Angle innovations are wrapped to $[-\pi,\pi]$ to handle wrap-around.

**IMU** (`/imu`, accelerometer + gyroscope):

The accelerometer in a multirotor measures the **specific force in the body frame** — the net non-gravitational force per unit mass. Near hover, this is dominated by the rotor thrust:

$$\mathbf{a}_{\text{pred}} = \begin{bmatrix}0 \\ 0 \\ T/m\end{bmatrix} \quad (\text{body frame, exact for any attitude})$$

Because this prediction depends on the control input $T$ rather than any state, the accelerometer Jacobian rows are zero: $H_{\text{imu}}[0:3,\,:] = 0$. The Kalman gain for those rows vanishes and the accelerometer innovation does not update the state. Only the **gyroscope rows** have non-zero Jacobian entries:

$$H_{\text{imu}}[3,9] = H_{\text{imu}}[4,10] = H_{\text{imu}}[5,11] = 1 \quad \Rightarrow \quad \mathbf{z}_{\text{gyro}} = [p,\, q,\, r]^\top$$

This design directly drives the angular rate states $\mathbf{x}[9:12]$ via gyroscope feedback, keeping body-rate estimation accurate even during dynamic manoeuvres.

**Joseph-form covariance update** (applied to both sensors):

$$K = P_{k|k-1}\,H^\top\,S^{-1}, \qquad S = H\,P_{k|k-1}\,H^\top + R$$

$$\mathbf{x}_k = \mathbf{x}_{k|k-1} + K\,\boldsymbol{\nu}$$

$$P_k = (I - KH)\,P_{k|k-1}\,(I - KH)^\top + K\,R\,K^\top$$

The **Joseph form** is used instead of the standard $P_k = (I-KH)P_{k|k-1}$ to guarantee $P_k$ remains symmetric positive semi-definite under floating-point accumulation errors, providing long-term numerical stability.

### 5.4 EKF Noise Parameters

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Prediction period | $\Delta t_{\text{EKF}}$ | 0.01 s (100 Hz) | 2× MPC rate for smoother estimate |
| Position process noise | $q_{\text{pos}}$ | 0.01 | Low: trust kinematic model |
| Attitude process noise | $q_{\text{ang}}$ | 0.005 | Low: angles change slowly |
| Velocity process noise | $q_{\text{vel}}$ | 0.1 | Higher: allows unmodelled accelerations |
| Ang. rate process noise | $q_{\dot\omega}$ | 0.05 | Moderate: gyroscopic effects |
| Odom position noise $\sigma$ | $r_{\text{odom,pos}}$ | 0.02 m | Near ground-truth Gazebo odom |
| Odom angle noise $\sigma$ | $r_{\text{odom,ang}}$ | 0.02 rad | Near ground-truth Gazebo odom |
| IMU accel noise $\sigma$ | $r_{\text{acc}}$ | 0.3 m/s² | High → effectively disables accel update |
| IMU gyro noise $\sigma$ | $r_{\text{gyro}}$ | 0.005 rad/s | Low → gyro dominates rate estimate |

### 5.5 Physical Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Mass | $m$ | 1.5 kg |
| Gravity | $g$ | 9.81 m/s² |
| Roll inertia | $I_{xx}$ | $3.476 \times 10^{-2}$ kg·m² |
| Pitch inertia | $I_{yy}$ | $7.0 \times 10^{-2}$ kg·m² |
| Yaw inertia | $I_{zz}$ | $9.77 \times 10^{-2}$ kg·m² |
| Thrust coefficient | $k_F$ | $8.549 \times 10^{-6}$ N/(rad/s)² |
| Moment coefficient | $k_M$ | $0.06$ (dimensionless) |
| Max rotor speed | $\omega_{\max}$ | 1500 rad/s |
| Hover rotor speed | $\omega_{\text{hov}}$ | $\approx 656$ rad/s |

---

## 6. Trajectory Design

All trajectories follow a three-phase sequence:

```
PRE_HOVER (5 s) ──► FLYING (traj_duration) ──► POST_HOVER
    stabilise              track reference           return
```

Reference yaw is fixed at $\psi_{\text{ref}} = 0$ for all trajectories to preserve MPC linearisation validity.

### 6.1 Part 2 — 2D Trajectories (x-z plane, $y = 0$)

#### Straight Line (`straight_2d`)

$$x(t) = v\,t, \quad y = 0, \quad z = z_0, \quad \dot{x} = v$$

#### Sine Wave (`sine_2d`)

$$x(t) = v\,t, \quad y = 0, \quad z(t) = z_0 + A\sin(\omega t), \quad \dot{x} = v, \quad \dot{z} = A\omega\cos(\omega t)$$

#### Staircase (`step_2d`)

$$x(t) = v\,t, \quad y = 0, \quad z(t) = z_0 + A\,\lfloor f\,t \rfloor \cdot 0.5 \mod 2A$$

Step height $A$, step frequency $f$ Hz. Provides a piecewise-constant altitude command that tests step-response rejection.

### 6.2 Part 3 — 3D Trajectories

#### Straight Line (`straight_3d`)

Direction vector $\hat{d} = [1, 0.5, 0.3]^\top$ (normalised):

$$\mathbf{p}(t) = [0, 0, z_0]^\top + \hat{d}\,v\,t, \qquad \dot{\mathbf{p}}(t) = \hat{d}\,v$$

#### Helix (`helix`)

Constant-radius helical spiral, phase-shifted to start at the origin:

$$x(t) = R\sin(\omega t), \quad y(t) = R(1-\cos(\omega t)), \quad z(t) = z_0 + c\,t$$

$$\dot{x} = R\omega\cos(\omega t), \quad \dot{y} = R\omega\sin(\omega t), \quad \dot{z} = c$$

with $R = 0.5$ m, $f = 0.2$ Hz, $c = 0.15$ m/s.

#### Conical Spiral (`cone_helix`)

Expanding-radius helix that forms a cone shape. Radius grows linearly with time at rate $\dot{R} = R_{\max}\,f$ (where $R_{\max}$ = `traj_radius`, $f$ = `traj_frequency`):

$$R(t) = R_{\max}\,f\,t, \quad x(t) = R(t)\sin(\omega t), \quad y(t) = R(t)(1-\cos(\omega t)), \quad z(t) = z_0 + c\,t$$

$$\dot{x} = \dot{R}\sin(\omega t) + R(t)\,\omega\cos(\omega t), \quad \dot{y} = \dot{R}(1-\cos(\omega t)) + R(t)\,\omega\sin(\omega t), \quad \dot{z} = c$$

where $\omega = 2\pi f$ and $\dot{R} = R_{\max}\,f$ is constant. With defaults $R_{\max}=0.5$ m, $f=0.2$ Hz: after the first revolution ($T=5$ s) the radius is $0.5$ m; after the second it is $1.0$ m — producing a conical path that starts at the hover point and expands outward.

---

## 7. Experimental Setup

### 7.1 Test Environments

| Environment | World File | Wind |
|-------------|-----------|------|
| No wind | `empty.sdf` | None |
| Wind | `wind.sdf` | 4 m/s in $-y$ direction (`WindEffects` plugin) |

### 7.2 Run Matrix

| Part | Trajectory | No wind | Wind |
|------|-----------|---------|------|
| 1 | hover | ✓ | ✓ |
| 2 | straight\_2d | ✓ | ✓ |
| 2 | sine\_2d | ✓ | ✓ |
| 2 | step\_2d | ✓ | ✓ |
| 3 | straight\_3d | ✓ | ✓ |
| 3 | helix | ✓ | ✓ |
| 3 | cone\_helix | ✓ | ✓ |

### 7.3 Data Collection

A dedicated `data_collector` node subscribes passively to all control and state topics and on shutdown produces:

- `state_data.csv` — time-aligned position, velocity, attitude (world frame) with tracking errors
- `control_data.csv` — wrench, motor speeds, and 19-field MPC debug telemetry
- `metrics.csv` — scalar summary (RMSE, settling time, motor utilisation, MPC cost, etc.)
- 11 diagnostic plots (position tracking, velocity tracking, 3D trajectory, position error, control wrench, motor speeds, Euler angles, motor utilisation histogram, MPC cost decomposition, wrench decomposition, torque scale, integral error)

### 7.4 Launch Commands

```bash
# Source workspace (run in every terminal)
cd ~/Mobile_Robot && source install/setup.bash

# Terminal 1 — Simulation
ros2 launch quad_description sim.launch.py world:=empty   # or world:=wind

# Terminal 2 — Controller + data collection
ros2 launch quad_controller controller.launch.py \
    trajectory_type:=<traj>     \
    hover_time:=5.0              \
    traj_duration:=30.0          \
    wind_y:=-4.0                 \   # wind runs only
    collect_data:=true           \
    collect_duration:=40.0
```

---

## 8. Results

> **Note:** This section will be populated with experimental data collected from the 14 simulation runs described in Section 7.2. Each subsection follows the structure below.

### 8.1 Part 1 — Hover

#### 8.1.1 Without Wind

*(Insert: z position vs. time, settling time, steady-state error, motor speeds. Include `position_tracking.png`, `euler_angles.png`.)*

| Metric | Value |
|--------|-------|
| Settling time (2%, $z = 1.0$ m) | — s |
| Steady-state position error (3D) | — m |
| Max $\|\phi\|$, $\|\theta\|$ | —°, —° |
| Mean motor utilisation | —% |

#### 8.1.2 With Wind (4 m/s −y)

*(Insert: same plots. Highlight integral action effect — compare z and y error before/after accumulation saturates.)*

| Metric | No wind | Wind |
|--------|---------|------|
| SS error $y$ [m] | — | — |
| Integral $\|e_I\|$ final [m·s] | — | — |
| Drift suppressed | — | — |

---

### 8.2 Part 2 — 2D Trajectory Tracking

*(For each trajectory: `position_tracking.png`, `velocity_tracking.png`, `position_error.png`. Tabulate RMSE.)*

#### 8.2.1 Straight Line

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE $x$ [m] | — | — |
| RMSE $z$ [m] | — | — |
| Max 3D error [m] | — | — |

#### 8.2.2 Sine Wave

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE $x$ [m] | — | — |
| RMSE $z$ [m] | — | — |
| RMSE $\dot{z}$ [m/s] | — | — |

#### 8.2.3 Staircase

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE $z$ [m] | — | — |
| Mean step rise time [s] | — | — |

---

### 8.3 Part 3 — 3D Trajectory Tracking

*(For each trajectory: `3d_trajectory.png`, `position_tracking.png`, `velocity_tracking.png`, `euler_angles.png`, `mpc_cost.png`. Tabulate RMSE.)*

#### 8.3.1 Straight Line 3D

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE 3D [m] | — | — |
| Max $\|\phi\|$, $\|\theta\|$ | —°, —° | —°, —° |

#### 8.3.2 Helix

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE 3D [m] | — | — |
| Max roll/pitch [°] | — | — |

#### 8.3.3 Conical Spiral

| Metric | No wind | Wind |
|--------|---------|------|
| RMSE 3D [m] | — | — |
| Max radius reached [m] | — | — |

---

### 8.4 MPC Internal Metrics

*(Insert: `mpc_cost.png`, `wrench_decomposition.png`, `torque_scale.png`, `integral_error.png` for representative runs.)*

| Metric | Hover | Helix | Cone |
|--------|-------|-------|------|
| Solve success rate [%] | — | — | — |
| Mean state cost $\bar{J}_x$ | — | — | — |
| Mean input cost $\bar{J}_u$ | — | — | — |
| Torque scale events [%] | — | — | — |
| RMS MPC thrust Δ [N] | — | — | — |
| RMS integral thrust [N] | — | — | — |

---

## 9. Discussion and Analysis

> **Note:** This section will be written after experimental data is collected. The outline below identifies the key discussion points.

### 9.1 Hover Performance

*(Discuss: How quickly does the drone reach z = 1.0 m? Does it overshoot? Are roll/pitch angles within the ±15° linearisation bound? Compare no-wind vs. wind cases: what is the steady-state y-offset without integral action? How much does the integral action reduce this?)*

### 9.2 Linearisation Validity

*(Discuss: The MPC is derived under a small-angle assumption. Report the percentage of flight time within ±15°. For trajectories that push the boundary (helix, cone), does tracking degrade? What does the MPC cost trajectory reveal about the periods where linearisation is strained?)*

### 9.3 Trajectory Tracking Quality

*(Discuss: Compare RMSE across trajectory types. Explain why 3D trajectories have higher error than 2D. For the helix, lateral acceleration demand from circular motion requires simultaneous roll and pitch, which the linearised model handles less accurately. The cone_helix exacerbates this as R grows. Velocity tracking errors should be discussed alongside position errors — does the MPC correctly anticipate reference velocity changes?)*

### 9.4 Wind Disturbance Rejection

*(Discuss: Without integral action (ki_xy = ki_z = 0), the proportional MPC alone cannot reject a constant aerodynamic offset — explain why (MPC has no internal model of constant disturbance). With integral action, show convergence of the integral accumulator to a steady value and corresponding reduction of the position offset. Quantify rejection quality using the signed steady-state offset.)*

### 9.5 MPC Wrench Decomposition

*(Discuss: The wrench decomposition plot separates hover feedforward, MPC perturbation, and integral contribution. During straight trajectories, the MPC Δ dominates. During wind runs, integral contribution grows to compensate. During agile 3D trajectories, examine whether torque scaling (`torque_scale < 1`) occurs — if so, motor saturation is reached and torque authority is reduced to protect thrust.)*

### 9.6 Closed-Form MPC vs. Iterative Solvers

*(Discuss: The precomputed gain approach guarantees 100% solve rate and constant 50 Hz execution with negligible online computation (two matrix-vector multiplies per tick). This contrasts with nonlinear MPC or QP-based constrained MPC, which carry a risk of solver timeout or infeasibility. The trade-off is the restriction to the linearisation domain and the absence of hard input constraints — saturation is handled heuristically by the thrust-priority allocator.)*

### 9.7 Limitations and Future Work

- The linearisation is valid only for near-hover attitudes. Agile manoeuvres exceeding ±15° roll/pitch will degrade tracking accuracy.
- Wind is modelled as a constant body-level force. A real aerodynamic model including induced drag and gust would require a more sophisticated disturbance observer.
- The integral action does not discriminate between wind disturbance and integral windup from trajectory transitions. A state-space disturbance observer (e.g., extended state observer) would provide faster and more principled rejection.
- The receding horizon length of 20 steps (0.4 s) may be insufficient for fast trajectories; a longer horizon or trajectory preview could improve tracking.

---

## 10. References

1. Coursera Robotics Specialization: Aerial Robotics — Prof. Vijay Kumar, University of Pennsylvania.
2. Lab material: [Formulation](./material/2C-1-Formulation.pdf) — Coordinate systems, motor model, rotation matrix, forces and moments.
3. Lab material: [Quadrotor Equations of Motion](./material/2C-4-Quadrotor-Equations-of-Motion.pdf) — Newton-Euler equations, 2D and 3D models.
4. Maciejowski, J. M. (2002). *Predictive Control with Constraints*. Prentice Hall.
5. Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill Publishing.
6. Welch, G., & Bishop, G. (1995). An Introduction to the Kalman Filter. UNC-Chapel Hill Technical Report TR 95-041.

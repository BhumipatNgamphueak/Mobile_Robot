# FRA532 Mobile Robotics – Lab 1: EKF / ICP / SLAM

**Student:** Bhumipat Ngamphueak
**ID:** 66340500043
**Course:** FRA532 Mobile Robotics
**Lab:** Lab 1 – EKF Odometry Fusion, ICP Refinement, and Full SLAM

---

## Table of Contents

1. [Setup](#1-setup)
2. [What We Do](#2-what-we-do)
   - [Differential Drive Model & Wheel Odometry](#21-differential-drive-model--wheel-odometry)
   - [Extended Kalman Filter (EKF)](#22-extended-kalman-filter-ekf)
   - [ICP Odometry Refinement](#23-icp-odometry-refinement)
   - [SLAM with slam_toolbox](#24-slam-with-slam_toolbox)
3. [Results](#3-results)
   - [Configuration Definition](#configuration-definition)
   - [Part 1 – EKF Odometry Fusion](#31-part-1--ekf-odometry-fusion)
   - [Part 2 – ICP Odometry Refinement](#32-part-2--icp-odometry-refinement)
   - [Part 3 – Full SLAM](#33-part-3--full-slam-with-slam_toolbox)
   - [Overall Comparison](#34-overall-comparison)

---

## 1. Setup

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble ([official install guide](https://docs.ros.org/en/humble/Installation.html))
- Python 3.10+
- TurtleBot3 Burger

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/BhumipatNgamphueak/Mobile_Robot.git -b Lab1
cd ~/Mobile_Robot
```

**2. Install ROS 2 Dependencies**
```bash
sudo apt update
sudo apt install -y ros-humble-slam-toolbox ros-humble-nav2-map-server
rosdep install --from-paths src --ignore-src -r -y
```

**3. Install Python Dependencies**
```bash
pip3 install numpy scipy pandas matplotlib pillow pyyaml
```

**4. Verify the Dataset**

The rosbag files are included in the repository under `src/FRA532_LAB1_DATASET/`:
```
src/FRA532_LAB1_DATASET/
├── fibo_floor3_seq00/
│   └── fibo_floor3_seq00_0.db3   (16 MB)
├── fibo_floor3_seq01/
│   └── fibo_floor3_seq01_0.db3   (12 MB)
└── fibo_floor3_seq02/
    └── fibo_floor3_seq02_0.db3   (19 MB)
```

> The default bag used when no `bag_path` argument is given is `fibo_floor3_seq02_0.db3`.

**5. Build the Workspace**
```bash
cd ~/Mobile_Robot
colcon build --symlink-install
```

**6. Source the Environment**
```bash
source install/setup.bash
```

**7. (Optional) Auto-source on every new terminal**
```bash
echo "source ~/Mobile_Robot/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Running the Experiments

You need **two terminals** for each part.

**Part 1 – EKF Odometry Fusion**
```bash
# Terminal 1
cd ~/Mobile_Robot && source install/setup.bash
ros2 launch ekf_filter part1_ekf_fusion.launch.py
```
Outputs saved to `results/`: `wheel_odometry.csv`, `ekf_odometry.csv`, `imu_data.csv`

**Part 2 – ICP Odometry Refinement**
```bash
# Terminal 1
cd ~/Mobile_Robot && source install/setup.bash
ros2 launch ekf_filter part2_icp_refinement.launch.py
```
Outputs saved to `results/`: `icp_odometry.csv`, `icp_map.pgm`, `icp_map.yaml`

**Part 3 – Full SLAM**
```bash
# Terminal 1 – without visualization
cd ~/Mobile_Robot && source install/setup.bash
ros2 launch ekf_filter part3_slam.launch.py

# Terminal 1 – with RViz2
cd ~/Mobile_Robot && source install/setup.bash
ros2 launch ekf_filter part3_slam_with_rviz.launch.py
```
Outputs saved to `results/`: `slam_trajectory.csv`, `slam_map.pgm`, `slam_map.yaml`

**Generate All Comparison Plots**
```bash
cd ~/Mobile_Robot
python3 scripts/plot_all_results.py
```

> **Switching datasets:** To use a different sequence, pass the `bag_path` parameter:
> ```bash
> ros2 launch ekf_filter part1_ekf_fusion.launch.py \
>     bag_path:=$HOME/Mobile_Robot/src/FRA532_LAB1_DATASET/fibo_floor3_seq01/fibo_floor3_seq01_0.db3
> ```

### Repository Structure

```
Mobile_Robot/
├── src/
│   ├── FRA532_LAB1_DATASET/               # Dataset rosbags (not tracked by git)
│   │   ├── fibo_floor3_seq00/
│   │   ├── fibo_floor3_seq01/
│   │   └── fibo_floor3_seq02/
│   ├── differential_drive_model/
│   │   └── scripts/
│   │       ├── read_data_node.py          # Rosbag replay + timestamp re-sync
│   │       └── wheel_odometry_node.py     # Dead-reckoning from wheel encoders
│   └── ekf_filter/
│       ├── config/
│       │   ├── mapper_params_online_async_B.yaml    # SLAM Config B (strict, active)
│       │   └── mapper_params_online_async_saved_not_use.yaml  # SLAM Config A (relaxed)
│       ├── launch/
│       │   ├── part1_ekf_fusion.launch.py
│       │   ├── part2_icp_refinement.launch.py
│       │   ├── part3_slam.launch.py
│       │   └── part3_slam_with_rviz.launch.py
│       └── scripts/
│           ├── part1/
│           │   ├── ekf_odometry_node.py   # 5-state EKF (wheel + IMU fusion)
│           │   └── wheel_odometry_node.py
│           ├── part2/
│           │   └── icp_odometry_node.py   # ICP scan matching + adaptive blending
│           ├── part3/
│           │   └── scan_republisher_node.py
│           ├── auto_map_saver_node.py
│           ├── icp_map_builder_node.py
│           ├── slam_trajectory_saver_node.py
│           └── read_data_node.py
├── scripts/
│   ├── plot_all_results.py                # All trajectory plots + metrics
│   ├── plot_parts.py                      # Part 1 and Part 2 focused comparisons
│   ├── plot_icp_vs_slam_maps.py           # ICP map vs SLAM map comparisons
│   └── plot_compare_drift.py              # compare_drift run analysis
└── results/                               # All output files (auto-created)
    ├── sequence_0/Config_A/ and Config_B/
    ├── sequence_1/Config_A/ and Config_B/
    └── sequence_2/Config_A/ and Config_B/
```

---

## 2. What We Do

This lab implements a 2D localization pipeline with four progressive methods, each adding more sensor information and algorithmic sophistication.

| Part | Method | Sensors |
|------|--------|---------|
| 1 | Wheel Odometry (baseline) | `/joint_states` |
| 1 | EKF Odometry Fusion | `/joint_states` + `/imu` |
| 2 | ICP Odometry Refinement | `/scan` (EKF as initial guess) |
| 3 | Full SLAM | `/scan` + EKF odometry |

---

### 2.1 Differential Drive Model & Wheel Odometry

The baseline method integrates wheel encoder ticks using the differential-drive kinematic model.

**Robot geometry (TurtleBot3 Burger):**

| Parameter | Value |
|-----------|-------|
| Wheel radius (`r`) | 0.033 m |
| Wheel separation (`L`) | 0.160 m |

**Velocity computation from encoder ticks:**
```
v     = r/2 * (dφ_R + dφ_L) / dt
omega = r/L * (dφ_R - dφ_L) / dt
```

**Dead-reckoning integration** (Euler, 20 Hz):
```
x_new = x + v·cos(θ)·dt
y_new = y + v·sin(θ)·dt
θ_new = θ + omega·dt
```

This accumulates unbounded error over time, especially in heading, because wheel slip and small calibration errors in `r` and `L` are never corrected.

---

### 2.2 Extended Kalman Filter (EKF)

The EKF fuses wheel encoder data with the IMU to reduce drift, particularly heading drift from the gyroscope.
**Code:** `src/ekf_filter/scripts/part1/ekf_odometry_node.py`

#### State Vector

```
state = [x, y, θ, v, ω]          # shape (5,)
```

| Index | Variable | Meaning | Unit |
|-------|----------|---------|------|
| 0 | `x` | position east | m |
| 1 | `y` | position north | m |
| 2 | `θ` | heading (yaw) | rad |
| 3 | `v` | linear velocity | m/s |
| 4 | `ω` | angular velocity | rad/s |

Initial covariance `P = 0.1 · I₅`.

---

#### Process Model — Prediction Step (20 Hz, triggered by `/joint_states`)

Called inside `joint_states_callback → ekf_predict(dt)`.
Uses the **differential-drive motion model**:

For curved motion `|ω| > 1×10⁻⁶ rad/s`:
```
x  ← x + (v/ω)·(sin(θ + ω·dt) − sin(θ))
y  ← y + (v/ω)·(−cos(θ + ω·dt) + cos(θ))
θ  ← θ + ω·dt
v, ω unchanged (velocities are updated by measurement)
```

For straight-line motion `|ω| ≈ 0`:
```
x  ← x + v·cos(θ)·dt
y  ← y + v·sin(θ)·dt
θ  unchanged
```

Covariance prediction:
```
P ← F · P · Fᵀ + Q
```

`F` is the **5×5 Jacobian** of the motion model (analytically derived in `ekf_predict`):
- `F[0,2]`, `F[1,2]` = ∂x/∂θ, ∂y/∂θ
- `F[0,3]`, `F[1,3]` = ∂x/∂v, ∂y/∂v
- `F[0,4]`, `F[1,4]`, `F[2,4]` = ∂x/∂ω, ∂y/∂ω, ∂θ/∂ω (only for curved case)

**Process noise** (tuned from sensor calibration):
```python
Q = diag([0.01,   # x position noise  (m²)
          0.01,   # y position noise  (m²)
          0.01,   # θ heading noise   (rad²)
          0.010,  # v velocity noise  (m/s)²
          0.01])  # ω angular noise   (rad/s)²
```

---

#### Measurement Models — Update Steps

**Update 1 – Wheel Odometry** `[v_odom, ω_odom]` (20 Hz, `/joint_states`)

```
d_left  = (Δφ_L) · r           r = 0.033 m
d_right = (Δφ_R) · r
v_odom  = (d_right + d_left) / (2·dt)
ω_odom  = (d_right − d_left) / (L·dt)   L = 0.160 m

z = [v_odom, ω_odom]
H = [[0, 0, 0, 1, 0],    # measures state[3] = v
     [0, 0, 0, 0, 1]]    # measures state[4] = ω

R_odom = diag([0.0005, 0.0031])   # (m/s)², (rad/s)²
```

**Update 2 – IMU Gyroscope** `ω_z` (20 Hz, `/imu`)

```
z = [ω_z − gyro_bias_z]
H = [[0, 0, 0, 0, 1]]    # measures state[4] = ω

R_gyro = diag([0.0030])   # (rad/s)²
```
Applied unconditionally every IMU callback.

**Update 3 – IMU Accelerometer** `a_x` (20 Hz, `/imu`, only when `|a_x| > 0.05 m/s²`)

```
v_from_accel = state[3] + (a_x − bias_x) · dt
z = [v_from_accel]
H = [[0, 0, 0, 1, 0]]    # measures state[3] = v

R_accel = diag([0.1907])  # (m/s²)²
```
Integrates forward acceleration to refine linear velocity estimate.

**Update 4 – Centripetal Acceleration** `a_y` (20 Hz, `/imu`, only when `|v| > 0.05` and `|a_y| > 0.02`)

```
ω_centripetal = (a_y − bias_y) / v      # from: a_y = v·ω
z = [ω_centripetal]
H = [[0, 0, 0, 0, 1]]    # measures state[4] = ω

R_centripetal = diag([2.0])    # high noise — indirect measurement
```
Uses lateral acceleration during turns to provide an independent ω estimate.

All four updates follow the standard EKF correction:
```
y = z − H·state                                       # innovation
S = H·P·Hᵀ + R                                       # innovation covariance
K = P·Hᵀ·S⁻¹                                         # Kalman gain
state ← state + K·y
P     ← (I − K·H)·P
```

---

#### Outlier Rejection (Mahalanobis Gating)

Every update is guarded by `mahalanobis_gate(innovation, S, threshold)`:
```
d² = yᵀ · S⁻¹ · y
if d² > threshold → reject measurement (no state update)
```
Threshold = 2,000,000 (effectively disabled — kept for safety only).

---

#### IMU Bias Calibration

The first **75 IMU samples** (robot must be stationary at start) are averaged:
```python
accel_bias_x = mean(accel_samples_x[:75])
accel_bias_y = mean(accel_samples_y[:75])
```
After calibration completes, all accelerometer readings are corrected by subtracting the bias before measurement updates.

---

#### Hyperparameters

| Parameter | Value | Where in Code |
|-----------|-------|--------------|
| `Q` | diag([0.01]×5) | `self.Q` in `__init__` |
| `R_odom` | diag([0.0005, 0.0031]) | `self.R_odom` |
| `R_gyro` | diag([0.0030]) | `self.R_imu_gyro` |
| `R_accel` | diag([0.1907]) | `self.R_imu_accel` |
| `R_centripetal` | diag([2.0]) | local in `imu_callback` |
| Bias samples | 75 | `self.bias_samples_needed` |
| TF publish rate | 50 Hz | `self.create_timer(0.02, ...)` |
| Covariance cap | P[i,i] ≤ 10 (pos), 1 (θ), 5 (vel) | end of `ekf_predict` |

---

### 2.3 ICP Odometry Refinement

ICP matches consecutive LiDAR scans to estimate the incremental robot transform and integrates it into a global pose.
**Code:** `src/ekf_filter/scripts/part2/icp_odometry_node.py`

The EKF delta is used **only as the initial guess** — the final output is the ICP result blended with the EKF initial guess based on match quality.

---

#### Step 1 — Scan-to-Points Conversion (`scan_to_points`)

Every incoming `LaserScan` message is converted to an `Nx2` NumPy array:

```
for each range r in scan.ranges:
    if range_min < r < range_max:
        x = r · cos(angle)
        y = r · sin(angle)
    angle += angle_increment
```

Scans with fewer than 10 valid points are discarded (`current_points.shape[0] < 10`).

---

#### Step 2 — EKF Delta as Initial Guess (`scan_callback`)

Before ICP runs, the EKF pose delta since the last scan is computed and transformed to the robot frame:

```
dx_global = ekf_x − x           # EKF position delta in global frame
dy_global = ekf_y − y
dtheta    = normalize(ekf_theta − theta)

dx = dx_global · cos(theta) + dy_global · sin(theta)   # rotate to robot frame
dy = −dx_global · sin(theta) + dy_global · cos(theta)
```

This pre-aligns the source scan before ICP begins, reducing the number of iterations needed.

---

#### Step 3 — ICP Loop (`icp`, max 200 iterations, tol = 1×10⁻⁵)

```
source_transformed = transform_points(source, dx, dy, dtheta)   # apply initial guess
tree = KDTree(target)                                            # build KD-tree once

for iteration in range(max_iterations):
    distances, indices = tree.query(source_transformed)          # nearest-neighbor search

    valid = distances < max_correspondence_dist (1.2 m)          # filter by distance
    if sum(valid) < 10: break (not converged)

    dt = compute_transformation(source_transformed[valid],
                                target[indices[valid]])           # SVD step (see below)

    # Compose incremental transform (2D rigid-body composition):
    dtheta_old = dtheta
    dtheta = normalize(dtheta + dt[2])
    dx += dt[0] · cos(dtheta_old) − dt[1] · sin(dtheta_old)
    dy += dt[0] · sin(dtheta_old) + dt[1] · cos(dtheta_old)

    source_transformed = transform_points(source, dx, dy, dtheta)

    error = mean(distances[valid])
    if |prev_error − error| < tol: converged = True; break
    prev_error = error
```

---

#### Step 3a — SVD Transformation (`compute_transformation`)

Given `N` matched point pairs `(source[i], target[i])`:

```
μ_src = mean(source, axis=0)                   # source centroid
μ_tgt = mean(target, axis=0)                   # target centroid
src_c = source − μ_src                         # centered source
tgt_c = target − μ_tgt                         # centered target

H = src_c.T @ tgt_c                            # 2×2 cross-covariance matrix
U, S, Vt = svd(H)                              # singular value decomposition
R = Vt.T @ U.T                                 # optimal rotation (2×2)

if det(R) < 0:                                 # correct reflection (det must = +1)
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

dtheta = atan2(R[1,0], R[0,0])                # extract rotation angle
t = μ_tgt − R @ μ_src                         # optimal translation
```

Returns `[t[0], t[1], dtheta]` — the transformation that best aligns the matched pairs.

---

#### Step 4 — Adaptive EKF Blending (`scan_callback`)

After ICP finishes, the result is blended with the EKF initial guess based on ICP match quality.

**Quality scoring:**
```
corr_score  = min(num_correspondences / 50.0, 1.0)     # ≥50 matches → score = 1.0
error_score = max(0, 1.0 − final_error / 0.3)          # <0.3 m error → score = 1.0
quality     = (corr_score + error_score) / 2.0          # combined [0, 1]
```

**Trust assignment:**
```
if not converged or num_correspondences < 20:
    ekf_trust = 1.0                                      # poor ICP → use EKF 100%
else:
    ekf_trust = max_ekf_trust − quality · (max_ekf_trust − min_ekf_trust)
    # high quality → ekf_trust ≈ min_ekf_trust = 0.05  (trust ICP 95%)
    # low quality  → ekf_trust ≈ max_ekf_trust = 0.35  (trust ICP 65%)

rotation_trust = min(1.0, ekf_trust + rotation_ekf_bonus)   # +0.10 for heading
```

**Blending:**
```
dx_icp    = ekf_trust · dx      + (1 − ekf_trust)    · dx_icp
dy_icp    = ekf_trust · dy      + (1 − ekf_trust)    · dy_icp
dtheta_icp= rotation_trust·dtheta + (1−rotation_trust) · dtheta_icp
```

---

#### Step 5 — Global Pose Integration

```
self.x     += dx_icp · cos(theta) − dy_icp · sin(theta)
self.y     += dx_icp · sin(theta) + dy_icp · cos(theta)
self.theta  = normalize(theta + dtheta_icp)
```

---

#### ICP Hyperparameters

| Parameter | Value | Variable in Code |
|-----------|-------|-----------------|
| `max_iterations` | 200 | `self.max_iterations` |
| `tolerance` | 1×10⁻⁵ | `self.tolerance` |
| `max_correspondence_dist` | 1.2 m | `self.max_correspondence_dist` |
| `min_ekf_trust` | 5% | `self.min_ekf_trust` |
| `max_ekf_trust` | 35% | `self.max_ekf_trust` |
| `rotation_ekf_bonus` | +10% | `self.rotation_ekf_bonus` |
| `min_correspondences` | 20 | hard-coded in `scan_callback` |
| corr normalizer | 50 | `min(num_corr / 50.0, 1.0)` |
| error normalizer | 0.3 m | `1.0 - final_error / 0.3` |

---

### 2.4 SLAM with slam_toolbox

Full SLAM uses `async_slam_toolbox_node` with EKF odometry as the odometry input and `/scan` as the laser source.

**Key features:**
- Scan-to-map matching at every pose (`use_scan_matching: true`)
- Pose graph optimization with loop closure (`do_loop_closing: true`)
- Ceres non-linear solver: `SPARSE_NORMAL_CHOLESKY` + `SCHUR_JACOBI` + `LEVENBERG_MARQUARDT`
- EKF odometry as the odometry source (high quality input)

#### SLAM Configurations

Two configurations were tested to study the effect of scan-matching search constraints:

| Parameter | Config A (Relaxed) | Config B (Strict) |
|-----------|-------------------|-------------------|
| `correlation_search_space_dimension` | 0.3 m (±15 cm) | 0.03 m (±1.5 cm) |
| `distance_variance_penalty` | 2.5 | **20.0** |
| `angle_variance_penalty` | 2.5 | **40.0** |
| `loop_match_minimum_response_fine` | 0.5 | 0.5 |
| `link_match_minimum_response_fine` | 0.3 | 0.3 |
| `resolution` | 0.05 m/px | 0.05 m/px |
| `max_laser_range` | 3.5 m | 3.5 m |
| `minimum_travel_distance` | 0.2 m | 0.2 m |

**Config A (Relaxed):** Allows the scan matcher to search a wide area (±15 cm). This can find better scan matches in open environments but is prone to false correspondences in symmetric corridor geometry (both walls look identical).

**Config B (Strict):** Constrains the search space to ±1.5 cm and applies strong variance penalties to force SLAM to closely follow EKF odometry. The high `distance_variance_penalty=20` and `angle_variance_penalty=40` prevent the pose graph from diverging from the EKF estimate, making SLAM more robust to corridor ambiguity.

> **Active configuration:** `Part 3` launch files use **Config B** (`mapper_params_online_async_B.yaml`).

---

## 3. Results

### Configuration Definition

The table below defines which SLAM configuration was used for each sequence and method.

| Sequence | Rosbag file | Wheel / EKF / ICP run | SLAM runs |
|----------|------------|----------------------|-----------|
| Sequence 0 | `fibo_floor3_seq00_0.db3` | Config A | **Both A and B** |
| Sequence 1 | `fibo_floor3_seq01_0.db3` | Config B | **Both A and B** |
| Sequence 2 | `fibo_floor3_seq02_0.db3` | Config A | **Both A and B** |

> Wheel, EKF, and ICP are pure odometry methods and do not depend on the SLAM config. SLAM was run independently with both configurations on all three sequences.

**Metric definitions used throughout:**
- **LCE** = Loop Closure Error (Euclidean distance start→end). Lower = better.
- **Drift Rate** = `LCE / Total Distance × 100%`. Lower = better.

---

### 3.1 Part 1 – EKF Odometry Fusion

#### Objective
Implement an Extended Kalman Filter (EKF) to fuse wheel odometry and IMU measurements, obtaining a filtered and more reliable odometry estimate compared to raw wheel odometry.

#### Description
Wheel odometry is computed from `/joint_states` and fused with IMU measurements from `/imu` using the EKF. The filter estimates robot pose by combining a differential-drive motion model with probabilistic sensor updates (gyroscope, accelerometer, and centripetal acceleration). The result is compared against the baseline dead-reckoning trajectory.

#### Trajectory Plots — All Methods per Sequence

Each figure shows all five methods on one sequence.
Top row: **Wheel Odometry** | **EKF Odometry** | **ICP Odometry**
Bottom row: **SLAM Config A** | **SLAM Config B** | **All Methods Overlay**

**Sequence 0 – Empty Hallway**
![All Methods Seq 0](results/sequence_0_all_methods.png)

**Sequence 1 – Sharp Turns**
![All Methods Seq 1](results/sequence_1_all_methods.png)

**Sequence 2 – Smooth Motion**
![All Methods Seq 2](results/sequence_2_all_methods.png)

#### Part 1 Quantitative Results

| Sequence | Method | Total Dist (m) | LCE (m) | Drift Rate (%) |
|----------|--------|---------------|---------|---------------|
| Seq 0 | Wheel Odometry | 61.82 | 8.877 | 14.36 |
| Seq 0 | **EKF Odometry** | 55.43 | **4.033** | **7.28** |
| Seq 1 | Wheel Odometry | 62.69 | 2.882 | 4.60 |
| Seq 1 | **EKF Odometry** | 56.39 | **1.711** | **3.03** |
| Seq 2 | Wheel Odometry | 68.59 | 8.761 | 12.77 |
| Seq 2 | **EKF Odometry** | 59.70 | **2.562** | **4.29** |

#### Observations
- EKF reduces drift rate by an average of **−54%** across all three sequences.
- The largest improvement is on Seq 0 (−49%) and Seq 2 (−66%), both long traversals where heading error dominates.
- The gyroscope update is the single most impactful correction — heading drift is the primary failure mode of dead-reckoning.
- Even with IMU fusion, EKF still accumulates drift because it has no absolute position reference.

---

### 3.2 Part 2 – ICP Odometry Refinement

#### Objective
Refine the EKF-based odometry using LiDAR scan matching (ICP) and evaluate the improvement in accuracy and drift compared to EKF alone.

#### Description
The EKF odometry from Part 1 is used as the initial guess for ICP scan matching on consecutive `/scan` messages. At each scan (5 Hz), ICP finds the optimal rigid transform between the current and previous scan. The result is blended adaptively with the EKF initial guess (5–35% EKF trust) and integrated to produce a LiDAR-based odometry estimate. ICP additionally builds a 2D occupancy map by projecting scans from the estimated poses.

#### Trajectory Comparison: EKF vs ICP Odometry

ICP trajectories are shown in the all-methods plots above (top row, right panel of each sequence).

#### 2D Occupancy Maps from ICP

The ICP node projects LiDAR scans at each estimated pose to build an occupancy grid. Map quality directly reflects positional accuracy.

| Sequence 0 | Sequence 1 | Sequence 2 |
|:----------:|:----------:|:----------:|
| ![ICP Map Seq 0](results/sequence_0/ICP_seq0.png) | ![ICP Map Seq 1](results/sequence_1/icp_seq1.png) | ![ICP Map Seq 2](results/sequence_2/icp_seq2.png) |

#### Part 2 Quantitative Results

| Sequence | Method | Total Dist (m) | LCE (m) | Drift Rate (%) |
|----------|--------|---------------|---------|---------------|
| Seq 0 | EKF Odometry | 55.43 | 4.033 | 7.28 |
| Seq 0 | **ICP Odometry** | 66.11 | **4.025** | **6.09** |
| Seq 1 | EKF Odometry | 56.39 | 1.711 | 3.03 |
| Seq 1 | **ICP Odometry** | 64.00 | **1.709** | **2.67** |
| Seq 2 | EKF Odometry | 59.70 | 2.562 | 4.29 |
| Seq 2 | **ICP Odometry** | 62.46 | **2.567** | **4.11** |

#### Observations
- ICP improves drift rate by an average of **−10.8%** over EKF.
- The improvement comes from 5 Hz scan matching that catches positional errors not corrected by the IMU alone.
- Adaptive blending (falling back to EKF when scan quality is low) prevents degradation in featureless corridor sections where ICP would otherwise diverge.
- ICP total distance is slightly higher than EKF because scan matching introduces small jitter on straight paths.
- ICP uniquely produces a 2D occupancy map alongside the trajectory.

---

### 3.3 Part 3 – Full SLAM with slam_toolbox

#### Objective
Perform full SLAM using `slam_toolbox` and compare its pose estimation and mapping performance with the ICP-based odometry from Part 2.

#### Description
`slam_toolbox` runs in async mode using `/scan` as laser input and EKF odometry (`/ekf/odometry`) as the odometry source. It builds a pose graph and applies Ceres non-linear optimization with loop closure to produce a globally consistent trajectory and map. Two configurations are tested to study the effect of scan-matching constraints on corridor environments.

#### Config A (Relaxed) vs Config B (Strict) – SLAM Trajectory

SLAM Config A and Config B trajectories are shown in the all-methods plots in §3.1 (bottom row, left and center panels of each sequence). Key differences are most visible in Sequence 1 where Config A diverges severely.

**All sequences overview:**

![All Sequences SLAM Grid](results/all_sequences_grid.png)

**Config A vs Config B – side-by-side summary:**

![SLAM Config Comparison](results/slam_config_comparison.png)

#### 2D Occupancy Maps: ICP vs SLAM Config A vs SLAM Config B

**Sequence 0 – Empty Hallway**

![Sequence 0 Map Comparison](results/sequence_0_icp_vs_slam_maps.png)

**Sequence 1 – Sharp Turns**

![Sequence 1 Map Comparison](results/sequence_1_icp_vs_slam_maps.png)

**Sequence 2 – Smooth Motion**

![Sequence 2 Map Comparison](results/sequence_2_icp_vs_slam_maps.png)

**All sequences map grid (ICP | SLAM-A | SLAM-B):**

![All Sequences Map Grid](results/all_sequences_icp_vs_slam_maps.png)

#### Compare-Drift Run (Sequence 0 – All 4 Methods on Identical Data)

The `results/sequence_0/compare_drift/` folder contains a dedicated run with all four methods active simultaneously, providing a direct apples-to-apples comparison. Both an ICP map and SLAM map were saved from this session.

**ICP Map vs SLAM Map:**

![Compare-Drift Maps](results/sequence_0/compare_drift/compare_drift_maps.png)

**All 4 method trajectories:**

![Compare-Drift Trajectories](results/sequence_0/compare_drift/compare_drift_trajectories.png)

**All 4 trajectories overlaid on occupancy maps:**

![Compare-Drift Trajectories on Map](results/sequence_0/compare_drift/compare_drift_traj_on_map.png)

#### Part 3 Quantitative Results

| Sequence | Method | Total Dist (m) | LCE (m) | Drift Rate (%) |
|----------|--------|---------------|---------|---------------|
| Seq 0 | ICP Odometry (Part 2 ref.) | 66.11 | 4.025 | 6.09 |
| Seq 0 | SLAM Config A | 62.05 | 4.567 | 7.36 |
| Seq 0 | **SLAM Config B** | 55.74 | **1.216** | **2.18** |
| Seq 1 | ICP Odometry (Part 2 ref.) | 64.00 | 1.709 | 2.67 |
| Seq 1 | SLAM Config A | 68.56 | 17.766 | 25.91 |
| Seq 1 | **SLAM Config B** | 56.41 | **0.643** | **1.14** |
| Seq 2 | ICP Odometry (Part 2 ref.) | 62.46 | 2.567 | 4.11 |
| Seq 2 | SLAM Config A | 69.62 | 5.883 | 8.45 |
| Seq 2 | **SLAM Config B** | 62.11 | **4.033** | **6.49** |

#### Observations
- **Config B (Strict) consistently outperforms Config A** in all three sequences.
- The most dramatic difference is Sequence 1: Config A drifts 25.91% (LCE = 17.77 m) while Config B achieves only 1.14% (LCE = 0.64 m). The relaxed ±15 cm search space causes false scan correspondences in the symmetric left/right corridor walls.
- Config B forces SLAM to stay close to EKF via high variance penalties (`distance=20`, `angle=40`), effectively using EKF as a strong prior and only refining with scan matching.
- Loop closure in Config B further removes accumulated drift when the robot revisits mapped areas.
- SLAM Config B produces the sharpest occupancy maps — walls are clean and consistent, reflecting accurate pose estimates throughout the run.

| Map Quality | ICP | SLAM Config A | SLAM Config B |
|-------------|-----|--------------|--------------|
| Seq 0 | Clear walls, slight blur | Blurry from drift | Sharp, clean walls |
| Seq 1 | Turn artifacts | **Badly distorted** | Best quality |
| Seq 2 | Smooth corridors | Moderate | Good quality |

---

### 3.4 Overall Comparison

#### Full Results Table (All Methods, All Sequences)

| Method | Config | Seq | Total Dist (m) | LCE (m) | Drift Rate (%) |
|--------|--------|-----|---------------|---------|---------------|
| Wheel  | A | 0 | 61.82 | 8.877 | 14.36 |
| EKF    | A | 0 | 55.43 | 4.033 |  7.28 |
| ICP    | A | 0 | 66.11 | 4.025 |  6.09 |
| SLAM   | A | 0 | 62.05 | 4.567 |  7.36 |
| SLAM   | **B** | 0 | 55.74 | **1.216** | **2.18** |
| Wheel  | B | 1 | 62.69 | 2.882 |  4.60 |
| EKF    | B | 1 | 56.39 | 1.711 |  3.03 |
| ICP    | B | 1 | 64.00 | 1.709 |  2.67 |
| SLAM   | A | 1 | 68.56 | 17.766 | 25.91 |
| SLAM   | **B** | 1 | 56.41 | **0.643** | **1.14** |
| Wheel  | A | 2 | 68.59 | 8.761 | 12.77 |
| EKF    | A | 2 | 59.70 | 2.562 |  4.29 |
| ICP    | A | 2 | 62.46 | 2.567 |  4.11 |
| SLAM   | A | 2 | 69.62 | 5.883 |  8.45 |
| SLAM   | **B** | 2 | 62.11 | **4.033** | **6.49** |

#### Average Drift Rate Summary

| Method | Avg Drift Rate | vs Wheel |
|--------|---------------|---------|
| Wheel Odometry | 10.58% | baseline |
| EKF Odometry | 4.87% | **−54%** |
| ICP Odometry | 4.29% | **−59%** |
| SLAM Config A | 13.91% | worse (fails Seq 1) |
| SLAM Config B | **3.27%** | **−69%** |

#### Ranking: SLAM Config B > ICP ≈ EKF > Wheel > SLAM Config A (Seq 1)

#### Robustness Comparison

| Method | Computation | Slip Robustness | Feature Dependency | Long-term Stability |
|--------|-------------|-----------------|-------------------|---------------------|
| Wheel Odometry | Very Fast | Poor | None | Poor |
| EKF Odometry | Fast | Moderate | None | Fair |
| ICP Odometry | Medium | Good | High | Fair |
| SLAM Config A | Slow | Good | High | Variable |
| SLAM Config B | Slow | Good | High | **Excellent** |

#### Key Findings

1. **EKF (Part 1) reduces drift by ~54% vs wheel-only.** The gyroscope update corrects heading drift, which is the dominant error source in dead-reckoning.
2. **ICP (Part 2) further reduces drift by ~11% vs EKF.** Adaptive blending prevents degradation in featureless sections while scan matching catches slip the IMU misses.
3. **SLAM Config B (Part 3) achieves the best overall accuracy (avg 3.27% drift).** Strict constraints keep the pose graph close to EKF, preventing corridor ambiguity. Loop closure provides global consistency.
4. **SLAM Config A is unreliable.** A wide search space (±15 cm) causes catastrophic failure on Seq 1 (25.91% drift). Config B is the recommended configuration for symmetric indoor corridors.
5. **Only SLAM produces a globally consistent 2D map**, making it uniquely suitable for autonomous navigation tasks beyond localization.

---

*Last updated: February 2026*

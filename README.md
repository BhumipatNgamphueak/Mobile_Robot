# Mobile Robot — Quadrotor UAV Control

Full LAB 2 report (dynamics, controller design, experimental results): [`src/README.md`](src/README.md)

## Quick Start

```bash
# Source workspace (run in every terminal)
cd ~/Mobile_Robot && source install/setup.bash

# Terminal 1 — Simulation (choose one)
ros2 launch quad_description sim.launch.py                  # no wind
ros2 launch quad_description sim.launch.py world:=wind      # 4 m/s in −y

# Terminal 2 — Controller + data collection
ros2 launch quad_controller controller.launch.py \
    trajectory_type:=<TYPE> traj_duration:=10.0 \
    collect_data:=true collect_duration:=20.0

# For wind runs, add: wind_y:=-4.0
```

**Trajectory types:** `hover`, `straight_2d`, `sine_2d`, `step_2d`, `lemniscate_2d`, `circle_2d`, `straight_3d`, `helix`, `figure8_3d`, `cone_helix`, `lissajous_3d`

## Data

Each run produces data in `data/<trajectory>_<timestamp>/`:
- `state_data.csv`, `control_data.csv` — time-series logs
- `metrics.csv` — scalar performance summary (RMSE, settling time, motor utilisation)
- 12 diagnostic plots

Cross-experiment comparison figures: `report/`

```bash
python3 compare_experiments.py   # regenerate comparison figures
```

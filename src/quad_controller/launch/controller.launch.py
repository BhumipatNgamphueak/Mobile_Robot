"""
controller.launch.py
====================
Launches the full quad_controller stack:
  1. ekf_node              -- state estimator (IMU + odom fusion)
  2. mpc_controller        -- linearised MPC setpoint tracker
  3. trajectory_generator  -- waypoint -> smooth reference state

All nodes load their parameters from config/params.yaml so gains can be
changed without recompiling.

All nodes use use_sim_time=True so timers are synchronised with the
Gazebo /clock topic published via the ros_gz_bridge.

Launch arguments
----------------
  trajectory_type   One of:
                      hover
                      straight_2d | sine_2d | step_2d          (Part 2 — 2D)
                      lemniscate_2d | circle_2d                 (Part 2 — MPC showcase)
                      straight_3d | helix | figure8_3d          (Part 3 — 3D)
                      cone_helix | lissajous_3d                 (Part 3 — MPC showcase)
                    Default: hover
  traj_plane        '2D plane selection: xz (altitude, default) or xy (horizontal)
  hover_time        Seconds to hold hover before flying trajectory  (default 5.0)
  traj_duration     Seconds to fly trajectory before returning to hover
                    0 = fly forever  (default 0)
  traj_accel        Acceleration for straight-line ramp [m/s²]; 0 = instant
                    (default 0.5)
  wind_x/y/z        Wind vector for RViz arrow visualisation  (default 0 0 0)
  collect_data      Enable data collection & analysis node  (default false)
  collect_duration  Data collection duration in seconds (0 = auto on POST_HOVER)

Usage
-----
  # Part 1 — hover
  ros2 launch quad_controller controller.launch.py trajectory_type:=hover collect_data:=true collect_duration:=15.0

  # Part 2 — 2D (x-z plane, 10 s each)
  ros2 launch quad_controller controller.launch.py trajectory_type:=straight_2d traj_duration:=10.0 collect_data:=true
  ros2 launch quad_controller controller.launch.py trajectory_type:=sine_2d     traj_duration:=10.0 collect_data:=true
  ros2 launch quad_controller controller.launch.py trajectory_type:=lemniscate_2d traj_duration:=10.0 collect_data:=true

  # Part 2 — x-y plane variant
  ros2 launch quad_controller controller.launch.py trajectory_type:=lemniscate_2d traj_plane:=xy traj_duration:=10.0 collect_data:=true

  # Part 3 — 3D (10 s each)
  ros2 launch quad_controller controller.launch.py trajectory_type:=straight_3d  traj_duration:=10.0 collect_data:=true
  ros2 launch quad_controller controller.launch.py trajectory_type:=helix         traj_duration:=10.0 collect_data:=true
  ros2 launch quad_controller controller.launch.py trajectory_type:=lissajous_3d  traj_duration:=10.0 collect_data:=true

  # With wind (add to any command above)
  ros2 launch quad_controller controller.launch.py trajectory_type:=lemniscate_2d traj_duration:=10.0 wind_y:=-4.0 collect_data:=true
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('quad_controller')
    params_file = os.path.join(pkg, 'config', 'params.yaml')

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    traj_type_arg = DeclareLaunchArgument(
        'trajectory_type',
        default_value='hover',
        description='Trajectory: hover | straight_2d | sine_2d | step_2d | '
                    'lemniscate_2d | circle_2d | straight_3d | helix | '
                    'figure8_3d | cone_helix | lissajous_3d')

    hover_time_arg = DeclareLaunchArgument(
        'hover_time',
        default_value='5.0',
        description='Seconds to hold hover before starting trajectory')

    traj_duration_arg = DeclareLaunchArgument(
        'traj_duration',
        default_value='0.0',
        description='Seconds to fly trajectory then return to hover (0 = forever)')

    traj_plane_arg = DeclareLaunchArgument(
        'traj_plane',
        default_value='xz',
        description='2D plane selection: xz (altitude change) or xy (horizontal)')

    traj_accel_arg = DeclareLaunchArgument(
        'traj_accel',
        default_value='0.5',
        description='Acceleration for straight-line ramp [m/s²]; 0 = instant speed')

    horizon_steps_arg = DeclareLaunchArgument(
        'horizon_steps',
        default_value='0',
        description='Trajectory preview steps (0 = disabled, use MPC horizon for best tracking)')

    wind_x_arg = DeclareLaunchArgument(
        'wind_x', default_value='0.0',
        description='Wind x-component [m/s] for RViz arrow')
    wind_y_arg = DeclareLaunchArgument(
        'wind_y', default_value='0.0',
        description='Wind y-component [m/s] for RViz arrow')
    wind_z_arg = DeclareLaunchArgument(
        'wind_z', default_value='0.0',
        description='Wind z-component [m/s] for RViz arrow')

    traj_type     = LaunchConfiguration('trajectory_type')
    hover_time    = LaunchConfiguration('hover_time')
    traj_duration = LaunchConfiguration('traj_duration')
    traj_plane    = LaunchConfiguration('traj_plane')
    traj_accel    = LaunchConfiguration('traj_accel')
    horizon_steps = LaunchConfiguration('horizon_steps')
    wind_x        = LaunchConfiguration('wind_x')
    wind_y        = LaunchConfiguration('wind_y')
    wind_z        = LaunchConfiguration('wind_z')

    collect_data_arg = DeclareLaunchArgument(
        'collect_data', default_value='false',
        description='Enable data collection & analysis node (true/false)')
    collect_duration_arg = DeclareLaunchArgument(
        'collect_duration', default_value='0.0',
        description='Data collection duration in seconds (0 = until Ctrl+C)')

    collect_data     = LaunchConfiguration('collect_data')
    collect_duration = LaunchConfiguration('collect_duration')

    # ------------------------------------------------------------------ #
    # EKF state estimator  (fuses IMU + odom, publishes /state_estimate)
    # ------------------------------------------------------------------ #
    ekf_node = Node(
        package='quad_controller',
        executable='ekf_node.py',
        name='ekf_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
    )

    # ------------------------------------------------------------------ #
    # MPC controller  (reads /state_estimate from EKF)
    # ------------------------------------------------------------------ #
    mpc_node = Node(
        package='quad_controller',
        executable='mpc_controller.py',
        name='mpc_controller',
        output='screen',
        parameters=[params_file, {
            'use_sim_time': True,
            'wind_x': wind_x,
            'wind_y': wind_y,
            'wind_z': wind_z,
        }],
    )

    # ------------------------------------------------------------------ #
    # Trajectory generator
    # trajectory_type, hover_time, traj_duration are overridden here so
    # ros2 launch arguments take precedence over params.yaml defaults.
    # ------------------------------------------------------------------ #
    traj_node = Node(
        package='quad_controller',
        executable='trajectory_generator.py',
        name='trajectory_generator',
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time':    True,
                'trajectory_type': traj_type,
                'hover_time':      hover_time,
                'traj_duration':   traj_duration,
                'traj_plane':      traj_plane,
                'traj_accel':      traj_accel,
                'horizon_steps':   horizon_steps,
            },
        ],
    )

    # ------------------------------------------------------------------ #
    # Data collector (optional -- enabled with collect_data:=true)
    # ------------------------------------------------------------------ #
    data_node = Node(
        package='quad_controller',
        executable='data_collector.py',
        name='data_collector',
        output='screen',
        parameters=[{
            'use_sim_time':     True,
            'collect_duration': collect_duration,
            'trajectory_type':  traj_type,
            'wind_x':           wind_x,
            'wind_y':           wind_y,
            'wind_z':           wind_z,
        }],
        condition=IfCondition(collect_data),
    )

    return LaunchDescription([
        traj_type_arg,
        hover_time_arg,
        traj_duration_arg,
        traj_plane_arg,
        traj_accel_arg,
        horizon_steps_arg,
        wind_x_arg,
        wind_y_arg,
        wind_z_arg,
        collect_data_arg,
        collect_duration_arg,
        ekf_node,
        mpc_node,
        traj_node,
        data_node,
    ])

"""
sim.launch.py
=============
Simulation-only launch file — spawns the world and robot.
Does NOT start the controller.

Run the controller separately in a second terminal:
  ros2 launch quad_controller controller.launch.py
  ros2 launch quad_controller controller.launch.py trajectory_type:=sine_2d

Launch arguments
----------------
  world   World to simulate:
            empty   — no wind  (default)
            wind    — 4 m/s in -y direction
          Full SDF paths are also accepted.

Usage
-----
  # Terminal 1 — simulation
  ros2 launch quad_description sim.launch.py
  ros2 launch quad_description sim.launch.py world:=wind

  # Terminal 2 — controller
  ros2 launch quad_controller controller.launch.py
  ros2 launch quad_controller controller.launch.py trajectory_type:=sine_2d
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_world(context, *args, **kwargs):
    """Resolve 'empty'/'wind' shorthand to full SDF paths."""
    pkg_desc = get_package_share_directory('quad_description')
    worlds_dir = os.path.join(pkg_desc, 'worlds')

    raw = LaunchConfiguration('world').perform(context)
    if raw in ('empty', 'empty.sdf'):
        sdf = os.path.join(worlds_dir, 'empty.sdf')
    elif raw in ('wind', 'wind.sdf'):
        sdf = os.path.join(worlds_dir, 'wind.sdf')
    elif os.path.isfile(raw):
        sdf = raw
    else:
        sdf = os.path.join(worlds_dir, 'empty.sdf')

    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'),
            'launch', 'gz_sim.launch.py')]),
        launch_arguments={
            'gz_args': f'-r -v4 {sdf}',
            'on_exit_shutdown': 'true',
        }.items()
    )]


def generate_launch_description():

    pkg_desc = 'quad_description'

    spawn_x = '0.0'
    spawn_y = '0.0'
    spawn_z = '0.0'

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty',
        description='World to load: "empty" (no wind) or "wind" (4 m/s -y). '
                    'Full SDF path also accepted.')

    # ------------------------------------------------------------------ #
    # Robot State Publisher  (publishes /robot_description for spawn)
    # ------------------------------------------------------------------ #
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(pkg_desc),
                         'launch', 'rsp.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # ------------------------------------------------------------------ #
    # ROS ↔ Gazebo bridge
    # ------------------------------------------------------------------ #
    bridge_params = os.path.join(
        get_package_share_directory(pkg_desc), 'config', 'gz_bridge.yaml')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen'
    )

    # ------------------------------------------------------------------ #
    # Spawn robot
    # ------------------------------------------------------------------ #
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_entity',
        arguments=[
            '-topic',   'robot_description',
            '-name',    'quadrotor',
            '-timeout', '120.0',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
        ],
        output='screen'
    )

    # ------------------------------------------------------------------ #
    # RViz
    # ------------------------------------------------------------------ #
    rviz_cfg = os.path.join(
        get_package_share_directory(pkg_desc), 'rviz', 'sim.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        OpaqueFunction(function=_resolve_world),
        rsp,
        bridge,
        spawn_entity,
        rviz,
    ])

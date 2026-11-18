"""Kinematics, odometry and a deterministic simulator for 4-wheel mecanum robots."""

from __future__ import annotations

__version__ = "0.1.0"

from .bridge import DriveBridge, OdomSample, pose_covariance_6x6, yaw_to_quaternion
from .config import (
    WHEEL_LABELS,
    ConfigError,
    RobotConfig,
    example_names,
    load_config,
    load_example,
)
from .kinematics import (
    ForwardResult,
    MecanumKinematics,
    Twist,
    WheelCommand,
    forward_kinematics,
    inverse_kinematics,
    saturate_wheel_speeds,
)
from .odometry import (
    MecanumOdometry,
    Pose2D,
    integrate_pose,
    integrate_pose_euler,
    se2_exp,
    wrap_angle,
)
from .simulator import DriftReport, SimulationOptions, SimulationResult, simulate
from .ticks_io import TickLog, TickLogError, integrate_tick_log, read_tick_csv
from .trajectories import (
    Segment,
    Trajectory,
    circle,
    figure_eight,
    line,
    make_trajectory,
    spin,
    square,
    strafe,
)

__all__ = [
    "WHEEL_LABELS",
    "ConfigError",
    "DriftReport",
    "DriveBridge",
    "ForwardResult",
    "MecanumKinematics",
    "MecanumOdometry",
    "OdomSample",
    "Pose2D",
    "RobotConfig",
    "Segment",
    "SimulationOptions",
    "SimulationResult",
    "TickLog",
    "TickLogError",
    "Trajectory",
    "Twist",
    "WheelCommand",
    "__version__",
    "circle",
    "example_names",
    "figure_eight",
    "forward_kinematics",
    "integrate_pose",
    "integrate_pose_euler",
    "integrate_tick_log",
    "inverse_kinematics",
    "line",
    "load_config",
    "load_example",
    "make_trajectory",
    "pose_covariance_6x6",
    "read_tick_csv",
    "saturate_wheel_speeds",
    "se2_exp",
    "simulate",
    "spin",
    "square",
    "strafe",
    "wrap_angle",
    "yaw_to_quaternion",
]

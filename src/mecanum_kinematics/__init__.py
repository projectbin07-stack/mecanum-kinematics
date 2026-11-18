"""Kinematics, odometry and a deterministic simulator for 4-wheel mecanum robots."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import WHEEL_LABELS, ConfigError, RobotConfig, example_names, load_config, load_example
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

__all__ = [
    "WHEEL_LABELS",
    "ConfigError",
    "ForwardResult",
    "MecanumKinematics",
    "MecanumOdometry",
    "Pose2D",
    "RobotConfig",
    "Twist",
    "WheelCommand",
    "__version__",
    "example_names",
    "forward_kinematics",
    "integrate_pose",
    "integrate_pose_euler",
    "inverse_kinematics",
    "load_config",
    "load_example",
    "saturate_wheel_speeds",
    "se2_exp",
    "wrap_angle",
]

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

__all__ = [
    "WHEEL_LABELS",
    "ConfigError",
    "ForwardResult",
    "MecanumKinematics",
    "RobotConfig",
    "Twist",
    "WheelCommand",
    "__version__",
    "example_names",
    "forward_kinematics",
    "inverse_kinematics",
    "load_config",
    "load_example",
    "saturate_wheel_speeds",
]

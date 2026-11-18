"""Kinematics, odometry and a deterministic simulator for 4-wheel mecanum robots."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import WHEEL_LABELS, ConfigError, RobotConfig, example_names, load_config, load_example

__all__ = [
    "WHEEL_LABELS",
    "ConfigError",
    "RobotConfig",
    "__version__",
    "example_names",
    "load_config",
    "load_example",
]

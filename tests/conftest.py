from __future__ import annotations

import pytest

from mecanum_kinematics import RobotConfig


@pytest.fixture
def robot() -> RobotConfig:
    """r = 0.03 m, lx = 0.08 m, ly = 0.09 m, so lx + ly = 0.17 m. 1320 counts per wheel rev."""
    return RobotConfig(
        wheel_radius=0.03,
        half_length=0.08,
        half_width=0.09,
        ticks_per_rev=44,
        gear_ratio=30.0,
        max_wheel_speed=30.0,
        name="test-robot",
    )

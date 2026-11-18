from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from mecanum_kinematics import (
    WHEEL_LABELS,
    ConfigError,
    RobotConfig,
    example_names,
    load_config,
    load_example,
)

BASE: dict[str, Any] = {"wheel_radius": 0.05, "half_length": 0.15, "half_width": 0.17}


def test_defaults_and_derived_quantities() -> None:
    cfg = RobotConfig.from_dict({**BASE, "ticks_per_rev": 2000, "gear_ratio": 19.0})
    assert cfg.wheel_order == WHEEL_LABELS
    assert cfg.roller_angle == pytest.approx(math.pi / 4)
    assert math.isinf(cfg.max_wheel_speed)
    assert cfg.ticks_per_wheel_rev == pytest.approx(38000.0)
    assert cfg.radians_per_tick == pytest.approx(2 * math.pi / 38000.0)
    # One count moves the rim by circumference / counts-per-rev.
    assert cfg.metres_per_tick == pytest.approx(2 * math.pi * 0.05 / 38000.0)
    # 45 degree rollers: tan = 1, so the yaw arm is lx + ly.
    assert cfg.rotation_arm == pytest.approx(0.32)


def test_rotation_arm_with_non_45_degree_rollers() -> None:
    cfg = RobotConfig(**BASE, roller_angle_deg=30.0)
    assert cfg.rotation_arm == pytest.approx(0.17 + 0.15 * math.tan(math.radians(30.0)))


def test_wheel_order_is_normalised_and_indexed() -> None:
    cfg = RobotConfig.from_dict({**BASE, "wheel_order": ["fr", "fl", "rr", "rl"]})
    assert cfg.wheel_order == ("FR", "FL", "RR", "RL")
    assert cfg.order_indices == (1, 0, 3, 2)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"wheel_radius": 0.0}, "wheel_radius"),
        ({"wheel_radius": -0.1}, "wheel_radius"),
        ({"half_length": float("nan")}, "half_length"),
        ({"half_width": float("inf")}, "half_width"),
        ({"wheel_radius": "big"}, "must be a number"),
        ({"roller_angle_deg": 90.0}, "roller_angle_deg"),
        ({"roller_angle_deg": 0.0}, "roller_angle_deg"),
        ({"ticks_per_rev": 0}, "ticks_per_rev"),
        ({"gear_ratio": -3}, "gear_ratio"),
        ({"max_wheel_speed": 0.0}, "max_wheel_speed"),
        ({"wheel_order": ["FL", "FR", "RL"]}, "permutation"),
        ({"wheel_order": ["FL", "FL", "RL", "RR"]}, "permutation"),
        ({"wheel_order": "FLFRRLRR"}, "list of four labels"),
        ({"wheel_raduis": 0.05}, "unknown config keys"),
    ],
)
def test_invalid_values_are_rejected(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        RobotConfig.from_dict({**BASE, **override})


def test_missing_required_key() -> None:
    with pytest.raises(ConfigError, match=r"missing required config keys.*half_width"):
        RobotConfig.from_dict({"wheel_radius": 0.05, "half_length": 0.1})


def test_toml_round_trip(tmp_path: Path) -> None:
    file = tmp_path / "robot.toml"
    file.write_text(
        'name = "bench"\n'
        "wheel_radius = 0.04\n"
        "half_length = 0.1\n"
        "half_width = 0.12\n"
        "ticks_per_rev = 1440\n"
        "max_wheel_speed = 20.0\n"
        'wheel_order = ["RL", "RR", "FL", "FR"]\n',
        encoding="utf-8",
    )
    cfg = RobotConfig.from_toml(file)
    assert cfg.name == "bench"
    assert cfg.wheel_order == ("RL", "RR", "FL", "FR")
    assert cfg.ticks_per_rev == 1440.0
    assert RobotConfig.from_dict(cfg.to_dict()) == cfg
    assert load_config(file) == cfg


def test_to_dict_omits_infinite_speed_limit() -> None:
    assert "max_wheel_speed" not in RobotConfig(**BASE).to_dict()
    assert RobotConfig(**BASE, max_wheel_speed=12.0).to_dict()["max_wheel_speed"] == 12.0


def test_toml_errors_name_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("wheel_radius = = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"broken\.toml.*invalid TOML"):
        RobotConfig.from_toml(broken)

    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text("wheel_radius = 0.03\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"incomplete\.toml.*missing required"):
        RobotConfig.from_toml(incomplete)


def test_bundled_examples_load_and_are_labelled_illustrative() -> None:
    names = example_names()
    assert names == ["generic-large", "generic-medium", "generic-small"]
    for name in names:
        cfg = load_example(name)
        assert cfg.name == name
        assert load_config(name) == cfg
        # Honesty rule: no example may pass for a measured commercial robot.
        assert "illustrative" in cfg.description.lower()


def test_unknown_example_and_path() -> None:
    with pytest.raises(ConfigError, match="no bundled example"):
        load_example("turtlebot")
    with pytest.raises(ConfigError, match="neither an existing TOML file nor a bundled example"):
        load_config("does-not-exist.toml")

"""Robot geometry and encoder configuration.

A :class:`RobotConfig` holds everything the kinematics need to know about one
robot. It can be built directly, from a plain ``dict`` or from a TOML file.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

#: Canonical wheel labels: front-left, front-right, rear-left, rear-right.
WHEEL_LABELS: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")


class ConfigError(ValueError):
    """Raised when a robot configuration is missing fields or physically invalid."""


def _positive(name: str, value: float, *, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc
    if math.isnan(number) or number <= 0.0 or (math.isinf(number) and not allow_inf):
        raise ConfigError(f"{name} must be a positive finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class RobotConfig:
    """Geometry, encoder and limit parameters of a 4-wheel mecanum base.

    All lengths are in metres and all angles handed to the kinematics are in
    radians; only ``roller_angle_deg`` is stored in degrees because that is how
    wheels are specified by vendors.

    Attributes:
        wheel_radius: Wheel radius ``r`` in metres.
        half_length: ``lx``, half the wheelbase: the distance along the body x
            axis from the robot centre to a wheel centre, in metres.
        half_width: ``ly``, half the track: the distance along the body y axis
            from the robot centre to a wheel centre, in metres.
        roller_angle_deg: Angle between the roller axis and the wheel's rolling
            direction, in degrees. 45 for ordinary mecanum wheels. Must lie
            strictly between 0 and 90.
        ticks_per_rev: Encoder counts per revolution of the encoder shaft, as
            counted by your decoder (so already multiplied by 4 if you count
            every quadrature edge).
        gear_ratio: Encoder-shaft revolutions per wheel revolution. Use 1 if
            the encoder sits on the wheel shaft.
        max_wheel_speed: Wheel speed limit in rad/s used for saturation.
            ``inf`` disables saturation.
        wheel_order: Order in which wheel values appear in every array this
            library accepts or returns for this robot. A permutation of
            ``FL, FR, RL, RR``.
        name: Short identifier.
        description: Free text.
    """

    wheel_radius: float
    half_length: float
    half_width: float
    roller_angle_deg: float = 45.0
    ticks_per_rev: float = 1024.0
    gear_ratio: float = 1.0
    max_wheel_speed: float = math.inf
    wheel_order: tuple[str, ...] = WHEEL_LABELS
    name: str = "unnamed"
    description: str = ""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "wheel_radius", _positive("wheel_radius", self.wheel_radius))
        set_(self, "half_length", _positive("half_length", self.half_length))
        set_(self, "half_width", _positive("half_width", self.half_width))
        set_(self, "ticks_per_rev", _positive("ticks_per_rev", self.ticks_per_rev))
        set_(self, "gear_ratio", _positive("gear_ratio", self.gear_ratio))
        set_(
            self,
            "max_wheel_speed",
            _positive("max_wheel_speed", self.max_wheel_speed, allow_inf=True),
        )
        angle = _positive("roller_angle_deg", self.roller_angle_deg)
        if angle >= 90.0:
            raise ConfigError(
                f"roller_angle_deg must be strictly between 0 and 90, got {self.roller_angle_deg!r}"
            )
        set_(self, "roller_angle_deg", angle)

        raw_order: object = self.wheel_order
        if isinstance(raw_order, str) or not isinstance(raw_order, (tuple, list)):
            raise ConfigError(
                "wheel_order must be a list of four labels, e.g. ['FL','FR','RL','RR']"
            )
        order = tuple(str(label).upper() for label in raw_order)
        if sorted(order) != sorted(WHEEL_LABELS):
            raise ConfigError(
                f"wheel_order must be a permutation of {list(WHEEL_LABELS)}, got {list(order)}"
            )
        set_(self, "wheel_order", order)
        set_(self, "name", str(self.name))
        set_(self, "description", str(self.description))

    # ------------------------------------------------------------------ derived
    @property
    def roller_angle(self) -> float:
        """Roller angle in radians."""
        return math.radians(self.roller_angle_deg)

    @property
    def ticks_per_wheel_rev(self) -> float:
        """Encoder counts per full wheel revolution."""
        return self.ticks_per_rev * self.gear_ratio

    @property
    def radians_per_tick(self) -> float:
        """Wheel rotation, in radians, represented by one encoder count."""
        return 2.0 * math.pi / self.ticks_per_wheel_rev

    @property
    def metres_per_tick(self) -> float:
        """Rim travel, in metres, represented by one encoder count."""
        return self.radians_per_tick * self.wheel_radius

    @property
    def rotation_arm(self) -> float:
        """Effective lever arm ``k = ly + lx * tan(alpha)`` for yaw, in metres.

        For 45 degree rollers this is the familiar ``lx + ly``.
        """
        return self.half_width + self.half_length * math.tan(self.roller_angle)

    @property
    def order_indices(self) -> tuple[int, ...]:
        """For each slot of ``wheel_order``, the index of that wheel in canonical order."""
        return tuple(WHEEL_LABELS.index(label) for label in self.wheel_order)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RobotConfig:
        """Build a config from a flat mapping using the attribute names as keys.

        Unknown keys are rejected so that typos (``wheel_raduis``) fail loudly
        instead of silently falling back to a default.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown config keys: {unknown}; allowed keys: {sorted(known)}")
        missing = [k for k in ("wheel_radius", "half_length", "half_width") if k not in data]
        if missing:
            raise ConfigError(f"missing required config keys: {missing}")
        return cls(**dict(data))

    @classmethod
    def from_toml(cls, path: str | Path) -> RobotConfig:
        """Load a config from a TOML file with the keys at the top level."""
        file = Path(path)
        try:
            with file.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file}: invalid TOML: {exc}") from exc
        try:
            return cls.from_dict(data)
        except ConfigError as exc:
            raise ConfigError(f"{file}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/TOML-friendly dict; an infinite speed limit is omitted."""
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        out["wheel_order"] = list(self.wheel_order)
        if math.isinf(self.max_wheel_speed):
            del out["max_wheel_speed"]
        return out


def example_names() -> list[str]:
    """Names of the illustrative configs bundled with the package."""
    root = resources.files("mecanum_kinematics") / "configs"
    return sorted(
        entry.name[: -len(".toml")] for entry in root.iterdir() if entry.name.endswith(".toml")
    )


def load_example(name: str) -> RobotConfig:
    """Load one of the bundled illustrative configs by name.

    These are generic, made-up geometries for trying the library out. They are
    not measurements of any real or commercial robot.
    """
    if name not in example_names():
        raise ConfigError(
            f"no bundled example config named {name!r}; choose from {example_names()}"
        )
    entry = resources.files("mecanum_kinematics") / "configs" / f"{name}.toml"
    with resources.as_file(entry) as path:
        return RobotConfig.from_toml(path)


def load_config(name_or_path: str | Path) -> RobotConfig:
    """Load a config from a TOML path, or a bundled example if no such file exists."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return RobotConfig.from_toml(candidate)
    text = str(name_or_path)
    if text in example_names():
        return load_example(text)
    raise ConfigError(
        f"{text!r} is neither an existing TOML file nor a bundled example ({example_names()})"
    )

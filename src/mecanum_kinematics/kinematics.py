"""Inverse and forward kinematics of a 4-wheel mecanum base.

Conventions (see ``docs/kinematics.md`` for the derivation):

* Body frame follows ROS REP-103: x forward, y left, yaw counter-clockwise.
* A positive wheel angular velocity drives that wheel's side of the robot
  forward.
* Rollers are mounted in the usual arrangement in which the roller axes form
  an "X" when the robot is viewed from above.
* Every wheel array is ordered according to ``RobotConfig.wheel_order``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import RobotConfig

FloatArray = npt.NDArray[np.float64]
ArrayLike4 = Sequence[float] | npt.NDArray[np.floating]

# Sign of the tan(alpha) term for each wheel in canonical FL, FR, RL, RR order.
_ROLLER_SIGNS = (-1.0, 1.0, 1.0, -1.0)
# Wheel centre positions in units of (lx, ly), canonical order.
_POSITION_SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


@dataclass(frozen=True)
class Twist:
    """Planar body velocity: ``vx``, ``vy`` in m/s and ``wz`` in rad/s."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def as_array(self) -> FloatArray:
        return np.array([self.vx, self.vy, self.wz], dtype=np.float64)

    @classmethod
    def from_array(cls, values: Sequence[float] | npt.NDArray[np.floating]) -> Twist:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"a twist needs exactly 3 values (vx, vy, wz), got shape {arr.shape}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def scaled(self, factor: float) -> Twist:
        return Twist(self.vx * factor, self.vy * factor, self.wz * factor)


@dataclass(frozen=True)
class WheelCommand:
    """Wheel speeds produced by saturating inverse kinematics.

    Attributes:
        speeds: Wheel angular velocities in rad/s, in ``wheel_order``.
        scale: Factor in ``(0, 1]`` that was applied to the requested speeds.
        saturated: True if ``scale < 1``.
        twist: The body twist these speeds actually produce (the request
            multiplied by ``scale``).
    """

    speeds: FloatArray
    scale: float
    saturated: bool
    twist: Twist


@dataclass(frozen=True)
class ForwardResult:
    """Result of the least-squares forward solve.

    Attributes:
        twist: Best-fit body twist.
        residual: ``wheel_speeds - J @ twist`` in rad/s, in ``wheel_order``.
            Zero when the four speeds are consistent with rigid no-slip motion.
        slip_rms: Root-mean-square of ``residual`` in rad/s.
        slip_ratio: ``|residual| / |wheel_speeds|``, a unitless number in
            ``[0, 1]``; 0 means fully consistent, 1 means the measured speeds
            contain no rigid-body motion at all. 0 when all wheels are stopped.
    """

    twist: Twist
    residual: FloatArray
    slip_rms: float
    slip_ratio: float


def saturate_wheel_speeds(speeds: ArrayLike4, max_wheel_speed: float) -> tuple[FloatArray, float]:
    """Uniformly scale wheel speeds so that none exceeds ``max_wheel_speed``.

    Because the inverse kinematics are linear, scaling all four wheels by the
    same factor scales the body twist by that factor: the robot follows the
    same path, only slower. Clipping wheels individually would instead bend
    the path.

    Returns:
        ``(scaled_speeds, scale)`` with ``0 < scale <= 1``.
    """
    arr = np.asarray(speeds, dtype=np.float64)
    if not max_wheel_speed > 0.0:
        raise ValueError(f"max_wheel_speed must be positive, got {max_wheel_speed!r}")
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= max_wheel_speed or math.isinf(max_wheel_speed):
        return arr.copy(), 1.0
    scale = max_wheel_speed / peak
    return arr * scale, scale


class MecanumKinematics:
    """Kinematic model of one robot, with the Jacobian precomputed.

    The model is ``wheel_speeds = J @ [vx, vy, wz]`` with a 4x3 Jacobian ``J``.
    """

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        r = config.wheel_radius
        t = math.tan(config.roller_angle)
        canonical = np.empty((4, 3), dtype=np.float64)
        for i, (sign, (sx, sy)) in enumerate(zip(_ROLLER_SIGNS, _POSITION_SIGNS, strict=True)):
            x = sx * config.half_length
            y = sy * config.half_width
            # r * w_i = vx - wz * y_i + s_i * tan(alpha) * (vy + wz * x_i)
            canonical[i] = (1.0, sign * t, -y + sign * t * x)
        canonical /= r
        idx = list(config.order_indices)
        self._jacobian: FloatArray = canonical[idx]
        self._pinv: FloatArray = np.linalg.pinv(self._jacobian)
        null_canonical = np.array([1.0, 1.0, -1.0, -1.0]) / 2.0
        self._null: FloatArray = null_canonical[idx]

    @property
    def jacobian(self) -> FloatArray:
        """4x3 matrix mapping ``[vx, vy, wz]`` to wheel speeds (a copy)."""
        return self._jacobian.copy()

    @property
    def pseudo_inverse(self) -> FloatArray:
        """3x4 Moore-Penrose pseudo-inverse of the Jacobian (a copy)."""
        return self._pinv.copy()

    @property
    def null_vector(self) -> FloatArray:
        """Unit vector spanning the wheel-speed pattern that produces no body motion."""
        return self._null.copy()

    # ----------------------------------------------------------------- inverse
    def inverse(self, twist: Twist) -> FloatArray:
        """Wheel angular velocities (rad/s) for a body twist, without saturation."""
        result: FloatArray = self._jacobian @ twist.as_array()
        return result

    def inverse_saturated(self, twist: Twist) -> WheelCommand:
        """Inverse kinematics followed by direction-preserving saturation."""
        speeds, scale = saturate_wheel_speeds(self.inverse(twist), self.config.max_wheel_speed)
        return WheelCommand(
            speeds=speeds, scale=scale, saturated=scale < 1.0, twist=twist.scaled(scale)
        )

    # ----------------------------------------------------------------- forward
    def solve(self, wheel_values: ArrayLike4) -> tuple[FloatArray, FloatArray]:
        """Least-squares solve shared by velocity and displacement use.

        Args:
            wheel_values: Four wheel speeds (rad/s) or four wheel angle
                increments (rad).

        Returns:
            ``(body, residual)``: the 3-vector ``[x, y, yaw]`` rate or
            increment in the body frame, and the 4-vector that the rigid-body
            model cannot explain.
        """
        w = np.asarray(wheel_values, dtype=np.float64).reshape(-1)
        if w.shape != (4,):
            raise ValueError(f"expected 4 wheel values, got shape {w.shape}")
        body: FloatArray = self._pinv @ w
        residual: FloatArray = w - self._jacobian @ body
        return body, residual

    def forward(self, wheel_speeds: ArrayLike4) -> ForwardResult:
        """Best-fit body twist for four measured wheel speeds (rad/s)."""
        body, residual = self.solve(wheel_speeds)
        norm_w = float(np.linalg.norm(np.asarray(wheel_speeds, dtype=np.float64)))
        norm_r = float(np.linalg.norm(residual))
        return ForwardResult(
            twist=Twist.from_array(body),
            residual=residual,
            slip_rms=norm_r / 2.0,
            slip_ratio=norm_r / norm_w if norm_w > 0.0 else 0.0,
        )


def inverse_kinematics(config: RobotConfig, twist: Twist, *, saturate: bool = True) -> WheelCommand:
    """One-shot inverse kinematics. Build a :class:`MecanumKinematics` for repeated use."""
    model = MecanumKinematics(config)
    if saturate:
        return model.inverse_saturated(twist)
    return WheelCommand(speeds=model.inverse(twist), scale=1.0, saturated=False, twist=twist)


def forward_kinematics(config: RobotConfig, wheel_speeds: ArrayLike4) -> ForwardResult:
    """One-shot forward kinematics. Build a :class:`MecanumKinematics` for repeated use."""
    return MecanumKinematics(config).forward(wheel_speeds)

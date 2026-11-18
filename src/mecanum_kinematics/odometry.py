"""Dead-reckoning odometry from wheel encoders.

Each update converts four encoder increments into a body-frame displacement
``(dx, dy, dtheta)`` by least squares, then composes it onto the pose with the
exact SE(2) exponential map. That is exact for a constant body twist over the
update interval, so a robot driving a perfect arc is integrated without the
chord error of a plain Euler step, however coarse the sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RobotConfig
from .kinematics import ArrayLike4, FloatArray, MecanumKinematics

_SMALL_ANGLE = 1e-6


@dataclass(frozen=True)
class Pose2D:
    """Planar pose: ``x``, ``y`` in metres, ``theta`` in radians (not wrapped)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)

    def compose(self, other: Pose2D) -> Pose2D:
        """Return ``self * other``: ``other`` expressed in the frame ``self`` is expressed in."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            self.theta + other.theta,
        )

    def distance_to(self, other: Pose2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def se2_exp(dx: float, dy: float, dtheta: float) -> Pose2D:
    """Exponential map of SE(2).

    ``(dx, dy, dtheta)`` is a body-frame twist multiplied by a time interval.
    The result is the pose reached, relative to the start, by holding that
    twist constant: a circular arc when ``dtheta != 0`` and a straight line
    otherwise.
    """
    if abs(dtheta) < _SMALL_ANGLE:
        # Taylor series of sin(a)/a and (1 - cos(a))/a, accurate to O(a^4).
        a = 1.0 - dtheta * dtheta / 6.0
        b = dtheta / 2.0 - dtheta**3 / 24.0
    else:
        a = math.sin(dtheta) / dtheta
        b = (1.0 - math.cos(dtheta)) / dtheta
    return Pose2D(a * dx - b * dy, b * dx + a * dy, dtheta)


def integrate_pose(pose: Pose2D, dx: float, dy: float, dtheta: float) -> Pose2D:
    """Advance ``pose`` by a body-frame displacement using exact arc integration."""
    return pose.compose(se2_exp(dx, dy, dtheta))


def integrate_pose_euler(pose: Pose2D, dx: float, dy: float, dtheta: float) -> Pose2D:
    """First-order (Euler) update, kept only as a baseline for comparison."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(pose.x + c * dx - s * dy, pose.y + s * dx + c * dy, pose.theta + dtheta)


class MecanumOdometry:
    """Integrates encoder increments into a pose with a simple covariance.

    Covariance model (documented in ``docs/kinematics.md``): each wheel's
    angle increment ``dphi_i`` is assumed to carry independent zero-mean noise
    with variance ``slip_noise**2 * |dphi_i|``. Variance proportional to the
    distance rolled means the accumulated error is a random walk whose size
    does not depend on how often the encoders are sampled. The pose
    covariance is propagated to first order::

        P <- F P F^T + G Q G^T

    with ``Q = diag(slip_noise**2 * |dphi|)``. With ``slip_noise == 0`` the
    covariance stays at its initial value. Systematic errors (wrong wheel
    radius, wrong track) and encoder quantisation are not modelled.
    """

    def __init__(
        self,
        config: RobotConfig,
        *,
        pose: Pose2D | None = None,
        covariance: FloatArray | None = None,
        slip_noise: float = 0.0,
    ) -> None:
        if slip_noise < 0.0:
            raise ValueError(f"slip_noise must be >= 0, got {slip_noise!r}")
        self.config = config
        self.kinematics = MecanumKinematics(config)
        self.slip_noise = float(slip_noise)
        self._pose = pose if pose is not None else Pose2D()
        self._cov = self._check_cov(covariance)
        self._distance = 0.0
        self._last_residual: FloatArray = np.zeros(4)
        self._last_delta = Pose2D()

    @staticmethod
    def _check_cov(covariance: FloatArray | None) -> FloatArray:
        if covariance is None:
            return np.zeros((3, 3), dtype=np.float64)
        cov = np.array(covariance, dtype=np.float64)
        if cov.shape != (3, 3):
            raise ValueError(f"covariance must be 3x3, got shape {cov.shape}")
        return cov

    # ------------------------------------------------------------------- state
    @property
    def pose(self) -> Pose2D:
        """Current pose estimate in the odometry frame."""
        return self._pose

    @property
    def covariance(self) -> FloatArray:
        """3x3 covariance of ``(x, y, theta)`` in the odometry frame (a copy)."""
        return self._cov.copy()

    @property
    def distance(self) -> float:
        """Accumulated translation path length in metres."""
        return self._distance

    @property
    def last_residual(self) -> FloatArray:
        """Wheel-angle residual (rad) of the most recent update, in ``wheel_order``."""
        return self._last_residual.copy()

    @property
    def last_delta(self) -> Pose2D:
        """Body-frame displacement ``(dx, dy, dtheta)`` of the most recent update."""
        return self._last_delta

    def reset(self, pose: Pose2D | None = None, covariance: FloatArray | None = None) -> None:
        """Reset pose (default origin), covariance (default zero) and distance."""
        self._pose = pose if pose is not None else Pose2D()
        self._cov = self._check_cov(covariance)
        self._distance = 0.0
        self._last_residual = np.zeros(4)
        self._last_delta = Pose2D()

    # ----------------------------------------------------------------- updates
    def update_ticks(self, delta_ticks: ArrayLike4) -> Pose2D:
        """Integrate one set of encoder count increments (in ``wheel_order``)."""
        ticks = np.asarray(delta_ticks, dtype=np.float64).reshape(-1)
        return self.update_wheel_angles(ticks * self.config.radians_per_tick)

    def update_wheel_speeds(self, wheel_speeds: ArrayLike4, dt: float) -> Pose2D:
        """Integrate wheel speeds (rad/s) held constant for ``dt`` seconds."""
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt!r}")
        return self.update_wheel_angles(np.asarray(wheel_speeds, dtype=np.float64) * dt)

    def update_wheel_angles(self, delta_angles: ArrayLike4) -> Pose2D:
        """Integrate one set of wheel angle increments in radians (in ``wheel_order``)."""
        dphi = np.asarray(delta_angles, dtype=np.float64).reshape(-1)
        body, residual = self.kinematics.solve(dphi)
        dx, dy, dtheta = float(body[0]), float(body[1]), float(body[2])

        if self.slip_noise > 0.0:
            self._propagate_covariance(dphi, dx, dy, dtheta)

        self._pose = integrate_pose(self._pose, dx, dy, dtheta)
        self._distance += math.hypot(dx, dy)
        self._last_residual = residual
        self._last_delta = Pose2D(dx, dy, dtheta)
        return self._pose

    def _propagate_covariance(self, dphi: FloatArray, dx: float, dy: float, dtheta: float) -> None:
        theta = self._pose.theta
        step = se2_exp(dx, dy, dtheta)
        c, s = math.cos(theta), math.sin(theta)
        wx = c * step.x - s * step.y  # world-frame displacement of this step
        wy = s * step.x + c * step.y
        f = np.array([[1.0, 0.0, -wy], [0.0, 1.0, wx], [0.0, 0.0, 1.0]])
        # Rotate body-frame increment noise into the world frame at the
        # mid-step heading.
        mid = theta + 0.5 * dtheta
        cm, sm = math.cos(mid), math.sin(mid)
        rot = np.array([[cm, -sm, 0.0], [sm, cm, 0.0], [0.0, 0.0, 1.0]])
        g = rot @ self.kinematics.pseudo_inverse
        q = np.diag(self.slip_noise**2 * np.abs(dphi))
        self._cov = f @ self._cov @ f.T + g @ q @ g.T

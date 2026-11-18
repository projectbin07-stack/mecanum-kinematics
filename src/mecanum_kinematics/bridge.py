"""Middleware-free glue between velocity commands, wheels and odometry.

This is the logic a ROS 2 node (or any other transport) needs around the
kinematics: a command watchdog, saturated wheel targets, and odometry samples
already shaped like a ``nav_msgs/Odometry`` message. It imports nothing from
ROS, so it is unit-tested here; ``examples/ros2/mecanum_node.py`` only moves
its values in and out of messages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RobotConfig
from .kinematics import ArrayLike4, FloatArray, MecanumKinematics, Twist, WheelCommand
from .odometry import MecanumOdometry, Pose2D


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion ``(x, y, z, w)`` of a rotation by ``yaw`` about the z axis."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def pose_covariance_6x6(covariance: FloatArray) -> list[float]:
    """Embed a 3x3 ``(x, y, yaw)`` covariance in the row-major 6x6 layout ROS uses.

    ROS orders pose covariance as ``(x, y, z, roll, pitch, yaw)``, so the
    planar terms land in rows and columns 0, 1 and 5. The unused diagonal
    entries are left at zero; set them yourself if a consumer needs them.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (3, 3):
        raise ValueError(f"covariance must be 3x3, got shape {cov.shape}")
    full = np.zeros((6, 6), dtype=np.float64)
    index = (0, 1, 5)
    for i, row in enumerate(index):
        for j, col in enumerate(index):
            full[row, col] = cov[i, j]
    return [float(v) for v in full.reshape(-1)]


@dataclass(frozen=True)
class OdomSample:
    """One odometry update in message-ready form.

    Attributes:
        pose: Pose in the odometry frame.
        twist: Body-frame velocity averaged over the update interval.
        quaternion: Orientation as ``(x, y, z, w)``.
        pose_covariance: Row-major 6x6 pose covariance (36 floats).
        slip_ratio: Wheel inconsistency of this update, 0 (none) to 1.
    """

    pose: Pose2D
    twist: Twist
    quaternion: tuple[float, float, float, float]
    pose_covariance: list[float]
    slip_ratio: float


class DriveBridge:
    """Turns velocity commands into wheel targets and encoder data into odometry.

    Args:
        config: Robot description.
        command_timeout: Seconds after which a stale command is replaced by a
            stop. This is the usual ``cmd_vel`` watchdog.
        slip_noise: Passed to :class:`MecanumOdometry` for covariance growth.
    """

    def __init__(
        self, config: RobotConfig, *, command_timeout: float = 0.5, slip_noise: float = 0.0
    ) -> None:
        if command_timeout <= 0.0:
            raise ValueError(f"command_timeout must be positive, got {command_timeout!r}")
        self.config = config
        self.command_timeout = float(command_timeout)
        self.kinematics = MecanumKinematics(config)
        self.odometry = MecanumOdometry(config, slip_noise=slip_noise)
        self._command = Twist()
        self._command_time: float | None = None
        self._last_ticks: FloatArray | None = None

    # ---------------------------------------------------------------- commands
    def set_command(self, twist: Twist, now: float) -> None:
        """Store a velocity command received at time ``now`` (seconds, any monotonic clock)."""
        self._command = twist
        self._command_time = now

    def command_is_fresh(self, now: float) -> bool:
        """True if a command arrived within ``command_timeout`` seconds before ``now``."""
        return self._command_time is not None and (now - self._command_time) <= self.command_timeout

    def wheel_targets(self, now: float) -> WheelCommand:
        """Saturated wheel speed targets for time ``now``; all zero if the command is stale."""
        twist = self._command if self.command_is_fresh(now) else Twist()
        return self.kinematics.inverse_saturated(twist)

    # ---------------------------------------------------------------- odometry
    def update_from_cumulative_ticks(self, ticks: ArrayLike4, dt: float) -> OdomSample:
        """Integrate cumulative encoder counts read ``dt`` seconds after the previous read.

        The first call only records the baseline and reports no motion.
        """
        current = np.asarray(ticks, dtype=np.float64).reshape(-1)
        if current.shape != (4,):
            raise ValueError(f"expected 4 tick counts, got shape {current.shape}")
        previous = self._last_ticks
        self._last_ticks = current.copy()
        if previous is None:
            return self._sample(np.zeros(4), dt)
        return self._sample((current - previous) * self.config.radians_per_tick, dt)

    def update_from_wheel_speeds(self, wheel_speeds: ArrayLike4, dt: float) -> OdomSample:
        """Integrate wheel speeds (rad/s) assumed constant over ``dt`` seconds.

        Feeding the commanded speeds back in gives open-loop dead reckoning
        for robots without encoders; it then knows nothing about what the
        wheels really did.
        """
        return self._sample(np.asarray(wheel_speeds, dtype=np.float64).reshape(-1) * dt, dt)

    def _sample(self, delta_angles: FloatArray, dt: float) -> OdomSample:
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt!r}")
        pose = self.odometry.update_wheel_angles(delta_angles)
        delta = self.odometry.last_delta
        twist = Twist(delta.x / dt, delta.y / dt, delta.theta / dt) if dt > 0.0 else Twist()
        norm = float(np.linalg.norm(delta_angles))
        residual = float(np.linalg.norm(self.odometry.last_residual))
        return OdomSample(
            pose=pose,
            twist=twist,
            quaternion=yaw_to_quaternion(pose.theta),
            pose_covariance=pose_covariance_6x6(self.odometry.covariance),
            slip_ratio=residual / norm if norm > 0.0 else 0.0,
        )

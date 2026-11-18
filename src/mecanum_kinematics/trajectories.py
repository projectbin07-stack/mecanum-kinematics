"""Open-loop velocity trajectories for the simulator.

A :class:`Trajectory` is a sequence of segments, each holding one body twist
for a duration. Piecewise-constant twists keep the simulator deterministic and
make the ideal path easy to reason about: every segment is a straight line, a
circular arc or a spin.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import accumulate

from .kinematics import Twist
from .odometry import Pose2D, integrate_pose

_STOP = Twist()


@dataclass(frozen=True)
class Segment:
    """Hold ``twist`` for ``duration`` seconds."""

    duration: float
    twist: Twist

    def __post_init__(self) -> None:
        if not (self.duration >= 0.0 and math.isfinite(self.duration)):
            raise ValueError(f"segment duration must be finite and >= 0, got {self.duration!r}")


@dataclass(frozen=True)
class Trajectory:
    """A named sequence of constant-twist segments."""

    name: str
    segments: tuple[Segment, ...]
    _ends: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "_ends", tuple(accumulate(s.duration for s in self.segments)))

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._ends[-1] if self._ends else 0.0

    def twist_at(self, t: float) -> Twist:
        """Commanded twist at time ``t``; zero before the start and after the end."""
        if t < 0.0 or not self.segments:
            return _STOP
        index = bisect_right(self._ends, t)
        if index >= len(self.segments):
            return _STOP
        return self.segments[index].twist

    def mean_twist(self, t0: float, t1: float) -> Twist:
        """Time-average of the commanded twist over ``[t0, t1]``.

        The simulator holds this average for one step, so a segment boundary
        that falls inside a step is weighted correctly instead of being
        rounded to the nearest step.
        """
        if not t1 > t0:
            raise ValueError("t1 must be greater than t0")
        vx = vy = wz = 0.0
        start = 0.0
        for seg, end in zip(self.segments, self._ends, strict=True):
            overlap = min(end, t1) - max(start, t0)
            if overlap > 0.0:
                vx += seg.twist.vx * overlap
                vy += seg.twist.vy * overlap
                wz += seg.twist.wz * overlap
            start = end
            if start >= t1:
                break
        span = t1 - t0
        return Twist(vx / span, vy / span, wz / span)

    def end_pose(self, start: Pose2D | None = None) -> Pose2D:
        """Pose reached by following the commands perfectly (no lag, no slip)."""
        pose = start if start is not None else Pose2D()
        for seg in self.segments:
            d = seg.duration
            pose = integrate_pose(pose, seg.twist.vx * d, seg.twist.vy * d, seg.twist.wz * d)
        return pose


def _check(speed: float, **lengths: float) -> None:
    if not (speed > 0.0 and math.isfinite(speed)):
        raise ValueError(f"speed must be positive and finite, got {speed!r}")
    for key, value in lengths.items():
        if not (value > 0.0 and math.isfinite(value)):
            raise ValueError(f"{key} must be positive and finite, got {value!r}")


def _build(name: str, segments: list[Segment], settle: float) -> Trajectory:
    if settle < 0.0:
        raise ValueError(f"settle must be >= 0, got {settle!r}")
    if settle > 0.0:
        segments = [*segments, Segment(settle, _STOP)]
    return Trajectory(name, tuple(segments))


def line(distance: float = 1.0, speed: float = 0.3, *, settle: float = 0.5) -> Trajectory:
    """Drive straight forward ``distance`` metres."""
    _check(speed, distance=distance)
    return _build("line", [Segment(distance / speed, Twist(vx=speed))], settle)


def strafe(distance: float = 1.0, speed: float = 0.3, *, settle: float = 0.5) -> Trajectory:
    """Slide ``distance`` metres to the left without turning."""
    _check(speed, distance=distance)
    return _build("strafe", [Segment(distance / speed, Twist(vy=speed))], settle)


def square(side: float = 1.0, speed: float = 0.3, *, settle: float = 0.5) -> Trajectory:
    """Trace a square holonomically: forward, left, backward, right, heading fixed."""
    _check(speed, side=side)
    t = side / speed
    return _build(
        "square",
        [
            Segment(t, Twist(vx=speed)),
            Segment(t, Twist(vy=speed)),
            Segment(t, Twist(vx=-speed)),
            Segment(t, Twist(vy=-speed)),
        ],
        settle,
    )


def circle(
    radius: float = 0.5, speed: float = 0.3, *, clockwise: bool = False, settle: float = 0.5
) -> Trajectory:
    """Drive one full circle of ``radius`` metres, nose tangent to the path."""
    _check(speed, radius=radius)
    wz = speed / radius * (-1.0 if clockwise else 1.0)
    return _build(
        "circle", [Segment(2.0 * math.pi * radius / speed, Twist(vx=speed, wz=wz))], settle
    )


def figure_eight(radius: float = 0.5, speed: float = 0.3, *, settle: float = 0.5) -> Trajectory:
    """Drive a counter-clockwise circle followed by a clockwise one."""
    _check(speed, radius=radius)
    t = 2.0 * math.pi * radius / speed
    wz = speed / radius
    return _build(
        "figure-eight",
        [Segment(t, Twist(vx=speed, wz=wz)), Segment(t, Twist(vx=speed, wz=-wz))],
        settle,
    )


def spin(turns: float = 1.0, rate: float = 1.0, *, settle: float = 0.5) -> Trajectory:
    """Rotate in place counter-clockwise by ``turns`` revolutions at ``rate`` rad/s."""
    _check(rate, turns=turns)
    return _build("spin", [Segment(2.0 * math.pi * turns / rate, Twist(wz=rate))], settle)


#: Generators keyed by CLI name. Each takes ``(size, speed)``: ``size`` is a
#: distance, side or radius in metres (turns for ``spin``) and ``speed`` is in
#: m/s (rad/s for ``spin``).
TRAJECTORIES: dict[str, Callable[[float, float], Trajectory]] = {
    "line": lambda size, speed: line(size, speed),
    "strafe": lambda size, speed: strafe(size, speed),
    "square": lambda size, speed: square(size, speed),
    "circle": lambda size, speed: circle(size, speed),
    "figure-eight": lambda size, speed: figure_eight(size, speed),
    "spin": lambda size, speed: spin(size, speed),
}

#: Default ``(size, speed)`` for each named trajectory.
TRAJECTORY_DEFAULTS: dict[str, tuple[float, float]] = {
    "line": (1.0, 0.3),
    "strafe": (1.0, 0.3),
    "square": (1.0, 0.3),
    "circle": (0.5, 0.3),
    "figure-eight": (0.5, 0.3),
    "spin": (1.0, 1.0),
}


def make_trajectory(name: str, size: float | None = None, speed: float | None = None) -> Trajectory:
    """Build a named trajectory, falling back to its defaults for omitted arguments."""
    if name not in TRAJECTORIES:
        raise ValueError(f"unknown trajectory {name!r}; choose from {sorted(TRAJECTORIES)}")
    default_size, default_speed = TRAJECTORY_DEFAULTS[name]
    return TRAJECTORIES[name](
        default_size if size is None else size, default_speed if speed is None else speed
    )

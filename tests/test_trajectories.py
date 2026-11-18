from __future__ import annotations

import math

import pytest

from mecanum_kinematics import (
    Segment,
    Trajectory,
    Twist,
    circle,
    figure_eight,
    line,
    make_trajectory,
    spin,
    square,
    strafe,
)
from mecanum_kinematics.trajectories import TRAJECTORIES


def test_line_and_strafe_end_points() -> None:
    forward = line(2.0, 0.5, settle=0.0)
    assert forward.duration == pytest.approx(4.0)
    assert forward.end_pose().x == pytest.approx(2.0)
    assert forward.end_pose().y == 0.0
    sideways = strafe(1.5, 0.3, settle=0.0)
    assert sideways.end_pose().y == pytest.approx(1.5)
    assert sideways.end_pose().x == 0.0


def test_square_is_closed_and_keeps_heading() -> None:
    traj = square(1.2, 0.4)
    end = traj.end_pose()
    assert math.hypot(end.x, end.y) < 1e-12
    assert end.theta == 0.0
    assert traj.duration == pytest.approx(4 * 3.0 + 0.5)
    assert all(seg.twist.wz == 0.0 for seg in traj.segments)


def test_circle_closes_after_one_turn() -> None:
    ccw = circle(0.5, 0.25, settle=0.0)
    assert ccw.duration == pytest.approx(2 * math.pi * 0.5 / 0.25)
    end = ccw.end_pose()
    assert math.hypot(end.x, end.y) < 1e-12
    assert end.theta == pytest.approx(2 * math.pi)
    assert circle(0.5, 0.25, clockwise=True).end_pose().theta == pytest.approx(-2 * math.pi)


def test_figure_eight_closes_with_zero_net_rotation() -> None:
    end = figure_eight(0.4, 0.2).end_pose()
    assert math.hypot(end.x, end.y) < 1e-12
    assert end.theta == pytest.approx(0.0, abs=1e-12)


def test_spin_turns_in_place() -> None:
    end = spin(1.5, 2.0).end_pose()
    assert end.theta == pytest.approx(3 * math.pi)
    assert (end.x, end.y) == (0.0, 0.0)


def test_twist_at_segment_lookup() -> None:
    traj = Trajectory("demo", (Segment(1.0, Twist(vx=1.0)), Segment(2.0, Twist(vy=2.0))))
    assert traj.twist_at(-0.1) == Twist()
    assert traj.twist_at(0.5) == Twist(vx=1.0)
    assert traj.twist_at(1.0) == Twist(vy=2.0)
    assert traj.twist_at(2.9) == Twist(vy=2.0)
    assert traj.twist_at(3.0) == Twist()
    assert Trajectory("empty", ()).twist_at(0.0) == Twist()
    assert Trajectory("empty", ()).duration == 0.0


def test_mean_twist_weights_segments_by_overlap() -> None:
    traj = Trajectory("demo", (Segment(1.0, Twist(vx=1.0)), Segment(2.0, Twist(vy=2.0))))
    assert traj.mean_twist(0.0, 1.0) == Twist(vx=1.0)
    # 0.25 s of vx = 1 and 0.75 s of vy = 2 in a 1 s window.
    mean = traj.mean_twist(0.75, 1.75)
    assert mean.vx == pytest.approx(0.25)
    assert mean.vy == pytest.approx(1.5)
    # Past the end the command is zero: 0.5 s of vy = 2 in a 2 s window.
    assert traj.mean_twist(2.5, 4.5).vy == pytest.approx(0.5)
    assert traj.mean_twist(5.0, 6.0) == Twist()
    with pytest.raises(ValueError, match="t1"):
        traj.mean_twist(1.0, 1.0)


def test_make_trajectory_registry_and_defaults() -> None:
    assert sorted(TRAJECTORIES) == ["circle", "figure-eight", "line", "spin", "square", "strafe"]
    for name in TRAJECTORIES:
        assert make_trajectory(name).name == name
    assert make_trajectory("line", 3.0, 1.0).end_pose().x == pytest.approx(3.0)
    assert make_trajectory("square", speed=0.6).duration == pytest.approx(4 / 0.6 + 0.5)
    with pytest.raises(ValueError, match="unknown trajectory"):
        make_trajectory("zigzag")


@pytest.mark.parametrize(
    "build",
    [
        lambda: line(-1.0, 0.3),
        lambda: line(1.0, 0.0),
        lambda: square(1.0, float("inf")),
        lambda: circle(0.0, 0.3),
        lambda: spin(1.0, -1.0),
        lambda: strafe(1.0, 0.3, settle=-1.0),
        lambda: Segment(-1.0, Twist()),
    ],
)
def test_invalid_parameters_raise(build: object) -> None:
    assert callable(build)
    with pytest.raises(ValueError, match="must be"):
        build()

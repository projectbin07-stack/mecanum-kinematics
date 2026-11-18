from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mecanum_kinematics import (
    RobotConfig,
    TickLogError,
    integrate_tick_log,
    read_tick_csv,
)

DATA = Path(__file__).parent / "data"
# One wheel revolution forward, then half a revolution of strafing left.
EXPECTED_X = 2 * math.pi * 0.03
EXPECTED_Y = math.pi * 0.03


def test_read_delta_csv() -> None:
    log = read_tick_csv(DATA / "ticks_delta.csv")
    assert log.ticks.shape == (15, 4)
    np.testing.assert_array_equal(log.ticks[0], [132, 132, 132, 132])
    np.testing.assert_array_equal(log.ticks[-1], [-132, 132, 132, -132])
    assert log.time is not None
    assert log.time[0] == pytest.approx(0.1)
    assert log.time[-1] == pytest.approx(1.5)


def test_read_cumulative_csv_with_shuffled_columns() -> None:
    log = read_tick_csv(DATA / "ticks_cumulative.csv")
    assert log.ticks.shape == (16, 4)
    # Columns come back as FL, FR, RL, RR whatever the file order was.
    np.testing.assert_array_equal(log.ticks[0], [1000, -250, 40, 7])
    np.testing.assert_array_equal(log.ticks[1], [1132, -118, 172, 139])
    assert log.time is not None
    assert log.time[-1] == pytest.approx(1.5)


def test_delta_and_cumulative_logs_give_the_same_pose(robot: RobotConfig) -> None:
    delta_poses, delta_odom = integrate_tick_log(robot, read_tick_csv(DATA / "ticks_delta.csv"))
    cum_poses, cum_odom = integrate_tick_log(
        robot, read_tick_csv(DATA / "ticks_cumulative.csv"), cumulative=True
    )
    for odom in (delta_odom, cum_odom):
        assert odom.pose.x == pytest.approx(EXPECTED_X)
        assert odom.pose.y == pytest.approx(EXPECTED_Y)
        assert odom.pose.theta == pytest.approx(0.0, abs=1e-12)
        assert odom.distance == pytest.approx(EXPECTED_X + EXPECTED_Y)
    assert delta_poses.shape == (15, 3)
    # The cumulative log has one extra leading row: the reference reading.
    assert cum_poses.shape == (16, 3)
    np.testing.assert_array_equal(cum_poses[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(cum_poses[1:], delta_poses, atol=1e-12)
    # Halfway through the forward leg.
    assert delta_poses[4, 0] == pytest.approx(EXPECTED_X / 2)


def test_csv_columns_are_mapped_to_the_robots_wheel_order(robot: RobotConfig) -> None:
    reordered = RobotConfig.from_dict({**robot.to_dict(), "wheel_order": ["RR", "RL", "FR", "FL"]})
    _, odom = integrate_tick_log(reordered, read_tick_csv(DATA / "ticks_delta.csv"))
    assert odom.pose.x == pytest.approx(EXPECTED_X)
    assert odom.pose.y == pytest.approx(EXPECTED_Y)


def test_slip_noise_gives_covariance(robot: RobotConfig) -> None:
    _, odom = integrate_tick_log(robot, read_tick_csv(DATA / "ticks_delta.csv"), slip_noise=0.02)
    assert odom.covariance[0, 0] > 0.0


def test_csv_without_time_column(tmp_path: Path) -> None:
    file = tmp_path / "no_time.csv"
    file.write_text("fl,fr,rl,rr\n1,2,3,4\n", encoding="utf-8")
    log = read_tick_csv(file)
    assert log.time is None
    np.testing.assert_array_equal(log.ticks, [[1.0, 2.0, 3.0, 4.0]])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("", "file is empty"),
        ("# only a comment\n", "file is empty"),
        ("fl,fr,rl\n1,2,3\n", r"missing \['rr'\]"),
        ("fl,fr,rl,rr\n", "no data rows"),
        ("fl,fr,rl,rr\n1,2,three,4\n", "data row 2 is malformed"),
        ("fl,fr,rl,rr\n1,2,3,4\n5,6\n", "data row 3 is malformed"),
    ],
)
def test_malformed_csv_is_reported(tmp_path: Path, content: str, match: str) -> None:
    file = tmp_path / "bad.csv"
    file.write_text(content, encoding="utf-8")
    with pytest.raises(TickLogError, match=match):
        read_tick_csv(file)

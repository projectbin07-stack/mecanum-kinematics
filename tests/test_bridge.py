from __future__ import annotations

import math

import numpy as np
import pytest

from mecanum_kinematics import (
    DriveBridge,
    RobotConfig,
    Twist,
    pose_covariance_6x6,
    yaw_to_quaternion,
)


def test_yaw_to_quaternion() -> None:
    assert yaw_to_quaternion(0.0) == (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert (x, y) == (0.0, 0.0)
    assert z == pytest.approx(math.sqrt(0.5))
    assert w == pytest.approx(math.sqrt(0.5))
    assert yaw_to_quaternion(math.pi)[2] == pytest.approx(1.0)


def test_pose_covariance_layout() -> None:
    cov = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    flat = pose_covariance_6x6(cov)
    assert len(flat) == 36
    full = np.array(flat).reshape(6, 6)
    assert full[0, 0] == 1.0  # var(x)
    assert full[1, 1] == 4.0  # var(y)
    assert full[5, 5] == 6.0  # var(yaw)
    assert full[0, 1] == full[1, 0] == 2.0
    assert full[0, 5] == full[5, 0] == 3.0
    assert full[1, 5] == full[5, 1] == 5.0
    assert float(np.sum(full[2:5, :])) == 0.0
    with pytest.raises(ValueError, match="3x3"):
        pose_covariance_6x6(np.zeros((6, 6)))


def test_command_watchdog(robot: RobotConfig) -> None:
    bridge = DriveBridge(robot, command_timeout=0.5)
    # No command yet: stopped.
    np.testing.assert_array_equal(bridge.wheel_targets(now=10.0).speeds, np.zeros(4))
    bridge.set_command(Twist(vx=0.3), now=10.0)
    assert bridge.command_is_fresh(10.4)
    np.testing.assert_allclose(bridge.wheel_targets(now=10.4).speeds, 10.0)
    # Stale after the timeout: stop rather than keep driving blind.
    assert not bridge.command_is_fresh(10.6)
    np.testing.assert_array_equal(bridge.wheel_targets(now=10.6).speeds, np.zeros(4))
    with pytest.raises(ValueError, match="command_timeout"):
        DriveBridge(robot, command_timeout=0.0)


def test_wheel_targets_are_saturated(robot: RobotConfig) -> None:
    bridge = DriveBridge(robot)
    bridge.set_command(Twist(vx=3.0), now=0.0)  # asks for 100 rad/s, limit is 30
    command = bridge.wheel_targets(now=0.1)
    assert command.saturated
    np.testing.assert_allclose(command.speeds, 30.0)
    assert command.twist.vx == pytest.approx(0.9)


def test_cumulative_ticks_to_odom_sample(robot: RobotConfig) -> None:
    bridge = DriveBridge(robot, slip_noise=0.02)
    first = bridge.update_from_cumulative_ticks([500, 500, 500, 500], dt=0.1)
    assert first.pose.x == 0.0  # baseline only
    assert first.twist == Twist()
    # +132 counts on every wheel is a tenth of a revolution in 0.1 s.
    sample = bridge.update_from_cumulative_ticks([632, 632, 632, 632], dt=0.1)
    step = 2 * math.pi * 0.03 / 10
    assert sample.pose.x == pytest.approx(step)
    assert sample.twist.vx == pytest.approx(step / 0.1)
    assert sample.twist.vy == pytest.approx(0.0, abs=1e-12)
    assert sample.quaternion == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)
    assert sample.slip_ratio == pytest.approx(0.0, abs=1e-12)
    assert sample.pose_covariance[0] > 0.0
    assert len(sample.pose_covariance) == 36
    with pytest.raises(ValueError, match="expected 4 tick counts"):
        bridge.update_from_cumulative_ticks([1, 2, 3], dt=0.1)


def test_open_loop_wheel_speed_integration_and_slip(robot: RobotConfig) -> None:
    bridge = DriveBridge(robot)
    # Pure CCW rotation at 1 rad/s for 0.5 s.
    w = 0.17 / 0.03
    sample = bridge.update_from_wheel_speeds([-w, w, -w, w], dt=0.5)
    assert sample.pose.theta == pytest.approx(0.5)
    assert sample.twist.wz == pytest.approx(1.0)
    assert sample.quaternion[2] == pytest.approx(math.sin(0.25))
    assert sample.quaternion[3] == pytest.approx(math.cos(0.25))
    # Fighting wheels: no motion, slip ratio 1.
    stuck = bridge.update_from_wheel_speeds([2.0, 2.0, -2.0, -2.0], dt=0.5)
    assert stuck.slip_ratio == pytest.approx(1.0)
    assert stuck.pose.theta == pytest.approx(0.5)
    # dt == 0 reports no velocity instead of dividing by zero.
    assert bridge.update_from_wheel_speeds([1.0] * 4, dt=0.0).twist == Twist()
    with pytest.raises(ValueError, match="dt"):
        bridge.update_from_wheel_speeds([1.0] * 4, dt=-1.0)

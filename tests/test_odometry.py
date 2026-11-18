from __future__ import annotations

import math

import numpy as np
import pytest

from mecanum_kinematics import (
    MecanumKinematics,
    MecanumOdometry,
    Pose2D,
    RobotConfig,
    Twist,
    integrate_pose,
    integrate_pose_euler,
    se2_exp,
    wrap_angle,
)


def test_se2_exp_straight_line() -> None:
    assert se2_exp(0.5, -0.2, 0.0) == Pose2D(0.5, -0.2, 0.0)


def test_se2_exp_quarter_circle() -> None:
    # Arc length pi/2 on a unit circle turning left ends at (1, 1) facing +y.
    pose = se2_exp(math.pi / 2, 0.0, math.pi / 2)
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(1.0)
    assert pose.theta == pytest.approx(math.pi / 2)


def test_se2_exp_small_angle_branch_is_continuous() -> None:
    # Just below the threshold the Taylor series is used; it must agree with
    # the closed form that is used just above it.
    for angle in (0.99e-6, -0.99e-6, 1.01e-6):
        pose = se2_exp(0.3, 0.1, angle)
        a = math.sin(angle) / angle
        b = (1 - math.cos(angle)) / angle
        assert pose.x == pytest.approx(a * 0.3 - b * 0.1, abs=1e-10)
        assert pose.y == pytest.approx(b * 0.3 + a * 0.1, abs=1e-10)


def test_wrap_angle() -> None:
    assert wrap_angle(0.0) == 0.0
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(2 * math.pi + 0.25) == pytest.approx(0.25)
    assert wrap_angle(-0.25) == pytest.approx(-0.25)


def test_pose_compose() -> None:
    # Facing +y at (1, 2); one metre ahead in the body frame is (1, 3) in the world.
    pose = Pose2D(1.0, 2.0, math.pi / 2).compose(Pose2D(1.0, 0.0, 0.1))
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(3.0)
    assert pose.theta == pytest.approx(math.pi / 2 + 0.1)
    assert Pose2D(0.0, 0.0).distance_to(Pose2D(3.0, 4.0)) == pytest.approx(5.0)


def test_ticks_for_one_wheel_revolution_move_one_circumference(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot)
    pose = odom.update_ticks([1320, 1320, 1320, 1320])  # 44 * 30 counts per wheel rev
    assert pose.x == pytest.approx(2 * math.pi * 0.03)
    assert pose.y == pytest.approx(0.0, abs=1e-15)
    assert pose.theta == pytest.approx(0.0, abs=1e-15)
    assert odom.distance == pytest.approx(2 * math.pi * 0.03)


def test_strafe_ticks(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot)
    pose = odom.update_ticks([-660, 660, 660, -660])  # half a rev, strafe-left pattern
    assert pose.x == pytest.approx(0.0, abs=1e-15)
    assert pose.y == pytest.approx(math.pi * 0.03)


def test_spin_ticks_full_turn(robot: RobotConfig) -> None:
    # A full body turn needs each wheel to roll (lx + ly) * 2 pi / r radians,
    # which is 0.17 / 0.03 wheel revolutions = 7480 counts.
    odom = MecanumOdometry(robot)
    pose = odom.update_ticks([-7480, 7480, -7480, 7480])
    assert pose.theta == pytest.approx(2 * math.pi)
    assert pose.x == pytest.approx(0.0, abs=1e-12)
    assert pose.y == pytest.approx(0.0, abs=1e-12)
    assert odom.distance == pytest.approx(0.0, abs=1e-12)


def test_arc_integration_matches_known_circle_with_coarse_steps(robot: RobotConfig) -> None:
    """Twelve 30 degree steps round a 0.5 m circle must land on the circle exactly."""
    radius, steps = 0.5, 12
    dtheta = 2 * math.pi / steps
    model = MecanumKinematics(robot)
    # Wheel angle increments for one step of arc length radius * dtheta.
    dphi = model.inverse(Twist(vx=radius * dtheta, wz=dtheta))
    odom = MecanumOdometry(robot)
    euler = Pose2D()
    for i in range(1, steps + 1):
        pose = odom.update_wheel_angles(dphi)
        euler = integrate_pose_euler(euler, radius * dtheta, 0.0, dtheta)
        angle = i * dtheta
        # Circle centred at (0, radius), starting at the origin heading +x.
        assert pose.x == pytest.approx(radius * math.sin(angle), abs=1e-12)
        assert pose.y == pytest.approx(radius * (1 - math.cos(angle)), abs=1e-12)
        assert pose.theta == pytest.approx(angle)
    assert math.hypot(odom.pose.x, odom.pose.y) < 1e-12
    assert odom.distance == pytest.approx(2 * math.pi * radius)
    # A single Euler step of the same size goes straight ahead instead of
    # following the arc, and is already centimetres off.
    one_euler = integrate_pose_euler(Pose2D(), radius * dtheta, 0.0, dtheta)
    one_exact = integrate_pose(Pose2D(), radius * dtheta, 0.0, dtheta)
    assert one_euler.distance_to(one_exact) > 0.05


def test_euler_error_shrinks_but_exact_does_not_depend_on_step() -> None:
    def end_error(steps: int) -> tuple[float, float]:
        dtheta = math.pi / steps  # half circle of radius 1: ends at (0, 2)
        exact = Pose2D()
        euler = Pose2D()
        for _ in range(steps):
            exact = integrate_pose(exact, dtheta, 0.0, dtheta)
            euler = integrate_pose_euler(euler, dtheta, 0.0, dtheta)
        target = Pose2D(0.0, 2.0)
        return exact.distance_to(target), euler.distance_to(target)

    exact_coarse, euler_coarse = end_error(4)
    exact_fine, euler_fine = end_error(64)
    assert exact_coarse < 1e-12
    assert exact_fine < 1e-12
    assert euler_coarse > 0.3
    assert euler_fine < euler_coarse / 10


def test_update_wheel_speeds_and_reset(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot, pose=Pose2D(1.0, 1.0, math.pi / 2))
    pose = odom.update_wheel_speeds([10.0, 10.0, 10.0, 10.0], 2.0)  # 0.3 m/s for 2 s
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(1.6)
    assert odom.last_delta.x == pytest.approx(0.6)
    with pytest.raises(ValueError, match="dt"):
        odom.update_wheel_speeds([1.0, 1.0, 1.0, 1.0], -0.1)
    odom.reset()
    assert odom.pose == Pose2D()
    assert odom.distance == 0.0


def test_residual_is_reported_for_inconsistent_ticks(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot)
    odom.update_wheel_angles([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(odom.last_residual, 0.0, atol=1e-15)
    odom.update_wheel_angles([1.0, 1.0, -1.0, -1.0])
    np.testing.assert_allclose(odom.last_residual, [1.0, 1.0, -1.0, -1.0], atol=1e-12)


def test_covariance_stays_zero_without_slip_noise(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot)
    odom.update_wheel_angles([3.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(odom.covariance, np.zeros((3, 3)))


def test_covariance_of_straight_line_matches_closed_form(robot: RobotConfig) -> None:
    """One forward step from zero covariance: P = G Q G^T with G = pinv(J).

    All four wheels roll dphi, so Q = k^2 dphi I and, because the rows of the
    pseudo-inverse are orthogonal, P is diagonal:
    var(x) = var(y) = 4 (r/4)^2 k^2 dphi, var(theta) = var(x) / (lx + ly)^2.
    """
    k, dphi, r, arm = 0.05, 2.0, 0.03, 0.17
    odom = MecanumOdometry(robot, slip_noise=k)
    odom.update_wheel_angles([dphi] * 4)
    var_xy = 4 * (r / 4) ** 2 * k**2 * dphi
    np.testing.assert_allclose(
        odom.covariance, np.diag([var_xy, var_xy, var_xy / arm**2]), atol=1e-18
    )


def test_covariance_is_independent_of_sampling_rate_on_a_line(robot: RobotConfig) -> None:
    coarse = MecanumOdometry(robot, slip_noise=0.05)
    coarse.update_wheel_angles([4.0] * 4)
    fine = MecanumOdometry(robot, slip_noise=0.05)
    for _ in range(400):
        fine.update_wheel_angles([0.01] * 4)
    # x and heading variance grow linearly with distance whatever the step.
    assert fine.covariance[0, 0] == pytest.approx(coarse.covariance[0, 0])
    assert fine.covariance[2, 2] == pytest.approx(coarse.covariance[2, 2])
    # Lateral variance additionally picks up heading error times distance,
    # which only the finely sampled run can see.
    assert fine.covariance[1, 1] > coarse.covariance[1, 1]


def test_covariance_grows_and_stays_symmetric_psd(robot: RobotConfig) -> None:
    odom = MecanumOdometry(robot, slip_noise=0.03)
    model = MecanumKinematics(robot)
    previous_trace = 0.0
    for _ in range(200):
        odom.update_wheel_angles(model.inverse(Twist(0.004, 0.001, 0.01)))
        cov = odom.covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)
        assert float(np.min(np.linalg.eigvalsh(cov))) > -1e-15
        assert float(np.trace(cov)) > previous_trace
        previous_trace = float(np.trace(cov))


def test_invalid_arguments(robot: RobotConfig) -> None:
    with pytest.raises(ValueError, match="slip_noise"):
        MecanumOdometry(robot, slip_noise=-1.0)
    with pytest.raises(ValueError, match="3x3"):
        MecanumOdometry(robot, covariance=np.zeros((2, 2)))
    initial = np.diag([0.01, 0.02, 0.03])
    odom = MecanumOdometry(robot, covariance=initial)
    np.testing.assert_array_equal(odom.covariance, initial)

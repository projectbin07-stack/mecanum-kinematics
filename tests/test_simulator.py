from __future__ import annotations

import math

import numpy as np
import pytest

from mecanum_kinematics import (
    RobotConfig,
    Segment,
    SimulationOptions,
    Trajectory,
    Twist,
    circle,
    figure_eight,
    line,
    simulate,
    spin,
    square,
    strafe,
)

IDEAL = SimulationOptions(dt=0.01, motor_time_constant=0.0)


def test_ideal_line_reaches_exact_distance(robot: RobotConfig) -> None:
    result = simulate(robot, line(1.0, 0.25), IDEAL)
    assert result.final_truth.x == pytest.approx(1.0, abs=1e-12)
    assert result.final_truth.y == pytest.approx(0.0, abs=1e-12)
    assert result.final_odometry.x == pytest.approx(1.0, abs=1e-12)
    assert result.drift.path_length == pytest.approx(1.0, abs=1e-12)
    assert result.drift.final_position_error < 1e-12
    assert result.drift.drift_percent == pytest.approx(0.0, abs=1e-9)
    assert result.steps == 450
    assert result.time[-1] == pytest.approx(4.5)
    # While cruising every wheel turns at v / r.
    np.testing.assert_allclose(result.wheel_speeds[100], 0.25 / 0.03)


def test_ideal_strafe(robot: RobotConfig) -> None:
    result = simulate(robot, strafe(0.8, 0.2), IDEAL)
    assert result.final_truth.x == pytest.approx(0.0, abs=1e-12)
    assert result.final_truth.y == pytest.approx(0.8, abs=1e-12)


def test_closed_square_returns_to_origin_noise_free(robot: RobotConfig) -> None:
    result = simulate(robot, square(1.0, 0.3), IDEAL)
    assert math.hypot(result.final_truth.x, result.final_truth.y) < 1e-9
    assert math.hypot(result.final_odometry.x, result.final_odometry.y) < 1e-9
    assert result.final_truth.theta == pytest.approx(0.0, abs=1e-12)

    # With 4.0 s sides every corner falls on a step edge, so the path is the
    # exact square: 4.8 m long and reaching the far corner (1.2, 1.2).
    exact = simulate(robot, square(1.2, 0.3), IDEAL)
    assert math.hypot(exact.final_truth.x, exact.final_truth.y) < 1e-9
    assert exact.drift.path_length == pytest.approx(4.8, abs=1e-9)
    assert float(np.max(exact.truth[:, 0])) == pytest.approx(1.2, abs=1e-9)
    assert float(np.max(exact.truth[:, 1])) == pytest.approx(1.2, abs=1e-9)


def test_closed_square_with_motor_lag_still_closes(robot: RobotConfig) -> None:
    """A linear lag only delays translation; after settling the square is closed.

    The 0.5 s settle segment is 10 time constants, so the leftover is about
    exp(-10) of one step of travel.
    """
    result = simulate(robot, square(1.0, 0.3), SimulationOptions(motor_time_constant=0.05))
    assert math.hypot(result.final_truth.x, result.final_truth.y) < 1e-5
    assert result.drift.final_position_error < 1e-12
    # Lag rounds the corners, so the path is shorter than the commanded 4 m.
    assert 3.9 < result.drift.path_length < 4.0


def test_ideal_circle_closes_and_stays_on_radius(robot: RobotConfig) -> None:
    result = simulate(robot, circle(0.5, 0.3), IDEAL)
    assert math.hypot(result.final_truth.x, result.final_truth.y) < 1e-9
    assert result.final_truth.theta == pytest.approx(2 * math.pi, abs=1e-9)
    # Every sample lies on the circle of radius 0.5 centred at (0, 0.5).
    radii = np.hypot(result.truth[:, 0], result.truth[:, 1] - 0.5)
    np.testing.assert_allclose(radii, 0.5, atol=1e-9)
    assert result.drift.path_length == pytest.approx(math.pi, rel=1e-4)


def test_ideal_figure_eight_and_spin_close(robot: RobotConfig) -> None:
    eight = simulate(robot, figure_eight(0.4, 0.3), IDEAL)
    # One step straddles the change of turning direction, hence the looser bound.
    assert math.hypot(eight.final_truth.x, eight.final_truth.y) < 1e-3
    assert eight.final_truth.theta == pytest.approx(0.0, abs=1e-9)
    assert float(np.min(eight.truth[:, 1])) == pytest.approx(-0.8, abs=1e-3)
    assert float(np.max(eight.truth[:, 1])) == pytest.approx(0.8, abs=1e-3)

    turn = simulate(robot, spin(2.0, 1.5), IDEAL)
    assert turn.final_truth.theta == pytest.approx(4 * math.pi, abs=1e-9)
    assert turn.drift.path_length == pytest.approx(0.0, abs=1e-12)
    assert turn.drift.drift_percent is None


def test_motor_lag_step_response(robot: RobotConfig) -> None:
    """After one time constant the wheel speed reaches 1 - 1/e of its target."""
    tau, dt = 0.2, 0.01
    traj = Trajectory("step", (Segment(2.0, Twist(vx=0.3)),))
    result = simulate(robot, traj, SimulationOptions(dt=dt, motor_time_constant=tau))
    target = 0.3 / 0.03
    np.testing.assert_allclose(result.wheel_targets[0], target)
    steps_per_tau = round(tau / dt)
    np.testing.assert_allclose(
        result.wheel_speeds[steps_per_tau - 1], target * (1 - math.exp(-1.0)), rtol=1e-12
    )
    # The robot ends short of the commanded 0.6 m by about v * tau.
    assert result.drift.tracking_error == pytest.approx(0.3 * tau, rel=0.05)
    assert result.final_commanded.x == pytest.approx(0.6)


def test_saturation_slows_the_robot_without_bending_the_path(robot: RobotConfig) -> None:
    # 1.2 m/s diagonal needs 80 rad/s on FR and RL; the limit is 30 rad/s.
    traj = Trajectory("fast-diagonal", (Segment(1.0, Twist(vx=1.2, vy=1.2)),))
    result = simulate(robot, traj, IDEAL)
    assert result.saturated_steps == 100
    assert float(np.max(np.abs(result.wheel_speeds))) == pytest.approx(30.0)
    scale = 30.0 / 80.0
    assert result.final_truth.x == pytest.approx(1.2 * scale)
    assert result.final_truth.y == pytest.approx(1.2 * scale)
    np.testing.assert_allclose(result.truth[:, 0], result.truth[:, 1], atol=1e-12)

    unlimited = simulate(robot, traj, SimulationOptions(motor_time_constant=0.0, saturate=False))
    assert unlimited.saturated_steps == 0
    assert unlimited.final_truth.x == pytest.approx(1.2)


def test_quantised_encoders_bound_the_odometry_error(robot: RobotConfig) -> None:
    result = simulate(
        robot, square(1.0, 0.3), SimulationOptions(motor_time_constant=0.0, quantise_encoders=True)
    )
    # Flooring loses less than one count per wheel; one count is 0.14 mm of rim travel.
    assert robot.metres_per_tick == pytest.approx(1.428e-4, rel=1e-3)
    assert 0.0 < result.drift.max_position_error < 2 * robot.metres_per_tick
    # Truth is unaffected by how the encoders are read.
    assert math.hypot(result.final_truth.x, result.final_truth.y) < 1e-9


def test_same_seed_is_bit_identical_and_seeds_differ(robot: RobotConfig) -> None:
    opts = SimulationOptions(slip_noise=0.02, seed=7, quantise_encoders=True)
    first = simulate(robot, figure_eight(0.4, 0.3), opts)
    second = simulate(robot, figure_eight(0.4, 0.3), opts)
    np.testing.assert_array_equal(first.truth, second.truth)
    np.testing.assert_array_equal(first.odometry, second.odometry)
    assert first.drift == second.drift

    other = simulate(robot, figure_eight(0.4, 0.3), SimulationOptions(slip_noise=0.02, seed=8))
    assert not np.array_equal(first.truth, other.truth)


def test_slip_makes_odometry_drift_but_not_the_encoders(robot: RobotConfig) -> None:
    clean = simulate(robot, square(1.0, 0.3), SimulationOptions(seed=1))
    noisy = simulate(robot, square(1.0, 0.3), SimulationOptions(slip_noise=0.03, seed=1))
    # Slip changes where the robot really goes, not what the encoders count.
    np.testing.assert_array_equal(clean.odometry, noisy.odometry)
    assert noisy.drift.final_position_error > 1e-3
    assert noisy.drift.max_position_error >= noisy.drift.final_position_error
    assert noisy.drift.rms_position_error <= noisy.drift.max_position_error
    assert noisy.covariance[0, 0] > 0.0
    np.testing.assert_array_equal(clean.covariance, np.zeros((3, 3)))


def test_predicted_covariance_matches_monte_carlo_spread(robot: RobotConfig) -> None:
    """The odometry covariance should describe the real scatter of the error.

    300 seeded runs of a 1 m line; the sample standard deviation of 300
    draws has a relative standard error of about 4%, so 20% is a wide margin.
    """
    traj = line(1.0, 0.5, settle=0.0)
    errors = []
    predicted = np.zeros((3, 3))
    for seed in range(300):
        run = simulate(
            robot,
            traj,
            SimulationOptions(dt=0.02, motor_time_constant=0.0, slip_noise=0.03, seed=seed),
        )
        errors.append(run.odometry[-1] - run.truth[-1])
        predicted = run.covariance
    empirical_std = np.std(np.array(errors), axis=0)
    predicted_std = np.sqrt(np.diag(predicted))
    np.testing.assert_allclose(empirical_std, predicted_std, rtol=0.2)
    assert abs(float(np.mean(np.array(errors)[:, 0]))) < 3 * predicted_std[0] / math.sqrt(300)


def test_odometry_noise_override_only_changes_covariance(robot: RobotConfig) -> None:
    base = simulate(robot, line(1.0, 0.5), SimulationOptions(slip_noise=0.02, seed=3))
    doubled = simulate(
        robot,
        line(1.0, 0.5),
        SimulationOptions(slip_noise=0.02, seed=3, odometry_slip_noise=0.04),
    )
    np.testing.assert_array_equal(base.truth, doubled.truth)
    np.testing.assert_allclose(doubled.covariance, 4.0 * base.covariance)


def test_empty_trajectory_and_invalid_options(robot: RobotConfig) -> None:
    result = simulate(robot, Trajectory("nothing", ()))
    assert result.steps == 0
    assert result.truth.shape == (1, 3)
    assert result.drift.final_position_error == 0.0
    assert result.drift.as_dict()["drift_percent"] is None
    for bad in (
        {"dt": 0.0},
        {"dt": float("inf")},
        {"motor_time_constant": -0.1},
        {"slip_noise": -1.0},
    ):
        with pytest.raises(ValueError, match="must be"):
            SimulationOptions(**bad)  # type: ignore[arg-type]

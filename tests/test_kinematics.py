from __future__ import annotations

import math

import numpy as np
import pytest

from mecanum_kinematics import (
    MecanumKinematics,
    RobotConfig,
    Twist,
    forward_kinematics,
    inverse_kinematics,
    load_example,
    saturate_wheel_speeds,
)

# The fixture robot has r = 0.03 m and lx + ly = 0.17 m.


def test_pure_forward_gives_equal_wheel_speeds_v_over_r(robot: RobotConfig) -> None:
    speeds = MecanumKinematics(robot).inverse(Twist(vx=0.3))
    np.testing.assert_allclose(speeds, [10.0, 10.0, 10.0, 10.0])


def test_pure_strafe_left_sign_pattern(robot: RobotConfig) -> None:
    # Strafing left: FL and RR spin backwards, FR and RL forwards, |w| = v / r.
    speeds = MecanumKinematics(robot).inverse(Twist(vy=0.3))
    np.testing.assert_allclose(speeds, [-10.0, 10.0, 10.0, -10.0])


def test_pure_rotation_ccw(robot: RobotConfig) -> None:
    # CCW yaw: left wheels backwards, right wheels forwards, |w| = (lx + ly) * wz / r.
    speeds = MecanumKinematics(robot).inverse(Twist(wz=1.5))
    expected = 0.17 * 1.5 / 0.03
    np.testing.assert_allclose(speeds, [-expected, expected, -expected, expected])


def test_combined_twist_hand_computed(robot: RobotConfig) -> None:
    # FL = (0.5 - 0.2 - 0.17) / 0.03, FR = (0.5 + 0.2 + 0.17) / 0.03,
    # RL = (0.5 + 0.2 - 0.17) / 0.03, RR = (0.5 - 0.2 + 0.17) / 0.03
    speeds = MecanumKinematics(robot).inverse(Twist(0.5, 0.2, 1.0))
    np.testing.assert_allclose(speeds, [13 / 3, 29.0, 53 / 3, 47 / 3])


def test_jacobian_matches_closed_form(robot: RobotConfig) -> None:
    model = MecanumKinematics(robot)
    r, k = 0.03, 0.17
    expected = np.array([[1, -1, -k], [1, 1, k], [1, 1, -k], [1, -1, k]]) / r
    np.testing.assert_allclose(model.jacobian, expected)
    # Closed-form pseudo-inverse for 45 degree rollers.
    pinv = (r / 4) * np.array(
        [[1, 1, 1, 1], [-1, 1, 1, -1], [-1 / k, 1 / k, -1 / k, 1 / k]], dtype=float
    )
    np.testing.assert_allclose(model.pseudo_inverse, pinv, atol=1e-15)


def test_general_roller_angle_against_first_principles() -> None:
    """Check J against the rolling constraint evaluated wheel by wheel."""
    cfg = RobotConfig(wheel_radius=0.04, half_length=0.11, half_width=0.2, roller_angle_deg=30.0)
    model = MecanumKinematics(cfg)
    twist = Twist(0.4, -0.25, 0.7)
    # Ground-roller axis angles gamma from the body x axis: FL -a, FR +a, RL +a, RR -a.
    gammas = np.radians([-30.0, 30.0, 30.0, -30.0])
    positions = [(0.11, 0.2), (0.11, -0.2), (-0.11, 0.2), (-0.11, -0.2)]
    expected = []
    for gamma, (x, y) in zip(gammas, positions, strict=True):
        contact_v = np.array([twist.vx - twist.wz * y, twist.vy + twist.wz * x])
        axis = np.array([math.cos(gamma), math.sin(gamma)])
        # No slip along the roller axis: axis . contact_v == axis . (r * w, 0)
        expected.append(float(axis @ contact_v) / (cfg.wheel_radius * math.cos(gamma)))
    np.testing.assert_allclose(model.inverse(twist), expected)


def test_fk_of_ik_is_identity_over_random_twists() -> None:
    rng = np.random.default_rng(20260101)
    for _ in range(300):
        cfg = RobotConfig(
            wheel_radius=rng.uniform(0.02, 0.15),
            half_length=rng.uniform(0.05, 0.5),
            half_width=rng.uniform(0.05, 0.5),
            roller_angle_deg=rng.uniform(20.0, 70.0),
            wheel_order=tuple(rng.permutation(["FL", "FR", "RL", "RR"]).tolist()),
        )
        model = MecanumKinematics(cfg)
        twist = Twist(*rng.uniform(-3.0, 3.0, size=3).tolist())
        result = model.forward(model.inverse(twist))
        np.testing.assert_allclose(result.twist.as_array(), twist.as_array(), atol=1e-10)
        assert result.slip_rms < 1e-9
        assert result.slip_ratio < 1e-9


def test_wheel_order_permutes_outputs(robot: RobotConfig) -> None:
    reordered = RobotConfig.from_dict({**robot.to_dict(), "wheel_order": ["RR", "FL", "RL", "FR"]})
    twist = Twist(0.5, 0.2, 1.0)
    canonical = MecanumKinematics(robot).inverse(twist)  # FL, FR, RL, RR
    permuted = MecanumKinematics(reordered).inverse(twist)
    np.testing.assert_allclose(permuted, canonical[[3, 0, 2, 1]])
    back = MecanumKinematics(reordered).forward(permuted).twist
    np.testing.assert_allclose(back.as_array(), twist.as_array(), atol=1e-12)


def test_null_vector_produces_no_motion_and_full_slip(robot: RobotConfig) -> None:
    model = MecanumKinematics(robot)
    np.testing.assert_allclose(model.pseudo_inverse @ model.null_vector, 0.0, atol=1e-15)
    # FL and FR forward, RL and RR backward: the rollers fight each other.
    result = model.forward([4.0, 4.0, -4.0, -4.0])
    np.testing.assert_allclose(result.twist.as_array(), 0.0, atol=1e-12)
    assert result.slip_ratio == pytest.approx(1.0)
    assert result.slip_rms == pytest.approx(4.0)


def test_slip_residual_for_one_fast_wheel(robot: RobotConfig) -> None:
    # 10 rad/s everywhere plus 2 rad/s extra on FR. Projecting [0, 2, 0, 0] on
    # the null direction [1, 1, -1, -1] / 2 leaves a residual of
    # [0.5, 0.5, -0.5, -0.5].
    result = forward_kinematics(robot, [10.0, 12.0, 10.0, 10.0])
    np.testing.assert_allclose(result.residual, [0.5, 0.5, -0.5, -0.5], atol=1e-12)
    assert result.slip_rms == pytest.approx(0.5)
    assert result.slip_ratio == pytest.approx(1.0 / math.sqrt(444.0))
    # The remaining [-0.5, 1.5, 0.5, 0.5] is rigid motion: r/4 * column sums.
    assert result.twist.vx == pytest.approx(0.03 / 4 * 42.0)
    assert result.twist.vy == pytest.approx(0.03 / 4 * 2.0)
    assert result.twist.wz == pytest.approx(0.03 / 4 * 2.0 / 0.17)


def test_stationary_wheels_have_zero_slip_ratio(robot: RobotConfig) -> None:
    result = forward_kinematics(robot, [0.0, 0.0, 0.0, 0.0])
    assert result.slip_ratio == 0.0
    assert result.twist == Twist()


def test_saturation_scales_uniformly_and_preserves_direction(robot: RobotConfig) -> None:
    twist = Twist(1.5, 0.6, 3.0)  # 3x the hand-computed case: peak wheel = 87 rad/s
    command = inverse_kinematics(robot, twist)
    assert command.saturated
    assert command.scale == pytest.approx(30.0 / 87.0)
    assert float(np.max(np.abs(command.speeds))) == pytest.approx(30.0)
    # The achieved twist is parallel to the request.
    achieved = MecanumKinematics(robot).forward(command.speeds).twist.as_array()
    np.testing.assert_allclose(achieved, twist.as_array() * command.scale, atol=1e-12)
    np.testing.assert_allclose(command.twist.as_array(), achieved, atol=1e-12)
    np.testing.assert_allclose(np.cross(achieved, twist.as_array()), 0.0, atol=1e-12)


def test_no_saturation_below_limit_or_when_disabled(robot: RobotConfig) -> None:
    below = inverse_kinematics(robot, Twist(vx=0.3))
    assert not below.saturated
    assert below.scale == 1.0
    raw = inverse_kinematics(robot, Twist(vx=3.0), saturate=False)
    assert not raw.saturated
    np.testing.assert_allclose(raw.speeds, 100.0)


def test_saturate_wheel_speeds_function() -> None:
    scaled, scale = saturate_wheel_speeds([10.0, -40.0, 20.0, 0.0], 20.0)
    assert scale == pytest.approx(0.5)
    np.testing.assert_allclose(scaled, [5.0, -20.0, 10.0, 0.0])
    unchanged, one = saturate_wheel_speeds([1.0, 2.0, 3.0, 4.0], math.inf)
    assert one == 1.0
    np.testing.assert_allclose(unchanged, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="max_wheel_speed"):
        saturate_wheel_speeds([1.0, 1.0, 1.0, 1.0], 0.0)


def test_bad_shapes_are_rejected(robot: RobotConfig) -> None:
    with pytest.raises(ValueError, match="expected 4 wheel values"):
        MecanumKinematics(robot).forward([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="exactly 3 values"):
        Twist.from_array([1.0, 2.0])


def test_example_config_with_custom_order_is_consistent() -> None:
    cfg = load_example("generic-large")  # order FR, FL, RR, RL
    speeds = MecanumKinematics(cfg).inverse(Twist(vy=0.38))
    # v / r = 5 rad/s; strafing left makes FR and RL positive.
    np.testing.assert_allclose(speeds, [5.0, -5.0, -5.0, 5.0])

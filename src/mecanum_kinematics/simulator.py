"""Deterministic time-stepped simulator for a mecanum base.

The simulator is a kinematic model, not a physics engine. Per step it:

1. takes the commanded body twist, averaged over the step, from the trajectory,
2. converts it to wheel speed targets (inverse kinematics + saturation),
3. passes each target through a first-order motor lag,
4. accumulates wheel angles and reads them through (optionally quantised)
   encoders, feeding the increments to :class:`MecanumOdometry`,
5. perturbs the wheel increments with seeded Gaussian slip to obtain the
   ground-truth motion of the body.

Nothing depends on wall-clock time, and all randomness comes from one seeded
``numpy.random.Generator``, so a run is reproducible bit for bit on one
machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RobotConfig
from .kinematics import FloatArray, MecanumKinematics, saturate_wheel_speeds
from .odometry import MecanumOdometry, Pose2D, integrate_pose, wrap_angle
from .trajectories import Trajectory


@dataclass(frozen=True)
class SimulationOptions:
    """Settings of one simulation run.

    Attributes:
        dt: Step size in seconds.
        motor_time_constant: Time constant ``tau`` of the first-order wheel
            speed response in seconds. 0 makes the wheels track instantly.
        quantise_encoders: If true, the odometry sees whole encoder counts
            (the cumulative angle is floored to counts, as a real counter
            does); otherwise it sees exact fractional counts.
        slip_noise: Slip intensity in rad^0.5. Each step, every wheel's ground
            motion differs from its encoder motion by a zero-mean Gaussian
            angle with variance ``slip_noise**2 * |dphi|``. 0 disables slip.
        seed: Seed of the slip random generator.
        saturate: Apply ``max_wheel_speed`` saturation to the wheel targets.
        odometry_slip_noise: ``slip_noise`` assumed by the odometry covariance
            model. ``None`` uses the simulator's true ``slip_noise``.
    """

    dt: float = 0.01
    motor_time_constant: float = 0.05
    quantise_encoders: bool = False
    slip_noise: float = 0.0
    seed: int = 0
    saturate: bool = True
    odometry_slip_noise: float | None = None

    def __post_init__(self) -> None:
        if not (self.dt > 0.0 and math.isfinite(self.dt)):
            raise ValueError(f"dt must be positive and finite, got {self.dt!r}")
        if self.motor_time_constant < 0.0:
            raise ValueError(f"motor_time_constant must be >= 0, got {self.motor_time_constant!r}")
        if self.slip_noise < 0.0:
            raise ValueError(f"slip_noise must be >= 0, got {self.slip_noise!r}")


@dataclass(frozen=True)
class DriftReport:
    """Summary of how far odometry ended up from ground truth.

    Attributes:
        path_length: Ground-truth translation path length in metres.
        final_position_error: Distance between final odometry and final
            ground-truth positions, in metres.
        final_heading_error: Final odometry heading minus ground-truth
            heading, wrapped to ``(-pi, pi]``, in radians.
        max_position_error: Largest position error over the run, in metres.
        rms_position_error: RMS position error over the run, in metres.
        drift_percent: ``100 * final_position_error / path_length``, or
            ``None`` when the robot did not translate.
        tracking_error: Distance between the final ground-truth position and
            where perfect command following would have ended, in metres. This
            measures motor lag, saturation and slip, not odometry.
    """

    path_length: float
    final_position_error: float
    final_heading_error: float
    max_position_error: float
    rms_position_error: float
    drift_percent: float | None
    tracking_error: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "path_length_m": self.path_length,
            "final_position_error_m": self.final_position_error,
            "final_heading_error_rad": self.final_heading_error,
            "max_position_error_m": self.max_position_error,
            "rms_position_error_m": self.rms_position_error,
            "drift_percent": self.drift_percent,
            "tracking_error_m": self.tracking_error,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Time histories of one run. All pose arrays have shape ``(steps + 1, 3)``.

    Attributes:
        trajectory_name: Name of the trajectory that was run.
        wheel_order: Labels of the wheel columns, copied from the config.
        options: The options used.
        time: Sample times in seconds, shape ``(steps + 1,)``.
        commanded: Poses from perfectly following the commands.
        truth: Ground-truth poses (with lag, saturation and slip).
        odometry: Poses estimated from the encoders.
        wheel_targets: Wheel speed targets in rad/s, shape ``(steps, 4)``.
        wheel_speeds: Wheel speeds after motor lag in rad/s, shape ``(steps, 4)``.
        covariance: Final 3x3 odometry covariance.
        saturated_steps: Number of steps in which saturation scaled the command.
        drift: Summary of odometry error.
    """

    trajectory_name: str
    wheel_order: tuple[str, ...]
    options: SimulationOptions
    time: FloatArray
    commanded: FloatArray
    truth: FloatArray
    odometry: FloatArray
    wheel_targets: FloatArray
    wheel_speeds: FloatArray
    covariance: FloatArray
    saturated_steps: int
    drift: DriftReport

    @property
    def steps(self) -> int:
        return int(self.time.shape[0] - 1)

    @property
    def final_truth(self) -> Pose2D:
        return Pose2D(*(float(v) for v in self.truth[-1]))

    @property
    def final_odometry(self) -> Pose2D:
        return Pose2D(*(float(v) for v in self.odometry[-1]))

    @property
    def final_commanded(self) -> Pose2D:
        return Pose2D(*(float(v) for v in self.commanded[-1]))


def _drift(commanded: FloatArray, truth: FloatArray, odom: FloatArray) -> DriftReport:
    errors = np.hypot(odom[:, 0] - truth[:, 0], odom[:, 1] - truth[:, 1])
    path = float(np.sum(np.hypot(np.diff(truth[:, 0]), np.diff(truth[:, 1]))))
    final = float(errors[-1])
    return DriftReport(
        path_length=path,
        final_position_error=final,
        final_heading_error=wrap_angle(float(odom[-1, 2] - truth[-1, 2])),
        max_position_error=float(np.max(errors)),
        rms_position_error=float(np.sqrt(np.mean(errors**2))),
        drift_percent=100.0 * final / path if path > 1e-9 else None,
        tracking_error=float(
            math.hypot(truth[-1, 0] - commanded[-1, 0], truth[-1, 1] - commanded[-1, 1])
        ),
    )


def simulate(
    config: RobotConfig,
    trajectory: Trajectory,
    options: SimulationOptions | None = None,
) -> SimulationResult:
    """Run ``trajectory`` on the robot described by ``config``."""
    opts = options if options is not None else SimulationOptions()
    dt = opts.dt
    steps = max(0, math.ceil(trajectory.duration / dt - 1e-9))
    model = MecanumKinematics(config)
    pinv = model.pseudo_inverse
    odom_noise = opts.slip_noise if opts.odometry_slip_noise is None else opts.odometry_slip_noise
    odometry = MecanumOdometry(config, slip_noise=odom_noise)
    rng = np.random.default_rng(opts.seed)
    # Exact discretisation of dw/dt = (target - w) / tau for a constant target.
    tau = opts.motor_time_constant
    blend = 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau)
    rad_per_tick = config.radians_per_tick

    time = np.arange(steps + 1, dtype=np.float64) * dt
    commanded = np.zeros((steps + 1, 3))
    truth = np.zeros((steps + 1, 3))
    odom = np.zeros((steps + 1, 3))
    targets = np.zeros((steps, 4))
    speeds = np.zeros((steps, 4))

    cmd_pose = Pose2D()
    true_pose = Pose2D()
    wheel_speed: FloatArray = np.zeros(4)
    wheel_angle: FloatArray = np.zeros(4)
    counts: FloatArray = np.zeros(4)
    saturated_steps = 0

    for k in range(steps):
        # Hold the step-averaged command, so segment boundaries inside a step
        # are weighted correctly.
        twist = trajectory.mean_twist(k * dt, (k + 1) * dt)
        cmd_pose = integrate_pose(cmd_pose, twist.vx * dt, twist.vy * dt, twist.wz * dt)

        target = model.inverse(twist)
        if opts.saturate:
            target, scale = saturate_wheel_speeds(target, config.max_wheel_speed)
            saturated_steps += scale < 1.0
        wheel_speed = wheel_speed + blend * (target - wheel_speed)
        dphi = wheel_speed * dt
        wheel_angle = wheel_angle + dphi

        # Encoders: difference of cumulative counts, so quantisation error
        # does not accumulate.
        new_counts = wheel_angle / rad_per_tick
        if opts.quantise_encoders:
            new_counts = np.floor(new_counts)
        odometry.update_ticks(new_counts - counts)
        counts = new_counts

        # Ground truth: the wheels' effective ground motion includes slip.
        if opts.slip_noise > 0.0:
            slip = opts.slip_noise * np.sqrt(np.abs(dphi)) * rng.standard_normal(4)
            body = pinv @ (dphi + slip)
        else:
            body = pinv @ dphi
        true_pose = integrate_pose(true_pose, float(body[0]), float(body[1]), float(body[2]))

        targets[k] = target
        speeds[k] = wheel_speed
        commanded[k + 1] = cmd_pose.as_array()
        truth[k + 1] = true_pose.as_array()
        odom[k + 1] = odometry.pose.as_array()

    return SimulationResult(
        trajectory_name=trajectory.name,
        wheel_order=config.wheel_order,
        options=opts,
        time=time,
        commanded=commanded,
        truth=truth,
        odometry=odom,
        wheel_targets=targets,
        wheel_speeds=speeds,
        covariance=odometry.covariance,
        saturated_steps=int(saturated_steps),
        drift=_drift(commanded, truth, odom),
    )

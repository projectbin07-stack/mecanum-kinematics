"""Command line interface: ``mecanum ik | fk | simulate | odom | configs``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, RobotConfig, example_names, load_config, load_example
from .kinematics import MecanumKinematics, Twist
from .odometry import Pose2D
from .simulator import SimulationOptions, simulate
from .svg import COLOR_ODOMETRY, Series, render_xy_svg, simulation_svg, write_svg
from .ticks_io import TickLogError, integrate_tick_log, read_tick_csv
from .trajectories import TRAJECTORIES, make_trajectory

DEFAULT_CONFIG = "generic-small"
_RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        metavar="NAME_OR_PATH",
        help="TOML robot config, or the name of a bundled illustrative example "
        f"(default: {DEFAULT_CONFIG}; see 'mecanum configs')",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mecanum",
        description="Kinematics, odometry and simulation for 4-wheel mecanum drive robots. "
        "Body frame: x forward, y left, yaw counter-clockwise (ROS REP-103).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ik = sub.add_parser(
        "ik", help="body twist -> wheel speeds", description="Inverse kinematics with saturation."
    )
    _add_config(ik)
    ik.add_argument("--vx", type=float, default=0.0, help="forward velocity [m/s]")
    ik.add_argument("--vy", type=float, default=0.0, help="leftward velocity [m/s]")
    ik.add_argument("--wz", type=float, default=0.0, help="yaw rate, CCW positive [rad/s]")
    ik.add_argument(
        "--no-saturate", action="store_true", help="do not apply the max_wheel_speed limit"
    )

    fk = sub.add_parser(
        "fk",
        help="wheel speeds -> body twist",
        description="Least-squares forward kinematics with a slip/consistency residual.",
    )
    _add_config(fk)
    fk.add_argument(
        "wheels",
        type=float,
        nargs=4,
        metavar="W",
        help="four wheel speeds [rad/s] in the config's wheel_order "
        "(put '--' before them if any is negative)",
    )

    sim = sub.add_parser(
        "simulate",
        help="run a simulated trajectory and report odometry drift",
        description="Deterministic simulation: motor lag, optional encoder quantisation and "
        "seeded wheel slip. Reports ground truth versus odometry.",
    )
    _add_config(sim)
    sim.add_argument(
        "-t", "--trajectory", choices=sorted(TRAJECTORIES), default="square", help="path to drive"
    )
    sim.add_argument(
        "--size",
        type=float,
        help="distance, side or radius [m]; number of turns for 'spin' (default depends on path)",
    )
    sim.add_argument("--speed", type=float, help="speed [m/s]; yaw rate [rad/s] for 'spin'")
    sim.add_argument("--dt", type=float, default=0.01, help="time step [s] (default: 0.01)")
    sim.add_argument(
        "--tau", type=float, default=0.05, help="motor time constant [s] (default: 0.05)"
    )
    sim.add_argument(
        "--slip-noise",
        type=float,
        default=0.0,
        help="wheel slip intensity [rad^0.5]; 0 disables slip (default: 0)",
    )
    sim.add_argument("--seed", type=int, default=0, help="random seed for slip (default: 0)")
    sim.add_argument("--quantise", action="store_true", help="quantise encoders to whole counts")
    sim.add_argument(
        "--plot",
        type=Path,
        metavar="FILE",
        help="write a path plot; .svg needs no extra packages, other types need matplotlib",
    )
    sim.add_argument(
        "--csv", type=Path, metavar="FILE", help="write the time histories of all poses as CSV"
    )

    odom = sub.add_parser(
        "odom",
        help="integrate an encoder tick CSV into a pose",
        description="Replay encoder counts through the odometry. The CSV needs a header row "
        "naming the columns fl, fr, rl, rr (any order, optional t column).",
    )
    _add_config(odom)
    odom.add_argument("ticks", type=Path, help="CSV file of encoder counts")
    odom.add_argument(
        "--cumulative",
        action="store_true",
        help="rows are running encoder totals (default: each row is an increment)",
    )
    odom.add_argument(
        "--slip-noise",
        type=float,
        default=0.0,
        help="slip intensity [rad^0.5] for the covariance model (default: 0)",
    )
    odom.add_argument("-o", "--output", type=Path, metavar="FILE", help="write poses as CSV")
    odom.add_argument("--plot", type=Path, metavar="FILE", help="write the path as an SVG plot")

    configs = sub.add_parser(
        "configs",
        help="list the bundled illustrative configs",
        description="List the bundled example configs. They are illustrative, made-up "
        "geometries, not measurements of any real robot.",
    )
    configs.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def _pose_text(pose: Pose2D) -> str:
    # Round first and add 0.0 so that a tiny negative value does not print as "-0.0000".
    x, y, theta = (round(v, 4) + 0.0 for v in (pose.x, pose.y, pose.theta))
    return f"x={x:+.4f} m  y={y:+.4f} m  theta={theta:+.4f} rad"


def _emit(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2))


def _wheel_table(config: RobotConfig, speeds: Sequence[float]) -> str:
    lines = ["wheel   rad/s        rpm"]
    for label, speed in zip(config.wheel_order, speeds, strict=True):
        lines.append(f"{label:<5} {speed:9.4f}  {speed * _RAD_S_TO_RPM:9.2f}")
    return "\n".join(lines)


def _cmd_ik(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = MecanumKinematics(config)
    twist = Twist(args.vx, args.vy, args.wz)
    if args.no_saturate:
        speeds, scale = model.inverse(twist), 1.0
    else:
        command = model.inverse_saturated(twist)
        speeds, scale = command.speeds, command.scale
    achieved = twist.scaled(scale)
    if args.json:
        _emit(
            {
                "config": config.name,
                "wheel_order": list(config.wheel_order),
                "wheel_speeds_rad_s": [float(v) for v in speeds],
                "scale": scale,
                "saturated": scale < 1.0,
                "achieved_twist": {"vx": achieved.vx, "vy": achieved.vy, "wz": achieved.wz},
            }
        )
        return 0
    print(f"config: {config.name}")
    print(f"twist:  vx={twist.vx:.4f} m/s  vy={twist.vy:.4f} m/s  wz={twist.wz:.4f} rad/s")
    print(_wheel_table(config, [float(v) for v in speeds]))
    if scale < 1.0:
        print(
            f"saturated: all wheels scaled by {scale:.4f} to respect "
            f"max_wheel_speed={config.max_wheel_speed:g} rad/s"
        )
        print(
            f"achieved: vx={achieved.vx:.4f} m/s  vy={achieved.vy:.4f} m/s  "
            f"wz={achieved.wz:.4f} rad/s"
        )
    return 0


def _cmd_fk(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = MecanumKinematics(config).forward(args.wheels)
    twist = result.twist
    if args.json:
        _emit(
            {
                "config": config.name,
                "wheel_order": list(config.wheel_order),
                "twist": {"vx": twist.vx, "vy": twist.vy, "wz": twist.wz},
                "residual_rad_s": [float(v) for v in result.residual],
                "slip_rms_rad_s": result.slip_rms,
                "slip_ratio": result.slip_ratio,
            }
        )
        return 0
    print(f"config: {config.name}")
    print(f"twist:  vx={twist.vx:.4f} m/s  vy={twist.vy:.4f} m/s  wz={twist.wz:.4f} rad/s")
    print(f"slip:   rms residual {result.slip_rms:.4f} rad/s, ratio {result.slip_ratio:.4f}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    trajectory = make_trajectory(args.trajectory, args.size, args.speed)
    options = SimulationOptions(
        dt=args.dt,
        motor_time_constant=args.tau,
        quantise_encoders=args.quantise,
        slip_noise=args.slip_noise,
        seed=args.seed,
    )
    result = simulate(config, trajectory, options)
    drift = result.drift
    truth, odom = result.final_truth, result.final_odometry

    written: dict[str, str] = {}
    if args.plot is not None:
        if args.plot.suffix.lower() == ".svg":
            write_svg(args.plot, simulation_svg(result))
        else:
            from .plotting import plot_simulation

            plot_simulation(result, args.plot)
        written["plot"] = str(args.plot)
    if args.csv is not None:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["t", "cmd_x", "cmd_y", "cmd_theta", "true_x", "true_y", "true_theta"]
            writer.writerow([*header, "odom_x", "odom_y", "odom_theta"])
            for i in range(result.steps + 1):
                row = [result.time[i], *result.commanded[i], *result.truth[i], *result.odometry[i]]
                writer.writerow([f"{float(v):.9g}" for v in row])
        written["csv"] = str(args.csv)

    if args.json:
        _emit(
            {
                "config": config.name,
                "trajectory": trajectory.name,
                "duration_s": trajectory.duration,
                "steps": result.steps,
                "saturated_steps": result.saturated_steps,
                "final_truth": {"x": truth.x, "y": truth.y, "theta": truth.theta},
                "final_odometry": {"x": odom.x, "y": odom.y, "theta": odom.theta},
                "drift": drift.as_dict(),
                "odometry_std": {
                    "x_m": math.sqrt(max(0.0, float(result.covariance[0, 0]))),
                    "y_m": math.sqrt(max(0.0, float(result.covariance[1, 1]))),
                    "theta_rad": math.sqrt(max(0.0, float(result.covariance[2, 2]))),
                },
                "written": written,
            }
        )
        return 0

    print(f"config:      {config.name}")
    print(
        f"trajectory:  {trajectory.name}, {trajectory.duration:.2f} s, {result.steps} steps "
        f"of {options.dt:g} s"
    )
    print(
        f"model:       tau={options.motor_time_constant:g} s, slip_noise={options.slip_noise:g}, "
        f"seed={options.seed}, quantised={'yes' if options.quantise_encoders else 'no'}"
    )
    if result.saturated_steps:
        print(f"saturation:  wheel speeds were scaled down in {result.saturated_steps} steps")
    print(f"ground truth final: {_pose_text(truth)}")
    print(f"odometry final:     {_pose_text(odom)}")
    print(f"path length:        {drift.path_length:.4f} m")
    percent = "" if drift.drift_percent is None else f" ({drift.drift_percent:.3f}% of path)"
    print(f"odometry error:     final {drift.final_position_error * 1000:.2f} mm{percent}")
    print(
        f"                    max {drift.max_position_error * 1000:.2f} mm, "
        f"rms {drift.rms_position_error * 1000:.2f} mm, "
        f"heading {math.degrees(drift.final_heading_error):+.3f} deg"
    )
    print(f"tracking error:     {drift.tracking_error * 1000:.2f} mm from the commanded end point")
    if options.slip_noise > 0.0:
        cov = result.covariance
        print(
            f"odometry 1-sigma:   x {math.sqrt(max(0.0, float(cov[0, 0]))) * 1000:.2f} mm, "
            f"y {math.sqrt(max(0.0, float(cov[1, 1]))) * 1000:.2f} mm, "
            f"theta {math.degrees(math.sqrt(max(0.0, float(cov[2, 2])))):.3f} deg"
        )
    for kind, path in written.items():
        print(f"wrote {kind}: {path}")
    return 0


def _cmd_odom(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    log = read_tick_csv(args.ticks)
    poses, odometry = integrate_tick_log(
        config, log, cumulative=args.cumulative, slip_noise=args.slip_noise
    )
    pose = odometry.pose
    cov = odometry.covariance

    written: dict[str, str] = {}
    if args.output is not None:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            has_time = log.time is not None
            writer.writerow((["t"] if has_time else []) + ["x", "y", "theta"])
            for i, row in enumerate(poses):
                prefix = [f"{float(log.time[i]):.9g}"] if log.time is not None else []
                writer.writerow(prefix + [f"{float(v):.9g}" for v in row])
        written["poses"] = str(args.output)
    if args.plot is not None:
        if args.plot.suffix.lower() != ".svg":
            raise ConfigError("'mecanum odom --plot' only writes .svg files")
        xs = [0.0, *(float(v) for v in poses[:, 0])]
        ys = [0.0, *(float(v) for v in poses[:, 1])]
        document = render_xy_svg(
            [Series("odometry", xs, ys, COLOR_ODOMETRY)],
            title=f"Odometry from {args.ticks.name}",
            subtitle=f"{poses.shape[0]} rows, path {odometry.distance:.2f} m, "
            f"config '{config.name}'. Cross = start, dot = end.",
        )
        write_svg(args.plot, document)
        written["plot"] = str(args.plot)

    if args.json:
        _emit(
            {
                "config": config.name,
                "rows": int(poses.shape[0]),
                "mode": "cumulative" if args.cumulative else "delta",
                "pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
                "distance_m": odometry.distance,
                "covariance": [[float(v) for v in row] for row in cov],
                "written": written,
            }
        )
        return 0
    print(f"config:   {config.name}")
    print(
        f"rows:     {poses.shape[0]} ({'cumulative counts' if args.cumulative else 'increments'})"
    )
    print(f"pose:     {_pose_text(pose)}")
    print(f"distance: {odometry.distance:.4f} m")
    if args.slip_noise > 0.0:
        print(
            f"1-sigma:  x {math.sqrt(max(0.0, float(cov[0, 0]))) * 1000:.2f} mm, "
            f"y {math.sqrt(max(0.0, float(cov[1, 1]))) * 1000:.2f} mm, "
            f"theta {math.degrees(math.sqrt(max(0.0, float(cov[2, 2])))):.3f} deg"
        )
    for kind, path in written.items():
        print(f"wrote {kind}: {path}")
    return 0


def _cmd_configs(args: argparse.Namespace) -> int:
    configs = [load_example(name) for name in example_names()]
    if args.json:
        _emit({"configs": [c.to_dict() for c in configs]})
        return 0
    print("Bundled example configs (illustrative, made-up geometries - not real robots):")
    for c in configs:
        print(
            f"  {c.name:<15} r={c.wheel_radius:g} m, lx={c.half_length:g} m, ly={c.half_width:g} m,"
            f" order={','.join(c.wheel_order)}"
        )
    return 0


_COMMANDS = {
    "ik": _cmd_ik,
    "fk": _cmd_fk,
    "simulate": _cmd_simulate,
    "odom": _cmd_odom,
    "configs": _cmd_configs,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 2 bad input)."""
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (ConfigError, TickLogError, ValueError, RuntimeError, OSError) as exc:
        print(f"mecanum: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

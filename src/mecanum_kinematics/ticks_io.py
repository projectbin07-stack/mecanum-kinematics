"""Reading encoder logs from CSV and replaying them through the odometry."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import WHEEL_LABELS, RobotConfig
from .kinematics import FloatArray
from .odometry import MecanumOdometry


class TickLogError(ValueError):
    """Raised when an encoder CSV file cannot be interpreted."""


@dataclass(frozen=True)
class TickLog:
    """Encoder counts read from a CSV file.

    Attributes:
        ticks: Array of shape ``(rows, 4)`` in canonical ``FL, FR, RL, RR``
            column order, whatever order the file used.
        time: Timestamps of shape ``(rows,)`` if the file had a ``t`` or
            ``time`` column, else ``None``.
    """

    ticks: FloatArray
    time: FloatArray | None


def read_tick_csv(path: str | Path) -> TickLog:
    """Read a CSV with a header row naming the columns ``fl``, ``fr``, ``rl``, ``rr``.

    Column names are case-insensitive and may come in any order. An optional
    ``t`` (or ``time``) column is kept as timestamps; other columns are
    ignored. Blank lines and lines starting with ``#`` are skipped.
    """
    file = Path(path)
    with file.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.reader(handle)
            if row and any(c.strip() for c in row) and not row[0].lstrip().startswith("#")
        ]
    if not rows:
        raise TickLogError(f"{file}: file is empty")
    header = [cell.strip().lower() for cell in rows[0]]
    missing = [label.lower() for label in WHEEL_LABELS if label.lower() not in header]
    if missing:
        raise TickLogError(
            f"{file}: header must name the columns fl, fr, rl, rr; missing {missing} in {header}"
        )
    columns = [header.index(label.lower()) for label in WHEEL_LABELS]
    time_col = next((header.index(n) for n in ("t", "time") if n in header), None)

    ticks: list[list[float]] = []
    times: list[float] = []
    for number, row in enumerate(rows[1:], start=2):
        try:
            ticks.append([float(row[c]) for c in columns])
            if time_col is not None:
                times.append(float(row[time_col]))
        except (ValueError, IndexError) as exc:
            raise TickLogError(f"{file}: data row {number} is malformed: {row}") from exc
    if not ticks:
        raise TickLogError(f"{file}: no data rows")
    return TickLog(
        ticks=np.array(ticks, dtype=np.float64),
        time=np.array(times, dtype=np.float64) if time_col is not None else None,
    )


def integrate_tick_log(
    config: RobotConfig,
    log: TickLog,
    *,
    cumulative: bool = False,
    slip_noise: float = 0.0,
) -> tuple[FloatArray, MecanumOdometry]:
    """Replay a tick log through the odometry.

    Args:
        config: Robot description.
        log: Counts from :func:`read_tick_csv`.
        cumulative: If true, rows are running encoder totals and the first row
            is the starting reference. Otherwise each row is an increment.
        slip_noise: Passed to :class:`MecanumOdometry`.

    Returns:
        ``(poses, odometry)``. ``poses`` has one ``(x, y, theta)`` row per
        input row; the integrator is returned for its covariance and distance.
    """
    canonical = log.ticks
    deltas = np.diff(canonical, axis=0, prepend=canonical[:1]) if cumulative else canonical
    # Reorder the canonical columns into this robot's wheel order.
    ordered = deltas[:, list(config.order_indices)]
    odometry = MecanumOdometry(config, slip_noise=slip_noise)
    poses = np.zeros((ordered.shape[0], 3), dtype=np.float64)
    for i, row in enumerate(ordered):
        poses[i] = odometry.update_ticks(row).as_array()
    return poses, odometry

"""Optional matplotlib plots (``pip install "mecanum-kinematics[plot]"``).

The dependency-free SVG writer in :mod:`mecanum_kinematics.svg` covers the
common case; this module exists for raster output and for a time-series view
of wheel speeds that the SVG writer does not draw.
"""

from __future__ import annotations

from pathlib import Path

from .simulator import SimulationResult
from .svg import COLOR_ODOMETRY, COLOR_REFERENCE, COLOR_TRUTH


def plot_simulation(result: SimulationResult, path: str | Path) -> Path:
    """Save a two-panel figure: top-down paths, and wheel speeds over time.

    The file type follows the extension of ``path`` (``.png``, ``.pdf`` ...).

    Raises:
        RuntimeError: if matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for non-SVG plots; install it with "
            "'pip install \"mecanum-kinematics[plot]\"' or use a .svg output path"
        ) from exc

    target = Path(path)
    fig, (ax_path, ax_wheels) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax_path.plot(
        result.commanded[:, 0],
        result.commanded[:, 1],
        "--",
        color=COLOR_REFERENCE,
        label="commanded",
    )
    ax_path.plot(result.truth[:, 0], result.truth[:, 1], color=COLOR_TRUTH, label="ground truth")
    ax_path.plot(
        result.odometry[:, 0], result.odometry[:, 1], color=COLOR_ODOMETRY, label="odometry"
    )
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_title(f"'{result.trajectory_name}' path")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend()

    for i, label in enumerate(result.wheel_order):
        ax_wheels.plot(result.time[1:], result.wheel_speeds[:, i], label=label)
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel speeds after motor lag")
    ax_wheels.grid(True, alpha=0.3)
    ax_wheels.legend()

    fig.tight_layout()
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return target

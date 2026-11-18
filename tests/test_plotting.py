from __future__ import annotations

from pathlib import Path

import pytest

from mecanum_kinematics import RobotConfig, SimulationOptions, simulate, square
from mecanum_kinematics.cli import main
from mecanum_kinematics.plotting import plot_simulation

pytest.importorskip("matplotlib", reason="optional 'plot' extra not installed")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_plot_simulation_writes_a_png(robot: RobotConfig, tmp_path: Path) -> None:
    result = simulate(robot, square(1.0, 0.3), SimulationOptions(slip_noise=0.02, seed=1))
    target = plot_simulation(result, tmp_path / "run.png")
    data = target.read_bytes()
    assert data.startswith(PNG_MAGIC)
    assert len(data) > 5000


def test_cli_uses_matplotlib_for_non_svg_output(tmp_path: Path) -> None:
    target = tmp_path / "cli.png"
    assert main(["simulate", "-t", "circle", "--plot", str(target)]) == 0
    assert target.read_bytes().startswith(PNG_MAGIC)

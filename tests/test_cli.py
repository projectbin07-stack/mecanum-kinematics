from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from mecanum_kinematics import __version__
from mecanum_kinematics.cli import main

DATA = Path(__file__).parent / "data"
SRC = Path(__file__).parent.parent / "src"


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    assert main([*argv, "--json"]) == 0
    data: dict[str, Any] = json.loads(capsys.readouterr().out)
    return data


def test_ik_json(capsys: pytest.CaptureFixture[str]) -> None:
    data = run_json(capsys, "ik", "--vx", "0.5", "--vy", "0.2", "--wz", "1.0")
    assert data["config"] == "generic-small"
    assert data["wheel_order"] == ["FL", "FR", "RL", "RR"]
    assert data["wheel_speeds_rad_s"] == pytest.approx([13 / 3, 29.0, 53 / 3, 47 / 3])
    assert data["saturated"] is False


def test_ik_text_reports_saturation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ik", "--vx", "2.0"]) == 0
    out = capsys.readouterr().out
    assert "FL      30.0000     286.48" in out
    assert "saturated: all wheels scaled by 0.4500" in out
    assert "achieved: vx=0.9000 m/s" in out

    assert main(["ik", "--vx", "2.0", "--no-saturate"]) == 0
    out = capsys.readouterr().out
    assert "66.6667" in out
    assert "saturated" not in out


def test_fk_json_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fk", "--json", "--", "-10", "10", "10", "-10"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["twist"]["vx"] == pytest.approx(0.0, abs=1e-12)
    assert data["twist"]["vy"] == pytest.approx(0.3)
    assert data["slip_ratio"] == pytest.approx(0.0, abs=1e-12)

    assert main(["fk", "10", "12", "10", "10"]) == 0
    out = capsys.readouterr().out
    assert "vx=0.3150 m/s  vy=0.0150 m/s  wz=0.0882 rad/s" in out
    assert "rms residual 0.5000 rad/s" in out


def test_ik_fk_respect_custom_wheel_order(capsys: pytest.CaptureFixture[str]) -> None:
    data = run_json(capsys, "ik", "-c", "generic-large", "--vy", "0.38")
    assert data["wheel_order"] == ["FR", "FL", "RR", "RL"]
    assert data["wheel_speeds_rad_s"] == pytest.approx([5.0, -5.0, -5.0, 5.0])


def test_simulate_square_writes_svg_and_csv(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    plot = tmp_path / "out.svg"
    table = tmp_path / "out.csv"
    data = run_json(
        capsys, "simulate", "--trajectory", "square", "--plot", str(plot), "--csv", str(table)
    )
    assert data["trajectory"] == "square"
    assert data["steps"] == 1384
    assert math.hypot(data["final_truth"]["x"], data["final_truth"]["y"]) < 1e-5
    assert data["drift"]["final_position_error_m"] < 1e-12
    assert data["written"] == {"plot": str(plot), "csv": str(table)}

    root = ET.parse(plot).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(root.findall("{http://www.w3.org/2000/svg}polyline")) == 3

    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:4] == ["t", "cmd_x", "cmd_y", "cmd_theta"]
    assert len(rows) == 1 + 1385
    assert float(rows[-1][0]) == pytest.approx(13.84)


def test_simulate_text_with_noise_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["simulate", "-t", "figure-eight", "--slip-noise", "0.02", "--seed", "3", "--quantise"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first
    assert "trajectory:  figure-eight" in first
    assert "quantised=yes" in first
    assert "odometry 1-sigma:" in first
    assert "% of path)" in first


def test_simulate_reports_saturation_and_spin(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "-t", "line", "--speed", "2.0", "--size", "2.0"]) == 0
    assert "saturation:  wheel speeds were scaled down in" in capsys.readouterr().out
    data = run_json(capsys, "simulate", "-t", "spin")
    assert data["drift"]["drift_percent"] is None
    assert data["final_truth"]["theta"] == pytest.approx(2 * math.pi, abs=1e-4)


def test_simulate_non_svg_plot_without_matplotlib_is_a_clean_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make "import matplotlib" fail whether or not it is installed.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert main(["simulate", "-t", "line", "--plot", str(tmp_path / "out.png")]) == 2
    assert "matplotlib is required" in capsys.readouterr().err


def test_odom_delta_and_cumulative(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    poses = tmp_path / "poses.csv"
    plot = tmp_path / "path.svg"
    data = run_json(
        capsys, "odom", str(DATA / "ticks_delta.csv"), "-o", str(poses), "--plot", str(plot)
    )
    assert data["rows"] == 15
    assert data["mode"] == "delta"
    assert data["pose"]["x"] == pytest.approx(2 * math.pi * 0.03)
    assert data["pose"]["y"] == pytest.approx(math.pi * 0.03)
    with poses.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["t", "x", "y", "theta"]
    assert len(rows) == 16
    assert float(rows[-1][1]) == pytest.approx(2 * math.pi * 0.03)
    assert ET.parse(plot).getroot().tag.endswith("svg")

    assert main(["odom", str(DATA / "ticks_cumulative.csv"), "--cumulative"]) == 0
    out = capsys.readouterr().out
    assert "rows:     16 (cumulative counts)" in out
    assert "pose:     x=+0.1885 m  y=+0.0942 m  theta=+0.0000 rad" in out
    assert "distance: 0.2827 m" in out


def test_odom_with_slip_noise_prints_sigma(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["odom", str(DATA / "ticks_delta.csv"), "--slip-noise", "0.02"]) == 0
    assert "1-sigma:" in capsys.readouterr().out


def test_configs_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["configs"]) == 0
    out = capsys.readouterr().out
    assert "illustrative" in out
    assert "generic-medium" in out
    data = run_json(capsys, "configs")
    assert [c["name"] for c in data["configs"]] == [
        "generic-large",
        "generic-medium",
        "generic-small",
    ]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["ik", "-c", "no-such-robot", "--vx", "1"], "neither an existing TOML file"),
        (["odom", "missing.csv"], "missing.csv"),
        (["simulate", "--dt", "0"], "dt must be positive"),
        (["simulate", "--speed", "-1"], "speed must be positive"),
        (["odom", str(DATA / "ticks_delta.csv"), "--plot", "x.png"], "only writes .svg"),
    ],
)
def test_bad_input_exits_2_with_message(
    capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_argparse_errors_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["simulate", "--trajectory", "zigzag"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"mecanum {__version__}"


def test_module_entry_point_runs_in_a_subprocess() -> None:
    import os

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
    }
    done = subprocess.run(
        [sys.executable, "-m", "mecanum_kinematics", "ik", "--vx", "0.3"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "FL      10.0000      95.49" in done.stdout

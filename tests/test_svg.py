from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mecanum_kinematics import RobotConfig, SimulationOptions, simulate, spin, square
from mecanum_kinematics.svg import (
    COLOR_ODOMETRY,
    COLOR_TRUTH,
    Series,
    nice_ticks,
    render_xy_svg,
    simulation_svg,
    write_svg,
)

NS = {"svg": "http://www.w3.org/2000/svg"}


def _points(polyline: ET.Element) -> list[tuple[float, float]]:
    pairs = polyline.attrib["points"].split()
    return [(float(p.split(",")[0]), float(p.split(",")[1])) for p in pairs]


def test_nice_ticks() -> None:
    assert nice_ticks(0.0, 1.0, target=5) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert nice_ticks(-0.3, 1.3, target=4) == pytest.approx([0.0, 0.5, 1.0])
    assert nice_ticks(0.0, 100.0, target=4) == pytest.approx([0.0, 50.0, 100.0])
    assert nice_ticks(-12.0, 12.0, target=5) == pytest.approx([-10.0, -5.0, 0.0, 5.0, 10.0])
    with pytest.raises(ValueError, match="greater"):
        nice_ticks(1.0, 1.0)


def test_svg_is_well_formed_with_equal_axis_scale() -> None:
    # A 2 m x 1 m rectangle must be drawn exactly twice as wide as it is tall.
    box = Series("box", [0.0, 2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0], COLOR_TRUTH)
    document = render_xy_svg([box], title="Box <test> & more", subtitle="sub")
    root = ET.fromstring(document)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.find("svg:title", NS).text == "Box <test> & more"  # type: ignore[union-attr]
    polylines = root.findall("svg:polyline", NS)
    assert len(polylines) == 1
    pts = _points(polylines[0])
    width = pts[1][0] - pts[0][0]
    height = pts[1][1] - pts[2][1]
    assert width == pytest.approx(2 * height, rel=1e-3)
    # SVG y grows downwards, so the top edge (y = 1) has the smaller pixel y.
    assert pts[2][1] < pts[1][1]
    # Everything stays inside the canvas.
    assert all(0 <= x <= 720 and 0 <= y <= 600 for x, y in pts)


def test_simulation_svg_has_three_series_legend_and_markers(robot: RobotConfig) -> None:
    result = simulate(robot, square(1.0, 0.3), SimulationOptions(slip_noise=0.02, seed=2))
    root = ET.fromstring(simulation_svg(result))
    polylines = root.findall("svg:polyline", NS)
    assert [p.attrib["stroke"] for p in polylines][1:] == [COLOR_TRUTH, COLOR_ODOMETRY]
    assert "stroke-dasharray" in polylines[0].attrib  # commanded path is dashed
    texts = [t.text for t in root.findall("svg:text", NS)]
    for label in ("commanded", "ground truth", "odometry", "x [m]", "y [m]"):
        assert label in texts
    assert any(t is not None and "final odometry error" in t for t in texts)
    circles = root.findall("svg:circle", NS)
    assert len(circles) == 3
    tooltip = circles[2].find("svg:title", NS)
    assert tooltip is not None
    assert tooltip.text is not None
    assert re.match(r"odometry end: x=-?\d+\.\d{3} m, y=-?\d+\.\d{3} m", tooltip.text)


def test_long_series_are_decimated_but_keep_their_end_point() -> None:
    n = 10001
    xs = [i / (n - 1) for i in range(n)]
    document = render_xy_svg([Series("ramp", xs, xs, COLOR_TRUTH)], title="ramp")
    root = ET.fromstring(document)
    pts = _points(root.findall("svg:polyline", NS)[0])
    assert len(pts) <= 1501
    circle = root.findall("svg:circle", NS)[0]
    assert pts[-1][0] == pytest.approx(float(circle.attrib["cx"]), abs=0.01)


def test_degenerate_path_still_renders(robot: RobotConfig) -> None:
    # Spinning in place never leaves the origin: the plot must not divide by zero.
    result = simulate(robot, spin(1.0, 2.0))
    root = ET.fromstring(simulation_svg(result, title="spin"))
    assert len(root.findall("svg:polyline", NS)) == 3


def test_invalid_series_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        Series("bad", [0.0, 1.0], [0.0], COLOR_TRUTH)
    with pytest.raises(ValueError, match="nothing to plot"):
        render_xy_svg([Series("empty", [], [], COLOR_TRUTH)], title="t")
    with pytest.raises(ValueError, match="non-finite"):
        render_xy_svg([Series("nan", [0.0, float("nan")], [0.0, 1.0], COLOR_TRUTH)], title="t")


def test_write_svg(tmp_path: Path) -> None:
    target = write_svg(tmp_path / "plot.svg", "<svg/>\n")
    assert target.read_bytes() == b"<svg/>\n"

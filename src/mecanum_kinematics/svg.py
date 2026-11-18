"""Dependency-free SVG writer for top-down path plots.

Only the standard library is used, so ``mecanum simulate --plot out.svg``
works on a robot's companion computer without matplotlib. The plot keeps a
1:1 aspect ratio, because a path drawn with unequal axis scales misrepresents
drift.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .kinematics import FloatArray
from .simulator import SimulationResult

# Colours: a neutral reference plus the first two slots of a categorical
# palette chosen to stay distinguishable under colour-vision deficiency.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
AXIS = "#b5b3ab"
COLOR_REFERENCE = "#8a8983"
COLOR_TRUTH = "#2a78d6"
COLOR_ODOMETRY = "#eb6834"

_FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_MAX_POINTS = 1500


@dataclass(frozen=True)
class Series:
    """One polyline of the plot."""

    label: str
    x: Sequence[float] | FloatArray
    y: Sequence[float] | FloatArray
    color: str
    dashed: bool = False

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"series {self.label!r}: x and y differ in length ({len(self.x)} != {len(self.y)})"
            )


def nice_ticks(low: float, high: float, target: int = 6) -> list[float]:
    """Round tick positions (steps of 1, 2 or 5 times a power of ten) covering ``[low, high]``."""
    if not high > low:
        raise ValueError("high must be greater than low")
    raw = (high - low) / max(1, target)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = magnitude * 10.0
    for multiple in (1.0, 2.0, 5.0):
        if raw <= multiple * magnitude:
            step = multiple * magnitude
            break
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    return [0.0 if i == 0 else i * step for i in range(first, last + 1)]


def _format_tick(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) if step < 1.0 else 0
    return f"{value:.{decimals}f}"


def _decimate(values: Sequence[float] | FloatArray) -> list[float]:
    n = len(values)
    if n <= _MAX_POINTS:
        return [float(v) for v in values]
    stride = math.ceil(n / _MAX_POINTS)
    picked = [float(values[i]) for i in range(0, n, stride)]
    if (n - 1) % stride:
        picked.append(float(values[-1]))
    return picked


def render_xy_svg(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str = "",
    x_label: str = "x [m]",
    y_label: str = "y [m]",
    width: int = 720,
    height: int = 600,
) -> str:
    """Render polylines on equal-scale axes and return the SVG document as a string."""
    drawable = [s for s in series if len(s.x) > 0]
    if not drawable:
        raise ValueError("nothing to plot: every series is empty")

    xs = [float(v) for s in drawable for v in s.x]
    ys = [float(v) for s in drawable for v in s.y]
    if not all(math.isfinite(v) for v in xs + ys):
        raise ValueError("cannot plot non-finite coordinates")

    left, right, top, bottom = 64.0, 24.0, 86.0, 52.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Data window: pad the bounding box, then widen one axis so that one
    # metre spans the same number of pixels horizontally and vertically.
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 0.1)
    pad = 0.08 * span
    x_min, x_max, y_min, y_max = x_min - pad, x_max + pad, y_min - pad, y_max + pad
    scale = min(plot_w / (x_max - x_min), plot_h / (y_max - y_min))
    extra_x = plot_w / scale - (x_max - x_min)
    extra_y = plot_h / scale - (y_max - y_min)
    x_min, x_max = x_min - extra_x / 2.0, x_max + extra_x / 2.0
    y_min, y_max = y_min - extra_y / 2.0, y_max + extra_y / 2.0

    def px(x: float) -> float:
        return left + (x - x_min) * scale

    def py(y: float) -> float:
        return top + (y_max - y) * scale

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" font-family="{_FONT}">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{left:.0f}" y="28" font-size="16" font-weight="600" '
        f'fill="{TEXT_PRIMARY}">{escape(title)}</text>',
    ]
    if subtitle:
        out.append(
            f'<text x="{left:.0f}" y="47" font-size="12" '
            f'fill="{TEXT_SECONDARY}">{escape(subtitle)}</text>'
        )

    # Grid and tick labels. Both axes share one step because they share one scale.
    ticks_x = nice_ticks(x_min, x_max, target=max(2, round(plot_w / 90)))
    step = ticks_x[1] - ticks_x[0] if len(ticks_x) > 1 else (x_max - x_min)
    first_y = math.ceil(y_min / step - 1e-9)
    last_y = math.floor(y_max / step + 1e-9)
    ticks_y = [0.0 if i == 0 else i * step for i in range(first_y, last_y + 1)]
    for tx in ticks_x:
        out.append(
            f'<line x1="{px(tx):.1f}" y1="{top:.1f}" x2="{px(tx):.1f}" y2="{top + plot_h:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{px(tx):.1f}" y="{top + plot_h + 18:.1f}" font-size="11" '
            f'text-anchor="middle" fill="{TEXT_SECONDARY}">{_format_tick(tx, step)}</text>'
        )
    for ty in ticks_y:
        out.append(
            f'<line x1="{left:.1f}" y1="{py(ty):.1f}" x2="{left + plot_w:.1f}" y2="{py(ty):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 8:.1f}" y="{py(ty) + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{TEXT_SECONDARY}">{_format_tick(ty, step)}</text>'
        )
    out.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
        f'fill="none" stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 12:.1f}" font-size="12" '
        f'text-anchor="middle" fill="{TEXT_SECONDARY}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(16 {top + plot_h / 2:.1f}) rotate(-90)" font-size="12" '
        f'text-anchor="middle" fill="{TEXT_SECONDARY}">{escape(y_label)}</text>'
    )

    # Legend: one row above the plot. Labels stay in text colour; the swatch
    # carries identity.
    cursor = left
    for s in series:
        dash = ' stroke-dasharray="6 4"' if s.dashed else ""
        out.append(
            f'<line x1="{cursor:.1f}" y1="68" x2="{cursor + 22:.1f}" y2="68" '
            f'stroke="{s.color}" stroke-width="2" stroke-linecap="round"{dash}/>'
        )
        out.append(
            f'<text x="{cursor + 28:.1f}" y="72" font-size="12" '
            f'fill="{TEXT_PRIMARY}">{escape(s.label)}</text>'
        )
        cursor += 28 + 7.0 * len(s.label) + 22

    # Paths, then end markers on top with a surface-coloured ring.
    for s in drawable:
        points = " ".join(
            f"{px(x):.2f},{py(y):.2f}" for x, y in zip(_decimate(s.x), _decimate(s.y), strict=True)
        )
        dash = ' stroke-dasharray="6 4"' if s.dashed else ""
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{s.color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
    start = drawable[0]
    sx, sy = px(float(start.x[0])), py(float(start.y[0]))
    out.append(
        f'<path d="M{sx - 6:.1f},{sy:.1f} H{sx + 6:.1f} M{sx:.1f},{sy - 6:.1f} V{sy + 6:.1f}" '
        f'stroke="{TEXT_PRIMARY}" stroke-width="1.5"><title>start</title></path>'
    )
    for s in drawable:
        ex, ey = float(s.x[-1]), float(s.y[-1])
        out.append(
            f'<circle cx="{px(ex):.2f}" cy="{py(ey):.2f}" r="4.5" fill="{s.color}" '
            f'stroke="{SURFACE}" stroke-width="2">'
            f"<title>{escape(s.label)} end: x={ex:.3f} m, y={ey:.3f} m</title></circle>"
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def simulation_svg(result: SimulationResult, *, title: str | None = None) -> str:
    """Top-down plot of commanded path, ground truth and odometry for one run."""
    drift = result.drift
    percent = "" if drift.drift_percent is None else f" ({drift.drift_percent:.2f}% of path)"
    subtitle = (
        f"final odometry error {drift.final_position_error * 1000:.1f} mm{percent}, "
        f"heading error {math.degrees(drift.final_heading_error):.2f} deg, "
        f"path {drift.path_length:.2f} m. Cross = start, dots = end."
    )
    series = [
        Series("commanded", result.commanded[:, 0], result.commanded[:, 1], COLOR_REFERENCE, True),
        Series("ground truth", result.truth[:, 0], result.truth[:, 1], COLOR_TRUTH),
        Series("odometry", result.odometry[:, 0], result.odometry[:, 1], COLOR_ODOMETRY),
    ]
    return render_xy_svg(
        series,
        title=title or f"Simulated '{result.trajectory_name}' trajectory",
        subtitle=subtitle,
    )


def write_svg(path: str | Path, document: str) -> Path:
    """Write an SVG document to ``path`` as UTF-8 and return the path."""
    target = Path(path)
    target.write_text(document, encoding="utf-8", newline="\n")
    return target

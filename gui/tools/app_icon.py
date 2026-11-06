# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Draw the application's icon: a sun rising over a few-cycle pulse.

Run ``python gui/tools/app_icon.py`` to rewrite ``solprop.svg`` and its PNG renderings
in ``gui/src/solprop_gui/ui/assets/icon/``; a test checks the committed SVG is this
script's. The SVG is the master, drawn on a 1024 grid with the rounded square and margin
macOS gives an application icon, so the dock shows it at the size of its neighbours.
The PNGs, 16 to 1024 pixels, are rendered from it by Qt's own SVG renderer, so a
platform that asks for a bitmap gets one drawn for its size; up to 64 pixels from a
compact variant (``solprop-compact.svg``) that fills the square, as Windows and Linux
draw their small icons, with thicker strokes that stay clear at 16 pixels.

The pulse is a real few-cycle pulse, a cosine carrier under a Gaussian envelope, sampled
finely enough that its curve is smooth at 1024 pixels; the sun is the "sol" of solprop
and the pulse its "prop".
"""

import math
import sys
from pathlib import Path

ICON = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "solprop_gui"
    / "ui"
    / "assets"
    / "icon"
)
"""Where the icon is written."""

SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
"""The PNG renderings' sizes, pixels."""

COMPACT = 64
"""The largest size drawn from the compact variant, pixels."""

_SKY = ("#0a2540", "#0b6e99")
"""The background's gradient, top to bottom: deep blue to the window's accent."""

_SUN = ("#ffe08a", "#f59e0b")
"""The sun's gradient, centre to edge."""

_PULSE = "#f3fbff"
"""The pulse's colour."""


def _pulse_path() -> str:
    """Return the pulse as an SVG path: a carrier under a Gaussian envelope.

    Returns
    -------
    str
        The path's data, in the 1024 grid.
    """
    left, right, middle, axis = 170.0, 854.0, 512.0, 668.0
    width, period, height = 128.0, 92.0, 150.0
    points = []
    samples = 400
    for index in range(samples + 1):
        x = left + (right - left) * index / samples
        envelope = math.exp(-(((x - middle) / width) ** 2))
        y = axis - height * envelope * math.cos(2.0 * math.pi * (x - middle) / period)
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def _rays() -> str:
    """Return the sun's rays as SVG lines.

    Returns
    -------
    str
        One ``<line>`` per ray, the lower ones left out behind the pulse's horizon.
    """
    cx, cy, inner, outer = 512.0, 430.0, 200.0, 262.0
    lines = []
    for index in range(12):
        angle = math.pi * index / 6.0
        if math.sin(angle) > 0.2:  # below the sun: the horizon hides them
            continue
        x1, y1 = cx + inner * math.cos(angle), cy + inner * math.sin(angle)
        x2, y2 = cx + outer * math.cos(angle), cy + outer * math.sin(angle)
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    return "".join(lines)


def svg(*, compact: bool = False) -> str:
    """Return the icon's SVG.

    Parameters
    ----------
    compact : bool, optional
        Whether to fill the square, with no margin and thicker strokes, for the small
        sizes.

    Returns
    -------
    str
        The document.
    """
    box = "100 100 824 824" if compact else "0 0 1024 1024"
    rays, pulse = ("40", "40") if compact else ("30", "30")
    outline = (
        ""
        if compact
        else f'<path d="{_pulse_path()}" fill="none" stroke="{_SKY[0]}" '
        'stroke-opacity="0.55" stroke-width="58" stroke-linecap="round" '
        'stroke-linejoin="round"/>\n'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}"
 width="1024" height="1024">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{_SKY[0]}"/><stop offset="1" stop-color="{_SKY[1]}"/>
</linearGradient>
<radialGradient id="sun" cx="0.5" cy="0.45" r="0.55">
<stop offset="0" stop-color="{_SUN[0]}"/><stop offset="1" stop-color="{_SUN[1]}"/>
</radialGradient>
<clipPath id="tile"><rect x="100" y="100" width="824" height="824" rx="185"/></clipPath>
</defs>
<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#sky)"/>
<g clip-path="url(#tile)">
<g stroke="{_SUN[0]}" stroke-width="{rays}" stroke-linecap="round">{_rays()}</g>
<circle cx="512" cy="430" r="165" fill="url(#sun)"/>
{outline}<path d="{_pulse_path()}" fill="none" stroke="{_PULSE}" stroke-width="{pulse}"
 stroke-linecap="round" stroke-linejoin="round"/>
</g>
</svg>
"""


def render(source: Path, size: int, target: Path) -> None:
    """Render the SVG to a square PNG.

    Parameters
    ----------
    source : pathlib.Path
        The SVG.
    size : int
        The PNG's side, pixels.
    target : pathlib.Path
        Where it is written.
    """
    from PySide6.QtCore import Qt  # noqa: PLC0415 - Qt only when rendering
    from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: PLC0415
    from PySide6.QtSvg import QSvgRenderer  # noqa: PLC0415

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(str(source)).render(painter)
    painter.end()
    image.save(str(target))


def main() -> None:
    """Write the SVG and its renderings."""
    ICON.mkdir(parents=True, exist_ok=True)
    source = ICON / "solprop.svg"
    source.write_text(svg(), encoding="utf-8")
    small = ICON / "solprop-compact.svg"
    small.write_text(svg(compact=True), encoding="utf-8")
    for size in SIZES:
        render(small if size <= COMPACT else source, size, ICON / f"solprop-{size}.png")
    sys.stdout.write(f"wrote {source.name} and {len(SIZES)} PNGs in {ICON}\n")


if __name__ == "__main__":
    main()

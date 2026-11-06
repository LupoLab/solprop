# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Draw the start page's geometry pictures: a simple cross-section of each class.

Run ``python gui/tools/geometry_icons.py`` to rewrite
``gui/src/solprop_gui/help/figures/<card key>.svg``; a test checks the committed
pictures are this script's. They are after the user guide's geometry figures, reduced
to what tells the classes apart, and styled with the guide's palette as CSS custom
properties (light, and dark under ``prefers-color-scheme``).
"""

import math
import sys
from pathlib import Path

FIGURES = (
    Path(__file__).resolve().parents[1] / "src" / "solprop_gui" / "help" / "figures"
)

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SPDX-FileCopyrightText: 2026 John C. Travers and contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150" viewBox="0 0 200 150">
<style>
svg {
    --ink: #1b1f24;
    --dim: #5b6673;
    --gas-fill: #e9f1fb;
    --glass-fill: #d5e9dc;
    --glass-edge: #2c7a55;
    --core-fill: #cfe0f4;
    --core-edge: #2f5eb0;
    --beam: #b8334f;
    --beam-b: #6f45a6;
    --beam-c: #a96a15;
}
@media (prefers-color-scheme: dark) {
  svg {
    --ink: #e7eaef;
    --dim: #9aa5b3;
    --gas-fill: #18222f;
    --glass-fill: #173528;
    --glass-edge: #5cb98a;
    --core-fill: #1d3150;
    --core-edge: #7ba6e8;
    --beam: #ee8ca0;
    --beam-b: #b99ae6;
    --beam-c: #e0a44f;
  }
}
.glass { fill: var(--glass-fill); stroke: var(--glass-edge); stroke-width: 2; }
.gas { fill: var(--gas-fill); stroke: var(--glass-edge); stroke-width: 1.2; }
.core { fill: var(--core-fill); stroke: var(--core-edge); stroke-width: 1.5; }
.axis { fill: none; stroke: var(--dim); stroke-width: 1.2; }
.node { fill: var(--dim); stroke: none; }
.ring { fill: none; stroke: var(--dim); stroke-width: 1; stroke-dasharray: 3 3; }
.curve { fill: none; stroke: var(--beam); stroke-width: 2.4; }
.curve-b { fill: none; stroke: var(--beam-b); stroke-width: 2; }
.curve-c { fill: none; stroke: var(--beam-c); stroke-width: 2; }
.label { fill: var(--dim); font-family: sans-serif; font-size: 13px;
  font-style: italic; }
</style>
<defs>
<radialGradient id="spot">
  <stop offset="0" stop-color="var(--beam)" stop-opacity="0.95"/>
  <stop offset="0.55" stop-color="var(--beam)" stop-opacity="0.45"/>
  <stop offset="1" stop-color="var(--beam)" stop-opacity="0"/>
</radialGradient>
<radialGradient id="graded">
  <stop offset="0" stop-color="var(--core-edge)" stop-opacity="0.75"/>
  <stop offset="1" stop-color="var(--core-edge)" stop-opacity="0"/>
</radialGradient>
<linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="var(--beam)" stop-opacity="0"/>
  <stop offset="0.5" stop-color="var(--beam)" stop-opacity="0.9"/>
  <stop offset="1" stop-color="var(--beam)" stop-opacity="0"/>
</linearGradient>
</defs>
"""
CX, CY = 100, 75


def spot(r: float, cx: float = CX, cy: float = CY, rx: float | None = None) -> str:
    """Return a soft spot of light, the beam's place in a cross-section."""
    rx = r if rx is None else rx
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{r}" fill="url(#spot)"/>\n'


def arrow(
    start: tuple[float, float],
    tip: tuple[float, float],
    name: str,
    offset: tuple[float, float] = (4, 4),
) -> str:
    """Return an axis arrow from one point to another, labelled at its tip."""
    (x1, y1), (x2, y2), (dx, dy) = start, tip, offset
    a = math.atan2(y2 - y1, x2 - x1)
    h = [
        (x2 - 7 * math.cos(a - 0.45), y2 - 7 * math.sin(a - 0.45)),
        (x2 - 7 * math.cos(a + 0.45), y2 - 7 * math.sin(a + 0.45)),
    ]
    return (
        f'<path class="axis" d="M{x1},{y1} L{x2},{y2} M{h[0][0]:.1f},{h[0][1]:.1f} '
        f'L{x2},{y2} L{h[1][0]:.1f},{h[1][1]:.1f}"/>\n'
        f'<text class="label" x="{x2 + dx}" y="{y2 + dy}">{name}</text>\n'
    )


def capillary(body: str = "") -> str:
    """Return a capillary's section: a glass ring around its gas, and what is in it."""
    return (
        f'<circle class="glass" cx="{CX}" cy="{CY}" r="64"/>\n'
        f'<circle class="gas" cx="{CX}" cy="{CY}" r="50"/>\n' + body
    )


icons: dict[str, str] = {}
icons["capillary-averaged"] = capillary(spot(34))
icons["capillary-modal"] = capillary(
    spot(22)
    + f'<circle class="curve-b" cx="{CX}" cy="{CY}" r="29"/>\n'
    + f'<circle class="curve-c" cx="{CX}" cy="{CY}" r="40"/>\n'
)
tubes = "".join(
    f'<circle class="glass" cx="{CX + 47 * math.cos(angle):.1f}" '
    f'cy="{CY + 47 * math.sin(angle):.1f}" r="13"/>\n'
    for angle in (2 * math.pi * k / 7 + math.pi / 2 for k in range(7))
)
icons["antiresonant"] = (
    f'<circle class="glass" cx="{CX}" cy="{CY}" r="70"/>\n'
    f'<circle class="gas" cx="{CX}" cy="{CY}" r="62"/>\n' + tubes + spot(26)
)
icons["gnlse"] = (
    '<rect class="glass" x="10" y="86" width="180" height="40" rx="4"/>\n'
    '<rect class="core" x="10" y="100" width="180" height="12"/>\n'
    + '<path class="curve" d="'
    + " ".join(
        f"{'M' if k == 0 else 'L'}{x},{78 - 52 / math.cosh((x - 100) / 9):.1f}"
        for k, x in enumerate(range(20, 182, 2))
    )
    + '"/>\n'
    + '<text class="label" x="150" y="40">β(ω), γ</text>\n'  # noqa: RUF001
)
icons["step-index"] = (
    f'<circle class="glass" cx="{CX}" cy="{CY}" r="64"/>\n'
    f'<circle class="core" cx="{CX}" cy="{CY}" r="18"/>\n' + spot(20)
)
icons["grin"] = (
    f'<circle class="glass" cx="{CX}" cy="{CY}" r="64"/>\n'
    f'<circle cx="{CX}" cy="{CY}" r="34" fill="url(#graded)"/>\n' + spot(20)
)
icons["rectangular"] = (
    '<rect class="glass" x="22" y="27" width="156" height="96" rx="3"/>\n'
    '<rect class="gas" x="36" y="41" width="128" height="68"/>\n' + spot(26, rx=50)
)
icons["slab"] = (
    '<rect class="glass" x="8" y="22" width="166" height="22"/>\n'
    '<rect class="glass" x="8" y="106" width="166" height="22"/>\n'
    '<rect class="gas" x="8" y="44" width="166" height="62" stroke="none"/>\n'
    '<rect x="8" y="50" width="166" height="50" fill="url(#band)"/>\n'
    + arrow((36, 140), (150, 140), "y", (6, 4))
    + arrow((186, 100), (186, 50), "x", (-4, -6))
)
dots = "".join(
    f'<circle class="node" cx="{40 + 12 * i}" cy="{27 + 12 * j}" r="1.6"/>\n'
    for i in range(11)
    for j in range(9)
)
icons["cartesian"] = (
    dots
    + spot(30, rx=40)
    + arrow((28, 137), (72, 137), "x")
    + arrow((28, 137), (28, 93), "y", (-12, 2))
)
xs = [30 + 2 * k for k in range(71)]
curve = " ".join(
    f"{'M' if k == 0 else 'L'}{x},{98 - 70 / math.cosh((x - 100) / 14):.1f}"
    for k, x in enumerate(xs)
)
icons["line"] = (
    "".join(
        f'<circle class="node" cx="{30 + 7 * i}" cy="105" r="1.8"/>\n'
        for i in range(21)
    )
    + f'<path class="curve" d="{curve}"/>\n'
    + arrow((30, 125), (174, 125), "x")
)
icons["radial"] = (
    "".join(
        f'<circle class="ring" cx="{CX}" cy="{CY}" r="{r}"/>\n'
        for r in (14, 28, 42, 56, 68)
    )
    + spot(32)
    + arrow((CX, CY), (CX + 66, CY - 24), "r", (2, -2))
)


def pictures() -> dict[str, str]:
    """Return each class's picture, by its card's key.

    Returns
    -------
    dict
        The SVG text of each.
    """
    return {key: HEAD + body + "</svg>\n" for key, body in icons.items()}


def main(out: Path = FIGURES) -> None:
    """Write the pictures.

    Parameters
    ----------
    out : pathlib.Path, optional
        The folder; the package's by default.
    """
    out.mkdir(parents=True, exist_ok=True)
    for key, text in pictures().items():
        (out / f"{key}.svg").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES)

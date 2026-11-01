# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Draw the cross-section figures of ``docs/guide/setup/spatial-decompositions.md``.

This script is **not** run by the docs build. ``poe figures`` runs it, beside the
structure figures' ``docs/dev/figures/build.py``, whenever a figure changes; the SVGs it
writes next to it are committed and the page only includes them, so the manual builds
on an install with no drawing library. ``tests/test_geometry_figures.py`` redraws every
figure and compares it with what is committed, and checks that every section of the
page shows its figure.

Every figure is a cross-section of one spatial decomposition, drawn to scale unless its
standfirst says otherwise, and everything in it that is a number comes from solprop:
the mode fields from ``Mode.field``, ``StepIndexMode.electromagnetic_field`` and the
providers' ``solve``, the node sets from the modes' own quadrature rules, the providers'
node rules and the grids' axes, the collar windows from the grids. The outlines --
walls, bores, tubes, cores, claddings, plates and collars -- are the shapes
:func:`solprop.plotting.geometry.outline` returns, so these figures and the library's
own drawings (``draw``, ``thumbnail``) show one geometry; only how a region is painted
is this script's. Only the radial-and-azimuthal figure, which draws something solprop
does not build, reads Bessel zeros of order one and two from SciPy.

The style is the structure figures': the same sans-serif stack with no embedded font,
the same neutral colours, stroke width, corner radius and arrow heads, read from
``build.py`` by path rather than copied; the regions' colours are the library's light
and dark palettes. Colours are CSS custom properties with a
``prefers-color-scheme`` override, because an SVG included as an image is an isolated
document that sees none of the page's stylesheet. Nothing is drawn through ``<defs>``
-- no marker, gradient, pattern or clip path -- because those mint generated
identifiers and the committed SVGs are compared byte for byte; and every coordinate is
rounded to a hundredth of a user unit, so that a last-place difference in a Bessel
function between two machines cannot move a byte either.
"""

import importlib.util
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
from scipy.special import jn_zeros

from solprop.constants import C
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    MarcatiliMode,
    ModalGuide,
    PlanarGrid,
    PowerLawPermittivity,
    RectMode,
    SlabMode,
    StepIndexMode,
    TubeLattice,
    VincettiMode,
)
from solprop.modes.quadrature import (
    cartesian_nodes,
    quadrature_rule,
    transverse_quadrature,
)
from solprop.plotting import geometry
from solprop.plotting.geometry import Disc, Palette, Ring, Shape
from solprop.profiles import PiecewiseLinear

try:
    import drawsvg
except ImportError:  # pragma: no cover - only on an install without the dev group
    drawsvg = None  # type: ignore[assignment]

__all__ = [
    "DARK",
    "FIGURES",
    "LIGHT",
    "PENDING",
    "UNFIGURED",
    "WIDTH",
    "Figure",
    "antiresonant",
    "capillary_full",
    "capillary_radial",
    "capillary_wall",
    "cartesian_free",
    "general_modal",
    "graded_index",
    "line_free",
    "main",
    "mode_averaged",
    "moving_guide",
    "planar_hybrid",
    "radial_azimuthal",
    "radial_free",
    "rectangular",
    "render",
    "slab",
    "step_index",
    "write",
]

HERE = Path(__file__).resolve().parent
"""Where the SVGs are written, which is beside this script and beside the page."""


def _structure_style() -> ModuleType:
    """Load the structure figures' script, whose style constants these figures share.

    It is a script beside a page rather than a module of the package, so it is loaded
    by path, as ``tests/test_structure_figures.py`` loads it. Loading it draws and
    writes nothing.

    Returns
    -------
    types.ModuleType
        The loaded ``docs/dev/figures/build.py``.
    """
    path = HERE.parents[1] / "dev" / "figures" / "build.py"
    spec = importlib.util.spec_from_file_location("_structure_figure_style", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # NamedTuple resolves its deferred annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STYLE = _structure_style()

WIDTH = 560
"""Width of every figure's ``viewBox``: narrower than the structure figures' 1200, so
that a figure renders at its own size in the prose column and its labels stay legible
on a phone."""

MARGIN = 20
"""Space between the figure's edge and anything drawn in it."""

FONT_STACK: str = _STYLE.FONT_STACK
"""The structure figures' font stack."""

STROKE: float = _STYLE.STROKE
"""The structure figures' stroke width, used for every outline and leader."""

DATA_STROKE = 2 * STROKE
"""Stroke width of a computed profile curve and of the polarisation arrow."""

RADIUS: float = _STYLE.RADIUS
"""The structure figures' corner radius, used for every rounded box."""

TITLE_SIZE = 15
"""Font size of a figure's own title."""

LABEL_SIZE = 12.5
"""Font size of a region's name and of a boundary condition's first line."""

SMALL_SIZE = 11
"""Font size of the lines under a label, of axis labels and of the standfirst."""

LINE = 14.0
"""Baseline-to-baseline distance of the lines of one label, in user units."""

DOT = 1.9
"""Radius of a quadrature node, in user units."""

NEUTRALS = ("ground", "ink", "dim", "rule")
"""The colours taken over from the structure figures' palettes."""

REGION_CLASSES: dict[str, str] = {
    "medium": "gas",
    "wall": "glass",
    "core": "core",
    "cladding": "glass",
    "collar": "collar",
}
"""The CSS class that paints each region of a library outline; a figure whose cladding
is air paints it as gas instead."""

THINNEST = 1.6
"""The thinnest wall drawn, in user units: a thinner one, such as a tube's, is drawn
this thick, as its figure's standfirst says."""


def _regions(palette: Palette) -> dict[str, str]:
    """Return the region colours of one library palette as CSS custom properties.

    Parameters
    ----------
    palette : solprop.plotting.geometry.Palette
        The library's light or dark palette.

    Returns
    -------
    dict of str
        Property name to colour for the gas and the glass, a fill and an edge each.
    """
    return {
        "gas-fill": palette.medium,
        "gas-edge": palette.medium_edge,
        "glass-fill": palette.wall,
        "glass-edge": palette.wall_edge,
    }


LIGHT = {
    **{name: _STYLE.LIGHT[name] for name in NEUTRALS},
    **_regions(geometry.LIGHT),
    "metal-fill": "#e4e0d6",
    "metal-edge": "#7a6848",
    "core-fill": geometry.LIGHT.core,
    "core-edge": geometry.LIGHT.core_edge,
    "collar-fill": geometry.LIGHT.collar,
    "collar-edge": geometry.LIGHT.collar_edge,
    "curve-a": "#b8334f",
    "curve-b": "#6f45a6",
    "curve-c": "#a96a15",
    "pol": "#b8334f",
}
"""The palette on a light ground: the neutrals, one fill and edge per region (the
library's light palette, and a metal of this script's own), and three curve colours."""

DARK = {
    **{name: _STYLE.DARK[name] for name in NEUTRALS},
    **_regions(geometry.DARK),
    "metal-fill": "#35302a",
    "metal-edge": "#c9b48c",
    "core-fill": geometry.DARK.core,
    "core-edge": geometry.DARK.core_edge,
    "collar-fill": geometry.DARK.collar,
    "collar-edge": geometry.DARK.collar_edge,
    "curve-a": "#ee8ca0",
    "curve-b": "#b99ae6",
    "curve-c": "#e0a44f",
    "pol": "#ee8ca0",
}
"""The same palette on a dark ground, with exactly the same keys as :data:`LIGHT`."""

REGIONS = ("gas", "glass", "metal", "core", "collar")
"""The region classes: each has a fill and an edge colour in both palettes."""

CURVES = ("curve-a", "curve-b", "curve-c")
"""The curve colours, in the order a figure uses them."""


# ---------------------------------------------------------------------------- style


def _variables(palette: dict[str, str]) -> str:
    """Return one palette as a block of CSS custom property declarations.

    Parameters
    ----------
    palette : dict of str
        A mapping of property name to colour, either :data:`LIGHT` or :data:`DARK`.

    Returns
    -------
    str
        The declarations, one per line.
    """
    return "\n".join(f"    --{name}: {value};" for name, value in palette.items())


def _stylesheet() -> str:
    """Return the stylesheet every figure carries, with both palettes.

    Returns
    -------
    str
        The contents of the figure's ``<style>`` element.
    """
    regions = "\n".join(
        f".{name} {{ fill: var(--{name}-fill); stroke: var(--{name}-edge); "
        f"stroke-width: {STROKE}; }}"
        for name in REGIONS
    )
    curves = "\n".join(
        f".{name} {{ fill: none; stroke: var(--{name}); stroke-width: {DATA_STROKE}; "
        f"stroke-linejoin: round; stroke-linecap: round; }}\n"
        f".ink-{name} {{ fill: var(--{name}); }}\n"
        f".dot-{name} {{ fill: var(--{name}); stroke: none; }}"
        for name in CURVES
    )
    return f"""
svg {{
{_variables(LIGHT)}
}}
@media (prefers-color-scheme: dark) {{
  svg {{
{_variables(DARK)}
  }}
}}
.ground {{ fill: var(--ground); stroke: none; }}
.card {{ fill: var(--ground); stroke: var(--rule); stroke-width: {STROKE}; }}
.outline {{ fill: none; stroke: var(--ink); stroke-width: {STROKE}; }}
.boundary {{ fill: none; stroke: var(--curve-a); stroke-width: {STROKE};
  stroke-dasharray: 5 3; }}
.guide {{ fill: none; stroke: var(--dim); stroke-width: {STROKE};
  stroke-dasharray: 2 3; }}
.ring {{ fill: none; stroke: var(--rule); stroke-width: {STROKE}; }}
.leader {{ fill: none; stroke: var(--dim); stroke-width: {STROKE}; }}
.axis {{ fill: none; stroke: var(--dim); stroke-width: {STROKE}; }}
.head {{ fill: var(--dim); stroke: none; }}
.node {{ fill: var(--ink); stroke: none; }}
.node-dim {{ fill: var(--dim); stroke: none; }}
.pol {{ fill: none; stroke: var(--pol); stroke-width: {DATA_STROKE};
  stroke-linecap: round; }}
.pol-head {{ fill: var(--pol); stroke: none; }}
text {{ font-family: {FONT_STACK}; fill: var(--ink); white-space: pre; }}
.dim {{ fill: var(--dim); }}
.mid {{ text-anchor: middle; }}
.end {{ text-anchor: end; }}
.bold {{ font-weight: 600; }}
.ink-pol {{ fill: var(--pol); }}
{regions}
{curves}
"""


def _require_drawsvg() -> None:
    """Raise a clear error when the drawing library is not installed.

    Raises
    ------
    RuntimeError
        When ``drawsvg`` cannot be imported. Only drawing needs it.
    """
    if drawsvg is None:  # pragma: no cover - only on an install without the dev group
        raise RuntimeError(
            "drawsvg is not installed, so the geometry figures cannot be drawn. It is "
            "in the project's dev group: run `uv sync` and try again. The committed "
            "SVGs are what the documentation shows; this script only rewrites them."
        )


def _q(value: float) -> float:
    """Round a coordinate to a hundredth of a user unit, so that bytes are stable.

    Parameters
    ----------
    value : float
        A coordinate or a length, in user units.

    Returns
    -------
    float
        The value rounded to two decimals, with a negative zero made positive.
    """
    return round(float(value), 2) + 0.0


# --------------------------------------------------------------------- primitives


def _canvas(height: float, title: str, subtitle: Sequence[str]) -> drawsvg.Drawing:
    """Return a drawing of the standard width with its ground, title and standfirst.

    Parameters
    ----------
    height : float
        The figure's height in user units; the width is always :data:`WIDTH`.
    title : str
        The figure's own title, so that it reads when it is opened on its own.
    subtitle : sequence of str
        The standfirst, one string per line.

    Returns
    -------
    drawsvg.Drawing
        A drawing carrying the stylesheet, a full-bleed ground and the header.
    """
    _require_drawsvg()
    drawing = drawsvg.Drawing(WIDTH, height, origin=(0, 0))
    drawing.append(drawsvg.Raw(f"<style>{_stylesheet()}</style>"))
    drawing.append(drawsvg.Rectangle(0, 0, WIDTH, height, class_="ground"))
    _text(drawing, title, MARGIN, 26, size=TITLE_SIZE, klass="bold")
    for index, line in enumerate(subtitle):
        _text(drawing, line, MARGIN, 44 + index * LINE, size=SMALL_SIZE, klass="dim")
    return drawing


def _text(
    drawing: drawsvg.Drawing,
    content: str,
    x: float,
    y: float,
    *,
    size: float = LABEL_SIZE,
    klass: str = "",
) -> None:
    """Draw one line of text at a baseline position.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    content : str
        The line to draw.
    x, y : float
        The anchor and the baseline, in user units.
    size : float, optional
        The font size; :data:`LABEL_SIZE` by default.
    klass : str, optional
        CSS classes, space separated: ``mid``, ``end``, ``bold``, ``dim`` or an ink.
    """
    drawing.append(drawsvg.Text(content, size, _q(x), _q(y), class_=klass.strip()))


def _label(
    drawing: drawsvg.Drawing,
    lines: Sequence[str],
    x: float,
    y: float,
    *,
    klass: str = "",
) -> float:
    """Draw a label: its first line at label size, the rest smaller and dimmer.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    lines : sequence of str
        The label's lines; the first names the thing, the rest say more.
    x, y : float
        The anchor and the first baseline, in user units.
    klass : str, optional
        Extra classes for every line, such as ``mid`` or ``end``.

    Returns
    -------
    float
        The baseline of the last line drawn.
    """
    baseline = y
    for index, line in enumerate(lines):
        baseline = y + index * LINE
        if index == 0:
            _text(drawing, line, x, baseline, klass=klass)
        else:
            _text(drawing, line, x, baseline, size=SMALL_SIZE, klass=f"dim {klass}")
    return baseline


def _line(
    drawing: drawsvg.Drawing, points: Sequence[tuple[float, float]], klass: str
) -> None:
    """Draw an open polyline.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    points : sequence of tuple of float
        Two or more ``(x, y)`` vertices, in user units.
    klass : str
        The CSS class that strokes it.
    """
    path = drawsvg.Path(class_=klass)
    path.M(_q(points[0][0]), _q(points[0][1]))
    for x, y in points[1:]:
        path.L(_q(x), _q(y))
    drawing.append(path)


def _circle(
    drawing: drawsvg.Drawing, x: float, y: float, radius: float, klass: str
) -> None:
    """Draw a circle.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The centre, in user units.
    radius : float
        The radius, in user units.
    klass : str
        The CSS class that colours it.
    """
    drawing.append(drawsvg.Circle(_q(x), _q(y), _q(radius), class_=klass))


def _annulus(
    drawing: drawsvg.Drawing,
    x: float,
    y: float,
    inner: float,
    outer: float,
    klass: str,
) -> None:
    """Draw a filled ring between two radii, as one path with an even-odd hole.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The centre, in user units.
    inner, outer : float
        The two radii, in user units.
    klass : str
        The region class that fills and edges it.
    """
    path = drawsvg.Path(class_=klass, fill_rule="evenodd")
    for radius in (outer, inner):
        path.M(_q(x + radius), _q(y))
        path.A(_q(radius), _q(radius), 0, 1, 0, _q(x - radius), _q(y))
        path.A(_q(radius), _q(radius), 0, 1, 0, _q(x + radius), _q(y))
        path.Z()
    drawing.append(path)


def _rect(
    drawing: drawsvg.Drawing,
    corner: tuple[float, float],
    size: tuple[float, float],
    klass: str,
    *,
    rounded: bool = False,
) -> None:
    """Draw a rectangle.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    corner : tuple of float
        The top-left corner, in user units.
    size : tuple of float
        Its width and height, in user units.
    klass : str
        The CSS class that colours it.
    rounded : bool, optional
        Give it the structure figures' corner radius.
    """
    radius = RADIUS if rounded else 0
    drawing.append(
        drawsvg.Rectangle(
            _q(corner[0]),
            _q(corner[1]),
            _q(size[0]),
            _q(size[1]),
            rx=radius,
            ry=radius,
            class_=klass,
        )
    )


def _outline(
    drawing: drawsvg.Drawing,
    shapes: Sequence[Shape],
    section: Section,
    *,
    y_scale: float | None = None,
    cladding: str = "glass",
) -> None:
    """Draw a library outline, each region painted by its CSS class.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    shapes : sequence of solprop.plotting.geometry.Shape
        What :func:`solprop.plotting.geometry.outline` returned, in m, back to front.
    section : Section
        Where, and at what scale.
    y_scale : float or None, optional
        User units per metre of y when y is drawn schematically; the section's scale
        by default. Only a box may then be drawn.
    cladding : str, optional
        The class that paints a cladding: ``glass``, or ``gas`` for air.
    """
    classes = {**REGION_CLASSES, "cladding": cladding}
    vertical = section.scale if y_scale is None else y_scale

    def page_y(y: float) -> float:
        """Return the page y of a transverse y, at the vertical scale.

        Parameters
        ----------
        y : float
            Transverse y, in m.

        Returns
        -------
        float
            The page coordinate, in user units.
        """
        return section.cy - y * vertical

    for shape in shapes:
        klass = classes[shape.region]
        if isinstance(shape, Disc):
            x, y = shape.centre
            _circle(
                drawing, section.px(x), page_y(y), shape.radius * section.scale, klass
            )
        elif isinstance(shape, Ring):
            x, y = shape.centre
            outer = shape.outer * section.scale
            inner = min(shape.inner * section.scale, outer - THINNEST)
            _annulus(drawing, section.px(x), page_y(y), inner, outer, klass)
        else:
            left, right = section.px(shape.lower[0]), section.px(shape.upper[0])
            top, bottom = page_y(shape.upper[1]), page_y(shape.lower[1])
            _rect(drawing, (left, top), (right - left, bottom - top), klass)


def _head(
    start: tuple[float, float], tip: tuple[float, float], scale: float = 1.0
) -> tuple[float, ...]:
    """Return the three corners of an arrow head pointing from ``start`` to ``tip``.

    The head is an explicit triangle rather than an SVG marker, which would live in
    ``<defs>`` under a generated identifier; its size is the structure figures'.

    Parameters
    ----------
    start : tuple of float
        A point behind the tip, which sets the direction.
    tip : tuple of float
        Where the head points.
    scale : float, optional
        A multiple of the structure figures' head size.

    Returns
    -------
    tuple of float
        Six rounded coordinates, ``x0, y0, x1, y1, x2, y2``.
    """
    dx, dy = tip[0] - start[0], tip[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    long, half = scale * _STYLE.HEAD_LENGTH, scale * _STYLE.HEAD_HALF_WIDTH
    base = (tip[0] - ux * long, tip[1] - uy * long)
    corners = (
        tip[0],
        tip[1],
        base[0] - uy * half,
        base[1] + ux * half,
        base[0] + uy * half,
        base[1] - ux * half,
    )
    return tuple(_q(value) for value in corners)


def _arrow(
    drawing: drawsvg.Drawing,
    start: tuple[float, float],
    tip: tuple[float, float],
    *,
    paint: tuple[str, str] = ("axis", "head"),
    both: bool = False,
    scale: float = 1.0,
) -> None:
    """Draw a straight arrow with a solid head, or a head at each end.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    start, tip : tuple of float
        The two ends, in user units.
    paint : tuple of str, optional
        The CSS class that strokes the shaft and the one that fills the head.
    both : bool, optional
        Put a head at the start as well, for a polarisation direction.
    scale : float, optional
        A multiple of the structure figures' head size.
    """
    shaft, head = paint
    _line(drawing, [start, tip], shaft)
    drawing.append(drawsvg.Lines(*_head(start, tip, scale), close=True, class_=head))
    if both:
        drawing.append(
            drawsvg.Lines(*_head(tip, start, scale), close=True, class_=head)
        )


def _leader(
    drawing: drawsvg.Drawing,
    target: tuple[float, float],
    anchor: tuple[float, float],
    lines: Sequence[str],
) -> None:
    """Draw a label with a thin line from its first baseline to what it names.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    target : tuple of float
        The point on the drawing the label is about: a boundary, a region.
    anchor : tuple of float
        The start of the label's first baseline.
    lines : sequence of str
        The label, as for :func:`_label`.
    """
    _circle(drawing, *target, 2.2, "node-dim")
    _line(drawing, [target, (anchor[0] - 5, anchor[1] - 4)], "leader")
    _label(drawing, lines, *anchor)


def _polarisation(
    drawing: drawsvg.Drawing,
    centre: tuple[float, float],
    angle: float,
    length: float,
    label: str,
) -> None:
    """Draw the linear polarisation direction as a double-headed arrow, labelled.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    centre : tuple of float
        The arrow's midpoint, in user units.
    angle : float
        Its direction on the page, in rad, anticlockwise from +x.
    length : float
        Its length, in user units.
    label : str
        What it shows, drawn above it.
    """
    dx, dy = 0.5 * length * math.cos(angle), -0.5 * length * math.sin(angle)
    # A card behind the arrow and its label keeps both legible over nodes and rings.
    half = 0.5 * max(length, _STYLE.text_width(label, SMALL_SIZE)) + 5
    _rect(
        drawing,
        (centre[0] - half, centre[1] - 22),
        (2 * half, 30),
        "card",
        rounded=True,
    )
    start = (centre[0] - dx, centre[1] - dy)
    tip = (centre[0] + dx, centre[1] + dy)
    _arrow(drawing, start, tip, paint=("pol", "pol-head"), both=True)
    _text(
        drawing, label, centre[0], centre[1] - 9, size=SMALL_SIZE, klass="mid ink-pol"
    )


def _axes_key(
    drawing: drawsvg.Drawing,
    x: float,
    y: float,
    names: tuple[str, str] = ("x", "y"),
) -> None:
    """Draw the transverse coordinate system as two short arrows from a corner.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The origin of the key, in user units.
    names : tuple of str, optional
        The names of the horizontal and the vertical coordinate.
    """
    _arrow(drawing, (x, y), (x + 30, y))
    _arrow(drawing, (x, y), (x, y - 30))
    _text(drawing, names[0], x + 34, y + 4, size=SMALL_SIZE, klass="dim")
    _text(drawing, names[1], x, y - 35, size=SMALL_SIZE, klass="dim mid")


class Section(NamedTuple):
    """Where a cross-section is drawn and at what scale."""

    cx: float
    """Its centre on the page, x, in user units."""

    cy: float
    """Its centre on the page, y, in user units."""

    scale: float
    """User units per metre of the cross-section."""

    def px(self, x: float) -> float:
        """Return the page x of a transverse coordinate.

        Parameters
        ----------
        x : float
            Transverse x, in m.

        Returns
        -------
        float
            The page coordinate, in user units.
        """
        return self.cx + x * self.scale

    def py(self, y: float) -> float:
        """Return the page y of a transverse coordinate, y pointing up the page.

        Parameters
        ----------
        y : float
            Transverse y, in m.

        Returns
        -------
        float
            The page coordinate, in user units.
        """
        return self.cy - y * self.scale


class Panel(NamedTuple):
    """A plotting frame for a 1-D profile, beneath or beside a cross-section."""

    left: float
    """Its left edge on the page, in user units."""

    right: float
    """Its right edge on the page, in user units."""

    top: float
    """Its top edge on the page, in user units."""

    bottom: float
    """Its bottom edge on the page, in user units."""

    x_range: tuple[float, float]
    """The data interval the horizontal extent spans."""

    y_range: tuple[float, float]
    """The data interval the vertical extent spans, bottom to top."""

    def px(self, x: float) -> float:
        """Return the page x of a data abscissa.

        Parameters
        ----------
        x : float
            The abscissa, in the panel's data units.

        Returns
        -------
        float
            The page coordinate, in user units.
        """
        low, high = self.x_range
        return self.left + (x - low) / (high - low) * (self.right - self.left)

    def py(self, y: float) -> float:
        """Return the page y of a data ordinate.

        Parameters
        ----------
        y : float
            The ordinate, in the panel's data units.

        Returns
        -------
        float
            The page coordinate, in user units.
        """
        low, high = self.y_range
        return self.bottom - (y - low) / (high - low) * (self.bottom - self.top)


def _frame(
    drawing: drawsvg.Drawing,
    panel: Panel,
    xlabel: str,
    ylabel: str,
    *,
    zero: bool = True,
) -> None:
    """Draw a panel's axes: the abscissa at zero or at the bottom, and a left rule.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    panel : Panel
        The frame.
    xlabel : str
        What the horizontal axis is, drawn at its right end.
    ylabel : str
        What the curves are, drawn above the left rule.
    zero : bool, optional
        Draw the abscissa through the data zero rather than along the bottom.
    """
    base = panel.py(0.0) if zero else panel.bottom
    _arrow(drawing, (panel.left, base), (panel.right + 12, base))
    _line(drawing, [(panel.left, panel.bottom), (panel.left, panel.top)], "axis")
    _text(
        drawing, xlabel, panel.right + 12, base + 15, size=SMALL_SIZE, klass="dim end"
    )
    _text(drawing, ylabel, panel.right, panel.top - 4, size=SMALL_SIZE, klass="dim end")


def _curve(
    drawing: drawsvg.Drawing,
    panel: Panel,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    klass: str,
) -> None:
    """Draw one sampled curve in a panel.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    panel : Panel
        The frame.
    x, y : sequence of float
        The samples, in the panel's data units.
    klass : str
        The curve class.
    """
    _line(
        drawing,
        [(panel.px(a), panel.py(b)) for a, b in zip(x, y, strict=True)],
        klass,
    )


def _key(
    drawing: drawsvg.Drawing,
    entries: Sequence[tuple[str, str]],
    x: float,
    y: float,
) -> None:
    """Draw a legend: a short stroke of each curve class and its name.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    entries : sequence of tuple of str
        ``(class, name)`` pairs, top to bottom.
    x, y : float
        The first entry's stroke start and baseline, in user units.
    """
    for index, (klass, name) in enumerate(entries):
        baseline = y + index * LINE
        _line(drawing, [(x, baseline - 4), (x + 18, baseline - 4)], klass)
        _text(drawing, name, x + 24, baseline, size=SMALL_SIZE)


def _guides(
    drawing: drawsvg.Drawing, xs: Sequence[float], top: float, bottom: float
) -> None:
    """Draw dotted verticals that carry a wall from the section down to its profile.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    xs : sequence of float
        The page abscissae, in user units.
    top, bottom : float
        The vertical extent, in user units.
    """
    for x in xs:
        _line(drawing, [(x, top), (x, bottom)], "guide")


def _ray(values: np.ndarray) -> np.ndarray:
    """Return points on the positive x axis, as ``Mode.field`` takes them.

    Parameters
    ----------
    values : numpy.ndarray
        Positions along x, in m.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``: ``(x, 0)`` pairs, which a polar mode reads as ``(r, 0)``
        for ``x >= 0``.
    """
    return np.stack([values, np.zeros_like(values)], axis=-1)


def _polar_profile(
    mode: MarcatiliMode, x: np.ndarray, component: int = 0
) -> np.ndarray:
    """Return one Cartesian component of a polar mode along the whole x axis.

    A polar mode's ``field`` takes ``(r, theta)``; the negative half-axis is
    ``theta = pi``.

    Parameters
    ----------
    mode : solprop.modes.MarcatiliMode
        The mode, whose profile is normalised to unit power by its closed form.
    x : numpy.ndarray
        Positions along x, in m, inside the core.
    component : int, optional
        0 for E_x, 1 for E_y.

    Returns
    -------
    numpy.ndarray
        The component in V/(m sqrt(W)).
    """
    points = np.stack([np.abs(x), np.where(x < 0, math.pi, 0.0)], axis=-1)
    return mode.field(points)[:, component] / math.sqrt(mode.normalisation())


# ------------------------------------------------------------------- the figures

CORE_CX = 170.0
"""Page x of a circular section's centre."""

CORE_CY = 205.0
"""Page y of a circular section's centre."""

CORE_PX = 118.0
"""Page radius of a circular section's outermost drawn radius."""

LABEL_X = 330.0
"""Where the labels to the right of a section begin."""

WALL_EDGE = 1 + geometry.WALL_FRACTION
"""A capillary's drawn outer radius, in bore radii: the library draws its wall
:data:`solprop.plotting.geometry.WALL_FRACTION` of the bore radius thick."""


def _capillary(radius: float, **changes: object) -> MarcatiliMode:
    """Return an evacuated capillary mode on a constant silica-like wall.

    Parameters
    ----------
    radius : float
        Core radius, in m.
    **changes
        Mode indices and kind: ``m``, ``n``, ``kind``, ``phi``.

    Returns
    -------
    solprop.modes.MarcatiliMode
        The mode, whose field and normalisation the figures read.
    """
    return MarcatiliMode(radius, cladding=1.45, loss=False, **changes)  # type: ignore[arg-type]


def _capillary_section(
    drawing: drawsvg.Drawing, section: Section, mode: MarcatiliMode
) -> None:
    """Draw a capillary's outline: a gas-filled bore in a glass wall.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    section : Section
        Where, and at what scale.
    mode : solprop.modes.MarcatiliMode
        A mode of the capillary.
    """
    _outline(drawing, geometry.outline(mode), section)


def mode_averaged() -> drawsvg.Drawing:
    """Draw the mode-averaged decomposition: one node, one area.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 50e-6
    mode = _capillary(radius)
    area = float(mode.aeff())
    disc = math.sqrt(area / math.pi)
    x = np.linspace(-radius, radius, 161)
    power = 2.0 / (8.8541878128e-12 * C)  # int |e|^2 dA of a unit-power profile
    intensity = _polar_profile(mode, x) ** 2 / power * area
    on_axis = float(intensity[80])

    drawing = _canvas(
        500,
        "Mode-averaged: one field and one area",
        (
            "A capillary's HE₁₁ to scale, and the uniform disc of equal effective area "
            "that the",
            "mode-averaged route propagates in its place.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / (WALL_EDGE * radius))
    _capillary_section(drawing, section, mode)
    _circle(drawing, section.cx, section.cy, disc * section.scale, "boundary")
    _circle(drawing, section.cx, section.cy, 3.2, "node")
    _polarisation(drawing, (section.cx - 38, section.cy - 60), 0.0, 44, "A(t), √W")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.cx + 1, section.cy),
        (LABEL_X, 92),
        ("one node: the mode itself", "E = A / √(ε₀c A_eff / 2)"),
    )
    _leader(
        drawing,
        (section.px(disc * 0.7071), section.py(disc * 0.7071)),
        (LABEL_X, 140),
        (
            "disc of area A_eff",
            f"A_eff = {area / radius**2:.3f} a²  (the GNLSE's",
            "surrogate profile)",
        ),
    )
    _leader(
        drawing,
        (section.px(1.09 * radius), section.py(0.0)),
        (LABEL_X, 206),
        (
            "no transverse boundary",
            "the wall is only in n_eff(ω)",
            "and its loss",
        ),
    )

    panel = Panel(
        section.px(-radius), section.px(radius), 375, 465, (-radius, radius), (0, 2.0)
    )
    _guides(drawing, [panel.left, panel.right], section.py(0) + 2, panel.bottom)
    _frame(drawing, panel, "x", "intensity × A_eff / P", zero=False)  # noqa: RUF001 - the glyph is what the figure prints
    _curve(drawing, panel, x, intensity, "curve-a")
    flat = [(-radius, 0.0), (-disc, 0.0), (-disc, 1.0), (disc, 1.0), (disc, 0.0)]
    flat.append((radius, 0.0))
    _curve(drawing, panel, [p[0] for p in flat], [p[1] for p in flat], "curve-b")
    _key(
        drawing,
        (("curve-a", "HE₁₁, on its own profile"), ("curve-b", "the disc: P / A_eff")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        (
            f"on axis: {on_axis:.3f} × P / A_eff",  # noqa: RUF001 - the glyph is what the figure prints
            "self-phase modulation agrees on",
            "either; an ionisation rate does not",
        ),
        LABEL_X,
        432,
    )
    return drawing


def _rule_dots(
    drawing: drawsvg.Drawing,
    section: Section,
    points: np.ndarray,
    klass: str = "node",
) -> None:
    """Draw quadrature nodes given as polar ``(r, theta)`` pairs.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    section : Section
        Where, and at what scale.
    points : numpy.ndarray
        Shape ``(n, 2)``: radius in m and angle in rad.
    klass : str, optional
        The dot class.
    """
    for r, theta in points:
        _circle(
            drawing,
            section.px(r * math.cos(theta)),
            section.py(r * math.sin(theta)),
            DOT,
            klass,
        )


def capillary_radial() -> drawsvg.Drawing:
    """Draw the capillary's radial decomposition: HE1m channels on a radial rule.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 50e-6
    modes = [_capillary(radius, m=order) for order in (1, 2, 3)]
    rule = transverse_quadrature(modes, n_r=12)
    nodes = rule.nodes(modes[0].dimlimits())

    drawing = _canvas(
        500,
        "Capillary modes on a radial rule",
        (
            "HE₁₁, HE₁₂ and HE₁₃ of one bore on the rule transverse_quadrature builds "
            "for them:",
            "twelve Gauss nodes in r/a and one azimuthal node carrying the whole 2π.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / (WALL_EDGE * radius))
    _capillary_section(drawing, section, modes[0])
    for r in rule.xi_r * radius:
        _circle(drawing, section.cx, section.cy, r * section.scale, "ring")
    _rule_dots(drawing, section, nodes)
    _circle(drawing, section.cx, section.cy, radius * section.scale, "boundary")
    _polarisation(drawing, (section.cx - 30, section.cy - 70), 0.0, 44, "E along x")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(0.6 * radius), section.py(0.8 * radius)),
        (LABEL_X, 92),
        (
            "E truncated at r = a",
            "the wall enters only through",
            "the complex n_eff (Marcatili)",
        ),
    )
    middle = nodes[nodes.shape[0] // 2]
    _leader(
        drawing,
        (section.px(middle[0]), section.py(0.0)),
        (LABEL_X, 152),
        (
            "12 radial nodes, one angle",
            "each node stands for a ring:",
            "its weight carries 2π r",
        ),
    )
    _leader(
        drawing,
        (section.px(1.09 * radius * 0.94), section.py(-1.09 * radius * 0.34)),
        (LABEL_X, 212),
        ("glass wall", "a ≫ λ: grazing reflection"),
    )

    x = np.linspace(-radius, radius, 161)
    profiles = [_polar_profile(mode, x) for mode in modes]
    peak = max(float(np.max(np.abs(profile))) for profile in profiles)
    panel = Panel(
        section.px(-WALL_EDGE * radius),
        section.px(WALL_EDGE * radius),
        372,
        470,
        (-WALL_EDGE * radius, WALL_EDGE * radius),
        (-0.6, 1.05),
    )
    _guides(
        drawing,
        [section.px(-radius), section.px(radius)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "E_x(x, 0)")
    for profile, klass in zip(profiles, CURVES, strict=True):
        padded_x = np.concatenate([[-WALL_EDGE * radius], x, [WALL_EDGE * radius]])
        padded = np.concatenate([[0.0], profile / peak, [0.0]])
        _curve(drawing, panel, padded_x, padded, klass)
    _key(
        drawing,
        (("curve-a", "HE₁₁"), ("curve-b", "HE₁₂"), ("curve-c", "HE₁₃")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("|E| does not depend on θ,", "so no azimuthal node is needed"),
        LABEL_X,
        448,
    )
    return drawing


def capillary_full() -> drawsvg.Drawing:
    """Draw the capillary on the full polar rule, with a vector mode's field on it.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 50e-6
    te01 = _capillary(radius, kind="TE", n=0, m=1)
    he31 = _capillary(radius, n=3, m=1)
    he11 = _capillary(radius)
    rule = transverse_quadrature([he11, te01, he31], n_r=6, n_theta=9)
    nodes = rule.nodes(he11.dimlimits())
    fields = te01.field(nodes) / math.sqrt(te01.normalisation())
    longest = float(np.max(np.hypot(fields[:, 0], fields[:, 1])))

    drawing = _canvas(
        500,
        "Capillary modes on the full cross-section",
        (
            "The product rule for a set whose highest azimuthal order is h = 2 (HE₃₁): "
            "6 radial",
            "nodes by n_θ = 4h + 1 = 9 angles, with TE₀₁'s transverse field drawn at "
            "every node.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / (WALL_EDGE * radius))
    _capillary_section(drawing, section, he11)
    for r in rule.xi_r * radius:
        _circle(drawing, section.cx, section.cy, r * section.scale, "ring")
    for (r, theta), (ex, ey) in zip(nodes, fields, strict=True):
        x, y = section.px(r * math.cos(theta)), section.py(r * math.sin(theta))
        length = 20.0 * math.hypot(ex, ey) / longest
        if length > 3.0:
            angle = math.atan2(ey, ex)
            tip = (x + length * math.cos(angle), y - length * math.sin(angle))
            _arrow(drawing, (x, y), tip, paint=("curve-b", "dot-curve-b"), scale=0.6)
    _rule_dots(drawing, section, nodes)
    _circle(drawing, section.cx, section.cy, radius * section.scale, "boundary")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(0.45 * radius), section.py(0.893 * radius)),
        (LABEL_X, 92),
        (
            "E truncated at r = a",
            "wall in the complex n_eff",
        ),
    )
    first = rule.xi_theta[0]
    _leader(
        drawing,
        (
            section.px(rule.xi_r[-1] * radius * math.cos(first)),
            section.py(rule.xi_r[-1] * radius * math.sin(first)),
        ),
        (LABEL_X, 138),
        (
            "6 × 9 nodes",  # noqa: RUF001 - the glyph is what the figure prints
            "angles offset by half a step:",
            "exact for degree ≤ n_θ − 1",  # noqa: RUF001 - the glyph is what the figure prints
        ),
    )
    _label(
        drawing,
        (
            "TE₀₁: E along θ̂",
            "HE, EH, TE and TM of any order;",
            "x and y resolved at every node",
        ),
        LABEL_X,
        204,
    )
    _line(drawing, [(LABEL_X - 24, 200), (LABEL_X - 6, 200)], "curve-b")

    x = np.linspace(-radius, radius, 161)
    profiles = (_polar_profile(he11, x), _polar_profile(te01, x, component=1))
    peak = max(float(np.max(np.abs(profile))) for profile in profiles)
    panel = Panel(
        section.px(-WALL_EDGE * radius),
        section.px(WALL_EDGE * radius),
        372,
        470,
        (-WALL_EDGE * radius, WALL_EDGE * radius),
        (-1.05, 1.05),
    )
    _guides(
        drawing,
        [section.px(-radius), section.px(radius)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "along y = 0")
    for profile, klass in zip(profiles, ("curve-a", "curve-b"), strict=True):
        padded_x = np.concatenate([[-WALL_EDGE * radius], x, [WALL_EDGE * radius]])
        padded = np.concatenate([[0.0], profile / peak, [0.0]])
        _curve(drawing, panel, padded_x, padded, klass)
    _key(
        drawing,
        (("curve-a", "HE₁₁, E_x"), ("curve-b", "TE₀₁, E_y")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("a quartic of order-h modes has", "degree 4h: warned below 4h + 1"),
        LABEL_X,
        432,
    )
    return drawing


def _wall_slope(mode: MarcatiliMode, radius: float) -> float:
    """Return the radial slope of a unit-power profile at the wall, by a difference.

    Parameters
    ----------
    mode : solprop.modes.MarcatiliMode
        The mode.
    radius : float
        Its core radius, in m.

    Returns
    -------
    float
        ``dE_x/dr`` at ``r = a``, in V/(m^2 sqrt(W)), by a one-sided difference of
        ``Mode.field`` a millionth of the radius inside the wall.
    """
    step = 1e-6 * radius
    values = _polar_profile(mode, np.array([radius - step, radius]))
    return float((values[1] - values[0]) / step)


def capillary_wall() -> drawsvg.Drawing:
    """Draw the capillary's wall two ways: truncated, and the condensed Robin wall.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 50e-6
    modes = [_capillary(radius, m=order) for order in (1, 2, 3)]
    rule = transverse_quadrature(modes, n_r=12)
    nodes = rule.nodes(modes[0].dimlimits())
    x = np.linspace(0.0, radius, 101)
    profiles = [_polar_profile(mode, x) for mode in modes]
    peak = max(float(np.max(np.abs(profile))) for profile in profiles)
    slopes = [_wall_slope(mode, radius) for mode in modes]

    drawing = _canvas(
        540,
        "The capillary wall, two ways",
        (
            "One HE₁₁, HE₁₂, HE₁₃ basis on one radial rule. Left: each channel "
            "attenuates on its own.",
            "Right: the outgoing boundary condition, condensed onto the basis, couples "
            "every pair.",
        ),
    )
    for side, (cx, title) in enumerate(
        ((140.0, "truncated at the wall"), (420.0, "the coherent wall"))
    ):
        section = Section(cx, 170, 88 / (WALL_EDGE * radius))
        _capillary_section(drawing, section, modes[0])
        for r in rule.xi_r * radius:
            _circle(drawing, cx, section.cy, r * section.scale, "ring")
        _rule_dots(drawing, section, nodes)
        _circle(drawing, cx, section.cy, radius * section.scale, "boundary")
        # Power leaves through the glass: per channel on the left, through one
        # boundary condition shared by the whole basis on the right.
        for angle in (0.6, 1.6, 2.6, 3.6, 4.6, 5.6):
            inner = 0.92 * radius * section.scale
            outer = 1.3 * radius * section.scale
            start = (cx + inner * math.cos(angle), section.cy - inner * math.sin(angle))
            tip = (cx + outer * math.cos(angle), section.cy - outer * math.sin(angle))
            _arrow(drawing, start, tip, scale=0.7)
        _text(drawing, title, cx, 292, klass="mid bold")
        lines = (
            (
                "E = 0 at r = a",
                "each channel: its own Im n_eff",
                "(Marcatili; incoherent)",
            )
            if side == 0
            else (
                "∂_r E = i k_eff E at r = a",
                "condensed: G = D + g(a) h hᵀ",
                "(CoherentWall; te_tm: two impedances)",
            )
        )
        _label(drawing, lines, cx, 310, klass="mid")
        panel = Panel(cx - 110, cx + 100, 392, 492, (0.0, 1.25 * radius), (-0.7, 1.05))
        _guides(drawing, [panel.px(radius)], 372, panel.bottom)
        _frame(drawing, panel, "r", "E_x(r)")
        for profile, klass in zip(profiles, CURVES, strict=True):
            _curve(drawing, panel, x, profile / peak, klass)
        if side == 1:
            for slope, klass in zip(slopes, CURVES, strict=True):
                # The slope at the wall in the panel's units: the boundary trace h_m
                # the Robin condition couples, continued past the wall as a tangent.
                gradient = slope / peak * radius
                reach = 0.2
                _curve(
                    drawing,
                    panel,
                    [radius, radius * (1 + reach)],
                    [0.0, gradient * reach],
                    klass,
                )
            _text(
                drawing,
                "tangents at the wall: h_m",
                cx + 100,
                512,
                size=SMALL_SIZE,
                klass="dim end",
            )
    _key(
        drawing,
        (("curve-a", "HE₁₁"), ("curve-b", "HE₁₂"), ("curve-c", "HE₁₃")),
        247,
        140,
    )
    return drawing


def antiresonant() -> drawsvg.Drawing:
    """Draw a single-ring tube lattice to scale, with Marcatili's HE11 in its core.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    core = 15e-6
    lattice = TubeLattice(wall_thickness=180e-9, tube_radius=9e-6, n_tubes=6)
    mode = VincettiMode(core, cladding=1.45, loss=False, lattice=lattice)
    tube = lattice.tube_radius
    count = lattice.n_tubes
    gap = 2 * ((core + tube) * math.sin(math.pi / count) - tube)
    jacket = core + 2 * tube
    outer = jacket + geometry.WALL_FRACTION * core
    x = np.linspace(0.0, core, 401)
    profile = _polar_profile(mode, x)
    levels = (0.9, 0.5, 0.1)
    contour = [
        float(np.interp(-level, -(profile**2) / profile[0] ** 2, x)) for level in levels
    ]

    drawing = _canvas(
        500,
        "Antiresonant tube lattice",
        (
            "Six tubes of 9 µm outer radius around a 15 µm core, to scale (the 180 nm "
            "walls are",
            "drawn thicker). The propagated field is Marcatili's HE₁₁ in the core "
            "alone.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / outer)
    _outline(drawing, geometry.outline(mode), section)
    for level, r in zip(levels, contour, strict=True):
        del level
        _circle(drawing, section.cx, section.cy, r * section.scale, "ring")
    _circle(drawing, section.cx, section.cy, core * section.scale, "boundary")
    _polarisation(drawing, (section.cx, section.cy + 10), 0.0, 34, "E along x")
    _axes_key(drawing, MARGIN + 8, 132)

    tube_angle = math.pi / count
    _leader(
        drawing,
        (
            section.px((core + 2 * tube) * math.cos(tube_angle) - 0.4e-6),
            section.py((core + 2 * tube) * math.sin(tube_angle)),
        ),
        (LABEL_X, 92),
        ("tube: outer radius r, wall t", "N = 6; the jacket is not modelled"),
    )
    _leader(
        drawing,
        (section.px(core + 0.5 * gap), section.py(0.0)),
        (LABEL_X, 136),
        (
            f"gap δ = 2[(a + r) sin(π/N) − r] = {gap * 1e6:.1f} µm",  # noqa: RUF001 - the glyph is what the figure prints
            "negative δ (overlap) is refused",
        ),
    )
    _leader(
        drawing,
        (section.px(-0.7071 * core), section.py(0.7071 * core)),
        (LABEL_X, 180),
        (
            "field truncated at r = a",
            "the tubes enter only through the",
            "fitted n_eff and loss (Vincetti)",
        ),
    )
    _label(
        drawing,
        ("contours: |E|² at 0.9, 0.5, 0.1", "of its peak"),
        LABEL_X,
        236,
    )

    xs = np.concatenate([-x[::-1], x[1:]])
    values = np.concatenate([profile[::-1], profile[1:]]) / profile[0]
    panel = Panel(
        section.px(-outer), section.px(outer), 372, 470, (-outer, outer), (-0.15, 1.05)
    )
    _guides(
        drawing, [section.px(-core), section.px(core)], section.py(0) + 2, panel.bottom
    )
    _frame(drawing, panel, "x", "E_x(x, 0)")
    _curve(
        drawing,
        panel,
        np.concatenate([[-outer], xs, [outer]]),
        np.concatenate([[0.0], values, [0.0]]),
        "curve-a",
    )
    _label(
        drawing,
        (
            "the resonances of the tube walls",
            "shape n_eff(ω) and the loss, not",
            "the transverse field",
        ),
        LABEL_X,
        400,
    )
    return drawing


def step_index() -> drawsvg.Drawing:
    """Draw an exact step-index mode: its core, its exterior tail, and its continuity.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 0.3e-6
    mode = StepIndexMode(radius, core=1.45, cladding=1.0, profile_wavelength=800e-9)
    outer = mode.dimlimits()[2][0]
    shown = 3.5 * radius
    inside = np.linspace(0.0, radius, 120)
    outside = np.linspace(radius * (1 + 1e-9), shown, 160)
    # E_x along x is the radial field, which jumps at the interface; along y it is
    # the tangential field, which is continuous there.
    radial = [mode.field(_ray(part))[:, 0] for part in (inside, outside)]
    tangential = [
        mode.field(np.stack([part, np.full_like(part, 0.5 * math.pi)], axis=-1))[:, 0]
        for part in (inside, outside)
    ]
    peak = float(radial[0][0])

    drawing = _canvas(
        500,
        "Step-index fibres, exact",
        (
            "HE₁₁ of a silica strand of 0.3 µm radius in air at 800 nm, solved "
            "exactly: the field",
            "is not truncated, and its exterior tail is integrated to a declared "
            "outer radius.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / shown)
    _outline(drawing, geometry.outline(mode, extent=shown), section, cladding="gas")
    for r in (1.5, 2.0, 2.5, 3.0):
        _circle(drawing, section.cx, section.cy, r * radius * section.scale, "ring")
    _line(
        drawing,
        [(section.px(-shown), section.py(0)), (section.px(shown), section.py(0))],
        "guide",
    )
    _line(
        drawing,
        [(section.px(0), section.py(-shown)), (section.px(0), section.py(shown))],
        "guide",
    )
    _circle(drawing, section.cx, section.cy, radius * section.scale, "boundary")
    _polarisation(
        drawing, (section.px(-1.8 * radius), section.py(1.9 * radius)), 0.0, 30, "E_x"
    )
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(0.7071 * radius), section.py(0.7071 * radius)),
        (LABEL_X, 92),
        (
            "core, n = 1.45: J_n(u r/a)",
            "E_t, H_t continuous at r = a;",
            "radial D continuous, so radial E jumps",
        ),
    )
    _leader(
        drawing,
        (section.px(2.4 * radius), section.py(-1.7 * radius)),
        (LABEL_X, 152),
        (
            "exterior, n = 1: K_n(w r/a)",
            f"integrated to R_out = {outer / radius:.2f} a",
            "= a max(2, 1 + ln 10⁸ / w), off the page",
        ),
    )
    _label(
        drawing,
        (
            "the profile wavelength is declared:",
            "the field is frozen in frequency",
        ),
        LABEL_X,
        212,
    )

    panel = Panel(
        section.px(-shown), section.px(shown), 372, 470, (-shown, shown), (-0.02, 1.05)
    )
    _guides(
        drawing,
        [section.px(-radius), section.px(radius)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x or y", "E_x, unit peak", zero=False)
    for parts, klass in ((radial, "curve-a"), (tangential, "curve-b")):
        for position, values in zip((inside, outside), parts, strict=True):
            _curve(drawing, panel, position, values / peak, klass)
            _curve(drawing, panel, -position, values / peak, klass)
    _key(
        drawing,
        (("curve-a", "along x: radial, jumps"), ("curve-b", "along y: tangential")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("mode-averaged via prop_gnlse(fibre=);", "vector: the general route below"),
        LABEL_X,
        432,
    )
    return drawing


def graded_index() -> drawsvg.Drawing:
    """Draw a graded-index provider's node rule, its law and its two TE channels.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 25e-6
    omega = 2 * math.pi * C / 1.03e-6
    law = PowerLawPermittivity(IndexMaterial(1.45), 0.005)
    provider = GradedIndexProvider(
        permittivity=law,
        exterior=IndexMaterial(1.45 * math.sqrt(0.99)),
        radius=radius,
        family="TE",
        count=2,
        elements=8,
        extent=2.0,
    )
    rule = provider.node_rule(omega, n_r=2)
    sample = provider.solve(omega, 0.0, nodes=rule)
    outer = provider.extent * radius
    xy = rule.xy
    radii = np.hypot(xy[:, 0], xy[:, 1])
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    ray = np.isclose(angles, angles[0])
    order = np.argsort(radii[ray])
    ray_r = radii[ray][order]
    theta0 = float(angles[0])
    # TE_0m: E is azimuthal, so E_theta along any ray is the whole transverse field.
    azimuthal = (
        -math.sin(theta0) * sample.electric[ray][order, 0, :].real
        + math.cos(theta0) * sample.electric[ray][order, 1, :].real
    )
    peak = float(np.max(np.abs(azimuthal)))

    drawing = _canvas(
        500,
        "Graded-index providers",
        (
            "A parabolic core (Δ = 0.005, a = 25 µm) solved by finite elements over "
            "8 core elements",
            "and 8 exterior ones: the rule splits at every element, and beyond the "
            "matching radius is Bessel.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / (1.12 * outer))
    _outline(drawing, geometry.outline(provider, extent=1.12 * outer), section)
    edges = np.concatenate(
        [
            np.linspace(0, 1, provider.elements + 1)[1:],
            np.linspace(1, provider.extent, provider.elements + 1)[1:],
        ]
    )
    for edge in edges:
        _circle(drawing, section.cx, section.cy, edge * radius * section.scale, "ring")
    for (px, py), region in zip(xy, rule.region_index, strict=True):
        _circle(
            drawing,
            section.px(px),
            section.py(py),
            DOT * 0.8,
            "node" if region == 0 else "node-dim",
        )
    _circle(drawing, section.cx, section.cy, outer * section.scale, "boundary")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(0.45 * radius), section.py(0.3 * radius)),
        (LABEL_X, 92),
        (
            "core: ε(r) = ε_axis [1 − 2Δ (r/a)²]",  # noqa: RUF001 - the glyph is what the figure prints
            "P1 finite elements, TE₀ sector",
        ),
    )
    _leader(
        drawing,
        (section.px(-0.7071 * outer), section.py(0.7071 * outer)),
        (LABEL_X, 138),
        (
            "matching radius, extent × a",  # noqa: RUF001 - the glyph is what the figure prints
            "exact decaying Bessel exterior;",
            "its tail power is certified",
        ),
    )
    _leader(
        drawing,
        (section.px(1.06 * outer), section.py(0.0)),
        (LABEL_X, 198),
        (
            "beyond: carried analytically,",
            "not by nodes (refused above 1e-4",
            "of a channel's power)",
        ),
    )

    panel = Panel(
        section.px(-1.12 * outer),
        section.px(1.12 * outer),
        372,
        470,
        (-1.12 * outer, 1.12 * outer),
        (-1.05, 1.05),
    )
    _guides(
        drawing,
        [section.px(-radius), section.px(radius)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "E_y(x, 0)")
    for channel, klass in zip(range(azimuthal.shape[1]), CURVES, strict=False):
        values = azimuthal[:, channel] / peak
        xs = np.concatenate([-ray_r[::-1], ray_r])
        ys = np.concatenate([-values[::-1], values])
        _curve(drawing, panel, xs, ys, klass)
    _key(
        drawing,
        (("curve-a", "TE₀₁"), ("curve-b", "TE₀₂")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("E is azimuthal: E_y on the x axis", "(LP is labelled weak guidance)"),
        LABEL_X,
        432,
    )
    return drawing


def general_modal() -> drawsvg.Drawing:
    """Draw a provider's vector channel: transverse E at its nodes, and E_z beside it.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    radius = 250e-9
    wavelength = 835e-9
    omega = 2 * math.pi * C / wavelength
    mode = StepIndexMode(radius, core=1.45, cladding=1.0, profile_wavelength=wavelength)
    guide = ModalGuide.from_step_index(
        [mode], core=IndexMaterial(1.45), cladding=IndexMaterial(1.0)
    )
    provider = guide.field_provider
    rule = provider.node_rule(omega, n_r=4)
    sample = provider.solve(omega, 0.0, nodes=rule)
    electric = sample.electric[:, :, 0]
    shown = 3.0 * radius
    rule_outer = float(np.max(np.hypot(rule.xy[:, 0], rule.xy[:, 1])))
    x = np.concatenate(
        [np.linspace(-shown, -radius, 90), np.linspace(-radius, radius, 120)[1:-1]]
    )
    x = np.concatenate([x, -x[::-1]])
    field, _ = mode.electromagnetic_field(_ray(x), omega)
    peak = float(np.max(np.abs(field)))

    drawing = _canvas(
        500,
        "The general modal route: vector fields",
        (
            "HE₁₁ of a 250 nm silica strand in air at 835 nm, from the step-index "
            "provider: the transverse",
            "field on the provider's own node rule (inner part shown), and |E_x| and "
            "|E_z| along x.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / shown)
    _outline(drawing, geometry.outline(guide, extent=shown), section, cladding="gas")
    transverse = np.hypot(electric[:, 0].real, electric[:, 1].real)
    longest = float(np.max(transverse))
    for (px, py), (ex, ey, _) in zip(rule.xy, electric, strict=True):
        if math.hypot(px, py) > shown:
            continue
        x0, y0 = section.px(px), section.py(py)
        length = 13.0 * math.hypot(ex.real, ey.real) / longest
        if length > 3.0:
            angle = math.atan2(ey.real, ex.real)
            tip = (x0 + length * math.cos(angle), y0 - length * math.sin(angle))
            _arrow(drawing, (x0, y0), tip, paint=("curve-b", "dot-curve-b"), scale=0.6)
        _circle(drawing, x0, y0, DOT, "node")
    _circle(drawing, section.cx, section.cy, radius * section.scale, "boundary")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(-0.7071 * radius), section.py(0.7071 * radius)),
        (LABEL_X, 92),
        (
            "E, H continuous at r = a",
            "exterior K_n(γr), exact",  # noqa: RUF001 - the glyph is what the figure prints
        ),
    )
    _leader(
        drawing,
        (section.px(0.7071 * shown * 0.97), section.py(-0.7071 * shown * 0.97)),
        (LABEL_X, 136),
        (
            f"the rule runs on to {rule_outer / radius:.1f} a",
            "its discarded tail power is",
            "certified at the lowest frequency",
        ),
    )
    _label(
        drawing,
        (
            "three components at every node",
            "E and H per √W; the flux metric W",
            "pairs e_n × h_m* over the section",  # noqa: RUF001 - the glyph is what the figure prints
        ),
        LABEL_X,
        196,
    )

    panel = Panel(
        section.px(-shown), section.px(shown), 372, 470, (-shown, shown), (-0.05, 1.05)
    )
    _guides(
        drawing,
        [section.px(-radius), section.px(radius)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "|E| along y = 0", zero=False)
    half = x.size // 2
    for component, klass in ((0, "curve-a"), (2, "curve-b")):
        values = np.abs(field[:, component]) / peak
        for part in (slice(0, 89), slice(89, half), slice(half, -89), slice(-89, None)):
            _curve(drawing, panel, x[part], values[part], klass)
    _key(
        drawing,
        (("curve-a", "|E_x|"), ("curve-b", "|E_z|: not small here")),
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("vector γ 25 % above the scalar", "n₂ω/(c A_eff) on this strand"),  # noqa: RUF001 - the glyph is what the figure prints
        LABEL_X,
        432,
    )
    return drawing


def moving_guide() -> drawsvg.Drawing:
    """Draw a taper in the (z, x) plane, with the radial nodes riding its radius.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    stages = (0.0, 6e-5, 1.4e-4, 2e-4)
    taper = PiecewiseLinear.from_stages(stages, [50e-6, 50e-6, 40e-6, 40e-6])
    modes = [_capillary(50e-6, m=order) for order in (1, 2, 3)]
    rule = transverse_quadrature(modes, n_r=8)
    z = np.linspace(stages[0], stages[-1], 201)
    radius = taper(z)
    wall = 12e-6
    left, right, mid = 80.0, 500.0, 205.0
    scale = (right - left) / (stages[-1] - stages[0])

    def page(along: float, across: float) -> tuple[float, float]:
        """Return the page position of a point of the (z, x) plane, to scale.

        Parameters
        ----------
        along : float
            Position along the guide, z, in m.
        across : float
            Transverse position, x, in m.

        Returns
        -------
        tuple of float
            The page coordinates, in user units.
        """
        return left + along * scale, mid - across * scale

    drawing = _canvas(
        590,
        "Guides that move along z",
        (
            "A capillary tapering from 50 to 40 µm radius over 80 µm, to scale in the "
            "(z, x) plane:",
            "the radial nodes ride r = a(z) ξ, so the basis moves with the bore.",
        ),
    )
    for sign in (1.0, -1.0):
        inner = [page(a, sign * r) for a, r in zip(z, radius, strict=True)]
        outer = [page(a, sign * (r + wall)) for a, r in zip(z, radius, strict=True)]
        path = drawsvg.Path(class_="glass")
        path.M(*(_q(v) for v in inner[0]))
        for point in inner[1:] + outer[::-1]:
            path.L(*(_q(v) for v in point))
        path.Z()
        drawing.append(path)
    upper = [page(a, r) for a, r in zip(z, radius, strict=True)]
    lower = [page(a, -r) for a, r in zip(z, radius, strict=True)]
    gas = drawsvg.Path(class_="gas")
    gas.M(*(_q(v) for v in upper[0]))
    for point in upper[1:] + lower[::-1]:
        gas.L(*(_q(v) for v in point))
    gas.Z()
    drawing.append(gas)
    for xi in rule.xi_r:
        for sign in (1.0, -1.0):
            _line(
                drawing,
                [page(a, sign * xi * r) for a, r in zip(z, radius, strict=True)],
                "ring",
            )
    for position in (0.0, stages[-1]):
        local = float(taper(np.array([position]))[0])
        for xi in rule.xi_r:
            for sign in (1.0, -1.0):
                _circle(drawing, *page(position, sign * xi * local), DOT, "node")
    for kink in stages[1:-1]:
        _line(drawing, [page(kink, 66e-6), page(kink, -66e-6)], "guide")
    # The local HE_11 at the entrance and the exit, drawn sideways: the same J_0 of
    # r/a(z), so the profile scales with the bore and does not change shape.
    for position, sign in ((0.0, 1.0), (stages[-1], -1.0)):
        local = float(taper(np.array([position]))[0])
        r = np.linspace(-local, local, 81)
        profile = _polar_profile(_capillary(local), r)
        profile = profile / float(np.max(profile))
        points = [
            (page(position, rr)[0] + sign * (8.0 + 34.0 * value), page(position, rr)[1])
            for rr, value in zip(r, profile, strict=True)
        ]
        _line(drawing, points, "curve-a")
    _axes_key(drawing, MARGIN + 8, 132, ("z", "x"))
    below = -71e-6
    _text(drawing, "z = 0", *page(0.0, below), size=SMALL_SIZE, klass="dim mid")
    _text(drawing, "0.2 mm", *page(stages[-1], below), size=SMALL_SIZE, klass="dim mid")
    _text(
        drawing,
        "kinks: section edges; a′ ≠ 0 between them",  # noqa: RUF001 - the glyph is what the figure prints
        *page(0.5 * (stages[1] + stages[2]), below),
        size=SMALL_SIZE,
        klass="dim mid",
    )
    _key(
        drawing,
        (("curve-a", "the local HE₁₁: a function of r/a(z) alone"),),
        MARGIN + 6,
        384,
    )

    notes = (
        (
            "scale invariant (capillary, Vincetti, a rectangle scaled whole):",
            "S(z) = S(0) a(0)/a(z) and W(z) = W(0) a(z)/a(0), exactly;",
            "one scalar, taper_scale, and their product does not move",
        ),
        (
            "the connection between the channels, C_mn proportional to a′/a:",  # noqa: RUF001 - the glyph is what the figure prints
            "carried by default on the capillary's HE_1m block, with or without",
            "CoherentWall; taper_coupling=False is the adiabatic opt-out",
        ),
        (
            "a solid guide's modes change shape, so it is re-solved along z:",
            "moving S(z), Q(z) and the connection Γ = W⁻¹S;",
            "on numpy_ref, numba_cpu and native",
        ),
    )
    for index, lines in enumerate(notes):
        _label(drawing, lines, MARGIN + 6, 414 + index * 56)
    return drawing


def rectangular() -> drawsvg.Drawing:
    """Draw a rectangular hollow guide to scale, with its Cartesian rule and profiles.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    half_x, half_y = 50e-6, 25e-6
    mode = RectMode(half_x, half_y, cladding=1.45, loss=False)
    order = cartesian_nodes(mode.azimuthal_order)
    rule = quadrature_rule("cartesian", full=True, n_r=order, n_theta=order)
    nodes = rule.nodes(mode.dimlimits())
    wall = geometry.WALL_FRACTION * max(half_x, half_y)
    x = np.linspace(-half_x, half_x, 121)
    y = np.linspace(-half_y, half_y, 81)
    along_x = mode.field(np.stack([x, np.zeros_like(x)], axis=-1))[:, 0]
    along_y = mode.field(np.stack([np.zeros_like(y), y], axis=-1))[:, 0]
    peak = float(np.max(along_x))

    drawing = _canvas(
        440,
        "Rectangular hollow guides",
        (
            "Half-widths a = 50 µm and b = 25 µm, to scale, with the x-polarised "
            "fundamental's",
            f"Cartesian Gauss rule: {order} × {order} nodes, the measured 6h + 6 per "  # noqa: RUF001 - the glyph is what the figure prints
            "axis for h = 1.",
        ),
    )
    section = Section(165.0, 202.0, 120.0 / (half_x + wall))
    _outline(drawing, geometry.outline(mode), section)
    for px, py in nodes:
        _circle(drawing, section.px(px), section.py(py), DOT * 0.9, "node")
    _polarisation(drawing, (section.cx, section.cy + 4), 0.0, 44, "E along x")
    _axes_key(drawing, MARGIN + 8, 112)
    # The profile along y, drawn sideways against the right wall.
    base = section.px(half_x + wall) + 14
    _line(
        drawing,
        [(base, section.py(-half_y - wall)), (base, section.py(half_y + wall))],
        "axis",
    )
    _line(
        drawing,
        [
            (base + 40.0 * value / peak, section.py(yy))
            for yy, value in zip(y, along_y, strict=True)
        ],
        "curve-b",
    )
    _text(
        drawing,
        "E_x(0, y)",
        base + 2,
        section.py(half_y + wall) - 6,
        size=SMALL_SIZE,
        klass="dim",
    )

    _leader(
        drawing,
        (section.px(half_x + 0.5 * wall), section.py(-0.8 * half_y)),
        (LABEL_X + 30, 246),
        (
            "x = ±a: E normal to the wall",
            "TM-like: a_c = ε_cl / (ε_co s)",
        ),
    )
    _leader(
        drawing,
        (section.px(0.4 * half_x), section.py(half_y + 0.5 * wall)),
        (LABEL_X + 30, 92),
        (
            "y = ±b: E parallel to the wall",
            "TE-like: b_c = 1/s, s = √(ε_cl − ε_co)",  # noqa: RUF001 - the glyph is what the figure prints
        ),
    )
    _label(
        drawing,
        (
            "E zero outside the rectangle;",
            "each wall pair adds its complex κ²",
            "to β² (Laakmann–Steier)",  # noqa: RUF001 - the glyph is what the figure prints
            "the y partner swaps a_c and b_c:",
            "not degenerate, its own row",
        ),
        LABEL_X + 30,
        150,
    )

    panel = Panel(
        section.px(-half_x - wall),
        section.px(half_x + wall),
        318,
        408,
        (-half_x - wall, half_x + wall),
        (-0.1, 1.05),
    )
    _guides(
        drawing,
        [section.px(-half_x), section.px(half_x)],
        section.py(-half_y - wall) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "E_x(x, 0)")
    _curve(
        drawing,
        panel,
        np.concatenate([[-half_x - wall], x, [half_x + wall]]),
        np.concatenate([[0.0], along_x / peak, [0.0]]),
        "curve-a",
    )
    _label(
        drawing,
        (
            "cos(πx/2a) cos(πy/2b)",
            "no Kronrod extension: the node",
            "bound stands in for the alarm",
        ),
        LABEL_X + 30,
        336,
    )
    return drawing


def slab() -> drawsvg.Drawing:
    """Draw a hollow planar guide, uniform in y, with its x rule and profiles.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    half_gap = 63.5e-6
    modes = [SlabMode(half_gap, cladding=1.45, order=order) for order in (1, 3)]
    order = cartesian_nodes(modes[-1].azimuthal_order)
    rule = quadrature_rule("slab", full=False, n_r=order)
    nodes = rule.nodes(modes[0].dimlimits())
    plate = geometry.WALL_FRACTION * half_gap
    x = np.linspace(-half_gap, half_gap, 161)
    profiles = [mode.field(_ray(x))[:, 1] for mode in modes]

    drawing = _canvas(
        440,
        "Planar hollow guides, uniform in y",
        (
            "Plates at x = ±a, a = 63.5 µm (Akturk et al. 2009) to scale in x, and "
            "the slab rule:",
            f"{order} Gauss nodes in x on y = 0, the measured 6h + 6 for h = 3; one "
            "y node of unit width.",
        ),
    )
    section = Section(170.0, 208.0, 105.0 / (half_gap + plate))
    top, bottom = section.cy - 80.0, section.cy + 80.0
    _outline(drawing, geometry.outline(modes[0], extent=80.0 / section.scale), section)
    # The free direction: nothing bounds the field in y, and nothing varies along it.
    for end, tip in ((top + 16, top + 2), (bottom - 16, bottom - 2)):
        _arrow(
            drawing,
            (section.px(0.62 * half_gap), end),
            (section.px(0.62 * half_gap), tip),
        )
    _text(
        drawing,
        "uniform in y (per unit width)",
        section.cx,
        top - 8,
        size=SMALL_SIZE,
        klass="dim mid",
    )
    for px, py in nodes:
        _circle(drawing, section.px(px), section.py(py), DOT * 0.9, "node")
    # TE's arrow stands upright, so its label sits beside it rather than above.
    te = (section.px(-0.55 * half_gap), section.cy - 40)
    _arrow(
        drawing,
        (te[0], te[1] + 16),
        (te[0], te[1] - 16),
        paint=("pol", "pol-head"),
        both=True,
    )
    _text(
        drawing, "TE: E along y", te[0] + 8, te[1] + 4, size=SMALL_SIZE, klass="ink-pol"
    )
    _polarisation(drawing, (section.cx, section.cy + 48), 0.0, 44, "TM: E along x")
    _axes_key(drawing, MARGIN + 8, 112)

    _leader(
        drawing,
        (section.px(half_gap + 0.5 * plate), section.cy - 50),
        (LABEL_X + 30, 96),
        (
            "x = ±a: one wall pair,",
            "TE-like 1/s for TE, TM-like",
            "ε_cl / (ε_co s) for TM",
        ),
    )
    _label(
        drawing,
        (
            "E zero outside the gap; the",
            "wall pair's complex κ² in β²",
            "(Laakmann–Steier, b → ∞)",  # noqa: RUF001 - the glyph is what the figure prints
            "N = ε₀c a/2 and A_eff = 4a/3",
            "per metre of y: W/m and J/m",
        ),
        LABEL_X + 30,
        164,
    )

    panel = Panel(
        section.px(-half_gap - plate),
        section.px(half_gap + plate),
        318,
        408,
        (-half_gap - plate, half_gap + plate),
        (-1.1, 1.1),
    )
    _guides(
        drawing, [section.px(-half_gap), section.px(half_gap)], bottom + 2, panel.bottom
    )
    _frame(drawing, panel, "x", "E_y(x)")
    for profile, klass in zip(profiles, ("curve-a", "curve-b"), strict=True):
        _curve(
            drawing,
            panel,
            np.concatenate([[-half_gap - plate], x, [half_gap + plate]]),
            np.concatenate([[0.0], profile, [0.0]]),
            klass,
        )
    _key(drawing, (("curve-a", "TE1"), ("curve-b", "TE3")), LABEL_X + 30, 330)
    _label(
        drawing,
        (
            "a centred beam reaches odd",
            "orders only; no self-focusing",
            "or instability in y here",
        ),
        LABEL_X + 30,
        366,
    )
    return drawing


def planar_hybrid() -> drawsvg.Drawing:
    """Draw the hybrid planar grid: slab modes in x, a collared periodic y axis.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    half_gap = 63.5e-6
    modes = [SlabMode(half_gap, cladding=1.45, order=order) for order in (1, 3)]
    grid = PlanarGrid(modes, 1e-3, 12, window_factor=0.35, n_x=24, x_rule="gauss")
    outer = grid.radius_y * (1 + grid.window_factor)
    plate = geometry.WALL_FRACTION * half_gap
    x = np.linspace(-half_gap, half_gap, 161)
    profiles = [mode.field(_ray(x))[:, 1] for mode in modes]

    drawing = _canvas(
        470,
        "Planar guides resolved in y",
        (
            "The same plates, a = 63.5 µm, to scale in x; y schematic. The slab "
            "rule's nodes in x",
            f"({grid.x.size} here) on every one of {grid.n_y} periodic y samples, "
            "the outer rows in the collar.",
        ),
    )
    section = Section(170.0, 212.0, 105.0 / (half_gap + plate))
    top, bottom = section.cy - 84.0, section.cy + 84.0
    height = bottom - top

    # y is drawn schematically: the periodic cell spans the section's height.
    per_metre = 0.5 * height / outer
    _outline(drawing, geometry.outline(grid), section, y_scale=per_metre)
    for value, window in zip(grid.y, grid.window_y, strict=True):
        for position in grid.x:
            _circle(
                drawing,
                section.px(position),
                section.cy - value * per_metre,
                DOT * 0.7,
                "node" if window == 1.0 else "node-dim",
            )
    # Periodic in y: what leaves through one end re-enters through the other.
    x_arrow = section.px(0.62 * half_gap)
    _arrow(drawing, (x_arrow, top + 14), (x_arrow, top - 12))
    _arrow(drawing, (x_arrow, bottom + 12), (x_arrow, bottom - 14))
    # The nodes fill the gap, so the polarisation stands beside the left plate.
    te = (section.px(-half_gap - plate) - 30, section.cy + 8)
    _arrow(
        drawing,
        (te[0], te[1] + 16),
        (te[0], te[1] - 16),
        paint=("pol", "pol-head"),
        both=True,
    )
    _text(drawing, "TE: E_y", te[0], te[1] - 24, size=SMALL_SIZE, klass="ink-pol mid")
    _axes_key(drawing, MARGIN + 8, 112)

    _leader(
        drawing,
        (section.px(half_gap + 0.5 * plate), section.cy - 30),
        (LABEL_X + 30, 96),
        (
            "x = ±a: the slab modes,",
            "one wall pair each; complex",
            "κ_p²(ω) in the square below",
        ),
    )
    _leader(
        drawing,
        (
            section.px(0.2 * half_gap),
            section.cy - 0.5 * (grid.radius_y + outer) * per_metre,
        ),
        (LABEL_X + 30, 150),
        ("periodic + absorbing", "collar in y (applied)"),
    )
    _label(
        drawing,
        (
            "columns (p, k_y), p fastest",
            "k_z² = k²ε − κ_p² − k_y²:",  # noqa: RUF001 - the glyph is what the figure prints
            "exact (non-paraxial) in y",
            "energy J, power W",
            "constant gap and media",
        ),
        LABEL_X + 30,
        196,
    )

    panel = Panel(
        section.px(-half_gap - plate),
        section.px(half_gap + plate),
        336,
        428,
        (-half_gap - plate, half_gap + plate),
        (-1.1, 1.1),
    )
    _guides(
        drawing, [section.px(-half_gap), section.px(half_gap)], bottom + 2, panel.bottom
    )
    _frame(drawing, panel, "x", "E_y(x)")
    for profile, klass in zip(profiles, ("curve-a", "curve-b"), strict=True):
        _curve(
            drawing,
            panel,
            np.concatenate([[-half_gap - plate], x, [half_gap + plate]]),
            np.concatenate([[0.0], profile, [0.0]]),
            klass,
        )
    _key(drawing, (("curve-a", "TE1"), ("curve-b", "TE3")), LABEL_X + 30, 350)
    _label(
        drawing,
        (
            "self-focusing and the",
            "modulational instability",
            "in y are in the model",
        ),
        LABEL_X + 30,
        386,
    )
    return drawing


def radial_free() -> drawsvg.Drawing:
    """Draw radial free space: the QDHT nodes, the collar and a beam on them.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    grid = RadialGrid(1e-3, 24)
    waist = 0.3e-3
    radius = grid.radius
    inner = radius * (1 - grid.window_factor)

    drawing = _canvas(
        500,
        "Radial free space",
        (
            "RadialGrid(1 mm, 24): the order-zero QDHT samples at the zeros of J₀, and "
            "a Planck",
            "collar occupies the outer tenth. A Gaussian beam of 0.3 mm waist on the "
            "nodes.",
        ),
    )
    section = Section(CORE_CX, CORE_CY + 5, CORE_PX / radius)
    _outline(drawing, geometry.outline(grid), section)
    for r in grid.r:
        _circle(drawing, section.cx, section.cy, r * section.scale, "ring")
    for r in grid.r:
        _circle(drawing, section.px(r), section.py(0.0), DOT, "node")
    _circle(drawing, section.cx, section.cy, radius * section.scale, "boundary")
    _polarisation(drawing, (section.cx - 40, section.cy - 60), 0.0, 44, "E_x")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(0.7071 * radius), section.py(0.7071 * radius)),
        (LABEL_X, 92),
        (
            "E = 0 at R",
            "r_i = R α_i / α_(N+1),",  # noqa: RUF001 - the glyph is what the figure prints
            "α_i the zeros of J₀",  # noqa: RUF001 - the glyph is what the figure prints
        ),
    )
    _leader(
        drawing,
        (
            section.px(-0.7071 * 0.5 * (inner + radius)),
            section.py(-0.7071 * 0.5 * (inner + radius)),
        ),
        (LABEL_X, 152),
        (
            "absorbing collar, outer 10 %",
            "a rate in every stage, not a mask",
        ),
    )
    _label(
        drawing,
        (
            "each Cartesian component",
            "cylindrically symmetric;",
            "a gas may vary along z",
        ),
        LABEL_X,
        200,
    )

    panel = Panel(
        section.px(-radius),
        section.px(radius),
        372,
        470,
        (-radius, radius),
        (-0.05, 1.1),
    )
    _guides(
        drawing,
        [section.px(-inner), section.px(inner)],
        section.py(0) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "on the nodes", zero=False)
    r = np.concatenate([-grid.r[::-1], grid.r])
    window = np.concatenate([grid.window[::-1], grid.window])
    beam = np.exp(-((r / waist) ** 2))
    _curve(drawing, panel, r, window, "curve-c")
    _curve(drawing, panel, r, beam, "curve-a")
    for position, value in zip(r, beam, strict=True):
        _circle(drawing, panel.px(position), panel.py(value), DOT, "dot-curve-a")
    _key(
        drawing,
        (("curve-a", "beam, exp(−r²/w²)"), ("curve-c", "collar window")),  # noqa: RUF001 - the glyph is what the figure prints
        LABEL_X,
        392,
    )
    _label(
        drawing,
        ("the finite transform is not", "exactly orthogonal (bounded)"),
        LABEL_X,
        432,
    )
    return drawing


def cartesian_free() -> drawsvg.Drawing:
    """Draw x-y free space: the periodic box, its collar and its FFT samples.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    grid = FreeGrid(3e-4, 16, 2e-4, 12, window_factor=0.1)
    waist = 80e-6
    outer_x = grid.radius_x * (1 + grid.window_factor)
    outer_y = grid.radius_y * (1 + grid.window_factor)

    drawing = _canvas(
        500,
        "Cartesian free space",
        (
            "FreeGrid(0.3 mm, 16, 0.2 mm, 12): a periodic box whose extra tenth on "
            "each side is",
            "the absorbing collar, sampled by the FFT, with x fastest in the "
            "flattened columns.",
        ),
    )
    section = Section(CORE_CX, CORE_CY, 130.0 / outer_x)
    _outline(drawing, geometry.outline(grid), section)
    for level in (0.9, 0.5, 0.1):
        _circle(
            drawing,
            section.cx,
            section.cy,
            waist * math.sqrt(-0.5 * math.log(level)) * section.scale,
            "ring",
        )
    columns = grid.x.size
    for index, (window, (px, py)) in enumerate(
        zip(
            grid.window,
            (
                (grid.x[i % columns], grid.y[i // columns])
                for i in range(grid.window.size)
            ),
            strict=True,
        )
    ):
        del index
        _circle(
            drawing,
            section.px(px),
            section.py(py),
            DOT,
            "node" if window == 1.0 else "node-dim",
        )
    _rect(
        drawing,
        (section.px(-outer_x), section.py(outer_y)),
        (2 * outer_x * section.scale, 2 * outer_y * section.scale),
        "boundary",
    )
    # Periodic: what leaves through one face re-enters through the opposite one.
    y_arrow = section.py(0.55 * grid.radius_y)
    _arrow(
        drawing,
        (section.px(outer_x) - 12, y_arrow),
        (section.px(outer_x) + 16, y_arrow),
    )
    _arrow(
        drawing,
        (section.px(-outer_x) - 16, y_arrow),
        (section.px(-outer_x) + 12, y_arrow),
    )
    _polarisation(drawing, (section.cx, section.cy - 2), 0.0, 36, "E_x")
    _axes_key(drawing, MARGIN + 8, 132)

    _leader(
        drawing,
        (section.px(outer_x), section.py(-0.4 * grid.radius_y)),
        (LABEL_X + 20, 92),
        (
            "periodic + absorbing collar",
            "half-width radius × (1 + 0.1)",  # noqa: RUF001 - the glyph is what the figure prints
        ),
    )
    _leader(
        drawing,
        (section.px(0.5 * grid.radius_x), section.py(-1.05 * grid.radius_y)),
        (LABEL_X + 20, 138),
        (
            "collar: product of two Planck",
            "windows; dim nodes are in it",
        ),
    )
    _label(
        drawing,
        (
            "constant media only",
            "no symmetry assumed: any",
            "azimuthal structure, any",
            "polarisation pattern",
        ),
        LABEL_X + 20,
        186,
    )

    panel = Panel(
        section.px(-outer_x),
        section.px(outer_x),
        372,
        470,
        (-outer_x, outer_x),
        (-0.05, 1.1),
    )
    _guides(
        drawing,
        [section.px(-grid.radius_x), section.px(grid.radius_x)],
        section.py(-outer_y) + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "along the middle row", zero=False)
    row = grid.window[: grid.x.size]
    middle = np.abs(grid.y).argmin()
    row = grid.window[middle * columns : (middle + 1) * columns]
    beam = np.exp(-((grid.x / waist) ** 2))
    _curve(drawing, panel, grid.x, row, "curve-c")
    _curve(drawing, panel, grid.x, beam, "curve-a")
    for position, value in zip(grid.x, beam, strict=True):
        _circle(drawing, panel.px(position), panel.py(value), DOT, "dot-curve-a")
    _key(
        drawing,
        (("curve-a", "beam at the nodes"), ("curve-c", "collar window")),
        LABEL_X + 20,
        392,
    )
    _label(
        drawing,
        ("memory: n_x n_y columns per", "time sample; n_x n_y log cost"),
        LABEL_X + 20,
        432,
    )
    return drawing


def line_free() -> drawsvg.Drawing:
    """Draw the line free space: one periodic x axis, uniform and unbounded in y.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    grid = LineGrid(3e-4, 16, window_factor=0.1)
    waist = 80e-6
    outer = grid.radius_x * (1 + grid.window_factor)

    drawing = _canvas(
        460,
        "Line free space: x only",
        (
            "LineGrid(0.3 mm, 16): the Cartesian grid's x axis with one y sample of "
            "unit width,",
            "so the beam is uniform in y and every energy and power is per metre of y.",
        ),
    )
    section = Section(CORE_CX, CORE_CY, 130.0 / outer)
    top, bottom = section.cy - 80.0, section.cy + 80.0
    _outline(drawing, geometry.outline(grid, extent=80.0 / section.scale), section)
    # The free direction: nothing bounds the field in y, and nothing varies along it.
    for end, tip in ((top + 16, top + 2), (bottom - 16, bottom - 2)):
        _arrow(
            drawing,
            (section.px(0.5 * grid.radius_x), end),
            (section.px(0.5 * grid.radius_x), tip),
        )
    _text(
        drawing,
        "uniform in y (per unit width)",
        section.cx,
        top - 8,
        size=SMALL_SIZE,
        klass="dim mid",
    )
    for position, window in zip(grid.x, grid.window, strict=True):
        _circle(
            drawing,
            section.px(position),
            section.cy,
            DOT,
            "node" if window == 1.0 else "node-dim",
        )
    # Periodic in x: what leaves through one end re-enters through the other.
    y_arrow = section.cy + 0.55 * (bottom - section.cy)
    _arrow(
        drawing,
        (section.px(outer) - 12, y_arrow),
        (section.px(outer) + 16, y_arrow),
    )
    _arrow(
        drawing,
        (section.px(-outer) - 16, y_arrow),
        (section.px(-outer) + 12, y_arrow),
    )
    _polarisation(drawing, (section.cx, section.cy - 30), 0.0, 36, "E_x")
    _axes_key(drawing, MARGIN + 8, 116)

    _leader(
        drawing,
        (section.px(outer), section.cy - 40),
        (LABEL_X + 20, 92),
        (
            "periodic + absorbing collar",
            "in x only; no y collar",
        ),
    )
    _label(
        drawing,
        (
            "one y column: k_y = 0,",
            "unit measure; energy J/m,",
            "power W/m, intensity W/m²",
        ),
        LABEL_X + 20,
        140,
    )
    _label(
        drawing,
        (
            "constant media only",
            "no collapse: the Kerr",
            "beam is a sech soliton",
        ),
        LABEL_X + 20,
        196,
    )

    panel = Panel(
        section.px(-outer),
        section.px(outer),
        340,
        430,
        (-outer, outer),
        (-0.05, 1.1),
    )
    _guides(
        drawing,
        [section.px(-grid.radius_x), section.px(grid.radius_x)],
        bottom + 2,
        panel.bottom,
    )
    _frame(drawing, panel, "x", "the line's samples", zero=False)
    beam = np.exp(-((grid.x / waist) ** 2))
    _curve(drawing, panel, grid.x, grid.window, "curve-c")
    _curve(drawing, panel, grid.x, beam, "curve-a")
    for position, value in zip(grid.x, beam, strict=True):
        _circle(drawing, panel.px(position), panel.py(value), DOT, "dot-curve-a")
    _key(
        drawing,
        (("curve-a", "beam at the nodes"), ("curve-c", "collar window")),
        LABEL_X + 20,
        362,
    )
    _label(
        drawing,
        ("memory: n_x columns per", "time sample; n_x log cost"),
        LABEL_X + 20,
        402,
    )
    return drawing


def radial_azimuthal() -> drawsvg.Drawing:
    """Draw why an expansion in azimuthal orders has no common radial grid.

    Returns
    -------
    drawsvg.Drawing
        The figure.
    """
    count = 12
    grid = RadialGrid(1.0, count, window_factor=0.0)
    orders = (0, 1, 2)
    left, right = 78.0, 520.0

    drawing = _canvas(
        330,
        "Radial and azimuthal free space: considered, not built",
        (
            "The QDHT of order m samples at the zeros of J_m, so each azimuthal order "
            "has its own",
            "radial nodes (12 each, same R). The pointwise nonlinear step needs one "
            "common grid.",
        ),
    )
    for row, order in enumerate(orders):
        y = 100 + row * 52
        zeros = jn_zeros(order, count + 1)
        nodes = grid.r if order == 0 else zeros[:count] / zeros[count]
        _line(drawing, [(left, y), (right, y)], "axis")
        _text(
            drawing, f"m = {order}", left - 12, y + 4, size=SMALL_SIZE, klass="end dim"
        )
        klass = CURVES[row]
        for node in nodes:
            _circle(drawing, left + node * (right - left), y, 3.0, f"dot-{klass}")
        if order == 0:
            for node in nodes:
                _line(
                    drawing,
                    [
                        (left + node * (right - left), y + 5),
                        (left + node * (right - left), 100 + 2 * 52 + 6),
                    ],
                    "guide",
                )
    _text(drawing, "r = 0", left, 100 + 2 * 52 + 22, size=SMALL_SIZE, klass="dim mid")
    _text(drawing, "R", right, 100 + 2 * 52 + 22, size=SMALL_SIZE, klass="dim mid")
    _label(
        drawing,
        (
            "the full expansion: one transform per order, and an interpolation or a",
            "non-orthogonal transform between them at every stage",
        ),
        MARGIN,
        258,
    )
    _label(
        drawing,
        (
            "one order alone: exact for radially or azimuthally polarised beams;",
            "for a scalar vortex only as an envelope without THG (a real field's",
            "third harmonic carries 3m). Use x–y free space for azimuthal structure.",  # noqa: RUF001 - the glyph is what the figure prints
        ),
        MARGIN,
        292,
    )
    return drawing


# ------------------------------------------------------------------- the registry


class Figure(NamedTuple):
    """One figure: the function that draws it and the page section that shows it."""

    draw: Callable[[], drawsvg.Drawing]
    """The function that draws it."""

    section: str
    """The heading, as written on the page, of the section it belongs to."""


FIGURES = {
    "mode_averaged": Figure(mode_averaged, "Mode-averaged: one field and one area"),
    "capillary_radial": Figure(capillary_radial, "Capillary modes on a radial rule"),
    "capillary_full": Figure(
        capillary_full, "Capillary modes on the full cross-section"
    ),
    "capillary_wall": Figure(capillary_wall, "The capillary wall, two ways"),
    "antiresonant": Figure(antiresonant, "Antiresonant tube lattice"),
    "step_index": Figure(step_index, "Step-index fibres, exact"),
    "graded_index": Figure(graded_index, "Graded-index providers"),
    "general_modal": Figure(general_modal, "The general modal route: vector fields"),
    "moving_guide": Figure(moving_guide, "Guides that move along z"),
    "rectangular": Figure(rectangular, "Rectangular hollow guides"),
    "slab": Figure(slab, "Planar hollow guides, uniform in y"),
    "planar_hybrid": Figure(planar_hybrid, "Planar guides resolved in y"),
    "radial_free": Figure(radial_free, "Radial free space"),
    "cartesian_free": Figure(cartesian_free, "Cartesian free space"),
    "line_free": Figure(line_free, "Line free space: x only"),
    "radial_azimuthal": Figure(
        radial_azimuthal, "Radial and azimuthal free space: considered, not built"
    ),
}
"""File stem to the figure; the page shows exactly these, each in its own section."""

PENDING: dict[str, str] = {}
"""Sections of the page whose decomposition, and so whose figure, a later slice adds."""

UNFIGURED = ("The map", "Choosing a decomposition")
"""Sections of the page that are about every decomposition and draw none."""


def render(name: str) -> str:
    """Return one figure as SVG text.

    Parameters
    ----------
    name : str
        The figure's file stem, a key of :data:`FIGURES`.

    Returns
    -------
    str
        The complete SVG document, stylesheet included.

    Raises
    ------
    KeyError
        If no figure has that name.
    """
    return FIGURES[name].draw().as_svg()


def write(directory: Path | None = None) -> list[Path]:
    """Draw every figure and write it out, one SVG per figure.

    Parameters
    ----------
    directory : pathlib.Path, optional
        Where to write; :data:`HERE`, beside this script, by default.

    Returns
    -------
    list of pathlib.Path
        The files written, in the order of :data:`FIGURES`.
    """
    target = HERE if directory is None else directory
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name in FIGURES:
        path = target / f"{name}.svg"
        path.write_text(render(name), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    """Redraw every figure and say what was written.

    This is what ``poe figures`` runs, after the structure figures. The files it writes
    are committed; ``tests/test_geometry_figures.py`` fails if they are not what this
    draws.
    """
    for path in write():
        # A hand-run script says what it changed: the files are committed, so the
        # next thing the developer does is read `git status`.
        print(f"wrote {path.relative_to(Path.cwd())}")  # noqa: T201


if __name__ == "__main__":
    main()

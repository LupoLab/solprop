# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Draw the cross-section of a guide or a free-space grid.

The drawing is the **outline** of a cross-section: a hollow guide's wall and gas-filled
bore, an antiresonant fibre's tubes and jacket, a solid fibre's core and cladding, a
planar guide's plates, and a free-space grid's medium with its absorbing collar. Mode
fields, node sets and labels are not drawn: they belong to the figure that shows them.

The geometry is computed once, by :func:`outline`, as a tuple of plain shapes in
metres (:class:`Disc`, :class:`Ring`, :class:`Box`), each naming the region it fills.
:func:`draw` paints those shapes into a Matplotlib axes, and :func:`thumbnail` makes a
small figure of a route's representative guide for a front end's route picker. The user
guide's cross-section figures draw the same shapes as SVG, so the guide and the library
show one geometry. Colours come from an explicit :class:`Palette` of ``#rrggbb``
strings, which Matplotlib, SVG viewers and Qt all read the same way; :data:`LIGHT` and
:data:`DARK` are the two built in.

What a model does not describe is drawn to a stated convention, never inferred: a
hollow guide's wall, and an antiresonant fibre's jacket, are :data:`WALL_FRACTION` of
the guide's size thick, and a direction a geometry leaves unbounded (a fibre's cladding,
a planar guide's or a line grid's y) is drawn to an ``extent`` the caller may choose.
Everything else is to scale. The manual page is ``docs/guide/results/plotting.md``.
"""

import functools
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Final, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Annulus, Circle, Patch, Rectangle

from solprop.exceptions import ProcessingError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import MarcatiliMode
from solprop.modes.collection import ModeCollection
from solprop.modes.graded_index import GradedIndexProvider
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.rectangular import RectMode
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.modes.stepindex import StepIndexMode
from solprop.profiles import ZProfile, as_profile

__all__ = [
    "DARK",
    "DRAWN_CLADDING",
    "LIGHT",
    "THUMBNAILS",
    "WALL_FRACTION",
    "Box",
    "Disc",
    "Palette",
    "Region",
    "Ring",
    "Shape",
    "draw",
    "outline",
    "thumbnail",
]

Region = Literal["medium", "wall", "core", "cladding", "collar"]
"""What a shape is: the propagation medium (a hollow guide's gas, free space), a
guide's glass or metal wall (a capillary's, a tube's, a plate), a solid guide's core or
cladding, or a free-space grid's absorbing collar."""

WALL_FRACTION: Final = 0.18
"""Drawn thickness of a hollow guide's wall, as a fraction of its size: of a capillary's
or an antiresonant fibre's core radius (the jacket), of a rectangular guide's larger
half-width and of a planar guide's half gap. None of these models has a wall thickness
of its own: the wall enters only through the effective index."""

DRAWN_CLADDING: Final = 3.0
"""Default drawn radius of a solid fibre's cladding, in core radii: the cladding is
unbounded in the model."""

_TO_UM: Final = 1e6
"""Metres to micrometres, the unit of the axes :func:`draw` labels."""

_LINE_WIDTH: Final = 0.8
"""Edge width of every region, in points."""

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True, slots=True)
class Palette:
    """The colours of a cross-section drawing, one fill and one edge per region.

    Every colour is a ``#rrggbb`` string, which Matplotlib, SVG and Qt read alike.

    Attributes
    ----------
    background : str
        The ground behind the drawing; :func:`thumbnail` paints it, and :func:`draw`
        leaves the axes' own.
    medium, medium_edge : str
        The propagation medium: a hollow guide's gas, free space.
    wall, wall_edge : str
        A guide's wall: a capillary's, a tube's or the jacket of an antiresonant
        fibre, a plate.
    core, core_edge : str
        A solid guide's core.
    cladding, cladding_edge : str
        A solid guide's cladding.
    collar, collar_edge : str
        A free-space grid's absorbing collar.
    boundary : str
        The dashed outline of what a representation propagates, in :func:`thumbnail`.

    Raises
    ------
    ProcessingError
        If a colour is not a ``#rrggbb`` string.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from solprop.plotting.geometry import LIGHT
    >>> replace(LIGHT, wall="#cccccc").fill("wall")
    '#cccccc'
    """

    background: str
    medium: str
    medium_edge: str
    wall: str
    wall_edge: str
    core: str
    core_edge: str
    cladding: str
    cladding_edge: str
    collar: str
    collar_edge: str
    boundary: str

    def __post_init__(self) -> None:
        """Require every colour to be a ``#rrggbb`` string.

        Raises
        ------
        ProcessingError
            If one is not, naming it.
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, str) or _HEX.fullmatch(value) is None:
                raise ProcessingError(
                    f"Palette.{item.name} must be a '#rrggbb' colour, got {value!r}"
                )

    def fill(self, region: Region) -> str:
        """Return the fill colour of a region.

        Parameters
        ----------
        region : {'medium', 'wall', 'core', 'cladding', 'collar'}
            The region.

        Returns
        -------
        str
            A ``#rrggbb`` colour.
        """
        return getattr(self, region)

    def edge(self, region: Region) -> str:
        """Return the edge colour of a region.

        Parameters
        ----------
        region : {'medium', 'wall', 'core', 'cladding', 'collar'}
            The region.

        Returns
        -------
        str
            A ``#rrggbb`` colour.
        """
        return getattr(self, f"{region}_edge")


LIGHT: Final = Palette(
    background="#fbfbfd",
    medium="#e9f1fb",
    medium_edge="#2f5eb0",
    wall="#d5e9dc",
    wall_edge="#2c7a55",
    core="#cfe0f4",
    core_edge="#2f5eb0",
    cladding="#d5e9dc",
    cladding_edge="#2c7a55",
    collar="#f8e8d2",
    collar_edge="#a96a15",
    boundary="#b8334f",
)
"""The palette on a light ground, the default: the user guide's figures use it."""

DARK: Final = Palette(
    background="#171a1f",
    medium="#18222f",
    medium_edge="#7ba6e8",
    wall="#173528",
    wall_edge="#5cb98a",
    core="#1d3150",
    core_edge="#7ba6e8",
    cladding="#173528",
    cladding_edge="#5cb98a",
    collar="#3a2b15",
    collar_edge="#e0a44f",
    boundary="#ee8ca0",
)
"""The same palette on a dark ground."""


@dataclass(frozen=True, slots=True)
class Disc:
    """A filled circle.

    Attributes
    ----------
    centre : tuple of float
        ``(x, y)`` in m.
    radius : float
        In m.
    region : {'medium', 'wall', 'core', 'cladding', 'collar'}
        What it is.
    """

    centre: tuple[float, float]
    radius: float
    region: Region


@dataclass(frozen=True, slots=True)
class Ring:
    """A filled ring between two radii.

    Attributes
    ----------
    centre : tuple of float
        ``(x, y)`` in m.
    inner, outer : float
        The two radii, in m, ``0 < inner < outer``.
    region : {'medium', 'wall', 'core', 'cladding', 'collar'}
        What it is.
    """

    centre: tuple[float, float]
    inner: float
    outer: float
    region: Region


@dataclass(frozen=True, slots=True)
class Box:
    """A filled axis-aligned rectangle.

    Attributes
    ----------
    lower : tuple of float
        The corner of least x and least y, in m.
    upper : tuple of float
        The corner of greatest x and greatest y, in m.
    region : {'medium', 'wall', 'core', 'cladding', 'collar'}
        What it is.
    """

    lower: tuple[float, float]
    upper: tuple[float, float]
    region: Region


Shape = Disc | Ring | Box
"""One shape of an outline."""

_ORIGIN: Final = (0.0, 0.0)


def _at(value: float | ZProfile, z: float) -> float:
    """Return a constant or a z path at one position.

    Parameters
    ----------
    value : float or ZProfile
        A length, in m, constant or along z.
    z : float
        Position along the guide, in m.

    Returns
    -------
    float
        The length at ``z``, in m.
    """
    return float(as_profile(value)(np.array([z], dtype=float))[0])


def _capillary(radius: float) -> tuple[Shape, ...]:
    """Return a gas-filled bore in a wall of the conventional thickness.

    Parameters
    ----------
    radius : float
        Bore radius, in m.

    Returns
    -------
    tuple of Shape
        The wall, then the bore.
    """
    return (
        Ring(_ORIGIN, radius, radius * (1 + WALL_FRACTION), "wall"),
        Disc(_ORIGIN, radius, "medium"),
    )


def _antiresonant(radius: float, lattice: TubeLattice) -> tuple[Shape, ...]:
    """Return a single ring of tubes around a core, inside a jacket.

    The tubes are to scale, their walls included; they touch the jacket, whose own
    thickness the lattice does not model. The first tube sits at an angle of
    ``pi / N``, so that a gap lies on the x axis.

    Parameters
    ----------
    radius : float
        Core radius, the inscribed radius of the tubes, in m.
    lattice : solprop.modes.antiresonant.TubeLattice
        The tubes.

    Returns
    -------
    tuple of Shape
        The jacket, the gas inside it, then each tube.
    """
    tube, count = lattice.tube_radius, lattice.n_tubes
    jacket = radius + 2 * tube
    shapes: list[Shape] = [
        Ring(_ORIGIN, jacket, jacket + WALL_FRACTION * radius, "wall"),
        Disc(_ORIGIN, jacket, "medium"),
    ]
    for index in range(count):
        angle = math.pi / count + 2 * math.pi * index / count
        centre = ((radius + tube) * math.cos(angle), (radius + tube) * math.sin(angle))
        shapes.append(Ring(centre, tube - lattice.wall_thickness, tube, "wall"))
    return tuple(shapes)


def _fibre(radius: float, extent: float | None) -> tuple[Shape, ...]:
    """Return a solid core in its cladding, drawn to a finite radius.

    Parameters
    ----------
    radius : float
        Core radius, in m.
    extent : float or None
        Drawn cladding radius, in m; :data:`DRAWN_CLADDING` core radii by default.

    Returns
    -------
    tuple of Shape
        The cladding, then the core.
    """
    outer = DRAWN_CLADDING * radius if extent is None else extent
    if outer <= radius:
        raise ProcessingError(
            f"the drawn cladding radius ({outer:g} m) must exceed the core radius "
            f"({radius:g} m)"
        )
    return Disc(_ORIGIN, outer, "cladding"), Disc(_ORIGIN, radius, "core")


def _rectangle(half_x: float, half_y: float) -> tuple[Shape, ...]:
    """Return a rectangular bore in a wall of the conventional thickness.

    Parameters
    ----------
    half_x, half_y : float
        The bore's half-widths, in m.

    Returns
    -------
    tuple of Shape
        The wall, then the bore.
    """
    wall = WALL_FRACTION * max(half_x, half_y)
    return (
        Box((-half_x - wall, -half_y - wall), (half_x + wall, half_y + wall), "wall"),
        Box((-half_x, -half_y), (half_x, half_y), "medium"),
    )


def _plates(half_gap: float, height: float) -> tuple[Shape, ...]:
    """Return two plates at ``x = +-a`` and the gap between them, over ``|y| < h``.

    Parameters
    ----------
    half_gap : float
        The plates' half separation ``a``, in m.
    height : float
        The drawn half-height ``h`` in y, in m.

    Returns
    -------
    tuple of Shape
        The two plates, left then right.
    """
    plate = WALL_FRACTION * half_gap
    return (
        Box((-half_gap - plate, -height), (-half_gap, height), "wall"),
        Box((half_gap, -height), (half_gap + plate, height), "wall"),
    )


def _slab(half_gap: float, extent: float | None) -> tuple[Shape, ...]:
    """Return a planar guide: two plates and the gas between them.

    Parameters
    ----------
    half_gap : float
        The plates' half separation, in m.
    extent : float or None
        Drawn half-height in y, in m; the plates' outer half-width by default, so
        that the drawing is square.

    Returns
    -------
    tuple of Shape
        The plates, then the gap.
    """
    height = half_gap * (1 + WALL_FRACTION) if extent is None else extent
    return (
        *_plates(half_gap, height),
        Box((-half_gap, -height), (half_gap, height), "medium"),
    )


def _planar_grid(grid: PlanarGrid) -> tuple[Shape, ...]:
    """Return a planar guide resolved in y: plates, the periodic y cell and its collar.

    Parameters
    ----------
    grid : solprop.modes.slab.PlanarGrid
        The grid.

    Returns
    -------
    tuple of Shape
        The plates, the collar over the whole periodic cell, then the medium inside
        the collar.
    """
    half_gap = grid.half_gap
    inner = grid.radius_y
    outer = inner * (1 + grid.window_factor)
    shapes: list[Shape] = list(_plates(half_gap, outer))
    if grid.window_factor > 0:
        shapes.append(Box((-half_gap, -outer), (half_gap, outer), "collar"))
    shapes.append(Box((-half_gap, -inner), (half_gap, inner), "medium"))
    return tuple(shapes)


def _radial_grid(grid: RadialGrid) -> tuple[Shape, ...]:
    """Return radial free space: the medium to ``R``, its outer part the collar.

    Parameters
    ----------
    grid : solprop.grid.RadialGrid
        The grid.

    Returns
    -------
    tuple of Shape
        The medium, then the collar when there is one.
    """
    radius = grid.radius
    shapes: list[Shape] = [Disc(_ORIGIN, radius, "medium")]
    if grid.window_factor > 0:
        shapes.append(
            Ring(_ORIGIN, radius * (1 - grid.window_factor), radius, "collar")
        )
    return tuple(shapes)


def _free_grid(grid: FreeGrid, extent: float | None) -> tuple[Shape, ...]:
    """Return Cartesian free space: the periodic box and the collar around it.

    A line grid is uniform in y, so its y extent is drawn to ``extent``.

    Parameters
    ----------
    grid : solprop.grid.FreeGrid
        The grid; a :class:`~solprop.grid.LineGrid` is drawn as a band.
    extent : float or None
        A line grid's drawn half-height in y, in m; its periodic half-width in x by
        default.

    Returns
    -------
    tuple of Shape
        The collar when there is one, then the medium inside it.
    """
    factor = 1 + grid.window_factor
    outer_x = grid.radius_x * factor
    if isinstance(grid, LineGrid):
        inner_y = outer_y = outer_x if extent is None else extent
    else:
        inner_y, outer_y = grid.radius_y, grid.radius_y * factor
    shapes: list[Shape] = []
    if grid.window_factor > 0:
        shapes.append(Box((-outer_x, -outer_y), (outer_x, outer_y), "collar"))
    shapes.append(Box((-grid.radius_x, -inner_y), (grid.radius_x, inner_y), "medium"))
    return tuple(shapes)


def _unbounded(guide: object) -> bool:
    """Return whether a geometry leaves a direction unbounded, so takes an extent.

    Parameters
    ----------
    guide : object
        The guide or grid, after unwrapping a collection or a modal guide.

    Returns
    -------
    bool
        True for a solid fibre, a planar guide and a line grid.
    """
    return isinstance(guide, StepIndexMode | GradedIndexProvider | SlabMode | LineGrid)


def _unwrap(guide: object) -> object:
    """Return the object whose geometry a collection or a modal guide carries.

    Parameters
    ----------
    guide : object
        What was passed to :func:`outline`.

    Returns
    -------
    object
        A collection's first mode, which shares its guide with the rest; a modal
        guide's provider, or a step-index provider's first mode; else ``guide``.
    """
    if isinstance(guide, ModeCollection):
        return _unwrap(guide.modes[0])
    if isinstance(guide, ModalGuide):
        return _unwrap(guide.field_provider)
    if isinstance(guide, StepIndexProvider):
        return guide.modes[0]
    return guide


def outline(
    guide: object, *, z: float = 0.0, extent: float | None = None
) -> tuple[Shape, ...]:
    """Return the cross-section of a guide or a free-space grid, as shapes in metres.

    The shapes are in drawing order, back to front, centred on the guide's axis. A
    hollow guide's wall and an antiresonant fibre's jacket are :data:`WALL_FRACTION` of
    its size thick, since no model carries a wall thickness; everything else is to
    scale.

    Parameters
    ----------
    guide : object
        What to draw: a capillary (:class:`~solprop.modes.capillary.MarcatiliMode`,
        and the tabulated modes built on it, drawn as their field's bore), an
        antiresonant fibre (:class:`~solprop.modes.antiresonant.VincettiMode`), a
        step-index fibre (:class:`~solprop.modes.stepindex.StepIndexMode`), a
        graded-index provider
        (:class:`~solprop.modes.graded_index.GradedIndexProvider`), a rectangular
        guide (:class:`~solprop.modes.rectangular.RectMode`), a planar guide
        (:class:`~solprop.modes.slab.SlabMode`), a
        :class:`~solprop.modes.slab.PlanarGrid`, a
        :class:`~solprop.modes.collection.ModeCollection` or a
        :class:`~solprop.modes.guide.ModalGuide` of any of these, or a free-space grid
        (:class:`~solprop.grid.RadialGrid`, :class:`~solprop.grid.FreeGrid`,
        :class:`~solprop.grid.LineGrid`).
    z : float, optional
        Position along the guide, in m, where a radius or a width that varies along z
        is read. Default 0.
    extent : float or None, optional
        How far to draw what the geometry leaves unbounded, in m: the cladding radius
        of a solid fibre (:data:`DRAWN_CLADDING` core radii by default), the
        half-height in y of a planar guide or a line grid (square by default). Refused
        for a geometry bounded in every direction.

    Returns
    -------
    tuple of Shape
        The regions, back to front.

    Raises
    ------
    ProcessingError
        If the guide is of a kind this module does not draw, if ``extent`` is given
        for a bounded geometry or is not positive and finite, or if a fibre's drawn
        cladding would not enclose its core.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.plotting.geometry import outline
    >>> wall, bore = outline(MarcatiliMode(50e-6, cladding=1.45))
    >>> wall.region, bore.region, bore.radius
    ('wall', 'medium', 5e-05)
    """
    target = _unwrap(guide)
    if extent is not None:
        if not _unbounded(target):
            raise ProcessingError(
                f"extent applies to a geometry unbounded in some direction (a solid "
                f"fibre, a planar guide, a line grid), not to a "
                f"{type(guide).__name__}"
            )
        if not math.isfinite(extent) or extent <= 0:
            raise ProcessingError(f"extent must be positive and finite, got {extent}")
    shapes = _shapes(target, z, extent)
    if shapes is None:
        raise ProcessingError(
            f"cannot draw a {type(guide).__name__}: solprop.plotting.geometry "
            "draws capillary, antiresonant, step-index, graded-index, rectangular "
            "and planar guides, mode collections and modal guides of them, and "
            "the radial, Cartesian, line and planar grids"
        )
    return shapes


def _shapes(  # noqa: PLR0911 - one return per kind of geometry
    target: object, z: float, extent: float | None
) -> tuple[Shape, ...] | None:
    """Return the shapes of an unwrapped guide or grid, or None for an unknown kind.

    Parameters
    ----------
    target : object
        The guide or grid, after :func:`_unwrap`.
    z : float
        Position along the guide, in m.
    extent : float or None
        How far to draw an unbounded direction, in m, already validated.

    Returns
    -------
    tuple of Shape or None
        The regions, back to front; None when this module does not draw the kind.
    """
    match target:
        case VincettiMode():
            return _antiresonant(target.dimlimits(z)[2][0], target.lattice)
        case MarcatiliMode():
            return _capillary(target.dimlimits(z)[2][0])
        case StepIndexMode():
            return _fibre(_at(target.core_radius, z), extent)
        case GradedIndexProvider():
            return _fibre(_at(target.radius, z), extent)
        case RectMode():
            _, _, (half_x, half_y) = target.dimlimits(z)
            return _rectangle(half_x, half_y)
        case SlabMode():
            return _slab(target.dimlimits(z)[2][0], extent)
        case PlanarGrid():
            return _planar_grid(target)
        case RadialGrid():
            return _radial_grid(target)
        case FreeGrid():
            return _free_grid(target, extent)
        case _:
            return None


def _patch(shape: Shape, palette: Palette) -> Patch:
    """Return the Matplotlib patch of one shape, in micrometres.

    Parameters
    ----------
    shape : Shape
        The shape, in m.
    palette : Palette
        Its colours.

    Returns
    -------
    matplotlib.patches.Patch
        A circle, an annulus or a rectangle, filled and edged by its region.
    """
    style = {
        "facecolor": palette.fill(shape.region),
        "edgecolor": palette.edge(shape.region),
        "linewidth": _LINE_WIDTH,
    }
    if isinstance(shape, Disc):
        x, y = shape.centre
        return Circle((x * _TO_UM, y * _TO_UM), shape.radius * _TO_UM, **style)
    if isinstance(shape, Ring):
        x, y = shape.centre
        return Annulus(
            (x * _TO_UM, y * _TO_UM),
            shape.outer * _TO_UM,
            (shape.outer - shape.inner) * _TO_UM,
            **style,
        )
    (x0, y0), (x1, y1) = shape.lower, shape.upper
    return Rectangle(
        (x0 * _TO_UM, y0 * _TO_UM), (x1 - x0) * _TO_UM, (y1 - y0) * _TO_UM, **style
    )


def draw(
    ax: Axes,
    guide: object,
    *,
    palette: Palette = LIGHT,
    z: float = 0.0,
    extent: float | None = None,
) -> list[Patch]:
    """Draw the cross-section of a guide or a free-space grid into an axes.

    The shapes of :func:`outline`, filled and edged by region, in micrometres, on
    equal axes labelled ``x (um)`` and ``y (um)``. The axes' own background is left
    alone; :func:`thumbnail` paints the palette's.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    guide : object
        The guide or grid, any that :func:`outline` takes.
    palette : Palette, optional
        The colours; :data:`LIGHT` by default.
    z : float, optional
        Position along the guide, in m. Default 0.
    extent : float or None, optional
        How far to draw an unbounded direction, in m, as for :func:`outline`.

    Returns
    -------
    list of matplotlib.patches.Patch
        The patches added, back to front.

    Raises
    ------
    ProcessingError
        As :func:`outline` does.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.modes import MarcatiliMode, TubeLattice, VincettiMode
    >>> from solprop.plotting.geometry import draw
    >>> lattice = TubeLattice(wall_thickness=180e-9, tube_radius=9e-6, n_tubes=6)
    >>> fibre = VincettiMode(15e-6, cladding=1.45, lattice=lattice)
    >>> figure, ax = plt.subplots()
    >>> len(draw(ax, fibre))
    8
    >>> plt.close(figure)
    """
    patches = [
        ax.add_patch(_patch(shape, palette))
        for shape in outline(guide, z=z, extent=extent)
    ]
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    return patches


@functools.cache
def _example(route: str, representation: str) -> object:
    """Build the representative guide or grid a thumbnail draws.

    Parameters
    ----------
    route, representation : str
        A key of :data:`THUMBNAILS`.

    Returns
    -------
    object
        The guide or grid, built once per session.
    """
    builders: dict[tuple[str, str], Callable[[], object]] = {
        ("gnlse", "mode_averaged"): lambda: StepIndexMode(
            1.25e-6, core=1.45, cladding=1.0, profile_wavelength=835e-9
        ),
        ("capillary", "mode_averaged"): lambda: MarcatiliMode(50e-6, cladding=1.45),
        ("capillary", "multimode"): lambda: MarcatiliMode(50e-6, cladding=1.45),
        ("waveguide", "multimode"): lambda: RectMode(50e-6, 25e-6, cladding=1.45),
        ("radial", "radial"): lambda: RadialGrid(1e-3, 8),
        ("free", "free"): lambda: FreeGrid(3e-4, 4, 2e-4, 4),
        ("planar", "slab"): lambda: PlanarGrid(
            [SlabMode(63.5e-6, cladding=1.45)], 1e-4, 4
        ),
    }
    return builders[route, representation]()


THUMBNAILS: Final[Mapping[tuple[str, str], str]] = {
    ("gnlse", "mode_averaged"): (
        "a step-index strand, 1.25 um in radius, and the disc of its mode's "
        "effective area"
    ),
    ("capillary", "mode_averaged"): (
        "a capillary of 50 um bore radius, and the disc of its HE11 mode's "
        "effective area"
    ),
    ("capillary", "multimode"): "a capillary of 50 um bore radius",
    ("waveguide", "multimode"): "a rectangular hollow guide, 100 um by 50 um",
    ("radial", "radial"): "radial free space with its absorbing collar",
    ("free", "free"): "Cartesian free space: the periodic box and its collar",
    ("planar", "slab"): "a planar guide resolved in y, with its periodic y collar",
}
"""The ``(route, representation)`` pairs :func:`thumbnail` draws, each with what it
shows. A route is a ``prop_*`` entry point's name and a representation one of
:data:`~solprop.output.schema.SimulationKind`."""


def thumbnail(
    route: str,
    representation: str,
    size: int = 128,
    *,
    palette: Palette = LIGHT,
) -> Figure:
    """Return a small square figure of a route's representative cross-section.

    For a front end's route picker: the outline of a typical guide or grid on the
    route (:data:`THUMBNAILS` says which), with no axes, on the palette's background.
    A mode-averaged representation adds the dashed disc of the mode's effective area,
    the one area the route propagates.

    The figure is a bare :class:`matplotlib.figure.Figure`, made without
    :mod:`matplotlib.pyplot`, so it touches no global state and needs no closing:
    save it with ``figure.savefig(target, format="png")`` (``size`` pixels square at
    its own dpi) or ``format="svg"``, or show it in Qt through
    ``matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(figure)``.

    Parameters
    ----------
    route : str
        The route: ``gnlse``, ``capillary``, ``waveguide``, ``radial``, ``free`` or
        ``planar``.
    representation : str
        The transverse representation, as a run records it
        (:data:`~solprop.output.schema.SimulationKind`).
    size : int, optional
        Width and height in pixels, at least 16. Default 128.
    palette : Palette, optional
        The colours; :data:`LIGHT` by default.

    Returns
    -------
    matplotlib.figure.Figure
        A figure of ``size / 100`` inches square at 100 dpi, with one axes filling
        it.

    Raises
    ------
    ProcessingError
        If the pair is not in :data:`THUMBNAILS` or ``size`` is too small.

    Examples
    --------
    >>> from solprop.plotting.geometry import DARK, thumbnail
    >>> figure = thumbnail("capillary", "mode_averaged", 64, palette=DARK)
    >>> figure.canvas.get_width_height()
    (64, 64)
    """
    if (route, representation) not in THUMBNAILS:
        pairs = ", ".join(f"({a!r}, {b!r})" for a, b in THUMBNAILS)
        raise ProcessingError(
            f"no thumbnail for route {route!r} with representation "
            f"{representation!r}; the pairs drawn are {pairs}"
        )
    if isinstance(size, bool) or not isinstance(size, int) or size < 16:
        raise ProcessingError(
            f"size must be an integer of at least 16 pixels, got {size!r}"
        )
    guide = _example(route, representation)
    figure = Figure(figsize=(size / 100, size / 100), dpi=100)
    figure.set_facecolor(palette.background)
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor(palette.background)
    draw(ax, guide, palette=palette)
    if representation == "mode_averaged" and isinstance(
        guide, MarcatiliMode | StepIndexMode
    ):
        area = guide.aeff()
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                math.sqrt(area / math.pi) * _TO_UM,
                fill=False,
                edgecolor=palette.boundary,
                linewidth=_LINE_WIDTH,
                linestyle=(0, (4, 2)),
            )
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.04)
    ax.set_axis_off()
    return figure

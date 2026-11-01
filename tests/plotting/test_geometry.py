# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import math
from dataclasses import replace

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Annulus, Circle, Rectangle

from solprop.exceptions import ProcessingError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    MarcatiliMode,
    ModalGuide,
    ModeCollection,
    PlanarGrid,
    PowerLawPermittivity,
    RectMode,
    SlabMode,
    StepIndexMode,
    TubeLattice,
    VincettiMode,
)
from solprop.plotting import geometry
from solprop.plotting.geometry import (
    DARK,
    LIGHT,
    THUMBNAILS,
    WALL_FRACTION,
    Box,
    Disc,
    Ring,
    draw,
    outline,
    thumbnail,
)
from solprop.profiles import PiecewiseLinear

LATTICE = TubeLattice(wall_thickness=180e-9, tube_radius=9e-6, n_tubes=6)


def _capillary():
    return MarcatiliMode(50e-6, cladding=1.45)


def _strand():
    return StepIndexMode(250e-9, core=1.45, cladding=1.0, profile_wavelength=835e-9)


def _graded():
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * math.sqrt(0.99)),
        radius=25e-6,
        family="TE",
        count=2,
        elements=8,
        extent=2.0,
    )


def _slab():
    return SlabMode(63.5e-6, cladding=1.45)


# Each kind of geometry, with the regions its outline paints, back to front.
GEOMETRIES = {
    "capillary": (_capillary, ["wall", "medium"]),
    "antiresonant": (
        lambda: VincettiMode(15e-6, cladding=1.45, lattice=LATTICE),
        ["wall", "medium", *["wall"] * 6],
    ),
    "step_index": (_strand, ["cladding", "core"]),
    "graded_index": (_graded, ["cladding", "core"]),
    "modal_guide": (
        lambda: ModalGuide.from_step_index(
            [_strand()], core=IndexMaterial(1.45), cladding=IndexMaterial(1.0)
        ),
        ["cladding", "core"],
    ),
    "collection": (lambda: ModeCollection([_capillary()]), ["wall", "medium"]),
    "rectangular": (
        lambda: RectMode(50e-6, 25e-6, cladding=1.45),
        ["wall", "medium"],
    ),
    "slab": (_slab, ["wall", "wall", "medium"]),
    "planar_grid": (
        lambda: PlanarGrid([_slab()], 1e-4, 4),
        ["wall", "wall", "collar", "medium"],
    ),
    "radial": (lambda: RadialGrid(1e-3, 8), ["medium", "collar"]),
    "radial_no_collar": (lambda: RadialGrid(1e-3, 8, window_factor=0.0), ["medium"]),
    "free": (lambda: FreeGrid(3e-4, 4, 2e-4, 4), ["collar", "medium"]),
    "free_no_collar": (
        lambda: FreeGrid(3e-4, 4, 2e-4, 4, window_factor=0.0),
        ["medium"],
    ),
    "line": (lambda: LineGrid(3e-4, 4), ["collar", "medium"]),
}


def _as[T](kind: type[T], shape: object) -> T:
    """Return a shape as the kind it must be."""
    assert isinstance(shape, kind)
    return shape


def _extent(shape):
    """Return a shape's (x0, y0, x1, y1) in m."""
    if isinstance(shape, Box):
        return (*shape.lower, *shape.upper)
    radius = shape.radius if isinstance(shape, Disc) else shape.outer
    x, y = shape.centre
    return x - radius, y - radius, x + radius, y + radius


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_draw_paints_every_region_of_each_geometry(name):
    build, regions = GEOMETRIES[name]
    guide = build()
    shapes = outline(guide)
    assert [shape.region for shape in shapes] == regions
    figure, ax = plt.subplots()
    patches = draw(ax, guide)
    assert len(patches) == len(shapes)
    assert list(ax.patches) == patches
    assert ax.get_aspect() == 1.0
    # The view encloses every shape, drawn in micrometres; a part in 1e9 of the
    # drawing's size absorbs the conversion's rounding.
    low_x, low_y, high_x, high_y = (
        1e6 * min(_extent(shape)[0] for shape in shapes),
        1e6 * min(_extent(shape)[1] for shape in shapes),
        1e6 * max(_extent(shape)[2] for shape in shapes),
        1e6 * max(_extent(shape)[3] for shape in shapes),
    )
    slack = 1e-9 * max(high_x - low_x, high_y - low_y)
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    assert left <= low_x + slack
    assert right >= high_x - slack
    assert bottom <= low_y + slack
    assert top >= high_y - slack
    assert ax.get_xlabel() == "x (um)"
    plt.close(figure)


def test_capillary_outline_is_the_bore_in_a_conventional_wall():
    wall, bore = outline(_capillary())
    assert isinstance(wall, Ring)
    assert isinstance(bore, Disc)
    assert bore.radius == pytest.approx(50e-6, rel=1e-15)
    assert wall.inner == pytest.approx(50e-6, rel=1e-15)
    assert wall.outer == pytest.approx(50e-6 * (1 + WALL_FRACTION), rel=1e-15)


def test_a_tapered_radius_is_read_at_z():
    taper = PiecewiseLinear.from_stages([0.0, 1.0], [50e-6, 40e-6])
    mode = MarcatiliMode(taper, cladding=1.45)
    # Halfway along a linear taper; the profile's own interpolation sets the bound.
    bore = _as(Disc, outline(mode, z=0.5)[1])
    assert bore.radius == pytest.approx(45e-6, rel=1e-12)


def test_antiresonant_tubes_are_to_scale_and_inscribe_the_core():
    core = 15e-6
    shapes = outline(VincettiMode(core, cladding=1.45, lattice=LATTICE))
    jacket, inside = _as(Ring, shapes[0]), _as(Disc, shapes[1])
    tubes = [_as(Ring, shape) for shape in shapes[2:]]
    assert len(tubes) == LATTICE.n_tubes
    assert inside.radius == pytest.approx(core + 2 * LATTICE.tube_radius, rel=1e-15)
    assert jacket.inner == pytest.approx(inside.radius, rel=1e-15)
    for tube in tubes:
        assert tube.outer == LATTICE.tube_radius
        assert tube.outer - tube.inner == pytest.approx(
            LATTICE.wall_thickness, rel=1e-9
        )
        # Each tube touches the inscribed core circle (rounding of the trigonometry).
        assert math.hypot(*tube.centre) - tube.outer == pytest.approx(core, rel=1e-12)
    # A gap, not a tube, lies on the +x axis.
    assert min(abs(math.atan2(t.centre[1], t.centre[0])) for t in tubes) > 0.1


def test_free_space_collars_follow_the_window_factor():
    radial = RadialGrid(1e-3, 8, window_factor=0.2)
    shapes = outline(radial)
    assert len(shapes) == 2
    medium, collar = _as(Disc, shapes[0]), _as(Ring, shapes[1])
    assert medium.radius == 1e-3
    assert collar.inner == pytest.approx(0.8e-3, rel=1e-15)
    assert collar.outer == 1e-3
    free = FreeGrid(3e-4, 4, 2e-4, 4, window_factor=0.1)
    outer, inner = (_as(Box, shape) for shape in outline(free))
    assert outer.upper == pytest.approx((3.3e-4, 2.2e-4), rel=1e-15)
    assert inner.upper == pytest.approx((3e-4, 2e-4), rel=1e-15)


@pytest.mark.parametrize(
    ("build", "extent"),
    [(_strand, 1e-6), (_slab, 2e-5), (lambda: LineGrid(3e-4, 4), 5e-5)],
)
def test_extent_sets_how_far_an_unbounded_direction_is_drawn(build, extent):
    shapes = outline(build(), extent=extent)
    assert max(_extent(shape)[3] for shape in shapes) == pytest.approx(
        extent, rel=1e-15
    )


def test_a_fibre_cladding_is_drawn_to_three_core_radii_by_default():
    cladding, core = (_as(Disc, shape) for shape in outline(_strand()))
    assert cladding.radius == pytest.approx(3 * core.radius, rel=1e-15)


@pytest.mark.parametrize("extent", [0.0, -1e-6, math.inf, math.nan])
def test_extent_must_be_positive_and_finite(extent):
    with pytest.raises(ProcessingError, match="positive and finite"):
        outline(_strand(), extent=extent)


def test_extent_is_refused_for_a_bounded_geometry():
    with pytest.raises(ProcessingError, match="not to a MarcatiliMode"):
        outline(_capillary(), extent=1e-4)


def test_a_cladding_inside_its_core_is_refused():
    with pytest.raises(ProcessingError, match="must exceed the core radius"):
        outline(_strand(), extent=100e-9)


@pytest.mark.parametrize("guide", [object(), "capillary", 50e-6])
def test_an_unsupported_guide_is_refused_by_name(guide):
    figure, ax = plt.subplots()
    with pytest.raises(ProcessingError, match=f"cannot draw a {type(guide).__name__}"):
        draw(ax, guide)
    assert len(ax.patches) == 0
    plt.close(figure)


@pytest.mark.parametrize("palette", [LIGHT, DARK, replace(LIGHT, wall="#123456")])
def test_draw_paints_with_the_palette_given(palette):
    figure, ax = plt.subplots()
    guide = PlanarGrid([_slab()], 1e-4, 4)
    for shape, patch in zip(
        outline(guide), draw(ax, guide, palette=palette), strict=True
    ):
        assert patch.get_facecolor() == to_rgba(palette.fill(shape.region))
        assert patch.get_edgecolor() == to_rgba(palette.edge(shape.region))
    plt.close(figure)


def test_each_shape_becomes_its_own_kind_of_patch():
    figure, ax = plt.subplots()
    patches = draw(ax, VincettiMode(15e-6, cladding=1.45, lattice=LATTICE))
    assert isinstance(patches[0], Annulus)
    assert isinstance(patches[1], Circle)
    patches = draw(ax, _slab())
    assert all(isinstance(patch, Rectangle) for patch in patches)
    plt.close(figure)


@pytest.mark.parametrize("colour", ["red", "#12345", "#1234567", "#gggggg"])
def test_a_palette_takes_only_hex_colours(colour):
    with pytest.raises(ProcessingError, match=r"Palette\.core"):
        replace(LIGHT, core=colour)


def test_the_two_palettes_name_every_region():
    for palette in (LIGHT, DARK):
        for region in ("medium", "wall", "core", "cladding", "collar"):
            assert palette.fill(region).startswith("#")
            assert palette.edge(region).startswith("#")
    assert LIGHT != DARK


@pytest.mark.parametrize(("route", "representation"), list(THUMBNAILS))
def test_a_thumbnail_is_a_bare_square_figure_of_the_route(route, representation):
    figure = thumbnail(route, representation, 48, palette=DARK)
    assert isinstance(figure, Figure)
    assert figure.canvas.get_width_height() == (48, 48)
    assert figure.get_facecolor() == to_rgba(DARK.background)
    (ax,) = figure.axes
    assert not ax.axison
    shapes = outline(geometry._example(route, representation))
    # A mode-averaged thumbnail adds the dashed disc of the mode's effective area.
    extra = 1 if representation == "mode_averaged" else 0
    assert len(ax.patches) == len(shapes) + extra
    if extra:
        assert ax.patches[-1].get_edgecolor() == to_rgba(DARK.boundary)
        assert not ax.patches[-1].get_fill()
    for kind in ("png", "svg"):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=kind)
        assert buffer.tell() > 0


def test_a_thumbnail_png_is_size_pixels_square():
    figure = thumbnail("capillary", "multimode", 40)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    # The PNG header stores the width and height as big-endian integers.
    header = buffer.getvalue()[16:24]
    assert int.from_bytes(header[:4]) == 40
    assert int.from_bytes(header[4:]) == 40


@pytest.mark.parametrize(
    ("route", "representation"),
    [("capillary", "free"), ("tdse", "radial"), ("free", "mode_averaged")],
)
def test_a_thumbnail_of_an_unknown_pair_is_refused(route, representation):
    with pytest.raises(ProcessingError, match="the pairs drawn are"):
        thumbnail(route, representation)


@pytest.mark.parametrize("size", [8, 15, True])
def test_a_thumbnail_needs_at_least_sixteen_whole_pixels(size):
    with pytest.raises(ProcessingError, match="at least 16 pixels"):
        thumbnail("radial", "radial", size)

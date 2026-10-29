# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Draw the structure figures of ``docs/dev/structure.md``.

This script is **not** run by the docs build. It is run by hand, with ``poe figures``,
whenever a figure changes; the SVGs it writes next to it are committed, and
``structure.md`` only includes them, so the manual builds on an install that has no
drawing library. ``tests/test_structure_figures.py`` redraws every figure and compares
it with what is committed, so a stale SVG is a test failure rather than something a
reader notices.

The names the figures draw are kept as data -- ``MODULES``, ``PLANNED``,
``CONTRACT_SECTIONS``, ``SEAM_METHODS``, ``STAGE_PASSES``, ``DRIVER_PARTS`` and
``OUTPUT_GROUPS`` -- and the same test checks each of them against the package, the
kernel contract and the backend seam. A box that names something the code no longer has
fails the fast lane; nothing here is decorative.

One style throughout: a sans-serif system stack with no embedded font, one colour per
layer (P, T, B, D, X of ``architecture.md`` section 3 and ``decisions.md`` ADR-001) used
in every figure and stated in the legend of the first, one corner radius, one stroke
width, and a fixed ``viewBox`` 1200 wide so the page renders every figure at one scale.
Colours are CSS custom properties with a ``prefers-color-scheme`` override, so a figure
carries its own light and dark palettes rather than relying on a page whose stylesheet
it never sees: an SVG included as an image is an isolated document.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from solprop.translate.problem import SCHEMA_VERSION as _KERNEL_SCHEMA_VERSION

try:
    import drawsvg
except ImportError:  # pragma: no cover - only on an install without the dev group
    drawsvg = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- style

WIDTH = 1200
"""Width of every figure's ``viewBox``, so the page renders them all at one scale."""

MARGIN = 32
"""Space between the figure's edge and anything drawn in it."""

RADIUS = 7
"""Corner radius of every box."""

STROKE = 1.4
"""Stroke width of every box and connector."""

FONT_STACK = 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif'
"""The one font stack; no font file is embedded, so a figure carries no binary."""

GLYPH_ADVANCE = 0.55
"""Mean glyph advance as a fraction of the font size, used to size boxes to their text.

Measured across the stack's fallbacks on mixed-case identifiers; it only has to be an
over-estimate, because a box that is a little wide reads and a box that is too narrow
clips.
"""

TITLE_SIZE = 15
"""Font size of a box's own name."""

BODY_SIZE = 13
"""Font size of the lines under a box's name and of an arrow's label."""

SMALL_SIZE = 12
"""Font size of the smallest annotations: module lists, units, side notes."""

__all__ = [
    "CONTRACT_SECTIONS",
    "DRIVER_PARTS",
    "FIGURES",
    "LAYERS",
    "MODULES",
    "OUTPUT_GROUPS",
    "OUTPUT_READERS",
    "OUTPUT_STATS",
    "PACKAGE_WIDE",
    "PLANNED",
    "RESPONSE_COLUMN",
    "SEAM_METHODS",
    "STAGE_PASSES",
    "STEP_PASSES",
    "backends",
    "kernel_contract",
    "layers",
    "main",
    "one_run",
    "one_stage",
    "output_side",
    "render",
    "write",
]

LIGHT = {
    "ground": "#fbfbfd",
    "ink": "#1b1f24",
    "dim": "#5b6673",
    "rule": "#c9d0da",
    "p-fill": "#e2f0e8",
    "p-edge": "#2c7a55",
    "t-fill": "#fceedd",
    "t-edge": "#a96a15",
    "b-fill": "#e3ebfa",
    "b-edge": "#2f5eb0",
    "d-fill": "#eee6f8",
    "d-edge": "#6f45a6",
    "x-fill": "#fce7ec",
    "x-edge": "#af3450",
}
"""The palette on a light ground: one fill and one edge per layer, plus the neutrals."""

DARK = {
    "ground": "#171a1f",
    "ink": "#e7eaef",
    "dim": "#9aa5b3",
    "rule": "#3a424e",
    "p-fill": "#16342a",
    "p-edge": "#5cb98a",
    "t-fill": "#3a2b15",
    "t-edge": "#e0a44f",
    "b-fill": "#17253f",
    "b-edge": "#7ba6e8",
    "d-fill": "#2a1f3e",
    "d-edge": "#b18fe0",
    "x-fill": "#3a1e26",
    "x-edge": "#e8879a",
}
"""The same palette on a dark ground, with exactly the same keys as :data:`LIGHT`."""

LAYER_LETTERS = ("P", "T", "B", "D", "X")
"""The layer letters of ``architecture.md`` section 3, in the order imports flow."""


def _variables(palette: dict[str, str]) -> str:
    """Return one palette as a block of CSS custom property declarations.

    Parameters
    ----------
    palette : dict of str
        A mapping of property name to colour, either :data:`LIGHT` or :data:`DARK`.

    Returns
    -------
    str
        The declarations, one per line, indented for the stylesheet.
    """
    return "\n".join(f"    --{name}: {value};" for name, value in palette.items())


def _stylesheet() -> str:
    """Return the stylesheet every figure carries.

    An SVG included with ``![](...)`` is an isolated document that receives none of the
    page's CSS, so it cannot follow the theme switcher; what it can follow is the
    reader's own colour-scheme preference, which is what the theme's default setting
    follows too. Both palettes are written out, and every element is coloured through a
    custom property rather than a literal, so the two can never drift apart.

    Returns
    -------
    str
        The contents of the figure's ``<style>`` element.
    """
    layers = "\n".join(
        f".fill-{letter.lower()} {{ fill: var(--{letter.lower()}-fill); "
        f"stroke: var(--{letter.lower()}-edge); }}\n"
        f".edge-{letter.lower()} {{ stroke: var(--{letter.lower()}-edge); }}\n"
        f".ink-{letter.lower()} {{ fill: var(--{letter.lower()}-edge); }}"
        for letter in LAYER_LETTERS
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
.plain {{ fill: none; stroke: var(--rule); stroke-width: {STROKE}; }}
.planned {{ fill: none; stroke: var(--dim); stroke-width: {STROKE};
  stroke-dasharray: 5 4; }}
.arrow {{ fill: none; stroke: var(--dim); stroke-width: {STROKE};
  stroke-linecap: round; stroke-linejoin: round; }}
.head {{ fill: var(--dim); stroke: none; }}
.rule {{ fill: none; stroke: var(--rule); stroke-width: {STROKE}; }}
text {{ font-family: {FONT_STACK}; fill: var(--ink);
  white-space: pre; }}
.dim {{ fill: var(--dim); }}
.mid {{ text-anchor: middle; }}
.end {{ text-anchor: end; }}
.bold {{ font-weight: 600; }}
{layers}
"""


def _require_drawsvg() -> None:
    """Raise a clear error when the drawing library is not installed.

    Raises
    ------
    RuntimeError
        When ``drawsvg`` cannot be imported. The figure data in this module is readable
        without it; only drawing needs it.
    """
    if drawsvg is None:  # pragma: no cover - only on an install without the dev group
        raise RuntimeError(
            "drawsvg is not installed, so the structure figures cannot be drawn. "
            "It is in the project's dev group: run `uv sync` and try again. The "
            "committed SVGs are what the documentation shows; this script only "
            "rewrites them."
        )


def text_width(text: str, size: float) -> float:
    """Estimate how wide a string renders at a font size.

    Nothing measures glyphs here -- there is no font file to measure -- so boxes are
    sized from :data:`GLYPH_ADVANCE`, which over-estimates on purpose.

    Parameters
    ----------
    text : str
        The string to be drawn.
    size : float
        Its font size, in the figure's user units.

    Returns
    -------
    float
        The estimated width, in the figure's user units.
    """
    return len(text) * size * GLYPH_ADVANCE


def _canvas(height: float) -> drawsvg.Drawing:
    """Return a drawing of the standard width with an opaque ground.

    Parameters
    ----------
    height : float
        The figure's height in user units; the width is always :data:`WIDTH`.

    Returns
    -------
    drawsvg.Drawing
        A drawing carrying the stylesheet and a full-bleed background rectangle, ready
        for boxes.
    """
    _require_drawsvg()
    drawing = drawsvg.Drawing(WIDTH, height, origin=(0, 0))
    drawing.append(drawsvg.Raw(f"<style>{_stylesheet()}</style>"))
    drawing.append(drawsvg.Rectangle(0, 0, WIDTH, height, class_="ground"))
    return drawing


def _text(
    drawing: drawsvg.Drawing,
    content: str,
    x: float,
    y: float,
    *,
    size: float = BODY_SIZE,
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
        The font size; :data:`BODY_SIZE` by default.
    klass : str, optional
        Extra CSS classes, space separated: ``mid``, ``end``, ``bold``, ``dim`` or a
        layer ink such as ``ink-p``.
    """
    drawing.append(drawsvg.Text(content, size, x, y, class_=klass.strip()))


def _box(
    drawing: drawsvg.Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    klass: str = "card",
) -> None:
    """Draw one rounded box.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The top-left corner, in user units.
    width, height : float
        Its size, in user units.
    klass : str, optional
        The CSS class that colours it: ``card``, ``plain``, ``planned`` or a layer fill
        such as ``fill-p``.
    """
    drawing.append(
        drawsvg.Rectangle(x, y, width, height, rx=RADIUS, ry=RADIUS, class_=klass)
    )


def _arrow(
    drawing: drawsvg.Drawing,
    points: Sequence[tuple[float, float]],
    *,
    label: str = "",
    label_at: tuple[float, float] | None = None,
    label_class: str = "dim",
) -> None:
    """Draw a polyline with a solid head at its last point, and label what crosses it.

    The head is an explicit triangle rather than an SVG marker: a marker lives in
    ``<defs>`` under a generated identifier, and the committed SVGs are compared byte
    for byte with what this script draws.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    points : sequence of tuple of float
        Two or more ``(x, y)`` vertices; the arrow runs from the first to the last.
    label : str, optional
        What crosses the arrow -- an object, an array, a call. Empty draws no label.
    label_at : tuple of float, optional
        Where to put the label's anchor; the midpoint of the last segment, raised
        clear of the line, by default.
    label_class : str, optional
        CSS classes for the label.
    """
    path = drawsvg.Path(class_="arrow")
    path.M(*points[0])
    for point in points[1:]:
        path.L(*point)
    drawing.append(path)
    drawing.append(
        drawsvg.Lines(*_head(points[-2], points[-1]), close=True, class_="head")
    )
    if label:
        if label_at is None:
            (x0, y0), (x1, y1) = points[-2], points[-1]
            label_at = ((x0 + x1) / 2, (y0 + y1) / 2 - 6)
        _text(drawing, label, *label_at, size=SMALL_SIZE, klass=f"mid {label_class}")


HEAD_LENGTH = 9.0
"""Length of an arrow head, in user units."""

HEAD_HALF_WIDTH = 3.6
"""Half the width of an arrow head at its base, in user units."""


def _head(start: tuple[float, float], tip: tuple[float, float]) -> tuple[float, ...]:
    """Return the three corners of an arrow head pointing from ``start`` to ``tip``.

    Parameters
    ----------
    start : tuple of float
        The previous vertex of the line, which sets the direction.
    tip : tuple of float
        Where the head points.

    Returns
    -------
    tuple of float
        Six coordinates, ``x0, y0, x1, y1, x2, y2``, ready for ``drawsvg.Lines``.
    """
    dx, dy = tip[0] - start[0], tip[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    base = (tip[0] - ux * HEAD_LENGTH, tip[1] - uy * HEAD_LENGTH)
    return (
        tip[0],
        tip[1],
        base[0] - uy * HEAD_HALF_WIDTH,
        base[1] + ux * HEAD_HALF_WIDTH,
        base[0] + uy * HEAD_HALF_WIDTH,
        base[1] - ux * HEAD_HALF_WIDTH,
    )


class Slot(NamedTuple):
    """Where a box goes and how wide it is; its height comes from its content."""

    x: float
    """Its left edge, in user units."""

    y: float
    """Its top edge, in user units."""

    width: float
    """How wide to draw it, in user units."""


class Paint(NamedTuple):
    """The pair of CSS classes a box is drawn with."""

    fill: str = "card"
    """The class that colours the box itself."""

    ink: str = ""
    """The class that colours its title, usually a layer ink such as ``ink-p``."""


NO_PAINT = Paint()
"""The default pair of classes: a neutral card with no layer ink."""

PAD = 10.0
"""Padding between a box's edge and its text, in user units; the same everywhere."""

LINE = 15.5
"""Baseline-to-baseline distance of the lines under a box's title, in user units."""


def node_height(lines: Sequence[str]) -> float:
    """Return the height of a titled box with this many lines under its title.

    Parameters
    ----------
    lines : sequence of str
        The lines drawn under the title.

    Returns
    -------
    float
        The box height, in user units.
    """
    return PAD + TITLE_SIZE + len(lines) * LINE + PAD


def node_width(title: str, lines: Sequence[str]) -> float:
    """Return the width a titled box needs for its longest line.

    Parameters
    ----------
    title : str
        The box's name, drawn at :data:`TITLE_SIZE`.
    lines : sequence of str
        The lines under it, drawn at :data:`SMALL_SIZE`.

    Returns
    -------
    float
        The box width, in user units.
    """
    widths = [text_width(title, TITLE_SIZE)]
    widths += [text_width(line, SMALL_SIZE) for line in lines]
    return max(widths) + 2 * PAD


def _node(
    drawing: drawsvg.Drawing,
    slot: Slot,
    title: str,
    lines: Sequence[str] = (),
    paint: Paint = NO_PAINT,
) -> float:
    """Draw a box with a name and, under it, the things it holds.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    slot : Slot
        Where it goes and how wide it is; :func:`node_width` gives the natural width.
    title : str
        The box's name -- a module, a class, a function, a dataset.
    lines : sequence of str, optional
        What it holds, one per line, drawn smaller and dimmer.
    paint : Paint, optional
        The classes that colour the box and its title.

    Returns
    -------
    float
        The box's height, so a caller can stack the next one under it.
    """
    x, y = slot.x, slot.y
    height = node_height(lines)
    _box(drawing, x, y, slot.width, height, klass=paint.fill)
    _text(
        drawing,
        title,
        x + PAD,
        y + PAD + TITLE_SIZE - 3,
        size=TITLE_SIZE,
        klass=f"bold {paint.ink}",
    )
    for index, line in enumerate(lines):
        _text(
            drawing,
            line,
            x + PAD,
            y + PAD + TITLE_SIZE + (index + 1) * LINE - 3,
            size=SMALL_SIZE,
            klass="dim",
        )
    return height


# ------------------------------------------------------------------- what is drawn
#
# Everything below is data, and `tests/test_structure_figures.py` checks all of it
# against the package, the kernel contract and the backend seam. Add a box by adding a
# name here; if the name is wrong the fast lane says so.


class Module(NamedTuple):
    """One box of the layers figure: a module, or a package and its members."""

    path: str
    """Where it lives under ``src/solprop``; a package path ends with ``/``."""

    members: tuple[str, ...] = ()
    """Module names inside a package, without ``.py``; empty for a single module."""

    def files(self) -> tuple[str, ...]:
        """Return the paths under ``src/solprop`` this box claims exist.

        Returns
        -------
        tuple of str
            One path per member of a package, or the single module's own path. A
            package named without members claims only its directory.
        """
        if not self.path.endswith("/"):
            return (self.path,)
        if not self.members:
            return (self.path,)
        return tuple(f"{self.path}{member}.py" for member in self.members)


class Layer(NamedTuple):
    """One band of the layers figure: a layer of `architecture.md` section 3."""

    letter: str
    """Its one-letter code, ``P``, ``T``, ``B``, ``D`` or ``X``."""

    name: str
    """Its name, as the section 3 table writes it."""

    imports: str
    """What it may import, as the section 3 table writes it."""


LAYERS = (
    Layer("P", "physics / convenience", "numpy, scipy, stdlib, other P modules"),
    Layer("T", "translation", "P, numpy, scipy"),
    Layer("B", "backends", "numpy, numba, pyfftw; the native package in native/"),
    Layer("D", "driver", "P, T, B through the seam, h5py"),
    Layer(
        "X",
        "processing / plotting",
        "numpy, scipy, matplotlib, D (outputs), T (table identities)",
    ),
)
"""The five layers of `architecture.md` section 3, in the order imports flow."""

PACKAGE_WIDE = (
    Module("__init__.py"),
    Module("exceptions.py"),
)
"""The two modules `architecture.md` section 4 gives no layer letter."""

MODULES = {
    "P": (
        Module("constants.py"),
        Module("fourier.py"),
        Module("grid.py"),
        Module("hankel.py"),
        Module("spatial.py"),
        Module("profiles.py"),
        Module("adiabatic.py"),
        Module("gasflow.py"),
        Module("pulses.py"),
        Module("noise.py"),
        Module("linear.py"),
        Module("boundaries.py"),
        Module("monitor.py"),
        Module("polarisation.py"),
        Module("optics.py"),
        Module("tools.py"),
        Module(
            "coupling/",
            (
                "plane",
                "port",
                "remap",
                "models",
                "coupler",
                "outcouple",
                "envelopes",
                "transport",
                "elements",
            ),
        ),
        Module(
            "physdata/",
            (
                "sellmeier",
                "equation_of_state",
                "gases",
                "glasses",
                "metals",
                "nonlinear_data",
                "raman_types",
                "species",
                "mixtures",
                "transport",
                "selection",
            ),
        ),
        Module(
            "modes/",
            (
                "protocol",
                "base",
                "capillary",
                "antiresonant",
                "tabulated",
                "fill",
                "stepindex",
                "material",
                "vector",
                "guide",
                "graded_index",
                "connection",
                "rectangular",
                "slab",
                "collection",
                "quadrature",
            ),
        ),
        Module(
            "nonlinear/",
            (
                "responses",
                "ionisation",
                "optmat_rates",
                "envelope_plasma",
                "raman",
                "gnlse",
                "normalisation",
                "mixtures",
                "optmat_raman",
                "tdse",
            ),
        ),
    ),
    "T": (
        Module(
            "translate/",
            (
                "problem",
                "stage",
                "linop",
                "expansion",
                "chebyshev",
                "modal",
                "modal_tables",
                "modal_path",
                "coupled",
                "general",
                "radial",
                "free",
                "planar",
                "spatial",
                "flux",
                "coupling",
                "responses",
                "ionisation",
                "plasma_integration",
                "plasma_source",
                "envelope_plasma",
                "raman",
                "masks",
                "precision",
                "tdse",
                "tdse_source",
            ),
        ),
    ),
    "B": (
        Module(
            "backends/",
            (
                "protocol",
                "registry",
                "driver",
                "batch",
                "bodies",
                "spatial",
                "populations",
                "memory",
                "multistep",
                "multistep_general",
                "adams_core",
                "exponential",
                "low_storage",
            ),
        ),
        Module(
            "backends/numpy_ref/",
            (
                "backend",
                "batch",
                "passes",
                "transforms",
                "single",
                "admission",
                "envelope_plasma",
                "general_modal",
                "tdse",
            ),
        ),
        Module(
            "backends/mlx_gpu/",
            ("backend", "batch", "passes", "arithmetic", "envelope"),
        ),
        Module(
            "backends/numba_cpu/",
            (
                "backend",
                "batch",
                "kernels",
                "envelope_plasma",
                "general_modal",
                "coupled_flow",
                "field_plasma",
                "fft",
                "openmp",
                "spatial",
                "planar",
                "tables",
                "multistep",
                "dispersive",
                "dispersive_spatial",
                "dispersive_general",
                "low_storage",
                "pass_timers",
            ),
        ),
    ),
    "D": (
        Module("log.py"),
        Module("stats.py"),
        Module(
            "stepper/",
            (
                "tableau",
                "runge_kutta",
                "multistep",
                "low_storage",
                "controller",
                "schedule",
            ),
        ),
        Module(
            "output/",
            (
                "schema",
                "capability",
                "memory",
                "hdf5",
                "tdse_archive",
                "snapshots",
                "scan",
                "scan_artifacts",
            ),
        ),
        Module("scan.py"),
        Module("driver/executors/", ("local", "pool", "queue", "slurm", "batch")),
        Module(
            "driver/",
            (
                "simulation",
                "run",
                "summary",
                "interface",
                "scaffold",
                "mixtures",
                "material_output",
                "tdse",
                "tdse_consistency",
                "scan",
                "batch",
                "memory",
            ),
        ),
    ),
    "X": (
        Module(
            "processing/",
            (
                "axes",
                "fields",
                "spectra",
                "time",
                "coherence",
                "compression",
                "mixtures",
                "modal",
                "planes",
                "scans",
            ),
        ),
        Module(
            "plotting/",
            ("propagation", "statistics", "spectrograms", "style"),
        ),
        Module("sol.py"),
    ),
}
"""Every module of the package, in its layer's band. All of these exist."""

PLANNED = {
    "X": (Module("processing/", ("windows",)),),
}
"""Modules `architecture.md` section 4 names that are not written yet.

They are drawn dashed, and the test asserts each one is **absent**: a box turns solid
only when the module lands, and the figure cannot quietly promise code that is not
there.
"""


class Stage(NamedTuple):
    """One stage of the data flow of one run (`architecture.md` section 5)."""

    letter: str
    """The layer it belongs to."""

    title: str
    """What the stage is called on the page."""

    lines: tuple[str, ...]
    """What it does, one line each."""

    note: str = ""
    """The right-hand annotation section 5 carries beside it."""


RUN_STAGES = (
    Stage(
        "D",
        "prop_capillary(...) / prop_gnlse(...)",
        (
            "builds the physics objects: Grid, Modes, ZProfiles,",
            "Responses, Pulses, Noise, Boundaries, StepperSettings, Output",
        ),
        "a user script does the same by hand",
    ),
    Stage(
        "D",
        "Simulation(...)",
        (
            "translate.assemble(...)  →  KernelProblem",
            "backends.select(name, problem, grid=, modes=, profiles=)",
        ),
        "validated arrays, schema_version",
    ),
    Stage(
        "B",
        "StepDriver.propagate_to(z_save)",
        (
            "the march, the replay, the landing, the controller,",
            "the history, the noise, the per-stage bundle",
        ),
        "shared Python step loop",
    ),
    Stage(
        "D",
        "run loop over save planes",
        (
            "obs = backend.observe()        stats = backend.stats(spec)",
            "output.save(SavedPlane(z, field, stats, step_history, step_stats))",
        ),
        "small host arrays, one fused pass",
    ),
    Stage(
        "X",
        "MemoryOutput / HDF5Output  →  processing / plotting",
        ("energy, fwhm, bandpass, g12  →  prop_2d, spec_1d, spectrograms",),
        "device arrays never cross the seam",
    ),
)
"""The five stages of `architecture.md` section 5, with the annotations it carries."""


class Section(NamedTuple):
    """One section of the kernel contract, and what stands either side of it."""

    number: int
    """Its section number in `kernel-contract.md`."""

    title: str
    """Its heading there."""

    fields: tuple[str, ...]
    """The `KernelProblem` fields it documents."""

    leaf: str
    """The dataclass that carries them, as the contract names it."""

    feeds: tuple[str, ...]
    """The modules that build it."""

    feeds_layer: str
    """The layer those modules belong to, which colours them."""

    backend: str
    """What a backend does with it."""


CONTRACT_SECTIONS = (
    Section(
        1,
        "identity and validation",
        ("schema_version", "grid_kind", "shape", "flags", "precision"),
        "translate.problem.Flags",
        ("derived from the content of §2 to §8", "by translate.assemble"),
        "T",
        "specialises kernels at compile time; validate() refuses the rest",
    ),
    Section(
        2,
        "grids and transforms",
        ("grids",),
        "translate.masks.Grids",
        ("grid.py", "boundaries.py", "fourier.py"),
        "P",
        "crop/pad indices, masks, windows and the analytic temporal derivative",
    ),
    Section(
        3,
        "linear operator",
        ("linear", "flux"),
        "translate.linop.LinearOperator",
        ("linear.py", "modes/", "profiles.py"),
        "P",
        "terms uploaded once, scalars per stage: ΔΦ = Σ_j w_j · terms_j",
    ),
    Section(
        4,
        "multimode transform",
        ("transform",),
        "translate.modal.Transform",
        ("modes.quadrature", "modes.collection"),
        "P",
        "the two GEMMs: synthesis onto the nodes, projection back onto the modes",
    ),
    Section(
        5,
        "responses and the prefactor",
        ("responses", "tdse", "current_numerator"),
        "translate.responses.CompiledResponses",
        ("nonlinear/", "physdata/", "boundaries.py"),
        "P",
        "non-plasma P and held-current streams; independent TDSE P/J factors",
    ),
    Section(
        6,
        "initial state and noise",
        ("initial",),
        "translate.problem.Initial",
        ("pulses.py", "noise.py"),
        "P",
        "set_initial once; the noise advances as N0 · exp(i Im Φ)",
    ),
    Section(
        7,
        "stepper settings and schedule",
        ("stepper",),
        "translate.problem.StepperSettings",
        ("stepper/", "tableau · controller · schedule"),
        "D",
        "the controller's decision, and the planes and kinks to land on",
    ),
    Section(
        8,
        "statistics requests",
        ("stats_spec",),
        "solprop.stats.StatsSpec",
        ("stats.py", "the catalogue and the energy windows"),
        "D",
        "one fused pass per save plane, deterministic reductions",
    ),
)
"""The eight sections of `kernel-contract.md`, each with the fields it documents.

Section 9 of that document is not a field: it is `Observation`, what comes back.
"""

PHI_REPRESENTATIONS = ("constant", "expansion", "table", "separable")
"""The four representations of Phi and of the prefactor (`kernel-contract.md` 3, 5).

``device_ready`` is exactly "neither of them is ``table``".
"""


class Pass(NamedTuple):
    """One of the seven passes of an interaction-picture stage."""

    number: int
    """Its number in `architecture.md` section 8."""

    title: str
    """What that section calls it."""

    detail: str
    """The formula, or what the pass runs over."""

    functions: tuple[str, ...]
    """The `numpy_ref` passes that implement it, one function per pass."""


STAGE_PASSES = (
    Pass(
        1,
        "stage combine, lab frame, noise, pad",
        "u = exp(ΔΦ_i)·(y + h Σ_j a_ij k_j) + N(z_n + c_i h)",
        ("stage_input",),
    ),
    Pass(2, "batched inverse FFT", "fine grid, n_modes · n_pol columns", ("to_time",)),
    Pass(
        3,
        "synthesis GEMM",
        "(nodes · n_pol) × n_modes   by   n_modes × n_to",  # noqa: RUF001 - the multiplication sign is what the figure should print
        ("synthesise",),
    ),
    Pass(
        4,
        "fused response pass",
        "per column: collared non-plasma P, gJ and g'J",
        ("nonlinear_polarisation", "NonlinearSources", "nonlinear_sources"),
    ),
    Pass(
        5,
        "projection GEMM",
        "back onto the modes, accumulated over chunks",
        ("project",),
    ),
    Pass(6, "batched forward FFT", "three fine-grid streams", ("to_frequency",)),
    Pass(
        7,
        "crop, prefactor, windows, mask, frame",
        "k_i = exp(−ΔΦ_i)·p(ω)·W(ω)·crop(P̃ + A_J);"  # noqa: RUF001 - a minus sign, in a formula
        "   acc_sol += b_i k_i,   acc_err += e_i k_i",
        ("held_plasma_spectrum", "stage_output", "accumulate"),
    ),
)
"""The seven passes of `architecture.md` section 8, with their reference functions."""

RESPONSE_COLUMN = (
    ("intensity", "|E|² on the fine time axis", ()),
    ("ionisation_rate", "a cubic spline in the log field", ("ionisation_rate",)),
    (
        "plasma_scan",
        "independent channel exposures in t, by the running-integral rule",
        (
            "plasma_scan",
            "plasma_channel_fractions",
            "plasma_integration_diagnostics",
            "plasma_integration_statistics",
        ),
    ),
    (
        "plasma_current",
        "weighted currents; physical P retained for direct response evaluation",
        ("plasma_current", "plasma_channels_polarisation"),
    ),
    (
        "kerr",
        "χ⁽³⁾, with the third harmonic where the grid holds it",
        ("kerr_polarisation",),
    ),
    (
        "dispersive_kerr",
        "χ⁽³⁾ at every quartet's frequencies: filtered fields, products, weights",
        ("dispersive_kerr_polarisation",),
    ),
    (
        "raman_drive",
        "one/three real histories on doubled grids, per chunk",
        (
            "raman_polarisation",
            "raman_spectrum",
            "mixture_raman_spectrum",
            "tensor_raman_spectrum",
        ),
    ),
    ("accumulate", "gP_nonplasma, gJ and g'J in separate streams", ()),
)
"""`response_order` of `kernel-contract.md` section 5, with its reference passes.

The first and last are stages of the fused pass rather than functions of their own,
which is why they name none.
"""

STEP_PASSES = (
    ("accept_step", "y ← exp(ΔΦ)·(y + h·acc_sol)"),
    ("error_norm", "the embedded difference, reduced in index order"),
    ("apply_hard_mask", "active_mask, once per accepted step"),
    ("apply_temporal_rate", "the split-step absorber, when absorber_scheme is that"),
    ("absorber_term", "the in-stage absorber, masked to the active band"),
)
"""What runs once per step rather than per stage, and the reference pass for each."""

TABLEAU = (
    "DOPRI5: order 5, embedded order 4",
    "7 stages, 6 evaluations of N",
    "k1(n+1) = exp(ΔΦ(n))·k7(n)",
    "FSAL: numba_cpu takes it, numpy_ref not",
)
"""What `stepper.md` section 2 says about the tableau, for the panel beside it."""

SEAM_METHODS = (
    ("set_initial", "the state and the noise realisation, host to device, once"),
    ("set_progress", "the observer each attempted step reports to; never an input"),
    ("propagate_to", "the shared step loop, returning a StepSummary"),
    ("observe", "host arrays: the field at the plane and the step history"),
    (
        "restore_call_ledgers",
        "resume cumulative calls; numerical history starts afresh",
    ),
    ("stats", "one fused pass, a dict of small host arrays"),
    ("metadata", "what the execution was: pools, plans, digests, propagator mode"),
    ("memory_ledger", "every named buffer and its size"),
    ("available_device_memory", "free device bytes for batch admission; None on hosts"),
    ("close", "idempotent"),
)
"""The methods of `backends.protocol.Backend`, including resume call accounting."""

DRIVER_PARTS = (
    ("the march and the landing", ("_march_to",)),
    ("the replay", ("_replay_to", "_schedule_index_at")),
    ("the history", ("_record", "observe")),
    ("the progress reporter", ("set_progress",)),
    ("the statistics channel", ("_take_step_stats", "stats")),
    ("the noise", ("_noise_field", "_noise_at", "_noise_phase")),
    (
        "the per-stage bundle",
        ("_hoist", "_stage_scalars", "_delta_phi", "_prefactor", "_unitary_phase"),
    ),
    (
        "what a backend supplies",
        ("_trial_step", "_apply_mask", "_finish", "_begin_plane", "_on_accepted_step"),
    ),
)
"""What `backends/driver.py` owns, and the `StepDriver` members of each part."""

CONTROLLER_PARTS = ("propose", "accepts", "rejection_factor", "equivalent_norm")
"""The methods of `stepper.controller.PIController` the driver calls."""

SCHEDULE_PARTS = ("landing", "discover", "calibrate", "replay")
"""The functions of `stepper.schedule` behind the landing and the fixed schedules."""

NUMPY_REF = (
    "plain NumPy and scipy.fft",
    "one named function per pass",
    "the general representation, the z table included",
    "propagator_mode auto → tabulate",
)
"""What `backends.md` section 2 says the oracle is."""

NUMBA_CPU = (
    "fused @njit columns over bodies.py",
    "TransformPlans: pyfftw, in place, FFTW_MEASURE",
    "ThreadPools: numba, FFTW, BLAS, OMP",
    "host z-table adapter; auto → recompute",
)
"""What `backends.md` section 3 says the first optimised backend is."""


class Dataset(NamedTuple):
    """One dataset or group of a saved run (`output/hdf5.py`, `output/schema.py`)."""

    name: str
    """Its key in the file."""

    shape: str
    """Its shape and dtype, or what the group holds."""

    lines: tuple[str, ...] = ()
    """What sits under it: members of a group, or attributes of a dataset."""


OUTPUT_GROUPS = (
    Dataset("z", "(n_z,) float64", ()),
    Dataset(
        "field",
        "(n_z, n_modes, n_pol, n_omega) complex128",
        ('spectral_convention = "optics"  (ADR-017)',),
    ),
    Dataset(
        "noise_phase",
        "the same shape, float64",
        ('spectral_convention = "fft"; present when a noise model ran',),
    ),
    Dataset(
        "stats/",
        "one dataset per statistic, leading n_z",
        (
            "energy · energy_window · peak_power · peak_power_window ·",
            "peak_intensity · fwhm_t · fwhm_r · electron_density ·",
            "peak_ionisation_rate · density · pressure · core_radius ·",
            "zdw · quadrature_error · omega0",
        ),
    ),
    Dataset(
        "step_history/",
        "one dataset per column, plus counts",
        ("z, h, norm, accepted, wall_time  (schema.HISTORY_FIELDS)",),
    ),
    Dataset(
        "step_stats/",
        "the periodic subset between planes, plus counts",
        ("z, step  (schema.STEP_STATS_INDEX), then one dataset per statistic",),
    ),
    Dataset(
        "meta/json",
        "RunMetadata as one JSON dataset",
        (
            "grid, modes, simulation_kind, configuration, stepper, noise,",
            "backend (BackendRecord), versions, git (GitProvenance), created",
        ),
    ),
)
"""The datasets and groups `HDF5Output` writes, in the order it creates them."""

OUTPUT_STATS = (
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_intensity",
    "fwhm_t",
    "fwhm_r",
    "electron_density",
    "peak_ionisation_rate",
    "density",
    "pressure",
    "core_radius",
    "zdw",
    "quadrature_error",
    "omega0",
)
"""The statistics named in the ``stats/`` box; every one is in `stats.CATALOGUE`."""

OUTPUT_READERS = (
    ("output.hdf5", ("open_file", "resume", "check_cache")),
    ("processing", ("energy", "fwhm_t", "bandpass", "arrival_time", "g12")),
    ("plotting", ("prop_2d", "spec_1d", "energy", "spectral_display")),
)
"""What reads a saved run back, and the entry points each offers."""

RESUME_KEY = "RunMetadata.configuration_hash"
"""The resume digest: everything that was computed, and not how it was executed."""


# ------------------------------------------------------------------- the figures


def _wrap(
    items: Sequence[str], budget: float, separator: str = " · "
) -> tuple[str, ...]:
    """Join names with a separator, breaking into lines that fit a width.

    Parameters
    ----------
    items : sequence of str
        The names to join, in the order they should read.
    budget : float
        The width available for one line, in user units.
    separator : str, optional
        What goes between two names on a line; a middle dot by default, a space when
        the items are words of a sentence.

    Returns
    -------
    tuple of str
        One or more lines, each within the budget unless a single name exceeds it.
    """
    lines: list[str] = []
    current = ""
    for item in items:
        candidate = f"{current}{separator}{item}" if current else item
        if current and text_width(candidate, SMALL_SIZE) > budget:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


class Chip(NamedTuple):
    """A module box of the layers figure, already laid out."""

    title: str
    """The module or package path."""

    lines: tuple[str, ...]
    """Its members, wrapped."""

    width: float
    """How wide it must be drawn."""

    klass: str
    """The CSS class that colours it."""

    ink: str
    """The CSS class of its title."""


CHIP_BUDGET = 232.0
"""Widest a module chip's member list is allowed to run before it wraps."""

CHIP_GAP = 10.0
"""Space between two module chips, in user units."""


def _chips(letter: str) -> tuple[Chip, ...]:
    """Lay out every module chip of one layer's band, built ones then planned ones.

    Parameters
    ----------
    letter : str
        The layer letter, one of :data:`LAYER_LETTERS`.

    Returns
    -------
    tuple of Chip
        The chips, sized to their text.
    """
    lower = letter.lower()
    chips: list[Chip] = []
    for module in MODULES.get(letter, ()):
        lines = _wrap(module.members, CHIP_BUDGET)
        chips.append(
            Chip(
                module.path,
                lines,
                node_width(module.path, lines),
                f"fill-{lower}",
                f"ink-{lower}",
            )
        )
    for module in PLANNED.get(letter, ()):
        lines = _wrap(module.members, CHIP_BUDGET)
        chips.append(
            Chip(module.path, lines, node_width(module.path, lines), "planned", "dim")
        )
    return tuple(chips)


def _rows(chips: Sequence[Chip], width: float) -> tuple[tuple[Chip, ...], ...]:
    """Break a band's chips into rows that fit the width available.

    Parameters
    ----------
    chips : sequence of Chip
        The chips, in the order they should read.
    width : float
        The width of the band's content column, in user units.

    Returns
    -------
    tuple of tuple of Chip
        The rows, in order.
    """
    rows: list[list[Chip]] = [[]]
    used = 0.0
    for chip in chips:
        extra = chip.width + (CHIP_GAP if rows[-1] else 0.0)
        if rows[-1] and used + extra > width:
            rows.append([chip])
            used = chip.width
        else:
            rows[-1].append(chip)
            used += extra
    return tuple(tuple(row) for row in rows)


HEADER_WIDTH = 262.0
"""Width of a band's header: the letter, the layer's name and what it may import."""

GUTTER_WIDTH = 126.0
"""Width of the right-hand lane the import arrows run in."""

BAND_PAD = 13.0
"""Space between a band's edge and the chips inside it."""

BAND_GAP = 46.0
"""Vertical space between two bands, which is where the short import arrows live."""


def _row_height(row: Sequence[Chip]) -> float:
    """Return the height of one row of module chips: its tallest member.

    Parameters
    ----------
    row : sequence of Chip
        The chips on the row.

    Returns
    -------
    float
        The row's height, in user units.
    """
    return max((node_height(chip.lines) for chip in row), default=34.0)


def _band_header(
    drawing: drawsvg.Drawing, layer: Layer, x: float, y: float, height: float
) -> None:
    """Draw a band's letter badge, name and import rule.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    layer : Layer
        The layer this band shows.
    x, y : float
        The top-left corner of the header column, in user units.
    height : float
        The band's height, so the header can be centred against it.
    """
    lower = layer.letter.lower()
    _box(drawing, x, y + height / 2 - 17, 34, 34, klass=f"fill-{lower}")
    _text(
        drawing,
        layer.letter,
        x + 17,
        y + height / 2 + 6,
        size=19,
        klass=f"bold mid ink-{lower}",
    )
    rule = _wrap(layer.imports.split(", "), HEADER_WIDTH - 96)
    first = y + height / 2 - 2 - (len(rule) - 1) * LINE / 2
    _text(
        drawing, layer.name, x + 44, first, size=TITLE_SIZE, klass=f"bold ink-{lower}"
    )
    _text(drawing, "may import", x + 44, first + LINE + 2, size=SMALL_SIZE, klass="dim")
    for index, line in enumerate(rule):
        _text(
            drawing,
            line.replace(" · ", ", "),
            x + 122,
            first + LINE + 2 + index * LINE,
            size=SMALL_SIZE,
            klass="dim",
        )


def _legend(drawing: drawsvg.Drawing, x: float, y: float) -> None:
    """Draw the one legend the page needs: a swatch per layer, and the dashed key.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The top-left corner of the legend, in user units.
    """
    cursor = x
    for layer in LAYERS:
        lower = layer.letter.lower()
        _box(drawing, cursor, y, 16, 16, klass=f"fill-{lower}")
        _text(
            drawing,
            layer.letter,
            cursor + 22,
            y + 12,
            size=SMALL_SIZE,
            klass=f"bold ink-{lower}",
        )
        cursor += 40
    _box(drawing, cursor, y, 16, 16, klass="planned")
    _text(
        drawing,
        "unwritten candidate, not scheduled (architecture.md §4)",
        cursor + 22,
        y + 12,
        size=SMALL_SIZE,
        klass="dim",
    )


def layers() -> drawsvg.Drawing:
    """Draw figure 1: the five layers, every module in its band, and the import edges.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    content = WIDTH - 2 * MARGIN - HEADER_WIDTH - GUTTER_WIDTH
    bands = [(layer, _rows(_chips(layer.letter), content)) for layer in LAYERS]
    heights = []
    for _, rows in bands:
        tall = [_row_height(row) for row in rows]
        heights.append(2 * BAND_PAD + sum(tall) + (len(rows) - 1) * CHIP_GAP)
    notes = [
        f"{index + 1}   {label}" for index, (_, _, label) in enumerate(DISTANT_EDGES)
    ]
    notes.append(
        "The package-wide modules carry no layer letter:  "
        + "  ·  ".join(module.path for module in PACKAGE_WIDE)
    )
    subtitle = (
        "Bands and colours from architecture.md §3 and decisions.md ADR-001; modules "
        "from src/solprop. Arrows point from the layer that imports to the layer it "
        "imports."
    )
    top = header_height(subtitle) + 34
    height = (
        top
        + sum(heights)
        + (len(bands) - 1) * BAND_GAP
        + 22
        + len(notes) * LINE
        + MARGIN
    )
    drawing = _canvas(height)
    _header(drawing, "The layers of solprop, and every module in each", subtitle)
    _legend(drawing, MARGIN, header_height(subtitle) - 4)

    tops: dict[str, tuple[float, float]] = {}
    y = top
    for (layer, rows), band_height in zip(bands, heights, strict=True):
        _box(
            drawing,
            MARGIN,
            y,
            WIDTH - 2 * MARGIN - GUTTER_WIDTH,
            band_height,
            klass="plain",
        )
        _band_header(drawing, layer, MARGIN + 14, y, band_height)
        row_y = y + BAND_PAD
        for row in rows:
            tall = _row_height(row)
            cursor = MARGIN + HEADER_WIDTH + BAND_PAD
            for chip in row:
                _box(drawing, cursor, row_y, chip.width, tall, klass=chip.klass)
                _text(
                    drawing,
                    chip.title,
                    cursor + PAD,
                    row_y + PAD + TITLE_SIZE - 3,
                    size=TITLE_SIZE,
                    klass=f"bold {chip.ink}",
                )
                for index, line in enumerate(chip.lines):
                    _text(
                        drawing,
                        line,
                        cursor + PAD,
                        row_y + PAD + TITLE_SIZE + (index + 1) * LINE - 3,
                        size=SMALL_SIZE,
                        klass="dim",
                    )
                cursor += chip.width + CHIP_GAP
            row_y += tall + CHIP_GAP
        tops[layer.letter] = (y, y + band_height)
        y += band_height + BAND_GAP

    _import_edges(drawing, tops)
    note_y = height - MARGIN - (len(notes) - 1) * LINE
    for index, note in enumerate(notes):
        _text(
            drawing, note, MARGIN, note_y + index * LINE, size=SMALL_SIZE, klass="dim"
        )
    return drawing


ADJACENT_EDGES = (
    ("T", "P", "physics objects"),
    ("B", "T", "delta_phi, prefactor_at, the stage bundle"),
    ("D", "B", "backends.select, then the seam"),
    ("X", "D", "a saved run, through the Output protocol"),
)
"""Import edges between neighbouring bands, drawn in the gap between them."""

DISTANT_EDGES = (
    ("D", "T", "Simulation calls translate.assemble"),
    ("D", "P", "the driver builds the grid, the modes, the pulse and the responses"),
    (
        "B",
        "P",
        "the statistics path: a backend is handed grid, modes and profiles at "
        "construction: the one import of P that architecture.md §3 allows "
        "B, types and constants, never a physics object",
    ),
    (
        "X",
        "T",
        "processing.modal takes the run's own ModalTransformTables beside the "
        "handler: a general modal file records what its reference is and not the "
        "arrays, so the reference is rebuilt in T and read in X",
    ),
)
"""Import edges that skip a band, drawn in the right-hand gutter and keyed by number."""


def _import_edges(
    drawing: drawsvg.Drawing, tops: dict[str, tuple[float, float]]
) -> None:
    """Draw the import arrows of `architecture.md` section 3 between the bands.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    tops : dict
        Layer letter to the band's top and bottom, in user units.
    """
    inner_right = WIDTH - MARGIN - GUTTER_WIDTH
    for index, (importer, imported, label) in enumerate(ADJACENT_EDGES):
        x = MARGIN + HEADER_WIDTH + 32 + index * 8
        _arrow(drawing, [(x, tops[importer][0] - 5), (x, tops[imported][1] + 5)])
        _text(
            drawing,
            f"{importer} → {imported}   {label}",
            x + 12,
            (tops[importer][0] + tops[imported][1]) / 2 + 4,
            size=SMALL_SIZE,
            klass="dim",
        )
    for index, (importer, imported, _) in enumerate(DISTANT_EDGES):
        lane = inner_right + 26 + index * 32
        start = (tops[importer][0] + tops[importer][1]) / 2
        end = (tops[imported][0] + tops[imported][1]) / 2
        _arrow(
            drawing,
            [
                (inner_right + 4, start),
                (lane, start),
                (lane, end),
                (inner_right + 6, end),
            ],
        )
        _key_mark(drawing, lane, (start + end) / 2, str(index + 1))


def _key_mark(drawing: drawsvg.Drawing, x: float, y: float, mark: str) -> None:
    """Draw a small numbered tag on a connector, keyed to a note under the figure.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The centre of the tag, in user units.
    mark : str
        What to print in it, usually a single digit.
    """
    _box(drawing, x - 9, y - 9, 18, 18, klass="card")
    _text(drawing, mark, x, y + 4, size=SMALL_SIZE, klass="bold mid dim")


RUN_ARTEFACTS = (
    (
        "P",
        "Grid · Modes · ZProfiles · Responses",
        ("Pulses · Noise · Boundaries · StepperSettings · Output",),
    ),
    (
        "T",
        "KernelProblem",
        (f"plain arrays, frozen, validated, schema_version {_KERNEL_SCHEMA_VERSION}",),
    ),
    ("B", "StepSummary", ("z, n_steps, n_rejected, max_norm, mean_norm, wall_time",)),
    (
        "B",
        "Observation  →  SavedPlane",
        (
            "z, field, noise_phase, step_history, step_stats;",
            "stats: a dict of small host arrays, keyed by name",
        ),
    ),
    ("D", "MemoryOutput / HDF5Output", ("one save protocol, one metadata schema",)),
)
"""What each stage of `RUN_STAGES` hands on, and the layer that owns it."""

SPINE_WIDTH = 560.0
"""Width of the call spine of the run figure, in user units."""

SPINE_GAP = 190.0
"""Space between the spine and the column of things that cross, in user units."""


def one_run() -> drawsvg.Drawing:
    """Draw figure 2: the data flow of one run, and what crosses at each step.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    right_x = MARGIN + SPINE_WIDTH + SPINE_GAP
    right_width = WIDTH - MARGIN - right_x
    subtitle = (
        "architecture.md §5, stage for stage. The spine is the call sequence; the "
        "column on the right is what each stage hands on, in the colour of the layer "
        "that owns it."
    )
    top = header_height(subtitle) + 26
    gap = 40.0
    spine = [node_height(stage.lines) for stage in RUN_STAGES]
    height = top + sum(spine) + (len(spine) - 1) * gap + MARGIN

    drawing = _canvas(height)
    _header(drawing, "The data flow of one run", subtitle)
    label_y = top - 12
    _text(drawing, "the calls", MARGIN, label_y, size=SMALL_SIZE, klass="bold dim")
    _text(drawing, "what crosses", right_x, label_y, size=SMALL_SIZE, klass="bold dim")

    y = top
    for index, stage in enumerate(RUN_STAGES):
        lower = stage.letter.lower()
        card = spine[index]
        _node(
            drawing,
            Slot(MARGIN, y, SPINE_WIDTH),
            stage.title,
            stage.lines,
            Paint(f"fill-{lower}", f"ink-{lower}"),
        )
        letter, title, lines = RUN_ARTEFACTS[index]
        _node(
            drawing,
            Slot(right_x, y, right_width),
            title,
            lines,
            Paint(f"fill-{letter.lower()}", f"ink-{letter.lower()}"),
        )
        middle = y + card / 2
        _arrow(drawing, [(MARGIN + SPINE_WIDTH + 6, middle), (right_x - 6, middle)])
        note = _wrap(stage.note.split(), SPINE_GAP - 22, separator=" ")
        for line_index, line in enumerate(note):
            _text(
                drawing,
                line,
                right_x - SPINE_GAP / 2,
                middle + 16 + line_index * LINE,
                size=SMALL_SIZE,
                klass="mid dim",
            )
        if index + 1 < len(RUN_STAGES):
            x = MARGIN + SPINE_WIDTH / 2
            _arrow(drawing, [(x, y + card + 6), (x, y + card + gap - 6)])
        y += card + gap
    return drawing


def _table(
    drawing: drawsvg.Drawing,
    slot: Slot,
    title: str,
    rows: Sequence[tuple[str, str]],
    key_width: float,
    paint: Paint = NO_PAINT,
) -> float:
    """Draw a card whose body is a two-column listing: a name and what it is.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    slot : Slot
        Where the card goes and how wide it is.
    title : str
        The card's heading.
    rows : sequence of tuple of str
        ``(name, note)`` per row; the note wraps to as many lines as it needs.
    key_width : float
        Width of the name column, in user units.
    paint : Paint, optional
        The classes that colour the card, its heading and its names.

    Returns
    -------
    float
        The card's height, so a caller can stack the next one under it.
    """
    x, y = slot.x, slot.y
    budget = slot.width - 2 * PAD - key_width
    wrapped = [_wrap(note.split(), budget, separator=" ") for _, note in rows]
    body = sum(max(len(lines), 1) for lines in wrapped) * LINE
    height = PAD + TITLE_SIZE + 6 + body + PAD
    _box(drawing, x, y, slot.width, height, klass=paint.fill)
    _text(
        drawing,
        title,
        x + PAD,
        y + PAD + TITLE_SIZE - 3,
        size=TITLE_SIZE,
        klass=f"bold {paint.ink}",
    )
    cursor = y + PAD + TITLE_SIZE + 6
    for (name, _), lines in zip(rows, wrapped, strict=True):
        _text(
            drawing,
            name,
            x + PAD,
            cursor + LINE - 3,
            size=SMALL_SIZE,
            klass=f"bold {paint.ink}",
        )
        for index, line in enumerate(lines):
            _text(
                drawing,
                line,
                x + PAD + key_width,
                cursor + (index + 1) * LINE - 3,
                size=SMALL_SIZE,
                klass="dim",
            )
        cursor += max(len(lines), 1) * LINE
    return height


def _table_height(
    rows: Sequence[tuple[str, str]], width: float, key_width: float
) -> float:
    """Return the height :func:`_table` would draw, without drawing it.

    Parameters
    ----------
    rows : sequence of tuple of str
        The same rows :func:`_table` would be given.
    width : float
        The card's width, in user units.
    key_width : float
        Width of the name column, in user units.

    Returns
    -------
    float
        The height, in user units.
    """
    budget = width - 2 * PAD - key_width
    body = sum(
        max(len(_wrap(note.split(), budget, separator=" ")), 1) for _, note in rows
    )
    return PAD + TITLE_SIZE + 6 + body * LINE + PAD


def note_height(lines: Sequence[str]) -> float:
    """Return the height of an untitled box holding this many lines.

    Parameters
    ----------
    lines : sequence of str
        The lines it holds.

    Returns
    -------
    float
        The box height, in user units.
    """
    return PAD + len(lines) * LINE + PAD


def _notes(
    drawing: drawsvg.Drawing,
    area: tuple[float, float, float, float],
    lines: Sequence[str],
    klass: str = "card",
) -> None:
    """Draw an untitled box of dim lines, vertically centred in the height given.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    area : tuple of float
        ``(x, y, width, height)``, in user units.
    lines : sequence of str
        What it says, one line each.
    klass : str, optional
        The CSS class that colours the box.
    """
    x, y, width, height = area
    _box(drawing, x, y, width, height, klass=klass)
    first = y + height / 2 - (len(lines) - 1) * LINE / 2 + 4
    for index, line in enumerate(lines):
        _text(
            drawing, line, x + PAD, first + index * LINE, size=SMALL_SIZE, klass="dim"
        )


CONTRACT_LEFT = 296.0
"""Width of the column of modules that feed the contract, in user units."""

CONTRACT_MIDDLE = 420.0
"""Width of the `KernelProblem` column, in user units."""

CONTRACT_GAP = 58.0
"""Space either side of the `KernelProblem` column, in user units."""


def kernel_contract() -> drawsvg.Drawing:
    """Draw figure 3: what feeds each section of the contract, and what a backend does.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    middle_x = MARGIN + CONTRACT_LEFT + CONTRACT_GAP
    right_x = middle_x + CONTRACT_MIDDLE + CONTRACT_GAP
    right_width = WIDTH - MARGIN - right_x
    feeds = [
        _wrap(section.feeds, CONTRACT_LEFT - 2 * PAD) for section in CONTRACT_SECTIONS
    ]
    uses = [
        _wrap(section.backend.split(), right_width - 2 * PAD, separator=" ")
        for section in CONTRACT_SECTIONS
    ]
    rows = [
        max(note_height(feed), note_height(use), node_height(("x",)))
        for feed, use in zip(feeds, uses, strict=True)
    ]
    gap = 12.0
    subtitle = (
        "kernel-contract.md §1 to §8. KernelProblem is the only object that crosses "
        "into a backend: eleven fields, frozen, validated, versioned. On the left, "
        "what builds each section; on the right, what a backend does with it."
    )
    top = header_height(subtitle) + 30
    inset = inset_height()
    height = top + sum(rows) + (len(rows) - 1) * gap + 34 + inset + MARGIN

    drawing = _canvas(height)
    _header(drawing, "The kernel contract, section by section", subtitle)
    label_y = top - 12
    _text(drawing, "built by", MARGIN, label_y, size=SMALL_SIZE, klass="bold dim")
    _text(
        drawing,
        "translate.problem.KernelProblem",
        middle_x,
        label_y,
        size=SMALL_SIZE,
        klass="bold ink-t",
    )
    _text(
        drawing,
        "read by a backend",
        right_x,
        label_y,
        size=SMALL_SIZE,
        klass="bold dim",
    )

    y = top
    for index, section in enumerate(CONTRACT_SECTIONS):
        tall = rows[index]
        _notes(
            drawing,
            (MARGIN, y, CONTRACT_LEFT, tall),
            feeds[index],
            f"fill-{section.feeds_layer.lower()}",
        )
        _node(
            drawing,
            Slot(middle_x, y, CONTRACT_MIDDLE),
            f"§{section.number}   {' · '.join(section.fields)}",
            (section.title, section.leaf),
            Paint("fill-t", "ink-t"),
        )
        _notes(drawing, (right_x, y, right_width, tall), uses[index], "fill-b")
        middle = y + tall / 2
        _arrow(drawing, [(MARGIN + CONTRACT_LEFT + 6, middle), (middle_x - 6, middle)])
        _arrow(
            drawing, [(middle_x + CONTRACT_MIDDLE + 6, middle), (right_x - 6, middle)]
        )
        y += tall + gap

    _inset(drawing, y + 22)
    return drawing


INSET_KINDS = (
    (
        "constant",
        "phi_constant · prefactor_constant",
        "one term, and the distance as its weight",
    ),
    (
        "expansion",
        "ParameterExpansion: terms, weights",
        "n_terms of them, over the mode set's scalar paths",
    ),
    (
        "table",
        "phi_coefficients · prefactor",
        "a Clenshaw pass over the state's size × the degree",  # noqa: RUF001 - the multiplication sign is what the figure should print
    ),
    (
        "separable",
        "SeparableOperator: material and transverse vectors",
        "corrected split-square subtraction; no coefficient volume",
    ),
)
"""The four representations of the inset, with what carries each and what it costs."""

INSET_CLOSING = (
    'device_ready is exactly "neither of them is the z table". translate.stage '
    "splits a stage in two: terms, the fixed arrays uploaded once, and scalars, a "
    "small record of host floats per stage. A backend evaluates "
    "ΔΦ = stage.combine(weights, terms), or evaluates the separable vectors "
    "without a full coefficient volume; translate.stage.require_device_ready "
    "is the refusal device backends raise. Numba CPU also accepts z tables through "
    "its host adapter."
)
"""The sentence under the inset: what the predicate means and what the split is."""

INSET_HEADING = (
    "Φ (§3) and the prefactor (§5) each take one of four representations, and "
    "device_ready is a predicate over the pair"
)
"""The inset's own heading."""


def inset_height() -> float:
    """Return the height of figure 3's inset.

    Returns
    -------
    float
        Its height, in user units.
    """
    box = PAD + 3 * LINE + PAD
    closing = _wrap(INSET_CLOSING.split(), WIDTH - 2 * MARGIN - 2 * PAD, separator=" ")
    return PAD + TITLE_SIZE + 10 + 2 * box + CHIP_GAP + 16 + len(closing) * LINE + PAD


def _inset(drawing: drawsvg.Drawing, y: float) -> None:
    """Draw the inset of figure 3: the four representations, and ``device_ready``.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    y : float
        The inset's top edge, in user units.
    """
    width = WIDTH - 2 * MARGIN
    _box(drawing, MARGIN, y, width, inset_height(), klass="plain")
    _text(
        drawing,
        INSET_HEADING,
        MARGIN + PAD,
        y + PAD + TITLE_SIZE - 3,
        size=TITLE_SIZE,
        klass="bold",
    )
    box_width = (width - 2 * PAD - CHIP_GAP) / 2
    box_height = PAD + 3 * LINE + PAD
    row = y + PAD + TITLE_SIZE + 10
    for index, (kind, carrier, note) in enumerate(INSET_KINDS):
        x = MARGIN + PAD + (index % 2) * (box_width + CHIP_GAP)
        box_y = row + (index // 2) * (box_height + CHIP_GAP)
        ready = kind != "table"
        _box(
            drawing,
            x,
            box_y,
            box_width,
            box_height,
            klass="fill-t" if ready else "planned",
        )
        _text(
            drawing,
            f'"{kind}"',
            x + PAD,
            box_y + PAD + LINE - 4,
            size=BODY_SIZE,
            klass="bold " + ("ink-t" if ready else "dim"),
        )
        _text(
            drawing,
            carrier,
            x + PAD,
            box_y + PAD + 2 * LINE - 4,
            size=SMALL_SIZE,
            klass="dim",
        )
        _text(
            drawing,
            note,
            x + PAD,
            box_y + PAD + 3 * LINE - 4,
            size=SMALL_SIZE,
            klass="dim",
        )
    closing = _wrap(INSET_CLOSING.split(), WIDTH - 2 * MARGIN - 2 * PAD, separator=" ")
    first = row + 2 * box_height + CHIP_GAP + 16 + LINE - 4
    for index, line in enumerate(closing):
        _text(
            drawing,
            line,
            MARGIN + PAD,
            first + index * LINE,
            size=SMALL_SIZE,
            klass="dim",
        )


TITLE_TOP = 44.0
"""Baseline of a figure's own title, in user units."""

SUBTITLE_TOP = 64.0
"""Baseline of the first line of a figure's standfirst, in user units."""


def header_height(subtitle: str) -> float:
    """Return where a figure's content may start, under its title and standfirst.

    Parameters
    ----------
    subtitle : str
        The standfirst, which wraps to the figure's width.

    Returns
    -------
    float
        The first free y, in user units.
    """
    return SUBTITLE_TOP + len(_header_lines(subtitle)) * LINE


def _header_lines(subtitle: str) -> tuple[str, ...]:
    """Return a figure's standfirst, wrapped to the figure's width.

    Parameters
    ----------
    subtitle : str
        The standfirst.

    Returns
    -------
    tuple of str
        The lines to draw.
    """
    return _wrap(subtitle.split(), WIDTH - 2 * MARGIN, separator=" ")


def _header(drawing: drawsvg.Drawing, title: str, subtitle: str) -> None:
    """Draw a figure's title and standfirst.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    title : str
        The figure's own title, so it reads when it is opened on its own.
    subtitle : str
        What it shows and where the text is; it wraps.
    """
    _text(drawing, title, MARGIN, TITLE_TOP, size=21, klass="bold")
    for index, line in enumerate(_header_lines(subtitle)):
        _text(
            drawing,
            line,
            MARGIN,
            SUBTITLE_TOP + index * LINE,
            size=SMALL_SIZE,
            klass="dim",
        )


def footer_height(text: str) -> float:
    """Return the height a figure's closing note needs, wrapped to its width.

    Parameters
    ----------
    text : str
        The note.

    Returns
    -------
    float
        Its height, in user units.
    """
    return len(_wrap(text.split(), WIDTH - 2 * MARGIN, separator=" ")) * LINE


def _footer(drawing: drawsvg.Drawing, text: str, bottom: float) -> None:
    """Draw a figure's closing note, its last line sitting on the baseline given.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    text : str
        The note; it wraps to the figure's width.
    bottom : float
        The baseline of its last line, in user units.
    """
    lines = _wrap(text.split(), WIDTH - 2 * MARGIN, separator=" ")
    for index, line in enumerate(lines):
        _text(
            drawing,
            line,
            MARGIN,
            bottom - (len(lines) - 1 - index) * LINE,
            size=SMALL_SIZE,
            klass="dim",
        )


PIPE_WIDTH = 430.0
"""Width of the pipeline column of the stage figure, in user units."""

PIPE_GAP = 36.0
"""Space between the stage figure's three columns, in user units."""

SIDE_WIDTH = 356.0
"""Width of the column holding the response column and the accumulators."""

PASS_GAP = 16.0
"""Vertical space between two passes of the pipeline, in user units."""


def one_stage() -> drawsvg.Drawing:
    """Draw figure 4: the seven passes of a stage, and the step they belong to.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    side_x = MARGIN + PIPE_WIDTH + PIPE_GAP
    panel_x = side_x + SIDE_WIDTH + PIPE_GAP
    panel_width = WIDTH - MARGIN - panel_x
    budget = PIPE_WIDTH - 2 * PAD
    bodies = [
        (
            *_wrap(step.detail.split(), budget, separator=" "),
            "numpy_ref  ·  " + " · ".join(step.functions),
        )
        for step in STAGE_PASSES
    ]
    subtitle = (
        "architecture.md §8: each numbered line is one pass over its data, and each "
        "is a named function in backends/numpy_ref/passes.py. stepper.md §2 is the "
        "step the seven passes sit inside."
    )
    tops: list[float] = []
    top = header_height(subtitle) + 26
    y = top
    for body in bodies:
        tops.append(y)
        y += node_height(body) + PASS_GAP
    pipeline_bottom = y - PASS_GAP

    rows = tuple(
        (name, note + (f"  —  {' · '.join(funcs)}" if funcs else ""))
        for name, note, funcs in RESPONSE_COLUMN
    )
    column = _table_height(rows, SIDE_WIDTH, 104)
    accumulators = (
        "acc_sol += b_i k_i        acc_err += e_i k_i",
        "on the last stage the weak error norm is reduced",
        "in the same pass, in index order",
    )
    accumulator_y = tops[3] + column + 34
    side_bottom = accumulator_y + node_height(accumulators)
    bottom = max(pipeline_bottom, side_bottom)
    footer = (
        "Per step: 6 stages, 12 modal batched FFTs and 6 node-chunk FFT pairs for "
        "Raman, one controller decision, one accept pass, one hard-mask multiply. "
        "Steps land on save planes and hard kinks (ADR-009)."
    )
    height = bottom + 18 + footer_height(footer) + MARGIN

    drawing = _canvas(height)
    _header(drawing, "One stage of the interaction-picture step", subtitle)
    _text(
        drawing,
        "the seven passes of one stage",
        MARGIN,
        top - 12,
        size=SMALL_SIZE,
        klass="bold dim",
    )

    for index, step in enumerate(STAGE_PASSES):
        card = node_height(bodies[index])
        _node(
            drawing,
            Slot(MARGIN, tops[index], PIPE_WIDTH),
            f"{step.number}   {step.title}",
            bodies[index],
            Paint("fill-b", "ink-b"),
        )
        if index + 1 < len(STAGE_PASSES):
            x = MARGIN + PIPE_WIDTH / 2
            _arrow(drawing, [(x, tops[index] + card + 3), (x, tops[index + 1] - 3)])

    fourth = tops[3]
    _table(
        drawing,
        Slot(side_x, fourth, SIDE_WIDTH),
        "response_order, per column",
        rows,
        104,
        Paint("fill-p", "ink-p"),
    )
    _arrow(drawing, [(MARGIN + PIPE_WIDTH + 4, fourth + 24), (side_x - 4, fourth + 24)])
    _text(
        drawing,
        "over a chunk of transform.node_chunk nodes, so the buffers are",
        side_x,
        fourth + column + 14,
        size=SMALL_SIZE,
        klass="dim",
    )
    _text(
        drawing,
        "sized by the chunk and not by the quadrature rule",
        side_x,
        fourth + column + 28,
        size=SMALL_SIZE,
        klass="dim",
    )

    seventh = tops[6]
    _node(
        drawing,
        Slot(side_x, accumulator_y, SIDE_WIDTH),
        "the two accumulators",
        accumulators,
        Paint("fill-b", "ink-b"),
    )
    _arrow(
        drawing,
        [
            (MARGIN + PIPE_WIDTH + 4, seventh + 24),
            (side_x - 18, seventh + 24),
            (side_x - 18, accumulator_y + 24),
            (side_x - 4, accumulator_y + 24),
        ],
    )

    _step_panel(drawing, panel_x, top, panel_width, bottom - top)
    _footer(drawing, footer, height - MARGIN)
    return drawing


STAGE_BOX = 26.0
"""Side of one tableau-stage square in the step panel, in user units."""


def _step_panel(
    drawing: drawsvg.Drawing, x: float, y: float, width: float, height: float
) -> None:
    """Draw the panel beside the pipeline: DOPRI5, FSAL, and what runs once per step.

    Parameters
    ----------
    drawing : drawsvg.Drawing
        The drawing to append to.
    x, y : float
        The panel's top-left corner, in user units.
    width, height : float
        Its size, in user units.
    """
    _box(drawing, x, y, width, height, klass="plain")
    cursor = y + PAD + TITLE_SIZE - 3
    _text(drawing, "the step around it", x + PAD, cursor, size=TITLE_SIZE, klass="bold")
    cursor += 8
    for line in TABLEAU:
        cursor += LINE
        _text(drawing, line, x + PAD, cursor, size=SMALL_SIZE, klass="dim")

    cursor += 24
    span = 7 * STAGE_BOX + 6 * 5
    left = x + (width - span) / 2
    for index in range(7):
        box_x = left + index * (STAGE_BOX + 5)
        fsal = index == 6
        _box(
            drawing,
            box_x,
            cursor,
            STAGE_BOX,
            STAGE_BOX,
            klass="planned" if fsal else "fill-d",
        )
        _text(
            drawing,
            str(index + 1),
            box_x + STAGE_BOX / 2,
            cursor + 17,
            size=BODY_SIZE,
            klass="bold mid " + ("dim" if fsal else "ink-d"),
        )
    arc = cursor + STAGE_BOX + 14
    _arrow(
        drawing,
        [
            (left + span - STAGE_BOX / 2, cursor + STAGE_BOX + 3),
            (left + span - STAGE_BOX / 2, arc),
            (left + STAGE_BOX / 2, arc),
            (left + STAGE_BOX / 2, cursor + STAGE_BOX + 3),
        ],
    )
    cursor = arc + LINE + 4
    _text(
        drawing,
        "FSAL:  k1(n+1) = exp(ΔΦ(n))·k7(n)",
        x + width / 2,
        cursor,
        size=SMALL_SIZE,
        klass="mid dim",
    )

    cursor += 26
    _text(
        drawing,
        "once per step, not per stage",
        x + PAD,
        cursor,
        size=TITLE_SIZE,
        klass="bold",
    )
    cursor += 8
    for name, formula in STEP_PASSES:
        cursor += LINE
        _text(drawing, name, x + PAD, cursor, size=SMALL_SIZE, klass="bold")
        for line in _wrap(formula.split(), width - 2 * PAD - 8, separator=" "):
            cursor += LINE
            _text(drawing, line, x + PAD + 8, cursor, size=SMALL_SIZE, klass="dim")


BACKEND_OVERRIDES = ("_trial_step", "_apply_mask", "_finish", "_observed_field")
"""What a backend supplies to `StepDriver`, and nothing else of the loop."""

DEVICE_SLOT = (
    "solprop-native distribution, native/",
    "C++20 Kokkos + persistent Kokkos-FFT plans",
    "one native call per seven-stage trial",
    "OpenMP on CPUs; CUDA and HIP on GPUs",
)
"""The native backend, the second distribution, built from ``native/``."""


def backends() -> drawsvg.Drawing:
    """Draw figure 5: the seam, the one step loop, and the backends behind it.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    full = WIDTH - 2 * MARGIN
    left_width = 560.0
    right_x = MARGIN + left_width + 48
    right_width = WIDTH - MARGIN - right_x
    card_width = (full - PIPE_GAP) / 2

    seam = tuple(SEAM_METHODS)
    driver_rows = tuple(
        (label, " · ".join(members)) for label, members in DRIVER_PARTS[:-1]
    )
    stepper_rows = (
        ("stepper.tableau", "DOPRI5, Tsit5, the order-condition checks"),
        ("stepper.controller", "PIController: " + ", ".join(CONTROLLER_PARTS)),
        ("stepper.schedule", ", ".join(SCHEDULE_PARTS)),
        (DRIVER_PARTS[-1][0], " · ".join(DRIVER_PARTS[-1][1])),
    )
    seam_key_width = max(
        128.0, *(text_width(name, SMALL_SIZE) + PAD for name, _ in seam)
    )
    seam_height = _table_height(seam, full, seam_key_width)
    driver_height = max(
        _table_height(driver_rows, left_width, 168),
        _table_height(stepper_rows, right_width, 166),
    )
    cards = (
        ("numpy_ref  ·  the oracle", NUMPY_REF, Paint("fill-b", "ink-b")),
        ("numba_cpu  ·  CPU fallback", NUMBA_CPU, Paint("fill-b", "ink-b")),
        ("native  ·  preferred if installed", DEVICE_SLOT, Paint("fill-b", "ink-b")),
    )
    card_height = max(node_height(lines) for _, lines, _ in cards) + 22
    row_gap = 44.0
    bottom_height = 2 * card_height + row_gap

    subtitle = (
        "backends.md §1: the protocol is drawn at the level of operations on the "
        "propagated state, never at the array-namespace level. A backend is "
        "constructed from a validated KernelProblem and host arrays and returns host "
        "arrays; its buffers, layout and plans are private."
    )
    footer = (
        "Three things every backend owes beyond the methods: deterministic "
        "reductions, the device form of the stage, and one step loop. The error norm "
        "is control flow, so a sum whose association depends on the thread count "
        "forks the trajectory."
    )
    top = header_height(subtitle) + 20
    gap = 44.0
    height = (
        top
        + seam_height
        + gap
        + driver_height
        + gap
        + bottom_height
        + 18
        + footer_height(footer)
        + MARGIN
    )

    drawing = _canvas(height)
    _header(drawing, "One seam, one step loop, and the backends behind them", subtitle)

    y = top
    _table(
        drawing,
        Slot(MARGIN, y, full),
        "the seam  ·  backends.protocol.Backend",
        seam,
        seam_key_width,
        Paint("fill-b", "ink-b"),
    )
    y += seam_height
    _arrow(
        drawing,
        [(WIDTH / 2, y + 6), (WIDTH / 2, y + gap - 6)],
        label="backends.registry.select(name, problem, grid=, modes=, profiles=)",
        label_at=(WIDTH / 2, y + gap / 2 - 8),
    )
    y += gap

    _table(
        drawing,
        Slot(MARGIN, y, left_width),
        "backends/driver.py  ·  StepDriver, and no backend reimplements it",
        driver_rows,
        168,
        Paint("fill-d", "ink-d"),
    )
    _table(
        drawing,
        Slot(right_x, y, right_width),
        "the stepper it drives",
        stepper_rows,
        166,
        Paint("fill-d", "ink-d"),
    )
    y += driver_height

    for index, (title, lines, paint) in enumerate(cards):
        x = MARGIN + (index % 2) * (card_width + PIPE_GAP)
        card_y = y + gap + (index // 2) * (card_height + row_gap)
        _arrow(
            drawing,
            [
                (WIDTH / 2, y + 6),
                (WIDTH / 2, card_y - gap / 2),
                (x + card_width / 2, card_y - gap / 2),
                (x + card_width / 2, card_y - 6),
            ],
        )
        _node(drawing, Slot(x, card_y, card_width), title, lines, paint)
        _text(
            drawing,
            " · ".join(BACKEND_OVERRIDES),
            x + PAD,
            card_y + node_height(lines) + 14,
            size=SMALL_SIZE,
            klass="bold dim",
        )
    _footer(drawing, footer, height - MARGIN)
    return drawing


FILE_WIDTH = 616.0
"""Width of the saved-file column of the output figure, in user units."""

FILE_GAP = 52.0
"""Space between the file and the things that read it, in user units."""

RESUME_NOTE = (
    "the digest of everything that was computed: the grid, the",
    "modes, the configuration, the stepper settings and the",
    "noise realisation. Not in it: the package versions, the git",
    "provenance, the wall clock, and which",
    "backend ran it, because a run continued on the other backend",
    "agrees with either uninterrupted run.",
)
"""What the resume key covers, and what it deliberately does not."""


def output_side() -> drawsvg.Drawing:
    """Draw figure 6: what a saved run holds, and what reads it back.

    Returns
    -------
    drawsvg.Drawing
        The figure, ready to be written out.
    """
    _require_drawsvg()
    right_x = MARGIN + FILE_WIDTH + FILE_GAP
    right_width = WIDTH - MARGIN - right_x
    subtitle = (
        "architecture.md §10 and output/schema.py. One save protocol and one metadata "
        "schema behind MemoryOutput and HDF5Output; the datasets below are the ones "
        "HDF5Output creates, in the order it creates them."
    )
    top = header_height(subtitle) + 26
    gap = 12.0
    entries = [
        (group, node_height(group.lines) if group.lines else node_height(("x",)))
        for group in OUTPUT_GROUPS
    ]
    file_height = (
        2 * PAD
        + TITLE_SIZE
        + 10
        + sum(tall for _, tall in entries)
        + (len(entries) - 1) * gap
    )

    resume_height = node_height(RESUME_NOTE)
    reader_rows = tuple((module, " · ".join(names)) for module, names in OUTPUT_READERS)
    readers_height = _table_height(reader_rows, right_width, 128)
    footer = (
        "Fields are stored in the canonical layout (n_z, n_modes, n_pol, n_omega) in "
        "the optics sign convention (ADR-017): the output layer conjugates on save "
        "and on resume, and processing accessors squeeze trivial axes and accept "
        'convention="fft".'
    )
    height = (
        top
        + max(file_height, resume_height + 40 + readers_height)
        + 18
        + footer_height(footer)
        + MARGIN
    )

    drawing = _canvas(height)
    _header(drawing, "A saved run, and what reads it back", subtitle)
    _text(drawing, "one file", MARGIN, top - 12, size=SMALL_SIZE, klass="bold dim")
    _text(
        drawing, "what it is for", right_x, top - 12, size=SMALL_SIZE, klass="bold dim"
    )

    _box(drawing, MARGIN, top, FILE_WIDTH, file_height, klass="plain")
    _text(
        drawing,
        "run.h5   ·   output.hdf5.HDF5Output",
        MARGIN + PAD,
        top + PAD + TITLE_SIZE - 3,
        size=TITLE_SIZE,
        klass="bold ink-d",
    )
    y = top + PAD + TITLE_SIZE + 10
    for group, tall in entries:
        _box(drawing, MARGIN + PAD, y, FILE_WIDTH - 2 * PAD, tall, klass="fill-d")
        _text(
            drawing,
            group.name,
            MARGIN + 2 * PAD,
            y + PAD + TITLE_SIZE - 3,
            size=TITLE_SIZE,
            klass="bold ink-d",
        )
        _text(
            drawing,
            group.shape,
            MARGIN + 2 * PAD + 150,
            y + PAD + TITLE_SIZE - 3,
            size=SMALL_SIZE,
            klass="dim",
        )
        for index, line in enumerate(group.lines):
            _text(
                drawing,
                line,
                MARGIN + 2 * PAD,
                y + PAD + TITLE_SIZE + (index + 1) * LINE - 3,
                size=SMALL_SIZE,
                klass="dim",
            )
        y += tall + gap

    _node(
        drawing,
        Slot(right_x, top, right_width),
        f"{RESUME_KEY}  ·  the resume key",
        RESUME_NOTE,
        Paint("fill-d", "ink-d"),
    )
    reader_top = top + resume_height + 40
    _table(
        drawing,
        Slot(right_x, reader_top, right_width),
        "what reads it back",
        reader_rows,
        128,
        Paint("fill-x", "ink-x"),
    )
    _arrow(
        drawing,
        [(MARGIN + FILE_WIDTH + 6, reader_top + 30), (right_x - 6, reader_top + 30)],
    )
    _footer(drawing, footer, height - MARGIN)
    return drawing


FIGURES: dict[str, Callable[[], drawsvg.Drawing]] = {
    "layers": layers,
    "one_run": one_run,
    "kernel_contract": kernel_contract,
    "one_stage": one_stage,
    "backends": backends,
    "output": output_side,
}
"""File stem to the function that draws it; the page shows exactly these."""


HERE = Path(__file__).resolve().parent
"""Where the SVGs are written, which is beside this script and beside the page."""


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
    return FIGURES[name]().as_svg()


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

    This is what ``poe figures`` runs. The files it writes are committed;
    ``tests/test_structure_figures.py`` fails if they are not what this draws.
    """
    for path in write():
        # A hand-run script says what it changed: the files are committed, so the
        # next thing the developer does is read `git status`.
        print(f"wrote {path.relative_to(Path.cwd())}")  # noqa: T201


if __name__ == "__main__":
    main()

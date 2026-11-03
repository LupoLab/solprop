# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Colours, grids, units and annotations shared by the standard figures.

The small plotting helpers, and the display units every figure in
this package uses: z in centimetres, time in femtoseconds, wavelength in nanometres,
frequency in petahertz, angular frequency in radians per femtosecond, energy in
microjoules and power auto-scaled between kilowatts and petawatts. Those are
conventions of the *figure*, not of the package: everything
:mod:`solprop.processing` returns is SI (``conventions.md`` section 1), and the
conversion happens here, once. A run uniform in y (a hollow planar guide) reports its
powers and energies per metre of y, and :func:`per_width` is the suffix its labels
carry.
"""

import math
from typing import Any, Literal

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.output.schema import Output, PlaneStore
from solprop.processing.axes import (
    SPECTRAL_AXES,
    SpectralAxis,
    _check_choice,
    transverse_units,
)
from solprop.processing.time import half_max_range

__all__ = [
    "CORNERS",
    "ENERGY_TO_UJ",
    "POWER_UNITS",
    "T_TO_FS",
    "Z_TO_CM",
    "Corner",
    "cmap_colours",
    "cmap_white",
    "corner_text",
    "field_modes",
    "fwhm_arrows",
    "mode_labels",
    "per_width",
    "power_unit",
    "spectral_display",
    "subplot_grid",
    "wavelength_to_axis",
]

Z_TO_CM = 1e2
"""Metres to centimetres, the distance unit of every propagation axis."""

T_TO_FS = 1e15
"""Seconds to femtoseconds, the unit of every time axis."""

ENERGY_TO_UJ = 1e6
"""Joules to microjoules, the unit of every energy axis."""

POWER_UNITS = ("kW", "MW", "GW", "TW", "PW")
"""The power units :func:`power_unit` chooses between."""

Corner = Literal["ul", "ur", "ll", "lr"]
"""Upper or lower, left or right."""

CORNERS: tuple[Corner, ...] = ("ul", "ur", "ll", "lr")
"""The four corners of an axis."""


def cmap_white(
    name: str = "viridis", *, stops: int = 8, levels: int = 4096
) -> Colormap:
    """Return a colour map whose lowest colour has been replaced by white.

    A propagation map is mostly empty, and a colour map that starts at white puts that
    emptiness where the eye expects it. The stops are interpolated linearly, which is
    what
    :meth:`~matplotlib.colors.LinearSegmentedColormap.from_list` does and is
    indistinguishable at the number of levels used here.

    Parameters
    ----------
    name : str
        Name of the matplotlib colour map to start from.
    stops : int
        How many colours to sample from it before replacing the first.
    levels : int
        How many levels the returned map holds.

    Returns
    -------
    matplotlib.colors.Colormap
        The modified colour map.

    Examples
    --------
    >>> from solprop.plotting.style import cmap_white
    >>> [float(channel) for channel in cmap_white()(0.0)]
    [1.0, 1.0, 1.0, 1.0]
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

    sampled = plt.get_cmap(name)(np.linspace(0.0, 1.0, stops))
    sampled[0] = (1.0, 1.0, 1.0, 1.0)
    return LinearSegmentedColormap.from_list(f"{name}_white", sampled, N=levels)


def cmap_colours(
    count: int, name: str = "viridis", *, low: float = 0.0, high: float = 0.8
) -> list[tuple[float, float, float, float]]:
    """Return evenly spaced colours from a colour map, for a family of lines.

    The default stops at 0.8 because the top of ``viridis`` is a pale yellow that
    disappears against a white page.

    Parameters
    ----------
    count : int
        How many colours to return, at least one.
    name : str
        Name of the matplotlib colour map.
    low : float
        Where in the map to start, in ``[0, 1]``.
    high : float
        Where to stop, in ``[0, 1]``.

    Returns
    -------
    list of tuple
        ``count`` RGBA tuples.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If fewer than one colour is asked for.

    Examples
    --------
    >>> from solprop.plotting.style import cmap_colours
    >>> len(cmap_colours(3))
    3
    """
    if count < 1:
        msg = f"at least one colour is needed, got {count}"
        raise ProcessingError(msg)
    positions = np.linspace(low, high, count)
    import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

    return [tuple(colour) for colour in plt.get_cmap(name)(positions)]


def subplot_grid(
    count: int, *, portrait: bool = True, width: float = 4.0, height: float = 2.5
) -> tuple[Figure, list[Axes]]:
    """Return a figure with a near-square grid of axes and the axes in row order.

    Surplus axes are removed rather than left blank, so a grid of seven panels is a
    figure of seven panels. The grid is filled in reading order, which is what
    :func:`matplotlib.pyplot.subplots` returns.

    Parameters
    ----------
    count : int
        How many axes are wanted, at least one.
    portrait : bool
        Whether to make the grid taller than it is wide.
    width : float
        Width of each column, inches.
    height : float
        Height of each row, inches.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure.
    axes : list of matplotlib.axes.Axes
        Exactly ``count`` axes, in reading order.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If fewer than one axis is asked for.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.style import subplot_grid
    >>> figure, axes = subplot_grid(5)
    >>> len(axes), figure.get_size_inches().tolist()
    (5, [8.0, 7.5])
    >>> plt.close(figure)
    """
    if count < 1:
        msg = f"at least one axis is needed, got {count}"
        raise ProcessingError(msg)
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    if portrait:
        rows, columns = columns, rows
    import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

    figure, grid = plt.subplots(
        rows, columns, figsize=(columns * width, rows * height), squeeze=False
    )
    axes = list(grid.ravel())
    for surplus in axes[count:]:
        surplus.remove()
    return figure, axes[:count]


def per_width(output: Output) -> str:
    """Return the suffix a power or energy label takes on a run uniform in y.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.

    Returns
    -------
    str
        ``"/m"`` when the run's powers and energies are per metre of y
        (:func:`solprop.processing.axes.transverse_units`), and ``""`` otherwise.

    Examples
    --------
    >>> from solprop.plotting.style import per_width
    >>> from solprop.processing._examples import example_output
    >>> per_width(example_output())
    ''
    """
    return "/m" if transverse_units(output).measure == "per_unit_width" else ""


def power_unit(power: Float[np.ndarray, "*shape"]) -> tuple[float, str]:
    """Return the factor and label that put a power in a readable unit.

    The unit is chosen from the largest value, in steps of a thousand from kilowatts to
    petawatts, so that a trace reads between one and a thousand of whatever it says on
    the axis.

    Parameters
    ----------
    power : numpy.ndarray
        Powers, W; only the maximum matters.

    Returns
    -------
    factor : float
        What to multiply a power in watts by.
    label : str
        The unit, one of :data:`POWER_UNITS`.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the array is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.plotting.style import power_unit
    >>> power_unit(np.array([2.5e9]))
    (1e-09, 'GW')
    """
    if power.size == 0:
        msg = "an empty array has no power scale"
        raise ProcessingError(msg)
    peak = float(np.max(power))
    # A plane with no light has no scale of its own; kilowatts is as good as any, and
    # taking the logarithm of zero would raise here rather than pick one.
    order = 1 if peak <= 0.0 else int(np.clip(math.floor(math.log10(peak) / 3), 1, 5))
    return 10.0 ** (-3 * order), POWER_UNITS[order - 1]


def spectral_display(axis: SpectralAxis) -> tuple[float, str]:
    """Return the display factor and axis label of a spectral abscissa.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa, in SI units as :mod:`solprop.processing` returns it.

    Returns
    -------
    factor : float
        What to multiply the SI abscissa by: nm, PHz or rad/fs.
    label : str
        The axis label, unit included.

    Examples
    --------
    >>> from solprop.plotting.style import spectral_display
    >>> spectral_display("wavelength")
    (1000000000.0, 'Wavelength (nm)')
    """
    _check_choice(axis, SPECTRAL_AXES, "axis")
    if axis == "wavelength":
        return 1e9, "Wavelength (nm)"
    if axis == "f":
        return 1e-15, "Frequency (PHz)"
    return 1e-15, "Angular frequency (rad/fs)"


def wavelength_to_axis(
    axis: SpectralAxis, wavelength_limits: tuple[float, float]
) -> tuple[float, float]:
    """Return a wavelength range on a spectral abscissa, in SI units.

    For a plot asked for on a frequency axis by someone who thinks in wavelength: the
    result is what the plots' ``spec_range`` takes.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa.
    wavelength_limits : tuple of float
        The range, m, in either order.

    Returns
    -------
    tuple of float
        The lower and upper limits in m, Hz or rad/s.

    Examples
    --------
    >>> from solprop.plotting.style import wavelength_to_axis
    >>> low, high = wavelength_to_axis("f", (400e-9, 2e-6))
    >>> f"{low:.4g} Hz, {high:.4g} Hz"
    '1.499e+14 Hz, 7.495e+14 Hz'
    """
    _check_choice(axis, SPECTRAL_AXES, "axis")
    low, high = min(wavelength_limits), max(wavelength_limits)
    if axis == "wavelength":
        return low, high
    turn = 2.0 * np.pi * C if axis == "omega" else C
    # Wavelength and frequency run in opposite directions, so the limits swap.
    return float(turn / high), float(turn / low)


def _spec_range(
    axis: SpectralAxis,
    spec_range: tuple[float, float] | None,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Return the spectral range of a plot, in SI units of its abscissa, ascending.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa.
    spec_range : tuple of float or None
        The range asked for, in m, Hz or rad/s as ``axis`` says, in either order.
    default : tuple of float
        The wavelength range, m, to show when none was asked for.

    Returns
    -------
    tuple of float
        The lower and upper limits in the units of ``axis``.
    """
    if spec_range is None:
        return wavelength_to_axis(axis, default)
    return float(min(spec_range)), float(max(spec_range))


def _display_limits(
    axis: SpectralAxis, spec_range: tuple[float, float]
) -> tuple[float, float]:
    """Return SI limits on a spectral abscissa in the display units of the figure.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa.
    spec_range : tuple of float
        The ascending limits, in m, Hz or rad/s.

    Returns
    -------
    tuple of float
        The limits in the units of :func:`spectral_display`.
    """
    factor, _ = spectral_display(axis)
    return float(factor * spec_range[0]), float(factor * spec_range[1])


def corner_text(
    ax: Axes, text: str, corner: Corner = "ul", *, pad: float = 0.02, **kwargs: Any
) -> None:
    """Place a label in one corner of an axis, in axis coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to write on.
    text : str
        The label.
    corner : {'ul', 'ur', 'll', 'lr'}
        Which corner: upper or lower, left or right.
    pad : float
        Distance from the corner, as a fraction of the axis.
    **kwargs : Any
        Passed to :meth:`matplotlib.axes.Axes.text`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.style import corner_text
    >>> figure, ax = plt.subplots()
    >>> corner_text(ax, "a)")
    >>> plt.close(figure)
    """
    _check_choice(corner, CORNERS, "corner")
    upper, left = corner[0] == "u", corner[1] == "l"
    ax.text(
        pad if left else 1.0 - pad,
        1.0 - pad if upper else pad,
        text,
        transform=ax.transAxes,
        ha="left" if left else "right",
        va="top" if upper else "bottom",
        **kwargs,
    )


def fwhm_arrows(
    ax: Axes,
    x: Float[np.ndarray, " n"],
    y: Float[np.ndarray, " n"],
    *,
    unit: str = "fs",
    outer: bool = False,
    colour: str = "k",
) -> float:
    """Mark the full width at half maximum of a trace with two arrows and a label.

    The abscissa is used as it stands, so pass it in the unit the label says: this draws
    on the axis, it does not convert. Nothing is drawn when the width is undefined.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    x : numpy.ndarray
        Abscissa, shape ``(n,)``, in the unit of ``unit``.
    y : numpy.ndarray
        The trace, shape ``(n,)``.
    unit : str
        Unit of ``x``, for the label.
    outer : bool
        Whether to mark the outermost crossings rather than the central feature.
    colour : str
        Colour of the arrows and the label.

    Returns
    -------
    float
        The width, in the unit of ``x``, or ``nan`` when there is none.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from solprop.plotting.style import fwhm_arrows
    >>> figure, ax = plt.subplots()
    >>> t = np.linspace(-50.0, 50.0, 401)
    >>> round(fwhm_arrows(ax, t, np.exp(-((t / 10) ** 2))), 2)
    16.65
    >>> plt.close(figure)
    """
    low, high = half_max_range(x, y, outer=outer)
    width = high - low
    if not math.isfinite(width):
        return width
    level = 0.5 * float(np.max(y))
    arrow = {"arrowstyle": "->", "color": colour}
    for tip, tail in ((low, low - width), (high, high + width)):
        ax.annotate("", xy=(tip, level), xytext=(tail, level), arrowprops=arrow)
    ax.text(
        0.5 * (low + high),
        1.05 * level,
        f"{width:.2f} {unit}",
        ha="center",
        va="bottom",
        color=colour,
    )
    return width


def field_modes(output: Output) -> int:
    """Return how many modes, or transverse columns, a run's field carries.

    Read from the shape the run recorded, so that a run that kept no field (SEP 22)
    still labels its statistics' columns.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding at least one plane.

    Returns
    -------
    int
        The field's leading axis, ``n_modes``.

    Examples
    --------
    >>> from solprop.plotting.style import field_modes
    >>> from solprop.processing._examples import example_output
    >>> field_modes(example_output(n_modes=2))
    2
    """
    if isinstance(output, PlaneStore) and output.field_shape:
        return int(output.field_shape[0])
    return int(output["field"].shape[1])


def mode_labels(output: Output, n_modes: int) -> list[str]:
    """Return a display name for each propagated mode, from the run's own metadata.

    A mode records the class it is and the arguments it was built from, so a capillary
    mode gives ``HE11``; anything else, and a mode-averaged run, gives ``mode 1``. When
    the set contains modes at more than one polarisation angle the name carries it, as
    ``(x)`` and ``(y)`` for the two axes and as a fraction of pi otherwise, read from
    the recorded mode arguments rather than from the printed form of the transform.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    n_modes : int
        How many modes the saved field carries.

    Returns
    -------
    list of str
        One label per mode.

    Examples
    --------
    >>> from solprop.plotting.style import mode_labels
    >>> from solprop.processing._examples import example_output
    >>> mode_labels(example_output(n_modes=2), 2)
    ['mode 1', 'mode 2']
    """
    metadata = output.metadata
    records = () if metadata is None else metadata.modes
    angles = [record.configuration.get("phi") for record in records[:n_modes]]
    show_angle = any(isinstance(one, float) and one != 0.0 for one in angles)
    labels = []
    for index in range(n_modes):
        name = f"mode {index + 1}"
        if index < len(records):
            configuration = records[index].configuration
            kind, radial, azimuthal = (
                configuration.get("kind"),
                configuration.get("n"),
                configuration.get("m"),
            )
            if (
                isinstance(kind, str)
                and isinstance(radial, int)
                and isinstance(azimuthal, int)
            ):
                name = f"{kind}{radial}{azimuthal}"
            if show_angle:
                name = f"{name} {_angle_label(angles[index])}"
        labels.append(name)
    return labels


def _angle_label(phi: object) -> str:
    """Return the polarisation suffix of a mode at a given angle.

    Parameters
    ----------
    phi : object
        The angle as the metadata records it, rad, or anything else.

    Returns
    -------
    str
        ``'(x)'``, ``'(y)'`` or the angle as a fraction of pi.
    """
    if not isinstance(phi, float):
        return "(?)"
    if phi == 0.0:
        return "(x)"
    if phi == 0.5 * math.pi:
        return "(y)"
    return f"({phi / math.pi:.2f}pi)"


def _figure_of(ax: Axes) -> Figure:
    """Return the figure an axis belongs to, narrowed to a root figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis.

    Returns
    -------
    matplotlib.figure.Figure
        Its root figure, so that a function drawing into a caller's sub-figure still
        returns something a caller can save.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the axis belongs to no figure.
    """
    figure = ax.get_figure(root=True)
    if figure is None:  # pragma: no cover - an axis always belongs to a figure
        msg = "the axis to draw on does not belong to a figure"
        raise ProcessingError(msg)
    return figure

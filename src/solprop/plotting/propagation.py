# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The standard figures of a propagation: the maps, the slices and the energy.

Drawn with matplotlib from whatever :mod:`solprop.processing` returns. Every function
takes an :class:`~solprop.output.schema.Output`, returns figures and never calls
:func:`~matplotlib.pyplot.show`; the slice plots accept an ``ax`` to draw into, so a
figure of several panels is assembled by the caller rather than by a keyword here.

The display units are :mod:`solprop.plotting.style`'s: z in centimetres, time in
femtoseconds, wavelength in nanometres, frequency in petahertz, energy in microjoules
and power auto-scaled.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from solprop.optics import TransferFunction
from solprop.output.schema import Output
from solprop.plotting.style import (
    ENERGY_TO_UJ,
    T_TO_FS,
    Z_TO_CM,
    _display_limits,
    _figure_of,
    _spec_range,
    cmap_colours,
    field_modes,
    mode_labels,
    per_width,
    power_unit,
    spectral_display,
)
from solprop.processing import spectra, time
from solprop.processing.axes import ModeSelection, SpectralAxis, _check_choice
from solprop.processing.fields import (
    NoiseFloor,
    get_field,
    get_power,
    get_spectral_density,
    nearest_plane,
)
from solprop.processing.time import energy as pulse_energy

__all__ = [
    "DEFAULT_SLICE_WAVELENGTH_RANGE",
    "DEFAULT_TRANGE",
    "DEFAULT_WAVELENGTH_RANGE",
    "SLICE_QUANTITIES",
    "SliceQuantity",
    "energy",
    "prop_2d",
    "spec_1d",
    "time_1d",
]

DEFAULT_WAVELENGTH_RANGE = (150e-9, 2e-6)
"""Wavelength range of a propagation map when no ``spec_range`` is given, m."""

DEFAULT_SLICE_WAVELENGTH_RANGE = (150e-9, 1.2e-6)
"""Wavelength range of a spectral slice when no ``spec_range`` is given, m."""

DEFAULT_TRANGE = (-50e-15, 50e-15)
"""Time range of a propagation map, s."""

SliceQuantity = Literal["power", "field", "field_squared"]
"""What a time-domain slice draws: the power, the real field, or its square."""

SLICE_QUANTITIES: tuple[SliceQuantity, ...] = ("power", "field", "field_squared")
"""The three time-domain quantities: power, field and squared field."""

# One decade below the colour floor, so that an empty bin plots at the bottom of the
# scale instead of raising a divide-by-zero on its way to minus infinity.
_DB_FLOOR_MARGIN = 10.0


def prop_2d(  # noqa: PLR0913 - the two axes, the two ranges and the colour scale
    output: Output,
    axis: SpectralAxis = "f",
    *,
    spec_range: tuple[float, float] | None = None,
    trange: tuple[float, float] = DEFAULT_TRANGE,
    db_min: float = -60.0,
    resolution: float | None = None,
    modes: ModeSelection = None,
    oversampling: int = 4,
    bandpass: spectra.Bandpass | None = None,
    cmap: str | Colormap | None = None,
    noise_floor: NoiseFloor = "keep",
) -> list[Figure]:
    """Return the false-colour propagation maps of a run, one figure per mode.

    Each figure carries the spectral energy density in decibels beside the instantaneous
    power, both against distance. A multimode run gets one figure per mode and a last
    one for the modal sum; a single-mode run gets one figure.

    The decibel scale is shared: every map is normalised by the peak of the modal sum,
    so the modes are comparable and none goes above 0 dB; normalising by the peak of
    the per-mode maps and then summing would take the summed map off the top of the
    scale. The bandpass reaches the temporal panel only: a window drawn on the spectrum
    it is displayed beside would be circular.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    axis : {'f', 'wavelength', 'omega'}
        Spectral abscissa of the left panel.
    spec_range : tuple of float or None
        Spectral limits in the units of ``axis``: m, Hz or rad/s. By default
        :data:`DEFAULT_WAVELENGTH_RANGE` on that axis;
        :func:`~solprop.plotting.style.wavelength_to_axis` converts a wavelength range.
    trange : tuple of float
        Time limits, s.
    db_min : float
        Bottom of the colour scale, dB below the peak.
    resolution : float or None
        Instrument resolution of the spectral panel, in the units of ``axis``.
    modes : int, sequence of int, 'sum' or None
        Which modes to draw.
    oversampling : int
        Time-domain resampling factor of the temporal panel.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of a window applied to the temporal panel.
    cmap : str, matplotlib.colors.Colormap or None
        Colour map of both panels.
    noise_floor : {'keep', 'subtract'}
        Whether the spectral panel keeps or removes the input noise's seed
        (:func:`~solprop.processing.fields.get_spectral_density`). A value at or below
        zero, which the subtraction can leave, is drawn at the bottom of the colour
        scale, as everything below ``db_min`` is.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per drawn mode, and one for the modal sum when there is more than
        one.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.propagation import prop_2d
    >>> from solprop.processing._examples import example_output
    >>> figures = prop_2d(example_output())
    >>> len(figures), len(figures[0].axes)
    (1, 4)
    >>> figures = prop_2d(example_output(), "f", spec_range=(0.0, 2e15))
    >>> figures[0].axes[0].get_xlim() == (0.0, 2.0)
    True
    >>> plt.close("all")
    """
    limits = _spec_range(axis, spec_range, DEFAULT_WAVELENGTH_RANGE)
    x, density = get_spectral_density(
        output,
        axis,
        modes=modes,
        spec_range=limits,
        resolution=resolution,
        noise_floor=noise_floor,
    )
    t, power = get_power(
        output, modes=modes, oversampling=oversampling, bandpass=bandpass
    )
    z = output["z"] * Z_TO_CM
    density, power = _with_mode_axis(density), _with_mode_axis(power)
    labels = _drawn_labels(output, density.shape[1], modes)
    if density.shape[1] > 1:
        density = np.concatenate([density, density.sum(axis=1, keepdims=True)], axis=1)
        power = np.concatenate([power, power.sum(axis=1, keepdims=True)], axis=1)
        labels = [*labels, "sum"]
    reference = float(np.max(density[:, -1]))
    factor, label = spectral_display(axis)
    figures = []
    for index, name in enumerate(labels):
        import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

        figure, (spectral, temporal) = plt.subplots(1, 2, figsize=(12, 4))
        _spectral_map(
            spectral,
            x * factor,
            z,
            density[:, index],
            reference=reference,
            db_min=db_min,
            cmap=cmap,
        )
        spectral.set_xlabel(label)
        spectral.set_xlim(*_display_limits(axis, limits))
        _temporal_map(
            temporal,
            t * T_TO_FS,
            z,
            power[:, index],
            cmap=cmap,
            width=per_width(output),
        )
        temporal.set_xlim(trange[0] * T_TO_FS, trange[1] * T_TO_FS)
        figure.suptitle(f"Propagation ({name})")
        figure.tight_layout()
        figures.append(figure)
    return figures


def time_1d(  # noqa: PLR0913 - the planes, the quantity, the window and the axis
    output: Output,
    z: float | Sequence[float] | None = None,
    *,
    quantity: SliceQuantity = "power",
    modes: ModeSelection = None,
    oversampling: int = 4,
    trange: tuple[float, float] = DEFAULT_TRANGE,
    bandpass: spectra.Bandpass | None = None,
    widths: bool = True,
    ax: Axes | None = None,
    transfer: TransferFunction | None = None,
    transform_limit: bool = False,
) -> Figure:
    """Return time-domain slices of a run at one or more positions.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    z : float, sequence of float or None
        Positions to draw, m; the end of the run by default.
    quantity : {'power', 'field', 'field_squared'}
        What to draw: the instantaneous power, the real field, or its square.
    modes : int, sequence of int, 'sum' or None
        Which modes to draw; only a power may be summed over modes.
    oversampling : int
        Time-domain resampling factor.
    trange : tuple of float
        Time limits, s.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of a window applied before the transform.
    widths : bool
        Whether to append each trace's width at half maximum to its legend entry.
    ax : matplotlib.axes.Axes or None
        Axis to draw into; a new figure is made when none is given.
    transfer : TransferFunction or None
        A linear optical element of :mod:`solprop.optics` applied before the transform:
        draw the pulse after a compressor.
    transform_limit : bool
        Whether to draw the transform-limited pulse of each spectrum.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the traces were drawn on.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.propagation import time_1d
    >>> from solprop.processing._examples import example_output
    >>> figure = time_1d(example_output(), [0.0, 0.1])
    >>> len(figure.axes[0].lines)
    2
    >>> plt.close(figure)
    """
    _check_choice(quantity, SLICE_QUANTITIES, "quantity")
    ax = _axis(ax, (8.5, 5.0))
    planes = _plane_indices(output, z)
    traces = []
    for index in planes:
        if quantity == "power":
            t, values = get_power(
                output,
                plane=index,
                modes=modes,
                oversampling=oversampling,
                bandpass=bandpass,
                transfer=transfer,
                transform_limit=transform_limit,
            )
        else:
            t, field = get_field(
                output,
                kind="real",
                plane=index,
                modes=modes,
                oversampling=oversampling,
                bandpass=bandpass,
                transfer=transfer,
                transform_limit=transform_limit,
            )
            values = field**2 if quantity == "field_squared" else field
        traces.append((index, t, _with_mode_axis(values, leading=False)))
    drawn = np.abs(np.concatenate([one[2] for one in traces]))
    # The unit always comes from a power: a field is the square root of one, so its
    # scale and its label are the square roots of the power's.
    scale, unit = power_unit(drawn**2 if quantity == "field" else drawn)
    unit += per_width(output)
    if quantity == "field":
        wrapped = f"({unit})" if "/" in unit else unit
        scale, unit = float(np.sqrt(scale)), f"{wrapped}$^{{1/2}}$"
    labels = _drawn_labels(output, traces[0][2].shape[0], modes)
    colours = cmap_colours(len(traces))
    for (index, t, values), colour in zip(traces, colours, strict=True):
        for mode, name in enumerate(labels):
            trace = values[mode] * scale
            entry = _legend_entry(float(output["z"][index]), name, len(labels))
            if widths:
                entry += f" [{time.fwhm(t * T_TO_FS, np.abs(trace)):.2f} fs]"
            ax.plot(t * T_TO_FS, trace, color=colour, dashes=_dashes(mode), label=entry)
    ax.set_xlim(trange[0] * T_TO_FS, trange[1] * T_TO_FS)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(("Field" if quantity == "field" else "Power") + f" ({unit})")
    if quantity != "field":
        ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    return _figure_of(ax)


def spec_1d(  # noqa: PLR0913 - the planes, the abscissa, the range and the scale
    output: Output,
    z: float | Sequence[float] | None = None,
    axis: SpectralAxis = "wavelength",
    *,
    modes: ModeSelection = None,
    spec_range: tuple[float, float] | None = None,
    log: bool = True,
    log_min: float = 1e-6,
    resolution: float | None = None,
    ax: Axes | None = None,
    noise_floor: NoiseFloor = "keep",
) -> Figure:
    """Return spectral slices of a run at one or more positions.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    z : float, sequence of float or None
        Positions to draw, m; the end of the run by default.
    axis : {'wavelength', 'f', 'omega'}
        Spectral abscissa.
    modes : int, sequence of int, 'sum' or None
        Which modes to draw.
    spec_range : tuple of float or None
        Spectral limits in the units of ``axis``: m, Hz or rad/s. By default
        :data:`DEFAULT_SLICE_WAVELENGTH_RANGE` on that axis;
        :func:`~solprop.plotting.style.wavelength_to_axis` converts a wavelength range.
    log : bool
        Whether to use a logarithmic ordinate.
    log_min : float
        Bottom of a logarithmic ordinate, as a fraction of the largest value drawn.
    resolution : float or None
        Instrument resolution, in the units of ``axis``.
    ax : matplotlib.axes.Axes or None
        Axis to draw into; a new figure is made when none is given.
    noise_floor : {'keep', 'subtract'}
        Whether the slices keep or remove the input noise's seed
        (:func:`~solprop.processing.fields.get_spectral_density`). On a logarithmic
        ordinate a value at or below zero, which the subtraction can leave, is masked
        rather than drawn.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the traces were drawn on.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.propagation import spec_1d
    >>> from solprop.processing._examples import example_output
    >>> figure = spec_1d(example_output(), 0.0)
    >>> figure.axes[0].get_xlabel()
    'Wavelength (nm)'
    >>> plt.close(figure)
    """
    ax = _axis(ax, (8.5, 5.0))
    planes = _plane_indices(output, z)
    limits = _spec_range(axis, spec_range, DEFAULT_SLICE_WAVELENGTH_RANGE)
    factor, label = spectral_display(axis)
    traces = [
        get_spectral_density(
            output,
            axis,
            plane=index,
            modes=modes,
            spec_range=limits,
            resolution=resolution,
            noise_floor=noise_floor,
        )
        for index in planes
    ]
    n_drawn = _with_mode_axis(traces[0][1], leading=False).shape[0]
    labels = _drawn_labels(output, n_drawn, modes)
    colours = cmap_colours(len(traces))
    peak = max(float(np.max(density)) for _, density in traces)
    for index, (x, density), colour in zip(planes, traces, colours, strict=True):
        values = _with_mode_axis(density, leading=False)
        if log:
            # A logarithmic axis has no place for a value at or below zero, which a
            # floor-subtracted density can hold; a gap says so where a clipped line
            # would draw a false dip to the axis.
            values = np.where(values > 0.0, values, np.nan)
        for mode, name in enumerate(labels):
            ax.plot(
                x * factor,
                values[mode],
                color=colour,
                dashes=_dashes(mode),
                label=_legend_entry(float(output["z"][index]), name, len(labels)),
            )
    if log:
        ax.set_yscale("log")
        ax.set_ylim(log_min * peak, 3.0 * peak)
    ax.set_xlim(*_display_limits(axis, limits))
    ax.set_xlabel(label)
    ax.set_ylabel(_density_label(axis, per_width(output)))
    ax.legend(frameon=False)
    return _figure_of(ax)


def energy(
    output: Output,
    *,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Return the energy along the propagation, with a conversion-efficiency axis.

    The right-hand axis is the same data as a percentage of the total energy of every
    mode at the first saved plane, before any bandpass, which is what a conversion
    efficiency means. Its limits are set from the left axis analytically rather than by
    drawing an invisible line for matplotlib to autoscale.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    modes : int, sequence of int, 'sum' or None
        Which modes to draw.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of a window applied before the measurement, which is how the energy in one
        spectral region is followed along z.
    ax : matplotlib.axes.Axes or None
        Axis to draw into; a new figure is made when none is given.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the traces were drawn on.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.propagation import energy
    >>> from solprop.processing._examples import example_output
    >>> figure = energy(example_output())
    >>> figure.axes[1].get_ylabel()
    'Conversion efficiency (%)'
    >>> plt.close(figure)
    """
    ax = _axis(ax, (7.0, 5.0))
    values = pulse_energy(output, modes=modes, bandpass=bandpass)
    if values.ndim == 1:
        # One value per plane, because one mode was asked for or they were summed.
        values = values[:, np.newaxis]
    reference = float(np.sum(pulse_energy(output, plane=0, modes="sum")))
    z = output["z"] * Z_TO_CM
    labels = _drawn_labels(output, values.shape[1], modes)
    for index, name in enumerate(labels):
        ax.plot(z, values[:, index] * ENERGY_TO_UJ, label=name)
    ax.set_xlim(float(z[0]), float(z[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(f"Energy (uJ{per_width(output)})")
    if len(labels) > 1:
        ax.legend(frameon=False)
    efficiency = ax.twinx()
    low, high = ax.get_ylim()
    per_cent = 100.0 / (ENERGY_TO_UJ * reference)
    efficiency.set_ylim(per_cent * low, per_cent * high)
    efficiency.set_ylabel("Conversion efficiency (%)")
    return _figure_of(ax)


# --- private helpers -----------------------------------------------------------------


def _axis(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return the axis to draw on, making a figure when none was given.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        The caller's axis, or ``None``.
    figsize : tuple of float
        Size of the new figure, inches.

    Returns
    -------
    matplotlib.axes.Axes
        The axis to draw on.
    """
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

    return plt.subplots(figsize=figsize)[1]


def _with_mode_axis(
    values: Float[np.ndarray, "*rest"], *, leading: bool = True
) -> Float[np.ndarray, "*with_modes"]:
    """Restore the mode axis an accessor squeezed away.

    Parameters
    ----------
    values : numpy.ndarray
        What an accessor returned, with or without a mode axis.
    leading : bool
        Whether a plane axis comes before the mode axis.

    Returns
    -------
    numpy.ndarray
        The same values with a mode axis, of length one when there was none.
    """
    wanted = 3 if leading else 2
    if values.ndim == wanted:
        return values
    return values[:, np.newaxis] if leading else values[np.newaxis]


def _drawn_labels(output: Output, count: int, modes: ModeSelection) -> list[str]:
    """Return the label of each drawn trace.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    count : int
        How many traces are drawn.
    modes : int, sequence of int, 'sum' or None
        The selection they came from.

    Returns
    -------
    list of str
        One label per trace: the mode's name, or ``'sum'`` for a modal sum.
    """
    if modes == "sum":
        return ["sum"]
    every = mode_labels(output, max(field_modes(output), count))
    if modes is None:
        return every[:count]
    chosen = [modes] if isinstance(modes, int) else list(modes)
    return [every[one] for one in chosen]


def _plane_indices(output: Output, z: float | Sequence[float] | None) -> list[int]:
    """Return the plane indices a slice plot draws.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    z : float, sequence of float or None
        Positions, m; ``None`` means the last saved plane.

    Returns
    -------
    list of int
        Indices into the saved planes, one per position.
    """
    if z is None:
        return [output.n_saved - 1]
    positions = [z] if isinstance(z, float | int) else list(z)
    return [nearest_plane(output, float(one)) for one in positions]


def _legend_entry(z: float, name: str, n_labels: int) -> str:
    """Return the legend entry of one trace.

    Parameters
    ----------
    z : float
        Position of the plane, m.
    name : str
        Label of the mode.
    n_labels : int
        How many modes are drawn; the mode is named only when there is more than one.

    Returns
    -------
    str
        The entry.
    """
    entry = f"{z * Z_TO_CM:.2f} cm"
    return entry if n_labels == 1 else f"{entry} ({name})"


def _dashes(index: int) -> tuple[float, ...]:
    """Return the dash pattern of a mode, so that modes differ within one colour.

    Parameters
    ----------
    index : int
        Index of the mode among those drawn.

    Returns
    -------
    tuple of float
        A dash pattern, solid for the first mode. The patterns repeat past the last
        one.
    """
    patterns: tuple[tuple[float, ...], ...] = (
        (),
        (4.0, 2.0),
        (1.0, 1.0),
        (6.0, 2.0, 1.0, 2.0),
        (3.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    return patterns[index % len(patterns)]


def _density_label(axis: SpectralAxis, width: str = "") -> str:
    """Return the ordinate label of a spectral energy density.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa it is quoted against.
    width : str
        :func:`solprop.plotting.style.per_width` of the run: ``"/m"`` makes the
        density one per metre of y as well.

    Returns
    -------
    str
        The label, with its unit.
    """
    units = {"wavelength": "J/m", "f": "J/Hz", "omega": "J/(rad/s)"}
    per_y = " per m of y" if width else ""
    return f"Spectral energy density ({units[axis]}{per_y})"


def _spectral_map(  # noqa: PLR0913 - an axis, two axes of data, and the colour scale
    ax: Axes,
    x: Float[np.ndarray, " n_x"],
    z: Float[np.ndarray, " n_z"],
    density: Float[np.ndarray, "n_z n_x"],
    *,
    reference: float,
    db_min: float,
    cmap: str | Colormap | None,
) -> None:
    """Draw one spectral density map in decibels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    x : numpy.ndarray
        Spectral abscissa in display units, shape ``(n_x,)``.
    z : numpy.ndarray
        Distance in centimetres, shape ``(n_z,)``.
    density : numpy.ndarray
        Spectral energy density, shape ``(n_z, n_x)``.
    reference : float
        The density that is 0 dB.
    db_min : float
        Bottom of the colour scale, dB.
    cmap : str, matplotlib.colors.Colormap or None
        Colour map.
    """
    floor = reference * 10.0 ** ((db_min - _DB_FLOOR_MARGIN) / 10.0)
    decibels = 10.0 * np.log10(np.maximum(density, floor) / reference)
    mesh = ax.pcolormesh(x, z, decibels, shading="auto", cmap=cmap)
    mesh.set_clim(db_min, 0.0)
    _figure_of(ax).colorbar(mesh, ax=ax).set_label("SED (dB)")
    ax.set_ylabel("Distance (cm)")


def _temporal_map(
    ax: Axes,
    t: Float[np.ndarray, " n_t"],
    z: Float[np.ndarray, " n_z"],
    power: Float[np.ndarray, "n_z n_t"],
    *,
    cmap: str | Colormap | None,
    width: str = "",
) -> None:
    """Draw one instantaneous-power map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    t : numpy.ndarray
        Time in femtoseconds, shape ``(n_t,)``.
    z : numpy.ndarray
        Distance in centimetres, shape ``(n_z,)``.
    power : numpy.ndarray
        Instantaneous power, W, shape ``(n_z, n_t)``.
    cmap : str, matplotlib.colors.Colormap or None
        Colour map.
    width : str
        ``"/m"`` for a power per metre of y (:func:`solprop.plotting.style.per_width`).
    """
    scale, unit = power_unit(power)
    mesh = ax.pcolormesh(t, z, power * scale, shading="auto", cmap=cmap)
    _figure_of(ax).colorbar(mesh, ax=ax).set_label(f"Power ({unit}{width})")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (cm)")

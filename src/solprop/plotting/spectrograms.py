# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time-frequency maps of a saved plane, through a Gaussian-gated short-time transform.

The map is :func:`solprop.processing.fields.get_spectrogram`'s, drawn. The gate width
and the number of gate positions have defaults, taken from the time range asked for;
and a multimode run must say which mode it means, rather than silently taking the
first.
"""

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from solprop.constants import C
from solprop.optics import TransferFunction
from solprop.output.schema import Output
from solprop.plotting.propagation import DEFAULT_TIME_RANGE, DEFAULT_WAVELENGTH_RANGE
from solprop.plotting.style import (
    T_TO_FS,
    _display_limits,
    _figure_of,
    _spec_range,
    spectral_display,
)
from solprop.processing.axes import ModeSelection, SpectralAxis
from solprop.processing.fields import (
    DEFAULT_GATE_POSITIONS,
    get_spectrogram,
    nearest_plane,
)

__all__ = ["DEFAULT_GATE_POSITIONS", "spectrogram"]


def spectrogram(  # noqa: PLR0913 - a plane, a mode, two axes and the colour scale
    output: Output,
    z: float | None = None,
    axis: SpectralAxis = "wavelength",
    *,
    modes: ModeSelection = None,
    time_range: tuple[float, float] = DEFAULT_TIME_RANGE,
    gate_fwhm: float | None = None,
    positions: int = DEFAULT_GATE_POSITIONS,
    spec_range: tuple[float, float] | None = None,
    log: bool = False,
    db_min: float = -40.0,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,
    transfer: TransferFunction | None = None,
) -> Figure:
    """Return the time-frequency map of one saved plane.

    The real field is gated by a Gaussian of the given full width at half maximum and
    transformed at evenly spaced positions, which trades time resolution against
    frequency resolution in the usual way: a gate of duration ``tau`` resolves
    frequencies no better than ``1/tau``.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    z : float or None
        Position of the plane, m; the end of the run by default.
    axis : {'wavelength', 'f', 'omega'}
        Spectral abscissa of the ordinate.
    modes : int, sequence of int, 'sum' or None
        Which mode to draw. A multimode run must name one: a spectrogram is of a field,
        and modal fields do not add away from a point in space.
    time_range : tuple of float
        Time range to gate over, s.
    gate_fwhm : float or None
        Width of the gate, s; a twentieth of the time range by default.
    positions : int
        How many gate positions to place across the time range.
    spec_range : tuple of float or None
        Spectral limits in the units of ``axis``: m, Hz or rad/s. By default
        :data:`~solprop.plotting.propagation.DEFAULT_WAVELENGTH_RANGE` on that axis;
        :func:`~solprop.plotting.style.wavelength_to_axis` converts a wavelength range.
    log : bool
        Whether to show decibels below the peak rather than a linear scale.
    db_min : float
        Bottom of a logarithmic colour scale, dB.
    cmap : str, matplotlib.colors.Colormap or None
        Colour map.
    ax : matplotlib.axes.Axes or None
        Axis to draw into; a new figure is made when none is given.
    transfer : TransferFunction or None
        A linear optical element of :mod:`solprop.optics` applied to the field first,
        so that the map is of the pulse after a compressor.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the map was drawn on.

    Raises
    ------
    solprop.exceptions.ModeError
        If a multimode run does not name a single mode.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.spectrograms import spectrogram
    >>> from solprop.processing._examples import example_output
    >>> figure = spectrogram(example_output(), 0.0)
    >>> figure.axes[0].get_xlabel()
    'Time (fs)'
    >>> plt.close(figure)
    """
    plane = output.n_saved - 1 if z is None else nearest_plane(output, z)
    # The map per unit frequency, drawn against the axis asked for as it stands:
    # the colour scale is arbitrary, and a figure keeps its look on every axis.
    times, frequency, intensity = get_spectrogram(
        output,
        "f",
        plane=plane,
        modes=modes,
        time_range=time_range,
        gate_fwhm=gate_fwhm,
        positions=positions,
        transfer=transfer,
    )
    values, ordinate = _ordinate(axis, frequency, intensity)
    import matplotlib.pyplot as plt  # noqa: PLC0415 - pyplot loads on first plot

    ax = plt.subplots(figsize=(8.0, 5.0))[1] if ax is None else ax
    if log:
        floor = float(np.max(values)) * 10.0 ** ((db_min - 10.0) / 10.0)
        values = 10.0 * np.log10(np.maximum(values, floor) / np.max(values))
    mesh = ax.pcolormesh(
        times * T_TO_FS, ordinate, values, shading="auto", cmap=cmap, rasterized=True
    )
    if log:
        mesh.set_clim(db_min, 0.0)
    label = "Intensity (dB)" if log else "Intensity (arb.)"
    figure = _figure_of(ax)
    figure.colorbar(mesh, ax=ax).set_label(label)
    ax.set_xlim(time_range[0] * T_TO_FS, time_range[1] * T_TO_FS)
    limits = _spec_range(axis, spec_range, DEFAULT_WAVELENGTH_RANGE)
    ax.set_ylim(*_display_limits(axis, limits))
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(spectral_display(axis)[1])
    return figure


def _ordinate(
    axis: SpectralAxis,
    frequency: Float[np.ndarray, " n_f"],
    intensity: Float[np.ndarray, "n_f n_slices"],
) -> tuple[Float[np.ndarray, "n_f n_slices"], Float[np.ndarray, " n_f"]]:
    """Return the map and its ordinate on the requested spectral abscissa.

    Parameters
    ----------
    axis : {'wavelength', 'f', 'omega'}
        The abscissa.
    frequency : numpy.ndarray
        Positive frequencies of the transform, Hz, shape ``(n_f,)``, increasing.
    intensity : numpy.ndarray
        The map on those frequencies, shape ``(n_f, n_slices)``.

    Returns
    -------
    values : numpy.ndarray
        The map, reordered so that the ordinate increases.
    ordinate : numpy.ndarray
        The abscissa in display units, increasing.
    """
    factor, _ = spectral_display(axis)
    if axis == "wavelength":
        # Wavelength decreases as frequency increases, so both are reversed.
        return intensity[::-1], factor * C / frequency[::-1]
    turn = 2.0 * np.pi if axis == "omega" else 1.0
    return intensity, factor * turn * frequency

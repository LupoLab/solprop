# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time-frequency maps of a saved plane, through a Gaussian-gated short-time transform.

:func:`spectrogram` draws :func:`solprop.processing.fields.get_spectrogram`'s map. The
gate width and the number of gate positions default from the time range asked for, and
a multimode run must say which mode it means, rather than silently taking the first.
"""

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from solprop.constants import C
from solprop.optics import TransferFunction
from solprop.output.store import Output
from solprop.plotting.propagation import _limits, _plane_time_range
from solprop.plotting.style import (
    T_TO_FS,
    _display_limits,
    _figure_of,
    spectral_display,
)
from solprop.processing.axes import ModeSelection, SpectralAxis, _plane_index
from solprop.processing.fields import (
    DEFAULT_GATE_POSITIONS,
    _check_spec_range,
    get_spectrogram,
)

__all__ = ["DEFAULT_GATE_POSITIONS", "spectrogram"]


def spectrogram(
    output: Output,
    z: float | None = None,
    axis: SpectralAxis = "wavelength",
    *,
    plane: int | None = None,
    modes: ModeSelection = None,
    time_range: tuple[float, float] | None = None,
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
    output : solprop.output.store.Output
        An opened output handler.
    z : float or None
        Position of the plane, m: the nearest saved plane; the end of the run by
        default.
    axis : {'wavelength', 'f', 'omega'}
        Spectral abscissa of the ordinate.
    plane : int or None
        The saved plane by index, negative from the end; refused beside `z`.
    modes : int, sequence of int, 'sum' or None
        Which mode to draw. A multimode run must name one: a spectrogram is of a field,
        and modal fields do not add away from a point in space.
    time_range : tuple of float or None
        Time range to gate over and show, s. By default where the plane's light is
        (:func:`~solprop.plotting.propagation.default_time_range`).
    gate_fwhm : float or None
        Width of the gate, s; a twentieth of the time range by default.
    positions : int
        How many gate positions to place across the time range.
    spec_range : tuple of float or None
        Spectral limits in the units of ``axis``: m, Hz or rad/s. By default the run's
        band (:func:`~solprop.plotting.propagation.default_spec_range`);
        :func:`~solprop.plotting.style.wavelength_to_axis` converts a wavelength range.
    log : bool
        Whether to show decibels below the peak rather than a linear scale.
    db_min : float
        Bottom of a logarithmic colour scale, dB below the peak: -40 dB by default,
        where :func:`~solprop.plotting.propagation.prop_2d` and
        :func:`~solprop.plotting.propagation.spec_1d` default to -60 dB.
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
    solprop.exceptions.ProcessingError
        If ``spec_range`` selects no sample of the spectral axis, or both `z` and
        `plane` are given.

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
    plane = _plane_index(output, plane, z)
    if time_range is None:
        time_range = _plane_time_range(output, plane, modes, transfer)
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
    if spec_range is not None:
        # The range is in SI units; the ordinate is in display units.
        _check_spec_range(ordinate / spectral_display(axis)[0], spec_range, axis)
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
    limits = _limits(output, axis, spec_range)
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

    - ``frequency``: positive frequencies of the transform, Hz, shape ``(n_f,)``,
      increasing.
    """
    factor, _ = spectral_display(axis)
    if axis == "wavelength":
        # Wavelength decreases as frequency increases, so both are reversed.
        return intensity[::-1], factor * C / frequency[::-1]
    turn = 2.0 * np.pi if axis == "omega" else 1.0
    return intensity, factor * turn * frequency

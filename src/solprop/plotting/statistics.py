# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Line plots of the statistics a run recorded at its saved planes.

Two figures: what the pulse did, and what the waveguide and the gas were doing while
it did it. Only the statistics the run actually recorded are drawn, and a figure with
nothing to put in it is not made at all rather than raising.

The catalogue of names, shapes and units is :data:`solprop.stats.CATALOGUE`; the display
units are :mod:`solprop.plotting.style`'s.
"""

from typing import Any

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from solprop.constants import C
from solprop.output.schema import Output
from solprop.plotting.style import (
    ENERGY_TO_UJ,
    T_TO_FS,
    Z_TO_CM,
    field_modes,
    mode_labels,
    per_width,
    power_unit,
    subplot_grid,
)

__all__ = ["OTHER_STATISTICS", "PULSE_STATISTICS", "stats"]

PULSE_STATISTICS = (
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "peak_intensity",
    "fwhm_t",
    "fwhm_t_outer",
    "fwhm_r",
    "omega0",
)
"""Statistics of the pulse, in the order they are drawn."""

OTHER_STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    "density",
    "pressure",
    "core_radius",
    "zdw",
    "quadrature_error",
)
"""Statistics of the waveguide, the gas and the numerics, in the order drawn."""

# Display factor and ordinate label of each statistic; a factor of None means the power
# scale is chosen from the data (kW to PW), and omega0 is converted to a wavelength.
_DISPLAY: dict[str, tuple[float | None, str]] = {
    "energy": (ENERGY_TO_UJ, "Energy (uJ)"),
    "energy_window": (ENERGY_TO_UJ, "Windowed energy (uJ)"),
    "peak_power": (None, "Peak power"),
    "peak_power_window": (None, "Windowed peak power"),
    "peak_power_total": (None, "Total peak power"),
    "peak_intensity": (1e-16, "Peak intensity (TW/cm^2)"),
    "fwhm_t": (T_TO_FS, "FWHM (fs)"),
    "fwhm_t_outer": (T_TO_FS, "Outer FWHM (fs)"),
    "fwhm_r": (1e6, "Radial FWHM (um)"),
    "omega0": (1e9, "Central wavelength (nm)"),
    "electron_density": (1e-6, "Electron density (cm^-3)"),
    "peak_ionisation_rate": (1.0, "Peak ionisation rate (1/s)"),
    "density": (1e-6, "Number density (cm^-3)"),
    "pressure": (1.0, "Pressure (bar)"),
    "core_radius": (1e6, "Core radius (um)"),
    "zdw": (1e9, "Zero-dispersion wavelength (nm)"),
    "quadrature_error": (1.0, "Quadrature error"),
}

_PER_WIDTH = ("energy", "energy_window")
"""The fixed-unit statistics that become per metre of y on a run uniform in y."""

# A panel gets a logarithmic ordinate when its largest value is more than this many
# times its smallest positive one.
_LOG_RATIO = 10.0


def stats(output: Output, **kwargs: Any) -> list[Figure]:
    """Return one figure per group of the statistics a run recorded.

    Where a statistic was recorded **between** planes as well as at them
    (:attr:`solprop.stats.StatsSpec.periodic`), the between-planes rows are drawn
    instead: they are the same numbers on a far denser grid of z, so a curve that a
    few save planes would render as a polygon is drawn as the trace it is. A run
    without that channel is drawn from its planes exactly as before.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    **kwargs : Any
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    list of matplotlib.figure.Figure
        A figure of pulse statistics and a figure of waveguide statistics, each present
        only if the run recorded at least one of its group. A run with no statistics at
        all gives an empty list and no error.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from solprop.plotting.statistics import stats
    >>> from solprop.processing._examples import example_output
    >>> figures = stats(example_output())
    >>> len(figures), len(figures[0].axes)
    (1, 5)
    >>> plt.close("all")
    """
    stored = output.stat_names
    figures = []
    for title, names in (
        ("Pulse statistics", PULSE_STATISTICS),
        ("Waveguide statistics", OTHER_STATISTICS),
    ):
        recorded = [name for name in names if name in stored]
        if not recorded:
            continue
        figure, axes = subplot_grid(len(recorded))
        figure.suptitle(title)
        for name, ax in zip(recorded, axes, strict=True):
            _panel(ax, output, name, kwargs)
        figure.tight_layout()
        figures.append(figure)
    return figures


def _trace(output: Output, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the abscissa and the values of one statistic, channel first.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    name : str
        Name of the statistic.

    Returns
    -------
    tuple of numpy.ndarray
        Distance in cm, shape ``(n,)``, and the values reshaped to ``(n, -1)``.
    """
    if name in output.step_stat_names:
        z = np.asarray(output["step_stats/z"])
        if z.size:
            values = np.asarray(output[f"step_stats/{name}"])
            return z * Z_TO_CM, values.reshape(z.size, -1)
    z = np.asarray(output["z"])
    return z * Z_TO_CM, np.asarray(output[f"stats/{name}"]).reshape(z.size, -1)


def _panel(
    ax: Axes,
    output: Output,
    name: str,
    kwargs: dict[str, Any],
) -> None:
    """Draw one statistic against distance, from the channel where there is one.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    output : solprop.output.schema.Output
        An opened output handler.
    name : str
        Name of the statistic.
    kwargs : dict
        Passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    z, values = _trace(output, name)
    factor, label = _DISPLAY[name]
    if name == "omega0":
        # The centroid is recorded in rad/s and read as the wavelength it belongs to.
        values = 2.0 * np.pi * C / values
    width = per_width(output)
    if factor is None:
        factor, unit = power_unit(values)
        label = f"{label} ({unit}{width})"
    elif width and name in _PER_WIDTH:
        # A run uniform in y reports energy per metre of y.
        label = label.replace("(uJ)", f"(uJ{width})")
    values = values * factor
    for column, entry in enumerate(_labels(output, values.shape[1])):
        ax.plot(z, values[:, column], label=entry, **kwargs)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(label)
    if _should_log(values):
        ax.set_yscale("log")
    if values.shape[1] > 1:
        ax.legend(frameon=False, fontsize="small")


def _labels(output: Output, count: int) -> list[str]:
    """Return the legend entries of a statistic's columns.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    count : int
        How many columns the statistic has once flattened.

    Returns
    -------
    list of str
        The mode names when the columns are the modes, and an index otherwise: a
        windowed statistic, or one that keeps its polarisation axis, has more columns
        than the run has modes.
    """
    n_modes = field_modes(output)
    if count == n_modes:
        return mode_labels(output, count)
    return [f"{index + 1}" for index in range(count)]


def _should_log(values: Float[np.ndarray, "n_z n_columns"]) -> bool:
    """Return whether a panel's ordinate should be logarithmic.

    Parameters
    ----------
    values : numpy.ndarray
        The values to be drawn, in display units.

    Returns
    -------
    bool
        Whether they span more than a decade and are everywhere positive.
    """
    if not np.all(values > 0.0):
        return False
    return bool(np.max(values) / np.min(values) > _LOG_RATIO)

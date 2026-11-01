# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standard plots of propagation results with matplotlib.

The standard figures, drawn from whatever :mod:`solprop.processing` reads out of a saved
run: the false-colour propagation maps, the time- and spectral-domain slices, the energy
along the fibre, the statistics panels and the spectrogram. Every function takes an
:class:`~solprop.output.schema.Output`, returns figures and never calls
:func:`~matplotlib.pyplot.show`, so a script decides for itself whether to save or to
display. The manual page is ``docs/guide/results/plotting.md``.

Display units are fixed across the package: z in centimetres,
time in femtoseconds, wavelength in nanometres, frequency in petahertz, angular
frequency in radians per femtosecond, energy in microjoules, power auto-scaled.
"""

from solprop.plotting.propagation import (
    DEFAULT_SLICE_WAVELENGTH_RANGE,
    DEFAULT_TIME_RANGE,
    DEFAULT_WAVELENGTH_RANGE,
    SLICE_QUANTITIES,
    SliceQuantity,
    plot_energy,
    prop_2d,
    spec_1d,
    time_1d,
)
from solprop.plotting.spectrograms import spectrogram
from solprop.plotting.statistics import OTHER_STATISTICS, PULSE_STATISTICS, plot_stats
from solprop.plotting.style import (
    CORNERS,
    POWER_UNITS,
    Corner,
    cmap_colours,
    cmap_white,
    corner_text,
    fwhm_arrows,
    mode_labels,
    per_width,
    power_unit,
    spectral_display,
    subplot_grid,
    wavelength_to_axis,
)

__all__ = [
    "CORNERS",
    "DEFAULT_SLICE_WAVELENGTH_RANGE",
    "DEFAULT_TIME_RANGE",
    "DEFAULT_WAVELENGTH_RANGE",
    "OTHER_STATISTICS",
    "POWER_UNITS",
    "PULSE_STATISTICS",
    "SLICE_QUANTITIES",
    "Corner",
    "SliceQuantity",
    "cmap_colours",
    "cmap_white",
    "corner_text",
    "fwhm_arrows",
    "mode_labels",
    "per_width",
    "plot_energy",
    "plot_stats",
    "power_unit",
    "prop_2d",
    "spec_1d",
    "spectral_display",
    "spectrogram",
    "subplot_grid",
    "time_1d",
    "wavelength_to_axis",
]

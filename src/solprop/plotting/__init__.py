# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standard plots of propagation results with matplotlib.

Every figure is drawn from what :mod:`solprop.processing` reads out of a saved run,
takes an :class:`~solprop.output.store.Output`, returns figures and never calls
:func:`~matplotlib.pyplot.show`, so a script decides whether to save or display. The
package re-exports the public names of:

- :mod:`~solprop.plotting.propagation` (the propagation maps, the time and spectral
  slices and the energy along the guide), :mod:`~solprop.plotting.statistics` (the
  statistics panels) and :mod:`~solprop.plotting.spectrograms`;
- :mod:`~solprop.plotting.style` (colours, grids, annotations and the fixed display
  units: z in cm, time in fs, wavelength in nm, frequency in PHz, angular frequency in
  rad/fs, energy in µJ, power auto-scaled).

The cross-section of a guide or a free-space grid, drawn from the guide rather than from
a run, is in :mod:`solprop.plotting.geometry`. The manual page is
``docs/guide/results/plotting.md``.
"""

from solprop.plotting.propagation import (
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

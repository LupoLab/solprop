# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The catalogue of plane statistics, with one definition of each.

The package, each module importing only those above it:

- :mod:`solprop.stats.definitions`: the reference definition of each statistic on
  internal spectra (:func:`energy`, :func:`peak_power`, :func:`fwhm_t`, ...).
- :mod:`solprop.stats.catalogue`: :data:`~solprop.stats.catalogue.CATALOGUE`, one
  :class:`StatDefinition` per name, and the name sets derived from it.
- :mod:`solprop.stats.spec`: :class:`StatsSpec`, the request a run records statistics
  by, its :class:`ApertureSet`, and the far-field tables every backend shares.
- :mod:`solprop.stats.evaluation`: :func:`collect` on a saved plane and
  :func:`default_spec`.

Their public names are importable from ``solprop.stats``, except
:func:`~solprop.stats.spec.far_field_tables`, which the translation layer reads from
its module.

Statistics belong to saved planes, not to steps; each is defined once, on internal
FFT-convention spectra, and a backend's fused pass is validated against that
definition. An undefined value is ``nan``; a missing dependency raises
:class:`solprop.exceptions.OutputError`. The manual pages are
``docs/guide/results/statistics.md`` and its definitions,
``docs/guide/results/statistics-definitions.md``.
"""

from solprop.stats.catalogue import (
    ACCESSOR_ADVICE,
    CATALOGUE,
    GENERAL_MODAL_ACCESSORS,
    METRIC_WEIGHTED_STATISTICS,
    ON_AXIS_PLASMA_STATISTICS,
    PLASMA_INTEGRATION_STATISTICS,
    PROFILE_STATISTICS,
    WITHHELD_WITH_A_FLOW,
    StatDefinition,
)
from solprop.stats.definitions import (
    DEFAULT_ZDW_RANGE,
    analytic_field,
    analytic_flux,
    energy,
    energy_window,
    fwhm_t,
    fwhm_t_outer,
    inband_wavelengths,
    omega0,
    peak_power,
    peak_power_total,
    peak_power_window,
    physical_energy,
    profile_value,
    species_statistics,
    spectral_flux,
    zdw,
)
from solprop.stats.evaluation import (
    WEIGHT_DEPENDENT_STATISTICS,
    collect,
    default_spec,
    weight_refusal_for,
)
from solprop.stats.spec import (
    MAX_APERTURES,
    MAX_REGIONS,
    ApertureSet,
    ApertureWeighting,
    StatsSpec,
    check_aperture_choices,
)

# isort: split
# Not part of the surface, and read from here by the backends: the reference backend's
# statistics (the Poynting-weighted set and its evaluator), solprop-native (the
# far-field names, defined in solprop.spatial) and processing.time (the half-maximum
# crossings).
from solprop.spatial import FAR_FIELD_STATISTICS as FAR_FIELD_STATISTICS
from solprop.stats.definitions import _half_max_range as _half_max_range
from solprop.stats.evaluation import _POYNTING_STATISTICS as _POYNTING_STATISTICS
from solprop.stats.evaluation import _poynting_statistics as _poynting_statistics

__all__ = [
    "ACCESSOR_ADVICE",
    "CATALOGUE",
    "DEFAULT_ZDW_RANGE",
    "GENERAL_MODAL_ACCESSORS",
    "MAX_APERTURES",
    "MAX_REGIONS",
    "METRIC_WEIGHTED_STATISTICS",
    "ON_AXIS_PLASMA_STATISTICS",
    "PLASMA_INTEGRATION_STATISTICS",
    "PROFILE_STATISTICS",
    "WEIGHT_DEPENDENT_STATISTICS",
    "WITHHELD_WITH_A_FLOW",
    "ApertureSet",
    "ApertureWeighting",
    "StatDefinition",
    "StatsSpec",
    "analytic_field",
    "analytic_flux",
    "check_aperture_choices",
    "collect",
    "default_spec",
    "energy",
    "energy_window",
    "fwhm_t",
    "fwhm_t_outer",
    "inband_wavelengths",
    "omega0",
    "peak_power",
    "peak_power_total",
    "peak_power_window",
    "physical_energy",
    "profile_value",
    "species_statistics",
    "spectral_flux",
    "weight_refusal_for",
    "zdw",
]

# The public classes report the package, their documented home, as their module: a
# run's configuration record names a class by its module, and that name -- and with it
# the resume key and the Python a recipe exports -- should not follow the package's
# internal layout.
for _public in (ApertureSet, StatDefinition, StatsSpec):
    _public.__module__ = __name__
del _public

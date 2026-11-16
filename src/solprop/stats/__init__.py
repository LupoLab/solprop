# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The catalogue of plane statistics, with one definition of each.

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

Statistics belong to saved planes, not to steps (see ``docs/dev/decisions.md``).
Evaluating every statistic on every accepted step, with a full inverse transform each
time and a root-finding zero-dispersion search among them, would cost about a fifth of a
step and dominate device runs; here they are evaluated where a field is saved, and an
optional periodic subset (:attr:`StatsSpec.periodic`) covers step-level diagnostics.

Every definition here is written once, on **internal** FFT-convention arrays of shape
``(n_modes, n_pol, n_omega)``, and reduces in a fixed order. There is **one
definition per statistic in the run**, not a host one and a device one that compute the
same quantity differently, and a backend's fused pass is validated against it. The
analytic signal comes from :func:`solprop.fourier.analytic_signal` alone.

One quantity is measured differently after the run, and deliberately: a run forms its
temporal power as the analytic Poynting flux from the internal amplitude *and* the
plane's index, while :mod:`solprop.processing.time`, given a saved plane and no mode,
measures the squared true-power envelope ``|b(t)|^2``. The two differ at second order in
the spread of ``Re n_eff`` across the spectrum and have identical time integrals; see
``docs/guide/results/processing.md`` and the docstrings of
:func:`solprop.processing.time.peak_power` and
:func:`solprop.processing.time.fwhm_t`.

Spatial statistics use field units and the geometry's transverse integration measure.
Their host definitions are in :mod:`solprop.spatial`; periodic backend reductions are
qualified against those definitions. Plasma statistics need the compiled response,
and spatial widths need the geometry declared at assembly.

**Undefined is not an error.** A mode that carries no light has no spectral centroid,
and a trace that never falls to half its maximum -- a continuous wave, an empty mode --
has no full width. Those return ``nan``, which the docstring of each states and the
manual page repeats. A *missing dependency*, by contrast -- a zero-dispersion
wavelength with no mode, a windowed energy with no windows -- raises
:class:`solprop.exceptions.OutputError`.

The manual page is ``docs/guide/results/statistics.md``.
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
# crossings). solprop-native also names the whitening and the crossings here as the
# definitions it mirrors.
from solprop.spatial import FAR_FIELD_STATISTICS as FAR_FIELD_STATISTICS
from solprop.stats.definitions import _half_max_range as _half_max_range
from solprop.stats.definitions import _whitened as _whitened
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

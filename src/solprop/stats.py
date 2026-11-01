# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The catalogue of plane statistics, with one definition of each.

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

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.exceptions import ApertureEdgeWarning, OutputError
from solprop.grid import TimeGrid
from solprop.modes import base as modes_base
from solprop.modes.protocol import Mode
from solprop.modes.slab import PlanarGrid
from solprop.output.schema import SavedPlane, SimulationKind
from solprop.profiles import ZProfile
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    OPERATOR_STATISTICS,
    ApertureProfile,
    SpatialFlux,
    aperture_statistics,
    beam_radius,
    check_aperture_profile,
    fwhm_r,
    on_axis_intensity,
    on_axis_spectrum,
    planar_intensity,
    spatial_energy,
    spatial_intensity,
)

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

logger = logging.getLogger(__name__)


def species_statistics(
    densities: Float[np.ndarray, " n_densities"],
    fractions: Float[np.ndarray, "n_species n_nodes"],
    peak_rates: Float[np.ndarray, " n_peaks"],
    potentials: Float[np.ndarray, " n_potentials"],
    weights: Float[np.ndarray, " n_weights"],
) -> dict[str, np.ndarray]:
    """Reduce species populations at the node of largest total electron density.

    Parameters
    ----------
    densities : numpy.ndarray
        Component number densities in m^-3, in mixture order.
    fractions : numpy.ndarray
        Final ionised fractions at each node, shape (n_species, n_nodes).
        Components without an ionisation channel have zero entries.
    peak_rates : numpy.ndarray
        Largest rate over all nodes per species in s^-1; cycle-averaged on
        envelope grids.
    potentials : numpy.ndarray
        Ionisation potentials in J, zero for components without channels.
    weights : numpy.ndarray
        Physical transverse quadrature weights in m². The mode-averaged case uses
        its effective area as its single weight.

    Returns
    -------
    dict
        The five species statistics plus total electron density and peak rate.
        ``species_ionisation_energy`` is final ionised inventory in J/m, including
        preionisation. Pulse-deposited ionisation work counts only newly freed
        electrons and must subtract the initial fractions before this reduction.

    Raises
    ------
    OutputError
        If a population, rate, potential or quadrature weight is outside its domain.
    """
    expected = (len(densities), len(weights))
    if (
        fractions.shape != expected
        or peak_rates.shape != densities.shape
        or potentials.shape != densities.shape
    ):
        raise OutputError(
            "species statistics require matching species and node extents; "
            f"expected fractions {expected}, got {fractions.shape}"
        )
    if (
        fractions.size == 0
        or not all(
            np.all(np.isfinite(v))
            for v in (densities, fractions, peak_rates, potentials, weights)
        )
        or any(
            np.any(v < 0)
            for v in (densities, fractions, peak_rates, potentials, weights)
        )
        or np.any(fractions > 1)
    ):
        raise OutputError(
            "species statistics require finite non-negative populations "
            "and fractions in [0, 1]"
        )
    electrons = densities[:, None] * fractions
    total = np.zeros(fractions.shape[1])
    for row in electrons:
        total += row
    densest = int(np.argmax(total))
    energy = np.zeros(len(densities))
    for index, (row, potential) in enumerate(zip(electrons, potentials, strict=True)):
        for population, weight in zip(row, weights, strict=True):
            energy[index] += weight * population * potential
    return {
        "species_ionisation": fractions.copy(),
        "species_electron_density": electrons[:, densest].copy(),
        "species_peak_ionisation_rate": peak_rates.copy(),
        "species_density": densities.copy(),
        "species_ionisation_energy": energy,
        "electron_density": np.asarray(total[densest]),
        "peak_ionisation_rate": np.asarray(np.max(peak_rates)),
    }


PROFILE_STATISTICS = ("density", "pressure", "core_radius")
"""Statistics that read a :class:`solprop.profiles.ZProfile` at the plane's position."""

ON_AXIS_PLASMA_STATISTICS = (
    "on_axis_electron_density",
    "on_axis_peak_ionisation_rate",
    "on_axis_species_electron_density",
    "on_axis_species_peak_ionisation_rate",
)
"""The plasma statistics evaluated on the waveguide's axis (SEP 19, section 5.2).

Opt-in and saved-plane only. The axis field is synthesised from the mode profiles, as
for ``peak_intensity``, and its populations are scanned with exactly the rate tables,
preionisation, densities and running-integral rule of the node scan, once per
observation; a rule with a node on the axis supplies that row instead. They sit beside
the unchanged node reductions -- ``electron_density`` is the densest node's -- because a
few-node field-scale rule may have no node near the axis, and a multimode field's
largest density need not lie on it. Defined on resolved modal routes; the spatial
routes, general modal guides, the mode-averaged route and TDSE refuse them by name.
"""

PLASMA_INTEGRATION_STATISTICS = (
    "plasma_undershoot",
    "plasma_work_defect",
    "plasma_boundary_level",
)
"""The field plasma column's run-time integration diagnostics (SEP 18, section 4.5).

Evaluated with the other plasma statistics at each save plane. Each is a ratio of two
maxima over every sample, node and population, so that a node the field barely
ionises cannot set it: the undershoot ``max(-dS) / max(dS)`` of the exposure
increments; the work defect ``max |paid - freed| / max freed``, the loss current's work
per ionisation potential against the fraction it freed; and the boundary level, the
free-current integrand's largest magnitude in the ``2K`` samples the filter reads past
the window's start against its largest anywhere.
"""

DEFAULT_ZDW_RANGE = (100e-9, 3e-6)
"""Search window of :func:`zdw` when no narrower one is given, m."""


def analytic_field(
    grid: TimeGrid, field: Complex[np.ndarray, "*batch n_omega"]
) -> Complex[np.ndarray, "*batch n_t"]:
    """Return the time-domain analytic signal of a stored spectrum.

    The one construction of the analytic signal in the package, so that the statistics,
    the backends' on-axis intensity and anything else that needs a power agree by
    construction. It serves both grids: an envelope grid's state already *is*
    the analytic signal with the carrier removed, so it only has to be transformed back;
    a field grid's is the analytic signal of the real field.
    :func:`solprop.fourier.analytic_signal` neither doubles the zero-frequency bin nor
    discards the Nyquist bin, because only that convention makes the real part an
    identity and the two energy functionals an exact Parseval pair.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(..., n_omega)``, internal convention.

    Returns
    -------
    numpy.ndarray
        Complex analytic signal, shape ``(..., n_t)``, whose modulus squared is the
        instantaneous power in watts.
    """
    samples = fourier.inverse(field, kind=grid.kind)
    if grid.kind == "field":
        return fourier.analytic_signal(samples)
    return samples


def _power(
    grid: TimeGrid, field: Complex[np.ndarray, "*batch n_omega"]
) -> Float[np.ndarray, "*batch n_t"]:
    """Return the instantaneous power of a stored spectrum, in watts.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(..., n_omega)``, internal convention, in the square
        root of watts.

    Returns
    -------
    numpy.ndarray
        ``P(t) = |E_a(t)|^2``, shape ``(..., n_t)``, W.
    """
    return np.abs(analytic_field(grid, field)) ** 2


def _whitened(
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Return the amplitudes of the same field in an orthonormal basis.

    The one normalisation every metric-weighted statistic shares. A basis whose flux
    Gram is ``W`` has physical same-frequency flux
    ``sum_mn conj(a_m) W_mn a_n``; the Hermitian positive-definite square root
    ``W^(1/2)`` turns that into ``sum_m |c_m|^2`` with ``c = W^(1/2) a``, because
    ``c.H c = a.H W^(H/2) W^(1/2) a = a.H W a`` and ``W^(1/2)`` is Hermitian. So a
    statistic of the physical flux is the **existing** statistic of ``c``, which is
    how ``physical_energy``, ``analytic_flux``, ``omega0``, ``fwhm_t`` and
    ``fwhm_t_outer`` stay one definition each rather than two
    (``docs/dev/conventions.md`` section 4).

    ``W^(1/2)`` is the unique positive-definite square root, so this is a *choice of
    orthonormal basis* fixed by ``W`` alone: it does not depend on how the basis was
    arrived at, and it reduces to the identity exactly where ``W`` does.

    ``metric=None`` is the identity and returns the field itself, unchanged and
    uncopied, so an unweighted run's arithmetic is bitwise what it was.

    Parameters
    ----------
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    metric : numpy.ndarray or None
        The basis flux metric ``W(omega)``, Hermitian positive definite, shape
        ``(n_modes, n_modes, n_omega)``, dimensionless; ``None`` is the identity.

    Returns
    -------
    numpy.ndarray
        ``c = W^(1/2) a``, of the field's own shape and units.

    Raises
    ------
    solprop.exceptions.OutputError
        If the metric's extents disagree with the field, or it is not positive
        definite at some frequency.
    """
    if metric is None:
        return field
    modes, _, frequencies = field.shape
    if metric.shape != (modes, modes, frequencies):
        msg = (
            f"a metric-weighted statistic needs W of shape "
            f"({modes}, {modes}, {frequencies}) for this field, got {metric.shape}"
        )
        raise OutputError(msg)
    # eigh reads the lower triangle of a Hermitian matrix, which is what a flux
    # metric is; the record that carries W refuses a block that is not.
    values, vectors = np.linalg.eigh(np.moveaxis(metric, (0, 1), (-2, -1)))
    if np.any(values <= 0.0):
        msg = (
            "a metric-weighted statistic needs a positive-definite flux metric; the "
            f"smallest eigenvalue is {float(np.min(values)):.6g}, so this basis "
            "carries a direction with no positive energy and no orthonormal basis "
            "to measure it in"
        )
        raise OutputError(msg)
    root = (vectors * np.sqrt(values)[..., None, :]) @ np.conj(
        np.swapaxes(vectors, -1, -2)
    )
    return np.einsum("kmn,npk->mpk", root, field)


def _window_masks(
    grid: TimeGrid, windows: Float[np.ndarray, "n_windows 2"]
) -> Bool[np.ndarray, "n_windows n_omega"]:
    """Return the hard spectral gate of each window, as a boolean mask per bin.

    The gate is half-open, ``lo <= omega < hi``, so that windows laid end to end assign
    every bin exactly once and their energies sum to the total. There is no tapered
    option, because a windowed *energy* never returns to the time domain and so has
    nothing to gain from a smooth edge.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid, whose ``omega`` axis the gate is taken on.
    windows : numpy.ndarray
        Window edges, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)``.

    Returns
    -------
    numpy.ndarray
        Boolean masks, shape ``(n_windows, n_omega)``.
    """
    lower = windows[:, 0, np.newaxis]
    upper = windows[:, 1, np.newaxis]
    omega = grid.omega[np.newaxis, :]
    return (omega >= lower) & (omega < upper)


def omega0(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, "n_modes n_pol"]:
    r"""Return the centroid of the spectral energy density, per mode and polarisation.

    .. math:: \omega_0 = \frac{\sum_k \omega_k\,S(\omega_k)}{\sum_k S(\omega_k)}

    with :math:`S` from :func:`solprop.fourier.spectral_density`. The prefactor of the
    density cancels in the ratio, so this is the first moment of
    :math:`|\tilde E|^2` over the frequency axis. The sum runs over every bin,
    including the zero-frequency one: outside the active band a propagated state is
    identically zero, so the bins there contribute nothing.

    Because the centroid is an energy-weighted mean, the whole field's centroid is the
    energy-weighted mean of these, which is why the statistic is per column rather than
    summed.

    **On a basis whose flux Gram is not the identity this takes the metric**, and the
    columns are then the whitened ones of ``_whitened``: a coordinate column of a
    nonorthogonal basis carries no separable spectral energy at all -- the columns
    interfere -- so a centroid of ``|a_m(omega)|^2`` would be a promise the basis
    cannot keep. ``W^(1/2)`` is the identity plus half the basis's Gram departure, so
    on a **unitary** change of basis of a certified orthonormal reference, which is
    the only kind this package executes, the whitened column is the declared channel
    to within that departure and the centroid moves by no more.

    **An empty column is empty relative to the plane.** A column whose density sums
    to at most one machine epsilon of the largest column's has no centroid and is
    ``nan``. That floor is relative rather than exactly zero because whitening
    mixes: the unlaunched partner of a launched circular channel carries
    ``W^(1/2)`` times the other's amplitude -- 1e-32 of it on this route's own
    fixture -- and an exact-zero test would report a finite centroid of the
    whitening's rounding noise for a channel nothing was launched into.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity and is the bitwise
        unchanged path.

    Returns
    -------
    numpy.ndarray
        Centroid frequency, shape ``(n_modes, n_pol)``, rad/s. A column carrying no
        energy has no centroid and is ``nan``.

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import omega0
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> f"{float(omega0(grid, pulse.build(grid))[0, 0]):.6e}"
    '2.354564e+15'
    """
    density = fourier.spectral_density(
        _whitened(field, metric), grid.dt, kind=grid.kind
    )
    total = np.sum(density, axis=-1)
    weighted = np.sum(density * grid.omega, axis=-1)
    centroid = np.full(total.shape, np.nan)
    # An *empty* column is empty relative to the plane, not to zero. On the identity
    # metric an unlaunched channel is identically zero and `> 0` catches it; after a
    # whitening it is not, because `c_0 = W^(1/2)_{01} a_1` is the off-diagonal's
    # share of the launched partner -- measured 1.1e-32 of it on the general-basis
    # test fixture, which divides out to a perfectly finite centroid of rounding noise.
    # One eps of the largest column is the floor: below it the numerator and the
    # denominator are both the whitening's own rounding and the ratio says nothing.
    # `initial=0.0` is free here -- a spectral density is nonnegative -- and keeps a
    # plane with no columns at all from having to be a branch.
    floor = float(np.finfo(float).eps) * float(np.max(total, initial=0.0))
    np.divide(weighted, total, out=centroid, where=total > floor)
    if np.isnan(centroid).any():
        logger.debug(
            "omega0 is undefined for %d empty columns", np.isnan(centroid).sum()
        )
    return centroid


def energy(
    grid: TimeGrid, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> Float[np.ndarray, "n_modes n_pol"]:
    """Return the pulse energy of each mode and polarisation, in joules.

    This is :func:`solprop.fourier.energy_omega`, the exact discrete form of
    ``docs/dev/conventions.md`` §4, which agrees with the time-domain functional to
    round-off. No quadrature weights (such as an extended Simpson rule on a field grid)
    are derived: those would differ from this by their fourth-order quadrature error.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention, in
        the square root of watts.

    Returns
    -------
    numpy.ndarray
        Energy, shape ``(n_modes, n_pol)``, J.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import energy
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> f"{float(energy(grid, pulse.build(grid))[0, 0]):.6e}"
    '1.000000e-06'
    """
    return fourier.energy_omega(field, grid.dt, kind=grid.kind)


def energy_window(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    windows: Float[np.ndarray, "n_windows 2"],
) -> Float[np.ndarray, "n_windows n_modes n_pol"]:
    """Return the energy inside each spectral window, in joules.

    The gate is hard and half-open, ``lo <= omega < hi``, so windows that tile the band
    partition the bins and their energies sum to the total exactly.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    windows : numpy.ndarray
        Window edges, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)``. Convert a
        wavelength pair with ``omega = 2 pi c / wavelength``, remembering that it
        reverses the order.

    Returns
    -------
    numpy.ndarray
        Energy per window, shape ``(n_windows, n_modes, n_pol)``, J.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import energy, energy_window
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> tiling = np.array([[-np.inf, 2.4e15], [2.4e15, np.inf]])
    >>> total = energy_window(grid, built, tiling).sum()
    >>> bool(np.isclose(total, energy(grid, built).sum(), rtol=1e-15))
    True
    """
    masks = _window_masks(grid, windows)
    gated = field[np.newaxis, ...] * masks[:, np.newaxis, np.newaxis, :]
    return fourier.energy_omega(gated, grid.dt, kind=grid.kind)


def peak_power(
    grid: TimeGrid, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> Float[np.ndarray, " n_modes"]:
    """Return the largest instantaneous power each mode reaches, in watts.

    ``max_t sum_pol |E_a(t)|^2``: the polarisation components of one mode are summed
    before the maximum is taken, because they are two components of one field and their
    powers add at each instant. The analytic signal is formed once for the whole call.
    This array function measures squared true-power amplitude; run statistics use
    analytic Poynting flux, with the arithmetic-mean frequency kernel instead.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.

    Returns
    -------
    numpy.ndarray
        Peak power, shape ``(n_modes,)``, W.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import peak_power
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=1e9)
    >>> f"{float(peak_power(grid, pulse.build(grid))[0]):.6e}"
    '1.000000e+09'
    """
    return np.max(np.sum(_power(grid, field), axis=-2), axis=-1)


def peak_power_total(
    grid: TimeGrid, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> Float[np.ndarray, ""]:
    """Return the largest instantaneous power of the whole field, in watts.

    ``max_t sum_modes sum_pol |E_a(t)|^2``: the modes are summed **before** the maximum,
    so this is the peak of the total power and not the largest of the per-mode peaks,
    which arrive at different times.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.

    Returns
    -------
    numpy.ndarray
        Peak total power, a zero-dimensional array, W.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import peak_power, peak_power_total
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=1e9)
    >>> built = pulse.build(grid)
    >>> f"{float(peak_power_total(grid, built)):.6e}"
    '1.000000e+09'
    """
    return np.asarray(np.max(np.sum(_power(grid, field), axis=(-3, -2)), axis=-1))


def physical_energy(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, ""]:
    """Return the physical energy through the plane, in joules.

    The general modal basis's name for an energy, and the one number of that family
    that survives a change of coordinates: the same-frequency product of the state
    with the basis's flux metric, ``sum_k sum_mn conj(a_m) W_mn a_n``, integrated
    over frequency. A coordinate energy is not additive when ``W`` is not the
    identity, and this total is, which is why it is a scalar and ``energy`` is an
    array.

    **On the identity metric this is exactly ``sum(energy(grid, field))``**, which is
    what ``metric=None`` evaluates, bit for bit. With a metric it is the same
    functional of the whitened amplitudes ``c = W^(1/2) a`` of ``_whitened``,
    which is the definition above and not a second one:
    ``sum_m |c_m|^2 = sum_mn conj(a_m) W_mn a_n`` at every frequency, and the
    frequency integral is the Parseval weight of ``docs/dev/conventions.md``
    section 4 either way.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention, in
        the square root of watts.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity.

    Returns
    -------
    numpy.ndarray
        Physical energy, a zero-dimensional array, J.

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import energy, physical_energy
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> bool(physical_energy(grid, built) == energy(grid, built).sum())
    True
    """
    return np.asarray(np.sum(energy(grid, _whitened(field, metric))))


def spectral_flux(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, " n_omega"]:
    """Return the physical energy's spectral density, in J/(rad/s).

    The spectrum analogue of :func:`physical_energy`, and the one array of the
    general modal family: the same same-frequency product of the state with the
    basis's flux metric, ``sum_mn conj(a_m) W_mn a_n``, published **per bin**
    instead of integrated over frequency. It is a density in the sense of
    ``docs/dev/conventions.md`` section 4 -- ``dU/domega``, the prefactor that
    pairs with the shot-noise amplitude so that one photon in a bin gives
    ``S(omega_k) domega = hbar omega_k`` -- and it is
    :func:`solprop.fourier.spectral_density` summed over the channel axis, which
    is what makes it additive where a coordinate spectrum is not.

    It is indexed by frequency and by nothing else. A per-channel spectral energy
    is no more additive than a per-channel energy on a basis whose Gram is not the
    identity, so the channel axis is summed **after** the whitening and not
    published.

    **The frequency sum is ``physical_energy``**, exactly on every interior bin:
    ``domega`` times this density is the ``4 dt / n_t`` weight of
    :func:`solprop.fourier.energy_omega` there. The zero-frequency and Nyquist
    bins are the documented exception -- the analytic signal does not double them,
    so summing the density over every bin overcounts those two -- and both are
    identically zero on any grid :mod:`solprop.grid` builds, whose spectral collar
    apodises them away. Ask :func:`physical_energy` when an exact energy is wanted.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention, in
        the square root of watts.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity.

    Returns
    -------
    numpy.ndarray
        Spectral energy density, shape ``(n_omega,)``, J/(rad/s).

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import physical_energy, spectral_flux
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> density = spectral_flux(grid, built)
    >>> density.shape == (grid.n_omega,)
    True
    >>> total = float(np.sum(density) * grid.d_omega)
    >>> bool(np.isclose(total, float(physical_energy(grid, built)), rtol=1e-14))
    True
    """
    return np.sum(
        fourier.spectral_density(_whitened(field, metric), grid.dt, kind=grid.kind),
        axis=(0, 1),
    )


def analytic_flux(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, ""]:
    """Return the peak of the plane's same-frequency analytic flux, in watts.

    The general modal basis's name for a peak power: ``max_t sum_m |a_m(t)|^2``, the
    **same-frequency diagonal** of the Poynting flux, which is all a modal state can
    say about a power on its own.

    **It is not the cross-section flux, and the difference is physics.** The physical
    ``int Re(E_a x conj(H_a))/2 . z dA`` of the reconstructed fields carries the
    guide's **cross-frequency** flux kernel
    ``K_ij = sum_q w_q [e_x(w_i) conj(h_y(w_j)) - e_y(w_i) conj(h_x(w_j))]/2``, whose
    diagonal a provider normalises to one and whose off-diagonal it says nothing
    about. The two peaks therefore differ, bounded term by term by
    ``sum_ij |B_i||B_j||K_ij - 1|`` and measured at 1.3e-5 of the peak on the
    general-basis test fixture, inside a bound of 3.4e-5: even a positive spectral
    metric does not make an instantaneous cross-frequency flux a sum of coordinate
    powers. The **time integrals** do agree,
    to 5.0e-7 there, because the cross-frequency terms are an integer number of
    cycles in the window. For the flux itself, reconstructed from the guide's stored
    reference, use :func:`solprop.processing.modal.reconstructed_flux`, a trace in
    time whose peak is this statistic plus the cross-frequency terms; its real-field
    partner with the sum-frequency term is
    :func:`solprop.processing.modal.instantaneous_flux`.

    **The arithmetic is exactly ``peak_power_total``'s** on the identity metric,
    which is what ``metric=None`` evaluates, bit for bit. With a metric it is the
    same peak taken on the whitened amplitudes ``c = W^(1/2) a`` of
    ``_whitened``, so that the quantity peaked is the physical same-frequency
    flux ``sum_mn conj(a_m) W_mn a_n`` and not a sum of coordinate powers. The old
    name is refused on this basis rather than redefined: ``peak_power_total``
    promises that sum of coordinate powers, and on a basis that is not orthonormal
    there is no such sum to promise.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention, in
        the square root of watts.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity.

    Returns
    -------
    numpy.ndarray
        Peak same-frequency analytic flux, a zero-dimensional array, W.

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import analytic_flux, peak_power_total
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=1e9)
    >>> built = pulse.build(grid)
    >>> bool(analytic_flux(grid, built) == peak_power_total(grid, built))
    True
    """
    return peak_power_total(grid, _whitened(field, metric))


def peak_power_window(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    windows: Float[np.ndarray, "n_windows 2"],
) -> Float[np.ndarray, "n_windows n_modes"]:
    """Return the peak power of each mode inside each spectral window, in watts.

    The spectrum is gated with the same hard half-open window as
    :func:`energy_window` and transformed back, so this is the peak power of the light
    in that band. One scalar per mode per window.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    windows : numpy.ndarray
        Window edges, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)``.

    Returns
    -------
    numpy.ndarray
        Peak power, shape ``(n_windows, n_modes)``, W.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import peak_power_window
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=1e9)
    >>> built = pulse.build(grid)
    >>> whole = np.array([[-np.inf, np.inf]])
    >>> f"{float(peak_power_window(grid, built, whole)[0, 0]):.6e}"
    '1.000000e+09'
    """
    masks = _window_masks(grid, windows)
    gated = field[np.newaxis, ...] * masks[:, np.newaxis, np.newaxis, :]
    return np.max(np.sum(_power(grid, gated), axis=-2), axis=-1)


def _crossing(
    x: Float[np.ndarray, " n"],
    y: Float[np.ndarray, " n"],
    level: float,
    pair: tuple[int, int],
) -> float:
    """Return the abscissa at which a straight line through two samples reaches a level.

    The stable form ``x1 + (level - y1)(x2 - x1)/(y2 - y1)``. Forming the slope and
    intercept of the line and evaluating ``(level - c)/m`` instead would, on a time axis
    whose values are of order ``1e-12`` and whose intercept is of order ``1e12``, lose
    most of the significand.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa, shape ``(n,)``.
    y : numpy.ndarray
        The trace, shape ``(n,)``.
    level : float
        The value to cross.
    pair : tuple of int
        Indices of the two samples that bracket the crossing.

    Returns
    -------
    float
        The crossing point, in the units of ``x``.
    """
    low, high = pair
    span = y[high] - y[low]
    return float(x[low] + (level - y[low]) * (x[high] - x[low]) / span)


def _half_max_range(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"], *, outer: bool
) -> tuple[float, float]:
    """Return the two points at which a trace crosses half its maximum.

    The one implementation of the half-maximum crossing: the temporal widths here and
    :func:`solprop.processing.time.half_max_range`, which documents the two
    conventions, both call it.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa, shape ``(n,)``, increasing.
    y : numpy.ndarray
        The trace, shape ``(n,)``, unclipped.
    outer : bool
        Take the first and last sample above the level anywhere, rather than walking
        outwards from the peak to the first sample below it on either side.

    Returns
    -------
    tuple of float
        The lower and upper crossings, both ``nan`` when the trace carries nothing, does
        not fall to half its maximum on both sides, or reaches the end of the axis still
        above it.
    """
    undefined = (float("nan"), float("nan"))
    peak = float(np.max(y))
    if peak <= 0.0:
        logger.debug("the half-maximum width is undefined: the trace carries nothing")
        return undefined
    level = 0.5 * peak
    apex = int(np.argmax(y))
    if outer:
        # The first and the last sample above half maximum anywhere on the axis, so
        # satellites and a pedestal are inside the width.
        above = np.flatnonzero(y > level)
        low, high = int(above[0]), int(above[-1])
        pairs = ((low - 1, low), (high + 1, high))
    else:
        # Walk outwards from the peak and stop at the first sample below half maximum,
        # so the width is that of the central feature alone.
        below_left = np.flatnonzero(y[:apex] < level)
        below_right = np.flatnonzero(y[apex:] < level)
        if below_left.size == 0 or below_right.size == 0:
            logger.debug(
                "the half-maximum width is undefined: no crossing beside the peak"
            )
            return undefined
        low, high = int(below_left[-1]), apex + int(below_right[0])
        pairs = ((low, low + 1), (high, high - 1))
    if pairs[0][0] < 0 or pairs[1][0] >= y.size:
        logger.debug("the half-maximum width is undefined: the trace fills the axis")
        return undefined
    return _crossing(x, y, level, pairs[0]), _crossing(x, y, level, pairs[1])


def _half_width(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    outermost: bool,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, ""]:
    """Return the full width at half maximum of the total power, in seconds.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    outermost : bool
        Whether to take the outermost pair of half-maximum crossings rather than the
        innermost pair on either side of the global peak.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``; ``None`` is the identity.

    Returns
    -------
    numpy.ndarray
        The width as a zero-dimensional array, s, or ``nan`` when the trace does not
        cross half its maximum on both sides of the peak.
    """
    return _power_half_width(
        grid,
        np.sum(_power(grid, _whitened(field, metric)), axis=(0, 1)),
        outermost=outermost,
    )


def _power_half_width(
    grid: TimeGrid, power: Float[np.ndarray, " n_t"], *, outermost: bool
) -> Float[np.ndarray, ""]:
    """Measure half-maximum crossings of a power trace without clipping its sign.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    power : numpy.ndarray
        Unclipped temporal flux, shape (n_t,), in watts.
    outermost : bool
        Use the outermost half-maximum crossings when true.

    Returns
    -------
    numpy.ndarray
        Scalar duration in seconds, NaN when crossings are absent.
    """
    start, end = _half_max_range(grid.t, power, outer=outermost)
    return np.asarray(end - start)


def fwhm_t(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, ""]:
    """Return the width of the central feature at half maximum, in seconds.

    The total power ``sum_modes sum_pol |E_a(t)|^2`` is formed, the half-maximum level
    taken from its global peak with no baseline subtraction, and the crossings found by
    walking outwards from that peak to the first sample below the level on either side
    and interpolating linearly. Structure further out -- a dispersive wave, a soliton
    that has walked off, a noise pedestal -- does not widen the answer, which is why
    this is the number to read as a pulse duration. Use :func:`fwhm_t_outer` for the
    total extent above half maximum.

    **On a basis whose flux Gram is not the identity this takes the metric**: the
    trace whose width is measured is then the physical same-frequency flux
    :func:`analytic_flux` peaks, evaluated on the whitened amplitudes of
    ``_whitened``, so that a run does not report the width of a quantity whose
    peak it does not report.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity.

    Returns
    -------
    numpy.ndarray
        The width as a zero-dimensional array, s. A trace that never falls to half its
        maximum -- a continuous wave, an empty plane -- has no width and gives ``nan``.

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import fwhm_t
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> bool(np.isclose(fwhm_t(grid, built), 30e-15, rtol=1e-4))
    True
    """
    return _half_width(grid, field, outermost=False, metric=metric)


def fwhm_t_outer(
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> Float[np.ndarray, ""]:
    """Return the total extent above half maximum, in seconds.

    The first and the last sample anywhere in the window that exceed half the peak
    power, interpolated outwards. On a clean single pulse this equals :func:`fwhm_t`;
    on a structured trace it spans every satellite, so the pair together says how much
    of the energy sits outside the central feature.

    It takes the metric for the reason :func:`fwhm_t` does.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    field : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless; ``None``, the default, is the identity.

    Returns
    -------
    numpy.ndarray
        The extent as a zero-dimensional array, s, or ``nan`` when the trace reaches
        the edge of the window still above half its maximum, or carries no power.

    Raises
    ------
    solprop.exceptions.OutputError
        If `metric` disagrees in shape with `field`, or is not positive definite at
        some frequency, so that no orthonormal basis exists to measure in.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import fwhm_t, fwhm_t_outer
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> bool(np.isclose(fwhm_t_outer(grid, built), fwhm_t(grid, built), rtol=1e-12))
    True
    """
    return _half_width(grid, field, outermost=True, metric=metric)


def profile_value(profile: ZProfile, z: float) -> Float[np.ndarray, ""]:
    """Return the value of a z-profile at one position.

    The one implementation behind the ``density``, ``pressure`` and ``core_radius``
    statistics, which differ only in which profile they read; the catalogue gives each
    its own units.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        The path along the waveguide.
    z : float
        Position, m.

    Returns
    -------
    numpy.ndarray
        The value as a zero-dimensional array, in the profile's own units. Those are SI
        except for a pressure path, which is in bar: pressures are the one documented
        exception at the user interface (``docs/dev/conventions.md`` §1) and a
        :class:`solprop.profiles.PressureGradient` is a user-interface object.

    Examples
    --------
    >>> from solprop.profiles import PressureGradient
    >>> from solprop.stats import profile_value
    >>> gradient = PressureGradient.two_point(1.0, 2.0, 0.0)
    >>> f"{float(profile_value(gradient, 0.75)):.6f}"
    '1.000000'
    """
    return np.asarray(profile(np.asarray([float(z)]))[0])


def inband_wavelengths(grid: TimeGrid) -> tuple[float, float]:
    """Return the shortest and longest in-band wavelength of a grid, in metres.

    The search window :func:`zdw` uses by default when :func:`collect` calls it: a
    zero-dispersion point outside the simulated band says nothing about the run, and
    hunting for one there drives every material model into extrapolation.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid.

    Returns
    -------
    tuple of float
        ``(shortest, longest)`` in m, from the grid's in-band frequency limits.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.stats import inband_wavelengths
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> [f"{value * 1e9:.1f}" for value in inband_wavelengths(grid)]
    ['400.0', '2000.0']
    """
    return (
        2.0 * np.pi * C / grid.omega_max,
        2.0 * np.pi * C / grid.omega_min,
    )


def zdw(
    modes: Sequence[Mode],
    z: float,
    *,
    wavelength_range: tuple[float, float] = DEFAULT_ZDW_RANGE,
) -> Float[np.ndarray, " n_modes"]:
    """Return the zero-dispersion wavelength of each mode at one position.

    Off by default: it is a root search on ``beta_2`` per mode, which is slow and does
    not belong on every accepted step. Each call stands alone: no previous answer is
    carried over as the next search's starting guess, so one failure cannot poison
    every later plane with ``nan``.

    Parameters
    ----------
    modes : collections.abc.Sequence
        The propagated modes, in state order.
    z : float
        Position along the waveguide, m.
    wavelength_range : tuple of float, optional
        Shortest and longest wavelength in m to bracket the root in. Defaults to a
        100 nm to 3 um window; :func:`collect` narrows it to the grid's own band with
        :func:`inband_wavelengths`, which is where the answer means anything and where
        the material models are not extrapolating.

    Returns
    -------
    numpy.ndarray
        Zero-dispersion wavelength, shape ``(n_modes,)``, m.

    Raises
    ------
    solprop.exceptions.ModeError
        If a mode has no zero-dispersion point in the search window, rather than a
        missing value that would propagate silently.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.stats import zdw
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> f"{zdw([mode], 0.0, wavelength_range=(150e-9, 900e-9))[0] * 1e9:.1f}"
    '376.7'
    """
    return np.asarray(
        [modes_base.zdw(mode, z, wavelength_range=wavelength_range) for mode in modes]
    )


@dataclass(frozen=True)
class StatDefinition:
    """What one statistic is called, what it returns and what it needs.

    Attributes
    ----------
    name : str
        Key the statistic is stored under.
    summary : str
        One line saying what the number is, used by the manual page.
    shape : str
        Shape of the value in the standard axis names, ``'()'`` for a scalar.
    units : str
        SI units of the value.
    requires : tuple
        What :func:`collect` must be given for this statistic: ``'windows'``,
        ``'modes'``, the name of a z-profile, ``'spatial_geometry'``, the far-field
        tables ``'apertures'`` and ``'regions'`` of the spec, or
        ``'separable_operator'``, the run's linear operator, which only a backend
        holds.
    implemented : bool
        Whether the statistic exists yet. A reserved name is in the catalogue so that a
        spec asking for it is refused now instead of silently returning nothing.
    backend : bool
        Whether only a backend can compute it. These are the statistics of quantities
        the saved field does not carry -- the electron density, the ionisation rate, the
        on-axis intensity, the quadrature error -- which a backend forms from its own
        buffers in one pass. :func:`collect` refuses them by name rather than
        pulling the state back across the seam to reconstruct them.
    between_planes : bool
        Whether a backend can also evaluate it *between* save planes, from what an
        accepted step already leaves behind, for the periodic channel of
        :attr:`StatsSpec.periodic`. What disqualifies a statistic is cost per step, not
        difficulty: ``zdw`` needs a root search per mode, and paying that on every step
        would cost a large share of the step; ``peak_intensity`` is in the subset
        because its on-axis matrix is built once and rescaled rather than rebuilt.
        :class:`StatsSpec` refuses a periodic name without it, by name.

    Examples
    --------
    >>> from solprop.stats import CATALOGUE
    >>> CATALOGUE["energy"].units, CATALOGUE["energy"].shape
    ('J', '(n_modes, n_pol); () for spatial beams')
    >>> CATALOGUE["electron_density"].backend
    True
    >>> CATALOGUE["energy"].between_planes, CATALOGUE["zdw"].between_planes
    (True, False)
    """

    name: str
    summary: str
    shape: str
    units: str
    requires: tuple[str, ...] = ()
    implemented: bool = True
    backend: bool = False
    between_planes: bool = False


def _catalogue() -> dict[str, StatDefinition]:
    """Return the fixed catalogue of statistics, keyed by name.

    Returns
    -------
    dict
        Every statistic solprop knows about, implemented or reserved.
    """
    entries = (
        StatDefinition(
            name="omega0",
            summary="Centroid of the spectral energy density.",
            shape="(n_modes, n_pol)",
            units="rad/s",
            between_planes=True,
        ),
        StatDefinition(
            name="energy",
            summary="Modal column energies or whole spatial beam energy.",
            shape="(n_modes, n_pol); () for spatial beams",
            units="J",
            between_planes=True,
        ),
        StatDefinition(
            name="energy_window",
            summary="Pulse energy inside each spectral window.",
            shape="(n_windows, n_modes, n_pol)",
            units="J",
            requires=("windows",),
            between_planes=True,
        ),
        StatDefinition(
            name="peak_power",
            summary="Peak analytic Poynting power of each mode during a run.",
            shape="(n_modes,)",
            units="W",
            between_planes=True,
        ),
        StatDefinition(
            name="peak_power_window",
            summary="Peak analytic Poynting power per mode and spectral window.",
            shape="(n_windows, n_modes)",
            units="W",
            requires=("windows",),
            between_planes=True,
        ),
        StatDefinition(
            name="peak_power_total",
            summary="Peak total analytic Poynting power during a run.",
            shape="()",
            units="W",
            between_planes=True,
        ),
        StatDefinition(
            name="physical_energy",
            summary="Physical energy of a general modal basis, metric-weighted.",
            shape="()",
            units="J",
            between_planes=True,
        ),
        StatDefinition(
            name="spectral_flux",
            summary="Spectral density of a general modal basis's physical energy.",
            shape="(n_omega,)",
            units="J/(rad/s)",
        ),
        StatDefinition(
            name="analytic_flux",
            summary="Peak total analytic Poynting flux of a general modal basis.",
            shape="()",
            units="W",
            between_planes=True,
        ),
        StatDefinition(
            name="fwhm_t",
            summary="Central half-maximum width of the total Poynting power.",
            shape="()",
            units="s",
            between_planes=True,
        ),
        StatDefinition(
            name="fwhm_t_outer",
            summary="Total extent of Poynting power above half its maximum.",
            shape="()",
            units="s",
            between_planes=True,
        ),
        StatDefinition(
            name="density",
            summary="Number density of the filling gas at the plane.",
            shape="()",
            units="1/m^3",
            requires=("density",),
            between_planes=True,
        ),
        StatDefinition(
            name="pressure",
            summary="Pressure of the filling gas at the plane.",
            shape="()",
            units="bar",
            requires=("pressure",),
            between_planes=True,
        ),
        StatDefinition(
            name="core_radius",
            summary="Core radius of the waveguide at the plane.",
            shape="()",
            units="m",
            requires=("core_radius",),
            between_planes=True,
        ),
        StatDefinition(
            name="zdw",
            summary="Zero-dispersion wavelength of each mode at the plane.",
            shape="(n_modes,)",
            units="m",
            requires=("modes",),
        ),
        StatDefinition(
            name="species_ionisation",
            summary="Final ionised fraction of each species at every node.",
            shape="(n_species, n_nodes)",
            units="1",
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="species_electron_density",
            summary="Species electron densities at the densest total-electron node.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="species_peak_ionisation_rate",
            summary="Species peak rate over all nodes; cycle-averaged for envelopes.",
            shape="(n_species,)",
            units="1/s",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="species_density",
            summary="Component number densities at this plane.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="species_ionisation_energy",
            summary="Final ionised inventory energy per propagation length.",
            shape="(n_species,)",
            units="J/m",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="electron_density",
            summary="Free-electron density left behind by the pulse.",
            shape="()",
            units="1/m^3",
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="peak_ionisation_rate",
            summary="Peak ionisation rate; cycle-averaged for envelopes.",
            shape="()",
            units="1/s",
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="on_axis_electron_density",
            summary="Free-electron density left behind on the axis.",
            shape="()",
            units="1/m^3",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="on_axis_peak_ionisation_rate",
            summary="Peak rate on the axis; cycle-averaged for envelopes.",
            shape="()",
            units="1/s",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="on_axis_species_electron_density",
            summary="Each species' electron density left behind on the axis.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="on_axis_species_peak_ionisation_rate",
            summary="Species peak rate on the axis; cycle-averaged for envelopes.",
            shape="(n_species,)",
            units="1/s",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="plasma_undershoot",
            summary="Largest exposure undershoot of the plasma column's band integral.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="plasma_work_defect",
            summary="Loss-current work against the ionisation it paid, relative.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="plasma_boundary_level",
            summary="Free-current integrand at the window start, relative to its peak.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
        ),
        StatDefinition(
            name="peak_intensity",
            summary="Spatial vacuum intensity; modal on-axis analytic Poynting flux.",
            shape="()",
            units="W/m^2",
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="fwhm_r",
            summary="Central fluence FWHM; NaN if undefined.",
            shape="()",
            units="m",
            requires=("spatial_geometry",),
        ),
        StatDefinition(
            name="on_axis_intensity",
            summary="Largest intensity in time on the spatial beam axis.",
            shape="()",
            units="W/m^2",
            requires=("spatial_geometry",),
            between_planes=True,
        ),
        StatDefinition(
            name="on_axis_spectrum",
            summary="On-axis spectral fluence density summed over polarisation.",
            shape="(n_omega,)",
            units="J s/m^2",
            requires=("spatial_geometry",),
        ),
        StatDefinition(
            name="beam_radius",
            summary="Fluence second-moment and 1/e^2 radii; NaN if undefined.",
            shape="(2,)",
            units="m",
            requires=("spatial_geometry",),
        ),
        StatDefinition(
            name="aperture_spectrum",
            summary="Spectrum collected through each far-field aperture.",
            shape="(n_apertures, n_omega)",
            units="J/(rad/s)",
            requires=("apertures", "spatial_geometry"),
        ),
        StatDefinition(
            name="aperture_on_axis_spectrum",
            summary="On-axis spectral fluence behind each aperture in a Fourier plane.",
            shape="(n_apertures, n_omega)",
            units="J s/m^2",
            requires=("apertures", "spatial_geometry"),
        ),
        StatDefinition(
            name="region_spectrum",
            summary="Spectrum in each box of transverse wavevector.",
            shape="(n_regions, n_omega)",
            units="J/(rad/s)",
            requires=("regions", "spatial_geometry"),
        ),
        StatDefinition(
            name="beam_spectrum",
            summary="Whole-beam spectral energy density.",
            shape="(n_omega,)",
            units="J/(rad/s)",
            requires=("spatial_geometry",),
        ),
        StatDefinition(
            name="absorbed_power",
            summary="Power each absorbing channel of the linear operator removes.",
            shape="(4,)",
            units="J/m",
            requires=("spatial_geometry", "separable_operator"),
            backend=True,
            between_planes=True,
        ),
        StatDefinition(
            name="quadrature_error",
            summary="Relative error of the transverse quadrature on the polarisation.",
            shape="()",
            units="dimensionless",
            backend=True,
            between_planes=True,
        ),
    )
    return {entry.name: entry for entry in entries}


CATALOGUE: Mapping[str, StatDefinition] = _catalogue()
"""Every statistic solprop knows about, implemented or reserved, keyed by name."""

GENERAL_MODAL_ACCESSORS: frozenset[str] = frozenset(
    {
        # Statistics of the general basis, from the catalogue above.
        "omega0",
        "physical_energy",
        "spectral_flux",
        "analytic_flux",
        "fwhm_t",
        "fwhm_t_outer",
        "zdw",
        # Processing accessors of the general basis (solprop.processing.modal).
        "reconstructed_flux",
        "channel_labels",
        "coordinate_intensity",
        "instantaneous_flux",
        "electric_field",
        "magnetic_field",
        "plane_nodes",
    }
)
"""Every accessor a three-component general modal basis admits, by name.

The sets live here, beside the catalogue whose names they select from, and
:mod:`solprop.output.capability` imports them to build the gate: they are a statement
about statistics, and keeping them here is what lets
:mod:`solprop.translate.general` record the list a file publishes without a
translation module importing an output one.

``omega0``, ``fwhm_t`` and ``fwhm_t_outer`` are functionals of the same-frequency flux
``a.H W a`` and take the basis metric
(:data:`solprop.stats.METRIC_WEIGHTED_STATISTICS`), so they are exact on a declared
basis as well as on the identity one;
``zdw`` is a property of the channels' propagation constants and of no basis at all
-- until a declared basis couples channels whose propagation constants differ, when
the block executes through a coupled flow and each channel's constant is the block's
mean; the route then withholds it (:data:`WITHHELD_WITH_A_FLOW`).

``spectral_flux`` is the one array of the family -- the same flux as a density in
frequency rather than integrated. It
is **admitted but not a default** of the route: it is ``n_omega`` wide per plane where
the other five are scalars, so a run asks for it by name.

Two names of :mod:`solprop.processing.modal` are deliberately absent.
``quadrature_error`` is refused separately, by the driver, because what disqualifies
it is the absent embedded coarse rule and not the basis -- a general guide's node rule
has no embedded pair to difference, so such a run has no transverse-resolution alarm,
which is a known limitation -- and it sits
inside the run's resume key, so publishing it here would promise a statistic the
same record refuses. ``channel_amplitudes`` takes two arrays and a table rather than
an opened handler, so no door ever asks the gate about it; it is in
``solprop.output.capability._GENERAL_ONLY`` all the same, so that the one
direction that could be reached -- an ordinary run asked for it -- would be refused
for the right reason.
"""

WITHHELD_WITH_A_FLOW: frozenset[str] = frozenset({"zdw"})
"""The general-basis accessors a route withholds when its block couples, by name.

A declared change of basis over channels whose propagation constants differ executes
through ``linear.coupled_flow``, and the per-channel propagation constant the
statistics would read is then the block's mean: ``zdw`` from it is a number that
propagates nothing. The route leaves these names out of the record's published list and
every accessor door refuses them with :func:`solprop.output.capability.refusal_for`'s
``withheld``.
"""

ACCESSOR_ADVICE: Mapping[str, str] = {
    "energy": "physical_energy, which integrates the same-frequency metric product",
    "energy_window": (
        "physical_energy with an explicit band, or the reconstructed E/H of a saved "
        "plane"
    ),
    "peak_power": (
        "analytic_flux or reconstructed E/H with an explicit detector definition"
    ),
    "peak_power_total": (
        "analytic_flux or reconstructed E/H with an explicit detector definition"
    ),
    "peak_power_window": (
        "analytic_flux or reconstructed E/H with an explicit detector definition"
    ),
    "peak_intensity": (
        "analytic_flux or reconstructed E/H with an explicit detector definition"
    ),
}
"""What to use instead of each legacy name on a general modal basis.

A name with no entry here gets the generic advice of
:mod:`solprop.output.capability`: a general modal basis has ``physical_energy``,
``analytic_flux`` and the reconstruction accessors, and nothing else of that family.
"""

_GAS_OR_SPATIAL_NAMES: frozenset[str] = (
    frozenset({"electron_density", "peak_ionisation_rate"})
    | frozenset(PLASMA_INTEGRATION_STATISTICS)
    | frozenset(PROFILE_STATISTICS)
    | frozenset(name for name in CATALOGUE if name.startswith("species_"))
    | frozenset(
        name
        for name, entry in CATALOGUE.items()
        if "spatial_geometry" in entry.requires
    )
)
"""Names a general modal guide fails for a reason that is not its basis at all.

A plasma density, a gas pressure and a beam radius are unavailable because this guide
has no gas fill and no spatial geometry, not because its coordinate powers do not add
up; advising a reader to use ``analytic_flux`` instead of ``electron_density`` would
be nonsense. The gate names that reason instead.
"""

_STEP_DIAGNOSTICS: frozenset[str] = frozenset({"quadrature_error"})
"""Diagnostics of the step, which are the same quantity in any basis.

Basis-independent step diagnostics are admitted on a general basis, so the gate never
refuses one: whoever owns the real reason refuses it
instead. ``quadrature_error`` is refused on a general guide by the run loop, because a
provider's node rule carries no embedded coarse rule to difference --
a statement about the rule, not about the basis -- and it is deliberately not in
:data:`GENERAL_MODAL_ACCESSORS`, so no run record publishes as admitted a name the
same record's run refuses.
"""

_FIELD_STATISTICS: Mapping[
    str, Callable[[TimeGrid, Complex[np.ndarray, "n_modes n_pol n_omega"]], np.ndarray]
] = {
    "energy": energy,
    "peak_power": peak_power,
    "peak_power_total": peak_power_total,
}
"""Statistics of the saved field alone, each taking the grid and the field."""

_METRIC_STATISTICS: Mapping[
    str,
    Callable[..., np.ndarray],
] = {
    "omega0": omega0,
    "physical_energy": physical_energy,
    "spectral_flux": spectral_flux,
    "analytic_flux": analytic_flux,
    "fwhm_t": fwhm_t,
    "fwhm_t_outer": fwhm_t_outer,
}
"""Statistics of the saved field that also take the basis's flux metric.

The annotation is ``Callable[..., ndarray]`` rather than the exact signature because
a keyword-only parameter cannot be written in a ``Callable`` at all; every member's
signature is ``(grid, field, *, metric=None)`` and :func:`collect` calls them that way.
"""

METRIC_WEIGHTED_STATISTICS: frozenset[str] = frozenset(_METRIC_STATISTICS)
"""Statistics whose value depends on the basis's flux metric, by name.

These are the functionals of the physical same-frequency flux
``sum_mn conj(a_m) W_mn a_n`` -- its integral (``physical_energy``), its density in
frequency (``spectral_flux``), its peak (``analytic_flux``), its two widths
(``fwhm_t``, ``fwhm_t_outer``) and the spectral
centroid of its per-channel share (``omega0``). On an orthonormal basis ``W`` is the
identity and they are the unweighted quantities they have always been, bit for bit;
on a general modal basis whose Gram is not the identity they are wrong without it,
so a caller that has a metric must hand it over (:func:`collect`) and one that
cannot must refuse rather than answer (:mod:`solprop.processing.modal`).

``zdw`` is deliberately absent: it is a property of the channels' propagation
constants and of no basis at all, and a unitary change of basis of an exactly
degenerate block leaves every one of them unchanged.
"""

_WINDOW_STATISTICS: Mapping[
    str,
    Callable[
        [
            TimeGrid,
            Complex[np.ndarray, "n_modes n_pol n_omega"],
            Float[np.ndarray, "n_windows 2"],
        ],
        np.ndarray,
    ],
] = {"energy_window": energy_window, "peak_power_window": peak_power_window}
"""Statistics that also take the spectral windows of the spec."""


ApertureWeighting = Literal["amplitude", "quadrature"]
"""How an aperture's profile weights the collected spectrum (SEP 23 section 4.5)."""

MAX_APERTURES = 16
"""The most apertures one :class:`ApertureSet` carries."""

MAX_REGIONS = 16
"""The most rows :attr:`StatsSpec.spectral_regions` carries."""


def _aperture_array(
    values: object, shape: tuple[int, ...] | None, name: str
) -> np.ndarray:
    """Own a finite float64 copy of one aperture table, read-only.

    Parameters
    ----------
    values : object
        The table as given.
    shape : tuple of int or None
        The shape it must have; ``None`` asks for any flat vector.
    name : str
        The field, named in the refusal.

    Returns
    -------
    numpy.ndarray
        A read-only, C-contiguous float64 copy.

    Raises
    ------
    OutputError
        If the table is not numeric, finite and of the stated shape.
    """
    stated = "a flat vector" if shape is None else f"shape {shape}"
    try:
        array = np.array(values, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as error:
        msg = f"ApertureSet.{name} must be numeric, {stated}"
        raise OutputError(msg) from error
    fits = array.ndim == 1 if shape is None else array.shape == shape
    if not fits or not np.all(np.isfinite(array)):
        msg = f"ApertureSet.{name} must be finite, {stated}, got {array.shape}"
        raise OutputError(msg)
    array.setflags(write=False)
    return array


def check_aperture_choices(profile: object, weighting: object) -> None:
    """Refuse, by name, an aperture profile or weighting that does not exist.

    The check :class:`ApertureSet` makes of its two choices. It takes any object,
    because a caller outside the test session's runtime type checker can pass
    anything.

    Parameters
    ----------
    profile : object
        The edge profile asked for.
    weighting : object
        The weighting asked for.

    Raises
    ------
    OutputError
        If either is not one of its choices.

    Examples
    --------
    >>> from solprop.stats import check_aperture_choices
    >>> check_aperture_choices("tanh", "quadrature") is None
    True
    """
    check_aperture_profile(profile)
    if weighting not in ("amplitude", "quadrature"):
        msg = (
            f"unknown aperture weighting {weighting!r}; choose 'amplitude' or "
            "'quadrature'"
        )
        raise OutputError(msg)


@dataclass(frozen=True, eq=False)
class ApertureSet:
    """Circular apertures in a far-field or Fourier plane behind a free-space run.

    The tables of ``aperture_spectrum`` and ``aperture_on_axis_spectrum`` (SEP 23):
    each aperture collects the light whose paraxial image ``x = z c kappa / omega``
    falls inside it, a chromatic weight on the stored transverse spectrum. Every
    position is physical: a beam travelling towards positive x lands at positive x.
    Equality is by identity; the arrays are owned read-only copies.

    Attributes
    ----------
    centres : numpy.ndarray
        Aperture centres ``(x, y)`` in the mask plane, in m, shape
        ``(n_apertures, 2)``.
    diameters : numpy.ndarray
        Diameters in m, finite and positive, shape ``(n_apertures,)``. On a line the
        aperture is a slit of this full width along x.
    distances : numpy.ndarray
        Distance from the exit face to the mask plane, or a lens's focal length for
        a Fourier plane, in m, finite and positive, shape ``(n_apertures,)``.
    edges : numpy.ndarray or None
        Edge widths ``w`` in m for ``tanh`` and ``rcos``, orders ``n >= 2`` for
        ``supergauss``, shape ``(n_apertures,)``; ``None`` for ``hard``, and only
        for ``hard``.
    profile : {'hard', 'tanh', 'rcos', 'supergauss'}
        The edge profile the set shares (:func:`solprop.spatial.aperture_profile`).
        The default is ``hard``.
    weighting : {'amplitude', 'quadrature'}
        ``amplitude`` (the default) models a mask of amplitude transmission ``W``
        and weights the collected spectrum by ``W**2``; ``quadrature`` uses a smooth
        profile once, at the corrected diameter, as a quadrature weight for a hard
        hole. Quadrature is refused with ``supergauss``.
    labels : tuple of str
        One unique non-empty label per aperture, recorded with the run's
        configuration; ``None`` (the default) labels them ``"0"``, ``"1"``, ...

    Raises
    ------
    OutputError
        If there are no apertures or more than :data:`MAX_APERTURES`; an array is
        not finite or not of its stated shape; a diameter or distance is not
        positive; an edge is missing, given for ``hard``, or out of its range (``0 <
        w <= D/4`` for ``rcos`` and for ``tanh`` under amplitude weighting, ``0 < w
        <= D/8`` for ``tanh`` under quadrature weighting, ``n >= 2`` for
        ``supergauss``); quadrature weighting is asked of ``supergauss``; an
        aperture reaches the paraxial disc's edge, ``|X| + D/2 (+ w for rcos) >=
        z``; or the labels are not unique non-empty strings, one per aperture.

    Warns
    -----
    solprop.exceptions.ApertureEdgeWarning
        Once per set, for ``tanh`` under amplitude weighting with ``r/w < 4``
        (``w > D/8``): the profile slopes at its centre, so a bright beam there is
        collected with a lattice error near ``1e-5``. The set is built as asked;
        ``rcos`` is the recommended smooth profile.

    Examples
    --------
    >>> from solprop.stats import ApertureSet
    >>> holes = ApertureSet(
    ...     centres=[[4e-3, 4e-3], [-4e-3, -4e-3]],
    ...     diameters=[1.5e-3, 1.5e-3],
    ...     distances=[0.25, 0.25],
    ...     edges=[40e-6, 40e-6],
    ...     profile="tanh",
    ...     weighting="quadrature",
    ...     labels=("probe", "signal"),
    ... )
    >>> holes.n_apertures, holes.labels
    (2, ('probe', 'signal'))
    """

    centres: Sequence[Sequence[float]] | np.ndarray
    diameters: Sequence[float] | np.ndarray
    distances: Sequence[float] | np.ndarray
    edges: Sequence[float] | np.ndarray | None = None
    profile: ApertureProfile = "hard"
    weighting: ApertureWeighting = "amplitude"
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Own the tables and check them against the list of SEP 23 section 5.

        Raises
        ------
        OutputError
            Naming the field and the rule it breaks.
        """
        diameters = _aperture_array(self.diameters, None, "diameters")
        count = diameters.size
        if not 1 <= count <= MAX_APERTURES:
            msg = f"an ApertureSet holds 1 to {MAX_APERTURES} apertures, got {count}"
            raise OutputError(msg)
        object.__setattr__(self, "diameters", diameters)
        object.__setattr__(
            self, "centres", _aperture_array(self.centres, (count, 2), "centres")
        )
        object.__setattr__(
            self, "distances", _aperture_array(self.distances, (count,), "distances")
        )
        if np.any(np.asarray(self.diameters) <= 0) or np.any(
            np.asarray(self.distances) <= 0
        ):
            msg = "every aperture diameter and distance must be positive"
            raise OutputError(msg)
        check_aperture_choices(self.profile, self.weighting)
        self._check_edges(count)
        self._check_paraxial()
        labels = (
            tuple(str(index) for index in range(count))
            if self.labels is None
            else tuple(self.labels)
        )
        if (
            len(labels) != count
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != count
        ):
            msg = f"ApertureSet.labels must be {count} unique non-empty strings"
            raise OutputError(msg)
        object.__setattr__(self, "labels", labels)

    def _check_edges(self, count: int) -> None:
        """Own the edge table and check it against the profile and the weighting.

        Parameters
        ----------
        count : int
            The number of apertures.

        Raises
        ------
        OutputError
            If the edges disagree with the profile or the weighting.
        """
        if self.profile == "hard":
            if self.edges is not None:
                msg = (
                    "a hard aperture has no edge, and an edge given beside it would "
                    "be silently unused; pass edges=None"
                )
                raise OutputError(msg)
            return
        if self.edges is None:
            msg = f"a {self.profile} aperture needs edges, one per aperture"
            raise OutputError(msg)
        edges = _aperture_array(self.edges, (count,), "edges")
        object.__setattr__(self, "edges", edges)
        diameters = np.asarray(self.diameters)
        if self.profile == "supergauss":
            if self.weighting == "quadrature":
                msg = (
                    "quadrature weighting needs a symmetric edge, and a supergauss "
                    "profile has none; use amplitude weighting"
                )
                raise OutputError(msg)
            if np.any(edges < 2):
                msg = "every supergauss order must be at least 2"
                raise OutputError(msg)
            return
        # tanh under quadrature weighting keeps its tail's area bias below 2e-5.
        fraction = 8 if self.profile == "tanh" and self.weighting == "quadrature" else 4
        if np.any(edges <= 0) or np.any(edges > diameters / fraction):
            msg = (
                f"a {self.profile} aperture under {self.weighting} weighting needs "
                f"0 < w <= D/{fraction} for every aperture"
            )
            raise OutputError(msg)
        if self.profile == "tanh" and self.weighting == "amplitude":
            # tanh is not flat at its centre: W'(0) = -sech^2(r/w) / 2w, so for
            # r/w < 4 a bright beam there converges only at third order in the mask
            # plane's spacing (about 1e-5 of the collected energy at r/w = 2). At
            # r/w >= 4 the slope is 0.02 of that and the error near 2e-7.
            wide = np.flatnonzero(diameters / 2 < 4 * edges)
            if wide.size:
                warnings.warn(
                    f"tanh apertures {wide.tolist()} under amplitude weighting have "
                    "r/w < 4: the profile slopes at its centre, so light there is "
                    "collected with a lattice error near 1e-5; rcos is flat inside "
                    "r - w and is preferred (or w <= D/8)",
                    ApertureEdgeWarning,
                    stacklevel=4,
                )

    def _check_paraxial(self) -> None:
        """Check that every aperture lies inside the paraxial disc ``|x| < z``.

        Raises
        ------
        OutputError
            Naming the apertures that reach it.
        """
        centres = np.asarray(self.centres)
        reach = np.hypot(centres[:, 0], centres[:, 1]) + np.asarray(self.diameters) / 2
        if self.profile == "rcos":
            reach = reach + np.asarray(self.edges)
        outside = np.flatnonzero(reach >= np.asarray(self.distances))
        if outside.size:
            msg = (
                f"apertures {outside.tolist()} reach the paraxial disc's edge, "
                "|X| + D/2 (+ w for rcos) >= z; a component there cannot leave the "
                "exit face (SEP 23 section 4.2)"
            )
            raise OutputError(msg)

    @property
    def n_apertures(self) -> int:
        """Return how many apertures the set holds.

        Returns
        -------
        int
            The number of apertures.
        """
        return int(np.asarray(self.diameters).size)

    def same_as(self, other: "ApertureSet | None") -> bool:  # noqa: UP037
        """Return whether another set describes the same apertures, entry for entry.

        Parameters
        ----------
        other : ApertureSet or None
            The set to compare.

        Returns
        -------
        bool
            True when every table, the profile, the weighting and the labels agree
            exactly: a structural comparison of two requests.
        """
        if other is None:
            return False
        tables = ("centres", "diameters", "distances")
        return (
            all(
                np.array_equal(np.asarray(getattr(self, name)), getattr(other, name))
                for name in tables
            )
            and (self.edges is None) == (other.edges is None)
            and (
                self.edges is None
                or np.array_equal(np.asarray(self.edges), np.asarray(other.edges))
            )
            and (self.profile, self.weighting, self.labels)
            == (other.profile, other.weighting, other.labels)
        )


def _no_windows() -> Float[np.ndarray, "n_windows 2"]:
    """Return the empty window list a spec defaults to.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(0, 2)``.
    """
    return np.empty((0, 2))


def _no_regions() -> Float[np.ndarray, "n_regions 4"]:
    """Return the empty region table a spec defaults to.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(0, 4)``.
    """
    return np.empty((0, 4))


_APERTURE_NAMES = frozenset({"aperture_spectrum", "aperture_on_axis_spectrum"})
"""The statistics that read :attr:`StatsSpec.apertures`."""


@dataclass(frozen=True, eq=False)
class StatsSpec:
    """Which statistics a run asks for, and the windows and period they need.

    This is the object of the kernel contract §8: the translation layer carries it into
    a backend, the driver hands it to :func:`collect`, and both read the same names.
    Equality is by identity (``eq=False``) because ``energy_windows`` is an array.

    Attributes
    ----------
    names : tuple
        Statistics to evaluate at every saved plane, in the order they are evaluated.
        Every name must be in :data:`CATALOGUE` and must be implemented.
    energy_windows : numpy.ndarray
        Spectral windows, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)`` with
        ``lo < hi``. Infinite edges are allowed, so that a pair of windows can tile the
        whole axis. Required by ``energy_window`` and ``peak_power_window``.
    periodic : tuple
        The subset of ``names`` a backend also evaluates between planes, for step-level
        diagnostics. Must be a subset of ``names``, and every entry must carry
        :attr:`StatDefinition.between_planes`: a statistic whose cost is only bearable
        once per plane is refused here rather than made the run's per-step cost.
    period : int
        How many accepted steps apart the periodic subset is evaluated, at least 1.
    apertures : ApertureSet or None
        Far-field apertures, required by ``aperture_spectrum`` and
        ``aperture_on_axis_spectrum`` (SEP 23). ``None`` by default, and then left
        out of the run's configuration record, so every earlier record keeps its
        hash.
    spectral_regions : numpy.ndarray
        Boxes of physical transverse wavevector in rad/m, shape ``(n_regions, 4)``,
        rows ``(kx_lo, kx_hi, ky_lo, ky_hi)``, each a half-open box ``[lo, hi)`` on
        both axes, at most :data:`MAX_REGIONS` rows, infinite edges allowed.
        Required by ``region_spectrum``. Empty by default, and then left out of the
        configuration record.

    Raises
    ------
    OutputError
        If a name is unknown, reserved or repeated; if a window or a region is
        malformed; if ``periodic`` is not a subset of ``names`` or names a statistic
        no backend evaluates between planes; if ``period`` is below 1; or if a name
        needs a far-field table the spec does not carry.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stats import StatsSpec
    >>> spec = StatsSpec(
    ...     names=("energy", "energy_window"),
    ...     energy_windows=np.array([[2.0e15, 3.0e15]]),
    ... )
    >>> spec.n_windows
    1
    """

    names: tuple[str, ...] = ()
    energy_windows: Float[np.ndarray, "n_windows 2"] = field(
        default_factory=_no_windows
    )
    periodic: tuple[str, ...] = ()
    period: int = 1
    apertures: ApertureSet | None = None
    spectral_regions: Float[np.ndarray, "n_regions 4"] = field(
        default_factory=_no_regions
    )

    def __post_init__(self) -> None:
        """Check the names, the tables and the period, and freeze the tables.

        Raises
        ------
        OutputError
            If a name is unknown, reserved or repeated; if a window or a region is
            malformed; if ``periodic`` is not a subset of ``names`` or names a
            statistic no backend evaluates between planes; if ``period`` is below 1;
            or if a name needs a far-field table the spec does not carry.
        """
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "periodic", tuple(self.periodic))
        _check_names(self.names)
        extra = sorted(set(self.periodic) - set(self.names))
        if extra:
            msg = f"the periodic subset asks for statistics the run does not: {extra}"
            raise OutputError(msg)
        _check_between_planes(self.periodic)
        if self.period < 1:
            msg = f"the statistics period must be at least 1, got {self.period}"
            raise OutputError(msg)
        windows = np.array(self.energy_windows, dtype=np.float64, copy=True)
        _check_windows(windows)
        windows.setflags(write=False)
        object.__setattr__(self, "energy_windows", windows)
        regions = np.array(self.spectral_regions, dtype=np.float64, copy=True)
        _check_regions(regions)
        regions.setflags(write=False)
        object.__setattr__(self, "spectral_regions", regions)
        missing = (
            sorted(set(self.names) & _APERTURE_NAMES) if self.apertures is None else []
        )
        if "region_spectrum" in self.names and self.n_regions == 0:
            missing.append("region_spectrum")
        if missing:
            msg = (
                f"the statistics {missing} need a far-field table this spec does not "
                "carry: apertures=ApertureSet(...) for the aperture spectra, "
                "spectral_regions for region_spectrum"
            )
            raise OutputError(msg)

    @property
    def n_apertures(self) -> int:
        """Return how many far-field apertures the spec carries.

        Returns
        -------
        int
            The apertures of :attr:`apertures`, or zero without them.
        """
        return 0 if self.apertures is None else self.apertures.n_apertures

    @property
    def n_regions(self) -> int:
        """Return how many regions of transverse wavevector the spec carries.

        Returns
        -------
        int
            The rows of :attr:`spectral_regions`.
        """
        return int(self.spectral_regions.shape[0])

    def same_tables(self, other: "StatsSpec") -> bool:  # noqa: UP037
        """Return whether another spec carries the same far-field tables.

        Parameters
        ----------
        other : StatsSpec
            The spec to compare, for example a lockstep batch member's.

        Returns
        -------
        bool
            True when both carry no apertures or the same apertures entry for entry,
            and identical regions: a structural comparison.
        """
        same_apertures = (
            other.apertures is None
            if self.apertures is None
            else self.apertures.same_as(other.apertures)
        )
        return same_apertures and np.array_equal(
            self.spectral_regions, other.spectral_regions
        )

    @property
    def n_windows(self) -> int:
        """Return how many spectral windows the spec carries.

        Returns
        -------
        int
            The number of windows.
        """
        return int(self.energy_windows.shape[0])


def _check_names(names: Sequence[str]) -> None:
    """Check that every requested statistic exists, is implemented and appears once.

    Parameters
    ----------
    names : collections.abc.Sequence
        The requested names.

    Raises
    ------
    OutputError
        If a name is unknown, reserved for physics that does not exist yet, or repeated.
    """
    unknown = sorted(set(names) - set(CATALOGUE))
    if unknown:
        msg = f"unknown statistics {unknown}; the catalogue holds {sorted(CATALOGUE)}"
        raise OutputError(msg)
    reserved = sorted(name for name in names if not CATALOGUE[name].implemented)
    if reserved:
        msg = (
            f"the statistics {reserved} are reserved but not implemented: they need "
            "physics that is not on main yet"
        )
        raise OutputError(msg)
    repeated = sorted({name for name in names if list(names).count(name) > 1})
    if repeated:
        msg = f"each statistic may be asked for once; these repeat: {repeated}"
        raise OutputError(msg)


def _check_between_planes(names: Sequence[str]) -> None:
    """Check that every periodic statistic is one a backend evaluates between planes.

    Parameters
    ----------
    names : collections.abc.Sequence
        The periodic subset, already known to be a subset of the requested names.

    Raises
    ------
    OutputError
        Naming the statistics that cannot be evaluated between planes, and the ones
        that can. The refusal is by name and at construction because the alternative --
        evaluating them anyway -- can cost a fifth of a step: ``zdw`` is a root search
        per mode and does not belong on every accepted step.
    """
    refused = sorted(name for name in names if not CATALOGUE[name].between_planes)
    if refused:
        available = sorted(
            name for name, entry in CATALOGUE.items() if entry.between_planes
        )
        msg = (
            f"the statistics {refused} cannot be evaluated between save planes, only "
            f"at them; the periodic subset may name {available}"
        )
        raise OutputError(msg)


def _check_windows(windows: Float[np.ndarray, "n_windows 2"]) -> None:
    """Check that every spectral window is a properly ordered pair.

    Parameters
    ----------
    windows : numpy.ndarray
        Window edges, shape ``(n_windows, 2)``, rad/s.

    Raises
    ------
    OutputError
        If the array holds a NaN, or has a row whose lower edge is not below its upper
        edge. That it is a list of pairs is fixed by the annotation, as it is for
        :class:`solprop.stepper.tableau.Tableau`.
    """
    if np.isnan(windows).any():
        msg = "energy_windows must not hold a NaN edge"
        raise OutputError(msg)
    bad = np.flatnonzero(windows[:, 0] >= windows[:, 1])
    if bad.size:
        msg = (
            f"every energy window needs lo < hi; rows {bad.tolist()} do not: "
            f"{windows[bad].tolist()}"
        )
        raise OutputError(msg)


def _check_regions(regions: Float[np.ndarray, "n_regions 4"]) -> None:
    """Check that every region of transverse wavevector is a properly ordered box.

    Parameters
    ----------
    regions : numpy.ndarray
        Region edges, shape ``(n_regions, 4)``, rad/m.

    Raises
    ------
    OutputError
        If there are more than :data:`MAX_REGIONS` rows, an edge is NaN, or a row's
        lower edge is not below its upper edge on either axis. That it is a list of
        quadruples is fixed by the annotation, as for the energy windows.
    """
    if regions.shape[0] > MAX_REGIONS:
        msg = (
            f"spectral_regions holds at most {MAX_REGIONS} rows, got {regions.shape[0]}"
        )
        raise OutputError(msg)
    if np.isnan(regions).any():
        msg = "spectral_regions must not hold a NaN edge"
        raise OutputError(msg)
    bad = np.flatnonzero(
        (regions[:, 0] >= regions[:, 1]) | (regions[:, 2] >= regions[:, 3])
    )
    if bad.size:
        msg = (
            f"every spectral region needs lo < hi on both axes; rows {bad.tolist()} "
            f"do not: {regions[bad].tolist()}"
        )
        raise OutputError(msg)


def _profile_for(profiles: Mapping[str, ZProfile] | None, name: str) -> ZProfile:
    """Return the z-profile one statistic needs, or say which one is missing.

    Parameters
    ----------
    profiles : collections.abc.Mapping or None
        The profiles the run has, keyed by statistic name.
    name : str
        The statistic asking.

    Returns
    -------
    solprop.profiles.ZProfile
        The profile.

    Raises
    ------
    OutputError
        If no profile of that name was given.
    """
    if profiles is None or name not in profiles:
        available = sorted(profiles) if profiles else []
        msg = (
            f"the {name!r} statistic needs a {name!r} z-profile; "
            f"collect was given {available}"
        )
        raise OutputError(msg)
    return profiles[name]


def collect(  # noqa: PLR0913 - a spec, a grid, a plane and four optional sources
    spec: StatsSpec,
    grid: TimeGrid,
    plane: SavedPlane,
    *,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    flux: SpatialFlux | None = None,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate every statistic a spec asks for at one saved plane.

    Each statistic uses its named public array function and deterministic reductions.
    For saved true-power modal amplitudes, temporal array functions use ``|b(t)|²``.
    Backend run statistics use analytic Poynting flux: their time integrals agree,
    but broadband peaks and widths can differ (``docs/dev/conventions.md`` section 4).

    Parameters
    ----------
    spec : StatsSpec
        Which statistics to evaluate, and the windows they need.
    grid : solprop.grid.TimeGrid
        The grid the plane's field lives on.
    plane : solprop.output.schema.SavedPlane
        The plane, whose field is in the internal FFT convention.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order. Needed by ``zdw``.
    profiles : collections.abc.Mapping, optional
        The z-profiles, keyed by statistic name. Needed by ``density``, ``pressure``
        and ``core_radius``.
    flux : solprop.spatial.SpatialFlux or None, optional
        The plane's measure for a radial or Cartesian spectrum, resolved at the
        plane's own position (and jump side) in its medium: every spatial statistic
        is that medium's axial Poynting flux (SEP 34), and widths use the signed
        time-integrated fluence.
        The far-field statistics ``aperture_spectrum``,
        ``aperture_on_axis_spectrum``, ``region_spectrum`` and ``beam_spectrum``
        need a Cartesian grid or a line (SEP 23), and are formed in one pass by
        :func:`solprop.spatial.aperture_statistics`, the function a run uses.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)`` at this plane, shape
        ``(n_modes, n_modes, n_omega)``, for the names in
        :data:`METRIC_WEIGHTED_STATISTICS`; ``None``, the default, is the identity
        and every statistic is then bitwise what it was. A caller that holds a
        general modal problem gets it from
        :func:`solprop.translate.coupled.metric_at`.

    Returns
    -------
    dict
        The value of each requested statistic, keyed by name, always as a NumPy array
        and zero-dimensional for a scalar. The shape of each is the ``shape`` field of
        its :class:`StatDefinition`.

    Raises
    ------
    OutputError
        If a statistic's dependency is missing -- no windows for a windowed statistic,
        no modes for ``zdw``, or no profile for a profile statistic -- or if it is one
        a backend owns, which this function cannot form from a saved plane
        (``absorbed_power`` among them, refused by name because a plane does not
        carry the linear operator's rates); if a far-field statistic
        is asked of a radial grid, or of a line with an aperture off ``y = 0`` or a
        region excluding ``kappa_y = 0``; or if the plane kept no field (SEP 22).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.schema import SavedPlane
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import collect, default_spec
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> plane = SavedPlane(z=0.0, field=built)
    >>> values = collect(default_spec("mode_averaged"), grid, plane)
    >>> sorted(values)
    ['energy', 'fwhm_t', 'fwhm_t_outer', 'omega0', 'peak_power']
    >>> values["energy"].shape, values["fwhm_t"].shape
    ((1, 1), ())
    """
    field = plane.field
    if field is None:
        msg = (
            "this plane kept no field (a run saved with fields='none' or 'planes'), "
            "so no statistic can be formed from it; read the statistics the run "
            "stored, output['stats/<name>']"
        )
        raise OutputError(msg)
    values: dict[str, np.ndarray] = {}
    # The far-field statistics share one pass over the columns (SEP 23).
    far_field = (
        _far_field_statistics(spec, flux, field)
        if flux is not None and _SAVED_FAR_FIELD & set(spec.names)
        else {}
    )
    for name in spec.names:
        if flux is not None and name not in PROFILE_STATISTICS:
            values[name] = (
                far_field[name]
                if name in far_field
                else _spatial_statistic(name, flux, field)
            )
        elif "spatial_geometry" in CATALOGUE[name].requires:
            msg = f"the {name!r} statistic requires spatial geometry"
            raise OutputError(msg)
        elif name in _METRIC_STATISTICS:
            values[name] = _METRIC_STATISTICS[name](grid, field, metric=metric)
        elif name in _FIELD_STATISTICS:
            values[name] = _FIELD_STATISTICS[name](grid, field)
        elif name in _WINDOW_STATISTICS:
            if spec.n_windows == 0:
                msg = f"the {name!r} statistic needs at least one energy window"
                raise OutputError(msg)
            values[name] = _WINDOW_STATISTICS[name](grid, field, spec.energy_windows)
        elif CATALOGUE[name].backend:
            msg = (
                f"the {name!r} statistic is computed by the backend from its own "
                "buffers, not from a saved plane; ask for it through "
                "solprop.backends.protocol.Backend.stats"
            )
            raise OutputError(msg)
        elif name in PROFILE_STATISTICS:
            values[name] = profile_value(_profile_for(profiles, name), plane.z)
        else:
            # Only "zdw" is left: the spec has already checked every name against the
            # catalogue, and every other implemented entry is routed above.
            if modes is None:
                msg = "the 'zdw' statistic needs the propagated modes"
                raise OutputError(msg)
            values[name] = zdw(
                modes, plane.z, wavelength_range=inband_wavelengths(grid)
            )
    logger.debug("collected %d statistics at z = %.6e m", len(values), plane.z)
    return values


_POYNTING_STATISTICS = frozenset(
    (
        "energy",
        "energy_window",
        "omega0",
        "peak_power",
        "peak_power_total",
        "peak_power_window",
        "fwhm_t",
        "fwhm_t_outer",
        "peak_intensity",
    )
)

_GENERAL_FLUX_STATISTICS = frozenset(
    ("physical_energy", "spectral_flux", "analytic_flux")
)
"""The three general modal statistics, which are **not** in `_POYNTING_STATISTICS`.

They are evaluated by :func:`collect` from the saved field alone, because the general
modal route's ``linear.power_weight`` is the constant one: the
Poynting amplitude is the true-power amplitude there. A weight that is not one would
make their arithmetic silently wrong rather than obviously so -- none of them is in
the set a backend routes through :func:`_poynting_statistics` -- so
:func:`weight_refusal_for` guards them.
"""

WEIGHT_DEPENDENT_STATISTICS = _POYNTING_STATISTICS | _GENERAL_FLUX_STATISTICS
"""Names whose value depends on the modal power weight, so a backend evaluates it."""


def weight_refusal_for(
    names: Sequence[str], weight: Float[np.ndarray, "n_modes n_pol n_omega"]
) -> str | None:
    """Return why a general statistic cannot be formed under this weight, or ``None``.

    :func:`physical_energy`, :func:`spectral_flux` and :func:`analytic_flux` evaluate
    the identity-metric,
    unit-weight arithmetic, which is what the general modal route certifies before it
    will run (``linear.power_weight`` is the constant one there). This is
    **unreachable today**: no route carries both a modal table and a non-constant
    weight, and the translation layer refuses to build one. It is written because the
    failure would be silent rather than loud if a later change loosened either side
    -- the three names are outside the set a backend routes through the
    weighted Poynting pass, so they would be answered from the unweighted field.

    Parameters
    ----------
    names : collections.abc.Sequence
        The statistic names asked for.
    weight : numpy.ndarray
        The modal power weight at this plane, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    str or None
        The refusal, or ``None`` when no general statistic is asked for or the
        weight is the constant one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stats import weight_refusal_for
    >>> weight_refusal_for(["physical_energy"], np.ones((1, 1, 4))) is None
    True
    >>> weight_refusal_for(["energy"], np.full((1, 1, 4), 2.0)) is None
    True
    """
    asked = [name for name in names if name in _GENERAL_FLUX_STATISTICS]
    if not asked or bool(np.all(weight == 1.0)):
        return None
    return (
        f"{sorted(asked)[0]} is unavailable for this run: it is the identity-metric, "
        "unit-power-weight quantity of a general modal basis, and this problem's "
        "linear.power_weight is not the constant one, so the arithmetic would be the "
        "unweighted field's and the answer would be silently wrong. A general modal "
        "guide's Poynting amplitude is its true-power amplitude; use energy or "
        "peak_power_total, which carry the weight"
    )


def _poynting_statistics(
    spec: StatsSpec,
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    weight: Float[np.ndarray, "n_modes n_pol n_omega"],
    axis: Float[np.ndarray, "2 n_modes"] | None = None,
) -> dict[str, np.ndarray]:
    """Share cycle-averaged analytic Poynting power and on-axis intensity.

    The real instantaneous sum-frequency term is excluded. Negative broadband
    excursions of the Hermitian flux are retained, without clipping.

    Parameters
    ----------
    spec : StatsSpec
        Requested statistics and half-open spectral windows.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    field : numpy.ndarray
        Internal modal spectrum, shape (n_modes, n_pol, n_omega).
    weight : numpy.ndarray
        Positive power weight, with the same modal shape.
    axis : numpy.ndarray or None
        Physical on-axis synthesis matrix, shape (2, n_modes).

    Returns
    -------
    dict
        Requested energy, temporal flux and axial intensity statistics.
    """
    names = set(spec.names) & _POYNTING_STATISTICS
    values: dict[str, np.ndarray] = {}
    density = np.abs(field) ** 2 * weight
    if "omega0" in names:
        total = np.sum(density, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values["omega0"] = np.sum(density * grid.omega, axis=-1) / total
    spectral = (
        density
        if grid.kind == "envelope"
        else density * np.concatenate(([1.0], np.full(grid.n_omega - 2, 4.0), [1.0]))
    )
    if "energy" in names:
        values["energy"] = (grid.dt / grid.n_t) * np.sum(spectral, axis=-1)
    if names & {"energy_window", "peak_power_window"} and spec.n_windows == 0:
        msg = "windowed Poynting statistics need at least one energy window"
        raise OutputError(msg)
    masks = _window_masks(grid, spec.energy_windows)
    if "energy_window" in names:
        values["energy_window"] = np.stack(
            [(grid.dt / grid.n_t) * np.sum(spectral * mask, axis=-1) for mask in masks]
        )
    if names - {"omega0", "energy", "energy_window", "peak_power_window"}:
        analytic = analytic_field(grid, field)
        weighted = analytic_field(grid, field * weight)
        power = np.sum(np.real(analytic * np.conj(weighted)), axis=1)
        total_power = np.sum(power, axis=0)
        if "peak_power" in names:
            values["peak_power"] = np.max(power, axis=-1)
        if "peak_power_total" in names:
            values["peak_power_total"] = np.asarray(np.max(total_power))
        for name in ("fwhm_t", "fwhm_t_outer"):
            if name in names:
                values[name] = _power_half_width(
                    grid, total_power, outermost=name.endswith("outer")
                )
        if "peak_intensity" in names:
            if axis is None:
                msg = "Poynting peak_intensity requires the on-axis synthesis matrix"
                raise OutputError(msg)
            electric = np.tensordot(axis, analytic, axes=([1], [0]))
            magnetic = np.tensordot(axis, weighted, axes=([1], [0]))
            flux = np.sum(np.real(electric * np.conj(magnetic)), axis=(0, 1))
            values["peak_intensity"] = np.asarray(0.5 * EPS0 * C * np.max(flux))
    if "peak_power_window" in names:
        values["peak_power_window"] = np.stack(
            [
                np.max(
                    np.sum(
                        np.real(
                            analytic_field(grid, field * mask)
                            * np.conj(analytic_field(grid, field * weight * mask))
                        ),
                        axis=1,
                    ),
                    axis=-1,
                )
                for mask in masks
            ]
        )
    return values


_SAVED_FAR_FIELD = frozenset(FAR_FIELD_STATISTICS) - frozenset(OPERATOR_STATISTICS)
"""The far-field statistics a saved Cartesian plane determines (SEP 23 section 5)."""


def _far_field_statistics(
    spec: StatsSpec,
    flux: SpatialFlux,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> dict[str, np.ndarray]:
    """Form every saved-field far-field statistic a spec asks for, in one pass.

    Parameters
    ----------
    spec : StatsSpec
        The request, with its far-field tables.
    flux : solprop.spatial.SpatialFlux
        The plane's measure; a radial grid is refused by name.
    field : numpy.ndarray
        Internal Cartesian spectrum in V/m, shape ``(n_x*n_y, n_pol, n_omega)``.

    Returns
    -------
    dict
        The requested statistics of :data:`_SAVED_FAR_FIELD`, keyed by name.
    """
    # The translation layer imports this module, so its tables are reached lazily.
    from solprop.translate.spatial import aperture_tables  # noqa: PLC0415

    names = [name for name in spec.names if name in _SAVED_FAR_FIELD]
    space = flux.space
    if isinstance(space, PlanarGrid):
        msg = "the far-field statistics read a Cartesian or line grid, not a planar one"
        raise OutputError(msg)
    tables = aperture_tables(spec, flux.grid, space)
    assert tables is not None  # the spec names a far-field statistic
    return aperture_statistics(
        names,
        flux.grid,
        field,
        tables,
        axial=lambda columns: (flux.admittance(columns), None),
    )


def _spatial_statistic(
    name: str,
    flux: SpatialFlux,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> np.ndarray:
    """Evaluate a spatial saved-field statistic through its physics reference.

    Parameters
    ----------
    name : str
        Requested spatial catalogue name.
    flux : solprop.spatial.SpatialFlux
        The plane's measure, binding its grids and medium.
    field : numpy.ndarray
        Internal spectrum, shape (n_modes, n_pol, n_omega): a Hankel spectrum in
        V m or a raw Cartesian DFT spectrum in V/m.

    Returns
    -------
    numpy.ndarray
        Scalar or array value in the catalogue's units.
    """
    if name == "energy":
        return np.asarray(spatial_energy(field, flux))
    if name == "on_axis_intensity":
        return np.asarray(on_axis_intensity(field, flux))
    if name == "on_axis_spectrum":
        return on_axis_spectrum(field, flux)
    if name == "peak_intensity":
        local = (
            planar_intensity(field, flux)
            if isinstance(flux.space, PlanarGrid)
            else spatial_intensity(field, flux)
        )
        return np.asarray(np.max(local))
    if name == "beam_radius":
        return np.asarray(
            [beam_radius(field, flux), beam_radius(field, flux, method="1/e2")]
        )
    if name == "fwhm_r":
        return np.asarray(fwhm_r(field, flux))
    if name in OPERATOR_STATISTICS:
        msg = (
            f"the {name!r} statistic needs the run's separable linear operator, which "
            "a saved plane does not carry; a backend forms it during the run, so read "
            f"the value the run stored, output['stats/{name}']"
        )
        raise OutputError(msg)
    msg = f"the {name!r} statistic is not a spatial saved-field statistic"
    raise OutputError(msg)


def default_spec(
    kind: SimulationKind, *, with_profiles: bool = False, with_zdw: bool = False
) -> StatsSpec:
    """Return the statistics a run of the given kind records unless told otherwise.

    The field statistics every run wants, plus -- for a multimode run, where the modes
    peak at different positions in time -- the peak of the total power. The z-profile
    statistics and the zero-dispersion wavelength are opt-in: the first needs profiles
    the driver may not have, and the second is a root search per mode per plane.

    Parameters
    ----------
    kind : {'mode_averaged', 'multimode', 'radial', 'free'}
        Which transform the run uses.
    with_profiles : bool, optional
        Whether to record the gas density, the pressure and the core radius for
        modal runs, or only the gas density for spatial runs.
    with_zdw : bool, optional
        Whether to record the zero-dispersion wavelength of each mode. Spatial
        runs reject this option because they carry no modal dispersion.

    Returns
    -------
    StatsSpec
        The spec, with no energy windows; add those for the run's bands of interest.

    Examples
    --------
    >>> from solprop.stats import default_spec
    >>> default_spec("mode_averaged").names
    ('omega0', 'energy', 'peak_power', 'fwhm_t', 'fwhm_t_outer')
    >>> "peak_power_total" in default_spec("multimode").names
    True
    """
    if kind in ("radial", "free"):
        if with_zdw:
            msg = "spatial statistics do not define a modal zdw"
            raise OutputError(msg)
        names = [
            "energy",
            "on_axis_intensity",
            "peak_intensity",
            "beam_radius",
            "fwhm_r",
            "on_axis_spectrum",
        ]
        if with_profiles:
            names.append("density")
        return StatsSpec(names=tuple(names))
    names = ["omega0", "energy", "peak_power"]
    if kind == "multimode":
        names.append("peak_power_total")
    names += ["fwhm_t", "fwhm_t_outer"]
    if with_profiles:
        names += list(PROFILE_STATISTICS)
    if with_zdw:
        names.append("zdw")
    return StatsSpec(names=tuple(names))

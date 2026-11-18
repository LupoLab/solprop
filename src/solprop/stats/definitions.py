# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference definition of each plane statistic, on internal spectra.

Every definition is written once, on internal FFT-convention arrays of shape
``(n_modes, n_pol, n_omega)``, reduces in a fixed order and is what a backend's fused
pass is validated against; an undefined value is ``nan``, a missing dependency an
:class:`solprop.exceptions.OutputError`.

- The analytic signal and centroid: :func:`analytic_field`, :func:`omega0`.
- Energy and power: :func:`energy`, :func:`energy_window`, :func:`peak_power`,
  :func:`peak_power_total`, :func:`peak_power_window`, and the flux of a general modal
  run, :func:`physical_energy`, :func:`spectral_flux`, :func:`analytic_flux`.
- Widths and the dispersion: :func:`fwhm_t`, :func:`fwhm_t_outer`, :func:`zdw`,
  :func:`inband_wavelengths`; z-profiles and species: :func:`profile_value`,
  :func:`species_statistics`.
"""

import logging
from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop import fourier
from solprop.constants import C
from solprop.exceptions import OutputError
from solprop.grid import TimeGrid
from solprop.modes import base as modes_base
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile

__all__ = [
    "DEFAULT_ZDW_RANGE",
    "analytic_field",
    "analytic_flux",
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
    # A running sum over the nodes: the same additions, in the same order, as a Python
    # loop over them, which took 0.2 s per species at 768 x 768 nodes.
    energy = np.zeros(len(densities))
    for index, (row, potential) in enumerate(zip(electrons, potentials, strict=True)):
        energy[index] = np.add.accumulate(weights * row * potential)[-1]
    return {
        "species_ionisation": fractions.copy(),
        "species_electron_density": electrons[:, densest].copy(),
        "species_peak_ionisation_rate": peak_rates.copy(),
        "species_density": densities.copy(),
        "species_ionisation_energy": energy,
        "electron_density": np.asarray(total[densest]),
        "peak_ionisation_rate": np.asarray(np.max(peak_rates)),
    }


DEFAULT_ZDW_RANGE = (150e-9, 3e-6)
"""Search window of :func:`zdw` when no narrower one is given, m: the default of
:func:`solprop.modes.base.zdw`."""


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

    - ``field``: stored spectrum, shape ``(..., n_omega)``, internal convention, in the
      square root of watts.
    - Returns: ``P(t) = |E_a(t)|^2``, shape ``(..., n_t)``, W.
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

    - ``field``: stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal
      convention.
    - ``metric``: the basis flux metric ``W(omega)``, Hermitian positive definite, shape
      ``(n_modes, n_modes, n_omega)``, dimensionless; ``None`` is the identity.
    - Returns: ``c = W^(1/2) a``, of the field's own shape and units.
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

    - ``grid``: the grid, whose ``omega`` axis the gate is taken on.
    - ``windows``: window edges, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)``.
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

    - Returns: the lower and upper crossings, both ``nan`` when the trace carries
      nothing, does not fall to half its maximum on both sides, or reaches the end of
      the axis still above it.
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

    - ``field``: stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal
      convention.
    - ``metric``: the basis flux metric ``W(omega)``; ``None`` is the identity.
    - Returns: the width as a zero-dimensional array, s, or ``nan`` when the trace does
      not cross half its maximum on both sides of the peak.
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

    - ``outermost``: use the outermost half-maximum crossings when true.
    - Returns: scalar duration in seconds, NaN when crossings are absent.
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

    The search window :func:`zdw` uses by default when :func:`~solprop.stats.collect`
    calls it: a zero-dispersion point outside the simulated band says nothing about the
    run, and hunting for one there drives every material model into extrapolation.

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
    wavelength_limits: tuple[float, float] = DEFAULT_ZDW_RANGE,
    near: float | None = None,
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
    wavelength_limits : tuple of float, optional
        Shortest and longest wavelength in m to search. Defaults to
        :data:`DEFAULT_ZDW_RANGE`, 150 nm to 3 um; :func:`~solprop.stats.collect`
        narrows it to the grid's own
        band with :func:`inband_wavelengths`, which is where the answer means anything
        and where the material models are not extrapolating.
    near : float, optional
        Reference wavelength in m: of several zeros, return the one nearest it that
        no resonance separates from it (:func:`solprop.modes.base.zdw`).
        :func:`~solprop.stats.collect` passes the grid's reference wavelength.
        ``None`` refuses several zeros.

    Returns
    -------
    numpy.ndarray
        Zero-dispersion wavelength, shape ``(n_modes,)``, m.

    Raises
    ------
    solprop.exceptions.NoZeroDispersionError
        If the search finds no zero-dispersion point of a mode in the window, rather
        than a missing value that would propagate silently.
        :func:`~solprop.stats.collect` records ``nan`` with a warning instead, so that
        a run is not stopped by its band.
    solprop.exceptions.ModeError
        If a mode has several zeros in the window and `near` is ``None``.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.stats import zdw
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> f"{zdw([mode], 0.0, wavelength_limits=(150e-9, 900e-9))[0] * 1e9:.1f}"
    '376.7'
    """
    return np.asarray(
        [
            modes_base.zdw(mode, z, wavelength_limits=wavelength_limits, near=near)
            for mode in modes
        ]
    )

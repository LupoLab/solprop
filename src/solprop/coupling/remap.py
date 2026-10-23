# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Move a field plane onto another temporal lattice, or another reference.

The one controlled temporal conversion of SEP 31 (section 4.5). Three things are
kept apart: the sign convention (a plane's samples are already optics-convention),
the raw DFT's time origin (the window centre) and its amplitude scaling. With
:math:`s_k` a plane's stored spectrum, its normalised analytic spectral density is

.. math:: f_k = \frac{\Delta t}{\sqrt{2\pi}}\,q_k s_k,

:math:`q_k = 2` on interior positive bins of a field grid, 1 at its zero-frequency
and Nyquist bins, and 1 on an envelope grid, so that
:math:`\Delta\omega\sum_k f_k^\dagger Q_k f_k` is the energy in the native metric Q.
Copying raw DFT amplitudes between lattices of different N or Δt is not this
operation; copying f between identical physical frequencies is.

**The map.** The source record is the analytic signal at its own time samples,
:math:`a_j = N^{-1}\sum_k q_k s_k e^{-i\omega_k t_j}`, zero outside its window. A
target bin of absolute frequency :math:`\omega'_m` receives its finite-time
Fourier sum

.. math:: s'_m = \frac{\Delta t}{\Delta t'\,q'_m}\,e^{i\Delta\phi}
          \sum_j a_j\,e^{i\omega'_m (t_j + \Delta)},

with :math:`\Delta = \tau - \tau'` and :math:`\Delta\phi = \phi - \phi'` the change of
reference (:class:`~solprop.coupling.PlaneReference`), so that the absolute spectrum is
unchanged. The sum is evaluated only on target bins inside the source lattice's own
span: beyond it the sum of samples spaced Δt repeats with period 2π/Δt and returns
aliases of the record, not light. The direct sum is the reference implementation;
it is evaluated in bounded chunks of target bins. The one acceleration is the
identical lattice in an unchanged reference, where the map is the identity and the
samples are copied; the tests check it against the direct sum. Complex spectral
zeros need no special case: no phase is unwrapped and nothing is interpolated.

**What is lost is reported, and what cannot be represented is refused.** Light
outside the target's active band, or outside its time window after the shift, is
removed only on request (`crop_band`, `truncate`), with its share in the report. A
record whose edges carry light, or whose exterior is unknown, is not extended into a
larger window without `truncate`, which declares it zero outside. Nothing is
renormalised: the energy out is whatever the map gives.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float

from solprop.coupling.plane import (
    FieldPlane,
    PlaneMetric,
    PlaneReference,
    lattice_identity,
)
from solprop.exceptions import CouplingError
from solprop.grid import TimeGrid
from solprop.spatial import SpatialFlux

__all__ = ["DEFAULT_TOLERANCE", "RegridReport", "regrid"]

DEFAULT_TOLERANCE = 1e-10
"""Share of a record's norm below which a band or a window is unoccupied.

Ten digits below the record: well above the rounding of a double-precision sum over
a few thousand samples (about 1e-13 relative), and well below any share a person
would call physical light. A share above it is occupied, and removing it is lossy.
"""

_CHUNK_ELEMENTS = 2**22
"""Largest number of complex exponentials held at once by the direct sum (64 MiB)."""


@dataclass(frozen=True)
class RegridReport:
    """What a temporal conversion did, and what it removed.

    The shares are of the source's **record norm**: the time-integrated squared
    analytic signal of every column, weighted by the column's transverse weight
    (one for a modal basis, the Parseval weight times ``eps0 c / 2`` for a spatial
    one), so that on an identity metric it is the energy in joules, and on a
    spatial one the vacuum-equivalent energy. The energies are physical.

    Attributes
    ----------
    method : {'copy', 'direct'}
        Identity copy on an identical lattice and reference, or the direct sum.
    source_lattice, target_lattice : str
        :func:`~solprop.coupling.lattice_identity` of the two grids.
    shift : float
        Record-time shift applied, ``old delay - new delay``, s.
    phase_shift : float
        Phase applied, ``old phase - new phase``, rad.
    record_norm : float
        The source's record norm, in its units (above).
    window_loss : float
        Share of the record norm outside the target window after the shift.
    band_loss : float
        Share of the record norm outside the target's active band.
    edge_share : float
        Share of the record norm inside the source's temporal collar.
    truncated : bool
        Whether the record was declared zero outside its window or cropped to the
        target's (`truncate`).
    energy_in : float or None
        Physical energy of the source, J; None when its metric is unknown.
    energy_out : float or None
        Physical energy of the result, J; None when its metric is unknown.
    """

    method: Literal["copy", "direct"]
    source_lattice: str
    target_lattice: str
    shift: float
    phase_shift: float
    record_norm: float
    window_loss: float
    band_loss: float
    edge_share: float
    truncated: bool
    energy_in: float | None
    energy_out: float | None


def _q(grid: TimeGrid) -> Float[np.ndarray, " n_omega"]:
    """Return the analytic-signal factor of each stored bin.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid.

    Returns
    -------
    numpy.ndarray
        Two on interior bins of a field grid, one at its end bins, one on an
        envelope grid; shape ``(n_omega,)``.
    """
    factor = np.ones(grid.n_omega)
    if grid.kind == "field":
        factor[1:-1] = 2.0
    return factor


def _analytic(
    samples: Complex[np.ndarray, "n_columns n_omega"], grid: TimeGrid
) -> Complex[np.ndarray, "n_columns n_t"]:
    """Return a record's analytic signal at its own time samples.

    ``a_j = sum_k q_k s_k exp(-i omega_k t_j) / N`` in the optics convention. With
    ``omega_k = omega_c + kappa_k d_omega`` (``omega_c`` the carrier of an envelope
    grid, zero on a field grid) and ``t_j = t_0 + j dt`` it is one FFT.

    Parameters
    ----------
    samples : numpy.ndarray
        Stored spectra, shape ``(n_columns, n_omega)``.
    grid : solprop.grid.TimeGrid
        Their grid.

    Returns
    -------
    numpy.ndarray
        The analytic signal, shape ``(n_columns, n_t)``.
    """
    centre = grid.omega0 if grid.kind == "envelope" else 0.0
    kappa = np.rint((grid.omega - centre) / grid.d_omega).astype(np.int64)
    full = np.zeros((samples.shape[0], grid.n_t), dtype=np.complex128)
    origin = np.exp(-1j * kappa * grid.d_omega * float(grid.t[0]))
    full[:, kappa % grid.n_t] = samples * (_q(grid) * origin)[None, :]
    carrier = np.exp(-1j * centre * grid.t)
    return carrier[None, :] * np.fft.fft(full, axis=-1) / grid.n_t


def _column_weights(plane: FieldPlane) -> Float[np.ndarray, " n_modes"]:
    """Return the transverse weight each column's record norm carries.

    Parameters
    ----------
    plane : FieldPlane
        The source.

    Returns
    -------
    numpy.ndarray
        One per column: the metric's Parseval weights times its scale for a spatial
        plane, ones otherwise.
    """
    metric = plane.metric
    if metric.weights is not None:
        return metric.scale * metric.weights
    return np.ones(plane.samples.shape[0])


def _target_metric(plane: FieldPlane, grid: TimeGrid, same: bool) -> PlaneMetric:
    """Return the result's metric on the target lattice.

    Parameters
    ----------
    plane : FieldPlane
        The source.
    grid : solprop.grid.TimeGrid
        The target grid.
    same : bool
        Whether the lattices are identical.

    Returns
    -------
    PlaneMetric
        The identity kept, an admittance resolved again from the plane's medium, the
        source's metric on an identical lattice, or ``unknown``.
    """
    metric = plane.metric
    if metric.kind in ("identity", "unknown") or same:
        return metric
    if metric.flux is not None:
        flux = metric.flux
        return PlaneMetric.from_flux(
            SpatialFlux.resolve(grid, flux.space, flux.medium, z=flux.z, side=flux.side)
        )
    return PlaneMetric(
        "unknown",
        reason=(
            "the source metric is tabulated on its own frequencies and the plane "
            "carries no medium to evaluate it on the new lattice"
        ),
    )


def _within_band(grid: TimeGrid, omega: np.ndarray) -> np.ndarray:
    """Return which absolute frequencies lie inside a grid's active band.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The receiving grid.
    omega : numpy.ndarray
        Absolute angular frequencies, rad/s.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the same shape.
    """
    active = grid.omega[grid.active_mask]
    return (omega >= float(np.min(active))) & (omega <= float(np.max(active)))


def regrid(  # noqa: PLR0913 - the target and each explicitly lossy permission
    plane: FieldPlane,
    grid: TimeGrid,
    *,
    phase: float | None = None,
    delay: float | None = None,
    truncate: bool = False,
    crop_band: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    method: Literal["auto", "direct"] = "auto",
) -> tuple[FieldPlane, RegridReport]:
    r"""Convert a plane to another temporal lattice and reference.

    The module's notes give the map. The plane's transverse representation, basis,
    geometry and medium are unchanged; only the temporal record moves.

    Parameters
    ----------
    plane : FieldPlane
        The source.
    grid : solprop.grid.TimeGrid
        The receiving temporal grid, field or envelope, of any length, spacing and
        carrier.
    phase : float or None, optional
        The result's reference phase, rad; None keeps the plane's.
    delay : float or None, optional
        The result's reference delay (time origin), s; None keeps the plane's. A
        later delay moves the record earlier in the window.
    truncate : bool, optional
        Declare the record zero outside its window, so that a larger or shifted
        target may extend it, and allow the part outside the target window to be
        cut, with its share reported. Default False refuses both.
    crop_band : bool, optional
        Allow light outside the target's active band to be removed, with its share
        reported. Default False refuses it.
    tolerance : float, optional
        Share of the record norm below which a band or a window counts as
        unoccupied; default :data:`DEFAULT_TOLERANCE`.
    method : {'auto', 'direct'}, optional
        ``'auto'`` copies on an identical lattice in an unchanged reference and
        evaluates the direct sum otherwise; ``'direct'`` always evaluates it.

    Returns
    -------
    tuple
        The converted :class:`~solprop.coupling.FieldPlane` and its
        :class:`RegridReport`.

    Raises
    ------
    CouplingError
        If the target band cannot hold the occupied frequencies, the target window
        the record, or a nonlocalised or unknown-exterior record would be extended,
        without the matching permission; or a value is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.coupling import FieldPlane, PlaneBasis, PlaneGeometry, regrid
    >>> from solprop.coupling import basis_identity
    >>> from solprop.fourier import spectrum_to_optics
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse
    >>> source = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    >>> target = EnvelopeGrid.from_limits(1.0, 800e-9, (500e-9, 1.5e-6), 2e-12)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> samples = spectrum_to_optics(pulse.build(source), kind="envelope")
    >>> basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["one"]}))
    >>> plane = FieldPlane.from_samples(samples, source, PlaneGeometry(0.0),
    ...                                 exterior="zero", basis=basis)
    >>> moved, report = regrid(plane, target)
    >>> bool(np.isclose(moved.energy(), plane.energy(), rtol=1e-9))
    True
    >>> report.method, report.band_loss < 1e-10
    ('direct', True)
    """
    if not (math.isfinite(tolerance) and 0.0 <= tolerance < 1.0):
        msg = f"tolerance is a share in [0, 1), got {tolerance!r}"
        raise CouplingError(msg)
    old = plane.reference
    new = PlaneReference(
        old.phase if phase is None else phase,
        old.delay if delay is None else delay,
        old.calibrated,
    )
    shift = old.delay - new.delay
    phase_shift = old.phase - new.phase
    source_grid = plane.grid
    same = lattice_identity(source_grid) == lattice_identity(grid) and np.array_equal(
        source_grid.omega, grid.omega
    )
    n_modes, n_pol, _ = plane.samples.shape
    columns = plane.samples.reshape(n_modes * n_pol, source_grid.n_omega)
    weights = np.repeat(_column_weights(plane), n_pol)

    q_source = _q(source_grid)
    per_bin = np.sum(weights[:, None] * np.abs(columns * q_source) ** 2, axis=0)
    total_bins = float(np.sum(per_bin))
    # Parseval on the source lattice: sum_j |a_j|^2 dt = (dt/N) sum_k |q s|^2.
    record_norm = total_bins * source_grid.dt / source_grid.n_t
    if total_bins == 0.0:
        band_loss = 0.0
    else:
        outside = ~_within_band(grid, source_grid.omega)
        band_loss = float(np.sum(per_bin[outside]) / total_bins)
    if band_loss > tolerance and not crop_band:
        msg = (
            f"{band_loss:.3e} of the record lies outside the target's active band "
            f"[{np.min(grid.omega[grid.active_mask]):.6e}, "
            f"{np.max(grid.omega[grid.active_mask]):.6e}] rad/s, which cannot "
            "represent it; widen the target or allow the loss with crop_band=True"
        )
        raise CouplingError(msg)

    if method == "auto" and same and shift == 0.0 and phase_shift == 0.0:
        # The identity is exact: what lies outside the band, below the tolerance, is
        # left for the receiving run's own hard mask, as the sum would zero it.
        result = columns.copy()
        report_method: Literal["copy", "direct"] = "copy"
        window_loss = edge_share = 0.0
    else:
        report_method = "direct"
        analytic = _analytic(columns, source_grid)
        inside, window_loss, edge_share = _record_window(
            plane,
            analytic,
            weights,
            grid,
            shift,
            truncate=truncate,
            tolerance=tolerance,
        )
        times = source_grid.t + shift
        result = _direct_sum(
            analytic[:, inside], times[inside], grid, source_grid, phase_shift
        )

    samples = result.reshape(n_modes, n_pol, grid.n_omega)
    history = (
        f"regrid({report_method}; n_t {source_grid.n_t}->{grid.n_t}, "
        f"{source_grid.kind}->{grid.kind}, shift {shift:.6e} s, "
        f"phase {phase_shift:.6e} rad)"
    )
    moved = FieldPlane(
        plane.kind,
        samples,
        grid,
        plane.units,
        plane.basis,
        new,
        plane.geometry,
        _target_metric(plane, grid, same),
        plane.exterior,
        plane.medium,
        plane.certificate,
        plane.nodal if report_method == "copy" else None,
        plane.provenance.then(history),
    )
    energy_in = None if plane.metric.kind == "unknown" else plane.energy()
    energy_out = None if moved.metric.kind == "unknown" else moved.energy()
    report = RegridReport(
        report_method,
        lattice_identity(source_grid),
        lattice_identity(grid),
        shift,
        phase_shift,
        record_norm,
        window_loss,
        band_loss,
        edge_share,
        truncate,
        energy_in,
        energy_out,
    )
    return moved, report


def _record_window(  # noqa: PLR0913 - the record and each permission it is checked by
    plane: FieldPlane,
    analytic: Complex[np.ndarray, "n_columns n_t"],
    weights: Float[np.ndarray, " n_columns"],
    grid: TimeGrid,
    shift: float,
    *,
    truncate: bool,
    tolerance: float,
) -> tuple[np.ndarray, float, float]:
    """Return which source samples the target window keeps, and check the record.

    Parameters
    ----------
    plane : FieldPlane
        The source.
    analytic : numpy.ndarray
        Its analytic signal, shape ``(n_columns, n_t)``.
    weights : numpy.ndarray
        Each column's transverse weight.
    grid : solprop.grid.TimeGrid
        The target grid.
    shift : float
        Record-time shift, s.
    truncate : bool
        Whether the record may be cut and declared zero outside its window.
    tolerance : float
        Share below which a window counts as unoccupied.

    Returns
    -------
    tuple
        The kept-sample mask, the share outside the target window and the share in
        the source's temporal collar.

    Raises
    ------
    CouplingError
        If light falls outside the target window, or a nonlocalised or
        unknown-exterior record would be extended, without `truncate`.
    """
    source = plane.grid
    power = np.sum(weights[:, None] * np.abs(analytic) ** 2, axis=0)
    total = float(np.sum(power))
    collar = source.t_collar
    edge = (source.t < collar.one_low) | (source.t > collar.one_high)
    edge_share = 0.0 if total == 0.0 else float(np.sum(power[edge]) / total)
    times = source.t + shift
    half = 0.5 * grid.dt
    inside = (times >= grid.t[0] - half) & (times <= grid.t[-1] + half)
    window_loss = 0.0 if total == 0.0 else float(np.sum(power[~inside]) / total)
    if window_loss > tolerance and not truncate:
        msg = (
            f"{window_loss:.3e} of the record falls outside the target window after "
            f"a shift of {shift:.6e} s; widen the target, change the delay, or cut "
            "it with truncate=True"
        )
        raise CouplingError(msg)
    source_half = 0.5 * source.dt
    extends = (grid.t[0] - half < times[0] - source_half) or (
        grid.t[-1] + half > times[-1] + source_half
    )
    if (
        extends
        and not truncate
        and (edge_share > tolerance or plane.exterior == "unknown")
    ):
        reason = (
            "its exterior is unknown"
            if plane.exterior == "unknown"
            else f"{edge_share:.3e} of it lies in the temporal collar"
        )
        msg = (
            f"the target window extends this record beyond its own, and {reason}: "
            "declare it zero outside with truncate=True"
        )
        raise CouplingError(msg)
    return inside, window_loss, edge_share


def _direct_sum(
    analytic: Complex[np.ndarray, "n_columns n_inside"],
    times: Float[np.ndarray, " n_inside"],
    grid: TimeGrid,
    source: TimeGrid,
    phase_shift: float,
) -> Complex[np.ndarray, "n_columns n_omega"]:
    """Evaluate the finite-time Fourier sum at the target's in-band bins.

    The sum of samples spaced ``dt`` is periodic in frequency with period
    ``2 pi / dt``; outside the source lattice's own span it returns aliases of the
    record, not light, so a target bin outside that span is zero: the source
    represents nothing there.

    Parameters
    ----------
    analytic : numpy.ndarray
        Source analytic signal at the kept samples, shape ``(n_columns, n_inside)``.
    times : numpy.ndarray
        Their shifted times, s.
    grid : solprop.grid.TimeGrid
        The target grid.
    source : solprop.grid.TimeGrid
        The source grid.
    phase_shift : float
        Phase applied, rad.

    Returns
    -------
    numpy.ndarray
        Target stored spectra, zero outside the target's active band and outside
        the source lattice's span, shape ``(n_columns, n_omega)``.
    """
    result = np.zeros((analytic.shape[0], grid.n_omega), dtype=np.complex128)
    half = 0.5 * source.d_omega
    represented = (grid.omega >= float(np.min(source.omega)) - half) & (
        grid.omega <= float(np.max(source.omega)) + half
    )
    bins = np.flatnonzero(_within_band(grid, grid.omega) & represented)
    scale = source.dt * np.exp(1j * phase_shift) / (grid.dt * _q(grid))
    size = max(1, _CHUNK_ELEMENTS // max(1, times.size))
    for start in range(0, bins.size, size):
        chosen = bins[start : start + size]
        kernel = np.exp(1j * np.outer(times, grid.omega[chosen]))
        result[:, chosen] = (analytic @ kernel) * scale[chosen][None, :]
    return result

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assemble independently projected TDSE polarisation and current sources.

The native closure's gated, edge-free source (SEP 26) multiplies the remainder current
``S`` by its gate ``g`` before the transform and subtracts the edge term
``DFT[g' S] / (i omega)`` on the active bins, so that the current is held after the
window instead of switched off.
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.backends.numpy_ref import passes
from solprop.exceptions import BackendError
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEResult, lattice_times, waveform_identity
from solprop.translate.tdse_source import steady_current_certificate

__all__ = [
    "FFT_ERROR_CONSTANT",
    "atomic_edge_source",
    "atomic_sources",
    "project_atomic_sources",
]

FFT_ERROR_CONSTANT = 8.0
"""Constant ``c_F`` of the forward FFT's normwise relative bound ``c_F eps log2 N``.

Higham (2002), Theorem 24.2: a radix-2 Cooley-Tukey transform with twiddles accurate
to ``mu`` has ``||y - fl(y)|| / ||y|| <= log2(N) eta / (1 - log2(N) eta)``,
``eta = mu + gamma_4 (sqrt 2 + mu)``, about ``3.4 eps`` per level with correctly
rounded twiddles. A radix-3 or radix-5 butterfly does more operations per level, but
spans ``log2 3`` or ``log2 5`` levels of ``log2 N``; 8 covers the 2-, 3- and
5-smooth sizes solprop's lattices use (SEP 26 section 4.5). The SEP 26 T3 test
measures the constant at a power-of-two and at a ``2^a 3^b 5^c`` size; any other size
is flagged in the diagnostic as unqualified. doi:10.1137/1.9780898718027
"""


def atomic_sources(
    fields: Float[np.ndarray, "n_nodes n_to"],
    result: TDSEResult,
    problem: KernelProblem,
    bundle: StageScalars,
) -> tuple[
    Float[np.ndarray, "n_nodes 1 n_to"],
    Float[np.ndarray, "n_nodes 1 n_to"],
]:
    """Subtract discrete shadows and apply the declared density and scales once.

    Parameters
    ----------
    fields : numpy.ndarray
        Chronological real physical fields in V/m, shape ``(n_nodes, n_to)``.
    result : TDSEResult
        Owning microscopic SI histories for these exact inputs.
    problem : KernelProblem
        Compiled optical and atomic scientific specification.
    bundle : StageScalars
        Density and other scalars at this evaluation's absolute position.

    Returns
    -------
    tuple of numpy.ndarray
        Polarisation in C/m² and current in A/m², each with a scalar component
        axis. These new arrays never alias or mutate the provider's histories.

    Raises
    ------
    BackendError
        If an identity, shape or required observable does not match the model.
    """
    descriptor = problem.tdse
    if descriptor is None:
        raise BackendError("atomic source assembly requires a TDSE descriptor")
    if (
        result.configuration_identity != descriptor.identity
        or result.input_identity != waveform_identity(fields)
        or result.total_current.shape != fields.shape
    ):
        raise BackendError("TDSE result does not identify this configuration and field")
    # These are the unscaled shadows of the same discrete map. Scaling either
    # before subtraction restores a large linear/cubic artifact to the residual.
    remainder = result.total_current - result.first_order_current
    if descriptor.kerr_model != "tdse_native":
        if result.third_order_current is None or 3 not in result.shadow_orders:
            raise BackendError("cubic replacement requires the third-order current")
        remainder = remainder - result.third_order_current
    if descriptor.gated:
        # Gate the subtracted source before its finite-record FFT. Applying a
        # spectral mask afterwards cannot undo endpoint leakage already in band.
        # The owning provider histories remain unchanged. An atomic window's gate
        # replaces the grid collar: it is the same gate that tapered the drive, so
        # the source vanishes wherever the atom was not driven (SEP 20, 4.6).
        remainder = remainder * _source_gate(problem)
    current = bundle.density * descriptor.remainder_scale * remainder[:, None, :]
    if descriptor.kerr_model == "tdse_shadow":
        if result.third_order_dipole is None or descriptor.cubic_scale is None:
            raise BackendError("tdse_shadow requires the third-order charge dipole")
        polarisation = (
            bundle.density
            * descriptor.cubic_scale
            * result.third_order_dipole[:, None, :]
        )
    elif descriptor.kerr_model == "optmat_cubic":
        polarisation = passes.nonlinear_polarisation(
            fields[:, None, :],
            problem.responses,
            problem.grids.t_window_fine,
            kind="field",
            density=bundle.density,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
            carrier=None,
            scalars=bundle,
        )
    else:
        polarisation = np.zeros_like(current)
    # The existing in-stage field absorber remains separate. An explicit source
    # gate modifies only this macroscopic source, never the raw atomic histories.
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(polarisation)):
        raise BackendError("TDSE source assembly produced a non-finite value")
    return polarisation, current


def atomic_edge_source(
    result: TDSEResult, problem: KernelProblem, bundle: StageScalars
) -> Float[np.ndarray, "n_nodes 1 n_to"] | None:
    """Form a gated source's edge array ``g' S``, scaled exactly as its current.

    ``S = J - j_1`` is the same unscaled remainder :func:`atomic_sources` gates (the
    gated filter admits ``tdse_native`` only), ``g'`` the analytic derivative of the
    same gate (SEP 26 section 4.2), and density and ``remainder_scale`` enter once,
    as for the current. Nothing here is differentiated numerically.

    Parameters
    ----------
    result : TDSEResult
        Owning microscopic SI histories, already matched to this configuration.
    problem : KernelProblem
        Compiled optical and atomic scientific specification.
    bundle : StageScalars
        Density and other scalars at this evaluation's absolute position.

    Returns
    -------
    numpy.ndarray or None
        ``rho s g'(t) S(t)`` in A/(m² s), shape ``(n_nodes, 1, n_to)``, for
        a gated (native) source; None for a cubic replacement, which has no gate.

    Raises
    ------
    BackendError
        If the descriptor is absent, the result is not this configuration's, or the
        array is not finite.
    """
    descriptor = problem.tdse
    if descriptor is None:
        raise BackendError("atomic edge assembly requires a TDSE descriptor")
    if not descriptor.gated:
        return None
    if (
        result.configuration_identity != descriptor.identity
        or result.total_current.shape[-1] != descriptor.samples
    ):
        raise BackendError("TDSE result does not identify this configuration")
    remainder = result.total_current - result.first_order_current
    edge = (
        bundle.density
        * descriptor.remainder_scale
        * (remainder * descriptor.source_gate_derivative())[:, None, :]
    )
    if not np.all(np.isfinite(edge)):
        raise BackendError("TDSE edge assembly produced a non-finite value")
    return edge


def _source_gate(problem: KernelProblem) -> Float[np.ndarray, " n_to"]:
    """Return the temporal gate of a gated source.

    Parameters
    ----------
    problem : KernelProblem
        Problem whose descriptor may carry an atomic window.

    Returns
    -------
    numpy.ndarray
        The atomic window's gate when one is set, else the fine temporal collar,
        shape ``(n_to,)``.
    """
    descriptor = problem.tdse
    gate = None if descriptor is None else descriptor.source_gate()
    return problem.grids.t_window_fine if gate is None else gate


def _project_spectrum(
    values: Float[np.ndarray, "n_nodes 1 n_to"],
    projection: Float[np.ndarray, "n_modes n_nodes"],
    problem: KernelProblem,
    bundle: StageScalars,
) -> Complex[np.ndarray, "n_modes 1 n_omega"]:
    """Apply the same modal projection and unscaled crop to either source.

    Parameters
    ----------
    values : numpy.ndarray
        Nodal P or J source, shape ``(n_nodes, 1, n_to)``, in C/m² or A/m².
    projection : numpy.ndarray
        Modal weights, shape ``(n_modes, n_nodes)``.
    problem : KernelProblem
        Crop indices of the field grid.
    bundle : StageScalars
        Stage taper scale.

    Returns
    -------
    numpy.ndarray
        Raw FFT source, shape ``(n_modes, 1, n_omega)`` before source factors.
    """
    projected = passes.project(values, projection, bundle.taper_scale, 1)
    return passes.to_frequency(projected, problem.grids.crop_index, kind="field")


def _roundoff_floor(
    magnitudes: Float[np.ndarray, "n_nodes 1 n_to"],
    projection: Float[np.ndarray, "n_modes n_nodes"],
    factor: Complex[np.ndarray, "n_modes 1 n_omega"]
    | Float[np.ndarray, "n_modes 1 n_omega"],
    problem: KernelProblem,
    bundle: StageScalars,
) -> float:
    """Bound operand cancellation through absolute projection and Fourier sums.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Sum of absolute microscopic operands after density and calibration,
        shape ``(n_nodes, 1, n_to)``, in C/m² or A/m².
    projection : numpy.ndarray
        Modal weights, shape ``(n_modes, n_nodes)``.
    factor : numpy.ndarray
        P or J spectral factor including crop scale, shape
        ``(n_modes, 1, n_omega)``.
    problem : KernelProblem
        Spectral collar and transform size.
    bundle : StageScalars
        Stage taper scale.

    Returns
    -------
    float
        Nonnegative absolute rounding floor in spectral derivative norm units.
    """
    scaled, accumulation = _roundoff_terms(
        magnitudes, projection, factor, problem, bundle
    )
    return float(np.linalg.norm(scaled)) * accumulation / (1 - accumulation)


def _roundoff_terms(
    magnitudes: Float[np.ndarray, "n_nodes 1 n_to"],
    projection: Float[np.ndarray, "n_modes n_nodes"],
    factor: Complex[np.ndarray, "n_modes 1 n_omega"]
    | Float[np.ndarray, "n_modes 1 n_omega"],
    problem: KernelProblem,
    bundle: StageScalars,
) -> tuple[Float[np.ndarray, "n_modes 1 n_omega"], float]:
    """Return the per-bin operand bound and the accumulation of :func:`_roundoff_floor`.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Absolute operands, shape ``(n_nodes, 1, n_to)``, as :func:`_roundoff_floor`.
    projection : numpy.ndarray
        Modal weights, shape ``(n_modes, n_nodes)``.
    factor : numpy.ndarray
        Spectral factor, shape ``(n_modes, 1, n_omega)``.
    problem : KernelProblem
        Spectral collar and transform size.
    bundle : StageScalars
        Stage taper scale.

    Returns
    -------
    scaled : numpy.ndarray
        Per-bin absolute operand bound, shape ``(n_modes, 1, n_omega)``.
    accumulation : float
        Relative accumulation ``operations * eps``.
    """
    absolute_projected = passes.project(
        magnitudes, np.abs(projection), abs(bundle.taper_scale), 1
    )
    # A Fourier coefficient is bounded by the temporal L1 sum. Keep the raw
    # operands in this bound: using the already cancelled remainder would hide
    # precisely the weak-field loss of significance the diagnostic must flag.
    spectral_bound = np.sum(absolute_projected, axis=-1, keepdims=True)
    scaled = np.abs(factor * problem.grids.omega_window) * spectral_bound
    # Account conservatively for subtraction/scaling, the node reduction and
    # radix-transform depth. This is a floating-point indicator, not a bound on
    # the native atomic integration, spectral truncation or physical model.
    operations = 8 + 2 * projection.shape[1] + 5 * np.ceil(np.log2(problem.grids.n_to))
    if problem.tdse is not None and problem.tdse.gated:
        # The gate's multiplication.
        operations += 1
    return scaled, float(operations * np.finfo(float).eps)


def _inverse_i_omega(problem: KernelProblem) -> Complex[np.ndarray, " n_omega"]:
    """Return ``1 / (i omega)`` on the active bins and zero elsewhere.

    Parameters
    ----------
    problem : KernelProblem
        Coarse frequencies in rad/s and the active mask; every active bin is above
        ``omega_min / 2 > 0``, so no division by zero occurs.

    Returns
    -------
    numpy.ndarray
        ``-i / omega`` in s on active bins, exactly zero elsewhere, shape
        ``(n_omega,)``.
    """
    grids = problem.grids
    inverse = np.zeros(grids.n_omega, dtype=np.complex128)
    inverse[grids.active_mask] = -1j / grids.omega[grids.active_mask]
    return inverse


@dataclass(frozen=True)
class _CurrentOperands:
    """Hold one evaluation's nodal current operands for every projection.

    Attributes
    ----------
    current : numpy.ndarray
        Scaled (gated) remainder current, shape ``(n_nodes, 1, n_to)``.
    edge : numpy.ndarray or None
        Scaled ``g' S`` of a gated source.
    magnitude : numpy.ndarray
        Absolute operands of today's floor, ``|rho s| |g| (|J| + |j_1|)``.
    edge_magnitude, derivative_magnitude : numpy.ndarray or None
        ``|rho s| |g'| (|J| + |j_1|)`` and ``|rho s| delta g' (|J| + |j_1|)``.
    factor : numpy.ndarray
        Current factor, shape ``(n_modes, 1, n_omega)``.
    inverse : numpy.ndarray or None
        ``1 / (i omega)`` on active bins, for a gated source.
    """

    current: Float[np.ndarray, "n_nodes 1 n_to"]
    edge: Float[np.ndarray, "n_nodes 1 n_to"] | None
    magnitude: Float[np.ndarray, "n_nodes 1 n_to"]
    edge_magnitude: Float[np.ndarray, "n_nodes 1 n_to"] | None
    derivative_magnitude: Float[np.ndarray, "n_nodes 1 n_to"] | None
    factor: Float[np.ndarray, "n_modes 1 n_omega"]
    inverse: Complex[np.ndarray, " n_omega"] | None


@dataclass(frozen=True)
class _CurrentPart:
    """Hold the current's spectra and floors under one projection.

    Attributes
    ----------
    truncating : numpy.ndarray
        ``X``, the raw transform of the projected gated current.
    edge : numpy.ndarray or None
        ``E / (i omega)`` of a gated source, zero off the active bins.
    source : numpy.ndarray
        The current's contribution after its factor: ``factor (X - E/(i omega))``
        for a gated source, ``factor X`` for an ungated one.
    floor : float
        Its rounding floor, the edge term's included.
    bounds : tuple of numpy.ndarray or None
        Per-bin bounds of ``c X``, the edge term, and the division and subtraction.
    """

    truncating: Complex[np.ndarray, "n_modes 1 n_omega"]
    edge: Complex[np.ndarray, "n_modes 1 n_omega"] | None
    source: Complex[np.ndarray, "n_modes 1 n_omega"]
    floor: float
    bounds: (
        tuple[
            Float[np.ndarray, "n_modes 1 n_omega"],
            Float[np.ndarray, "n_modes 1 n_omega"],
            Float[np.ndarray, "n_modes 1 n_omega"],
        ]
        | None
    )


def _current_part(
    operands: _CurrentOperands,
    projection: Float[np.ndarray, "n_modes n_nodes"],
    problem: KernelProblem,
    bundle: StageScalars,
) -> _CurrentPart:
    """Project, transform and crop the current, and its edge term, under one map.

    Parameters
    ----------
    operands : _CurrentOperands
        The evaluation's nodal arrays and factors.
    projection : numpy.ndarray
        Modal weights, shape ``(n_modes, n_nodes)``: the fine or the embedded map.
    problem : KernelProblem
        Crop, masks, collar and transform size.
    bundle : StageScalars
        Stage taper scale.

    Returns
    -------
    _CurrentPart
        Spectra, source and floors.
    """
    factor = operands.factor
    truncating = _project_spectrum(operands.current, projection, problem, bundle)
    scaled, accumulation = _roundoff_terms(
        operands.magnitude, projection, factor, problem, bundle
    )
    floor = float(np.linalg.norm(scaled)) * accumulation / (1 - accumulation)
    if operands.edge is None:
        return _CurrentPart(truncating, None, factor * truncating, floor, None)
    assert operands.inverse is not None
    assert operands.edge_magnitude is not None
    assert operands.derivative_magnitude is not None
    edge = (
        _project_spectrum(operands.edge, projection, problem, bundle) * operands.inverse
    )
    edge_bins, division_bins = _edge_rounding(
        (operands.edge_magnitude, operands.derivative_magnitude),
        projection,
        factor,
        (truncating, edge),
        problem,
        bundle,
    )
    bounds = (scaled * (accumulation / (1 - accumulation)), edge_bins, division_bins)
    # Subtract the edge term before the current factor; X and E share every linear
    # factor that follows (SEP 26 section 4.6).
    floor += float(np.linalg.norm(edge_bins)) + float(np.linalg.norm(division_bins))
    return _CurrentPart(truncating, edge, factor * (truncating - edge), floor, bounds)


def _edge_rounding(
    magnitudes: tuple[
        Float[np.ndarray, "n_nodes 1 n_to"], Float[np.ndarray, "n_nodes 1 n_to"]
    ],
    projection: Float[np.ndarray, "n_modes n_nodes"],
    factor: Float[np.ndarray, "n_modes 1 n_omega"],
    spectra: tuple[
        Complex[np.ndarray, "n_modes 1 n_omega"],
        Complex[np.ndarray, "n_modes 1 n_omega"],
    ],
    problem: KernelProblem,
    bundle: StageScalars,
) -> tuple[
    Float[np.ndarray, "n_modes 1 n_omega"], Float[np.ndarray, "n_modes 1 n_omega"]
]:
    r"""Bound, per bin, the rounding the edge term adds (SEP 26 section 4.5).

    In the units of the source after the current factor and the spectral collar, with
    ``c_k = |factor_k W_k|`` and ``omega_lo`` the lowest active frequency:

    * the edge transform, ``c_k sqrt(N) (c_F eps log2 N + (n_nodes + 6) u) ||y||_2 /
      omega_lo``. ``y`` is the absolute projection of the raw operands
      ``|rho s| |g'| (|J| + |j_1|)``, never of the cancelled remainder, so shadow
      cancellation is accounted for. ``(n_nodes + 6) u`` covers the remainder's
      subtraction, the products by ``g'``, density and scale, the taper division and
      the node reduction before the transform. ``sqrt(N)`` is Cauchy-Schwarz from the
      2-norm to one bin;
    * the evaluation of ``g'``, ``c_k sqrt(N) ||y_d||_2 / omega_lo``, with ``y_d`` the
      absolute projection of ``|rho s| delta g' (|J| + |j_1|)`` and ``delta g'`` the
      derivative's own rounding bound;
    * the division by ``i omega``, the subtraction and the current factor,
      ``2 eps c_k (|X_k| + |E_k| / omega_k)``.

    The truncating part keeps :func:`_roundoff_floor`, unchanged; aliasing is not a
    rounding error and is governed by the certificate (section 4.4).

    Parameters
    ----------
    magnitudes : tuple of numpy.ndarray
        ``|rho s| |g'| (|J| + |j_1|)`` and ``|rho s| delta g' (|J| + |j_1|)``, each
        of shape ``(n_nodes, 1, n_to)``.
    projection : numpy.ndarray
        Modal weights, shape ``(n_modes, n_nodes)``.
    factor : numpy.ndarray
        Current factor, shape ``(n_modes, 1, n_omega)``.
    spectra : tuple of numpy.ndarray
        ``X``, the projected raw spectrum of the gated current, and ``E / (i
        omega)``, of the edge array, each ``(n_modes, 1, n_omega)``.
    problem : KernelProblem
        Coarse frequencies, masks, collar and transform size.
    bundle : StageScalars
        Stage taper scale.

    Returns
    -------
    edge : numpy.ndarray
        Bound of the edge transform and the derivative evaluation, per bin.
    division : numpy.ndarray
        Bound of the division, subtraction and factor, per bin.
    """
    grids = problem.grids
    edge_magnitude, derivative_magnitude = magnitudes
    epsilon = float(np.finfo(float).eps)
    weights = np.abs(projection)
    taper = abs(bundle.taper_scale)
    operands = np.linalg.norm(
        passes.project(edge_magnitude, weights, taper, 1), axis=-1, keepdims=True
    )
    evaluation = np.linalg.norm(
        passes.project(derivative_magnitude, weights, taper, 1),
        axis=-1,
        keepdims=True,
    )
    collar = np.abs(factor * grids.omega_window)
    omega_lo = float(np.min(grids.omega[grids.active_mask]))
    root = math.sqrt(grids.n_to)
    transform = FFT_ERROR_CONSTANT * epsilon * math.ceil(math.log2(grids.n_to)) + (
        projection.shape[1] + 6
    ) * (epsilon / 2)
    edge = collar * root * (transform * operands + evaluation) / omega_lo
    truncating, divided = spectra
    division = 2 * epsilon * collar * (np.abs(truncating) + np.abs(divided))
    return edge, division


def _ratio(
    numerator: float, denominator: float, floor: float
) -> tuple[float | None, str | None]:
    """Return a ratio, or None with its reason when the denominator is noise.

    Parameters
    ----------
    numerator, denominator : float
        Nonnegative norms.
    floor : float
        Rounding floor of the denominator.

    Returns
    -------
    tuple
        The ratio and None, or None and ``"denominator below rounding floor"``.
    """
    if denominator <= floor:
        return None, "denominator below rounding floor"
    return numerator / denominator, None


def _band_entry(
    spectra: tuple[
        Complex[np.ndarray, "n_modes 1 n_omega"],
        Complex[np.ndarray, "n_modes 1 n_omega"],
    ],
    bounds: tuple[
        Float[np.ndarray, "n_modes 1 n_omega"],
        Float[np.ndarray, "n_modes 1 n_omega"],
        Float[np.ndarray, "n_modes 1 n_omega"],
    ],
    band: Bool[np.ndarray, " n_omega"],
) -> dict[str, object]:
    """Describe the edge term over one band, with explicit denominators and floors.

    Parameters
    ----------
    spectra : tuple of numpy.ndarray
        ``c X`` and ``c E / (i omega)``, the truncating source and the edge term
        after the current factor and the spectral collar.
    bounds : tuple of numpy.ndarray
        Per-bin rounding bounds of ``c X``, of the edge term and of the division and
        subtraction.
    band : numpy.ndarray
        Coarse bins of the band.

    Returns
    -------
    dict
        Norms of the truncating, edge and corrected sources, their floors and the two
        ratios, each None with a reason when its denominator is at or below its floor.
    """
    truncating, edge = spectra
    truncating_bins, edge_bins, division_bins = bounds
    corrected = truncating - edge
    norms = {
        name: float(np.linalg.norm(values[..., band]))
        for name, values in (
            ("truncating", truncating),
            ("edge", edge),
            ("corrected", corrected),
        )
    }
    truncating_floor = float(np.linalg.norm(truncating_bins[..., band]))
    edge_floor = float(np.linalg.norm(edge_bins[..., band])) + float(
        np.linalg.norm(division_bins[..., band])
    )
    corrected_floor = truncating_floor + edge_floor
    over_truncating, truncating_reason = _ratio(
        norms["edge"], norms["truncating"], truncating_floor
    )
    over_corrected, corrected_reason = _ratio(
        norms["edge"], norms["corrected"], corrected_floor
    )
    return {
        "bins": int(np.count_nonzero(band)),
        "truncating_norm": norms["truncating"],
        "edge_norm": norms["edge"],
        "corrected_norm": norms["corrected"],
        "truncating_floor": truncating_floor,
        "edge_floor": edge_floor,
        "corrected_floor": corrected_floor,
        "edge_over_truncating": over_truncating,
        "edge_over_truncating_reason": truncating_reason,
        "edge_over_corrected": over_corrected,
        "edge_over_corrected_reason": corrected_reason,
    }


def _projection_comparison(
    fine: Complex[np.ndarray, "n_modes 1 n_omega"],
    coarse: Complex[np.ndarray, "n_modes 1 n_omega"] | None,
    fine_floor: float,
    coarse_floor: float,
) -> dict[str, object]:
    """Describe one embedded comparison without dividing by cancellation noise.

    Parameters
    ----------
    fine : numpy.ndarray
        Fine modal source after the spectral collar, shape
        ``(n_modes, 1, n_omega)``.
    coarse : numpy.ndarray or None
        Embedded source on the same coarse spectral bins, or None if absent.
    fine_floor, coarse_floor : float
        Nonnegative absolute rounding bounds in spectral derivative norm units.

    Returns
    -------
    dict
        Norms, differences, combined absolute floor and a relative difference
        only when a distinct embedded source exists above the floor.
    """
    norm = float(np.linalg.norm(fine))
    floor = fine_floor + coarse_floor
    dominated = norm <= floor
    difference = None if coarse is None else float(np.linalg.norm(coarse - fine))
    return {
        "norm": norm,
        "embedded_norm": None if coarse is None else float(np.linalg.norm(coarse)),
        "absolute_difference": difference,
        "roundoff_floor": floor,
        "difference_above_floor": None
        if difference is None
        else max(0.0, difference - floor),
        "floor_dominated": dominated,
        "relative_difference": None
        if dominated or difference is None
        else difference / norm,
    }


def _leading_taper(
    result: TDSEResult,
    problem: KernelProblem,
    bundle: StageScalars,
    band_weights: dict[str, Float[np.ndarray, "n_modes 1 n_omega"]],
) -> dict[str, object]:
    r"""Record the leading taper's remainder ratio and its band bound (SEP 26, 4.3).

    ``rho_-`` is ``max |S|`` over every row and every sample strictly inside the
    leading taper ``(z_-, o_-)``, over ``max |S|`` over every row and every sample of
    the gate's support ``[z_-, z_+]``, with ``S = J - j_1`` unscaled. The band bound
    is ``sum_lead |g'_-(t_n)| max_lead x_m (sum_B (c_k / omega_k)^2)^{1/2}``, in the
    units of the ``edge`` norms, where ``x_m`` bounds the projected modal current from
    above by the absolute projection of ``|rho s S|``: it bounds the leading edge's
    share of the edge term over the band.

    Parameters
    ----------
    result : TDSEResult
        Unscaled microscopic histories.
    problem : KernelProblem
        Gate breakpoints, lattice and projection.
    bundle : StageScalars
        Density and taper scale.
    band_weights : dict
        ``c_k / omega_k`` on each band's bins, zero elsewhere, by band name.

    Returns
    -------
    dict
        ``ratio`` and its ``reason`` (``"zero remainder"`` when ``max |S| = 0``, or
        ``"empty leading taper"``), the sampled weight ``W_- / dt`` and the bound per
        band.
    """
    descriptor = problem.tdse
    assert descriptor is not None
    remainder = np.abs(result.total_current - result.first_order_current)
    zero_low, one_low, _, zero_high = descriptor.source_breakpoints()
    times = lattice_times(descriptor.samples, descriptor.spacing)
    lead = (times > zero_low) & (times < one_low)
    support = (times >= zero_low) & (times <= zero_high)
    peak = float(np.max(remainder[:, support], initial=0.0))
    lead_peak = float(np.max(remainder[:, lead], initial=0.0))
    weight = float(np.sum(np.abs(descriptor.source_gate_derivative()[lead])))
    if not np.any(lead):
        ratio, reason = None, "empty leading taper"
    elif peak == 0.0:
        ratio, reason = None, "zero remainder"
    else:
        ratio, reason = lead_peak / peak, None
    scaled = abs(bundle.density * descriptor.remainder_scale) * remainder[:, None, lead]
    projected = passes.project(
        scaled,
        np.abs(problem.transform.projection),
        abs(bundle.taper_scale),
        1,
    )
    largest = np.max(projected, axis=-1, keepdims=True, initial=0.0)
    return {
        "ratio": ratio,
        "reason": reason,
        "lead_samples": int(np.count_nonzero(lead)),
        "sampled_weight": weight * descriptor.spacing,
        "band_bound": {
            name: weight * float(np.linalg.norm(largest * values))
            for name, values in band_weights.items()
        },
    }


def _edge_diagnostic(
    spectra: tuple[
        Complex[np.ndarray, "n_modes 1 n_omega"],
        Complex[np.ndarray, "n_modes 1 n_omega"],
    ],
    bounds: tuple[
        Float[np.ndarray, "n_modes 1 n_omega"],
        Float[np.ndarray, "n_modes 1 n_omega"],
        Float[np.ndarray, "n_modes 1 n_omega"],
    ],
    factor: Float[np.ndarray, "n_modes 1 n_omega"],
    result: TDSEResult,
    problem: KernelProblem,
    bundle: StageScalars,
) -> dict[str, object]:
    """Describe the last evaluation's edge term on the active and in-band bins.

    Parameters
    ----------
    spectra : tuple of numpy.ndarray
        Projected raw spectra ``X`` of the gated current and ``E / (i omega)`` of the
        edge array, before the current factor.
    bounds : tuple of numpy.ndarray
        Per-bin rounding bounds, after the factor and collar, of ``c X``, of the edge
        term and of the division and subtraction.
    factor : numpy.ndarray
        Current factor ``current_numerator / Re n_eff``.
    result : TDSEResult
        Unscaled microscopic histories, for ``rho_-``.
    problem : KernelProblem
        Masks, collar, lattice and descriptor.
    bundle : StageScalars
        Density and taper scale.

    Returns
    -------
    dict
        JSON-compatible record of SEP 26 section 5. A diagnostic, never a gate.
    """
    descriptor, grids = problem.tdse, problem.grids
    assert descriptor is not None
    weight = factor * grids.omega_window
    truncating, divided = spectra
    bands = {"active": grids.active_mask, "inband": grids.inband_mask}
    magnitude = np.abs(weight)
    inverse = np.abs(_inverse_i_omega(problem))
    band_weights = {
        name: np.where(mask, magnitude * inverse, 0.0) for name, mask in bands.items()
    }
    certificate = steady_current_certificate(descriptor, grids.omega, grids.active_mask)
    return {
        "units": "raw FFT modal current derivative after the spectral collar",
        "steady_current_certificate": certificate,
        "fft_constant_qualified": _smooth(grids.n_to),
        "bands": {
            name: _band_entry((weight * truncating, weight * divided), bounds, mask)
            for name, mask in bands.items()
        },
        "leading_taper": _leading_taper(result, problem, bundle, band_weights),
    }


def _smooth(samples: int) -> bool:
    """Return whether a transform size is 2-, 3- and 5-smooth.

    Parameters
    ----------
    samples : int
        Positive transform size.

    Returns
    -------
    bool
        True when ``samples`` has no prime factor above 5, the sizes for which
        ``FFT_ERROR_CONSTANT`` is assumed.
    """
    for prime in (2, 3, 5):
        while samples % prime == 0:
            samples //= prime
    return samples == 1


def project_atomic_sources(  # noqa: PLR0913 - the existing seam plus the edge array.
    polarisation: Float[np.ndarray, "n_nodes_p 1 n_to_p"],
    current: Float[np.ndarray, "n_nodes_j 1 n_to_j"],
    result: TDSEResult,
    problem: KernelProblem,
    bundle: StageScalars,
    prefactor: Complex[np.ndarray, "n_modes 1 n_omega"],
    *,
    edge: Float[np.ndarray, "n_nodes_e 1 n_to_e"] | None = None,
) -> tuple[Complex[np.ndarray, "n_modes 1 n_omega"], dict[str, object]]:
    """Project both sources and reuse atomic rows for an embedded radial indicator.

    For a gated (native) source the current's spectrum is the edge-free
    ``X - E / (i omega)`` on the active bins (SEP 26 section 4.4), ``X`` and ``E`` the
    raw transforms of the projected gated current and edge array, before the current
    factor; ``1/(i omega)`` is never formed at the zero or inactive bins, where the
    current factor is zero anyway.

    Parameters
    ----------
    polarisation : numpy.ndarray
        Already calibrated macroscopic polarisation in C/m², shape
        ``(n_nodes, 1, n_to)``.
    current : numpy.ndarray
        Already scaled macroscopic remainder current in A/m², same shape.
    result : TDSEResult
        Unscaled microscopic operands used to derive an absolute cancellation floor.
    problem : KernelProblem
        Resolved modal maps, crop, spectral collar and independent current numerator.
    bundle : StageScalars
        Stage position in m, density in m⁻³ and modal taper scale.
    prefactor : numpy.ndarray
        Stage polarisation factor, including the crop scale and effective index,
        shape ``(n_modes, 1, n_omega)``.
    edge : numpy.ndarray or None, optional
        :func:`atomic_edge_source`'s ``g' S``, same shape as ``current``. Required
        for a gated (native) source and refused for an ungated one.

    Returns
    -------
    source : numpy.ndarray
        Complete spectral derivative before the spectral collar and interaction
        propagator, shape ``(n_modes, 1, n_omega)``. The caller applies these once.
    diagnostics : dict
        JSON-compatible complete-source and remainder-source norms after the
        spectral collar, in the same raw FFT modal derivative units. Each reports
        the embedded discrepancy, an absolute floating-point floor, and no relative
        score when the fine source is floor dominated. ``available=False`` means
        there is no distinct embedded map. This same-row indicator is not the
        independent 17-versus-33-node qualification screen. A gated source
        adds the ``edge`` entry of SEP 26 section 5, on the fine projection.

    Raises
    ------
    BackendError
        If a required descriptor, map or source shape is absent or nonfinite.
    """
    descriptor, transform, grids = problem.tdse, problem.transform, problem.grids
    if descriptor is None or problem.current_numerator is None:
        raise BackendError(
            "TDSE projection requires independent current and polarisation factors"
        )
    expected = (transform.n_nodes, 1, grids.n_to)
    if polarisation.shape != expected or current.shape != expected:
        raise BackendError(
            "TDSE nodal source shapes disagree with the resolved transform"
        )
    if result.total_current.shape != (transform.n_nodes, grids.n_to):
        raise BackendError(
            "TDSE cancellation operands disagree with the resolved transform"
        )
    if not np.all(np.isfinite(polarisation)) or not np.all(np.isfinite(current)):
        raise BackendError("TDSE nodal sources must be finite")
    operands = _current_operands(current, edge, result, problem, bundle)
    fine_p = prefactor * _project_spectrum(
        polarisation, transform.projection, problem, bundle
    )
    fine = _current_part(operands, transform.projection, problem, bundle)
    fine_j = fine.source
    source = fine_p + fine_j
    if not np.all(np.isfinite(source)):
        raise BackendError("TDSE projected source is nonfinite")
    fine_j_floor = fine.floor
    fine_p_floor = _roundoff_floor(
        np.abs(polarisation), transform.projection, prefactor, problem, bundle
    )
    available = not np.array_equal(transform.projection, transform.projection_coarse)
    coarse_j = coarse_source = None
    coarse_j_floor = coarse_p_floor = 0.0
    if available:
        projection = transform.projection_coarse
        coarse_p = prefactor * _project_spectrum(
            polarisation, projection, problem, bundle
        )
        coarse = _current_part(operands, projection, problem, bundle)
        coarse_source = (coarse_p + coarse.source) * grids.omega_window
        coarse_j = coarse.source * grids.omega_window
        coarse_j_floor = coarse.floor
        coarse_p_floor = _roundoff_floor(
            np.abs(polarisation), projection, prefactor, problem, bundle
        )
    floors = (fine_j_floor, fine_p_floor, coarse_j_floor, coarse_p_floor)
    if not all(np.isfinite(value) for value in floors) or (
        coarse_source is not None and not np.all(np.isfinite(coarse_source))
    ):
        raise BackendError("TDSE projected diagnostic is nonfinite")
    diagnostics: dict[str, object] = {
        "available": available,
        "kind": "embedded_same_rows",
        "includes_spectral_collar": True,
        "complete_source": _projection_comparison(
            source * grids.omega_window,
            coarse_source,
            fine_p_floor + fine_j_floor,
            coarse_p_floor + coarse_j_floor,
        ),
        "remainder_source": _projection_comparison(
            fine_j * grids.omega_window,
            coarse_j,
            fine_j_floor,
            coarse_j_floor,
        ),
    }
    if fine.edge is not None:
        assert fine.bounds is not None
        diagnostics["edge"] = _edge_diagnostic(
            (fine.truncating, fine.edge),
            fine.bounds,
            operands.factor,
            result,
            problem,
            bundle,
        )
    return source, diagnostics


def _current_operands(
    current: Float[np.ndarray, "n_nodes 1 n_to"],
    edge: Float[np.ndarray, "n_nodes_e 1 n_to_e"] | None,
    result: TDSEResult,
    problem: KernelProblem,
    bundle: StageScalars,
) -> _CurrentOperands:
    """Gather one evaluation's current operands and check the edge array.

    Parameters
    ----------
    current : numpy.ndarray
        Scaled (gated) remainder current, shape ``(n_nodes, 1, n_to)``.
    edge : numpy.ndarray or None
        Scaled ``g' S``: required for a gated source, refused for an ungated one.
    result : TDSEResult
        Unscaled microscopic operands.
    problem : KernelProblem
        Descriptor, transform and current numerator.
    bundle : StageScalars
        Density, scale, position and taper scale.

    Returns
    -------
    _CurrentOperands
        The operands shared by the fine and embedded projections.

    Raises
    ------
    BackendError
        If the edge array's presence, shape or finiteness is wrong, or a cubic
        replacement lacks its third-order operand.
    """
    descriptor = problem.tdse
    assert descriptor is not None and problem.current_numerator is not None
    gated = descriptor.gated
    if gated != (edge is not None):
        raise BackendError(
            "a gated source requires its edge array, and an ungated source has none"
        )
    if edge is not None and (
        edge.shape != current.shape or not np.all(np.isfinite(edge))
    ):
        raise BackendError("TDSE edge array must be finite and match the transform")
    raw = np.abs(result.total_current) + np.abs(result.first_order_current)
    operands = raw
    if descriptor.kerr_model != "tdse_native":
        if result.third_order_current is None:
            raise BackendError(
                "TDSE cubic replacement requires third-order cancellation operands"
            )
        operands = operands + np.abs(result.third_order_current)
    if gated:
        operands = operands * _source_gate(problem)
    scale = abs(bundle.density * descriptor.remainder_scale)
    return _CurrentOperands(
        current=current,
        edge=edge,
        magnitude=scale * operands[:, None, :],
        edge_magnitude=None
        if not gated
        else scale * (raw * np.abs(descriptor.source_gate_derivative()))[:, None, :],
        derivative_magnitude=None
        if not gated
        else scale * (raw * descriptor.source_gate_derivative_error())[:, None, :],
        factor=problem.current_numerator / power_weight_at(problem, bundle.z1),
        inverse=_inverse_i_omega(problem) if gated else None,
    )

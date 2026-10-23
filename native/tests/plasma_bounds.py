# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The field plasma column's rounding bound, carried through its scans (solprop SEP 25).

The column forms two running integrals per node (solprop SEP 27): the
exposure ``X = int W``, the free current ``j = int w E`` with ``w = sum_c N_c f_c
(q^2/m)_c e_c`` and ``f = 1 - (1 - p) exp(-max(X, 0))``. The total current adds
``L = sum_c N_c I_c W_c (1 - f_c) E / |E|^2`` without a third integral. Each
integral is a band filter of ``2K + 1`` taps (one identity tap for the trapezoid)
followed by a cumulative trapezoid. Two implementations of the same discrete column,
this backend's and the reference's, are held to a bound fixed from their operation
counts and the magnitudes entering each sum, before either runs:

1. **A scan.** For increments ``y`` known to within ``e``, let ``M[a]_n = sum_{i <= n}
   dt (A_i + A_{i-1}) / 2`` with ``A_i = |c_0| a_i + sum_k |c_k| (a_{i-k} + a_{i+k})``,
   a reflected sample ``2 y_0 - y_m`` entering as ``2 a_0 + a_m``. Forming each
   increment costs ``gamma_{2K+6}`` of its magnitudes (the stencil's ``2K + 2``, a
   reflection's one, the trapezoid's three): ``gamma_{2K+6} M[|y|]``. Adding them, any
   tree's addition forms a sum over a contiguous range of increments, a difference of
   two prefixes, so at most ``2 R_n`` with ``R_n`` the largest prefix magnitude so far,
   and a prefix receives at most ``2 (n + T)`` additions (a chunked host team scan of
   ``T`` threads, a device's up and down sweeps, or the tiled column's tile scan,
   reduction of the earlier tiles' totals and the addition joining them, a tree of
   ``n - 1`` additions): ``4 u (n + T) R_n`` (Higham's
   recursive-summation bound, ``u`` times the partial sums, rather than ``gamma_D``
   times the increments' magnitudes: the currents' prefixes cancel over carrier cycles
   and ``gamma_D M[|y|]`` with ``D = n_to + T + 2K + 6`` would exceed them by orders of
   magnitude). Two implementations then differ by twice those, to first order in ``u``
   (a factor ``1 + gamma_D`` covers the second), plus ``(1 + gamma_D) M[e]``.
2. **The field** each side synthesises from the same state, through its own inverse
   transforms: within :func:`bounds.fft_bound` of the modal fields' 2-norms per
   sample (a normwise bound is a pointwise one), carried through the synthesis
   weights' magnitudes, plus the synthesis sum's ``gamma``.
3. **The rate** at a magnitude known to ``delta``: the rate's change over ``[|E| -
   delta, |E| + delta]`` (a rate is monotone in the field, so the ends enclose it), plus
   the operand-derived logarithm, cubic and exponential bound below. This retains
   both adjacent polynomials when interval selection can straddle a knot.
4. **The fraction** moves by at most ``(1 - p) exp(-max(X - dX, 0)) dX``, plus each
   side's own roundings, ``2 u (4 s + f)`` with ``s = (1 - p) exp(-X)``: the
   exponential's two, the product's and ``1 - p``'s on ``s``, the difference's on
   ``f``.
5. **The integrands** ``w E`` and ``L``: their inputs' errors through their
   derivatives (``|d(E_p / |E|^2)| <= 3 delta / |E|^2`` for a vector field), and a few
   roundings each on their magnitudes (``4 + 2C`` for ``C`` channels on ``w``, eight on
   ``L``).
6. **The held source.** The products gJ and g'J each carry the current error,
   multiplication rounding and the FFT normwise error. Their independent errors
   enter as ``e_X/omega + e_Z/omega^2``; neither cancellation nor the global
   steady-current certificate bounds this rounding. The projection sum's gamma and
   reciprocal, square and assembly roundings enter through the absolute operands.
   Projecting in time or frequency has the same absolute-operand bound by the
   triangle inequality. The final crop scaling and source prefactor are explicit.

The result is an absolute bound on the stage's departure, returned relative to the
reference stage's largest magnitude, beside :func:`bounds.fft_bound` for the transforms,
the Kerr term and the products the scans do not carry.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from bounds import UNIT_ROUNDOFF, fft_bound, higham_gamma
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.grid import FieldGrid
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.translate.ionisation import RateTable
from solprop.translate.problem import KernelProblem
from solprop.translate.spatial import separable_values

type Real = npt.NDArray[np.float64]

U = UNIT_ROUNDOFF


def scan_depth(n_to: int, threads: int, half: int) -> int:
    """Return ``D = n_to + T + 2K + 6``, the column's deepest prefix (module item 1).

    The scan's share is ``n_to + T``. A chunked host team scan of ``T`` threads
    reaches a prefix through at most ``2m + T - 3`` additions with ``m = ceil(n_to /
    T)``, within it. The tiled column (tiles of ``L`` samples, ``B = ceil(n_to / L)``
    of them) joins, in one addition, a tile's scan (a tree over at most ``L``
    increments, depth at most ``L - 1``) and the reduction of at most ``B - 1``
    earlier totals (depth at most ``B - 2`` over totals of depth at most ``L - 1``):
    at most ``L + B <= n_to + 1`` for ``1 <= L <= n_to``, so within ``n_to + T`` for
    every ``T >= 1``.
    """
    return n_to + threads + 2 * half + 6


def scan_magnitude(magnitude: Real, rule: RunningIntegral, dt: float) -> Real:
    """Return ``M[a]``: the prefix sums of the filtered trapezoid's magnitudes.

    `magnitude` is nonnegative along its last axis; a reflected sample ``2 y_0 - y_m``
    enters as ``2 a_0 + a_m`` (module item 1).
    """
    a = np.asarray(magnitude, dtype=float)
    half = rule.half_width
    if half:
        taps = np.abs(np.asarray(rule.taps, dtype=float))
        n_to = a.shape[-1]
        reach = np.arange(half, 0, -1)
        head = 2.0 * a[..., :1] + a[..., reach]
        tail = 2.0 * a[..., -1:] + a[..., n_to - 1 - reach[::-1]]
        extended = np.concatenate([head, a, tail], axis=-1)
        filtered = taps[0] * a
        for k in range(1, half + 1):
            filtered = filtered + taps[k] * (
                extended[..., half - k : half - k + n_to]
                + extended[..., half + k : half + k + n_to]
            )
        a = filtered
    increments = dt * (a[..., 1:] + a[..., :-1]) / 2.0
    zero = np.zeros((*a.shape[:-1], 1))
    return np.concatenate([zero, np.cumsum(increments, axis=-1)], axis=-1)


def scan_bound(
    prefix: Real,
    magnitude: Real,
    error: Real,
    rule: RunningIntegral,
    dt: float,
    threads: int,
) -> Real:
    """Bound two implementations' prefixes of one scan apart (module item 1).

    `prefix` is the scan's value (the reference's), `magnitude` its integrand's
    magnitude and `error` the integrand's difference between the two sides.
    """
    n_to = prefix.shape[-1]
    depth = scan_depth(n_to, threads, rule.half_width)
    # Forming each increment: the stencil's 2K + 2, a reflection's one, the
    # trapezoid's three, on each side.
    formed = (
        2.0
        * higham_gamma(2 * rule.half_width + 6)
        * scan_magnitude(magnitude, rule, dt)
    )
    # Adding them: every addition forms a sum over a contiguous range of increments,
    # a difference of two prefixes, so at most twice the largest prefix so far; a
    # prefix receives at most 2 (n + T) of them. Both sides, to first order in u.
    largest = np.maximum.accumulate(np.abs(prefix), axis=-1)
    count = np.arange(1, n_to + 1, dtype=float)
    added = 2.0 * (1.0 + higham_gamma(depth)) * U * 4.0 * (count + threads) * largest
    return (
        formed + added + (1.0 + higham_gamma(depth)) * scan_magnitude(error, rule, dt)
    )


def rate_roundoff(table: RateTable, magnitude: Real, rate: Real) -> Real:
    """Bound one evaluation of exp(cubic(log |E|)) from its actual operands.

    With normal field components, higham_gamma(8) bounds the norm's relative error.
    Assuming one-epsilon log/exp, delta(log E) <= higham_gamma(8) + eps |log E|.
    Forming x = log E - (log_min + index*d) adds higham_gamma(3) times the sum
    of those absolute operands. The cubic contributes its derivative bound
    times delta(x), plus higham_gamma(6) times its absolute Horner polynomial.
    Exponentiation turns delta(p) into expm1(delta(p)) relative error.

    Near an interval boundary, either adjacent stored polynomial can be
    selected. Their evaluated difference and both evaluation bounds cover
    any rounding discontinuity without assuming exact spline continuity.
    ``rate_spread`` also encloses the norm uncertainty across either table clip.
    No compiled result or observed discrepancy enters this calculation.
    """
    out = np.zeros_like(rate)
    inside = (magnitude >= table.field_min) & (magnitude <= table.field_max)
    logarithm = np.log(magnitude[inside])
    log_error = higham_gamma(8) + (2 * U) * abs(logarithm)
    coordinate = (logarithm - table.log_field_min) / table.d_log_field
    coordinate_error = (
        log_error + higham_gamma(2) * (abs(logarithm) + abs(table.log_field_min))
    ) / table.d_log_field
    last = table.coefficients.shape[1] - 1
    index = np.clip(coordinate.astype(np.intp), 0, last)

    def polynomial(indices: npt.NDArray[np.intp]) -> tuple[Real, Real]:
        centre = table.log_field_min + indices * table.d_log_field
        local = logarithm - centre
        local_error = log_error + higham_gamma(3) * (
            abs(logarithm) + abs(table.log_field_min) + abs(indices * table.d_log_field)
        )
        c = table.coefficients[:, indices]
        radius = abs(local) + local_error
        condition = (
            (abs(c[0]) * radius + abs(c[1])) * radius + abs(c[2])
        ) * radius + abs(c[3])
        derivative = (3 * abs(c[0]) * radius + 2 * abs(c[1])) * radius + abs(c[2])
        value = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
        return value, local_error * derivative + higham_gamma(6) * condition

    value, base_error = polynomial(index)
    error = base_error.copy()
    for neighbour in (coordinate - coordinate_error, coordinate + coordinate_error):
        alternate = np.clip(np.floor(neighbour).astype(np.intp), 0, last)
        other_value, other_error = polynomial(alternate)
        error = np.maximum(
            error,
            np.where(
                alternate == index,
                base_error,
                abs(other_value - value) + other_error + base_error,
            ),
        )
    # Account for rounding of the reference exponential used to express the
    # absolute scale, and round the bound outward after its own arithmetic.
    out[inside] = rate[inside] * (np.expm1(error) + (2 * U)) / (1 - (2 * U))
    # A clipped zero rate is exact. Rounding it outward to the smallest
    # subnormal would invent loss work at zero field and force a singular
    # division in the downstream loss-current bound.
    return np.where(inside, np.nextafter(out * (1 + higham_gamma(32)), np.inf), 0.0)


def rate_spread(table: RateTable, magnitude: Real, delta: Real) -> tuple[Real, Real]:
    """Return the rate at `magnitude` and its bound over ``|E| +- delta`` (item 3)."""
    delta = delta + higham_gamma(8) * magnitude
    rate = np.asarray(passes.ionisation_rate(magnitude[:, None, :], table), float)
    high = np.asarray(
        passes.ionisation_rate((magnitude + delta)[:, None, :], table), float
    )
    low = np.asarray(
        passes.ionisation_rate(np.maximum(magnitude - delta, 0.0)[:, None, :], table),
        float,
    )
    spread = np.maximum(np.abs(high - rate), np.abs(rate - low))
    return rate, spread + 2 * rate_roundoff(table, magnitude, rate)


def node_field_error(
    reference: NumpyRefBackend, problem: KernelProblem, fields: Real
) -> Real:
    """Bound each node field's difference between two syntheses (module item 2)."""
    transform = problem.transform
    grids = problem.grids
    if problem.linear.separable is not None:
        # One transverse and one temporal transform of the whole field: normwise
        # within fft_bound of the field's 2-norm over every node and sample.
        size = grids.n_to * transform.n_nodes
        return np.full(fields.shape, fft_bound(size) * float(np.linalg.norm(fields)))
    state = reference._require_state()[0]  # pyright: ignore[reportPrivateUsage]
    kind = reference._kind  # pyright: ignore[reportPrivateUsage]
    bundle = reference._stage_scalars(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
    taper = float(bundle.taper_scale)
    modal = np.real_if_close(
        passes.to_time(state, grids.crop_index, grids.n_to, kind=kind)
    )
    weights = np.abs(np.asarray(transform.synthesis, dtype=float)) * taper
    norms = np.linalg.norm(modal, axis=-1)  # (modes, n_pol_state)
    per_row = fft_bound(grids.n_to) * (weights @ norms)  # (rows, n_pol_state)
    summed = np.tensordot(weights, np.abs(modal), axes=([1], [0]))
    rows = per_row[..., None] + higham_gamma(2 * modal.shape[0]) * summed
    n_rows, n_pol_state, n_to = rows.shape
    return rows.reshape(n_rows // transform.n_pol, transform.n_pol * n_pol_state, n_to)


def current_error(
    problem: KernelProblem,
    grid: FieldGrid,
    fields: Real,
    field_error: Real,
    densities: tuple[float, ...],
    preionisation: tuple[float, ...],
    threads: int,
) -> tuple[Real, Real]:
    """Return the total current's error and absolute operands (items 3 to 5)."""
    responses = problem.responses
    rule = responses.plasma_integral
    dt = grid.dt_fine
    channels = responses.plasma_channels
    if len(channels) <= 1:
        coefficients = responses.plasma_coefficients
        assert coefficients is not None
        groups = [(coefficients, 1.0)]
    else:
        groups = [(channel, channel.electrons_per_event) for channel in channels]
    # The magnitude |E| and its spread: a 2-norm over components moves by at most the
    # 2-norm of their errors, which their sum bounds.
    magnitude = np.sqrt(np.sum(fields**2, axis=1))
    delta = np.sum(field_error, axis=1)
    low = np.maximum(magnitude - delta, np.finfo(float).tiny)
    weight = np.zeros_like(magnitude)
    weight_error = np.zeros_like(magnitude)
    work = np.zeros_like(magnitude)
    work_error = np.zeros_like(magnitude)
    for (group, electrons), density, initial in zip(
        groups, densities, preionisation, strict=True
    ):
        rate, rate_error = rate_spread(group.rate, magnitude, delta)
        exposure = rule.floor(rule.integrate(rate, dt))
        exposure_error = scan_bound(
            rule.integrate(rate, dt), rate, rate_error, rule, dt, threads
        )
        survival = (1.0 - initial) * np.exp(-exposure)
        fraction = 1.0 - survival
        fraction_error = (1.0 - initial) * np.exp(
            -np.maximum(exposure - exposure_error, 0.0)
        ) * exposure_error + 2 * U * (4 * survival + fraction)
        scale = density * group.charge_mass_ratio * electrons
        weight += scale * fraction
        weight_error += scale * fraction_error
        loss = density * group.ionisation_potential
        work += loss * rate * (1.0 - fraction)
        work_error += loss * (rate_error * (1.0 - fraction) + rate * fraction_error)
    weight_error += 2 * (4 + 2 * len(groups)) * U * weight
    components = np.abs(fields)
    # The free current's integrand w E_p and its scan.
    integrand = weight[:, None, :] * components
    integrand_error = (
        weight_error[:, None, :] * components
        + weight[:, None, :] * field_error
        + 4 * U * integrand
    )
    signed_current = rule.integrate(weight[:, None, :] * fields, dt)
    current = np.abs(signed_current)
    current_error = scan_bound(
        signed_current, integrand, integrand_error, rule, dt, threads
    )
    # The loss current, work E_p / |E|^2, at the smallest magnitude the field can have.
    live = (work > 0.0) | (work_error > 0.0)
    squared = np.zeros_like(low)
    np.divide(1.0, low**2, out=squared, where=live)
    inverse = squared[:, None, :]
    loss = work[:, None, :] * components * inverse
    loss_error = (
        work_error[:, None, :] * components * inverse
        + work[:, None, :]
        * (field_error + 2.0 * components * (delta / low)[:, None, :])
        * inverse
        + 16 * U * loss
    )
    # The density and the final sum each contribute rounding on both sides.
    total = current + loss
    return current_error + loss_error + 6 * U * total, total


def stage_error(
    reference: NumpyRefBackend, problem: KernelProblem, error: Real, magnitude: Real
) -> float:
    """Carry independent current-product errors through the held source (item 6)."""
    transform, grids = problem.transform, problem.grids
    projected_depth = 2 * transform.n_nodes * transform.n_pol
    # Pointwise product roundings; normwise FFT error dominates any one output bin.
    errors: list[Real] = []
    operands: list[Real] = []
    for gate in (grids.t_window_fine, grids.t_window_fine_derivative):
        product = abs(gate) * magnitude
        product_error = abs(gate) * error + 2 * U * product
        operand = np.sum(product, axis=-1)
        propagated = np.sum(product_error, axis=-1) + (
            fft_bound(grids.n_to)
            * np.sqrt(grids.n_to)
            * np.linalg.norm(product, axis=-1)
        )
        errors.append(propagated + 2 * higham_gamma(projected_depth) * operand)
        operands.append(operand)
    inverse = np.zeros_like(grids.omega)
    np.divide(1.0, grids.omega, out=inverse, where=grids.active_mask)
    # Each implementation's reciprocal, squared reciprocal and final sum.
    nodes = (errors[0] + 8 * U * operands[0])[..., None] * inverse
    nodes += (errors[1] + 12 * U * operands[1])[..., None] * inverse**2
    window = np.asarray(grids.omega_window, dtype=float)
    if problem.linear.separable is not None:
        _, prefactor = separable_values(problem.linear.separable)
        # The transverse forward transform has unit-modulus entries; the planar
        # transform's absolute projection weights precede it.
        if transform.geometry is not None and transform.geometry.kind == "slab":
            nx, ny = transform.geometry.shape
            nodes = nodes.reshape(ny, nx, *nodes.shape[1:])
            weights = np.sum(abs(transform.projection), axis=0)
            joined = np.sum(nodes * weights[None, :, None, None], axis=(0, 1))
        else:
            joined = np.sum(nodes, axis=0)
        return (
            grids.crop_scale
            * float(np.max(abs(prefactor * window)))
            * float(np.max(joined))
        )
    bundle = reference._stage_scalars(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
    prefactor = reference._prefactor(bundle)  # pyright: ignore[reportPrivateUsage]
    n_nodes, n_pol_node, n_omega = nodes.shape
    columns = nodes.reshape(
        n_nodes * transform.n_pol, n_pol_node // transform.n_pol, n_omega
    )
    weights = abs(np.asarray(transform.projection, dtype=float)) / float(
        bundle.taper_scale
    )
    projected = np.einsum("mn,ncw->mcw", weights, columns)
    return grids.crop_scale * float(np.max(abs(prefactor * window) * projected))


def plasma_stage_bound(problem: KernelProblem, grid: FieldGrid, threads: int) -> float:
    """Return the column's bound on one stage, relative to the stage's largest value.

    At the launch field and ``z = 0``: one evaluation, or each stage of a short march
    whose field and plasma stay close to the launch's (module docstring).
    """
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)  # pyright: ignore[reportPrivateUsage] - oracle probe
        state = reference._require_state()  # pyright: ignore[reportPrivateUsage]
        bundle = reference._stage_scalars(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        stage = reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
            state, bundle, np.ones(problem.shape, complex)
        )[0]
        fields = np.real(
            np.concatenate(
                list(reference._plasma_nodes(state[0]))  # pyright: ignore[reportPrivateUsage]
            )
        )
        field_error = node_field_error(reference, problem, fields)
        if len(problem.responses.plasma_channels) > 1:
            densities = tuple(bundle.plasma_densities)
            preionisation = tuple(bundle.plasma_preionisation)
        else:
            densities = (
                bundle.plasma_densities[0]
                if bundle.plasma_densities
                else float(bundle.density),
            )
            preionisation = (float(bundle.preionisation),)
        node_error, node_magnitude = current_error(
            problem, grid, fields, field_error, densities, preionisation, threads
        )
        absolute = stage_error(reference, problem, node_error, node_magnitude)
    finally:
        reference.close()
    scale = float(np.max(np.abs(stage)))
    assert scale > 0.0
    return absolute / scale


REGRESSION_GUARD = 1e-11
"""The band column's regression guard, relative to the reference's largest value.

Not a rounding bound: an empirical tolerance, kept beside the derived bounds of this
module by the maintainer's decision (solprop SEP 25, Phase 4 report). The derived chain
must assume no cancellation in the currents' scans and so allows up to 3e-3 on the
stress set, too loose to see a subtle defect; this guard catches one. Set on
2026-10-18 from the departures measured on Apple silicon (Clang, OpenMP, four
threads), at most 6.6e-14 (the leading-edge stress case's one evaluation) and 1e-15 or
less on most cases, times a margin of 150. A deliberate change of the column's
arithmetic that exceeds it is re-measured and the guard reset in its report, never
widened silently; a build that exceeds it otherwise has a defect to find.
"""


def assert_regression_guard(
    actual: npt.ArrayLike, expected: npt.ArrayLike, what: str
) -> None:
    """Hold `actual` within :data:`REGRESSION_GUARD` of `expected`'s largest value.

    Parameters
    ----------
    actual, expected : array_like
        The backend's and the reference's results.
    what : str
        What is compared, for the message.
    """
    have, want = np.asarray(actual), np.asarray(expected)
    scale = float(np.max(np.abs(want)))
    assert scale > 0.0, what
    departure = float(np.max(np.abs(have - want))) / scale
    assert departure <= REGRESSION_GUARD, (
        f"{what}: relative departure {departure:.3e} exceeds the regression guard "
        f"{REGRESSION_GUARD:.0e}"
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Numerical comparison bounds: rounding budgets, accuracy envelopes, assertions.

SEP 25 replaced bitwise agreement between independent floating-point computations by
two criteria, and this module holds the reusable parts of both (``docs/dev/testing.md``
section 3). Case-specific reference data and sensitivity assumptions stay with the
tests that use them.

- **Rounding bound (class R).** Two computations with the same discrete method and
  history -- two backends, thread counts, a batch member and its solo run, a scan point
  and its run -- differ by rounding alone. The bound is fixed from operation and
  reduction counts, transform lengths, magnitudes and a stated margin *before* the
  comparison runs, never fitted to the discrepancy it then accepts:
  :func:`higham_gamma`, :func:`fft_bound`, :func:`scan_bound`, :func:`prefix_bound`,
  :func:`stencil_bound`, :func:`propagation_growth`.
- **Accuracy envelope (class T).** Adaptive runs, ordinary replays and resumed runs
  may take different steps; each is compared with a tighter-tolerance or stored
  converged reference at an envelope fixed in advance (:class:`AccuracyEnvelope`).
  A local ``rtol`` is not a global error bound.

:func:`assert_within` is the comparison: finite values, a nontrivial reference and a
normwise departure inside the bound, with the observed margin in the failure message.
Exact comparison remains right for structural contracts (class P): lossless storage,
integer accounting, identity metadata, hard masks and explicitly exact no-ops.

``u`` is the unit roundoff ``2**-53`` and ``EPS = 2 u`` NumPy's machine epsilon; the
two are not interchangeable (SEP 25 section 4.1). Every formula takes its roundoff as a
parameter, binary64's by default, so that SEP 33's binary32 bounds are the same formulas
evaluated at :data:`UNIT_ROUNDOFF_32` (or :data:`EPS_32`) rather than binary64 bounds
scaled: ``gamma_k``'s denominator matters at binary32, and each formula refuses a
roundoff outside its domain (``k u < 1``). SEP 36's mixed bound composes them per
stage, step and run (:func:`source_pass_bound`, :func:`rate_condition`,
:func:`mixed_step_bound`, :func:`mixed_run_bound`). The established ladder of
``docs/dev/accuracy.md`` section 7 is written in ``EPS``: :func:`gamma`,
:func:`fft_bound` and :func:`scan_bound` keep its constants, which were fixed before
the first device result. The CPU rows retain their established bounds; the Kokkos
device rows add only the FFT and reduction envelopes declared for device qualification.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPS = np.finfo(float).eps
"""NumPy's machine epsilon, ``2 u``."""

UNIT_ROUNDOFF = 2.0**-53
"""The unit roundoff ``u`` of binary64 round-to-nearest arithmetic."""

UNIT_ROUNDOFF_32 = 2.0**-24
"""The unit roundoff of binary32 round-to-nearest arithmetic (SEP 33)."""

EPS_32 = 2.0**-23
"""Binary32's machine epsilon, ``2 u``: the ladder's unit at binary32 (SEP 33)."""


def gamma(operations: int, eps: float = EPS) -> float:
    """Return the ladder's ``k eps / (1 - k eps)``: twice Higham's gamma, in ``EPS``."""
    scaled = operations * eps
    if scaled >= 1.0:
        raise ValueError(f"{operations} operations are beyond gamma's range at {eps}")
    return scaled / (1.0 - scaled)


def fft_bound(samples: int, eps: float = EPS) -> float:
    """Bound one FFT round trip and its associated pointwise operations."""
    return 512 * eps * math.ceil(math.log2(samples))


def scan_bound(samples: int, eps: float = EPS) -> float:
    """Bound the complete fixed-geometry deterministic plasma scan chain."""
    return 192 * eps * math.ceil(math.log2(samples))


def rate_log_sensitivity(rate, value: float, field_range: tuple[float, float]) -> float:
    """Return ``d ln W / d ln F`` of an ionisation rate where it equals ``value``.

    A statistic ``W(F)`` of the field amplitude ``F`` moves by ``S`` times the field's
    relative change, so a rate statistic's rounding bound is its field's times this
    ``S`` (the chain rule). ``F`` is found by bisection in
    ``ln F`` inside ``field_range`` (V/m), where the rate is monotone, and ``S`` by a
    central difference of relative step 1e-4, whose truncation error is far below the
    factor of two any use of it can tolerate.
    """
    low, high = math.log(field_range[0]), math.log(field_range[1])
    for _ in range(200):
        middle = 0.5 * (low + high)
        if float(rate(math.exp(middle))) < value:
            low = middle
        else:
            high = middle
    field = math.exp(0.5 * (low + high))
    step = 1e-4
    rise = math.log(float(rate(field * (1 + step)))) - math.log(
        float(rate(field * (1 - step)))
    )
    return rise / (math.log1p(step) - math.log1p(-step))


def device_backend(backend_name: str) -> str:
    """Identify the execution space without initializing a native device runtime."""
    if backend_name != "native":
        return "none"
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        build_info,
    )

    return build_info().device_backend


def record_backend(backend_name: str) -> str:
    """Distinguish accelerator evidence from the established native CPU series."""
    device = device_backend(backend_name)
    return backend_name if device == "none" else f"{backend_name}_{device}"


def ladder_bound(
    backend_name: str,
    established: float,
    samples: int,
    fft_round_trips: int,
) -> float:
    """Return an unchanged CPU bound or the larger device envelope."""
    if device_backend(backend_name) == "none":
        return established
    return max(established, fft_round_trips * fft_bound(samples))


def analytic_bound(backend_name: str, samples: int) -> float:
    """Return the approved analytic-family bound for this execution space."""
    if device_backend(backend_name) == "none":
        return 1e-13
    return max(1e-13, fft_bound(samples))


def dispersive_device_bound(cpu_bound: float, problem) -> float:
    """Return a dispersive comparison's CPU bound, or its device envelope.

    SEP 6's dispersive stage counts its transforms and node reductions in its CPU bound
    (``evaluation_bound`` of ``test_numba_dispersive_kerr.py``) at ``8 eps
    ceil(log2 n_to)`` per transform. A device build's vendor FFT and reductions are
    held to this module's predeclared envelope instead, :func:`fft_bound` per transform,
    with the same count and the reference's factor of two; the larger bound holds.
    """
    if device_backend("native") == "none":
        return cpu_bound
    leaf = problem.responses.dispersive_kerr
    symmetric = leaf.evaluator == "symmetric"
    transforms = 2 + (3 if symmetric else 6) * leaf.n_channels
    reductions = 2 + (2 if symmetric else 5) * leaf.n_channels
    device = 2 * (
        transforms * fft_bound(problem.grids.n_to)
        + reductions * gamma(problem.transform.n_nodes)
    )
    return max(cpu_bound, device)


def step_evaluations(problem) -> float:
    """Return how many source evaluations' rounding one replayed step carries.

    DOPRI5's stages for the Runge-Kutta march, and the low-storage march's four
    evaluations (SEP 24), whose weights in the step sum to one. An Adams step makes one
    evaluation, but every sample enters the step's predictor and corrector through
    their weights: at the method's top order on equal steps, the sum of their absolute
    values (12.2 and 2.3 for PEC5/6).
    """
    from solprop.stepper import DOPRI5, LowStorageRungeKutta  # noqa: PLC0415
    from solprop.stepper.multistep import integrated_weights  # noqa: PLC0415

    method = problem.stepper.integrator
    if method is None:
        return float(DOPRI5.n_stages)
    if isinstance(method, LowStorageRungeKutta):
        return float(method.evaluations)
    past = -np.arange(method.order, dtype=np.float64)
    predictor = integrated_weights(past)
    corrector = integrated_weights(np.concatenate(([1.0], past)))
    return float(np.sum(np.abs(predictor)) + np.sum(np.abs(corrector)))


def higham_gamma(depth: int, u: float = UNIT_ROUNDOFF) -> float:
    """Return Higham's ``gamma_k = k u / (1 - k u)`` for an addition depth `depth`.

    A sum of fixed inputs evaluated by any tree of depth ``k`` is within
    ``gamma_k sum |x_i|`` of the exact sum (SEP 25 section 4.1); a real dot product of
    length ``n`` within ``gamma_n sum |a_i b_i|``.
    """
    scaled = depth * u
    if scaled >= 1.0:
        raise ValueError(f"depth {depth} is beyond gamma's range at u = {u}")
    return scaled / (1.0 - scaled)


def prefix_bound(increments, depth: int, uncertainty=None, u: float = UNIT_ROUNDOFF):
    """Return the absolute rounding bound of every prefix of a scan, elementwise.

    ``gamma_D sum_{i<=j} |x_i|`` for a scan of addition depth `depth` over already
    computed `increments` along the last axis (SEP 25 section 4.6), plus, when given,
    the prefix sum of the increments' own `uncertainty` scaled by ``1 + gamma_D``. A
    chunked scan of chunk length ``m`` and offset depth ``d`` has ``D = m + d + 1``.
    """
    magnitude = np.cumsum(np.abs(increments), axis=-1)
    bound = higham_gamma(depth, u) * magnitude
    if uncertainty is not None:
        bound = bound + (1.0 + higham_gamma(depth, u)) * np.cumsum(
            np.abs(uncertainty), axis=-1
        )
    return bound


def stencil_bound(taps, values, u: float = UNIT_ROUNDOFF):
    """Return the rounding bound of a symmetric stencil applied along the last axis.

    For ``y_n = c_0 x_n + sum_k c_k (x_{n-k} + x_{n+k})`` the paired evaluation is
    within ``gamma_{2K+2} A_n`` with ``A_n = |c_0 x_n| + sum_k |c_k| (|x_{n-k}| +
    |x_{n+k}|)`` (SEP 25 section 4.6). `taps` is ``(c_0, c_1, ..., c_K)``; the edges
    use the available neighbours (zero padding), which only lowers ``A_n``.
    """
    taps = np.asarray(taps, dtype=np.float64)
    magnitude = np.abs(np.asarray(values))
    radius = taps.size - 1
    total = abs(taps[0]) * magnitude
    for k in range(1, radius + 1):
        shifted = np.zeros_like(magnitude)
        shifted[..., k:] += magnitude[..., :-k]
        shifted[..., :-k] += magnitude[..., k:]
        total = total + abs(taps[k]) * shifted
    return higham_gamma(2 * radius + 2, u) * total


def propagation_growth(lipschitz: float, distance: float) -> float:
    """Return ``exp(L Z)``, the growth a step's rounding can suffer over `distance`.

    The guide of SEP 25 section 4.3: with ``G_j <= 1 + h_j L`` the perturbations of
    ``N`` steps add up to at most ``exp(L Z) (e_0 + N c u Y)``. `lipschitz` is in
    1/m and is a stated assumption of the case, not a measured certainty; a strongly
    amplifying case is compared over a short window instead.
    """
    return math.exp(lipschitz * distance)


def step_budget(
    operations: float, steps: int, growth: float = 1.0, u: float = UNIT_ROUNDOFF
) -> float:
    """Return a relative budget of ``steps`` steps of ``operations`` roundings each.

    ``growth * steps * operations * u``: each step contributes its operation count of
    unit roundoffs relative to the state's scale, carried by `growth`
    (:func:`propagation_growth`). The count is the case's own, stated with its margin.
    """
    budget = growth * steps * operations * u
    if steps * operations * u >= 1.0:
        raise ValueError(f"{steps} steps of {operations} roundings exceed u = {u}")
    return budget


@dataclass(frozen=True)
class AccuracyEnvelope:
    """An accuracy envelope fixed before the candidate is measured (class T).

    Attributes
    ----------
    reference : str
        What the run is compared with: a run at a tighter tolerance or a stored
        converged reference, with its problem, method, distance and observable.
    error : float
        The reference run's own measured error against that reference, or an
        established error of the method at the run's settings, relative.
    factor : float
        The safety factor, fixed in advance and justified beside the test.
    """

    reference: str
    error: float
    factor: float = 4.0

    @property
    def bound(self) -> float:
        """The relative departure a run may have from the reference."""
        return self.factor * self.error


def departure(actual, expected, *, scale=None) -> float:
    """Return ``||actual - expected|| / scale``, scale ``||expected||`` by default."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    denominator = np.linalg.norm(expected) if scale is None else scale
    return float(np.linalg.norm(actual - expected) / denominator)


def assert_within(
    actual,
    expected,
    bound: float | np.ndarray,
    *,
    scale=None,
    what: str = "",
    elementwise: bool = False,
) -> float:
    """Assert two results agree inside a bound fixed beforehand; return the margin.

    Both arrays must be finite and the reference nontrivial (a zero reference makes
    any relative comparison vacuous: test zero results structurally instead). The
    comparison is normwise relative to ``||expected||``, or to `scale` when given
    (an absolute bound passes ``scale=1``). With `elementwise`, `bound` is an array
    of absolute per-element bounds and every element must satisfy it.

    Returns
    -------
    float
        The observed departure as a fraction of the bound: the margin the record
        keeps (SEP 25 section 7.1).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    label = f"{what}: " if what else ""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}shapes differ, {actual.shape} != {expected.shape}"
        )
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))):
        raise AssertionError(f"{label}non-finite values")
    if elementwise:
        error = np.abs(actual - expected)
        limit = np.broadcast_to(np.asarray(bound, dtype=np.float64), error.shape)
        if not np.any(np.abs(expected)):
            raise AssertionError(f"{label}the reference is identically zero")
        worst = float(np.max(error / np.where(limit > 0, limit, np.inf), initial=0.0))
        if np.any(error > limit):
            raise AssertionError(
                f"{label}{int(np.sum(error > limit))} elements outside their bound "
                f"(worst {worst:.3g} of it)"
            )
        return worst
    if scale is None and not np.linalg.norm(expected):
        raise AssertionError(f"{label}the reference is identically zero")
    bound = float(bound)
    if bound <= 0:
        raise AssertionError(f"{label}a bound must be positive, got {bound}")
    found = departure(actual, expected, scale=scale)
    if found > bound:
        raise AssertionError(
            f"{label}departure {found:.3e} exceeds the bound {bound:.3e}"
        )
    return found / bound


def assert_work_within(actual: int, expected: int, *, factor: float, what: str = ""):
    """Assert an adaptive work count is within a band of a baseline (SEP 25 4.4).

    Independent adaptive runs may take different steps, orders and rejections; their
    counts are performance diagnostics, held to a band declared from baseline data,
    never to equality. Exact accounting of the calls actually made is tested
    separately, exactly.
    """
    label = f"{what}: " if what else ""
    if not expected / factor <= actual <= expected * factor:
        raise AssertionError(
            f"{label}{actual} outside [{expected / factor:g}, {expected * factor:g}]"
        )


def source_pass_bound(
    samples: int, *, scans: int = 0, transverse: int = 0, eps: float = EPS_32
) -> float:
    """Return SEP 33's relative bound of one source evaluation's passes, ``E_pass``.

    The ladder SEP 33's class R tests fixed for a stage: the source's transform pair
    and its pointwise response at ``16 eps d`` with ``d = ceil(log2 n) + transverse``,
    the depth of the temporal transform plus that of a transverse transform or matrix
    product, plus, for each of `scans` running integrals of the plasma, twice Higham's
    ``gamma_d`` at that depth (SEP 25 section 4.6). Evaluated at binary32's ``eps`` by
    default, as the source of both reduced modes runs (SEP 36 section 7.1, item 1).
    """
    depth = math.ceil(math.log2(samples)) + transverse
    return 16 * eps * depth + scans * 2 * higham_gamma(depth, eps / 2)


def rate_condition(rate, fields) -> float:
    """Return the plasma's condition number ``1 + max |d ln W / d ln |E||``.

    SEP 36 section 7.1, item 1: a relative change ``delta`` of the field moves the
    rate by its logarithmic slope times ``delta``, the ionised fraction (a running
    integral of the rate) by at most as much, and the current, which also carries the
    field, by one more ``delta``. The slope is read from the rate on the host by a
    central difference of relative step 1e-4 at every field magnitude in `fields`
    (V/m) where the rate is positive, so the steepness near a table's cutoffs enters
    explicitly; magnitudes where the rate vanishes on both sides contribute nothing.
    """
    magnitudes = np.unique(np.abs(np.asarray(fields, dtype=np.float64)))
    magnitudes = magnitudes[magnitudes > 0]
    step = 1e-4
    upper = np.asarray(rate(magnitudes * (1 + step)), dtype=np.float64)
    lower = np.asarray(rate(magnitudes * (1 - step)), dtype=np.float64)
    live = (upper > 0) & (lower > 0)
    if not np.any(live):
        return 1.0
    slopes = (np.log(upper[live]) - np.log(lower[live])) / (
        math.log1p(step) - math.log1p(-step)
    )
    return 1.0 + float(np.max(np.abs(slopes)))


def mixed_step_bound(  # noqa: PLR0913 - the terms of SEP 36 section 7.1
    phase: float,
    *,
    kappa: float,
    passes: float,
    weights,
    stages: int,
    absolute: float = 0.0,
    u: float = UNIT_ROUNDOFF_32,
) -> float:
    """Return one step's relative state error under mixed precision (SEP 36 7.1).

    Item 1: a stage's source differs from binary64's by ``kappa u + E_pass`` relative
    to its normwise magnitude (the narrowing of the binary64 input carried by the
    response's condition number `kappa`, the passes by `passes`), plus the absolute
    underflow term `absolute` relative to the state; the binary64 inverse propagator
    scales the absolute bound bin by bin and leaves the relative one unchanged.
    Item 2: the step's state error is ``phase sum_i |b_i| (kappa u + E_pass)``, the
    stages' errors weighted by the solution weights `weights` and carried into the
    step by its nonlinear phase ``phase = dz ||N|| / ||A||``, times
    ``(1 + kappa phase)**stages`` for their propagation into later stages.
    """
    source = kappa * u + passes
    transport = (1.0 + kappa * phase) ** stages
    return (
        phase * float(np.sum(np.abs(np.asarray(weights)))) * source * transport
        + absolute
    )


def mixed_run_bound(
    step_bounds,
    *,
    samples: int,
    stages: int,
    growth: float = 1.0,
    eps: float = EPS,
) -> float:
    """Return a mixed run's relative bound on a fixed schedule (SEP 36 7.1, item 3).

    The per-step bounds of :func:`mixed_step_bound` summed and multiplied by the
    case's `growth` (:func:`propagation_growth`), plus the binary64 state path's own
    rounding per step at the ladder's binary64 constants (the state rounded twice and
    each stage's binary64 linear passes at ``16 eps ceil(log2 n)``), which is
    negligible beside the binary32 terms but stated. For a comparison of two sides,
    add each side's bound (item 4).
    """
    bounds = np.asarray(step_bounds, dtype=np.float64)
    depth = math.ceil(math.log2(samples))
    state_path = bounds.size * (2 * eps + stages * 16 * eps * depth)
    return growth * (float(np.sum(bounds)) + state_path)


Bound = Callable[..., float]
"""A bound function, for parametrised tests that carry one per case."""


PHI_ULPS = 256
"""Unit roundoffs of the phi functions' natural scale ``exp(max(Re x, 0)) / r!``.

The witnesses of ``tests/backends/test_exponential_weights.py`` hold both algorithms of
:func:`solprop.backends.exponential.phi_functions` within 256 ulp of that scale."""


def carrier_increment_budget(  # noqa: PLR0913 - the increment's operands
    coefficients, moments, samples, nodes, shift, shifted, *, phase_rounding=0.0
):
    """Bound the rounding of one carrier-frame increment ``sum_k M_k sum_j c_jk G_j``.

    SEP 35's step adds three operations to the exponential increment (section 6.2):
    the rotation ``G_j = exp(sigma (1 - theta_j)) F_j``, the subtraction
    ``x' = x - sigma`` before the moments, and the shift's own phase, read from the
    exponent. Elementwise, with ``u`` the unit roundoff and ``n`` the nodes:

    - a rotated sample is within ``|F_j| r_j``,
      ``r_j = (u |sigma| + d) |1 - theta_j| + 6 u``: the argument's product, a complex
      exponential (three correctly rounded real functions) and the multiply, plus the
      shift's phase uncertainty ``d`` (`phase_rounding`);
    - a coefficient sum ``S_k`` within ``sum_j |c_jk| |F_j| (r_j + gamma_n + n^2 u)``,
      the last term the Lagrange coefficients' own formation;
    - a moment ``M_k = k! phi_{k+1}(x')`` within its evaluation's ``PHI_ULPS`` of the
      natural scale ``E / (k + 1)``, ``E = exp(max(Re x', 0))``, plus its argument's
      ``u |x'| + d`` carried by ``|M_k'| <= E / (k + 1)``;
    - the outer sum within ``gamma_n sum_k |M_k| |S_k|``.

    The rotation's unit modulus keeps ``|G_j| = |F_j|``, so the interpolation's
    conditioning is the unshifted formulation's; an imaginary shift never amplifies.
    Two implementations of the same step differ by at most twice the bound.

    Parameters
    ----------
    coefficients : array_like
        ``c_jk``, shape ``(n_nodes, n_k)``.
    moments : array_like
        ``M_k`` at ``x'``, shape ``(n_k, *shape)``.
    samples : array_like
        ``F_j``, shape ``(n_nodes, *shape)``.
    nodes : array_like
        ``theta_j``, shape ``(n_nodes,)``.
    shift : array_like
        ``sigma``, broadcasting against an element.
    shifted : array_like
        ``x'``, shape ``shape``.
    phase_rounding : float or array_like, optional
        An absolute bound on the shift's phase as the backend forms it; 0 when the
        exponent is given exactly.

    Returns
    -------
    numpy.ndarray
        The absolute bound per element, shape ``shape``.
    """
    coefficients = np.abs(np.asarray(coefficients, dtype=np.complex128))
    moments = np.abs(np.asarray(moments))
    samples = np.abs(np.asarray(samples))
    nodes = np.asarray(nodes, dtype=np.float64)
    shift = np.abs(np.asarray(shift))
    shifted = np.asarray(shifted)
    n_nodes, n_k = coefficients.shape
    u = UNIT_ROUNDOFF
    extra = (1,) * (samples.ndim - 1)
    spans = np.abs(1.0 - nodes).reshape((n_nodes, *extra))
    rotation = (u * shift + phase_rounding) * spans + 6 * u
    scale = np.exp(np.maximum(np.real(shifted), 0.0))
    argument = u * np.abs(shifted) + phase_rounding
    total = np.zeros(samples.shape[1:])
    for k in range(n_k):
        weights = coefficients[:, k].reshape((n_nodes, *extra))
        magnitude = np.sum(weights * samples, axis=0)
        sum_error = np.sum(
            weights * samples * (rotation + higham_gamma(n_nodes) + n_nodes**2 * u),
            axis=0,
        )
        moment_error = (PHI_ULPS * u + argument) * scale / (k + 1)
        total = (
            total
            + moments[k] * sum_error
            + moment_error * magnitude
            + higham_gamma(n_k) * moments[k] * magnitude
        )
    return total

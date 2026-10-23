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
two are not interchangeable (SEP 25 section 4.1). The established ladder of
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


def gamma(operations: int) -> float:
    """Return the ladder's ``k eps / (1 - k eps)``: twice Higham's gamma, in ``EPS``."""
    scaled = operations * EPS
    return scaled / (1.0 - scaled)


def fft_bound(samples: int) -> float:
    """Bound one FFT round trip and its associated pointwise operations."""
    return 512 * EPS * math.ceil(math.log2(samples))


def scan_bound(samples: int) -> float:
    """Bound the complete fixed-geometry deterministic plasma scan chain."""
    return 192 * EPS * math.ceil(math.log2(samples))


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


def higham_gamma(depth: int) -> float:
    """Return Higham's ``gamma_k = k u / (1 - k u)`` for an addition depth `depth`.

    A sum of fixed inputs evaluated by any tree of depth ``k`` is within
    ``gamma_k sum |x_i|`` of the exact sum (SEP 25 section 4.1); a real dot product of
    length ``n`` within ``gamma_n sum |a_i b_i|``.
    """
    scaled = depth * UNIT_ROUNDOFF
    if scaled >= 1.0:
        raise ValueError(f"depth {depth} is beyond gamma's range")
    return scaled / (1.0 - scaled)


def prefix_bound(increments, depth: int, uncertainty=None):
    """Return the absolute rounding bound of every prefix of a scan, elementwise.

    ``gamma_D sum_{i<=j} |x_i|`` for a scan of addition depth `depth` over already
    computed `increments` along the last axis (SEP 25 section 4.6), plus, when given,
    the prefix sum of the increments' own `uncertainty` scaled by ``1 + gamma_D``. A
    chunked scan of chunk length ``m`` and offset depth ``d`` has ``D = m + d + 1``.
    """
    magnitude = np.cumsum(np.abs(increments), axis=-1)
    bound = higham_gamma(depth) * magnitude
    if uncertainty is not None:
        bound = bound + (1.0 + higham_gamma(depth)) * np.cumsum(
            np.abs(uncertainty), axis=-1
        )
    return bound


def stencil_bound(taps, values):
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
    return higham_gamma(2 * radius + 2) * total


def propagation_growth(lipschitz: float, distance: float) -> float:
    """Return ``exp(L Z)``, the growth a step's rounding can suffer over `distance`.

    The guide of SEP 25 section 4.3: with ``G_j <= 1 + h_j L`` the perturbations of
    ``N`` steps add up to at most ``exp(L Z) (e_0 + N c u Y)``. `lipschitz` is in
    1/m and is a stated assumption of the case, not a measured certainty; a strongly
    amplifying case is compared over a short window instead.
    """
    return math.exp(lipschitz * distance)


def step_budget(operations: float, steps: int, growth: float = 1.0) -> float:
    """Return a relative budget of ``steps`` steps of ``operations`` roundings each.

    ``growth * steps * operations * u``: each step contributes its operation count of
    unit roundoffs relative to the state's scale, carried by `growth`
    (:func:`propagation_growth`). The count is the case's own, stated with its margin.
    """
    return growth * steps * operations * UNIT_ROUNDOFF


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


Bound = Callable[..., float]
"""A bound function, for parametrised tests that carry one per case."""

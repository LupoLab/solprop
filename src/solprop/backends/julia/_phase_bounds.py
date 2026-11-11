# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound represented phase operands without certifying a DF32 arithmetic graph.

All envelopes concern the exact real values of the supplied binary64/binary32
coefficients. Nonnegative floating accumulations round outward after every operation;
scalar geometry and endpoint jumps use exact rational arithmetic. See
``julia/PHASE_ADMISSION.md`` for the distinction from an assembly-error certificate.
No function in this module admits a precision, backend, device or restored history.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.backends.julia._descriptors import CoefficientTransfer, TableTransfer
from solprop.backends.julia._lower import RunTransfer
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot
from solprop.stepper.multistep import MAX_ADAMS_ORDER
from solprop.stepper.schedule import LANDING_ULPS

__all__ = [
    "ComplexEnvelope",
    "OperatorEnvelope",
    "PhaseOperands",
    "PhaseSpans",
    "operator_envelope",
    "phase_operands",
    "phase_spans",
]


@dataclass(frozen=True)
class ComplexEnvelope:
    """Bound the absolute real and imaginary components independently.

    Attributes
    ----------
    real, imag : float
        Nonnegative outward bounds, in the represented operand's units. These
        are maxima over all computed cells, including inactive spectral bins.
    """

    real: float
    imag: float

    def __post_init__(self) -> None:
        """Reject nonfinite or negative component bounds."""
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.real, self.imag)
        ):
            raise _failure("complex envelopes must be finite and nonnegative")


@dataclass(frozen=True)
class PhaseSpans:
    """Describe the position and distance operands of a resolved march.

    Attributes
    ----------
    start, end, oldest : float
        Actual starting anchor, requested end and earliest retained sample, metres.
    step_max, step_min : float or None
        Largest prescribed remaining replay step or adaptive ceiling, metres.
        The smallest actual replay step is known; adaptive ``step_min`` is None,
        since a forced landing can be shorter than ``dz_min``.
    stage_scale : float
        Maximum of one and the absolute actual tableau nodes.
    transport_span : float
        Conservative distance covering configured Adams order and retained age,
        including landing allowances, metres. This alone is not a certificate.
    position_lower, position_upper : float
        Required complete represented-coordinate span, metres, including actual
        replay arithmetic, possible adaptive endpoint rounding and old samples.
    landing_slack : float
        Conservative shared landing/replay and endpoint arithmetic allowance, metres.
    compensation_factor_max : float or None
        Bound on ``abs(1-(sample_z-z)/h)``. None means no positive actual lower
        step bound is known; an adaptive floor is never substituted for one.
    """

    start: float
    end: float
    oldest: float
    step_max: float
    step_min: float | None
    stage_scale: float
    transport_span: float
    position_lower: float
    position_upper: float
    landing_slack: float
    compensation_factor_max: float | None


@dataclass(frozen=True)
class OperatorEnvelope:
    """Keep exact-representation envelopes and uncancelled assembly operands.

    Attributes
    ----------
    kind : str
        ``constant``, ``represented`` or ``expansion``.
    cumulative : ComplexEnvelope or None
        Absolute cumulative Phi envelope, dimensionless; absent for a stored
        constant generator, which has no cumulative evaluation operand.
    rate : ComplexEnvelope
        Derivative/generator envelope, inverse metres, over complete intervals.
    jumps : ComplexEnvelope
        Sum of represented boundary-jump magnitudes, dimensionless. Exact
        continuity is not inferred from physical intent or sampled values.
    difference : ComplexEnvelope
        Endpoint-difference envelope over the complete transport span.
    absolute_contributions : ComplexEnvelope
        Sum of absolute assembly contributions before term cancellation. This
        retains cumulative endpoint operands even if their difference is tiny.
    coefficient_magnitudes : ComplexEnvelope
        Largest componentwise cumulative coefficient or fixed-term magnitude in
        stored units. An expansion's weight coefficients need separate conversion.
        Conversion and intermediate representability still need separate proofs.
    term_count, max_degree : int
        Number of fixed expansion terms and maximum represented polynomial degree.
    """

    kind: Literal["constant", "represented", "expansion"]
    cumulative: ComplexEnvelope | None
    rate: ComplexEnvelope
    jumps: ComplexEnvelope
    difference: ComplexEnvelope
    absolute_contributions: ComplexEnvelope
    coefficient_magnitudes: ComplexEnvelope
    term_count: int
    max_degree: int


@dataclass(frozen=True)
class PhaseOperands:
    """Report operand bounds and the unresolved conditions for future certification.

    Attributes
    ----------
    spans : PhaseSpans
        Recorded and possible march geometry.
    operator : OperatorEnvelope
        Complete represented value/rate/jump/assembly envelopes.
    step_argument : ComplexEnvelope
        Bound for stage/step endpoint exponents as exact represented differences.
    transported_argument : ComplexEnvelope or None
        Bound for exact compensated Adams transport arguments, when available.
        None explicitly records unbounded jump amplification without a lower h.
    restored_rates : ComplexEnvelope or None
        Actual selected stored rate rows, inverse metres; never replaced by the
        current represented derivative or inferred from inactive rows.
    unresolved : tuple of str
        Missing arithmetic-graph, conversion, conditioning and device proofs.
        This record is deliberately not an admission certificate.
    """

    spans: PhaseSpans
    operator: OperatorEnvelope
    step_argument: ComplexEnvelope
    transported_argument: ComplexEnvelope | None
    restored_rates: ComplexEnvelope | None
    unresolved: tuple[str, ...]


def _failure(message: str) -> BackendError:
    """Build the named operand-analysis refusal.

    Parameters
    ----------
    message : str
        Specific invalid operand or unresolved input.

    Returns
    -------
    BackendError
        Named analysis refusal.
    """
    return BackendError(f"julia phase operands: {message}")


def _rational(value: float) -> Fraction:
    """Recover a finite binary float's exact real value.

    Parameters
    ----------
    value : float
        Finite binary32 or binary64 value.

    Returns
    -------
    fractions.Fraction
        Exact represented real value.
    """
    return Fraction.from_float(float(value))


def _directed(value: Fraction, *, upper: bool = True) -> float:
    """Round a finite rational outward, refusing binary64 overflow.

    Parameters
    ----------
    value : fractions.Fraction
        Exact finite bound.
    upper : bool, optional
        Select rounding toward positive infinity; otherwise negative infinity.

    Returns
    -------
    float
        Outward binary64 bound, or a named overflow refusal.
    """
    try:
        rounded = float(value)
    except OverflowError as error:
        raise _failure("an outward bound exceeds the finite binary64 range") from error
    represented = _rational(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    if not math.isfinite(rounded):
        raise _failure("an outward bound exceeds the finite binary64 range")
    return rounded


def _positive(
    first: Float[np.ndarray, "*shape"],
    second: Float[np.ndarray, "*shape"],
    *,
    multiply: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Add or multiply equal-shaped nonnegative envelopes with outward rounding.

    Parameters
    ----------
    first, second : numpy.ndarray
        Equal-shaped finite nonnegative binary64 component bounds.
    multiply : bool, optional
        Multiply instead of adding the bounds.

    Returns
    -------
    numpy.ndarray
        Fresh outward component bounds with structural zeros preserved.
    """
    # Overflow is converted to a named refusal, not left as a warning or an Inf
    # envelope. A positive underflowed product rounds UP to the least subnormal.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result = first * second if multiply else first + second
        nonzero = (
            (first != 0) & (second != 0) if multiply else (first != 0) | (second != 0)
        )
        result = np.where(nonzero, np.nextafter(result, np.inf), 0.0)
    if not np.all(np.isfinite(result)):
        raise _failure("an outward accumulation exceeds the finite binary64 range")
    return result


def _maximum(real: np.ndarray, imag: np.ndarray) -> ComplexEnvelope:
    """Reduce nonnegative component arrays to their independent maxima.

    Parameters
    ----------
    real, imag : numpy.ndarray
        Nonempty nonnegative real and imaginary component bounds.

    Returns
    -------
    ComplexEnvelope
        Independent maxima over all supplied cells.
    """
    return ComplexEnvelope(float(np.max(real)), float(np.max(imag)))


def _scale(bound: ComplexEnvelope, factor: float) -> ComplexEnvelope:
    """Multiply both component bounds by an exact nonnegative binary64 factor.

    Parameters
    ----------
    bound : ComplexEnvelope
        Finite nonnegative component bounds.
    factor : float
        Finite nonnegative exact binary64 multiplier.

    Returns
    -------
    ComplexEnvelope
        Outward component products.
    """
    return ComplexEnvelope(
        _directed(_rational(bound.real) * _rational(factor)),
        _directed(_rational(bound.imag) * _rational(factor)),
    )


def _add(first: ComplexEnvelope, second: ComplexEnvelope) -> ComplexEnvelope:
    """Add component bounds in exact rational arithmetic and round upward.

    Parameters
    ----------
    first, second : ComplexEnvelope
        Finite nonnegative component bounds in identical units.

    Returns
    -------
    ComplexEnvelope
        Outward component sums.
    """
    return ComplexEnvelope(
        _directed(_rational(first.real) + _rational(second.real)),
        _directed(_rational(first.imag) + _rational(second.imag)),
    )


def _minimum(first: ComplexEnvelope, second: ComplexEnvelope) -> ComplexEnvelope:
    """Take the tighter of two independently valid component bounds.

    Parameters
    ----------
    first, second : ComplexEnvelope
        Two valid bounds on the same operands.

    Returns
    -------
    ComplexEnvelope
        Componentwise tighter bounds.
    """
    return ComplexEnvelope(min(first.real, second.real), min(first.imag, second.imag))


def _numeric(values: np.ndarray, rank: int, name: str) -> None:
    """Validate nonempty finite binary32/binary64 real or complex storage.

    Parameters
    ----------
    values : numpy.ndarray
        Storage whose shape and values will be read.
    rank : int
        Required number of axes.
    name : str
        Operand name for the refusal.
    """
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != rank
        or values.dtype not in (np.float32, np.float64, np.complex64, np.complex128)
        or not values.size
        or not np.all(np.isfinite(values))
    ):
        raise _failure(f"{name} must be nonempty finite real/complex floating storage")


def _history_geometry(
    history: MarchSnapshot | None, run: RunTransfer, start: float
) -> None:
    """Validate compatible full-store history without changing sample coordinates.

    Parameters
    ----------
    history : MarchSnapshot or None
        Already-compatible oldest-first history, or no restored history.
    run : RunTransfer
        Resolved target march policy.
    start : float
        Actual accepted anchor in metres.
    """
    if history is None:
        return
    policy = run.adams
    if (
        policy is None
        or run.history_key is None
        or history.history_key != run.history_key
    ):
        raise _failure("only already-compatible Adams history may be analyzed")
    positions, slots = history.positions, history.slots
    if not isinstance(positions, np.ndarray) or not isinstance(slots, np.ndarray):
        raise _failure("history coordinates and slots must be numerical vectors")
    count, capacity = positions.size, policy.order + 2
    if (
        history.z != start
        or not math.isfinite(history.proposal)
        or history.proposal <= 0
        or type(history.order) is not int
        or not 1 <= history.order <= policy.order
        or type(history.rejections) is not int
        or history.rejections < 0
        or positions.ndim != 1
        or positions.dtype != np.float64
        or not 1 <= count <= policy.order + 1
        or not np.all(np.isfinite(positions))
        or not np.all(positions[1:] > positions[:-1])
        or slots.shape != positions.shape
        or slots.dtype != np.int64
        or np.unique(slots).size != count
        or np.any((slots < 0) | (slots >= capacity))
        or len(history.identities) != count
        or any(kind not in ("accepted", "predicted") for kind in history.identities)
        or not math.isclose(float(positions[-1]), start, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise _failure("malformed retained history coordinates, slots or metadata")
    _numeric(history.state, 3, "history state")
    if not np.iscomplexobj(history.state):
        raise _failure("history state must use complex storage")
    rows = history.rows
    if (
        set(rows) - {"store", "states", "rates"}
        or "store" not in rows
        or (("states" in rows) != ("rates" in rows))
    ):
        raise _failure("history needs a source store and paired sampled states/rates")
    for name, values in rows.items():
        _numeric(values, 5, f"history {name}")
        if values.shape != (capacity, 1, *history.state.shape) or not np.iscomplexobj(
            values
        ):
            raise _failure(f"history {name} has the wrong full-store layout")


def _positions(values: np.ndarray, name: str, minimum: int) -> None:
    """Validate a strictly increasing finite binary64 position vector.

    Parameters
    ----------
    values : numpy.ndarray
        Position vector in metres.
    name : str
        Position family for the refusal.
    minimum : int
        Smallest admitted number of entries.
    """
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype != np.float64
        or len(values) < minimum
        or not np.all(np.isfinite(values))
        or not np.all(values[1:] > values[:-1])
    ):
        raise _failure(f"{name} must be increasing finite binary64 positions")


def _coordinate(origin: float, node: float, step: float) -> tuple[float, float]:
    """Enclose binary64 multiply-then-add stage coordinates in metres.

    Parameters
    ----------
    origin : float
        Accepted binary64 coordinate in metres.
    node : float
        Dimensionless actual tableau node.
    step : float
        Actual binary64 step in metres.

    Returns
    -------
    tuple of float
        Lower and upper bounds in metres after two rounded operations.
    """
    product = _rational(node) * _rational(step)
    lower = _directed(product, upper=False)
    upper = _directed(product)
    return (
        _directed(_rational(origin) + _rational(lower), upper=False),
        _directed(_rational(origin) + _rational(upper)),
    )


def _replay_geometry(
    run: RunTransfer, start: float, end: float
) -> tuple[float, float | None, float, float, float]:
    """Bound the remaining actual prescribed lengths and their stage coordinates.

    Parameters
    ----------
    run : RunTransfer
        Resolved replay with actual lengths and authoritative edges.
    start, end : float
        Actual starting anchor and requested end in metres.

    Returns
    -------
    tuple
        Maximum/minimum length, lower/upper coordinate and slack in metres.
    """
    steps, edges = run.steps, run.edges
    _positions(edges, "replay edges", 2)
    if (
        steps.ndim != 1
        or len(edges) != len(steps) + 1
        or not np.all(np.isfinite(steps) & (steps > 0))
    ):
        raise _failure(
            "replay steps must be finite positive lengths matching the edges"
        )
    factor = max(LANDING_ULPS, len(steps))
    running = 0.0
    for step, edge in zip(steps, edges[1:], strict=True):
        running += float(step)
        expected = float(edges[0]) + running
        if not math.isfinite(expected) or abs(
            _rational(float(edge)) - _rational(expected)
        ) > factor * _rational(math.ulp(max(running, abs(expected)))):
            raise _failure("replay edges disagree with accumulated prescribed lengths")
    total = _rational(float(edges[-1])) - _rational(float(edges[0]))
    magnitude = max(abs(start), abs(end), abs(float(edges[0])), abs(float(edges[-1])))
    slack = _directed(factor * _rational(math.ulp(max(_directed(total), magnitude))))
    first = min(
        range(len(edges)),
        key=lambda index: abs(_rational(float(edges[index])) - _rational(start)),
    )
    if abs(float(edges[first]) - start) > slack or float(edges[-1]) < end - slack:
        raise _failure(
            "replay does not cover the actual starting anchor and requested end"
        )
    last = first
    while last < len(steps) and float(edges[last]) < end - slack:
        last += 1
    if float(edges[last]) > end + slack:
        raise _failure("replay crosses the requested end")
    selected = steps[first:last]
    low, high = start, end
    for index in range(first, last):
        origin = start if index == first else float(edges[index])
        low, high = min(low, origin), max(high, float(edges[index + 1]))
        for node in (*run.nodes, 1.0):
            lower, upper = _coordinate(origin, float(node), float(steps[index]))
            low, high = min(low, lower), max(high, upper)
    return (
        float(np.max(selected)) if selected.size else 0.0,
        float(np.min(selected)) if selected.size else None,
        low,
        high,
        slack,
    )


def _adaptive_geometry(
    run: RunTransfer, start: float, end: float
) -> tuple[float, None, float, float, float]:
    """Bound capped adaptive coordinates while leaving the minimum step unknown.

    Parameters
    ----------
    run : RunTransfer
        Resolved adaptive maximum and actual tableau nodes.
    start, end : float
        Actual starting anchor and requested end in metres.

    Returns
    -------
    tuple
        Maximum length, absent minimum, coordinate bounds and slack in metres.
    """
    maximum = run.control.dz_max
    if not math.isfinite(maximum) or maximum <= 0:
        raise _failure("adaptive dz_max must be finite and positive")
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)
    distance = _rational(end) - _rational(start)
    gap_error = unit * distance + tiny
    raw_end = _rational(end) + gap_error
    raw_end += unit * abs(raw_end) + tiny
    low, high = start, _directed(raw_end)
    for node in run.nodes:
        point = float(node)
        if point < 0 or point > 1:
            lower = _rational(start) + min(Fraction(0), _rational(point)) * _rational(
                maximum
            )
            upper = (
                _rational(end)
                + gap_error
                + max(Fraction(0), _rational(point) - 1) * _rational(maximum)
            )
            error = unit * abs(_rational(point)) * _rational(maximum) + tiny
            error += unit * (max(abs(lower), abs(upper)) + error) + tiny
            low = min(low, _directed(lower - error, upper=False))
            high = max(high, _directed(upper + error))
    ulps = LANDING_ULPS if run.adams is None else run.adams.landing_ulps
    slack = _directed(ulps * _rational(math.ulp(max(abs(low), abs(high)))))
    return maximum, None, low, high, slack


def phase_spans(
    run: RunTransfer, start: float, *, history: MarchSnapshot | None = None
) -> PhaseSpans:
    """Bound march geometry without treating the adaptive floor as a lower step.

    Parameters
    ----------
    run : RunTransfer
        Already resolved and compatible request; all distances are metres.
    start : float
        Actual fresh or resumed accepted anchor, metres.
    history : MarchSnapshot or None, optional
        Already-compatible history that will actually be restored. Its key, full
        stores, slots, sample kinds and unchanged coordinates are checked again;
        incompatible/malformed history refuses rather than silently restarting.

    Returns
    -------
    PhaseSpans
        Outward geometry, including actual replay lengths and endpoint rounding.

    Raises
    ------
    BackendError
        For malformed input, incompatible history or a nonfinite outward bound.
    """
    _positions(run.planes, "save planes", 1)
    _positions(run.breakpoints, "operator breakpoints", 2)
    end = float(run.planes[-1])
    if not math.isfinite(start) or not float(
        run.breakpoints[0]
    ) <= start <= end <= float(run.breakpoints[-1]):
        raise _failure(
            "the actual start and requested end must lie inside the operator"
        )
    if run.nodes.ndim != 1 or not run.nodes.size or not np.all(np.isfinite(run.nodes)):
        raise _failure("tableau nodes must be a nonempty finite vector")
    if run.adams is not None and (
        run.adams.dz_max != run.control.dz_max
        or run.adams.z_end != end
        or not 1 <= run.adams.order <= MAX_ADAMS_ORDER
        or run.adams.landing_ulps != LANDING_ULPS
    ):
        raise _failure("Adams geometry must match the shared translated march policy")
    _history_geometry(history, run, start)
    maximum, minimum, low, high, slack = (
        _replay_geometry(run, start, end)
        if run.steps.size
        else _adaptive_geometry(run, start, end)
    )
    if not run.steps.size and run.edges.size:
        raise _failure("adaptive requests cannot retain replay edges")
    oldest = start if history is None else min(start, float(history.positions[0]))
    low = min(low, oldest)
    if history is not None:
        high = max(high, float(history.positions[-1]))
    stage_scale = max(1.0, float(np.max(np.abs(run.nodes))))
    age = _rational(start) - _rational(oldest)
    order_span = run.adams.order + 1 if run.adams is not None else 1
    span = age + order_span * (
        _rational(stage_scale) * _rational(maximum) + 2 * _rational(slack)
    )
    span = min(span, _rational(high) - _rational(low))
    transport = _directed(span)
    multiplier = None if minimum is None else _directed(1 + span / _rational(minimum))
    return PhaseSpans(
        start,
        end,
        oldest,
        maximum,
        minimum,
        stage_scale,
        transport,
        low,
        high,
        slack,
        multiplier,
    )


@dataclass(frozen=True, eq=False)
class _TableEnvelope:
    """Retain per-cell positive bounds for a complete represented table."""

    values: Float[np.ndarray, "n_values 2"]
    rates: Float[np.ndarray, "n_values 2"]
    jumps: Float[np.ndarray, "n_values 2"]
    coefficients: ComplexEnvelope
    degree: int


def _growth(
    coordinate: Fraction, degree: int, width: Fraction
) -> tuple[list[float], list[float]]:
    """Bound Chebyshev values and physical derivatives on an enlarged interval.

    Parameters
    ----------
    coordinate : fractions.Fraction
        Exact maximum absolute mapped coordinate, at least one.
    degree : int
        Maximum represented Chebyshev degree.
    width : fractions.Fraction
        Exact positive interval width in metres.

    Returns
    -------
    tuple of list of float
        Outward value and inverse-metre derivative factors for each degree.
    """
    # For X>=1, |T_k(x)|<=T_k(X) and |T'_k(x)|<=k U_(k-1)(X)
    # whenever |x|<=X. Exact rational recurrence avoids unstable interval wrapping
    # at X=1 and certifies slight endpoint extrapolation without sampled extrema.
    polynomials = [Fraction(1)]
    derivatives = [Fraction(0)]
    previous, current = Fraction(1), coordinate
    before_u, current_u = Fraction(0), Fraction(1)
    for order in range(1, degree + 1):
        polynomials.append(current)
        derivatives.append(2 * order * current_u / width)
        previous, current = current, 2 * coordinate * current - previous
        before_u, current_u = current_u, 2 * coordinate * current_u - before_u
    return (
        [_directed(value) for value in polynomials],
        [_directed(value) for value in derivatives],
    )


def _jump(coefficients: np.ndarray, interval: int) -> Float[np.ndarray, "n_values 2"]:
    """Bound one represented endpoint jump using exact coefficient cancellation.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Finite coefficients shaped (n_values, n_deg, n_intervals).
    interval : int
        Zero-based interval to the left of the boundary.

    Returns
    -------
    numpy.ndarray
        Absolute jump bounds shaped (n_values, 2), in cumulative units.
    """
    result = np.zeros((coefficients.shape[0], 2))
    for value in range(coefficients.shape[0]):
        for component, words in enumerate(
            (np.real(coefficients), np.imag(coefficients))
        ):
            difference = Fraction(0)
            for degree in range(coefficients.shape[1]):
                difference += _rational(float(words[value, degree, interval]))
                difference -= (-1) ** degree * _rational(
                    float(words[value, degree, interval + 1])
                )
            result[value, component] = _directed(abs(difference))
    return result


def _table_envelope(table: TableTransfer, spans: PhaseSpans) -> _TableEnvelope:
    """Bound values, derivatives and jumps over every required table interval.

    Parameters
    ----------
    table : TableTransfer
        Exact represented coefficients and admitted span in metres.
    spans : PhaseSpans
        Complete required coordinate span in metres.

    Returns
    -------
    _TableEnvelope
        Per-cell envelopes plus coefficient magnitudes and degree.
    """
    edges, coefficients = table.edges, table.coefficients
    _positions(edges, "table edges", 2)
    _numeric(coefficients, 3, "table coefficients")
    if coefficients.shape[2] != len(edges) - 1 or not (
        math.isfinite(table.lower)
        and math.isfinite(table.upper)
        and table.lower <= float(edges[0])
        and float(edges[-1]) <= table.upper
        and table.lower <= spans.position_lower <= spans.position_upper <= table.upper
    ):
        raise _failure("required phase coordinates leave the represented table span")
    degree = coefficients.shape[1] - 1
    values = np.zeros((coefficients.shape[0], 2))
    rates = np.zeros_like(values)
    jumps = np.zeros_like(values)
    real = np.abs(np.asarray(np.real(coefficients), dtype=float))
    imag = np.abs(np.asarray(np.imag(coefficients), dtype=float))
    for interval in range(len(edges) - 1):
        left = table.lower if interval == 0 else float(edges[interval])
        right = (
            table.upper if interval == len(edges) - 2 else float(edges[interval + 1])
        )
        left, right = max(left, spans.position_lower), min(right, spans.position_upper)
        if right < left:
            continue
        a, b = _rational(float(edges[interval])), _rational(float(edges[interval + 1]))
        width = b - a
        coordinate = max(
            Fraction(1),
            abs((2 * _rational(left) - a - b) / width),
            abs((2 * _rational(right) - a - b) / width),
        )
        polynomial, derivative = _growth(coordinate, degree, width)
        interval_values, interval_rates = np.zeros_like(values), np.zeros_like(rates)
        for index in range(degree + 1):
            coefficient = np.column_stack(
                (real[:, index, interval], imag[:, index, interval])
            )
            for output, weight in (
                (interval_values, polynomial[index]),
                (interval_rates, derivative[index]),
            ):
                product = _positive(
                    coefficient, np.full_like(coefficient, weight), multiply=True
                )
                output[:] = _positive(output, product)
        values = np.maximum(values, interval_values)
        rates = np.maximum(rates, interval_rates)
        if (
            interval < len(edges) - 2
            and spans.position_lower <= edges[interval + 1] <= spans.position_upper
        ):
            jumps = _positive(jumps, _jump(coefficients, interval))
    return _TableEnvelope(values, rates, jumps, _maximum(real, imag), degree)


def _weighted(terms: np.ndarray, weights: np.ndarray) -> Float[np.ndarray, " n_values"]:
    """Sum absolute fixed-term contributions for nonnegative scalar weight bounds.

    Parameters
    ----------
    terms : numpy.ndarray
        Real component of fixed terms, with a final term axis.
    weights : numpy.ndarray
        One nonnegative bound for each scalar weight.

    Returns
    -------
    numpy.ndarray
        Flattened outward absolute-contribution sums.
    """
    flattened = np.abs(np.asarray(terms, dtype=float)).reshape(
        -1, len(weights), order="F"
    )
    total = np.zeros(flattened.shape[0])
    for index, weight in enumerate(weights):
        product = _positive(
            flattened[:, index], np.full_like(total, weight), multiply=True
        )
        total = _positive(total, product)
    return total


def _table_difference(table: _TableEnvelope, distance: float) -> np.ndarray:
    """Bound differences by endpoint sums and integrated derivatives plus jumps.

    Parameters
    ----------
    table : _TableEnvelope
        Per-cell value, derivative and jump envelopes.
    distance : float
        Maximum endpoint separation in metres.

    Returns
    -------
    numpy.ndarray
        Component bounds shaped (n_values, 2), in cumulative units.
    """
    endpoints = _positive(table.values, table.values)
    derivative = _positive(
        table.rates, np.full_like(table.rates, distance), multiply=True
    )
    return np.minimum(endpoints, _positive(derivative, table.jumps))


def _matrix_envelope(values: np.ndarray) -> ComplexEnvelope:
    """Bound absolute components of already-validated finite stored values.

    Parameters
    ----------
    values : numpy.ndarray
        Nonempty finite real/complex storage of arbitrary shape.

    Returns
    -------
    ComplexEnvelope
        Absolute component maxima in stored units.
    """
    return _maximum(np.abs(np.real(values)), np.abs(np.imag(values)))


def _representation(
    operator: CoefficientTransfer,
) -> Literal["constant", "represented", "expansion"]:
    """Require exactly one of the three translated linear representations.

    Parameters
    ----------
    operator : CoefficientTransfer
        Translated cumulative or constant representation.

    Returns
    -------
    str
        One of constant, represented or expansion.
    """
    if operator.values is not None and operator.table is None:
        return "constant" if operator.weights is None else "expansion"
    if (
        operator.values is None
        and operator.table is not None
        and operator.weights is None
    ):
        return "represented"
    raise _failure(
        "the operator needs exactly one constant, table or expansion representation"
    )


def operator_envelope(
    operator: CoefficientTransfer, spans: PhaseSpans
) -> OperatorEnvelope:
    """Bound complete represented operators and their uncancelled evaluation operands.

    Parameters
    ----------
    operator : CoefficientTransfer
        A constant generator, cumulative Chebyshev table, or fixed terms with
        real cumulative weights. No physics object or callback is evaluated.
    spans : PhaseSpans
        Required coordinates and transport distance from :func:`phase_spans`.

    Returns
    -------
    OperatorEnvelope
        Exact-representation bounds, with outward accumulation. Full represented
        jumps are retained even when a physical cumulative operator is continuous.

    Raises
    ------
    BackendError
        For malformed representations, out-of-span coordinates or overflow.
    """
    numeric_spans = (
        spans.start,
        spans.end,
        spans.oldest,
        spans.step_max,
        spans.stage_scale,
        spans.transport_span,
        spans.position_lower,
        spans.position_upper,
        spans.landing_slack,
    )
    if not all(math.isfinite(value) for value in numeric_spans) or not (
        spans.position_lower
        <= spans.oldest
        <= spans.start
        <= spans.end
        <= spans.position_upper
        and spans.step_max >= 0
        and spans.stage_scale >= 1
        and 0
        <= spans.transport_span
        <= _directed(_rational(spans.position_upper) - _rational(spans.position_lower))
        and spans.landing_slack >= 0
        and (
            spans.step_min is None
            or (math.isfinite(spans.step_min) and 0 < spans.step_min <= spans.step_max)
        )
        and (
            spans.compensation_factor_max is None
            or (
                math.isfinite(spans.compensation_factor_max)
                and spans.compensation_factor_max >= 1
            )
        )
    ):
        raise _failure(
            "phase spans must be finite ordered geometry with nonnegative distances"
        )
    kind = _representation(operator)
    zero = ComplexEnvelope(0.0, 0.0)
    if kind == "constant":
        assert operator.values is not None
        _numeric(operator.values, 2, "constant generator")
        rate = _matrix_envelope(operator.values)
        difference = _scale(rate, spans.transport_span)
        return OperatorEnvelope(
            kind, None, rate, zero, difference, difference, rate, 1, 0
        )
    if kind == "represented":
        assert operator.table is not None
        table = _table_envelope(operator.table, spans)
        return OperatorEnvelope(
            kind,
            _maximum(table.values[:, 0], table.values[:, 1]),
            _maximum(table.rates[:, 0], table.rates[:, 1]),
            _maximum(table.jumps[:, 0], table.jumps[:, 1]),
            _maximum(*_table_difference(table, spans.transport_span).T),
            _maximum(*_positive(table.values, table.values).T),
            table.coefficients,
            1,
            table.degree,
        )
    assert operator.values is not None and operator.weights is not None
    terms = operator.values
    _numeric(terms, 3, "fixed expansion terms")
    if np.iscomplexobj(operator.weights.coefficients):
        raise _failure("cumulative expansion weights must be real")
    table = _table_envelope(operator.weights, spans)
    if table.values.shape[0] != terms.shape[2]:
        raise _failure("fixed terms and cumulative weight counts disagree")
    difference = _table_difference(table, spans.transport_span)[:, 0]
    operands = _positive(table.values[:, 0], table.values[:, 0])

    def combine(weights: np.ndarray) -> ComplexEnvelope:
        """Contract nonnegative scalar envelopes against absolute complex terms.

        Parameters
        ----------
        weights : numpy.ndarray
            One nonnegative bound for each scalar cumulative weight.

        Returns
        -------
        ComplexEnvelope
            Outward absolute-contribution maxima over every spectral cell.
        """
        return _maximum(
            _weighted(np.real(terms), weights), _weighted(np.imag(terms), weights)
        )

    return OperatorEnvelope(
        kind,
        combine(table.values[:, 0]),
        combine(table.rates[:, 0]),
        combine(table.jumps[:, 0]),
        combine(difference),
        combine(operands),
        _matrix_envelope(terms),
        terms.shape[2],
        table.degree,
    )


def phase_operands(
    operator: CoefficientTransfer,
    run: RunTransfer,
    start: float,
    *,
    history: MarchSnapshot | None = None,
) -> PhaseOperands:
    """Analyze phase operands while leaving missing assembly proofs explicit.

    Parameters
    ----------
    operator : CoefficientTransfer
        Exact stored linear representation, including all spectral cells.
    run : RunTransfer
        Already resolved compatible march policy; no runtime is initialized.
    start : float
        Actual accepted starting coordinate, metres.
    history : MarchSnapshot or None, optional
        Compatible portable history that will be restored, without changing its
        sample coordinates, slots or stored rates. None explicitly selects a
        fresh/field-only analysis; no failed history is silently dropped.

    Returns
    -------
    PhaseOperands
        Operand bounds, not a DF32 certificate or an admission decision. Missing
        bounds are None with reasons in ``unresolved``. The function-domain
        limits and independent input-error budget remain downstream obligations.

    Raises
    ------
    BackendError
        For malformed input/history, unsupported layouts, out-of-span positions
        or a bound that cannot be represented by a finite binary64 value.
    """
    spans = phase_spans(run, start, history=history)
    envelope = operator_envelope(operator, spans)
    step_distance = min(
        spans.transport_span,
        _directed(
            _rational(spans.stage_scale) * _rational(spans.step_max)
            + 2 * _rational(spans.landing_slack)
        ),
    )
    step = _minimum(
        envelope.difference,
        _add(_scale(envelope.rate, step_distance), envelope.jumps),
    )
    compensated = (
        run.adams is not None and envelope.kind != "constant" and not run.source_absent
    )
    restored = None
    if history is not None:
        state = history.state
        cells = state.shape[2], state.shape[0] * state.shape[1]
        if (
            state.shape[2] != run.active.size
            or (operator.values is not None and operator.values.shape[:2] != cells)
            or (
                operator.table is not None
                and operator.table.coefficients.shape[0] != state.size
            )
        ):
            raise _failure(
                "restored state and represented operator cell counts disagree"
            )
        has_rates = "rates" in history.rows
        if has_rates != compensated:
            raise _failure(
                "restored rates disagree with this operator's compensation policy"
            )
        if has_rates:
            restored = _matrix_envelope(history.rows["rates"][history.slots])
    missing = [
        "binding prepared operator/history arrays to an identified arithmetic graph",
        "DF32 coefficient/term conversion and intermediate representability",
        "identified coordinate, Clenshaw, subtraction and expansion arithmetic graph",
        "independently propagated argument error and reserved function-domain margin",
        "selected device RN-even/FMA/division and gradual-underflow model",
    ]
    transported: ComplexEnvelope | None = step
    if compensated:
        multiplier = spans.compensation_factor_max
        if multiplier is None:
            transported = None
            missing.append("adaptive compensation has no positive actual lower h bound")
        else:
            transported = _scale(step, multiplier)
        missing.append(
            "compensation multiplier and cumulative-subtraction/endpoint rounding error"
        )
    if restored is not None:
        missing.append(
            "conversion and error provenance of the selected restored rate/state rows"
        )
    return PhaseOperands(spans, envelope, step, transported, restored, tuple(missing))

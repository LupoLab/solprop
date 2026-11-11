# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check represented operand envelopes independently of any Julia arithmetic graph."""

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from solprop.backends.julia import _phase_bounds
from solprop.backends.julia._descriptors import CoefficientTransfer, TableTransfer
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._phase_bounds import (
    ComplexEnvelope,
    PhaseSpans,
    operator_envelope,
    phase_operands,
    phase_spans,
)
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep, march_policy
from solprop.translate.run_request import ErrorControlSpec


def _run(*, steps=None, order=None, maximum=0.1, nodes=(0.0, 1.0)):
    method = None if order is None else Multistep(order=order)
    policy = (
        None
        if method is None
        else march_policy(
            method,
            dz_min=0.01,
            dz_max=maximum,
            stretch=1.2,
            z_end=1.0,
            save_planes=[1.0],
            hard_kinks=[],
            breakpoints=[0.0, 1.0],
            interpolate=False,
        )
    )
    prescribed = np.array([] if steps is None else steps, dtype=float)
    edges = np.r_[0.0, np.cumsum(prescribed)] if steps is not None else np.empty(0)
    return RunTransfer(
        ErrorControlSpec(
            PIController(1e-6, 1e-9), 0.01, maximum, 0.01, 1.2, False, True
        ),
        np.array(nodes),
        np.zeros((len(nodes), len(nodes))),
        np.ones(len(nodes)),
        np.zeros(len(nodes)),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1),
        np.array([1.0]),
        np.array([0.0, 1.0]),
        np.empty(0),
        prescribed,
        edges,
        0,
        2.0,
        adams=policy,
        history_key=None if method is None else method.history_key,
    )


def _history(run, positions=(0.2, 0.5), *, rates=False):
    assert run.adams is not None
    capacity = run.adams.order + 2
    shape = (1, 1, 1)
    store = np.zeros((capacity, 1, *shape), dtype=complex)
    rows = {"store": store}
    if rates:
        rows.update(states=store.copy(), rates=store.copy())
        rows["rates"][0] = 100 + 200j
        rows["rates"][1] = -300 + 400j
        rows["rates"][-1] = 1e10 + 1e11j
    return MarchSnapshot(
        z=0.5,
        state=np.ones(shape, dtype=complex),
        proposal=0.1,
        method="M()",
        order=run.adams.order,
        rejections=0,
        last_rejected=False,
        positions=np.array(positions),
        identities=("accepted", "predicted"),
        slots=np.array([0, 1], dtype=np.int64),
        rows=rows,
        history_key=run.history_key,
    )


def _spans(lower=0.0, upper=1.0, distance=1.0):
    return PhaseSpans(
        lower,
        upper,
        lower,
        distance,
        None,
        1.0,
        distance,
        lower,
        upper,
        0.0,
        None,
    )


def _table(coefficients, edges=(0.0, 1.0), *, lower=None, upper=None):
    array = np.asarray(coefficients)
    if array.ndim == 1:
        array = array[None, :, None]
    return TableTransfer(
        np.array(edges),
        edges[0] if lower is None else lower,
        edges[-1] if upper is None else upper,
        array,
    )


def _represented(table):
    return CoefficientTransfer(None, table, None)


def _q(value):
    return Fraction.from_float(float(value))


def _changed(record, **changes):
    # Construct a valid dataclass first; then test its production validation past
    # the optional test-session annotation wrapper, without weakening that wrapper.
    result = replace(record)
    for name, value in changes.items():
        object.__setattr__(result, name, value)
    return result


def _cheb(coefficients, coordinate):
    """Use an independent exact defining polynomial recurrence, not Clenshaw."""
    polynomials = [Fraction(1), coordinate]
    for _unused in range(2, len(coefficients)):
        polynomials.append(2 * coordinate * polynomials[-1] - polynomials[-2])
    return sum(
        _q(value) * polynomial
        for value, polynomial in zip(
            coefficients, polynomials[: len(coefficients)], strict=True
        )
    )


def _encloses(bound, exact):
    assert math.isfinite(bound) and bound >= 0
    assert _q(bound) >= abs(exact)


def test_actual_replay_ignores_adaptive_ceiling_and_used_prefix():
    run = _run(steps=[0.25, 0.25, 0.5], maximum=0.1)
    spans = phase_spans(run, 0.0)
    assert spans.step_max == 0.5
    assert spans.step_min == 0.25
    assert spans.transport_span >= 0.5
    resumed = phase_spans(run, 0.5)
    assert resumed.step_min == resumed.step_max == 0.5
    assert resumed.position_lower == 0.5
    ended = phase_spans(run, 1.0)
    assert ended.step_max == 0 and ended.step_min is None
    operator = CoefficientTransfer(np.array([[2.0 - 3.0j]]), None, None)
    result = phase_operands(operator, run, 0.0)
    _encloses(result.step_argument.real, Fraction(1))
    _encloses(result.step_argument.imag, Fraction(3, 2))
    assert result.operator.cumulative is None
    assert result.operator.jumps == ComplexEnvelope(0.0, 0.0)
    assert result.transported_argument == result.step_argument
    assert result.unresolved and result.restored_rates is None


def test_adaptive_floor_is_not_a_lower_step_and_stages_are_explicit():
    run = _run(maximum=0.1, nodes=(0.0, -0.5, 1.5))
    spans = phase_spans(run, 0.0)
    assert spans.step_min is None and spans.compensation_factor_max is None
    assert spans.stage_scale == 1.5
    assert spans.position_lower <= -0.05
    assert spans.position_upper >= 1.05
    assert spans.transport_span >= 0.15
    ordinary = phase_spans(_run(), 0.0)
    assert ordinary.position_lower == 0
    assert ordinary.position_upper > 1.0
    assert ordinary.landing_slack > 0


def test_retained_actual_positions_maximum_order_and_sampled_rates():
    run = _run(order=3, maximum=0.01)
    snapshot = _history(run, positions=(0.2, np.nextafter(0.5, 1.0)), rates=True)
    spans = phase_spans(run, 0.5, history=snapshot)
    assert spans.oldest == 0.2
    _encloses(spans.transport_span, _q(0.5) - _q(0.2) + 4 * _q(0.01))
    table = _table([0.5, 0.5], upper=1.0 + 1e-12)
    result = phase_operands(_represented(table), run, 0.5, history=snapshot)
    assert result.restored_rates == ComplexEnvelope(300.0, 400.0)
    assert result.transported_argument is None
    assert any("lower h" in reason for reason in result.unresolved)
    assert any("restored" in reason for reason in result.unresolved)
    assert snapshot.positions[-1] == np.nextafter(0.5, 1.0)
    # History produced by another wide type can have complex64 state and stored
    # complex128 rates: this analyzer reads exact stored operands, not precision
    # admission, and never replaces those rates with the current derivative.
    mixed = replace(snapshot, state=snapshot.state.astype(np.complex64))
    assert (
        phase_operands(_represented(table), run, 0.5, history=mixed).restored_rates
        == result.restored_rates
    )


def test_retained_positive_sample_offset_dominates_tiny_replay_steps():
    run = _run(order=3)
    assert run.adams is not None
    step = 2.0**-49
    end = 0.5 + 2 * step
    run = replace(
        run,
        adams=replace(run.adams, z_end=end, save_planes=(end,)),
        planes=np.array([end]),
        steps=np.array([step, step]),
        edges=np.array([0.5, 0.5 + step, end]),
    )
    sample = 0.5 + 2.0**-43
    history = replace(
        _history(run, rates=True),
        positions=np.array([sample]),
        slots=np.array([0], dtype=np.int64),
        identities=("predicted",),
        order=1,
    )
    assert math.isclose(sample, 0.5, rel_tol=1e-12, abs_tol=0.0)
    spans = phase_spans(run, 0.5, history=history)
    _encloses(spans.transport_span, _q(sample) - _q(0.5))
    assert spans.compensation_factor_max is not None
    _encloses(spans.compensation_factor_max, 1 + (_q(sample) - _q(0.5)) / _q(step))


def test_compensation_retains_replay_ratio_and_cumulative_conditioning():
    small = 2.0**-30
    run = _run(steps=[0.5, small, 0.5 - small], order=3)
    snapshot = _history(run, rates=True)
    table = _table([2.0**40, 0.5], upper=1.0 + 1e-12)
    result = phase_operands(_represented(table), run, 0.5, history=snapshot)
    assert result.spans.step_min == small
    assert result.spans.compensation_factor_max is not None
    assert result.spans.compensation_factor_max >= 0.3 / small
    assert result.transported_argument is not None
    assert result.operator.absolute_contributions.real >= 2.0**41
    assert result.operator.difference.real < 1.0
    assert result.operator.jumps.real == 0.0
    assert any("cumulative-subtraction" in reason for reason in result.unresolved)


def test_replay_position_rounding_and_resumed_anchor_are_included():
    run = _run(steps=[0.1] * 10)
    start = np.nextafter(float(run.edges[2]), 1.0)
    spans = phase_spans(run, float(start))
    origins = [float(start), *run.edges[3:-1]]
    for origin, step in zip(origins, run.steps[2:], strict=True):
        for node in run.nodes:
            point = float(origin + node * step)
            assert spans.position_lower <= point <= spans.position_upper
        assert spans.position_lower <= origin + step <= spans.position_upper
    assert spans.landing_slack >= 10 * math.ulp(1.0)


@pytest.mark.parametrize("node", [1048578.0, -1048578.0])
@pytest.mark.parametrize("shift", [0.0, 734004.3])
def test_replay_large_stage_rounding_is_kept_in_distance_bounds(node, shift):
    # #675: the actual stage addition can have a larger ulp than replay edges.
    # Compare exact rational differences of the binary64 coordinates, including
    # a shifted origin that nearly cancels the negative stage product.
    run = _run(steps=[0.3, 0.7], nodes=(0.0, node))
    run = replace(
        run,
        planes=run.planes + shift,
        breakpoints=run.breakpoints + shift,
        edges=run.edges + shift,
    )
    origin = float(run.edges[1])
    endpoint = origin + node * float(run.steps[1])
    spans = phase_spans(run, origin)
    difference = _q(endpoint) - _q(origin)
    _encloses(spans.transport_span, difference)
    constant = CoefficientTransfer(np.array([[1.0j]]), None, None)
    result = phase_operands(constant, run, origin)
    _encloses(result.step_argument.imag, difference)


def test_full_chebyshev_interval_interior_extrema_and_endpoint_growth():
    # 1-T_2 vanishes at both endpoints but has a positive interior maximum. A sampled
    # endpoint-only admission would miss it; exact rational values probe the span.
    table = _table(np.array([1.0, 0.0, -1.0]))
    envelope = operator_envelope(_represented(table), _spans())
    assert envelope.cumulative is not None
    for numerator in range(-32, 33):
        x = Fraction(numerator, 32)
        _encloses(envelope.cumulative.real, _cheb([1, 0, -1], x))
        _encloses(envelope.rate.real, -8 * x)
    assert envelope.cumulative.real >= 2
    assert envelope.coefficient_magnitudes.real == 1
    assert envelope.max_degree == 2
    edge = np.nextafter(1.0, math.inf)
    coefficients = np.r_[np.zeros(64), 1.0]
    table = _table(coefficients, lower=-1e-15, upper=edge)
    envelope = operator_envelope(_represented(table), _spans(0.0, edge, edge))
    assert envelope.cumulative is not None
    coordinate = 2 * _q(edge) - 1
    exact = _cheb(coefficients, coordinate)
    assert exact > 1
    _encloses(envelope.cumulative.real, exact)
    assert envelope.cumulative.real > 1


def test_piecewise_exact_jumps_and_small_differences_are_separate():
    coefficients = np.zeros((1, 2, 2), dtype=complex)
    coefficients[0, :, 0] = [1e12 + 2e12j, 0.5 + 0.25j]
    coefficients[0, :, 1] = [1e12 + 4 + 2e12j, 0.5 + 0.25j]
    table = _table(coefficients, edges=(0.0, 0.5, 1.0))
    envelope = operator_envelope(_represented(table), _spans(distance=1e-6))
    left = sum(_q(value) for value in np.real(coefficients)[0, :, 0])
    right = sum(
        (-1) ** degree * _q(value)
        for degree, value in enumerate(np.real(coefficients)[0, :, 1])
    )
    _encloses(envelope.jumps.real, left - right)
    assert 3 <= envelope.jumps.real < 3.01
    assert envelope.absolute_contributions.real >= 2e12
    assert envelope.difference.real < 3.01
    # Equal constant pieces have exactly zero represented jump: no artificial
    # epsilon is injected into structural zero by the outward accumulation.
    constant = _table(np.full((1, 1, 2), 1.0), edges=(0.0, 0.5, 1.0))
    bound = operator_envelope(_represented(constant), _spans(distance=0.25))
    assert bound.jumps == ComplexEnvelope(0.0, 0.0)
    assert bound.rate == bound.difference == ComplexEnvelope(0.0, 0.0)


def test_expansion_absolute_contributions_survive_cancelled_result():
    terms = np.array([[[1e12 + 2e12j, -1e12 - 2e12j]]])
    coefficients = np.zeros((2, 2, 1))
    coefficients[:, :, 0] = [100.0, 0.5]
    operator = CoefficientTransfer(terms, None, _table(coefficients))
    envelope = operator_envelope(operator, _spans(distance=1e-3))
    assert np.sum(terms, axis=2)[0, 0] == 0
    assert envelope.absolute_contributions.real >= 4e14
    assert envelope.absolute_contributions.imag >= 8e14
    assert envelope.difference.real >= 2e9
    assert envelope.term_count == 2
    assert envelope.coefficient_magnitudes == ComplexEnvelope(1e12, 2e12)


def test_independent_exact_rational_piecewise_envelopes():
    rng = np.random.default_rng(48)
    coefficients = rng.integers(-32, 33, size=(3, 7, 3)) / 16
    table = _table(coefficients, edges=(0.0, 0.25, 0.75, 1.0))
    envelope = operator_envelope(_represented(table), _spans(distance=0.05))
    assert envelope.cumulative is not None
    for value in range(3):
        for interval, (a, b) in enumerate(
            zip(table.edges[:-1], table.edges[1:], strict=True)
        ):
            for numerator in range(-16, 17):
                x = Fraction(numerator, 16)
                exact = _cheb(coefficients[value, :, interval], x)
                _encloses(envelope.cumulative.real, exact)
                # A difference entirely inside one interval has no jump term.
                second = min(Fraction(1), x + Fraction(1, 64))
                distance = (second - x) * (_q(b) - _q(a)) / 2
                if distance <= _q(0.05):
                    difference = _cheb(coefficients[value, :, interval], second) - exact
                    _encloses(envelope.difference.real, difference)


@pytest.mark.parametrize(
    "change",
    [
        {"history_key": "foreign"},
        {"z": 0.4},
        {"order": 0},
        {"rejections": -1},
        {"positions": np.array([0.5, 0.2])},
        {"positions": np.array([0.2, 0.4])},
        {"slots": np.array([0, 0])},
        {"slots": np.array([-1, 1])},
        {"slots": np.array([0, 8])},
        {"slots": np.array([0, 1], dtype=np.int32)},
        {"identities": ("unknown", "predicted")},
        {"proposal": 0.0},
    ],
)
def test_malformed_or_incompatible_history_never_disappears(change):
    run = _run(order=3)
    history = _changed(_history(run), **change)
    with pytest.raises(BackendError, match="phase operands"):
        phase_spans(run, 0.5, history=history)


@pytest.mark.parametrize(
    "change",
    [
        {"rows": {}},
        {"rows": {"store": np.zeros((1, 1, 1, 1, 1), complex)}},
        {"rows": {"store": np.zeros((5, 1, 1, 1, 1), complex), "foreign": np.zeros(1)}},
        {"rows": {"store": np.zeros((5, 1, 1, 1, 1), complex), "states": np.zeros(1)}},
        {"rows": {"store": np.full((5, 1, 1, 1, 1), complex(np.inf))}},
        {"state": np.ones((1, 1, 1))},
    ],
)
def test_invalid_full_stores_refuse_even_if_bad_row_is_inactive(change):
    run = _run(order=3)
    history = _changed(_history(run), **change)
    with pytest.raises(BackendError, match="phase operands"):
        phase_spans(run, 0.5, history=history)


def test_history_compensation_layout_must_match_the_operator():
    run = _run(order=3)
    constant = CoefficientTransfer(np.array([[1.0j]]), None, None)
    with pytest.raises(BackendError, match="compensation policy"):
        phase_operands(constant, run, 0.5, history=_history(run, rates=True))
    table = _table([0.5, 0.5], upper=1 + 1e-12)
    with pytest.raises(BackendError, match="compensation policy"):
        phase_operands(_represented(table), run, 0.5, history=_history(run))


@pytest.mark.parametrize(
    "operator",
    [
        CoefficientTransfer(None, None, None),
        CoefficientTransfer(np.array([[np.nan]]), None, None),
        CoefficientTransfer(np.empty((0, 1)), None, None),
        _changed(
            CoefficientTransfer(np.ones((1, 1)), None, None),
            values=np.ones((1, 1), dtype=int),
        ),
        _represented(_table([1.0], upper=0.5)),
        _represented(_table(np.ones((1, 1, 2)))),
        CoefficientTransfer(np.ones((1, 1, 2)), None, _table([1.0])),
        CoefficientTransfer(np.ones((1, 1, 1)), None, _table([1.0j])),
    ],
)
def test_malformed_operator_refusal(operator):
    with pytest.raises(BackendError, match="phase operands"):
        operator_envelope(operator, _spans())


def test_overflow_refusal_and_subnormal_outward_rounding():
    largest = np.finfo(float).max
    operator = _represented(_table([largest, largest]))
    with pytest.raises(BackendError, match="finite binary64 range"):
        operator_envelope(operator, _spans())
    tiny = np.nextafter(0.0, 1.0)
    constant = CoefficientTransfer(np.array([[tiny]]), None, None)
    result = operator_envelope(constant, _spans(distance=0.5))
    assert result.difference.real == tiny
    _encloses(result.difference.real, _q(tiny) / 2)
    explosive = _represented(_table(np.r_[np.zeros(1024), 1.0], upper=2.0))
    with pytest.raises(BackendError, match="finite binary64 range"):
        operator_envelope(explosive, _spans(0.0, 2.0, 2.0))


def test_exact_directed_rounding_boundaries():
    # Exact dyadic midpoints and underflow witnesses constrain the outward helper
    # independently of the polynomial bounds which consume it.
    tiny = Fraction(1, 2**1075)
    for exact in (tiny, -tiny, Fraction(1) + Fraction(1, 2**53)):
        lower = _phase_bounds._directed(exact, upper=False)
        upper = _phase_bounds._directed(exact)
        assert _q(lower) <= exact <= _q(upper)
        assert upper == math.nextafter(lower, math.inf)
    for exact in (Fraction(2**1024), _q(np.finfo(float).max) + 1):
        with pytest.raises(BackendError, match="finite binary64 range"):
            _phase_bounds._directed(exact)


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_component_bound_refuses(value):
    with pytest.raises(BackendError, match="finite and nonnegative"):
        ComplexEnvelope(value, 0.0)


@pytest.mark.parametrize(
    "change",
    [
        {"start": math.nan},
        {"transport_span": -1.0},
        {"step_min": 0.0},
        {"step_min": math.inf},
        {"compensation_factor_max": 0.5},
        {"compensation_factor_max": math.inf},
    ],
)
def test_invalid_external_span_refuses(change):
    operator = CoefficientTransfer(np.array([[1.0]]), None, None)
    with pytest.raises(BackendError, match="finite ordered geometry"):
        operator_envelope(operator, replace(_spans(), **change))


@pytest.mark.parametrize(
    "change",
    [
        {"planes": np.empty(0)},
        {"breakpoints": np.array([0.0, 0.0])},
        {"nodes": np.array([math.inf])},
        {"edges": np.array([0.0, 1.0])},
    ],
)
def test_malformed_adaptive_geometry_refuses(change):
    with pytest.raises(BackendError, match="phase operands"):
        phase_spans(_changed(_run(), **change), 0.0)


def test_invalid_adaptive_anchor_ceiling_and_duplicate_policy_refuse():
    run = _run()
    with pytest.raises(BackendError, match="inside the operator"):
        phase_spans(run, -1.0)
    invalid = replace(run, control=replace(run.control, dz_max=math.inf))
    with pytest.raises(BackendError, match="adaptive dz_max"):
        phase_spans(invalid, 0.0)
    run = _run(order=3)
    invalid = replace(run, control=replace(run.control, dz_max=0.2))
    with pytest.raises(BackendError, match="shared translated march"):
        phase_spans(invalid, 0.0)


@pytest.mark.parametrize(
    "steps,edges,start,end",
    [
        ([0.5, 0.0], [0.0, 0.5, 1.0], 0.0, 1.0),
        ([0.5], [0.0, 0.5, 1.0], 0.0, 1.0),
        ([0.5, 0.5], [0.0, 0.4, 1.0], 0.0, 1.0),
        ([0.5, 0.5], [0.0, 0.5, 1.0], 0.2, 1.0),
        ([0.5], [0.0, 0.5], 0.0, 1.0),
        ([0.5, 0.5], [0.0, 0.5, 1.0], 0.0, 0.75),
    ],
)
def test_malformed_replay_geometry_refuses(steps, edges, start, end):
    run = replace(
        _run(), steps=np.array(steps), edges=np.array(edges), planes=np.array([end])
    )
    with pytest.raises(BackendError, match="phase operands"):
        phase_spans(run, start)


def test_unused_intervals_do_not_hide_required_subinterval_extrema():
    coefficients = np.zeros((1, 3, 3))
    coefficients[0, :, 1] = [1.0, 0.0, -1.0]
    coefficients[0, 0, [0, 2]] = 1e10
    table = _table(coefficients, edges=(0.0, 0.25, 0.75, 1.0))
    envelope = operator_envelope(_represented(table), _spans(0.3, 0.7, 0.25))
    assert envelope.cumulative is not None
    assert 2 <= envelope.cumulative.real < 2.01
    assert envelope.jumps == ComplexEnvelope(0.0, 0.0)


def test_restored_history_shape_and_non_adams_refusals():
    run = _run(order=3)
    history = _history(run)
    with pytest.raises(BackendError, match="already-compatible"):
        phase_spans(_run(), 0.5, history=history)
    with pytest.raises(BackendError, match="numerical vectors"):
        phase_spans(run, 0.5, history=_changed(history, positions=[0.2, 0.5]))
    constant = CoefficientTransfer(np.zeros((1, 1), complex), None, None)
    result = phase_operands(constant, run, 0.5, history=history)
    assert result.restored_rates is None
    assert result.transported_argument == ComplexEnvelope(0.0, 0.0)
    for operator in (
        CoefficientTransfer(np.zeros((2, 1), complex), None, None),
        _represented(_table(np.zeros((2, 1, 1)), upper=1 + 1e-12)),
    ):
        with pytest.raises(BackendError, match="cell counts"):
            phase_operands(operator, run, 0.5, history=history)
    with pytest.raises(BackendError, match="cell counts"):
        phase_operands(
            constant, _changed(run, active=np.ones(2, bool)), 0.5, history=history
        )

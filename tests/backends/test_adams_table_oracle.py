# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams trial's coefficient tables against exact ones (SEP 39 section 7.2, R7).

No condition-number budget bounds these tables over the histories the march admits
(SEP 39 section 4.7): a landing may shorten a step by any ratio while a plane keeps the
history, and rejections compound the smallest factor. Instead every helper record's
tables are compared with the same tables computed in rational arithmetic
(:class:`fractions.Fraction`) from the same binary64 nodes, and the error is measured in
the weights a trial applies: per row, the sum of absolute errors over the sum of
absolute exact weights.

The node sets are the stencils the march actually formed on the decision-trace corpus
(``tests/data/adams_traces.json.gz``: landings shortened down to ``1e-9`` of a step,
rejection cascades, orders one to six, non-default factors), and the review's extreme
example, four unit steps followed by a trial of ``1e-10``. A record's error must not
exceed the larger of four times the plain composition's on the same nodes and 16 ulp;
the plain composition's own error is the reference, reported rather than bounded.
"""

from __future__ import annotations

import gzip
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from solprop._helpers import NUMBA, PLAIN, Helpers
from solprop.backends.multistep import fused_tables, trial_tables

try:
    from solprop_native._helpers import NATIVE  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - the native package is optional
    NATIVE = None

CORPUS = Path(__file__).parents[1] / "data" / "adams_traces.json.gz"

ULPS = 16 * np.finfo(np.float64).eps
"""The floor of a record's allowed error, relative to the exact weights."""

FACTOR = 4.0
"""How far a record's error may exceed the plain composition's on the same nodes."""


def _stencils() -> list[tuple[tuple[float, ...], int, tuple[int, ...]]]:
    """Return the distinct stencils of the corpus, and the review's extreme one."""
    with gzip.open(CORPUS, "rt") as handle:
        corpus = json.load(handle)
    found: dict[tuple[float, ...], tuple[int, tuple[int, ...]]] = {}
    for entry in corpus["scenarios"]:
        for event in entry["trace"]["events"]:
            if event.get("nodes"):
                key = tuple(event["nodes"])
                found.setdefault(key, (event["order"], tuple(event["orders"])))
    stencils = [(nodes, order, orders) for nodes, (order, orders) in found.items()]
    # Four unit steps, then a trial shortened to 1e-10 of a step (SEP 39 R7): the
    # corrector's nodes are (1, 0, -1e10, ..., -4e10).
    extreme = (0.0, -1e10, -2e10, -3e10, -4e10)
    stencils += [(extreme, 4, (4, 3, 5)), (extreme, 5, (5, 4))]
    return stencils


STENCILS = _stencils()


def _records() -> list[Helpers]:
    return [NUMBA] if NATIVE is None else [NUMBA, NATIVE]


# ------------------------------------------------------------------ exact tables


def _lagrange(nodes: list[Fraction]) -> list[list[Fraction]]:
    """Return the monomial coefficients of the Lagrange basis, exactly."""
    rows = []
    for j, node in enumerate(nodes):
        coefficients = [Fraction(1)]
        denominator = Fraction(1)
        for i, other in enumerate(nodes):
            if i == j:
                continue
            product = [Fraction(0), *coefficients]
            for power, value in enumerate(coefficients):
                product[power] -= other * value
            coefficients = product
            denominator *= node - other
        rows.append([value / denominator for value in coefficients])
    return rows


def _weights(nodes: list[Fraction]) -> list[Fraction]:
    """Return the integrated weights of the Lagrange basis over ``[0, 1]``, exactly."""
    return [
        sum((value / (power + 1) for power, value in enumerate(row)), Fraction(0))
        for row in _lagrange(nodes)
    ]


def _defect(nodes: list[Fraction], width: int) -> list[Fraction]:
    """Return one factored defect row: divided-difference coefficients, then the
    node polynomial's ascending coefficients, each padded to `width`."""
    points = [Fraction(1), *nodes]
    row = [Fraction(0)] * (2 * width)
    for t, node in enumerate(points):
        denominator = Fraction(1)
        for s, other in enumerate(points):
            if s != t:
                denominator *= node - other
        row[t] = 1 / denominator
    coefficients = [Fraction(1)]
    for root in nodes:
        product = [Fraction(0), *coefficients]
        for power, value in enumerate(coefficients):
            product[power] -= root * value
        coefficients = product
    row[width : width + len(coefficients)] = coefficients
    return row


def _exact(nodes: tuple[float, ...], order: int, orders: tuple[int, ...]) -> dict:
    """Return every table a trial applies, exactly, as rows of fractions."""
    theta = [Fraction(node) for node in nodes]
    width = 1 + max(orders)
    one = Fraction(1)

    def padded(values: list[Fraction], length: int) -> list[Fraction]:
        return [*values, *([Fraction(0)] * (length - len(values)))]

    predictor = _lagrange(theta[:order])
    corrector = [padded(row, width) for row in _lagrange([one, *theta[:order]])]
    corrector += [[Fraction(0)] * width] * (width - len(corrector))
    defects = [_defect(theta[:k], width) for k in orders]
    spans = [[one - node for node in theta[: width - 1]]]
    weights = {k: (_weights(theta[:k]), _weights([one, *theta[:k]])) for k in orders}
    weights.setdefault(
        order, (_weights(theta[:order]), _weights([one, *theta[:order]]))
    )
    comp_rows = []
    for k in orders:
        predicted, through = weights[k]
        row = padded(through, width)
        for index, value in enumerate(predicted):
            row[index + 1] -= value
        comp_rows.append(row)
    fused_rows = comp_rows
    return {
        "predictor": predictor,
        "corrector": corrector,
        "factored": defects,
        "spans": spans,
        "carried": [weights[order][0]],
        "endpoint": [weights[order][1]],
        "compensation": comp_rows,
        "fused_predictor": [weights[order][0]],
        "fused_corrector": [padded(weights[orders[0]][1], width)],
        "fused_defects": fused_rows,
    }


def _computed(
    helpers: Helpers, nodes: tuple[float, ...], order: int, orders: tuple[int, ...]
) -> dict:
    """Return the same tables as `helpers` forms them, as rows of floats."""
    listed = list(nodes)
    predictor, corrector, factored, spans, carried, endpoint, defects = trial_tables(
        listed, order, list(orders), compensated=True, helpers=helpers
    )
    fused_predictor, fused_corrector, fused_defects = fused_tables(
        listed, order, list(orders), helpers=helpers
    )
    return {
        "predictor": np.atleast_2d(predictor),
        "corrector": np.atleast_2d(corrector),
        "factored": np.atleast_2d(factored),
        "spans": np.atleast_2d(spans),
        "carried": np.atleast_2d(carried),
        "endpoint": np.atleast_2d(endpoint),
        "compensation": np.atleast_2d(defects),
        "fused_predictor": np.atleast_2d(fused_predictor),
        "fused_corrector": np.atleast_2d(fused_corrector),
        "fused_defects": np.atleast_2d(fused_defects),
    }


def _errors(computed: np.ndarray, exact: list[list[Fraction]]) -> list[float]:
    """Return each row's relative error in the weights it applies."""
    found = []
    for values, row in zip(computed, exact, strict=True):
        scale = sum(abs(value) for value in row)
        error = sum(
            abs(Fraction(float(value)) - target)
            for value, target in zip(values, row, strict=True)
        )
        found.append(float(error / scale) if scale else float(error))
    return found


def _table_errors(helpers: Helpers, stencil) -> dict[str, list[float]]:
    exact = _exact(*stencil)
    computed = _computed(helpers, *stencil)
    return {name: _errors(computed[name], exact[name]) for name in exact}


def test_the_corpus_supplies_extreme_histories():
    # The oracle's node sets include landings shortened far below a step and every
    # order: the histories a condition-number budget could not have covered.
    orders = {order for _, order, _ in STENCILS}
    assert orders >= {1, 2, 3, 4, 5, 6}
    spread = max(max(abs(node) for node in nodes) for nodes, _, _ in STENCILS)
    assert spread >= 1e9


@pytest.mark.parametrize("helpers", _records(), ids=lambda helpers: helpers.name)
def test_every_record_is_within_its_bound_of_the_exact_tables(helpers):
    worst = 0.0
    for stencil in STENCILS:
        reference = _table_errors(PLAIN, stencil)
        record = _table_errors(helpers, stencil)
        for name, rows in record.items():
            for found, plain in zip(rows, reference[name], strict=True):
                assert found <= max(FACTOR * plain, ULPS), (name, stencil, found, plain)
                worst = max(worst, found)
    assert math.isfinite(worst)


def test_the_plain_composition_is_accurate_on_the_corpus():
    # The reference's own error. Measured at SEP 39 phase 0, on every stencil the
    # march formed and on the review's extreme history, it was at most 4.5e-16 of a
    # row's weights, about two ulp: the products of roots and divided differences do
    # not inherit the monomial Vandermonde's conditioning. A rise past 16 ulp is a
    # finding about the method's numerics, for the author, not a bound to loosen.
    worst: dict[str, float] = {}
    for stencil in STENCILS:
        for name, rows in _table_errors(PLAIN, stencil).items():
            worst[name] = max(worst.get(name, 0.0), *rows)
    assert all(value <= ULPS for value in worst.values()), worst

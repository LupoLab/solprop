# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The helper records: the per-stage Chebyshev rows and the Adams basis (SEP 29, R2).

Every record performs the operations of the plain Python forms in their order under
strict arithmetic, so each is held to the rounding bound of those operations against
the plain forms (SEP 25: bounds, not bits), derived here before any comparison. That
the native record also agrees to the last bit is recorded as information only.
"""

from typing import Any

import numpy as np
import pytest

from bounds import EPS, assert_within, gamma
from solprop._helpers import NUMBA, PLAIN, Helpers
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper.multistep import integrated_weights, lagrange_coefficients
from solprop.translate.chebyshev import build

try:
    from solprop_native import (  # pyright: ignore[reportMissingImports]
        NativeBackend,
        _core,  # pyright: ignore[reportMissingModuleSource]
    )
    from solprop_native._helpers import NATIVE  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - the native package is optional
    NATIVE = None
    NativeBackend = None
    _core = None

needs_native = pytest.mark.skipif(NATIVE is None, reason="solprop-native is absent")

EDGES = np.array([0.0, 0.3, 0.45, 1.0, 1.75])
"""Four unequal intervals of a 1.75 m span."""

SLACK = 1e-10 * (EDGES[-1] - EDGES[0])
"""The span's tolerated overreach, as ``PiecewiseChebyshev.at_positions`` takes it."""

POSITIONS = np.array(
    [
        EDGES[0],  # the span's left end
        EDGES[0] - 0.5 * SLACK,  # within the slack below
        0.1,
        EDGES[1],  # an interior breakpoint: the interval starting there
        EDGES[2],
        0.7,
        EDGES[-1],  # the right end: the last interval
        EDGES[-1] + 0.5 * SLACK,  # within the slack above
    ]
)


def coefficients(degree: int, n_values: int, dtype: type) -> np.ndarray:
    """Return random coefficients ``(n_intervals, degree + 1, n_values)``."""
    rng = np.random.default_rng(degree * 100 + n_values)
    shape = (EDGES.size - 1, degree + 1, n_values)
    values = rng.standard_normal(shape)
    if dtype is np.complex128:
        values = values + 1j * rng.standard_normal(shape)
    # Decaying, as a resolved table's coefficients do, but not to zero.
    return np.ascontiguousarray(values / (1.0 + np.arange(degree + 1))[:, None])


def clenshaw_bound(
    c: np.ndarray, positions: np.ndarray, edges: np.ndarray = EDGES
) -> np.ndarray:
    """Return the absolute bound between two Clenshaw passes, per row and value.

    Each pass's departure from the exact sum at the same ``x`` is at most
    ``gamma(4 n) n sum_k |c_k|`` for ``n`` terms and ``|x| <= 1``: every step rounds
    a product and two sums of values bounded by ``n sum |c|`` (the Chebyshev functions
    of the second kind are at most ``n`` on [-1, 1]). The map onto [-1, 1] adds its own
    three roundings times the derivative, at most ``n**2 sum |c|``. Two passes differ
    by at most twice the sum.
    """
    n_terms = c.shape[1]
    total = np.sum(np.abs(c), axis=1)  # (n_intervals, n_values)
    per_interval = (
        2.0 * (gamma(4 * n_terms) * n_terms + gamma(4) * n_terms**2) * total
        + np.finfo(float).tiny
    )
    index = np.searchsorted(edges, positions, side="right") - 1
    return per_interval[np.clip(index, 0, edges.size - 2)]


def basis_bound(nodes: np.ndarray, upper: float, *, integrate: bool) -> np.ndarray:
    """Return the absolute bound between two Lagrange bases on `nodes`.

    Row ``j``'s numerator is built by ``n - 1`` multiplications by ``(x - theta_m)``;
    each coefficient's rounding is at most ``gamma(2 n)`` times the coefficient of the
    numerator with every root's magnitude, ``prod (x + |theta_m|)``. The denominator's
    ``n - 1`` differences and products add ``gamma(2 n)`` relative, Horner's
    integration ``gamma(2 n)`` times the integral of that magnitude polynomial, and
    the division one more rounding: ``gamma(8 n)`` covers the chain, twice for two
    implementations.
    """
    count = nodes.size
    rows = []
    for j in range(count):
        others = np.delete(nodes, j)
        magnitude = np.polynomial.polynomial.polyfromroots(-np.abs(others))
        denominator = abs(float(np.prod(nodes[j] - others)))
        if integrate:
            powers = np.abs(upper) ** np.arange(1, count + 1) / np.arange(1, count + 1)
            rows.append(np.sum(magnitude * powers) / denominator)
        else:
            rows.append(magnitude / denominator)
    return 2.0 * gamma(8 * count) * np.array(rows)


def node_sets() -> list[np.ndarray]:
    """Return variable-step Adams stencils of 1 to 8 nodes, the endpoint first."""
    rng = np.random.default_rng(29)
    sets = []
    for count in range(1, 9):
        steps = rng.uniform(0.4, 1.6, size=count)
        sets.append(np.concatenate(([1.0], -np.cumsum(steps)))[:count])
    return sets


def records() -> list[Helpers]:
    """Return the records under test against the plain forms."""
    found = [NUMBA]
    if NATIVE is not None:
        found.append(NATIVE)
    return found


# ------------------------------------------------------------------ ownership


def test_each_backend_names_its_own_record():
    assert NumpyRefBackend.helpers is PLAIN
    assert NumbaCpuBackend.helpers is NUMBA
    if NativeBackend is not None:
        assert NativeBackend.helpers is NATIVE


def test_a_record_loads_its_functions_at_first_use():
    calls = []

    def load() -> Any:
        calls.append(1)
        return lambda *_: -1

    record = Helpers("probe", load, load)
    assert not calls
    assert record.points is record.points
    assert len(calls) == 1


# ------------------------------------------------------------ Chebyshev rows


@pytest.mark.parametrize("helpers", records(), ids=lambda record: record.name)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("n_values", [1, 33])
@pytest.mark.parametrize("degree", [0, 1, 2, 8, 64])
def test_points_agree_with_the_plain_form(
    helpers, dtype, n_values, degree, record_property
):
    c = coefficients(degree, n_values, dtype)
    bound = clenshaw_bound(c, POSITIONS)
    lower, upper = EDGES[0] - SLACK, EDGES[-1] + SLACK
    expected = np.empty((POSITIONS.size, n_values), dtype=dtype)
    actual = np.empty_like(expected)
    assert PLAIN.points(POSITIONS, EDGES, lower, upper, c, expected) == -1
    assert helpers.points(POSITIONS, EDGES, lower, upper, c, actual) == -1
    for part in (np.real, np.imag) if dtype is np.complex128 else (np.real,):
        assert_within(
            part(actual), part(expected), bound, elementwise=True, what="rows"
        )
    record_property("bitwise", bool(np.array_equal(actual, expected)))


@pytest.mark.parametrize("helpers", records(), ids=lambda record: record.name)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_points_refuse_outside_and_carry_nan(helpers, dtype):
    c = coefficients(8, 3, dtype)
    lower, upper = EDGES[0] - SLACK, EDGES[-1] + SLACK
    outside = np.array([0.2, 0.9, EDGES[-1] + 2.0 * SLACK, 0.5])
    rows = np.zeros((outside.size, 3), dtype=dtype)
    assert helpers.points(outside, EDGES, lower, upper, c, rows) == 2
    # The rows from the refused position on are not written.
    assert not np.any(rows[2:])
    below = np.array([EDGES[0] - 2.0 * SLACK])
    assert helpers.points(below, EDGES, lower, upper, c, rows[:1]) == 0
    # A NaN is not refused: it sorts last, takes the last interval, and is NaN.
    missing = np.array([np.nan, 0.5])
    values = np.empty((2, 3), dtype=dtype)
    assert helpers.points(missing, EDGES, lower, upper, c, values) == -1
    assert np.all(np.isnan(values[0]))
    assert np.all(np.isfinite(values[1]))


@needs_native
@pytest.mark.parametrize("imaginary", [0.0, 0.5])
def test_a_tables_positions_through_the_native_record(imaginary):
    table = build(
        lambda z: np.cos(3.0 * z) + imaginary * 1j * np.sin(2.0 * z), 0.0, 1.0
    )
    positions = np.array([0.0, 0.25, 0.5, 1.0])
    assert NATIVE is not None
    plain = table.at_positions(positions, helpers=PLAIN)
    native = table.at_positions(positions, helpers=NATIVE)
    flat = table.coefficients.reshape(*table.coefficients.shape[:2], -1)
    bound = clenshaw_bound(flat, positions, table.breakpoints)[:, 0]
    for part in (np.real, np.imag) if imaginary else (np.real,):
        assert_within(part(native), part(plain), bound, elementwise=True)


# ---------------------------------------------------------------- Adams basis


@pytest.mark.parametrize("helpers", records(), ids=lambda record: record.name)
@pytest.mark.parametrize("upper", [1.0, 0.37, 1.6])
@pytest.mark.parametrize("nodes", node_sets(), ids=lambda nodes: f"{nodes.size}")
def test_basis_agrees_with_the_plain_form(helpers, upper, nodes, record_property):
    rows_plain, weights_plain = PLAIN.basis(nodes, upper, True)
    rows, weights = helpers.basis(nodes, upper, True)
    # Integrating, the rows are zero: a structural contract, exact.
    assert not np.any(rows) and not np.any(rows_plain)
    assert_within(
        weights,
        weights_plain,
        basis_bound(nodes, upper, integrate=True),
        elementwise=True,
        what="weights",
    )
    coefficients_plain, _ = PLAIN.basis(nodes, upper, False)
    coefficients_, _ = helpers.basis(nodes, upper, False)
    assert_within(
        coefficients_,
        coefficients_plain,
        basis_bound(nodes, upper, integrate=False),
        elementwise=True,
        what="coefficients",
    )
    record_property(
        "bitwise",
        bool(
            np.array_equal(weights, weights_plain)
            and np.array_equal(coefficients_, coefficients_plain)
        ),
    )


@pytest.mark.parametrize("helpers", records(), ids=lambda record: record.name)
def test_the_shared_functions_take_the_record(helpers):
    nodes = np.array([1.0, 0.0, -1.0, -2.0])
    # Adams-Moulton 4, times 24: a few operations on quarters, well inside 64 EPS.
    assert_within(
        24.0 * integrated_weights(nodes, helpers=helpers),
        np.array([9.0, 19.0, -5.0, 1.0]),
        64.0 * EPS * 20.0,
        scale=1.0,
    )
    linear = lagrange_coefficients([1.0, 0.0], helpers=helpers)
    np.testing.assert_array_equal(linear, [[0.0, 1.0], [1.0, -1.0]])


# ---------------------------------------------------------- native refusals


@needs_native
def test_the_native_bindings_refuse_hidden_conversions():
    # Untyped on purpose: every call below hands the bindings what they must refuse.
    core: Any = _core
    c = coefficients(2, 3, np.float64)
    out = np.empty((2, 3))
    z = np.array([0.1, 0.2])
    with pytest.raises(TypeError):
        core.clenshaw_points(z, EDGES, 0.0, 1.75, c.astype(np.float32), out)
    with pytest.raises(TypeError):
        core.clenshaw_points(np.array([0.1, 0.5, 0.2])[::2], EDGES, 0.0, 1.75, c, out)
    with pytest.raises(TypeError):
        core.clenshaw_points(z, EDGES, 0.0, 1.75, c, np.empty((2, 3), np.complex128))
    held = np.empty((2, 3))
    held.flags.writeable = False
    with pytest.raises(TypeError):
        core.clenshaw_points(z, EDGES, 0.0, 1.75, c, held)
    with pytest.raises(ValueError, match="out has shape"):
        core.clenshaw_points(z, EDGES, 0.0, 1.75, c, np.empty((3, 3)))
    with pytest.raises(ValueError, match="do not match"):
        core.clenshaw_points(z, EDGES[:3], 0.0, 1.75, c, out)
    with pytest.raises(ValueError, match="at least two"):
        core.clenshaw_points(z, EDGES[:1], 0.0, 1.75, c[:0], out)
    points = np.zeros(33)
    with pytest.raises(ValueError, match="exceed"):
        core.basis_rows(points, 1.0, True, np.empty((33, 33)), np.empty(33))
    with pytest.raises(ValueError, match="must be"):
        core.basis_rows(points[:3], 1.0, True, np.empty((3, 2)), np.empty(3))
    with pytest.raises(TypeError):
        core.basis_rows(
            points[:3].astype(np.float32), 1.0, True, np.empty((3, 3)), np.empty(3)
        )

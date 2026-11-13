# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general z table on the native backend's CPU build (SEP 29 section 6.6).

The native backend holds a z table's coefficients resident and evaluates them at a
trial's positions into the slots its stage kernels read as terms; the stage weights
then select them, ``Phi(z1) - Phi(z0)`` as ``e_1 - e_0`` and a value as ``e_1``. These
tests hold the slots to the reference's own host evaluation
(:class:`solprop.translate.chebyshev.PiecewiseChebyshev`, through NumPy's ``chebval``)
within a Clenshaw rounding budget, the stage kernels' phase sum over two slots to the
table's difference and the compensation's rate slot to solprop's ``phi_rate``. A device
build refuses the table, naming its size and the CPU backends; no CPU configuration is
refused. Ten fixed steps of each march against the reference, with ``Phi``, the
prefactor or both on the table, are the catalogue's ``z table`` rows
(``tests/backends/session_rows.py``).
"""

import numpy as np
import pytest

from backends.session_rows import MARCHES, TABLES, capillary, tabulate
from backends.session_rows import Z_TABLE_LENGTH as LENGTH
from bounds import EPS
from march_bounds import FIXED_ORDER
from solprop.exceptions import BackendError
from solprop.translate.linop import phi_rate

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")
NativeBackend = native.NativeBackend


def clenshaw_bound(table, z):
    """Return a value's rounding budget per element: ``8 n^2 eps sum_i |c_i|``.

    The budget ``tests/backends/test_host_tables.py`` derives for one evaluation of
    the recursion (at most eight real operations per complex coefficient and step,
    each step's error carried by the remaining ones), here per element of the interval
    `z` lies in, for each of the two evaluations compared.
    """
    index = table.interval_containing(z)
    coefficients = table.coefficients[index]
    terms = coefficients.shape[0]
    return 8.0 * terms**2 * EPS * np.sum(np.abs(coefficients), axis=0)


@pytest.fixture(scope="module")
def tabulated(field_grid):
    """The capillary with both quantities on the z table, under the Adams march."""
    return tabulate(capillary(field_grid, FIXED_ORDER))


def test_the_slots_hold_the_tables_values(tabulated, field_grid):
    backend = NativeBackend(*tabulated, grid=field_grid)
    tables = backend._z_tables
    try:
        assert tables is not None and tables.phi and tables.prefactor
        phi = tabulated.problem.linear.polynomial
        prefactor = tabulated.problem.responses.prefactor
        # The span's ends, an interior breakpoint and points between.
        positions = sorted(
            {0.0, float(phi.breakpoints[1]), 0.3 * LENGTH, LENGTH, phi.z_span[1]}
        )
        for z in positions:
            for kind, table in (("phi", phi), ("prefactor", prefactor)):
                got = tables.values(kind, z).reshape(table.value_shape)
                want = table(z)
                bound = clenshaw_bound(table, z)
                # Each value against the host's within both evaluations' budgets.
                assert np.all(np.abs(got - want) <= 2 * bound), (kind, z)
        assert tables.evaluations == len(positions)
        # A position held in a slot is not evaluated again.
        tables.load(positions[-tables.slots :])
        assert tables.evaluations == len(positions)
    finally:
        backend.close()


def test_the_phase_sum_of_two_slots_is_the_tables_difference(tabulated, field_grid):
    backend = NativeBackend(*tabulated, grid=field_grid)
    tables = backend._z_tables
    native_state = backend._native_state
    try:
        phi = tabulated.problem.linear.polynomial
        z0, z1 = 0.25 * LENGTH, 0.4 * LENGTH
        slot_of = tables.load([z0, z1])
        row = tables.phi_row(slot_of, z0, z1)
        assert sorted(row[row != 0].tolist()) == [-1.0, 1.0]
        got = native_state.phase_sum_to_host(row).reshape(phi.value_shape)
        want = phi(z1) - phi(z0)
        # Two evaluations on each side, each within its budget, and one rounding of
        # the difference each.
        bound = 2 * (clenshaw_bound(phi, z0) + clenshaw_bound(phi, z1)) + 2 * EPS * (
            np.abs(want)
        )
        assert np.all(np.abs(got - want) <= bound)
        # The host's exponent for a host engine: the same two slots, subtracted once.
        bundle = backend._stage_scalars(z0, z1)
        assert np.array_equal(backend._delta_phi(bundle).reshape(got.shape), got)
        # A step of no length weighs both slots by zero: an exact zero, structural.
        still = native_state.phase_sum_to_host(tables.phi_row(slot_of, z1, z1))
        assert not np.any(still)
    finally:
        backend.close()


def test_the_rate_slot_holds_the_derivative_table(tabulated, field_grid):
    # The exponential Adams march's compensation reads D = dPhi/dz from the
    # derivative table, on the side of an interior breakpoint a step came from.
    backend = NativeBackend(*tabulated, grid=field_grid)
    tables = backend._z_tables
    try:
        assert tables.rate_slot == tables.slots
        rate = tabulated.problem.linear.rate
        edge = float(rate.breakpoints[1])
        for z, before in ((edge, True), (edge, False), (0.5 * LENGTH, True)):
            row = backend._rate_weights(z, before=before)
            assert row.tolist() == [0.0] * tables.slots + [1.0]
            got = backend._native_state.table_terms_to_host("phi", tables.rate_slot)
            want = phi_rate(tabulated.problem.linear, z, before=before)
            index = rate.interval_containing(z)
            if before and index > 0 and z <= float(rate.breakpoints[index]):
                index -= 1
            terms = rate.coefficients.shape[1]
            bound = (
                2
                * 8.0
                * terms**2
                * EPS
                * np.sum(np.abs(rate.coefficients[index]), axis=0)
            )
            assert np.all(np.abs(got.reshape(want.shape) - want) <= bound), (z, before)
    finally:
        backend.close()


def test_a_device_build_refuses_the_table_naming_the_cpu_backends(
    tabulated, field_grid, monkeypatch
):
    monkeypatch.setattr(backend_module, "_on_device", lambda: True)
    with pytest.raises(BackendError, match="on an accelerator") as refused:
        NativeBackend(*tabulated, grid=field_grid)
    message = str(refused.value)
    size = (
        tabulated.problem.linear.phi_coefficients.nbytes
        + tabulated.problem.responses.prefactor.coefficients.nbytes
    )
    assert f"{size / 1e6:.3g} MB" in message
    # The native CPU build is named first, the reference after it.
    assert message.index("a CPU build of native") < message.index('backend="numpy_ref"')


@pytest.mark.parametrize("tables", TABLES)
def test_no_cpu_configuration_refuses_a_z_table(field_grid, tables, monkeypatch):
    # SEP 29 section 7: on a CPU build the native backend admits every z table, under
    # every march.
    def refuse(*_):
        raise AssertionError("a CPU build reached the device refusal")

    monkeypatch.setattr(backend_module, "require_device_ready", refuse)
    phi, prefactor = TABLES[tables]
    for integrator in MARCHES.values():
        problem, run_request = tabulate(
            capillary(field_grid, integrator), phi=phi, prefactor=prefactor
        )
        NativeBackend(problem, run_request, grid=field_grid).close()


def test_the_ledger_counts_the_resident_tables(tabulated, field_grid):
    backend = NativeBackend(*tabulated, grid=field_grid)
    try:
        ledger = backend.memory_ledger()
        tables = backend._z_tables
        edges = 8 * (
            tabulated.problem.linear.polynomial.breakpoints.size
            + tabulated.problem.linear.rate.breakpoints.size
            + tabulated.problem.responses.prefactor.breakpoints.size
        )
        assert ledger["z_tables"] == tables.nbytes + edges
        field = 16 * int(np.prod(tabulated.problem.shape))
        assert ledger["fixed_phi_terms"] == tables.n_phi * field
        assert ledger["fixed_prefactor_terms"] == tables.slots * field
    finally:
        backend.close()


def test_the_slots_are_reused_least_recently_used_first(tabulated, field_grid):
    backend = NativeBackend(*tabulated, grid=field_grid)
    tables = backend._z_tables
    try:
        count = tables.slots
        first = [LENGTH * index / (2 * count) for index in range(count)]
        held = tables.load(first)
        assert sorted(held.values()) == list(range(count))
        # One new position takes the least recently used slot, the first position's;
        # a call's own positions are never evicted by each other.
        assert tables.load([LENGTH])[LENGTH] == held[first[0]]
        again = tables.load([first[1], 0.9 * LENGTH])
        assert again[first[1]] == held[first[1]]
        assert again[0.9 * LENGTH] == held[first[2]]
        assert tables.evaluations == count + 2
        with pytest.raises(BackendError, match="more than the"):
            tables.load([LENGTH * index / (3 * count) for index in range(count + 1)])
    finally:
        backend.close()

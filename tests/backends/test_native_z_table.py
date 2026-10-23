# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general z table on the native backend's CPU build (SEP 29 section 6.6).

The native backend holds a z table's coefficients resident and evaluates them at a
trial's positions into the slots its stage kernels read as terms; the stage weights
then select them, ``Phi(z1) - Phi(z0)`` as ``e_1 - e_0`` and a value as ``e_1``. These
tests hold the slots to the reference's own host evaluation
(:class:`solprop.backends.numba_cpu.tables.HostTable`, NumPy's ``chebval`` operation
order) within a Clenshaw rounding budget, the stage kernels' phase sum over two slots to
the table's difference, the compensation's rate slot to solprop's ``phi_rate``, and
ten fixed steps of each march to the reference within the matched-march budget of
``tests/march_bounds.py`` -- with ``Phi``, the prefactor or both on the table, since
either may be a table while the other is an expansion. A device build refuses the
table, naming its size and the CPU backends; no CPU configuration is refused.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from bounds import EPS
from march_bounds import FIXED_ORDER, assert_matched, march_bound
from solprop.backends.memory import estimate_memory, ledger_groups, z_table_slots
from solprop.backends.numba_cpu.tables import HostTable
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.exceptions import BackendError, MaterialRangeWarning
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController
from solprop.translate.chebyshev import build
from solprop.translate.linop import delta_phi, phi_rate
from solprop.translate.problem import assemble
from solprop.translate.responses import prefactor_at

native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")
NativeBackend = native.NativeBackend

LENGTH = 0.02
"""Two centimetres of the gradient capillary, in :data:`STEPS` fixed steps."""

STEPS = 10

MARCHES = {
    "adams": FIXED_ORDER,
    "runge_kutta": None,
    "low_storage": LowStorageRungeKutta(),
}
"""Each march on the prescribed schedule: the exponential Adams method (a varying
operator's compensated march), Runge-Kutta (``None`` replays the schedule with it) and
the low-storage march."""

TABLES = {
    "both": (True, True),
    "phi": (True, False),
    "prefactor": (False, True),
}
"""Which of the two quantities take the z table; the other keeps its expansion."""


def capillary(field_grid, integrator):
    """Return the helium gradient capillary on ten fixed steps, its expansions intact.

    The fixture of ``tests/backends/conftest.py`` (``capillary_problem``) on a
    prescribed schedule, with the collar integrated exactly so that the explicit
    marches are stable at its steps.
    """
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    schedule = FixedSchedule.uniform(
        0.0, LENGTH, LENGTH / STEPS * 1.0000001, save_planes=[LENGTH]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            field_grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[LENGTH],
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            schedule=schedule,
            integrator=integrator,
            absorber=absorber(field_grid),
            absorber_scheme="split_step",
        )


def tabulate(problem, substitute, *, phi=True, prefactor=True):
    """Return `problem` with ``Phi``, the prefactor or both on the z table.

    The tables are built from the values the expansions give, as
    ``tests/backends/conftest.py``'s ``tabulated_problem`` builds them, the way a mode
    set that declares no parameters would present them.
    """
    linear, responses = problem.linear, problem.responses
    if phi:
        table = build(
            lambda z: np.stack([delta_phi(linear, 0.0, float(one)) for one in z]),
            *linear.z_span,
            tol=1e-10,
        )
        linear = dataclasses.replace(
            linear,
            phi_expansion=None,
            phi_representation="table",
            phi_breakpoints=table.breakpoints,
            phi_coefficients=table.coefficients,
            phi_degrees=table.degrees,
            hard_kinks=table.hard_kinks,
            phi_constant=None,
        )
    if prefactor:
        table = build(
            lambda z: np.stack([prefactor_at(responses, float(one)) for one in z]),
            *responses.prefactor_span,
            tol=1e-10,
        )
        for values in (
            table.coefficients,
            table.breakpoints,
            table.degrees,
            table.hard_kinks,
        ):
            values.setflags(write=False)
        responses = dataclasses.replace(
            responses,
            prefactor_representation="table",
            prefactor=table,
            prefactor_expansion=None,
            prefactor_constant=None,
        )
    return substitute(problem, linear=linear, responses=responses)


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


def launch_phase(problem, grid):
    """Return the run's nonlinear phase ``L ||N(a_0)|| / ||a_0||`` by the reference.

    ``tests/march_bounds.py``'s ``measured_scales``, at the seam.
    """
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        state = problem.initial.field0
        backend._hoist(0.0)
        source = backend._nonlinear(
            state, backend._stage_scalars(0.0, 0.0), np.ones_like(state)
        )
    finally:
        backend.close()
    return LENGTH * float(np.linalg.norm(source) / np.linalg.norm(state))


def replay(backend_class, problem, grid):
    """Replay the schedule to its plane; return the observed field."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        field = backend.observe().require_field().copy()
    finally:
        backend.close()
    assert np.all(np.isfinite(field))
    return field


@pytest.fixture(scope="module")
def tabulated(field_grid, substitute):
    """The capillary with both quantities on the z table, under the Adams march."""
    return tabulate(capillary(field_grid, FIXED_ORDER), substitute)


def test_the_slots_hold_the_tables_values(tabulated, field_grid):
    backend = NativeBackend(tabulated, grid=field_grid)
    tables = backend._z_tables
    try:
        assert tables is not None and tables.phi and tables.prefactor
        phi = tabulated.linear.polynomial
        prefactor = tabulated.responses.prefactor
        # The span's ends, an interior breakpoint and points between.
        positions = sorted(
            {0.0, float(phi.breakpoints[1]), 0.3 * LENGTH, LENGTH, phi.z_span[1]}
        )
        for z in positions:
            for kind, table in (("phi", phi), ("prefactor", prefactor)):
                got = tables.values(kind, z).reshape(table.value_shape)
                want = HostTable(table).at(z)
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
    backend = NativeBackend(tabulated, grid=field_grid)
    tables = backend._z_tables
    native_state = backend._native_state
    try:
        phi = tabulated.linear.polynomial
        host = HostTable(phi)
        z0, z1 = 0.25 * LENGTH, 0.4 * LENGTH
        slot_of = tables.load([z0, z1])
        row = tables.phi_row(slot_of, z0, z1)
        assert sorted(row[row != 0].tolist()) == [-1.0, 1.0]
        got = native_state.phase_sum_to_host(row).reshape(phi.value_shape)
        want = host.delta(z0, z1)
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
    backend = NativeBackend(tabulated, grid=field_grid)
    tables = backend._z_tables
    try:
        assert tables.rate_slot == tables.slots
        rate = tabulated.linear.rate
        edge = float(rate.breakpoints[1])
        for z, before in ((edge, True), (edge, False), (0.5 * LENGTH, True)):
            row = backend._rate_weights(z, before=before)
            assert row.tolist() == [0.0] * tables.slots + [1.0]
            got = backend._native_state.table_terms_to_host("phi", tables.rate_slot)
            want = phi_rate(tabulated.linear, z, before=before)
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


@pytest.mark.medium
@pytest.mark.parametrize("tables", TABLES)
@pytest.mark.parametrize("march", MARCHES)
def test_ten_steps_match_the_reference(field_grid, substitute, march, tables):
    phi, prefactor = TABLES[tables]
    problem = tabulate(
        capillary(field_grid, MARCHES[march]),
        substitute,
        phi=phi,
        prefactor=prefactor,
    )
    assert not problem.device_ready
    bound = march_bound(
        problem, phi=max(launch_phase(problem, field_grid), 0.1), backend="native"
    )
    expected = replay(NumpyRefBackend, problem, field_grid)
    got = replay(NativeBackend, problem, field_grid)
    assert_matched(got, expected, bound, what=f"{march}, {tables} on the table")


def test_a_device_build_refuses_the_table_naming_the_cpu_backends(
    tabulated, field_grid, monkeypatch
):
    monkeypatch.setattr(backend_module, "_on_device", lambda: True)
    with pytest.raises(BackendError, match="on an accelerator") as refused:
        NativeBackend(tabulated, grid=field_grid)
    message = str(refused.value)
    size = (
        tabulated.linear.phi_coefficients.nbytes
        + tabulated.responses.prefactor.coefficients.nbytes
    )
    assert f"{size / 1e6:.3g} MB" in message
    # The native CPU build is named first; numba_cpu is one option among the others.
    assert message.index("a CPU build of native") < message.index("numba_cpu")
    assert 'backend="numpy_ref"' in message


@pytest.mark.parametrize("tables", TABLES)
def test_no_cpu_configuration_refuses_a_z_table(
    field_grid, substitute, tables, monkeypatch
):
    # SEP 29 section 7: no supported configuration directs a user to numba_cpu. On a
    # CPU build the native backend admits every z table, under every march.
    def refuse(*_):
        raise AssertionError("a CPU build reached the device refusal")

    monkeypatch.setattr(backend_module, "require_device_ready", refuse)
    phi, prefactor = TABLES[tables]
    for integrator in MARCHES.values():
        problem = tabulate(
            capillary(field_grid, integrator), substitute, phi=phi, prefactor=prefactor
        )
        NativeBackend(problem, grid=field_grid).close()


def test_the_ledger_counts_the_resident_tables(tabulated, field_grid):
    backend = NativeBackend(tabulated, grid=field_grid)
    try:
        ledger = backend.memory_ledger()
        tables = backend._z_tables
        edges = 8 * (
            tabulated.linear.polynomial.breakpoints.size
            + tabulated.linear.rate.breakpoints.size
            + tabulated.responses.prefactor.breakpoints.size
        )
        assert ledger["z_tables"] == tables.nbytes + edges
        field = 16 * int(np.prod(tabulated.shape))
        assert ledger["fixed_phi_terms"] == tables.n_phi * field
        assert ledger["fixed_prefactor_terms"] == tables.slots * field
    finally:
        backend.close()


def test_the_slots_are_reused_least_recently_used_first(tabulated, field_grid):
    backend = NativeBackend(tabulated, grid=field_grid)
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


def test_the_adams_history_estimate_counts_the_rate_slot(tabulated, field_grid):
    # The compensated history's rate weights cover every Phi term: the position slots
    # and the rate's own (solprop's estimate, to the byte).
    estimate = estimate_memory(tabulated, "native").groups["adams_history"]
    backend = NativeBackend(tabulated, grid=field_grid)
    try:
        held = ledger_groups(backend.memory_ledger())["adams_history"]
        assert backend._z_tables.slots == z_table_slots(tabulated)
    finally:
        backend.close()
    assert estimate == held > 0

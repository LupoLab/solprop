# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march's resident engine on native CPU: SEP 24 7.2 and 7.3.

Phase 3 runs the march in solprop-native: one native call a step, the four register
passes in the route's own scratch, accepted in place. Its parity with ``numpy_ref`` and
``numba_cpu`` runs on every case of ``test_numba_low_storage.py`` (its
``compiled_backends``); this module holds what is the engine's own: the ledger and the
estimate, which agree to the byte, the varying operator's flows, the field-free plane
held in the native memory space and its resume, the in-flight guard, and the commit a
signal waits for. No agreement here is bitwise (SEP 25): the resume meets the
uninterrupted run within the parity bound.
"""

import math
import signal
from pathlib import Path

import numpy as np
import pytest

from backends.matched import native_thread_fields
from backends.test_low_storage import MARCH, cartesian_march
from backends.test_numba_low_storage import (
    parity_bound,
    replay,
    uniform,
    varying_capillary,
    with_mode,
)
from bounds import assert_within
from solprop._interrupts import handled
from solprop.backends.memory import (
    estimate_memory,
    low_storage_flows,
    low_storage_stage_register,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.noise import spatial_noise
from solprop.output.memory import MemoryOutput
from solprop.spatial import SpatialFlux, SpatialMedium

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

MARCH_ROWS = (
    "state",
    "transverse_scratch",
    "register_accumulator",
    "register_stage",
    "propagators",
)
"""The ledger's field-sized rows of the march on a Cartesian grid."""

PROBE = Path(__file__).with_name("native_low_storage_thread_probe.py")

RUNGE_KUTTA_ROWS = (
    "advanced",
    "stages",
    "accumulator_solution",
    "accumulator_error",
    "weighted_residual",
    "recycled_stage",
    "stage_inverse",
)
"""The Runge-Kutta march's native views, none allocated under the low-storage one."""


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_native_state_holds_the_registers_and_no_runge_kutta_view(mode, collar):
    """Section 7.3 on Kokkos: three or four volumes beside the state, none of DOPRI5's.

    The accumulator, the stage register under an in-stage collar only (Q4, as on
    ``numba_cpu``), the table when tabulating, the state and the transverse scratch:
    four volumes tabulated without a collar and three recomputed, five and four with
    one. The estimate is the ledger to the byte for these rows.
    """
    problem, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[-1]))
    ledger = backend.memory_ledger()
    volumes = {row: ledger[row] for row in MARCH_ROWS}
    assert volumes == {
        "state": field,
        "transverse_scratch": field,
        "register_accumulator": field,
        "register_stage": field if collar else 0,
        "propagators": field if mode == "tabulate" else 0,
    }
    assert low_storage_stage_register(problem, "native") == collar
    assert low_storage_flows(problem, "native") == (mode == "tabulate")
    assert sum(volumes.values()) == field * (3 + int(collar) + (mode == "tabulate"))
    for row in (*RUNGE_KUTTA_ROWS, "noise", "held_plane", "adams_history"):
        assert ledger[row] == 0, row
    estimate = estimate_memory(problem, "native")
    assert estimate.groups["adams_history"] == 0
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert engine.atomic and math.isnan(engine.stiffness)
    metadata = backend.metadata()
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert metadata["propagator_mode"] == mode
    assert metadata["source_evaluations"] == 4 * schedule.n_steps
    backend.close()


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_estimate_is_the_ledger_to_the_byte(mode, collar):
    """The march's field-sized rows of the estimate, row by row, as numpy_ref counts.

    The estimate's base is the reference's rows less the elided register; its march
    rows -- the state, the scratch, the registers and the flows -- are the native
    ledger's own, to the byte.
    """
    from solprop.backends.memory import (  # noqa: PLC0415
        reference_base_memory,
    )

    problem, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    reference = reference_base_memory(
        problem,
        tabulating=mode == "tabulate",
        embedded=False,
        noise=False,
        statistics_weight=0,
        spatial_absorber=0,
    )
    backend = NativeBackend(problem, grid=grid)
    ledger = backend.memory_ledger()
    backend.close()
    for row in MARCH_ROWS:
        expected = 0 if row == "register_stage" and not collar else reference[row]
        assert ledger[row] == expected, row
    estimate = estimate_memory(problem, "native")
    assert estimate.groups["base"] == sum(reference.values()) - (0 if collar else field)


def test_a_varying_operator_holds_its_pair_tabulated_and_none_recomputed():
    """A varying modal operator's ``P`` and ``Q``: two volumes, or formed per element.

    Recomputed, the flows are formed in the finish from the phase terms, so nothing
    of the state's size is held for them: no transient outside the ledger.
    """
    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 0)):
        problem = with_mode(simulation.translate(), mode)
        backend = NativeBackend(problem, grid=simulation.grid, modes=simulation.modes)
        field = 16 * math.prod(problem.shape)
        ledger = backend.memory_ledger()
        assert ledger["propagators"] == held * field
        assert low_storage_flows(problem, "native") == held
        # The modal route holds C: its input is a padded fine spectrum.
        assert ledger["register_stage"] == field
        assert all(ledger[row] == 0 for row in RUNGE_KUTTA_ROWS)
        backend.close()


def test_a_field_free_plane_is_held_natively_and_resumes_to_rounding():
    """SEP 22's held plane in the native memory space: one volume, then one copy.

    Nothing crosses at the field-free plane; the ledger names the held volume; a
    resume from it meets the uninterrupted run within the parity bound (the table is
    built again from the resumed run's own step, section 4.3).
    """
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
    noise = spatial_noise(
        SpatialFlux.resolve(grid, space, SpatialMedium()),
        np.random.default_rng(7),
        model="modified",
    )
    problem, grid, _ = cartesian_march("tabulate", noise=noise)
    field = 16 * math.prod(problem.shape)
    end = float(problem.stepper.save_planes[-1])
    schedule = problem.stepper.schedule
    assert schedule is not None
    middle = float(schedule.edges[2])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    assert backend.memory_ledger()["noise"] == field
    backend.propagate_to(middle)
    assert backend.held_observation() is None
    assert backend.memory_ledger()["held_plane"] == 0
    plane = backend.observe(field=False)
    assert plane.field is None
    assert backend.memory_ledger()["held_plane"] == field
    held = backend.held_observation()
    assert held is not None and held.noise_phase is not None and held.z == middle
    assert held.field is not None and held.field.shape == problem.shape
    backend.propagate_to(end)
    uninterrupted = backend.observe().require_field().copy()
    # A plane that keeps its field releases the hold; the volume stays charged.
    assert backend.held_observation() is None
    assert backend.memory_ledger()["held_plane"] == field
    backend.close()
    resumed = NativeBackend(problem, grid=grid)
    resumed.set_initial(held.field, noise, z=held.z)
    resumed.propagate_to(end)
    got = resumed.observe().require_field()
    resumed.close()
    assert_within(got, uninterrupted, parity_bound(problem), what="resume")


def test_the_soliton_resumes_from_a_held_plane_on_its_reference(
    soliton_problem, narrow_grid, scales
):
    """The modal route's held plane, and its resume against the reference's run."""
    length = scales.dispersion_length
    schedule = uniform(length, 20, save_planes=[length / 2, length])
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=[length / 2, length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
    )
    backend = NativeBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(length / 2)
    backend.observe(field=False)
    held = backend.held_observation()
    backend.close()
    assert held is not None and held.field is not None
    resumed = NativeBackend(problem, grid=narrow_grid)
    resumed.set_initial(held.field, None, z=held.z)
    resumed.propagate_to(length)
    got = resumed.observe().require_field()
    resumed.close()
    expected = replay(NumpyRefBackend, problem, narrow_grid)
    assert_within(got, expected, parity_bound(problem), what="resume")


def test_a_native_exception_leaves_the_backend_refusing(monkeypatch):
    """Section 6.2's guard: a step that raises after its passes stays in flight."""
    problem, grid, _ = cartesian_march("tabulate")
    end = float(problem.stepper.save_planes[-1])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    original = engine.step
    calls = []

    def failing(z, dz):
        calls.append(z)
        original(z, dz)
        if len(calls) == 2:
            raise RuntimeError("abandoned")
        return 1.0

    monkeypatch.setattr(engine, "step", failing)
    with pytest.raises(RuntimeError, match="abandoned"):
        backend.propagate_to(end)
    for entry in (backend.observe, backend.march_snapshot):
        with pytest.raises(BackendError, match="in flight"):
            entry()
    with pytest.raises(BackendError, match="in flight"):
        backend.propagate_to(end)
    # The abandoned step's start was the first edge: the guard names it.
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert backend._in_flight == (calls[-1], float(schedule.steps[1]))
    backend.set_initial(problem.initial.field0, None)
    backend.observe()
    backend.close()


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="Ctrl-C is delivered by a POSIX signal"
)
def test_a_signal_during_the_native_step_waits_for_its_commit(monkeypatch):
    """Section 6.7: the native step, the finish and the move commit together.

    A Ctrl-C raised while the engine steps is held to the end of the commit: the
    step lands, the backend is not in flight, and the run can be observed.
    """
    problem, grid, _ = cartesian_march("tabulate")
    end = float(problem.stepper.save_planes[-1])
    backend = NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    original = engine.step

    def interrupted(z, dz):
        squared = original(z, dz)
        # Ctrl-C in-process; os.kill would terminate the process on Windows.
        signal.raise_signal(signal.SIGINT)
        return squared

    monkeypatch.setattr(engine, "step", interrupted)
    with handled(), pytest.raises(KeyboardInterrupt):
        backend.propagate_to(end)
    assert backend._in_flight is None
    schedule = problem.stepper.schedule
    assert schedule is not None
    assert backend._z == float(schedule.edges[1])
    assert np.all(np.isfinite(backend.observe().require_field()))
    backend.close()


def test_auto_runs_the_march_on_native_when_installed():
    """``backend="auto"`` resolves to Kokkos for the march, as for every integrator."""
    from solprop.driver import Simulation  # noqa: PLC0415

    length = 0.01
    schedule = uniform(length, 5)
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=2,
        schedule=schedule,
        integrator=MARCH,
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        # Five replayed steps: no collar, as before the default one.
        absorber=False,
        noise=None,
    )
    assert simulation.settings.backend == "native"
    output = simulation.run()
    assert isinstance(output, MemoryOutput)
    assert output.outcome is not None
    assert output.outcome.execution["backend"] == "native"
    assert output.outcome.execution["source_evaluations"] == 4 * schedule.n_steps


@pytest.mark.medium
def test_the_march_agrees_across_thread_counts(tmp_path):
    """One thread twice and four threads, each in a fresh process (SEP 24 7.2).

    The same passes on the same schedule, differing by the order of their threaded
    transforms, folds and reductions alone: within the parity bound fixed for the
    case, as against the reference.
    """
    names = ["tabulate-plain", "recompute-collar"]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    again = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    for name in names:
        mode, collar = name.split("-")
        problem, _, _ = cartesian_march(mode, collar=collar == "collar")
        bound = parity_bound(problem, transverse=problem.transform.n_nodes)
        assert_within(again[name], one[name], bound, what=f"repeat, {name}")
        assert_within(four[name], one[name], bound, what=f"four threads, {name}")

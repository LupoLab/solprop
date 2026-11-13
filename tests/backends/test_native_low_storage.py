# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march's resident engine on native CPU: SEP 24 7.2 and 7.3.

Phase 3 runs the march in solprop-native: one native call a step, the four register
passes in the route's own scratch, accepted in place. Its parity with ``numpy_ref`` on
every route the translation admits the march on is the catalogue's ``low storage``
rows (``tests/backends/session_rows.py``), each case one schedule of at most ten steps
within a rounding bound fixed before the comparison
(:func:`low_storage_cases.parity_bound`). Here is what is the engine's own: the ledger
and the estimate, which agree to the byte, the varying operator's flows, the load, the
march across thread counts and the refused trial. Every run is the native session's
(SEP 47 phase 5), which owns the loop: a step left in flight and the poisoned session
are ``tests/sessions/test_native_session_low_storage.py``'s, and a signal during a
step, which lands after its native call has committed it,
``tests/backends/test_low_storage.py``'s
``test_an_atomic_engine_commits_its_step_whole[atomic engine]``.
"""

import gc
import math
import weakref
from pathlib import Path

import numpy as np
import pytest

from backends.forced_threads import assert_threaded
from backends.low_storage_cases import (
    parity_bound,
    uniform,
    varying_capillary,
    with_mode,
)
from backends.matched import native_thread_fields
from backends.test_low_storage import CARTESIAN_CHUNK, MARCH, cartesian_march
from bounds import assert_within
from sessions.runs import launch, load_native, run
from solprop.backends.memory import (
    low_storage_flows,
    low_storage_stage_register,
)
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
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


# --- the engine's own --------------------------------------------------------------


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_native_state_holds_the_registers_and_no_runge_kutta_view(mode, collar):
    """Section 7.3 on Kokkos: three or four volumes beside the state, none of DOPRI5's.

    The accumulator, the stage register under an in-stage collar only (Q4), the table
    when tabulating, the state and the transverse scratch:
    four volumes tabulated without a collar and three recomputed, five and four with
    one. The estimate is the ledger to the byte for these rows.
    """
    problem, run_request, grid, _ = cartesian_march(mode, collar=collar)
    field = 16 * math.prod(problem.shape)
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    record: dict[str, object] = {}
    held: list[dict[str, int]] = []

    def at_end(_session, batch):
        # The ledger while the session holds the backend, at its terminal batch.
        if batch.terminal is not None:
            held.append(backend.memory_ledger())

    run(backend, problem, run_request, record=record, between=at_end)
    [ledger] = held
    assert ledger is not None
    volumes = {row: ledger[row] for row in MARCH_ROWS}
    assert volumes == {
        "state": field,
        "transverse_scratch": field,
        "register_accumulator": field,
        "register_stage": field if collar else 0,
        "propagators": field if mode == "tabulate" else 0,
    }
    assert low_storage_stage_register(problem, "native") == collar
    assert low_storage_flows(problem, run_request.stepper, "native") == (
        mode == "tabulate"
    )
    assert sum(volumes.values()) == field * (3 + int(collar) + (mode == "tabulate"))
    for row in (*RUNGE_KUTTA_ROWS, "noise", "adams_history"):
        assert ledger[row] == 0, row
    schedule = run_request.stepper.schedule
    assert schedule is not None
    assert record["propagator_mode"] == mode
    assert record["source_evaluations"] == 4 * schedule.n_steps
    backend.close()


def test_a_varying_operator_holds_its_pair_tabulated_and_none_recomputed():
    """A varying modal operator's ``P`` and ``Q``: two volumes, or formed per element.

    Recomputed, the flows are formed in the finish from the phase terms, so nothing
    of the state's size is held for them: no transient outside the ledger.
    """
    simulation = varying_capillary()
    for mode, held in (("tabulate", 2), ("recompute", 0)):
        problem, run_request = with_mode(
            simulation.translate(), simulation._run_request(), mode
        )
        backend = NativeBackend(
            problem, run_request, grid=simulation.grid, modes=simulation.modes
        )
        field = 16 * math.prod(problem.shape)
        ledger = backend.memory_ledger()
        assert ledger["propagators"] == held * field
        assert low_storage_flows(problem, run_request.stepper, "native") == held
        # The modal route holds C: its input is a padded fine spectrum.
        assert ledger["register_stage"] == field
        assert all(ledger[row] == 0 for row in RUNGE_KUTTA_ROWS)
        backend.close()


@pytest.mark.parametrize("model", [None, "input"], ids=["no noise", "input noise"])
def test_a_loaded_state_is_not_held_after_the_load(monkeypatch, model):
    """The upload copies and fences, so a temporary dies with the load; no ledger row.

    A delay scan's input, the input noise model's sum and a resumed plane are all
    temporaries: each would otherwise stay alive until the next load. The field is read
    back to show that the load took.
    """
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    noise = (
        None
        if model is None
        else spatial_noise(
            SpatialFlux.resolve(
                grid, FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0), SpatialMedium()
            ),
            np.random.default_rng(7),
            model=model,
        )
    )
    problem, run_request, grid, _ = cartesian_march("tabulate", noise=noise)
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    temporary = np.full(problem.shape, 1.0 + 0.5j)
    given = weakref.ref(temporary)
    start = Start(field=temporary, z=0.0, noise=noise)
    del temporary
    session = start_session(backend, problem, run_request, start, owns=False)
    gc.collect()
    assert given() is None
    session.close()
    # A resumed plane, which takes no input draw: the field crosses unchanged.
    loaded_again = np.full(problem.shape, 2.0 + 0.5j)
    again = weakref.ref(loaded_again)
    start = Start(field=loaded_again, z=0.0, noise=noise, resume=True)
    del loaded_again
    observed = launch(backend, problem, run_request, start).field
    gc.collect()
    assert again() is None
    # Exact: the state crosses twice without arithmetic, and the mask keeps or zeroes.
    mask = problem.grids.active_mask
    assert np.all(observed[..., mask] == 2.0 + 0.5j)
    assert np.all(observed[..., ~mask] == 0.0)
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
    assert_threaded(four["sizing"], 4, "kernels.stage", "fft.fft")
    for name in names:
        mode, collar = name.split("-")
        problem, run_request, _, _ = cartesian_march(mode, collar=collar == "collar")
        bound = parity_bound(problem, run_request, transverse=problem.transform.n_nodes)
        assert_within(again[name], one[name], bound, what=f"repeat, {name}")
        assert_within(four[name], one[name], bound, what=f"four threads, {name}")


def test_a_trial_step_on_a_low_storage_state_is_refused_by_name():
    # #492: this state allocates none of the views the trial step writes to.
    problem, run_request, grid, _ = cartesian_march("tabulate")
    backend = NativeBackend(problem, run_request, grid=grid, options=CARTESIAN_CHUNK)
    try:
        state = load_native(backend, problem.initial.field0, clear_fsal=False)
        backend._staged_rows()
        nodes = [float(node) for node in run_request.stepper.tableau.nodes]
        with pytest.raises(RuntimeError, match=r"low-storage state .* no Runge-Kutta"):
            state.rk_trial(0.0, run_request.stepper.dz_init, nodes, prepare_only=False)
    finally:
        backend.close()

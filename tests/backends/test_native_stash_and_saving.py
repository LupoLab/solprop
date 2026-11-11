# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's two savings between steps change no answer (SEP 43, items 22 and 40).

**The stash.** Between planes the plasma statistics are read off what the accepted
step's last stage left behind, because re-synthesising every node costs a seventh of a
run. The Numba tests held that stash against a fresh scan of one state, at the bound
derived for it, :func:`matched.stash_bound`, which had lost its caller. Here the stash
is the last row of the periodic channel, and the fresh scan is a new backend started
from the accepted state, on native and on ``numpy_ref``.

**The recycled stage.** A step's last stage is the next one's first, which saves one
evaluation in seven. The saving against no saving is a comparison of one backend with
itself by design: the feature is what is under test, and ``numpy_ref`` has no such
saving to compare. It is not parity, and the ladder holds that separately. The bound is
the original's, 1e-14 of the peak after 200 fixed steps, and each run's evaluations are
counted, so that the two runs do differ.

**Weaker than the original.** Native reports the largest fraction and rate over a
plane's nodes, not the stash per node. The stash of a mode-averaged node is exact in
this; on the planar route only the largest node is held.
"""

import dataclasses
import itertools
import os
import warnings

import numpy as np
import pytest

from backends.matched import stash_bound
from backends.planar_cases import planar_case
from bounds import assert_within
from sessions.executors import committed_steps
from sessions.runs import load_native
from sessions.runs import run as session_run
from solprop.backends.events import join_attempts
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.gases import Gas
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.stepper.controller import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

NAMES = ("electron_density", "peak_ionisation_rate")


def field_grid():
    """The original's grid for the gradient capillary."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


def ionising_capillary(grid):
    """The original's gradient capillary of argon, ionising, one 1 mm step at a time."""
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, run_request = assemble(
            grid,
            mode,
            [
                KerrResponse(chi3=2.2e-52, thg=True),
                PlasmaResponse(Gas.Ar, table, preionisation=1e-3),
            ],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[2e-3],
            schedule=FixedSchedule.uniform(0.0, 2e-3, 1e-3),
            dz_max=0.01,
            dz_init=1e-4,
        )
    return problem, rows_on_the_runge_kutta_march(run_request)


def rows_on_the_runge_kutta_march(run_request):
    """Return `run_request` on the Runge-Kutta march, with the plasma rows on."""
    return run_request.with_stepper(
        dataclasses.replace(run_request.stepper, integrator=None)
    ).for_run(stats=StatsSpec(names=NAMES, periodic=NAMES, period=1))


def stash_and_fresh(problem, run_request, grid, length):
    """Return the stash's last row and fresh scans of the accepted state on both."""
    assert float(run_request.stepper.save_planes[-1]) == length
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        run = session_run(backend, problem, run_request)
    finally:
        backend.close()
    rows = join_attempts(run.chunks)[1]
    assert rows.n_rows == run_request.stepper.schedule.n_steps
    stashed = {name: np.asarray(rows.values[name])[-1] for name in NAMES}
    accepted = run.fields[-1]
    fresh = {}
    for implementation in (NativeBackend, NumpyRefBackend):
        scanning = implementation(problem, run_request, grid=grid)
        try:
            # A new session has no stash, so its statistics scan afresh: started at
            # the last plane, which it owes, it observes the state there and ends.
            at_end = Start(
                field=accepted,
                z=length,
                plane_index=len(run_request.stepper.save_planes),
            )
            observed = session_run(scanning, problem, run_request, at_end)
            assert [event.z for event in observed.observations] == [length]
            fresh[implementation.name] = observed.all_planes()[-1].stats
        finally:
            scanning.close()
    return stashed, fresh


def assert_stash_is_the_fresh_scan(stashed, fresh, rate, field_range, n_to):
    """Hold the stash within the derived bound of each fresh scan, and of the oracle."""
    peak = float(np.max(fresh["numpy_ref"]["peak_ionisation_rate"]))
    assert peak > 0.0
    bound = stash_bound(rate, peak, field_range, n_to)
    for name in NAMES:
        for side in ("native", "numpy_ref"):
            assert_within(
                stashed[name],
                fresh[side][name],
                bound,
                what=f"{name} against {side}",
            )


def test_the_stashed_plasma_scan_is_the_synthesised_one():
    """The ionising capillary, as the original took it, at ``stash_bound``."""
    grid = field_grid()
    problem, run_request = ionising_capillary(grid)
    stashed, fresh = stash_and_fresh(problem, run_request, grid, 2e-3)
    rate = OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov")
    assert_stash_is_the_fresh_scan(stashed, fresh, rate, (1e9, 2e11), grid.n_to)


@pytest.mark.medium
def test_the_stash_on_the_planar_route_is_the_fresh_scan():
    """The planar field-plasma case: the original's second family, at the same bound."""
    grid, _, problem, run_request = planar_case("field-plasma")
    run_request = rows_on_the_runge_kutta_march(run_request)
    stashed, fresh = stash_and_fresh(problem, run_request, grid, grid.length)
    rate = problem.responses.plasma_coefficients.rate  # pyright: ignore[reportOptionalMemberAccess]
    assert float(np.max(stashed["electron_density"])) > 0.0
    assert_stash_is_the_fresh_scan(
        stashed, fresh, rate, (rate.field_min, rate.field_max), grid.n_to
    )


def step_by_step(backend, problem, run_request, *, recycle):
    """Take a schedule's steps on the backend's native state, the stage recycled or not.

    The native Runge-Kutta march's own step, called on the state directly: a staged
    trial (``rk_trial``), its accept (``finish``), and the last stage promoted to the
    next step's first, or dropped so that the next trial evaluates it afresh. Returns
    the state after the steps, and the steps taken.
    """
    backend._staged_rows()
    state = load_native(backend, problem.initial.field0)
    nodes = [float(node) for node in run_request.stepper.tableau.nodes]
    schedule = run_request.stepper.schedule
    assert schedule is not None
    edges = np.asarray(schedule.edges)
    for z, end in itertools.pairwise(edges):
        dz = float(end - z)
        state.rk_trial(float(z), dz, nodes, prepare_only=False)
        state.finish(dz)
        if recycle:
            state.promote_fsal()
        else:
            state.clear_fsal()
    field = np.asarray(state.field_to_host()).reshape(problem.shape).copy()
    return field, edges.size - 1


def test_the_saving_does_not_change_the_answer():
    """Six evaluations in seven, and the same numbers to the reduction's own round-off.

    The original's case and bound: the gradient capillary over 200 fixed steps to 0.1 m,
    a replayed schedule so that the step count cannot move, and 1e-14 of the peak. The
    Numba run measured 2.9e-16 and 5.8e-16 over 400 steps; native measured 1.0e-15 over
    these 200 on one thread. It finishes in seconds here, which is why it is not
    shortened to ten steps (AGENTS.md). Both runs step the native state directly
    (:func:`step_by_step`): the native session always recycles the stage, so the run
    without it is not one the session can make; the recycled run is then held to the
    session's own.
    """
    grid = field_grid()
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, run_request = assemble(
            grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=1e-4,
            absorber=absorber(grid),
            schedule=FixedSchedule.uniform(0.0, 0.1, 0.0005, save_planes=[0.1]),
        )
    run_request = run_request.with_stepper(
        dataclasses.replace(run_request.stepper, integrator=None)
    )

    def run(recycle):
        backend = NativeBackend(problem, run_request, grid=grid)
        try:
            field, taken = step_by_step(backend, problem, run_request, recycle=recycle)
            return field, taken, backend._native_state.rk_source_evaluations
        finally:
            backend.close()

    with_it, taken, evaluated = run(recycle=True)
    without, alone, unsaved = run(recycle=False)
    assert taken == alone == 200
    stages = run_request.stepper.tableau.n_stages
    assert unsaved == stages * taken
    assert evaluated == (stages - 1) * taken + 1
    peak = float(np.max(np.abs(without)))
    assert float(np.max(np.abs(with_it - without))) / peak < 1e-14
    # The steps above are the native session's own: its run lands on the recycled
    # steps' state, with their evaluations.
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        session = session_run(backend, problem, run_request)
        assert committed_steps(session) == taken
        assert backend._native_state.rk_source_evaluations == evaluated
    finally:
        backend.close()
    if "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }:
        # One thread fixes every reduction's order: the same steps to the bit.
        np.testing.assert_array_equal(session.fields[-1], with_it)
    assert float(np.max(np.abs(session.fields[-1] - with_it))) / peak < 1e-14

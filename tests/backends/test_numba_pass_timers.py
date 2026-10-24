# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s opt-in per-pass timers (SEP 28 phase 0).

Off by default: no timers object exists, so the instrumented passes take the path
they took before. On (``SOLPROP_PASS_TIMERS=1``): every free-space evaluation's passes
and the steppers' traversals around them are charged, with exact call counts. Times
are wall clock and only asserted non-negative; the counts are structural and exact.
"""

import numpy as np
import pytest

from backends.planar_cases import planar_case
from backends.test_dispersive_kerr_routes import ARGON, free_grid, free_problem
from backends.test_envelope_plasma_geometries import envelope_mixture, geometry_case
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.pass_timers import (
    ENVIRONMENT,
    PassTimers,
    PassTotal,
    timers_from_environment,
)
from solprop.boundaries import Absorber
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, PIController
from solprop.translate.problem import assemble

__all__ = ["envelope_mixture"]  # the module fixture, imported for geometry_case

LENGTH = 2e-5


def test_the_timers_are_off_unless_the_variable_is_one():
    assert timers_from_environment({}) is None
    assert timers_from_environment({ENVIRONMENT: ""}) is None
    assert timers_from_environment({ENVIRONMENT: "0"}) is None
    assert isinstance(timers_from_environment({ENVIRONMENT: " 1 "}), PassTimers)
    with pytest.raises(BackendError, match="must be 0 or 1"):
        timers_from_environment({ENVIRONMENT: "yes"})


def test_laps_accumulate_per_name_and_reset_forgets_them():
    ticks = iter([0.0, 1.0, 3.0, 4.0, 10.0, 11.0])
    timers = PassTimers(clock=lambda: next(ticks))
    mark = timers.now()
    mark = timers.lap("b", mark)
    mark = timers.lap("a", mark)
    timers.lap("b", mark)
    assert timers.snapshot() == {"a": PassTotal(1, 2.0), "b": PassTotal(2, 2.0)}
    assert list(timers.snapshot()) == ["a", "b"]
    timers.reset()
    assert timers.snapshot() == {}
    mark = timers.now()
    timers.lap("c", mark)
    assert timers.snapshot() == {"c": PassTotal(1, 1.0)}


def cartesian(integrator, *, collar: bool):
    """Return a small Cartesian envelope Kerr problem of two prescribed steps."""
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 100e-15)
    space = FreeGrid(3e-4, 8, 3e-4, 6, window_factor=0.3 if collar else 0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 8e-5
    )
    # A lossless temporal absorber of a long reference length: the transverse collar's
    # in-stage term is then evaluated beside the material's.
    boundary = (
        Absorber(
            np.zeros(grid.n_omega),
            np.zeros(grid.n_t),
            np.zeros(grid.n_to),
            grid.active_mask,
            5 * LENGTH,
        )
        if collar
        else None
    )
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        absorber=boundary,
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH / 2,
        dz_init=LENGTH / 2,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 2),
        integrator=integrator,
    )
    return problem, grid


def timed(problem, grid, monkeypatch, *, enabled: bool = True):
    """Run a problem to its last plane; return the backend's timers and its flags."""
    if enabled:
        monkeypatch.setenv(ENVIRONMENT, "1")
    else:
        monkeypatch.delenv(ENVIRONMENT, raising=False)
    backend = NumbaCpuBackend(
        problem,
        grid=grid,
        plan_flag="estimate",
        threads=1,
        fftw_threads=1,
        blas_threads=1,
    )
    try:
        timers = backend.pass_timers
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        absorber = backend._in_stage_absorber
        return (None if timers is None else timers.snapshot()), absorber
    finally:
        backend.close()


def calls(snapshot):
    """Return each pass's call count, and check every time is non-negative."""
    assert all(entry.seconds >= 0.0 for entry in snapshot.values())
    return {name: entry.calls for name, entry in snapshot.items()}


def test_without_the_variable_a_backend_holds_no_timers(monkeypatch):
    problem, grid = cartesian(LowStorageRungeKutta(), collar=False)
    snapshot, _ = timed(problem, grid, monkeypatch, enabled=False)
    assert snapshot is None


@pytest.mark.parametrize("collar", [False, True])
def test_the_march_charges_its_input_folds_table_and_core(monkeypatch, collar):
    problem, grid = cartesian(LowStorageRungeKutta(), collar=collar)
    snapshot, absorber = timed(problem, grid, monkeypatch)
    assert absorber is collar
    counted = calls(snapshot)
    chunks = -(-problem.transform.n_nodes // problem.transform.node_chunk)
    # Two steps of four passes; with the collar each pass evaluates the material and
    # the absorber's term through the core and folds twice.
    cores = 8 * (2 if collar else 1)
    assert counted["free.transverse_inverse"] == cores
    assert counted["free.transverse_forward"] == cores
    assert counted["free.pack"] == counted["free.crop"] == cores * chunks
    assert counted["free.response"] == 8 * chunks
    assert counted["ls.fold"] == cores
    assert counted["ls.table"] == 1
    # The stage register is elided without the collar: one input a step.
    assert counted["ls.input"] == (8 if collar else 2)
    assert counted.get("free.absorber", 0) == (8 * chunks if collar else 0)


@pytest.mark.parametrize("collar", [False, True])
def test_the_replay_charges_its_stage_traversals(monkeypatch, collar):
    problem, grid = cartesian(None, collar=collar)
    snapshot, absorber = timed(problem, grid, monkeypatch)
    assert absorber is collar
    counted = calls(snapshot)
    stages = counted["rk.accumulate"]
    assert stages >= 2 * 6
    assert counted["rk.stage_clear"] == stages
    assert counted["rk.input"] == counted["rk.finish"] == stages * (2 if collar else 1)
    assert counted["free.transverse_inverse"] == counted["rk.input"]


def test_the_adams_sample_charges_its_fills_and_copy(monkeypatch):
    problem, grid = cartesian(Multistep(), collar=False)
    counted = calls(timed(problem, grid, monkeypatch)[0])
    assert counted["adams.identity_fill"] == counted["adams.sample_copy"] >= 2
    assert counted["free.transverse_inverse"] >= counted["adams.sample_copy"]


def test_the_envelope_plasma_streams_are_charged(monkeypatch, envelope_mixture):
    problem, grid = geometry_case(envelope_mixture, "free", 1)
    counted = calls(timed(problem, grid, monkeypatch)[0])
    assert counted["free.plasma_streams"] == counted["free.to_frequency"] > 0


def test_the_dispersive_response_is_charged(monkeypatch):
    grid = free_grid("envelope")
    problem = free_problem(DispersiveKerrResponse(ARGON, "polynomial", thg=False), grid)
    counted = calls(timed(problem, grid, monkeypatch)[0])
    assert counted["free.dispersive"] == counted["free.pack"] > 0


@pytest.mark.filterwarnings(
    # The shared planar cases' short window; the timers are what is tested.
    "ignore:the temporal absorber attenuates a packet walking off"
)
@pytest.mark.parametrize("name", ["kerr", "kerr-collar"])
def test_the_planar_core_charges_its_blocks(monkeypatch, name):
    grid, _, problem = planar_case(name, node_chunk=4 * 2)
    snapshot, absorber = timed(problem, grid, monkeypatch)
    assert snapshot is not None
    counted = calls(snapshot)
    assert counted["slab.transverse_inverse"] == counted["slab.transverse_forward"] > 0
    assert counted["slab.pack"] == counted["slab.to_time"] == counted["slab.material"]
    assert counted["slab.output"] == counted["slab.material"]
    assert counted.get("slab.absorber", 0) == (counted["slab.pack"] if absorber else 0)
    assert np.isfinite(sum(entry.seconds for entry in snapshot.values()))


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
def test_the_field_plasma_streams_are_charged(monkeypatch):
    from backends.test_band_plasma_parity import case  # noqa: PLC0415

    grid, problem = case("free")
    counted = calls(timed(problem, grid, monkeypatch)[0])
    assert counted["free.plasma_streams"] == counted["free.to_frequency"] > 0

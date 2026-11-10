# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One translated problem and one backend run for many inputs (SEP 22 section 5.2).

A run on the runner's reused backend repeats a fresh backend's run: on a replayed
schedule it agrees with a fresh ``Simulation.run`` to a rounding bound fixed before the
runs (class R of ``docs/dev/testing.md`` section 3), its Adams call ledger and the four
pieces of state a load restores (section 4.7) are structural and compared exactly.
"""

import copy
import dataclasses
import gc
import math
import threading
import types
import warnings
import weakref
from typing import cast

import numpy as np
import pytest

from bounds import assert_within, fft_bound, step_evaluations
from driver.test_field_free_planes import cartesian, low_storage, replay_bound
from runtime_types import unchecked
from solprop.backends import sessions
from solprop.backends.driver import StepDriver
from solprop.backends.host.driver import HostDriver
from solprop.backends.registry import available, select
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationRunner
from solprop.driver.simulation import check_derived_problem
from solprop.exceptions import DriverError, SolpropWarning, TranslationError
from solprop.log import NullReporter
from solprop.noise import NoiseRealisation, spatial_noise
from solprop.output import MemoryOutput, open_file
from solprop.output import schema as output_schema
from solprop.output.schema import (
    PEAK_FIELDS_ANNOTATION,
    ArrayDigest,
    Output,
    PlaneStore,
    RunMetadata,
)
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialFlux
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta
from solprop.translate.routes import GeneralRoute, ModalRoute, SpatialRoute, TDSESource
from solprop.translate.tdse import TDSEDescriptor

LENGTH = 1e-3
STEPS = 10
SCALES = (0.9, 1.0, 1.1)


def stored(output: Output) -> PlaneStore:
    """Return a standard handler as the store it is, for its typed readers."""
    assert isinstance(output, PlaneStore)
    return output


def bare_tdse() -> TDSESource:
    """Return a TDSE source whose descriptor is a bare record, for a route check.

    A check that reads only whether the route carries a TDSE extension needs no
    atomic descriptor, which a TDSE translation builds at some cost.
    """
    numerator = np.zeros((1, 1, 1))
    numerator.setflags(write=False)
    return TDSESource(
        descriptor=object.__new__(TDSEDescriptor), current_numerator=numerator
    )


def capillary(backend: str, *, dense: bool = False) -> Simulation:
    """Return a cheap two-mode helium capillary of ten replayed Adams steps.

    The collar is spectral-only, so the explicit Adams march has no stiff in-stage
    term to bound its replayed step; a dense run is adaptive.
    """
    simulation = Simulation.from_capillary(
        125e-6,
        LENGTH,
        "He",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_r=8,
        n_save=3,
        rtol=1e-6,
        modes=2,
        plasma=False,
        backend=backend,
        noise=None,
    )
    settings = simulation.settings
    simulation.settings = dataclasses.replace(
        settings,
        absorber=absorber(simulation.grid, temporal=False),
        schedule=None
        if dense
        else FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / STEPS, save_planes=settings.save_planes
        ),
        interpolate_planes=dense,
    )
    return simulation


def inputs(simulation: Simulation) -> list[np.ndarray]:
    """Return three launches: the run's own, a tenth weaker and a tenth stronger."""
    return [scale * simulation._field0 for scale in SCALES]


def bound(simulation: Simulation) -> float:
    """Return the relative rounding bound of a replayed run against another.

    Fixed before the runs: every step carries the rounding of the Adams weights'
    source evaluations (``step_evaluations``), each one a temporal transform pair on
    the fine grid and a quadrature over the nodes, over the ten steps.
    """
    problem = simulation.translate()
    return (
        STEPS
        * step_evaluations(simulation._run_request())
        * (fft_bound(problem.grids.n_to) + fft_bound(problem.transform.n_nodes))
    )


class _Stop(NullReporter):
    """Stop the run, or call a hook, at its first save plane."""

    def __init__(self, hook=None) -> None:
        self.hook = hook

    def plane(self, z, summary) -> None:
        if self.hook is not None:
            self.hook()
        msg = "stopped on purpose"
        raise RuntimeError(msg)


def restored(backend, problem, run_request) -> tuple:
    """Return the four pieces of state a load restores (SEP 22 section 4.7).

    The quadrature alarm as the entrance plane reads it, ``quadrature_error`` before
    any march: in-tree backends keep it on the host, Kokkos in its native state.
    """
    march = backend._adams
    spec = dataclasses.replace(
        run_request.observation.stats, names=("quadrature_error",)
    )
    entrance = float(backend.stats(spec)["quadrature_error"])
    return (
        backend._dense_ahead is None,
        "nan" if np.isnan(entrance) else entrance,
        None if march is None else dataclasses.asdict(march.ledger),
        None if march is None else march._last_rejected,
    )


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
def test_a_runner_repeats_fresh_runs_even_after_a_failed_one(backend):
    simulation = capillary(backend)
    budget = bound(simulation)
    fresh = [
        stored(simulation.with_input(field).run(progress="none"))
        for field in inputs(simulation)
    ]
    with simulation.runner() as runner:
        first = stored(runner.run(inputs(simulation)[0], progress="none"))
        # A run stopped part-way leaves its march, its ledger and its alarm behind.
        with pytest.raises(RuntimeError, match="on purpose"):
            runner.run(inputs(simulation)[1], progress=_Stop())
        assert not runner.closed
        reused = [first] + [
            stored(runner.run(field, progress="none"))
            for field in inputs(simulation)[1:]
        ]
        assert runner.runs == 3
    assert runner.closed
    for index, (one, other) in enumerate(zip(fresh, reused, strict=True)):
        assert_within(other["field"], one["field"], budget, what=f"field {index}")
        for name in ("energy", "peak_power"):
            assert_within(
                other[f"stats/{name}"], one[f"stats/{name}"], 2 * budget, what=name
            )
        # The entrance is read before any march: no alarm of an earlier run there.
        quadrature = (one["stats/quadrature_error"], other["stats/quadrature_error"])
        assert np.isnan(quadrature[0][0]) and np.isnan(quadrature[1][0])
        assert_within(quadrature[1][1:], quadrature[0][1:], 2 * budget)
        # Integer accounting of a replay: exact.
        assert one.outcome is not None and other.outcome is not None
        assert other.outcome.logical_calls == one.outcome.logical_calls
        assert np.array_equal(other["step_history/h"], one["step_history/h"])
        assert other.metadata is not None and one.metadata is not None
        assert other.metadata.configuration_hash() == one.metadata.configuration_hash()
    hashes = {
        cast("RunMetadata", output.metadata).configuration_hash() for output in reused
    }
    assert len(hashes) == len(SCALES)


def test_a_runner_counts_each_run_s_low_storage_evaluations_alone():
    simulation = low_storage()
    fresh = stored(simulation.run(progress="none"))
    assert fresh.outcome is not None
    expected = fresh.outcome.execution["source_evaluations"]
    # Four physical evaluations a committed step: an integer contract, exact.
    assert expected == 4 * len(fresh["step_history/h"])
    with simulation.runner() as runner:
        for scale in SCALES:
            output = stored(runner.run(scale * simulation._field0, progress="none"))
            assert output.outcome is not None
            assert output.outcome.execution["source_evaluations"] == expected


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "replayed Adams"])
def test_a_load_restores_what_a_failed_run_left_as_a_fresh_backend_holds(
    backend, dense, monkeypatch
):
    served: list[str] = []
    start = sessions.start_session

    def serving(*args, **keywords):
        session = start(*args, **keywords)
        served.append(type(session).__name__)
        return session

    monkeypatch.setattr(sessions, "start_session", serving)
    simulation = capillary(backend, dense=dense)
    if dense:
        # The Runge-Kutta march observes densely through the backend's look-ahead.
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=RungeKutta()
        )
    problem = simulation.translate()
    run_request = simulation._run_request()
    with simulation.runner() as runner:
        with pytest.raises(RuntimeError, match="on purpose"):
            runner.run(inputs(simulation)[0], progress=_Stop())
        left = cast("StepDriver | HostDriver", runner.backend)
        if dense:
            # The failed run held the march ahead of its interpolated plane; the
            # native session holds that look-ahead itself, and the backend's is unused.
            assert left._dense_ahead is not None or served == ["NativeSession"]
        else:
            assert left._adams is not None
            # The native session keeps the march's ledger itself and leaves the
            # backend's untouched; the transitional session's march books into it.
            assert left._adams.ledger.total > 0 or served == ["NativeSession"]
            # A failed run can stop just after a rejection; the load clears it.
            left._adams._last_rejected = True
        left.set_initial(problem.initial.field0, None, z=problem.initial.z)
        fresh = select(
            backend,
            problem,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            fresh.set_initial(problem.initial.field0, None, z=problem.initial.z)
            # Restored state is exact: the same flags, alarm and integer ledger.
            assert restored(left, problem, run_request) == restored(
                fresh, problem, run_request
            )
        finally:
            fresh.close()
        after = stored(runner.run(inputs(simulation)[1], progress="none"))
    assert after.n_saved == len(simulation.settings.save_planes) + 1


def test_a_runner_refuses_what_its_backend_cannot_carry(tmp_path):
    simulation = capillary("numpy_ref")
    with SimulationRunner(simulation) as runner:
        # Into a file, with the caller's key in its record.
        path = tmp_path / "run.h5"
        output = stored(
            runner.run(
                inputs(simulation)[0], output=path, configuration={"delay": 1e-15}
            )
        )
        assert output.closed
        with open_file(path) as written:
            assert written.metadata is not None
            assert written.metadata.configuration["delay"] == 1e-15
        # One input at a time: a run started from inside another is refused.
        with pytest.raises(DriverError, match="already running an input"):
            runner.run(
                inputs(simulation)[1],
                progress=_Stop(lambda: runner.run(inputs(simulation)[2])),
            )
        assert not runner.closed
        # Another noise model is one the backend would drop or misapply.
        realisation = NoiseRealisation(
            spectrum=np.zeros_like(inputs(simulation)[0]), seed=1, model="input"
        )
        with pytest.raises(DriverError, match="noise model 'input'"):
            runner.run(inputs(simulation)[0], noise=realisation)
        assert "runs" in repr(runner) and "open" in str(runner)
    with pytest.raises(DriverError, match="runner is closed"):
        runner.run(inputs(simulation)[0])
    runner.close()  # idempotent


def test_a_runner_closes_only_between_runs():
    simulation = cartesian("numpy_ref")
    runner = simulation.runner()
    # A run in progress on another thread holds the runner's lock; the close waits.
    assert runner._lock.acquire(blocking=False)
    closer = threading.Thread(target=runner.close)
    closer.start()
    closer.join(timeout=0.2)
    assert closer.is_alive() and not runner.closed
    runner._lock.release()
    closer.join()
    assert runner.closed
    with pytest.raises(DriverError, match="runner is closed"):
        runner.run(simulation._field0)


def test_a_backend_that_closed_itself_closes_its_runner():
    simulation = capillary("numpy_ref")
    runner = simulation.runner()
    with pytest.raises(RuntimeError, match="on purpose"):
        runner.run(inputs(simulation)[0], progress=_Stop(runner.backend.close))
    assert runner.closed


def test_a_tdse_run_has_no_runner(monkeypatch):
    simulation = capillary("numpy_ref")
    route = dataclasses.replace(simulation.translate().route, tdse=bare_tdse())
    monkeypatch.setattr(
        simulation, "translate", lambda: types.SimpleNamespace(route=route)
    )
    with pytest.raises(DriverError, match="TDSE response"):
        simulation.runner()


def test_a_derived_problem_is_the_constructed_one_but_for_its_input():
    simulation = capillary("numpy_ref")
    problem = simulation.translate()
    run_request = simulation._run_request()
    derived = simulation.with_input(inputs(simulation)[1])
    check_derived_problem(
        problem, derived.translate(), run_request, derived._run_request()
    )
    replaced = dataclasses.replace(problem, grids=dataclasses.replace(problem.grids))
    with pytest.raises(DriverError, match="share the runner's grids"):
        check_derived_problem(problem, replaced, run_request, run_request)
    moved = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, z=1e-6)
    )
    with pytest.raises(DriverError, match="different initial_z"):
        check_derived_problem(problem, moved, run_request, run_request)
    noisy = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial,
            noise0=np.zeros_like(problem.initial.field0),
            noise_model="input",
        ),
    )
    with pytest.raises(DriverError, match="noise model 'input'"):
        check_derived_problem(problem, noisy, run_request, run_request)
    stepped = run_request.with_stepper(
        dataclasses.replace(run_request.stepper, dz_max=run_request.stepper.dz_max / 2)
    )
    with pytest.raises(DriverError, match=r"stepper\.dz_max differs"):
        check_derived_problem(problem, problem, run_request, stepped)
    # The statistics are the request's, shared with the derived run like the leaves.
    counted = run_request.for_run(stats=StatsSpec(names=("energy",)))
    with pytest.raises(DriverError, match="share the runner's stats_spec"):
        check_derived_problem(problem, problem, run_request, counted)


def test_a_derived_problem_on_another_route_is_refused():
    # The transform is the same object, so only the route's variant, TDSE source or
    # plane measure can differ: each is the constructed problem's or refused.
    simulation = capillary("numpy_ref")
    problem, run_request = simulation.translate(), simulation._run_request()
    check_derived_problem(
        problem,
        dataclasses.replace(
            problem, route=ModalRoute(transform=problem.transform, tdse=None)
        ),
        run_request,
        run_request,
    )
    # A general modal route needs tables this transform lacks: built past its check.
    general = object.__new__(GeneralRoute)
    object.__setattr__(general, "transform", problem.transform)
    other = copy.copy(problem)
    object.__setattr__(other, "route", general)
    with pytest.raises(DriverError, match="does not share the runner's route"):
        check_derived_problem(problem, other, run_request, run_request)
    attached = copy.copy(problem)
    object.__setattr__(
        attached, "route", dataclasses.replace(problem.route, tdse=bare_tdse())
    )
    with pytest.raises(DriverError, match="does not share the runner's route"):
        check_derived_problem(problem, attached, run_request, run_request)
    planar = cartesian("numpy_ref")
    spatial, spatial_request = planar.translate(), planar._run_request()
    assert isinstance(spatial.route, SpatialRoute)
    remeasured = dataclasses.replace(
        spatial,
        route=dataclasses.replace(spatial.route, flux=copy.copy(spatial.route.flux)),
    )
    with pytest.raises(DriverError, match="does not share the runner's route"):
        check_derived_problem(spatial, remeasured, spatial_request, spatial_request)


def test_with_input_validates_the_replaced_input_alone():
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0]
    with pytest.raises(DriverError, match="finite initial field"):
        simulation.with_input(np.full_like(field, np.nan))
    # One sample settles it, and finite samples too large to add do not.
    one = field.copy()
    one.flat[one.size // 2] = complex(np.inf, 0.0)
    with pytest.raises(DriverError, match="finite initial field"):
        simulation.with_input(one)
    simulation.with_input(np.full_like(field, 1e308))
    with pytest.raises(DriverError, match="existing mode and grid shape"):
        simulation.with_input(field[:1])
    with pytest.raises(DriverError, match="recorded by the run itself"):
        simulation.with_input(field, configuration={"initial": 1.0})
    derived = simulation.with_input(field, configuration={"label": "a"})
    assert derived.metadata().configuration["label"] == "a"
    other = simulation.with_input(field, configuration={"label": "b"})
    assert derived.metadata().configuration_hash() != (
        other.metadata().configuration_hash()
    )
    # Every leaf but the input, the stepper and the flags is shared, unscanned.
    check_derived_problem(
        simulation.translate(),
        derived.translate(),
        simulation._run_request(),
        derived._run_request(),
    )


def test_a_large_array_input_is_recorded_by_its_digest(monkeypatch):
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0]
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", field.size - 1)
    one = simulation.with_input(field).metadata()
    recorded = one.configuration["initial"]
    assert isinstance(recorded, ArrayDigest)
    assert recorded.shape == field.shape
    changed = field.copy()
    changed[0, 0, 7] *= 1 + 1e-15
    two = simulation.with_input(changed).metadata()
    assert two.configuration["initial"] != recorded
    assert two.configuration_hash() != one.configuration_hash()
    # Below the limit an array is recorded inline, as before.
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", field.size)
    inline = simulation.with_input(field).metadata().configuration["initial"]
    assert isinstance(inline, np.ndarray)


class _Watching(MemoryOutput):
    """Note at every save whether each watched array is still alive."""

    def __init__(self, *watched: weakref.ref[np.ndarray]) -> None:
        super().__init__()
        self.watched = watched
        self.alive: list[bool] = []

    def save(self, plane) -> None:
        self.alive.extend(ref() is not None for ref in self.watched)
        super().save(plane)


@pytest.fixture
def no_collector():
    """Hold the cycle collector off: a volume must go by its count, not by a pass."""
    gc.disable()
    yield
    gc.enable()


def _spied_load(backend, monkeypatch) -> list[weakref.ref[np.ndarray]]:
    """Return a list that keeps a weak reference to each array the backend loads."""
    loads: list[weakref.ref[np.ndarray]] = []
    load = backend.set_initial

    def spy(field, noise, *, z=0.0, resume=False) -> None:
        loads.append(weakref.ref(field))
        load(field, noise, z=z, resume=resume)

    monkeypatch.setattr(backend, "set_initial", spy)
    return loads


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("route", ["modal", "spatial"])
@pytest.mark.parametrize("digested", [True, False], ids=["digest", "inline"])
@pytest.mark.usefixtures("no_collector")
def test_a_runner_holds_no_input_through_the_march(
    backend, route, digested, monkeypatch
):
    """Neither the caller's array, nor a copy, nor a launch outlives the load.

    Called past the runtime type checks' wrappers, as production calls it: their frames
    keep the caller's arguments for the call, the runner's own hold nothing.
    """
    simulation = capillary(backend) if route == "modal" else cartesian(backend)
    limit = 0 if digested else simulation._field0.size
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", limit)
    run = unchecked(SimulationRunner.run)
    with simulation.runner() as runner:
        loads = _spied_load(runner.backend, monkeypatch)
        field = np.array(simulation._field0)
        given = weakref.ref(field)
        handler = _Watching(given)
        # The call takes the only reference: the name is gone before the body runs.
        arguments = [field]
        del field
        output = run(runner, arguments.pop(), output=handler, progress="none")
        assert output is handler
        assert handler.n_saved > 1
        assert handler.alive == [False] * handler.n_saved
        # An inline record keeps the input it ran, which on the spatial route is the
        # very array loaded: once the record goes, nothing else holds the launch.
        del output, handler
        assert loads[0]() is None


@pytest.mark.parametrize("backend", available())
def test_a_runner_reads_a_digested_input_without_a_copy(backend, monkeypatch):
    """The derived simulation views the caller's array, which it leaves as it was."""
    simulation = cartesian(backend)
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", 0)
    field = 0.9 * simulation._field0
    before = field.copy()
    with simulation.runner() as runner:
        load = runner.backend.set_initial
        shared: list[bool] = []

        def spy(given, noise, *, z=0.0, resume=False) -> None:
            shared.append(np.shares_memory(given, field))
            load(given, noise, z=z, resume=resume)

        monkeypatch.setattr(runner.backend, "set_initial", spy)
        output = stored(runner.run(field, progress="none"))
        fresh = stored(simulation.with_input(field).run(progress="none"))
    assert shared == [True]
    assert field.flags.writeable
    assert np.array_equal(field, before)
    # One record with a fresh run, whichever route the input took to the load.
    assert output.metadata is not None and fresh.metadata is not None
    assert output.metadata.configuration_hash() == fresh.metadata.configuration_hash()
    assert_within(output["field"], fresh["field"], replay_bound(simulation))


def test_a_runner_s_inline_record_outlives_the_caller_s_changes():
    """An array small enough to be recorded inline is copied: the record stays."""
    simulation = cartesian("numpy_ref")
    field = 0.9 * simulation._field0
    before = field.copy()
    with simulation.runner() as runner:
        output = stored(runner.run(field, progress="none"))
    field *= 2.0
    assert output.metadata is not None
    recorded = cast("np.ndarray", output.metadata.configuration["initial"])
    assert np.array_equal(recorded, before)


@pytest.mark.parametrize("route", ["modal", "spatial"])
def test_a_resident_adams_march_holds_no_host_copy_of_the_state(route):
    """The native engine holds the state; a stopped run's report reads it on demand."""
    pytest.importorskip("solprop_native")
    simulation = capillary("native") if route == "modal" else cartesian("native")
    if route == "spatial":
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=Multistep()
        )
    with simulation.runner() as runner:
        backend = cast("StepDriver", runner.backend)
        assert getattr(backend, "_resident", None) is not None
        output = stored(runner.run(np.array(simulation._field0), progress="none"))
        assert getattr(backend, "_host_state", ...) is None
        assert runner.backend.memory_ledger()["adams_host_state"] == 0
        # What a budget stop reports is the engine's: the state, read when it is asked.
        assert np.array_equal(
            backend._adams_state(), backend._observed_field(backend._require_state())
        )
    assert output.n_saved > 1
    assert getattr(backend, "_host_state", ...) is None


def test_a_host_adams_march_lists_its_copy_of_the_state_and_closes_it(monkeypatch):
    """Solprop's host engine reads the state on the host, one volume in the ledger.

    A coupled flow the host engine steps (``_flow_on_host``: on a device build off
    the eigenbasis route) is what takes that engine
    (``_adams_engine``); the test asks for it on any build.
    """
    pytest.importorskip("solprop_native")
    flow_parity = pytest.importorskip("flow_parity")
    monkeypatch.setattr(
        "solprop_native.backend._flow_on_host", lambda *_: "the test's host engine"
    )
    with warnings.catch_warnings():
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build("scalar", schedule=None)
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=Multistep(), backend="native"
        )
        problem = simulation.translate()
        run_request = simulation._run_request()
        backend = select(
            "native", problem, run_request, grid=simulation.grid, modes=simulation.modes
        )
    try:
        assert getattr(backend, "_resident", ...) is None
        assert backend.memory_ledger()["adams_host_state"] == 0
        backend.set_initial(problem.initial.field0, None)
        volume = 16 * math.prod(problem.shape)
        assert backend.memory_ledger()["adams_host_state"] == volume
    finally:
        backend.close()
    assert getattr(backend, "_host_state", ...) is None


@pytest.mark.medium
def test_a_runner_carries_a_replaced_realisation_of_its_noise_model():
    simulation = cartesian("numpy_ref", noise=True)
    grid, space = simulation.grid, simulation.spatial_grid
    assert space is not None
    field = simulation._field0
    assert simulation.medium is not None
    other = spatial_noise(
        SpatialFlux.resolve(grid, space, simulation.medium), np.random.default_rng(11)
    )
    fresh = stored(simulation.with_input(field, noise=other).run(progress="none"))
    with simulation.runner() as runner:
        reused = stored(runner.run(field, noise=other, progress="none"))
        kept = stored(runner.run(field, progress="none"))
    assert_within(reused["field"], fresh["field"], replay_bound(simulation))
    # The kept realisation is the constructed one: another draw, another field.
    assert not np.allclose(kept["field"], fresh["field"])


def test_a_runner_runs_an_input_without_noise_when_it_has_none():
    simulation = capillary("numpy_ref")
    with simulation.runner() as runner:
        output = stored(runner.run(inputs(simulation)[0], noise=None, progress="none"))
    assert output.n_saved == len(simulation.settings.save_planes) + 1


@pytest.mark.medium
def test_a_runner_annotates_each_run_with_its_rates_peak_fields():
    simulation = Simulation.from_capillary(
        125e-6,
        LENGTH,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_save=2,
        rtol=1e-6,
        backend="numpy_ref",
        noise=None,
    )
    with simulation.runner() as runner:
        output = stored(runner.run(simulation._field0, progress="none"))
    assert PEAK_FIELDS_ANNOTATION in output.annotations


def test_with_input_refuses_a_non_finite_realisation_and_keeps_a_pulse_recipe():
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0]
    broken = NoiseRealisation(
        spectrum=np.full_like(field, np.nan), seed=1, model="input"
    )
    with pytest.raises(DriverError, match="finite noise realisation"):
        simulation.with_input(field, noise=broken)
    single = Simulation.from_capillary(
        125e-6,
        LENGTH,
        "He",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_save=2,
        plasma=False,
        backend="numpy_ref",
        noise=None,
    )
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=10e-6)
    derived = single.with_input(pulse)
    assert derived.metadata().configuration["pulse"] is pulse
    # A new launch keeps the stepper's checks: the launch precedes every plane.
    problem = single.translate()
    run_request = single._run_request()
    late = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, z=float(run_request.stepper.save_planes[0])
        ),
    )
    with pytest.raises(TranslationError, match="follow the initial position"):
        late.validate_launch(run_request)
    replayed = cartesian("numpy_ref")
    replayed.translate().validate_launch(replayed._run_request())  # a schedule

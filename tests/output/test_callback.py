# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A handler built from a callable: every run reaches it, and it keeps nothing.

Three things are held here. A run on either backend and any march delivers every plane
as that run made it. The handler costs no volume: what it allocates itself is measured
with ``tracemalloc``, which sees NumPy's allocations and so a copy on any backend, and
how long a plane lives with weak references, which also see the native backend's
observation that the tracer cannot. And what it cannot do it refuses by name.
"""

import dataclasses
import gc
import logging
import threading
import tracemalloc
import weakref
from collections.abc import Callable

import numpy as np
import pytest

from bounds import assert_within
from driver.test_field_free_planes import cartesian, keeping, replay_bound
from driver.test_plane_adoption import CASES
from driver.test_runner import bound, capillary, inputs
from runtime_types import unchecked
from solprop import capabilities, sol
from solprop.backends.protocol import Observation
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.driver.run import run_problem
from solprop.exceptions import DriverError, OutputError
from solprop.output import (
    CallbackOutput,
    MemoryOutput,
    check_cache,
    resumability,
    resume,
)
from solprop.output import schema as output_schema
from solprop.output.schema import (
    PEAK_FIELDS_ANNOTATION,
    CheckpointOutput,
    FieldRetention,
    Output,
    PlaneStore,
    RetainingOutput,
    SavedPlane,
    _example_metadata,
)
from solprop.plotting.style import field_modes
from solprop.processing import energy
from solprop.stats import StatsSpec, default_spec

# These cases run coarse grids for speed; their windows are not what is tested here.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

KEYS = (
    "field",
    "z",
    "noise_phase",
    "field_planes",
    "stats/energy",
    "step_history/z",
    "step_stats/z",
    "not a key",
)


def nothing(plane: SavedPlane) -> None:
    """Take a plane and let it go."""


def opened() -> CallbackOutput:
    """Return a handler with a run begun, for what needs one."""
    handler = CallbackOutput(nothing)
    handler.open(_example_metadata())
    return handler


def field_free_plane() -> SavedPlane:
    """Return the smallest plane a handler can be given."""
    return SavedPlane(z=0.0, field=None)


def fibre(backend: str, stats: StatsSpec | None = None) -> Simulation:
    """Return a short soliton of three planes, on the route that reads a run back."""
    return Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1.2e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=3,
        rtol=1e-6,
        backend=backend,
        stats=stats,
    )


@pytest.fixture
def no_collector():
    """Hold the cycle collector off: a volume must go by its count, not by a pass."""
    gc.disable()
    yield
    gc.enable()


# ------------------------------------------------------------- what it declares


def test_it_declares_only_what_it_can_do():
    handler = CallbackOutput(nothing)
    assert isinstance(handler, Output)
    assert isinstance(handler, RetainingOutput)
    # A checkpoint output promises a state that resume=True continues.
    assert not isinstance(handler, CheckpointOutput)
    assert not isinstance(handler, PlaneStore)
    assert sol.CallbackOutput is CallbackOutput


def test_a_handler_needs_a_callable():
    handler = CallbackOutput.__new__(CallbackOutput)
    with pytest.raises(OutputError, match="needs a callable"):
        unchecked(CallbackOutput.__init__)(handler, 3)


def test_it_holds_the_record_and_no_statistic():
    handler = CallbackOutput(nothing)
    assert handler.metadata is None
    metadata = _example_metadata()
    handler.open(metadata)
    handler.retain(FieldRetention("none"), (1, 1, 8))
    handler.save(SavedPlane(z=0.0, field=None, stats={"energy": np.ones(1)}))
    assert handler.metadata is metadata
    assert handler.n_saved == 1
    assert handler.stat_names == ()
    assert handler.step_stat_names == ()
    assert repr(handler) == "CallbackOutput(1 planes, open)"
    handler.close()
    handler.close()
    assert repr(handler) == "CallbackOutput(1 planes, closed)"


def test_the_callable_is_given_the_plane_itself_and_only_then_is_it_counted():
    given = SavedPlane(z=0.5, field=np.ones((1, 1, 4), dtype=np.complex128))
    received: list[tuple[SavedPlane, int]] = []
    handler = CallbackOutput(lambda plane: received.append((plane, handler.n_saved)))
    handler.open(_example_metadata())
    handler.save(given)
    assert received[0][0] is given
    assert received[0][1] == 0
    assert handler.n_saved == 1


def test_a_plane_needs_a_run_open():
    handler = CallbackOutput(nothing)
    with pytest.raises(OutputError, match=r"call open\(metadata\)"):
        handler.save(field_free_plane())
    handler.open(_example_metadata())
    handler.close()
    with pytest.raises(OutputError, match="is closed"):
        handler.save(field_free_plane())


def test_a_different_record_is_refused_unless_forced():
    handler = opened()
    other = dataclasses.replace(_example_metadata(), configuration={"gamma": 0.2})
    with pytest.raises(OutputError, match="different run"):
        handler.open(other)
    handler.open(other, force=True)
    assert handler.metadata is other


# ------------------------------------------------------------------ what it runs


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("case", CASES)
@pytest.mark.usefixtures("no_collector")
def test_every_march_delivers_its_planes_and_keeps_none_alive(backend, case):
    """Fields on the first two planes and none after: one run of each kind of plane.

    At each plane the plane before it must be gone, so that one observation lives at a
    time. This is the volume the native backend's observation costs, which it allocates
    outside the tracer's sight, and it would go unseen if the run or the handler held
    on.
    """
    simulation = keeping(CASES[case](backend), "planes", (0, 1))
    earlier: list[weakref.ref] = []
    kept: list[bool] = []
    positions: list[float] = []
    alive: list[bool] = []

    def receive(plane: SavedPlane) -> None:
        alive.extend(reference() is not None for reference in earlier)
        earlier.clear()
        earlier.append(weakref.ref(plane))
        if plane.field is not None:
            earlier.append(weakref.ref(plane.field))
        kept.append(plane.field is not None)
        positions.append(plane.z)
        assert (plane.noise_phase is None) or (plane.field is not None)
        assert plane.stats

    handler = CallbackOutput(receive)
    assert simulation.run(handler, progress="none") is handler
    planes = simulation.settings.save_planes
    assert handler.n_saved == len(kept) == len(planes) + 1
    assert kept == [index < 2 for index in range(len(kept))]
    # The positions are the save planes, landed on within a relative rounding.
    np.testing.assert_allclose(positions, [0.0, *planes], rtol=1e-12)
    assert alive
    assert not any(alive)


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize(
    ("fields", "carries"), [("all", True), ("none", False)], ids=["all", "none"]
)
def test_the_fields_setting_says_what_every_plane_carries(backend, fields, carries):
    simulation = keeping(cartesian(backend, noise=True), fields)
    planes: list[SavedPlane] = []
    simulation.run(CallbackOutput(planes.append), progress="none")
    assert planes
    for plane in planes:
        assert (plane.field is not None) is carries
        assert (plane.noise_phase is not None) is carries
        assert set(plane.stats) == set(default_spec("free").names)


@pytest.mark.parametrize("backend", available())
def test_the_planes_of_natural_steps_reach_the_callable(backend):
    simulation = cartesian(backend, replay=False)
    end = simulation.settings.save_planes[-1:]
    simulation.settings = dataclasses.replace(
        simulation.settings, step_planes=2, save_planes=end
    )
    planes: list[SavedPlane] = []
    handler = simulation.run(CallbackOutput(planes.append), progress="none")
    # More planes than were asked for: the entrance, the end and the steps' own.
    assert handler.n_saved == len(planes) > 2
    assert planes[-1].z == pytest.approx(end[0], rel=1e-12)


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("fields", ["all", "none"])
def test_a_periodic_subset_reaches_the_callable(backend, fields):
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=1)
    simulation = keeping(fibre(backend, spec), fields)
    planes: list[SavedPlane] = []
    handler = simulation.run(CallbackOutput(planes.append), progress="none")
    assert planes[0].step_stats.n_rows == 0
    for plane in planes[1:]:
        assert plane.step_stats.names == ("energy",)
        assert plane.step_stats.n_rows > 0
    assert handler.step_stat_names == ()


@pytest.mark.parametrize("backend", available())
def test_a_field_free_scan_takes_no_field_from_the_backend(backend, monkeypatch):
    simulation = keeping(cartesian(backend), "none")
    asked: list[bool] = []
    with simulation.runner() as runner:
        observe = runner.backend.observe

        def watched(*, field: bool = True) -> Observation:
            asked.append(field)
            return observe(field=field)

        monkeypatch.setattr(runner.backend, "observe", watched)
        runner.run(simulation._field0, output=CallbackOutput(nothing), progress="none")
    assert asked
    assert not any(asked)


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize(
    ("build", "rounding"),
    [(capillary, bound), (cartesian, replay_bound)],
    ids=["modal", "spatial"],
)
def test_a_runner_hands_over_what_a_store_would_hold(backend, build, rounding):
    """The same input on one runner, into a callable and a store, on a replayed march.

    The field the callable is given is the store's in the FFT convention, which is
    what says it arrives unconverted; the two agree within the rounding bound of two
    replayed runs, fixed before them, and a statistic quadratic in it within twice that.
    """
    simulation = build(backend)
    limit = rounding(simulation)
    with simulation.runner() as runner:
        for launch in inputs(simulation)[:2]:
            planes: list[SavedPlane] = []
            handler = runner.run(
                launch, output=CallbackOutput(planes.append), progress="none"
            )
            store = runner.run(launch, progress="none")
            assert isinstance(store, MemoryOutput)
            assert handler.n_saved == store.n_saved == len(planes)
            assert not handler.stat_names
            np.testing.assert_array_equal([p.z for p in planes], store["z"])
            for index, plane in enumerate(planes):
                assert plane.field is not None
                assert not plane.field.flags.writeable
                assert_within(
                    plane.field, store.fields("fft")[index], limit, what="field"
                )
                energies = plane.stats["energy"]
                assert_within(
                    energies, store["stats/energy"][index], 2 * limit, what="energy"
                )
    assert runner.runs == 4


# ------------------------------------------------------------------- what it costs


class _Measured(CallbackOutput):
    """Note what each save allocates and what it leaves allocated, beside the tracer."""

    def __init__(self, receive: Callable[[SavedPlane], None]) -> None:
        super().__init__(receive)
        self.peaks: list[int] = []
        self.left: list[int] = []

    def save(self, plane: SavedPlane) -> None:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        super().save(plane)
        after, peak = tracemalloc.get_traced_memory()
        self.peaks.append(peak - before)
        self.left.append(after - before)


@pytest.fixture
def traced():
    """Trace allocations for one test."""
    tracemalloc.start()
    yield
    tracemalloc.stop()


@pytest.mark.parametrize("backend", available())
@pytest.mark.usefixtures("traced")
def test_saving_a_plane_allocates_no_volume(backend):
    """Whatever the run does, ``save`` itself copies and converts nothing.

    A copy of the field or its conversion to the optics convention is a whole volume.
    What ``save`` should allocate is a handful of Python objects, so a tenth of a
    volume, tens of kilobytes on this grid, leaves room for them and none for an array.
    """
    simulation = cartesian(backend)
    volume = simulation._field0.nbytes
    handler = _Measured(nothing)
    simulation.run(handler, progress="none")
    assert handler.n_saved == len(simulation.settings.save_planes) + 1
    assert max(handler.peaks) < volume / 10
    assert max(handler.left) < volume / 10


def _live_levels_at_planes(
    simulation: Simulation, receive: Callable[[SavedPlane], None]
) -> list[int]:
    """Return the traced memory alive at each plane of a second run, in bytes.

    The first run on the runner settles what a backend and the run path build once. Each
    reading follows a collection, because what is measured is what stays alive: a run
    leaves cycles for the collector, and a pass may free an earlier run's at any moment.
    """
    levels: list[int] = []

    def noting(plane: SavedPlane) -> None:
        receive(plane)
        gc.collect()
        levels.append(tracemalloc.get_traced_memory()[0])

    with simulation.runner() as runner:
        runner.run(simulation._field0, output=CallbackOutput(nothing), progress="none")
        runner.run(simulation._field0, output=CallbackOutput(noting), progress="none")
    return levels


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.usefixtures("traced")
def test_a_reducing_run_keeps_no_more_alive_as_it_goes(backend, monkeypatch):
    """From the second plane to the last the live level stays under one volume.

    From the second, because the planes before it may still be building buffers. A run
    keeps about a kilobyte per plane, and this session's runtime type checks some tens
    more in their caches, on a volume of hundreds. Keeping a copy of every field is the
    control, which shows that the measure can see a volume at all: it adds one per
    plane, so three here, of which the bound takes a tenth off for what a pass frees.
    """
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", 0)
    volume = cartesian(backend)._field0.nbytes
    copies: list[np.ndarray] = []

    def keep_a_copy(plane: SavedPlane) -> None:
        assert plane.field is not None
        copies.append(plane.field.copy())

    reducing = _live_levels_at_planes(cartesian(backend), nothing)
    copying = _live_levels_at_planes(cartesian(backend), keep_a_copy)
    planes = len(reducing)
    assert planes >= 4
    assert reducing[-1] - reducing[1] < volume
    assert copying[-1] - copying[1] > 0.9 * (planes - 2) * volume


@pytest.mark.parametrize("backend", available())
def test_the_callable_holds_the_observation_itself(backend, monkeypatch):
    """No copy on the way to the callable, on a route with no conversion of units."""
    simulation = cartesian(backend)
    observed: list[Observation] = []
    shared: list[bool] = []

    def receive(plane: SavedPlane) -> None:
        assert plane.field is not None
        shared.append(np.shares_memory(plane.field, observed[-1].field))

    with simulation.runner() as runner:
        observe = runner.backend.observe

        def watched(*, field: bool = True) -> Observation:
            observation = observe(field=field)
            observed.append(observation)
            return observation

        monkeypatch.setattr(runner.backend, "observe", watched)
        runner.run(simulation._field0, output=CallbackOutput(receive), progress="none")
    assert shared
    assert all(shared)


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("case", ["runge-kutta replayed", "adams modal", "low-storage"])
def test_what_the_callable_keeps_stays_as_it_was(backend, case):
    """A reduction may keep a plane's statistics or history without copying them."""
    kept: list[tuple[np.ndarray, np.ndarray]] = []

    def receive(plane: SavedPlane) -> None:
        arrays = [*plane.stats.values(), plane.step_history.z, plane.step_history.h]
        kept.extend((values, values.copy()) for values in arrays)

    CASES[case](backend).run(CallbackOutput(receive), progress="none")
    assert kept
    for held, as_it_arrived in kept:
        assert np.array_equal(held, as_it_arrived, equal_nan=True)


# ---------------------------------------------------------------- what it refuses


@pytest.mark.parametrize("key", KEYS)
def test_reading_a_key_is_refused_by_name(key):
    handler = opened()
    handler.save(field_free_plane())
    with pytest.raises(OutputError, match=rf"does not hold '{key}'.*keeps nothing"):
        handler[key]
    assert key not in handler


@pytest.mark.parametrize("convention", ["optics", "fft"])
def test_reading_the_fields_is_refused_by_name(convention):
    with pytest.raises(OutputError, match="does not hold the fields"):
        opened().fields(convention)


def test_what_reads_a_run_is_refused_by_name():
    handler = fibre("numpy_ref").run(CallbackOutput(nothing), progress="none")
    assert handler.n_saved == 3
    with pytest.raises(OutputError, match="does not hold 'field_planes'"):
        capabilities.output(handler)
    with pytest.raises(OutputError, match="does not hold 'field'"):
        field_modes(handler)
    with pytest.raises(OutputError, match="does not hold"):
        energy(handler)


@pytest.mark.parametrize("choice", [True, "interpolated"])
def test_a_resume_is_refused_by_name(choice):
    planes: list[SavedPlane] = []
    handler = CallbackOutput(planes.append)
    simulation = cartesian("numpy_ref")
    with pytest.raises(DriverError, match="only a path can be resumed"):
        simulation.run(handler, resume=choice, progress="none")
    assert handler.metadata is None
    assert not planes


@pytest.mark.parametrize(
    ("reader", "name"),
    [
        (lambda handler: check_cache(handler, _example_metadata()), "check_cache"),
        (resume, "resume"),
        (resumability, "resumability"),
    ],
    ids=["check_cache", "resume", "resumability"],
)
@pytest.mark.parametrize("build", [opened, MemoryOutput], ids=["callback", "memory"])
def test_a_check_against_a_handler_is_refused_by_name(reader, name, build):
    handler = build()
    with pytest.raises(
        OutputError, match=rf"{name} reads a results file.*{type(handler).__name__}"
    ):
        reader(handler)


def test_a_handler_that_has_served_a_run_refuses_another():
    simulation = cartesian("numpy_ref")
    handler = simulation.run(CallbackOutput(nothing), progress="none")
    served = handler.n_saved
    with pytest.raises(OutputError, match="no run can continue it"):
        simulation.run(handler, progress="none")
    with simulation.runner() as runner:
        with pytest.raises(OutputError, match="no run can continue it"):
            runner.run(simulation._field0, output=handler, progress="none")
        # The runner is unharmed, and the next input takes a handler of its own.
        assert not runner.closed
        again = runner.run(
            simulation._field0, output=CallbackOutput(nothing), progress="none"
        )
    assert again.n_saved == handler.n_saved == served


def test_a_handler_holding_planes_is_not_continued():
    """The in-memory form of a resume: the run asks the handler where it stands."""
    simulation = cartesian("numpy_ref")
    problem = simulation.translate()
    metadata = simulation.metadata()
    planes: list[SavedPlane] = []
    handler = CallbackOutput(planes.append)
    handler.open(metadata)
    handler.save(field_free_plane())
    with (
        simulation.runner() as runner,
        pytest.raises(OutputError, match="no run can continue it"),
    ):
        run_problem(problem, runner.backend, handler, metadata=metadata)
    assert len(planes) == 1


# ---------------------------------------------------------- how a run ends early


class _RefusedError(Exception):
    """What the callable raises, which nothing in the package does."""


@pytest.mark.parametrize("backend", available())
def test_what_the_callable_raises_stops_the_run_and_propagates(backend):
    def refuse_the_second(plane: SavedPlane) -> None:
        if plane.z > 0.0:
            raise _RefusedError

    simulation = cartesian(backend)
    handler = CallbackOutput(refuse_the_second)
    with pytest.raises(_RefusedError):
        simulation.run(handler, progress="none")
    assert handler.n_saved == 1
    assert repr(handler) == "CallbackOutput(1 planes, closed)"
    # A run that raises leaves a runner open for the next input.
    with simulation.runner() as runner:
        with pytest.raises(_RefusedError):
            runner.run(
                simulation._field0,
                output=CallbackOutput(refuse_the_second),
                progress="none",
            )
        assert not runner.closed
        assert runner.run(
            simulation._field0, output=CallbackOutput(nothing), progress="none"
        ).n_saved == handler.n_saved + len(simulation.settings.save_planes)


@pytest.mark.parametrize("backend", available())
def test_a_stopped_run_returns_the_handler_with_no_state_written(backend, caplog):
    """Nothing can continue it, so the run does not say a state was written."""
    flag = threading.Event()
    simulation = cartesian(backend)

    def stop_at_the_entrance(plane: SavedPlane) -> None:
        flag.set()

    handler = CallbackOutput(stop_at_the_entrance)
    with caplog.at_level(logging.INFO, logger="solprop.driver.run"):
        assert simulation.run(handler, progress="none", stop=flag) is handler
    assert handler.n_saved == 1
    assert "no state was written there" in caplog.text
    assert "resume=True continues" not in caplog.text


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
def test_a_run_with_an_optmat_rate_finishes_with_no_report(backend):
    """A store is read back for the peak rates; this handler answers that it has none.

    The rate is there for the callable to read, which is what the guide tells a scan.
    """
    simulation = Simulation.from_capillary(
        125e-6,
        1e-3,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_save=2,
        rtol=1e-6,
        backend=backend,
        noise=None,
    )
    rates: list[np.ndarray] = []

    def receive(plane: SavedPlane) -> None:
        rates.append(plane.stats["peak_ionisation_rate"])

    handler = simulation.run(CallbackOutput(receive), progress="none")
    assert handler.n_saved == len(rates) == 2
    assert simulation._peak_fields == ()
    with simulation.runner() as runner:
        scanned = runner.run(
            simulation._field0, output=CallbackOutput(receive), progress="none"
        )
        assert scanned.n_saved == 2
        # The store answers on the same run: what the handler gives up is this.
        store = runner.run(simulation._field0, progress="none")
    assert isinstance(store, PlaneStore)
    assert PEAK_FIELDS_ANNOTATION in store.annotations

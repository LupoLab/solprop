# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The outer loop: what it saves, what it reports and what it refuses."""

import logging
from typing import Any

import h5py
import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import (
    FIXED_ORDER,
    PRESCRIBED,
    assert_matched,
    assert_stored_matched,
    simulation_bound,
)
from solprop import fourier
from solprop.backends.protocol import Backend, Observation, StepHistory, StepSummary
from solprop.backends.registry import select
from solprop.driver import Simulation, SimulationSettings, metadata_for, run_problem
from solprop.driver.run import plane_from_observation
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.nonlinear.gnlse import problem_from
from solprop.output import HDF5Output, MemoryOutput
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

# Every column but the wall time: two runs of the same prescribed march take the same
# steps and score the same norms to rounding, but they never take the same number of
# microseconds.
STORED_COLUMNS = (
    "field",
    "z",
    "stats/energy",
    "stats/omega0",
    "stats/peak_power",
    "stats/fwhm_t",
    "stats/fwhm_t_outer",
    "step_history/z",
    "step_history/h",
    "step_history/norm",
    "step_history/accepted",
    "step_history/counts",
)

# The between-planes channel, which only a run with a periodic subset records.
STORED_ROW_COLUMNS = (
    "step_stats/z",
    "step_stats/step",
    "step_stats/counts",
    "step_stats/energy",
)


def bare(length):
    """Return the soliton fibre's former grid and no collar, as keywords.

    A short fixed schedule is refused beside the default collar, and these tests
    are about the run loop, not boundaries.
    """
    grid = EnvelopeGrid.from_limits(
        length, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return {"grid": grid, "absorber": bare_absorber(grid)}


def matched(build_simulation, **keywords):
    """Return the soliton run on a prescribed schedule at the fixed Adams order.

    Two runs of it are one discrete march, so they agree within its rounding budget
    (SEP 25), whatever executes them.
    """
    planes = list(np.asarray(build_simulation(**keywords).settings.save_planes))
    schedule = FixedSchedule.uniform(
        0.0, planes[-1], planes[-1] / 40, save_planes=planes
    )
    return build_simulation(
        schedule=schedule, integrator=FIXED_ORDER, **bare(planes[-1]), **keywords
    )


class Recorder(NullReporter):
    """A progress reporter that remembers what it was told."""

    def __init__(self):
        self.events = []
        self.planes = []
        self.totals = []

    def started(self, z_start, z_end, n_planes):
        self.events.append("started")

    def step(self, z, dz, accepted):
        self.events.append("step")

    def plane(self, z, summary):
        self.events.append("plane")
        self.planes.append((z, summary))

    def finished(self, total):
        self.events.append("finished")
        self.totals.append(total)


class NullBackend:
    """A backend that satisfies the protocol and refuses to be stepped.

    The refusals of ``run_problem`` fire before the backend is touched, and a
    ``runtime_checkable`` protocol checks only that the methods are there, so this is
    enough to reach them without building a problem the reference backend would accept.
    """

    name = "null"
    max_batch = 1

    def __init__(self, failure=None, *, shape=(1, 1, 1, 8)):
        self.failure = failure
        self.shape = shape

    def set_initial(self, field, noise, *, z=0.0, resume=False):
        pass

    def set_progress(self, reporter):
        self.reporter = reporter

    def propagate_to(self, z):
        if self.failure is not None:
            raise self.failure
        return StepSummary(z, 1, 0, 0.5, 0.5, 0.0, True)

    def observe(self, *, field=True):
        history = StepHistory(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0)
        )
        state = np.zeros(self.shape, complex) if field else None
        return Observation(0.0, state, None, history, 0, 0)

    def stats(self, spec):
        return {}

    def metadata(self) -> dict[str, object]:
        return {"backend": self.name, "propagator_mode": "tabulate"}

    def restore_call_ledgers(self, logical, atomic=None):
        pass

    def memory_ledger(self):
        return {}

    def available_device_memory(self):
        return None

    def close(self):
        pass


@pytest.fixture
def batched_problem(narrow_grid, soliton):
    """A two-member batch, which a stored plane has no room for."""
    gnlse = problem_from(
        [0.0, 0.0, soliton.beta2],
        omega_ref=narrow_grid.omega0,
        gamma=soliton.gamma,
        shock=False,
    )
    start = SechPulse(
        wavelength=soliton.wavelength,
        width=soliton.tau0,
        peak_power=soliton.peak_power(1),
    ).build(narrow_grid)
    return assemble(
        narrow_grid,
        gnlse.mode,
        [gnlse.kerr],
        np.concatenate([start[np.newaxis], start[np.newaxis]]),
        controller=PIController(rtol=1e-8, atol=1e-18),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=1e-4,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )


def item(parent, *names) -> Any:
    """Return one h5py object or serialised entry, loosely typed for inspection."""
    value = parent
    for name in names:
        value = value[name]
    return value


def read_file(path, key):
    """Return one stored array out of a closed HDF5 output file."""
    with h5py.File(path, "r") as file:
        return np.asarray(item(file, *key.split("/"))[()])


# ------------------------------------------------------------------ what is saved


def test_the_stored_history_holds_start_positions_and_per_attempt_times():
    """The backend records end positions and a running clock; the file does not.

    ``output.schema.StepHistory`` documents ``z`` as the start of each attempt and
    ``wall_time`` as the time one attempt took, and the driver converts on the way.
    Exactly representable numbers, so the check is bitwise.
    """
    history = StepHistory(
        z=np.array([0.5, 1.0, 1.25]),
        dz=np.array([0.5, 0.5, 0.25]),
        error_norm=np.array([0.5, 2.0, 0.5]),
        accepted=np.array([True, False, True]),
        wall_time=np.array([0.25, 0.75, 1.0]),
    )
    observation = Observation(1.25, np.ones((1, 1, 1, 4), complex), None, history, 3, 1)
    stored = plane_from_observation(observation).step_history
    assert np.array_equal(stored.z, [0.0, 0.5, 1.0])
    assert np.array_equal(stored.h, history.dz)
    assert np.array_equal(stored.norm, history.error_norm)
    assert np.array_equal(stored.accepted, history.accepted)
    assert np.array_equal(stored.wall_time, [0.25, 0.5, 0.25])
    # An empty history, the entrance plane's, converts to an empty one.
    empty = StepHistory(
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0)
    )
    assert plane_from_observation(
        Observation(0.0, np.ones((1, 1, 1, 4), complex), None, empty, 0, 0)
    ).step_history.z.shape == (0,)


def test_the_initial_plane_is_saved_at_zero_with_an_empty_history(build_simulation):
    handler = build_simulation().run()
    assert handler["z"][0] == 0.0
    assert handler["step_history/counts"][0] == 0
    # The state at z = 0 is the input masked to the active band, so its energy is the
    # pulse's own to the mask's accuracy; here it is only asserted to be there.
    assert handler["stats/energy"].shape == (4, 1, 1)


def test_every_save_plane_is_landed_on_and_saved_with_its_statistics(build_simulation):
    simulation = build_simulation(n_save=6)
    handler = simulation.run()
    planes = np.asarray(simulation.settings.save_planes)
    # Steps land on the planes exactly (ADR-009), so the stored positions are the
    # requested ones bitwise, not to a tolerance.
    assert np.array_equal(handler["z"], np.concatenate([[0.0], planes]))
    assert handler.n_saved == 6
    for name in ("omega0", "energy", "peak_power", "fwhm_t", "fwhm_t_outer"):
        assert handler[f"stats/{name}"].shape[0] == 6
    # A step history that is empty only for the first plane: every later one was reached
    # by at least one step.
    counts = handler["step_history/counts"]
    assert counts[0] == 0
    assert np.all(counts[1:] > 0)


def test_the_stored_field_is_the_optics_spectrum_of_the_observation(build_simulation):
    simulation = matched(build_simulation, n_save=2)
    handler = simulation.run()
    problem = simulation.translate()
    # The backend the run itself used: the claim is that storing a plane is a
    # conjugation and a sign flip and nothing else, both exact, so the stored plane is
    # the observation's within the rounding of running the march a second time.
    backend = select(
        simulation.settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
    )
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(simulation.settings.save_planes[-1]))
    observation = backend.observe()
    backend.close()
    expected = fourier.spectrum_to_optics(
        observation.require_field()[0], kind="envelope"
    )
    # The same problem stepped the same way, a matched march; the storage boundary
    # is a conjugation and a sign flip, both exact in floating point (ADR-017).
    bound = simulation_bound(simulation)
    assert_matched(handler["field"][-1], expected, bound)
    assert_matched(handler.fields("fft")[-1], observation.require_field()[0], bound)
    # And on one stored array the two conventions are that exact map of each other.
    assert np.array_equal(
        handler["field"][-1],
        fourier.spectrum_to_optics(handler.fields("fft")[-1], kind="envelope"),
    )


def test_the_two_handlers_store_a_run_alike(build_simulation, tmp_path):
    # Two runs of one prescribed march, one into each handler: what the schedule
    # prescribes and the counts exactly, the values within the march's budget (the
    # handlers' lossless storage of one set of planes is tested exactly in
    # tests/output/test_handlers.py).
    names = ("energy", "peak_power")
    spec = StatsSpec(names=names, periodic=("energy",))
    simulation = matched(build_simulation, stats=spec)
    memory = simulation.run()
    path = tmp_path / "run.h5"
    simulation.run(path)
    keys = [
        key
        for key in (*STORED_COLUMNS, *STORED_ROW_COLUMNS)
        if not key.startswith("stats/") or key.split("/", 1)[1] in names
    ]
    bound = simulation_bound(simulation)
    norm = "step_history/norm"
    assert_stored_matched(
        {key: memory[key] for key in keys if key != norm},
        {key: read_file(path, key) for key in keys if key != norm},
        bound,
        exact=(*PRESCRIBED, "step_stats/step"),
    )
    # A scaled defect near round-off, ``||d|| / (rtol |y|)``: the budget relative to
    # the state moves it by up to ``bound / rtol`` absolutely.
    rtol = simulation.settings.controller.rtol
    np.testing.assert_allclose(
        read_file(path, norm), memory[norm], rtol=0, atol=bound / rtol
    )
    assert read_file(path, "step_history/wall_time").shape == (
        memory["step_history/wall_time"].shape
    )


def test_a_second_run_reuses_the_translated_problem(build_simulation):
    simulation = matched(build_simulation, n_save=2)
    first = simulation.run()
    second = simulation.run()
    assert simulation.translate() is simulation.translate()
    assert_matched(second["field"], first["field"], simulation_bound(simulation))


# --------------------------------------------------------------- what is reported


def test_a_step_too_coarse_for_its_tolerance_is_logged_with_its_plane(
    build_simulation, caplog
):
    length = 0.01
    simulation = build_simulation(
        length=length,
        n_save=3,
        # Four steps over a sixteenth of a soliton period is far too coarse for a
        # tolerance of 1e-12, and a replayed schedule takes them anyway: the embedded
        # estimate is computed regardless, so the driver reports it rather than leaving
        # it silent (stepper.md section 8). Measured largest scaled norm 91.7 against
        # the requested 1.0; twenty steps score 0.029 and pass.
        schedule=FixedSchedule.uniform(
            0.0, length, length / 4, save_planes=[length / 2]
        ),
        rtol=1e-12,
        **bare(length),
    )
    with caplog.at_level(logging.WARNING, logger="solprop.driver.run"):
        simulation.run()
    warnings = [
        record for record in caplog.records if record.name == "solprop.driver.run"
    ]
    assert warnings, caplog.text
    assert "save plane at z =" in warnings[0].getMessage()
    assert f"{length / 2:.6e}" in warnings[0].getMessage()


def test_the_reporter_is_called_once_per_plane_and_once_at_the_end(build_simulation):
    reporter = Recorder()
    simulation = build_simulation(n_save=4)
    simulation.run(progress=reporter)
    assert [z for z, _ in reporter.planes] == list(simulation.settings.save_planes)
    assert len(reporter.totals) == 1
    assert reporter.totals[0] >= 0.0
    assert all(summary.n_steps > 0 for _, summary in reporter.planes)
    # A user reporter reaches the step loop the same way the default one does: the
    # driver attaches whatever it was given, so the whole protocol is live.
    assert reporter.events[0] == "started"
    assert reporter.events[-1] == "finished"
    assert reporter.events.count("step") == sum(
        summary.n_steps for _, summary in reporter.planes
    )


def test_the_default_reporter_logs_one_line_per_plane(build_simulation, caplog):
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        build_simulation(n_save=3).run()
    lines = [record.getMessage() for record in caplog.records]
    assert sum(line.startswith("z = ") for line in lines) == 2
    assert sum(line.startswith("finished at z = ") for line in lines) == 1


# ------------------------------------------------------------------ what is refused


def test_a_statistics_spec_that_disagrees_about_the_channel_is_refused(
    build_simulation, narrow_grid
):
    """The channel is armed when the backend is built, from the problem's own spec.

    So a caller reaching `run_problem` directly with a different one would get plane
    statistics from one spec and rows from the other. Refused before anything is
    written rather than left to disagree.
    """
    spec = StatsSpec(names=("energy",), periodic=("energy",), period=10)
    problem = build_simulation(n_save=2, stats=spec).translate()
    with pytest.raises(DriverError, match="the between-planes channel is armed"):
        run_problem(
            problem,
            NullBackend(),
            MemoryOutput(),
            metadata=metadata_for(problem, narrow_grid),
            stats_spec=StatsSpec(names=("energy",)),
        )


def test_a_batch_axis_is_refused(batched_problem, narrow_grid):
    with pytest.raises(DriverError, match="plane has no batch axis"):
        run_problem(
            batched_problem,
            NullBackend(),
            MemoryOutput(),
            metadata=metadata_for(batched_problem, narrow_grid),
        )


def test_a_start_plane_outside_the_run_is_refused(build_simulation, narrow_grid):
    simulation = build_simulation(n_save=3)
    problem = simulation.translate()
    with pytest.raises(DriverError, match="between 0 and 2"):
        run_problem(
            problem,
            NullBackend(),
            MemoryOutput(),
            metadata=simulation.metadata(),
            start_plane=3,
        )


def test_the_output_is_closed_when_the_run_fails(build_simulation, tmp_path):
    simulation = build_simulation(n_save=2)
    problem = simulation.translate()
    handler = HDF5Output(tmp_path / "broken.h5")
    with pytest.raises(RuntimeError, match="the card caught fire"):
        run_problem(
            problem,
            NullBackend(RuntimeError("the card caught fire"), shape=problem.shape),
            handler,
            metadata=simulation.metadata(),
        )
    # The plane at z = 0 was written before the failure and the file is readable, which
    # is what closing in a `finally` buys.
    assert handler.closed
    assert read_file(tmp_path / "broken.h5", "z").tolist() == [0.0]


def test_a_run_without_statistics_saves_planes_all_the_same(soliton):
    """A spec that asks for nothing is a run that stores fields and history only."""
    simulation = Simulation.from_gnlse(
        soliton.gamma,
        0.005,
        [0.0, 0.0, soliton.beta2],
        wavelength=soliton.wavelength,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=soliton.tau0,
        peak_power=soliton.peak_power(1),
        shape="sech",
        shock=False,
        n_save=2,
        stats=StatsSpec(),
        noise=None,
    )
    handler = simulation.run()
    assert handler.n_saved == 2
    assert handler["field"].shape == (2, 1, 1, simulation.grid.n_omega)


def test_the_loop_runs_without_a_simulation(narrow_grid, soliton):
    """`run_problem` is usable on its own: a problem, a backend and a handler."""
    gnlse = problem_from(
        [0.0, 0.0, soliton.beta2],
        omega_ref=narrow_grid.omega0,
        gamma=soliton.gamma,
        shock=False,
    )
    start = SechPulse(
        wavelength=soliton.wavelength,
        width=soliton.tau0,
        peak_power=soliton.peak_power(1),
    ).build(narrow_grid)
    problem = assemble(
        narrow_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-18),
        save_planes=[0.005, 0.01],
        dz_max=0.01,
        dz_init=1e-4,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )
    backend = select("numpy_ref", problem, grid=narrow_grid, modes=[gnlse.mode])
    backend.set_initial(problem.initial.field0, None)
    handler = MemoryOutput()
    run_problem(
        problem,
        backend,
        handler,
        metadata=metadata_for(problem, narrow_grid, modes=[gnlse.mode]),
    )
    backend.close()
    assert handler["z"].tolist() == [0.0, 0.005, 0.01]


def test_the_settings_refuse_save_planes_that_are_not_increasing():
    with pytest.raises(DriverError, match="positive and increasing"):
        SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-18),
            save_planes=[1.0, 0.5],
            dz_max=0.1,
            dz_init=1e-3,
        )


def test_the_settings_refuse_a_plane_at_the_entrance():
    with pytest.raises(DriverError, match="saved anyway"):
        SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-18),
            save_planes=[0.0, 0.5],
            dz_max=0.1,
            dz_init=1e-3,
        )


def test_a_null_backend_satisfies_the_protocol():
    """The stub above is only a valid stand-in while it implements the protocol."""
    assert isinstance(NullBackend(), Backend)


# ------------------------------------------- the statistics channel between planes


def test_a_periodic_subset_is_recorded_end_to_end(build_simulation):
    """A `periodic` subset runs instead of being refused.

    The whole path in one test -- the spec, the driver's channel, the seam record, the
    plane and both handlers' layout -- on a real `prop_gnlse`-shaped run.
    """
    names = ("energy", "peak_power")
    spec = StatsSpec(names=names, periodic=names, period=2)
    output = build_simulation(n_save=3, stats=spec).run()

    counts = output["step_stats/counts"]
    assert counts.shape == output["z"].shape
    # The entrance plane is saved before a step is taken, so it carries no rows.
    assert counts[0] == 0
    assert counts[1:].sum() > 0
    assert output.step_stat_names == tuple(sorted(names))

    z = output["step_stats/z"]
    step = output["step_stats/step"]
    assert z.size == step.size == int(counts.sum())
    assert np.all(np.diff(z) > 0.0)
    for name in names:
        rows = output[f"step_stats/{name}"]
        assert rows.shape[0] == z.size
        # The stored rows have lost the batch axis, as the plane statistics have.
        assert rows.shape[1:] == output[f"stats/{name}"].shape[1:]

    # Every row stands at an accepted step of the run, at that step's own end.
    ends = output["step_history/z"] + output["step_history/h"]
    assert np.all(np.isin(np.round(z, 12), np.round(ends, 12)))


def test_a_run_without_a_periodic_subset_reads_back_as_an_empty_channel(
    build_simulation,
):
    """The default. Asking is not an error, so a reader may ask unconditionally."""
    output = build_simulation(n_save=2).run()
    assert output.step_stat_names == ()
    assert output["step_stats/z"].size == 0
    assert output["step_stats/step"].size == 0
    assert output["step_stats/counts"].tolist() == [0, 0]


def test_the_period_is_what_it_says_on_a_real_run(build_simulation):
    """Twice the period is half the rows, to the rounding a plane boundary forces."""
    names = ("energy",)

    def rows_at(period):
        spec = StatsSpec(names=names, periodic=names, period=period)
        output = build_simulation(n_save=2, stats=spec).run()
        return int(output["step_stats/counts"].sum()), output

    every, output = rows_at(1)
    every_third, _ = rows_at(3)
    accepted = int(np.count_nonzero(output["step_history/accepted"]))
    assert every == accepted
    # The phase runs over the whole run, not per plane, so the count is the floor of
    # the accepted-step count over the period, less one at most per plane boundary.
    assert every_third <= accepted // 3 + 1
    assert every_third >= accepted // 3 - 1


def test_a_run_that_raises_tells_the_reporter_it_was_interrupted(build_simulation):
    """A reporter holding a terminal gets it back before the error escapes."""

    class Events(NullReporter):
        def __init__(self):
            self.calls = []

        def started(self, z_start, z_end, n_planes):
            self.calls.append("started")

        def finished(self, total):
            self.calls.append("finished")

        def interrupted(self, z, error):
            self.calls.append(("interrupted", z, type(error).__name__))

    simulation = build_simulation(n_save=2)
    problem = simulation.translate()
    events = Events()
    with pytest.raises(RuntimeError, match="the card caught fire"):
        run_problem(
            problem,
            NullBackend(RuntimeError("the card caught fire"), shape=problem.shape),
            MemoryOutput(),
            metadata=simulation.metadata(),
            progress=events,
        )
    # The null backend never calls `started` (the real step driver does), so the
    # interruption is the only event, and it names the last plane reached: none.
    assert events.calls == [("interrupted", 0.0, "RuntimeError")]


def test_a_reporter_without_interrupted_is_still_accepted(build_simulation):
    """A reporter written to the four-call protocol is not required to grow."""

    class FourCalls:
        def started(self, z_start, z_end, n_planes):
            pass

        def step(self, z, dz, accepted):
            pass

        def plane(self, z, summary):
            pass

        def finished(self, total):
            pass

    simulation = build_simulation(n_save=2)
    problem = simulation.translate()
    with pytest.raises(RuntimeError, match="the card caught fire"):
        run_problem(
            problem,
            NullBackend(RuntimeError("the card caught fire"), shape=problem.shape),
            MemoryOutput(),
            metadata=simulation.metadata(),
            progress=FourCalls(),  # type: ignore[arg-type] - the older protocol on purpose
        )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One rule for the saved planes and the end of a run (SEP 51).

Every run records its entrance and ends at its length; ``n_save`` counts evenly spaced
observations over ``[initial_z, length]`` and ``save_planes`` lists positions inside it.
The placement, the endpoint rounding and every refusal are one helper's, reached by
every adoption: the constructor, the keyword builders, the settings setter and
``with_input``. Resolved planes are compared exactly: they are one ``linspace`` or one
normalized copy, a structural contract.
"""

import dataclasses
import math

import numpy as np
import pytest

from builders import grid_at
from solprop.backends.numpy_ref.workspace import NumpyWorkspace
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.settings import DEFAULT_N_SAVE, _observation_planes
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.output.hdf5 import read_resume_state
from solprop.recipe import Recipe, resume_file
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep, RungeKutta
from solprop.stepper.schedule import LANDING_ULPS, replay_tolerance

LENGTH = 1e-5
"""The tiny fibre's length, m: ten micrometres."""

STEPS = {"dz_max": 1e-6, "dz_init": 1e-6}

ULP = math.ulp(LENGTH)
"""One representable increment at the run's end, m."""


def grid(length: float = LENGTH) -> EnvelopeGrid:
    """Return the tiny fibre's grid."""
    return EnvelopeGrid.from_limits(length, 800e-9, (500e-9, 1.6e-6), 200e-15)


def fibre(length: float | None = LENGTH, **keywords) -> Simulation:
    """Return a GNLSE fibre that builds in milliseconds; nothing here runs it."""
    arguments = {
        "wavelength": 800e-9,
        "wavelength_limits": (500e-9, 1.6e-6),
        "trange": 200e-15,
        "fwhm": 20e-15,
        "peak_power": 1.0,
        "backend": "numpy_ref",
        "noise": None,
        **STEPS,
    } | keywords
    return Simulation.from_gnlse(0.0, length, [0.0, 0.0, -1e-26], **arguments)


def settings(**keywords) -> SimulationSettings:
    """Return object-route settings with the tiny fibre's steps."""
    return SimulationSettings(backend="numpy_ref", **STEPS, **keywords)


def adopted(simulation: Simulation, chosen: SimulationSettings, **keywords):
    """Return a direct Simulation of `simulation`'s physics adopting `chosen`."""
    return Simulation(
        keywords.pop("grid", simulation.grid),
        simulation.modes,
        simulation.responses,
        simulation._field0,
        settings=dataclasses.replace(chosen, controller=simulation.settings.controller),
        **keywords,
    )


# ------------------------------------------------------------------- the refusals


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"save_planes": 80}, r"pass n_save=80"),
        ({"save_planes": 0.25}, r"pass \[0\.25\], or n_save"),
        ({"save_planes": True}, r"takes positions in m, got True"),
        ({"save_planes": [0.1], "n_save": 3}, r"give one of n_save and save_planes"),
        ({"n_save": 1}, r"n_save is an integer of at least 2, got 1"),
        ({"n_save": 2.5}, r"n_save is an integer of at least 2, got 2\.5"),
        ({"n_save": True}, r"n_save is an integer of at least 2, got True"),
        ({"save_planes": [0.1, math.nan]}, r"save plane nan is not finite"),
        ({"save_planes": [0.2, 0.1]}, r"save plane 0\.1 m follows 0\.2 m"),
        ({"save_planes": [0.1, 0.1]}, r"save plane 0\.1 m repeats 0\.1 m"),
    ],
)
def test_a_malformed_selector_is_refused_by_name_with_its_fix(given, match):
    with pytest.raises(DriverError, match=match) as refused:
        settings(**given)
    assert refused.value.parameter in (
        "save_planes",
        "n_save",
        ("save_planes", "n_save"),
    )


@pytest.mark.parametrize(
    ("selectors", "shown"),
    [({"n_save": 5}, "5 planes"), ({}, f"{DEFAULT_N_SAVE} planes (default)")],
    ids=["count", "default"],
)
def test_unadopted_settings_print_their_count(selectors, shown):
    settings = SimulationSettings(dz_max=1e-3, dz_init=1e-4, **selectors)
    assert settings._planes_text() == shown


def test_both_selectors_name_how_to_carry_a_count():
    with pytest.raises(DriverError, match=r"dataclasses\.replace\(settings, n_save=N"):
        settings(save_planes=[], n_save=2)


def test_a_numpy_count_is_a_count():
    assert settings(n_save=np.int64(5)).n_save == 5


# ------------------------------------------------------------- placement and ends


def test_a_count_places_that_many_observations_over_the_whole_run():
    simulation = fibre(n_save=5)
    planes = np.asarray(simulation.settings.save_planes)
    # The entrance is observation 0, so four positions follow it, the last the end.
    assert planes.tolist() == np.linspace(0.0, LENGTH, 5)[1:].tolist()
    assert planes[-1] == LENGTH == simulation.grid.length
    assert simulation.settings.n_save is None


def test_neither_selector_takes_the_default_count():
    planes = np.asarray(fibre().settings.save_planes)
    assert planes.size + 1 == DEFAULT_N_SAVE
    assert planes[-1] == LENGTH


def test_positions_short_of_the_length_gain_the_end():
    assert np.asarray(
        fibre(save_planes=[LENGTH / 2]).settings.save_planes
    ).tolist() == [
        LENGTH / 2,
        LENGTH,
    ]


def test_an_empty_list_records_the_two_endpoints_alone():
    assert np.asarray(fibre(save_planes=[]).settings.save_planes).tolist() == [LENGTH]


def test_listed_endpoints_are_kept_once():
    planes = np.asarray(
        fibre(save_planes=[0.0, LENGTH / 2, LENGTH]).settings.save_planes
    )
    assert planes.tolist() == [LENGTH / 2, LENGTH]


@pytest.mark.parametrize(
    "end",
    [np.nextafter(0.3, 0.0), 0.3, 0.1 + 0.2, 0.3 + LANDING_ULPS * math.ulp(0.3)],
    ids=["below", "equal", "sum", "bound"],
)
def test_a_position_within_the_landing_slack_of_the_end_is_the_end(end):
    planes = _observation_planes(np.array([0.1, end]), None, entrance=0.0, length=0.3)
    assert planes.tolist() == [0.1, 0.3]


def test_a_position_just_beyond_the_slack_is_refused_and_one_below_it_kept():
    ulp = math.ulp(0.3)
    beyond = 0.3 + (LANDING_ULPS + 1) * ulp
    with pytest.raises(DriverError, match="beyond the end of the run"):
        _observation_planes(np.array([beyond]), None, entrance=0.0, length=0.3)
    below = 0.3 - (LANDING_ULPS + 1) * ulp
    planes = _observation_planes(np.array([below]), None, entrance=0.0, length=0.3)
    assert planes.tolist() == [below, 0.3]


def test_a_position_beyond_the_end_names_the_interval_and_the_fix():
    with pytest.raises(
        DriverError, match=r"raise length \(or grid\.length\)"
    ) as refused:
        fibre(save_planes=[2 * LENGTH])
    assert "[0 m, 1e-05 m]" in str(refused.value)
    assert refused.value.parameter == "save_planes"


def test_resolving_resolved_planes_adds_nothing():
    first = _observation_planes(np.array([0.1]), None, entrance=0.0, length=0.3)
    again = _observation_planes(first, None, entrance=0.0, length=0.3)
    assert again.tolist() == first.tolist() == [0.1, 0.3]
    simulation = fibre(n_save=4)
    resolved = simulation.settings
    simulation.settings = resolved
    assert simulation.settings is resolved


def test_a_count_and_the_positions_it_resolves_to_are_one_run():
    counted = fibre(n_save=4)
    listed = fibre(save_planes=counted.settings.save_planes)
    assert (
        np.asarray(listed.settings.save_planes).tolist()
        == np.asarray(counted.settings.save_planes).tolist()
    )
    assert (
        listed.metadata().configuration_hash()
        == counted.metadata().configuration_hash()
    )


def test_an_object_route_runs_to_its_grid_length():
    simulation = fibre()
    direct = adopted(simulation, settings(n_save=3))
    assert np.asarray(direct.settings.save_planes).tolist() == [LENGTH / 2, LENGTH]


def test_steps_left_unset_take_the_keyword_routes_fractions_of_the_grid_length():
    # The same defaults as the keyword routes: half the length, a ten-thousandth of
    # it refined by the start-up rule. Exact: one product each.
    simulation = fibre()
    direct = adopted(simulation, SimulationSettings(backend="numpy_ref"))
    assert (direct.settings.dz_max, direct.settings.dz_init) == (
        0.5 * LENGTH,
        1e-4 * LENGTH,
    )
    assert direct.settings.auto_init
    keyword = fibre(dz_max=None, dz_init=None).settings
    assert (keyword.dz_max, keyword.dz_init) == (
        direct.settings.dz_max,
        direct.settings.dz_init,
    )
    given = adopted(simulation, settings())
    assert (given.settings.dz_max, given.settings.dz_init) == (1e-6, 1e-6)
    assert not given.settings.auto_init
    assert "dz_max half the length" in str(SimulationSettings())


# ------------------------------------------------------------- interior launches


def test_a_count_at_an_interior_launch_spans_the_launched_interval():
    simulation = fibre()
    launched = adopted(simulation, settings(n_save=4), initial_z=4e-6)
    # Three steps of 2 um from the launch at 4 um: not the zero-based lattice.
    np.testing.assert_array_equal(
        launched.settings.save_planes, np.linspace(4e-6, LENGTH, 4)[1:]
    )
    assert np.asarray(launched.settings.save_planes)[-1] == LENGTH


def test_an_interior_launch_on_the_zero_based_lattice_still_spans_its_interval():
    simulation = fibre()
    launched = adopted(simulation, settings(n_save=3), initial_z=5e-6)
    # 5 um is on the lattice of a zero-based count of three; the count is not.
    np.testing.assert_array_equal(
        launched.settings.save_planes, np.linspace(5e-6, LENGTH, 3)[1:]
    )


def test_a_position_before_an_interior_launch_is_refused():
    simulation = fibre()
    with pytest.raises(DriverError, match="before the entrance, initial_z"):
        adopted(simulation, settings(save_planes=[2e-6]), initial_z=4e-6)


def test_with_input_keeps_the_resolved_observations_beyond_the_launch():
    simulation = fibre(n_save=5)
    derived = simulation.with_input(simulation._field0, initial_z=3e-6)
    # No new count: the parent's resolved positions beyond 3 um, the end among them.
    np.testing.assert_array_equal(
        derived.settings.save_planes, np.linspace(0.0, LENGTH, 5)[2:]
    )
    assert derived.initial_z == 3e-6


# --------------------------------------------------------- adoption and replacement


def test_replaced_settings_are_resolved_against_the_run():
    simulation = fibre(n_save=2)
    simulation.settings = dataclasses.replace(
        simulation.settings, save_planes=None, n_save=3
    )
    assert np.asarray(simulation.settings.save_planes).tolist() == [LENGTH / 2, LENGTH]


def test_after_translation_a_count_equal_to_the_planes_is_no_change():
    simulation = fibre(n_save=3)
    simulation.translate()
    simulation.settings = dataclasses.replace(
        simulation.settings, save_planes=None, n_save=3
    )
    with pytest.raises(DriverError, match="a new save_planes would be ignored"):
        simulation.settings = dataclasses.replace(
            simulation.settings, save_planes=None, n_save=4
        )


def test_resolved_settings_reused_at_another_length_are_validated_afresh():
    simulation = fibre(save_planes=[LENGTH / 2])
    resolved = simulation.settings
    longer = adopted(simulation, resolved, grid=grid_at(simulation.grid, 2 * LENGTH))
    assert np.asarray(longer.settings.save_planes).tolist() == [
        LENGTH / 2,
        LENGTH,
        2 * LENGTH,
    ]
    with pytest.raises(DriverError, match="beyond the end of the run"):
        adopted(simulation, resolved, grid=grid_at(simulation.grid, LENGTH / 4))


# ------------------------------------------------------------- the length, once


def test_a_grid_alone_gives_the_length():
    simulation = fibre(None, grid=grid(), wavelength_limits=None, trange=None)
    assert np.asarray(simulation.settings.save_planes)[-1] == LENGTH


def test_a_length_within_the_slack_of_the_grid_s_is_the_grid_s():
    near = np.nextafter(LENGTH, 0.0)
    simulation = fibre(near, grid=grid(), wavelength_limits=None, trange=None)
    assert np.asarray(simulation.settings.save_planes)[-1] == LENGTH


def test_a_length_and_a_grid_that_disagree_are_refused_naming_both():
    with pytest.raises(DriverError, match=r"length = 2e-05 m and grid\.length = 1e-05"):
        fibre(2 * LENGTH, grid=grid(), wavelength_limits=None, trange=None)


@pytest.mark.parametrize("route", ["gnlse", "capillary", "waveguide"])
def test_a_keyword_route_without_a_length_or_a_grid_is_refused(route):
    with pytest.raises(DriverError, match="a length or a grid is required"):
        if route == "gnlse":
            fibre(None)
        elif route == "capillary":
            Simulation.from_capillary(125e-6, None, "He", wavelength=800e-9)
        else:
            Simulation.from_waveguide([], None, "He", wavelength=800e-9)


# ---------------------------------------------------------------- replay schedules


def history(steps: np.ndarray, z_start: float = 0.0) -> FixedSchedule:
    """Return the schedule a run that accepted `steps` from `z_start` records."""
    starts = z_start + np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    return FixedSchedule.from_history(starts, steps)


def drift(schedule: FixedSchedule) -> float:
    """Return how far a schedule's step sum ends from the run's length, in ulps."""
    return (schedule.z_end - LENGTH) / math.ulp(LENGTH)


@pytest.mark.parametrize(
    ("schedule", "below"),
    [
        # 160 and 300 equal steps of a 10 um run end 13 ulps short and 21 ulps past
        # it: beyond the landing slack, within replay_tolerance's step count.
        (history(np.full(160, LENGTH / 160)), True),
        (history(np.full(300, LENGTH / 300)), False),
        (FixedSchedule.uniform(0.0, LENGTH - 3 * math.ulp(LENGTH), LENGTH / 100), True),
        (
            FixedSchedule.uniform(0.0, LENGTH + 3 * math.ulp(LENGTH), LENGTH / 100),
            False,
        ),
    ],
    ids=["history-below", "history-above", "uniform-below", "uniform-above"],
)
def test_a_replayed_schedule_whose_sum_drifts_within_its_tolerance_is_accepted(
    schedule, below
):
    assert (drift(schedule) < 0) == below and drift(schedule) != 0
    assert abs(schedule.z_end - LENGTH) <= replay_tolerance(schedule, LENGTH)
    simulation = fibre(schedule=schedule, integrator=LowStorageRungeKutta(), n_save=2)
    assert simulation.settings.schedule is schedule
    assert np.asarray(simulation.settings.save_planes)[-1] == LENGTH


def test_an_adaptive_run_s_own_history_replays_over_the_same_run():
    # No collar, whose in-stage stability would cap the replayed steps.
    run = fibre(n_save=2, integrator=RungeKutta(), absorber=False).run(progress="none")
    schedule = FixedSchedule.from_history(
        np.asarray(run["step_history/z"]),
        np.asarray(run["step_history/h"]),
        np.asarray(run["step_history/accepted"]),
    )
    replayed = fibre(
        n_save=2, integrator=RungeKutta(), absorber=False, schedule=schedule
    )
    assert replayed.run(progress="none")["z"][-1] == LENGTH


@pytest.mark.parametrize(
    "schedule",
    [
        history(np.full(10, LENGTH / 20)),
        history(np.full(30, LENGTH / 20)),
        history(np.full(10, LENGTH / 10), z_start=LENGTH / 10),
    ],
    ids=["shorter", "longer", "late-start"],
)
def test_a_history_that_does_not_cover_the_run_is_refused_before_it_runs(schedule):
    with pytest.raises(DriverError, match="the replayed schedule") as refused:
        fibre(schedule=schedule, integrator=LowStorageRungeKutta(), n_save=2)
    assert refused.value.rule == "replay-interval"


def test_a_uniform_schedule_over_the_run_is_accepted():
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 7)
    assert (
        fibre(schedule=schedule, integrator=Multistep()).settings.schedule is schedule
    )


@pytest.mark.parametrize(
    ("schedule", "match"),
    [
        (FixedSchedule.uniform(0.0, LENGTH / 2, LENGTH / 10), "ends at"),
        (FixedSchedule.uniform(0.0, 2 * LENGTH, LENGTH / 10), "ends at"),
        (FixedSchedule(np.full(10, LENGTH / 10), 1e-7), "starts at"),
    ],
    ids=["shorter", "longer", "late-start"],
)
def test_a_schedule_that_does_not_cover_the_run_is_refused_at_adoption(schedule, match):
    with pytest.raises(DriverError, match=f"the replayed schedule {match}") as refused:
        fibre(schedule=schedule, integrator=Multistep())
    assert refused.value.parameter == "schedule"


def test_an_interior_launch_still_refuses_a_schedule():
    simulation = fibre()
    replay = settings(schedule=FixedSchedule.uniform(4e-6, LENGTH, 1e-6))
    with pytest.raises(
        DriverError, match="refuses the modified noise model and a fixed"
    ):
        adopted(simulation, replay, initial_z=4e-6)


# ------------------------------------------------------- resume to the declared end


class CrashError(RuntimeError):
    """What an interrupted run raises, standing in for a machine going away."""


class Interrupt(NullReporter):
    """Stop a run once it has saved `after` planes."""

    def __init__(self, after: int) -> None:
        self.after, self.seen = after, 0

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.seen += 1
        if self.seen >= self.after:
            raise CrashError(f"stopped after plane {self.seen}")


def resumable(**keywords) -> Simulation:
    """Return the tiny fibre with five observations, one statistic at each."""
    return fibre(
        n_save=5,
        integrator=RungeKutta(),
        stats=StatsSpec(names=("energy",)),
        **keywords,
    )


def ends_complete(path) -> None:
    """Assert a file holds its whole run, ending at the grid's length."""
    state = read_resume_state(path)
    assert state.plane.z == LENGTH
    assert state.n_saved == 5
    assert state.checkpoint is None
    assert state.outcome is not None and state.outcome.status == "complete"


def test_an_interrupted_file_resumes_to_the_declared_end(tmp_path):
    path = tmp_path / "run.h5"
    with pytest.raises(CrashError):
        resumable().run(path, progress=Interrupt(2))
    assert read_resume_state(path).plane.z < LENGTH
    resumable().run(path, resume=True, progress="none")
    ends_complete(path)


def test_a_file_holding_a_terminal_checkpoint_resumes_to_the_declared_end(
    tmp_path, monkeypatch
):
    path = tmp_path / "run.h5"
    original = NumpyWorkspace.stats
    calls: list[int] = []

    def fail_at_the_end(self, spec, z):
        calls.append(1)
        if len(calls) == 5:
            raise CrashError("the last plane's statistics failed")
        return original(self, spec, z)

    with monkeypatch.context() as patch:
        patch.setattr(NumpyWorkspace, "stats", fail_at_the_end)
        with pytest.raises(CrashError):
            resumable().run(path, progress="none")
    stopped = read_resume_state(path)
    # The march reached the end; its last plane was never saved.
    assert stopped.checkpoint is not None and stopped.n_saved < 5
    resumable().run(path, resume=True, progress="none")
    ends_complete(path)


def test_a_completed_file_resumes_to_its_end_and_saves_nothing_more(tmp_path):
    path = tmp_path / "run.h5"
    resumable().run(path, progress="none")
    resumable().run(path, resume=True, progress="none")
    ends_complete(path)


def test_a_recipe_s_file_resumes_to_the_declared_end_from_the_file_alone(tmp_path):
    recipe = Recipe.from_keywords(
        "gnlse",
        gamma=0.0,
        length=LENGTH,
        betas=[0.0, 0.0, -1e-26],
        wavelength=800e-9,
        wavelength_limits=[500e-9, 1.6e-6],
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        n_save=5,
        backend="numpy_ref",
        noise=None,
        dz_max=1e-6,
        dz_init=1e-6,
    )
    path = tmp_path / "run.h5"
    with pytest.raises(CrashError):
        recipe.build().run(path, progress=Interrupt(2))
    resume_file(path, progress="none")
    ends_complete(path)

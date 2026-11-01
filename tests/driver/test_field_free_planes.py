# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statistics-only planes leave the march as it was (SEP 22 section 4.7, section 7).

Skipping a plane's field removes a copy, a conversion and the storage, and no operation
of the march, so the same run is made keeping every field, two planes' fields and none.
On a replayed schedule the prescribed step history is a structural contract (exact) and
everything computed is compared within a rounding bound fixed before the runs; an
adaptive run is compared with a tighter-tolerance reference within an accuracy envelope
(``docs/dev/testing.md`` section 3).
"""

import dataclasses
import logging
from typing import cast

import h5py
import numpy as np
import pytest

from bounds import AccuracyEnvelope, assert_within, fft_bound, gamma
from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.protocol import BackendHistory, Observation
from solprop.backends.registry import available, select
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.memory import admit_memory, field_bytes, plane_bytes
from solprop.driver.run import field_retention, run_problem
from solprop.exceptions import BackendError, ConvergenceError, DriverError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.log import NullReporter
from solprop.nonlinear.responses import KerrResponse
from solprop.output import MemoryOutput, open_file
from solprop.output.schema import FieldRetention, FieldsChoice, Output, PlaneStore
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    PIController,
    RungeKutta,
)

LENGTH = 2e-4
STEPS = 10
PLANES = np.linspace(LENGTH / 4, LENGTH, 4)
RULES: tuple[tuple[FieldsChoice, tuple[int, ...]], ...] = (
    ("all", ()),
    ("planes", (0, -1)),
    ("none", ()),
)


def stored(output: Output) -> PlaneStore:
    """Return a standard handler as the store it is, for its typed readers."""
    assert isinstance(output, PlaneStore)
    return output


def cartesian(
    backend: str,
    *,
    replay: bool = True,
    noise: bool = False,
    interpolate: bool = False,
    rtol: float = 1e-6,
) -> Simulation:
    """Return a small Cartesian Kerr run of ten replayed (or adaptive) steps.

    On the former grid and without a collar: ten replayed steps are refused beside
    the default collar, and these tests are about field retention, not boundaries.
    """
    grid = EnvelopeGrid.from_limits(
        LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False, **LEGACY_GRID
    )
    space = FreeGrid(3e-4, 12, 4e-4, 12, window_factor=0.25)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=2e-6), 40e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=rtol, atol=1e-14),
        save_planes=PLANES,
        dz_max=LENGTH,
        dz_init=LENGTH / STEPS,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS, save_planes=PLANES)
        if replay
        else None,
        interpolate_planes=interpolate,
        integrator=RungeKutta(),
        backend=backend,
        stats=default_spec("free"),
        absorber=bare_absorber(grid),
    )
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(2e-22, thg=False)],
        pulse,
        settings=settings,
        noise="input" if noise else None,
        seed=7 if noise else None,
    )


def low_storage(length: float = 0.02, steps: int = 8) -> Simulation:
    """Return a short soliton replayed by the low-storage march on ``numpy_ref``.

    On the former grid and without a collar, as :func:`cartesian`.
    """
    schedule = FixedSchedule.uniform(
        0.0, length, length / steps, save_planes=[length / 2]
    )
    grid = EnvelopeGrid.from_limits(
        length, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        save_planes=[length / 2, length],
        schedule=schedule,
        integrator=LowStorageRungeKutta(),
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        noise=None,
    )


def keeping(
    simulation: Simulation, fields: FieldsChoice, field_planes=()
) -> Simulation:
    """Return the simulation with another field retention: an execution setting."""
    simulation.settings = dataclasses.replace(
        simulation.settings, fields=fields, field_planes=field_planes
    )
    return simulation


def replay_bound(simulation: Simulation) -> float:
    """Return the relative rounding bound of a field after the replayed window.

    Fixed before any run: the ladder's sixteen roundings of a DOPRI5 step per step
    (``docs/dev/accuracy.md`` section 7), each carrying the transverse and temporal
    transform pair of every stage, over the ten steps. The Kerr phase of this case is
    a few milliradians, so the propagation does not amplify a perturbation.
    """
    problem = simulation.translate()
    columns = problem.shape[1]
    return STEPS * (gamma(16) + 7 * fft_bound(columns * problem.grids.n_to))


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("noise", [False, True])
def test_a_replayed_run_keeps_its_march_whatever_fields_it_keeps(backend, noise):
    runs = {}
    simulation = cartesian(backend, noise=noise)
    bound = replay_bound(simulation)
    for fields, field_planes in RULES:
        runs[fields] = stored(
            keeping(simulation, fields, field_planes).run(progress="none")
        )
    full, planes, none = runs["all"], runs["planes"], runs["none"]
    assert np.array_equal(none["field_planes"], [])
    assert np.array_equal(planes["field_planes"], [0, len(PLANES)])
    controller = cast("PIController", simulation.settings.controller)
    for output in (planes, none):
        # A prescribed schedule is a structural contract: the same steps, in order.
        for column in ("z", "h", "accepted", "counts"):
            assert np.array_equal(
                output[f"step_history/{column}"], full[f"step_history/{column}"]
            )
        # The scaled norm divides a difference of two solutions by rtol |y|: its
        # rounding is the field's over rtol, absolutely.
        assert_within(
            output["step_history/norm"],
            full["step_history/norm"],
            bound / controller.rtol,
            scale=1.0,
            what="norms",
        )
        assert np.array_equal(output["z"], full["z"])
        for name in full.stat_names:
            reference = full[f"stats/{name}"]
            if not np.any(reference):
                continue
            # A quadratic statistic doubles the field's relative bound.
            assert_within(output[f"stats/{name}"], reference, 2 * bound, what=name)
    for index in (0, len(PLANES)):
        assert_within(
            planes.plane_field(index),
            full.plane_field(index),
            bound,
            what=f"field {index}",
        )
    if noise:
        # The noise phase is kept with the field: on the named planes and no others.
        for output, kept in (
            (full, range(len(PLANES) + 1)),
            (planes, (0, len(PLANES))),
            (none, ()),
        ):
            assert isinstance(output, MemoryOutput)
            for index, plane in enumerate(output.planes()):
                assert (plane.noise_phase is not None) == (index in kept), index


@pytest.mark.medium
@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("interpolate", [False, True])
def test_an_adaptive_run_meets_its_envelope_whatever_fields_it_keeps(
    backend, interpolate
):
    reference = stored(
        cartesian(backend, replay=False, rtol=1e-10).run(progress="none")
    )
    simulation = cartesian(backend, replay=False, interpolate=interpolate)
    full = stored(keeping(simulation, "all").run(progress="none"))
    energy = reference["stats/energy"]
    # The envelope is the full run's own error against the tighter reference, with a
    # factor of four (AccuracyEnvelope's default) for another run of the same method;
    # an error floor of 1e-12 keeps it positive if the two happen to agree closely.
    envelope = AccuracyEnvelope(
        "rtol 1e-10 run of the same case, energy at every plane",
        max(
            float(np.max(np.abs(full["stats/energy"] - energy) / np.abs(energy))),
            1e-12,
        ),
    )
    field = AccuracyEnvelope(
        "rtol 1e-10 run of the same case, the last plane's field",
        max(
            float(
                np.linalg.norm(full.plane_field(-1) - reference.plane_field(-1))
                / np.linalg.norm(reference.plane_field(-1))
            ),
            1e-12,
        ),
    )
    for fields, field_planes in RULES[1:]:
        output = stored(keeping(simulation, fields, field_planes).run(progress="none"))
        assert_within(output["stats/energy"], energy, envelope.bound, what=fields)
        if fields == "planes":
            assert_within(
                output.plane_field(-1),
                reference.plane_field(-1),
                field.bound,
                what="last field",
            )


class _StopAfter(NullReporter):
    """A reporter that stops the run after it has seen a number of planes."""

    def __init__(self, planes: int) -> None:
        self.planes = planes
        self.seen = 0

    def plane(self, z, summary) -> None:
        self.seen += 1
        if self.seen == self.planes:
            msg = "stopped on purpose"
            raise RuntimeError(msg)


@pytest.mark.medium
def test_a_field_free_run_resumes_from_its_checkpoint_alone(tmp_path):
    simulation = cartesian("numpy_ref")
    uninterrupted = stored(simulation.run(progress="none"))
    keeping(simulation, "none")
    path = tmp_path / "run.h5"
    with pytest.raises(RuntimeError, match="on purpose"):
        simulation.run(path, progress=_StopAfter(2))
    with open_file(path) as written:
        assert written.n_saved == 3 and "field" not in written
        checkpoint = written.terminal_checkpoint
        assert checkpoint is not None
        assert checkpoint.plane.field is not None
        # The checkpoint holds the field of the plane the run stopped at.
        assert checkpoint.plane.z == written["z"][-1]
    resumed = stored(simulation.run(path, resume=True, progress="none"))
    with open_file(path) as written:
        assert written.n_saved == len(PLANES) + 1
        assert written.outcome is not None and written.outcome.status == "complete"
        energy = written["stats/energy"]
    # A resumed replay evaluates the first stage again instead of recycling it, so it
    # agrees with the uninterrupted run to rounding (class R, the replay bound).
    assert_within(
        energy,
        uninterrupted["stats/energy"],
        2 * replay_bound(simulation),
        what="energy",
    )
    assert resumed.closed
    # A finished field-free file resumes as finished, running nothing.
    again = stored(simulation.run(path, resume=True, progress="none"))
    assert again.closed


@pytest.mark.medium
def test_a_file_keeping_some_fields_resumes_through_its_last_kept_plane(tmp_path):
    simulation = cartesian("numpy_ref")
    uninterrupted = stored(simulation.run(progress="none"))
    keeping(simulation, "planes", (0, 2, -1))
    path = tmp_path / "run.h5"
    # Stopped at the plane after the entrance, then after the second kept plane: each
    # clean close trims the field rows to those written, and each resume grows them
    # back before its next kept field.
    with pytest.raises(RuntimeError, match="on purpose"):
        simulation.run(path, progress=_StopAfter(1))
    with open_file(path) as written:
        assert written.n_saved == 2
        assert written.terminal_checkpoint is not None
    with pytest.raises(RuntimeError, match="on purpose"):
        simulation.run(path, resume=True, progress=_StopAfter(2))
    with open_file(path) as written:
        assert written.n_saved == 4
    stored(simulation.run(path, resume=True, progress="none"))
    bound = 2 * replay_bound(simulation)
    with open_file(path) as written:
        assert written.n_saved == len(PLANES) + 1
        assert written.outcome is not None and written.outcome.status == "complete"
        assert np.array_equal(written["field_planes"], [0, 2, len(PLANES)])
        for index in (0, 2, len(PLANES)):
            # A resumed replay evaluates the first stage again instead of recycling
            # it, so it agrees with the uninterrupted run to rounding (class R).
            assert_within(
                written.plane_field(index),
                uninterrupted.plane_field(index),
                bound,
                what=f"field {index}",
            )


def test_field_free_planes_are_admitted_beside_the_low_storage_march():
    """SEP 22, Q14: the march holds the last field-free plane, so every rule runs."""
    uninterrupted = stored(low_storage().run(progress="none"))
    for fields, field_planes in RULES[1:]:
        simulation = keeping(low_storage(), fields, field_planes)
        output = stored(simulation.run(progress="none"))
        assert output.n_saved == 3
        assert output.retention == FieldRetention(fields, field_planes).resolved(3)
        # The same replay, observed with or without its field: the march is the one
        # history (exact), and so are its statistics' inputs.
        assert np.array_equal(output["z"], uninterrupted["z"])
        assert_within(
            output["stats/energy"],
            uninterrupted["stats/energy"],
            replay_bound(simulation),
            what="energy",
        )


@pytest.mark.medium
def test_a_file_whose_last_plane_kept_no_field_and_no_checkpoint_is_refused(tmp_path):
    simulation = keeping(cartesian("numpy_ref"), "planes", (0,))
    path = tmp_path / "run.h5"
    with pytest.raises(RuntimeError, match="on purpose"):
        simulation.run(path, progress=_StopAfter(2))
    with h5py.File(path, "r+") as file:
        # What a run killed outright leaves: its planes and no checkpoint.
        del file["terminal_checkpoint"]
    with pytest.raises(DriverError, match="kept no field"):
        simulation.run(path, resume=True, progress="none")


def test_planes_beside_step_planes_are_refused_and_retention_is_validated():
    simulation = cartesian("numpy_ref", replay=False)
    settings = simulation.settings
    with pytest.raises(DriverError, match="step_planes"):
        dataclasses.replace(settings, step_planes=2, fields="planes", field_planes=(0,))
    with pytest.raises(DriverError, match="required exactly when"):
        dataclasses.replace(settings, fields="planes")
    problem = simulation.translate()
    stepped = dataclasses.replace(
        problem, stepper=dataclasses.replace(problem.stepper, step_planes=2)
    )
    with pytest.raises(DriverError, match="step_planes"):
        field_retention(stepped, "planes", (0,))
    assert field_retention(stepped, "none", ()).fields == "none"
    with pytest.raises(DriverError, match="required exactly when"):
        field_retention(problem, "none", (0,))
    with pytest.raises(DriverError, match="outside a run of 5"):
        field_retention(problem, "planes", (7,))
    # An execution setting: it may change after translation.
    keeping(simulation, "none")
    assert simulation.settings.fields == "none"


def test_an_output_that_cannot_store_field_free_planes_is_refused():
    class _NoRetention(MemoryOutput):
        retain = None  # pyright: ignore[reportAssignmentType] - a store without it

    simulation = cartesian("numpy_ref")
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(DriverError, match="cannot store field-free planes"):
            run_problem(
                problem,
                backend,
                _NoRetention(),
                metadata=simulation.metadata(),
                progress="none",
                fields="none",
            )
        with pytest.raises(DriverError, match="required exactly when"):
            run_problem(
                problem,
                backend,
                MemoryOutput(),
                metadata=simulation.metadata(),
                progress="none",
                fields="planes",
            )
    finally:
        backend.close()


def test_the_memory_admission_charges_only_the_kept_fields():
    simulation = cartesian("numpy_ref")
    problem = simulation.translate()
    n_planes = len(PLANES) + 1
    groups = {}
    for fields, field_planes in RULES:
        retention = FieldRetention(fields, field_planes).resolved(n_planes)
        admission = admit_memory(
            problem, "numpy_ref", "none", output=MemoryOutput(), retention=retention
        )
        groups[fields] = admission.estimate.groups["output"]
    per_plane, field = plane_bytes(problem), field_bytes(problem)
    # Exact arithmetic of the store's allocation: the statistics of every plane, the
    # field of the kept ones.
    assert groups["all"] == n_planes * per_plane
    assert groups["planes"] == n_planes * (per_plane - field) + 2 * field
    assert groups["none"] == n_planes * (per_plane - field)


# --- the seam: observe(field=False) (SEP 22 phase 2) --------------------------------


@pytest.mark.parametrize("backend", available())
def test_a_field_free_observation_crosses_no_field(backend, monkeypatch):
    simulation = cartesian(backend, noise=True)
    problem = simulation.translate()
    observations = []
    for field in (True, False):
        owner = select(backend, problem, grid=simulation.grid)
        copies = []
        original = cast("type[StepDriver]", type(owner))._observed_field

        def counted(self, state, copies=copies, original=original):
            copies.append(1)
            return original(self, state)

        monkeypatch.setattr(type(owner), "_observed_field", counted)
        try:
            owner.set_initial(problem.initial.field0, None)
            owner.propagate_to(float(PLANES[0]))
            observations.append(owner.observe(field=field))
        finally:
            owner.close()
            monkeypatch.undo()
        # The field-free observation makes no copy at all (on ``native``, no transfer).
        assert len(copies) == (1 if field else 0)
    kept, free = observations
    assert kept.field is not None and kept.noise_phase is not None
    assert free.field is None and free.noise_phase is None
    # The same replayed march: its history is one structure, observed either way.
    assert np.array_equal(free.step_history.z, kept.step_history.z)
    assert np.array_equal(free.step_history.dz, kept.step_history.dz)
    with pytest.raises(BackendError, match=r"observe\(field=False\)"):
        free.require_field()


def test_the_run_asks_the_backend_for_exactly_the_kept_fields(monkeypatch):
    simulation = keeping(cartesian("numpy_ref"), "planes", (0, -1))
    asked = []
    original = NumpyRefBackend.observe

    def observe(self, *, field=True):
        asked.append(field)
        return original(self, field=field)

    monkeypatch.setattr(NumpyRefBackend, "observe", observe)
    stored(simulation.run(progress="none"))
    assert asked == [True, False, False, False, True]


def test_the_collar_diagnostic_is_skipped_on_field_free_planes_and_recorded(caplog):
    simulation = keeping(cartesian("numpy_ref"), "none")
    with caplog.at_level(logging.INFO, logger="solprop.backends.numpy_ref.backend"):
        output = stored(simulation.run(progress="none"))
    skipped = [
        record for record in caplog.records if "collar diagnostic" in record.message
    ]
    # Logged once, at the first field-free plane; every one counted in the record.
    assert len(skipped) == 1
    assert output.outcome is not None
    assert output.outcome.execution["collar_diagnostic_skipped"] == len(PLANES) + 1
    # A run that keeps every field evaluates it everywhere and records nothing.
    kept = stored(keeping(cartesian("numpy_ref"), "all").run(progress="none"))
    assert kept.outcome is not None
    assert "collar_diagnostic_skipped" not in kept.outcome.execution


def test_a_load_clears_the_skipped_diagnostic_count():
    simulation = cartesian("numpy_ref")
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.observe(field=False)
        assert backend.metadata()["collar_diagnostic_skipped"] == 1
        backend.set_initial(problem.initial.field0, None)
        assert "collar_diagnostic_skipped" not in backend.metadata()
    finally:
        backend.close()


def test_a_field_free_observation_relies_on_the_step_loop_for_finiteness():
    simulation = cartesian("numpy_ref")
    problem = simulation.translate()
    poisoned = problem.initial.field0.copy()
    poisoned[0, 0, 0, np.flatnonzero(problem.grids.active_mask)[0]] = np.nan
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(poisoned, None)
        # Nothing scans a field that is not observed...
        assert backend.observe(field=False).field is None
        # ...while an observed one is scanned, and the replay refuses to step on.
        with pytest.raises(BackendError, match="not finite"):
            backend.observe()
        with (
            np.errstate(invalid="ignore"),
            pytest.raises((ConvergenceError, BackendError)),
        ):
            backend.propagate_to(float(PLANES[0]))
    finally:
        backend.close()


def test_an_observation_refuses_a_noise_phase_without_its_field():
    history = BackendHistory(
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0)
    )
    with pytest.raises(BackendError, match="without a field"):
        Observation(0.0, None, np.zeros((1, 1, 1, 4)), history, 0, 0)
    with pytest.raises(BackendError, match="n_steps is 1"):
        Observation(0.0, None, None, history, 1, 0)


def test_a_lockstep_batch_still_refuses_a_shared_observation():
    from solprop.backends.batch import LockstepRunner  # noqa: PLC0415

    with pytest.raises(BackendError, match="observe_members"):
        LockstepRunner.observe(object.__new__(LockstepRunner), field=False)


def test_the_low_storage_march_holds_its_field_free_plane_with_its_noise_phase():
    simulation = cartesian("numpy_ref", noise=True)
    simulation.settings = dataclasses.replace(
        simulation.settings, integrator=LowStorageRungeKutta()
    )
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        assert backend.held_observation() is None
        backend.propagate_to(float(PLANES[0]))
        assert backend.observe(field=False).field is None
        held = backend.held_observation()
        assert held is not None and held.z == float(PLANES[0])
        observed = backend.observe()
        # The held state is a copy of the state there (exact), its noise phase the
        # same evaluation; a kept plane then releases it.
        assert np.array_equal(held.require_field(), observed.require_field())
        assert held.noise_phase is not None and observed.noise_phase is not None
        assert np.array_equal(held.noise_phase, observed.noise_phase)
        assert held.step_history.n_steps == 0
        assert backend.held_observation() is None
    finally:
        backend.close()


def test_a_lockstep_split_refuses_a_field_free_observation():
    from solprop.driver.batch import split_observation  # noqa: PLC0415

    history = BackendHistory(
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0)
    )
    observation = Observation(0.0, None, None, history, 0, 0)
    with pytest.raises(DriverError, match="every member's field"):
        split_observation(observation, {}, histories=[history])

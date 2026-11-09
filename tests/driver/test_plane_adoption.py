# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run hands its output the observation itself, and nothing changes it afterwards.

The plane holds the observation's arrays rather than a copy, so a handler that reduces
a plane and drops it pays for no volume of its own. What makes that sound is that the
backend never writes to what it observed, whichever march it runs, and a handler may
keep ``plane.field``: each case keeps every plane and compares it exactly, a structural
contract, with a copy taken as the plane arrived.
"""

import dataclasses
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from driver.test_field_free_planes import cartesian, low_storage
from driver.test_runner import capillary
from solprop import fourier
from solprop.backends.driver import StepDriver
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ObservationReady
from solprop.backends.export import ExportRecipe
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.backends.protocol import Backend, BackendHistory, Observation
from solprop.backends.registry import available, select
from solprop.backends.session import Start
from solprop.driver import Simulation
from solprop.driver.run import plane_from_observation
from solprop.exceptions import OutputError
from solprop.fourier import power_amplitude
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.schema import PlaneStore, SavedPlane, _adopted
from solprop.stepper import Multistep, RungeKutta
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import KernelProblem
from solprop.translate.run_request import describe_run

# These cases run coarse grids for speed; their windows are not what is tested here.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")


class _Keeping(MemoryOutput):
    """Keep every plane's arrays by reference, with a copy of each as it arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: list[tuple[np.ndarray, np.ndarray]] = []

    def save(self, plane: SavedPlane) -> None:
        for values in (plane.field, plane.noise_phase):
            if values is not None:
                assert not values.flags.writeable
                self.kept.append((values, values.copy()))
        super().save(plane)


def _integrated(simulation: Simulation, integrator: object) -> Simulation:
    simulation.settings = dataclasses.replace(
        simulation.settings, integrator=integrator
    )
    return simulation


def _single_mode(backend: str) -> Simulation:
    """Return a short soliton, whose single mode has unit power weights."""
    simulation = _integrated(low_storage(), RungeKutta())
    simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
    return simulation


def _single(simulation: Simulation) -> Simulation:
    simulation.settings = dataclasses.replace(simulation.settings, precision="single")
    return simulation


CASES: dict[str, Callable[[str], Simulation]] = {
    "runge-kutta replayed": cartesian,
    "runge-kutta adaptive": lambda name: cartesian(name, replay=False),
    "runge-kutta interpolated": lambda name: cartesian(
        name, replay=False, interpolate=True
    ),
    "runge-kutta with noise": lambda name: cartesian(name, noise=True),
    "low-storage": lambda name: _integrated(cartesian(name), LowStorageRungeKutta()),
    "low-storage with noise": lambda name: _integrated(
        cartesian(name, noise=True), LowStorageRungeKutta()
    ),
    "adams": lambda name: _integrated(cartesian(name), Multistep()),
    "adams modal": capillary,
    "adams modal interpolated": lambda name: capillary(name, dense=True),
    "runge-kutta modal": lambda name: _integrated(capillary(name), RungeKutta()),
    "runge-kutta single-mode": _single_mode,
}


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("case", CASES)
def test_a_kept_plane_never_changes(backend, case):
    simulation = CASES[case](backend)
    handler = _Keeping()
    simulation.run(output=handler, progress="none")
    assert handler.n_saved > 1
    assert len(handler.kept) >= handler.n_saved
    for held, as_it_arrived in handler.kept:
        assert np.array_equal(held, as_it_arrived)


@pytest.mark.parametrize(
    "case", ["runge-kutta replayed", "runge-kutta modal", "runge-kutta single-mode"]
)
def test_a_kept_single_precision_plane_never_changes(case):
    """The widened state is the observation's own (SEP 47 section 5.4)."""
    simulation = _single(CASES[case]("numpy_ref"))
    handler = _Keeping()
    simulation.run(output=handler, progress="none")
    assert handler.n_saved > 1
    for held, as_it_arrived in handler.kept:
        assert np.array_equal(held, as_it_arrived)


@pytest.mark.parametrize("backend", available())
def test_a_kept_plane_survives_the_field_free_planes_after_it(backend):
    """The low-storage march steps in place again after a plane without a field."""
    simulation = _integrated(cartesian(backend), LowStorageRungeKutta())
    simulation.settings = dataclasses.replace(
        simulation.settings, fields="planes", field_planes=(0, 1)
    )
    handler = _Keeping()
    simulation.run(output=handler, progress="none")
    assert handler.n_saved > 2
    assert len(handler.kept) == 2
    for held, as_it_arrived in handler.kept:
        assert np.array_equal(held, as_it_arrived)


@pytest.mark.parametrize("backend", available())
def test_a_run_hands_over_the_observation_itself(backend, monkeypatch):
    """No copy on the way to the handler, on a route with no conversion of units."""
    simulation = cartesian(backend)
    observed: list[Observation] = []
    shared: list[bool] = []

    class Sharing(MemoryOutput):
        def save(self, plane: SavedPlane) -> None:
            assert plane.field is not None
            shared.append(np.shares_memory(plane.field, observed[-1].field))
            super().save(plane)

    with simulation.runner() as runner:
        observe = runner.backend.observe

        def watched(*, field: bool = True) -> Observation:
            observation = observe(field=field)
            observed.append(observation)
            return observation

        monkeypatch.setattr(runner.backend, "observe", watched)
        runner.run(simulation._field0.copy(), output=Sharing(), progress="none")
    assert shared
    assert all(shared)


def test_a_plane_built_by_hand_copies_what_it_is_given():
    field = np.ones((1, 1, 4), dtype=np.complex128)
    phase = np.zeros((1, 1, 4))
    plane = SavedPlane(z=0.0, field=field, noise_phase=phase)
    assert plane.field is not None and plane.noise_phase is not None
    assert not np.shares_memory(plane.field, field)
    assert not np.shares_memory(plane.noise_phase, phase)
    assert not plane.field.flags.writeable and field.flags.writeable


def test_the_public_observation_route_copies_a_member():
    """A member of a batch stays a copy, or its view would pin the batch."""
    observation = _observation(np.ones((2, 1, 1, 4), dtype=np.complex128))
    plane = plane_from_observation(observation, member=1)
    assert plane.field is not None
    assert not np.shares_memory(plane.field, observation.require_field())


def test_an_adopted_array_is_taken_as_it_is_and_checked():
    field = np.ones((1, 1, 4), dtype=np.complex128)
    plane = SavedPlane(z=0.0, field=_adopted(field))
    assert plane.field is field and not field.flags.writeable
    # Replacing another of the plane's fields does not copy it either.
    assert dataclasses.replace(plane, stats={"energy": np.ones(1)}).field is field
    with pytest.raises(OutputError, match="noise_phase has shape"):
        SavedPlane(z=0.0, field=field, noise_phase=_adopted(np.zeros((1, 1, 2))))


def _observation(field: np.ndarray) -> Observation:
    """Return an observation of a batch that took no step."""
    empty = np.empty(0)
    return Observation(
        0.0,
        field,
        None,
        BackendHistory(empty, empty, empty, np.empty(0, bool), empty),
        0,
        0,
    )


# SEP 47 section 5.4: the copies a saved plane no longer makes. Every stored field keeps
# the bits of the conversion in two steps, the true-power weight into a new array and
# then the optics convention.

STORED: dict[str, Callable[[str], Simulation]] = {
    "spatial": cartesian,
    "modal": lambda name: _integrated(capillary(name), RungeKutta()),
    "single-mode": _single_mode,
}


def _copied_observations(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Copy each observed field as its backend returned it, before any conversion."""
    copies: list[np.ndarray] = []
    observe = StepDriver.observe

    def copying(self: StepDriver, *, field: bool = True) -> Observation:
        observation = observe(self, field=field)
        if observation.field is not None:
            copies.append(np.array(observation.field))
        return observation

    monkeypatch.setattr(StepDriver, "observe", copying)
    return copies


@pytest.mark.parametrize("handler", ["memory", "hdf5"])
@pytest.mark.parametrize(
    ("backend", "case", "precision"),
    [
        *((backend, case, "double") for backend in available() for case in STORED),
        *(("numpy_ref", case, "single") for case in STORED),
    ],
)
def test_a_stored_field_has_the_bits_of_the_conversion_in_two_steps(
    backend, case, precision, handler, tmp_path, monkeypatch
):
    simulation = STORED[case](backend)
    if precision == "single":
        simulation = _single(simulation)
    problem = simulation.translate()
    copies = _copied_observations(monkeypatch)
    path = tmp_path / "run.h5"
    output = MemoryOutput() if handler == "memory" else HDF5Output(path)
    simulation.run(output=output, progress="none")
    reader = output if isinstance(output, MemoryOutput) else open_file(path)
    assert isinstance(reader, PlaneStore)
    stored = reader.fields("optics")
    positions = np.asarray(reader["z"])
    assert reader.metadata is not None
    kind = reader.metadata.grid.kind
    assert len(copies) == stored.shape[0] > 1
    for index, observed in enumerate(copies):
        field = observed
        if problem.transform.geometry is None:
            weight = power_weight_at(problem, float(positions[index]))
            # Both of the run's modal routes are exercised: weights of one and others.
            assert bool(np.all(weight == 1.0)) is (case == "single-mode")
            field = power_amplitude(observed, weight)
        expected = fourier.spectrum_to_optics(field[0], kind=kind)
        assert expected.tobytes() == stored[index].tobytes()


def _observed_at_its_first_plane(
    name: str, simulation: Simulation
) -> tuple[KernelProblem, Backend, Observation]:
    """Return a loaded backend stepped to the first save plane, and its observation."""
    problem = simulation.translate()
    backend = select(name, problem, grid=simulation.grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[0]))
    return problem, backend, backend.observe()


@pytest.mark.parametrize("backend", available())
@pytest.mark.parametrize("case", ["modal", "single-mode"])
def test_an_observed_field_is_converted_in_place_only_where_it_is_owned(backend, case):
    """The session converts an owned field where it is; the reference's is detached."""
    simulation = STORED[case](backend)
    problem = simulation.translate()
    stepping = select(
        backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    raws: list[np.ndarray] = []
    seen: list[Observation] = []
    observe = stepping.observe

    def recorded(*, field: bool = True) -> Observation:
        observation = observe(field=field)
        if observation.field is not None:
            raws.append(observation.field.copy())
            seen.append(observation)
        return observation

    stepping.observe = recorded  # pyright: ignore[reportAttributeAccessIssue]
    session = DriverSession(
        stepping,
        problem,
        describe_run(problem),
        Start(field=problem.initial.field0[0], z=float(problem.initial.z)),
    )
    try:
        session.acknowledge(session.advance().receipt)  # the launch plane
        batch = session.advance()
        event = batch.events[-1]
        assert isinstance(event, ObservationReady)
        held = session.export(event.ticket, ExportRecipe(power=True, optics=False))
        owned = backend == "native"
        assert seen[-1].field_owned is owned
        expected = power_amplitude(raws[-1], power_weight_at(problem, event.z))[0]
        assert held.tobytes() == expected.tobytes()
        # The observation's own array stays frozen; a fresh export is the caller's.
        assert held.flags.writeable is not owned
        # Converted where it is when owned: the observation's own array then holds it.
        assert np.shares_memory(held, seen[-1].field) is owned
        if not owned:
            state = cast("StepDriver[np.ndarray]", stepping)._state
            assert not np.shares_memory(held, state)
    finally:
        session.close()


def test_the_widened_single_precision_state_is_owned_and_has_the_same_bits():
    _, stepping, observation = _observed_at_its_first_plane(
        "numpy_ref", _single(cartesian("numpy_ref"))
    )
    try:
        assert isinstance(stepping, NumpySingleBackend)
        state, scale = stepping._state, stepping._scale
        assert isinstance(state, np.ndarray) and state.dtype == np.complex64
        assert observation.field_owned
        field = observation.require_field()
        assert field.tobytes() == (state.astype(np.complex128) * scale).tobytes()
        assert not np.shares_memory(field, state)
    finally:
        stepping.close()


def test_an_observation_without_a_field_owns_none():
    empty = np.empty(0)
    observation = Observation(
        0.0,
        None,
        None,
        BackendHistory(empty, empty, empty, np.empty(0, bool), empty),
        0,
        0,
        field_owned=True,
    )
    assert not observation.field_owned

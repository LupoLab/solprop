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

import numpy as np
import pytest

from driver.test_field_free_planes import cartesian
from driver.test_runner import capillary
from solprop.backends.protocol import BackendHistory, Observation
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.driver.run import plane_from_observation
from solprop.exceptions import OutputError
from solprop.output import MemoryOutput
from solprop.output.schema import SavedPlane, _adopted
from solprop.stepper import Multistep, RungeKutta
from solprop.stepper.low_storage import LowStorageRungeKutta

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

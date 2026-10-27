# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check common batch preparation independently of native execution."""

import dataclasses

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from driver.test_scans import build_point
from runtime_types import requires_typecheck
from solprop.backends.protocol import Observation, StepHistory, StepStats
from solprop.driver.batch import (
    BatchMember,
    _structure,
    compatible_groups,
    refuse_mixed_guides,
    shared_table_bytes,
    split_observation,
)
from solprop.driver.scan import Scan
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError
from solprop.grid import RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule


def member(index, power=1.0, backend="native", fixed=True, **settings):
    simulation = build_point(Scan("fixture", power=[power]).point(0))
    schedule = (
        FixedSchedule.uniform(
            0.0,
            simulation.grid.length,
            0.00025,
            save_planes=simulation.settings.save_planes,
        )
        if fixed
        else None
    )
    simulation.settings = dataclasses.replace(
        simulation.settings, backend=backend, schedule=schedule, **settings
    )
    return BatchMember(index, simulation)


def test_batch_groups_keep_global_indices_and_partial_tail():
    members = [member(i, float(i + 1)) for i in (7, 9, 12, 15, 20)]
    groups = list(compatible_groups(members, 2))
    assert [[m.index for m in group] for group in groups] == [[7, 9], [12, 15], [20]]
    assert all(
        m.simulation.settings.backend == "native" for group in groups for m in group
    )


def test_batch_groups_do_not_build_past_a_full_group():
    built = []

    def points():
        for index in range(5):
            built.append(index)
            yield member(index)

    groups = compatible_groups(points(), 2)
    assert [point.index for point in next(groups)] == [0, 1]
    assert built == [0, 1]
    assert [point.index for point in next(groups)] == [2, 3]
    assert built == [0, 1, 2, 3]
    assert [point.index for point in next(groups)] == [4]


def test_batch_groups_reject_negative_global_index():
    with pytest.raises(DriverError, match="nonnegative global indices"):
        list(compatible_groups([member(-1)], 2))


def test_batch_different_schedule_starts_a_separate_group():
    first, second = member(4), member(6)
    second.simulation.settings = dataclasses.replace(
        second.simulation.settings,
        schedule=FixedSchedule.uniform(
            0.0,
            second.simulation.grid.length,
            0.000125,
            save_planes=second.simulation.settings.save_planes,
        ),
    )
    assert [len(group) for group in compatible_groups([first, second], 4)] == [1, 1]


@pytest.mark.parametrize("backend", ["jax", "numba", "unregistered_backend"])
def test_batch_refuses_unsupported_backends_by_name(backend):
    # On the Runge-Kutta march: the default Adams integrator refuses a backend that
    # does not implement it when the settings are built, before a batch is formed.
    point = member(0, backend=backend, integrator=RungeKutta())
    with pytest.raises(DriverError, match=backend):
        list(compatible_groups([point], 2))


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_batch_prepares_each_backend_without_changing_selection(backend):
    points = [member(0, backend=backend), member(1, backend=backend)]
    assert list(compatible_groups(points, 2)) == [tuple(points)]
    assert all(point.simulation.settings.backend == backend for point in points)


def test_batch_refuses_mixed_backend_scan():
    points = [member(0, backend="numpy_ref"), member(1, backend="numba_cpu")]
    with pytest.raises(DriverError, match="mixed backends"):
        list(compatible_groups(points, 2))


def test_batch_refuses_implicit_adaptive_execution():
    with pytest.raises(DriverError, match="explicit fixed schedule"):
        list(compatible_groups([member(0, fixed=False)], 2))


def test_batch_refuses_an_already_batched_input_contract():
    point = member(0, backend="numpy_ref")
    problem = point.simulation.translate()
    point.simulation._problem = dataclasses.replace(
        problem, shape=(2, *problem.shape[1:])
    )
    with pytest.raises(DriverError, match="independently translated singletons"):
        list(compatible_groups([point], 2))


def test_batch_refuses_spatial_geometry_before_selecting_a_backend():
    base = member(0, backend="numpy_ref").simulation
    simulation = Simulation.from_radial(
        base.grid,
        RadialGrid(0.6e-3, 8),
        SpatialMedium(),
        [],
        SpatialPulse(
            GaussianPulse(wavelength=835e-9, fwhm=100e-15, energy=1e-9), 60e-6
        ),
        settings=dataclasses.replace(
            base.settings,
            stats=StatsSpec(names=("energy",)),
            shock=True,
            prefactor_index=None,
        ),
        noise=None,
    )
    with pytest.raises(DriverError, match="fibre geometry"):
        list(compatible_groups([BatchMember(0, simulation)], 2))


def test_a_native_batch_groups_z_table_members():
    # The native backend's CPU build evaluates the z table itself (SEP 29 section
    # 6.6), so grouping no longer refuses it; a device build refuses at construction.
    point = member(0)
    problem = point.simulation.translate()
    point.simulation._problem = dataclasses.replace(
        problem,
        linear=dataclasses.replace(
            problem.linear, phi_representation="table", phi_constant=None
        ),
    )
    assert [
        [one.index for one in group] for group in compatible_groups([point], 2)
    ] == [[0]]


@pytest.mark.parametrize("size", [0, -1, True])
def test_batch_refuses_invalid_size(size):
    with pytest.raises(DriverError, match="positive integer"):
        list(compatible_groups([], size))


@requires_typecheck
@pytest.mark.parametrize("size", [1.5])
def test_batch_size_rejects_fractional_type(size):
    with pytest.raises(TypeCheckError, match="size"):
        list(compatible_groups([], size))


def test_batch_memory_counts_shared_tables_once():
    simulation = member(0).simulation
    first = simulation.translate()
    field = first.initial.field0[0] * 1.25
    second = simulation.with_input(field).translate()
    single = shared_table_bytes([first])
    assert shared_table_bytes([first, first]) == single
    assert shared_table_bytes([first, second]) == single + field.nbytes
    np.testing.assert_array_equal(second.initial.field0[0], field)


def test_named_table_layouts_ignore_values_but_preserve_names_and_shapes():
    first = {"rate": np.zeros(3), "enabled": True}
    second = {"rate": np.ones(3), "enabled": True}
    assert _structure(first) == _structure(second)
    assert _structure(first) != _structure({"rate": np.ones(4), "enabled": True})
    assert _structure(first) != _structure({"other_rate": np.zeros(3), "enabled": True})


def test_batch_memory_includes_parameter_arrays_and_shared_views():
    problem = member(0).simulation.translate()
    storage = np.arange(8, dtype=float)
    first = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, batch_params={"parameter": storage[:1]}
        ),
    )
    second = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, batch_params={"parameter": storage[1:2]}
        ),
    )
    assert (
        shared_table_bytes([first, second])
        == shared_table_bytes([problem]) + storage.nbytes
    )


@pytest.fixture
def observed_batch():
    histories = tuple(
        StepHistory(
            z=np.array([0.25, 0.5]),
            dz=np.array([0.25, 0.25]),
            error_norm=np.array([0.125, 0.25]) * (index + 1),
            accepted=np.ones(2, dtype=bool),
            wall_time=np.array([0.5, 1.0]),
        )
        for index in range(3)
    )
    observation = Observation(
        z=0.5,
        field=np.arange(24).reshape(3, 2, 1, 4).astype(complex),
        noise_phase=np.arange(24, dtype=float).reshape(3, 2, 1, 4),
        step_history=histories[-1],
        n_steps=2,
        n_rejected=0,
        step_stats=StepStats(
            z=np.array([0.25, 0.5]),
            step=np.array([0, 1]),
            values={"populations": np.arange(30).reshape(2, 3, 5)},
        ),
    )
    statistics = {"energy": np.arange(6).reshape(3, 2, 1)}
    return observation, statistics, histories


def test_batch_split_preserves_member_fields_noise_statistics_and_norms(observed_batch):
    observation, statistics, histories = observed_batch
    planes = split_observation(observation, statistics, histories=histories)
    assert len(planes) == 3
    for index, plane in enumerate(planes):
        assert plane.field is not None
        assert plane.field.shape == (2, 1, 4)
        np.testing.assert_array_equal(plane.field, observation.field[index])
        np.testing.assert_array_equal(plane.noise_phase, observation.noise_phase[index])
        np.testing.assert_array_equal(
            plane.stats["energy"], statistics["energy"][index]
        )
        np.testing.assert_array_equal(
            plane.step_stats.values["populations"],
            observation.step_stats.values["populations"][:, index],
        )
        np.testing.assert_array_equal(
            plane.step_history.norm, histories[index].error_norm
        )
        np.testing.assert_array_equal(plane.step_history.z, np.array([0.0, 0.25]))


def test_batch_split_requires_every_member_history(observed_batch):
    observation, statistics, histories = observed_batch
    with pytest.raises(DriverError, match="3 members but 2 histories"):
        split_observation(observation, statistics, histories=histories[:-1])


def test_batch_split_refuses_member_with_different_itinerary(observed_batch):
    observation, statistics, histories = observed_batch
    different = dataclasses.replace(histories[-1], dz=np.array([0.25, 0.125]))
    with pytest.raises(DriverError, match=r"member 2.*different step itinerary"):
        split_observation(
            observation, statistics, histories=(*histories[:-1], different)
        )


@pytest.mark.parametrize("value", [np.ones(2), np.asarray(1.0)])
def test_batch_split_refuses_wrong_statistic_member_axis(observed_batch, value):
    observation, _, histories = observed_batch
    with pytest.raises(DriverError, match=r"energy.*3 members"):
        split_observation(observation, {"energy": value}, histories=histories)


@pytest.mark.parametrize("value", [np.ones((2, 2, 5)), np.ones(2)])
def test_batch_split_refuses_wrong_periodic_member_axis(observed_batch, value):
    observation, statistics, histories = observed_batch
    rows = dataclasses.replace(observation.step_stats, values={"populations": value})
    observation = dataclasses.replace(observation, step_stats=rows)
    with pytest.raises(DriverError, match=r"periodic statistic.*3 members"):
        split_observation(observation, statistics, histories=histories)


def test_a_group_with_no_members_or_no_guide_has_nothing_to_disagree_about():
    """`refuse_mixed_guides` passes what it has no complaint about.

    Nothing to compare, and members that carry no general modal guide at all -- every
    ordinary batch -- are the two cases it must let through, so that the refusal is
    about guides and not about batches.
    """
    refuse_mixed_guides([])
    ordinary = [member(index).simulation.translate() for index in range(2)]
    refuse_mixed_guides(ordinary)

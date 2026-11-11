# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve species inventory units, route measures and diagnostic node budgets."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import material_case
from backends.julia.spatial_cases import spatial_case
from backends.julia.test_node_plan import _modal_nodes, _workspace_payload
from solprop.backends.julia._node_plan import plan_julia_nodes
from solprop.backends.julia._observations import lower_observations
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.exceptions import BackendError
from solprop.stats import StatsSpec

INVENTORY = (
    "species_density",
    "species_ionisation_energy",
    "species_peak_ionisation_rate",
)
DIAGNOSTICS = ("plasma_undershoot", "plasma_work_defect", "plasma_boundary_level")


def _request(case, names):
    return case.request.for_run(stats=StatsSpec(names=tuple(names), periodic=()))


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_complete_inventory_owns_potentials_and_mode_averaged_area(kind):
    case, _, _ = material_case(kind, "plasma")
    request = _request(case, INVENTORY)
    lowered = lower_observations(case.problem, request)
    population = lowered.populations
    assert population is not None and population.measure is not None
    assert population.species_labels == case.problem.responses.labels
    assert population.species_potentials.tolist() == [
        case.problem.responses.population_channels[0].ionisation_potential
    ]
    measure = population.measure
    assert measure.weights.tolist() == [1.0] and measure.taper == 1.0
    area = case.problem.transform.aeff
    assert area is not None and measure.area == float(area(0.0))
    assert [slot.shape for slot in lowered.slots] == [(1,)] * 3
    for array in (population.species_potentials, measure.weights):
        assert not array.flags.writeable
    assert lower_observations(case.problem, _request(case, ())).populations is None


def test_resolved_inventory_uses_physical_weights_without_map_inference():
    original, _, _ = material_case("field", "plasma")
    case = _modal_nodes(original, count=7)
    lowered = lower_observations(case.problem, _request(case, INVENTORY))
    population = lowered.populations
    assert population is not None and population.measure is not None
    measure = population.measure
    np.testing.assert_array_equal(measure.weights, case.problem.transform.node_weights)
    assert not np.shares_memory(measure.weights, case.problem.transform.node_weights)
    assert measure.area == measure.taper == 1.0
    missing = replace(case.problem.transform, node_weights=None)
    problem = replace(
        case.problem, route=replace(case.problem.route, transform=missing)
    )
    with pytest.raises(BackendError, match="physical node weights"):
        lower_observations(problem, _request(case, INVENTORY))


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_spatial_passive_inventory_retains_its_physical_measure(geometry):
    case, grid = spatial_case("field", geometry)
    lowered = lower_observations(case.problem, _request(case, INVENTORY), grid=grid)
    population = lowered.populations
    assert population is not None and population.measure is not None
    measure = population.measure
    spatial = case.problem.transform.geometry
    assert spatial is not None
    np.testing.assert_array_equal(measure.weights, spatial.spatial_weights)
    assert measure.area == measure.taper == 1.0
    assert not np.shares_memory(measure.weights, spatial.spatial_weights)
    assert population.channel_species == ()
    assert np.all(population.species_potentials == 0.0)


def test_default_diagnostics_add_exact_bounded_maximum_storage():
    case, _, _ = material_case("field", "plasma")
    case = _modal_nodes(case, count=7)
    old = _request(case, ("species_ionisation",))
    new = _request(case, ("species_ionisation", *DIAGNOSTICS))
    before = plan_julia_nodes(case.problem, old)
    after = plan_julia_nodes(case.problem, new, ExecutionOptions(node_chunk=3))
    entry = next(
        entry
        for entry in after.inventory
        if entry.name == "observations.plasma_diagnostics"
    )
    assert entry.per_node_bytes == entry.fixed_bytes == 6 * 8
    assert entry.full_volume_bytes == 0
    assert after.per_node_bytes == before.per_node_bytes + 6 * 8
    assert after.plan.chunk == 3
    # Two owned packed rows gain three scalar entries; maxima gain six.
    assert after.fixed_workspace_bytes == before.fixed_workspace_bytes + (2 * 3 + 6) * 8
    lowered = lower_observations(case.problem, new)
    assert lowered.periodic_indices == ()


@pytest.mark.parametrize("kind,response", [("envelope", "plasma"), ("field", "kerr")])
def test_diagnostic_admission_refuses_absent_field_plasma(kind, response):
    case, _, _ = material_case(kind, response)
    with pytest.raises(BackendError, match="field plasma channels"):
        lower_observations(case.problem, _request(case, DIAGNOSTICS))


@pytest.mark.skipif(
    os.environ.get("SOLPROP_TEST_JULIA") != "1",
    reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
)
@pytest.mark.medium
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_fresh_diagnostic_buffers_fit_the_complete_node_inventory(chunk):
    case, _, _ = material_case("field", "plasma")
    case = _modal_nodes(case, count=7)
    request = _request(case, (*INVENTORY, *DIAGNOSTICS))
    options = ExecutionOptions(
        node_chunk=chunk,
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
    )
    plan = plan_julia_nodes(case.problem, request, options)
    owner = JuliaBackend(case.problem, request, options=options)
    try:
        prepared = owner.prepare(case.problem, request)
        with prepared.start(Start.entrance(case.problem), owns=False) as session:
            ledger = session.memory_ledger()
            parts = [
                size
                for key, size in ledger.items()
                if key.endswith("diagnostics.parts")
            ]
            maxima = [
                size
                for key, size in ledger.items()
                if key.endswith("diagnostics.maxima")
            ]
            assert parts == [6 * chunk * 8] and maxima == [6 * 8]
            bound = (
                plan.per_node_bytes * chunk
                + plan.fixed_workspace_bytes
                + plan.full_volume_workspace_bytes
            )
            assert _workspace_payload(ledger) <= bound
            batch = session.advance()
            assert batch.events
            session.acknowledge(batch.receipt)
    finally:
        owner.close()

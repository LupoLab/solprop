# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify independent on-axis population transfer and unit-weight flux rows."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import InstantaneousRate, cpu_case, material_case
from backends.julia.test_node_plan import _modal_nodes, _workspace_payload
from solprop import fourier
from solprop.backends.events import ObservationReady
from solprop.backends.julia import _observations
from solprop.backends.julia._node_plan import plan_julia_nodes
from solprop.backends.julia._observations import lower_observations
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.exceptions import BackendError
from solprop.modes import MarcatiliMode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.responses import PlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stats import ON_AXIS_PLASMA_STATISTICS, StatsSpec
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Translation, assemble

AXIS = ON_AXIS_PLASMA_STATISTICS
FLUX = ("physical_energy", "analytic_flux", "spectral_flux")


def _request(case, names):
    return case.request.for_run(stats=StatsSpec(names=tuple(names), periodic=()))


def _axis_case(kind):
    """Keep an actual resolved mode and its independent scalar plasma channels."""
    original, grid, _ = material_case(kind, "plasma")
    mode = MarcatiliMode(30e-6, loss=False)
    response = PlasmaResponse if kind == "field" else EnvelopePlasmaResponse
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-9)
    case = assemble(
        grid,
        mode,
        [response("Ar", InstantaneousRate(scale=1e13), preionisation=0.02)],
        pulse.build(grid),
        density=1e24,
        controller=original.request.error_control.controller,
        save_planes=[grid.length],
        dz_init=grid.length / 8,
        dz_max=grid.length / 8,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 8),
        stats_spec=StatsSpec(names=(), periodic=()),
        ionisation_field_range=(1e7, 1e11),
    )
    return case, grid, mode


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_axis_transfer_owns_fine_synthesis_and_independent_species(kind):
    original, grid, mode = _axis_case(kind)
    case = _modal_nodes(original)
    request = _request(case, AXIS)
    lowered = lower_observations(case.problem, request, grid=grid, modes=(mode,))
    axis = lowered.axis_populations
    assert axis is not None and lowered.populations is None
    assert lowered.axis_synthesis is None and lowered.periodic_indices == ()
    rule = on_axis_rule(mode.dimlimits(0.0)[0])
    expected = mode_matrix((mode,), rule, 0.0, components=("x", "y")).synthesis
    # Preparation copies the same translated map and applies one exact pad scale.
    np.testing.assert_array_equal(
        axis.synthesis, expected * case.problem.grids.pad_scale
    )
    assert not axis.synthesis.flags.writeable
    assert axis.channel_species == (1,) and axis.species_densities == (1e24,)
    assert [slot.shape for slot in lowered.slots] == [(), (), (1,), (1,)]
    assert not np.shares_memory(axis.synthesis, case.problem.transform.synthesis)
    plans = [
        plan_julia_nodes(case.problem, request, ExecutionOptions(node_chunk=chunk))
        for chunk in (1, 3, 7)
    ]
    entries = [
        next(
            row for row in plan.inventory if row.name == "observations.axis_populations"
        )
        for plan in plans
    ]
    assert entries[0] == entries[1] == entries[2]
    assert entries[0].per_node_bytes == 0 and entries[0].fixed_bytes > 0


@pytest.mark.parametrize("problem_kind", ["averaged", "no_plasma", "no_modes"])
def test_axis_refuses_missing_context_before_map_reconstruction(
    problem_kind, monkeypatch
):
    case, _, mode = material_case("field", "plasma")
    if problem_kind != "averaged":
        case = _modal_nodes(case)
    if problem_kind == "no_plasma":
        empty, _, _ = material_case("field", "kerr")
        case = Translation(
            replace(case.problem, responses=empty.problem.responses), case.request
        )

    def forbidden(*_):
        raise AssertionError("axis reconstruction preceded admission")

    monkeypatch.setattr(_observations, "_axis", forbidden)
    with pytest.raises(BackendError, match="julia on-axis plasma statistics"):
        lower_observations(
            case.problem,
            _request(case, AXIS),
            modes=None if problem_kind == "no_modes" else (mode,),
        )


def test_spectral_flux_owns_its_density_weights_and_public_shape():
    case = cpu_case()
    lowered = lower_observations(case.problem, _request(case, FLUX))
    grid = case.problem.grids
    expected = fourier.spectral_density(
        np.ones(grid.omega.size, dtype=complex), grid.dt, kind=case.problem.grid_kind
    )
    np.testing.assert_array_equal(lowered.grid.spectral_density, expected)
    assert lowered.grid.spectral_density is not None
    assert not lowered.grid.spectral_density.flags.writeable
    assert [slot.shape for slot in lowered.slots] == [(), (), (grid.omega.size,)]
    assert lowered.axis_populations is None


@pytest.mark.parametrize("name", FLUX)
def test_unit_flux_refuses_nonidentity_weight_before_preparation(name):
    case = cpu_case()
    linear = replace(
        case.problem.linear, power_weight_constant=np.full((1, 1, 1), 1.01)
    )
    problem = replace(case.problem, linear=linear)
    with pytest.raises(BackendError, match="constant unit modal power weights"):
        lower_observations(problem, _request(case, (name,)))


@pytest.mark.skipif(
    os.environ.get("SOLPROP_TEST_JULIA") != "1",
    reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
)
@pytest.mark.medium
@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_axis_zero_field_retains_preionisation_and_complete_memory_bound(kind, chunk):
    original, grid, mode = _axis_case(kind)
    case = _modal_nodes(original)
    problem = replace(
        case.problem,
        initial=replace(
            case.problem.initial, field0=np.zeros_like(case.problem.initial.field0)
        ),
    )
    request = _request(case, AXIS)
    options = ExecutionOptions(
        node_chunk=chunk,
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
    )
    plan = plan_julia_nodes(problem, request, options)
    owner = JuliaBackend(problem, request, grid=grid, modes=(mode,), options=options)
    try:
        with owner.prepare(problem, request).start(
            Start.entrance(problem), owns=False
        ) as session:
            ledger = session.memory_ledger()
            assert any("axis_populations" in name for name in ledger)
            assert _workspace_payload(ledger) <= (
                plan.per_node_bytes * chunk
                + plan.fixed_workspace_bytes
                + plan.full_volume_workspace_bytes
            )
            batch = session.advance()
            (event,) = batch.events
            assert isinstance(event, ObservationReady)
            values = event.stats
            # Zero field has zero exposure and preserves the supplied initial
            # fraction. One density product and the sum of one species need 4u.
            np.testing.assert_allclose(
                values["on_axis_electron_density"],
                0.02 * 1e24,
                rtol=4 * np.finfo(float).eps,
                atol=0,
            )
            np.testing.assert_array_equal(
                values["on_axis_species_electron_density"],
                np.asarray([values["on_axis_electron_density"]]),
            )
            assert float(values["on_axis_peak_ionisation_rate"]) == 0.0
            assert np.all(values["on_axis_species_peak_ionisation_rate"] == 0.0)
            assert all(not value.flags.writeable for value in values.values())
            session.acknowledge(batch.receipt)
    finally:
        owner.close()


@pytest.mark.skipif(
    os.environ.get("SOLPROP_TEST_JULIA") != "1",
    reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
)
@pytest.mark.medium
def test_managed_unit_flux_preserves_spectral_density_and_scalar_units():
    case = cpu_case()
    request = _request(case, (*FLUX, "energy", "peak_power_total"))
    options = ExecutionOptions(
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
    )
    owner = JuliaBackend(case.problem, request, options=options)
    try:
        with owner.prepare(case.problem, request).start(
            Start.entrance(case.problem), owns=False
        ) as session:
            batch = session.advance()
            (event,) = batch.events
            assert isinstance(event, ObservationReady)
            values = event.stats
            field = np.where(
                case.problem.grids.active_mask, case.problem.initial.field0, 0.0
            )
            density = fourier.spectral_density(
                np.ones(field.shape[-1], dtype=complex),
                case.problem.grids.dt,
                kind=case.problem.grid_kind,
            )
            expected = density * np.sum(np.abs(field) ** 2, axis=(0, 1))
            # One mode, abs2 and the supplied density need fewer than 16 rounded
            # operations per bin. Allow 32u for the independent complex magnitude.
            np.testing.assert_allclose(
                values["spectral_flux"],
                expected,
                rtol=32 * np.finfo(float).eps,
                atol=0,
            )
            # These aliases use the same admitted identity metric. Their separate
            # sums are bounded by the number of frequency terms, not by bits.
            budget = 8 * field.shape[-1] * np.finfo(float).eps
            np.testing.assert_allclose(
                values["physical_energy"],
                np.sum(values["energy"]),
                rtol=budget,
                atol=0,
            )
            np.testing.assert_allclose(
                values["analytic_flux"],
                values["peak_power_total"],
                rtol=budget,
                atol=0,
            )
            assert all(not value.flags.writeable for value in values.values())
            session.acknowledge(batch.receipt)
    finally:
        owner.close()

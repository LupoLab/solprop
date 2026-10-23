# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent source and complete-run witnesses for the envelope host route."""

from contextlib import closing, nullcontext
from dataclasses import replace

import numpy as np
import pytest

from bounds import assert_within, scan_bound
from rate_oracles import adk_oracle
from solprop.backends.bodies import PLAIN, RateArrays
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.envelope_plasma import EnvelopeWorkspace
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.envelope_plasma import envelope_sources
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import (
    ConvergenceError,
    MaterialRangeWarning,
    SolpropWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.envelope_plasma import cycle_moments
from solprop.translate.problem import assemble
from solprop.translate.stage import scalars


@pytest.fixture(scope="module")
def host_case():
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (500e-9, 1.2e-6), 80e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-5).build(grid)
    field = pulse * np.array([1.0, 0.35j])[None, :, None] / np.sqrt(1 + 0.35**2)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        problem = assemble(
            grid,
            mode,
            [EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=0.1)],
            field,
            controller=PIController(rtol=1e-7, atol=1e-14),
            save_planes=[1e-6, 8e-6],
            dz_max=1e-6,
            dz_init=1e-6,
            ionisation_field_range=(1e9, 1e11),
        )
    return problem, grid


@pytest.fixture
def workspace(host_case):
    problem, _ = host_case
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    return EnvelopeWorkspace(
        assembly,
        (3, 2, 2, 9),
        embedded=True,
        parallel=False,
        flag="estimate",
        threads=1,
    )


def test_shared_envelope_bodies_resolve_degenerate_axes(host_case, workspace):
    problem, _ = host_case
    assert problem.responses.envelope_plasma is not None
    cycle = problem.responses.envelope_plasma.channels[0].cycle
    for ratio in (0.0, 1e-12, 0.35, 1 - 1e-12, 1.0):
        a = 3.713e10 * np.array([1, 1j * ratio]) * np.exp(0.317j)
        axes = PLAIN.envelope_axes(*a)
        mc, ms, status = PLAIN.envelope_moments(
            axes[0], axes[1], RateArrays.of(cycle.rate), *workspace.rules[0]
        )
        assert status == 0
        # Independent stable Gram/SVD arithmetic changes the table argument by
        # rounding; this budget is below the independent 2e-6 angular budget.
        np.testing.assert_allclose(
            [mc, ms], cycle_moments(3.713e10, ratio, cycle), rtol=2e-10, atol=1e-30
        )
        current = np.array(PLAIN.envelope_loss(axes, mc, ms, 1.0))
        np.testing.assert_allclose(
            np.vdot(a, current).real / 2, mc + ratio**2 * ms, rtol=2e-13, atol=1e-30
        )
    assert PLAIN.envelope_loss(PLAIN.envelope_axes(0j, 0j), 0, 0, 0) == (0j, 0j)


def test_one_source_evaluation_matches_independent_oracle(host_case, workspace):
    problem, _ = host_case
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    time = np.linspace(-1, 1, 9)
    field = np.array(
        [
            np.array([1, 0.35j])[:, None]
            * np.exp(-(time**2))
            * 3.713e10
            * np.exp(1j * phase)
            for phase in (0.0, 0.31, 1.1)
        ]
    )
    bundle = scalars(problem, 0.0, 0.0)
    window = np.ones(9)
    window.setflags(write=False)
    reference = envelope_sources(field, assembly, bundle, 1e-15, window)
    workspace.respond(field.reshape(6, 9), bundle, 1e-15, window)
    for stream in range(2):
        # Compiled/plain sums must agree more tightly than the physical oracle.
        np.testing.assert_allclose(
            workspace.sources[stream].reshape(field.shape),
            reference[stream],
            rtol=2e-10,
            atol=0,
        )
    response = EnvelopePlasmaResponse(
        "Ar", assembly.channels[0].rate, preionisation=0.1
    )
    physical = response.sources(field[0], bundle.plasma_densities[0], dt=1e-15)
    np.testing.assert_allclose(reference[0][0], physical.drude_drive, rtol=2e-6, atol=0)
    np.testing.assert_allclose(
        reference[1][0], physical.loss_current, rtol=2e-6, atol=0
    )


def test_population_recomputed_and_partial_rows_zero(host_case, workspace):
    problem, _ = host_case
    bundle = scalars(problem, 0.0, 0.0)
    window = np.ones(9)
    window.setflags(write=False)
    field = np.full((6, 9), 4e10 + 0j)
    workspace.respond(field, bundle, 1e-15, window)
    first = workspace.sources.copy()
    workspace.respond(field[:2] * 0, bundle, 1e-15, window)
    # Structural: a zero field has exactly zero sources, and the rows beyond the
    # partial chunk are cleared.
    np.testing.assert_array_equal(workspace.sources, 0)
    workspace.respond(field, bundle, 1e-15, window)
    # The same evaluation repeated, the population recomputed rather than kept:
    # within one evaluation's scan chain.
    assert_within(workspace.sources, first, scan_bound(field.shape[-1]), what="repeat")
    assert sum(workspace.memory_ledger().values()) > workspace.sources.nbytes


@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
def test_envelope_population_values_do_not_depend_on_retaining_node_scans(
    host_case, backend_class
):
    """The optional on-axis reuse record must not alter the node statistics."""
    problem, grid = host_case
    with closing(backend_class(problem, grid=grid)) as backend:
        backend.set_initial(problem.initial.field0, None)
        field = problem.initial.field0[0]
        plain = backend._species_statistics(field)
        scans = {}
        retained = backend._species_statistics(field, scans, member=3)
        assert set(scans) == {3}
        fractions, peaks = scans[3]
        assert (
            fractions.shape
            == peaks.shape
            == (
                len(problem.responses.population_channels),
                problem.transform.n_nodes,
            )
        )
        assert np.all(fractions >= 0.1)
        assert np.any(peaks > 0.0)
        for name, value in plain.items():
            # Retaining copies happens after the same population calculation, which
            # is repeated: within one statistics evaluation's scan chain.
            np.testing.assert_allclose(
                retained[name],
                value,
                rtol=scan_bound(problem.grids.n_to),
                atol=0,
                err_msg=name,
            )


@pytest.mark.medium
@pytest.mark.parametrize("amplitude_scale", [1.0, 10.0])
def test_one_eight_steps_and_complete_host_run(host_case, amplitude_scale):
    problem, grid = host_case
    problem = replace(
        problem,
        initial=replace(
            problem.initial, field0=problem.initial.field0 * amplitude_scale
        ),
        stepper=replace(
            problem.stepper, schedule=FixedSchedule.uniform(0.0, 8e-6, 1e-6)
        ),
    )
    if amplitude_scale == 10.0:
        assembly = problem.responses.envelope_plasma
        assert assembly is not None
        modal = np.fft.ifft(problem.initial.field0[0], axis=-1)
        physical = (problem.transform.synthesis @ modal.reshape(1, -1)).reshape(
            2, grid.n_to
        )
        response = EnvelopePlasmaResponse(
            "Ar", assembly.channels[0].rate, preionisation=0.1
        )
        # Qualify a run with substantial new electrons as well as the weaker
        # prescribed-density case. This is a model-execution, not regime, witness.
        assert response.ionised_fraction(physical, dt=grid.dt_fine)[-1] > 0.2
    results = []
    for cls in (NumpyRefBackend, NumbaCpuBackend):
        backend = (
            NumpyRefBackend(problem, grid=grid)
            if cls is NumpyRefBackend
            else NumbaCpuBackend(
                problem,
                grid=grid,
                plan_flag="estimate",
                threads=1,
                fftw_threads=1,
                blas_threads=1,
            )
        )
        try:
            backend.set_initial(problem.initial.field0, None)
            first = backend.propagate_to(1e-6)
            one = backend.observe().require_field().copy()
            final = backend.propagate_to(8e-6)
            eight = backend.observe().require_field().copy()
            assert first.met_tolerance and final.met_tolerance
            assert first.n_steps == 1 and final.n_steps == 7
            assert np.all(np.isfinite(eight))
            assert np.linalg.norm(eight - problem.initial.field0) > 0
            results.append((one, eight))
        finally:
            backend.close()
    # Source discrepancy is below 2e-10 and perturbs the field weakly over 8 um.
    for level in range(2):
        scale = np.max(np.abs(results[0][level]))
        np.testing.assert_allclose(
            results[1][level], results[0][level], rtol=2e-10, atol=2e-12 * scale
        )


def test_prescribed_density_off_carrier_drude_assembly(host_case):
    problem, grid = host_case
    initial = replace(problem.initial, field0=problem.initial.field0 * 1e-6)
    window = np.ones(problem.grids.n_to)
    window.setflags(write=False)
    problem = replace(
        problem, initial=initial, grids=replace(problem.grids, t_window_fine=window)
    )
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        bundle = scalars(problem, 0.0, 0.0)
        lab = problem.initial.field0
        actual = backend._nonlinear(lab, bundle, np.ones_like(lab))
        assembly = problem.responses.envelope_plasma
        assert assembly is not None
        channel = assembly.channels[0]
        transform = problem.transform
        assert np.all(problem.grids.t_window_fine == 1)
        coefficient = channel.charge_mass_ratio * bundle.plasma_densities[0] * 0.1
        overlap = float((transform.projection @ transform.synthesis)[0, 0])
        expected = (
            backend._prefactor(bundle)
            * problem.grids.omega_window
            * assembly.factors.drude
            * coefficient
            * overlap
            * lab
        )
        # Prescribed constant density diagonalises the source; error is FFT rounding.
        np.testing.assert_allclose(
            actual, expected, rtol=2e-12, atol=1e-14 * np.max(np.abs(expected))
        )
    finally:
        backend.close()


def test_problem_rejects_source_factors_from_another_grid(host_case):
    problem, _ = host_case
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    for factors, message in (
        (replace(assembly.factors, omega0=assembly.factors.omega0 * 1.1), "grid"),
        (replace(assembly.factors, drude=assembly.factors.drude * 1.1), "multipliers"),
    ):
        changed = replace(assembly, factors=factors)
        responses = replace(
            problem.responses, envelope_plasma=changed, spectral_responses=(changed,)
        )
        with pytest.raises(TranslationError, match=message):
            replace(problem, responses=responses).validate()


@pytest.mark.medium
def test_direct_simulation_records_carrier_approximation_and_runs(host_case):
    original, grid = host_case
    assembly = original.responses.envelope_plasma
    assert assembly is not None
    mode = MarcatiliMode(125e-6, "Ar", 1.0, loss=False)
    simulation = Simulation(
        grid,
        mode,
        [EnvelopePlasmaResponse("Ar", assembly.channels[0].rate, preionisation=0.1)],
        original.initial.field0[0],
        settings=SimulationSettings(
            controller=original.stepper.controller,
            save_planes=[8e-6],
            dz_init=1e-6,
            dz_max=1e-6,
            shock=False,
            backend="numba_cpu",
        ),
    )
    with pytest.warns((MaterialRangeWarning, SolpropWarning)) as recorded:
        problem = simulation.translate()
    assert sum("shock=False" in str(item.message) for item in recorded) == 1
    metadata = simulation.metadata()
    record = metadata.configuration["envelope_plasma"]
    assert isinstance(record, dict)
    assert record["carrier_policy"] == "frozen-carrier"
    assert record["rate_convention"] == "cycle-averaged-from-instantaneous-isotropic"
    compiled = problem.responses.envelope_plasma
    assert compiled is not None
    assert (
        record["channels"][0]["cycle_identity"] == compiled.channels[0].cycle.identity
    )
    active = original.grids.active_mask
    assert original.responses.prefactor_constant is not None
    assert problem.responses.prefactor_constant is not None
    expected = original.responses.prefactor_constant[..., active] * (
        grid.omega0 / grid.omega[active]
    )
    # Only the explicit numerator is frozen: both paths retain the same
    # frequency-dependent refractive index. Division rearrangement costs 2 ulps.
    np.testing.assert_allclose(
        problem.responses.prefactor_constant[..., active], expected, rtol=5e-16, atol=0
    )
    handler = simulation.run()
    assert handler.n_saved == 2


def test_node_budget_accounts_for_envelope_histories(host_case):
    original, grid = host_case
    assembly = original.responses.envelope_plasma
    assert assembly is not None
    # The registry's argon fit range warned here; optmat's argon is qualified over
    # the band (SEP 3, Phase 4a).
    with pytest.raises(TranslationError, match="one envelope plasma source history"):
        assemble(
            grid,
            MarcatiliMode(125e-6, "Ar", 1.0, loss=False),
            [EnvelopePlasmaResponse("Ar", assembly.channels[0].rate)],
            original.initial.field0[0],
            controller=original.stepper.controller,
            save_planes=[8e-6],
            dz_init=1e-6,
            dz_max=1e-6,
            node_budget=1,
        )


def test_host_refuses_unresolved_angular_source_and_recovers(host_case, workspace):
    problem, _ = host_case
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    bundle = scalars(problem, 0.0, 0.0)
    window = np.ones(9)
    window.setflags(write=False)
    field = np.zeros((3, 2, 9), complex)
    field[:, 0] = 1e16
    with pytest.raises(ConvergenceError, match="512"):
        envelope_sources(field, assembly, bundle, 1e-15, window)
    with pytest.raises(ConvergenceError, match="angular refinement"):
        workspace.respond(field.reshape(6, 9), bundle, 1e-15, window)
    populations = np.full((2, 3), np.nan)
    with pytest.raises(ConvergenceError, match="angular refinement"):
        workspace.populations(
            field.reshape(6, 9),
            bundle.plasma_preionisation,
            1e-15,
            window,
            populations,
        )
    assert np.all(np.isnan(populations))
    field[:] = 0
    workspace.respond(field.reshape(6, 9), bundle, 1e-15, window)
    workspace.populations(
        field.reshape(6, 9), bundle.plasma_preionisation, 1e-15, window, populations
    )
    # Structural: a zero field leaves the prescribed populations and has exactly zero
    # sources.
    np.testing.assert_array_equal(populations[0], bundle.plasma_preionisation[0])
    np.testing.assert_array_equal(populations[1], 0)
    np.testing.assert_array_equal(workspace.status, 0)
    np.testing.assert_array_equal(workspace.sources, 0)
    drive, loss = envelope_sources(field, assembly, bundle, 1e-15, window)
    np.testing.assert_array_equal(drive, 0)
    np.testing.assert_array_equal(loss, 0)
    assert (
        PLAIN.envelope_moments(np.nan, 0.0, workspace.rates[0], *workspace.rules[0])[2]
        == 1
    )


def test_subnormal_zero_rate_field_has_finite_sources(host_case, workspace):
    problem, _ = host_case
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    bundle = scalars(problem, 0.0, 0.0)
    window = np.ones(9)
    window.setflags(write=False)
    field = np.full((3, 2, 9), 1e-320 + 0j)
    reference = envelope_sources(field, assembly, bundle, 1e-15, window)
    workspace.respond(field.reshape(6, 9), bundle, 1e-15, window)
    assert np.all(np.isfinite(workspace.sources))
    np.testing.assert_array_equal(reference[1], 0)
    np.testing.assert_array_equal(workspace.sources[1], 0)
    # Both paths use the same few products for the prescribed initial population;
    # no reciprocal of a subnormal field should enter either source. Four roundings,
    # and two subnormal quanta for a result that underflows.
    np.testing.assert_allclose(
        workspace.sources[0].reshape(field.shape),
        reference[0],
        rtol=4 * np.finfo(float).eps,
        atol=2 * np.nextafter(0.0, 1.0),
    )


@pytest.mark.medium
def test_paired_backend_executes_envelope_sources(host_case):
    native = pytest.importorskip("solprop_native")
    problem, grid = host_case
    reference = NumpyRefBackend(problem, grid=grid)
    actual = native.NativeBackend(problem, grid=grid)
    try:
        for owner in (reference, actual):
            owner.set_initial(problem.initial.field0, None)
            owner.propagate_to(1e-6)
        target = reference.observe().require_field()
        # One seven-stage step includes temporal FFT and modal projection sums;
        # the independent source witnesses separately bound angular error.
        bound = 128 * grid.n_to * 7 * np.finfo(float).eps
        np.testing.assert_allclose(
            actual.observe().require_field(),
            target,
            rtol=bound,
            atol=bound * np.max(np.abs(target)),
        )
    finally:
        actual.close()
        reference.close()

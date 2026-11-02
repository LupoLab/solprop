# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s envelope host: its workspace and the bodies its kernels compile."""

from contextlib import closing

import numpy as np
import pytest

from backends import test_envelope_plasma_geometries as geometries
from backends import test_envelope_plasma_host as host
from backends.host_cases_envelope_plasma import geometry_case
from bounds import assert_within, scan_bound
from solprop.backends.bodies import PLAIN, RateArrays
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.envelope_plasma import EnvelopeWorkspace
from solprop.backends.numpy_ref.envelope_plasma import envelope_sources
from solprop.exceptions import ConvergenceError
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.translate.envelope_plasma import cycle_moments
from solprop.translate.stage import scalars

host_case = host.host_case
envelope_mixture = geometries.envelope_mixture


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


def test_envelope_population_values_do_not_depend_on_retaining_node_scans(host_case):
    """The optional on-axis reuse record must not alter the node statistics."""
    problem, grid = host_case
    with closing(NumbaCpuBackend(problem, grid=grid)) as backend:
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


@pytest.mark.parametrize("geometry", ["modal", "radial", "free"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_source_streams_are_sized_by_the_chunk(envelope_mixture, geometry, n_pol):
    problem, grid = geometry_case(envelope_mixture, geometry, n_pol)
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert ledger["envelope_sources"] == (
        2 * problem.transform.node_chunk * n_pol * grid.n_to * 16
    )

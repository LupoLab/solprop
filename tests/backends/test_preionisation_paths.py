# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preionisation along z, per species, on every backend.

The seam already carried the initial ionised fraction as a per-channel table along z;
these tests make the front end supply a real path and check that each backend reads
it per stage: a constant path is the number, to the run's rounding, a stepped and a
smooth path agree across the backends within the derived plasma bounds the mixture
ladder already uses,
steps land on the path's kinks, and at a field too weak to ionise the final electron
density is the analytic ``f0(z) rho(z)``.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from bounds import assert_within
from mixture_cases import AnalyticRate, mixture_case
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.profiles import Constant, FromCallable, PiecewiseLinear
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

NAMES = (
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "electron_density",
    "peak_ionisation_rate",
)

FIELD_LENGTH = 1e-3
"""The mixture case's fibre, m; its fixed ladder takes steps of ``dz_init = 1e-5``."""

FIELD_STEPS = 8
FIELD_DZ = 1e-5


def stepped(length, low, high, start, end):
    """Return a fraction that ramps from `low` to `high` between two kinks, in m."""
    return PiecewiseLinear(
        np.array([0.0, start, end, length]), np.array([low, low, high, high])
    )


def smooth(length, low, high):
    """Return a fraction rising smoothly along the fibre, with no kink at all."""
    return FromCallable(
        lambda z: (
            low + (high - low) * np.sin(0.5 * np.pi * np.asarray(z) / length) ** 2
        ),
        (0.0, length),
    )


def run(backend_class, problem, grid, stop, *, stats=NAMES):
    """Propagate to `stop` and return the field, the summary, the history and stats."""
    backend = (
        NumbaCpuBackend(problem, grid=grid, threads=1)
        if backend_class is NumbaCpuBackend
        else backend_class(problem, grid=grid)
    )
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(stop)
        observed = backend.observe()
        return (
            observed.require_field().copy(),
            summary,
            observed.step_history,
            backend.stats(StatsSpec(names=stats)),
        )
    finally:
        backend.close()


def field_case(preionisation):
    """Return the two-population field case on its fixed eight-step ladder."""
    _, grid, _, _, problem = mixture_case(
        preionisation=preionisation,
        schedule=FixedSchedule.uniform(0.0, FIELD_LENGTH, FIELD_DZ),
    )
    return problem, grid


# Kinks on the ladder's edges (a fixed schedule must land on every one), inside the
# eight steps that are taken.
FIELD_PATHS = {
    "stepped-smooth": (
        stepped(FIELD_LENGTH, 0.02, 0.2, 3 * FIELD_DZ, 5 * FIELD_DZ),
        smooth(FIELD_LENGTH, 0.05, 0.3),
    ),
    "smooth-stepped": (
        smooth(FIELD_LENGTH, 0.01, 0.1),
        stepped(FIELD_LENGTH, 0.3, 0.05, 2 * FIELD_DZ, 6 * FIELD_DZ),
    ),
}


@pytest.mark.medium
def test_a_constant_path_is_the_number_on_every_backend(backend_class):
    """A constant path and its number: one method on one schedule, so one run twice.

    Held to the mixture ladder's bound below; the schedule's positions are exact.
    """
    numbers, number_grid = field_case((0.03, 0.1))
    paths, _ = field_case((Constant(0.03), Constant(0.1)))
    stop = FIELD_STEPS * FIELD_DZ
    bound = 64 * number_grid.n_to * 2 * 7 * FIELD_STEPS * np.finfo(float).eps
    expected = run(backend_class, numbers, number_grid, stop)
    actual = run(backend_class, paths, number_grid, stop)
    assert_within(actual[0], expected[0], bound, what="field")
    np.testing.assert_array_equal(actual[2].z, expected[2].z)
    for name in NAMES:
        np.testing.assert_allclose(
            actual[3][name],
            expected[3][name],
            rtol=bound,
            atol=bound if name == "species_ionisation" else 0.0,
            err_msg=name,
        )


@pytest.mark.medium
@pytest.mark.parametrize("paths", sorted(FIELD_PATHS))
def test_field_plasma_paths_agree_with_the_oracle(paths, equivalence_backend_class):
    problem, grid = field_case(FIELD_PATHS[paths])
    stop = FIELD_STEPS * FIELD_DZ
    # The bound of tests/backends/test_mixture_ladder.py, unchanged: seven stages per
    # step, each with two FFTs, a cubic evaluation and two accumulated current
    # integrals per channel, 64 eps per fine sample, per channel, per stage and step.
    # A path changes only the per-stage scalar each backend reads from the same table.
    bound = 64 * grid.n_to * 2 * 7 * FIELD_STEPS * np.finfo(float).eps
    assert bound > 0
    ours = run(equivalence_backend_class, problem, grid, stop)
    oracle = run(NumpyRefBackend, problem, grid, stop)
    assert ours[1].n_steps == oracle[1].n_steps == FIELD_STEPS
    difference = float(np.linalg.norm(ours[0] - oracle[0]) / np.linalg.norm(oracle[0]))
    assert difference < bound
    for name in NAMES:
        np.testing.assert_allclose(
            ours[3][name],
            oracle[3][name],
            rtol=bound,
            atol=bound if name == "species_ionisation" else 0.0,
        )
    # The path is read: the final fraction starts from the local initial fraction.
    initial = np.array(
        [float(path(np.array([stop]))[0]) for path in FIELD_PATHS[paths]]
    )
    # Shape (n_batch, n_species, n_nodes); one member, one node here.
    final = oracle[3]["species_ionisation"][0, :, 0]
    assert np.all(final >= initial * (1 - 1e-12))


ENVELOPE_LENGTH = 8e-6


@dataclass(frozen=True)
class _InstantaneousRate(AnalyticRate):
    """The synthetic power law, declared an instantaneous isotropic rate."""

    convention: str = "instantaneous-isotropic"


def envelope_case(preionisation):
    """Return the two-population envelope case on a four-step fixed ladder."""
    mixture, _, _, _, _ = mixture_case(kerr=False, preionisation=preionisation)
    responses = tuple(
        EnvelopePlasmaResponse(
            component.species,
            build_rate_table(
                _InstantaneousRate(scale=(index + 1) * 1e13), (1e7, 1e11), rtol=1e-7
            ),
            preionisation=component.preionisation,
            species=component.label,
        )
        for index, component in enumerate(mixture.components)
    )
    grid = EnvelopeGrid.from_limits(ENVELOPE_LENGTH, 800e-9, (500e-9, 1.2e-6), 80e-15)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    initial = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7).build(grid)
    step = ENVELOPE_LENGTH / 4
    problem = assemble(
        grid,
        mode,
        responses,
        initial,
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[ENVELOPE_LENGTH],
        dz_max=step,
        dz_init=step,
        schedule=FixedSchedule.uniform(0.0, ENVELOPE_LENGTH, step),
    )
    return problem, grid


ENVELOPE_PATHS = {
    "stepped-smooth": (
        stepped(ENVELOPE_LENGTH, 0.02, 0.2, ENVELOPE_LENGTH / 4, ENVELOPE_LENGTH / 2),
        smooth(ENVELOPE_LENGTH, 0.05, 0.3),
    ),
    "smooth-stepped": (
        smooth(ENVELOPE_LENGTH, 0.01, 0.1),
        stepped(
            ENVELOPE_LENGTH, 0.3, 0.05, ENVELOPE_LENGTH / 2, 3 * ENVELOPE_LENGTH / 4
        ),
    ),
}


@pytest.mark.medium
@pytest.mark.parametrize("paths", sorted(ENVELOPE_PATHS))
def test_envelope_plasma_paths_agree_with_the_oracle(paths, equivalence_backend_class):
    problem, grid = envelope_case(ENVELOPE_PATHS[paths])
    assert problem.flags.n_channels == 2
    # The bound of tests/backends/test_envelope_plasma_geometries.py for the same
    # two-population envelope case: the polynomial rate's integrals converge to
    # rounding, and FFTs, two-channel scans and projections contribute under 128 eps
    # per sample per RK stage; here over four steps rather than two.
    bound = 128 * grid.n_to * 7 * 4 * np.finfo(float).eps
    assert bound > 0
    names = ("species_ionisation", "species_electron_density", "electron_density")
    ours = run(equivalence_backend_class, problem, grid, ENVELOPE_LENGTH, stats=names)
    oracle = run(NumpyRefBackend, problem, grid, ENVELOPE_LENGTH, stats=names)
    assert ours[1].n_steps == oracle[1].n_steps == 4
    np.testing.assert_allclose(
        ours[0], oracle[0], rtol=bound, atol=bound * np.max(np.abs(oracle[0]))
    )
    for name in names:
        np.testing.assert_allclose(ours[3][name], oracle[3][name], rtol=bound, atol=0)


@pytest.mark.medium
def test_a_constant_envelope_path_is_the_number(backend_class):
    """One run twice, held to the envelope geometry ladder's four-step bound."""
    numbers, grid = envelope_case((0.03, 0.1))
    paths, _ = envelope_case((Constant(0.03), Constant(0.1)))
    bound = 128 * grid.n_to * 7 * 4 * np.finfo(float).eps
    expected = run(backend_class, numbers, grid, ENVELOPE_LENGTH, stats=())
    actual = run(backend_class, paths, grid, ENVELOPE_LENGTH, stats=())
    assert_within(actual[0], expected[0], bound)


@pytest.mark.medium
def test_adaptive_steps_land_on_an_interior_kink(backend_class):
    # Kinks that are no multiple of any step the controller would choose: without
    # them among Phi's hard kinks the adaptive run steps straight across both.
    first, second = 0.3712345e-3, 0.3812345e-3
    _, grid, _, _, problem = mixture_case(
        preionisation=(
            stepped(FIELD_LENGTH, 0.02, 0.2, first, second),
            Constant(0.1),
        )
    )
    assert problem.linear.phi_representation == "constant"
    # Structural: the declared kinks are carried as given.
    np.testing.assert_array_equal(problem.linear.hard_kinks, [first, second, 1e-3])
    _, summary, history, _ = run(backend_class, problem, grid, FIELD_LENGTH, stats=())
    assert summary.met_tolerance
    accepted = np.asarray(history.z)[np.asarray(history.accepted)]
    assert first in accepted
    assert second in accepted


@pytest.mark.medium
def test_the_fraction_follows_the_path_where_the_pulse_cannot_ionise(backend_class):
    # The synthetic rate is exactly zero below its 1e7 V/m threshold, and this pulse
    # peaks far below it, so the final fraction is the initial one as the
    # preionisation table gives it: species electron density = f0(z) rho(z).
    paths = (
        stepped(FIELD_LENGTH, 0.02, 0.2, 3 * FIELD_DZ, 5 * FIELD_DZ),
        smooth(FIELD_LENGTH, 0.05, 0.3),
    )
    problem, grid = field_case(paths)
    weak = dataclasses.replace(problem.initial, field0=problem.initial.field0 * 1e-9)
    problem = dataclasses.replace(problem, initial=weak)
    names = ("species_ionisation", "species_electron_density", "species_density")
    for plane in (2, 4, 7):
        stop = plane * FIELD_DZ
        _, _, _, stats = run(backend_class, problem, grid, stop, stats=names)
        f0 = np.array([float(path(np.array([stop]))[0]) for path in paths])
        # The tables reproduce a linear piece to rounding and the smooth path to
        # the translation tolerance (1e-12 relative, DEFAULT_TOL's order).
        np.testing.assert_allclose(
            stats["species_ionisation"][0, :, 0],
            f0,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            stats["species_electron_density"],
            f0 * stats["species_density"],
            rtol=1e-11,
        )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preionisation along z, per species, on every backend.

The seam already carried the initial ionised fraction as a per-channel table along z;
these tests make the front end supply a real path and check that each backend reads
it per stage: a constant path is the number, to the run's rounding, steps land on the
path's kinks, and at a field too weak to ionise the final electron density is the
analytic ``f0(z) rho(z)``. A stepped and a smooth path on native against the reference,
within the derived plasma bounds the mixture ladder uses, are the catalogue's rows
(``backends/mixture_rows.py``, ``mixture preionisation ...``).
"""

import dataclasses

import numpy as np
import pytest

from backends.preionisation_cases import (
    ENVELOPE_LENGTH,
    FIELD_DZ,
    FIELD_LENGTH,
    FIELD_PATHS,
    FIELD_STEPS,
    NAMES,
    envelope_case,
    field_case,
    smooth,
    stepped,
)
from bounds import assert_within
from mixture_cases import mixture_translation
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.profiles import Constant
from solprop.stats import StatsSpec


def run(backend_class, problem, run_request, grid, stop, *, stats=NAMES):
    """Propagate to `stop` through the backend's own session, `stop` made a plane of
    the run's schedule, and return the field, the summary, the history and the
    statistics there."""
    planes = np.asarray(run_request.stepper.save_planes, dtype=float)
    if stop not in planes:
        run_request = run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper, save_planes=np.sort(np.append(planes, stop))
            )
        )
    run_request = run_request.for_run(stats=StatsSpec(names=stats))
    backend = backend_class(problem, run_request, grid=grid)
    plane = session_run(backend, problem, run_request, owns=True, until=stop).final
    assert plane.z == stop
    return plane.field, plane.summary, plane.history, plane.stats


@pytest.mark.medium
def test_a_constant_path_is_the_number_on_every_backend(backend_class):
    """A constant path and its number: one method on one schedule, so one run twice.

    Held to the mixture ladder's bound below; the schedule's positions are exact.
    """
    numbers, numbers_request, number_grid = field_case((0.03, 0.1))
    paths, paths_request, _ = field_case((Constant(0.03), Constant(0.1)))
    stop = FIELD_STEPS * FIELD_DZ
    bound = 64 * number_grid.n_to * 2 * 7 * FIELD_STEPS * np.finfo(float).eps
    expected = run(backend_class, numbers, numbers_request, number_grid, stop)
    actual = run(backend_class, paths, paths_request, number_grid, stop)
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


@pytest.mark.parametrize("paths", sorted(FIELD_PATHS))
def test_the_final_fraction_starts_from_the_path(paths):
    """The path is read: each final fraction is at least its local initial fraction."""
    problem, run_request, grid = field_case(FIELD_PATHS[paths])
    stop = FIELD_STEPS * FIELD_DZ
    oracle = run(NumpyRefBackend, problem, run_request, grid, stop)
    assert oracle[1].n_steps == FIELD_STEPS
    initial = np.array(
        [float(path(np.array([stop]))[0]) for path in FIELD_PATHS[paths]]
    )
    # Shape (n_species, n_nodes); one node here.
    final = oracle[3]["species_ionisation"][:, 0]
    assert np.all(final >= initial * (1 - 1e-12))


@pytest.mark.medium
def test_a_constant_envelope_path_is_the_number(backend_class):
    """One run twice, held to the envelope geometry ladder's four-step bound."""
    numbers, numbers_request, grid = envelope_case((0.03, 0.1))
    paths, paths_request, _ = envelope_case((Constant(0.03), Constant(0.1)))
    bound = 128 * grid.n_to * 7 * 4 * np.finfo(float).eps
    expected = run(
        backend_class, numbers, numbers_request, grid, ENVELOPE_LENGTH, stats=()
    )
    actual = run(backend_class, paths, paths_request, grid, ENVELOPE_LENGTH, stats=())
    assert_within(actual[0], expected[0], bound)


@pytest.mark.medium
def test_adaptive_steps_land_on_an_interior_kink(backend_class):
    # Kinks that are no multiple of any step the controller would choose: without
    # them among Phi's hard kinks the adaptive run steps straight across both.
    first, second = 0.3712345e-3, 0.3812345e-3
    _, grid, _, _, (problem, run_request) = mixture_translation(
        preionisation=(
            stepped(FIELD_LENGTH, 0.02, 0.2, first, second),
            Constant(0.1),
        )
    )
    assert problem.linear.phi_representation == "constant"
    # Structural: the declared kinks are carried as given.
    np.testing.assert_array_equal(problem.linear.hard_kinks, [first, second, 1e-3])
    _, summary, history, _ = run(
        backend_class, problem, run_request, grid, FIELD_LENGTH, stats=()
    )
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
    problem, run_request, grid = field_case(paths)
    weak = dataclasses.replace(problem.initial, field0=problem.initial.field0 * 1e-9)
    problem = dataclasses.replace(problem, initial=weak)
    names = ("species_ionisation", "species_electron_density", "species_density")
    for plane in (2, 4, 7):
        stop = plane * FIELD_DZ
        _, _, _, stats = run(
            backend_class, problem, run_request, grid, stop, stats=names
        )
        f0 = np.array([float(path(np.array([stop]))[0]) for path in paths])
        # The tables reproduce a linear piece to rounding and the smooth path to
        # the translation tolerance (1e-12 relative, DEFAULT_TOL's order).
        np.testing.assert_allclose(
            stats["species_ionisation"][:, 0],
            f0,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            stats["species_electron_density"],
            f0 * stats["species_density"],
            rtol=1e-11,
        )

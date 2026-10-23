# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One convolution with independent Raman amplitudes and decay rates."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mixture_cases import mixture_case
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.exceptions import BackendError
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.stepper import FixedSchedule
from solprop.translate.problem import Flags
from solprop.translate.responses import compile_responses
from solprop.translate.stage import scalars


def raman_refreshes(backend):
    """Read the selected backend's owned Raman kernel refresh counter."""
    return backend._raman_transforms


def prepare_raman(backend, rates, densities):
    """Exercise the actual cached Raman owner without a propagation."""
    backend._raman_spectrum(rates, densities)


@pytest.fixture
def case():
    mixture, grid, mode, responses, problem = mixture_case()
    extra = tuple(
        RamanResponse(
            models=(
                DampedOscillator((i + 1) * 1e-37, (i + 1) * 2e14, undamped_dephasing()),
            ),
            thg=False,
            species=label,
        )
        for i, label in enumerate(mixture.labels)
    )
    compiled = compile_responses(
        (*responses, *extra),
        grid,
        mode,
        density=mode.density,
        z_max=grid.length,
        mixture=mixture,
    )
    problem = dataclasses.replace(problem, responses=compiled)
    problem = dataclasses.replace(
        problem,
        flags=Flags.from_content(
            shape=problem.shape,
            linear=problem.linear,
            transform=problem.transform,
            responses=compiled,
            initial=problem.initial,
        ),
    )
    return grid, problem


def test_one_raman_convolution_matches_the_sum_of_component_convolutions(case):
    _, problem = case
    bundle = scalars(problem, 0.0, 0.0)
    n_to = problem.grids.n_to
    t = np.arange(n_to) * problem.grids.dt_fine
    field = (2e9 * np.exp(-(((t - np.mean(t)) / 10e-15) ** 2)) * np.cos(2e15 * t))[
        None, None, :
    ]
    # Two FFTs per component and three factors in the final product give a
    # conservative 64 eps per radix-2 level plus 32 eps per component.
    rounding = (64 * np.log2(2 * n_to) + 32 * problem.flags.n_raman) * np.finfo(
        float
    ).eps
    separate = []
    for term, density in zip(
        problem.responses.raman_terms, bundle.raman_densities, strict=True
    ):
        kernel = passes.raman_spectrum(
            term.undamped, (0.0,), term.fraction, problem.grids.dt_fine
        )
        separate.append(
            passes.raman_polarisation(
                field, kernel, kind="field", thg=False, density=density
            )
        )
    bound = rounding * sum(float(np.max(np.abs(value))) for value in separate)
    assert bound > 0
    kernel = passes.mixture_raman_spectrum(
        problem.responses.raman_terms,
        bundle.raman_rates,
        bundle.raman_densities,
        problem.grids.dt_fine,
    )
    actual = passes.raman_polarisation(
        field, kernel, kind="field", thg=False, density=1.0
    )
    assert_allclose(actual, sum(separate), rtol=0, atol=bound)


def test_raman_cache_tracks_every_amplitude_and_decay_rate(case):
    grid, problem = case
    backend = NumpyRefBackend(problem, grid=grid)
    bundle = scalars(problem, 0.0, 0.0)
    first = backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
    assert backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities) is first
    densities = list(bundle.raman_densities)
    for index in range(len(densities)):
        densities[index] *= 1.1
        fresh = backend._raman_spectrum(bundle.raman_rates, tuple(densities))
        assert fresh is not first
        first = fresh
    rates = list(bundle.raman_rates)
    for index in range(len(rates)):
        rates[index] += 1e12
        fresh = backend._raman_spectrum(tuple(rates), tuple(densities))
        assert fresh is not first
        first = fresh
    assert backend._raman_transforms == 5
    backend.close()
    assert backend._raman_densities is None


def test_raman_component_extents_are_checked(case):
    _, problem = case
    bundle = scalars(problem, 0.0, 0.0)
    with pytest.raises(BackendError, match="Raman"):
        passes.mixture_raman_spectrum(
            problem.responses.raman_terms,
            (),
            bundle.raman_densities,
            problem.grids.dt_fine,
        )
    with pytest.raises(BackendError, match="Raman"):
        passes.mixture_raman_spectrum(
            problem.responses.raman_terms, bundle.raman_rates, (), problem.grids.dt_fine
        )


def test_multiple_raman_components_propagate_with_one_cached_kernel(
    case, equivalence_backend_class
):

    grid, original = case
    steps = 16
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
        ),
    )
    # Each step has seven evaluations; two component kernels each carry one
    # convolution and two plasma exposures. The same 64-eps/sample budget used
    # by the channel ladder is summed over all four response components.
    bound = 64 * grid.n_to * 4 * 7 * steps * np.finfo(float).eps
    assert bound > 0.0
    outputs = []
    for backend_type in (NumpyRefBackend, equivalence_backend_class):
        backend = backend_type(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            summary = backend.propagate_to(grid.length)
            assert summary.n_steps == steps
            assert raman_refreshes(backend) == 1
            outputs.append(backend.observe().require_field().copy())
        finally:
            backend.close()
    relative = np.linalg.norm(outputs[1] - outputs[0]) / np.linalg.norm(outputs[0])
    assert relative < bound


def test_compiled_raman_cache_keys_include_every_component(
    case, equivalence_backend_class
):
    grid, problem = case
    backend = equivalence_backend_class(problem, grid=grid)
    bundle = scalars(problem, 0.0, 0.0)
    try:
        prepare_raman(backend, bundle.raman_rates, bundle.raman_densities)
        prepare_raman(backend, bundle.raman_rates, bundle.raman_densities)
        assert raman_refreshes(backend) == 1
        densities = list(bundle.raman_densities)
        rates = list(bundle.raman_rates)
        for index in range(2):
            densities[index] *= 1.1
            prepare_raman(backend, tuple(rates), tuple(densities))
            assert raman_refreshes(backend) == 2 + index
        for index in range(2):
            rates[index] += 1e12
            prepare_raman(backend, tuple(rates), tuple(densities))
            assert raman_refreshes(backend) == 4 + index
    finally:
        backend.close()

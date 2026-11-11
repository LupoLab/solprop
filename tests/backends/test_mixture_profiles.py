# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent density baselines and propagation under labelled population splitting."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends._session_probe_b1 import history, last_field
from backends._session_probe_b1 import run as session_run
from mixture_cases import mixture_translation
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Component, Mixture
from solprop.profiles import Constant, Linear
from solprop.stepper import FixedSchedule
from solprop.translate.problem import assemble
from solprop.translate.stage import require_device_ready


def construct(mixture, responses, grid, original, original_request):
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        original.initial.field0,
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=original_request.stepper.dz_max,
        dz_init=original_request.stepper.dz_init,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 16),
    )
    return mode, problem, run_request


def final(backend_class, problem, run_request, grid):
    backend = backend_class(problem, run_request, grid=grid)
    try:
        result = session_run(backend, problem)
        assert history(result).n_steps == 16
        return last_field(result)
    finally:
        backend.close()


def test_three_independent_profiles_run_on_every_cpu_backend(backend_class):
    mixture, grid, _, responses, (original, original_request) = mixture_translation()
    components = tuple(
        dataclasses.replace(
            component,
            density=Linear((index + 1) * 1e24, (4 - index) * 1e24, grid.length),
        )
        for index, component in enumerate(mixture.components)
    )
    third = Component(
        dataclasses.replace(components[0].species, gamma3=0.0, ionisation=None),
        Linear(7e24, 0.8e24, grid.length),
        "inert",
    )
    mixture = Mixture.of(*components, third)
    mode, problem, run_request = construct(
        mixture, responses, grid, original, original_request
    )
    assert problem.linear.phi_representation == "table"
    assert problem.responses.prefactor_representation == "table"
    # The contract refusal is testable even when the optional backend is absent.
    with pytest.raises(
        BackendError,
        match=r"independent profiles.*population-0.*population-1.*inert.*z table",
    ):
        require_device_ready(problem, "device")
    positions = np.array([0.0, grid.length * 0.137, grid.length * 0.631, grid.length])
    assert mode.mixture is not None
    assert problem.responses.prefactor is not None
    for channel in problem.responses.plasma_channels:
        expected = mixture[channel.label].density(positions)
        # The declared profiles are linear; only their Chebyshev endpoint mapping
        # and the three-term evaluation incur rounding.
        assert_allclose(
            np.array([float(channel.density(float(z))) for z in positions]),
            expected,
            rtol=32 * np.finfo(float).eps,
            atol=0.0,
        )
        assert mode.mixture[channel.label].density is mixture[channel.label].density
    # A GPU build's refusal names the labels; the native CPU build evaluates the
    # table itself (SEP 29 section 6.6).
    with pytest.raises(
        BackendError,
        match=r"independent profiles.*population-0.*population-1.*inert.*z table",
    ):
        require_device_ready(problem, "native")
    # Clenshaw evaluation contributes 8(d+1)^2 eps per table. Each of sixteen
    # seven-stage steps contributes the two-channel ladder's 64 eps/sample.
    degree = max(
        int(problem.linear.phi_degrees.max()),
        int(problem.responses.prefactor.degrees.max()),
    )
    bound = (16 * (degree + 1) ** 2 + 64 * grid.n_to * 2) * 7 * 16 * np.finfo(float).eps
    assert bound > 0.0
    expected = final(NumpyRefBackend, problem, run_request, grid)
    actual = final(backend_class, problem, run_request, grid)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < bound


def test_splitting_a_population_preserves_propagation(backend_class):
    mixture, grid, _, responses, (original, original_request) = mixture_translation()
    material = mixture.components[0].species
    rate = next(
        response.rate for response in responses if isinstance(response, PlasmaResponse)
    )
    single = Mixture.of(Component(material, Constant(2e24), "whole"))
    split = Mixture.of(
        Component(material, Constant(1e24), "left"),
        Component(material, Constant(1e24), "right"),
    )
    # Two populations share the same physical susceptibility, Kerr and rate. The
    # only discrepancy can be redistribution of floating-point products/sums:
    # budget the same two-channel, seven-stage, sixteen-step accumulation bound.
    bound = 64 * grid.n_to * 2 * 7 * 16 * np.finfo(float).eps
    assert bound > 0.0
    fields = []
    for populations in (single, split):
        assert material.gamma3 is not None
        declared = tuple(
            response
            for component in populations.components
            for response in (
                KerrResponse(material.gamma3, thg=False, species=component.label),
                PlasmaResponse(material, rate, species=component.label),
            )
        )
        _, problem, run_request = construct(
            populations, declared, grid, original, original_request
        )
        fields.append(final(backend_class, problem, run_request, grid))
    assert np.linalg.norm(fields[1] - fields[0]) / np.linalg.norm(fields[0]) < bound

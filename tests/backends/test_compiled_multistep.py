# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 2: the Adams march on the compiled backends, against the reference.

The march is shared (``solprop.backends.multistep``); only the one source evaluation
and the propagator differ: ``numba_cpu`` evaluates through its fused stage kernels and
Kokkos through its general engine's stage pipeline. Parity is the reference's
trajectory to round-off: the same steps, orders and calls, and fields that agree to
the accumulated rounding of the evaluation orders.
"""

import math

import numpy as np
import pytest

from bounds import assert_within
from legacy_geometry import LEGACY_GRID
from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.noise import modified_noise
from solprop.stepper import PIController
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.translate.linop import delta_phi

COMPILED: list[type] = [NumbaCpuBackend]
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED.append(NativeBackend)

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27


def run(backend_class, problem, grid):
    """Propagate to the last plane; return the field, the ledger and the step sizes."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    steps, field = [], None
    for plane in problem.stepper.save_planes:
        backend.propagate_to(float(plane))
        observed = backend.observe()
        steps += list(observed.step_history.dz)
        field = observed.field
    assert field is not None
    ledger = backend._adams.ledger
    backend.close()
    return field, ledger, np.array(steps)


@pytest.mark.parametrize(
    "method",
    [
        Multistep(formulation="interaction"),
        Multistep(order=3, variable_order=False, formulation="interaction"),
        ComparisonMultistep(corrector="pece", order=4),
        "interpolated",
    ],
    ids=["selected", "pec3/4", "pece4", "dense"],
)
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_follows_the_reference_trajectory(
    narrow_grid, soliton_problem, method, backend_class
):
    interpolate = method == "interpolated"
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8, SOLITON_PERIOD / 4],
        controller=PIController(rtol=1e-6, atol=1e-18),
        integrator=Multistep(formulation="interaction") if interpolate else method,
        interpolate_planes=interpolate,
    )
    reference, reference_ledger, reference_steps = run(
        NumpyRefBackend, problem, narrow_grid
    )
    field, ledger, steps = run(backend_class, problem, narrow_grid)
    # The same decisions: every step, order and call.
    assert ledger.orders == reference_ledger.orders
    assert ledger.total == reference_ledger.total
    # The evaluation orders round differently and the controller passes that on to
    # the step sizes: measured at most 1.5e-10 relative over 47-99 steps (Kokkos's
    # PECE4; 1.0e-10 on numba_cpu).
    np.testing.assert_allclose(steps, reference_steps, rtol=1e-9)
    # Measured 8e-16 to 1.8e-15 of the peak over the quarter period.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_steps_the_modified_noise_as_the_reference(
    narrow_grid, soliton_problem, formulation, backend_class
):
    # SEP 16 phase 3: the draw enters each sample at its own position, through the
    # same stage pipeline the Runge-Kutta march uses.
    noise = modified_noise(narrow_grid, np.random.default_rng(3))
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8],
        noise=noise,
        controller=PIController(rtol=1e-7, atol=1e-18),
        integrator=Multistep(formulation=formulation),  # pyright: ignore[reportArgumentType]
    )

    def noisy_run(backend_class):
        backend = backend_class(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, noise)
        backend.propagate_to(SOLITON_PERIOD / 8)
        field = backend.observe().require_field()
        calls = backend._adams.ledger.total
        backend.close()
        return field, calls

    reference, calls = noisy_run(NumpyRefBackend)
    field, compiled_calls = noisy_run(backend_class)
    assert compiled_calls == calls
    # As for the noiseless trajectory (above): rounding of the evaluation orders.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


def test_the_settings_admit_the_compiled_backends():
    control = PIController(rtol=1e-6, atol=1e-12)
    common = {"save_planes": [1.0], "dz_max": 0.1, "dz_init": 1e-3}
    for backend in ("numba_cpu", "native"):
        settings = SimulationSettings(
            control,
            backend=backend,
            integrator=Multistep(formulation="interaction"),
            **common,
        )
        assert settings.backend == backend
    with pytest.raises(DriverError, match="numba_cpu and native only"):
        SimulationSettings(
            control,
            backend="cuda",
            integrator=Multistep(formulation="interaction"),
            **common,
        )


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
def test_numba_marches_a_problem_without_responses(narrow_grid, linear_problem, collar):
    # No nonlinear response: the source is zero, or the in-stage collar's term alone.
    extra = (
        {"absorber": absorber(narrow_grid), "absorber_scheme": "nonlinear"}
        if collar
        else {}
    )
    problem = linear_problem(
        narrow_grid,
        save_planes=[SOLITON_PERIOD / 8],
        integrator=Multistep(formulation="interaction"),
        **extra,
    )
    reference, _, _ = run(NumpyRefBackend, problem, narrow_grid)
    field, _, _ = run(NumbaCpuBackend, problem, narrow_grid)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize(
    "route",
    [{"thg": False, "plasma": False}, {"envelope": True, "plasma": True}],
    ids=["analytic field", "envelope plasma"],
)
def test_numba_sources_the_analytic_field_and_the_envelope_plasma(route):
    # The Adams source on numba_cpu through the stage pipeline's two other column
    # families: the analytic signal of a field without THG, and the envelope with its
    # plasma assembly. One evaluation is compared, at a field that reaches into the
    # temporal collar: whole runs part by round-off in the discrete order choice.
    # The former grid, on which the tolerance below was measured.
    limits = (0.01, 800e-9, (200e-9, 2e-6), 100e-15)
    grid = (
        EnvelopeGrid.from_limits(*limits, thg=False, **LEGACY_GRID)
        if route.get("envelope")
        else FieldGrid.from_limits(*limits, **LEGACY_GRID)
    )
    simulation = Simulation.from_capillary(
        150e-6,
        0.01,
        "He",
        1.0,
        lambda0=800e-9,
        fwhm=10e-15,
        energy=5e-5,
        grid=grid,
        n_save=2,
        noise=None,
        integrator=Multistep(formulation="interaction"),
        **route,
    )
    problem = simulation.translate()
    start = problem.initial.field0
    pulse = np.fft.ifft(start, axis=-1)
    edge = np.roll(pulse, pulse.shape[-1] * 2 // 5, axis=-1)
    lab = np.fft.fft(pulse + 0.5 * edge, axis=-1)
    found = []
    for backend_class in (NumpyRefBackend, NumbaCpuBackend):
        backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
        backend.set_initial(start, None)
        source, back = backend._adams_source(lab, 0.0, 1e-3)
        found.append((np.array(source), np.array(back)))
        backend.close()
    (reference, reference_back), (source, back) = found
    # Measured 8.3e-12 (analytic field) and 1.6e-15 (envelope plasma) of the peak.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(source, reference, rtol=0, atol=1e-10 * scale)
    # The propagator back to the anchor, exp(Phi(anchor) - Phi(position)), from one
    # table on both sides: an exponent rounded within a few eps of its size and an
    # exponential's few eps, so 16 eps (1 + max |delta Phi|) relative.
    exponent = np.abs(delta_phi(problem.linear, 1e-3, 0.0))
    bound = 16 * np.finfo(float).eps * (1 + float(np.max(exponent)))
    assert_within(back, reference_back, bound, what="propagator")

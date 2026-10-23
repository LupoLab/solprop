# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Numerical parity, stepping, FSAL, replay, and resume."""

from __future__ import annotations

import dataclasses
import math
from typing import overload

import numpy as np
import numpy.typing as npt
import pytest
from bounds import assert_within, assert_work_within, departure
from conftest import PhaseACase, RamanCase
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.protocol import StepSummary
from solprop.grid import EnvelopeGrid
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import KernelProblem

from solprop_native import NativeBackend, build_info

EPS = np.finfo(np.float64).eps
DEVICE_BUILD = build_info().device_backend != "none"
type ComplexArray = npt.NDArray[np.complex128]
type PhaseBackend = NumpyRefBackend | NativeBackend
type PhaseBackendClass = type[NumpyRefBackend] | type[NativeBackend]
pytestmark = pytest.mark.gpu


def _gamma(operations: int) -> float:
    """Derived fixed-order dot-product round-off bound."""
    scaled = operations * EPS
    return scaled / (1.0 - scaled)


def _fft_bound(samples: int) -> float:
    """Derived FFT round-trip and associated pointwise-operation bound."""
    return 512 * EPS * math.ceil(math.log2(samples))


def _scan_bound(samples: int) -> float:
    """Derived complete deterministic plasma-scan-chain bound."""
    return 192 * EPS * math.ceil(math.log2(samples))


def _phase_factor(cpu_factor: int, samples: int, fft_round_trips: int) -> float:
    """Retain CPU bounds and apply the predeclared CUDA FFT envelope on devices."""
    established = cpu_factor * EPS
    if not DEVICE_BUILD:
        return established
    return max(established, fft_round_trips * _fft_bound(samples))


def test_phase_d_roundoff_formulas_are_fixed_before_device_measurement() -> None:
    """Keep the approved dot, FFT, and plasma-scan formulas executable."""
    assert _gamma(4) == (4 * EPS) / (1.0 - 4 * EPS)
    assert _fft_bound(8192) == 512 * EPS * 13
    assert _scan_bound(8192) == 192 * EPS * 13


@overload
def _loaded(
    backend_class: type[NumpyRefBackend], problem: KernelProblem, grid: EnvelopeGrid
) -> NumpyRefBackend: ...


@overload
def _loaded(
    backend_class: type[NativeBackend], problem: KernelProblem, grid: EnvelopeGrid
) -> NativeBackend: ...


def _loaded(
    backend_class: PhaseBackendClass, problem: KernelProblem, grid: EnvelopeGrid
) -> PhaseBackend:
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    return backend


def test_one_trial_matches_numpy_within_sixteen_epsilon(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    reference = _loaded(NumpyRefBackend, problem, grid)
    backend = _loaded(NativeBackend, problem, grid)
    try:
        # The numerical seam test intentionally bypasses the shared march.
        _, expected = reference._trial_step(  # pyright: ignore[reportPrivateUsage]
            0.0, 1e-4
        )
        _, state = backend._trial_step(  # pyright: ignore[reportPrivateUsage]
            0.0, 1e-4
        )
        backend._finish(state, 1e-4)  # pyright: ignore[reportPrivateUsage]
        actual = backend.observe().require_field()
        bound = _phase_factor(16, problem.grids.n_to, 7) * float(
            np.max(np.abs(expected))
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)
    finally:
        reference.close()
        backend.close()


def test_eight_fixed_steps_match_within_128_epsilon(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    schedule = FixedSchedule(np.full(8, 1.25e-4))
    fixed = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(problem.stepper, schedule=schedule),
    )
    fields: list[ComplexArray] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = _loaded(backend_class, fixed, grid)
        backend.propagate_to(1e-3)
        fields.append(backend.observe().require_field())
        backend.close()
    bound = _phase_factor(128, problem.grids.n_to, 8 * 7) * float(
        np.max(np.abs(fields[0]))
    )
    np.testing.assert_allclose(fields[1], fields[0], rtol=0.0, atol=bound)


def _tight_reference(problem: KernelProblem, grid: EnvelopeGrid) -> ComplexArray:
    """Return the case's field at 1 mm from ``numpy_ref`` at a much tighter tolerance.

    ``rtol`` 1e-12 against the cases' 1e-8, and steps of at most 10 um against the
    cases' millimetre cap (their steps are set by the cap and the growth limit, not by
    the error estimate): the reference's own error is far below the runs' it judges.
    """
    controller = dataclasses.replace(problem.stepper.controller, rtol=1e-12, atol=1e-20)
    tight = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper, controller=controller, dz_max=1e-5, dz_init=1e-5
        ),
    )
    backend = _loaded(NumpyRefBackend, tight, grid)
    try:
        backend.propagate_to(1e-3)
        return np.array(backend.observe().require_field(), copy=True)
    finally:
        backend.close()


def _run_adaptively(
    backend_class: PhaseBackendClass, problem: KernelProblem, grid: EnvelopeGrid
) -> tuple[StepSummary, ComplexArray]:
    backend = _loaded(backend_class, problem, grid)  # pyright: ignore[reportArgumentType]
    try:
        summary = backend.propagate_to(1e-3)
        return summary, np.array(backend.observe().require_field(), copy=True)
    finally:
        backend.close()


ENVELOPE_FACTOR = 4.0
"""Safety factor of the adaptive runs' accuracy envelope, fixed before measurement."""


def test_adaptive_propagation_meets_the_reference_s_accuracy_envelope(
    phase_a_case: PhaseACase,
) -> None:
    """Two independent adaptive Kokkos runs, each judged by its accuracy.

    Adaptive runs may fork on a rounding difference, so neither run is required to
    repeat the other or ``numpy_ref``'s trajectory (solprop SEP 25 section 4.4). The
    envelope: ``numpy_ref``'s own error at the case's tolerance against a much tighter
    reference, times four; each Kokkos run must be as accurate, and its step count
    within a quarter of the reference run's.
    """
    problem, grid = phase_a_case
    reference = _tight_reference(problem, grid)
    results = [
        _run_adaptively(backend_class, problem, grid)
        for backend_class in (NumpyRefBackend, NativeBackend, NativeBackend)
    ]
    error = departure(results[0][1], reference)
    assert error > 0.0
    for summary, field in results[1:]:
        assert_work_within(summary.n_steps, results[0][0].n_steps, factor=1.25)
        assert_within(field, reference, ENVELOPE_FACTOR * error, what="adaptive run")


def test_a_kerr_leaf_density_path_scales_the_single_column_response(
    phase_a_case: PhaseACase,
) -> None:
    """A mode-averaged Kerr run takes the Kerr leaf's own density, as a mixture does.

    The leaf's density path is doubled and its coefficient halved, the split a
    multi-component Kerr leaf has: the physics is unchanged, so the reference's split
    run and the backend's are each as accurate as the reference's unsplit run, against
    a much tighter reference, within the factor of four of the adaptive envelope
    (solprop SEP 25 section 4.4). The unreachable single-column engine (`State`)
    ignores the path and misses by about one per cent on such a case.
    """
    problem, grid = phase_a_case
    kerr = problem.responses.kerr_coefficients
    assert kerr is not None
    density = problem.responses.density if kerr.density is None else kerr.density
    doubled = np.array(density.coefficients * 2.0)
    doubled.setflags(write=False)
    split = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            kerr_coefficients=dataclasses.replace(
                kerr,
                density=dataclasses.replace(density, coefficients=doubled),
                coefficient=kerr.coefficient / 2.0,
            ),
        ),
    )
    reference = _tight_reference(problem, grid)
    results = [
        _run_adaptively(backend_class, case, grid)
        for backend_class, case in (
            (NumpyRefBackend, problem),
            (NumpyRefBackend, split),
            (NativeBackend, split),
        )
    ]
    error = departure(results[0][1], reference)
    assert error > 0.0
    names = ("reference split", "native split")
    for name, (_, field) in zip(names, results[1:], strict=True):
        assert_within(field, reference, ENVELOPE_FACTOR * error, what=name)


def test_resume_matches_an_uninterrupted_replay(phase_a_case: PhaseACase) -> None:
    problem, grid = phase_a_case
    schedule = FixedSchedule(np.full(8, 1.25e-4))
    fixed = dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=schedule,
            save_planes=(5e-4, 1e-3),
        ),
    )
    whole = _loaded(NativeBackend, fixed, grid)
    whole.propagate_to(1e-3)
    expected = whole.observe().require_field()
    whole.close()
    first = _loaded(NativeBackend, fixed, grid)
    first.propagate_to(5e-4)
    halfway = first.observe().require_field()
    first.close()
    resumed = NativeBackend(fixed, grid=grid)
    resumed.set_initial(halfway, None, z=5e-4)
    resumed.propagate_to(1e-3)
    actual = resumed.observe().require_field()
    resumed.close()
    bound = 32 * EPS * float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


def test_fsal_is_promoted_after_acceptance_and_cleared_at_a_load(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    backend = _loaded(NativeBackend, problem, grid)
    assert backend.metadata()["fsal_ready"] is False
    backend.propagate_to(5e-4)
    assert backend.metadata()["fsal_ready"] is True
    backend.propagate_to(1e-3)
    assert backend.metadata()["fsal_ready"] is True
    backend.set_initial(problem.initial.field0, None)
    assert backend.metadata()["fsal_ready"] is False
    backend.close()


def test_one_complete_raman_response_matches_the_reference(
    raman_case: RamanCase,
) -> None:
    """The native doubled-grid convolution matches the independent NumPy path."""
    problem, grid = raman_case
    reference = _loaded(NumpyRefBackend, problem, grid)
    backend = _loaded(NativeBackend, problem, grid)
    try:
        dz = 1e-6
        _, expected = reference._trial_step(  # pyright: ignore[reportPrivateUsage]
            0.0, dz
        )
        _, state = backend._trial_step(  # pyright: ignore[reportPrivateUsage]
            0.0, dz
        )
        backend._finish(state, dz)  # pyright: ignore[reportPrivateUsage]
        actual = backend.observe().require_field()
        established = 512 * EPS * np.log2(2 * problem.grids.n_to)
        device = 7 * _fft_bound(problem.grids.n_to) + 8 * _fft_bound(
            2 * problem.grids.n_to
        )
        bound = max(established, device if DEVICE_BUILD else 0.0) * max(
            1.0, float(np.max(np.abs(expected)))
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)
    finally:
        reference.close()
        backend.close()

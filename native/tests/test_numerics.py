# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Numerical parity, stepping, FSAL, replay, and resume."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from bounds import assert_within, assert_work_within, departure
from conftest import PhaseACase, RamanCase
from session_runs import finish, steps
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.grid import EnvelopeGrid
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

from solprop_native import NativeBackend, build_info

EPS = np.finfo(np.float64).eps
DEVICE_BUILD = build_info().device_backend != "none"
type ComplexArray = npt.NDArray[np.complex128]
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


def _run(
    backend_class: PhaseBackendClass,
    problem: PropagationProblem,
    run_request: RunRequest,
    grid: EnvelopeGrid,
    start: Start | None = None,
) -> list[dict[str, Any]]:
    """Run the problem through the backend's session; return its observations."""
    backend = backend_class(problem, run_request, grid=grid)
    return finish(backend, problem, run_request, start).planes


def _one_step(run_request: RunRequest, dz: float) -> RunRequest:
    """Return the request replaying one step of `dz` to a plane at its end."""
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule(np.full(1, dz)),
            save_planes=(dz,),
        )
    )


def test_one_trial_matches_numpy_within_sixteen_epsilon(
    phase_a_case: PhaseACase,
) -> None:
    """One replayed step of 0.1 mm: the trial, accepted and finished, on each side."""
    problem, run_request, grid = phase_a_case
    stepped = _one_step(run_request, 1e-4)
    expected, actual = (
        _run(backend_class, problem, stepped, grid)[-1]["field"]
        for backend_class in (NumpyRefBackend, NativeBackend)
    )
    bound = _phase_factor(16, problem.grids.n_to, 7) * float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


def test_eight_fixed_steps_match_within_128_epsilon(
    phase_a_case: PhaseACase,
) -> None:
    problem, run_request, grid = phase_a_case
    schedule = FixedSchedule(np.full(8, 1.25e-4))
    fixed, fixed_request = (
        problem,
        run_request.with_stepper(
            dataclasses.replace(run_request.stepper, schedule=schedule)
        ),
    )
    fields: list[ComplexArray] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        planes = _run(backend_class, fixed, fixed_request, grid)
        assert planes[-1]["z"] == pytest.approx(1e-3)
        fields.append(planes[-1]["field"])
    bound = _phase_factor(128, problem.grids.n_to, 8 * 7) * float(
        np.max(np.abs(fields[0]))
    )
    np.testing.assert_allclose(fields[1], fields[0], rtol=0.0, atol=bound)


def _tight_reference(
    problem: PropagationProblem, run_request: RunRequest, grid: EnvelopeGrid
) -> ComplexArray:
    """Return the case's field at 1 mm from ``numpy_ref`` at a much tighter tolerance.

    ``rtol`` 1e-12 against the cases' 1e-8, and steps of at most 10 um against the
    cases' millimetre cap (their steps are set by the cap and the growth limit, not by
    the error estimate): the reference's own error is far below the runs' it judges.
    """
    controller = dataclasses.replace(
        run_request.stepper.controller, rtol=1e-12, atol=1e-20
    )
    tight, tight_request = (
        problem,
        run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper, controller=controller, dz_max=1e-5, dz_init=1e-5
            )
        ),
    )
    planes = _run(NumpyRefBackend, tight, tight_request, grid)
    assert planes[-1]["z"] == 1e-3
    return planes[-1]["field"]


def _run_adaptively(
    backend_class: PhaseBackendClass,
    problem: PropagationProblem,
    run_request: RunRequest,
    grid: EnvelopeGrid,
) -> tuple[int, ComplexArray]:
    """Return an adaptive run's accepted steps to 1 mm and its field there."""
    planes = _run(backend_class, problem, run_request, grid)
    assert planes[-1]["z"] == 1e-3
    return steps(planes)[0], planes[-1]["field"]


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
    problem, run_request, grid = phase_a_case
    reference = _tight_reference(problem, run_request, grid)
    results = [
        _run_adaptively(backend_class, problem, run_request, grid)
        for backend_class in (NumpyRefBackend, NativeBackend, NativeBackend)
    ]
    error = departure(results[0][1], reference)
    assert error > 0.0
    for accepted, field in results[1:]:
        assert_work_within(accepted, results[0][0], factor=1.25)
        assert_within(field, reference, ENVELOPE_FACTOR * error, what="adaptive run")


def test_a_kerr_leaf_density_path_scales_the_single_column_response(
    phase_a_case: PhaseACase,
) -> None:
    """A mode-averaged Kerr run takes the Kerr leaf's own density, as a mixture does.

    The leaf's density path is doubled and its coefficient halved, the split a
    multi-component Kerr leaf has: the physics is unchanged, so the reference's split
    run and the backend's are each as accurate as the reference's unsplit run, against
    a much tighter reference, within the factor of four of the adaptive envelope
    (solprop SEP 25 section 4.4).
    """
    problem, run_request, grid = phase_a_case
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
    reference = _tight_reference(problem, run_request, grid)
    results = [
        _run_adaptively(backend_class, case, run_request, grid)
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
    problem, run_request, grid = phase_a_case
    schedule = FixedSchedule(np.full(8, 1.25e-4))
    fixed, fixed_request = (
        problem,
        run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper,
                schedule=schedule,
                save_planes=(5e-4, 1e-3),
            )
        ),
    )
    whole = _run(NativeBackend, fixed, fixed_request, grid)
    assert [plane["z"] for plane in whole] == [0.0, 5e-4, 1e-3]
    expected = whole[-1]["field"]
    # A resume from the saved plane halfway.
    halfway = whole[1]["field"]
    start = Start(
        field=halfway, z=5e-4, plane_index=1, plane_pending=False, resume=True
    )
    resumed = _run(NativeBackend, fixed, fixed_request, grid, start)
    assert [plane["z"] for plane in resumed] == [1e-3]
    actual = resumed[-1]["field"]
    bound = 32 * EPS * float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


def test_fsal_is_promoted_after_acceptance_and_cleared_at_a_load(
    phase_a_case: PhaseACase,
) -> None:
    problem, run_request, grid = phase_a_case
    run_request = run_request.with_stepper(
        dataclasses.replace(run_request.stepper, save_planes=(5e-4, 1e-3))
    )
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        assert backend.metadata()["fsal_ready"] is False
        finish(backend, problem, run_request, owns=False)
        assert backend.metadata()["fsal_ready"] is True
        # A second run's start loads the launch again, which drops the stage.
        session = backend.prepare(problem, run_request).start(
            Start.entrance(problem), owns=False
        )
        try:
            assert backend.metadata()["fsal_ready"] is False
        finally:
            session.close()
    finally:
        backend.close()


def test_one_complete_raman_response_matches_the_reference(
    raman_case: RamanCase,
) -> None:
    """The native doubled-grid convolution matches the independent NumPy path."""
    problem, run_request, grid = raman_case
    stepped = _one_step(run_request, 1e-6)
    expected, actual = (
        _run(backend_class, problem, stepped, grid)[-1]["field"]
        for backend_class in (NumpyRefBackend, NativeBackend)
    )
    established = 512 * EPS * np.log2(2 * problem.grids.n_to)
    device = 7 * _fft_bound(problem.grids.n_to) + 8 * _fft_bound(2 * problem.grids.n_to)
    bound = max(established, device if DEVICE_BUILD else 0.0) * max(
        1.0, float(np.max(np.abs(expected)))
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent-oracle checks on the actual standard translated workloads."""

import dataclasses

import numpy as np
import pytest
from benchmarks.builders import capillary
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.stats import PROFILE_STATISTICS, StatsSpec
from solprop.stepper import FixedSchedule

from solprop_cpunative.backend import CPUNativeBackend, _stage


@pytest.fixture(params=["rdw", "ctc", "argon_blueshift"], scope="module")
def problem(request):
    simulation = capillary(request.param)
    return simulation, simulation.translate()


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", "1")
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")


def backends(simulation, problem):
    return [
        cls(problem, grid=simulation.grid, modes=simulation.modes)
        for cls in (CPUNativeBackend, NumpyRefBackend)
    ]


def relative_error(candidate, reference):
    return np.linalg.norm(candidate - reference) / max(
        np.linalg.norm(reference), np.finfo(float).tiny
    )


@pytest.mark.parametrize("amplitude", [1.0, 3.0])
@pytest.mark.parametrize("threads", [1, 8])
def test_source(problem, amplitude, threads, monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", str(threads))
    simulation, problem = problem
    candidate, reference = backends(simulation, problem)
    # Fixed before comparing: sequential scans have O(n_to*eps) accumulated
    # rounding, with 64 operations per sample covering both filters and transforms.
    budget = 64 * np.finfo(float).eps * problem.grids.n_to
    field = amplitude * problem.initial.field0
    for backend in (candidate, reference):
        backend.set_initial(field, None)
    got = np.asarray(
        candidate._cpu.evaluate(_stage(candidate._stage_scalars(0.0, 0.0)))
    ).reshape(problem.shape)
    expected = reference._adams_lab_source(reference._state, 0.0)
    assert relative_error(got, expected) < budget
    assert np.isfinite(got).all()
    assert np.count_nonzero(got[..., ~problem.grids.active_mask]) == 0
    candidate.close()
    reference.close()


@pytest.mark.parametrize(
    ("modes", "chunk"),
    [(modes, 11) for modes in range(1, 9)]
    + [(3, 1), (3, 8), (5, 16), (6, 32), (7, 32), (8, 64)],
)
def test_raman_modal_pairs_and_tail(modes, chunk, monkeypatch):
    simulation = Simulation.from_capillary(
        16e-6,
        0.13,
        "H2",
        40.0,
        lambda0=515e-9,
        wavelength_limits=(220e-9, 4e-6),
        trange=2e-12,
        fwhm=400e-15,
        energy=10e-6,
        modes=modes,
        loss=0.0,
        plasma=False,
        seed=2016,
        stats=StatsSpec(names=()),
    )
    original = simulation.translate()
    kerr = original.responses.kerr_coefficients
    assert kerr is not None
    problem = dataclasses.replace(
        original,
        transform=dataclasses.replace(original.transform, node_chunk=chunk),
        responses=dataclasses.replace(
            original.responses,
            kerr_coefficients=dataclasses.replace(kerr, coefficient=0.0),
        ),
    )
    monkeypatch.setenv("SOLPROP_THREADS", "4")
    candidate, reference = backends(simulation, problem)
    try:
        # Populate every mode appreciably so the comparison exercises cross
        # products; the ordinary seeded launch is dominated by its first mode.
        field = np.array(problem.initial.field0, copy=True)
        launch = field[:, 0].copy()
        for mode in range(modes):
            field[:, mode] = launch * np.exp(0.37j * mode) / np.sqrt(modes)
        for backend in (candidate, reference):
            backend.set_initial(field, None)
        got = np.asarray(
            candidate._cpu.evaluate(_stage(candidate._stage_scalars(0.0, 0.0)))
        ).reshape(problem.shape)
        expected = reference._adams_lab_source(reference._state, 0.0)
        # The pair expansion reorders a real contraction and linear convolution;
        # isolate Raman so a larger Kerr/plasma source cannot hide an error.
        budget = 64 * np.finfo(float).eps * problem.grids.n_to
        assert relative_error(got, expected) < budget
    finally:
        candidate.close()
        reference.close()


@pytest.mark.parametrize("method", ["rk", "adams"])
def test_prescribed_steps(problem, method):
    simulation, original = problem
    dz = min(original.stepper.dz_init, 1e-4)
    target = 8 * dz
    # Keep the full translated domain and its hard endpoints, but execute only
    # the short prefix. The appended step is never executed.
    tail = max(float(k) for k in [target, *original.linear.hard_kinks])
    schedule = FixedSchedule.uniform(
        0.0, tail, dz, save_planes=[target], hard_kinks=original.linear.hard_kinks
    )
    settings = dataclasses.replace(
        original.stepper,
        schedule=schedule,
        save_planes=[target],
        integrator=None if method == "rk" else original.stepper.integrator,
    )
    problem = dataclasses.replace(original, stepper=settings)
    candidate, reference = backends(simulation, problem)
    for backend in (candidate, reference):
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(target)
    # Eight matched updates: a conservative rounding allowance for FFTs, scans,
    # strict defects and history cancellation, fixed independently of differences.
    assert relative_error(candidate.observe().field, reference.observe().field) < 1e-10
    if method == "adams":
        assert (
            candidate.metadata()["response_calls"]
            == reference.metadata()["response_calls"]
        )
    candidate.close()
    reference.close()


def test_statistics(problem):
    simulation, problem = problem
    candidate, reference = backends(simulation, problem)
    field = 3 * problem.initial.field0
    for backend in (candidate, reference):
        backend.set_initial(field, None)
    spec = dataclasses.replace(
        problem.stats_spec,
        names=tuple(
            n
            for n in problem.stats_spec.names
            if n not in PROFILE_STATISTICS and n != "quadrature_error"
        ),
        periodic=(),
    )
    got, expected = candidate.stats(spec), reference.stats(spec)
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name].shape == expected[name].shape
        # Population fractions subtract nearly equal values; their absolute error
        # has a density factor. Compare each diagnostic at its own overall scale.
        scale = max(float(np.max(np.abs(expected[name]))), np.finfo(float).tiny)
        # A defect subtracts two O(1) quantities before normalising. Its
        # rounding budget is absolute, not relative to the near-zero defect.
        atol = (
            64 * np.finfo(float).eps * problem.grids.n_to
            if name == "plasma_work_defect"
            else 1e-10 * scale
        )
        np.testing.assert_allclose(
            got[name],
            expected[name],
            rtol=1e-9,
            atol=atol,
            equal_nan=True,
            err_msg=name,
        )
    candidate.close()
    reference.close()

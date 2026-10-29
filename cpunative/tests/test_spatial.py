# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Qualify the unchanged radial filament recipe against the NumPy oracle."""

import dataclasses

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.registry import CASES
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stepper import FixedSchedule

from solprop_cpunative.backend import CPUNativeBackend, _stage, memory_estimate


@pytest.fixture(scope="module")
def radial():
    case = CASES["radial_filament_256"]
    simulation = build(case.configuration("default", 1, "numpy_ref", 1.0))
    return simulation, simulation.translate()


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", "1")
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")


@pytest.mark.parametrize("chunk", [17, 128, 256])
@pytest.mark.parametrize("threads", [1, 4])
def test_radial_source_and_statistics(radial, chunk, threads, monkeypatch):
    simulation, original = radial
    problem = dataclasses.replace(
        original,
        transform=dataclasses.replace(original.transform, node_chunk=chunk),
    )
    monkeypatch.setenv("SOLPROP_THREADS", str(threads))
    candidate = CPUNativeBackend(problem, grid=simulation.grid)
    reference = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        # Fixed summation allowance includes the dense QDHT, temporal Hilbert
        # transform and the causal band-integral scans; the oracle is independent.
        bound = 64 * np.finfo(float).eps * (problem.grids.n_to + problem.shape[1])
        field = 2 * problem.initial.field0
        for backend in (candidate, reference):
            backend.set_initial(field, None)
        got = np.asarray(
            candidate._cpu.evaluate(_stage(candidate._stage_scalars(0.0, 0.0)))
        ).reshape(problem.shape)
        expected = reference._adams_lab_source(reference._state, 0.0)
        assert np.linalg.norm(got - expected) <= bound * np.linalg.norm(expected)
        assert np.isfinite(got).all()
        assert np.count_nonzero(got[..., ~problem.grids.active_mask]) == 0
        got_stats = candidate.stats(problem.stats_spec)
        expected_stats = reference.stats(problem.stats_spec)
        assert got_stats.keys() == expected_stats.keys()
        for name, value in expected_stats.items():
            assert got_stats[name].shape == value.shape, name
            scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
            np.testing.assert_allclose(
                got_stats[name], value, rtol=bound, atol=bound * scale, err_msg=name
            )
        assert sum(candidate.memory_ledger().values()) == candidate._cpu.memory_bytes
        assert memory_estimate(problem) == candidate._cpu.memory_bytes
    finally:
        candidate.close()
        reference.close()


@pytest.mark.parametrize("method", ["default", "rk"])
def test_radial_eight_prescribed_steps(radial, method):
    simulation, original = radial
    dz = min(original.stepper.dz_init, 3e-6)
    target = 8 * dz
    schedule = FixedSchedule.uniform(0.0, 0.03, dz, save_planes=[target])
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper,
            integrator=original.stepper.integrator if method == "default" else None,
            schedule=schedule,
            save_planes=[target],
        ),
    )
    fields = []
    for backend_type in (NumpyRefBackend, CPUNativeBackend):
        backend = backend_type(problem, grid=simulation.grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(target)
            fields.append(backend.observe().require_field())
        finally:
            backend.close()
    # Eight matched updates and dense transforms, fixed before viewing differences.
    bound = 512 * np.finfo(float).eps * (problem.grids.n_to + problem.shape[1]) * 8
    assert np.linalg.norm(fields[1] - fields[0]) <= bound * np.linalg.norm(fields[0])


@pytest.mark.parametrize("method", ["default", "rk"])
def test_radial_ordinary_driver(method):
    case = CASES["radial_filament_256"]
    simulation = build(case.configuration(method, 1, "cpunative", 0.0025))
    result = simulation.run(progress="none")
    assert result.n_saved == case.planes + 1
    for name in simulation.translate().stats_spec.names:
        assert np.isfinite(result[f"stats/{name}"]).all(), name

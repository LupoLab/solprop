# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify spatial execution choices without changing numerical contracts."""

import dataclasses
import math

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from backends import test_free_ladders
from backends.test_radial_nonlinear import derivative, make_problem
from bounds import assert_within, gamma
from legacy_geometry import LEGACY_GRID
from solprop.backends.numba_cpu import NumbaCpuBackend, fft, kernels, spatial
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from spatial_records import record_metric

free_case = test_free_ladders.free_case


@pytest.mark.parametrize(
    ("n_r", "n_pol", "chunk", "duration"),
    [
        (256, 1, 256, 300e-15),
        (256, 2, 256, 300e-15),
        (256, 1, 73, 300e-15),
        (128, 1, 128, 300e-15),
        (256, 1, 256, 150e-15),
        (384, 1, 384, 600e-15),
    ],
)
def test_radial_real_blas_preserves_transform_and_memory_contract(
    spatial_backend_class, n_r, n_pol, chunk, duration, request
):
    grid = EnvelopeGrid.from_limits(
        1e-4, 800e-9, (600e-9, 1.2e-6), duration, thg=False, **LEGACY_GRID
    )
    space = RadialGrid(6e-4, n_r, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1 + 0j,) if n_pol == 1 else (1 + 0j, 0.5j),
    )
    problem = make_problem(
        grid,
        space,
        # make_problem's medium is the vacuum.
        pulse.build(grid, space, medium=SpatialMedium()),
        [KerrResponse(1e-22, thg=False)],
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )
    # Two radial reductions and temporal transforms, with the existing chunk
    # ladder's association allowance, fixed before either backend runs.
    bound = gamma(64 * n_r + 32 * grid.n_to)
    reference = derivative(NumpyRefBackend, problem, grid)
    actual = derivative(spatial_backend_class, problem, grid)
    np.testing.assert_allclose(
        actual, reference, rtol=0, atol=bound * np.max(np.abs(reference))
    )
    record_metric(
        request,
        "radial real BLAS derivative (relative)",
        f"r={n_r}, p={n_pol}, chunk={chunk}, time={grid.n_to}",
        np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
        bound,
        spatial_backend_class.name,
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        metadata, ledger = backend.metadata(), backend.memory_ledger()
        if backend.name == "numba_cpu":
            assert metadata["radial_real_gemm"]
            assert ledger["synthesis"] == 2 * problem.transform.synthesis.nbytes
        if backend.name == "native":
            # Portable/custom and device builds keep their existing provider.
            packed = str(metadata["synthesis_provider"]).endswith("-packed-real")
            eligible = (
                metadata["device_backend"] == "none"
                and metadata["gemm_provider"] in {"HYBRID", "KOKKOSBLAS_TPL"}
                and n_pol == 1
                and chunk == n_r
                and n_r >= 256
                and grid.n_to >= 256
                and 22 * 16 * n_r * grid.n_to <= 64 * 1024 * 1024
            )
            assert packed == eligible
            if packed:
                assert n_pol == 1 and chunk == n_r
                assert ledger["radial_blas_buffers"] == 32 * n_r * grid.n_to
                assert 22 * 16 * n_r * grid.n_to <= 64 * 1024 * 1024
            else:
                assert ledger["radial_blas_buffers"] == 0
    finally:
        backend.close()


@pytest.fixture
def two_workers():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    try:
        yield
    finally:
        set_num_threads(previous)


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("flag", ["estimate", "measure", "patient"])
def test_transverse_layout_scales_each_component_without_an_extra_volume(
    n_pol, layout, flag
):
    shape = (7, 9, n_pol, 11)
    # Two transforms over 63 samples, bounded by their arithmetic depth. The
    # integer round-trip scale is explicit because execute() is unnormalised.
    bound = 64 * np.finfo(float).eps * math.ceil(math.log2(63))
    rng = np.random.default_rng(30)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    plans = fft.build_transverse_plans(
        shape, flag=flag, threads=1, spatial_contiguous=layout
    )
    assert plans.spectrum.nbytes == data.nbytes
    assert np.shares_memory(plans.spectrum.reshape(-1, shape[-1]), plans.spectrum)
    plans.spectrum[:] = data
    plans.forward.execute()
    expected = np.fft.fftn(data, axes=(0, 1))
    np.testing.assert_allclose(
        plans.spectrum, expected, rtol=0, atol=bound * np.max(np.abs(expected))
    )
    plans.inverse.execute()
    np.testing.assert_allclose(
        plans.spectrum / 63, data, rtol=0, atol=bound * np.max(np.abs(data))
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_spatial_copies_preserve_padding_crop_and_chunk_boundaries(
    parallel, analytic, layout, two_workers
):
    operations = spatial.make_kernels(parallel=False, parallel_transfers=parallel)
    rng = np.random.default_rng(31)
    storage = np.array(
        rng.normal(size=(23, 17)) + 1j * rng.normal(size=(23, 17)), order=layout
    )
    source = storage[2:21]
    crop = rng.permutation(32)[:17]
    fine = np.full((21, 32), 2 + 3j)
    expected = np.zeros_like(fine)
    factors = np.ones(17)
    if analytic:
        # The analytic signal doubles the fine bins strictly between the zero
        # frequency and the fine Nyquist bin (16 of 32), wherever the crop puts them.
        factors[(crop > 0) & (crop < 16)] = 2
    expected[:19, crop] = source * (factors / 32)
    operations.pack_columns(source, crop, fine, 1 / 32, analytic)
    # Structural: copies and binary scalings, which round nothing in any grouping.
    np.testing.assert_array_equal(fine, expected)
    target = np.full_like(storage, 11 + 13j)
    expected_crop = target.copy(order="K")
    expected_crop[2:21] = fine[:19, crop]
    operations.crop_columns(fine, crop, target[2:21])
    np.testing.assert_array_equal(target, expected_crop)


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_parallel_combine_agrees_with_the_serial_one(noise, contiguous, two_workers):
    """One combination by two loops: a dozen roundings of uncancelled terms, 16 eps."""
    rng = np.random.default_rng(32)
    state = rng.normal(size=(19, 17)) + 1j * rng.normal(size=(19, 17))
    stages = np.stack([state * complex(i, 0.5) for i in range(4)])
    crop = np.arange(17) if contiguous else rng.permutation(32)[:17]
    weights = np.array([0.2, -0.5, 0.7, 0.1])
    values = []
    for parallel in (False, True):
        target = np.empty((19, 32), complex, order="F")
        operation = kernels.make_combine_and_pad(
            contiguous=contiguous, noise=noise, parallel=parallel
        )
        operation(state, stages, weights, 4, 1e-5, state, state, crop, target, 1 / 32)
        values.append(target)
    assert_within(values[1], values[0], 16 * np.finfo(float).eps)


def test_spatial_execution_metadata_and_independent_fft_override(
    thread_budget_lifted, free_case, monkeypatch
):
    _, grid, _, problem = free_case
    monkeypatch.setattr(fft, "REGIME_ELEMENTS", 1)
    monkeypatch.setattr(fft, "_thread_budget", lambda: fft.ThreadBudget(4, "affinity"))
    monkeypatch.setenv("SOLPROP_TRANSVERSE_FFTW_THREADS", "3")
    backend = NumbaCpuBackend(problem, grid=grid, threads=2, fftw_threads=1)
    try:
        metadata = backend.metadata()
        assert metadata["plasma_fraction_reuse"] is False
        assert metadata["kernel_regime_elements"] == fft.REGIME_ELEMENTS
        assert metadata["parallel_spatial"]
        assert metadata["parallel_spatial_transfers"] == (grid.kind == "envelope")
        assert metadata["fftw_threads"] == 1
        assert metadata["transverse_fftw_threads"] == 3
        assert backend._transverse is not None
        assert np.shares_memory(
            backend._transverse.spectrum.reshape(problem.shape),
            backend._transverse.spectrum,
        )
    finally:
        backend.close()
    with pytest.raises(BackendError, match="positive"):
        NumbaCpuBackend(problem, grid=grid, transverse_fftw_threads=0)


@pytest.mark.parametrize("steps", [0, 1, 8])
def test_parallel_spatial_passes_follow_the_existing_physics_ladder(
    thread_budget_lifted, free_case, steps, monkeypatch
):
    family, grid, space, problem = free_case
    monkeypatch.setattr(fft, "REGIME_ELEMENTS", 1)
    monkeypatch.setenv("SOLPROP_NUMBA_THREADS", "2")
    monkeypatch.setenv("SOLPROP_FFTW_THREADS", "1")
    monkeypatch.setenv("SOLPROP_TRANSVERSE_FFTW_THREADS", "1")
    bound = test_free_ladders.amplitude_bound(family, grid, space, "numba_cpu", steps)
    if steps == 0:
        values = [
            derivative(implementation, problem, grid)
            for implementation in (NumpyRefBackend, NumbaCpuBackend)
        ]
    else:
        schedule = problem.stepper.schedule
        assert schedule is not None
        problem = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper,
                schedule=type(schedule).uniform(0, grid.length, grid.length / steps),
            ),
        )
        values = []
        for implementation in (NumpyRefBackend, NumbaCpuBackend):
            backend = implementation(problem, grid=grid)
            try:
                backend.set_initial(problem.initial.field0, None)
                backend.propagate_to(grid.length)
                values.append(backend.observe().require_field().copy())
            finally:
                backend.close()
    np.testing.assert_allclose(
        values[1], values[0], rtol=0, atol=bound * np.max(np.abs(values[0]))
    )

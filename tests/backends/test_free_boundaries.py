# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise Cartesian backend diagnostics and allocator accounting boundaries."""

import copy
import ctypes
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backends.test_radial_nonlinear import make_problem
from solprop.backends.numba_cpu import NumbaCpuBackend, fft
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError, GridError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.spatial import cartesian_fluence_widths


@pytest.fixture
def free_backend_problem():
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 4, 4e-4, 6, window_factor=0)
    problem = make_problem(
        grid,
        space,
        np.zeros((space.n_columns, 1, grid.n_omega), complex),
        [KerrResponse(1e-22, thg=False)],
    )
    return grid, problem


@pytest.mark.parametrize("missing", ["plans", "coefficients", "geometry"])
def test_cartesian_numba_stage_names_missing_resources(
    free_backend_problem, missing, monkeypatch
):
    grid, problem = free_backend_problem
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        bundle = backend._stage_scalars(0.0, 0.0)
        if missing == "geometry":
            broken = copy.copy(problem)
            transform = copy.copy(problem.transform)
            object.__setattr__(transform, "geometry", None)
            object.__setattr__(broken, "transform", transform)
            monkeypatch.setattr(backend, "_problem", broken)
        else:
            monkeypatch.setattr(
                backend, "_transverse" if missing == "plans" else "_separable", None
            )
        with pytest.raises(BackendError, match="Cartesian stage requires"):
            backend._stage_free(backend._require_state(), 0, 0.0, bundle)
    finally:
        backend.close()


def test_cartesian_reference_stage_names_missing_geometry(free_backend_problem):
    grid, problem = free_backend_problem
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        bundle = backend._stage_scalars(0.0, 0.0)
        transform = copy.copy(problem.transform)
        object.__setattr__(transform, "geometry", None)
        object.__setattr__(problem, "transform", transform)
        with pytest.raises(BackendError, match=r"Cartesian nonlinear.*geometry"):
            backend._nonlinear(
                problem.initial.field0, bundle, np.ones(problem.shape, complex)
            )
    finally:
        backend.close()


def test_transverse_planner_rejects_unknown_effort():
    with pytest.raises(BackendError, match="planning flag"):
        fft.build_transverse_plans((4, 6, 1, 8), flag="unknown", threads=1)


def test_allocator_counter_counts_glibc_heap_and_mapped_bytes(monkeypatch):
    def mallinfo2():
        return SimpleNamespace(uordblks=37, hblkhd=91)

    allocator = SimpleNamespace(mallinfo2=mallinfo2)
    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "CDLL", lambda _: allocator)
        patch.setattr(sys, "platform", "linux")
        assert fft._heap_bytes() == 128


def test_allocator_counter_counts_macos_used_bytes_on_any_host(monkeypatch):
    def statistics(_zone, output):
        result = ctypes.cast(output, ctypes.POINTER(fft._MallocStatistics)).contents
        result.size_in_use = 37
        result.size_allocated = 91

    allocator = SimpleNamespace(malloc_zone_statistics=statistics)
    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "CDLL", lambda _: allocator)
        patch.setattr(sys, "platform", "darwin")
        assert fft._heap_bytes() == 37


def test_allocator_counter_is_zero_on_an_unsupported_platform(monkeypatch):
    # The row is a measurement, not a requirement: the default backend must still
    # construct where neither allocator interface exists.
    with monkeypatch.context() as patch:
        # A C library without either interface, whatever the real platform has.
        patch.setattr(ctypes, "CDLL", lambda _: SimpleNamespace())
        patch.setattr(sys, "platform", "freebsd")
        assert fft._heap_bytes() == 0


def test_allocator_counter_is_zero_on_windows_without_a_process_handle(monkeypatch):
    def no_handle(_name):
        raise TypeError("Windows has no process-wide library handle")

    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "CDLL", no_handle)
        patch.setattr(sys, "platform", "win32")
        assert fft._heap_bytes() == 0


@pytest.mark.parametrize("radius", [1.4e308, float(np.nextafter(0.0, 1.0))])
def test_free_grid_names_unrepresentable_extents(radius):
    with pytest.raises(GridError, match="extent or spacing"):
        FreeGrid(radius, 4, 1.0, 6, window_factor=0.5)


def test_cartesian_widths_reject_nonnumeric_samples():
    with pytest.raises(GridError, match="must be real"):
        cartesian_fluence_widths([0.0, 1.0], [0.0, 1.0], ["bad"] * 4)


def test_cartesian_edge_peak_has_no_inscribed_threshold_width():
    widths = cartesian_fluence_widths([0.0, 1.0], [0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    assert widths[0] == 0.0
    assert np.all(np.isnan(widths[1:]))


def test_cartesian_widths_report_nonrepresentable_moment_arithmetic():
    axis = np.array([-8e307, 0.0, 8e307])
    fluence = np.zeros(9)
    fluence[0] = 1.0
    # Every input is finite. The far, zero-fluence corner's radius overflows;
    # both NumPy diagnostics remain visible and the public function raises.
    with (
        pytest.warns(RuntimeWarning, match="overflow|invalid"),
        pytest.raises(GridError, match="moments are not representable"),
    ):
        cartesian_fluence_widths(axis, axis, fluence)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Private Cartesian native boundary and persistent buffer accounting."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest
from conftest import PhaseACase
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import KernelProblem, assemble

from solprop_native import NativeBackend, _core
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def free_payload(phase_a_case: PhaseACase) -> dict[str, object]:
    original, grid = phase_a_case
    space = FreeGrid(1e-3, 4, 1.5e-3, 6, window_factor=0)
    problem = assemble(
        grid,
        None,
        [],
        np.zeros((24, 1, grid.n_omega), complex),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=original.stepper.controller,
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=5)
    )
    backend = NativeBackend(problem, grid=grid)
    try:
        return _native_problem(problem, backend._terms)  # pyright: ignore[reportPrivateUsage]
    finally:
        backend.close()


@pytest.mark.parametrize(
    "name",
    [
        "spatial_k_squared",
        "spatial_k_squared_low",
        "spatial_numerator",
        "spatial_clamped",
        "spatial_frame",
        "spatial_spectral_rate",
        "spatial_kperp_sq",
        "spatial_kperp_sq_low",
        "spatial_absorber_rate",
        "spatial_kperp_rate",
        "spatial_k_x",
        "spatial_k_y",
    ],
)
@pytest.mark.parametrize("spoiling", ["length", "nonfinite", "dtype"])
def test_free_native_vectors_reject_invalid_data(
    free_payload: dict[str, object], name: str, spoiling: str
) -> None:
    values = np.array(cast(np.ndarray, free_payload[name]), copy=True)
    if spoiling == "length":
        values = values[:-1]
    elif spoiling == "nonfinite":
        values[0] = np.nan
    else:
        values = values.astype(np.complex64 if np.iscomplexobj(values) else np.float32)
    free_payload[name] = values
    with pytest.raises((TypeError, ValueError)):
        _core.State(free_payload)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_x", 3),
        ("n_x", 1),
        ("n_y", 1),
        ("n_y", 0),
        ("spatial_evanescent_rate", 0.0),
        ("spatial_evanescent_rate", np.inf),
    ],
)
def test_free_native_axes_and_clamp_are_checked(
    free_payload: dict[str, object], name: str, value: float
) -> None:
    free_payload[name] = value
    with pytest.raises((TypeError, ValueError)):
        _core.State(free_payload)


def test_the_far_field_payload_belongs_to_free_space(
    free_payload: dict[str, object], planar_payload: dict[str, object]
) -> None:
    """solprop SEP 23: the axes and the slot count are free space's alone."""
    assert free_payload["spatial_far_field_slots"] == 0
    assert cast(np.ndarray, free_payload["spatial_k_x"]).shape == (4,)
    assert cast(np.ndarray, free_payload["spatial_k_y"]).shape == (6,)
    with pytest.raises(ValueError, match="belong to free space"):
        _core.State(planar_payload | {"spatial_far_field_slots": 3})


def test_free_native_ledger_retains_one_transverse_volume_and_chunked_time(
    free_payload: dict[str, object],
) -> None:
    state = _core.State(free_payload)
    try:
        ledger = state.memory_ledger()
        omega, time = (
            int(cast(int, free_payload["n_omega"])),
            int(cast(int, free_payload["n_to"])),
        )
        assert ledger["transverse_scratch"] == 24 * omega * 16
        assert ledger["modal_time"] == 5 * time * 16
        assert ledger["fine_spectrum"] == 5 * time * 16
        assert ledger["projected_spectrum"] == 5 * time * 16
        assert ledger["fixed_phi_terms"] == ledger["fixed_prefactor_terms"] == 0
        assert ledger["transform_matrices"] == 0
        expected = sum(
            cast(np.ndarray, free_payload[name]).nbytes
            for name in (
                "spatial_k_squared",
                "spatial_k_squared_low",
                "spatial_numerator",
                "spatial_clamped",
                "spatial_frame",
                "spatial_spectral_rate",
                "spatial_kperp_sq",
                "spatial_kperp_sq_low",
                "spatial_absorber_rate",
                "spatial_kperp_rate",
            )
        )
        assert ledger["spatial_coefficients"] == expected
    finally:
        state.close()


@pytest.mark.parametrize("slots", [3, 7, 8])
def test_adams_dense_storage_is_optional_and_reallocation_is_idempotent(
    free_payload: dict[str, object], slots: int
) -> None:
    state = _core.State(free_payload)
    try:
        field_bytes = state.memory_ledger()["state"]
        omega = int(cast(int, free_payload["n_omega"]))
        initial = np.zeros(field_bytes // 16, dtype=np.complex128)
        state.set_initial(initial, np.empty(0, dtype=np.complex128))
        for dense in (False, True, True, False):
            state.adams_allocate(slots, False, dense)
            expected = (2 * slots + 4 + (slots + 4 if dense else 0)) * field_bytes
            assert state.memory_ledger()["adams_history"] == expected + 8 * omega
            if dense:
                state.adams_hold()
                state.adams_restore()
                np.testing.assert_array_equal(
                    state.field_to_host().reshape(-1), initial
                )
            else:
                with pytest.raises(RuntimeError, match="dense observation storage"):
                    state.adams_hold()
                with pytest.raises(RuntimeError, match="dense observation storage"):
                    state.adams_restore()
    finally:
        state.close()
        state.close()


def test_adams_rows_cross_to_the_host_and_back_exactly(
    free_payload: dict[str, object],
) -> None:
    # solprop SEP 29 G2: the resident history's rows are exported for a resume and
    # installed again; a copy, so exact, and refused by name, extent, dtype and order
    # before anything is copied.
    state = _core.State(free_payload)
    try:
        columns = state.memory_ledger()["state"] // 16
        omega = int(cast(int, free_payload["n_omega"]))
        state.set_initial(
            np.zeros(columns, dtype=np.complex128), np.empty(0, dtype=np.complex128)
        )
        state.adams_allocate(3, False, False)
        assert state.adams_row_names() == ["store"]
        shape = (3, columns // omega, omega)
        rng = np.random.default_rng(29)
        rows = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        state.adams_rows_from_host("store", rows)
        back = state.adams_rows_to_host("store")
        assert back.shape == shape
        assert back.dtype == np.complex128
        np.testing.assert_array_equal(back, rows)
        with pytest.raises(ValueError, match="must have shape"):
            state.adams_rows_from_host("store", rows[:2])
        with pytest.raises(ValueError, match="must have shape"):
            state.adams_rows_from_host("store", rows.reshape(3, -1))
        with pytest.raises(TypeError):
            state.adams_rows_from_host(
                "store",
                rows.real.copy(),  # pyright: ignore[reportArgumentType]
            )
        with pytest.raises(TypeError):
            state.adams_rows_from_host("store", np.asfortranarray(rows))
        with pytest.raises(ValueError, match="unknown Adams history row"):
            state.adams_rows_to_host("slopes")
        with pytest.raises(RuntimeError, match="not allocated"):
            state.adams_rows_to_host("states")
        # Nothing was copied by a refused call.
        np.testing.assert_array_equal(state.adams_rows_to_host("store"), rows)
    finally:
        state.close()


@pytest.mark.parametrize("collar", [0.0, 0.3])
def test_a_line_of_one_y_row_matches_the_reference(collar: float) -> None:
    """Solprop's LineGrid is the Cartesian contract with n_y = 1.

    The two-axis transverse plan then has a unit y extent, the collar and absorber
    act along x only, and the statistics read the same packed columns. Ten fixed
    Kerr steps and the spatial statistics agree with numpy_ref.
    """
    length = 1e-4
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = LineGrid(3e-4, 24, window_factor=collar)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-3),
        80e-6,
        jones=(1, 0.5j),
    )
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length,
        dz_init=length / 10,
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.shape == (24, 1)
    names = ("energy", "peak_intensity", "on_axis_intensity", "beam_radius", "fwhm_r")
    rows: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for implementation in (NumpyRefBackend, NativeBackend):
        backend = implementation(problem, grid=grid)
        try:
            if isinstance(backend, NativeBackend):
                ledger = backend.memory_ledger()
                assert ledger["transverse_scratch"] == 2 * 24 * grid.n_omega * 16
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(length)
            field = np.array(backend.observe().require_field(), copy=True)
            statistics = {
                name: np.asarray(value)
                for name, value in backend.stats(StatsSpec(names=names)).items()
            }
            rows.append((field, statistics))
        finally:
            backend.close()
    (reference, reference_stats), (actual, actual_stats) = rows
    # Solprop's Cartesian ladder bound for ten steps (tests/backends/test_free_ladders):
    # 70 evaluations of 32 eps per FFT level over log2(n_to) + log2(24) levels.
    levels = int(np.ceil(np.log2(grid.n_to))) + int(np.ceil(np.log2(24)))
    bound = 70 * 32 * np.finfo(float).eps * levels
    scale = float(np.max(np.abs(reference)))
    assert scale > 0
    np.testing.assert_allclose(actual, reference, rtol=0, atol=bound * scale)
    for name in names:
        expected = np.asarray(reference_stats[name])
        np.testing.assert_allclose(
            actual_stats[name],
            expected,
            rtol=0,
            atol=2 * bound * float(np.max(np.abs(expected))),
        )


def _planar_problem(
    grid: EnvelopeGrid, controller: PIController, *, collar: float = 0.0
) -> tuple[PlanarGrid, KernelProblem]:
    """Return a small Kerr slab problem: three orders, 5 y rows, two rows per chunk."""
    # A constant plate index: the fixture's band reaches beyond silica's fit.
    modes = [SlabMode(63.5e-6, order=p, cladding=1.45) for p in (1, 2, 3)]
    space = PlanarGrid(modes, 1e-3, 5, window_factor=collar)
    pulse = PlanarPulse(
        GaussianPulse(wavelength=835e-9, fwhm=100e-15, energy=1e-6),
        2e-4,
        waist_x=0.735 * 63.5e-6,
        offset_x=10e-6,
    )
    length = 1e-3
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=controller,
        save_planes=[length],
        dz_max=length,
        dz_init=length / 10,
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
    )
    chunk = 2 * space.x.size
    return space, dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )


def test_the_hybrid_planar_kind_matches_the_reference(phase_a_case: PhaseACase) -> None:
    """Schema 15's ``"slab"`` geometry (the hybrid planar kind) against numpy_ref.

    Columns ``(p, k_y)``, nodes ``(x, y)``: the y FFT around the slab rule's
    synthesis and projection, each column's operator with its channel's wall term.
    Ten fixed Kerr steps, with a partial last chunk of y rows, and the spatial
    statistics agree under solprop's planar ladder bound
    (``tests/backends/test_planar_native.py``).
    """
    original, grid = phase_a_case
    space, problem = _planar_problem(grid, original.stepper.controller)
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "slab"
    names = ("energy", "peak_intensity", "on_axis_intensity", "beam_radius", "fwhm_r")
    rows: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    length = 1e-3
    for implementation in (NumpyRefBackend, NativeBackend):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(length)
            field = np.array(backend.observe().require_field(), copy=True)
            statistics = {
                name: np.asarray(value)
                for name, value in backend.stats(StatsSpec(names=names)).items()
            }
            rows.append((field, statistics))
        finally:
            backend.close()
    (reference, reference_stats), (actual, actual_stats) = rows
    # 70 evaluations of 32 eps per FFT level and per term of the two x contractions,
    # and the operator's square root over the run (8 eps beta_max L).
    depth = (
        int(np.ceil(np.log2(grid.n_to)))
        + int(np.ceil(np.log2(space.n_y)))
        + space.n_p
        + int(space.x.size)
    )
    beta = float(np.max(np.abs(grid.omega))) / 299792458.0
    bound = 70 * 32 * np.finfo(float).eps * depth
    bound += 8 * np.finfo(float).eps * beta * length
    scale = float(np.max(np.abs(reference)))
    assert scale > 0
    np.testing.assert_allclose(actual, reference, rtol=0, atol=bound * scale)
    for name in names:
        expected = np.asarray(reference_stats[name])
        np.testing.assert_allclose(
            actual_stats[name],
            expected,
            rtol=0,
            atol=2 * bound * float(np.max(np.abs(expected))),
        )


@pytest.fixture
def planar_payload(phase_a_case: PhaseACase) -> dict[str, object]:
    original, grid = phase_a_case
    _, problem = _planar_problem(grid, original.stepper.controller)
    backend = NativeBackend(problem, grid=grid)
    try:
        return _native_problem(problem, backend._terms)  # pyright: ignore[reportPrivateUsage]
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_p", 2),
        ("n_y", 4),
        ("n_x", 5),
        ("node_chunk", 7),
        ("n_pol_state", 2),
        ("projection_coarse_matrix", "present"),
        ("spatial_kappa_sq", "short"),
        ("spatial_kappa_sq", "nonfinite"),
    ],
)
def test_the_planar_native_payload_is_checked(
    planar_payload: dict[str, object], name: str, value: int | str
) -> None:
    """The planar axes, whole-row chunks, scalar state and wall terms are validated."""
    kappa = np.array(cast(np.ndarray, planar_payload["spatial_kappa_sq"]), copy=True)
    if value == "present":
        planar_payload[name] = np.array(
            cast(np.ndarray, planar_payload["projection_matrix"]), copy=True
        )
    elif value == "short":
        planar_payload[name] = kappa[:-1]
    elif value == "nonfinite":
        kappa[0] = np.nan
        planar_payload[name] = kappa
    else:
        planar_payload[name] = value
    with pytest.raises((TypeError, ValueError)):
        _core.State(planar_payload)


def test_a_free_payload_with_slab_channels_is_refused(
    free_payload: dict[str, object],
) -> None:
    """Slab channels turn a plane into the planar kind, whose shapes it lacks."""
    free_payload["n_p"] = 1
    free_payload["spatial_kappa_sq"] = np.zeros(
        int(cast(int, free_payload["n_omega"])), dtype=np.complex128
    )
    with pytest.raises(ValueError, match="hybrid planar"):
        _core.State(free_payload)

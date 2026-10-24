# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optional Julia qualification against the independent NumPy oracle."""

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bounds import EPS, fft_bound, higham_gamma
from solprop.backends import select
from solprop.backends.julia._backend import JuliaBackend, _configuration, _refuse
from solprop.backends.julia._runtime import runtime, thread_limit
from solprop.backends.memory import estimate_memory
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import error_norm
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.responses import ResponseStage
from solprop.translate.spatial import separable_values

ROOT = Path(__file__).resolve().parents[2]
READY = (
    importlib.util.find_spec("juliacall") is not None
    and (ROOT / ".julia-runtime/julia-1.13.2/bin/julia").is_file()
)


@pytest.fixture(scope="module")
def julia_runtime():
    """Keep the optional runtime in one worker and skip unprepared installations."""
    if not READY:
        pytest.skip("private Julia runtime and solprop[julia] are required")
    return runtime()


def simulation(*, kind="resolved", **options):
    """Use a short public capillary with optmat's existing argon coefficients."""
    if kind in {"radial", "free"}:
        return radial_simulation(free=kind == "free", **options)
    mode_averaged = kind == "averaged"
    settings = {
        "lambda0": 800e-9,
        "wavelength_limits": (300e-9, 2e-6),
        "trange": 100e-15,
        "fwhm": 15e-15,
        "energy": 1e-6,
        "modes": 4,
        "plasma": False,
        "raman": False,
        "integrator": RungeKutta(),
        "backend": "julia",
        "stats": default_spec("multimode"),
        "save_planes": [1e-4],
        "rtol": 1e-8,
        "atol": 1e-14,
    } | options
    if mode_averaged:
        settings.update(modes=1, stats=default_spec("mode_averaged"))
    sim = Simulation.from_capillary(125e-6, 0.01, "Ar", 1.0, **settings)
    if mode_averaged:
        sim.settings = dataclasses.replace(
            sim.settings, quadrature=None, components=None
        )
    return sim


def radial_simulation(
    *, thg=True, absorber_scheme="nonlinear", schedule=None, free=False
):
    """Use a broad beam so the spatial collar contributes to every source check."""
    grid = FieldGrid.from_limits(0.01, 800e-9, (300e-9, 2e-6), 100e-15)
    constructor = Simulation.from_free if free else Simulation.from_radial
    return constructor(
        grid,
        FreeGrid(600e-6, 8, 700e-6, 10) if free else RadialGrid(600e-6, 24),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=thg)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=1e-6), 420e-6
        ),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1e-4],
            dz_init=1e-5,
            dz_max=1e-4,
            integrator=RungeKutta(),
            schedule=schedule,
            backend="julia",
            absorber=absorber(grid),
            absorber_scheme=absorber_scheme,
            stats=default_spec("free" if free else "radial"),
        ),
    )


def pair(problem, sim):
    """Construct both engines on the exact same translated contract."""
    engines = [
        select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in ("julia", "numpy_ref")
    ]
    for engine in engines:
        engine.set_initial(problem.initial.field0, None)
    candidate, reference = engines
    assert isinstance(candidate, JuliaBackend)
    assert isinstance(reference, NumpyRefBackend)
    return candidate, reference


def observed(engine):
    """Assert the default observation includes its host field."""
    field = engine.observe().field
    assert field is not None
    return field


def close_field(actual, expected, operations):
    """Bound normwise accumulation and transforms, including spectral nulls."""
    # Each pass has at most operations rounding contributions at unit conditioning;
    # the scale is global because spectral nulls have no meaningful relative error.
    bound = higham_gamma(operations) * max(float(np.linalg.norm(expected)), EPS)
    assert float(np.linalg.norm(actual - expected)) <= bound


@pytest.mark.parametrize("thg", [True, False])
@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
@pytest.mark.parametrize("kind", ["resolved", "averaged", "radial", "free"])
def test_source_trial_and_eight_steps(julia_runtime, thg, scheme, kind):
    sim = simulation(thg=thg, absorber_scheme=scheme, kind=kind)
    p = sim.translate()
    assert p.flags.mode_averaged == (kind == "averaged")
    # A short last chunk must use only its own quadrature rows.
    if kind != "averaged":
        p = dataclasses.replace(
            p, transform=dataclasses.replace(p.transform, node_chunk=7)
        )
    candidate, reference = pair(p, sim)
    try:
        lab = p.initial.field0
        got = np.asarray(
            julia_runtime.evaluate_b(candidate._engine, lab[0, :, 0, :].T.copy())
        ).T[None, :, None, :]
        bundle = reference._stage_scalars(0.0, 0.0)
        want = reference._nonlinear(lab, bundle, np.ones_like(lab))
        if scheme == "nonlinear":
            # The FFT of the rate times a nearly zero pulse tail is ill-conditioned.
            # Bound each transform's backward error before the multiplication by
            # the collar rate, rather than by the tiny final absorber spectrum.
            rate = p.responses.temporal_absorber_fine
            assert rate is not None
            scale = np.linalg.norm(want) + 0.5 * np.max(rate) * np.linalg.norm(lab)
            assert np.linalg.norm(got - want) <= fft_bound(p.grids.n_to) * scale
        else:
            close_field(got, want, 2000 * p.grids.n_to)
        for index in range(8):
            z, dz = index * 1e-5, 1e-5
            before = observed(candidate).copy()
            norm, raw = candidate._trial_step(z, dz)
            other_norm, other = reference._trial_step(z, dz)
            got = np.asarray(julia_runtime.candidate(candidate._engine)).T[
                None, :, None, :
            ]
            close_field(got, other, (index + 1) * 3000 * p.grids.n_to)
            assert np.isfinite(norm) and np.isfinite(other_norm)
            error = np.asarray(julia_runtime.error_snapshot(candidate._engine)).T[
                None, :, None, :
            ]
            reduced = error_norm(
                error, before, got, p.grids.norm_mask, p.stepper.controller
            )
            # Independent reduction of the same embedded defect isolates the norm
            # from the cancellation-sensitive difference of stage histories.
            close_field(np.asarray(norm), np.asarray(reduced), 16 * before.size)
            # A trial never mutates accepted state, including one subsequently rejected.
            np.testing.assert_array_equal(observed(candidate), before)
            candidate._state = candidate._finish(raw, dz)
            reference._state = reference._finish(other, dz)
        close_field(observed(candidate), observed(reference), 24000 * p.grids.n_to)
    finally:
        candidate.close()
        reference.close()


@pytest.mark.parametrize("kind", ["resolved", "averaged", "radial", "free"])
def test_fixed_schedule_statistics_and_reinitialize(julia_runtime, kind):
    sim = simulation(schedule=FixedSchedule(np.full(8, 1e-4 / 8)), kind=kind)
    p = sim.translate()
    candidate, reference = pair(p, sim)
    try:
        for engine in (candidate, reference):
            engine.propagate_to(1e-4)
        original = observed(candidate).copy()
        close_field(original, observed(reference), 24000 * p.grids.n_to)
        got, want = candidate.stats(p.stats_spec), reference.stats(p.stats_spec)
        for name in got:
            close_field(got[name], want[name], 24000 * p.grids.n_to)
        intensity = StatsSpec(names=("peak_intensity",))
        close_field(
            candidate.stats(intensity)["peak_intensity"],
            reference.stats(intensity)["peak_intensity"],
            24000 * p.grids.n_to,
        )
        first = observed(candidate)
        assert not first.flags.writeable
        candidate.set_initial(p.initial.field0, None)
        np.testing.assert_array_equal(first, original)
        candidate.propagate_to(1e-4)
        close_field(observed(candidate), original, 24000 * p.grids.n_to)
        assert candidate.metadata()["source_evaluations"] == 49
        assert all(value > 0 for value in candidate.memory_ledger().values())
        assert estimate_memory(p, "julia").total == sum(
            candidate.memory_ledger().values()
        )
    finally:
        candidate.close()
        candidate.close()
        reference.close()
    with pytest.raises(BackendError, match="closed"):
        candidate.observe()


@pytest.mark.parametrize("kind", ["resolved", "averaged", "radial", "free"])
def test_adaptive_public_run(julia_runtime, kind):
    sim = simulation(kind=kind)
    p = sim.translate()
    candidate, reference = pair(p, sim)
    try:
        for engine in (candidate, reference):
            engine.propagate_to(1e-4)
        # Short, weakly nonlinear case: tighten the reference 100-fold for a global
        # accuracy envelope, independent of the adaptive trajectory's last bits.
        tight = dataclasses.replace(
            p,
            stepper=dataclasses.replace(
                p.stepper,
                controller=dataclasses.replace(
                    p.stepper.controller, rtol=1e-10, atol=1e-16
                ),
            ),
        )
        oracle = select("numpy_ref", tight, grid=sim.grid)
        try:
            oracle.set_initial(p.initial.field0, None)
            oracle.propagate_to(1e-4)
            error = np.linalg.norm(observed(candidate) - observed(oracle))
            assert error <= 1e-7 * np.linalg.norm(observed(oracle))
        finally:
            oracle.close()
        output = sim.run(progress="none")
        try:
            assert output.metadata is not None
            assert (
                output.metadata.simulation_kind
                == (
                    {
                        "averaged": "mode_averaged",
                        "resolved": "multimode",
                        "radial": "radial",
                        "free": "free",
                    }[kind]
                )
            )
            assert np.all(np.isfinite(output.fields("fft")))
        finally:
            output.close()
    finally:
        candidate.close()
        reference.close()


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"integrator": Multistep(formulation="interaction")}, "Adams"),
        ({"noise": "input", "seed": 3}, "noise"),
        ({"interpolate_planes": True}, "interpolated"),
    ],
)
def test_unsupported_contracts_refuse_before_runtime(options, reason):
    p = simulation(**(options | {"backend": "numpy_ref"})).translate()
    with pytest.raises(BackendError, match=reason):
        _refuse(p)


def test_statistics_refusal_and_nonfinite_input(julia_runtime):
    sim = simulation()
    p = sim.translate()
    engine = JuliaBackend(p, grid=sim.grid)
    try:
        bad = p.initial.field0.copy()
        bad.flat[0] = np.nan
        with pytest.raises(BackendError, match="finite"):
            engine.set_initial(bad, None)
        engine.set_initial(p.initial.field0, None)
        with pytest.raises(BackendError, match="statistics"):
            engine.stats(StatsSpec(names=("beam_radius",)))
    finally:
        engine.close()


@pytest.mark.parametrize("threads", [1, 4])
def test_isolated_interop(julia_runtime, threads):
    # Strip only settings installed by this test process's private runtime; the
    # subprocess must initialize a fresh runtime at its requested task width.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTHON_JULIACALL_")
    } | {"SOLPROP_JULIA_THREADS": str(threads)}
    subprocess.run(
        [sys.executable, str(ROOT / "julia/interop.py")],
        env=env,
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", ["resolved", "averaged", "radial", "free"])
def test_rejected_candidate_and_linear_mask(julia_runtime, kind):
    sim = simulation(kind=kind)
    original = sim.translate()
    if kind in {"radial", "free"}:
        geometry = original.transform.geometry
        assert geometry is not None
        original = dataclasses.replace(
            original,
            transform=dataclasses.replace(
                original.transform,
                geometry=dataclasses.replace(
                    geometry, absorber_rate=np.zeros_like(geometry.absorber_rate)
                ),
            ),
        )
    p = dataclasses.replace(
        original,
        responses=dataclasses.replace(
            original.responses,
            temporal_absorber=None,
            temporal_absorber_fine=None,
            kerr_coefficients=None,
            response_order=(ResponseStage.ACCUMULATE,),
        ),
        flags=dataclasses.replace(original.flags, has_temporal_absorber=False),
    )
    engine = JuliaBackend(p, grid=sim.grid)
    try:
        engine.set_initial(p.initial.field0, None)
        before = observed(engine).copy()
        engine._trial_step(0.0, 1e-3)  # Discard this candidate deliberately.
        np.testing.assert_array_equal(observed(engine), before)
        norm, candidate = engine._trial_step(0.0, 1e-5)
        assert norm == 0.0  # Zero source gives an exactly zero embedded defect.
        engine._state = engine._finish(candidate, 1e-5)
        linear = p.linear.phi_constant
        if p.linear.separable is not None:
            linear, _ = separable_values(p.linear.separable)
        assert linear is not None
        expected = before * np.exp(1e-5 * linear)
        # One complex exponential and product, no source accumulation.
        close_field(observed(engine), expected, 32)
        np.testing.assert_array_equal(observed(engine)[..., ~p.grids.active_mask], 0)
    finally:
        engine.close()


def test_default_capillary_statistics(julia_runtime):
    sim = simulation(stats=None, schedule=FixedSchedule(np.full(8, 1e-4 / 8)))
    p = sim.translate()
    engines = [
        select(name, p, grid=sim.grid, modes=sim.modes, profiles=sim.profiles)
        for name in ("julia", "numpy_ref")
    ]
    try:
        results = []
        for engine in engines:
            engine.set_initial(p.initial.field0, None)
            engine.propagate_to(1e-4)
            results.append(engine.stats(p.stats_spec))
        assert results[0].keys() == results[1].keys()
        for name in results[0]:
            got, expected = results[0][name], results[1][name]
            if name == "quadrature_error":
                # Difference of two near-equal projections, normalised by source.
                assert np.max(np.abs(got - expected)) <= higham_gamma(
                    16 * p.transform.n_nodes * p.grids.n_to
                )
            else:
                close_field(got, expected, 24000 * p.grids.n_to)
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_radial_absorber_without_material_source(julia_runtime, scheme):
    sim = radial_simulation(absorber_scheme=scheme)
    original = sim.translate()
    prefactor = original.responses.prefactor
    assert isinstance(prefactor, PiecewiseChebyshev)
    coefficients = np.zeros_like(prefactor.coefficients)
    coefficients.setflags(write=False)
    p = dataclasses.replace(
        original,
        responses=dataclasses.replace(
            original.responses,
            temporal_absorber=None,
            temporal_absorber_fine=None,
            prefactor_constant=np.zeros_like(original.responses.prefactor_constant),
            prefactor=dataclasses.replace(prefactor, coefficients=coefficients),
            kerr_coefficients=None,
            response_order=(ResponseStage.ACCUMULATE,),
        ),
        flags=dataclasses.replace(original.flags, has_temporal_absorber=False),
    )
    engine = JuliaBackend(p, grid=sim.grid)
    try:
        field = p.initial.field0[0, :, 0, :]
        transform, geometry = p.transform, p.transform.geometry
        assert geometry is not None
        # A time-independent loss commutes with the temporal transform. Apply
        # it directly at radial nodes, without using the backend's composed map.
        synthesis = transform.synthesis / p.grids.pad_scale
        physical = synthesis @ field
        expected = transform.projection @ (
            -0.5 * geometry.absorber_rate[:, None] * physical
        )
        expected *= p.grids.active_mask
        got = np.asarray(julia_runtime.evaluate_b(engine._engine, field.T.copy())).T
        assert np.linalg.norm(expected) > 0
        scale = (
            np.linalg.norm(transform.projection, 2)
            * 0.5
            * np.max(geometry.absorber_rate)
            * np.linalg.norm(synthesis, 2)
            * np.linalg.norm(field)
        )
        bound = fft_bound(p.grids.n_to) + higham_gamma(4 * transform.n_nodes)
        assert np.linalg.norm(got - expected) <= bound * scale
    finally:
        engine.close()


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_thread_limit(monkeypatch, value):
    monkeypatch.setenv("SOLPROP_JULIA_THREADS", value)
    with pytest.raises(BackendError, match="positive integer"):
        thread_limit("SOLPROP_JULIA_THREADS")


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_cartesian_absorber_without_material_source(julia_runtime, scheme):
    sim = radial_simulation(free=True, absorber_scheme=scheme)
    original = sim.translate()
    data = original.linear.separable
    assert data is not None
    data = dataclasses.replace(
        data,
        prefactor_numerator=np.zeros_like(data.prefactor_numerator),
        prefactor_clamped=np.zeros_like(data.prefactor_clamped),
    )
    p = dataclasses.replace(
        original,
        linear=dataclasses.replace(original.linear, separable=data),
        responses=dataclasses.replace(
            original.responses,
            separable=data,
            temporal_absorber=None,
            temporal_absorber_fine=None,
            kerr_coefficients=None,
            response_order=(ResponseStage.ACCUMULATE,),
        ),
        flags=dataclasses.replace(original.flags, has_temporal_absorber=False),
    )
    engine = JuliaBackend(p, grid=sim.grid)
    try:
        geometry = p.transform.geometry
        assert geometry is not None
        nx, ny = geometry.shape
        field = p.initial.field0[0, :, 0, :]
        shaped = field.reshape(ny, nx, -1)
        rate = geometry.absorber_rate.reshape(ny, nx, 1)
        expected = (
            np.fft.fft2(
                -0.5 * rate * np.fft.ifft2(shaped, axes=(0, 1)), axes=(0, 1)
            ).reshape(field.shape)
            * p.grids.active_mask
        )
        got = np.asarray(julia_runtime.evaluate_b(engine._engine, field.T.copy())).T
        assert np.linalg.norm(expected) > 0
        scale = 0.5 * np.max(rate) * np.linalg.norm(field)
        bound = fft_bound(p.grids.n_to) + fft_bound(nx) + fft_bound(ny)
        assert np.linalg.norm(got - expected) <= bound * scale
    finally:
        engine.close()


def test_cartesian_separable_elements(julia_runtime):
    sim = simulation(kind="free")
    p = sim.translate()
    data = p.linear.separable
    assert data is not None
    # Deliberately cancelled leading squares: low residuals determine which side
    # of grazing each column occupies. Include lossy roots and the attenuation cap.
    square = np.full_like(data.k_squared, 1e16)
    square[::3] += 2j
    low = np.full_like(square, 0.25)
    transverse = np.resize(np.array([0.0, 1e16, 1e16 + 2, 2e16]), p.shape[1])
    transverse_low = np.resize(np.array([0.0, 0.125, -4.0, 0.0]), p.shape[1])
    data = dataclasses.replace(
        data,
        k_squared=square,
        k_squared_low=low,
        kperp_sq=transverse,
        kperp_sq_low=transverse_low,
    )
    p = dataclasses.replace(
        p,
        linear=dataclasses.replace(p.linear, separable=data),
        responses=dataclasses.replace(p.responses, separable=data),
    )
    # Exercise scalar seam arithmetic on synthetic squares, independently of the
    # physical grid validator (whose FFT axes intentionally differ here).
    engine = julia_runtime.create(_configuration(p, 1))
    try:
        linear, prefactor = separable_values(data)
        for m in range(4):
            for f in (0, 1, 2, 30, 100):
                rate, source = julia_runtime.free_element(engine.config, f + 1, m + 1)
                # Same compensated input and principal root; allowance covers root,
                # division and final complex construction with no reassociation.
                close_field(np.asarray(rate), np.asarray(linear[0, m, 0, f]), 32)
                close_field(np.asarray(source), np.asarray(prefactor[m, 0, f]), 32)
    finally:
        engine = None

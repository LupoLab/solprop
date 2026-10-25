# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Radial Kerr normalisation, parity, and complete nonlinear backend ladders."""

import dataclasses
import math
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import j0

from backends.test_polarised_runs import QuadraticRate as EnergyRate
from backends.test_polarised_runs import plasma_work_bound
from bounds import gamma, ladder_bound
from legacy_geometry import LEGACY_GRID
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import ionisation_potential
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble
from spatial_records import record_metric


@pytest.mark.parametrize("spacing", [0.04e-15, 0.02e-15, 0.01e-15])
def test_radial_circular_plasma_pays_for_electrons_in_the_area_measure(
    spatial_backend_class, spacing, request
):
    tau, amplitude, density = 6e-15, 1e10, 1e22
    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (200e-9, 4e-6), 100e-15, dt=spacing, **LEGACY_GRID
    )
    # One Bessel eigenfunction makes the transverse adjoint identity exact for
    # the raw quadrature weights: wk_j F_ji = wr_i B_ij. No T^2=I assumption.
    space = RadialGrid(0.1, 16, window_factor=0.0)
    profile = j0(space.k_r[0] * space.r)
    potential = ionisation_potential("He") - ionisation_potential("Ar")
    exposure = 3e12 * tau * math.sqrt(math.pi / 2)
    expected = (
        density
        * potential
        * np.dot(space.radial_weights, -np.expm1(-exposure * profile**2))
    )
    # Exact k_z adds k/k_z to the vacuum-measure work. Its maximum deviation on
    # the active band multiplies a Cauchy-Schwarz bound on |integral E' P|.
    # ||P||_2 <= T ||J||_2 and 1-exp(-x) >= x exp(-x) give the last factor.
    wave_min = float(grid.omega[grid.active_mask].min()) / C
    ratio = (space.k_r[0] / wave_min) ** 2
    multiplier = ratio / (math.sqrt(1 - ratio) * (1 + math.sqrt(1 - ratio)))
    nonparaxial = multiplier * np.pi * grid.n_to * math.exp(exposure)
    bound = plasma_work_bound(grid, tau, amplitude, potential)
    bound += nonparaxial + gamma(128 * space.n_r)
    bound = ladder_bound(spatial_backend_class.name, bound, grid.n_to, 1)
    assert 0 < bound < 0.1
    temporal = (
        amplitude
        * np.exp(-((grid.t / tau) ** 2))
        * np.stack([np.cos(grid.omega0 * grid.t), np.sin(grid.omega0 * grid.t)])
    )
    start = np.zeros((space.n_r, 2, grid.n_omega), complex)
    start[0] = 2 * np.pi / space.spectral_weights[0] * np.fft.rfft(temporal, axis=-1)
    table = build_rate_table(EnergyRate(), (1e3, 1e11))
    derivatives = []
    for gas in ("He", "Ar"):
        problem = make_problem(
            grid, space, start, [PlasmaResponse(gas, table)], density=density
        )
        derivatives.append(derivative(spatial_backend_class, problem, grid))
    difference = derivatives[0] - derivatives[1]
    assert np.all(difference[..., (0, -1)] == 0)
    work = (
        -4
        * EPS0
        * C
        * grid.dt
        / grid.n_t
        * np.sum(
            space.spectral_weights[:, None, None]
            * np.real(np.conj(start[..., 1:-1]) * difference[..., 1:-1])
        )
    )
    assert abs(work / expected - 1) < bound
    record_metric(
        request,
        "circular plasma area energy (relative)",
        grid.n_to,
        abs(work / expected - 1),
        bound,
        spatial_backend_class.name,
    )


LENGTH = 1e-4
CHI3 = 1e-22


def derivative(spatial_backend_class, problem, grid):
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if isinstance(backend, NumpyRefBackend):
            backend._hoist(0.0)
            return backend._nonlinear(
                backend._require_state(),
                backend._stage_scalars(0.0, 0.0),
                np.ones(problem.shape, complex),
            )[0]
        # This probe requests N(E, z=0), not an advanced field. A zero-length
        # trial exposes the same first stage without later stages inventing an
        # unstable finite step near a resolved Cartesian cutoff.
        backend._trial_step(0.0, 0.0)
        values = (
            backend._native_state.stage_to_host(0)
            if backend.name == "native"
            else backend._stages[0].copy()
        )
        return values.reshape(problem.shape[1:])
    finally:
        backend.close()


def make_problem(grid, space, start, responses, *, steps=8, density=1.0):
    return assemble(
        grid,
        None,
        responses,
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        density=density,
    )


def test_radial_vector_raman_keeps_the_named_refusal(spatial_backend_class):
    grid = EnvelopeGrid.from_limits(
        LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    response = RamanResponse(
        (DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=False
    )
    with pytest.raises(TranslationError, match="Raman"):
        make_problem(
            grid, space, np.zeros((space.n_r, 2, grid.n_omega), complex), [response]
        )
    scalar = make_problem(
        grid, space, np.zeros((space.n_r, 1, grid.n_omega), complex), [response]
    )
    problem = dataclasses.replace(scalar, shape=(1, space.n_r, 2, grid.n_omega))
    with pytest.raises(BackendError, match="Raman"):
        spatial_backend_class(problem, grid=grid)


@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
@pytest.mark.parametrize("circular", [False, True])
def test_monochromatic_kerr_uses_the_exact_spatial_prefactor(
    spatial_backend_class, family, circular, request
):
    constructor = (
        partial(EnvelopeGrid.from_limits, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits
    )
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 32, window_factor=0.0)
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    amplitude = 2e8
    spectrum = amplitude * (80e-6) ** 2 / 2 * np.exp(-((80e-6 * space.k_r) ** 2) / 4)
    jones = np.array([1, 1j]) / math.sqrt(2) if circular else np.array([1 + 0j])
    temporal_scale = grid.n_t if grid.kind == "envelope" else grid.n_t / 2
    start = np.zeros((space.n_r, len(jones), grid.n_omega), complex)
    start[:, :, index] = temporal_scale * spectrum[:, None] * jones
    # Independent Fourier-Bessel coefficients. Expanding real (Ex^2+Ey^2)Ep
    # at the fundamental gives 3/4 for linear and 1/2 for circular analytic light.
    kernel = j0(space.r[:, None] * space.k_r[None, :])
    backward = kernel * space.spectral_weights[None] / (2 * np.pi)
    forward = kernel.T * space.radial_weights[None] / (2 * np.pi)
    physical = backward @ spectrum
    cubic_factor = 0.5 if circular else 0.75
    wave = grid.omega[index] / C
    prefactor = (
        -0.5j * grid.omega[index] ** 2 / (EPS0 * C**2 * np.sqrt(wave**2 - space.k_r**2))
    )
    # The nonlinear response is multiplied by the fine temporal window before
    # its forward FFT. A monochromatic field reaches that collar; include its
    # exactly sampled modulation, including the real cubic's third-harmonic tail.
    phase = np.exp(
        2j * np.pi * int(grid.crop_index[index]) * np.arange(grid.n_to) / grid.n_to
    )
    analytic = jones[:, None] * phase
    if family == "field_real":
        real = np.real(analytic)
        temporal_cubic = np.sum(real**2, axis=0) * real
    else:
        temporal_cubic = cubic_factor * analytic
        if grid.kind == "field":
            temporal_cubic = temporal_cubic.real
    windowed = temporal_cubic * grid.t_window_fine
    temporal_spectrum = (
        np.fft.fft(windowed, axis=-1)
        if grid.kind == "envelope"
        else np.fft.rfft(windowed, axis=-1)
    )[:, grid.crop_index]
    all_wave = grid.omega[grid.active_mask] / C
    all_prefactor = np.zeros((space.n_r, grid.n_omega), complex)
    all_prefactor[:, grid.active_mask] = (
        -0.5j
        * grid.omega[grid.active_mask] ** 2
        / (EPS0 * C**2 * np.sqrt(all_wave**2 - space.k_r[:, None] ** 2))
    )
    expected = (
        (forward @ (EPS0 * CHI3 * physical**3))[:, None, None]
        * temporal_spectrum[None]
        * all_prefactor[:, None, :]
        * grid.crop_scale
    )
    # Absolute product majorant remains valid when oscillating Bessel terms cancel.
    majorant = temporal_scale * np.max(
        np.abs(prefactor)
        * (
            np.abs(forward)
            @ (cubic_factor * EPS0 * CHI3 * (np.abs(backward) @ spectrum) ** 3)
        )
    )
    bound = (
        ladder_bound(
            spatial_backend_class.name,
            gamma(32 * space.n_r + 128 * math.ceil(math.log2(grid.n_to))),
            grid.n_to,
            1,
        )
        * majorant
    )
    problem = make_problem(
        grid, space, start, [KerrResponse(CHI3, thg=family == "field_real")]
    )
    actual = derivative(spatial_backend_class, problem, grid)
    assert_allclose(actual, expected, rtol=0, atol=bound)
    record_metric(
        request,
        "Kerr derivative (V)",
        f"{family}, circular={circular}",
        np.max(np.abs(actual - expected)),
        bound,
        spatial_backend_class.name,
    )


class QuadraticRate:
    """An analytic ionisation law, independent of material fit data."""

    threshold = 1e3
    maximum = 1e17
    citation = "Analytic test rate W=3e12*(E/1e9)^2 s^-1"
    parameters = (
        ("model", "quadratic-radial-oracle"),
        ("reference", 1e9),
        ("scale", 3e12),
    )

    def __call__(self, field_amplitude):
        values = np.abs(np.asarray(field_amplitude))
        return np.where(
            values < self.threshold,
            0.0,
            np.minimum(self.maximum, 3e12 * (values / 1e9) ** 2),
        )


@pytest.fixture(
    params=["field_kerr", "envelope_kerr", "field_plasma", "envelope_raman"]
)
def nonlinear_case(request):
    kind = request.param
    constructor = (
        FieldGrid.from_limits
        if kind.startswith("field")
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    # Resolve the held-current collar while preserving the non-plasma fixtures.
    window = 200e-15 if kind == "field_plasma" else 150e-15
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    jones = (1 + 0j,) if kind.endswith("raman") else (1 + 0j, 0.5j)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6, jones=jones
    )
    density = 1.0
    if kind.endswith("plasma"):
        responses = [
            PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))
        ]
        density = 1e22
    elif kind.endswith("raman"):
        responses = [
            RamanResponse(
                (DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=False
            )
        ]
    else:
        responses = [KerrResponse(CHI3, thg=False)]
    return (
        kind,
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        density,
    )


@pytest.mark.parametrize("steps", [0, 1, 8])
def test_radial_nonlinear_ladder_matches_reference(
    nonlinear_case, spatial_backend_class, steps, request
):
    _, grid, space, start, responses, density = nonlinear_case
    evaluations = max(1, 7 * steps)
    # Same CPU FFT-depth envelope as the modal ladders, two length-N radial
    # reductions per evaluation; plasma adds three sequential temporal integrals.
    per_call = 16 * np.finfo(float).eps * math.ceil(math.log2(grid.n_to)) + 2 * gamma(
        space.n_r
    )
    if isinstance(responses[0], PlasmaResponse):
        per_call += gamma(12 * grid.n_to)
    bound = ladder_bound(
        spatial_backend_class.name, evaluations * per_call, grid.n_to, evaluations
    )
    problem = make_problem(
        grid, space, start, responses, steps=max(1, steps), density=density
    )
    if steps == 0:
        reference = derivative(NumpyRefBackend, problem, grid)
        actual = derivative(spatial_backend_class, problem, grid)
        assert_allclose(
            actual, reference, rtol=0, atol=bound * np.max(np.abs(reference))
        )
        record_metric(
            request,
            f"nonlinear ladder {nonlinear_case[0]} (relative)",
            "N",
            np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
            bound,
            spatial_backend_class.name,
        )
        return
    values = []
    statistics = []
    spec = StatsSpec(names=("energy", "on_axis_intensity", "peak_intensity"))
    for implementation in (NumpyRefBackend, spatial_backend_class):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            summary = backend.propagate_to(LENGTH)
            assert summary.n_steps == steps
            values.append(backend.observe().require_field().copy())
            statistics.append(backend.stats(spec))
        finally:
            backend.close()
    assert_allclose(
        values[1], values[0], rtol=0, atol=bound * np.max(np.abs(values[0]))
    )
    record_metric(
        request,
        f"nonlinear ladder {nonlinear_case[0]} (relative)",
        steps,
        np.max(np.abs(values[1] - values[0])) / np.max(np.abs(values[0])),
        bound,
        spatial_backend_class.name,
    )
    # Squared moduli double the relative amplitude perturbation; statistic
    # transforms and positive reductions contribute their own gamma bound.
    statistic_bound = 2 * bound + gamma(
        128 * space.n_r + 32 * int(np.ceil(np.log2(grid.n_t))) + 8 * grid.n_t
    )
    for name in spec.names:
        assert_allclose(
            statistics[1][name], statistics[0][name], rtol=statistic_bound, atol=0
        )
        record_metric(
            request,
            f"nonlinear ladder {nonlinear_case[0]} {name} (relative)",
            steps,
            np.max(np.abs(statistics[1][name] - statistics[0][name]))
            / np.max(np.abs(statistics[0][name])),
            statistic_bound,
            spatial_backend_class.name,
        )


def matched_parity_grids():
    """Construct a common temporal period and an exact carrier-bin shift."""
    field_grid = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 400e-15)
    carrier_index = int(np.argmin(np.abs(field_grid.omega - field_grid.omega0)))
    carrier = field_grid.omega[carrier_index]
    wavelength = 2 * np.pi * C / carrier
    envelope_grid = EnvelopeGrid.from_limits(
        LENGTH, wavelength, (600e-9, 1.2e-6), 400e-15, dt=field_grid.dt_fine, thg=False
    )
    assert envelope_grid.n_to == field_grid.n_to
    assert envelope_grid.d_omega == field_grid.d_omega
    return field_grid, envelope_grid, carrier_index


@pytest.mark.parametrize("steps", [1, 8])
def test_radial_field_envelope_and_scalar_vector_parity(
    spatial_backend_class, steps, request
):
    field_grid, envelope_grid, carrier_index = matched_parity_grids()
    carrier = field_grid.omega[carrier_index]
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    waist = 80e-6
    spatial = 2e8 * waist**2 / 2 * np.exp(-((waist * space.k_r) ** 2) / 4)
    duration = 30e-15 / math.sqrt(2 * math.log(2))
    temporal = np.exp(-((field_grid.t_fine / duration) ** 2))
    carrier_phase = (-1) ** carrier_index
    # Envelope and field share the temporal period, exact carrier-bin shift and
    # physical band. The real-field opposite-frequency tail is bounded by the
    # Gaussian Fourier tail plus the finite temporal wall, before either run.
    tail = 4 * math.exp(-((duration * carrier / 2) ** 2))
    tail += 4 * math.exp(-((field_grid.t_fine[0] / duration) ** 2))
    evaluations = 7 * steps
    rounding = evaluations * (
        16 * np.finfo(float).eps * math.ceil(math.log2(field_grid.n_to))
        + 2 * gamma(space.n_r)
    )
    rounding += gamma(64) * carrier / C * LENGTH + gamma(16 * field_grid.n_to)
    bound = ladder_bound(
        spatial_backend_class.name, rounding + tail, field_grid.n_to, evaluations
    )
    rows = []
    statistics = []
    for grid, n_pol in (
        (field_grid, 1),
        (envelope_grid, 1),
        (field_grid, 2),
        (envelope_grid, 2),
    ):
        if grid.kind == "field":
            samples = (
                temporal
                * np.cos(2 * np.pi * carrier_index * np.arange(grid.n_to) / grid.n_to)
                * carrier_phase
            )
            spectrum = np.fft.rfft(samples)[grid.crop_index] * grid.crop_scale
        else:
            spectrum = np.fft.fft(temporal)[grid.crop_index] * grid.crop_scale
        start = np.zeros((space.n_r, n_pol, grid.n_omega), complex)
        start[:, 0] = spatial[:, None] * spectrum
        problem = make_problem(
            grid, space, start, [KerrResponse(CHI3, thg=False)], steps=steps
        )
        backend = spatial_backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            state = backend.observe().require_field()[0]
            if n_pol == 2:
                np.testing.assert_array_equal(state[:, 1], np.zeros_like(state[:, 1]))
            indices = np.rint((grid.omega - carrier) / grid.d_omega).astype(int)
            active = grid.active_mask
            ordering = np.argsort(indices[active])
            values = state[:, 0, active][:, ordering] / grid.n_t
            if grid.kind == "field":
                values *= 2 * carrier_phase
            rows.append((indices[active][ordering], values))
            statistics.append(
                backend.stats(
                    StatsSpec(names=("energy", "on_axis_intensity", "peak_intensity"))
                )
            )
        finally:
            backend.close()
    for label, (indices, values) in zip(
        ("envelope scalar", "field vector", "envelope vector"), rows[1:], strict=True
    ):
        np.testing.assert_array_equal(indices, rows[0][0])
        assert_allclose(
            values, rows[0][1], rtol=0, atol=bound * np.max(np.abs(rows[0][1]))
        )
        record_metric(
            request,
            f"field-scalar parity with {label} (relative)",
            steps,
            np.max(np.abs(values - rows[0][1])) / np.max(np.abs(rows[0][1])),
            bound,
            spatial_backend_class.name,
        )
    for result in statistics[1:]:
        for name, reference in statistics[0].items():
            assert_allclose(result[name], reference, rtol=2 * bound, atol=0)


@pytest.mark.parametrize("chunk", [1, 3])
def test_radial_chunked_nonlinear_transform_matches_the_full_rule(
    nonlinear_case, spatial_backend_class, chunk, request
):
    _, grid, space, start, responses, density = nonlinear_case
    problem = make_problem(grid, space, start, responses, density=density)
    chunked = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )
    # Repartitioning changes the reduction association only; its two dense sums
    # and temporal transforms are bounded before either derivative is evaluated.
    bound = ladder_bound(
        spatial_backend_class.name, gamma(64 * space.n_r + 32 * grid.n_to), grid.n_to, 1
    )
    reference = derivative(spatial_backend_class, problem, grid)
    actual = derivative(spatial_backend_class, chunked, grid)
    assert_allclose(actual, reference, rtol=0, atol=bound * np.max(np.abs(reference)))
    record_metric(
        request,
        f"chunked {nonlinear_case[0]} (relative)",
        chunk,
        np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
        bound,
        spatial_backend_class.name,
    )

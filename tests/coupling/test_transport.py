# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test homogeneous transport, observation planes and the sequential ledger."""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, j1
from tests.coupling.conftest import apply_sequence

from solprop import fourier
from solprop.constants import C
from solprop.coupling.elements import (
    Alignment,
    Aperture,
    FreeSpace,
    SpatialState,
    ThinLens,
)
from solprop.exceptions import CouplingError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.optics import Chain, TaylorPhase
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    aperture_profile,
    spatial_energy,
    spatial_power,
)

BeamFactory = Callable[..., SpatialState]


def _radial_field(space: RadialGrid, state: SpatialState, bin_index: int) -> np.ndarray:
    """Return the optics-convention field samples of one bin."""
    return np.conj(space.hankel.inverse(state.spectrum[:, 0, bin_index]))


def _on_axis(state: SpatialState, bin_index: int) -> complex:
    """Return the optics-convention on-axis field of one bin, from the spectrum."""
    column = state.spectrum[:, 0, bin_index]
    space = state.space
    if isinstance(space, RadialGrid):
        return complex(np.conj(np.sum(space.spectral_weights / (2 * np.pi) * column)))
    return complex(np.conj(np.sum(space.on_axis * column)))


def _second_moment_radius(space: RadialGrid, field: np.ndarray) -> float:
    """Return the 1/e^2 radius of a Gaussian from its second intensity moment."""
    intensity = np.abs(field) ** 2
    weights = space.radial_weights
    return math.sqrt(
        2 * np.sum(weights * space.r**2 * intensity) / np.sum(weights * intensity)
    )


def _wavenumber(grid: EnvelopeGrid, bin_index: int) -> float:
    """Return the vacuum wave number of one bin."""
    return float(grid.omega[bin_index] / C)


@pytest.mark.parametrize("fraction", [0.5, 1.0, 2.0])
def test_gaussian_waist_curvature_and_gouy_phase(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium, centre: int, fraction
):
    waist = 100e-6
    space = RadialGrid(1.2e-3, 128)
    state = beam(space, waist)
    k = _wavenumber(grid, centre)
    rayleigh = k * waist**2 / 2
    distance = fraction * rayleigh
    result = FreeSpace(distance, air, frame_delay=0.0).apply(state)
    radius = waist * math.hypot(1.0, distance / rayleigh)
    curvature = distance * (1 + (rayleigh / distance) ** 2)
    gouy = math.atan(distance / rayleigh)
    ratio = _on_axis(result.state, centre) / _on_axis(state, centre)
    field = _radial_field(space, result.state, centre)
    inside = space.r < radius
    phase = np.unwrap(np.angle(field[inside] / field[0]))
    fitted = k / (2 * np.polyfit(space.r[inside] ** 2, phase, 1)[0])
    # The exact angular spectrum departs from the paraxial Gaussian by
    # O(theta^2) ~ (lambda/(pi w0))^2 = 6.5e-6 of the width and curvature, and the
    # on-axis phase by k L theta^4 / 8 < 1e-6 rad; the grid is converged beyond that.
    assert abs(ratio) == pytest.approx(waist / radius, rel=2e-5)
    assert np.angle(ratio * np.exp(-1j * k * distance)) == pytest.approx(
        -gouy, abs=2e-5
    )
    assert _second_moment_radius(space, field) == pytest.approx(radius, rel=2e-5)
    assert fitted == pytest.approx(curvature, rel=2e-5)
    # Free space conserves the physical energy; the ledger closes at rounding.
    assert result.report.specified_loss == 0.0
    assert abs(result.report.approximation_error) <= 64 * np.finfo(float).eps * (
        result.report.energy_in
    )
    assert result.report.diagnostics["wrap_share"] <= 1e-6


def test_direct_fresnel_matches_the_angular_spectrum_and_quadrature(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium, centre: int
):
    """Two independent references for the direct Bessel evaluation of an output."""
    waist = 20e-6
    k = _wavenumber(grid, centre)
    rayleigh = k * waist**2 / 2
    source = RadialGrid(0.4e-3, 256)
    state = beam(source, waist)
    output = RadialGrid(0.2e-3, 128)
    distance = 2 * rayleigh
    angular = FreeSpace(distance, air, output=output).apply(state)
    fresnel = FreeSpace(distance, air, output=output, output_method="fresnel").apply(
        state
    )
    first = angular.state.spectrum[:, 0, centre]
    second = fresnel.state.spectrum[:, 0, centre]
    # The two differ by the paraxial approximation alone: k L theta^4/8 with
    # theta ~ 2 lambda/(pi w0) is ~1e-4 rad at the beam's edge.
    assert np.linalg.norm(first - second) <= 2e-4 * np.linalg.norm(first)
    assert fresnel.report.diagnostics["fresnel_phase_bound"] > 0

    # A soft-edged flat top at Fresnel number ~1, against adaptive quadrature of
    # the same Fresnel integral.
    radius, edge = 100e-6, 5e-6
    source = RadialGrid(0.5e-3, 512)
    flat = beam(source, 0.3e-3, order=10)
    clipped = Aperture(
        radius, model_error=0.05, edge=edge, kind="rcos", weighting="quadrature"
    ).apply(flat)
    distance = radius**2 * k / (2 * math.pi)
    observed = RadialGrid(0.25e-3, 64)
    # The 5 um edge leaves 2e-6 of the field near the source grid's Nyquist rate,
    # where the kernel's chirp would alias it; the comparison below is coarser.
    result = FreeSpace(
        distance,
        air,
        output=observed,
        output_method="fresnel",
        frame_delay=0.0,
        sampling_tolerance=1e-5,
    ).apply(clipped.state)
    field = _radial_field(observed, result.state, centre)
    amplitude = _radial_field(source, flat, centre)[0]
    expected = []
    for rho in observed.r[[0, 10, 30]]:

        def integrand(r, rho=rho, part=np.real):
            profile = math.sqrt(aperture_profile(r, radius, edge, "rcos"))
            value = profile * np.exp(1j * k * r**2 / (2 * distance))
            return float(part(value * j0(k * rho * r / distance) * r))

        upper = radius + edge
        real = quad(integrand, 0, upper, limit=400, epsabs=1e-16)[0]
        imag = quad(integrand, 0, upper, args=(rho, np.imag), limit=400)[0]
        factor = k / (1j * distance) * np.exp(1j * k * distance)
        factor *= np.exp(1j * k * rho**2 / (2 * distance))
        expected.append(amplitude * factor * (real + 1j * imag))
    # The QDHT node rule integrates the 5 um edge on 1 um nodes; the comparison is
    # limited by that quadrature and by the flat top's residual curvature.
    np.testing.assert_allclose(
        field[[0, 10, 30]], expected, rtol=2e-3, atol=2e-3 * abs(expected[0])
    )


def test_finite_distance_converges_to_fraunhofer_at_first_order(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium, centre: int
):
    waist = 20e-6
    k = _wavenumber(grid, centre)
    rayleigh = k * waist**2 / 2
    source = RadialGrid(0.12e-3, 96)
    state = beam(source, waist)
    amplitude = _radial_field(source, state, centre)[0] / math.exp(
        -((source.r[0] / waist) ** 2)
    )
    errors = []
    for ratio in (10.0, 20.0, 40.0):
        distance = ratio * rayleigh
        output = RadialGrid(4 * waist * ratio, 128)
        result = FreeSpace(
            distance, air, output=output, output_method="fresnel", frame_delay=0.0
        ).apply(state)
        field = _radial_field(output, result.state, centre)
        # SEP 31 section 4.6 with the continuous Hankel transform of the Gaussian,
        # (w^2/2) exp(-q^2 w^2/4) at q = k r / L.
        spatial_frequency = k * output.r / distance
        fraunhofer = (
            k
            * np.exp(1j * k * distance)
            / (1j * distance)
            * np.exp(1j * k * output.r**2 / (2 * distance))
            * amplitude
            * waist**2
            / 2
            * np.exp(-((spatial_frequency * waist) ** 2) / 4)
        )
        weights = output.radial_weights
        errors.append(
            math.sqrt(
                np.sum(weights * np.abs(field - fraunhofer) ** 2)
                / np.sum(weights * np.abs(fraunhofer) ** 2)
            )
        )
    # The neglected source phase k r^2/(2L) makes the error z_R/L: first order.
    assert errors[0] < 0.1
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.05)


def test_aperture_far_field_is_the_airy_pattern(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium, centre: int
):
    radius, distance = 100e-6, 0.5
    source = RadialGrid(0.5e-3, 512)
    flat = beam(source, 0.3e-3, order=10)
    output = RadialGrid(8e-3, 256)
    result = apply_sequence(
        flat,
        [
            Aperture(radius, model_error=0.05, edge=2e-6, kind="rcos"),
            # The 2 um edge puts 4e-6 of the field where the kernel aliases it.
            FreeSpace(
                distance,
                air,
                output=output,
                output_method="fresnel",
                frame_delay=0.0,
                sampling_tolerance=1e-5,
            ),
        ],
    )
    k = _wavenumber(grid, centre)
    field = _radial_field(output, result.state, centre)
    peak = abs(_on_axis(result.state, centre)) ** 2
    argument = k * radius * output.r / distance
    airy = (2 * j1(argument) / argument) ** 2
    inside = output.r < 6e-3
    # Fresnel number a^2/(lambda L) = 0.025 and a 2 um soft edge set the departure
    # from the hard-edge Fraunhofer pattern at a few 1e-3 of the peak.
    assert np.max(np.abs(np.abs(field[inside]) ** 2 / peak - airy[inside])) < 1e-2
    first_zero = 3.8317059702 * distance / (k * radius)
    dark = np.argmin(np.where(output.r < 1.5 * first_zero, np.abs(field), np.inf))
    assert output.r[dark] == pytest.approx(first_zero, rel=0.03)


def test_thin_lens_focus_matches_the_fourier_plane_waist(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium, centre: int
):
    waist, focal = 1e-3, 0.2
    space = RadialGrid(4e-3, 512)
    state = beam(space, waist)
    k = _wavenumber(grid, centre)
    expected = 2 * focal / (k * waist)  # lambda f / (pi w0)
    focused = apply_sequence(
        state, [ThinLens(focal, model_error=1e-3), FreeSpace(focal, air)]
    )
    spot = _second_moment_radius(space, _radial_field(space, focused.state, centre))
    # The back focal plane is the paraxial Fourier plane, exactly; the remainder
    # is the angular spectrum's O(theta^2) = 2.5e-5 correction.
    assert spot == pytest.approx(expected, rel=5e-5)
    zoom = RadialGrid(0.3e-3, 128)
    zoomed = apply_sequence(
        state, [ThinLens(focal, model_error=1e-3), FreeSpace(focal, air, output=zoom)]
    )
    zoom_spot = _second_moment_radius(zoom, _radial_field(zoom, zoomed.state, centre))
    assert zoom_spot == pytest.approx(spot, rel=1e-6)
    # Resampling onto a finer, smaller plane keeps the energy to its window.
    assert zoomed.report.energy_out == pytest.approx(zoomed.report.energy_in, rel=1e-5)
    certificate = focused.report.elements[0].certificate
    assert certificate is not None
    assert certificate.ratio <= certificate.model_error
    assert certificate.scope == "single field"


@pytest.mark.parametrize("n_samples", [48, 64])
def test_radial_and_cartesian_agree_for_a_symmetric_beam(
    beam: BeamFactory, air: SpatialMedium, centre: int, n_samples
):
    radial = RadialGrid(1.5e-3, 128)
    cartesian = FreeGrid(1.5e-3, n_samples, 1.5e-3, n_samples)
    elements = [ThinLens(0.3, model_error=1e-3), FreeSpace(0.1, air)]
    first = apply_sequence(beam(radial, 0.3e-3), elements)
    second = apply_sequence(beam(cartesian, 0.3e-3), elements)
    # Both representations are spectrally converged at these sizes (the 32-sample
    # grid still departs by 3e-5).
    assert _on_axis(second.state, centre) == pytest.approx(
        _on_axis(first.state, centre), rel=1e-8
    )
    assert second.report.energy_out == pytest.approx(first.report.energy_out, rel=1e-12)


def test_cartesian_displacement_and_tilt_follow_the_synthesis_map(
    beam: BeamFactory, air: SpatialMedium, centre: int
):
    space = FreeGrid(1.5e-3, 64, 1.5e-3, 64)
    state = beam(space, 0.3e-3)
    shift, angle, distance = (0.1e-3, -0.05e-3), (2e-3, 0.0), 0.1
    result = apply_sequence(
        state,
        [
            Alignment(displacement=shift, tilt=angle, model_error=1e-3),
            FreeSpace(distance, air),
        ],
    )
    samples = np.fft.ifft2(
        result.state.spectrum[:, 0, centre].reshape(space.n_y, space.n_x)
    ).ravel()
    intensity = np.abs(samples) ** 2
    x = np.tile(space.x, space.n_y)
    y = np.repeat(space.y, space.n_x)
    # Positive displacement moves the beam towards +x; a positive tilt sends it
    # towards +x by L tan(theta) after propagation.
    expected = shift[0] + distance * math.tan(angle[0])
    assert np.sum(x * intensity) / np.sum(intensity) == pytest.approx(
        expected, rel=1e-5
    )
    assert np.sum(y * intensity) / np.sum(intensity) == pytest.approx(
        shift[1], rel=1e-9
    )


def test_element_order_matters_and_a_chain_commutes_only_internally(
    beam: BeamFactory, air: SpatialMedium
):
    space = RadialGrid(1.5e-3, 128)
    state = beam(space, 0.3e-3)
    aperture = Aperture(0.25e-3, model_error=0.05, edge=20e-6, kind="rcos")
    # The aperture's diffraction reaches the window over 0.2 m; its removal is
    # declared, and the 1.9e-3 left in the outer band admitted, which keeps both
    # orders on the same small grid.
    gap = FreeSpace(0.2, air, beyond_window="remove", edge_tolerance=1e-2)
    first = apply_sequence(state, [aperture, gap])
    second = apply_sequence(state, [gap, aperture])
    assert not np.allclose(first.state.spectrum, second.state.spectrum, rtol=1e-3)
    gdd = TaylorPhase((0.0, 0.0, 200e-30), 800e-9)
    tod = TaylorPhase((0.0, 0.0, 0.0, 3000e-45), 800e-9)
    forward = apply_sequence(state, [Chain((gdd, tod)), gap])
    backward = apply_sequence(state, [Chain((tod, gdd)), gap])
    np.testing.assert_allclose(
        forward.state.spectrum,
        backward.state.spectrum,
        rtol=0,
        atol=1e-15 * np.max(np.abs(forward.state.spectrum)),
    )
    report = forward.report.elements[0]
    assert report.position == 0
    assert forward.report.elements[1].position == 1
    assert report.element == "SpectralTransfer(Chain)"
    assert len(report.notes) == 2
    assert len(report.calibration) == 2


def test_sequential_ledger_adds_losses_in_joules(beam: BeamFactory, air: SpatialMedium):
    # The clipped beam's edge diffraction must stay inside the window: the lens's
    # certificate counts the field at the window's edge as domain error.
    space = RadialGrid(3e-3, 256)
    state = beam(space, 0.3e-3)
    result = apply_sequence(
        state,
        [
            Aperture(0.3e-3, model_error=0.05, edge=20e-6, kind="rcos"),
            FreeSpace(0.1, air),
            ThinLens(0.5, model_error=1e-3),
            Aperture(0.2e-3, model_error=0.05, edge=20e-6, kind="rcos"),
        ],
    )
    report = result.report
    reports = report.elements
    for before, after in itertools.pairwise(reports):
        assert after.energy_in == before.energy_out
    assert report.specified_loss == pytest.approx(
        sum(item.specified_loss for item in reports), rel=1e-15
    )
    assert abs(report.ledger_residual) <= 1e-14 * report.energy_in
    assert report.energy_out < report.energy_in
    # Every report keeps its per-bin spectra, which sum to its energy.
    for item in reports:
        assert np.sum(item.spectrum_out) == pytest.approx(item.energy_out, rel=1e-12)
    # The losses are not a product of broadband efficiencies.
    product = math.prod(item.energy_out / item.energy_in for item in reports)
    assert report.energy_out / report.energy_in == pytest.approx(product, rel=1e-12)


def test_energy_spectrum_splits_spatial_energy(beam: BeamFactory, air: SpatialMedium):
    space = RadialGrid(1.5e-3, 64)
    state = beam(space, 0.3e-3)
    report = FreeSpace(0.0, air).apply(state).report
    flux = SpatialFlux.resolve(state.grid, space, air)
    assert np.sum(report.spectrum_in) == pytest.approx(
        spatial_energy(state.spectrum, flux), rel=1e-13
    )
    assert report.energy_in == pytest.approx(1e-6, rel=1e-12)


def test_default_frame_delay_keeps_a_field_grid_pulse_in_its_window():
    grid = FieldGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 300e-15)
    argon = SpatialMedium("Ar", 1.0)
    space = RadialGrid(2e-3, 32)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    state = SpatialState(
        grid, space, SpatialPulse(pulse, 0.5e-3).build(grid, space, medium=argon), argon
    )
    result = FreeSpace(1.0, argon).apply(state)

    def arrival(plane: SpatialState) -> float:
        flux = SpatialFlux.resolve(grid, space, argon)
        power = spatial_power(plane.spectrum, flux)
        return float(np.sum(grid.t * power) / np.sum(power))

    # beta1 L = 3.3 ns for a metre of argon, ten thousand windows; removing it
    # leaves only the dispersive and diffractive reshaping, well under a femtosecond.
    assert result.report.diagnostics["delay_removed"] > 3e-9
    assert abs(arrival(result.state) - arrival(state)) <= 1e-15
    assert result.report.energy_out == pytest.approx(result.report.energy_in, rel=1e-12)


def test_backward_propagation_round_trips_and_is_flagged(
    beam: BeamFactory, air: SpatialMedium
):
    space = RadialGrid(1.5e-3, 128)
    state = beam(space, 0.3e-3)
    there = FreeSpace(0.2, air).apply(state)
    back = FreeSpace(-0.2, air).apply(there.state)
    assert back.report.reconstruction
    assert not there.report.reconstruction
    np.testing.assert_allclose(
        back.state.spectrum,
        state.spectrum,
        rtol=0,
        atol=1e-12 * np.max(np.abs(state.spectrum)),
    )


def _narrow_state(grid: EnvelopeGrid, air: SpatialMedium) -> SpatialState:
    """Return a beam narrower than a wavelength, built without a launch check."""
    space = RadialGrid(20e-6, 64)
    profile = np.exp(-((space.r / 0.3e-6) ** 2))
    envelope = np.exp(-((grid.t / 30e-15) ** 2)) + 0j
    temporal = fourier.forward(envelope, kind=grid.kind)
    temporal = temporal * (grid.active_mask & (grid.omega > 0))
    spectrum = space.hankel.forward(profile)[:, None, None] * temporal[None, None, :]
    return SpatialState(grid, space, spectrum, air)


def test_evanescent_content_is_refused_or_removed_explicitly(
    grid: EnvelopeGrid, air: SpatialMedium
):
    state = _narrow_state(grid, air)
    with pytest.raises(CouplingError, match="nonpropagating"):
        FreeSpace(1e-6, air).apply(state)
    # The sub-wavelength beam rings across this 64-node window, 1.4e-3 of it in the
    # outer band before any propagation; only the band removal is under test.
    limited = FreeSpace(1e-6, air, band_limit=0.5, edge_tolerance=1e-2).apply(state)
    report = limited.report
    assert report.removed > 0
    assert abs(report.approximation_error) <= 1e-12 * report.energy_in
    assert report.energy_out == pytest.approx(
        report.energy_in - report.removed, rel=1e-12
    )


def test_a_same_grid_field_in_the_outer_band_is_refused_by_its_own_tolerance(
    beam: BeamFactory, air: SpatialMedium
):
    space = RadialGrid(0.5e-3, 64)
    # A 0.4 mm beam on a 0.5 mm window fills its collar before it moves: no ray
    # leaves over a micrometre, so only the edge share can see it.
    filling = beam(space, 0.4e-3)
    with pytest.raises(CouplingError, match="outer band") as refusal:
        FreeSpace(1e-6, air).apply(filling)
    message = str(refusal.value)
    assert "enlarge the window" in message
    assert "beyond_window='remove'" in message
    report = FreeSpace(1e-6, air, edge_tolerance=1.0).apply(filling).report
    assert report.diagnostics["edge_share"] > 1e-4
    assert report.diagnostics["wrap_share"] <= 1e-6
    # A beam well inside the window passes the default.
    inside = FreeSpace(1e-2, air).apply(beam(space, 0.1e-3)).report
    assert inside.diagnostics["edge_share"] <= 1e-4


def test_output_plane_refusals(beam: BeamFactory, air: SpatialMedium):
    space = RadialGrid(0.5e-3, 64)
    state = beam(space, 0.1e-3)
    wide = RadialGrid(5e-3, 64)
    # The beam spreads past the source window: the angular output cannot see it,
    # and the refusal points to the Fresnel route.
    with pytest.raises(CouplingError, match='output_method="fresnel"'):
        FreeSpace(1.0, air, output=wide).apply(state)
    # A short Fresnel distance undersamples the kernel on the source nodes.
    # A beam filling the window's outer band cannot be read off it by the angular
    # sum even without any spreading; the refusal points to the Fresnel route.
    filling = beam(space, 0.4e-3)
    with pytest.raises(CouplingError, match=r'edge.*output_method="fresnel"'):
        FreeSpace(1e-6, air, output=wide).apply(filling)
    with pytest.raises(CouplingError, match="Fresnel kernel"):
        FreeSpace(1e-3, air, output=wide, output_method="fresnel").apply(state)
    with pytest.raises(CouplingError, match="positive distance"):
        FreeSpace(-1.0, air, output=wide, output_method="fresnel").apply(state)
    with pytest.raises(CouplingError, match="same kind"):
        FreeSpace(0.1, air, output=FreeGrid(1e-3, 8, 1e-3, 8)).apply(state)
    with pytest.raises(CouplingError, match="PlanarInterface"):
        FreeSpace(0.1, SpatialMedium(1.5)).apply(state)
    with pytest.raises(CouplingError, match="band_limit"):
        FreeSpace(0.1, air, band_limit=2.0).apply(state)


def test_an_empty_sequence_and_a_zero_field_are_exact(
    grid: EnvelopeGrid, air: SpatialMedium
):
    space = RadialGrid(1e-3, 32)
    zero = SpatialState(grid, space, np.zeros((32, 1, grid.n_omega), complex), air)
    identity = apply_sequence(zero, [])
    assert identity.report.energy_in == identity.report.energy_out == 0.0
    result = apply_sequence(
        zero, [ThinLens(0.5, model_error=1e-3), FreeSpace(0.1, air)]
    )
    assert not np.any(result.state.spectrum)
    certificate = result.report.elements[0].certificate
    assert certificate is not None
    assert certificate.exact_zero
    with pytest.raises(CouplingError, match="neither an element"):
        apply_sequence(zero, [object()])


def test_cartesian_output_planes_agree_with_radial(
    beam: BeamFactory, air: SpatialMedium, centre: int
):
    """The Cartesian direct summations against the radial ones, for one beam."""
    lens = ThinLens(0.5, model_error=1e-3)
    radial = beam(RadialGrid(1.5e-3, 128), 0.3e-3)
    cartesian = beam(FreeGrid(1.5e-3, 64, 1.5e-3, 64), 0.3e-3)
    zoom_r = RadialGrid(0.9e-3, 96)
    zoom_c = FreeGrid(0.9e-3, 48, 0.9e-3, 48)
    angular_r = apply_sequence(radial, [lens, FreeSpace(0.1, air, output=zoom_r)])
    angular_c = apply_sequence(cartesian, [lens, FreeSpace(0.1, air, output=zoom_c)])
    # Both are converged trigonometric/Fourier-Bessel interpolants of converged
    # propagations; the on-axis values agree to their transforms' accuracy.
    assert _on_axis(angular_c.state, centre) == pytest.approx(
        _on_axis(angular_r.state, centre), rel=1e-8
    )
    far_r = RadialGrid(2.5e-3, 128)
    far_c = FreeGrid(2.5e-3, 48, 2.5e-3, 48)
    # A finer Cartesian source keeps the wider kernel's chirp below its Nyquist rate.
    cartesian = beam(FreeGrid(1.5e-3, 96, 1.5e-3, 96), 0.3e-3)
    fresnel_r = apply_sequence(
        radial, [lens, FreeSpace(1.0, air, output=far_r, output_method="fresnel")]
    )
    fresnel_c = apply_sequence(
        cartesian, [lens, FreeSpace(1.0, air, output=far_c, output_method="fresnel")]
    )
    # The radial and Cartesian node rules integrate the same smooth integrand;
    # each output window holds the beam to 1e-7 of its energy.
    assert _on_axis(fresnel_c.state, centre) == pytest.approx(
        _on_axis(fresnel_r.state, centre), rel=1e-6
    )
    assert fresnel_c.report.energy_out == pytest.approx(
        fresnel_r.report.energy_out, rel=1e-6
    )
    # The exact propagation on a window wide enough to hold the beam agrees with
    # both to the paraxial bound the report states.
    exact = apply_sequence(
        beam(RadialGrid(3e-3, 256), 0.3e-3), [lens, FreeSpace(1.0, air)]
    )
    bound = fresnel_r.report.elements[1].diagnostics["fresnel_phase_bound"]
    assert _on_axis(fresnel_r.state, centre) == pytest.approx(
        _on_axis(exact.state, centre), rel=max(bound, 1e-6)
    )


def test_windows_without_a_collar_use_their_outer_tenth(
    beam: BeamFactory, air: SpatialMedium
):
    for space in (
        RadialGrid(0.6e-3, 64, window_factor=0.0),
        FreeGrid(0.6e-3, 32, 0.6e-3, 32, window_factor=0.0),
    ):
        state = beam(space, 0.15e-3)
        # Half a metre spreads the 150 um beam well past this window: the wrap is
        # refused unless its removal is declared, and then it is reported.
        with pytest.raises(CouplingError, match="beyond_window='remove'"):
            FreeSpace(0.5, air).apply(state)
        # What stays still fills the outer tenth, which is refused by its own
        # tolerance, and the refusal no longer offers the removal already made.
        removing = FreeSpace(0.5, air, beyond_window="remove")
        with pytest.raises(
            CouplingError, match=r"outer band.*window, or ev"
        ) as refusal:
            removing.apply(state)
        assert "beyond_window" not in str(refusal.value)
        admitted = FreeSpace(0.5, air, beyond_window="remove", edge_tolerance=1.0)
        report = admitted.apply(state).report
        assert report.diagnostics["edge_share"] > 1e-3
        assert report.diagnostics["wrap_share"] > 0
        assert report.removed == report.diagnostics["window_removed"] > 0
        assert report.numerical_error == 0.0
        assert abs(report.approximation_error) <= 64 * np.finfo(float).eps * (
            report.energy_in
        )


def test_an_absorbing_gap_reports_its_loss(beam: BeamFactory):
    lossy = SpatialMedium(1.0 + 1e-7j)
    space = RadialGrid(1.5e-3, 64)
    state = beam(space, 0.3e-3, medium=lossy)
    report = FreeSpace(0.1, lossy).apply(state).report
    # Im(n) = 1e-7 attenuates the power by exp(-2 k Im(n) L) ~ 0.86 over 0.1 m.
    assert 0.1 < report.specified_loss / report.energy_in < 0.2
    assert abs(report.approximation_error) <= 1e-12 * report.energy_in
    explicit = FreeSpace(0.1, lossy, evanescent_tolerance=0.0).apply(state).report
    assert explicit.energy_out == report.energy_out


@pytest.mark.parametrize(
    "space",
    [
        RadialGrid(1e-3, 48),
        FreeGrid(1e-3, 24, 1e-3, 24),
        FreeGrid(1e-3, 25, 1e-3, 23),
    ],
)
def test_resampling_onto_the_source_nodes_is_the_identity(
    beam: BeamFactory, air: SpatialMedium, space
):
    state = beam(space, 0.2e-3)
    result = FreeSpace(0.0, air, output=space).apply(state)
    # Direct summation at the source's own nodes is its inverse transform, so the
    # round trip is the transforms' rounding (the raw QDHT is near-self-inverse).
    np.testing.assert_allclose(
        result.state.spectrum,
        state.spectrum,
        rtol=0,
        atol=1e-12 * np.max(np.abs(state.spectrum)),
    )


def test_a_zero_field_reaches_a_fresnel_plane(grid: EnvelopeGrid, air: SpatialMedium):
    space = RadialGrid(1e-3, 32)
    zero = SpatialState(grid, space, np.zeros((32, 1, grid.n_omega), complex), air)
    result = FreeSpace(
        1.0, air, output=RadialGrid(2e-3, 16), output_method="fresnel"
    ).apply(zero)
    assert not np.any(result.state.spectrum)
    assert result.report.diagnostics["kernel_alias_share"] == 0.0

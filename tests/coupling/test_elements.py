# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test reflection frames, interfaces, spectral optics, thin screens and refusals."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from solprop.constants import C
from solprop.coupling.elements import (
    Alignment,
    Aperture,
    FreeSpace,
    Mirror,
    PlanarInterface,
    PolarisationTransform,
    SpatialState,
    SpectralCalibration,
    SpectralTransfer,
    ThinLens,
    TransmissionMask,
)
from solprop.coupling.transport import transport
from solprop.exceptions import (
    ConvergenceError,
    GridError,
    MaterialError,
    OpticsGeometryError,
    PolarisationError,
)
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.optics import (
    Chain,
    ChirpedMirror,
    GratingPair,
    Material,
    TaylorPhase,
    littrow_angle,
)
from solprop.profiles import Linear
from solprop.spatial import SpatialMedium

BeamFactory = Callable[..., SpatialState]
EPS = np.finfo(float).eps


def _lab_field(state: SpatialState, column: int, bin_index: int) -> np.ndarray:
    """Return a column's optics-convention field vector in laboratory axes."""
    stored = state.spectrum[column, :, bin_index]
    jones = np.conj(stored)
    if jones.size == 1:
        jones = np.array([jones[0], 0.0])
    return jones[0] * state.frame[0] + jones[1] * state.frame[1]


def _reflection(incident: np.ndarray, outgoing: np.ndarray) -> np.ndarray:
    """Return the improper reflection taking one direction into the other."""
    normal = outgoing - incident
    normal = normal / np.linalg.norm(normal)
    return np.eye(3) - 2 * np.outer(normal, normal)


CONDUCTOR = {"reflectivity_p": 1.0 + 0j, "reflectivity_s": -1.0 + 0j}


@pytest.mark.parametrize("s_sign", [1.0, -1.0])
def test_perfect_mirror_reverses_the_tangential_field_at_normal_incidence(
    beam: BeamFactory, centre: int, s_sign
):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3, jones=(1.0, 0.4 + 0.3j))
    mirror = Mirror(
        **CONDUCTOR, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(0, s_sign, 0)
    )
    result = mirror.apply(state)
    # A perfect conductor cancels the tangential field: E_out = -E_in in the lab.
    np.testing.assert_allclose(
        _lab_field(result.state, 3, centre),
        -_lab_field(state, 3, centre),
        rtol=0,
        atol=8 * EPS * np.max(np.abs(state.spectrum)),
    )
    np.testing.assert_allclose(result.state.frame[2], [0, 0, -1])
    assert np.linalg.det(result.state.frame) == pytest.approx(1.0)
    back = Mirror(
        **CONDUCTOR, incident=(0, 0, -1), outgoing=(0, 0, 1), s_axis=(0, 1, 0)
    )
    twice = back.apply(result.state)
    np.testing.assert_allclose(twice.state.frame, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        twice.state.spectrum,
        state.spectrum,
        rtol=0,
        atol=8 * EPS * np.max(np.abs(state.spectrum)),
    )
    assert result.report.specified_loss == pytest.approx(0.0, abs=1e-15 * 1e-6)


def test_dielectric_mirror_reflects_isotropically_at_normal_incidence(
    beam: BeamFactory, centre: int
):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3, jones=(0.6, 0.8j))
    r_s = (1 - 1.5) / (1 + 1.5)
    # With p = s x k on both beams, normal incidence has r_p = -r_s.
    mirror = Mirror(
        -r_s, r_s, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(1, 0, 0)
    )
    result = mirror.apply(state)
    np.testing.assert_allclose(
        _lab_field(result.state, 0, centre),
        r_s * _lab_field(state, 0, centre),
        rtol=1e-14,
    )
    assert result.report.specified_loss == pytest.approx(
        (1 - r_s**2) * result.report.energy_in, rel=1e-12
    )


def test_fold_mirrors_follow_the_physical_reflection(beam: BeamFactory, centre: int):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3, jones=(0.6, 0.8j))
    z, x, y = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
    first = Mirror(**CONDUCTOR, incident=z, outgoing=x, s_axis=y).apply(state)
    # A perfect conductor's reflected field is -M E for the reflection M.
    np.testing.assert_allclose(
        _lab_field(first.state, 0, centre),
        -_reflection(z, x) @ _lab_field(state, 0, centre),
        atol=1e-12 * np.max(np.abs(state.spectrum)),
    )
    np.testing.assert_allclose(first.state.frame[2], x, atol=1e-15)
    second = Mirror(**CONDUCTOR, incident=x, outgoing=z, s_axis=y).apply(first.state)
    np.testing.assert_allclose(second.state.frame, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        _lab_field(second.state, 0, centre),
        _reflection(x, z) @ _reflection(z, x) @ _lab_field(state, 0, centre),
        atol=1e-12 * np.max(np.abs(state.spectrum)),
    )


def test_mirror_images_a_displaced_cartesian_beam(beam: BeamFactory, centre: int):
    space = FreeGrid(1.5e-3, 32, 1.5e-3, 32)
    shift = 0.2e-3
    state = Alignment(displacement=(shift, 0.0)).apply(beam(space, 0.3e-3)).state
    result = Mirror(
        **CONDUCTOR, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(0, 1, 0)
    ).apply(state)
    samples = np.fft.ifft2(
        result.state.spectrum[:, 0, centre].reshape(space.n_y, space.n_x)
    ).ravel()
    intensity = np.abs(samples) ** 2
    local = np.sum(np.tile(space.x, space.n_y) * intensity) / np.sum(intensity)
    # The outgoing x axis is the lab -x, so the beam sits at -d locally and at the
    # same lab position.
    assert local == pytest.approx(-shift, rel=1e-9)
    np.testing.assert_allclose(result.state.frame[0], [-1, 0, 0], atol=1e-15)


def test_mirror_and_jones_refusals(beam: BeamFactory):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3)
    with pytest.raises(OpticsGeometryError, match="perpendicular"):
        Mirror(**CONDUCTOR, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(0, 1, 1))
    with pytest.raises(OpticsGeometryError, match="three-vector"):
        Mirror(**CONDUCTOR, incident=(0, 0, 0), outgoing=(0, 0, -1), s_axis=(0, 1, 0))
    with pytest.raises(OpticsGeometryError, match="incident direction"):
        Mirror(
            **CONDUCTOR, incident=(1, 0, 0), outgoing=(0, 0, 1), s_axis=(0, 1, 0)
        ).apply(state)
    with pytest.raises(OpticsGeometryError, match="changes the propagation"):
        Mirror(
            **CONDUCTOR, incident=(0, 0, 1), outgoing=(0, 0, 1), s_axis=(0, 1, 0)
        ).apply(state)
    with pytest.raises(PolarisationError, match="singular values"):
        Mirror(
            1.2, 1.0, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(0, 1, 0)
        ).apply(state)
    with pytest.raises(PolarisationError, match="singular values"):
        PolarisationTransform(np.array([[1.0, 1.0], [0.0, 1.0]])).apply(state)
    with pytest.raises(PolarisationError, match="2x2"):
        PolarisationTransform(np.eye(3)).apply(state)
    cartesian = FreeGrid(1e-3, 16, 1e-3, 16)
    with pytest.raises(GridError, match="oblique"):
        Mirror(
            **CONDUCTOR,
            incident=(0, 0, 1),
            outgoing=(0, 0, -1),
            s_axis=(1, 1, 0),
        ).apply(beam(cartesian, 0.3e-3))
    odd = FreeGrid(1e-3, 15, 1e-3, 16)
    with pytest.raises(GridError, match="odd sample count"):
        Mirror(
            **CONDUCTOR, incident=(0, 0, 1), outgoing=(0, 0, -1), s_axis=(0, 1, 0)
        ).apply(beam(odd, 0.3e-3))


def test_polarisation_transform_promotes_a_scalar_state_and_keeps_energy(
    beam: BeamFactory, centre: int
):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3)
    quarter = np.array([[1.0, 0.0], [0.0, 1j]])
    kept = PolarisationTransform(quarter).apply(state)
    assert kept.state.n_pol == 1
    rotation = np.array(
        [[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]]
    )
    rotated = PolarisationTransform(rotation).apply(state)
    assert rotated.state.n_pol == 2
    field = _lab_field(rotated.state, 0, centre)
    original = _lab_field(state, 0, centre)
    np.testing.assert_allclose(field[:2], rotation @ original[:2], rtol=1e-14)
    report = rotated.report
    assert abs(report.specified_loss) <= 1e-14 * report.energy_in
    assert abs(report.approximation_error) <= 1e-14 * report.energy_in
    polariser = PolarisationTransform(np.diag([math.sqrt(0.3), 0.0])).apply(state)
    assert polariser.report.energy_out == pytest.approx(
        0.3 * polariser.report.energy_in, rel=1e-12
    )


def test_fresnel_interface_conserves_energy_and_tracks_the_medium(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium
):
    glass = SpatialMedium(1.5)
    space = RadialGrid(4e-3, 64)
    state = beam(space, 1e-3)
    interface = PlanarInterface(air, glass)
    result = interface.apply(state)
    report = result.report
    reflectance, transmittance = 0.04, 1.5 * (2 / 2.5) ** 2
    assert reflectance + transmittance == pytest.approx(1.0, abs=4 * EPS)
    assert report.diagnostics["fresnel_defect"] <= 4 * EPS
    assert report.specified_loss == pytest.approx(
        reflectance * report.energy_in, rel=1e-12
    )
    # A 1 mm beam's obliquity theta^2 ~ 6e-8 is all that departs from R + T = 1.
    assert report.energy_out == pytest.approx(
        transmittance * report.energy_in, rel=1e-7
    )
    assert abs(report.approximation_error) <= 1e-7 * report.energy_in
    assert result.state.medium is glass
    with pytest.raises(MaterialError, match="applied once"):
        interface.apply(result.state)
    with pytest.raises(MaterialError, match="PlanarInterface"):
        FreeSpace(0.1, air).apply(result.state)
    FreeSpace(0.1, glass).apply(result.state)
    with pytest.raises(MaterialError, match="lossless"):
        PlanarInterface(air, SpatialMedium(1.5 + 0.01j)).apply(state)
    assert grid.n_omega > 0


def test_spectral_transfer_attenuates_by_its_squared_modulus(
    beam: BeamFactory, grid: EnvelopeGrid
):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3)
    lossy = Material("SiO2", 0.02, 800e-9, loss=True)
    result = SpectralTransfer(lossy).apply(state)
    band = grid.active_mask & (grid.omega > 0)
    factor = np.abs(lossy(grid.omega[band])) ** 2
    report = result.report
    np.testing.assert_allclose(
        report.spectrum_out[band], factor * report.spectrum_in[band], rtol=1e-12
    )
    assert report.specified_loss == pytest.approx(
        report.energy_in - report.energy_out, rel=1e-12
    )
    assert report.calibration[0].removes_delay is True
    assert report.calibration[0].removes_phase is False


def test_phase_only_transfer_is_invariant_and_efficiency_is_per_frequency(
    beam: BeamFactory, grid: EnvelopeGrid
):
    space = RadialGrid(1e-3, 32)
    gdd = TaylorPhase((0.0, 0.0, 500e-30), 800e-9)
    narrow = beam(space, 0.3e-3, fwhm=60e-15)
    broad = beam(space, 0.3e-3, fwhm=15e-15, wavelength=790e-9)
    invariant = SpectralTransfer(gdd).apply(narrow).report
    assert invariant.energy_out == pytest.approx(invariant.energy_in, rel=1e-14)
    assert invariant.specified_loss == pytest.approx(0.0, abs=1e-14 * 1e-6)

    def efficiency(omega: np.ndarray) -> np.ndarray:
        return 0.5 + 0.4 * np.tanh((omega - 2.355e15) / 5e13)

    band = grid.active_mask & (grid.omega > 0)
    ratios = []
    for plane in (narrow, broad):
        report = (
            SpectralTransfer(gdd, efficiency=efficiency, predict_loss=True)
            .apply(plane)
            .report
        )
        np.testing.assert_allclose(
            report.spectrum_out[band],
            efficiency(grid.omega[band]) * report.spectrum_in[band],
            rtol=1e-12,
        )
        ratios.append(report.energy_out / report.energy_in)
    # The broadband efficiency depends on the spectrum it is applied to.
    assert abs(ratios[0] - ratios[1]) > 1e-2


def test_spectral_transfer_refusals_and_records(beam: BeamFactory, grid: EnvelopeGrid):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3)
    gdd = TaylorPhase((0.0, 0.0, 500e-30), 800e-9)
    with pytest.raises(MaterialError, match="explicit efficiency"):
        SpectralTransfer(gdd, predict_loss=True).apply(state)
    with pytest.raises(MaterialError, match="twice"):
        SpectralTransfer(Material("SiO2", 1e-3, loss=True), efficiency=0.9).apply(state)
    with pytest.raises(MaterialError, match=r"\[0, 1\]"):
        SpectralTransfer(gdd, efficiency=1.2).apply(state)
    with pytest.raises(MaterialError, match="calibration"):
        transport(state, [lambda omega: np.ones_like(omega, dtype=complex)])
    record = SpectralCalibration(True, True, 800e-9, "measured compressor")
    custom = SpectralTransfer(
        lambda omega: np.exp(1j * 1e-30 * (omega - 2.35e15) ** 2), calibration=record
    ).apply(state)
    assert custom.report.calibration == (record,)
    inverse = SpectralTransfer(ChirpedMirror.lookup("PC70").bounces(-2)).apply(state)
    assert inverse.report.reconstruction
    assert inverse.report.calibration[0].removes_phase is True
    negative = SpectralTransfer(Material("SiO2", -1e-3, 800e-9)).apply(state)
    assert negative.report.reconstruction
    chained = SpectralTransfer(Chain((gdd, Material("SiO2", -1e-3)))).apply(state)
    assert chained.report.reconstruction
    assert chained.report.calibration[1].removes_delay is False
    grating = GratingPair(
        600e3,
        0.01,
        littrow_angle(600e3, 800e-9),
        800e-9,
        band=(790e-9, 810e-9),
    )
    selected = SpectralTransfer(grating).apply(state).report
    band = grid.active_mask & (grid.omega > 0)
    wavelength = 2 * math.pi * C / grid.omega[band]
    outside = (wavelength < 790e-9) | (wavelength > 810e-9)
    assert selected.removed == pytest.approx(
        float(np.sum(selected.spectrum_in[band][outside])), rel=1e-12
    )
    assert selected.calibration[0].removes_phase is True
    assert abs(selected.approximation_error) <= 1e-14 * selected.energy_in


def test_thin_screens_stay_inside_their_certificates(beam: BeamFactory):
    space = RadialGrid(4e-3, 256)
    state = beam(space, 1e-3)
    lens = ThinLens(0.5, model_error=1e-3).apply(state)
    certificate = lens.report.certificate
    assert certificate is not None
    # A phase screen keeps U_0; its axial-energy change stays inside the defects.
    assert lens.report.specified_loss == 0.0
    assert abs(lens.report.approximation_error) <= (
        certificate.defect_in + certificate.defect_out
    )
    aperture = Aperture(1e-3, model_error=1e-2, edge=0.1e-3, kind="tanh").apply(state)
    certificate = aperture.report.certificate
    assert certificate is not None
    assert aperture.report.specified_loss > 0
    # L0 minus the axial decrement is bounded by the same defects (SEP 31 4.6).
    assert abs(aperture.report.approximation_error) <= (
        certificate.defect_in + certificate.defect_out
    )
    chromatic = ThinLens(lambda omega: 0.5 * (omega / 2.35e15) ** 2, model_error=1e-3)
    assert chromatic.apply(state).report.element == "ThinLens(chromatic f)"


def test_thin_screen_refusals(beam: BeamFactory, grid: EnvelopeGrid):
    space = RadialGrid(4e-3, 256)
    state = beam(space, 1e-3)
    with pytest.raises(ConvergenceError, match="model_error"):
        ThinLens(0.5, model_error=1e-7).apply(state)
    with pytest.raises(ConvergenceError, match=r"\(0, 1\)"):
        ThinLens(0.5, model_error=0.0).apply(state)
    with pytest.raises(GridError, match="Nyquist"):
        ThinLens(5e-3, model_error=0.5).apply(state)
    with pytest.raises(MaterialError, match="nonzero focal"):
        ThinLens(0.0, model_error=1e-3).apply(state)
    with pytest.raises(MaterialError, match="passive"):
        TransmissionMask(np.full(space.n_r, 1.1), model_error=1e-3).apply(state)
    with pytest.raises(MaterialError, match="unit modulus"):
        TransmissionMask(
            np.full(space.n_r, 0.5), model_error=1e-3, phase_only=True
        ).apply(state)
    with pytest.raises(MaterialError, match="n_nodes"):
        TransmissionMask(np.ones(3), model_error=1e-3).apply(state)
    with pytest.raises(MaterialError, match="broadcast"):
        TransmissionMask(lambda x, y, omega: np.ones(3), model_error=1e-3).apply(
            state.with_spectrum(state.spectrum, space=space)
        )


def test_radial_grids_refuse_symmetry_breaking(beam: BeamFactory):
    space = RadialGrid(4e-3, 64)
    state = beam(space, 1e-3)
    with pytest.raises(GridError, match="azimuth"):
        TransmissionMask(
            lambda x, y, omega: np.exp(-((x / 2e-3) ** 2))[:, None] + 0 * omega,
            model_error=1e-2,
        ).apply(state)
    symmetric = TransmissionMask(
        lambda x, y, omega: np.exp(-((x**2 + y**2) / 4e-6))[:, None] + 0 * omega,
        model_error=1e-2,
    ).apply(state)
    assert symmetric.report.specified_loss > 0
    with pytest.raises(GridError, match="off-axis"):
        Aperture(1e-3, model_error=1e-2, centre=(1e-4, 0.0)).apply(state)
    with pytest.raises(GridError, match="cylindrical symmetry"):
        Alignment(displacement=(1e-4, 0.0)).apply(state)
    with pytest.raises(GridError, match="cylindrical symmetry"):
        Alignment(tilt=(1e-3, 0.0), model_error=1e-3).apply(state)
    assert Alignment().apply(state).report.element == "Alignment(none)"


def test_cartesian_masks_and_alignment_refusals(beam: BeamFactory):
    space = FreeGrid(1.5e-3, 32, 1.5e-3, 32)
    state = beam(space, 0.3e-3)
    offset = Aperture(
        0.3e-3, model_error=0.05, edge=50e-6, kind="rcos", centre=(0.1e-3, 0.0)
    ).apply(state)
    assert offset.report.specified_loss > 0
    phase = TransmissionMask(
        lambda x, y, omega: np.exp(1j * 1e3 * x)[:, None] + 0 * omega,
        model_error=1e-3,
        phase_only=True,
    ).apply(state)
    assert phase.report.specified_loss == 0.0
    with pytest.raises(ConvergenceError, match="tilt"):
        Alignment(tilt=(1e-3, 0.0)).apply(state)
    with pytest.raises(GridError, match="finite"):
        Alignment(tilt=(2.0, 0.0), model_error=1e-3).apply(state)


def test_state_validation(grid: EnvelopeGrid, air: SpatialMedium, beam: BeamFactory):
    space = RadialGrid(1e-3, 16)
    zero = np.zeros((16, 1, grid.n_omega), complex)
    with pytest.raises(GridError, match="line grid"):
        SpatialState(grid, LineGrid(1e-3, 16), zero, air)
    with pytest.raises(GridError, match="shape"):
        SpatialState(grid, space, np.zeros((15, 1, grid.n_omega), complex), air)
    outside = zero.copy()
    outside[0, 0, np.flatnonzero(~grid.active_mask)[0]] = 1.0
    with pytest.raises(GridError, match="outside the active band"):
        SpatialState(grid, space, outside, air)
    taper = SpatialMedium("Ar", Linear(1.0, 2.0, 0.1))
    with pytest.raises(MaterialError, match="homogeneous"):
        SpatialState(grid, space, zero, taper)
    with pytest.raises(GridError, match="right-handed"):
        SpatialState(grid, space, zero, air, np.diag([1.0, 1.0, -1.0]))
    state = beam(space, 0.3e-3)
    assert not state.spectrum.flags.writeable
    assert not state.frame.flags.writeable
    with pytest.raises(GridError, match="neither"):
        transport(state, [3])


def test_nonfinite_and_out_of_range_inputs_are_refused(
    beam: BeamFactory, grid: EnvelopeGrid, air: SpatialMedium
):
    space = RadialGrid(1e-3, 32)
    state = beam(space, 0.3e-3)
    with pytest.raises(MaterialError, match="finite on every band"):
        ThinLens(lambda omega: np.full(omega.shape, np.nan), model_error=1e-3).apply(
            state
        )
    with pytest.raises(MaterialError, match="transmission must be finite"):
        TransmissionMask(np.full(space.n_r, np.nan), model_error=1e-3).apply(state)
    with pytest.raises(PolarisationError, match="finite"):
        PolarisationTransform(np.full((2, 2), np.inf)).apply(state)
    record = SpectralCalibration(None, None, None)
    with pytest.raises(MaterialError, match=r"\|H\| <= 1"):
        SpectralTransfer(
            lambda omega: np.full(omega.shape, 2.0 + 0j), calibration=(record,)
        ).apply(state)
    unknown_member = Chain((lambda omega: np.ones(omega.shape, complex),))
    declared = SpectralTransfer(unknown_member, calibration=record).apply(state)
    assert declared.report.energy_out == pytest.approx(
        declared.report.energy_in, rel=1e-14
    )
    with pytest.raises(MaterialError, match="known optics element"):
        SpectralTransfer(Chain((lambda omega: np.ones(omega.shape, complex),))).apply(
            state
        )
    with pytest.raises(GridError, match="finite"):
        FreeSpace(math.inf, air).apply(state)
    unknown = FreeSpace(0.1, air)
    object.__setattr__(unknown, "output_method", "spherical")
    with pytest.raises(GridError, match="output_method"):
        unknown.apply(state)
    samples = np.ones((space.n_r, int(np.sum(grid.active_mask & (grid.omega > 0)))))
    sampled = TransmissionMask(0.5 * samples, model_error=1e-3).apply(state)
    assert sampled.report.energy_out == pytest.approx(
        0.25 * sampled.report.energy_in, rel=1e-10
    )


def test_a_field_with_no_forward_flux_cannot_be_certified(
    grid: EnvelopeGrid, air: SpatialMedium
):
    space = RadialGrid(10e-6, 64)
    band = grid.active_mask & (grid.omega > 0)
    spectrum = np.zeros((space.n_r, 1, grid.n_omega), complex)
    # Every column beyond the largest band wave number is nonpropagating.
    beyond = space.k_r > 1.1 * np.max(grid.omega[band]) / C
    spectrum[beyond, 0, :] = band
    state = SpatialState(grid, space, spectrum, air)
    with pytest.raises(GridError, match="no forward flux"):
        ThinLens(1.0, model_error=1e-3).apply(state)

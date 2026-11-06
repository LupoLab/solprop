# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A free-space beam's displacement and tilt (SEP 41, S6)."""

import math

import numpy as np
import pytest

from solprop.exceptions import PulseError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy

ENERGY = 3e-9
WAIST = 60e-6


def _grid():
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 300e-15, thg=False)


def _space(**changes):
    return FreeGrid(
        **{"radius_x": 4e-4, "n_x": 64, "radius_y": 4e-4, "n_y": 48} | changes
    )


def _beam(**changes):
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY)
    return SpatialPulse(temporal, WAIST, **changes)


def _cube(spectrum, space):
    return spectrum.reshape(space.n_y, space.n_x, *spectrum.shape[1:])


def test_a_displacement_is_alignment_s_exact_shift():
    grid, space, medium = _grid(), _space(), SpatialMedium()
    centred = _beam().build(grid, space, medium=medium)
    moved = _beam(displacement=(50e-6, -30e-6)).build(grid, space, medium=medium)
    shift = np.exp(
        -1j
        * (
            np.tile(space.k_x, space.n_y) * 50e-6
            + np.repeat(space.k_y, space.n_x) * -30e-6
        )
    )
    assert np.allclose(
        moved,
        shift[:, None, None] * centred,
        rtol=0,
        atol=1e-12 * np.max(np.abs(centred)),
    )
    fluence = np.sum(
        np.abs(np.fft.ifft2(_cube(moved, space), axes=(0, 1))) ** 2, axis=(2, 3)
    )
    iy, ix = np.unravel_index(np.argmax(fluence), fluence.shape)
    assert space.x[ix] == pytest.approx(50e-6, abs=space.x[1] - space.x[0])
    assert space.y[iy] == pytest.approx(-30e-6, abs=space.y[1] - space.y[0])


def test_every_frequency_of_a_tilted_beam_leaves_at_the_one_angle():
    grid, space, medium = _grid(), _space(n_x=128), SpatialMedium()
    tilt = 0.02
    spectrum = _beam(tilt=(tilt, 0.0)).build(grid, space, medium=medium)
    power = np.sum(np.abs(_cube(spectrum, space)) ** 2, axis=(0, 2))  # (n_x, n_omega)
    band = grid.active_mask & (grid.omega > 0)
    carried = band & (np.sum(power, axis=0) > 1e-3 * np.max(np.sum(power, axis=0)))
    wavenumber = (
        np.real(medium.index(grid.omega[carried])) * grid.omega[carried] / 299_792_458.0
    )
    centroid = (space.k_x @ power[:, carried]) / np.sum(power[:, carried], axis=0)
    # Towards +x is the spectrum's -k_x side, as Alignment's tilt has it.
    angle = np.arcsin(-centroid / wavenumber)
    assert np.allclose(angle, tilt, rtol=1e-3)


@pytest.mark.parametrize(
    "changes", [{"tilt": (0.02, -0.01)}, {"displacement": (4e-5, 2e-5)}]
)
def test_the_energy_is_set_after_the_screen(changes):
    grid, space, medium = _grid(), _space(), SpatialMedium()
    spectrum = _beam(**changes).build(grid, space, medium=medium)
    energy = spatial_energy(spectrum, SpatialFlux.resolve(grid, space, medium))
    assert energy == pytest.approx(ENERGY, rel=1e-12)


def test_a_tilt_matches_alignment_s_screen_on_the_same_beam():
    grid, space, medium = _grid(), _space(), SpatialMedium()
    centred = _beam().build(grid, space, medium=medium)
    tilted = _beam(tilt=(0.01, 0.0)).build(grid, space, medium=medium)
    band = grid.active_mask & (grid.omega > 0)
    k = np.zeros(grid.n_omega)
    k[band] = np.real(medium.index(grid.omega[band])) * grid.omega[band] / 299_792_458.0
    real = np.fft.ifft2(_cube(centred, space), axes=(0, 1))
    screen = np.exp(-1j * space.x[None, :, None] * math.sin(0.01) * k[None, None, :])
    expected = np.fft.fft2(real * screen[:, :, None, :], axes=(0, 1)).reshape(
        centred.shape
    )
    # The energy is set after the screen: oblique columns carry less axial flux, so the
    # tilted beam is the screened one up to one real factor of about 1 + theta^2 / 4.
    scale = np.vdot(expected, tilted) / np.vdot(expected, expected)
    assert abs(scale.imag) < 1e-12
    assert scale.real == pytest.approx(1 + 0.01**2 / 4, rel=1e-5)
    assert np.allclose(
        tilted, scale * expected, rtol=0, atol=1e-9 * np.max(np.abs(centred))
    )


def test_a_tilted_beam_travels_the_way_alignment_tilts_it():
    # A positive tilt sends the beam towards +x: the same beam, tilted by its own
    # tilt or by Alignment's, then carried 16 cm, lands at +L tan(theta) either way.
    from solprop.coupling.elements import (  # noqa: PLC0415 - only here
        Alignment,
        FreeSpace,
        SpatialState,
    )
    from solprop.coupling.transport import _sequence  # noqa: PLC0415 - only here

    grid, medium = _grid(), SpatialMedium()
    space = FreeGrid(radius_x=2.6e-3, n_x=240, radius_y=0.8e-3, n_y=48)
    tilt, distance = 5e-3, 0.16

    def centre(spectrum):
        samples = np.fft.ifft2(_cube(spectrum, space), axes=(0, 1))
        across = np.sum(np.abs(samples) ** 2, axis=(0, 2, 3))
        return float(space.x @ across / np.sum(across))

    def carried(state):
        return _sequence(state, [FreeSpace(distance, medium)])[0].spectrum

    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY)
    own = SpatialPulse(temporal, 150e-6, tilt=(tilt, 0.0)).build(
        grid, space, medium=medium
    )
    plain = SpatialPulse(temporal, 150e-6).build(grid, space, medium=medium)
    aligned = _sequence(
        SpatialState(grid, space, plain, medium),
        [Alignment(tilt=(tilt, 0.0), model_error=1e-3)],
    )[0]
    expected = distance * math.tan(tilt)
    assert centre(carried(SpatialState(grid, space, own, medium))) == pytest.approx(
        expected, rel=1e-3
    )
    assert centre(carried(aligned)) == pytest.approx(expected, rel=1e-3)


def test_a_line_beam_moves_along_x():
    grid, space, medium = _grid(), LineGrid(4e-4, 64), SpatialMedium()
    moved = _beam(displacement=(5e-5, 0.0)).build(grid, space, medium=medium)
    fluence = np.sum(np.abs(np.fft.ifft(moved, axis=0)) ** 2, axis=(1, 2))
    step = space.x[1] - space.x[0]
    assert space.x[np.argmax(fluence)] == pytest.approx(5e-5, abs=step)


def test_an_unmoved_beam_is_built_as_before():
    grid, space, medium = _grid(), _space(), SpatialMedium()
    assert np.array_equal(
        _beam().build(grid, space, medium=medium),
        _beam(displacement=(0.0, 0.0), tilt=(0.0, 0.0)).build(
            grid, space, medium=medium
        ),
    )


@pytest.mark.parametrize(
    ("space", "changes", "match"),
    [
        (RadialGrid(4e-4, 32), {"tilt": (0.01, 0.0)}, "cylindrical symmetry"),
        (LineGrid(4e-4, 64), {"displacement": (0.0, 1e-5)}, "uniform in y"),
        (LineGrid(4e-4, 64), {"tilt": (0.0, 0.01)}, "uniform in y"),
        (FreeGrid(4e-4, 16, 4e-4, 16), {"tilt": (0.3, 0.0)}, "transverse band"),
        (_space(), {"displacement": (3.5e-4, 0.0)}, "window's edge"),
        (FreeGrid(8e-3, 1024, 8e-3, 16), {"tilt": (0.01, 0.0)}, "pulse front"),
    ],
)
def test_each_launch_the_grid_cannot_carry_is_refused(space, changes, match):
    beam = _beam(**changes)
    if match == "pulse front":
        beam = SpatialPulse(beam.temporal, 2e-3, tilt=changes["tilt"])
    with pytest.raises(PulseError, match=match):
        beam.build(_grid(), space, medium=SpatialMedium())


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"displacement": (math.inf, 0.0)}, "two finite values"),
        ({"tilt": (0.0, math.nan)}, "two finite values"),
        ({"tilt": (math.pi / 2, 0.0)}, "below pi/2"),
    ],
)
def test_a_displacement_or_tilt_must_be_finite_and_below_a_right_angle(changes, match):
    with pytest.raises(PulseError, match=match):
        _beam(**changes)


def test_a_planar_beam_is_placed_by_its_own_offset():
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY)
    with pytest.raises(PulseError, match="offset_x"):
        PlanarPulse(temporal, 2e-4, waist_x=4e-5, tilt=(0.01, 0.0))
    assert PlanarGrid([SlabMode(63.5e-6, cladding=1.45)], 1e-3, 8).n_y == 8


def test_a_beam_into_a_guide_is_misaligned_by_alignment_not_the_beam():
    grid = _grid()
    index = np.where(grid.omega > 0, 1.0, 0.0)
    with pytest.raises(PulseError, match=r"coupling\.Alignment"):
        _beam(displacement=(1e-5, 0.0)).incident(grid, np.zeros((1, 2)), index=index)


def test_the_display_names_a_displacement_and_a_tilt():
    text = str(_beam(displacement=(1e-5, 0.0), tilt=(0.0, 0.02)))
    assert "displacement" in text
    assert "0.02) rad" in text
    assert "displacement" not in str(_beam())

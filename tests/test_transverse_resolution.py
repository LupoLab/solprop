# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A free-space run's transverse boundary, resolved at translation (SEP 40, 5.1-5.2).

The default is the safety ramp alone, on every geometry; a requested collar is
designed against the grid, the medium over the requested band and the noise-free
launch, and refused by name where it cannot be.
"""

import dataclasses
import math

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import DriverError, TranslationError
from solprop.grid import FieldGrid, RadialGrid
from solprop.linear import Frame
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stepper import PIController, RungeKutta
from solprop.translate.transverse import resolve_transverse

SPATIAL = ("radial", "cartesian", "line", "planar")
LENGTH = 0.01
W0, RADIUS = 35.36e-6, 106.07e-6
"""The suite's ``spatial_absorber`` case scaled paraxially by 1/8 (Part II)."""


def suite(
    fraction: float,
    specification: SpatialAbsorber | bool | None,
    *,
    length: float = LENGTH,
    energy: float = 1e-9,
) -> Simulation:
    """Return the scaled suite case on a grid with `fraction` of room."""
    grid = FieldGrid.from_limits(length, 800e-9, (500e-9, 1.6e-6), 250e-15)
    space = RadialGrid(RADIUS, round(64 * (1 + fraction)), collar_fraction=fraction)
    pulse = (
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), W0)
        if energy > 0
        else np.zeros((space.n_r, 1, grid.n_omega), dtype=complex)
    )
    return Simulation.from_radial(
        grid,
        space,
        SpatialMedium("Ar", 1.0),
        [],
        pulse,
        settings=SimulationSettings(
            integrator=RungeKutta(),
            controller=PIController(rtol=1e-6, atol=0.0),
            save_planes=[length],
            dz_max=length,
            dz_init=length * 1e-4,
            backend="numpy_ref",
        ),
        noise=None,
        spatial_absorber=specification,  # pyright: ignore[reportArgumentType]
    )


@pytest.mark.parametrize("route", SPATIAL)
def test_the_default_is_the_safety_ramp_alone(route: str) -> None:
    problem = ROUTES[route](None).translate()
    design = problem.transverse
    assert design is not None
    assert not design.has_collar
    geometry = problem.transform.geometry
    assert geometry is not None
    assert not np.any(geometry.absorber_rate)
    assert design.ramp_strength > 0.0
    assert float(np.max(design.ramp_rate)) == pytest.approx(design.ramp_strength)
    if route == "radial":
        return
    assert problem.linear.separable is not None
    ramp = problem.linear.separable.kperp_rate
    assert ramp is not None
    assert np.array_equal(ramp, design.ramp_rate)


def test_the_radial_ramp_rides_the_dense_operator_exactly() -> None:
    # The radial operator is diagonal in k_r, so the ramp enters it as -alpha_s / 2
    # on each column's real part, inside the active band, and nowhere else.
    plain = suite(0.0, False).translate()
    ramped = suite(0.0, None).translate()
    design = ramped.transverse
    assert design is not None
    before, after = plain.linear.phi_constant, ramped.linear.phi_constant
    assert before is not None and after is not None
    active = plain.grids.active_mask
    difference = np.real(after - before)
    expected = np.broadcast_to(-0.5 * design.ramp_rate[:, None, None], before.shape)
    assert np.allclose(
        difference[..., active],
        expected[..., active],
        rtol=1e-12,
        atol=1e-9 * design.ramp_strength,
    )
    assert np.all(difference[..., ~active] == 0.0)


def test_the_ramp_covers_the_outer_tenth_of_the_wavenumbers() -> None:
    problem = suite(0.0, None).translate()
    design = problem.transverse
    assert design is not None
    geometry = problem.transform.geometry
    assert geometry is not None
    k_r = geometry.spectral_axes[0]
    (start,) = design.kappa_s
    assert start == pytest.approx(0.9 * float(k_r[-1]))
    assert np.all(design.ramp_rate[k_r <= start] == 0.0)
    assert np.all(design.ramp_rate[k_r > start] > 0.0)


def test_the_scaled_suite_collar_is_the_full_case_s_design_scaled() -> None:
    # At full size f = 2.6 resolves to 242 1/m (Part II); a paraxial scaling of the
    # lengths by 1/8 scales the rate by 8.
    problem = suite(2.6, SpatialAbsorber()).translate()
    design = problem.transverse
    assert design is not None and design.has_collar
    assert design.alpha0 == pytest.approx(8 * 242, rel=0.01)
    assert design.kappa_p[0] == pytest.approx(3 * math.sqrt(2) / W0, rel=1e-3)
    geometry = problem.transform.geometry
    assert geometry is not None
    outside = geometry.window < 1
    assert np.all(geometry.absorber_rate[~outside] == 0.0)
    assert float(np.max(geometry.absorber_rate)) == pytest.approx(
        design.alpha0, rel=1e-6
    )
    # The crossing limit holds an adaptive run's longest step.
    assert problem.stepper.dz_max == pytest.approx(design.crossing_step)
    assert design.crossing_step < LENGTH


def test_the_resolved_operator_does_not_depend_on_the_run_length() -> None:
    # Section 7.7: the design is the same for L and 2L. Checked directly, before the
    # design check, which does depend on L: it decides how much slow content reaches
    # the edge, not the collar's strength.
    designs = []
    for length in (LENGTH, 2 * LENGTH):
        simulation = suite(2.6, SpatialAbsorber(), length=length)
        problem = suite(2.6, False, length=length).translate()
        space = simulation.spatial_grid
        assert isinstance(space, RadialGrid)
        dispersion = SpatialDispersion(
            SpatialMedium("Ar", 1.0), simulation.grid, Frame.comoving(simulation.grid)
        )
        designs.append(
            resolve_transverse(
                SpatialAbsorber(), space, dispersion, problem.initial.field0, length
            )
        )
    short, long = designs
    assert long.alpha0 == short.alpha0
    assert long.ramp_strength == short.ramp_strength
    assert np.array_equal(long.ramp_rate, short.ramp_rate)


def test_a_declared_band_is_used_as_given() -> None:
    kappa = 2.5e5
    design = suite(2.6, SpatialAbsorber(kappa=kappa)).translate().transverse
    assert design is not None
    assert design.kappa_p == (kappa,)
    angle = suite(2.6, SpatialAbsorber(angle=0.03)).translate().transverse
    assert angle is not None
    low, high = angle.notes["kappa_p_band"][0]  # pyright: ignore[reportIndexIssue]
    assert high > low > 0.0


def test_a_collar_without_room_is_refused_by_name() -> None:
    with pytest.raises(TranslationError, match="collar_fraction > 0"):
        suite(0.0, SpatialAbsorber()).translate()


def test_a_collar_narrower_than_the_profile_s_floor_is_refused() -> None:
    with pytest.raises(TranslationError, match="floor") as caught:
        suite(0.5, SpatialAbsorber()).translate()
    assert "collar_fraction of at least" in str(caught.value)


def test_a_band_reaching_the_ramp_is_refused() -> None:
    with pytest.raises(TranslationError, match="safety ramp"):
        suite(2.6, SpatialAbsorber(kappa=1.75e6)).translate()


def test_a_launch_derived_band_needs_a_launch() -> None:
    with pytest.raises(TranslationError, match="kappa or angle"):
        suite(2.6, SpatialAbsorber(), energy=0.0).translate()


def test_the_keyword_and_the_settings_cannot_both_name_a_boundary() -> None:
    simulation = suite(2.6, None)
    settings = dataclasses.replace(
        simulation.settings, spatial_absorber=SpatialAbsorber(kappa=2e5)
    )
    with pytest.raises(DriverError, match="not both"):
        Simulation.from_radial(
            simulation.grid,
            simulation.spatial_grid,  # pyright: ignore[reportArgumentType]
            simulation.medium,  # pyright: ignore[reportArgumentType]
            [],
            np.zeros((1, 1)),
            settings=settings,
            spatial_absorber=SpatialAbsorber(kappa=3e5),
        )


def test_a_modal_run_refuses_a_specification() -> None:
    simulation = ROUTES["gnlse"](None)
    simulation.settings = dataclasses.replace(
        simulation.settings, spatial_absorber=SpatialAbsorber()
    )
    with pytest.raises(DriverError, match="modal route"):
        simulation.translate()


def test_the_record_holds_the_specification_and_the_design() -> None:
    simulation = suite(2.6, SpatialAbsorber())
    metadata = simulation.metadata()
    assert metadata.configuration["spatial_absorber"] == SpatialAbsorber()
    assert metadata.transverse is not None
    assert metadata.transverse["collar"] is True
    assert metadata.transverse["alpha0"] == pytest.approx(8 * 242, rel=0.01)
    default = suite(0.0, None).metadata()
    assert "spatial_absorber" not in default.configuration
    assert default.transverse is not None
    assert default.transverse["collar"] is False

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Several beams launched together: a SpatialPulseSum is the sum of its beams."""

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import PulseError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse, SpatialPulseSum
from solprop.recipe import Recipe
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy
from solprop.stepper import PIController

ENERGY = 1e-9


def _grid() -> EnvelopeGrid:
    return EnvelopeGrid.from_limits(0.02, 800e-9, (650e-9, 1000e-9), 300e-15)


def _space() -> FreeGrid:
    return FreeGrid(radius_x=2e-3, n_x=96, radius_y=1e-3, n_y=48)


def _beam(**changes: object) -> SpatialPulse:
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY)
    return SpatialPulse(temporal, 150e-6, **changes)  # pyright: ignore[reportArgumentType]


def _energy(spectrum: np.ndarray, space: object) -> float:
    medium = SpatialMedium()
    return spatial_energy(spectrum, SpatialFlux.resolve(_grid(), space, medium))  # pyright: ignore[reportArgumentType]


def test_apart_the_energies_add_and_together_the_fields_do():
    grid, space, medium = _grid(), _space(), SpatialMedium()
    apart = SpatialPulseSum(
        (_beam(displacement=(-0.8e-3, 0.0)), _beam(displacement=(0.8e-3, 0.0)))
    )
    built = apart.build(grid, space, medium=medium)
    left = _beam(displacement=(-0.8e-3, 0.0)).build(grid, space, medium=medium)
    right = _beam(displacement=(0.8e-3, 0.0)).build(grid, space, medium=medium)
    assert np.allclose(built, left + right, rtol=0, atol=1e-12 * np.abs(built).max())
    assert _energy(built, space) == pytest.approx(2 * ENERGY, rel=1e-9)
    # Two copies in one place: the field doubles, so the energy is four times one.
    together = SpatialPulseSum((_beam(), _beam())).build(grid, space, medium=medium)
    assert _energy(together, space) == pytest.approx(4 * ENERGY, rel=1e-9)


def test_a_radial_sum_of_co_centred_beams_builds():
    grid, medium = _grid(), SpatialMedium()
    space = RadialGrid(1e-3, 32)
    wide = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY), 3e-4
    )
    built = SpatialPulseSum((_beam(), wide)).build(grid, space, medium=medium)
    assert built.shape == _beam().build(grid, space, medium=medium).shape


@pytest.mark.parametrize(
    ("beams", "match"),
    [
        ((), "at least one beam"),
        ((_beam(jones=(1, 0)), _beam(jones=(0, 1))), "share one Jones vector"),
        ((_beam(jones=(1, 0)), _beam()), "share one Jones vector"),
    ],
)
def test_a_sum_refuses_what_cannot_be_launched_together(beams, match):
    with pytest.raises(PulseError, match=match):
        SpatialPulseSum(beams)


def test_a_member_that_is_no_beam_is_refused():
    # The tests' type checker refuses a wrong member at construction, before the sum's
    # own check: a caller without it meets the check, run here directly.
    total = SpatialPulseSum((_beam(),))
    object.__setattr__(
        total, "beams", (GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY),)
    )
    with pytest.raises(PulseError, match="member 0 of the SpatialPulseSum"):
        total.__post_init__()


def test_a_sum_names_its_beams_and_their_polarisation():
    total = SpatialPulseSum((_beam(jones=(1, 1j)), _beam(jones=(2, 2j))))
    assert total.jones == (1, 1j)
    assert repr(total).startswith("SpatialPulseSum(SpatialPulse(")
    title, rows = total._describe()
    assert title == "SpatialPulseSum" and [key for key, _ in rows] == [
        "beam 0",
        "beam 1",
    ]


def test_a_free_run_launches_the_sum_and_a_recipe_holds_it():
    grid, space, medium = _grid(), _space(), SpatialMedium()
    total = SpatialPulseSum(
        (
            _beam(displacement=(-0.4e-3, 0.0), tilt=(5e-3, 0.0)),
            _beam(displacement=(0.4e-3, 0.0), tilt=(-5e-3, 0.0)),
        )
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[0.02],
        dz_init=1e-3,
        dz_max=5e-3,
    )
    simulation = Simulation.from_free(grid, space, medium, (), total, settings=settings)
    assert np.allclose(
        simulation._field0, total.build(grid, space, medium=medium), rtol=0, atol=1e-30
    )
    text = Recipe.example("cartesian_beam").to_text()
    start, end = text.index("[build.initial]"), text.index("[build.settings]")
    beam = (
        '{ type = "SpatialPulse", temporal = { type = "GaussianPulse", wavelength = '
        '"800 nm", fwhm = "30 fs", energy = "0.75 nJ" }, waist = "1 mm", '
        'displacement = ["%s mm", "0 mm"] }'
    )
    launch = (
        '[build.initial]\ntype = "SpatialPulseSum"\nbeams = [\n'
        f"    {beam % -1.0},\n    {beam % 1.0},\n]\n\n"
    )
    recipe = Recipe.loads(text[:start] + launch + text[end:])
    built = recipe.build().initial
    assert isinstance(built, SpatialPulseSum) and len(built.beams) == 2
    assert "SpatialPulseSum" in recipe.to_python()

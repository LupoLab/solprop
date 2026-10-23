# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test physical flux through radial gas paths on every backend (SEP 34)."""

import numpy as np
import pytest

from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.exceptions import MaterialError, TranslationError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.profiles import FromCallable, Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController

LENGTH = 1e-3
ENERGY = 3e-9


def _step(z: np.ndarray) -> np.ndarray:
    """Return a 1 to 3 bar pressure step at half the length."""
    return np.where(z < LENGTH / 2, 1.0, 3.0)


def _case(backend: str, medium: SpatialMedium, planes: list[float]) -> Simulation:
    """Return a short linear radial run of a 3 nJ beam through `medium`."""
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY), 60e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=planes,
        dz_max=LENGTH / 4,
        dz_init=LENGTH / 4,
        backend=backend,
        stats=StatsSpec(names=("energy", "peak_intensity")),
    )
    return Simulation.from_radial(
        grid, RadialGrid(6e-4, 16), medium, [], pulse, settings=settings
    )


MEDIA = {
    "ramp": SpatialMedium("Ar", pressure=Linear(0.5, 3.0, LENGTH)),
    "step": SpatialMedium(
        "Ar",
        pressure=FromCallable(
            _step,
            (0.0, 1.2 * LENGTH),
            jumps=np.array([LENGTH / 2]),
            turning=np.empty(0),
        ),
    ),
}


@pytest.mark.parametrize("profile", sorted(MEDIA))
def test_linear_flux_is_conserved_through_a_gas_path(spatial_backend_class, profile):
    """A lossless path keeps the launched physical energy at every plane."""
    medium = MEDIA[profile]
    simulation = _case(spatial_backend_class.name, medium, [0.25e-3, 0.75e-3, LENGTH])
    admission = simulation.translate().flux
    assert admission is not None
    assert admission.admission is not None
    # The certificate is exact for these monotone paths; the step attains it.
    if profile == "step":
        index = medium.index(np.array([2.35e15]), 0.0)[0].real
        after = medium.index(np.array([2.35e15]), LENGTH)[0].real
        fresnel = ((after - index) / (after + index)) ** 2
        assert admission.admission.reflection == pytest.approx(fresnel, rel=2e-2)
    output = simulation.run()
    stored = np.asarray(output["stats/energy"]).ravel()
    # The linear flow conserves Y_z |E|^2 to its representation budget (about
    # 1e-14 relative here) and the stepper's rounding.
    assert stored == pytest.approx(np.full(stored.shape, ENERGY), rel=1e-11)
    grid = simulation.grid
    space = simulation.spatial_grid
    assert isinstance(space, RadialGrid)
    for field, z in zip(
        output.fields("fft"), [0.0, 0.25e-3, 0.75e-3, LENGTH], strict=True
    ):
        flux = SpatialFlux.resolve(grid, space, medium, z=z)
        # In-run and saved-plane measures share one definition: rounding of the sum
        # plus the material table's interpolation (relative 1e-12 of k^2).
        assert spatial_energy(field, flux) == pytest.approx(ENERGY, rel=1e-11)


def test_a_plane_or_flux_on_a_jump_needs_a_side():
    """A save plane on a declared jump refuses; a flux there names its side."""
    medium = MEDIA["step"]
    with pytest.raises(TranslationError, match="declared jump"):
        _case("numpy_ref", medium, [LENGTH / 2, LENGTH]).translate()
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 16)
    with pytest.raises(MaterialError, match="must say which"):
        SpatialFlux.resolve(grid, space, medium, z=LENGTH / 2)
    before = SpatialFlux.resolve(grid, space, medium, z=LENGTH / 2, side="before")
    after = SpatialFlux.resolve(grid, space, medium, z=LENGTH / 2, side="after")
    assert np.all(before.admittance().real <= after.admittance().real)
    assert np.any(before.admittance().real < after.admittance().real)

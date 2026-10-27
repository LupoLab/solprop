# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify physical flux through gas gradients and loss on every backend (SEP 34)."""

import numpy as np
import pytest

from bounds import gamma
from solprop.backends.spatial import collar_field_fraction
from solprop.constants import C
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.stepper.controller import PIController
from solprop.translate.problem import KernelProblem

LENGTH = 1e-3
SPACE = RadialGrid(6e-4, 32)


def _grid() -> EnvelopeGrid:
    """Return the 800 nm envelope grid every case here shares."""
    return EnvelopeGrid.from_limits(
        LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )


def _run(  # noqa: PLR0913 - one case's physics and its numerical settings
    backend: str,
    medium: SpatialMedium,
    initial: SpatialPulse | np.ndarray,
    *,
    responses: tuple[object, ...] = (),
    names: tuple[str, ...] = ("energy",),
    rtol: float = 1e-8,
    field_range: tuple[float, float] | None = None,
    space: RadialGrid = SPACE,
    steps: int | None = None,
) -> tuple[dict[str, np.ndarray], KernelProblem]:
    """Run 1 mm of `medium`; return the stored statistics, exit field and problem."""
    settings = SimulationSettings(
        controller=PIController(rtol=rtol, atol=1e-14),
        save_planes=[LENGTH / 2, LENGTH],
        dz_max=LENGTH / 4,
        dz_init=LENGTH / 16,
        backend=backend,
        ionisation_field_range=field_range,
        stats=StatsSpec(names=names),
        schedule=None
        if steps is None
        else FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
    )
    simulation = Simulation.from_radial(
        _grid(), space, medium, list(responses), initial, settings=settings, noise=None
    )
    output = simulation.run()
    values = {name: np.asarray(output[f"stats/{name}"]) for name in names}
    values["exit"] = np.asarray(list(output.fields("fft"))[-1])
    return values, simulation.translate()


def _pulse(energy: float) -> SpatialPulse:
    """Return a 30 fs, 60 um beam of `energy` joules."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), 60e-6
    )


def test_a_falling_path_undoes_the_amplitude_of_a_rising_one(spatial_backend_class):
    """Exchanging the planes of a path inverts its endpoint amplitude exactly."""
    backend = spatial_backend_class.name
    rising = SpatialMedium("Ar", pressure=Linear(0.5, 3.0, LENGTH))
    falling = SpatialMedium("Ar", pressure=Linear(3.0, 0.5, LENGTH))
    grid, space = _grid(), SPACE
    start = _pulse(3e-9).build(grid, space, medium=rising)
    there = _run(backend, rising, start)[0]["exit"]
    back = _run(backend, falling, there)[0]["exit"]
    active = grid.active_mask
    # Each column's |E| scales by sqrt(Y_z(0) / Y_z(L)) out and by its inverse back:
    # the phases accumulate, the amplitudes cancel. The ratio departs from one at
    # each leg's endpoint representation (relative 1e-9 of a log ratio below 1e-3
    # here) and the rounding of the steps.
    outward = SpatialFlux.resolve(grid, space, rising, z=LENGTH).admittance()
    inward = SpatialFlux.resolve(grid, space, rising, z=0.0).admittance()
    exponent = float(
        np.max(np.abs(np.log(np.real(outward[:, active] / inward[:, active]))))
    )
    budget = 2 * (1e-9 * exponent + gamma(256))
    initial = np.abs(start[:, 0][:, active])
    scale = float(np.max(initial))
    assert np.max(np.abs(np.abs(back[:, 0][:, active]) - initial)) <= budget * scale
    # One leg alone changes the amplitude by far more than the round trip's residual.
    assert (
        np.max(np.abs(np.abs(there[:, 0][:, active]) - initial)) > 100 * budget * scale
    )


def test_kerr_keeps_its_energy_through_a_gradient(spatial_backend_class):
    """Instantaneous Kerr does no work, so physical energy survives a ramp."""
    backend = spatial_backend_class.name
    kerr = (KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False),)
    ramp = SpatialMedium("Ar", pressure=Linear(0.5, 3.0, LENGTH))
    # About 0.2 rad of nonlinear phase on axis: the dynamics are not trivial.
    stored = _run(backend, ramp, _pulse(3e-4), responses=kerr)[0]["energy"].ravel()
    # The local source prefactor's weight derivative cancels the endpoint term (SEP
    # 34, section 4.5); what remains is the controller's error, rtol 1e-8 per step.
    assert stored == pytest.approx(np.full(stored.shape, 3e-4), rel=1e-8)
    # The matched uniform limit: a constant path is the uniform medium.
    flat = _run(
        backend,
        SpatialMedium("Ar", pressure=Linear(1.75, 1.75, LENGTH)),
        _pulse(3e-4),
        responses=kerr,
    )[0]
    uniform = _run(
        backend, SpatialMedium("Ar", pressure=1.75), _pulse(3e-4), responses=kerr
    )[0]
    assert np.max(np.abs(flat["exit"] - uniform["exit"])) <= gamma(64) * np.max(
        np.abs(uniform["exit"])
    )


def test_plasma_loss_through_a_gradient_follows_the_local_density(
    spatial_backend_class,
):
    """A ramp's plasma loss is the average of its local media's (SEP 34, section 7)."""
    backend = spatial_backend_class.name
    plasma = (EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 800e-9)),)

    def loss(medium: SpatialMedium) -> float:
        values, _ = _run(
            backend,
            medium,
            _pulse(3e-4),
            responses=plasma,
            field_range=(1e8, 1e11),
            space=RadialGrid(6e-4, 16),
            steps=16,
        )
        energy = values["energy"].ravel()
        return float(1 - energy[-1] / energy[0])

    ramp = loss(SpatialMedium("Ar", pressure=Linear(1.0, 3.0, LENGTH)))
    low = loss(SpatialMedium("Ar", pressure=1.0))
    high = loss(SpatialMedium("Ar", pressure=3.0))
    middle = loss(SpatialMedium("Ar", pressure=2.0))
    # The uniform losses are concave in density: 1.9, 3.2 and 4.3 % at 1, 2 and 3 bar.
    assert 0.01 < low < middle < high
    assert middle > (low + high) / 2
    # Physical E drives the rate at the local density, so the ramp, whose density is
    # linear in z, loses the average of its local loss rates: between the chord of
    # its endpoints and the loss at its mean pressure, as concavity requires. Sixteen
    # steps resolve that ordering (32 move the ramp's loss by 6e-4 of itself).
    assert (low + high) / 2 < ramp < middle
    assert ramp == pytest.approx(middle, rel=1e-2)


def test_uniform_loss_is_the_exponential_and_not_the_collar(spatial_backend_class):
    """Material loss attenuates each column as exp(-2 Im k_z L); the collar is apart."""
    backend = spatial_backend_class.name
    medium = SpatialMedium(1.5 + 2e-5j)
    grid, space = _grid(), SPACE
    start = _pulse(3e-9).build(grid, space, medium=medium)
    flux = SpatialFlux.resolve(grid, space, medium)
    # Im(c k_z / omega) omega / c = Im k_z on the forward ports; zero elsewhere.
    rate = np.imag(flux.admittance()) * grid.omega / C
    values, problem = _run(backend, medium, start)
    expected = [
        spatial_energy(start * np.exp(-rate * z)[:, None, :], flux)
        for z in (0.0, LENGTH / 2, LENGTH)
    ]
    # Im n = 2e-5 at 800 nm gives 2 Im k_z L ~ 0.31: about a quarter of the energy is
    # absorbed by the material, while the beam stays well inside the collar.
    assert expected[-1] < 0.8 * expected[0]
    assert values["energy"].ravel() == pytest.approx(expected, rel=1e-9)
    # The collar's field-norm diagnostic is separate accounting, and here empty.
    assert collar_field_fraction(problem, values["exit"]) < 1e-12

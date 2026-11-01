# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 40's physics and scheme witnesses on numpy_ref (sections 4.7, 7.4 and 7.5)."""

import math
import warnings

import numpy as np
import pytest

from solprop.boundaries import SpatialAbsorber, no_absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryWarning, TranslationError
from solprop.grid import EnvelopeGrid, LineGrid
from solprop.output.schema import PlaneStore
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController, RungeKutta

K = 2 * math.pi / 800e-9
CROSSING = 3e-3
"""One transverse circuit of the counterexample's box."""


def crossing_run(schedule: FixedSchedule | None) -> Simulation:
    """Return section 4.7's counterexample on the line route, with a requested collar.

    512 samples, periodic half-width 300 um, requested 200 um; the launch
    exp(-(x/40 um)^2 + i kappa_0 x), kappa_0 = 0.2 k at 800 nm, over 3 mm.
    """
    grid = EnvelopeGrid.from_limits(CROSSING, 800e-9, (790e-9, 810e-9), 1e-12)
    space = LineGrid(200e-6, 512, collar_fraction=0.5)
    x = np.asarray(space.x)
    kappa0 = 0.2 * K
    spectrum = GaussianPulse(wavelength=800e-9, fwhm=200e-15, energy=1e-9).build(grid)[
        0, 0
    ]
    physical = np.exp(-((x / 40e-6) ** 2) - 1j * kappa0 * x)
    launch = np.fft.fft(physical)[:, None, None] * spectrum[None, None, :]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=0.0),
        save_planes=[CROSSING],
        dz_max=CROSSING,
        dz_init=CROSSING * 1e-4,
        integrator=RungeKutta(),
        backend="numpy_ref",
        schedule=schedule,
        spatial_absorber=SpatialAbsorber(kappa=1.1 * kappa0),
    )
    return Simulation.from_free(
        grid, space, SpatialMedium(1.0), [], launch, settings=settings, noise=None
    )


def aperture_share(simulation: Simulation) -> float:
    """Return the energy left in the requested aperture at the end, over the launch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    space = simulation.spatial_grid
    assert isinstance(space, LineGrid)
    inside = np.abs(np.asarray(space.x)) <= space.radius_x
    fields = [
        np.fft.ifft(np.asarray(output["field"][i])[:, 0, :], axis=0) for i in (0, -1)
    ]
    start, end = (float(np.sum(np.abs(f[inside]) ** 2)) for f in fields)
    return end / start


def test_the_crossing_counterexample_is_refused_on_one_step() -> None:
    # Section 4.7: one exact-diffraction step carries the packet round the period
    # unseen; in-stage under a fixed schedule it is refused, never silently wrong.
    with pytest.raises(TranslationError, match=r"stability limit|crossing"):
        crossing_run(
            FixedSchedule.uniform(0.0, CROSSING, CROSSING, save_planes=[CROSSING])
        ).translate()


@pytest.mark.medium
def test_the_crossing_counterexample_is_accurate_when_adaptive() -> None:
    # The exact generator leaves about 1e-6 of the launch in the aperture after one
    # circuit (section 4.7's table, for its own collar); adaptive in-stage RK, held
    # to the crossing limit, absorbs the packet rather than returning it.
    assert aperture_share(crossing_run(None)) < 1e-4


def test_the_ledger_closes_with_the_ramp_in_a_lossless_medium() -> None:
    # Section 4.9 and 7.7: U(z) plus ramp_removed equals the launch's energy in a
    # lossless linear medium uniform along z. The ramp is diagonal in the columns,
    # which do not couple here, so its credit from each interval's starting power is
    # exact whatever the read spacing. Content above discard_above is declared, so the
    # launch's share in the ramp's band is recorded at INFO, not refused.
    grid = EnvelopeGrid.from_limits(2e-3, 800e-9, (790e-9, 810e-9), 1e-12)
    space = LineGrid(100e-6, 128, collar_fraction=0.0)
    x = np.asarray(space.x)
    top = float(np.max(np.abs(space.k_x)))
    spectrum = GaussianPulse(wavelength=800e-9, fwhm=200e-15, energy=1e-9).build(grid)[
        0, 0
    ]
    physical = np.exp(-((x / 20e-6) ** 2)) * (1 + 0.1 * np.exp(-1j * 0.97 * top * x))
    launch = np.fft.fft(physical)[:, None, None] * spectrum[None, None, :]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=0.0),
        save_planes=[2e-3],
        dz_max=2e-3,
        dz_init=2e-7,
        integrator=RungeKutta(),
        backend="numpy_ref",
        spatial_absorber=SpatialAbsorber(real_space=False, discard_above=0.9 * top),
        # No spectral or temporal collar: the ramp is the only loss to account for.
        absorber=no_absorber(grid),
    )
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(1.0), [], launch, settings=settings, noise=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    assert output.outcome is not None
    record = output.outcome.execution["boundary_monitor"]
    assert isinstance(record, dict)
    edge = record["transverse"]
    fields = [np.asarray(output["field"][i]) for i in (0, -1)]
    start, end = (float(np.sum(np.abs(f) ** 2)) for f in fields)
    removed = edge["raw_ramp_removed"]
    assert removed > 1e-3
    assert end / start + removed == pytest.approx(1.0, abs=1e-9)

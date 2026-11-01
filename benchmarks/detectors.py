# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract the far-field-statistics tutorial's two-beam slab and detectors."""

import numpy as np

from solprop.boundaries import absorber, custom_absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata.nonlinear_data import chi3_from_n2
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController

__all__ = ["far_field"]


def far_field() -> Simulation:
    """Keep the tutorial's illustrative medium, launch, apertures and absorbers.

    Index 1.45 and nonlinear index 3e-20 m²/W are the declared illustrative
    sample in ``tutorials/free_space/plot_07_far_field_statistics.py``;
    neither is presented as a measured material coefficient.
    """
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (740e-9, 870e-9), 300e-15)
    space = FreeGrid(0.4e-3, 224, 0.4e-3, 64, collar_fraction=0.0)
    x, y = np.meshgrid(space.x, space.y)
    # The tutorial's launch: each beam's physical field, peak 7.5051e8 V/m (SEP 34).
    pulse = GaussianPulse(wavelength=800e-9, fwhm=50e-15, peak_power=1.0)
    profile = np.exp(-(x**2 + y**2) / (40e-6) ** 2)
    beams = []
    for direction in (1, -1):
        tilt = np.exp(
            -1j * direction * x[..., None] * grid.omega * 1.45 * np.sin(0.010) / C
        )
        physical = 7.5051e8 * profile[..., None] * tilt * pulse.build(grid)[0, 0]
        beams.append(np.fft.fft2(physical, axes=(0, 1)).reshape(-1, 1, grid.n_omega))
    image_x = 0.25 * C * (2 * np.pi * 1.45 / 800e-9 * np.sin(0.010)) / grid.omega0
    holes = ApertureSet(
        centres=[[image_x, 0], [-image_x, 0], [3 * image_x, 0], [0, 1.6 * image_x]],
        diameters=[2.4e-3, 2.4e-3, 2.4e-3, 1.5e-3],
        distances=[0.25] * 4,
        edges=[0.5e-3, 0.5e-3, 0.5e-3, 0.3e-3],
        profile="rcos",
        weighting="quadrature",
        labels=("beam 1", "beam 2", "signal", "dark"),
    )
    quadrants = np.array(
        [
            [-np.inf, 0, -np.inf, 0],
            [0, np.inf, -np.inf, 0],
            [-np.inf, 0, 0, np.inf],
            [0, np.inf, 0, np.inf],
        ]
    )
    edge = 2 * np.pi * C / 770e-9
    band = custom_absorber(
        grid,
        lambda omega: 3e3 / (1 + np.exp(-(omega - edge) / 2e13)),
        collar=absorber(grid, temporal=False),
    )
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium(1.45),
        [KerrResponse(chi3_from_n2(3e-20, 1.45), thg=False)],
        beams[0] + beams[1],
        settings=SimulationSettings(
            PIController(rtol=1e-6, atol=1e-12),
            save_planes=np.linspace(grid.length / 8, grid.length, 8),
            dz_init=grid.length / 8,
            dz_max=grid.length / 8,
            stats=StatsSpec(
                names=(
                    "energy",
                    "beam_spectrum",
                    "aperture_spectrum",
                    "region_spectrum",
                    "absorbed_power",
                ),
                apertures=holes,
                spectral_regions=quadrants,
                periodic=("absorbed_power",),
                period=1,
            ),
            absorber=band,
            fields="planes",
            field_planes=(-1,),
            backend="numpy_ref",
        ),
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The axial-flux statistics' cases (SEP 34 sections 4.2 and 5.1) and their budgets.

One state at one plane per case: lossy glass on the radial, Cartesian and line routes
(so the admittance is complex), an argon gradient read past the entrance on the radial
route, and the planar plates. Each statistic's rounding budget (:func:`budget`) is fixed
here before any comparison (SEP 25). The case catalogue's geometry rows
(``geometry_rows.py``) compare native with the reference on them;
``test_flux_parity.py`` keeps a width's refusal and native's ledger.
"""

import numpy as np

from backends.planar_cases import planar_case
from bounds import gamma
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController, RungeKutta

LOCAL = (
    "energy",
    "on_axis_spectrum",
    "on_axis_intensity",
    "peak_intensity",
    "beam_radius",
    "fwhm_r",
)
FAR = (
    "beam_spectrum",
    "region_spectrum",
    "aperture_spectrum",
    "aperture_on_axis_spectrum",
    "absorbed_power",
)
HOLES = ApertureSet(
    centres=[[0.0, 0.0], [1e-3, 0.0]],
    diameters=[2e-3, 1e-3],
    distances=[0.25, 0.25],
    edges=[2e-4, 1e-4],
    profile="tanh",
)
HALVES = np.array([[-np.inf, 0.0, -np.inf, np.inf], [0.0, np.inf, -np.inf, np.inf]])
LOSSY = SpatialMedium(1.45 + 2e-3j)
"""Glass with a loss length of tens of micrometres: Im(eta)/Re(eta) about 1e-3."""


def flux_grid():
    """Return the cases' envelope grid: a millimetre, 600 nm to 1.2 um, 150 fs."""
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)


def flux_pulse(waist, energy=3e-9):
    """Return a 30 fs beam of `waist` and `energy`, elliptically polarised."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy),
        waist,
        jones=(1, 0.5j),
    )


def flux_settings(names, **extra):
    """Return four fixed Runge-Kutta steps over a millimetre, observing `names`."""
    return SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2.5e-4,
        dz_init=2.5e-4,
        absorber=None,
        integrator=RungeKutta(),
        stats=StatsSpec(names=names, **extra),
    )


def radial_case(medium):
    """Return ``(grid, problem, run_request)`` of a radial beam in `medium`."""
    grid = flux_grid()
    space = RadialGrid(6e-4, 16)
    simulation = Simulation.from_free(
        grid,
        space,
        medium,
        [],
        flux_pulse(60e-6),
        settings=flux_settings(LOCAL),
        noise=None,
    )
    return grid, simulation.translate(), simulation._run_request()


def free_case(space):
    """Return ``(grid, problem, run_request)`` of a beam in lossy glass on `space`."""
    grid = flux_grid()
    names = LOCAL + FAR
    simulation = Simulation.from_free(
        grid,
        space,
        LOSSY,
        [],
        flux_pulse(60e-6, 3e-9 if isinstance(space, FreeGrid) else 1e-3),
        settings=flux_settings(names, apertures=HOLES, spectral_regions=HALVES),
        noise=None,
    )
    return grid, simulation.translate(), simulation._run_request()


def planar_flux_case():
    """Return ``(grid, problem, run_request)`` of the planar Kerr case at its launch."""
    grid, _, problem, run_request = planar_case("kerr", steps=0)
    return grid, problem, run_request.for_run(stats=StatsSpec(names=LOCAL))


CASES = {
    # name: (builder, position of the plane, m); the state is the launch with a
    # chirp-like phase on every column, so that the frequencies and the columns
    # interfere at the nodes and the closure's cross terms enter every local value.
    "radial-lossy": (lambda: radial_case(LOSSY), 0.0),
    "radial-gradient": (
        lambda: radial_case(SpatialMedium("Ar", pressure=Linear(0.5, 3.0, 1e-3))),
        6e-4,
    ),
    "cartesian-lossy": (lambda: free_case(FreeGrid(3e-4, 24, 3e-4, 16)), 0.0),
    "line-lossy": (lambda: free_case(LineGrid(4e-4, 24)), 0.0),
    "planar": (planar_flux_case, 0.0),
}


def budget(name, grid, problem):
    """The rounding budget of one statistic, relative to its largest value."""
    n_columns = problem.shape[0]
    n_pol = problem.shape[1]
    if name in ("energy", "beam_spectrum", "region_spectrum", "aperture_spectrum"):
        # A sum over n_pol * n_columns * n_omega products of a squared modulus,
        # the admittance's root and division and the weights, in any order.
        return gamma(16 * (n_pol * n_columns + grid.n_omega))
    if name in ("absorbed_power", "aperture_on_axis_spectrum", "on_axis_spectrum"):
        # As above, and each amplitude's coherent sum over the columns before its
        # pairing; absorbed_power adds the rate and the band sum.
        return gamma(32 * (n_pol * n_columns + grid.n_omega))
    # The local statistics synthesise E and G at the nodes (n_columns terms, a
    # transverse FFT or a dense synthesis) and invert the time transform
    # (log2 n_t stages); their pairing sums n_pol * n_t products. Widths
    # interpolate a fluence whose relative slope is of order one at the crossing,
    # which a factor of eight covers.
    synthesis = 16 * (n_columns + n_pol * grid.n_t + 64 * int(np.log2(grid.n_t)))
    return gamma(8 * synthesis if name in ("beam_radius", "fwhm_r") else synthesis)

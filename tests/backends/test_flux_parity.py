# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Three-backend parity of the axial-flux statistics (SEP 34 sections 4.2 and 5.1).

The same state, loaded at the same plane, is measured by ``numpy_ref``,
``numba_cpu`` and, when it is installed, the native CPU backend: energy, the beam,
aperture and region spectra, the aperture's on-axis spectrum, ``absorbed_power``,
the on-axis spectrum and intensity, ``peak_intensity`` and the fluence widths. The
media are lossy (or the planar walls are), so the admittance is complex and the
local statistics' cross terms ``Im(eta)`` are far above rounding; the radial argon
gradient reads the plane measure's material table at a position past the entrance.
Every comparison is a matched discrete computation, held to a rounding budget fixed
here before the comparison (SEP 25).
"""

import dataclasses
import importlib.util

import numpy as np
import pytest

from backends.planar_cases import planar_case
from bounds import assert_within, gamma
from solprop.backends import select
from solprop.backends.spatial import fluence_budget, signed_fluence_widths
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController, RungeKutta

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

BACKENDS = ["numpy_ref", "numba_cpu"] + (
    ["native"] if importlib.util.find_spec("solprop_native") is not None else []
)

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


def _grid():
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)


def _pulse(waist, energy=3e-9):
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy),
        waist,
        jones=(1, 0.5j),
    )


def _settings(names, **extra):
    return SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2.5e-4,
        dz_init=2.5e-4,
        absorber=None,
        integrator=RungeKutta(),
        stats=StatsSpec(names=names, **extra),
    )


def _radial(medium):
    grid = _grid()
    space = RadialGrid(6e-4, 16)
    simulation = Simulation.from_radial(
        grid, space, medium, [], _pulse(60e-6), settings=_settings(LOCAL)
    )
    return grid, simulation.translate()


def _free(space):
    grid = _grid()
    names = LOCAL + FAR
    simulation = Simulation.from_free(
        grid,
        space,
        LOSSY,
        [],
        _pulse(60e-6, 3e-9 if isinstance(space, FreeGrid) else 1e-3),
        settings=_settings(names, apertures=HOLES, spectral_regions=HALVES),
    )
    return grid, simulation.translate()


def _planar():
    grid, _, problem = planar_case("kerr", steps=0)
    return grid, dataclasses.replace(
        problem, stats_spec=StatsSpec(names=LOCAL[:1] + LOCAL[2:4])
    )


CASES = {
    # name: (builder, position of the plane, m)
    "radial-lossy": (lambda: _radial(LOSSY), 0.0),
    "radial-gradient": (
        lambda: _radial(SpatialMedium("Ar", pressure=Linear(0.5, 3.0, 1e-3))),
        6e-4,
    ),
    "cartesian-lossy": (lambda: _free(FreeGrid(3e-4, 24, 3e-4, 16)), 0.0),
    "line-lossy": (lambda: _free(LineGrid(4e-4, 24)), 0.0),
    "planar": (_planar, 0.0),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def measured(request):
    """Every backend's statistics of one state at one plane."""
    build, z = CASES[request.param]
    grid, problem = build()
    rng = np.random.default_rng(34)
    field = problem.initial.field0
    # A chirp-like phase on every column makes the frequencies and the columns
    # interfere at the nodes, so the closure's cross terms enter every local value.
    field = field * np.exp(1j * rng.uniform(-0.3, 0.3, field.shape[-1]))
    values = {}
    for name in BACKENDS:
        backend = select(name, problem, grid=grid)
        try:
            backend.set_initial(field, None, z=z)
            values[name] = backend.stats(problem.stats_spec)
        finally:
            backend.close()
    return request.param, grid, problem, values


def _budget(name, grid, problem):
    """The rounding budget of one statistic, relative to its largest value."""
    n_columns = problem.shape[1]
    n_pol = problem.shape[2]
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


def test_every_backend_measures_the_same_flux(measured):
    case, grid, problem, values = measured
    reference = values["numpy_ref"]
    assert set(reference) == set(problem.stats_spec.names)
    for backend in BACKENDS[1:]:
        for name in problem.stats_spec.names:
            assert_within(
                values[backend][name],
                reference[name],
                _budget(name, grid, problem),
                what=f"{case} {backend} {name}",
            )



def test_a_width_refuses_a_fluence_negative_beyond_its_normwise_budget():
    """Rounding in a beam's wings is read as zero; a real negative lobe is refused."""
    _, problem = _free(LineGrid(4e-4, 24))
    geometry = problem.transform.geometry
    x = geometry.axes[0]
    fluence = np.exp(-((x / 6e-5) ** 2))
    magnitude = np.abs(fluence)
    budget = fluence_budget(magnitude, 1000)
    # Synthesis rounding is normwise: a wing value below zero by far more than its
    # own size, but within the budget of the peak, is rounding.
    wing = fluence.copy()
    wing[0] = -0.5 * budget[0]
    assert np.isfinite(signed_fluence_widths(geometry, wing, budget, 1.0)[0])
    lobe = fluence.copy()
    lobe[0] = -1e-6
    with pytest.raises(BackendError, match="nonnegative fluence profile"):
        signed_fluence_widths(geometry, lobe, budget, 1.0)

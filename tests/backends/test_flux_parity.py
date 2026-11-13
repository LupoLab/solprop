# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend parity of the axial-flux statistics (SEP 34 sections 4.2 and 5.1).

The same state, loaded at the same plane, is measured by ``numpy_ref`` and, when it is
installed, the native CPU backend: energy, the beam,
aperture and region spectra, the aperture's on-axis spectrum, ``absorbed_power``,
the on-axis spectrum and intensity, ``peak_intensity`` and the fluence widths. The
media are lossy (or the planar walls are), so the admittance is complex and the
local statistics' cross terms ``Im(eta)`` are far above rounding; the radial argon
gradient reads the plane measure's material table at a position past the entrance.
Every comparison is a matched discrete computation, held to a rounding budget fixed
here before the comparison (SEP 25). The comparisons are the catalogue's rows
(``backends/geometry_rows.py``), which build these cases and read these budgets; what
stays here is a width's refusal and native's ledger.
"""

import importlib.util

import numpy as np
import pytest

from backends.planar_cases import planar_case
from bounds import gamma
from sessions.runs import launch_statistics
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
from solprop.translate.routes import flux_of

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

BACKENDS = ["numpy_ref"] + (
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
    simulation = Simulation.from_free(
        grid, space, medium, [], _pulse(60e-6), settings=_settings(LOCAL), noise=None
    )
    return grid, simulation.translate(), simulation._run_request()


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
        noise=None,
    )
    return grid, simulation.translate(), simulation._run_request()


def _planar():
    grid, _, problem, run_request = planar_case("kerr", steps=0)
    return grid, problem, run_request.for_run(stats=StatsSpec(names=LOCAL))


CASES = {
    # name: (builder, position of the plane, m); the state is the launch with a
    # chirp-like phase on every column, so that the frequencies and the columns
    # interfere at the nodes and the closure's cross terms enter every local value.
    "radial-lossy": (lambda: _radial(LOSSY), 0.0),
    "radial-gradient": (
        lambda: _radial(SpatialMedium("Ar", pressure=Linear(0.5, 3.0, 1e-3))),
        6e-4,
    ),
    "cartesian-lossy": (lambda: _free(FreeGrid(3e-4, 24, 3e-4, 16)), 0.0),
    "line-lossy": (lambda: _free(LineGrid(4e-4, 24)), 0.0),
    "planar": (_planar, 0.0),
}


def _budget(name, grid, problem):
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


def test_a_width_refuses_a_fluence_negative_beyond_its_normwise_budget():
    """Rounding in a beam's wings is read as zero; a real negative lobe is refused."""
    _, problem, _ = _free(LineGrid(4e-4, 24))
    geometry = problem.transform.geometry
    assert geometry is not None
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


@pytest.mark.skipif("native" not in BACKENDS, reason="solprop-native is not installed")
def test_the_native_closure_buffers_are_in_the_ledger():
    """The closure's transverse volume, the radial plane measure and its table."""
    grid, problem, run_request = _free(FreeGrid(3e-4, 24, 3e-4, 16))
    # One state's bytes beside the transverse scratch, sized for the problem's own
    # local statistics.
    closure = 16 * int(np.prod(problem.shape))
    owner = select("native", problem, run_request, grid=grid)
    try:
        ledger = owner.memory_ledger()
        assert ledger["flux_closure_transverse"] == closure
        # A separable run's plane measure is its operator's own views.
        assert ledger["flux_coefficients"] == ledger["flux_table"] == 0
        # Its fluence takes no time transform, so it holds no magnitude per sample.
        assert ledger["statistics_flux_magnitude"] == 0
    finally:
        owner.close()
    # Without a local statistic nothing is held until a call first needs it.
    bare = run_request.for_run(stats=StatsSpec(names=("energy",)))
    owner = select("native", problem, bare, grid=grid)
    try:
        assert owner.memory_ledger()["flux_closure_transverse"] == 0
        launch_statistics(
            owner,
            problem,
            field=problem.initial.field0,
            stats=StatsSpec(names=("peak_intensity",)),
        )
        assert owner.memory_ledger()["flux_closure_transverse"] == closure
    finally:
        owner.close()
    grid, radial, radial_request = _radial(
        SpatialMedium("Ar", pressure=Linear(0.5, 3.0, 1e-3))
    )
    flux = flux_of(radial.route)
    assert flux is not None and flux.material is not None
    owner = select("native", radial, radial_request, grid=grid)
    try:
        ledger = owner.memory_ledger()
        # The entrance squares (two complex rows of n_omega, two real of n_modes)
        # and the material table's coefficients and breakpoints.
        n_omega, n_modes = radial.shape[2], radial.shape[0]
        assert ledger["flux_coefficients"] == 32 * n_omega + 16 * n_modes
        assert ledger["flux_table"] == (
            flux.material.coefficients.size * 16 + flux.material.breakpoints.size * 8
        )
        assert ledger["flux_closure_transverse"] == 0
        assert ledger["statistics_flux_magnitude"] > 0
    finally:
        owner.close()

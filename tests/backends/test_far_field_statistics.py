# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The far-field statistics in a run: the operator's two, the doors and the host path.

SEP 23 section 7, items 8 (the axial flux) and 9 (the absorbed power) on
``numpy_ref``, the backend-level refusals of item 12, the solprop-side admission that
refuses the native backend until its native pass, the native statistics of a launch
beside the reference's, and the slots and bands every pass shares.
"""

import dataclasses
import importlib.util
import math
from contextlib import ExitStack

import numpy as np
import pytest
from scipy import integrate

from backends import test_batch_lockstep as lockstep
from backends.planar_cases import planar_case
from bounds import EPS, assert_within, gamma
from legacy_geometry import bare_absorber
from solprop.backends import registry, select
from solprop.backends import spatial as backend_spatial
from solprop.backends.host.group import HostGroup
from solprop.backends.registry import statistics_refusal
from solprop.backends.session import Start
from solprop.backends.spatial import (
    FarFieldLayout,
    FarFieldTables,
    collect_spatial,
    far_field_bands,
    far_field_layout,
    far_field_partials,
)
from solprop.boundaries import SpatialAbsorber, absorber, custom_absorber
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.run import run_problem
from solprop.exceptions import (
    BackendError,
    DriverError,
    OutputError,
    TranslationError,
)
from solprop.fourier import energy_weights, spectral_density
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.output import MemoryOutput
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import ABSORBED_CHANNELS, FAR_FIELD_STATISTICS, SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController, RungeKutta
from solprop.translate.spatial import axial_admittance, separable_values

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

ALL_SIX = FAR_FIELD_STATISTICS

HOLES = ApertureSet(
    centres=[[0.0, 0.0], [1e-3, 0.0]],
    diameters=[2e-3, 1e-3],
    distances=[0.25, 0.25],
    edges=[2e-4, 1e-4],
    profile="tanh",
)
QUADRANTS = np.array(
    [
        [-np.inf, 0.0, -np.inf, 0.0],
        [0.0, np.inf, -np.inf, 0.0],
        [-np.inf, 0.0, 0.0, np.inf],
        [0.0, np.inf, 0.0, np.inf],
    ]
)


def far_spec(names=ALL_SIX, **options):
    return StatsSpec(
        names=tuple(names), apertures=HOLES, spectral_regions=QUADRANTS, **options
    )


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 900e-9), 100e-15, thg=False)


def free_simulation(grid, space, launch, *, medium=None, spec=None, **settings):
    options = {
        "controller": PIController(rtol=1e-6, atol=1e-14),
        "save_planes": [1e-4],
        "dz_max": 1e-4,
        "dz_init": 1e-4,
        # No collar unless declared, as an undeclared absorber meant before SEP 21.
        "absorber": bare_absorber(grid),
        "backend": "numpy_ref",
        "integrator": RungeKutta(),
        "stats": far_spec() if spec is None else spec,
    }
    options.update(settings)
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium() if medium is None else medium,
        [],
        launch,
        settings=SimulationSettings(**options),
        noise=None,
    )


def entrance(simulation, backend="numpy_ref", spec=None):
    """The statistics of the launch on a backend, and the problem they belong to."""
    problem = simulation.translate()
    owner = select(backend, problem, grid=simulation.grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        values = owner.stats(problem.stats_spec if spec is None else spec)
    finally:
        owner.close()
    return values, problem


def gaussian(waist, *, energy=1e-6):
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), waist
    )


# --------------------------------------------------------- the launch on both backends


def test_native_forms_the_far_field_statistics_as_the_reference_does(grid):
    if importlib.util.find_spec("solprop_native") is None:
        pytest.skip("solprop-native is not installed")
    space = FreeGrid(3e-4, 24, 3e-4, 16, collar_fraction=0.1)
    simulation = free_simulation(
        grid,
        space,
        gaussian(60e-6),
        medium=SpatialMedium(1 + 1e-6j),
        absorber=absorber(grid),
    )
    values = {name: entrance(simulation, name)[0] for name in ("numpy_ref", "native")}
    assert set(values["native"]) == set(ALL_SIX)
    for name in ALL_SIX:
        # One pass over the same state on both: the summation bound.
        assert_within(
            values["native"][name],
            values["numpy_ref"][name],
            gamma(4 * (space.n_columns + 16)),
            what=name,
        )
        assert values["numpy_ref"][name].shape[0] == 1  # the batch axis


# ------------------------------------------------------------- the layout and bands


def test_the_layout_takes_only_the_present_slots():
    layout = far_field_layout(
        (*ALL_SIX, "on_axis_spectrum"), n_apertures=3, n_regions=4, n_pol=2
    )
    # Each amplitude holds its electric sum and its magnetic closure, 4 n_pol slots.
    assert layout == FarFieldLayout(
        beam=0,
        absorbed=1,
        regions=6,
        apertures=10,
        stride=9,
        collected=0,
        amplitude=1,
        axis=37,
        n_slots=45,
    )
    assert layout is not None
    assert layout.array().tolist() == [0, 1, 6, 10, 9, 0, 1, 37, 45]
    alone = far_field_layout(
        ("aperture_on_axis_spectrum",), n_apertures=2, n_regions=0, n_pol=1
    )
    assert alone is not None
    assert (alone.apertures, alone.stride, alone.collected, alone.amplitude) == (
        0,
        4,
        -1,
        0,
    )
    assert far_field_layout(("energy",), n_apertures=0, n_regions=0, n_pol=1) is None
    # A region request without regions takes no slot.
    assert (
        far_field_layout(("region_spectrum",), n_apertures=0, n_regions=0, n_pol=1)
        is None
    )


def test_the_bands_follow_the_shapes_and_the_budget(monkeypatch):
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 5, 10, 15, 20, 25, 30, 35]
    assert far_field_bands(10, 1, 3, 10).tolist() == [0, 10]  # one block of a line
    # A budget of two bands over seven rows: three and four rows.
    monkeypatch.setattr(backend_spatial, "FAR_FIELD_BAND_BUDGET", 2 * 8 * 3 * 10)
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 15, 35]
    # A budget below one band's partials still takes one band.
    monkeypatch.setattr(backend_spatial, "FAR_FIELD_BAND_BUDGET", 1)
    assert far_field_bands(5, 7, 3, 10).tolist() == [0, 35]


def test_a_radial_problem_takes_no_partials(grid):
    radial = Simulation.from_radial(
        grid,
        RadialGrid(3e-4, 8),
        SpatialMedium(),
        [],
        gaussian(1e-4, energy=1e-9),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-4,
            absorber=None,
            backend="numpy_ref",
        ),
        noise=None,
    ).translate()
    assert far_field_partials(radial) == (0, 0)


# -------------------------------------------------------------- item 8: axial flux


def field_spectrum(problem, grid):
    """The beam's spectrum with no admittance: ``eps0 c/2 sum_c w_c sd(E_c)``."""
    field = problem.initial.field0[0]
    weights = problem.transform.geometry.spectral_weights
    density = np.sum(spectral_density(field, grid.dt, kind=grid.kind), axis=1)
    return 0.5 * EPS0 * C * np.sum(weights[:, None] * density, axis=0)


def paraxial_obliquity(k_waist):
    """The mean of sqrt(1 - u), u exponential of mean 2 / (k w0)^2, cut at u = 1."""
    mean = 2 / k_waist**2
    value, _ = integrate.quad(
        lambda u: math.sqrt(1 - u) * math.exp(-u / mean) / mean,
        0,
        1,
        epsabs=0,
        epsrel=1e-13,
        limit=200,
    )
    return value


def test_the_axial_flux_of_a_vacuum_gaussian_is_its_mean_obliquity(grid):
    # k w0 = 20 at 800 nm: 1 - O = 2.5e-3, far from paraxial and far above rounding.
    waist = 20 * 800e-9 / (2 * np.pi)
    # Twelve waists sampled at a quarter waist: truncation exp(-36), aliasing
    # exp(-79), and no light within exp(-160) of the cut-off.
    space = FreeGrid(6 * waist, 48, 6 * waist, 48, collar_fraction=0)
    simulation = free_simulation(
        grid,
        space,
        gaussian(waist),
        spec=StatsSpec(names=("beam_spectrum",)),
    )
    values, problem = entrance(simulation)
    band = np.flatnonzero(grid.inband_mask)
    # The beam spectrum is the axial flux (SEP 34): the field's spectrum weighted
    # per column by the real admittance, here the vacuum's obliquity.
    ratio = values["beam_spectrum"][0, band] / field_spectrum(problem, grid)[band]
    expected = np.array([paraxial_obliquity(grid.omega[k] / C * waist) for k in band])
    assert np.all(ratio < 1)
    # The lattice sum of a smooth integrand (aliasing below exp(-70)), quad 1e-13 and
    # the rounding of two sums and a root per column: 1e-10 with a margin.
    assert np.all(np.abs(ratio - expected) < 1e-10)
    assert np.min(1 - expected) > 1e-3


def test_the_axial_flux_in_a_gas_is_its_index(grid):
    waist = 100e-6
    medium = SpatialMedium("Ar", pressure=5.0)
    space = FreeGrid(5e-4, 32, 5e-4, 32, collar_fraction=0)
    simulation = free_simulation(
        grid,
        space,
        gaussian(waist),
        medium=medium,
        spec=StatsSpec(names=("beam_spectrum",)),
    )
    values, problem = entrance(simulation)
    band = np.flatnonzero(grid.inband_mask)
    ratio = values["beam_spectrum"][0, band] / field_spectrum(problem, grid)[band]
    index = np.real(medium.index(grid.omega[band], 0.0))
    # O = sqrt(n^2 - u) averages to n - mean(u)/(2n): within the obliquity term
    # 2/(k w0)^2 of Re n, and below it.
    obliquity = 2 / (grid.omega[band] / C * waist) ** 2
    assert np.all(ratio < index)
    assert np.all(index - ratio < obliquity)
    assert np.min(index) - 1 > 100 * np.max(obliquity)  # the index, not the obliquity


# ---------------------------------------------------------- item 9: absorbed power


@pytest.fixture(scope="module")
def lossy_launch(grid):
    # Sixteen columns either way at 0.27 um: the outer columns are evanescent at
    # every band frequency, so every channel carries power.
    space = FreeGrid(2e-6, 16, 2e-6, 16, collar_fraction=0)
    rng = np.random.default_rng(29)
    shape = (space.n_columns, 1, grid.n_omega)
    field = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e3
    return space, field * grid.active_mask


def channel_powers(problem, grid, field):
    """Each channel's -dU/dz from the operator's coefficients, recomputed here."""
    data = problem.linear.separable
    square = (data.k_squared[None, :] - data.kperp_sq[:, None]) + (
        data.k_squared_low[None, :] - data.kperp_sq_low[:, None]
    )
    root = np.sqrt(square)
    live = data.active[None, :]
    propagating = np.real(square) > 0
    rates = np.stack(
        [
            np.broadcast_to(data.spectral_rate - data.user_rate, root.shape),
            # The transverse collar's rate on the active band (SEP 22, Q6).
            np.where(live, data.kperp_rate[:, None], 0.0),
            np.where(
                live & ~propagating,
                2 * np.minimum(np.abs(np.imag(root)), data.evanescent_rate),
                0.0,
            ),
            np.where(live & propagating, 2 * np.imag(root), 0.0),
            np.broadcast_to(data.user_rate, root.shape),
        ]
    )
    weights = problem.transform.geometry.spectral_weights
    # Each channel's power is measured by the axial flux (SEP 34): the real
    # admittance, zero on evanescent columns, weighs every bin.
    admittance = np.real(axial_admittance(data, grid.omega, slice(None)))
    measure = (
        weights[:, None]
        * admittance
        * np.abs(field[:, 0]) ** 2
        * energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    )
    terms = 0.5 * EPS0 * C * rates * measure[None]
    return np.sum(terms, axis=(1, 2)), np.sum(np.abs(terms), axis=(1, 2))


@pytest.mark.parametrize(
    "collar", [False, SpatialAbsorber(real_space=False, safety_fraction=0.3)]
)
@pytest.mark.parametrize("loss", [1e-5, -1e-5])
def test_each_absorbed_channel_is_its_share_of_the_linear_decay(
    grid, lossy_launch, loss, collar
):
    space, field = lossy_launch
    simulation = free_simulation(
        grid,
        space,
        field,
        medium=SpatialMedium(1 + 1j * loss),
        absorber=absorber(grid, temporal=False),
        spec=StatsSpec(names=("energy", "absorbed_power")),
        spatial_absorber=collar,
    )
    values, problem = entrance(simulation)
    power = values["absorbed_power"][0]
    expected, magnitude = channel_powers(problem, grid, problem.initial.field0[0])
    # Each channel's power is a sum of n terms with their roundings: 2 gamma_n of
    # the sum of their magnitudes; without a collar the transverse channel is an
    # exact zero.
    if collar is False:
        assert power[1] == 0.0
    else:
        assert power[1] > 0
    # An evanescent column carries no axial flux, so its channel removes none.
    assert power[2] == 0.0
    for channel in (0, 1, 3) if collar is not False else (0, 3):
        assert_within(
            power[channel],
            expected[channel],
            2
            * gamma(power.size * problem.shape[1] * grid.n_omega + 16)
            * magnitude[channel]
            / abs(expected[channel]),
        )
    assert np.sign(power[3]) == np.sign(loss)  # loss, or gain with the sign reversed
    assert power[0] > 0
    # Their sum is the whole real part of the stages' own operator.
    data, geometry = problem.linear.separable, problem.transform.geometry
    assert data is not None and geometry is not None
    operator, _ = separable_values(data)
    total = (
        0.5
        * EPS0
        * C
        * np.sum(
            geometry.spectral_weights[:, None, None]
            * np.real(axial_admittance(data, grid.omega, slice(None)))[:, None]
            * (-2 * np.real(operator[0]))
            * np.abs(problem.initial.field0[0]) ** 2
            * energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
        )
    )
    assert_within(
        np.sum(power),
        total,
        2 * gamma(4 * field.size + 16) * np.sum(magnitude) / abs(total),
    )


def test_gain_reverses_the_material_channel(grid, lossy_launch):
    space, field = lossy_launch
    powers = {}
    for loss in (2e-5, -2e-5):
        simulation = free_simulation(
            grid,
            space,
            field,
            medium=SpatialMedium(1 + 1j * loss),
            spec=StatsSpec(names=("absorbed_power",)),
        )
        powers[loss] = entrance(simulation)[0]["absorbed_power"][0]
    # sqrt(conj q) = conj sqrt(q): the material rate flips sign, to rounding.
    assert_within(powers[-2e-5][3], -powers[2e-5][3], 4 * gamma(field.size + 16))
    # No flux on the evanescent columns and no spectral absorber.
    assert powers[2e-5][2] == powers[-2e-5][2] == 0.0
    assert powers[2e-5][0] == powers[-2e-5][0] == 0.0


def test_the_absorbed_power_predicts_the_energy_step_at_its_orders(grid, lossy_launch):
    space, field = lossy_launch
    medium = SpatialMedium(1 + 1e-5j)
    boundary = absorber(grid, temporal=False)
    probe = free_simulation(grid, space, field, medium=medium, absorber=boundary)
    data = probe.translate().linear.separable
    assert data is not None
    fastest = max(float(np.max(data.spectral_rate)), 2 * data.evanescent_rate)
    errors = []
    steps = [0.02 / fastest / 2**level for level in range(3)]
    for step in steps:
        simulation = free_simulation(
            grid,
            space,
            field,
            medium=medium,
            absorber=boundary,
            save_planes=[step],
            dz_max=step,
            dz_init=step,
            # The subject is the absorbed power; the random launch fills the window.
            boundary_check=None,
            spec=StatsSpec(
                names=("energy", "absorbed_power"),
                periodic=("absorbed_power",),
                period=1,
            ),
        )
        output = simulation.run(progress="none")
        energy = np.asarray(output["stats/energy"])
        power = np.sum(np.asarray(output["stats/absorbed_power"]), axis=-1)
        rows = np.asarray(output["step_stats/absorbed_power"])
        # The row taken after the plane's accepted step is that plane's own number.
        assert_within(np.sum(rows[-1]), power[-1], gamma(4 * field.size + 16))
        explicit = energy[1] - energy[0] + step * power[0]
        trapezoid = energy[1] - energy[0] + step * (power[0] + power[1]) / 2
        errors.append((explicit / step, trapezoid / step))
    errors = np.abs(np.asarray(errors))
    # Halving h: the rectangle rule's error per unit length falls at first order and
    # the trapezoid's at second (testing.md section 1's three levels and gate).
    for column, order in ((0, 1), (1, 2)):
        observed = np.log2(errors[:-1, column] / errors[1:, column])
        assert np.all(np.abs(observed - order) < 0.15), (column, observed)


# ------------------------------------------------- the custom spectral rate (#231)


def split_rates(problem, grid):
    """Each channel's rate on every column and bin, and the total, recomputed here.

    Everything acting on this linear beam is a power rate applied as ``-alpha/2``, and
    diagonal in the stored transverse spectrum, so a bin's power decays at the sum of
    the rates inside it: the collar's, the user's and the material's.
    """
    data = problem.linear.separable
    square = (data.k_squared[None, :] - data.kperp_sq[:, None]) + (
        data.k_squared_low[None, :] - data.kperp_sq_low[:, None]
    )
    root = np.sqrt(square)
    live = data.active[None, :] & (np.real(square) > 0)
    collar = np.broadcast_to(data.spectral_rate - data.user_rate, root.shape)
    user = np.broadcast_to(data.user_rate, root.shape)
    material = np.where(live, 2 * np.imag(root), 0.0)
    measure = (
        0.5
        * EPS0
        * C
        * problem.transform.geometry.spectral_weights[:, None]
        * np.real(axial_admittance(data, grid.omega, slice(None)))
        * np.sum(np.abs(problem.initial.field0[0]) ** 2, axis=1)
        * energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    )
    return collar, user, material, measure


def exact_shares(rate, total, measure, length):
    """A channel's energy over `length`: ``a_j/R (1 - exp(-R L))``, per bin."""
    fraction = np.divide(
        rate * -np.expm1(-total * length),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return float(np.sum(measure * fraction))


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_the_collar_and_the_custom_rate_integrate_to_exact_exponential_weights(
    backend, grid, lossy_launch
):
    # On a linear beam the state is exact to rounding: the propagator is the operator's
    # own exponential and there is no source, so the only error here is the z quadrature
    # of the statistic, an instantaneous J/m sampled at the saved planes.
    if backend == "native":
        pytest.importorskip("solprop_native")
    space, field = lossy_launch
    collar = absorber(grid, temporal=False)
    # A band across the middle of the spectrum, of the collar's own strength.
    centre, width = grid.omega0, 0.5 * np.ptp(grid.omega)
    strength = 0.5 * float(np.max(collar.spectral_rate))
    boundary = custom_absorber(
        grid,
        lambda omega: strength * np.exp(-(((omega - centre) / width) ** 2)),
        collar=collar,
    )
    assert boundary.user_rate is not None
    assert np.any(boundary.user_rate > 0)
    assert np.any(boundary.spectral_rate - boundary.user_rate > 0)
    fastest = float(np.max(boundary.spectral_rate))
    planes = 2**5
    length = 2.0 / fastest  # the fastest bin falls by e^-2, so the sum is well resolved
    planes_z = np.linspace(length / planes, length, planes)
    simulation = free_simulation(
        grid,
        space,
        field,
        medium=SpatialMedium(1 + 1e-6j),
        absorber=boundary,
        save_planes=planes_z,
        dz_max=length / planes,
        dz_init=length / planes,
        backend=backend,
        boundary_check=None,
        spec=StatsSpec(names=("energy", "absorbed_power")),
    )
    output = simulation.run(progress="none")
    powers = np.asarray(output["stats/absorbed_power"])
    z = np.asarray(output["z"])
    assert powers.shape == (planes + 1, 5)
    problem = simulation.translate()
    collar_rate, user_rate, material, measure = split_rates(problem, grid)
    total = collar_rate + user_rate + material
    assert ABSORBED_CHANNELS[0] == "spectral" and ABSORBED_CHANNELS[4] == "custom"
    h = length / planes
    for channel, rate in ((0, collar_rate), (4, user_rate), (3, material)):
        integral = integrate.simpson(powers[:, channel], x=z)
        expected = exact_shares(rate, total, measure, length)
        assert expected > 0
        # Composite Simpson on P_j(z) = sum m r_j exp(-R z): L h^4 / 180 sum m r_j R^4,
        # plus the state's rounding over the planes, a few eps of the channel per plane.
        simpson = length * h**4 / 180 * float(np.sum(measure * rate * total**4))
        rounding = 64 * planes * EPS * expected
        assert abs(integral - expected) <= simpson + rounding, channel
    # The two parts of the spectral rate partition it, plane by plane.
    assert_within(
        powers[0, 0] + powers[0, 4],
        float(np.sum(measure * (collar_rate + user_rate))),
        4 * gamma(field.size + 16),
    )
    # -sum P_j stays dE/dz: the drop of the energy statistic, by the same quadrature.
    energy = np.asarray(output["stats/energy"]).reshape(planes + 1, -1).sum(axis=1)
    drop = energy[0] - energy[-1]
    simpson = length * h**4 / 180 * float(np.sum(measure * total * total**4))
    assert abs(integrate.simpson(np.sum(powers, axis=1), x=z) - drop) <= (
        simpson + 64 * planes * EPS * drop
    )


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_a_run_without_a_custom_rate_reports_zero_and_moves_no_bit(
    backend, grid, lossy_launch
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    space, field = lossy_launch
    simulation = free_simulation(
        grid,
        space,
        field,
        absorber=absorber(grid, temporal=False),
        spec=StatsSpec(names=("absorbed_power",)),
    )
    values, problem = entrance(simulation, backend)
    power = values["absorbed_power"][0]
    assert power.shape == (5,) and power[4] == 0.0 and power[0] > 0
    data = problem.linear.separable
    assert data is not None and not np.any(data.user_rate)
    # A zero share leaves channel 0 the total rate's sum, on the one rate read before.
    expected, magnitude = channel_powers(problem, grid, problem.initial.field0[0])
    assert_within(
        power[0],
        expected[0],
        2 * gamma(field.size + 16) * magnitude[0] / abs(expected[0]),
    )


def test_a_custom_rate_moves_no_bit_of_the_operator(grid, lossy_launch):
    # The stages read the total, and only the statistic reads the share.
    space, field = lossy_launch
    boundary = custom_absorber(grid, 1e3, collar=absorber(grid, temporal=False))
    assert boundary.user_rate is not None
    lumped = dataclasses.replace(boundary, user_rate=None)
    split = free_simulation(grid, space, field, absorber=boundary).translate()
    plain = free_simulation(grid, space, field, absorber=lumped).translate()
    split_data, plain_data = split.linear.separable, plain.linear.separable
    assert split_data is not None
    assert plain_data is not None
    np.testing.assert_array_equal(split_data.spectral_rate, plain_data.spectral_rate)
    np.testing.assert_array_equal(
        separable_values(split_data)[0], separable_values(plain_data)[0]
    )
    np.testing.assert_array_equal(split_data.user_rate, boundary.user_rate)
    assert not np.any(plain_data.user_rate)


def test_the_operator_refuses_a_share_above_the_total(grid, lossy_launch):
    space, field = lossy_launch
    data = free_simulation(grid, space, field).translate().linear.separable
    assert data is not None
    with pytest.raises(TranslationError, match="user's share of spectral_rate"):
        dataclasses.replace(data, user_rate=np.ones(grid.n_omega))
    with pytest.raises(TranslationError, match="user's share of spectral_rate"):
        dataclasses.replace(data, user_rate=np.zeros(3))


# ------------------------------------------------------------- doors and refusals


# The planar fixture's family warns about its own collar and fill, as in the planar
# backend tests; neither is what these refusals are about.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
def test_radial_and_planar_geometry_are_refused_before_any_step(grid):
    spec = StatsSpec(names=("energy", "beam_spectrum"))
    radial = Simulation.from_radial(
        grid,
        RadialGrid(3e-4, 16),
        SpatialMedium(),
        [],
        gaussian(60e-6),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-4,
            backend="numpy_ref",
            stats=spec,
        ),
        noise=None,
    )
    with pytest.raises(OutputError, match="radial grid"):
        select("numpy_ref", radial.translate(), grid=grid)
    planar_grid, _, problem = planar_case("kerr", steps=0)
    problem = dataclasses.replace(problem, stats_spec=spec)
    native = importlib.util.find_spec("solprop_native") is not None
    for name in ("numpy_ref", "native") if native else ("numpy_ref",):
        with pytest.raises(OutputError, match="hybrid planar"):
            select(name, problem, grid=planar_grid)


def test_an_older_native_build_is_refused_the_far_field_statistics_by_name(
    grid, monkeypatch
):
    space = FreeGrid(3e-4, 8, 3e-4, 8, collar_fraction=0.1)
    simulation = free_simulation(grid, space, gaussian(60e-6))
    problem = simulation.translate()
    # A build that declares none of them, as every build before SEP 23's third phase:
    # the refusal precedes the native build and names the backends that form them.
    monkeypatch.setattr(registry, "_plugin_far_field", lambda name: frozenset())
    with pytest.raises(BackendError, match="or run them on numpy_ref"):
        select("native", problem, grid=grid)
    assert statistics_refusal("native", ["energy"]) is None
    assert statistics_refusal("numpy_ref", list(ALL_SIX)) is None
    # A build that declares some of them is refused the others, by name.
    monkeypatch.setattr(
        registry, "_plugin_far_field", lambda name: frozenset({"beam_spectrum"})
    )
    refusal = statistics_refusal("native", ["beam_spectrum", "region_spectrum"])
    assert refusal is not None
    assert "['region_spectrum']" in refusal
    assert statistics_refusal("native", ["beam_spectrum"]) is None
    monkeypatch.setattr(registry, "_plugin_far_field", lambda name: frozenset())

    plain = free_simulation(
        grid, space, gaussian(60e-6), spec=StatsSpec(names=("energy",))
    )
    problem = plain.translate()
    owner = select("numpy_ref", problem, grid=grid)
    try:
        monkeypatch.setattr(owner, "name", "native")
        # The run loop's door, for a spec handed to it rather than the problem's own.
        with pytest.raises(DriverError, match="far-field statistics"):
            run_problem(
                problem,
                owner,
                MemoryOutput(),
                start=Start.entrance(problem),
                metadata=plain.metadata(),
                stats_spec=far_spec(),
            )
    finally:
        owner.close()


def test_the_plugin_probe_reads_what_an_installed_build_declares(monkeypatch):
    # Not installed: no refusal here, and selecting it reports the missing package.
    monkeypatch.setattr(registry, "_satisfied", lambda entry: False)
    assert registry._plugin_far_field("native") is None
    assert statistics_refusal("native", ["beam_spectrum"]) is None
    assert registry._plugin_far_field("numpy_ref") is None
    monkeypatch.undo()
    if importlib.util.find_spec("solprop_native") is None:
        pytest.skip("solprop-native is not installed")
    # The optional plugin is imported only after its availability check.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    declared = registry._plugin_far_field("native")
    assert declared == frozenset(getattr(NativeBackend, "far_field_statistics", ()))
    # A build that predates the attribute declares none.
    monkeypatch.delattr(NativeBackend, "far_field_statistics", raising=False)
    assert registry._plugin_far_field("native") == frozenset()
    assert statistics_refusal("native", list(ALL_SIX)) is not None


def test_the_host_pass_refuses_what_it_was_not_given(grid):
    space = FreeGrid(3e-4, 8, 3e-4, 8, collar_fraction=0.1)
    simulation = free_simulation(grid, space, gaussian(60e-6))
    problem = simulation.translate()
    field = problem.initial.field0[0]
    with pytest.raises(BackendError, match="tables"):
        collect_spatial(
            far_spec(), grid, problem.transform, field, z=0.0, flux=problem.flux
        )
    tables = FarFieldTables(problem).for_spec(far_spec())
    # The saved-field names need no operator.
    beam = collect_spatial(
        far_spec(("beam_spectrum",)),
        grid,
        problem.transform,
        field,
        z=0.0,
        flux=problem.flux,
        tables=tables,
    )
    assert beam["beam_spectrum"].shape == (grid.n_omega,)
    with pytest.raises(BackendError, match="separable linear operator"):
        collect_spatial(
            far_spec(("absorbed_power",)),
            grid,
            problem.transform,
            field,
            z=0.0,
            flux=problem.flux,
            tables=tables,
        )


def test_the_tables_are_formed_once_and_reused(grid):
    space = FreeGrid(3e-4, 8, 3e-4, 8, collar_fraction=0.1)
    problem = free_simulation(grid, space, gaussian(60e-6)).translate()
    held = FarFieldTables(problem)
    own = held.for_spec(problem.stats_spec)
    assert own is not None
    # The periodic channel and the reference backend's filtered specs carry the
    # problem's own tables (a structural identity, not a recomputation).
    assert (
        held.for_spec(dataclasses.replace(problem.stats_spec, names=("beam_spectrum",)))
        is own
    )
    other = StatsSpec(
        names=("aperture_spectrum",),
        apertures=ApertureSet(centres=[[0.0, 0.0]], diameters=[1e-3], distances=[0.5]),
    )
    fresh = held.for_spec(other)
    assert fresh is not None and fresh is not own
    assert np.array_equal(fresh.distances, [0.5])
    assert held.for_spec(StatsSpec(names=("energy",))) is None
    modal = lockstep.simulation(0, "numpy_ref").translate()
    with pytest.raises(OutputError, match="without spatial geometry"):
        FarFieldTables(modal).for_spec(StatsSpec(names=("beam_spectrum",)))


@pytest.mark.parametrize("table", ["apertures", "regions"])
def test_lockstep_members_must_carry_equal_tables(table):
    with ExitStack() as cleanup:
        first, second = [
            lockstep.loaded(lockstep.simulation(index, "numpy_ref"), cleanup)
            for index in range(2)
        ]
        problem = second._problem
        change = (
            {"apertures": HOLES}
            if table == "apertures"
            else {"spectral_regions": QUADRANTS}
        )
        second._problem = dataclasses.replace(
            problem, stats_spec=dataclasses.replace(problem.stats_spec, **change)
        )
        with pytest.raises(BackendError, match="same statistics request"):
            HostGroup([first, second])


def test_a_run_records_every_far_field_statistic_at_every_plane(grid):
    space = FreeGrid(3e-4, 16, 3e-4, 16, collar_fraction=0.1)
    simulation = free_simulation(
        grid,
        space,
        gaussian(60e-6),
        save_planes=[5e-5, 1e-4],
        dz_max=5e-5,
        dz_init=5e-5,
        absorber=absorber(grid),
        spec=far_spec((*ALL_SIX, "energy"), periodic=("absorbed_power",), period=1),
    )
    output = simulation.run(progress="none")
    shapes = {
        "aperture_spectrum": (3, 2, grid.n_omega),
        "aperture_on_axis_spectrum": (3, 2, grid.n_omega),
        "region_spectrum": (3, 4, grid.n_omega),
        "beam_spectrum": (3, grid.n_omega),
        "absorbed_power": (3, 5),
    }
    for name, shape in shapes.items():
        assert np.asarray(output[f"stats/{name}"]).shape == shape, name
    assert np.asarray(output["step_stats/absorbed_power"]).shape[1:] == (5,)
    beam = np.asarray(output["stats/beam_spectrum"])
    # Parseval at every plane: 2 gamma_m of the total.
    assert_within(
        np.sum(beam, axis=-1) * grid.d_omega,
        np.asarray(output["stats/energy"]),
        2 * gamma(space.n_columns * grid.n_omega + 16),
    )


def test_a_launched_tilt_moves_towards_the_aperture_that_collects_it(grid):
    # The closed-form direction tests build the stored spectrum with fft2 by hand;
    # this ties the aperture's physical +x to the way a run propagates a launch.
    waist, angle, length = 60e-6, 20e-3, 1e-3
    space = FreeGrid(3e-4, 64, 3e-4, 32, collar_fraction=0)
    x, y = np.meshgrid(space.x, space.y)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    profile = np.exp(-(x**2 + y**2) / waist**2)
    # The crossing-beam lesson's +x beam: internal phase exp(-i kappa x).
    tilt = np.exp(-1j * x[..., None] * grid.omega * np.sin(angle) / C)
    launch = np.fft.fft2(profile[..., None] * tilt * temporal[0, 0], axes=(0, 1))
    distance = 0.25
    image = distance * np.sin(angle)
    holes = ApertureSet(
        centres=[[image, 0.0], [-image, 0.0]],
        diameters=[image, image],
        distances=[distance, distance],
        labels=("towards +x", "mirror"),
    )
    spec = StatsSpec(names=("beam_spectrum", "aperture_spectrum"), apertures=holes)
    simulation = free_simulation(
        grid,
        space,
        launch.reshape(-1, 1, grid.n_omega),
        spec=spec,
        save_planes=[length],
    )
    output = simulation.run(progress="none")
    assert isinstance(output, MemoryOutput)
    field = list(output.planes())[-1].field
    assert field is not None
    physical = np.fft.ifft2(
        field.reshape(space.n_y, space.n_x, grid.n_omega), axes=(0, 1)
    )
    fluence = np.sum(np.abs(physical) ** 2, axis=-1)
    centroid = np.sum(x * fluence) / np.sum(fluence)
    # The beam walks off by length tan(theta) towards physical +x: 20 um, or 1.6
    # pixels, against a centroid error of the lattice far below a per cent of that.
    assert centroid == pytest.approx(length * math.tan(angle), rel=1e-2)
    band = grid.inband_mask
    beam = np.asarray(output["stats/beam_spectrum"])[-1, band]
    collected = np.asarray(output["stats/aperture_spectrum"])[-1][:, band] / beam
    assert np.all(collected[0] > 0.99)
    assert np.all(collected[1] < 1e-6)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gas temperature along z through the public interface."""

import functools
import warnings
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from march_bounds import FIXED_ORDER, STATISTIC_FACTOR, assert_matched, simulation_bound
from solprop.boundaries import absorber
from solprop.constants import ROOM_TEMPERATURE
from solprop.driver import Simulation
from solprop.driver.simulation import _refuse_gas_keywords
from solprop.exceptions import (
    DriverError,
    MaterialError,
    MaterialRangeWarning,
    RamanTemperatureWarning,
    ToolsError,
)
from solprop.gasflow import steady_fill
from solprop.modes import MarcatiliMode
from solprop.output.schema import serialise
from solprop.physdata import Mixture, species
from solprop.profiles import Constant, Linear, PressureGradient
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta
from solprop.tools import soliton_params

LENGTH = 1e-2
STEPS = 10
RADIUS = 125e-6
WARM = Linear(294.0, 360.0, LENGTH)
PAIR = {"He": 0.5, "Ar": 0.5}
TRIPLE = {"He": 0.4, "Ar": 0.3, "Ne": 0.3}


@pytest.fixture(autouse=True)
def _quiet_gas_index():
    """Silence the gas indices' extrapolation notice, as the capillary tests do."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        yield


def keywords(**overrides):
    """Return a cheap capillary's keywords: laminar, a continuum, no plasma."""
    return {
        "lambda0": 800e-9,
        "wavelength_limits": (200e-9, 2e-6),
        "trange": 100e-15,
        "fwhm": 10e-15,
        "energy": 20e-6,
        "n_r": 8,
        "n_save": 3,
        "rtol": 1e-6,
        # No order choice on the fixed schedule: two runs of one problem are one march.
        "integrator": FIXED_ORDER,
        "plasma": False,
        "stats": StatsSpec(names=("energy", "density", "pressure")),
    } | overrides


def build(gas, temperature, *, radius=RADIUS, pressure=(0.5, 0.0), **overrides):
    """Build the capillary on a fixed ten-step schedule, without running it."""
    simulation = Simulation.from_capillary(
        radius,
        LENGTH,
        gas,
        pressure,
        temperature=temperature,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        **keywords(**overrides),
    )
    # A collar referred to a metre, weak enough for ten explicit millimetre steps
    # (ADR-025), as the tapered-gradient run of tests/driver/test_prop_capillary.py.
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=1.0)
    )
    return simulation


@functools.cache
def run(gas_key, temperature_key, backend):
    """Run one cached case: a gas and a temperature by key, on a backend."""
    gas = {"pair": PAIR, "triple": TRIPLE}[gas_key]
    temperature = {"warm": WARM, "number": 294.0, "constant": Constant(294.0)}[
        temperature_key
    ]
    return build(gas, temperature, backend=backend).run()


def test_a_shared_pair_under_a_temperature_path_becomes_two_populations():
    # A uniform bore: the two populations are the only varying parameters, so the
    # two-parameter expansion carries them (a taper would make three: the table).
    simulation = build(PAIR, WARM)
    problem = simulation.translate()
    assert problem.linear.phi_representation == "expansion"
    expansion = problem.linear.phi_expansion
    assert expansion is not None
    assert sorted(expansion.parameters) == ["density:Ar", "density:He"]
    fill = simulation.profiles["pressure"]
    assert isinstance(fill, PressureGradient) and fill.weight is not None
    expected = steady_fill(
        {species("He"): 0.5, species("Ar"): 0.5},
        [0.0, LENGTH],
        [0.5, 0.0],
        radius=RADIUS,
        temperature=WARM,
    )
    z = np.linspace(0.0, LENGTH, 17)
    np.testing.assert_array_equal(fill(z), expected(z))


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_temperature_path_runs_ten_steps_on_every_backend(backend):
    if backend == "native":
        pytest.importorskip("solprop_native")
    output = run("pair", "warm", backend)
    simulation = build(PAIR, WARM)
    z = np.asarray(output["z"])
    # The recorded statistics are the declared paths at the saved planes: the
    # pressure fill and the total density rho(p(z), T(z)) of both populations.
    np.testing.assert_allclose(
        output["stats/pressure"], simulation.profiles["pressure"](z), rtol=1e-13
    )
    np.testing.assert_allclose(
        output["stats/density"], simulation.profiles["density"](z), rtol=1e-12
    )
    reference = np.asarray(run("pair", "warm", "numpy_ref")["field"])
    field = np.asarray(output["field"])
    # The same ten fixed steps on every backend: only the order of the
    # floating-point sums differs, a few ulps per step compounded over ten.
    scale = float(np.linalg.norm(reference[-1]))
    assert np.linalg.norm(field[-1] - reference[-1]) / scale < 1e-10
    # A warmer gas is thinner: the density at the exit end is below the room-
    # temperature fill's at the same pressure.
    room = build(PAIR, 294.0)
    assert simulation.profiles["density"](z)[-2] < room.profiles["density"](z)[-2]


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_constant_temperature_path_runs_like_the_number(backend):
    # The same problem either way, so the two runs are one march, within its rounding
    # budget (SEP 25); the prescribed planes exactly.
    if backend == "native":
        pytest.importorskip("solprop_native")
    number = run("pair", "number", backend)
    constant = run("pair", "constant", backend)
    bound = simulation_bound(build(PAIR, 294.0, backend=backend), backend=backend)
    assert_matched(constant["field"], number["field"], bound, what="field")
    np.testing.assert_array_equal(constant["z"], number["z"])
    assert_matched(
        constant["stats/density"],
        number["stats/density"],
        STATISTIC_FACTOR * bound,
        what="density",
    )


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
def test_three_populations_take_the_z_table_on_every_backend(backend):
    # Three independent density paths exceed the two-parameter expansion, so Phi
    # takes the z table, which the native CPU build evaluates natively (SEP 29
    # section 6.6): the same matched march as the reference's, within its budget.
    if backend == "native":
        pytest.importorskip("solprop_native")
    simulation = build(TRIPLE, WARM, backend=backend)
    assert simulation.translate().linear.phi_representation == "table"
    output = run("triple", "warm", backend)
    reference = run("triple", "warm", "numpy_ref")
    bound = simulation_bound(simulation, backend=backend)
    assert_matched(output["field"], reference["field"], bound, what="field")


@pytest.mark.medium
@pytest.mark.parametrize(
    "integrator",
    [RungeKutta(), LowStorageRungeKutta()],
    ids=["runge_kutta", "low_storage"],
)
def test_three_populations_on_the_native_z_table_with_each_march(integrator):
    # The other two marches over the same ten fixed steps (the Adams one is above):
    # the native stages and low-storage flows read the table's slots, matched to the
    # reference within the march's budget.
    pytest.importorskip("solprop_native")
    simulation = build(TRIPLE, WARM, backend="native", integrator=integrator)
    assert simulation.translate().linear.phi_representation == "table"
    native = simulation.run()
    reference = build(TRIPLE, WARM, backend="numpy_ref", integrator=integrator).run()
    bound = simulation_bound(simulation, backend="native")
    assert_matched(native["field"], reference["field"], bound, what="field")


def test_the_run_records_the_temperature_path():
    metadata = build(PAIR, WARM).metadata()
    configuration = cast("dict", serialise(metadata.configuration))
    assert metadata.schema_version == 26
    # The constructor's record keeps the path as the dataclass it is, and the
    # medium's record samples it with its breakpoints, as it does a density.
    assert configuration["capillary"]["temperature"] == {
        "__class__": "solprop.profiles.Linear",
        "start": 294.0,
        "end": 360.0,
        "length": LENGTH,
    }
    record = configuration["medium"]["temperature_K"]
    assert record["kind"] == "Linear" and record["breakpoints_m"] == [0.0, LENGTH]
    number = cast("dict", serialise(build(PAIR, 294.0).metadata().configuration))
    assert number["capillary"]["temperature"] == 294.0
    assert number["medium"]["temperature_K"] == 294.0


def test_a_molecular_fill_under_a_temperature_path_warns_for_its_raman_populations():
    # optmat's response (solprop's own rotational ladder, which this once tested, is
    # deleted, SEP 3 Phase 4b) is held at one reference temperature too.
    with pytest.warns(RamanTemperatureWarning, match="N2: the gas temperature varies"):
        simulation = build("N2", WARM, radius=RADIUS, pressure=1.0)
    policies: Any = simulation.material_manifest().policies
    # A constant pressure: the density-weighted mean leans to the cool, dense end.
    assert 294.0 < policies["raman_reference_temperature"] < 327.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RamanTemperatureWarning)
        build("N2", WARM, radius=RADIUS, pressure=1.0, raman=False)
        build("N2", Constant(300.0), radius=RADIUS, pressure=1.0)


def test_the_free_space_routes_refuse_a_temperature_path_by_name():
    mixture = Mixture.fill(PAIR, 1.0, temperature=WARM, length=LENGTH)
    with pytest.raises(MaterialError, match="free-space routes do not carry one"):
        SpatialMedium(mixture)


def test_the_general_route_refuses_a_temperature_path_by_name():
    with pytest.raises(DriverError, match="temperature="):
        _refuse_gas_keywords({"temperature": WARM})


def test_a_supplied_gradient_without_the_flow_weight_is_refused_by_the_constructor():
    boreless = PressureGradient.two_point(LENGTH, 0.5, 0.1)
    with pytest.raises(MaterialError, match="without the flow weight"):
        build(PAIR, WARM, pressure=boreless)


def test_the_design_calculators_refuse_a_temperature_path():
    """The tools take one temperature: a mixture whose temperature varies is refused."""

    mixture = Mixture.fill(
        {species("He"): 1.0}, 1.0, temperature=Linear(294.0, 350.0, 1.0)
    )
    mode = MarcatiliMode(125e-6, "He", 1.0)
    with pytest.raises(ToolsError, match="one temperature"):
        soliton_params(mode, mixture, energy=1e-6, fwhm=10e-15, wavelength=800e-9)


def test_a_temperature_beside_a_mixture_is_refused_unless_it_is_the_mixtures():
    cool = Mixture.fill(PAIR, 0.5, length=LENGTH)
    warm = Mixture.fill(PAIR, 0.5, temperature=WARM, length=LENGTH)
    for mixture, other in ((cool, WARM), (cool, 350.0), (warm, 294.0)):
        with pytest.raises(DriverError, match="Mixture carries its own temperature"):
            build(mixture, other, pressure=None)
    # Omitted (the default), or the mixture's own number or path, is accepted.
    assert build(cool, None, pressure=None).configuration["medium"] is cool
    build(cool, ROOM_TEMPERATURE, pressure=None)
    simulation = build(warm, WARM, pressure=None)
    assert all(getattr(mode, "temperature", None) is WARM for mode in simulation.modes)


def test_the_default_temperature_is_room_temperature():
    medium = build(PAIR, None).configuration["medium"]
    assert isinstance(medium, Mixture) and medium.temperature == ROOM_TEMPERATURE


def test_from_waveguide_replaces_a_modes_temperature_path_with_its_own():
    mode = MarcatiliMode(RADIUS, "He", 1.0, temperature=WARM)
    simulation = Simulation.from_waveguide([mode], LENGTH, "He", 1.0, **keywords(n_r=8))
    (rebound,) = simulation.modes
    assert getattr(rebound, "temperature", None) == ROOM_TEMPERATURE

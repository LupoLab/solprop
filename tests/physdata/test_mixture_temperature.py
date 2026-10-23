# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mixtures and gas-filled modes under a temperature path."""

import math

import numpy as np
import pytest

from solprop.constants import ROOM_TEMPERATURE
from solprop.driver.mixtures import mixture_profiles
from solprop.exceptions import MaterialError, ModeError
from solprop.gasflow import steady_fill
from solprop.modes import MarcatiliMode, RectMode, TubeLattice, VincettiMode
from solprop.modes.capillary import _wall_channel_settings
from solprop.physdata import Mixture, Species, species
from solprop.profiles import (
    Constant,
    DensityPath,
    Linear,
    PiecewiseLinear,
    PressureGradient,
    SeriesOfPath,
)

LENGTH = 1.0
WARM = Linear(294.0, 360.0, LENGTH)
FRACTIONS = {"He": 0.5, "Ar": 0.5}


def value(path, z):
    """Return a path's value at one position."""
    return float(path(np.array([z]))[0])


def test_a_fill_under_a_temperature_path_follows_it_in_every_density():
    mixture = Mixture.fill(FRACTIONS, 2.0, temperature=WARM, length=LENGTH)
    assert mixture.temperature is WARM
    for component, fraction in zip(mixture.components, (0.5, 0.5), strict=True):
        assert isinstance(component.density, DensityPath)
        assert component.density.temperature is WARM
        z = 0.7
        expected = component.species.equation_of_state.density(
            2.0 * fraction, value(WARM, z)
        )
        assert value(component.density, z) == pytest.approx(expected, rel=1e-9)
    assert "Linear(294 to 360 over 1 m) K" in repr(mixture)
    assert "Linear(294 to 360 over 1 m) K" in str(mixture)


def test_a_constant_temperature_path_is_the_number_in_a_mixture_exactly():
    # A constant path is canonicalised to the number at construction, so the two
    # fills are the same object state and evaluate identically (structural).
    z = np.linspace(0.0, LENGTH, 17)
    number = Mixture.fill(FRACTIONS, (2.0, 0.5), temperature=300.0, length=LENGTH)
    constant = Mixture.fill(
        FRACTIONS, (2.0, 0.5), temperature=Constant(300.0), length=LENGTH
    )
    assert constant.temperature == 300.0 and isinstance(constant.temperature, float)
    np.testing.assert_array_equal(constant.density(z), number.density(z))
    assert constant.serialise() == number.serialise()
    assert Mixture.of(*number.components, temperature=300).temperature == 300


def test_a_gradient_under_a_temperature_path_is_weighted_even_without_a_bore():
    mixture = Mixture.fill(FRACTIONS, (2.0, 0.5), temperature=WARM, length=LENGTH)
    fill = mixture.shared
    assert isinstance(fill, PressureGradient) and fill.radius is None
    assert isinstance(fill.weight, SeriesOfPath)
    stages = Mixture.fill(
        FRACTIONS, ([0.0, 0.5, LENGTH], [2.0, 1.5, 0.5]), temperature=WARM
    )
    assert isinstance(stages.shared, PressureGradient)
    assert stages.shared.weight is not None
    with_bore = Mixture.fill(
        FRACTIONS, (2.0, 0.5), temperature=WARM, length=LENGTH, radius=125e-6
    )
    expected = steady_fill(
        {species("He"): 0.5, species("Ar"): 0.5},
        [0.0, LENGTH],
        [2.0, 0.5],
        radius=125e-6,
        temperature=WARM,
    )
    z = np.linspace(0.0, LENGTH, 9)
    assert isinstance(with_bore.shared, PressureGradient)
    np.testing.assert_array_equal(with_bore.shared(z), expected(z))


def test_a_varying_profile_without_the_weight_is_refused_beside_a_temperature_path():
    boreless = PressureGradient.two_point(LENGTH, 2.0, 0.5)
    with pytest.raises(MaterialError, match="without the flow weight"):
        Mixture.fill(FRACTIONS, boreless, temperature=WARM)
    # A declared pressure path that is not a steady-flow fill makes no flow claim:
    # it is taken as given.
    measured = PiecewiseLinear(np.array([0.0, LENGTH]), np.array([2.0, 0.5]))
    assert Mixture.fill(FRACTIONS, measured, temperature=WARM).shared is measured
    # A constant pressure, or a fill that carries the weight, is accepted.
    Mixture.fill(FRACTIONS, Constant(2.0, (0.0, LENGTH)), temperature=WARM)
    weighted = steady_fill(
        "He", [0.0, LENGTH], [2.0, 0.5], radius=125e-6, temperature=WARM
    )
    assert Mixture.fill({"He": 1.0}, weighted, temperature=WARM).shared is weighted


def test_a_custom_species_gradient_under_a_temperature_path_is_refused_by_name():
    inert = Species("inert", species("He").susceptibility)
    with pytest.raises(MaterialError, match="carries no viscosity data"):
        Mixture.fill({inert: 1.0}, (2.0, 0.5), temperature=WARM, length=LENGTH)


def test_pressure_sampling_is_refused_under_a_temperature_path():
    mixture = Mixture.fill(FRACTIONS, (2.0, 0.5), temperature=WARM, length=LENGTH)
    with pytest.raises(MaterialError, match="density:<label>"):
        mixture.at_parameters({"pressure": 1.0})
    frozen = mixture.at_parameters({"density:He": 1e25, "density:Ar": 2e25})
    np.testing.assert_array_equal(frozen.density(np.array([0.3])), [[1e25], [2e25]])


@pytest.mark.parametrize(
    "temperature", [0.0, -1.0, math.inf, math.nan, Linear(294.0, -4.0, LENGTH)]
)
def test_a_mixture_refuses_a_temperature_that_is_not_positive(temperature):
    with pytest.raises(MaterialError, match="temperature of the"):
        Mixture.fill(FRACTIONS, 1.0, temperature=temperature)


def test_the_serialised_mixture_records_a_temperature_path():
    record = Mixture.fill(FRACTIONS, 1.0, temperature=WARM, length=LENGTH).serialise()
    temperature = record["temperature_K"]
    assert isinstance(temperature, dict)
    assert temperature["kind"] == "Linear"
    assert temperature["breakpoints_m"] == [0.0, LENGTH]
    assert temperature["values"][0] == 294.0 and temperature["values"][-1] == 360.0
    assert Mixture.fill(FRACTIONS, 1.0).serialise()["temperature_K"] == ROOM_TEMPERATURE


def test_partial_pressures_and_labelled_populations_take_a_temperature_path():
    mixture = Mixture.partial_pressures({"He": 1.0, "Ar": 0.5}, temperature=WARM)
    assert mixture.shared is None
    assert all(
        isinstance(c.density, DensityPath) and c.density.temperature is WARM
        for c in mixture.components
    )
    shared = Mixture.partial_pressures(
        dict.fromkeys(("He", "Ar"), Constant(1.0)), temperature=WARM
    )
    assert shared.shared is not None


# ------------------------------------------------------------------ gas-filled modes


def test_a_shared_fill_under_a_temperature_path_declares_its_populations():
    shared = Mixture.fill(FRACTIONS, 2.0, length=LENGTH)
    warm = Mixture.fill(FRACTIONS, 2.0, temperature=WARM, length=LENGTH)
    assert sorted(MarcatiliMode(125e-6, mixture=shared).parameter_paths) == [
        "core_radius",
        "pressure",
    ]
    mode = MarcatiliMode(125e-6, mixture=warm)
    assert sorted(mode.parameter_paths) == ["core_radius", "density:Ar", "density:He"]
    assert not mode.z_constant
    assert mode.temperature is WARM
    # Freezing the declared populations reproduces the mode at that position.
    z = 0.6
    frozen = mode.at_parameters(
        {name: value(path, z) for name, path in mode.parameter_paths.items()}
    )
    omega = np.array([2.356e15])
    np.testing.assert_allclose(frozen.neff(omega), mode.neff(omega, z), rtol=1e-15)


def test_a_single_gas_under_a_temperature_path_keeps_one_density_parameter():
    mode = MarcatiliMode(125e-6, "Ar", 2.0, temperature=WARM)
    assert sorted(mode.parameter_paths) == ["core_radius", "density"]
    assert isinstance(mode.density, DensityPath)
    assert mode.density.temperature is WARM
    rows = dict(mode._describe()[1])
    assert rows["temperature"] == "Linear(294 to 360 over 1 m) K"
    omega = np.array([2.356e15])
    assert np.real(mode.dneff_dz(omega, 0.5))[0] < 0.0


def test_a_constant_temperature_path_gives_the_same_mode_exactly():
    # The canonicalised constant makes the two modes the same state: structural.
    omega = np.linspace(2.0e15, 3.0e15, 7)
    fill = PressureGradient.two_point(LENGTH, 2.0, 0.5)
    number = MarcatiliMode(125e-6, "Ar", fill, temperature=300.0)
    constant = MarcatiliMode(125e-6, "Ar", fill, temperature=Constant(300.0))
    assert constant.temperature == 300.0
    for z in (0.0, 0.3, LENGTH):
        np.testing.assert_array_equal(constant.neff(omega, z), number.neff(omega, z))
        np.testing.assert_array_equal(
            constant.dneff_dz(omega, z), number.dneff_dz(omega, z)
        )
    assert _wall_channel_settings(constant) == _wall_channel_settings(number)


def test_a_mode_refuses_a_temperature_path_that_is_not_positive():
    with pytest.raises(ModeError, match="temperature must be positive"):
        MarcatiliMode(125e-6, "Ar", 1.0, temperature=Linear(294.0, 0.0, LENGTH))
    with pytest.raises(ModeError, match="temperature must be positive"):
        RectMode(50e-6, 20e-6, "Ar", 1.0, temperature=Linear(294.0, -1.0, LENGTH))


def test_the_wall_channels_compare_a_temperature_path_by_its_identity():
    first = MarcatiliMode(125e-6, "Ar", 1.0, temperature=WARM)
    same = MarcatiliMode(125e-6, "Ar", 1.0, temperature=Linear(294.0, 360.0, LENGTH))
    other = MarcatiliMode(125e-6, "Ar", 1.0, temperature=Linear(294.0, 361.0, LENGTH))
    assert _wall_channel_settings(first) == _wall_channel_settings(same)
    assert _wall_channel_settings(first) != _wall_channel_settings(other)


def test_a_rectangular_guide_carries_a_temperature_path():
    mode = RectMode(50e-6, 20e-6, "Ar", 1.0, temperature=WARM)
    assert isinstance(mode.density, DensityPath)
    assert mode.density.temperature is WARM
    assert dict(mode._describe()[1])["temperature"] == "Linear(294 to 360 over 1 m) K"


def test_the_partial_pressure_statistic_inverts_each_state_at_its_temperature():
    mixture = Mixture.partial_pressures({"He": 1.0, "Ar": 0.5}, temperature=WARM)
    total = mixture_profiles(mixture, pressure=True)["pressure"]
    z = np.array([0.0, 0.4, LENGTH])
    # Each partial density was built at T(z); inverting at T(z) returns 1.5 bar, to
    # the density table's 1e-9 and the inversion's own tolerance.
    np.testing.assert_allclose(total(z), [1.5, 1.5, 1.5], rtol=1e-8)


# ------------------------------------------------------------------------- the doors

UNWEIGHTED = PressureGradient.two_point(LENGTH, 2.0, 0.1)
"""A steady-flow fill built without a flow weight: p^2 linear in z."""


def lattice_mode(**keywords):
    """Return a Vincetti mode of a small argon-filled lattice fibre."""
    return VincettiMode(
        15e-6, lattice=TubeLattice(180e-9, 10.38e-6, 6), cladding=1.45, **keywords
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda p, t: MarcatiliMode(100e-6, "He", p, temperature=t),
        lambda p, t: RectMode(50e-6, 20e-6, "He", p, temperature=t),
        lambda p, t: lattice_mode(gas="He", pressure=p, temperature=t),
    ],
    ids=["marcatili", "rectangular", "vincetti"],
)
def test_every_gas_filled_mode_refuses_an_unweighted_gradient_beside_a_path(build):
    with pytest.raises(ModeError, match="without the flow weight"):
        build(UNWEIGHTED, WARM)
    # The same fill at one temperature, or a matching weighted fill, is accepted.
    build(UNWEIGHTED, 294.0)
    weighted = steady_fill(
        "He", [0.0, LENGTH], [2.0, 0.1], radius=100e-6, temperature=WARM
    )
    assert build(weighted, WARM).temperature is WARM


def test_partial_pressures_refuse_an_unweighted_gradient_beside_a_path():
    with pytest.raises(MaterialError, match=r"He population.*without the flow weight"):
        Mixture.partial_pressures({"He": UNWEIGHTED}, temperature=WARM)
    with pytest.raises(MaterialError, match="without the flow weight"):
        Mixture.partial_pressures(
            {"He": UNWEIGHTED, "Ar": UNWEIGHTED}, temperature=WARM
        )
    own = steady_fill("He", [0.0, LENGTH], [2.0, 0.1], radius=100e-6, temperature=WARM)
    Mixture.partial_pressures({"He": own}, temperature=WARM)
    # A declared partial-pressure path that is not a flow is taken as given.
    Mixture.partial_pressures({"He": Linear(2.0, 0.1, LENGTH)}, temperature=WARM)


def test_a_weight_for_another_gas_or_temperature_is_refused():
    hot = Linear(294.0, 600.0, LENGTH)
    for_hot_helium = steady_fill(
        "He",
        [0.0, LENGTH],
        [2.0, 0.1],
        radius=100e-6,
        temperature=hot,
        vacuum_end="floor",
    )
    with pytest.raises(MaterialError, match="another gas or temperature path"):
        Mixture.fill({"He": 1.0}, for_hot_helium, temperature=WARM)
    for_argon = steady_fill(
        "Ar",
        [0.0, LENGTH],
        [2.0, 0.1],
        radius=100e-6,
        temperature=WARM,
        vacuum_end="floor",
    )
    with pytest.raises(MaterialError, match="another gas or temperature path"):
        Mixture.fill({"He": 1.0}, for_argon, temperature=WARM)
    with pytest.raises(ModeError, match="another gas or temperature path"):
        MarcatiliMode(100e-6, "He", for_argon, temperature=WARM)
    # The same gas and path, rebuilt, is accepted: the weight is compared by value.
    again = steady_fill(
        "He",
        [0.0, LENGTH],
        [2.0, 0.1],
        radius=100e-6,
        temperature=Linear(294.0, 360.0, LENGTH),
    )
    Mixture.fill({"He": 1.0}, again, temperature=WARM)


def test_new_fractions_refuse_a_retained_weighted_fill():
    mixture = Mixture.fill(
        {"He": 0.5, "Xe": 0.5}, (2.0, 0.1), length=LENGTH, temperature=WARM
    )
    assert isinstance(mixture.shared, PressureGradient)
    assert mixture.shared.weight is not None
    with pytest.raises(MaterialError, match="flow weight eta\\(T\\) T computed"):
        mixture.with_fill({"He": 0.95, "Xe": 0.05})
    rebuilt = mixture.with_fill({"He": 0.95, "Xe": 0.05}, (2.0, 0.1), length=LENGTH)
    assert rebuilt.shared is not mixture.shared
    # A fill at one temperature carries no weight and is kept, as before.
    cool = Mixture.fill({"He": 0.5, "Xe": 0.5}, (2.0, 0.1), length=LENGTH)
    assert cool.with_fill({"He": 0.95, "Xe": 0.05}).shared is cool.shared


def test_a_mode_refuses_a_temperature_beside_a_mixture_unless_it_is_the_mixtures():
    cool = Mixture.fill({"He": 1.0}, 1.0)
    warm = Mixture.fill({"He": 1.0}, 1.0, temperature=WARM, length=LENGTH)
    with pytest.raises(ModeError, match="mixture carries its own temperature"):
        MarcatiliMode(100e-6, mixture=cool, temperature=WARM)
    with pytest.raises(ModeError, match="mixture carries its own temperature"):
        MarcatiliMode(100e-6, mixture=cool, temperature=350.0)
    with pytest.raises(ModeError, match="mixture carries its own temperature"):
        RectMode(50e-6, 20e-6, mixture=warm, temperature=294.0)
    # Omitted, the same number, a constant path of it, or the same path object.
    room = ROOM_TEMPERATURE
    assert MarcatiliMode(100e-6, mixture=cool).temperature == room
    assert MarcatiliMode(100e-6, mixture=cool, temperature=room).temperature == room
    assert MarcatiliMode(100e-6, mixture=cool, temperature=Constant(room)).temperature
    assert MarcatiliMode(100e-6, mixture=warm, temperature=WARM).temperature is WARM
    # Rebinding replaces the gas, the pressure and the temperature path.
    gas = MarcatiliMode(100e-6, "He", 1.0, temperature=WARM)
    assert gas.with_mixture(cool).temperature == room
    assert MarcatiliMode(100e-6, "He", 1.0).temperature == room

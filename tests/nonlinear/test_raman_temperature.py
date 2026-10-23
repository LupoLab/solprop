# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""One reference temperature for the rotational Raman populations.

Of a caller's molecular constants, on the per-component route a mixture of supplied
species takes: solprop's former N2 constants (``former_raman.FORMER_N2``, from its
registry at dc4ca0d) on a species named ``nitrogen`` that carries optmat's N2 equation
of state and transport (the pressure gradient is a gas flow), so that its density, and
with it every reference temperature here, is the built-in gas's. optmat's composition
route, whose reference is taken over the summed density, is tested in
test_optmat_raman.py.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from former_raman import FORMER_N2
from solprop.exceptions import MaterialError, RamanTemperatureWarning
from solprop.nonlinear.mixtures import raman_temperature
from solprop.nonlinear.mixtures import responses_for as _responses_for
from solprop.nonlinear.raman import MolecularOptions, RamanResponse, RotationalLines
from solprop.physdata import (
    Component,
    Gas,
    Mixture,
    OptmatGas,
    PoleSumSusceptibility,
    ReferenceState,
    Species,
)
from solprop.profiles import Constant, FromCallable, Linear, PiecewiseLinear, ZProfile

LENGTH = 0.5
WARM = PiecewiseLinear(np.array([0.0, 0.2, LENGTH]), np.array([294.0, 380.0, 320.0]))


def supplied(name: str, gas: Gas, **data) -> Species:
    """Return a caller's species named `name` with optmat's state for `gas`.

    Named off the :class:`Gas` names, so it is never resolved as the built-in gas:
    its Raman response is built from the constants it carries.
    """
    law = PoleSumSusceptibility(
        gas=gas,
        source="supplied",
        reference_state=ReferenceState(pressure_bar=1.0, temperature=273.15),
        wavelength_range=(200e-9, 5e-6),
        citation="placeholder: these tests read only the density and the Raman data",
        b=(5.8e-4,),
        c=(7.4e-3,),
    )
    return Species(name, law, equation_of_state=OptmatGas(gas.value), **data)


# A caller's nitrogen carrying solprop's former Raman constants, and a helium with no
# Raman data to sit beside it.
NITROGEN = supplied("nitrogen", Gas.N2, raman=FORMER_N2)
HELIUM = supplied("helium", Gas.He)


def responses_for(mixture, **keywords):
    """Select the per-component Raman responses of a mixture of supplied species.

    The Kerr response, which needs a supplied coefficient, is not the point here.
    """
    keywords.setdefault("kerr", False)
    return _responses_for(mixture, **keywords)


def gradient_mixture(temperature: float | ZProfile = WARM) -> Mixture:
    """Return nitrogen from 3 bar to 0.2 bar along the fibre, at a temperature."""
    return Mixture.fill(
        {NITROGEN: 1.0}, (3.0, 0.2), temperature=temperature, length=LENGTH
    )


def mean_by_quadrature(path, temperature, length=LENGTH, points=(0.2,)):
    """Return int rho T dz / int rho dz over [0, length] by independent quadrature."""

    def density(z):
        return float(path(np.array([z]))[0])

    def weighted(z):
        return density(z) * float(temperature(np.array([z]))[0])

    top, _ = quad(weighted, 0.0, length, points=points, epsabs=0.0, epsrel=1e-12)
    bottom, _ = quad(density, 0.0, length, points=points, epsabs=0.0, epsrel=1e-12)
    return top / bottom


def rotational_temperature(responses):
    """Return the temperature the rotational ladder's populations were built at."""
    (raman,) = [one for one in responses if isinstance(one, RamanResponse)]
    (ladder,) = [
        term.model
        for term in raman.tensor_terms
        if isinstance(term.model, RotationalLines)
    ]
    return ladder.temperature


def test_the_reference_is_the_density_weighted_mean_and_it_warns():
    mixture = gradient_mixture()
    expected = mean_by_quadrature(mixture["nitrogen"].density, WARM)
    with pytest.warns(RamanTemperatureWarning) as caught:
        responses = responses_for(mixture, raman=True, length=LENGTH)
    # Two adaptive quadratures of the same smooth integrands, each to 1e-10.
    assert rotational_temperature(responses) == pytest.approx(expected, rel=1e-9)
    (message,) = [str(record.message) for record in caught]
    assert "from 294 K to 380 K" in message
    assert f"{expected:.6g} K" in message
    assert "weighted by the density of nitrogen" in message
    assert "populations of nitrogen" in message
    assert "do not follow" in message
    # The mean sits between the extremes and leans to the dense, cool entrance.
    assert 294.0 < expected < 0.5 * (294.0 + 380.0) + 20.0


def test_an_explicit_reference_is_taken_and_logged_not_warned(caplog):
    # The caller chose the reference knowingly, so it is logged rather than warned.
    mixture = gradient_mixture()
    options = MolecularOptions(temperature=330.0)
    with warnings.catch_warnings(), caplog.at_level("INFO", "solprop.nonlinear"):
        warnings.simplefilter("error")
        responses = responses_for(mixture, raman={"nitrogen": options}, length=LENGTH)
    assert rotational_temperature(responses) == 330.0
    assert "330 K (MolecularOptions.temperature)" in caplog.text
    assert "do not follow" in caplog.text


def test_a_constant_temperature_path_takes_the_number_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        number = responses_for(gradient_mixture(300.0), raman=True)
        constant = responses_for(gradient_mixture(Constant(300.0)), raman=True)
    assert rotational_temperature(number) == rotational_temperature(constant) == 300.0


def test_no_rotational_ladder_means_no_warning():
    mixture = gradient_mixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        responses = responses_for(
            mixture, raman={"nitrogen": MolecularOptions(rotation=False)}, length=LENGTH
        )
    assert not any(
        isinstance(term.model, RotationalLines)
        for one in responses
        if isinstance(one, RamanResponse)
        for term in one.tensor_terms
    )
    atoms = Mixture.fill({"Ar": 1.0}, 1.0, temperature=WARM, length=LENGTH)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        responses_for(atoms, raman=True, length=LENGTH)


def test_a_constant_temperature_is_returned_as_it_is():
    mixture = gradient_mixture(310.0)
    assert raman_temperature(mixture, "nitrogen") == 310.0
    assert raman_temperature(mixture, "nitrogen", 320.0) == 320.0


def test_the_mean_needs_a_finite_span_and_some_gas():
    nitrogen = NITROGEN
    endless_warm = FromCallable(lambda z: 300.0 + z, (0.0, np.inf), np.empty(0))
    mixture = Mixture.of(Component(nitrogen, Constant(1e25)), temperature=endless_warm)
    with pytest.raises(MaterialError, match="no finite run"):
        raman_temperature(mixture, "nitrogen", warn=False)
    with pytest.raises(MaterialError, match="run length must be positive"):
        raman_temperature(mixture, "nitrogen", length=0.0, warn=False)
    # An explicit reference needs no finite run: the spread is probed on the path.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert raman_temperature(mixture, "nitrogen", 320.0) == 320.0
    empty = Mixture.of(
        Component(nitrogen, Constant(0.0, (0.0, LENGTH))),
        temperature=Linear(294.0, 330.0, LENGTH),
    )
    with pytest.raises(MaterialError, match="vanishes"):
        raman_temperature(empty, "nitrogen", warn=False)


def test_each_species_takes_the_mean_over_its_own_density():
    # Nitrogen fed from one end, helium from the other, the gas warming from 294 K to
    # 400 K over a centimetre. Nitrogen sits where it is cool.
    length = 1e-2
    temperature = Linear(294.0, 400.0, length)
    mixture = Mixture.partial_pressures(
        {NITROGEN: Linear(1.0, 0.1, length), HELIUM: Linear(0.1, 3.0, length)},
        temperature=temperature,
    )
    own = raman_temperature(mixture, "nitrogen", length=length, warn=False)
    expected = mean_by_quadrature(
        mixture["nitrogen"].density, temperature, length=length, points=()
    )
    total = mean_by_quadrature(
        mixture.total_density, temperature, length=length, points=()
    )
    # Adaptive quadratures of smooth integrands, each to 1e-10 or better.
    assert own == pytest.approx(expected, rel=1e-9)
    assert own == pytest.approx(330.4, abs=0.05)
    assert total == pytest.approx(352.9, abs=0.05)
    with pytest.warns(
        RamanTemperatureWarning, match="weighted by the density of nitrogen"
    ):
        responses = responses_for(mixture, raman=True, length=length)
    assert rotational_temperature(responses) == pytest.approx(expected, rel=1e-9)


def test_the_mean_is_taken_over_the_run_and_never_beyond_the_fibre():
    # A constant pressure has an unbounded density span, and this temperature path is
    # twice as long as the fibre: only [0, L] counts.
    temperature = Linear(294.0, 494.0, 2 * LENGTH)
    mixture = Mixture.fill({NITROGEN: 1.0}, 2.0, temperature=temperature)
    mean = raman_temperature(mixture, "nitrogen", length=LENGTH, warn=False)
    expected = mean_by_quadrature(
        mixture["nitrogen"].density, temperature, length=LENGTH, points=()
    )
    assert mean == pytest.approx(expected, rel=1e-9)
    assert mean < 394.0
    with pytest.warns(RamanTemperatureWarning, match="from 294 K to 394 K"):
        raman_temperature(mixture, "nitrogen", length=LENGTH)

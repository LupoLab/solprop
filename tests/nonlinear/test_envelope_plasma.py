# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical-source oracles independent of the compiled ellipse implementation."""

from dataclasses import dataclass, replace
from math import comb
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import beta

from rate_oracles import AdkOracle, adk_oracle
from solprop.constants import CHARGE_MASS_RATIO
from solprop.exceptions import ConvergenceError, GridError, MaterialError
from solprop.nonlinear import EnvelopePlasmaResponse, EnvelopePlasmaSources
from solprop.nonlinear import envelope_plasma as envelope_module
from solprop.nonlinear.ionisation import (
    bind_rate_energy,
    rate_convention,
    rate_species,
)
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import ionisation_potential
from solprop.physdata.species import species
from solprop.translate.ionisation import build_rate_table


@dataclass(frozen=True)
class PolynomialRate:
    power: int = 4
    scale: float = 1.0
    convention: str = "instantaneous-isotropic"

    def __call__(self, field_amplitude):
        return self.scale * np.asarray(field_amplitude) ** self.power


@dataclass(frozen=True)
class ThresholdRate:
    threshold: float = 1e7
    convention: str = "instantaneous-isotropic"

    def __call__(self, field_amplitude):
        amplitude = np.asarray(field_amplitude)
        return np.where(amplitude >= self.threshold, 1e-9 * amplitude**2, 0.0)


def test_cartesian_oracle_resolves_thin_cutoff_interval():
    major, ratio = 1e11, 1e-6
    rate = ThresholdRate()
    theta = np.arccos(
        np.sqrt(((rate.threshold / major) ** 2 - ratio**2) / (1 - ratio**2))
    )
    moments = (
        1e-9
        * major**2
        / np.pi
        * np.array([theta + np.sin(2 * theta) / 2, theta - np.sin(2 * theta) / 2])
    )
    response = EnvelopePlasmaResponse("Ar", rate)
    field = np.array([major, 1j * major * ratio])[:, None]
    source = response.sources(field, 1.0, dt=1e-15)
    expected = 2 * response.potential / major * moments * np.array([1, 1j * ratio])
    # The exact truncated polynomial integral resolves a 1e-4 rad missing arc;
    # 2e-10 is the independent adaptive oracle's error budget on each component.
    np.testing.assert_allclose(source.loss_current[:, 0], expected, rtol=2e-10, atol=0)


@pytest.mark.parametrize("minor", [0.0, 1e-12, 0.7, 2 * (1 - 1e-12), 2.0])
def test_loss_fundamental_and_cycle_work(minor):
    a = np.array([2, 1j * minor])[:, None]
    response = EnvelopePlasmaResponse("Ar", PolynomialRate())
    actual = response.sources(a, 1.0, dt=0.01)
    exact_rate = (3 * 2**4 + 2 * 2**2 * minor**2 + 3 * minor**4) / 8
    exact_current = (
        np.array([2 * (3 * 2**2 + minor**2), 1j * minor * (2**2 + 3 * minor**2)]) / 4
    )
    # Degree-four trigonometric polynomials resolve to rounding. The absolute
    # current budget covers the zero component of the linear limit.
    np.testing.assert_allclose(actual.ionisation_rate, exact_rate, rtol=2e-13, atol=0)
    np.testing.assert_allclose(
        actual.loss_current[:, 0] / response.potential,
        exact_current,
        rtol=2e-13,
        atol=2e-14,
    )
    work = np.vdot(a, actual.loss_current).real / (2 * response.potential)
    assert work == pytest.approx(exact_rate, rel=2e-13, abs=0)
    if minor == 0.7:
        collinear = 2 * exact_rate * a[:, 0] / np.vdot(a, a).real
        assert (
            np.linalg.norm(collinear - exact_current) / np.linalg.norm(exact_current)
            > 0.18
        )


@pytest.mark.parametrize("power", [1, 2, 3])
@pytest.mark.parametrize("minor", [0.0, 0.7, 2.0])
def test_phase_average_linear_circular_elliptical(power, minor):
    response = EnvelopePlasmaResponse("Ar", PolynomialRate(2 * power))
    field = np.array([2.0, 1j * minor])[:, None]
    actual = response.sources(field, 1.0, dt=0.01)
    exact = sum(
        comb(power, index)
        * 2.0 ** (2 * index)
        * minor ** (2 * (power - index))
        * beta(index + 0.5, power - index + 0.5)
        / np.pi
        for index in range(power + 1)
    )
    # The analytic beta integral of cos^(2k) sin^(2p-2k) independently
    # integrates each polynomial term; the adaptive rule resolves these to rounding.
    np.testing.assert_allclose(actual.ionisation_rate, exact, rtol=2e-13, atol=0)


def test_cycle_current_phase_covariance_and_real_rotations():
    a = np.array([2, 0.7j])[:, None]
    rotation = np.array([[np.cos(0.61), -np.sin(0.61)], [np.sin(0.61), np.cos(0.61)]])
    phase = np.exp(0.317j)
    response = EnvelopePlasmaResponse("Ar", PolynomialRate())
    before = response.sources(a, 1, dt=0.01)
    after = response.sources(phase * rotation @ a, 1, dt=0.01)
    # Independent integrations accumulate a few hundred weighted terms.
    np.testing.assert_allclose(
        after.ionisation_rate, before.ionisation_rate, rtol=2e-13, atol=0
    )
    np.testing.assert_allclose(
        after.loss_current, phase * rotation @ before.loss_current, rtol=2e-13, atol=0
    )


@pytest.mark.parametrize("initial", [0.0, 0.3, 1.0])
def test_fraction_depletion_and_preionisation(initial):
    rate, amplitude, dt = 1e14, 1e10, 1e-14
    model = PolynomialRate(2, rate / amplitude**2)
    a = np.tile(np.array([amplitude, 1j * amplitude])[:, None], (1, 9))
    response = EnvelopePlasmaResponse("Ar", model, preionisation=initial)
    actual = response.sources(a, 2.5e25, dt=dt)
    exposure = rate * dt * np.arange(a.shape[1])
    survival = (1 - initial) * np.exp(-exposure)
    expected = initial + (1 - initial) * -np.expm1(-exposure)
    assert actual.fraction[0] == initial
    # Constant circular intensity has exact trapezoidal exposure; only quadrature
    # rounding and the nine-point prefix sum remain.
    np.testing.assert_allclose(actual.fraction, expected, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(
        actual.drude_drive,
        CHARGE_MASS_RATIO * 2.5e25 * expected * a,
        rtol=2e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        actual.loss_current,
        response.potential * 2.5e25 * rate * survival * a / amplitude**2,
        rtol=2e-13,
        atol=0,
    )
    # One fraction by two entry points: the running sum's 4 n eps.
    np.testing.assert_allclose(
        response.ionised_fraction(a, dt=dt),
        actual.fraction,
        rtol=4 * a.shape[-1] * np.finfo(float).eps,
        atol=0,
    )


def test_survival_is_retained_after_fraction_rounds_to_one():
    response = EnvelopePlasmaResponse("Ar", PolynomialRate(2))
    a = np.tile(np.array([1, 1j])[:, None], (1, 2))
    actual = response.sources(a, 1, dt=40.0)
    assert actual.fraction[-1] == 1
    assert abs(actual.loss_current[0, -1]) > 0
    # The analytic survival is exp(-40), even though subtraction 1-f gives zero.
    np.testing.assert_allclose(
        actual.loss_current[:, -1] / actual.loss_current[:, 0],
        np.exp(-40),
        rtol=2e-13,
        atol=0,
    )


def test_vacuum_zero_field_and_complete_preionisation():
    response = EnvelopePlasmaResponse("He", adk_oracle("He"), preionisation=0.2)
    zero = response.sources(np.zeros((2, 2), complex), 1e25, dt=1e-16)
    # Structural: no field, no density or a full fraction give exact zeros.
    np.testing.assert_array_equal(zero.fraction, [0.2, 0.2])
    np.testing.assert_array_equal(zero.drude_drive, 0)
    np.testing.assert_array_equal(zero.loss_current, 0)
    field = np.array([[3e10 + 0j]])
    vacuum = response.sources(field, 0, dt=1e-16)
    np.testing.assert_array_equal(vacuum.drude_drive, 0)
    np.testing.assert_array_equal(vacuum.loss_current, 0)
    assert vacuum.ionisation_rate[0] > 0
    full = replace(response, preionisation=1.0).sources(field, 1e25, dt=1e-16)
    np.testing.assert_array_equal(full.loss_current, 0)
    # The same constant-times-field expression differs only by multiplication order.
    np.testing.assert_allclose(
        full.drude_drive, CHARGE_MASS_RATIO * 1e25 * field, rtol=2e-16, atol=0
    )


def test_sources_are_fresh_and_rhs_calls_do_not_share_population():
    response = EnvelopePlasmaResponse("Ar", PolynomialRate(2), species="argon-a")
    field = np.ones((1, 2), complex)
    first = response.sources(field, 1, dt=0.1)
    first.fraction[:] = 1
    second = response.sources(field, 1, dt=0.1)
    assert second.fraction[0] == 0
    for name in ("ionisation_rate", "fraction", "drude_drive", "loss_current"):
        assert not np.shares_memory(getattr(first, name), getattr(second, name))
    assert response.traits.grids == ("envelope",)
    assert response.traits.domain == "memory"
    assert response.traits.polarisation == "vector_isotropic"
    assert response.traits.z_profiles == ("density:argon-a", "preionisation:argon-a")


@pytest.mark.parametrize(
    "model",
    [
        adk_oracle("Ar"),
        OptmatRate("Ar", 800e-9, model="ppt_ilkov"),
        OptmatRate("N2", 800e-9),
    ],
    ids=["adk-oracle", "optmat-ppt", "optmat-n2"],
)
def test_rate_convention_survives_table_building_and_caching(model):
    table = build_rate_table(model, (1e9, 5e10), rtol=1e-5)
    assert rate_convention(model) == rate_convention(table) == "instantaneous-isotropic"
    assert rate_species(model) == rate_species(table)
    assert build_rate_table(model, (1e9, 5e10), rtol=1e-5) is table
    name = rate_species(model)[0]
    assert name is not None
    # A table keeps the rate's potential, not its declared energy per event: the
    # response's species is bound to the rate's energy explicitly.
    EnvelopePlasmaResponse(bind_rate_energy(name, model), table)
    if isinstance(model, AdkOracle):
        averaged = build_rate_table(
            replace(model, cycle_averaged=True), (1e9, 5e10), rtol=1e-5
        )
        assert averaged.key != table.key
        with pytest.raises(MaterialError, match="instantaneous-isotropic"):
            EnvelopePlasmaResponse(bind_rate_energy(name, model), averaged)


@pytest.mark.parametrize(
    "model",
    [
        adk_oracle("Ar", cycle_averaged=True),
        PolynomialRate(convention="unspecified"),
        lambda amplitude: np.asarray(amplitude) ** 4,
    ],
)
def test_rate_conventions_and_invalid_inputs(model):
    with pytest.raises(MaterialError, match="instantaneous-isotropic"):
        EnvelopePlasmaResponse("Ar", model)
    # The field-plasma response still accepts each of these rates.
    PlasmaResponse("Ar", model)


@dataclass(frozen=True)
class DeclaredRate(PolynomialRate):
    """A rate declaring a species and potential but no energy per event."""

    species_identity: str = "Ar"
    ionisation_potential: float = 0.0


def test_inconsistent_species_and_binding_energy_are_refused():
    with pytest.raises(MaterialError, match="rate is Ar's"):
        EnvelopePlasmaResponse("He", adk_oracle("Ar"))
    table = build_rate_table(adk_oracle("Ar"), (1e9, 5e10), rtol=1e-5)
    with pytest.raises(MaterialError, match="rate is Ar's"):
        EnvelopePlasmaResponse("He", table)
    # The table keeps the oracle's energy per event, which the response pays.
    bound = EnvelopePlasmaResponse("Ar", table)
    assert bound.potential == adk_oracle("Ar").plasma_energy
    gas = species("Ar")
    assert gas.ionisation is not None
    declared = DeclaredRate(ionisation_potential=gas.ionisation.potential)
    EnvelopePlasmaResponse(gas, declared)
    altered = replace(
        gas,
        ionisation=replace(gas.ionisation, potential=1.01 * gas.ionisation.potential),
    )
    with pytest.raises(MaterialError, match="ionisation potential"):
        EnvelopePlasmaResponse(altered, declared)
    assert rate_species(OptmatRate("Ar", 800e-9)) == ("Ar", ionisation_potential("Ar"))
    with pytest.raises(MaterialError, match="no ionisation data"):
        EnvelopePlasmaResponse(replace(gas, ionisation=None), PolynomialRate())


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_initial_fraction_is_refused(value):
    with pytest.raises(MaterialError, match="preionisation"):
        EnvelopePlasmaResponse("Ar", PolynomialRate(), preionisation=value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_dt_is_refused(value):
    with pytest.raises(GridError, match="dt"):
        EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(
            np.zeros((1, 1), complex), 1, dt=value
        )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_density_is_refused(value):
    with pytest.raises(GridError, match="density"):
        EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(
            np.zeros((1, 1), complex), value, dt=1
        )


@pytest.mark.parametrize(
    "field",
    [
        np.zeros((3, 1), complex),
        np.zeros((1, 0), complex),
        np.full((1, 1), np.nan + 0j),
        np.full((1, 1), np.inf + 0j),
    ],
)
def test_invalid_field_is_refused(field):
    with pytest.raises(GridError, match="field"):
        EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(field, 1, dt=1)


def test_source_container_copies_inputs_and_validates():
    rate = np.zeros(2)
    field = np.zeros((1, 2), complex)
    source = EnvelopePlasmaSources(rate, rate, field, field)
    assert not np.shares_memory(source.fraction, source.ionisation_rate)
    assert not np.shares_memory(source.loss_current, source.drude_drive)
    with pytest.raises(GridError, match="same nonempty"):
        EnvelopePlasmaSources(rate, np.zeros(3), field, field)
    with pytest.raises(GridError, match="currents"):
        EnvelopePlasmaSources(rate, rate, field, np.zeros((2, 3), complex))
    with pytest.raises(GridError, match="fractions"):
        EnvelopePlasmaSources(rate, np.full(2, 1.1), field, field)
    with pytest.raises(GridError, match="finite"):
        EnvelopePlasmaSources(np.full(2, np.inf), rate, field, field)


@dataclass(frozen=True)
class MalformedRate:
    value: complex | float | np.ndarray
    convention: str = "instantaneous-isotropic"

    def __call__(self, field_amplitude):
        return np.asarray(0.0 if field_amplitude == 0 else self.value)


@pytest.mark.parametrize("value", [np.array([1.0]), 1j, -1.0, np.nan, np.inf])
def test_invalid_rate_values_fail_before_population(value):
    response = EnvelopePlasmaResponse("Ar", MalformedRate(value))
    with pytest.raises(GridError, match="rate must"):
        response.sources(np.array([[1 + 0j]]), 1.0, dt=1.0)


def test_invalid_binding_and_nonzero_dark_rate_are_refused():
    with pytest.raises(MaterialError, match="label"):
        EnvelopePlasmaResponse("Ar", PolynomialRate(), species=" ")
    with pytest.raises(MaterialError, match="zero at zero"):
        EnvelopePlasmaResponse("Ar", PolynomialRate(power=0))


def test_unrepresentable_physical_magnitude_is_refused():
    field = np.full((2, 1), np.finfo(float).max, complex)
    with pytest.raises(GridError, match="magnitude is not representable"):
        EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(field, 1.0, dt=1.0)


def test_adaptive_cycle_failure_is_reported(monkeypatch):
    def failed_integral(*args, **kwargs):
        return np.zeros(2, complex), np.inf, SimpleNamespace(success=False)

    monkeypatch.setattr(envelope_module.integrate, "quad_vec", failed_integral)
    with pytest.raises(ConvergenceError, match="Cartesian"):
        EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(
            np.ones((1, 1), complex), 1.0, dt=1.0
        )


def test_real_field_zero_has_a_finite_zero_integrand(monkeypatch):
    original = envelope_module.integrate.quad_vec

    def inspect_zero(integrand, *args, **kwargs):
        np.testing.assert_array_equal(integrand(0.0), 0)
        return original(integrand, *args, **kwargs)

    monkeypatch.setattr(envelope_module.integrate, "quad_vec", inspect_zero)
    EnvelopePlasmaResponse("Ar", PolynomialRate()).sources(
        np.full((1, 1), 1j), 1.0, dt=1.0
    )


def test_subnormal_field_preserves_the_continuous_zero_rate_limit():
    field = np.full((2, 3), 1e-320 + 0j)
    response = EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=0.1)
    sources = response.sources(field, 1e25, dt=1e-15)
    np.testing.assert_array_equal(sources.ionisation_rate, 0)
    np.testing.assert_array_equal(sources.loss_current, 0)
    np.testing.assert_array_equal(sources.fraction, 0.1)
    # Multiplying density by the initial fraction in a different order changes
    # one ulp. The subnormal input must still give a finite prescribed-density drive.
    np.testing.assert_allclose(
        sources.drude_drive, CHARGE_MASS_RATIO * 1e24 * field, rtol=5e-16, atol=0
    )

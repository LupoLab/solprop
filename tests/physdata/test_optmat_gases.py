# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gas linear laws from optmat: chi = N alpha / eps0, ranges and the dilute error."""

import dataclasses
import math
import re
from typing import Any

import numpy as np
import pytest
from former_gas_laws import AR_BORZSONYI, XE_BORZSONYI
from optmat.conditions import State
from optmat.exceptions import StateError
from optmat.gas.air import air_model
from optmat.gas.eos import air_equation_of_state, equation_of_state
from optmat.gas.models.registry import default_model, model

from solprop.constants import EPS0, K_B, C
from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialError,
    MaterialRangeWarning,
)
from solprop.modes.capillary import MarcatiliMode
from solprop.physdata import Gas, Mixture, Species, gamma, species
from solprop.physdata.mixtures import DILUTE_SCALING_THRESHOLD
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import OptmatSusceptibility
from solprop.physdata.species import OptmatGas

GASES = [gas for gas in Gas if gas is not Gas.Air]
SELECTION = MaterialSelection()
# Wavelengths inside every built-in gas's qualified intervals: the infrared-active
# molecules' first window starts at 320 nm (N2O) and every fit reaches 1123 nm.
INSIDE = np.array([350e-9, 532e-9, 800e-9, 1030e-9])


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength, dtype=float)


def reference(gas):
    return air_model("dry air (N2, O2, Ar)") if gas is Gas.Air else default_model(gas)


@pytest.mark.parametrize("gas", list(Gas))
def test_susceptibility_is_the_polarizability_over_eps0_for_every_gas(gas):
    adapted = OptmatSusceptibility.resolve(gas)
    alpha = reference(gas).complex_polarizability(INSIDE)
    # One division by eps0 and the omega-to-wavelength round trip: a few roundings.
    np.testing.assert_allclose(
        adapted.susceptibility(omega_of(INSIDE)) * EPS0, np.real(alpha), rtol=4e-15
    )
    dalpha = reference(gas).dpolarizability_domega(INSIDE)
    np.testing.assert_allclose(
        adapted.dsusceptibility_domega(omega_of(INSIDE)) * EPS0,
        np.real(dalpha),
        rtol=4e-15,
    )
    assert adapted.susceptibility(omega_of(INSIDE)).dtype == np.float64
    assert not adapted.absorbing
    assert adapted.material_records[0].digest == reference(gas).identity.digest


def test_the_static_limit_is_zero_frequency():
    argon = OptmatSusceptibility.resolve("Ar")
    static = np.real(default_model("Ar").complex_polarizability(math.inf)) / EPS0
    assert float(argon.susceptibility(0.0)) == pytest.approx(static, rel=1e-15)
    assert float(argon.dsusceptibility_domega(0.0)) == 0.0


@pytest.mark.parametrize("gas", ["He", "Ar", "N2", "CH4", "SF6", "Air"])
def test_the_derivative_agrees_with_independent_finite_differences(gas):
    # Central differences at three relative steps, each Richardson-extrapolated: the
    # truncation error falls as h^4 and the roundoff grows as eps/h, so the steps
    # bracket the balance, and every one of them must agree with the analytic value,
    # not just the best (a single difference never certifies a rough derivative).
    adapted = OptmatSusceptibility.resolve(gas)
    omega = omega_of(np.array([400e-9, 800e-9]))
    analytic = adapted.dsusceptibility_domega(omega)
    for relative in (1e-3, 3e-4, 1e-4):
        step = relative * omega
        wide = (
            adapted.susceptibility(omega + step) - adapted.susceptibility(omega - step)
        ) / (2 * step)
        narrow = (
            adapted.susceptibility(omega + step / 2)
            - adapted.susceptibility(omega - step / 2)
        ) / step
        # (4 narrow - wide) / 3 is fourth order; at these steps its truncation is
        # below 1e-9 relative and its roundoff near 1e-16 / 1e-4.
        np.testing.assert_allclose((4 * narrow - wide) / 3, analytic, rtol=1e-8)


@pytest.mark.parametrize("gas", ["He", "Ar", "N2", "O2"])
def test_a_band_inside_the_qualified_interval_raises_no_warning(gas):
    adapted = OptmatSusceptibility.resolve(gas)
    low = default_model(gas).intervals[0][0]
    # The interval edges themselves are inside: the band ends exactly on them.
    adapted.susceptibility(omega_of(np.array([low * (1 + 1e-12), 2e-6])))


def test_the_band_below_the_interval_warns_once_per_context():
    argon = OptmatSusceptibility.resolve("Ar")
    with pytest.warns(MaterialRangeWarning, match="below its shortest") as caught:
        argon.susceptibility(omega_of(np.array([80e-9, 800e-9])))
    assert len(caught) == 1
    assert "fit/Ar" in str(caught[0].message)


@pytest.mark.parametrize(
    ("gas", "wavelength", "reason"),
    [
        ("CH4", 1130e-9, "a gap (width)"),
        ("CH4", 1160e-9, "a gap (absorption)"),
        ("CH4", 5e-6, "a gap (representation)"),
        ("N2O", 1517e-9, "a gap (absorption)"),
        ("N2O", 3e-6, "a gap (representation)"),
    ],
)
def test_a_band_in_a_gap_between_intervals_names_the_gap_and_its_reason(
    gas, wavelength, reason
):
    adapted = OptmatSusceptibility.resolve(gas)
    with pytest.warns(MaterialRangeWarning, match=re.escape(reason)):
        adapted.susceptibility(omega_of(np.array([800e-9, wavelength])))


def test_beyond_the_last_interval_is_named():
    sf6 = OptmatSusceptibility.resolve("SF6")
    with pytest.warns(MaterialRangeWarning, match="beyond its longest"):
        sf6.susceptibility(omega_of(np.array([800e-9, 7e-6])))


def test_air_is_the_named_composition_and_its_alternatives():
    default = OptmatSusceptibility.resolve(Gas.Air)
    assert default.source == "air/dry air (N2, O2, Ar)"
    argon_free = OptmatSusceptibility.resolve(Gas.Air, "dry air (N2, O2)")
    assert argon_free.source == "air/dry air (N2, O2)"
    with pytest.raises(MaterialError, match="air compositions"):
        OptmatSusceptibility.resolve(Gas.Air, "fit/N2")
    # The composition's summed polarizabilities, and so its own identity.
    alpha = sum(
        c.mole_fraction
        * np.real(default_model(c.material).complex_polarizability(800e-9))
        for c in air_model().constituents
    )
    assert float(default.susceptibility(omega_of(800e-9))) * EPS0 == pytest.approx(
        alpha, rel=1e-15
    )


def test_a_named_or_supplied_model_is_checked_against_its_gas():
    named = OptmatSusceptibility.resolve("Ar", "borzsonyi_2008/sellmeier_Ar")
    assert named.source == "borzsonyi_2008/sellmeier_Ar"
    supplied = OptmatSusceptibility.resolve("Ar", model("fit/Ar"))
    assert supplied.model is model("fit/Ar")
    with pytest.raises(MaterialError, match="describes N2, not Ar"):
        OptmatSusceptibility.resolve("Ar", "fit/N2")
    with pytest.raises(MaterialError, match="describes dry air"):
        OptmatSusceptibility.resolve("N2", air_model())
    with pytest.raises(MaterialError, match="no linear model 'fit/Zz'"):
        OptmatSusceptibility.resolve("Ar", "fit/Zz")
    with pytest.raises(MaterialError, match="unknown gas"):
        OptmatSusceptibility.resolve("Zz")


def test_the_reference_state_is_the_dilute_limit():
    argon = OptmatSusceptibility.resolve("Ar")
    assert argon.reference_state.pressure_bar == 0.0
    assert argon.reference_density == 0.0
    with pytest.raises(MaterialError, match="no reference-state susceptibility"):
        argon._chi_and_derivative(np.array([1.0]))
    assert not gamma("Ar").absorbing


@dataclasses.dataclass(frozen=True)
class _AbsorbingArgon:
    """A test double of an absorbing optmat dispersion model: argon with Im alpha."""

    base: Any = dataclasses.field(default_factory=lambda: default_model("Ar"))
    absorbing: bool = True
    gas: str = "Ar"

    @property
    def identity(self):
        return self.base.identity

    @property
    def intervals(self):
        return self.base.intervals

    @property
    def wavelength_range(self):
        return self.base.wavelength_range

    @property
    def citation(self):
        return "test double"

    def complex_polarizability(self, wavelength):
        return self.base.complex_polarizability(wavelength) * (1 + 1e-3j)

    def dpolarizability_domega(self, wavelength):
        return self.base.dpolarizability_domega(wavelength) * (1 + 1e-3j)


def test_an_absorbing_model_keeps_its_imaginary_part_and_its_absorption():
    absorbing = OptmatSusceptibility.resolve("Ar", _AbsorbingArgon())
    assert absorbing.absorbing
    values = absorbing.susceptibility(omega_of(INSIDE))
    assert np.iscomplexobj(values) and np.all(np.imag(values) > 0)
    argon = Species("Ar", absorbing, OptmatGas("Ar"))
    fill = Mixture.fill({argon: 1.0}, 1.0)
    mode = MarcatiliMode(50e-6, mixture=fill)
    attributed = mode.absorption_by_species(omega_of(INSIDE))
    np.testing.assert_array_less(0.0, attributed)


def test_dilute_error_matches_optmat_refractivity_at_the_same_density():
    argon = OptmatSusceptibility.resolve("Ar")
    eos = equation_of_state("Ar")
    state = State(10e5, 294.0)
    density = eos.number_density(state)
    error, wavelength = argon.dilute_scaling_error(density, 294.0, omega_of(INSIDE))
    alpha = default_model("Ar").polarizability(wavelength)
    chi = density * alpha / EPS0
    exact = default_model("Ar").refractivity(wavelength, state)
    expected = (chi / (np.sqrt(1 + chi) + 1)) / exact - 1
    # The pressure inverse reproduces the density to 1e-12; the error is ~2e-3.
    assert error == pytest.approx(float(expected), rel=1e-6)
    # SEP 3, section 4.2: argon at 10 bar, 294 K and 800 nm is -0.19 %.
    assert error == pytest.approx(-1.9e-3, rel=0.05)


def test_dilute_error_of_air_uses_the_pseudo_pure_density():
    air = OptmatSusceptibility.resolve(Gas.Air)
    density = air_equation_of_state().number_density(State(1e5, 294.0))
    error, _ = air.dilute_scaling_error(density, 294.0, omega_of(np.array([800e-9])))
    assert -1e-3 < error < 0


def test_dilute_error_outside_the_intervals_is_used_only_when_nothing_is_inside():
    argon = OptmatSusceptibility.resolve("Ar")
    error, wavelength = argon.dilute_scaling_error(
        2.5e25, 294.0, omega_of(np.array([50e-9, 90e-9]))
    )
    assert wavelength == pytest.approx(90e-9) and math.isfinite(error)
    with pytest.raises(MaterialError, match="positive band"):
        argon.dilute_scaling_error(2.5e25, 294.0, np.array([0.0]))


def test_dilute_error_is_measured_at_the_density_without_an_equation_of_state():
    # optmat 0.1.0's density inverse fails for N2 below 284 K and Ar below 255 K;
    # the check compares the two laws at the density itself, so it never needs it.
    for gas, temperature in (("N2", 280.0), ("Ar", 250.0)):
        adapted = OptmatSusceptibility.resolve(gas)
        density = OptmatGas(gas).density(5.0, temperature)
        error, wavelength = adapted.dilute_scaling_error(
            density, temperature, omega_of(np.array([800e-9]))
        )
        state = State(density * K_B * temperature, temperature)
        reference = default_model(gas)
        chi = density * reference.polarizability(wavelength) / EPS0
        exact = reference.refractivity(wavelength, state, ideal=True)
        assert error == pytest.approx(
            float((chi / (np.sqrt(1 + chi) + 1)) / exact - 1), rel=1e-12
        )
    # And at room temperature the ideal-gas state is the real-gas one's density.
    argon = default_model("Ar")
    real = State(10e5, 294.0)
    density = equation_of_state("Ar").number_density(real)
    ideal = State(density * K_B * 294.0, 294.0)
    assert argon.refractivity(800e-9, ideal, ideal=True) == pytest.approx(
        argon.refractivity(800e-9, real), rel=4e-16
    )


def test_a_cooled_optmat_fill_propagates_through_its_mode():
    band = omega_of(np.linspace(400e-9, 1200e-9, 64))
    for gas, temperature in (("N2", 280.0), ("Ar", 250.0)):
        fill = Mixture.fill(
            {gas: 1.0}, 1.0, temperature=temperature, selection=SELECTION
        )
        mode = MarcatiliMode(100e-6, mixture=fill)
        assert np.all(np.isfinite(mode.neff(band)))
        (check,) = fill.dilute_scaling_checks(band)
        assert check.measured and check.temperature == temperature


@dataclasses.dataclass(frozen=True)
class _RefusingArgon(_AbsorbingArgon):
    """A test double whose finite-density law optmat refuses."""

    absorbing: bool = False

    def complex_polarizability(self, wavelength):
        return self.base.complex_polarizability(wavelength)

    def dpolarizability_domega(self, wavelength):
        return self.base.dpolarizability_domega(wavelength)

    def polarizability(self, wavelength, warn=True):
        return self.base.polarizability(wavelength, warn=warn)

    def refractivity(self, *args, **kwargs):
        raise StateError("refused for the test")


def test_a_refused_dilute_check_is_recorded_and_never_fails_the_fill():
    refusing = OptmatSusceptibility.resolve("Ar", _RefusingArgon())
    with pytest.raises(MaterialError, match="dilute-scaling check"):
        refusing.dilute_scaling_error(2.5e25, 294.0, omega_of(np.array([800e-9])))
    argon = Species("Ar", refusing, OptmatGas("Ar"))
    fill = Mixture.fill({argon: 1.0}, 30.0)
    mode = MarcatiliMode(100e-6, mixture=fill)
    band = omega_of(np.array([600e-9, 800e-9]))
    assert np.all(np.isfinite(mode.neff(band)))
    (check,) = fill.dilute_scaling_checks(band)
    assert not check.measured and not check.exceeded
    assert "refused for the test" in check.reason


def test_a_fill_warns_once_per_gas_where_dilute_scaling_exceeds_its_bound():
    band = omega_of(np.linspace(400e-9, 1200e-9, 200))
    low = MarcatiliMode(
        50e-6, mixture=Mixture.fill({"Ar": 1.0}, 1.0, selection=SELECTION)
    )
    low.neff(band)
    assert low.mixture is not None
    (check,) = low.mixture.dilute_scaling_checks(band)
    assert not check.exceeded and check.threshold == DILUTE_SCALING_THRESHOLD
    # Below the 1 % bound the error is recorded but not warned about: argon and
    # xenon at 10 bar each are about -2e-3 and -6e-3.
    middle = Mixture.fill({"Ar": 0.5, "Xe": 0.5}, 20.0, selection=SELECTION)
    MarcatiliMode(50e-6, mixture=middle).neff(band)
    checks = middle.dilute_scaling_checks(band)
    assert all(c.measured and not c.exceeded for c in checks)
    assert all(1e-3 < abs(c.relative_error) < DILUTE_SCALING_THRESHOLD for c in checks)
    # At 70 bar each, argon is about -1.4e-2 and krypton -2.5e-2.
    high = MarcatiliMode(
        50e-6, mixture=Mixture.fill({"Ar": 0.5, "Kr": 0.5}, 140.0, selection=SELECTION)
    )
    with pytest.warns(DiluteScalingWarning) as caught:
        high.neff(band)
    assert {str(w.message).split(":")[0] for w in caught} == {"Ar", "Kr"}
    assert high.mixture is not None
    checks = high.mixture.dilute_scaling_checks(band)
    assert [c.label for c in checks] == ["Ar", "Kr"]
    assert all(c.exceeded for c in checks)


def test_a_single_gas_slot_with_an_optmat_law_is_checked_too():
    band = omega_of(np.linspace(400e-9, 1000e-9, 50))
    mode = MarcatiliMode(
        50e-6,
        gas="Xe",
        pressure=25.0,
        susceptibility=OptmatSusceptibility.resolve("Xe"),
    )
    with pytest.warns(DiluteScalingWarning, match="Xe"):
        mode.neff(band)
    # A caller's own law is not checked (its dilute scaling is the caller's); a named
    # gas without one is optmat's, and is.
    supplied = MarcatiliMode(
        50e-6, gas="Xe", pressure=25.0, susceptibility=XE_BORZSONYI
    )
    supplied.neff(band)
    assert supplied._dilute_checks(band) == ()
    default = MarcatiliMode(50e-6, gas="Xe", pressure=25.0)
    with pytest.warns(DiluteScalingWarning, match="Xe"):
        default.neff(band)


def test_a_vacuum_population_is_not_checked():
    fill = Mixture.partial_pressures({"Ar": 0.0, "He": 1.0}, selection=SELECTION)
    checks = fill.dilute_scaling_checks(omega_of(INSIDE))
    assert [c.label for c in checks] == ["He"]


def test_species_takes_the_linear_law_equation_of_state_and_records_from_optmat():
    argon = species("Ar", selection=SELECTION)
    assert isinstance(argon.susceptibility, OptmatSusceptibility)
    assert argon.equation_of_state == OptmatGas("Ar")
    kinds = [r.kind for r in argon.material_records]
    assert kinds == [
        "gas_refractive",
        "equation_of_state",
        "transport",
        "gas_hyperpolarizability",
    ]
    # The Kerr coefficient is optmat's at a carrier, not a fixed number; the
    # ionisation potential is optmat's, and the Raman response is optmat's too,
    # outside the species.
    assert argon.gamma3 is None
    assert argon.kerr is not None
    assert argon.raman is None
    assert argon.ionisation is not None
    # selection=None is MaterialSelection().
    assert species("Ar").material_records == argon.material_records
    named = species("Ar", selection=MaterialSelection(linear="peck_1964/peck_15C"))
    assert named.susceptibility.source == "peck_1964/peck_15C"


def test_species_refuses_overriding_a_custom_species():
    custom = Species("Ar", XE_BORZSONYI)
    with pytest.raises(MaterialError, match="cannot override custom species"):
        species(custom, selection=SELECTION)


def test_a_supplied_species_in_a_selected_fill_is_never_matched_by_name():
    custom = Species("Ar", AR_BORZSONYI)
    fill = Mixture.fill({custom: 0.5, "He": 0.5}, 1.0, selection=SELECTION)
    assert fill.components[0].species is custom
    assert isinstance(fill.components[1].species.susceptibility, OptmatSusceptibility)
    assert {r.material for r in fill.material_records} == {"He"}


def test_an_override_runs_through_a_species_with_status_user_and_its_base():
    narrowed = default_model("Ar").with_parameters(intervals=((120e-9, math.inf),))
    argon = species("Ar", selection=MaterialSelection(linear=narrowed))
    (record,) = [r for r in argon.material_records if r.kind == "gas_refractive"]
    assert record.status == "user"
    assert record.base == default_model("Ar").identity.digest
    # Its narrower interval is what the range policy now checks.
    with pytest.warns(MaterialRangeWarning, match="below its shortest"):
        argon.susceptibility.susceptibility(omega_of(np.array([110e-9])))

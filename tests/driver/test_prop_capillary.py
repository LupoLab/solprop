# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`prop_capillary`: the keyword surface, and the RDW regression that is milestone M2.

The surface tests run a millimetre or two of capillary, where the pulse barely evolves
and what is under test is the wiring: which mode set, which quadrature, which density
path, which rate table and which statistics come out of a set of keywords. The
milestone tests are in the slow lane and are documented where they sit.

Every helium capillary asks its gas index for a band wider than the band any published
fit covers -- a run needs the index out to several microns and the Ermolov expansion is
fitted to 1 um -- so the extrapolation warning is silenced here as it is in
``tests/backends/test_numpy_ref_multimode.py``. It is reported once per model and says
nothing about the run.
"""

import functools
import json
import logging
import math
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI  # pyright: ignore[reportMissingImports]
from coverage import Coverage
from optmat.conditions import State
from optmat.gas.eos import equation_of_state
from scipy.constants import N_A
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.special import jn_zeros, jv

from bounds import assert_within, record_backend
from convergence_records import Column, Study, record
from convergence_studies import BACKEND_EQUIVALENCE
from former_raman import FORMER_H2
from gated_source import _gated_source
from march_bounds import (
    FIXED_ORDER,
    STATISTIC_FACTOR,
    assert_matched,
    evaluation_rounding,
    simulation_bound,
)
from rate_oracles import HISTORICAL_ATOMS, AdkOracle, PptOracle, adk_oracle, ppt_oracle
from selected_backend import selected_backend
from solprop import tools
from solprop.backends.numpy_ref import QUADRATURE_ALARM, NumpyRefBackend, passes
from solprop.backends.registry import select
from solprop.boundaries import absorber
from solprop.constants import AU_EFIELD, AU_ENERGY, EPS0, C
from solprop.driver import Simulation, prop_capillary
from solprop.driver.mixtures import capillary_fill
from solprop.driver.simulation import (
    BACKEND_STATISTICS,
    PLASMA_STATISTICS,
    _in_lowest_mode,
    default_field_range,
    default_radial_order,
)
from solprop.exceptions import (
    DriverError,
    GasFlowWarning,
    MaterialError,
    MaterialRangeWarning,
    ModeError,
    SolpropWarning,
    TranslationError,
)
from solprop.gasflow import VacuumEnd, choked_pressure, steady_fill
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.log import NullReporter
from solprop.modes import MarcatiliMode, RectMode
from solprop.modes.quadrature import RadialRule, mode_matrix, transverse_quadrature
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, RotationalLines
from solprop.nonlinear.responses import KerrResponse
from solprop.output import MemoryOutput, open_file
from solprop.output.hdf5 import resume
from solprop.physdata import Gas, Mixture, ReferenceState, gamma, n2
from solprop.physdata._optmat import composition_fractions
from solprop.physdata.glasses import TabulatedGlass, fused_silica
from solprop.physdata.mixtures import Component, Pressure
from solprop.physdata.sellmeier import InversePoleIndex, PoleSumSusceptibility
from solprop.physdata.species import IonisationData, Species, species
from solprop.processing import get_power_weight, get_spectral_density
from solprop.profiles import Constant, DensityPath, Linear, PressureGradient, ZProfile
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec, analytic_field, energy
from solprop.stepper import FixedSchedule, RungeKutta, observed_order
from solprop.translate.responses import KerrCoefficients, ResponseStage

# A millimetre of capillary on a small grid: the surface is what is measured, not the
# physics, so everything is as cheap as it can be while still being a real run.
SHORT = 1e-3
RADIUS = 125e-6
LAMBDA0 = 800e-9

# A run whose operator varies along z tabulates it, and the tabulation of a *short*
# gradient costs a wasted 512-interval search before `build_phi` relaxes its tolerance
# and rebuilds: measured 52 s at 1 mm against 0.04 s at 5 cm for this case, with the
# same 22-interval answer either way. Nothing about that is under test
# here, so the one test that has to propagate a gradient uses a length past it.
GRADIENT_LENGTH = 0.05


@pytest.fixture(autouse=True)
def _quiet_gas_index():
    """Silence the gas index's extrapolation notice for every test in this file.

    Every capillary run asks its gas index for a band wider than any published fit
    covers, so the notice is reported once per model and says nothing about the run;
    ``tests/backends/test_numpy_ref_multimode.py`` silences it the same way. It is
    asserted on its own in ``tests/physdata/test_sellmeier.py``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        yield


# ------------------------------------------------------------- the walk-off warning


def _four_modes_over_a_metre_and_a_half(trange, modes):
    return Simulation.from_capillary(
        125e-6,
        1.5,
        "He",
        (0.8, 0.1),
        lambda0=800e-9,
        wavelength_limits=(90e-9, 4e-6),
        trange=trange,
        fwhm=7.5e-15,
        energy=100e-6,
        modes=modes,
        n_save=3,
        noise=None,
    )


def test_a_mode_set_walking_out_of_its_window_is_warned_about():
    """One frame for the set makes the walk-off real.

    Four HE_1m modes of a 125 um capillary spread by 346 fs of group delay over 1.5 m,
    the M2 case's own number, whatever the fill; on a 100 fs window the collar
    attenuates a packet drifting that fast by 1.7 nepers, a quarter of what stops it
    wrapping around, and the warning names the span.
    """
    with pytest.warns(SolpropWarning, match="walking off by 346 fs over 1.5 m"):
        _four_modes_over_a_metre_and_a_half(100e-15, 4)


@pytest.mark.parametrize(("trange", "modes"), [(400e-15, 4), (4e-12, 4), (400e-15, 1)])
def test_a_window_whose_collar_holds_the_walk_off_is_not_warned_about(trange, modes):
    """The M2 window absorbs the drift, the M4 window has room, one mode has none."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SolpropWarning)
        warnings.simplefilter("ignore", MaterialRangeWarning)
        _four_modes_over_a_metre_and_a_half(trange, modes)


def keywords(**overrides):
    """Return the cheap capillary's keywords, with any of them replaced."""
    return {
        "lambda0": LAMBDA0,
        "wavelength_limits": (200e-9, 2e-6),
        "trange": 100e-15,
        "fwhm": 10e-15,
        "energy": 20e-6,
        "n_r": 8,
        "n_save": 2,
        "rtol": 1e-6,
    } | overrides


def build(
    *,
    radius: float = RADIUS,
    length: float = SHORT,
    gas: str = "He",
    pressure: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    **overrides,
):
    """Build the cheap capillary as a Simulation, without running it."""
    return Simulation.from_capillary(
        radius, length, gas, pressure, **{"noise": None, **keywords(**overrides)}
    )


def run(
    *,
    radius: float = RADIUS,
    length: float = SHORT,
    gas: str = "He",
    pressure: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    **overrides,
):
    """Run the cheap capillary through the whole interface."""
    return prop_capillary(
        radius, length, gas, pressure, **{"noise": None, **keywords(**overrides)}
    )


# ------------------------------------------------------------------ the three fills


def test_a_number_is_a_constant_fill():
    profile = build(pressure=0.8).profiles["pressure"]
    assert profile.is_constant
    np.testing.assert_allclose(profile(np.array([0.0, SHORT])), 0.8)


def test_a_pair_is_the_two_point_gradient():
    # A millimetre carrying a bar of difference is far from laminar, which the fill
    # says; the numerical floor keeps the declared ends of Luna's law.
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        profile = build(pressure=(2.0, 1.0), vacuum_end="floor").profiles["pressure"]
    got = profile(np.array([0.0, SHORT / 2, SHORT]))
    # Luna's steady-flow law, p(z) = sqrt(p0^2 + z/L (p1^2 - p0^2)), exactly.
    expected = np.sqrt(4.0 + np.array([0.0, 0.5, 1.0]) * (1.0 - 4.0))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_a_pair_of_sequences_is_a_staged_fill():
    stages = ([0.0, SHORT / 2, SHORT], [1.0, 3.0, 1.0])
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        profile = build(pressure=stages, vacuum_end="floor").profiles["pressure"]
    got = profile(np.array([0.0, SHORT / 2, SHORT]))
    np.testing.assert_allclose(got, [1.0, 3.0, 1.0], rtol=1e-12)


def test_a_vacuum_end_is_floored_and_says_so(caplog):
    # The floor is a documented choice, so it is logged, not silent.
    with (
        caplog.at_level(logging.WARNING, logger="solprop.profiles"),
        pytest.warns(GasFlowWarning, match="Reynolds"),
    ):
        profile = build(pressure=(1.0, 0.0), vacuum_end="floor").profiles["pressure"]
    assert float(profile(np.array([SHORT]))[0]) > 0.0
    assert any("raised from" in record.message for record in caplog.records)


def test_a_true_vacuum_end_is_reached_by_asking_for_it():
    # Luna's gradient goes to zero exactly, so a comparison against one needs this.
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        simulation = build(pressure=(1.0, 0.0), pressure_floor=0.0, vacuum_end="floor")
    profile = simulation.profiles["pressure"]
    assert float(profile(np.array([SHORT]))[0]) == 0.0


def test_a_linear_fill_to_exact_vacuum_runs_and_ends_in_vacuum():
    # Fitted alone, helium's density table at 1.5 bar rounds to -1.0e11 m^-3 at this
    # fill's vacuum end and the translation refused it (SEP 3, Q23); the end now
    # tabulates to an exact zero. Ten steps, under the weak collar of `_flow_run`.
    simulation = _flow_build(
        length=SHORT,
        pressure=Linear(1.5, 0.0, SHORT),
        schedule=FixedSchedule(np.full(10, SHORT / 10)),
        backend="numpy_ref",
    )
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=1.0)
    )
    density = np.asarray(simulation.run(progress=None)["stats/density"])
    assert density[0] > 0.0
    assert density[-1] == 0.0


def test_an_unreadable_pressure_is_refused_by_name():
    with pytest.raises(DriverError, match="p_in, p_out"):
        build(pressure=(1.0, 2.0, 3.0))


def test_a_malformed_staged_fill_is_refused_by_the_profile_that_knows_the_rule():
    # One pressure for two positions: the profile names the rule, so the driver does
    # not restate it in a message of its own.
    with pytest.raises(TranslationError, match="one pressure per position"):
        build(pressure=([0.0, SHORT], [1.0]))


# ------------------------------------------------------------------- the mode set


def capillary_modes(simulation) -> tuple[MarcatiliMode, ...]:
    """Return a simulation's modes as the capillary modes they are."""
    return cast("tuple[MarcatiliMode, ...]", simulation.modes)


def test_an_integer_builds_that_many_he1m_modes():
    simulation = build(modes=3)
    assert len(simulation.modes) == 3
    assert [mode.m for mode in capillary_modes(simulation)] == [1, 2, 3]


def test_a_given_set_is_adopted_as_it_is():
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=m, cladding=1.45) for m in (1, 2)]
    simulation = build(modes=modes)
    modes = capillary_modes(simulation)
    assert [mode.cladding_index for mode in modes] == [1.45, 1.45]


def test_one_given_mode_needs_no_orthonormality_check():
    # A set of one is orthonormal by construction, and the check integrates adaptively,
    # so it is not paid for nothing.
    simulation = build(modes=[MarcatiliMode(RADIUS, "He", 1.0, m=2)])
    assert len(simulation.modes) == 1
    assert capillary_modes(simulation)[0].m == 2


def test_a_set_that_is_not_orthonormal_is_refused():
    # A mode and a partner turned by 0.3 rad are two distinct channels whose
    # overlap is cos(0.3), and one synthesis matrix and its dual cannot represent
    # that. Two *copies* of one mode never reach this gate: a collection refuses a
    # repeated channel by its label first.
    mode = MarcatiliMode(RADIUS, "He", 1.0, m=1)
    with pytest.raises(DriverError, match="not an orthonormal set"):
        build(modes=[mode, mode.rotated(0.3)])
    with pytest.raises(ModeError, match="duplicate physical channel"):
        build(modes=[mode, mode])


def test_the_input_enters_the_lowest_mode_alone():
    simulation = build(modes=3)
    state = simulation.translate().initial.field0[0]
    assert np.any(state[0] != 0.0)
    assert not np.any(state[1:] != 0.0)


# ------------------------------------------------------------------ the quadrature


def test_the_default_radial_order_follows_the_mode_count():
    assert default_radial_order(1) == 32
    assert default_radial_order(4) == 32
    assert default_radial_order(8) == 64


def test_the_rule_is_kronrod_so_the_alarm_has_an_estimate():
    rule = build(modes=2, n_r=None).settings.quadrature
    assert rule is not None
    assert rule.kronrod
    assert rule.has_embedded_rule
    assert rule.n_nodes == 33


def test_azimuthal_nodes_are_refused_for_a_radial_set():
    with pytest.raises(DriverError, match="azimuthal order zero"):
        build(modes=2, n_theta=16)


def test_azimuthal_nodes_are_taken_when_a_mode_has_angular_structure():
    # HE_21 has azimuthal order one, so its quartic reaches degree four and the rule
    # has to resolve the angle; the exactness bound is 4 * 1 + 1. No plasma, because
    # this mode is dark on the axis (below).
    simulation = build(
        modes=[MarcatiliMode(RADIUS, "He", 1.0, n=2, m=1)], n_theta=8, plasma=False
    )
    rule = simulation.settings.quadrature
    assert rule is not None
    assert rule.full
    assert rule.n_theta == 8


def test_an_input_cannot_be_placed_in_a_mode_that_is_not_propagated():
    # Unreachable through the interface, which builds one pulse; a guard on the helper
    # so that a second input route cannot arrive without noticing the set is too small.
    with pytest.raises(DriverError, match="cannot be placed"):
        _in_lowest_mode(np.zeros((3, 1, 8), dtype=complex), 2)


# ------------------------------------------------------------------- the responses


# The default rate is optmat's for every built-in gas since SEP 3's Phase 4a; the
# registry's policy (PPT for atoms, ADK and isotropic MO-ADK for molecules) was deleted
# with its formulas at Phase 4b. optmat's rates have their own tests in
# tests/driver/test_optmat_rates.py; what is left here is the wiring.


@pytest.mark.parametrize("gas", ["He", "CH4", "H2", "D2", "N2", "O2"])
@pytest.mark.parametrize("plasma", [None, True])
def test_the_default_plasma_rate_is_optmats_for_atoms_and_molecules(gas, plasma):
    rate = _rate(build(gas=gas, plasma=plasma))
    assert isinstance(rate, OptmatRate)
    assert rate.species_identity == gas
    assert rate.wavelength == LAMBDA0


@pytest.mark.parametrize("gas", ["H2", "D2", "N2", "O2"])
def test_molecular_plasma_choices_remain_explicit(gas):
    rate = AdkOracle(gas, 2.5e-18, hydrogenic=True)
    assert _rate(build(gas=gas, plasma=rate)) is rate
    # An optmat rate keeps its model; the driver binds the input's peak field and the
    # tabulated span to it, which is all that changes.
    chosen = OptmatRate(gas, LAMBDA0, model="moadk_tong_2002")
    bound = _rate(build(gas=gas, plasma=chosen))
    assert isinstance(bound, OptmatRate)
    assert (bound.source, bound.wavelength, bound.model) == (gas, LAMBDA0, chosen.model)
    assert bound.reference_field is not None
    for simulation in (build(gas=gas, plasma=False), build(gas=gas, envelope=True)):
        assert not any(hasattr(one, "rate") for one in simulation.responses)


@pytest.mark.parametrize("gas", ["H2", "D2", "N2", "O2"])
def test_capillary_interface_default_matches_explicit_optmat_rate(gas):
    # A short plasma-only run exercises the selected rate through translation
    # and propagation. Identical explicit/automatic inputs make one problem, and on a
    # prescribed schedule at a fixed order one march, so the two runs agree within
    # its rounding budget (SEP 25); a different rate would move them far more.
    options = {
        "gas": gas,
        "radius": 30e-6,
        "length": 1e-4,
        "energy": 8e-6,
        "kerr": False,
        "loss": False,
        "backend": "numpy_ref",
        "schedule": FixedSchedule.uniform(0.0, 1e-4, 1e-5, save_planes=[]),
        "integrator": FIXED_ORDER,
        # Ten steps of 10 um are past the in-stage collar's explicit limit; split
        # off, it is integrated exactly.
        "absorber_scheme": "split_step",
    }
    automatic = run(**options)
    explicit = run(**options, plasma=OptmatRate(gas, LAMBDA0))
    assert_matched(
        explicit.fields("fft"),
        automatic.fields("fft"),
        simulation_bound(build(**options)),
    )
    density = np.asarray(automatic["stats/electron_density"])
    assert np.all(np.isfinite(density))
    assert float(np.max(density)) > 0.0


def _rate(simulation):
    """Return the ionisation rate of a simulation's plasma response."""
    return next(
        response.rate for response in simulation.responses if hasattr(response, "rate")
    )


def _raman(simulation) -> RamanResponse:
    """Return a simulation's Raman response, typed: `responses` is a list of objects."""
    return next(
        cast("RamanResponse", response)
        for response in simulation.responses
        if isinstance(response, RamanResponse)
    )


def test_a_rate_object_is_used_as_it_is():
    rate = adk_oracle("He")
    assert _rate(build(plasma=rate)) is rate


def test_plasma_defaults_to_on_for_a_field_grid():
    assert any(hasattr(one, "rate") for one in build().responses)
    assert not any(hasattr(one, "rate") for one in build(plasma=False).responses)


def kerr_of(simulation) -> KerrResponse:
    """Return a simulation's Kerr response, which is always the first."""
    return cast("KerrResponse", simulation.responses[0])


def test_thg_defaults_to_on_for_a_field_grid():
    assert kerr_of(build()).thg
    assert not kerr_of(build(thg=False)).thg


def test_a_run_with_no_nonlinearity_is_refused():
    with pytest.raises(DriverError, match="no nonlinearity"):
        build(kerr=False, plasma=False)


# ------------------------------------------------------------- the CtC keywords
#
# The keywords milestone M4's target asks for, in solprop's names. Luna's call is
#
#     prop_capillary(16e-6, 1.5, :H2, 40.0; lambda0=515e-9,
#                    pulses=Pulses.GaussPulse(lambda0, tau_fwhm=400e-15, energy=10e-6),
#                    lambda_lims=(170e-9, 4e-6), trange=4e-12, loss=false, modes=6)
#
# and every keyword of it already existed here except one: Luna turns the Raman response
# on by itself for a molecular gas, so `raman=True` is solprop's only addition to the
# call. Nothing below propagates -- a 1.5 m six-mode field-grid run is the benchmark at
# the end of this file -- but the grid, the mode set, the quadrature, the rate and the
# compiled responses are all the ones the real case gets.

CTC = {
    "radius": 16e-6,
    "gas": "H2",
    "pressure": 40.0,
    "lambda0": 515e-9,
    "wavelength_limits": (170e-9, 4e-6),
    "trange": 4e-12,
    "fwhm": 400e-15,
    "energy": 10e-6,
    "modes": 6,
    "loss": False,
    "raman": True,
}


def ctc_build(length=SHORT, *, luna=False, **overrides):
    """Build the CtC case's keywords at whatever length, without running it.

    By default the keywords as a user writes them, on optmat's hydrogen. With `luna`,
    the fill and the rate are the witness's: Luna's material inputs supplied
    explicitly (:func:`luna_hydrogen_fill` and :func:`luna_hydrogen_rate`), the
    inputs the stored runs were compared on.
    """
    case: dict[str, Any] = CTC | overrides
    radius, gas, pressure = case.pop("radius"), case.pop("gas"), case.pop("pressure")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        if luna:
            temperature = case.pop("temperature", CTC_WITNESS["temperature_k"])
            return Simulation.from_capillary(
                radius,
                length,
                luna_hydrogen_fill(pressure, temperature),
                n_save=2,
                temperature=temperature,
                plasma=(plasma := case.pop("plasma", luna_hydrogen_rate())),
                # The witnesses were made by Luna's cumulative trapezoid (SEP 18).
                plasma_integration=None if plasma is False else "trapezoid",
                **{"noise": None, **case},
            )
        return Simulation.from_capillary(
            radius, length, gas, pressure, n_save=2, **{"noise": None, **case}
        )


@pytest.mark.parametrize("luna", [False, True])
def test_the_ctc_keywords_build_the_case_luna_runs(luna):
    simulation = ctc_build(luna=luna)
    # Kerr, plasma and Raman: Luna turns the last two on by itself, from the gas being
    # molecular and the grid being a field grid, and solprop only the plasma. The
    # Raman response is optmat's for the named gas, the supplied molecular one for
    # the witness's own hydrogen.
    kinds = [type(response).__name__ for response in simulation.responses]
    raman = "RamanResponse" if luna else "OptmatRamanResponse"
    assert kinds == ["KerrResponse", "PlasmaResponse", raman]
    # The realised grid is solprop's default for these keywords: the 4 ps request
    # widened so that the temporal collar lies outside it, on the cheapest smooth
    # transform size. Luna's own grid, 32768 samples on 16385 bins, is the witness
    # grid (`WITNESS_GRID_RULES`), which the stored comparisons are run on.
    assert (simulation.grid.n_t, simulation.grid.n_omega) == (27648, 13825)


def test_the_ctc_keywords_select_optmats_rate_unless_the_comparison_rate_is_given():
    assert isinstance(_rate(ctc_build()), OptmatRate)
    rate = _rate(ctc_build(luna=True))
    assert isinstance(rate, AdkOracle)
    # The form Luna's own molecular runs use, on the potential alone, and the one the
    # stored witness is sampled from.
    assert rate.hydrogenic
    assert rate.occupancy == 2
    assert rate.ionisation_potential == LUNA_HYDROGEN_POTENTIAL


def test_the_ctc_raman_response_is_hydrogens_two_terms():
    simulation = ctc_build(luna=True)
    raman = _raman(simulation)
    assert [type(model).__name__ for model in raman.models] == [
        "RotationalLines",
        "DampedOscillator",
    ]
    # An absolute molecular response, so no fraction is taken out of the Kerr term.
    assert raman.fraction is None
    problem = simulation.translate()
    assert problem.flags.has_raman
    assert ResponseStage.RAMAN_DRIVE in problem.responses.response_order


def test_six_modes_take_eight_radial_nodes_each():
    # The default was measured up to four modes and its slope above that is not;
    # six modes is the first set that exercises it, and this is the order the M4 case
    # and the stored benchmark both run at.
    assert default_radial_order(6) == 48
    rule = ctc_build().settings.quadrature
    assert rule is not None
    assert rule.n_nodes == 49


def test_the_ctc_temperature_reaches_the_rotational_populations():
    # The rotational ladder is Boltzmann-weighted, and for hydrogen those lines carry
    # the bulk of the Raman response, so a run's own temperature has to reach it.
    warm = cast(
        "RotationalLines", _raman(ctc_build(luna=True, temperature=400.0)).models[0]
    )
    cool = cast(
        "RotationalLines", _raman(ctc_build(luna=True, temperature=294.0)).models[0]
    )
    assert warm.temperature == 400.0
    assert not np.array_equal(warm.populations, cool.populations)


# ------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("gas", ["H2", "O2"])
def test_raman_true_adds_the_molecular_response_of_the_fill(gas):
    # plasma=False so that this is about the Raman response alone; the plasma beside it
    # is the CtC case's business, below.
    simulation = build(gas=gas, raman=True, plasma=False)
    kinds = [type(response).__name__ for response in simulation.responses]
    # optmat's response since SEP 3's Phase 4a, a RamanResponse of its own type.
    assert kinds == ["KerrResponse", "OptmatRamanResponse"]
    assert simulation.translate().flags.has_raman


def test_raman_true_is_refused_for_a_gas_without_raman_data():
    # optmat names the atom (SEP 3, Phase 1c).
    with pytest.raises(MaterialError, match="no molecular Raman response"):
        build(raman=True)


@pytest.mark.parametrize("gas", ["N2", "O2", "H2", "D2", "CH4", "SF6"])
def test_raman_is_on_by_default_for_a_molecule_with_data(gas):
    # raman=None means on where optmat's response is validated; False switches it off.
    kinds = [type(r).__name__ for r in build(gas=gas, plasma=False).responses]
    assert kinds == ["KerrResponse", "OptmatRamanResponse"]
    quiet = [
        type(r).__name__ for r in build(gas=gas, plasma=False, raman=False).responses
    ]
    assert quiet == ["KerrResponse"]


def test_raman_stays_off_for_an_atom_and_warns_for_a_molecule_without_data():
    assert [type(r).__name__ for r in build(plasma=False).responses] == ["KerrResponse"]
    with pytest.warns(SolpropWarning, match="N2O.*provisionally"):
        simulation = build(gas="N2O", plasma=False)
    assert [type(r).__name__ for r in simulation.responses] == ["KerrResponse"]
    # optmat's N2O response is provisional and admitted only under that policy.
    with pytest.raises(MaterialError, match="provisional"):
        build(gas="N2O", plasma=False, raman=True)


@pytest.mark.parametrize("name", ["air", "Air", "AIR"])
def test_air_is_a_nitrogen_oxygen_mixture_with_every_channel(name):
    """The convenience: 'air' is N2 and O2 with Kerr, plasma and Raman each."""
    simulation = build(gas=name, pressure=6.0)
    kinds = [type(r).__name__ for r in simulation.responses]
    labels = [getattr(r, "species", None) for r in simulation.responses]
    # optmat's one Raman response of the composition sits at its leaf, N2, and carries
    # both molecules' lines (SEP 3, D16).
    assert kinds == [
        "KerrResponse",
        "PlasmaResponse",
        "OptmatRamanResponse",
        "KerrResponse",
        "PlasmaResponse",
    ]
    assert labels == ["N2"] * 3 + ["O2"] * 2
    raman = simulation.responses[2]
    assert isinstance(raman, OptmatRamanResponse) and raman.labels == ("N2", "O2")
    mode = simulation.modes[0]
    assert isinstance(mode, MarcatiliMode)
    mixture = mode.mixture
    assert mixture is not None
    assert mixture.labels == ("N2", "O2")
    assert mixture.fractions is not None
    # optmat's CIPM-2007 "dry air (N2, O2)", renormalised over the two.
    named = composition_fractions("dry air (N2, O2)")
    np.testing.assert_allclose(
        mixture.fractions, (named["N2"], named["O2"]), rtol=1e-12
    )
    assert mixture.composition == "dry air (N2, O2)"
    with pytest.raises(DriverError, match="omit susceptibility"):
        build(gas="air", pressure=6.0, susceptibility=gamma("N2"))


def test_oxygen_raman_propagates_through_both_cpu_backends():
    options = {
        "gas": "O2",
        "radius": 30e-6,
        "length": 2e-3,
        "pressure": 6.0,
        "temperature": 295.0,
        "fwhm": 30e-15,
        "trange": 300e-15,
        "energy": 1e-6,
        "plasma": False,
        "loss": False,
        "rtol": 1e-10,
        "n_r": 24,
    }
    reference = run(**options, raman=True, backend="numpy_ref")
    compiled = run(**options, raman=True, backend="numba_cpu")
    without = run(**options, raman=False, backend="numpy_ref")
    scale = float(np.linalg.norm(reference.fields("fft")[-1]))
    assert scale > 0.0
    assert np.all(np.isfinite(compiled.fields("fft")))
    # Same equations, independent execution backends: the 1e-9 bound allows
    # accumulated integration error at rtol=1e-10. Radial quadrature is resolved.
    assert (
        np.linalg.norm(compiled.fields("fft")[-1] - reference.fields("fft")[-1]) / scale
        < 1e-9
    )
    # At this modest nonlinear exposure the Raman signal must be observable;
    # this guards accidentally omitting it while passing a backend parity test.
    assert (
        np.linalg.norm(without.fields("fft")[-1] - reference.fields("fft")[-1]) / scale
        > 1e-5
    )


def test_raman_true_needs_the_kerr_term_it_drives_beside():
    with pytest.raises(DriverError, match="needs kerr=True"):
        build(gas="H2", raman=True, kerr=False)


def test_a_vector_request_is_refused_as_scalar_only():
    with pytest.raises(DriverError, match="this path is scalar"):
        build(polarisation="circular")


def test_an_envelope_grid_with_explicit_plasma_selects_cycle_averaging():
    simulation = build(envelope=True, plasma=True)
    assert any(isinstance(one, EnvelopePlasmaResponse) for one in simulation.responses)
    assert not kerr_of(simulation).thg


def test_envelope_plasma_with_thg_is_refused_by_name():
    with pytest.raises(
        DriverError, match="envelope plasma cannot be combined with THG"
    ):
        build(envelope=True, plasma=True, thg=True)


def test_an_envelope_grid_turns_plasma_and_thg_off_by_itself():
    # Luna's defaults: `plasma` and `thg` are both `!envelope`, so an envelope run
    # asks for neither unless it is told to.
    simulation = build(envelope=True)
    assert not any(hasattr(one, "rate") for one in simulation.responses)
    assert not kerr_of(simulation).thg


def test_an_envelope_grid_without_plasma_is_allowed():
    simulation = build(envelope=True, plasma=False)
    assert isinstance(simulation.grid, EnvelopeGrid)
    assert not any(hasattr(one, "rate") for one in simulation.responses)


def test_a_field_grid_is_the_default():
    assert isinstance(build().grid, FieldGrid)


def test_a_grid_beside_the_keywords_that_build_one_is_refused():
    grid = FieldGrid.from_limits(SHORT, LAMBDA0, (200e-9, 2e-6), 100e-15)
    with pytest.raises(DriverError, match=r"not\s+both"):
        build(grid=grid)


def test_a_grid_of_ones_own_is_adopted():
    grid = FieldGrid.from_limits(SHORT, LAMBDA0, (200e-9, 2e-6), 100e-15)
    simulation = build(grid=grid, wavelength_limits=None, trange=None)
    assert simulation.grid is grid


# -------------------------------------------------------- the ionisation field range


def test_the_default_field_range_brackets_the_input_peak():
    """The range is anchored to a number the run itself reports at its entrance."""
    simulation = build(plasma=True)
    covered = simulation.settings.ionisation_field_range
    assert covered is not None
    low, high = covered
    # peak_intensity is eps0 c/2 times the square of the on-axis field amplitude
    # so inverting it recovers the field the range is built from.
    reported = math.sqrt(2.0 * _peak_intensity(simulation) / (EPS0 * C))
    assert low == pytest.approx(reported * 1e-3, rel=1e-12)
    assert high == pytest.approx(reported * 30.0, rel=1e-12)
    assert low < high


def _peak_intensity(simulation):
    """Return the peak on-axis intensity of a simulation's input, in W/m^2."""
    problem = simulation.translate()
    backend = NumpyRefBackend(
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend.set_initial(problem.initial.field0, None)
    return float(
        backend.stats(StatsSpec(names=("peak_intensity",)))["peak_intensity"][0]
    )


def test_the_field_range_is_not_derived_without_a_plasma():
    assert build(plasma=False).settings.ionisation_field_range is None


def test_an_explicit_field_range_wins():
    given = (1e8, 1e12)
    assert build(ionisation_field_range=given).settings.ionisation_field_range == given


def test_an_empty_input_has_no_range_to_cover():
    with pytest.raises(DriverError, match="carries no field"):
        default_field_range(0.0)


@pytest.mark.parametrize(
    ("kind", "order", "extra"),
    [("TE", 0, {}), ("TM", 0, {}), ("HE", 2, {"n_theta": 8})],
)
def test_a_mode_set_dark_on_the_axis_takes_its_range_from_the_nodes(kind, order, extra):
    """A family with no field on the axis still derives its range.

    ``TE_01``, ``TM_01`` and ``HE_21`` all carry ``J_1(j_11 r/a)``, zero on the axis,
    so the peak is read over the run's own nodes. Against ``HE_11`` launched with the
    same energy, the ratio of the two flux-normalised peaks is analytic:
    ``max J_1 * |J_1(j_01)| / |J_2(j_11)|`` (each profile normalised by
    ``int_0^1 J_n(u x)^2 x dx = J_{n+1}(u)^2 / 2`` at a zero of ``J_n``).
    """
    dark = [MarcatiliMode(RADIUS, "He", 1.0, kind=kind, n=order, m=1)]
    covered = build(modes=dark, plasma=True, **extra).settings.ionisation_field_range
    reference = build(plasma=True).settings.ionisation_field_range
    assert covered is not None
    assert reference is not None
    j01 = jn_zeros(0, 1)[0]
    j11 = jn_zeros(1, 1)[0]
    peak_of_j1 = -float(
        minimize_scalar(
            lambda x: -jv(1, x),
            bounds=(1.0, 3.0),
            method="bounded",
            options={"xatol": 1e-10},
        ).fun
    )
    expected = peak_of_j1 * abs(jv(1, j01)) / abs(jv(2, j11))
    ratio = covered[0] / reference[0]
    # The nodes sample a smooth maximum, so they can only fall short of it, by the
    # square of their spacing there: measured 1.9e-3 with the default rule. The
    # two modes' n_eff differ by ~1e-5, which moves their true-power launch by less.
    assert ratio <= expected * (1.0 + 1e-4)
    assert ratio == pytest.approx(expected, rel=1e-2)
    given = build(modes=dark, ionisation_field_range=(1e8, 1e12), **extra)
    assert given.settings.ionisation_field_range == (1e8, 1e12)


# ------------------------------------------------------------------- the statistics


def stat_names(simulation) -> tuple[str, ...]:
    """Return the names a simulation's statistics spec asks for."""
    spec = simulation.settings.stats
    assert spec is not None
    return spec.names


def test_the_default_spec_carries_the_profiles_and_the_backend_statistics():
    names = stat_names(build())
    assert {"density", "pressure", "core_radius"} <= set(names)
    assert set(BACKEND_STATISTICS) <= set(names)


def test_a_run_without_plasma_does_not_ask_for_the_plasma_statistics():
    names = stat_names(build(plasma=False))
    assert not set(PLASMA_STATISTICS) & set(names)
    assert "quadrature_error" in names


def test_the_profile_statistics_reach_the_output():
    output = run(length=GRADIENT_LENGTH, pressure=(2.0, 1.0))
    assert float(output["stats/pressure"][0]) == pytest.approx(2.0)
    assert float(output["stats/core_radius"][0]) == pytest.approx(RADIUS)
    assert float(output["stats/density"][0]) > 0.0


# ------------------------------------------------------------------ planes and files


def test_n_save_counts_the_entrance():
    assert run(n_save=4)["z"].size == 4


def test_save_planes_and_n_save_are_exclusive():
    with pytest.raises(DriverError, match="not both"):
        run(save_planes=[SHORT], n_save=3)


def test_explicit_planes_are_used_as_given():
    planes = [SHORT / 2, SHORT]
    np.testing.assert_allclose(run(save_planes=planes, n_save=None)["z"][1:], planes)


def test_a_path_is_written_and_read_back(tmp_path):
    path = tmp_path / "capillary.h5"
    handler = run(output=path)
    assert handler.n_saved == 2
    with open_file(path) as stored:
        assert stored["field"].shape[1] == 1
        assert "stats/peak_intensity" in _names(stored)


def _names(handler):
    """Return every key a handler will answer to, for a membership assertion."""
    return {"z", "field", *(f"stats/{name}" for name in handler.stat_names)}


def test_memory_is_the_default_handler():
    assert isinstance(run(), MemoryOutput)


class _Interrupt(NullReporter):
    """A progress reporter that stops a run once it has saved enough planes."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise _CrashError(f"stopped after plane {self.seen}")

    def finished(self, total):  # pragma: no cover - the run never gets this far
        raise AssertionError("an interrupted run does not finish")


class _CrashError(RuntimeError):
    """What an interrupted run raises, standing in for a machine going away."""


def test_a_replayed_schedule_resumes_onto_the_same_trajectory(tmp_path):
    """The same configuration either way, so the file continues rather than is refused.

    A replayed schedule takes the same steps whatever it is resumed from: the step
    sizes, the positions and the counts it prescribes are exactly equal here. The
    planes written before the interruption are a second run of the same fixed-order
    march, and where the run keeps the march's state (the host engines, ADR-046) so
    is the plane computed *after* the resume: both within the march's rounding budget
    (SEP 25). An engine that starts the Adams history afresh on a resume (Kokkos)
    restarts at order one, and that plane differs by the step error: 2.9e-10 of the
    peak, measured, bounded below with a margin.
    """
    path = tmp_path / "resumed.h5"
    planes = [SHORT / 3, 2 * SHORT / 3, SHORT]
    # The collar split off: ADR-025's in-stage absorber makes the collar's ceiling rate
    # an eigenvalue of the linearised problem, so an explicit method has to keep
    # ``h alpha / 2`` inside its stability region, 0.0877 on the negative real axis
    # for the default Adams history. On this fibre that is a step of 0.7 um, and
    # twelve steps of 83 um diverge; integrated exactly, the collar takes any step.
    schedule = FixedSchedule.uniform(0.0, SHORT, SHORT / 12, save_planes=planes)
    settings = {
        "save_planes": planes,
        "n_save": None,
        "schedule": schedule,
        "integrator": FIXED_ORDER,
        "absorber_scheme": "split_step",
    }
    bound = simulation_bound(build(**settings))
    whole = run(**settings)
    with pytest.raises(_CrashError):
        run(output=path, progress=_Interrupt(2), **settings)
    with open_file(path) as partial:
        # The entrance plus the two planes the reporter saw before it stopped.
        assert partial["z"].size == 3
    continued = resume(path).march is not None
    run(output=path, resume=True, **settings)
    with open_file(path) as stored:
        np.testing.assert_array_equal(stored["z"], whole["z"])
        for key in ("step_history/h", "step_history/z", "step_history/counts"):
            np.testing.assert_array_equal(stored[key], whole[key], err_msg=key)
        # The planes the interrupted run had already written are untouched by the
        # resume; the last one is the one it recomputed.
        assert_matched(stored["field"][:3], whole["field"][:3], bound, what="field")
        if continued:
            assert_matched(stored["field"], whole["field"], bound, what="field")
            assert_matched(
                stored["stats/energy"],
                whole["stats/energy"],
                STATISTIC_FACTOR * bound,
                what="energy",
            )
        else:
            peak = float(np.max(np.abs(whole["field"])))
            worst = float(np.max(np.abs(stored["field"] - whole["field"]))) / peak
            assert worst < 1e-8
            np.testing.assert_allclose(
                stored["stats/energy"], whole["stats/energy"], rtol=1e-8
            )


# ===================================================== the intermodal Kerr coupling
#
# Which code is right about the higher modes: a comparison of the two once found Luna
# putting two to four orders of magnitude more energy into the higher HE_1m modes of a
# multimode capillary than solprop, while the two agreed on the fundamental (the fault
# was solprop's). Everything in this section is fast, and none of it compares solprop
# with Luna -- each code is measured against an
# analytic answer instead, so that "they disagree" can become "this one is wrong".
#
# A field that is the fundamental alone, `a_1(t) ehat_1(r)`, drives a Kerr polarisation
# `ehat_1(r)^3 f(t)` whose projection onto mode p is `T_p f(t)` with
#
#     T_p = int ehat_1^3 ehat_p dA,   ehat_m = J_0(u_m r/a) / sqrt(N_m).
#
# The *ratio* `T_p / T_1` is therefore the whole of the intermodal coupling at this
# order: it survives every prefactor, the response, the shock term, the third harmonic
# and the transform normalisation, and it is a pure Bessel number. `T_2/T_1` is
# 0.4785423.


def he1m_set(count):
    """Return the first `count` HE_1m modes of the M2 geometry, at a constant fill."""
    return tuple(MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in range(1, count + 1))


@functools.cache
def overlap_tensor(count):
    """Return the quartic overlap tensor of the first `count` HE_1m modes.

    ``Gamma_plmn = int ehat_p ehat_l ehat_m ehat_n dA`` with
    ``ehat_m = J_0(u_m r/a) / sqrt(N_m)``, by adaptive quadrature. Deliberately
    independent of everything under test: the zeros come from
    ``scipy.special.jn_zeros``, the profiles from ``scipy.special.jv``, and both the
    norms and the quartics from ``scipy.integrate.quad``, so nothing in solprop's mode,
    normalisation or fixed rule is reused to check solprop's mode, normalisation or
    fixed rule. The angular factor of an HE_1m mode has unit modulus, so the whole
    cross-section integral is ``2 pi int_0^a ... r dr``, and the tensor is symmetric in
    all four indices, so only the sorted entries are integrated.

    With modal amplitudes in the square root of watts the Kerr drive projected into
    mode p is ``eps0 chi3 rho sum_lmn Gamma_plmn a_l a_m a_n``
    (``docs/guide/media/nonlinear.md``).
    """
    zeros = jn_zeros(0, count)

    def profile(index, radius):
        return jv(0, zeros[index] * radius / RADIUS)

    def radial(entry, power=1):
        """Integrate a product of profiles over the cross-section."""
        integrand = lambda r: (  # noqa: E731
            math.prod(profile(index, r) for index in entry) ** power * 2.0 * np.pi * r
        )
        return quad(integrand, 0.0, RADIUS, limit=400)[0]

    # N = (1/2) eps0 c int |e|^2 dA: the power carried by unit field amplitude
    # (docs/dev/conventions.md section 4).
    norms = [0.5 * EPS0 * C * radial((index,), power=2) for index in range(count)]
    values = {}
    tensor = np.zeros((count,) * 4)
    for entry in np.ndindex(tensor.shape):
        sorted_entry = tuple(sorted(entry))
        if sorted_entry not in values:
            scale = math.sqrt(math.prod(norms[index] for index in sorted_entry))
            values[sorted_entry] = radial(sorted_entry) / scale
        tensor[entry] = values[sorted_entry]
    return tensor


def independent_overlaps(count):
    """Return ``T_p / T_1`` with ``T_p = int ehat_1^3 ehat_p dA``, the drive ratios."""
    drives = overlap_tensor(count)[0, 0, 0, :]
    return drives / drives[0]


def projected_drive(modes, *, n_r, radial_rule: RadialRule = "gauss_kronrod"):
    """Return the Kerr drive projected into each mode by a field that is mode 1 alone.

    The production path and nothing else: :func:`solprop.modes.quadrature.mode_matrix`
    for the two matrices, then passes 3, 4 and 5 of the reference backend, on a state
    of one time sample carrying unit amplitude in the fundamental. The Kerr constant is
    set to one because only the ratio between modes is wanted, and the real cubic is
    the form a field grid compiles (``translate.responses``).
    """
    rule = transverse_quadrature(modes, n_r=n_r, radial_rule=radial_rule)
    matrices = mode_matrix(modes, rule, components=("x",))
    state = np.zeros((len(modes), 1, 1))
    state[0, 0, 0] = 1.0
    field = passes.synthesise(state, matrices.synthesis, 1.0, matrices.n_pol)
    coefficients = KerrCoefficients(1.0, "real", 0.0, 0.0, 0.0, False, True)
    polarisation = passes.kerr_polarisation(field, coefficients, 1.0)
    return passes.project(polarisation, matrices.projection, 1.0, matrices.n_pol)[
        :, 0, 0
    ]


@pytest.mark.parametrize("count", [2, 4])
def test_the_projected_drive_is_the_analytic_quartic_overlap(count):
    """solprop's inter-mode Kerr drive is the Bessel overlap, to the rule's own floor.

    The reduced M2 case's two modes and the full case's four, at the radial order the
    driver picks for each. The bound is the measured agreement with a margin: 8.9e-16
    on two modes and 1.6e-13 on four, where the four-mode figure is the reference's
    own accuracy -- ``quad`` is asked for the *smallest* ratio, -2.4e-3 for mode four,
    against integrands of order one. The 0.4785423 is the second entry.
    """
    modes = he1m_set(count)
    expected = independent_overlaps(count)
    drive = projected_drive(modes, n_r=default_radial_order(count))
    np.testing.assert_allclose(drive / drive[0], expected, rtol=1e-12, atol=0.0)
    assert float(drive[1] / drive[0]) == pytest.approx(0.4785423, abs=5e-8)


def test_the_drive_ratio_converges_with_the_radial_order():
    """The four-mode ratios are converged well below the order the driver picks.

    Measured on the full case's mode set: 9.6e-3 at ``n_r = 8``, 7.7e-11 at 16 and
    1.6e-13 at 32, which is the reference's floor: converged from ``n_r = 16``.
    """
    expected = independent_overlaps(4)
    modes = he1m_set(4)
    drives = [projected_drive(modes, n_r=n_r) for n_r in (8, 16, 32)]
    worst = [
        float(np.max(np.abs(drive / drive[0] / expected - 1.0))) for drive in drives
    ]
    assert worst[0] < 2e-2
    assert worst[1] < 1e-9
    assert worst[2] < 1e-12


def test_the_fundamental_drive_is_the_effective_area_in_closed_form():
    """``T_1 = 4 / ((eps0 c)^2 A_eff)``: the tensor's tie to the mode-averaged limit.

    ``T_1 = int ehat_1^4 dA`` and ``A_eff = (int e^2)^2 / int e^4``, so the two are the
    same statement; it is the identity that makes a one-mode multimode run agree with a
    mode-averaged one, and Luna writes the same closed form. Equality here is exact to
    rounding, so the tolerance is a few ulp of the reduction rather than a physical
    bound.
    """
    mode = he1m_set(1)[0]
    rule = transverse_quadrature((mode,), n_r=32, radial_rule="gauss_kronrod")
    matrices = mode_matrix((mode,), rule, components=("x",))
    fields = matrices.synthesis[:, 0]
    weights = rule.weights(mode.dimlimits(0.0))
    quartic = float(np.sum(fields**4 * weights))
    assert quartic == pytest.approx(4.0 / ((EPS0 * C) ** 2 * mode.aeff(0.0)), rel=1e-14)


# ------------------------------------- oracle 1: Luna's own projection of a known field
#
# `tests/data/generate_luna_overlap.py` asks Luna's own `TransModalFixed` for the same
# ratio, by handing it a spectrum that lives in the fundamental alone and reading the
# drive it projects into every mode. Nothing here runs Julia.

OVERLAP = json.loads((Path(__file__).parents[1] / "data/luna_overlap.json").read_text())
OVERLAP_PROVENANCE = json.loads(
    (Path(__file__).parents[1] / "data/luna_overlap_provenance.json").read_text()
)
OVERLAP_CASES = [("REDUCED", 2), ("FULL", 4)]
OVERLAP_ORDERS = [(nr, rule) for nr in (64, 128, 256) for rule in ("true", "false")]

# Luna's constants are an older CODATA release: its eps0 is smaller than solprop's by
# 1.33e-10 relative, which is exactly the difference seen in its mode normalisations and
# profiles below. The generator emits the constants so this can be asserted rather than
# absorbed into a tolerance.
LUNA_CONSTANTS_OFFSET = 1.4e-10


def luna_ratios(tag, count, nr, rule):
    """Return the drive ratios Luna's transform realised on one quadrature."""
    return np.array(
        [OVERLAP[f"{tag}_{nr}_{rule}_RATIO_{p}"][0] for p in range(1, count + 1)]
    )


def test_the_overlap_witness_names_the_luna_it_came_from():
    """The stored oracle records which Luna, which Julia and which case built it."""
    assert len(OVERLAP_PROVENANCE["luna_commit"]) == 40
    assert OVERLAP_PROVENANCE["luna_branch"] == "modal-fixed"
    assert OVERLAP_PROVENANCE["julia_version"].startswith("julia version")
    assert OVERLAP_PROVENANCE["temperature_k"] == pytest.approx(294.0)
    assert OVERLAP_PROVENANCE["radius"] == pytest.approx(RADIUS)
    assert OVERLAP_PROVENANCE["reduced_modes"] == 2
    assert OVERLAP_PROVENANCE["full_modes"] == 4


def test_lunas_constants_are_an_older_codata_release():
    """The one difference in Luna's mode data, named rather than tolerated.

    ``c`` is exact in both; Luna's ``eps0`` is 1.33e-10 smaller than solprop's, and
    ``N`` carries ``eps0 c`` so its normalisations are off by that and its profiles by
    half of it. Nothing else in the mode layer differs at all.
    """
    speed, permittivity, _ = OVERLAP["CONSTANTS"]
    assert speed == C
    assert permittivity / EPS0 - 1.0 == pytest.approx(-1.332e-10, rel=1e-3)


@pytest.mark.parametrize(("tag", "count"), OVERLAP_CASES)
def test_lunas_own_projection_is_the_analytic_quartic_overlap(tag, count):
    """Luna's inter-mode Kerr drive is the same Bessel overlap solprop's is.

    This is the first oracle of this section, and it exonerates Luna's projection: at
    Luna's own default radial order its ``TransModalFixed`` reproduces the analytic
    ratios to 1.9e-12 -- 0.4785423, 0.0221346 and -0.0023723 -- which is solprop's
    answer to every digit either code can state. Whatever the higher-mode disagreement
    is, it is not the synthesis, the normalisation, the projection or the quadrature.

    The bound is the measured agreement with a margin. It is above the 1.6e-13 of
    solprop's own check because Luna's ratio is read off a propagated spectrum rather
    than one time sample, so it carries the transform's rounding as well as the rule's.
    """
    expected = independent_overlaps(count)
    np.testing.assert_allclose(
        luna_ratios(tag, count, 64, "true"), expected, rtol=1e-11, atol=0.0
    )


@pytest.mark.parametrize(("tag", "count"), OVERLAP_CASES)
def test_lunas_ratio_is_the_same_at_every_frequency(tag, count):
    """The ratio is frequency-independent in Luna's code, as it is on paper.

    A field that is the fundamental alone gives a polarisation that factorises into
    ``ehat_1(r)^3`` times a function of time, so everything after the projection --
    the time window, the transform, the spectral window and ``-i omega / 4`` -- is
    common to all modes and cancels. Measured: the spread of the ratio across the band
    where the fundamental's own drive is above 1e-8 of its peak is at most 8.5e-11, and
    its imaginary part at most 3.5e-14. If Luna's chain had a per-mode or
    frequency-dependent factor anywhere, this is where it would show.
    """
    for nr, rule in OVERLAP_ORDERS:
        for p in range(1, count + 1):
            _, imaginary, spread, bins = OVERLAP[f"{tag}_{nr}_{rule}_RATIO_{p}"]
            assert bins > 100, f"{tag} {nr} {rule} mode {p}: too few bins to mean over"
            assert abs(imaginary) < 1e-13
            assert spread < 1e-10


@pytest.mark.parametrize(("tag", "count"), OVERLAP_CASES)
def test_lunas_projection_does_not_move_with_its_quadrature_order(tag, count):
    """Luna's own quadrature ladder, measured rather than inspected.

    Three radial orders spanning 64 to 256 nodes, with and without the Kronrod
    extension Luna turns on by default, all agree with the analytic ratios to 6.8e-12.
    Luna's rule is converged at its default order, so the quadrature is ruled out on
    Luna's side by measurement rather than by inspection.
    """
    expected = independent_overlaps(count)
    for nr, rule in OVERLAP_ORDERS:
        np.testing.assert_allclose(
            luna_ratios(tag, count, nr, rule),
            expected,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"{tag} nr={nr} kronrod={rule}",
        )
    # Luna's own high-order control, from its mode matrices without the transform.
    np.testing.assert_allclose(
        OVERLAP[f"{tag}_REFERENCE_512"], expected, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize(("tag", "count"), OVERLAP_CASES)
def test_lunas_basis_is_orthonormal_on_its_own_rule(tag, count):
    """No linear leak: mode one does not reach mode p without the nonlinearity.

    ``(1/2) eps0 c S Wp`` is the discrete Gram matrix of Luna's normalised basis on its
    own quadrature. It is the identity to 8.9e-16, so a discrete non-orthogonality --
    which would put the fundamental into the higher modes with no Kerr term at all --
    is excluded. The bound is a few ulp of the reduction over the rule's nodes.
    """
    for nr, rule in OVERLAP_ORDERS:
        gram = np.reshape(OVERLAP[f"{tag}_{nr}_{rule}_GRAM"], (count, count), order="F")
        np.testing.assert_allclose(
            gram, np.eye(count), rtol=0.0, atol=1e-14, err_msg=f"{tag} {nr} {rule}"
        )


@pytest.mark.parametrize(("tag", "count"), OVERLAP_CASES)
def test_lunas_modes_are_solprops_modes(tag, count):
    """The normalisation, the effective area, the Bessel zero and the profile all agree.

    ``A_eff`` and ``u_nm`` are geometry alone and agree to rounding; ``N`` carries
    ``eps0 c`` and the normalised profile carries its square root, so both sit at
    Luna's older CODATA offset and no further. There is no m dependence in any of it.

    The witness ``EXY`` is Luna's **y** component, because Luna's ``ϕ = 0`` HE_1m mode
    is y-polarised; solprop's is x-polarised (ADR-031), so the same numbers are read
    out of component 0 here. That quarter turn is a deliberate convention divergence,
    not a defect: a circular capillary is rotationally symmetric and the profile
    compared is the same function of ``r``.
    """
    modes = he1m_set(count)
    fractions = np.asarray(OVERLAP_PROVENANCE["profile_fractions"])
    points = np.stack([fractions * RADIUS, np.zeros_like(fractions)], axis=-1)
    np.testing.assert_allclose(
        OVERLAP[f"{tag}_UNM"], [mode.u_nm for mode in modes], rtol=1e-14
    )
    np.testing.assert_allclose(
        OVERLAP[f"{tag}_AEFF"], [mode.aeff(0.0) for mode in modes], rtol=1e-14
    )
    np.testing.assert_allclose(
        OVERLAP[f"{tag}_N"],
        [mode.normalisation(0.0) for mode in modes],
        rtol=LUNA_CONSTANTS_OFFSET,
    )
    for index, mode in enumerate(modes, start=1):
        ours = mode.field(points, 0.0)[:, 0] / math.sqrt(mode.normalisation(0.0))
        np.testing.assert_allclose(
            OVERLAP[f"{tag}_EXY_{index}"],
            ours,
            rtol=LUNA_CONSTANTS_OFFSET,
            err_msg=f"{tag} mode {index}",
        )


# ------------------------------ oracle 2: the undepleted-pump two-mode solution
#
# A weak pump in the fundamental, a weak seed in the second mode, a constant fill, no
# plasma, and a length short enough that the first order of perturbation theory holds.
# Each code is run twice, with the Kerr term and with it switched off by a zero
# susceptibility, so the *nonlinear change* of each mode is a difference of two runs of
# the same code: no frame, transform or amplitude convention has to be modelled, because
# every one of them cancels between the two runs and again between the two modes.
#
# The reference is the first-order solution of the coupled-mode equations,
#
#     dA_p/dz = D_p(omega) A_p + N_p(omega) F[sum_lmn Gamma_plmn a_l a_m a_n],
#
# integrated once along z with the pump propagating linearly. Its only content is the
# overlap tensor and the modes' propagation constants, and the ratio between two modes
# drops the Kerr constant, chi3, the shock term, the third harmonic and the transform
# normalisation with them. What is left is what the higher-mode disagreement is about.
#
# ``exp(D_p L)`` is not modelled either: it is *measured*, bin by bin, as the ratio of
# the linear run's exit to its input -- which is what the seed is for. Without a seed
# the second mode has no input to divide by and its linear propagation is unknown.

UNDEPLETED = OVERLAP_PROVENANCE["undepleted"]

# Where the pump's own spectrum is above this fraction of its peak. Below it the ratio
# of two nonlinear changes is a ratio of round-off.
UNDEPLETED_BAND = 1e-5

# Bins whose amplitude is below this fraction of the mode's peak carry no measurable
# linear transfer; they also carry no field, so the reference propagates them unchanged.
TRANSFER_FLOOR = 1e-8

# Gauss-Legendre nodes for the z integral and the oversampling of the cubic. Sixteen
# nodes integrate the reference to below the second-order term it is being compared
# against; the pad keeps the third harmonic off the fold.
FIRST_ORDER_NODES = 16
FIRST_ORDER_PAD = 2


def undepleted_grid():
    """Return the grid both codes run the undepleted-pump case on."""
    return FieldGrid.from_limits(
        UNDEPLETED["length"],
        OVERLAP_PROVENANCE["lambda0"],
        (UNDEPLETED["lambda_min"], UNDEPLETED["lambda_max"]),
        UNDEPLETED["trange"],
        **WITNESS_GRID_RULES,
    )


def undepleted_solprop(*, kerr, energy):
    """Run solprop's side of the undepleted-pump case, with or without the Kerr term.

    ``prop_capillary`` puts the whole input in the lowest mode, so the seeded initial
    field is built here and handed to :class:`solprop.driver.Simulation` directly; the
    grid, the modes, the response, the profiles and the numerics are the ones
    ``from_capillary`` builds, so only the input and the collar depart from the
    interface.

    **No temporal collar.** The oracle this case is measured against is the first-order
    solution of the coupled-mode equations through a flow that is diagonal in omega, and
    an absorber is diagonal in *t* instead: whatever it does to the perturbation the
    oracle cannot model. Over two centimetres nothing walks anywhere near the window's
    edge, so the collar has nothing to absorb and leaving it out measures the intermodal
    transfer and nothing else. With one, the two ADR-025 schemes disagree about the
    measurement at the 1e-5 level -- the split factor acts between steps and cancels in
    the difference of the two runs, the in-stage term acts on the perturbation itself --
    and neither number is about the physics being tested. Without one they agree to four
    digits, which is what this comment is here to record.
    """
    grid = undepleted_grid()
    base = Simulation.from_capillary(
        RADIUS,
        UNDEPLETED["length"],
        OVERLAP_PROVENANCE["gas"],
        UNDEPLETED["pressure"],
        grid=grid,
        lambda0=OVERLAP_PROVENANCE["lambda0"],
        fwhm=UNDEPLETED["fwhm"],
        energy=energy,
        modes=2,
        plasma=False,
        raman=False,
        n_save=2,
        rtol=UNDEPLETED["rtol"],
        atol=UNDEPLETED["atol"],
        # Deliberately solprop's own helium and silica (optmat's recommended models)
        # at solprop's room temperature, 293.15 K, where Luna ran at 294 K: the ratios
        # compared are the overlap and the mismatch, which the 0.3 % of density does
        # not reach at the bounds below. Everything else is stated, as for any witness.
        cladding=fused_silica(model="fit/SiO2"),
        temperature=293.15,
        loss=True,
        **{"noise": None, **witness_inputs(grid, 2)},
    )
    pulse = GaussianPulse(
        wavelength=OVERLAP_PROVENANCE["lambda0"], fwhm=UNDEPLETED["fwhm"], energy=energy
    )
    built = pulse.build(grid)
    field0 = np.zeros((2, built.shape[1], grid.n_omega), dtype=complex)
    field0[0] = built[0]
    field0[1] = UNDEPLETED["seed"] * built[0]
    declared = kerr_of(base)
    response = replace(declared, chi3=declared.chi3 * kerr)
    simulation = Simulation(
        grid,
        base.modes,
        [response],
        field0,
        settings=replace(base.settings, absorber=None),
        profiles=base.profiles,
    )
    return simulation.run().fields("fft")[:, :, 0, :], base.modes


def linear_exponent(initial, exit_linear):
    """Return ``log H_p`` per mode, the measured linear exponent over the whole fibre.

    ``H_p = A_p^lin(L) / A_p(0)``, so ``log H_p = D_p L`` with the code's own operator,
    frame and conventions in it. The phase is unwrapped along omega and re-anchored at
    the carrier, where it is within a turn: over two centimetres the second mode's
    accumulated mismatch alone is two radians, and the band edges pass pi.
    """
    peak = np.max(np.abs(initial), axis=-1, keepdims=True)
    known = np.abs(initial) > TRANSFER_FLOOR * peak
    transfer = np.where(known, exit_linear / np.where(known, initial, 1.0), 1.0)
    carrier = int(np.argmax(np.abs(initial[0])))
    phase = np.unwrap(np.angle(transfer), axis=-1)
    principal = np.angle(transfer[:, carrier])
    turns = np.round((phase[:, carrier] - principal) / (2.0 * np.pi))[:, np.newaxis]
    return np.log(np.abs(transfer)) + 1j * (phase - 2.0 * np.pi * turns)


def first_order_change(initial, log_h, tensor, index):
    """Return each mode's first-order nonlinear change at the exit, up to one constant.

    ``Delta A_p(L) = N_p int_0^L e^{D_p (L - z)} F[P_p(z)] dz`` with the modes
    propagating linearly inside the integral, evaluated by Gauss-Legendre in z. The
    common ``eps0 chi3 rho``, the transform normalisation and the ``-i omega / 4`` are
    left out because only the ratio between two modes is used; what is kept is the
    per-mode part of the prefactor, ``1 / n_eff``, which solprop carries and Luna does
    not (pass ones for Luna).
    """
    n_t = 2 * (initial.shape[-1] - 1)
    nodes, weights = np.polynomial.legendre.leggauss(FIRST_ORDER_NODES)
    change = np.zeros_like(initial)
    for node, weight in zip(0.5 * (nodes + 1.0), 0.5 * weights, strict=True):
        field = np.fft.irfft(
            initial * np.exp(log_h * node), n=FIRST_ORDER_PAD * n_t, axis=-1
        )
        cubic = np.einsum("plmn,lt,mt,nt->pt", tensor, field, field, field)
        drive = np.fft.rfft(cubic, axis=-1)[:, : initial.shape[-1]]
        change += weight * np.exp(log_h * (1.0 - node)) * drive
    return change / index


def undepleted_ratios(initial, exit_kerr, exit_linear, index):
    """Return the measured, first-order and closed-form mode-two to mode-one ratios."""
    # The output stores b; the independent source-chain oracle is derived for a.
    # The Luna witness, at index one, is compared unchanged.
    root = np.sqrt(index)
    initial, exit_kerr, exit_linear = (
        initial / root,
        exit_kerr / root,
        exit_linear / root,
    )
    band = np.abs(initial[0]) > UNDEPLETED_BAND * np.max(np.abs(initial[0]))
    log_h = linear_exponent(initial, exit_linear)
    predicted = first_order_change(initial, log_h, overlap_tensor(2), index)
    measured = exit_kerr - exit_linear
    mismatch = (log_h[1] - log_h[0])[band]
    drives = overlap_tensor(2)[0, 0, 0, :]
    # The dispersion-free closed form: the drive ratio times the mismatch factor
    # (exp(i dbeta L) - 1) / (i dbeta L), with (D_2 - D_1) L read off the same measured
    # transfer. It neglects the pump's own dispersion, which the integral above keeps.
    closed = (
        (drives[1] / drives[0])
        * (index[0] / index[1])[band]
        * np.expm1(mismatch)
        / mismatch
    )
    # Sliced before the division and not after it: outside the active mask both exit
    # fields are exactly zero and the ratio there is 0/0. It used to be 0/1e-300
    # instead, because ADR-025's split factor ran its transform pair *after* the hard
    # mask and put round-off back outside the band for the next step to remove; the
    # in-stage scheme leaves the state exactly zero there, which is what exposed this.
    return {
        "band": band,
        "measured": measured[1][band] / measured[0][band],
        "first_order": predicted[1][band] / predicted[0][band],
        "closed_form": closed,
        "nonlinear_phase": float(
            np.max(
                np.abs(exit_kerr[0] - exit_linear[0])[band]
                / np.abs(exit_linear[0])[band]
            )
        ),
    }


def refractive_index(modes, grid):
    """Return ``Re n_eff`` per mode on the grid, one outside the active band."""
    index = np.ones((len(modes), grid.n_omega))
    active = grid.active_mask
    for row, mode in enumerate(modes):
        index[row, active] = np.real(np.asarray(mode.neff(grid.omega[active], 0.0)))
    return index


@pytest.fixture(scope="module")
def undepleted_solprop_ratios():
    """solprop's undepleted-pump case: the Kerr run, the linear run and the ratios.

    Module-scoped, so it runs outside the file's autouse fixture and silences the gas
    index's extrapolation notice for itself, as the M2 fixtures below do.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        grid = undepleted_grid()
        kerr, modes = undepleted_solprop(kerr=1.0, energy=UNDEPLETED["energy"])
        linear, _ = undepleted_solprop(kerr=0.0, energy=UNDEPLETED["energy"])
        np.testing.assert_array_equal(kerr[0], linear[0])
        return undepleted_ratios(
            kerr[0], kerr[-1], linear[-1], refractive_index(modes, grid)
        )


@pytest.fixture(scope="module")
def undepleted_luna_ratios():
    """Luna's undepleted-pump case, from the stored spectra of the same two runs."""

    def stored(tag, plane, count=2):
        return np.stack(
            [
                np.asarray(OVERLAP[f"UNDEPLETED_{tag}_{plane}_{m}_RE"])
                + 1j * np.asarray(OVERLAP[f"UNDEPLETED_{tag}_{plane}_{m}_IM"])
                for m in range(1, count + 1)
            ]
        )

    initial = stored("NL", "IN")
    np.testing.assert_array_equal(initial, stored("LIN", "IN"))
    # Luna's modal transform divides by no effective index at all -- `norm_modal` is
    # -i omega / 4 -- so its prediction carries none either.
    ones = np.ones(initial.shape)
    return undepleted_ratios(initial, stored("NL", "OUT"), stored("LIN", "OUT"), ones)


def test_the_undepleted_case_runs_on_the_grid_luna_used():
    """Both codes step the same case on the same frequency axis, bitwise."""
    np.testing.assert_array_equal(
        OVERLAP["UNDEPLETED_NL_OMEGA"], undepleted_grid().omega
    )


@pytest.mark.parametrize("code", ["solprop", "luna"])
def test_the_undepleted_case_is_the_first_order_solution(
    code, undepleted_solprop_ratios, undepleted_luna_ratios
):
    """The second oracle: each code against perturbation theory, not against the other.

    Both codes are in the first-order regime -- the pump's own nonlinear phase is
    2.2e-3 -- and both reproduce the first-order solution of the coupled-mode equations
    across the whole pump band: solprop to 4.9e-6 and Luna to 1.3e-5 in the worst bin,
    against a second-order term of the size of that nonlinear phase. The bound is the
    measured agreement with a margin of four; it scales linearly with the pump energy,
    which is asserted below, so it is the neglected second order and not a tolerance.
    """
    both = {"solprop": undepleted_solprop_ratios, "luna": undepleted_luna_ratios}
    ratios = both[code]
    assert int(np.sum(ratios["band"])) == 77
    assert ratios["nonlinear_phase"] == pytest.approx(2.2e-3, rel=0.05)
    deviation = np.abs(ratios["measured"] - ratios["first_order"]) / np.abs(
        ratios["first_order"]
    )
    assert float(np.max(deviation)) < 5e-5


@pytest.mark.parametrize("code", ["solprop", "luna"])
def test_the_undepleted_ratio_is_the_analytic_mismatch_factor(
    code, undepleted_solprop_ratios, undepleted_luna_ratios
):
    """And it is the closed form, once the pump's own dispersion is allowed for.

    ``A_2 / Delta A_1 = (T_2/T_1) (n_1/n_2) (e^{i dbeta L} - 1) / (i dbeta L)`` with
    ``dbeta = beta_1 - beta_2``: the quartic overlap ratio times the mismatch factor,
    and nothing else. ``dbeta`` is 100.68 rad/m for these two modes at the carrier, so
    2 cm is 2.014 rad, the factor has modulus 0.8394 and the ratio is 0.4017 there.
    Averaged over the pump's band the closed form gives 0.3987 and both codes measure
    0.3995.

    The 5e-3 bound is the pump's own dispersion over that band, which this form drops
    and the integrated first-order solution above keeps -- which is why that one agrees
    four orders better. Measured 3.7e-3 in the worst bin on both codes.
    """
    both = {"solprop": undepleted_solprop_ratios, "luna": undepleted_luna_ratios}
    ratios = both[code]
    deviation = np.abs(ratios["measured"] - ratios["closed_form"]) / np.abs(
        ratios["closed_form"]
    )
    assert float(np.max(deviation)) < 5e-3
    assert float(np.abs(np.mean(ratios["measured"]))) == pytest.approx(0.3995, rel=1e-3)


def test_both_codes_now_give_the_same_intermodal_ratio(
    undepleted_solprop_ratios, undepleted_luna_ratios
):
    """A consequence of the two oracles rather than one of them, and their point.

    Each code above is measured against perturbation theory on its own; that they then
    agree with each other to 1e-5 in modulus and phase is what the earlier comparison
    said they did not do. With each mode in its own frame solprop could not have passed
    either test: its operator carried 201.17 rad/m of mismatch between the two modes
    where their propagation constants differ by 100.54, exactly twice over. For a hollow
    capillary the group-delay term is the mismatch -- the waveguide part of beta goes as
    1/omega, so ``beta_1,m omega = -beta_wg,m`` and ``(beta_1,2 - beta_1,1) omega_0`` is
    ``beta_1 - beta_2`` to a tenth of a per cent -- so subtracting it per mode counted
    it twice. Over a short fibre that is a wrong phase; over a long one the same term is
    the walk-off the modes really do suffer.
    """
    ours = np.mean(undepleted_solprop_ratios["measured"])
    theirs = np.mean(undepleted_luna_ratios["measured"])
    assert abs(abs(ours) - abs(theirs)) / abs(theirs) < 1e-4
    assert abs(np.angle(ours) - np.angle(theirs)) < 1e-4


def test_the_undepleted_ratio_does_not_depend_on_the_pump_energy():
    """First order means first order: a decade of pump leaves the ratio where it is.

    The ratio is homogeneous of degree zero in the pump at this order, so this is the
    measurement that the case is in the regime the oracle assumes rather than merely
    close to it. A tenth of the energy moves the ratio by 7e-7, from 0.39947707 to
    0.39947679, and takes the deviation from the first-order solution down with it,
    4.9e-6 to 6.7e-7 -- not quite the full decade, because at 2.2e-4 of nonlinear phase
    the reference's own z quadrature and the step tolerance start to show.
    """
    grid = undepleted_grid()
    energy = 0.1 * UNDEPLETED["energy"]
    kerr, modes = undepleted_solprop(kerr=1.0, energy=energy)
    linear, _ = undepleted_solprop(kerr=0.0, energy=energy)
    ratios = undepleted_ratios(
        kerr[0], kerr[-1], linear[-1], refractive_index(modes, grid)
    )
    assert float(np.abs(np.mean(ratios["measured"]))) == pytest.approx(0.3995, rel=1e-3)
    deviation = np.abs(ratios["measured"] - ratios["first_order"]) / np.abs(
        ratios["first_order"]
    )
    # A tenth of the energy, near enough a tenth of the second-order term.
    assert float(np.max(deviation)) < 2e-6


# ============================================================ milestone M2, the RDW
#
# The reduced case of `tests/data/generate_luna_rdw.py` against the Luna result stored
# beside it, and solprop against itself on the same case. Everything here is slow.
#
# What the comparison controls for, so that what is left is the solver:
#
# * the grid -- Luna's realised omega axis and solprop's are bitwise equal, asserted
#   below rather than assumed;
# * the helium susceptibility -- Luna divides the Ermolov expansion by the 1 bar
#   density although the fit is referenced to 1 atm, so the mode is given the
#   package fit at Luna's reference state, which reproduces Luna's values to 1.6e-8,
#   and that offset cancels against the equation of state (asserted below);
# * the cladding -- Luna's is an unreferenced 400-row table, so the mode is
#   given a spline through Luna's own values on this grid;
# * the temperature and the shot noise -- 294.0 K and none, on both sides, both
#   stated rather than defaulted, as is every other input of the run
#   (`witness_inputs`: the witness is frozen, docs/dev/testing.md);
# * the launch -- `witness_run` asks for the *true* energy of Luna's pulse rather than
#   for the nominal energy Luna asked for. A requested energy is a true one in solprop
#   (ADR-030, docs/dev/decisions.md) while Luna's is its vacuum amplitude's, so the
#   nominal request would launch 0.0036 % less than Luna did.
#
# The launch is worth a few parts in 1e4 here. Two further differences of the power
# normalisation remain on this case and no launch can undo them. This fill is a
# gradient, so the change of variables keeps its `-1/2[ln w(z) - ln w(0)]` term --
# where the reduced M4 case's constant fill has none -- and both codes' spectra are
# compared as true power, so a distance or a band ratio between them is weighted by `w`.
#
# What is left is a large and unexplained disagreement in the *higher* modes, measured
# and documented below. It is not the grid, the media, the linear operator, the
# quadrature order or either code's step size, and solprop's inter-mode drive is exactly
# the analytic quartic overlap of the Bessel profiles.

WITNESS = json.loads((Path(__file__).parents[1] / "data/luna_rdw.json").read_text())
MEDIA = json.loads((Path(__file__).parents[1] / "data/luna_rdw_media.json").read_text())
REDUCED = WITNESS["reduced"]
FULL = WITNESS["full"]
LUNA_FIELD = "Eω"
VUV = (100e-9, 140e-9)

# Luna's own defaults, which its interface cannot set and which every published
# Luna result therefore carries.
LUNA_TOLERANCES = {"rtol": 1e-6, "atol": 1e-10}
TEST_BACKEND = selected_backend()

# The Luna witnesses are frozen (``docs/dev/testing.md``): never regenerated, so every
# input a comparison with one depends on is stated in the run compared, and a change of
# a solprop default cannot move what is compared. These are the inputs such a run would
# otherwise take from a default, each today's default.
WITNESS_GRID_RULES = {
    "ffac": 6.0,
    "fft_size": "pow2",
    "collar_fraction": 0.05,
    "collar_placement": "inside",
}
"""The field-grid rules every witness grid is built with; they realise Luna's axes."""


def witness_inputs(grid, n_modes):
    """Return the run inputs a Luna-witness run states rather than takes by default.

    Parameters
    ----------
    grid : FieldGrid
        The run's grid, for its collar and its length.
    n_modes : int
        How many modes the run carries, for the radial order.

    Returns
    -------
    dict
        Keywords of :func:`prop_capillary` and
        :meth:`~solprop.driver.simulation.Simulation.from_capillary`: the full
        Marcatili model, a Gaussian launch, the Kerr term with its third harmonic, a
        Gauss--Kronrod radial rule of 32 nodes up to four modes and 48 for six, the
        grid's collar at ``boundary_N = 20``, no noise, and the step caps of half the
        fibre and 1e-4 of it.
    """
    return {
        "model": "full",
        "pulseshape": "gauss",
        "kerr": True,
        "thg": True,
        "n_r": 32 if n_modes <= 4 else 48,
        "radial_rule": "gauss_kronrod",
        "absorber": absorber(grid, boundary_N=20),
        "noise": None,
        "dz_max": grid.length / 2,
        "dz_init": grid.length * 1e-4,
    }


def luna_helium():
    """Return Luna's helium susceptibility: the Ermolov fit at the state Luna assumed.

    Luna's input, supplied as the caller's data: the Ermolov three-pole expansion, with
    the coefficients both codes carried (solprop's former registry at dc4ca0d, its
    default helium and the input of this witness). Luna divides it by the 1 bar
    density although the expansion is referenced to 1 atm, so it is built at that
    reference state. It reproduces Luna's sampled values to 3.6e-7, flat in frequency
    -- asserted below -- so the comparison needs no table for the gas.
    """
    return PoleSumSusceptibility(
        gas=Gas.He,
        source="ermolov (Luna's normalisation)",
        reference_state=ReferenceState(pressure_bar=1.0, temperature=273.15),
        wavelength_range=(90e-9, 1000e-9),
        citation=(
            "A. Ermolov et al., Phys. Rev. A 92, 033821 (2015), "
            "doi:10.1103/PhysRevA.92.033821, normalised by the 1 bar density as Luna "
            "does; the coefficients of solprop's former registry at dc4ca0d"
        ),
        b=(2.16463842e-05, 2.10561127e-07, 4.75092720e-05),
        c=(-6.80769781e-04, 5.13251289e-03, 3.18621354e-03),
    )


def luna_helium_gamma3():
    """Return Luna's helium hyperpolarisability, per particle, SI.

    Luna's ``PhysData.gamma3_gas(:He)``: Lehmeier's ``4 x 3.43e-28`` divided by
    CoolProp's number density at 1 atm and 0 C, Luna's own formula. It is solprop's
    former registry value (dc4ca0d) exactly, the input this witness's bounds were
    measured on; optmat's helium coefficient at 800 nm is 2.0 % above it, which alone
    moves the reduced case's fundamental shape from 0.0569 to 0.0823.
    """
    density = PropsSI("Dmolar", "P", 101325.0, "T", 273.15, "Helium") * N_A
    return 4.0 * 3.43e-28 / density


LUNA_HELIUM_POTENTIAL = HISTORICAL_ATOMS["He"][0] * AU_ENERGY
"""Helium's ionisation potential the M2 witness was compared on, J (0.9036 hartree)."""


def m2_helium_rate():
    """Return the M2 witness's helium rate: solprop's former default PPT, as an oracle.

    Luna's own ``IonRatePPT(:He, 800e-9)`` (the Ilkov form with the Stark shift and the
    ion's dipole factor, two electrons) times Tong and Lin's barrier-suppression
    factor, which Luna's branch lacks: the rate the witness's bounds were measured on,
    numerically solprop's former ``PPT("He", 800e-9)`` bit for bit. The barrier
    suppression is the one known departure from Luna's rate (``docs/dev/accuracy.md``
    section 5).
    """
    return ppt_oracle("He", WITNESS["lambda0"])


def luna_helium_fill(
    case,
    pressure=None,
    *,
    pressure_floor: float | None = 0.0,
    vacuum_end: VacuumEnd | None = "floor",
):
    """Return the M2 witness's helium fill: Luna's material inputs, supplied.

    The fill a named gas gets (:func:`solprop.driver.mixtures.capillary_fill`, the same
    density path and flow), with its species replaced by a caller's carrying
    :func:`luna_helium`, :func:`luna_helium_gamma3` and the witness's potential; the
    rate is :func:`m2_helium_rate`, given beside it.

    Parameters
    ----------
    case : dict
        The stored case, for its length, radius and pressures.
    pressure : float, pair or None, optional
        The fill in bar; ``None`` takes the case's gradient.
    pressure_floor, vacuum_end : optional
        As :meth:`~solprop.driver.simulation.Simulation.from_capillary`; Luna's true
        vacuum end by default.
    """
    fill = capillary_fill(
        "He",
        (case["p_in"], case["p_out"]) if pressure is None else pressure,
        length=case["length"],
        temperature=WITNESS["temperature_k"],
        floor=pressure_floor,
        susceptibility=luna_helium(),
        preionisation=None,
        radius=case["radius"],
        vacuum_end=vacuum_end,
    )
    helium = Species(
        "He",
        luna_helium(),
        gamma3=luna_helium_gamma3(),
        ionisation=IonisationData(
            LUNA_HELIUM_POTENTIAL, citation="solprop's former registry at dc4ca0d"
        ),
        citation="the M2 witness's material inputs (Luna's helium)",
    )
    (component,) = fill.components
    return Mixture.of(replace(component, species=helium), temperature=fill.temperature)


def luna_hydrogen():
    """Return Luna's hydrogen susceptibility: the Peck and Hung fit at 1 atm and 0 C.

    Luna's ``sellmeier_gas(:H2)``, supplied as the caller's data: the coefficients are
    the numbers both codes carried (solprop's former registry at dc4ca0d, its default
    hydrogen and the input of this witness), divided by the 1 atm density, the state
    Peck and Hung quote them at.
    """
    return InversePoleIndex(
        gas=Gas.H2,
        source="peck (Luna's)",
        reference_state=ReferenceState(pressure_bar=1.01325, temperature=273.15),
        wavelength_range=(168.0e-9, 1694.5e-9),
        citation=(
            "E. R. Peck and S. Hung, J. Opt. Soc. Am. 67, 1550 (1977), "
            "doi:10.1364/JOSA.67.001550; the coefficients of solprop's former "
            "registry at dc4ca0d"
        ),
        offset=0.0,
        b=(0.0148956, 0.0049037),
        c=(180.7, 92.0),
    )


LUNA_HYDROGEN_RAMAN = FORMER_H2
"""The hydrogen Raman constants the CtC witness was compared on.

solprop's former registry at dc4ca0d (``raman_parameters("H2")``), the input of this
witness, supplied as the caller's data (``tests/former_raman.py``). Luna's own
vibrational amplitude is half of this one's; ``ctc_run(luna_vibration=True)`` matches
it for the historical bounds.
"""


LUNA_HYDROGEN_POTENTIAL = 2.4715392830185813e-18
"""Hydrogen's ionisation potential the CtC witness was compared on, J.

solprop's former registry at dc4ca0d, the input of this witness: with it the oracle is
the former ``ADK("H2")`` bit for bit. Luna's own (``ADK_IONPOT``, an older CODATA
release) is 1.7e-8 below it.
"""


def luna_hydrogen_rate():
    """Return Luna's molecular hydrogen rate: potential-only ADK, occupancy two.

    ``Ionisation.IonRateADK(:H2)`` takes the potential alone (``Z = 1``, ``l* = 0``, no
    angular factor), which is the oracle's hydrogenic form, the one solprop's former
    ``ADK("H2")`` took for want of quantum numbers.
    """
    return AdkOracle("H2", LUNA_HYDROGEN_POTENTIAL, hydrogenic=True, occupancy=2)


def test_the_witness_rate_is_lunas_sampled_hydrogen_rate():
    """The oracle the CtC witness runs on is the rate Luna sampled beside it.

    Bound, stated first: the two codes' atomic units come from CODATA releases that
    differ by under 2e-8, and an offset ``delta`` in the field or potential moves the
    rate by its exponent ``2 (2 Ip)^(3/2) / (3 F)`` times ``delta``, 1.5 times for the
    potential; the comparison is over the fields above 1e9 V/m and below either
    code's ceiling, where neither threshold acts. Measured 3.4e-6.
    """
    rate = luna_hydrogen_rate()
    field = np.asarray(CTC_MEDIA["ADK_FIELD"])
    luna = np.asarray(CTC_MEDIA["ADK_RATE"])
    ours = rate(field)
    compared = (field > 1e9) & (luna > 0.0) & (ours < rate.maximum)
    assert int(np.count_nonzero(compared)) > 100
    potential = LUNA_HYDROGEN_POTENTIAL / AU_ENERGY
    exponent = 2.0 * (2.0 * potential) ** 1.5 / (3.0 * field[compared] / AU_EFIELD)
    bound = exponent * 2.5 * 2e-8
    assert np.all(np.abs(ours[compared] / luna[compared] - 1.0) < bound)


def luna_hydrogen_fill(pressure, temperature):
    """Return the CtC witness's hydrogen fill: Luna's material inputs, supplied.

    A caller's species carrying :func:`luna_hydrogen`, the hyperpolarisability the
    generator recorded beside the witness (``GAMMA3``, Luna's), the witness's
    ionisation potential and :data:`LUNA_HYDROGEN_RAMAN`. Its density is optmat's
    equation of state at the fill's pressure, the state :func:`luna_hydrogen` is
    referenced by, so the two carry the same offset from Luna's CoolProp and it
    cancels from the susceptibility as it did when both were solprop's.
    """
    hydrogen = Species(
        "H2",
        luna_hydrogen(),
        gamma3=CTC_MEDIA["GAMMA3"][0],
        ionisation=IonisationData(
            LUNA_HYDROGEN_POTENTIAL, citation="solprop's former registry at dc4ca0d"
        ),
        raman=LUNA_HYDROGEN_RAMAN,
        citation="the CtC witness's material inputs (luna_ctc_media.json)",
    )
    density = DensityPath(Gas.H2, Constant(float(pressure)), temperature)
    return Mixture.of(Component(hydrogen, density), temperature=temperature)


def test_the_m2_witness_runs_on_lunas_helium_and_its_former_rate():
    """The M2 witness's Kerr and rate are its own inputs, not optmat's defaults.

    The hyperpolarisability is Luna's formula and solprop's former registry number
    exactly (5.109203705817757e-53, dc4ca0d), and the rate the former default PPT;
    the run's responses carry both.
    """
    assert luna_helium_gamma3() == 5.109203705817757e-53
    simulation = reduced_m2_simulation()
    kinds = [type(response).__name__ for response in simulation.responses]
    assert kinds == ["KerrResponse", "PlasmaResponse"]
    assert kerr_of(simulation).chi3 == luna_helium_gamma3()
    rate = _rate(simulation)
    assert isinstance(rate, PptOracle)
    assert rate == m2_helium_rate()
    assert (rate.tong_lin_alpha, rate.polarisabilities) == (7.0, (1.3207, 0.2811))


def luna_cladding(tag, grid, media=None):
    """Return a silica index carrying Luna's own cladding values on this grid.

    Luna's cladding is a 400-row complex table with no citation anywhere in its
    repository, so it cannot be rebuilt from a source the way the gas can. The
    generator samples it on the run's own frequency axis instead, and this splines those
    samples; at the nodes -- which is every frequency the run uses -- the spline returns
    them exactly. `media` names which witness's samples to read, defaulting to the RDW
    one.
    """
    sampled = MEDIA if media is None else media
    band = grid.active_mask
    omega = grid.omega[band]
    index = (
        np.asarray(sampled[f"{tag}_CLAD_RE"])[band]
        + 1j * np.asarray(sampled[f"{tag}_CLAD_IM"])[band]
    )
    wavelength_um = 2.0e6 * math.pi * C / omega
    order = np.argsort(wavelength_um)
    log_wavelength = np.log(wavelength_um[order])
    return TabulatedGlass(
        name="SiO2",
        source="luna",
        wavelength_range=(
            float(wavelength_um[order][0]) * 1e-6,
            float(wavelength_um[order][-1]) * 1e-6,
        ),
        citation="Luna's shipped cladding table, sampled on this run's frequency axis",
        provenance={"source": "the media emitted beside this witness"},
        _real=CubicSpline(log_wavelength, np.real(index[order])),
        _log_extinction=CubicSpline(
            log_wavelength, np.log(np.maximum(np.imag(index[order]), 1e-30))
        ),
        _log_wavelength_limits=(float(log_wavelength[0]), float(log_wavelength[-1])),
    )


def witness_grid(case):
    """Return the grid a stored case was run on, built from its own parameters."""
    return FieldGrid.from_limits(
        case["length"],
        WITNESS["lambda0"],
        (case["lambda_min"], case["lambda_max"]),
        case["trange"],
        **WITNESS_GRID_RULES,
    )


def stored_field(name) -> np.ndarray:
    """Convert Luna's vacuum amplitudes with the solprop mode's plane index."""
    with h5py.File(Path(__file__).parents[1] / "data" / name, "r") as file:
        dataset = cast("h5py.Dataset", file[LUNA_FIELD])
        field = np.asarray(dataset[()])
        positions = np.asarray(cast("h5py.Dataset", file["z"])[()])
        stored_planes = json.loads(str(file.attrs.get("stored_planes", '"all"')))
        if stored_planes != "all":
            positions = positions[stored_planes]
        assert len(positions) == len(field)
    ctc = "ctc" in name
    tag = "REDUCED" if "reduced" in name else "FULL"
    witness = CTC_WITNESS if ctc else WITNESS
    case = witness[tag.lower()]
    grid = ctc_grid(case) if ctc else witness_grid(case)
    pressure = (
        case["pressure"]
        if ctc
        else PressureGradient.two_point(
            case["length"], case["p_in"], case["p_out"], floor=0.0
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        for m in range(field.shape[1]):
            mode = MarcatiliMode(
                case["radius"],
                "H2" if ctc else "He",
                pressure,
                m=m + 1,
                temperature=witness["temperature_k"],
                # Luna's runs, read with Luna's own hydrogen or helium law.
                susceptibility=luna_hydrogen() if ctc else luna_helium(),
                model="full",
                cladding=luna_cladding(tag, grid, CTC_MEDIA if ctc else MEDIA),
                loss=not ctc,
            )
            for plane, z in enumerate(positions):
                weight = np.ones(grid.n_omega)
                weight[grid.active_mask] = np.real(
                    mode.neff(grid.omega[grid.active_mask], float(z))
                )
                field[plane, m] *= np.sqrt(weight)
    # Only this reader converts to true power; the archived Luna data are not edited.
    return field


def luna_launch_energy(case, tag, *, ctc=False):
    """Return the true energy of the pulse Luna launched for this case's nominal one.

    Luna launches its nominal energy in the vacuum-normalised amplitude it steps in,
    so the *physical* pulse its archive was produced with carries
    ``sum w |a|^2 = w_bar E`` of true energy, with ``w = Re n_eff(omega, 0)`` and
    ``w_bar`` the launch-spectrum-weighted mean of it. In solprop a requested energy
    **is** true energy, so asking for the nominal one would launch ``1/w_bar`` of
    Luna's pulse and the comparison would not be like for like: 0.492 % down on the
    reduced CtC case and 0.0036 % on the reduced M2 one.

    The average is exact rather than first order in the 1e-4 spread of ``w`` across the
    band: both codes build the same Gaussian spectral shape, Luna as its ``a`` and
    solprop as its true-power ``b``, so weighting ``w`` by ``|b|^2`` here is weighting
    it by Luna's own ``|a|^2`` there. What the energy cannot remove is that the two
    launches still differ in *shape* by ``sqrt(w)``, which is that 1e-4.

    Measured against the archives' own plane zero -- the same quantity read off Luna's
    stored amplitudes, ``energy(grid, stored_field(name)[0])`` summed over the modes --
    ``w_bar`` agrees to 2e-16 on both RDW cases and the full CtC one and to 1.2e-11 on
    the reduced CtC one. Taking the launch from the witness field instead would also
    absorb Luna's own 1.6e-6 pulse-normalisation error on that short window -- its
    energy functional is an extended Simpson rule -- and so would make
    `test_the_two_codes_start_from_the_same_state` compare a number with itself. The
    launch is therefore built from the case and the mode, as a user builds one, and
    that 1.6e-6 stays measurable.

    Parameters
    ----------
    case : dict
        The stored case, out of ``luna_rdw.json`` or ``luna_ctc.json``.
    tag : {'REDUCED', 'FULL'}
        Which of the two, naming the cladding samples to read.
    ctc : bool, optional
        Whether this is the CtC witness rather than the RDW one. It selects the gas,
        the susceptibility, the media file and the mode's loss exactly as
        :func:`stored_field` does, so that the two agree on what ``w`` is.

    Returns
    -------
    float
        The energy to request of solprop, J.
    """
    witness = CTC_WITNESS if ctc else WITNESS
    grid = ctc_grid(case) if ctc else witness_grid(case)
    pressure = (
        case["pressure"]
        if ctc
        else PressureGradient.two_point(
            case["length"], case["p_in"], case["p_out"], floor=0.0
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        mode = MarcatiliMode(
            case["radius"],
            "H2" if ctc else "He",
            pressure,
            temperature=witness["temperature_k"],
            susceptibility=luna_hydrogen() if ctc else luna_helium(),
            model="full",
            cladding=luna_cladding(tag, grid, CTC_MEDIA if ctc else MEDIA),
            loss=not ctc,
        )
        # One outside the active band, which is what `translate.linop` assembles and
        # what `stored_field` applies to Luna's side, so the 4.4e-7 of the reduced
        # CtC launch that the hard time window throws out of band is treated alike.
        weight = np.ones(grid.n_omega)
        weight[grid.active_mask] = np.real(mode.neff(grid.omega[grid.active_mask], 0.0))
        spectrum = (
            np.abs(
                GaussianPulse(
                    wavelength=witness["lambda0"],
                    fwhm=witness["tau_fwhm"],
                    energy=witness["energy"],
                ).build(grid)[0, 0]
            )
            ** 2
        )
    return witness["energy"] * float(np.sum(weight * spectrum) / np.sum(spectrum))


def test_trimmed_luna_witness_uses_its_stored_plane_positions():
    """The archived field uses endpoint indices, not the first two z values."""
    name = "luna_rdw_full.h5"
    converted = stored_field(name)
    with h5py.File(Path(__file__).parents[1] / "data" / name, "r") as file:
        original = np.asarray(cast("h5py.Dataset", file[LUNA_FIELD])[()])
        positions = np.asarray(cast("h5py.Dataset", file["z"])[()])[[0, -1]]
    grid = witness_grid(FULL)
    pressure = PressureGradient.two_point(
        FULL["length"], FULL["p_in"], FULL["p_out"], floor=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        mode = MarcatiliMode(
            FULL["radius"],
            "He",
            pressure,
            temperature=WITNESS["temperature_k"],
            susceptibility=luna_helium(),
            model="full",
            cladding=luna_cladding("FULL", grid, MEDIA),
            loss=True,
        )
        for plane, z in enumerate(positions):
            weight = np.ones(grid.n_omega)
            weight[grid.active_mask] = np.real(
                mode.neff(grid.omega[grid.active_mask], float(z))
            )
            # The reader and direct endpoint evaluation use identical arithmetic.
            np.testing.assert_array_equal(
                converted[plane, 0], original[plane, 0] * np.sqrt(weight)
            )


def witness_run(
    case,
    tag,
    *,
    rtol=LUNA_TOLERANCES["rtol"],
    n_r=None,
    planes=None,
    backend=None,
    **overrides,
):
    """Run solprop's side of a stored case, with Luna's media in the modes."""
    grid = witness_grid(case)
    chosen = {"backend": TEST_BACKEND if backend is None else backend}
    stated = witness_inputs(grid, case["modes"])
    if n_r is not None:
        stated["n_r"] = n_r
    return prop_capillary(
        case["radius"],
        case["length"],
        # Luna's helium, hyperpolarisability and the witness's rate, supplied; the
        # gradient ends in a true vacuum, as Luna's does.
        luna_helium_fill(case),
        plasma=(plasma := overrides.pop("plasma", m2_helium_rate())),
        # The witnesses were made by Luna's cumulative trapezoid (SEP 18).
        plasma_integration=None if plasma is False else "trapezoid",
        grid=grid,
        lambda0=WITNESS["lambda0"],
        fwhm=WITNESS["tau_fwhm"],
        # Luna's nominal energy is its *vacuum* amplitude's, so the pulse its archive
        # was made with carries w_bar times as much true energy -- 1.0000357455 here.
        # A requested energy is true energy in solprop (ADR-030), so requesting the
        # nominal one would launch 0.0036 % less than Luna did.
        energy=luna_launch_energy(case, tag),
        modes=case["modes"],
        cladding=luna_cladding(tag, grid),
        loss=True,
        temperature=WITNESS["temperature_k"],
        raman=False,
        n_save=case["n_save"] if planes is None else planes,
        rtol=rtol,
        atol=LUNA_TOLERANCES["atol"],
        **chosen,
        **(stated | overrides),
    )


def gated_witness_run(case, tag, **keywords):
    """Compare an archive with its original equation and unchanged tolerance."""
    with _gated_source():
        return witness_run(case, tag, backend="numpy_ref", **keywords)


def band_mask(grid, limits):
    """Return the mask of a wavelength band on the stored frequency axis."""
    positive = grid.omega > 0.0
    wavelength = np.where(
        positive, 2.0 * math.pi * C / np.where(positive, grid.omega, 1.0), np.inf
    )
    return (wavelength >= limits[0]) & (wavelength <= limits[1])


def arrival_of(grid, spectrum, mask):
    """Return the intensity-weighted arrival time of a band, s."""
    field = np.fft.irfft(np.where(mask, spectrum, 0.0), n=grid.n_t, axis=-1)
    power = np.sum(np.abs(field) ** 2, axis=tuple(range(field.ndim - 1)))
    return float(np.sum(grid.t * power) / np.sum(power))


def in_band_distance(ours, theirs, band):
    """Return the relative L2 distance of two spectral energy densities in a band."""
    a, b = np.abs(ours)[..., band] ** 2, np.abs(theirs)[..., band] ** 2
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# The records behind the two witness-comparison tables of `docs/dev/accuracy.md`
# sections 5 and 6. These stay tables rather than figures -- they are scalars, not
# ladders -- but each number now comes from a file the suite maintains, together with
# the band its own test asserts, so a bound quietly widened to fit a measurement is a
# failure. The tolerance is a fiftieth: an adaptive run on another platform may put its
# steps in slightly different places, and what that is worth is the controller's own
# tolerance of 1e-6, four orders inside this.
WITNESS_COLUMNS = (
    Column(
        name="value",
        unit="1 for a ratio or a relative L2 distance, fs for an arrival",
        why="a fiftieth: an adaptive run's step sequence is not bitwise across "
        "platforms, and what that costs the answer is the controller's 1e-6, four "
        "orders inside this, while every band below is wider still",
        rtol=0.02,
    ),
    Column(
        name="low",
        unit="as the value",
        why="the lower end of the band the test asserts, compared exactly so that a "
        "bound widened to fit a measurement is itself a failure",
        rtol=0.0,
    ),
    Column(
        name="high",
        unit="as the value",
        why="the upper end of the same band, compared exactly for the same reason",
        rtol=0.0,
    ),
)

M2_WITNESS_METRICS = Study(
    name="m2_witness_metrics",
    title="Milestone M2 against its Luna witness, metric by metric",
    case={
        "witness": "tests/data/luna_rdw_reduced.h5 and luna_rdw_full.h5, production "
        "Luna at the rtol = 1e-6 its interface cannot change",
        "media": "Luna's own helium susceptibility and cladding table",
        "reduced case": "2 modes, 1 m",
        "full case": "4 modes, 1.5 m",
    },
    axis={"name": "metric", "unit": "1", "label": "metric"},
    columns=WITNESS_COLUMNS,
)

M4_WITNESS_METRICS = Study(
    name="m4_witness_metrics",
    title="Milestone M4 against its Luna witness, metric by metric",
    case={
        "witness": "tests/data/luna_ctc_reduced.h5",
        "media": "the package fit unchanged; Luna's cladding table",
        "reduced case": "1 mode, a 2 ps window, 0.1 m",
    },
    axis={"name": "metric", "unit": "1", "label": "metric"},
    columns=WITNESS_COLUMNS,
)

# The record behind the two "what an alarm costs" rows of section 2: the same run at
# three radial orders, with the observable the case is about beside the alarm that is
# supposed to warn about it.
M2_ALARM_COST = Study(
    name="m2_alarm_cost",
    title="What a coarse transverse rule costs the observable, on the M2 case",
    case={
        "case": "the reduced M2 capillary (2 modes, 1 m), adaptive at Luna's rtol",
        "plasma_source": "SEP 27 held current, A_J",
        "band": "100-140 nm, the VUV band milestone M2 is about",
        "rule": "Gauss-Legendre with the embedded Kronrod pair that gives the alarm",
    },
    axis={"name": "n_r", "unit": "nodes", "label": "radial quadrature order"},
    columns=(
        Column(
            name="vuv_energy",
            unit="arbitrary (summed spectral energy density in the band)",
            why="a thousandth: the whole point of the row is that the band moves by "
            "2e-5 across the ladder, so the comparison has to be tighter than that "
            "and looser than an adaptive run's platform-dependent step placement",
            rtol=1e-3,
        ),
        Column(
            name="alarm",
            unit="1 (relative, the embedded pair's disagreement)",
            why="a factor of two on a quantity that falls ten orders across the "
            "ladder, with atol at the projection's own round-off floor",
            bound=1e-6,
            rtol=1.0,
            atol=1e-14,
        ),
    ),
)


def metric_rows(series, entries):
    """Return the record rows of a witness-metric table.

    Parameters
    ----------
    series : str
        Which case the metrics are of.
    entries : sequence of (str, float, float, float)
        The metric's name, its measured value, and the band the test asserts.
    """
    return [
        {
            "series": series,
            "axis": name,
            "value": float(value),
            "low": low,
            "high": high,
        }
        for name, value, low, high in entries
    ]


@pytest.fixture(scope="module")
def reduced():
    """The reduced case at Luna's tolerance: the run the regression is measured on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return gated_witness_run(REDUCED, "REDUCED")


@pytest.fixture(scope="module")
def reduced_tight():
    """The same case two decades tighter, to separate solprop's own step error out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return gated_witness_run(REDUCED, "REDUCED", rtol=1e-8)


@pytest.fixture(scope="module")
def luna_reduced():
    """Luna's stored spectra for the reduced case."""
    return stored_field("luna_rdw_reduced.h5")


# ------------------------------------------------------ what the comparison controls


def test_the_witness_names_the_luna_it_came_from():
    assert len(WITNESS["luna_commit"]) == 40
    assert WITNESS["luna_branch"] == "modal-fixed"
    assert WITNESS["julia_version"].startswith("julia version")
    # The two departures from Luna's own defaults, both so that the comparison measures
    # the solver rather than a temperature or a noise realisation.
    assert WITNESS["temperature_k"] == 294.0
    assert WITNESS["shotnoise"] is False


@pytest.mark.parametrize("tag", ["REDUCED", "FULL"])
def test_the_grid_luna_realised_is_the_grid_solprop_builds(tag):
    """Bitwise, so nothing in the comparison is an interpolation between two grids."""
    case = REDUCED if tag == "REDUCED" else FULL
    grid = witness_grid(case)
    np.testing.assert_array_equal(grid.omega, np.asarray(MEDIA[f"{tag}_OMEGA"]))
    np.testing.assert_array_equal(grid.t, np.asarray(MEDIA[f"{tag}_T"]))


def test_lunas_helium_is_the_ermolov_fit_at_the_reference_state_luna_assumed():
    """Luna's helium differs by a normalisation, so no table is needed for the gas."""
    grid = witness_grid(REDUCED)
    band = grid.active_mask
    ours = luna_helium().susceptibility(grid.omega[band])
    theirs = np.asarray(MEDIA["REDUCED_GAMMA"])[band]
    # Measured 3.551e-7 everywhere, and flat in frequency (spread 6.7e-16): optmat's
    # helium reference density is 3.389e-7 below Luna's CoolProp molar density, and
    # the remaining 1.6e-8 is the fit's own agreement, unchanged since before SEP 3.
    ratio = ours / theirs
    assert np.ptp(ratio) < 1e-14
    # Luna's reference density is CoolProp's molar density at the reference state;
    # with that taken out the fit itself agrees to the 1.6e-8 it always did (the bound
    # this witness had before SEP 3).
    state = luna_helium().reference_state
    lunas = (
        PropsSI(
            "Dmolar", "P", state.pressure_bar * 1e5, "T", state.temperature, "Helium"
        )
        * N_A
    )
    optmats = float(
        equation_of_state("He").number_density(
            State(pressure=state.pressure_bar * 1e5, temperature=state.temperature)
        )
    )
    assert abs(float(np.mean(ratio)) * optmats / lunas - 1.0) < 3e-8


def test_the_equation_of_state_offset_cancels_out_of_the_susceptibility():
    """The run's density carries the same offset as the fit's, with the opposite sign.

    What enters the linear operator is ``chi = gamma * rho``, so a common factor in the
    reference density leaves it untouched: the comparison is of the solver even though
    optmat's equation of state and Luna's CoolProp differ in the seventh digit.
    """
    grid = witness_grid(REDUCED)
    band = grid.active_mask
    gas_ratio = float(
        np.mean(
            luna_helium().susceptibility(grid.omega[band])
            / np.asarray(MEDIA["REDUCED_GAMMA"])[band]
        )
    )
    positions = np.asarray(MEDIA["REDUCED_Z"])
    theirs = np.asarray(MEDIA["REDUCED_DENSITY"])
    path = PressureGradient.two_point(
        REDUCED["length"], REDUCED["p_in"], REDUCED["p_out"], floor=0.0
    )
    ours = DensityPath("He", path, WITNESS["temperature_k"])(positions)
    inside = theirs > 0.0
    density_ratio = float(np.mean(ours[inside] / theirs[inside]))
    # Each is 3.6e-7 from one and they multiply to one to rounding.
    assert abs(gas_ratio * density_ratio - 1.0) < 1e-14


# -------------------------------------------------------------- the M2 regression


@pytest.mark.slow
def test_the_fundamental_mode_matches_luna_within_the_measured_bound(
    reduced, luna_reduced, request
):
    """The mode that carries the pulse, over the in-band mask, at the last plane.

    Measured 6.330e-2 at Luna's tolerance, with the two codes' total content in that
    mode agreeing to 2.89 %; 5.685e-2 under the bounded temporal profile. The
    bound is that measurement with a margin, asserted from both sides: a distance that
    suddenly collapsed would mean the witness had been regenerated with different
    physics, not that the solver had improved.

    With each mode in its own frame these were 5.811e-2 and 3.47 %; the
    fundamental barely moved when the frame became shared, because its own frame was
    the shared one. The higher modes are the test below.
    """
    grid = witness_grid(REDUCED)
    ours = reduced.fields("fft")[-1, :, 0, :]
    distance = in_band_distance(ours[0], luna_reduced[-1][0], grid.inband_mask)
    assert 0.05 < distance < 0.08
    norms = float(
        np.sum(np.abs(ours[0]) ** 2) / np.sum(np.abs(luna_reduced[-1][0]) ** 2)
    )
    assert 1.01 < norms < 1.05
    # The launch is Luna's own pulse, so this is like for like: launching the
    # nominal energy instead moves the shape by 1.40e-4 and the content by 5.2e-6
    # relative, well inside both bands and the record's rtol=0.02.
    record(
        M2_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows(
            "reduced case",
            [
                ("fundamental, shape", distance, 0.05, 0.08),
                ("fundamental, content", norms, 1.01, 1.05),
            ],
        ),
    )


@pytest.mark.slow
def test_the_step_tolerance_is_not_what_separates_the_two_codes(
    reduced, reduced_tight, luna_reduced, request
):
    """Two decades of solprop's tolerance leave the distance where it is.

    The part of the gap that is discretisation error measures 1.6e-5 of itself on
    solprop's side (1.1e-6 before SEP 14's step limits, which leave the ``rtol =
    1e-6`` arm a little further from converged), far below the gap; Luna's own part is
    the same size, its stored result being unchanged to 1e-3 between ``rtol = 1e-6`` and
    ``1e-9`` (measured with `Luna.run` called directly, because Luna's interface cannot
    set a tolerance at all). This rules discretisation out as the cause of the
    higher-mode disagreement, and left the frame as the only remaining candidate.
    """
    grid = witness_grid(REDUCED)
    band = grid.inband_mask
    theirs = luna_reduced[-1][0]
    loose = in_band_distance(reduced.fields("fft")[-1, 0, 0, :], theirs, band)
    tight = in_band_distance(reduced_tight.fields("fft")[-1, 0, 0, :], theirs, band)
    assert abs(loose - tight) / loose < 1e-3
    assert loose > 0.03
    # Both arms move with the launch, not with the tolerance, which is the point: a
    # launch 0.0036 % lower moves the tight arm by 1.40e-4 relative, exactly as the
    # loose one, so their difference moves by 1.48e-4 of itself and stays four orders
    # inside the gap.
    record(
        M2_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows(
            "reduced case",
            [
                ("fundamental shape at rtol 1e-8", tight, 0.05, 0.08),
                (
                    "two decades of tolerance, relative move",
                    abs(loose - tight) / loose,
                    0.0,
                    1e-3,
                ),
            ],
        ),
    )


@pytest.mark.slow
def test_the_dispersive_wave_is_phase_matched_where_the_gradient_has_thinned(reduced):
    """The design tools against the run they are for, on the fixture that exists.

    The linear and the Kerr-corrected phase matching at the entrance pressure might be
    expected to bracket the simulated dispersive wave. They do not, and cannot: the fill
    is a gradient to vacuum, the pulse compresses on its way down it, and the wave is
    shed where the pressure has fallen. What holds is that the simulated peak is to the
    blue of both entrance estimates and is phase matched at a pressure the gradient
    passes through, well down the fibre.
    """
    density = get_spectral_density(reduced, "wavelength", plane=-1, modes=0)
    wavelength, values = density
    ultraviolet = (wavelength > 100e-9) & (wavelength < 250e-9)
    measured = float(wavelength[ultraviolet][np.argmax(values[ultraviolet])])
    pump, p_in, radius = WITNESS["lambda0"], REDUCED["p_in"], REDUCED["radius"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        entrance = MarcatiliMode(radius, "He", p_in)
        power = tools.pulse_peak_power(WITNESS["energy"], WITNESS["tau_fwhm"], "gauss")
        gamma_in = tools.nonlinear_coefficient(
            entrance, n2("He", pressure=p_in, wavelength=pump), pump
        )
        linear = tools.lambda_rdw(entrance, pump)
        kerr = tools.lambda_rdw(
            entrance,
            pump,
            peak_power=power,
            gamma=gamma_in,
            nonlinear_phase="supplied_power",
        )
        matched = tools.pressure_rdw(radius, "He", measured, pump, pressure_max=p_in)
    assert measured < kerr < linear
    # p(z) = p_in sqrt(1 - z/L): 0.62 bar is reached at 0.74 of the length. The window
    # is wide because the run carries Luna's helium and cladding and the tools the
    # package's own, and because the peak of a structured band is read to a bin.
    position = 1.0 - (matched / p_in) ** 2
    assert 0.5 < position < 0.9


@pytest.mark.slow
def test_the_vuv_band_is_the_measured_fraction_of_lunas(reduced, luna_reduced, request):
    """The 100-140 nm band: the quantity milestone M2 is about.

    In the fundamental alone solprop puts 1.426 times Luna's energy in the band, and
    summed over both modes 1.298 times as much. The arrival times are 4.730 fs and
    4.622 fs. All four numbers are stable to 1e-4 across two decades of step tolerance,
    so they are what the two codes mean, not what their steppers did.

    With each mode in its own frame these were 1.560 and 0.664, and an arrival of
    3.966 fs against the same 4.622 fs. The band summed over both modes was the number
    that moved most: with each mode in its own frame the second one was empty, so the
    total was *below* Luna's while the fundamental alone was above it. Both are now on
    the same side and the arrival agrees to 0.11 fs rather than 0.66 fs.
    """
    grid = witness_grid(REDUCED)
    vuv = band_mask(grid, VUV)
    ours = reduced.fields("fft")[-1, :, 0, :]
    theirs = luna_reduced[-1]
    fundamental = np.sum(np.abs(ours[0, vuv]) ** 2) / np.sum(
        np.abs(theirs[0, vuv]) ** 2
    )
    total = np.sum(np.abs(ours[:, vuv]) ** 2) / np.sum(np.abs(theirs[:, vuv]) ** 2)
    assert 1.35 < fundamental < 1.50
    assert 1.22 < total < 1.37
    ours_arrival = arrival_of(grid, ours, vuv)
    theirs_arrival = arrival_of(grid, theirs, vuv)
    assert 4.4e-15 < ours_arrival < 5.1e-15
    assert 4.2e-15 < theirs_arrival < 5.0e-15
    # Launching the nominal energy instead of Luna's own pulse moves the two band
    # ratios by 6.2e-5 and 1.05e-4 and solprop's arrival by 2.1e-5 relative. Luna's
    # arrival does not move at all, as it must not -- nothing on this side of the
    # comparison reaches it. All four are well inside the bands and the rtol=0.02.
    record(
        M2_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows(
            "reduced case",
            [
                ("VUV band in the fundamental", float(fundamental), 1.35, 1.50),
                ("VUV band over both modes", float(total), 1.22, 1.37),
                ("VUV arrival, solprop", 1e15 * ours_arrival, 4.4, 5.1),
                ("VUV arrival, Luna", 1e15 * theirs_arrival, 4.2, 5.0),
            ],
        ),
    )


@pytest.mark.slow
def test_the_higher_modes_agree_now_that_the_set_shares_one_frame(
    reduced, luna_reduced, request
):
    """The second mode carries 1.106 times Luna's, not 1/90.

    With each mode in its own frame this was 0.011, and a comparison alone could not
    say which code was right. The two oracles above settled it: Luna's own projection of
    a field that is the fundamental alone returns the analytic quartic overlap to
    1.9e-12 (above), and on the undepleted-pump case both codes reproduce first-order
    perturbation theory to 1e-5 (above). What was wrong was solprop's linear operator,
    which gave every mode its own co-moving frame and so let the modes walk apart in
    time by ``(beta_1,p - beta_1,1) z`` -- 43 fs per metre between these two -- while
    the nonlinear term went on multiplying them at the same time sample.

    On the full four-mode case the three higher modes go from 1.7e-3, 1.4e-5 and 1.7e-5
    of Luna's to 0.564, 1.080 and 1.165; that case is the ``benchmark``-marked test at
    the end of this file, and what is left of its 0.564 is **Luna's step control, not
    either code's physics**: at ``rtol = 1e-9`` Luna's second mode is 1.77 times smaller
    than at the ``1e-6`` its interface cannot change, and the two codes then agree on it
    to 0.44 %. The reduced case's 1.106 is against the same unconverged reference and
    is not corrected for it either: the witness is production Luna, deliberately.

    The band below is the measurement, not a tolerance: it fails if either code changes.
    """
    ours = reduced.fields("fft")[-1, :, 0, :]
    theirs = luna_reduced[-1]
    ratio = float(np.sum(np.abs(ours[1]) ** 2) / np.sum(np.abs(theirs[1]) ** 2))
    assert 1.0 < ratio < 1.2
    # Launching the nominal energy instead of Luna's own pulse moves this by
    # 2.54e-4 relative, the largest of the reduced M2 moves and two orders inside the
    # band.
    record(
        M2_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows("reduced case", [("second mode, content", ratio, 1.0, 1.2)]),
    )


# ------------------------------------------------------------------- the ladders
#
# Solprop against itself on the reduced case. Fixed schedules rather than tolerances,
# so that the refinement is a step size and an order can be read.
#
# `fixed_ladder` keeps the case's nominal 275 uJ rather than `luna_launch_energy`'s
# corrected one: nothing here is compared with Luna, an order is what it measures, and
# 0.0036 % of launch energy does not move one.


# The record of the order study of `docs/dev/accuracy.md` section 5. Four switches,
# each a three-level fixed-step ladder on the reduced case, and the point of the figure
# is that the case's order is its own physics rather than the scheme's. A fixed
# schedule is a deterministic trajectory, so above round-off the differences reproduce;
# the tolerance is a twentieth, which is the width of the claim each rung makes and
# leaves room for a different transform accumulated over six hundred steps.
M2_ORDER = Study(
    name="m2_order",
    title="The M2 case's convergence order, one switch at a time",
    case={
        "case": "the reduced M2 capillary (2 modes, 1 m) of tests/data/luna_rdw.json",
        "media": "Luna's own helium susceptibility and cladding table",
        "metric": "relative L2 difference of the in-band spectral energy density "
        "between successive levels, normalised by the finest level",
        "schedule": "uniform fixed steps at 250, 500 and 1000",
        "plasma_source": "SEP 27 held current, A_J, when plasma is enabled",
    },
    axis={"name": "n_steps", "unit": "steps", "label": "coarse level of the pair"},
    columns=(
        Column(
            name="difference",
            unit="1 (relative to the finest level's norm)",
            why="a twentieth: a fixed schedule reproduces above round-off, and this "
            "is the width of the claim each rung makes about the case's order",
            rtol=0.05,
            atol=1e-12,
        ),
        Column(
            name="order",
            unit="1",
            why="what a twentieth in each of two differences costs a log2 ratio; null "
            "on the coarser pair, which has nothing to be a ratio with",
            rtol=0.0,
            atol=0.2,
        ),
    ),
)

M2_SWITCHES = {
    "constant fill, no plasma": {"pressure": None, "plasma": False},
    "constant fill, plasma": {"pressure": None},
    "vacuum gradient, no plasma": {
        "pressure_floor": 0.0,
        "vacuum_end": "floor",
        "plasma": False,
    },
    "vacuum gradient, plasma": {"pressure_floor": 0.0, "vacuum_end": "floor"},
}
"""The four rows of the order table, as keyword sets for :func:`fixed_ladder`.

``pressure`` of ``None`` in the first two is filled in with the case's own inlet
pressure by :func:`order_rows`; ``fixed_ladder`` reads ``None`` as "take the gradient",
which is what the other two want.
"""

ORDER_COUNTS = (250, 500, 1000)


def order_rows(switch, differences):
    """Return the record rows of one switch of the order study."""
    return [
        {
            "series": switch,
            "axis": count,
            "difference": float(difference),
            "order": None
            if index == 0
            else float(observed_order(differences[index - 1], difference)),
        }
        for index, (count, difference) in enumerate(
            zip(ORDER_COUNTS[:-1], differences, strict=True)
        )
    ]


def switch_ladder(switch):
    """Run one switch of the order study and return its two successive differences."""
    grid = witness_grid(REDUCED)
    keywords = dict(M2_SWITCHES[switch])
    if "pressure" in keywords:
        keywords["pressure"] = REDUCED["p_in"]
    return fixed_ladder(
        ORDER_COUNTS, stop=REDUCED["length"], band=grid.inband_mask, **keywords
    )


def fixed_ladder(counts, *, stop, band, pressure=None, **overrides):
    """Return the successive differences of a three-level fixed-step ladder.

    Parameters
    ----------
    counts : sequence of int
        Step counts, each twice the last.
    stop : float
        Where the ladder's runs end, m.
    band : numpy.ndarray
        Mask of the frequencies the observable is measured over.
    pressure : float, pair of float or None, optional
        The fill, in bar; ``None`` takes the reduced case's own gradient.
    **overrides : object
        Keywords replacing the reduced case's own.

    Returns
    -------
    list of float
        ``[|Q0 - Q1|, |Q1 - Q2|]`` for the spectral energy density, relative to the
        finest level's norm.

    The order measured is the Runge-Kutta march's, so the march is named. Under the
    default Adams history a fixed ladder would start each rung at order one and keep
    the in-stage collar's ``h alpha / 2`` below 0.0877 -- about 1440 steps a rung on
    this case -- so its convergence is the adaptive regression's to show.
    """
    grid = witness_grid(REDUCED)
    fill = luna_helium_fill(
        REDUCED,
        pressure,
        pressure_floor=overrides.pop("pressure_floor", None),
        vacuum_end=overrides.pop("vacuum_end", None),
    )
    if overrides.get("plasma", True) is not False:
        overrides["plasma"] = m2_helium_rate()
    # The stored provenance and the selected-backend lane must identify the
    # implementation that actually marches, even when another plugin is installed.
    overrides.setdefault("backend", TEST_BACKEND)
    levels = []
    for n_steps in counts:
        output = prop_capillary(
            REDUCED["radius"],
            REDUCED["length"],
            fill,
            grid=grid,
            lambda0=WITNESS["lambda0"],
            fwhm=WITNESS["tau_fwhm"],
            energy=WITNESS["energy"],
            modes=REDUCED["modes"],
            cladding=luna_cladding("REDUCED", grid),
            save_planes=[stop],
            schedule=FixedSchedule.uniform(
                0.0, stop, stop / n_steps, save_planes=[stop]
            ),
            atol=LUNA_TOLERANCES["atol"],
            integrator=RungeKutta(),
            **{"noise": None, **overrides},
        )
        levels.append(np.abs(output.fields("fft")[-1, :, 0, :]) ** 2)
    scale = float(np.linalg.norm(levels[-1][:, band]))
    return [
        float(np.linalg.norm((levels[k] - levels[k + 1])[:, band]) / scale)
        for k in range(2)
    ]


@pytest.mark.slow
def test_the_scheme_shows_its_own_order_on_the_smooth_version_of_the_case(request):
    """A constant fill and no plasma: everything in z is smooth, and DOPRI5 is fifth.

    This is the control for the ladder below. Measured differences 2.922e-5 and
    3.938e-7 at 250, 500 and 1000 steps, so order 6.21 -- above five because the coarse
    rung is not yet asymptotic, which the reference backend's soliton ladder sees too.

    The counts are 250, 500 and 1000 and not 150, 300 and 600, because the in-stage
    absorber makes the collar's ceiling rate an eigenvalue of the linearised problem
    and 150 steps over this metre is past the stability limit of 5.5 mm. The problem
    says so at construction rather than diverging quietly.
    """
    switch = "constant fill, no plasma"
    differences = switch_ladder(switch)
    assert observed_order(differences[0], differences[1]) > 4.5
    record(
        M2_ORDER,
        request=request,
        backend=TEST_BACKEND,
        rows=order_rows(switch, differences),
    )


@pytest.mark.slow
def test_the_m2_case_is_first_order_and_the_two_reasons_are_known(request):
    """The RDW case itself converges at order one, and the reason is not the solver.

    Measured on this case at 250, 500 and 1000 steps, changing one thing at a time:

    ==============================  =====
    constant fill, no plasma         6.21
    constant fill, plasma            5.95
    vacuum gradient, no plasma       1.42
    vacuum gradient, plasma          1.59
    ==============================  =====

    **It is the vacuum end, alone.** A fill that reaches a true vacuum has ``dp/dz``
    diverging at the exit, which ``solprop.profiles`` warns about at construction and
    ``docs/dev/stepper.md`` section 10 records as trap 5; no step size fixes it. The
    plasma costs nothing here: at 150, 300 and 600 steps it appears to cost order
    (3.31 against 5.77), and at 250, 500 and 1000 it does not (5.95 against 6.21). The
    ionisation switch-on is resolved once the steps are fine enough and the vacuum end
    never is, so the coarser ladder attributes the gradient's cost to the plasma.

    The order gate of 4.5 is therefore met by the smooth version above and not by
    this one, which is a property of the physics and not a defect: the *answer* is
    converged to 1e-4 in the VUV band across two decades of tolerance, as the
    regression above measures.
    """
    switch = "vacuum gradient, plasma"
    differences = switch_ladder(switch)
    order = observed_order(differences[0], differences[1])
    assert 0.7 < order < 2.5
    record(
        M2_ORDER,
        request=request,
        backend=TEST_BACKEND,
        rows=order_rows(switch, differences),
    )


@pytest.mark.benchmark
def test_the_other_two_switches_of_the_order_study(request):
    """The two rows of the order table that no lane runs, for the record.

    The two slow tests above are the ends of the table -- the smooth control and the
    case itself -- and are what milestone M2's convergence claim rests on. The figure
    of `docs/dev/accuracy.md` section 5 needs the middle two, which isolate one cause
    each, and running all four in the slow lane would put another two and a half
    thousand fixed steps in a four-minute budget for no assertion.

    Run it, and rewrite its rows of ``tests/data/convergence/m2_order.json``, with

        SOLPROP_WRITE_RECORDS=1 uv run pytest \
            tests/driver/test_prop_capillary.py -m benchmark -p no:randomly \
            -k the_other_two_switches

    from a clean tree. Measured on this tree at 250, 500 and 1000 steps, with the four
    rows of the table run back to back on `numba_cpu` (not at 150, 300 and 600: the
    in-stage absorber has a stability limit of 5.5 mm on this metre, and 150 steps is
    past it):

    ==============================  =====
    constant fill, no plasma         6.21
    constant fill, plasma            5.95
    vacuum gradient, no plasma       1.42
    vacuum gradient, plasma          1.59
    ==============================  =====

    **The vacuum end is the whole of it, and the plasma costs nothing.** The coarser
    rungs read 5.77, 3.31, 4.49 and 1.08, and another run of them 5.08, 4.75, 1.52 and
    0.99; at the finer rungs the ionisation switch-on is resolved and the
    plasma row sits at the scheme's own order, while the vacuum end never is. The
    reading the page takes from it is the diverging ``dp/dz`` of
    `docs/dev/stepper.md` section 10, trap 5, alone.
    """
    orders = {}
    for switch in ("constant fill, plasma", "vacuum gradient, no plasma"):
        differences = switch_ladder(switch)
        orders[switch] = observed_order(differences[0], differences[1])
        record(
            M2_ORDER,
            request=request,
            backend=TEST_BACKEND,
            rows=order_rows(switch, differences),
        )
    # The plasma alone leaves the scheme near its own order; the vacuum end alone does
    # not. Both bands are wide, because what this test is for is the record.
    assert orders["constant fill, plasma"] > 4.0
    assert 1.0 < orders["vacuum gradient, no plasma"] < 2.5


@pytest.mark.slow
def test_the_quadrature_ladder_converges_and_the_alarm_falls_with_it(request):
    """The VUV band and the alarm against the radial order, and the threshold measured.

    On this two-mode set the fixed rule is already converged at nine nodes: the band's
    energy moves by 6.7e-5 between ``n_r = 8`` and ``n_r = 48``, while the alarm falls
    1.38e-1, 2.96e-9, 4.24e-11 (the committed record; an earlier tree read 2.0e-5 and
    1.22e-1, 1.44e-9, 2.37e-11). That is the measurement the alarm's threshold was
    chosen without: at an alarm five orders *above* its 1e-6 threshold, the observable
    the run is about is wrong by seven parts in a hundred thousand. The threshold is
    conservative here by a wide margin -- which is the right direction for an alarm,
    and is now a measurement rather than a judgement.
    """
    grid = witness_grid(REDUCED)
    vuv = band_mask(grid, VUV)
    energies, alarms = {}, {}
    for n_r in (8, 32, 48):
        output = witness_run(REDUCED, "REDUCED", n_r=n_r)
        density = np.abs(output.fields("fft")[-1, :, 0, :]) ** 2
        energies[n_r] = float(np.sum(density[:, vuv]))
        alarm = np.asarray(output["stats/quadrature_error"])
        finite = alarm[np.isfinite(alarm)]
        alarms[n_r] = float(np.max(finite)) if finite.size else math.nan
    # The rule converges: the two resolved orders agree far more closely than either
    # agrees with the coarse one.
    resolved = abs(energies[48] / energies[32] - 1.0)
    coarse = abs(energies[8] / energies[48] - 1.0)
    assert resolved < 1e-9
    assert 5e-6 < coarse < 1e-4
    # The alarm ranks the three rules correctly and spans eight orders.
    assert alarms[8] > 1e-2
    assert alarms[32] < QUADRATURE_ALARM
    assert alarms[48] < alarms[32]
    # Launching Luna's own pulse rather than the nominal energy moves the three band
    # energies by 1.05e-4 relative, in step, so neither the 6.7e-5 the ladder is about
    # nor the alarm's eight orders moves with them; the alarms move by 9.8e-6, 5.6e-4
    # and 5.1e-3, inside the record's rtol of 1e-3 and 1.0.
    record(
        M2_ALARM_COST,
        request=request,
        backend=TEST_BACKEND,
        rows=[
            {
                "series": "reduced M2 case",
                "axis": n_r,
                "vuv_energy": energies[n_r],
                "alarm": alarms[n_r],
            }
            for n_r in (8, 32, 48)
        ],
    )


# ======================================== milestone M4, CtC-type propagation
#
# The reduced case of `tests/data/generate_luna_ctc.py` against the Luna result stored
# beside it: hydrogen at 40 bar in a 16 um capillary, 400 fs and 10 uJ at 515 nm, the
# fundamental mode alone on a 2 ps, 250 nm-2 um field grid over 0.1 m, with the Raman
# response and the molecular plasma. The comparison with Luna is noise-free, and
# noise-free, with the launch clear of the window, neither code forms a continuum over
# this length: nothing seeds the Stokes band. Seeded with shot noise, solprop's run
# forms the Raman cascade and bridges the octave
# (`test_the_seeded_reduced_ctc_case_forms_the_raman_cascade`). Everything that
# propagates here is slow.
#
# What the comparison controls for, so that what is left is the solver:
#
# * the grid -- asserted below, to one unit in the last place;
# * the hydrogen susceptibility -- unlike helium, Luna's hydrogen entry is referenced to
#   the state its fit is quoted at (it divides the Peck expansion by the 1 atm density,
#   where the helium entry divides Ermolov by the 1 bar one), and the
#   coefficients are the same numbers solprop carries, so the package fit *is* Luna's
#   and no table is needed for the gas;
# * chi3 -- one number per particle, agreeing to the same 1.6e-8 reference-density
#   offset, which cancels against the equation of state;
# * the cladding -- Luna's is an unreferenced 400-row table, so the mode is
#   given a spline through Luna's own values on this grid;
# * the temperature and the shot noise -- 294.0 K and none, on both sides, both
#   stated rather than defaulted, as is every other input of the run
#   (`witness_inputs`: the witness is frozen, docs/dev/testing.md);
# * the launch -- `ctc_run` asks for the *true* energy of Luna's pulse, 1.0049222310
#   times the nominal energy Luna asked for, because a requested energy is a true one
#   in solprop (ADR-030, docs/dev/decisions.md) while Luna's is its vacuum amplitude's;
# * the input state -- the two codes' plane-zero spectra agree to 1.8e-13 in total
#   energy, asserted below, so the pulse, the mode and the normalisation are not what
#   any later difference is.
#
# What is left, under the witness's vibrational amplitude, is a case where the two
# codes agree on the fundamental's spectrum to 3.8 %, on its content and the exit
# energy to 1e-5, and on the plasma to 1e-6. Until 2026-10-24 the case ran on a 1 ps
# window with two modes; its launch reached the temporal collar, both codes' Raman
# cascades grew from that clipped edge, and the continuum and the disagreements
# recorded for it were the boundary's, not the solvers'.

CTC_WITNESS = json.loads((Path(__file__).parents[1] / "data/luna_ctc.json").read_text())
CTC_MEDIA = json.loads(
    (Path(__file__).parents[1] / "data/luna_ctc_media.json").read_text()
)
CTC_REDUCED = CTC_WITNESS["reduced"]


def ctc_grid(case):
    """Return the grid a stored CtC case was run on, built from its own parameters."""
    return FieldGrid.from_limits(
        case["length"],
        CTC_WITNESS["lambda0"],
        (case["lambda_min"], case["lambda_max"]),
        case["trange"],
        **WITNESS_GRID_RULES,
    )


def ctc_run(
    case,
    tag,
    *,
    planes=None,
    backend=TEST_BACKEND,
    luna_vibration=False,
    seed=None,
):
    """Run CtC with corrected vibration, or explicitly select the old witness model.

    A `seed` launches the case with input shot noise of that seed; without one the run
    is noise-free, as the witness is.
    """
    if luna_vibration and backend != "numpy_ref":
        raise ValueError("the historical CtC witness requires the NumPy gated hook")
    grid = ctc_grid(case)
    context = _gated_source() if luna_vibration else nullcontext()
    with warnings.catch_warnings(), context:
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            case["radius"],
            case["length"],
            # Luna's material inputs, supplied: hydrogen law, hyperpolarisability,
            # Raman constants and the potential-only ADK rate Luna's own runs use.
            luna_hydrogen_fill(case["pressure"], CTC_WITNESS["temperature_k"]),
            grid=grid,
            lambda0=CTC_WITNESS["lambda0"],
            fwhm=CTC_WITNESS["tau_fwhm"],
            # As in `witness_run`: Luna's nominal energy is its vacuum amplitude's, and
            # 40 bar of hydrogen in a 16 um core makes w_bar 1.0049222310, so
            # requesting the nominal energy would launch 0.492 % less than Luna did.
            # A cascade is nonlinear in its pump, which is why that half per cent is
            # worth several per cent on a Stokes rung (ADR-030).
            energy=luna_launch_energy(case, tag, ctc=True),
            modes=case["modes"],
            loss=False,
            raman=True,
            plasma=luna_hydrogen_rate(),
            # The witness was made by Luna's cumulative trapezoid (SEP 18).
            plasma_integration="trapezoid",
            temperature=CTC_WITNESS["temperature_k"],
            cladding=luna_cladding(tag, grid, CTC_MEDIA),
            n_save=case["n_save"] if planes is None else planes,
            rtol=LUNA_TOLERANCES["rtol"],
            atol=LUNA_TOLERANCES["atol"],
            backend=backend,
            **witness_inputs(grid, case["modes"])
            | ({"noise": None} if seed is None else {"noise": "input", "seed": seed}),
        )
        if luna_vibration:
            # Test-only matching of the archived witness's physical model, not
            # a production compatibility mode. The default fixture below retains
            # corrected vibration and still exercises the octave/plasma/grid
            # checks. Only the historical M4 numerical bounds use this override.
            responses = []
            for response in simulation.responses:
                updated_response = response
                if isinstance(response, RamanResponse):
                    terms = tuple(
                        replace(
                            term,
                            model=replace(
                                term.model, amplitude=term.model.amplitude / 2
                            ),
                        )
                        if isinstance(term.model, DampedOscillator)
                        else term
                        for term in response.tensor_terms
                    )
                    updated_response = replace(response, tensor_terms=terms)
                responses.append(updated_response)
            simulation.responses = tuple(responses)
        return simulation.run()


def ctc_field_range(case):
    """Return the ionisation table's field span, V/m, that a CtC case is built with."""
    grid = ctc_grid(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            case["radius"],
            case["length"],
            luna_hydrogen_fill(case["pressure"], CTC_WITNESS["temperature_k"]),
            grid=grid,
            lambda0=CTC_WITNESS["lambda0"],
            fwhm=CTC_WITNESS["tau_fwhm"],
            energy=CTC_WITNESS["energy"],
            modes=case["modes"],
            loss=False,
            raman=True,
            plasma=luna_hydrogen_rate(),
            # The witness was made by Luna's cumulative trapezoid (SEP 18).
            plasma_integration="trapezoid",
            temperature=CTC_WITNESS["temperature_k"],
            n_save=2,
            noise=None,
        )
    return simulation.settings.ionisation_field_range


@pytest.fixture(scope="module")
def ctc_reduced(test_backend_name):
    """The reduced CtC case at Luna's tolerance: the run every metric is measured on."""
    return ctc_run(CTC_REDUCED, "REDUCED", backend=test_backend_name)


CTC_SEED = 21
"""The shot-noise seed of the seeded reduced CtC runs."""

# Either side of the 515 nm pump: an octave about it is 364-1030 nm, so a run that fills
# both shoulders has bridged it.
CTC_BLUE = (330e-9, 430e-9)
CTC_RED = (900e-9, 1300e-9)


def spectral_span(grid, spectrum, floor=1e-4):
    """Return the shortest and longest wavelength, m, a spectrum clears a floor at."""
    positive = grid.omega > 0.0
    wavelength = np.where(
        positive, 2.0 * math.pi * C / np.where(positive, grid.omega, 1.0), np.inf
    )
    total = np.sum(np.abs(spectrum) ** 2, axis=tuple(range(spectrum.ndim - 1)))
    kept = positive & (total > floor * np.max(total[positive]))
    return float(np.min(wavelength[kept])), float(np.max(wavelength[kept]))


def band_share(grid, spectrum, limits):
    """Return the fraction of a spectrum's energy between two wavelengths."""
    mask = band_mask(grid, limits)
    return float(
        np.sum(np.abs(spectrum[..., mask]) ** 2) / np.sum(np.abs(spectrum) ** 2)
    )


def vibrational_rung(order):
    """Return the 12 % band about hydrogen's `order`-th vibrational Stokes line, m."""
    centre = C / (C / CTC_WITNESS["lambda0"] - order * 124.57e12)
    return (0.94 * centre, 1.06 * centre)


@pytest.mark.slow
def test_the_seeded_reduced_ctc_case_forms_the_raman_cascade():
    """With shot noise as its seed, the reduced case forms the cascade over 0.1 m.

    Noise-free it does not: nothing seeds the Stokes band but rounding, which grows to
    5 % of the energy by the exit. Input shot noise is the physical seed. Measured on
    `numba_cpu` at the exit for seeds 21, 22 and 23: the Stokes band (560 nm-2 um)
    holds 0.48, 0.44 and 0.44 of the energy; the first three vibrational rungs
    0.26/0.11/0.10, 0.21/0.16/0.058 and 0.24/0.10/0.086; the 330-430 nm shoulder
    0.026, 0.041 and 0.040; the 900-1300 nm one 0.042, 0.046 and 0.038; and at a
    ten-thousandth of its peak the exit spectrum spans 349-1614, 311-1610 and
    355-1606 nm, bridging the 364-1030 nm octave about the pump. The bounds
    hold for every seed measured rather than fitting this one, so the test asks that
    the cascade forms, not that one realisation repeats. solprop's own physics only:
    Luna's noise is its own realisation, so the comparison with Luna stays
    noise-free.
    """
    grid = ctc_grid(CTC_REDUCED)
    run = ctc_run(CTC_REDUCED, "REDUCED", seed=CTC_SEED)
    exit_spectrum = run.fields("fft")[-1, :, 0, :]
    assert band_share(grid, exit_spectrum, (560e-9, 2e-6)) > 0.3
    for order in (1, 2, 3):
        assert band_share(grid, exit_spectrum, vibrational_rung(order)) > 0.01
    for shoulder in (CTC_BLUE, CTC_RED):
        assert band_share(grid, exit_spectrum, shoulder) > 0.01
    low, high = spectral_span(grid, exit_spectrum)
    assert low < 364e-9
    assert high > 1030e-9


@pytest.mark.slow
def test_the_reduced_ctc_native_field_matches_fresh_numba(test_backend_name):
    """The native backend against a fresh ``numba_cpu`` run, seeded by shot noise.

    Hydrogen's Raman gain amplifies whatever seeds the Stokes band. Noise-free, that is
    each backend's own rounding: the two agree to 1.7e-15 at the first plane and only
    to 6e-6 at the exit, where 5 % of the energy is Stokes light grown from it. Input
    shot noise of a fixed seed is the same physical seed on both, about 1e9 times
    the rounding in amplitude, so the comparison at the exit measures the backends
    again: 5.2e-11 at this seed (1.4e-10 at the next), at identical accepted steps
    (4937), with 48 % of the energy in the Stokes band.
    """
    if test_backend_name != "native":
        pytest.skip("run with SOLPROP_TEST_BACKEND=native for the backend comparison")
    native = ctc_run(CTC_REDUCED, "REDUCED", backend="native", seed=CTC_SEED)
    numba = ctc_run(CTC_REDUCED, "REDUCED", backend="numba_cpu", seed=CTC_SEED)
    assert int(np.sum(native["step_history/accepted"])) == int(
        np.sum(numba["step_history/accepted"])
    )
    actual = native.fields("fft")[-1]
    expected = numba.fields("fft")[-1]
    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 < 1e-9


@pytest.fixture(scope="module")
def luna_ctc_density():
    """Return the peak electron density Luna's reduced run reached, m^-3."""
    with h5py.File(Path(__file__).parents[1] / "data" / "luna_ctc_reduced.h5") as file:
        return float(
            np.max(np.asarray(cast("h5py.Dataset", file["stats/electrondensity"])[()]))
        )


@pytest.fixture(scope="module")
def luna_ctc_reduced():
    """Luna's stored spectra for the reduced CtC case, every plane."""
    return stored_field("luna_ctc_reduced.h5")


def test_the_ctc_witness_names_the_luna_it_came_from():
    assert len(CTC_WITNESS["luna_commit"]) == 40
    assert CTC_WITNESS["luna_branch"] == "modal-fixed"
    assert CTC_WITNESS["julia_version"].startswith("julia version")
    assert CTC_WITNESS["temperature_k"] == 294.0
    assert CTC_WITNESS["shotnoise"] is False
    # One mode on a 2 ps window: five times the 400 fs pulse, so the launch is clear of
    # the temporal collar (a 1 ps window clipped it and seeded the Raman cascade).
    assert CTC_REDUCED["modes"] == 1
    assert CTC_REDUCED["trange"] == 2e-12


def test_the_ctc_grid_is_the_one_luna_realised():
    """To one unit in the last place.

    The two codes reach the same fine sample spacing by different associations --
    solprop as ``2 pi / (ffac * 2 pi c / lambda_min)`` and Luna as
    ``1 / (ffac * c / lambda_min)`` -- and the round trip through the reciprocal is
    inexact at the case's 250 nm, where it costs about 2e-16 in the bin spacing and
    3e-16 across the axis. Nothing is interpolated: the two axes have the same length
    and the same bins to that rounding.
    """
    reduced = ctc_grid(CTC_REDUCED)
    np.testing.assert_allclose(
        reduced.omega, np.asarray(CTC_MEDIA["REDUCED_OMEGA"]), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        reduced.t, np.asarray(CTC_MEDIA["REDUCED_T"]), rtol=1e-15, atol=1e-27
    )


def test_lunas_hydrogen_is_the_peck_fit_unchanged():
    """Luna's helium normalisation is not repeated for hydrogen: no table is needed.

    ``sellmeier_gas(:H2)`` divides the Peck expansion by the **1 atm** density, which is
    the state Peck and Hung quote it at, where the helium entry divides Ermolov by the
    1 bar density although that fit is referenced to 1 atm as well. The coefficients are
    the numbers :func:`luna_hydrogen` supplies, so the only difference left is each
    code's equation of state at the reference state.
    """
    grid = ctc_grid(CTC_REDUCED)
    band = grid.active_mask
    ours = luna_hydrogen().susceptibility(grid.omega[band])
    theirs = np.asarray(CTC_MEDIA["REDUCED_GAMMA"])[band]
    ratio = ours / theirs
    # Measured -1.112e-6 everywhere and flat in frequency (spread 2.4e-12): optmat's
    # hydrogen equation of state against Luna's CoolProp at 1 atm and 0 C, the same
    # kind of offset the helium comparison sees. (Before SEP 3, Phase 4b it was
    # solprop's own CoolProp call, 1.611e-8.)
    assert np.ptp(ratio) < 1e-11
    assert abs(float(np.mean(ratio)) - 1.0) < 2e-6


@pytest.mark.slow
def test_the_reduced_ctc_case_ionises(ctc_reduced, luna_ctc_density):
    """The molecular ADK rate runs, and makes the plasma Luna makes.

    Measured: peak electron density 1.7097e12 m^-3 here and in Luna, to 1.1e-6. The
    witness runs the ADK oracle of ``tests/rate_oracles.py`` on the former potential,
    which matches Luna's own sampled hydrogen rate to 3.4e-6
    (`test_the_witness_rate_is_lunas_sampled_hydrogen_rate`), and with the launch clear
    of the window edges the peak field stays at the input's 9.06e9 V/m in both codes.
    On the 1 ps window both made two to three orders of magnitude more, and disagreed by
    a factor of a thousand, from launched field compressing in each code's temporal
    collar.

    The rate table is not clipped: the run reaches 9.06e9 V/m against the 2.71e11 upper
    limit `default_field_range` derives from the input's own 9.04e9 peak, a thirtyfold
    margin.
    """
    density = float(np.max(np.asarray(ctc_reduced["stats/electron_density"])))
    assert density > 0.0
    assert abs(density / luna_ctc_density - 1.0) < 1e-3
    intensity = float(np.max(np.asarray(ctc_reduced["stats/peak_intensity"])))
    peak_field = math.sqrt(2.0 * intensity / (EPS0 * C))
    span = ctc_field_range(CTC_REDUCED)
    assert span is not None
    low, high = span
    assert low < peak_field < high
    # A thirtyfold margin, so the field would have to grow by an order of magnitude
    # more than it does before the table's silent clip could reach this case.
    assert high / peak_field > 5.0


@pytest.mark.slow
def test_the_reduced_ctc_quadrature_is_resolved(ctc_reduced):
    # Two modes at the default radial order of 32. Measured 3.49e-9, five orders below
    # the alarm; the plane-zero sample is nan by construction, no step having been
    # accepted yet, which is why this is a nanmax.
    alarm = np.asarray(ctc_reduced["stats/quadrature_error"])
    assert float(np.nanmax(alarm)) < QUADRATURE_ALARM
    assert int(np.isnan(alarm).sum()) == 1


@pytest.mark.slow
def test_the_two_codes_start_from_the_same_state(ctc_reduced, luna_ctc_reduced):
    # Plane zero, before anything has propagated: the pulse, the mode projection, the
    # grid and the energy normalisation, in one number. Measured 1.8e-13, which bounds
    # what any later difference can be blamed on.
    ours = np.sum(np.abs(ctc_reduced.fields("fft")[0, :, 0, :]) ** 2)
    theirs = np.sum(np.abs(luna_ctc_reduced[0]) ** 2)
    # Dividing the index out with the run's *own* recorded weight would pass whatever
    # that weight happened to be, so rebuild it from the modes, exactly as
    # `stored_field` converts Luna's archived amplitudes.
    grid = ctc_grid(CTC_REDUCED)
    independent = np.ones((CTC_REDUCED["modes"], grid.n_omega))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        for index in range(CTC_REDUCED["modes"]):
            mode = MarcatiliMode(
                CTC_REDUCED["radius"],
                "H2",
                CTC_REDUCED["pressure"],
                m=index + 1,
                temperature=CTC_WITNESS["temperature_k"],
                # The run's own hydrogen, Luna's law (not optmat's default).
                susceptibility=luna_hydrogen(),
                cladding=luna_cladding("REDUCED", grid, CTC_MEDIA),
                model="full",
                loss=False,
            )
            independent[index, grid.active_mask] = np.real(
                mode.neff(grid.omega[grid.active_mask], 0.0)
            )
    # One direct index evaluation against the record's factored expansion, which
    # resolves the same number through a Clenshaw sum: the gap is that sum's round-off,
    # bounded at 1e-14, which the translation tests measure at 6.7e-15.
    np.testing.assert_allclose(
        get_power_weight(ctc_reduced)[0, :, 0], independent, rtol=1e-14, atol=0
    )
    # The run asks for the true energy of Luna's launch, w_bar = 1.0049222310 times
    # the nominal one, so the two codes are pumped alike and the two plane-zero states
    # compare directly, with no predicted conversion between them (the nominal energy
    # would put the raw ratio 0.4900125 % out). The measurement is 1.8e-13 against the
    # bound of 1e-5. On the former 1 ps window it was 2.014e-6, most of it in Luna's
    # own pulse normalisation (an extended Simpson rule, whose stored plane zero then
    # carried 1 + 1.570e-6 of the nominal energy); the 2 ps window no longer shows it.
    # Building the launch from the case and the mode rather than from Luna's plane is
    # what keeps this a measurement at all.
    assert abs(float(ours / theirs) - 1.0) < 1e-5


@pytest.mark.slow
def test_the_reduced_ctc_metrics_are_bounded_at_milestone_m4(
    luna_ctc_reduced, request, test_backend_name
):
    """Bound the solver against Luna under an explicitly matched Raman model.

    solprop's scalar vibrational Raman amplitude is twice the witness's (Luna's
    prefactor has ``4 mu Omega`` where the induced-dipole Hamiltonian gives
    ``2 mu Omega``). Here only, the witness's weaker vibration is restored before
    propagation, so that these bounds compare the two solvers on the same physics; the
    default run (spectral distance 0.409, content 0.983) is exercised by the ionisation,
    quadrature and backend checks.

    Measured at Luna's tolerance on `numpy_ref`, solprop against Luna, at the exit
    plane: the fundamental's in-band spectral energy density differs by **0.0383**
    (relative L2), its content is **1 + 6.1e-6** of Luna's, and the exit energy is
    **1 + 6.9e-6** of the launch here against **1 + 7.4e-7** in Luna. None of these
    is a continuum: noise-free, the case forms none, and with shot noise the two codes'
    cascades grow from different realisations. The 1 ps window's clipped launch seeded
    a cascade in both codes alike, which is what these bounds once described (0.154,
    1.044, and a Stokes ladder); the seeded cascade is now solprop's own check.
    """
    del test_backend_name
    ctc_reduced = ctc_run(
        CTC_REDUCED, "REDUCED", backend="numpy_ref", luna_vibration=True
    )
    grid = ctc_grid(CTC_REDUCED)
    fields = ctc_reduced.fields("fft")
    ours = fields[-1, :, 0, :]
    theirs = luna_ctc_reduced[-1]
    fundamental = in_band_distance(ours[0], theirs[0], grid.inband_mask)
    content = float(np.sum(np.abs(ours[0]) ** 2) / np.sum(np.abs(theirs[0]) ** 2))
    surviving = float(
        np.sum(np.abs(ours) ** 2) / np.sum(np.abs(fields[0, :, 0, :]) ** 2)
    )
    luna_surviving = float(
        np.sum(np.abs(theirs) ** 2) / np.sum(np.abs(luna_ctc_reduced[0]) ** 2)
    )

    # ---------------------------------------------------- milestone M4's bounds
    #
    # The shape is bounded at +-10 % of its measurement, two-sided: these are
    # deterministic runs against a fixed witness, so only a change in solprop can move
    # it, and a collapse would mean the witness had been regenerated with different
    # physics rather than that the solver had improved. The two energies sit within
    # 1e-5 of Luna's; their bounds are 1e-4, which a step sequence that differs across
    # platforms at the controller's 1e-6 cannot reach.
    assert 0.0345 < fundamental < 0.0422
    assert abs(content - 1.0) < 1e-4
    assert abs(surviving / luna_surviving - 1.0) < 1e-4
    record(
        M4_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows(
            "reduced case",
            [
                ("fundamental, shape", fundamental, 0.0345, 0.0422),
                ("fundamental, content", content, 1.0 - 1e-4, 1.0 + 1e-4),
                (
                    "exit energy, ours over Luna's",
                    surviving / luna_surviving,
                    1.0 - 1e-4,
                    1.0 + 1e-4,
                ),
            ],
        ),
    )


# ====================================== the statistics channel between planes


PERIODIC_SUBSET = ("energy", "peak_power", "fwhm_t", "electron_density")
"""The cost gate's default subset: three field statistics and the plasma one."""

GATE = 0.25
"""The channel must cost under a quarter of a step at ``period = 1``.

Not a tenth, because the measurement is a wall-clock ratio taken on a shared machine,
where it has read 10.57 % under load for a change that costs 0.7 percentage points, and
a gate that fails on load says nothing worth tuning for.
"""


def reduced_m2_simulation(**overrides):
    """Return the reduced M2 case as a `Simulation`, so the problem can be reused."""
    case, tag = REDUCED, "REDUCED"
    grid = witness_grid(case)
    overrides.setdefault("backend", TEST_BACKEND)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            case["radius"],
            case["length"],
            # Luna's helium and the witness's rate, with Luna's true vacuum end, as
            # the witnesses and the records were made.
            luna_helium_fill(case),
            plasma=(plasma := overrides.pop("plasma", m2_helium_rate())),
            # The witnesses were made by Luna's cumulative trapezoid (SEP 18).
            plasma_integration=None if plasma is False else "trapezoid",
            grid=grid,
            lambda0=WITNESS["lambda0"],
            fwhm=WITNESS["tau_fwhm"],
            energy=WITNESS["energy"],
            modes=case["modes"],
            cladding=luna_cladding(tag, grid),
            loss=True,
            temperature=WITNESS["temperature_k"],
            raman=False,
            n_save=case["n_save"],
            rtol=LUNA_TOLERANCES["rtol"],
            atol=LUNA_TOLERANCES["atol"],
            **({"noise": None, **witness_inputs(grid, case["modes"]) | overrides}),
        )


@pytest.fixture(scope="module")
def reduced_m2_channel():
    """Return ``(simulation, plain, watched)``: one translation, two specs."""
    simulation = reduced_m2_simulation()
    with warnings.catch_warnings():
        # Helium at this span is outside the published fit, which every test of this
        # case accepts; the warning is the subject of its own test elsewhere.
        warnings.simplefilter("ignore", MaterialRangeWarning)
        plain = simulation.translate()
    spec = StatsSpec(names=PERIODIC_SUBSET, periodic=PERIODIC_SUBSET, period=1)
    return simulation, plain, replace(plain, stats_spec=spec)


@pytest.mark.slow
def test_the_statistics_channel_costs_under_a_quarter_of_a_step(
    record_property, tmp_path
):
    """The cost gate, on the reduced M2 case at ``period = 1``.

    Ten fixed Runge-Kutta steps to 2 mm retain the real grid, modes and plasma while
    keeping the measured propagation window short. The measurement uses
    three fresh processes per configuration, rotated paired order, and
    medians with unscaled MAD. Each process warms its arm before three timed runs.
    ``poe test-slow`` runs this check after its parallel functional tests finish;
    unrelated workers competing with only one arm can dominate a wall-clock ratio.

    The plasma statistics reuse the accepted Runge-Kutta step's last-stage scan.
    The default Adams step evaluates its source at the predicted state, so its rows
    scan the accepted field afresh. This cost witness therefore selects Runge-Kutta
    explicitly (``docs/guide/results/statistics.md``).
    """
    # Both coverage and the session's runtime type hook alter the ratio.
    # The same fixture and strict assertion run in production imports; functional
    # channel tests retain both instruments and complete source coverage.
    results = {"plain": [], "watched": []}
    for repetition in range(3):
        order = ("plain", "watched") if repetition % 2 == 0 else ("watched", "plain")
        for selection in order:
            report = tmp_path / f"statistics-cost-{repetition}-{selection}.json"
            command = [
                sys.executable,
                "-W",
                "error",
                str(Path(__file__).parents[1] / "statistics_cost_probe.py"),
                str(report),
                selection,
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            label = f"statistics_child_{repetition}_{selection}"
            record_property(label, result.stdout + result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
            results[selection].append(json.loads(report.read_text()))
    counts = {
        (row["steps"], tuple(row["accepted"]))
        for records in results.values()
        for process in records
        for row in process["runs"]
    }
    assert len(counts) == 1
    assert counts == {(10, (True,) * 10)}
    histories = {
        (tuple(row["z"]), tuple(row["dz"]))
        for records in results.values()
        for process in records
        for row in process["runs"]
    }
    assert len(histories) == 1
    samples = {
        name: [
            statistics.median(row["seconds"] for row in item["runs"]) for item in arm
        ]
        for name, arm in results.items()
    }
    medians = {name: statistics.median(values) for name, values in samples.items()}
    mads = {
        name: statistics.median(abs(value - medians[name]) for value in values)
        for name, values in samples.items()
    }
    ratio = medians["watched"] / medians["plain"] - 1.0
    record_property("statistics_process_seconds", json.dumps(samples))
    record_property("statistics_medians", json.dumps(medians))
    record_property("statistics_mads", json.dumps(mads))
    record_property("statistics_cost_ratio", ratio)
    record_property("statistics_cost_bound", GATE)
    assert ratio < GATE, f"the channel cost {100 * ratio:.1f} % of the run"


def statistics_channel_measurement(reduced_m2_channel, selection):
    """Measure one warmed configuration with production imports."""
    assert Coverage.current() is None, "statistics timing requires an untraced worker"
    assert selection in ("plain", "watched")
    simulation, plain, watched = reduced_m2_channel
    problem = plain if selection == "plain" else watched
    # The stash a row is read from is the Runge-Kutta march's last stage.
    target = 0.002
    problem = replace(
        problem,
        stepper=replace(
            problem.stepper,
            integrator=None,
            # The schedule includes the fixture's declared save planes, although
            # each measurement advances only its first ten steps, to 2 mm.
            schedule=FixedSchedule.uniform(0.0, simulation.grid.length, target / 10),
        ),
    )

    def arm():
        backend = select(
            TEST_BACKEND,
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            backend.set_initial(problem.initial.field0, None)
            cpu_started = time.process_time()
            started = time.perf_counter()
            summary = backend.propagate_to(target)
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - cpu_started
            plane = backend.observe()
            # A row per accepted step: a rejected trial records none.
            accepted = int(np.sum(plane.step_history.accepted))
            assert plane.step_stats.n_rows == (
                accepted if selection == "watched" else 0
            )
            return {
                "seconds": elapsed,
                "cpu_seconds": cpu_elapsed,
                "steps": summary.n_steps,
                "accepted": plane.step_history.accepted.tolist(),
                "z": plane.step_history.z.tolist(),
                "dz": plane.step_history.dz.tolist(),
                "rows": plane.step_stats.n_rows,
                "metadata": backend.metadata(),
            }
        finally:
            backend.close()

    arm()
    return [arm() for _ in range(3)]


@pytest.mark.medium
def test_a_periodic_row_at_a_plane_is_the_plane_s_own_statistic_on_the_m2_case(
    reduced_m2_channel,
):
    """On a real multimode run with a plasma and a gradient fill.

    The row goes through the same :meth:`~solprop.backends.protocol.Backend.stats` the
    save plane uses, so where a row lands on a plane the two are the same statistic of
    one state, evaluated twice -- including the electron density, whose cheap path
    must therefore agree with the synthesis it replaces. Two evaluations of one
    statistic need not agree in their last bits (SEP 25: a compiled reduction may be
    threaded); they agree within both evaluations' rounding, the source evaluation's
    count (``tests/march_bounds.py``) taken as the statistic's.
    """
    simulation, _, watched = reduced_m2_channel
    backend = select(
        TEST_BACKEND,
        watched,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        backend.set_initial(watched.initial.field0, None)
        backend.propagate_to(0.06)
        observed = backend.observe()
        at_plane = backend.stats(watched.stats_spec)
    finally:
        backend.close()
    rows = observed.step_stats
    assert rows.step[-1] == observed.step_history.n_steps - 1
    bound = STATISTIC_FACTOR * evaluation_rounding(
        watched.grids.n_to,
        int(watched.transform.n_nodes),
        int(watched.shape[1]),
        plasma=True,
    )
    for name in PERIODIC_SUBSET:
        row, plane = np.asarray(rows.values[name][-1]), np.asarray(at_plane[name])
        if not np.any(plane):
            assert not np.any(row), name
            continue
        assert_within(row, plane, bound, what=name)
    # And the plasma trace is a trace, not one number repeated.
    assert float(np.ptp(rows.values["electron_density"][:, 0])) > 0.0


# ============================================ milestone M3, the CPU backend

#: Luna's own wall time for the full M2 case on an Apple-silicon laptop, from the
#: witness's provenance (Luna de9db7c9 on modal-fixed). What M3 has to beat.
LUNA_M2_SECONDS = 104.96


@pytest.mark.slow
def test_the_reduced_m2_case_is_the_same_run_on_either_backend(request, reduced):
    """Milestone M3, the equivalence half: the two backends step the same trajectory.

    The bound is derived and then measured, and the measurement is the interesting
    part. Per step the two differ by round-off: every complex product costs 1.42 eps,
    because numpy's contracts into a fused multiply-add on this host and numba's does
    not, and a transform costs log2(n) eps. Over 570 steps that would be a few times
    1e-14 if the case were benign. It is not: this one self-compresses through a
    plasma into a vacuum end where dp/dz diverges, which caps it at first order
    and amplifies any perturbation. **Measured 6.5e-12 at the
    exit**, about four hundred times the per-step figure, and the bound is set at 1e-10
    -- fifteen times the measurement, and still four orders inside the 2.6e-1 at which
    this case agrees with Luna at all.

    That the amplification is round-off and not a different integration is what the
    step counts say: both backends take the same number of steps and reject the same
    number, because they read the same error norm.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        theirs = witness_run(REDUCED, "REDUCED", backend="numpy_ref")
        ours = witness_run(REDUCED, "REDUCED", backend=TEST_BACKEND)
    a = ours.fields("fft")[-1]
    b = theirs.fields("fft")[-1]
    measured = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    assert measured < 1e-10
    # The last rung of the equivalence ladder of `docs/dev/accuracy.md` section 7; the
    # other six are in `tests/backends/test_numba_cpu.py` and merge into one record.
    record_name = record_backend(TEST_BACKEND)
    # Launching the nominal energy instead of Luna's own pulse moves this from
    # 3.478e-14 to 3.543e-14: a round-off-limited quantity re-measured on a trajectory
    # perturbed in its fifth digit, which is why the record compares it at rtol=1.0.
    record(
        BACKEND_EQUIVALENCE,
        request=request,
        backend=record_name,
        rows=[
            {
                "series": f"{record_name} against numpy_ref",
                "axis": "the reduced M2 case",
                "deviation": measured,
                "bound": 1e-10,
                "order": 6.0,
            }
        ],
    )
    # And the same case against Luna, at the reference's own bound, so that agreeing
    # with the reference cannot stand in for agreeing with the physics.
    grid = witness_grid(REDUCED)
    luna = stored_field("luna_rdw_reduced.h5")[-1]
    mine = reduced.fields("fft")[-1, :, 0, :]
    # The reference's own bound for this case, which it meets at 6.330e-2;
    # numba_cpu meets it at 6.330e-2 too, to four figures.
    assert 0.05 < in_band_distance(mine[0], luna[0], grid.inband_mask) < 0.08


# ------------------------------------------------- the full case, on both codes
#
# Luna's side of the three ladders below is already stored: a separate campaign ran
# the full M2 case on Luna at four tolerances, at three quadrature orders and with the
# plasma switched off, and kept the per-mode energy at every one of 201 planes
# (`tests/data/luna_rdw_switches.json`, provenance beside it). Nothing here runs Luna;
# the rows are read out of that file and carried into the record with Luna's own
# commit, which is what lets one figure hold both codes.

SWITCHES = json.loads(
    (Path(__file__).parents[1] / "data/luna_rdw_switches.json").read_text()
)
SWITCHES_PROVENANCE = json.loads(
    (Path(__file__).parents[1] / "data/luna_rdw_switches_provenance.json").read_text()
)
LUNA_SWITCHES = {
    "commit": SWITCHES_PROVENANCE["luna_commit"],
    "branch": SWITCHES_PROVENANCE["luna_branch"],
    "source": "luna_rdw_switches.json",
    "sha256": SWITCHES_PROVENANCE["files"]["luna_rdw_switches.json"]["sha256"],
}

# The second mode's exit energy is what these ladders are about: it is the number
# that reads 0.564 of Luna's at Luna's own tolerance and 0.9958 once Luna is
# converged, because Luna's weak modes are not converged at its default tolerance. A
# half of
# one per cent is the tolerance for it: solprop's own answer moves by 2.3e-4 across
# three decades of tolerance, and an adaptive run on another platform may place its
# steps slightly differently, which is worth the controller's 1e-6 at most.
MODE2_ENERGY = Column(
    name="mode2_energy",
    unit="J",
    why="half of one per cent: solprop's own answer moves by 2.3e-4 across three "
    "decades of tolerance, and a different step placement on another platform is "
    "worth the controller's 1e-6, so anything this wide is a real change",
    rtol=5e-3,
)

M2_TOLERANCE = Study(
    name="m2_tolerance",
    title="The M2 second mode's exit energy against the step tolerance, both codes",
    case={
        "case": "the full M2 capillary, 4 modes, 1.5 m",
        "solprop": "prop_capillary with Luna's media, atol 1e-10",
        "luna": "Luna.run called directly, because prop_capillary's interface cannot "
        "set a tolerance at all; rows from luna_rdw_switches.json",
        "metric": "the second mode's energy at the exit plane",
    },
    axis={"name": "rtol", "unit": "1", "label": "relative step tolerance"},
    luna=LUNA_SWITCHES,
    columns=(
        MODE2_ENERGY,
        Column(
            name="steps",
            unit="accepted steps",
            why="recorded, never compared: Luna's step count is not reproducible even "
            "across two machines, and solprop's is a property of the "
            "controller rather than of the answer",
            rtol=None,
        ),
    ),
)

M2_QUADRATURE = Study(
    name="m2_quadrature",
    title="The M2 second mode's exit energy against the radial order, both codes",
    case={
        "case": "the full M2 capillary, 4 modes, 1.5 m, at Luna's own tolerance",
        "solprop": "a fixed Gauss-Legendre rule of this order (ADR-010)",
        "luna": "its own `nr` keyword, rows from luna_rdw_switches.json",
        "metric": "the second mode's energy at the exit plane",
    },
    axis={"name": "n_r", "unit": "nodes", "label": "radial quadrature order"},
    luna=LUNA_SWITCHES,
    columns=(MODE2_ENERGY,),
)

M2_MODE2_ALONG_Z = Study(
    name="m2_mode2_along_z",
    title="The M2 second mode's energy along the fibre, both codes, plasma on and off",
    case={
        "case": "the full M2 capillary, 4 modes, 1.5 m, at Luna's own tolerance",
        "planes": "201, the witness's own save grid",
        "why": "with the plasma off the two codes agree at every plane; with "
        "it on the difference is a beat whose envelope departs from about 0.65 m, "
        "which is why no z-averaged bound is asserted on it",
    },
    axis={"name": "plane", "unit": "1", "label": "save plane"},
    luna=LUNA_SWITCHES,
    columns=(
        Column(
            name="z",
            unit="m",
            why="the plane's position, an input of the run rather than a measurement, "
            "so it is compared exactly",
            rtol=0.0,
        ),
        Column(
            name="energy",
            unit="J",
            why="half of one per cent, as the exit energy above; the Kerr-only "
            "comparison this figure is for agrees between the codes to one per cent",
            rtol=5e-3,
        ),
    ),
)


def luna_mode2(tag):
    """Return the archived vacuum-norm scalar: these rows contain no spectra."""
    return SWITCHES[f"{tag}_E_2"]


def legacy_mode2_energy(output):
    """Evaluate the same historical functional as Luna's scalar-only records.

    These three convergence curves predate stored spectra. They cannot be converted
    frequency by frequency. Keep their explicit vacuum convention on both sides;
    all amplitude witnesses and public run statistics use true-power units.
    """
    weights = get_power_weight(output)
    grid = witness_grid(FULL)
    return np.asarray(
        [
            energy(grid, field / np.sqrt(weight))[1, 0]
            for field, weight in zip(output.fields("fft"), weights, strict=True)
        ]
    )


@pytest.mark.benchmark
def test_the_full_case_tolerance_ladder_on_both_codes(request):
    """Three decades of tolerance on the full M2 case, against Luna's own ladder.

    This is the figure that carries the story of `docs/dev/accuracy.md` section 5:
    Luna's second mode falls from 3.341 to 1.893 uJ over ``rtol`` 1e-6 to 1e-9, a
    factor of 1.77, while solprop's is flat -- so the production witness's 0.564 is
    solprop converged against Luna unconverged.

    Marked `benchmark`, so no lane runs it: four runs of the full four-mode case, the
    finest of them several times the coarsest. Run it, and rewrite
    ``tests/data/convergence/m2_tolerance.json``, with

        SOLPROP_WRITE_RECORDS=1 uv run pytest \
            tests/driver/test_prop_capillary.py -m benchmark -p no:randomly \
            -k the_full_case_tolerance_ladder

    from a clean tree.
    """
    exponents = (6, 7, 8, 9)
    ours = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        for exponent in exponents:
            output = gated_witness_run(FULL, "FULL", rtol=10.0**-exponent)
            accepted = np.asarray(output["step_history/accepted"])
            ours.append(
                (
                    float(legacy_mode2_energy(output)[-1]),
                    float(np.count_nonzero(accepted)),
                )
            )
    # solprop's answer does not move across the three decades; Luna's falls by 1.77.
    energies = [energy for energy, _ in ours]
    assert abs(energies[-1] / energies[0] - 1.0) < 1e-3
    theirs = [
        (luna_mode2(f"RTOL{exponent}")[-1], SWITCHES[f"RTOL{exponent}_STEPS"][0])
        for exponent in exponents
    ]
    assert theirs[0][0] / theirs[-1][0] > 1.5
    # The bounded temporal absorber profile takes the full M2 case from 2551 to 1904
    # accepted steps against the capped logarithm (2550/3935/5973/9093 ->
    # 1905/2952/4703/7458 on this ladder).
    # The stored rows were measured with the nominal launch energy rather than the
    # true energy of Luna's pulse (1.0000228218 times the nominal one on the full
    # case), so they will move by parts in 1e4 when next regenerated.
    record(
        M2_TOLERANCE,
        request=request,
        backend="numpy_ref",
        rows=[
            {
                "series": series,
                "axis": f"1e-{exponent}",
                "mode2_energy": energy,
                "steps": steps,
            }
            for series, measured in (
                ("solprop, full case", ours),
                ("Luna, full case", theirs),
            )
            for exponent, (energy, steps) in zip(exponents, measured, strict=True)
        ],
    )


@pytest.mark.benchmark
def test_the_full_case_quadrature_ladder_on_both_codes(request):
    """The quadrature ladder, run on solprop and read off Luna.

    Three radial orders leave the second mode's exit energy unchanged to seven digits
    on both codes, which is what excludes the transverse rule as a cause of the
    codes' remaining difference. Marked `benchmark`: three full-case runs. Run it, and
    rewrite ``tests/data/convergence/m2_quadrature.json``, with

        SOLPROP_WRITE_RECORDS=1 uv run pytest \
            tests/driver/test_prop_capillary.py -m benchmark -p no:randomly \
            -k the_full_case_quadrature_ladder

    from a clean tree.
    """
    orders = (32, 64, 128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        ours = [
            float(legacy_mode2_energy(gated_witness_run(FULL, "FULL", n_r=n_r))[-1])
            for n_r in orders
        ]
    theirs = [luna_mode2(f"NR{n_r}")[-1] for n_r in orders]
    # Seven digits on both codes: the fixed rule is converged well below 32 nodes.
    for measured in (ours, theirs):
        assert abs(measured[-1] / measured[0] - 1.0) < 1e-6, measured
    # The stored rows were measured with the nominal launch energy rather than the
    # true energy of Luna's pulse (1.0000228218 times the nominal one on the full
    # case), so they will move by parts in 1e4 when next regenerated.
    record(
        M2_QUADRATURE,
        request=request,
        backend="numpy_ref",
        rows=[
            {"series": series, "axis": n_r, "mode2_energy": energy}
            for series, measured in (
                ("solprop, full case", ours),
                ("Luna, full case", theirs),
            )
            for n_r, energy in zip(orders, measured, strict=True)
        ],
    )


@pytest.mark.benchmark
def test_the_second_mode_along_z_with_the_plasma_on_and_off(request):
    """The Kerr-only agreement, at every plane, on both codes.

    With the plasma off the two codes agree on all four modes to one per cent at every
    one of 201 planes; with it on the second mode's envelope departs from about 0.65 m
    and the difference is a beat, which is why the page carries a figure rather than a
    bound. Marked `benchmark`: two full-case runs. Run it, and rewrite
    ``tests/data/convergence/m2_mode2_along_z.json``, with

        SOLPROP_WRITE_RECORDS=1 uv run pytest \
            tests/driver/test_prop_capillary.py -m benchmark -p no:randomly \
            -k the_second_mode_along_z

    from a clean tree.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        for label, keywords in (("on", {}), ("off", {"plasma": False})):
            output = gated_witness_run(FULL, "FULL", **keywords)
            energy = legacy_mode2_energy(output)
            position = np.asarray(output["z"])
            rows += [
                {
                    "series": f"solprop, plasma {label}",
                    "axis": index,
                    "z": float(z),
                    "energy": float(value),
                }
                for index, (z, value) in enumerate(zip(position, energy, strict=True))
            ]
    for label, tag in (("on", "REFERENCE"), ("off", "NOPLASMA")):
        rows += [
            {
                "series": f"Luna, plasma {label}",
                "axis": index,
                "z": float(z),
                "energy": float(value),
            }
            for index, (z, value) in enumerate(
                zip(SWITCHES[f"{tag}_Z"], luna_mode2(tag), strict=True)
            )
        ]
    ours = np.array(
        [row["energy"] for row in rows if row["series"] == "solprop, plasma off"]
    )
    theirs = np.array(
        [row["energy"] for row in rows if row["series"] == "Luna, plasma off"]
    )
    # With the plasma off the codes agree at every plane. The first plane is
    # the input, where both are the same pulse, so the ratio starts at one.
    assert float(np.max(np.abs(ours[1:] / theirs[1:] - 1.0))) < 0.02
    # The stored rows were measured with the nominal launch energy rather than the
    # true energy of Luna's pulse (1.0000228218 times the nominal one on the full
    # case), so they will move by parts in 1e4 when next regenerated.
    record(
        M2_MODE2_ALONG_Z,
        request=request,
        backend="numpy_ref",
        rows=rows,
    )


@pytest.mark.benchmark
def test_the_historical_full_m2_case_against_luna_wall_time():
    """Retain the historical gated comparison and its original wall-time limit.

    SEP 27 confines this equation to the private NumPy hook. The measurements below
    predate that change; production performance was qualified on SEP 27's ten-step
    windows, separately from this historical witness.

    Measured on an Apple-silicon laptop: **12.2 s on a quiet machine and 14.0 s under
    a load average of eight**, against numpy_ref's 63.7 s in the same session and Luna's
    recorded 104.96 s -- a factor of 4.6 over the reference and 8.6 over Luna, at
    *identical* achieved error: both backends return the same in-band distance and the
    same in-band content to five figures, which is what "at equal achieved error" has to
    mean here.

    The bound below is deliberately loose. What M3 asks is that the case is faster
    than Luna's, and a factor of eight has room for a machine under load; asserting
    the measured number would make this a test of how busy the machine is.
    """
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        output = gated_witness_run(FULL, "FULL", planes=2)
    wall = time.perf_counter() - started
    grid = witness_grid(FULL)
    theirs = stored_field("luna_rdw_full.h5")[-1]
    ours = output.fields("fft")[-1, :, 0, :]
    assert wall < LUNA_M2_SECONDS
    # The same bounds the reference's own full-case test asserts: faster is only
    # milestone M3 if it is the same answer.
    assert 0.15 < in_band_distance(ours[0], theirs[0], grid.inband_mask) < 0.4
    norms = np.sum(np.abs(ours[0]) ** 2) / np.sum(np.abs(theirs[0]) ** 2)
    assert 0.98 < norms < 1.01


@pytest.mark.benchmark
def test_the_full_m2_case_against_its_witness(request):
    """Milestone M2 at full size, against the trimmed Luna witness beside it.

    Measured on one workstation at Luna's tolerance: 1761 accepted steps, 92 s; the
    fundamental agrees with Luna to 4.7e-3 in its total in-band content and 2.6e-1 in
    the shape of its spectral energy density, and the three higher modes to 0.564,
    1.080 and 1.165 of Luna's energy. Solprop's own answer is converged: two decades of
    tolerance move the VUV band's energy by 7.5e-5 and its arrival time by 1.2e-5.

    With a frame per mode this was 1824 steps, 88 s, 1.2e-3 in content, and higher
    modes at 1.7e-3, 1.4e-5 and 1.7e-5 of Luna's; giving the mode set one co-moving
    frame instead of one each is what closed the higher modes.

    **The second mode's 0.564 is a bound against an unconverged reference.** The
    witness is Luna at ``rtol = 1e-6``, which its interface cannot change, and here that
    is 1.77 times its own converged answer for this mode: 3.3408 uJ at 1e-6 against
    1.8929 at 1e-9, over 4365 steps against 117486, where solprop moves by 2.3e-4 across
    the same three decades. Against a converged Luna the ratio is 0.9956. Neither the
    witness nor this bound is changed for it -- the comparison is deliberately against
    the Luna every published result carries -- but the number is not a physics
    discrepancy and should not be read as one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        output = gated_witness_run(FULL, "FULL", planes=2)
    grid = witness_grid(FULL)
    theirs = stored_field("luna_rdw_full.h5")[-1]
    ours = output.fields("fft")[-1, :, 0, :]
    # Against production Luna, from the capped-logarithm absorber in the vacuum-
    # amplitude form to the bounded profile in the true-power form: shape .2594200275 ->
    # .2586611274 (-.2925372%); content .9953527961 -> .9946881720 (-.0667727%); VUV
    # 1.0895199672 -> 1.0894263605 (-.0085916%); modes 2/3/4 .5640228623/1.0796669968/
    # 1.1648044943 -> .5638943939/1.0797012283/1.1645067591 (-.0227772%, +.0031706%,
    # -.0255610%); all inside the two-sided bounds below. The full-case rows of
    # m2_witness_metrics were measured with the nominal launch energy rather than the
    # true energy of Luna's pulse (1.0000228218 times the nominal one here) and were
    # not regenerated with the reduced ones, this test being the `benchmark`-marked
    # writer that produces them.
    fundamental = in_band_distance(ours[0], theirs[0], grid.inband_mask)
    assert 0.15 < fundamental < 0.4
    norms = np.sum(np.abs(ours[0]) ** 2) / np.sum(np.abs(theirs[0]) ** 2)
    assert 0.98 < norms < 1.01
    vuv = band_mask(grid, VUV)
    band_ratio = np.sum(np.abs(ours[:, vuv]) ** 2) / np.sum(np.abs(theirs[:, vuv]) ** 2)
    assert 0.95 < band_ratio < 1.25
    higher = [
        np.sum(np.abs(ours[m]) ** 2) / np.sum(np.abs(theirs[m]) ** 2) for m in (1, 2, 3)
    ]
    assert 0.45 < higher[0] < 0.7
    assert 0.95 < higher[1] < 1.25
    assert 1.0 < higher[2] < 1.35

    # ------------------------------------------------ against *converged* Luna
    #
    # The same run against `luna_rdw_full_converged.h5`, which is Luna's own answer
    # to this case at `rtol = 1e-9` -- 117 486 accepted steps against the 4365 the
    # tolerance its interface cannot change gives it. The bounds above stay
    # exactly as they are: they are the regression against what a user of Luna
    # actually gets, and that is what a regression should be. These are the physics
    # numbers.
    #
    # Measured: the fundamental's in-band distance falls from 0.2594 to **0.0908**,
    # and the second mode goes from 0.564 of Luna's energy to **0.9958** -- agreement
    # to 0.42 %. Luna's second mode is 1.77 times larger unconverged than converged,
    # so almost all of the 0.564 was Luna's step control and none of it was solprop.
    # In the same two forms as above: converged-reference shape .09074989575 ->
    # .08923627499 (-1.6679036%); modes 2/3/4 .9954845118/1.0506817651/.8457296069
    # -> .9952546625/1.0507146353/.8455127507 (-.0230892%, +.0031285%, -.0256413%),
    # all inside the bounds.
    converged = stored_field("luna_rdw_full_converged.h5")[-1]
    assert 0.06 < in_band_distance(ours[0], converged[0], grid.inband_mask) < 0.13
    second = np.sum(np.abs(ours[1]) ** 2) / np.sum(np.abs(converged[1]) ** 2)
    assert 0.97 < second < 1.03
    # Modes three and four move much less between Luna's two tolerances and are
    # recorded rather than tightened: 1.051 and 0.846 against the converged witness,
    # where they read 1.080 and 1.165 against the unconverged one.
    third = float(np.sum(np.abs(ours[2]) ** 2) / np.sum(np.abs(converged[2]) ** 2))
    fourth = float(np.sum(np.abs(ours[3]) ** 2) / np.sum(np.abs(converged[3]) ** 2))
    assert 0.95 < third < 1.2
    assert 0.7 < fourth < 1.0
    shape = in_band_distance(ours[0], converged[0], grid.inband_mask)
    record(
        M2_WITNESS_METRICS,
        request=request,
        backend="numpy_ref",
        rows=metric_rows(
            "full case, against production Luna",
            [
                ("fundamental, shape", fundamental, 0.15, 0.4),
                ("fundamental, content", float(norms), 0.98, 1.01),
                ("VUV band over four modes", float(band_ratio), 0.95, 1.25),
                ("second mode, content", float(higher[0]), 0.45, 0.7),
                ("third mode, content", float(higher[1]), 0.95, 1.25),
                ("fourth mode, content", float(higher[2]), 1.0, 1.35),
            ],
        )
        + metric_rows(
            "full case, against converged Luna",
            [
                ("fundamental, shape", shape, 0.06, 0.13),
                ("second mode, content", float(second), 0.97, 1.03),
                ("third mode, content", third, 0.95, 1.2),
                ("fourth mode, content", fourth, 0.7, 1.0),
            ],
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", ["rdw", "ctc"])
def test_witness_poynting_cross_terms_and_negative_excursions(
    case, request, record_property
):
    output = request.getfixturevalue("reduced" if case == "rdw" else "ctc_reduced")
    grid = witness_grid(REDUCED) if case == "rdw" else ctc_grid(CTC_REDUCED)
    weights = get_power_weight(output)
    fields = output.fields("fft")
    minimum, discrepancy, maximum_bound = np.inf, 0.0, 0.0
    for b, weight in zip(fields, weights, strict=True):
        a = b / np.sqrt(weight)
        flux = np.real(
            analytic_field(grid, a) * np.conj(analytic_field(grid, a * weight))
        )
        geometric = np.abs(analytic_field(grid, b)) ** 2
        scale = float(np.max(np.abs(flux)))
        minimum = min(minimum, float(np.min(flux)) / scale)
        difference = float(np.max(np.abs(flux - geometric))) / scale
        discrepancy = max(discrepancy, difference)
        coefficients = a.copy()
        if grid.kind == "field":
            coefficients[..., 1:-1] *= 2
        amplitude_sum = np.sum(np.abs(coefficients), axis=-1) / grid.n_t
        root = np.sqrt(weight)
        support = np.abs(coefficients) > 0
        largest = np.max(np.where(support, root, -np.inf), axis=-1)
        smallest = np.min(np.where(support, root, np.inf), axis=-1)
        span = np.where(amplitude_sum > 0, largest - smallest, 0.0)
        bound = float(np.max(0.5 * span**2 * amplitude_sum**2)) / scale
        maximum_bound = max(maximum_bound, bound)
        # The kernel difference is (sqrt(w)-sqrt(w'))**2/2. The triangle
        # inequality gives this absolute trace bound, plus FFT rounding.
        assert difference <= bound + 64 * np.finfo(float).eps * np.log2(grid.n_t)
    record_property("poynting_minimum_relative", minimum)
    record_property("poynting_geometric_difference", discrepancy)
    record_property("poynting_geometric_analytic_bound", maximum_bound)
    # The predeclared gas estimate 1e-8 failed: CtC measured 7.9506872e-8.
    # The exact quadratic-kernel bound above replaces that estimate.
    # Negative excursions are reported, never clipped or assumed absent.


# -------------------------------------------------- a fill that knows gas and bore

# A taper and a gradient to vacuum over a centimetre of helium: laminar (peak Re about
# 900) and a continuum up to its choked exit (Kn about 0.005), so it warns about
# nothing.
FLOW_LENGTH = 1e-2
FLOW_TAPER = Linear(125e-6, 100e-6, FLOW_LENGTH)
FLOW_STEPS = 10


def _flow_build(
    *,
    radius: float | ZProfile = RADIUS,
    length: float = FLOW_LENGTH,
    gas: str | Mixture = "He",
    pressure: Pressure | None = 1.0,
    **overrides,
):
    """Build the cheap capillary on any radius, fill or material, without running."""
    return Simulation.from_capillary(
        radius, length, gas, pressure, **{"noise": None, **keywords(**overrides)}
    )


def test_a_gradient_to_vacuum_chokes_by_default(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.gasflow"):
        simulation = _flow_build(length=FLOW_LENGTH, pressure=(1.0, 0.0))
    fill = simulation.profiles["pressure"]
    assert isinstance(fill, PressureGradient)
    # The flow takes the fill's own (optmat's) transport properties.
    assert fill.pressures[-1] == choked_pressure(
        species("He"), RADIUS, FLOW_LENGTH, 1.0
    )
    assert any("choked pressure" in record.message for record in caplog.records)


def test_a_tapered_gradient_follows_the_bore():
    simulation = _flow_build(radius=FLOW_TAPER, length=FLOW_LENGTH, pressure=(1.0, 0.0))
    fill = simulation.profiles["pressure"]
    expected = steady_fill(
        species("He"), [0.0, FLOW_LENGTH], [1.0, 0.0], radius=FLOW_TAPER
    )
    z = np.linspace(0.0, FLOW_LENGTH, 33)
    assert isinstance(fill, PressureGradient) and fill.radius is FLOW_TAPER
    np.testing.assert_array_equal(fill(z), expected(z))


@functools.cache
def _flow_run(backend):
    simulation = _flow_build(
        radius=FLOW_TAPER,
        length=FLOW_LENGTH,
        pressure=(1.0, 0.0),
        plasma=False,
        n_save=3,
        schedule=FixedSchedule(np.full(FLOW_STEPS, FLOW_LENGTH / FLOW_STEPS)),
        backend=backend,
    )
    # A collar referred to a metre, weak enough for ten explicit millimetre steps
    # (ADR-025); the default refers it to the centimetre run and needs forty.
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=1.0)
    )
    return simulation.run()


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_tapered_gradient_to_vacuum_runs_ten_steps_on_every_backend(backend):
    if backend == "native":
        pytest.importorskip("solprop_native")
    output = _flow_run(backend)
    fill = steady_fill(species("He"), [0.0, FLOW_LENGTH], [1.0, 0.0], radius=FLOW_TAPER)
    z = np.asarray(output["z"])
    # The recorded statistic is the fill itself, evaluated at the saved planes.
    np.testing.assert_allclose(output["stats/pressure"], fill(z), rtol=1e-13, atol=0)
    assert float(output["stats/pressure"][-1]) == pytest.approx(fill.pressures[-1])
    reference = np.asarray(_flow_run("numpy_ref")["field"])
    field = np.asarray(output["field"])
    scale = float(np.linalg.norm(reference[-1]))
    # The same ten fixed steps on every backend: what differs is the order of the
    # floating-point sums, a few ulps per step compounded over ten.
    assert np.linalg.norm(field[-1] - reference[-1]) / scale < 1e-10


def test_a_boreless_gradient_in_a_taper_warns():
    boreless = PressureGradient.two_point(FLOW_LENGTH, 1.0, 0.1)
    with pytest.warns(GasFlowWarning, match="built without the bore"):
        _flow_build(radius=FLOW_TAPER, length=FLOW_LENGTH, pressure=boreless)
    bored = PressureGradient.two_point(FLOW_LENGTH, 1.0, 0.1, radius=FLOW_TAPER)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        _flow_build(radius=FLOW_TAPER, length=FLOW_LENGTH, pressure=bored)
        _flow_build(length=FLOW_LENGTH, pressure=boreless)


def test_vacuum_end_is_refused_where_nothing_can_honour_it():
    independent = Mixture.partial_pressures({"N2": 0.8, "O2": 0.2})
    with pytest.raises(DriverError, match="vacuum_end"):
        _flow_build(gas=independent, pressure=None, vacuum_end="choked")
    with pytest.raises(MaterialError, match="no vacuum end"):
        _flow_build(pressure=1.0, vacuum_end="floor")
    with pytest.raises(MaterialError, match="carries its own ends"):
        _flow_build(
            pressure=PressureGradient.two_point(SHORT, 1.0, 0.5), vacuum_end="floor"
        )


def test_prop_waveguide_takes_the_bore_of_circular_modes():
    mode = MarcatiliMode(FLOW_TAPER, "He", 1.0)
    simulation = Simulation.from_waveguide(
        [mode], FLOW_LENGTH, "He", (1.0, 0.0), **{"noise": None, **keywords()}
    )
    fill = simulation.profiles["pressure"]
    assert isinstance(fill, PressureGradient) and fill.radius is FLOW_TAPER
    floored = Simulation.from_waveguide(
        [mode],
        FLOW_LENGTH,
        "He",
        (1.0, 0.0),
        vacuum_end="floor",
        **{"noise": None, **keywords()},
    )
    fill = floored.profiles["pressure"]
    assert isinstance(fill, PressureGradient)
    assert fill.pressures[-1] == pytest.approx(1e-4)


def test_a_rectangular_guide_keeps_the_uniform_law_and_says_so():
    widths = Linear(100e-6, 80e-6, FLOW_LENGTH)
    mode = RectMode(widths, widths, "He", 1.0)
    with pytest.warns(GasFlowWarning, match="rectangular"):
        simulation = Simulation.from_waveguide(
            [mode],
            FLOW_LENGTH,
            "He",
            (1.0, 0.1),
            **{"noise": None, **keywords(n_r=12, n_theta=12)},
        )
    fill = simulation.profiles["pressure"]
    assert isinstance(fill, PressureGradient) and fill.radius is None
    assert fill.pressures[-1] == 0.1


def test_a_rectangular_guide_under_a_constant_fill_does_not_warn():
    widths = Linear(100e-6, 80e-6, FLOW_LENGTH)
    mode = RectMode(widths, widths, "He", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        Simulation.from_waveguide(
            [mode],
            FLOW_LENGTH,
            "He",
            1.0,
            **{"noise": None, **keywords(n_r=12, n_theta=12)},
        )


def test_an_empty_waveguide_mode_set_is_refused_before_the_fill_needs_it():
    with pytest.raises(DriverError, match="at least one mode"):
        Simulation.from_waveguide(
            [], FLOW_LENGTH, "He", (1.0, 0.0), **{"noise": None, **keywords()}
        )


@pytest.mark.medium
def test_with_mixture_warns_for_a_boreless_gradient_in_a_taper():
    simulation = _flow_build(radius=FLOW_TAPER, pressure=(1.0, 0.0), plasma=False)
    boreless = Mixture.fill(
        {"He": 1.0}, PressureGradient.two_point(FLOW_LENGTH, 1.0, 0.1)
    )
    with pytest.warns(GasFlowWarning, match="built without the bore"):
        simulation.with_mixture(boreless)
    bored = Mixture.fill({"He": 1.0}, (1.0, 0.1), length=FLOW_LENGTH, radius=FLOW_TAPER)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        simulation.with_mixture(bored)

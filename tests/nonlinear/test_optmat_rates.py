# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's rate models on solprop's rate protocol (SEP 3, Phase 1d).

Every check compares the adapter with direct calls of the same optmat model, so that a
later optmat release that changes a rate changes both sides alike: no optmat value is
written here. The one exception is the parity of optmat's hydrogenic form with the
test-only ADK oracle (``tests/rate_oracles.py``), which compares like with like.
"""

import dataclasses
import math
import warnings
from functools import partial
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from optmat.constants import E_CHARGE
from optmat.exceptions import ModelRangeWarning
from optmat.gas.ionisation_rates import IonisationRateModel, ionisation_rate_model
from optmat.gas.molecular_ionisation_rates import molecular_rate_model
from scipy import integrate

from bounds import gamma
from rate_oracles import AdkOracle, adk_oracle
from solprop.exceptions import (
    GridError,
    MaterialError,
    MaterialRangeWarning,
    MaterialStatusWarning,
    TranslationError,
)
from solprop.nonlinear.envelope_plasma import (
    EnvelopePlasmaResponse,
    _cycle_sources,
    linear_cycle_average,
    linear_cycle_averages,
)
from solprop.nonlinear.ionisation import (
    MAX_RATE,
    bind_rate_energy,
    ionisation_fraction,
    rate_energy,
    rate_species,
)
from solprop.nonlinear.optmat_rates import (
    NEGLIGIBLE_RATE,
    OptmatRate,
    bind_reference_field,
    peak_field_report,
    selected_rate,
)
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas, _optmat, ionisation_potential, species
from solprop.physdata.selection import MaterialSelection
from solprop.translate.ionisation import DEFAULT_RTOL, build_rate_table, rate_table_key

WAVELENGTH = 800e-9


def quiet(method, *arguments):
    """Call an optmat method with its range warnings silenced, for a reference."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ModelRangeWarning)
        return np.asarray(method(*arguments))


def window(rate, count=97):
    """Fields from just above the threshold to just below the maximum field."""
    return np.geomspace(rate.threshold * 1.01, rate.maximum_field * 0.99, count)


# --------------------------------------------------------------- the rate itself


@pytest.mark.parametrize("gas", ["He", "Ar", "Xe", "N2", "O2", "CH4"])
def test_the_rate_is_optmats_instantaneous_rate_above_the_threshold(gas):
    rate = OptmatRate(gas, WAVELENGTH)
    model = rate.optmat_model
    fields = window(rate)
    # The same call on the same array: the adapter adds nothing below the ceiling, so
    # this is structural and exact.
    expected = np.minimum(model.instantaneous_rate(fields), MAX_RATE)
    assert_array_equal(rate(fields), expected)
    assert_array_equal(rate(-fields), expected)
    below = rate.threshold * np.array([0.0, 0.5, 1.0 - 1e-9])
    assert_array_equal(rate(below), 0.0)
    assert rate(3e10).shape == ()
    assert rate(np.ones((2, 3)) * 3e10).shape == (2, 3)


def test_the_threshold_is_where_the_rate_reaches_the_negligible_rate():
    rate = OptmatRate("Ar", WAVELENGTH)
    model = rate.optmat_model
    assert float(model.instantaneous_rate(rate.threshold)) >= NEGLIGIBLE_RATE
    # The bisection stops at a relative width of 1e-12, so a field 1e-10 below the
    # threshold is on the negligible side.
    assert float(model.instantaneous_rate(rate.threshold * (1 - 1e-10))) < (
        NEGLIGIBLE_RATE
    )
    looser = OptmatRate("Ar", WAVELENGTH, negligible_rate=1e-10)
    assert looser.threshold > rate.threshold
    assert float(model.instantaneous_rate(looser.threshold)) >= 1e-10
    assert dict(looser.parameters)["negligible_rate"] == 1e-10


def test_the_rate_is_held_above_the_maximum_field_and_the_run_records_it():
    rate = OptmatRate("Ar", WAVELENGTH)
    held = float(rate.optmat_model.instantaneous_rate(rate.maximum_field))
    with _optmat.material_context() as log:
        above = rate(rate.maximum_field * np.array([1.5, 3.0, 30.0]))
        rate(rate.maximum_field * 0.5)
    assert_array_equal(above, held)
    entry: Any
    (entry,) = log.range_entries()
    assert entry["digest"] == rate.material_records[0].digest
    assert any("maximum field" in part for part in entry["outside"])
    # Outside any context nothing is recorded, and nothing is warned either way.
    rate(rate.maximum_field * 2.0)


def test_a_table_above_the_maximum_field_has_nothing_to_tabulate():
    rate = OptmatRate(Gas.Ar, WAVELENGTH)
    assert rate.species_identity == "Ar"
    span = (2.0 * rate.maximum_field, 3.0 * rate.maximum_field)
    with pytest.raises(TranslationError, match="held above its maximum field"):
        build_rate_table(rate, span, use_cache=False)


def test_the_ceiling_clips_and_a_field_must_be_finite():
    rate = OptmatRate("Ar", WAVELENGTH, maximum=1e10)
    assert float(rate(rate.maximum_field)) == 1e10
    with pytest.raises(MaterialError, match="finite"):
        rate(np.array([1e10, np.nan]))


def test_a_rate_table_reproduces_the_direct_rate_and_holds_above_the_maximum():
    rate = OptmatRate("Ar", WAVELENGTH)
    table = build_rate_table(rate, (1e8, 30 * rate.maximum_field), use_cache=False)
    # The table stops at the maximum field and holds its value above it.
    assert table.field_max == rate.maximum_field
    model = rate.optmat_model
    probes = np.geomspace(table.field_min * 1.001, table.field_max * 0.999, 1001)
    direct = model.instantaneous_rate(probes)
    # The build's bound is on the difference of logarithms, the relative error to
    # first order: twice the tolerance covers the second order.
    assert np.max(np.abs(table(probes) / direct - 1.0)) <= 2 * DEFAULT_RTOL
    beyond = table(rate.maximum_field * np.array([2.0, 10.0]))
    assert_allclose(beyond, rate(rate.maximum_field), rtol=2 * DEFAULT_RTOL)
    assert table.convention == "instantaneous-isotropic"
    assert table.species_identity == "Ar"
    assert table.ionisation_potential == rate.ionisation_potential


# ------------------------------------------------------------ energy per event


@pytest.mark.parametrize("gas", ["Ne", "Ar", "Kr", "Xe"])
def test_the_energy_per_event_is_the_energy_loss_over_the_rate(gas):
    rate = OptmatRate(gas, WAVELENGTH)
    model = rate.optmat_model
    fields = window(rate, 33)
    identity = model.energy_loss_rate(fields) / model.instantaneous_rate(fields)
    # Two evaluations of one channel sum, in logarithms and in rates: rounding.
    assert_allclose(rate.energy_per_event(fields), identity, rtol=1e-12)


@pytest.mark.parametrize("gas", ["He", "H2", "N2", "O2", "CH4", "N2O"])
def test_a_one_channel_model_is_exact(gas):
    rate = OptmatRate(gas, WAVELENGTH).bound(3e10, (1e7, 1e12))
    model = rate.optmat_model
    assert rate.energy_bound == 0.0
    fields = window(rate, 9)
    assert_array_equal(rate.energy_per_event(fields), rate.ionisation_potential)
    identity = model.energy_loss_rate(fields) / model.instantaneous_rate(fields)
    assert_allclose(identity, rate.ionisation_potential, rtol=1e-12)


def test_the_constant_energy_is_optmats_at_the_reference_field():
    unbound = OptmatRate("Ar", WAVELENGTH)
    model = unbound.optmat_model
    # Without a reference, the weak-field limit: the lowest channel's energy.
    lowest = min(channel.ionization_energy for channel in model.channels)
    assert unbound.ionisation_potential == float(model.energy_per_event(0.0))
    assert unbound.ionisation_potential == lowest
    assert unbound.effective_reference_field is None
    bound = unbound.bound(3e10, (3e7, 9e11))
    assert bound.effective_reference_field == 3e10
    assert bound.ionisation_potential == float(model.energy_per_event(3e10))
    assert bound.plasma_energy == bound.ionisation_potential
    assert rate_species(bound) == ("Ar", bound.ionisation_potential)
    assert rate_energy(bound) == ("Ar", bound.ionisation_potential)
    # Clipped to the tabulated fields above the threshold, which end at the
    # maximum field, where the rate is held.
    high = unbound.bound(10 * unbound.maximum_field, (3e7, 9e11))
    assert high.effective_reference_field == unbound.maximum_field
    low = unbound.bound(1e3, (1e9, 9e11))
    assert low.effective_reference_field == 1e9
    assert low.energy_span == (1e9, unbound.maximum_field)
    below = unbound.bound(1e3, (1.0, 2.0))
    assert below.energy_span == (unbound.threshold, unbound.threshold)
    assert below.energy_bound == 0.0


@pytest.mark.parametrize("gas", ["Ar", "Xe"])
def test_the_energy_bound_holds_against_direct_evaluation(gas):
    rate = OptmatRate(gas, WAVELENGTH).bound(2.5e10, (2.5e7, 7.5e11))
    low, high = rate.energy_span
    dense = np.geomspace(low, high, 20001)
    direct = rate.optmat_model.energy_per_event(dense)
    departure = np.max(np.abs(direct / rate.ionisation_potential - 1.0))
    # Taken on 257 samples including both ends; rounding only beyond that.
    assert departure <= rate.energy_bound * (1 + 1e-12)
    assert rate.energy_bound > 0.0
    ends = rate.optmat_model.energy_per_event(np.array([low, high]))
    assert rate.energy_bound == pytest.approx(
        float(np.max(np.abs(ends / rate.ionisation_potential - 1.0))), rel=1e-12
    )


# ------------------------------------------------------------- the responses


def test_a_response_pays_the_rates_energy_and_the_envelope_agrees():
    rate = OptmatRate("Ar", WAVELENGTH).bound(3e10, (3e7, 9e11))
    field = PlasmaResponse("Ar", rate)
    envelope = EnvelopePlasmaResponse(species("Ar"), rate, species="Ar")
    for response in (field, envelope):
        assert response.potential == rate.ionisation_potential
        assert cast("Any", response.gas).name == "Ar"
    # The rate replaced by its table keeps the bound energy, as translation does.
    table = build_rate_table(rate, (3e7, 9e11), use_cache=False)
    assert dataclasses.replace(envelope, rate=table).potential == (
        rate.ionisation_potential
    )
    assert dataclasses.replace(field, rate=table).potential == (
        rate.ionisation_potential
    )
    with pytest.raises(MaterialError, match="must name the response's species"):
        PlasmaResponse("N2", rate)


def test_a_rate_without_an_energy_leaves_the_species_as_given():
    def anonymous(field_amplitude):
        return np.zeros_like(np.asarray(field_amplitude, dtype=float))

    assert rate_energy(anonymous) is None
    assert bind_rate_energy("Ar", anonymous) == "Ar"
    custom = dataclasses.replace(species("Ar"), ionisation=None)
    rate = OptmatRate("Ar", WAVELENGTH)
    bound: Any = bind_rate_energy(custom, rate)
    assert bound.ionisation is not None
    assert bound.ionisation.potential == rate.ionisation_potential


@pytest.mark.parametrize(
    ("gas", "fields"),
    [("Ar", (2e10, 4.3e10)), ("He", (5e10, 1.05e11)), ("N2", (2e10, 4.2e10))],
)
def test_the_envelope_average_is_optmats_cycle_averaged_rate(gas, fields):
    # optmat's "consistent" averaging makes the cycle average of the instantaneous
    # rate its cycle-averaged rate to 3 % at any Keldysh parameter, as it states; the
    # envelope's adaptive quadrature of the same average is converged to 2e-10, so
    # the stated 3 % is the whole tolerance, up to the barrier-suppression field.
    rate = OptmatRate(gas, WAVELENGTH)
    response = EnvelopePlasmaResponse(gas, rate)
    amplitudes = np.geomspace(fields[0], fields[1], 9)
    envelope = amplitudes.astype(complex)[None, :]
    averaged = response.sources(envelope, 1.0, dt=1e-16).ionisation_rate
    model = rate.optmat_model
    expected = model.cycle_averaged_rate(amplitudes)
    assert np.max(np.abs(averaged / expected - 1.0)) <= 0.03
    # The fixed-order average and the sources' adaptive one are each within 2e-10.
    assert_allclose(
        [linear_cycle_average(rate, float(a)) for a in amplitudes], averaged, rtol=4e-10
    )
    assert linear_cycle_average(rate, 0.0) == 0.0
    for bad in (-1.0, math.nan):
        with pytest.raises(GridError, match="cycle amplitude"):
            linear_cycle_average(rate, bad)


def test_the_cycle_rule_takes_a_bare_rate_and_refuses_a_malformed_one():
    # A plain function declares no bounds, so the quarter cycle is one piece: the
    # average of (a cos theta)^2 is a^2/2, exact for a polynomial of this degree.
    def square(field_amplitude: Any) -> np.ndarray:
        return np.asarray(field_amplitude) ** 2

    amplitudes = np.array([0.0, 1e9, 3e10])
    assert_allclose(
        linear_cycle_averages(square, amplitudes), amplitudes**2 / 2, rtol=1e-14
    )
    with pytest.raises(GridError, match="real array of the field's shape"):
        linear_cycle_averages(cast(Any, lambda field_amplitude: 1.0), amplitudes)
    with pytest.raises(GridError, match="finite and non-negative"):
        linear_cycle_averages(
            lambda field_amplitude: -square(field_amplitude), amplitudes
        )


@pytest.mark.parametrize(
    ("gas", "factor"),
    [("He", 1.01), ("He", 2.0), ("Ar", 1.0001), ("Ar", 1.01), ("Ar", 2.0)],
)
def test_the_envelope_average_resolves_the_hold_at_the_maximum_field(gas, factor):
    # The held rate has a corner at the maximum field, which the adaptive rule is
    # now seeded with; without it the integral failed just above the maximum field.
    # (Helium within a few parts in 1e4 of it still fails on optmat's own rounding
    # noise, about 3e-9 of the rate, against the reference's 2e-10; a run integrates
    # its smooth table instead.)
    rate = OptmatRate(gas, WAVELENGTH)
    amplitude = factor * rate.maximum_field
    response = EnvelopePlasmaResponse(gas, rate)
    source = response.sources(np.array([[amplitude + 0j]]), 1.0, dt=1e-16)
    assert source.ionisation_rate[0] == pytest.approx(
        linear_cycle_average(rate, amplitude), rel=4e-10
    )
    assert 0.0 < source.ionisation_rate[0] <= float(rate(rate.maximum_field))
    table = build_rate_table(rate, (1e8, 30 * rate.maximum_field), use_cache=False)
    for near in (0.999, 1.0001):
        assert linear_cycle_average(table, near * rate.maximum_field) > 0.0


class _CountingRate:
    """A rate that counts its evaluations and declares its wrapped rate's bounds."""

    def __init__(self, rate: Any) -> None:
        self.rate = rate
        self.calls = 0
        for name in ("threshold", "maximum_field"):
            if hasattr(rate, name):
                setattr(self, name, getattr(rate, name))

    def __call__(self, field_amplitude):
        self.calls += 1
        return self.rate(field_amplitude)


@pytest.mark.parametrize("gas", ["Ar", "He", "Xe", "N2"])
def test_the_cycle_average_evaluates_an_optmat_rate_once(gas):
    # Q46: one evaluation of the rate for every amplitude, from the threshold to past
    # the hold at the maximum field, each within the adaptive quadrature's 2e-10.
    rate = OptmatRate(gas, WAVELENGTH)
    top = rate.maximum_field
    amplitudes = np.r_[
        0.0,
        0.5 * rate.threshold,
        rate.threshold * np.array([1.01, 1.5]),
        np.geomspace(0.1 * top, 3.0 * top, 6),
        top * np.array([0.999, 1.0, 1.0001, 1.01]),
    ]
    counting = _CountingRate(rate)
    averaged = linear_cycle_averages(counting, amplitudes)
    assert counting.calls == 1
    assert averaged.shape == amplitudes.shape
    assert averaged[0] == 0.0 and averaged[1] == 0.0
    adaptive = np.array(
        [_cycle_sources(np.array([complex(a)]), rate)[0] for a in amplitudes[2:]]
    )
    assert_allclose(averaged[2:], adaptive, rtol=4e-10, atol=1e-30)
    # An amplitude's average does not depend on the batch it is taken in: one
    # 32-point Gauss-Legendre sum of positive terms per piece, gamma(32) relative,
    # whatever order a batch sums it in (twice, one per side).
    singly = [linear_cycle_average(rate, float(a)) for a in amplitudes]
    assert_allclose(singly, averaged, rtol=2 * gamma(32), atol=0)


def test_the_cycle_average_falls_back_where_its_check_fails():
    # The oracle's ceiling on its rate is a corner it does not declare, so its two
    # fixed rules disagree there and those amplitudes take the adaptive quadrature.
    oracle = adk_oracle("Ar")
    amplitudes = np.array([2e10, 2e11])
    assert float(oracle(2e11)) == oracle.maximum
    averaged = linear_cycle_averages(oracle, amplitudes)
    for value, amplitude in zip(averaged, amplitudes, strict=True):
        exact = _cycle_sources(np.array([complex(amplitude)]), oracle)[0]
        assert value == pytest.approx(exact, rel=4e-10)
    assert averaged[1] == _cycle_sources(np.array([2e11 + 0j]), oracle)[0]
    with pytest.raises(GridError, match="cycle amplitudes"):
        linear_cycle_averages(oracle, np.array([1e10, -1.0]))


def test_bind_reference_field_binds_only_unbound_optmat_rates():
    unbound = PlasmaResponse("Ar", OptmatRate("Ar", WAVELENGTH))
    fixed = PlasmaResponse("Ar", OptmatRate("Ar", WAVELENGTH, reference_field=2e10))
    legacy = PlasmaResponse("Ar", adk_oracle("Ar"))
    kerr = KerrResponse(1e-50)
    first, second, third, fourth = cast("Any", bind_reference_field)(
        (unbound, fixed, legacy, kerr), 3e10, (3e7, 9e11)
    )
    assert first.rate.effective_reference_field == 3e10
    assert first.rate.field_range == (3e7, 9e11)
    assert first.potential == first.rate.ionisation_potential
    assert second is fixed
    assert third is legacy
    assert fourth is kerr


# --------------------------------------------------- statuses and the carrier


def test_an_untested_model_warns_once_per_run_and_a_comparison_form_is_silent():
    with _optmat.material_context():
        with pytest.warns(MaterialStatusWarning, match="untested") as caught:
            first = OptmatRate("SF6", WAVELENGTH)
            first.bound(3e10)
        assert len(caught) == 1
    assert first.status == "untested"
    comparison = OptmatRate("N2", WAVELENGTH, model="moadk_zhao_2010")
    assert comparison.status == "comparison"
    assert comparison.manifest_entry()["uncertainty_factor"] is None


@pytest.mark.parametrize("gas", ["Ar", "N2"])
def test_a_carrier_below_350_nm_warns_once_and_is_recorded(gas):
    with _optmat.material_context() as log:
        with pytest.warns(MaterialRangeWarning, match="350 nm") as caught:
            rate = OptmatRate(gas, 300e-9)
            rate(rate.maximum_field * 0.5)
        assert len(caught) == 1
    assert log.range_entries()


def test_optmats_other_warnings_pass_through():
    model = ionisation_rate_model("Ar", WAVELENGTH)

    class Noisy:
        """An optmat model whose rate warns about something else."""

        identity = model.identity

        def __getattr__(self, name):
            return getattr(model, name)

        def instantaneous_rate(self, field):
            warnings.warn("not a range warning", RuntimeWarning, stacklevel=2)
            return model.instantaneous_rate(field)

    with pytest.warns(RuntimeWarning, match="not a range warning"):
        OptmatRate(Noisy())


# ------------------------------------------------ overrides and custom species


def test_an_override_changes_the_digest_the_cache_key_and_the_rate():
    base = ionisation_rate_model("Ar", WAVELENGTH)
    override = base.with_parameters(electrons=2.0 * base.electrons)
    curated, changed = OptmatRate(base), OptmatRate(override)
    assert changed.status == "user"
    assert changed.material_records[0].base == curated.material_records[0].digest
    assert changed.material_records[0].digest != curated.material_records[0].digest
    span = (1e9, 5e10)
    assert rate_table_key(curated, span, 1e-6) != rate_table_key(changed, span, 1e-6)
    fields = window(curated, 9)
    assert_array_equal(changed(fields), override.instantaneous_rate(fields))
    assert curated.wavelength == WAVELENGTH
    named = OptmatRate("Ar", WAVELENGTH)
    assert rate_table_key(named, span, 1e-6) == rate_table_key(curated, span, 1e-6)


def test_a_custom_species_takes_optmats_explicit_parameters():
    energy = 15.0 * E_CHARGE
    model = IonisationRateModel.from_parameters(
        "X",
        WAVELENGTH,
        ionization_energy=energy,
        orbital_l=1,
        electrons=1.0,
        static_constants=(9.0,),
        citation="a test species",
    )
    rate = OptmatRate(model)
    assert rate.species_identity == "X"
    assert rate.status == "user"
    assert rate.ionisation_potential == energy
    assert rate.energy_kind == "not stated"
    custom = dataclasses.replace(species("Ar"), name="X")
    response = PlasmaResponse(custom, rate)
    assert response.potential == energy
    fields = window(rate, 9)
    assert_array_equal(rate(fields), model.instantaneous_rate(fields))


def test_construction_refuses_what_optmat_does_not_serve():
    with pytest.raises(MaterialError, match="no ionisation rate for 'X'"):
        OptmatRate("X", WAVELENGTH)
    with pytest.raises(MaterialError, match="other family, which refuses Ar"):
        OptmatRate("Ar", WAVELENGTH, model="moadk_zhao_2010")
    with pytest.raises(MaterialError, match="other family, which refuses N2"):
        OptmatRate("N2", WAVELENGTH, model="ppt_ilkov")
    with pytest.raises(MaterialError, match="not one of optmat's atomic rates"):
        OptmatRate("Ar", WAVELENGTH, model="nonsense")
    with pytest.raises(MaterialError, match="not given for SF6"):
        OptmatRate("SF6", WAVELENGTH, model="moadk_zhao_2010")
    with pytest.raises(MaterialError, match="needs the carrier wavelength"):
        OptmatRate("Ar")
    with pytest.raises(MaterialError, match="species name or an optmat rate model"):
        OptmatRate(adk_oracle("Ar"))
    model = ionisation_rate_model("Ar", WAVELENGTH)
    with pytest.raises(MaterialError, match="taken as it is"):
        OptmatRate(model, model="ppt_ilkov")
    with pytest.raises(MaterialError, match="at_wavelength"):
        OptmatRate(model, 1030e-9)
    assert OptmatRate(model, WAVELENGTH).wavelength == WAVELENGTH
    for keyword, value in (
        ("negligible_rate", 0.0),
        ("maximum", math.inf),
        ("reference_field", -1.0),
    ):
        with pytest.raises(MaterialError, match=keyword):
            OptmatRate("Ar", WAVELENGTH, **cast("Any", {keyword: value}))
    with pytest.raises(MaterialError, match="wavelength must be positive"):
        OptmatRate("Ar", True)
    for span in ((1e9, 1e8), (0.0, 1e9), (1e9, math.inf)):
        with pytest.raises(MaterialError, match="field_range"):
            OptmatRate("Ar", WAVELENGTH, field_range=span)


def test_a_rate_without_a_threshold_to_tabulate_from_is_refused():
    model = ionisation_rate_model("Ar", WAVELENGTH)

    class Flat:
        """An optmat model whose rate does not fall at low field."""

        identity = model.identity

        def __getattr__(self, name):
            return getattr(model, name)

        def instantaneous_rate(self, field):
            return np.ones_like(np.asarray(field, dtype=float))

    with pytest.raises(MaterialError, match="no threshold field"):
        OptmatRate(Flat())
    with pytest.raises(MaterialError, match="even at its barrier-suppression field"):
        OptmatRate("Ar", WAVELENGTH, negligible_rate=1e30)


def test_the_record_describes_the_model_and_the_policies():
    rate = OptmatRate("Ar", WAVELENGTH).bound(3e10, (3e7, 9e11))
    model = rate.optmat_model
    (record,) = rate.material_records
    assert record == _optmat.record(model)
    entry: Any = rate.manifest_entry()
    assert entry["digest"] == record.digest
    assert entry["threshold"] == {
        "kind": "negligible-rate",
        "negligible_rate": NEGLIGIBLE_RATE,
        "field": rate.threshold,
    }
    assert entry["energy_loss"]["reference_field"] == 3e10
    assert entry["energy_loss"]["energy"] == rate.ionisation_potential
    assert entry["energy_loss"]["bound"] == rate.energy_bound
    assert entry["maximum_field"] == float(model.maximum_field)
    assert entry["uncertainty_factor"] == float(model.uncertainty_factor)
    assert rate.energy_kind == model.energy_kind
    assert rate.keldysh(3e10) == float(model.keldysh(3e10))
    assert record.digest[:12] in rate.citation
    assert "3e+10 V/m" in rate.citation
    assert "weak-field limit" in OptmatRate("Ar", WAVELENGTH).citation
    parameters = dict(rate.parameters)
    assert parameters["digest"] == record.digest
    assert parameters["energy_loss"] == "peak-weighted"
    assert list(rate.parameters) == sorted(rate.parameters)


# ---------------------------------------------------- the input and the run


def test_the_input_is_checked_once_per_run():
    rate = OptmatRate("Ar", WAVELENGTH)
    gamma_above_one = 0.5 * rate.maximum_field * 0.1
    assert rate.keldysh(gamma_above_one) > 1.0
    rate.check_input(3e10)
    rate.check_input(gamma_above_one, circular=1e-9)
    with _optmat.material_context():
        with pytest.warns(MaterialRangeWarning, match="exceeds the maximum") as caught:
            rate.check_input(2.0 * rate.maximum_field)
            rate.check_input(2.0 * rate.maximum_field)
        assert len(caught) == 1
        with pytest.warns(MaterialRangeWarning, match="not linearly polarised"):
            rate.check_input(gamma_above_one, circular=1.0)
    # In the tunnelling regime the quasi-static average is the right one.
    tunnelling = 0.9 * rate.maximum_field
    assert rate.keldysh(tunnelling) < 1.0
    rate.check_input(tunnelling, circular=1.0)


@pytest.mark.parametrize("envelope", [False, True])
def test_the_peak_field_report_inverts_the_recorded_rate(envelope):
    rate = OptmatRate("Ar", WAVELENGTH).bound(3e10, (3e7, 9e11))
    table = build_rate_table(rate, (3e7, 9e11), use_cache=False)
    average = partial(linear_cycle_average, table) if envelope else None
    field = 2.2e10
    recorded = average(field) if average is not None else float(rate(field))
    report = peak_field_report(rate, recorded, label="argon", average=average)
    assert report.largest_field == pytest.approx(field, rel=1e-10)
    assert not report.reached_maximum
    assert report.energy_at_largest is not None
    assert report.energy_at_largest == pytest.approx(
        float(rate.optmat_model.energy_per_event(field)), rel=1e-9
    )
    assert report.departure == pytest.approx(
        report.energy_at_largest / rate.ionisation_potential - 1.0
    )
    assert report.to_dict()["label"] == "argon"
    held = (
        average(rate.maximum_field)
        if average is not None
        else float(rate(rate.maximum_field))
    )
    reached = peak_field_report(rate, held, label="argon", average=average)
    assert reached.reached_maximum
    assert reached.largest_field == rate.maximum_field
    none = peak_field_report(rate, 0.0, label="argon", average=average)
    assert none.largest_field is None
    assert none.energy_at_largest is None
    assert none.departure is None


# ----------------------------------------------------- the selection's rate


def test_a_selection_names_optmats_rate_never_solprops():
    selection = MaterialSelection()
    argon = selected_rate(species("Ar"), selection, WAVELENGTH)
    assert argon.material_records[0] == _optmat.record(
        ionisation_rate_model("Ar", WAVELENGTH)
    )
    nitrogen = selected_rate(species("N2"), selection, WAVELENGTH)
    assert nitrogen.material_records[0] == _optmat.record(
        molecular_rate_model("N2", WAVELENGTH)
    )
    named = selected_rate(
        species("Ar"), MaterialSelection(ionisation="ppt_ilkov"), WAVELENGTH
    )
    assert named.optmat_model.model == "ppt_ilkov"
    carried = selected_rate(
        species("Ar"), MaterialSelection(carrier_wavelength=1030e-9), WAVELENGTH
    )
    assert carried.wavelength == 1030e-9
    model = ionisation_rate_model("Ar", WAVELENGTH).with_parameters(electrons=2.0)
    chosen = selected_rate(
        species("Ar"), MaterialSelection(ionisation=model), WAVELENGTH
    )
    assert chosen.optmat_model is model


def test_a_selection_refuses_a_custom_species_and_a_missing_carrier():
    selection = MaterialSelection()
    renamed = dataclasses.replace(species("Ar"), name="X")
    with pytest.raises(MaterialError, match="custom species"):
        selected_rate(renamed, selection, WAVELENGTH, label="probe")
    argon = species("Ar")
    assert argon.ionisation is not None
    supplied = dataclasses.replace(
        argon,
        ionisation=dataclasses.replace(argon.ionisation, potential=2.6e-18),
    )
    with pytest.raises(MaterialError, match="custom species"):
        selected_rate(supplied, selection, WAVELENGTH)
    with pytest.raises(MaterialError, match="custom species"):
        selected_rate(species("Air"), selection, WAVELENGTH)
    with pytest.raises(MaterialError, match="carrier wavelength"):
        selected_rate(argon, selection, None)
    model = ionisation_rate_model("Ne", WAVELENGTH)
    with pytest.raises(MaterialError, match="describes Ne, not Ar"):
        selected_rate(argon, MaterialSelection(ionisation=model), WAVELENGTH)


# ------------------------------------------------- the hydrogenic form's parity
#
# optmat's potential-only hydrogenic form must be the ADK formula with l* = 0, Z = 1
# and no angular factor, the form the test-only oracle builds from the same energy and
# occupancy (bit for bit solprop's former potential-only ``ADK``). The residuals are
# rounding of the same formula assembled in a different order (1e-12 measured), so
# 1e-10 is the gate. Rates are compared below the oracle's ceiling, which optmat does
# not apply. (The parities of optmat's MO-ADK, molecular PPT and ppt_ilkov forms with
# solprop's own formulas were checked here before those formulas were deleted, SEP 3,
# Phase 4b; there is no oracle for them, so they went with the formulas.)

FIELDS = np.geomspace(2e9, 8e10, 41)


@pytest.mark.parametrize("gas", ["H2", "D2", "CH4", "SF6", "N2O"])
@pytest.mark.parametrize("occupancy", [1, 2])
def test_the_hydrogenic_form_is_the_potential_only_adk_oracle(gas, occupancy):
    energy = ionisation_potential(gas)
    model = molecular_rate_model(gas, WAVELENGTH, "adk_hydrogenic").with_parameters(
        ionization_energy=energy, electrons=float(occupancy)
    )
    oracle = AdkOracle(
        gas, energy, orbital=0, charge=1.0, occupancy=occupancy, hydrogenic=True
    )
    actual = np.minimum(quiet(model.instantaneous_rate, FIELDS), MAX_RATE)
    assert_allclose(actual, oracle(FIELDS), rtol=1e-10)


# ------------------------------------------- a molecular rate through its table


@pytest.mark.parametrize("gas", ["H2", "D2", "N2", "O2"])
def test_a_molecular_table_interpolates_and_its_plasma_pays_the_potential(gas):
    rate = OptmatRate(gas, WAVELENGTH)
    table = build_rate_table(rate, (1e9, 5e10), rtol=1e-6, use_cache=False)
    # The table reconstructs its lower bound through exp(log E); that can move by
    # ulps. Compare only inside the table's declared interval, and separately verify
    # the zero region rather than relaxing tiny-rate accuracy.
    fields = np.geomspace(table.field_min * (1 + 1e-12), table.field_max, 2001)
    assert float(table(np.nextafter(table.field_min, 0.0))) == 0.0
    assert_allclose(table(fields), rate(fields), rtol=1e-6, atol=0)
    plasma = PlasmaResponse(gas, table)
    # The table keeps the rate's energy per event, so its plasma pays the energy the
    # rate's exponent uses (SEP 3, D8): the vertical one, not the adiabatic potential.
    assert plasma.potential == rate.plasma_energy
    assert table.plasma_energy == rate.plasma_energy
    errors = []
    for n in (1025, 2049, 4097):
        time = np.linspace(0, 5e-15, n)
        field = np.full(n, 4e10)
        dt = float(time[1] - time[0])
        fraction = ionisation_fraction(table, field, dt)
        # A constant rate: the trapezoid is exact up to Higham's n * eps.
        expected = -np.expm1(-float(table(4e10)) * time)
        assert_allclose(fraction, expected, rtol=1e-12, atol=1e-14)
        assert np.all((fraction >= 0) & (fraction <= 1))
        loss = plasma._loss(field[None, :], field, table(field), fraction)[0]
        work = integrate.trapezoid(loss * field, time)
        energy = plasma.potential * fraction[-1]
        errors.append(abs(work / energy - 1))
    assert errors[-1] < 1e-7  # Constant-field trapezoid error at 4097 samples.
    # Nonzero loss integrates at second order, rather than an algebraic identity test.
    assert 3.7 < errors[0] / errors[1] < 4.3
    assert 3.7 < errors[1] / errors[2] < 4.3


@pytest.mark.parametrize("gas", ["Ar", "Xe", "He", "N2"])
@pytest.mark.parametrize("fraction", [0.90, 0.95])
def test_an_envelope_peak_near_the_maximum_field_is_recovered(gas, fraction):
    # An envelope run's statistic is its own cycle average of its table, which above
    # the barrier-suppression field departs from optmat's cycle-averaged rate.
    # Inverted through that same average, a known peak comes back to the inversion's
    # tolerance: the 2e-10 of the adaptive average over a rate whose logarithmic
    # slope is of order ten, well inside 1e-8.
    rate = OptmatRate(gas, WAVELENGTH)
    table = build_rate_table(rate, (1e8, 30 * rate.maximum_field), use_cache=False)
    field = fraction * rate.maximum_field
    average = partial(linear_cycle_average, table)
    report = peak_field_report(rate, average(field), label=gas, average=average)
    assert not report.reached_maximum
    assert report.largest_field == pytest.approx(field, rel=1e-8)

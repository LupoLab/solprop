# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from witness_materials import borzsonyi, malitson_silica

from solprop.constants import C
from solprop.exceptions import MaterialRangeWarning, ModeError
from solprop.modes import (
    EVANESCENT_INDEX_FLOOR,
    MarcatiliMode,
    Mode,
    TubeLattice,
    VincettiMode,
    core_radius_from_gap,
    tube_gap,
    tube_radius_from_gap,
)
from solprop.modes.antiresonant import (
    AREA_FACTOR,
    CONVENTIONS,
    LEGACY_CONVENTIONS,
    MAX_TERMS,
    RADIUS_COEFFICIENT,
)
from solprop.modes.base import alpha, beta2, db_per_m, loss_length, transmission
from solprop.modes.protocol import ParameterisedMode
from solprop.profiles import Linear

GEOMETRIES = [
    (15e-6, 180e-9, 9e-6, 6),
    (5e-6, 80e-9, 1e-6, 10),
    (core_radius_from_gap(10e-6, 8, 5e-6), 1e-6, 10e-6, 8),
]
LEGACY_RADIUS_COEFFICIENT = LEGACY_CONVENTIONS.radius_coefficient


class LunaFormMode(VincettiMode):
    """Luna's and pyJCMwave's form of the fits, from the production building blocks.

    Every accessor -- `neff`, `alpha`, `beta2`, `effective_area` -- reaches
    `_spectral_model`, so switching its default conventions is the whole change.
    The witness tests compare this form with Luna's stored values; the production
    form differs from it by the closed-form amounts checked below and is checked
    against the sympy and FEM witnesses instead.
    """

    def _spectral_model(self, terms, radius, density, conventions=LEGACY_CONVENTIONS):
        return super()._spectral_model(terms, radius, density, conventions)

    def effective_area(self, omega, z=0.0, *, conventions=LEGACY_CONVENTIONS):
        return super().effective_area(omega, z, conventions=conventions)


def make_mode(geometry=GEOMETRIES[0], cls=VincettiMode, **kwargs):
    radius, wall, outer, count = geometry
    return cls(radius, lattice=TubeLattice(wall, outer, count), cladding=1.45, **kwargs)


def test_luna_regression():
    omega = np.array([2 * math.pi * C / 1030e-9])
    # Luna's evacuated HE11 with its loss, stated (``docs/dev/testing.md``).
    luna = replace(
        make_mode(GEOMETRIES[2], LunaFormMode, kind="HE", n=1, m=1, loss=True),
        cladding=1.44,
    )
    result = luna.neff(omega)[0]
    # Rounding bounds for the constant-index Luna test. Luna's stored Im n_eff is twice
    # the convention (c alpha/omega there, not c alpha/(2 omega)), so it is halved.
    assert result.real == pytest.approx(0.9998598623672965, rel=1e-13)
    assert result.imag == pytest.approx(3.4579455755137454e-8 / 2, rel=1e-12)
    # Evacuated, so only the radius coefficient separates the production form: it
    # gives Re n_eff = 0.9998591986525931 here, 6.64e-7 below Luna, and the same
    # imaginary part. The difference is the closed form checked below.
    production = replace(make_mode(GEOMETRIES[2]), cladding=1.44).neff(omega)[0]
    assert production.imag == result.imag
    assert -7e-7 < production.real - result.real < -6e-7


@given(st.floats(1e-6, 100e-6), st.floats(0, 10e-6), st.integers(3, 20))
def test_geometry_roundtrip(outer, gap, count):
    radius = core_radius_from_gap(outer, count, gap)
    # Three elementary operations per inverse; cancellation is bounded by geometry size.
    np.testing.assert_allclose(
        tube_gap(radius, outer, count),
        gap,
        rtol=0,
        atol=16 * np.finfo(float).eps * (radius + outer),
    )
    assert tube_radius_from_gap(radius, count, gap) == pytest.approx(outer, rel=1e-14)
    lattice = TubeLattice.from_gap(radius, gap, count, outer / 10)
    assert lattice.core_radius_for(gap) == pytest.approx(radius, rel=1e-14)
    assert lattice.gap(radius) == pytest.approx(gap, abs=1e-18)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_loss_policy_and_wide_band(geometry):
    mode = make_mode(geometry)
    omega = 2 * math.pi * C / np.geomspace(100e-9, 20e-6, 200)
    values = mode.neff(omega)
    assert np.all(np.isfinite(values))
    assert np.all(np.imag(values) >= 0)
    assert not mode._gain_warned
    np.testing.assert_array_equal(np.imag(replace(mode, loss=False).neff(omega)), 0)
    # Half the loss is the lossy evaluation's imaginary part halved: a power-of-two
    # scale on another code path, held to a few roundings of the loss term.
    np.testing.assert_allclose(
        np.imag(replace(mode, loss=0.5).neff(omega)),
        np.imag(values) / 2,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    assert np.all(np.isfinite(mode.dneff_domega(omega)))


def test_protocol_and_freezing():
    mode = replace(
        make_mode(),
        radius=Linear(15e-6, 18e-6, 1.0),
        gas="Ar",
        pressure=Linear(1.0, 2.0, 1.0),
    )
    assert isinstance(mode, Mode)
    assert isinstance(mode, ParameterisedMode)
    assert list(mode.parameter_paths) == ["core_radius", "density"]
    omega = np.array([2e15, 3e15])
    point = 0.37
    frozen = mode.at_parameters(
        {
            key: float(path(np.array([point]))[0])
            for key, path in mode.parameter_paths.items()
        }
    )
    assert isinstance(frozen, VincettiMode)
    # The frozen mode evaluates the same closed form at the sampled parameters: the
    # ladder's one-evaluation budget of 16 eps (accuracy.md section 7) covers any
    # reordering of its few tens of roundings.
    np.testing.assert_allclose(
        frozen.neff(omega), mode.neff(omega, point), rtol=16 * np.finfo(float).eps
    )
    assert isinstance(mode.rotated(0.3), VincettiMode)
    assert isinstance(mode.orthogonal_polarisation, VincettiMode)
    parent = MarcatiliMode(mode.radius, mode.gas, mode.pressure, cladding=1.45)
    assert mode.aeff(point) == parent.aeff(point)
    assert mode.normalisation(point) == parent.normalisation(point)
    assert mode.dimlimits(point) == parent.dimlimits(point)
    points = np.array([[0.0, 0.0], [5e-6, 0.3]])
    # The field is the parent's own formula evaluated through the subclass.
    np.testing.assert_allclose(
        mode.field(points, point),
        parent.field(points, point),
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    # The polarisation convention is the parent's too: phi = 0 is the x-polarised
    # `even` member and the quarter turn its `odd` partner (ADR-031).
    assert mode.principal_component() == "x"
    assert mode.channel_label.display_label == "HE1,1:even"
    assert mode.orthogonal_polarisation.channel_label.display_label == "HE1,1:odd"
    assert mode.orthogonal_polarisation.principal_component() == "y"


def test_refusals_and_warnings(caplog):
    with pytest.raises(ModeError, match="reduced"):
        make_mode(model="reduced")
    with pytest.raises(ModeError, match="gap"):
        replace(make_mode(), radius=1e-6)
    with pytest.raises(ModeError, match="gap"):
        replace(make_mode(), radius=Linear(15e-6, 1e-6, 1.0))
    caplog.clear()
    mode = make_mode(GEOMETRIES[2])
    assert not caplog.records
    replace(mode, m=2)
    assert len(caplog.records) == 1
    caplog.clear()
    make_mode()
    assert len(caplog.records) == 1


DATA = Path(__file__).parents[1] / "data"
SYMBOLIC = json.loads((DATA / "symbolic_vincetti.json").read_text())


def as_complex(pairs):
    values = np.asarray(pairs)
    return values[:, 0] + 1j * values[:, 1]


@pytest.mark.parametrize("case", SYMBOLIC["cases"])
def test_symbolic_derivative(case):
    mode = make_mode(case["geometry"])
    wavelength = np.asarray(case["wavelength"])
    omega = 2 * math.pi * C / wavelength
    gas, dgas = as_complex(case["gas_index"]), as_complex(case["dgas_domega"])
    terms = replace(
        mode.sampled_terms(omega),
        tube_index=np.asarray(case["tube_index"]),
        dtube_index_domega=np.asarray(case["dtube_domega"]),
        core_susceptibility=gas**2 - 1,
        dcore_susceptibility_domega=2 * gas * dgas,
    )
    values, derivative = mode._model_values(terms, case["geometry"][0], 1.0)
    # Sympy differentiates the independent expression of the production form,
    # including all material slopes. The 1e-10 bound is declared in advance;
    # measured worst 1.07e-12 on the derivative and 5.0e-11 on the value, the
    # latter at a near-zero refractivity where the gas square root's rounding
    # dominates (hence the absolute allowance).
    np.testing.assert_allclose(
        derivative, as_complex(case["derivative"]), rtol=1e-10, atol=0
    )
    np.testing.assert_allclose(
        values, as_complex(case["refractivity"]), rtol=1e-10, atol=4e-16
    )
    fit = mode._spectral_model(terms, case["geometry"][0], 1.0)
    area = (
        AREA_FACTOR
        / (8 * math.pi * np.real(fit.core_index.value))
        * (mode.u_nm * wavelength) ** 2
        / np.real(fit.decrement.value - fit.correction.value)
    )
    # Measured worst 5.7e-13.
    np.testing.assert_allclose(area, case["effective_area"], rtol=1e-10, atol=0)


@pytest.mark.parametrize("wavelength", [800e-9, 1200e-9, 1800e-9])
def test_contour_derivative(wavelength):
    mode = make_mode()
    omega = np.array([2 * math.pi * C / wavelength])
    nodes = 128
    offset = 0.01 * omega * np.exp(2j * math.pi * np.arange(nodes) / nodes)
    shifted_omega = omega + offset
    # The materials are constant (vacuum, n_t = 1.45), so sampling the centre
    # frequency `nodes` times and moving only the frequency inside the terms
    # continues the same holomorphic function, as the parent's contour check does.
    terms = mode.sampled_terms(np.full(nodes, omega[0]))
    shifted = replace(
        terms, omega=shifted_omega, wavelength=2 * math.pi * C / shifted_omega
    )
    values = mode._model_values(shifted, GEOMETRIES[0][0], 0.0)[0]
    # A 1% circle avoids the narrow loss poles. 128 nodes resolve this holomorphic
    # annulus; rounding is amplified by 1/radius. Same 1e-11 bound as the parent.
    np.testing.assert_allclose(
        mode.dneff_domega(omega), [np.sum(values / offset) / nodes], rtol=1e-11, atol=0
    )


LUNA = json.loads((DATA / "luna_antiresonant.json").read_text())

# Luna was run at 294 K (``generate_luna_antiresonant.py``), solprop's former default;
# the witness is not regenerated, so every mode compared with it is built at 294 K.
LUNA_TEMPERATURE = 294.0


@pytest.mark.parametrize("case", LUNA["cases"].values())
def test_luna_witness(case):
    radius, wall, outer, count = case["geometry"]
    rows = np.asarray(case["samples"])
    omega = 2 * math.pi * C / rows[:, 0]
    silica = case["cladding"] == "silica"
    # Luna's stored values are compared with Luna's form of the fits assembled from
    # the production building blocks (LEGACY_CONVENTIONS); the production form is
    # checked against the sympy and FEM witnesses and differs from this one by the
    # closed forms of test_production_form_against_luna_form.
    mode = LunaFormMode(
        radius,
        "Ar",
        case["pressure_bar"],
        # Luna's argon index, Börzsönyi's fit, supplied explicitly.
        susceptibility=borzsonyi("Ar"),
        lattice=TubeLattice(wall, outer, count),
        # Malitson's silica, the real part of solprop's former default, on which the
        # measured differences below were taken, supplied explicitly since SEP 3's
        # Phase 4b. (optmat's absolute fit/SiO2 sits 2.7e-4 higher in the index, which
        # moves this near-resonant wall's loss by 45 % and is not what is compared.)
        cladding=malitson_silica() if silica else float(case["cladding"]),
        temperature=LUNA_TEMPERATURE,
        # Luna's HE11 with its loss, stated rather than defaulted.
        kind="HE",
        n=1,
        m=1,
        loss=True,
    )
    # The witness deliberately extrapolates the 400--1000 nm argon fit. Assert the
    # range warning; these points compare implementations, not material accuracy.
    with pytest.warns(MaterialRangeWarning, match="borzsonyi"):
        terms = mode.sampled_terms(omega)
        result = mode.neff_minus_one(omega)
        attenuation = alpha(mode, omega)
        decibels = db_per_m(mode, omega)
        area = mode.effective_area(omega)
        dispersion = beta2(mode, omega[:10])
    fit = mode._spectral_model(terms, *mode._scalars(0.0))
    if silica:
        # Measured material-data differences, not rounding: worst refractivity
        # 8.99e-7, loss 1.891%, area .03354%, radius .00968%, correction .674%.
        np.testing.assert_allclose(np.real(result), rows[:, 1], rtol=0, atol=1e-6)
        np.testing.assert_allclose(attenuation, rows[:, 2], rtol=0.02, atol=0)
        np.testing.assert_allclose(area, rows[:, 4], rtol=4e-4, atol=0)
        np.testing.assert_allclose(
            fit.effective_radius.value, rows[:, 5], rtol=1e-4, atol=0
        )
        np.testing.assert_allclose(fit.correction.value, rows[:, 6], rtol=0.007, atol=0)
        # The silica interpolation also changes its slope; max absolute GVD
        # difference is 1.436e-27 s^2/m (relative values are unstable at zero).
        np.testing.assert_allclose(dispersion, rows[:10, 10], rtol=0, atol=1.5e-27)
    else:
        # Luna subtracts unity after summing the index: absolute 4 eps is the
        # rounding bound, including the gas sqrt. Max measured is 3.06e-16.
        np.testing.assert_allclose(
            np.real(result), rows[:, 1], rtol=0, atol=4 * np.finfo(float).eps
        )
        # 128 Lorentzians magnify cutoff rounding; measured max 1.37e-13.
        np.testing.assert_allclose(
            attenuation, rows[:, 2], rtol=1024 * np.finfo(float).eps, atol=0
        )
        # Measured 1.09e-15 on the radius and 6.99e-14 on the correction.
        np.testing.assert_allclose(
            fit.effective_radius.value,
            rows[:, 5],
            rtol=16 * np.finfo(float).eps,
            atol=0,
        )
        np.testing.assert_allclose(
            fit.correction.value, rows[:, 6], rtol=1024 * np.finfo(float).eps, atol=0
        )
        # Compare the inverse area denominator: Luna's near-unity subtraction
        # loses relative precision at its zeros (area error reaches 2.23e-10).
        numerator = AREA_FACTOR / (8 * math.pi) * (mode.u_nm * rows[:, 0]) ** 2
        np.testing.assert_allclose(
            numerator / area,
            numerator / rows[:, 4],
            rtol=0,
            atol=4 * np.finfo(float).eps,
        )
        # Finite differences of beta/beta1 have different truncation near the
        # resonances. Measured worst 5.22e-4; this is not an analytic-slope gate.
        np.testing.assert_allclose(dispersion, rows[:10, 10], rtol=6e-4, atol=0)
    np.testing.assert_allclose(
        decibels,
        rows[:, 7],
        rtol=0.02 if silica else 1024 * np.finfo(float).eps,
        atol=0,
    )
    # Luna converts CL to alpha and back; the stored columns are one correctly
    # rounded product apart, which is exact on every IEEE platform, rather than an
    # exact floating-point inverse.
    np.testing.assert_array_equal(rows[:, 3], rows[:, 2] * (10 / math.log(10)))
    np.testing.assert_allclose(
        rows[:, 11], mode.lattice.gap(radius), rtol=1e-14, atol=1e-20
    )
    np.testing.assert_allclose(rows[:, 12], radius, rtol=1e-14, atol=0)
    np.testing.assert_allclose(rows[:, 13], outer, rtol=1e-14, atol=0)


def test_production_form_against_luna_form():
    """The five conventions change the fits by their closed forms, and only those.

    The dispersion width is common to both forms. Evacuated, the gas index is unity
    and the three gas-index choices are identities, so the radius coefficient is the
    whole difference; filled, each choice moves its own term by the amount its
    definition says (``docs/guide/waveguides/modes.md``, "The five conventions").
    """
    lattice = TubeLattice(180e-9, 9e-6, 6)
    radius, tube_index = 15e-6, 1.45
    omega = 2 * math.pi * C / np.geomspace(400e-9, 990e-9, 60)
    wavelength = 2 * math.pi * C / omega
    eps = np.finfo(float).eps

    def prefactor(frequency):
        return 1.027 + 0.001 * (frequency + 2 / frequency**4)

    # Evacuated: equal to rounding but for the radius coefficient, whose effect on
    # R_eff^2 is the coefficient difference times its own term. The rounding is the
    # complex square root of n_t^2 - n_g^2 with n_g = 1 + 0j against the real
    # square root of n_t^2 - 1: measured 4 ulps on the refractivity.
    mode = VincettiMode(radius, lattice=lattice, cladding=tube_index)
    terms = mode.sampled_terms(omega)
    luna = mode._spectral_model(terms, radius, 0.0, LEGACY_CONVENTIONS)
    production = mode._spectral_model(terms, radius, 0.0)
    luna_coefficient = mode._spectral_model(
        terms, radius, 0.0, replace(CONVENTIONS, radius_coefficient=20.0)
    )
    for name in ("refractivity", "effective_radius", "correction", "decrement"):
        for part in ("value", "slope"):
            expected = getattr(getattr(luna, name), part)
            # The correction's slope crosses zero, so the floor is relative to
            # the largest element rather than to each.
            np.testing.assert_allclose(
                getattr(getattr(luna_coefficient, name), part),
                expected,
                rtol=8 * eps,
                atol=8 * eps * float(np.max(np.abs(expected))),
            )
    np.testing.assert_allclose(
        mode.effective_area(omega, conventions=LEGACY_CONVENTIONS),
        mode.effective_area(
            omega, conventions=replace(CONVENTIONS, radius_coefficient=20.0)
        ),
        rtol=8 * eps,
        atol=0,
    )
    frequency = 2 * lattice.wall_thickness / wavelength * math.sqrt(tube_index**2 - 1)
    expected = (
        prefactor(frequency) ** 2
        * lattice.n_tubes
        / math.pi
        * 3
        / 64
        * lattice.tube_radius**2
        * (RADIUS_COEFFICIENT - LEGACY_RADIUS_COEFFICIENT)
        * wavelength
        / radius
        * lattice.gap(radius)
        / lattice.tube_radius
    )
    # The difference of squares is 0.9 % of R_eff^2, so its rounding is 1e-13.
    np.testing.assert_allclose(
        production.effective_radius.value**2 - luna.effective_radius.value**2,
        expected,
        rtol=1e-12,
        atol=0,
    )

    # Argon at 30 bar, one choice at a time from Luna's form.
    mode = VincettiMode(radius, "Ar", 30.0, lattice=lattice, cladding=tube_index)
    terms = mode.sampled_terms(omega)
    scalars = mode._scalars(0.0)
    luna = mode._spectral_model(terms, *scalars, LEGACY_CONVENTIONS)
    gas_index = np.real(luna.core_index.value)
    assert np.all(gas_index > 1.005)
    # The single-power decrement is Luna's times n_g: with D_0 the decrement in
    # vacuum, Luna's D_0 / n_g^2 becomes D_0 / n_g, and the refractivity moves by
    # -D_0 (n_g - 1) / n_g^2, which is -(n_g - 1) times Luna's decrement. Rounding
    # is a few eps of the decrement; the refractivity difference is 1e-4 of the
    # refractivity at 30 bar, so its rounding is 1e-12 relative.
    single = mode._spectral_model(
        terms, *scalars, replace(LEGACY_CONVENTIONS, single_power_decrement=True)
    )
    np.testing.assert_allclose(
        single.decrement.value, luna.decrement.value * gas_index, rtol=8 * eps, atol=0
    )
    np.testing.assert_allclose(
        single.refractivity.value - luna.refractivity.value,
        -luna.decrement.value * (gas_index - 1),
        rtol=1e-10,
        atol=0,
    )
    # The correction is the same term reached through the two forms' arrangements.
    np.testing.assert_allclose(
        single.correction.value, luna.correction.value, rtol=8 * eps, atol=0
    )
    # The area's prefactor divides by n_g and touches nothing else.
    np.testing.assert_allclose(
        mode.effective_area(
            omega, conventions=replace(LEGACY_CONVENTIONS, area_over_gas_index=True)
        ),
        mode.effective_area(omega, conventions=LEGACY_CONVENTIONS) / gas_index,
        rtol=4 * eps,
        atol=0,
    )
    # The filled frequency scales F by sqrt((n_t^2 - n_g^2) / (n_t^2 - 1)); R_eff
    # carries it only through its prefactor, whose ratio is the closed form.
    filled = mode._spectral_model(
        terms, *scalars, replace(LEGACY_CONVENTIONS, filled_frequency=True)
    )
    scale = np.sqrt((tube_index**2 - gas_index**2) / (tube_index**2 - 1))
    np.testing.assert_allclose(
        filled.effective_radius.value / luna.effective_radius.value,
        prefactor(scale * frequency) / prefactor(frequency),
        rtol=1e-12,
        atol=0,
    )
    # It moves the loss and the correction by more than rounding (1.3 % at 800 nm
    # for the loss at this pressure) and leaves the minimum loss's factors alone.
    filled_loss = np.imag(filled.refractivity.value)
    assert np.max(np.abs(filled_loss / np.imag(luna.refractivity.value) - 1)) > 1e-3
    assert np.max(np.abs(filled.correction.value / luna.correction.value - 1)) > 1e-3
    # All five together differ from Luna by more than rounding in every piece.
    production = mode._spectral_model(terms, *scalars)
    production_index = np.real(production.refractivity.value)
    assert np.max(np.abs(production_index - np.real(luna.refractivity.value))) > 1e-6
    assert np.all(production.effective_radius.value != luna.effective_radius.value)


@pytest.mark.parametrize("row", LUNA["stale"])
def test_stale_silica_is_not_reference(row):
    wavelength, stale, correct, stale_loss, correct_loss, stale_tube, correct_tube = row
    mode = LunaFormMode(
        15e-6,
        "Ar",
        1.0,
        susceptibility=borzsonyi("Ar"),
        lattice=TubeLattice(180e-9, 12e-6, 6),
        cladding=correct_tube,
        temperature=LUNA_TEMPERATURE,
        kind="HE",
        n=1,
        m=1,
        loss=True,
    )
    omega = np.array([2 * math.pi * C / wavelength])
    if wavelength < 400e-9:
        with pytest.warns(MaterialRangeWarning):
            result, attenuation = mode.neff_minus_one(omega), alpha(mode, omega)
    else:
        result, attenuation = mode.neff_minus_one(omega), alpha(mode, omega)
    # Hold the correct table index fixed to isolate Luna's stale silica spline
    # bracket from the silica data choice;
    # Luna's form, since the stored values are Luna's.
    np.testing.assert_allclose(
        np.real(result), correct, rtol=0, atol=4 * np.finfo(float).eps
    )
    np.testing.assert_allclose(
        attenuation, correct_loss, rtol=1024 * np.finfo(float).eps, atol=0
    )
    assert abs(float(np.real(result)[0]) - stale) > 1e-9
    assert abs(float(attenuation[0]) / stale_loss - 1) > 0.005
    assert abs(stale_tube - correct_tube) > 0.0004


@pytest.mark.parametrize("analytic", [False, True])
def test_loss_callable(analytic):
    class Curve:
        def __call__(self, omega):
            return (omega / 2e15) ** 2

        def derivative(self, omega):
            return 2 * omega / 2e15**2

    curve = Curve()
    mode = make_mode(loss=curve if analytic else curve.__call__)
    omega = np.array([2e15, 3e15])
    # Callable loss is absolute power loss; c/(2w) fixes the extinction convention.
    np.testing.assert_allclose(
        alpha(mode, omega), curve(omega), rtol=8 * np.finfo(float).eps, atol=0
    )
    expected = C / 2 * (curve.derivative(omega) / omega - curve(omega) / omega**2)
    # Parent's Richardson slope has ~1e-10 rounding floor without a supplied slope.
    np.testing.assert_allclose(
        np.imag(mode.dneff_domega(omega)), expected, rtol=1e-10, atol=0
    )


def test_floor_and_base_loss_helpers():
    mode = make_mode(n=200)
    omega = np.array([2 * math.pi * C / 800e-9])
    # A high azimuthal order is evanescent at this wavelength. The public floor's
    # subtraction/addition of unity costs at most one epsilon in the real index.
    np.testing.assert_allclose(
        np.real(mode.neff(omega)),
        EVANESCENT_INDEX_FLOOR,
        rtol=0,
        atol=np.finfo(float).eps,
    )
    # The floor is a constant by construction: its derivative is an explicit zero.
    np.testing.assert_array_equal(mode.dneff_domega(omega), 0)
    loss = alpha(mode, omega)
    np.testing.assert_allclose(
        db_per_m(mode, omega),
        10 / math.log(10) * loss,
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    np.testing.assert_allclose(
        loss_length(mode, omega), 1 / loss, rtol=8 * np.finfo(float).eps, atol=0
    )
    np.testing.assert_allclose(
        transmission(mode, omega, 0.1),
        np.exp(-loss * 0.1),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"wall_thickness": 0.0}, "wall_thickness"),
        ({"wall_thickness": float("nan")}, "wall_thickness"),
        ({"wall_thickness": 10e-6}, "wall_thickness"),
        ({"tube_radius": float("inf")}, "tube_radius"),
        ({"n_tubes": 2}, "n_tubes"),
        ({"n_tubes": True}, "n_tubes"),
        ({"n_terms": 0}, "n_terms"),
        ({"n_terms": True}, "n_terms"),
        ({"n_terms": MAX_TERMS + 1}, "pole at nu = 7.75"),
    ],
)
def test_invalid_lattice(kwargs, message):
    with pytest.raises(ModeError, match=message):
        TubeLattice(
            **({"wall_thickness": 1e-6, "tube_radius": 10e-6, "n_tubes": 8} | kwargs)
        )
    assert TubeLattice(1e-6, 10e-6, 8).n_terms == MAX_TERMS == 8


def test_invalid_samples_and_geometry():
    mode = make_mode()
    with pytest.raises(ModeError, match="zero frequency"):
        mode.neff(np.array([0.0]))
    with pytest.raises(ModeError, match="tube index"):
        replace(mode, cladding=0.9).neff(np.array([2e15]))
    with pytest.raises(ModeError, match="sampled_terms"):
        mode._model_values(
            MarcatiliMode(15e-6, cladding=1.45).sampled_terms(np.array([2e15])),
            15e-6,
            0.0,
        )
    with pytest.raises(ModeError, match="gap"):
        core_radius_from_gap(10e-6, 8, -1e-6)
    with pytest.raises(ModeError, match="tube_radius"):
        tube_radius_from_gap(10e-6, 8, 1.0)
    lattice = TubeLattice(1e-6, 10e-6, 8)
    touching = VincettiMode(
        lattice.core_radius_for(0.0), lattice=lattice, cladding=1.44
    )
    assert touching.lattice.gap(float(touching.core_radius(np.array([0.0]))[0])) >= 0
    assert repr(touching).startswith("VincettiMode(HE11, a=16.1313 um, t=1000 nm")
    assert repr(touching).endswith("N=8, evacuated, loss=True)")
    assert repr(make_mode(gas="Ar", pressure=1.0)).endswith("N=6, Ar 1 bar, loss=True)")
    assert "Ar 2 bar" in repr(make_mode(gas="Ar", pressure=Linear(2.0, 3.0, 1.0)))


def test_tube_count_warning_is_per_construction(caplog):
    mode = make_mode((15e-6, 1e-6, 9e-6, 3))
    assert len(caplog.records) == 1
    assert "n_tubes=3 outside [6, 10]" in caplog.text
    mode.neff(np.array([2e15]))
    mode.dneff_domega(np.array([3e15]))
    assert len(caplog.records) == 1

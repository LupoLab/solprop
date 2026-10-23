# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify rectangular fields, material derivatives and distinct polarisations."""

import inspect
import json
import math
import warnings
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import dblquad
from witness_materials import borzsonyi

from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning, ModeError
from solprop.modes import ModeCollection, RectMode
from solprop.modes.base import beta, db_per_m, zdw
from solprop.modes.quadrature import (
    cartesian_nodes,
    mode_matrix,
    transverse_quadrature,
)
from solprop.physdata import Mixture, silver
from solprop.profiles import Linear

EPS = np.finfo(float).eps


class VacuumContrastMode(RectMode):
    """Use Luna's evacuated core solely for witness comparisons.

    Setting eps_co to one restores Luna's form in both places the core
    permittivity enters: the wall contrast and the TM-like coefficient's
    permittivity ratio.
    """

    def _core_permittivity(self, terms):
        return np.ones_like(terms.chi), np.zeros_like(terms.dchi)


def test_rectangular_closed_forms_for_indices_one_through_six():
    a, b = 50e-6, 12.5e-6
    for n in range(1, 7):
        for m in range(1, 7):
            mode = RectMode(a, b, n=n, m=m)
            rule = transverse_quadrature([mode], n_r=64, n_theta=64)
            intensity = np.sum(mode.field(rule.nodes(mode.dimlimits())) ** 2, axis=1)
            weights = rule.weights(mode.dimlimits())
            square, quartic = weights @ intensity, weights @ intensity**2
            # 64-point Gauss resolves these smooth trigonometric products far
            # beyond 1e-12; comparison includes normalisation reduction rounding.
            assert_allclose(
                EPS0 * C / 2 * square, mode.normalisation(), rtol=1e-12, atol=0
            )
            assert_allclose(square**2 / quartic, mode.aeff(), rtol=1e-12, atol=0)
            for power, reference in ((2, a * b), (4, 9 * a * b / 16)):
                # Independently integrate the separated Cartesian eigenfunctions;
                # dimensionless coordinates avoid an absolute tolerance in m².
                def integrand(x, y, power=power, m=m, n=n):
                    first = (
                        math.cos(m * math.pi * x / 2)
                        if m % 2
                        else math.sin(m * math.pi * x / 2)
                    )
                    second = (
                        math.cos(n * math.pi * y / 2)
                        if n % 2
                        else math.sin(n * math.pi * y / 2)
                    )
                    return (first * second) ** power

                value, error = dblquad(integrand, -1, 1, -1, 1, epsabs=0, epsrel=1e-12)
                assert error < 2e-12
                assert_allclose(a * b * value, reference, rtol=3e-12, atol=0)


@pytest.mark.parametrize("polarisation", ["x", "y"])
def test_rectangular_contour_derivative_and_material_chain(polarisation):
    mode = RectMode(50e-6, 12.5e-6, cladding=0.05 + 3.5j, polarisation=polarisation)
    omega = 2 * np.pi * C / np.linspace(300e-9, 2e-6, 7)
    terms = mode._terms(omega, 0.0)
    total = np.zeros_like(omega, dtype=complex)
    for phase in np.arange(64) * 2 * np.pi / 64:
        offset = 0.1 * omega * np.exp(1j * phase)
        value = mode._model_values(
            replace(terms, omega=omega + offset), 50e-6, 12.5e-6
        )[0]
        total = total + value / offset
    # Existing capillary contour tolerance: float rounding amplified by the
    # Cauchy radius, with exponentially small 64-node trapezoid truncation.
    assert_allclose(mode.dneff_domega(omega), total / 64, rtol=1e-11, atol=0)
    material = RectMode(50e-6, 12.5e-6, "Ar", 10.0, polarisation=polarisation)
    omega = 2 * np.pi * C / np.linspace(450e-9, 900e-9, 7)
    h = omega * 1e-3
    wide = (material.neff_minus_one(omega + h) - material.neff_minus_one(omega - h)) / (
        2 * h
    )
    narrow = (
        material.neff_minus_one(omega + h / 2) - material.neff_minus_one(omega - h / 2)
    ) / h
    # Richardson h^4 truncation and cancellation at gas refractivities ~1e-3.
    assert_allclose(
        material.dneff_domega(omega), (4 * narrow - wide) / 3, rtol=1e-8, atol=0
    )
    # The metal wall's own slope: dpermittivity_domega -> dwall -> dneff_domega,
    # which a constant complex wall (dwall = 0) and a silica wall leave
    # unexercised. Measured 4.67e-12 (x) and 4.33e-12 (y) relative, limited by
    # the C2 table spline's fourth derivative under the stencil.
    metal = RectMode(
        50e-6, 12.5e-6, "Ar", 10.0, cladding=silver(), polarisation=polarisation
    )
    wide = (metal.neff_minus_one(omega + h) - metal.neff_minus_one(omega - h)) / (2 * h)
    narrow = (
        metal.neff_minus_one(omega + h / 2) - metal.neff_minus_one(omega - h / 2)
    ) / h
    assert_allclose(
        metal.dneff_domega(omega), (4 * narrow - wide) / 3, rtol=2e-11, atol=0
    )


def test_rectangular_gram_and_non_degenerate_partners():
    # Luna's own set and counts (test_modal_quadrature.jl): a = 50 um,
    # b = 30 um, helium at one bar, a silica wall, (1,1), (2,1), (1,2), (3,2)
    # on 16 nodes per axis. The Gram matrix is a quadratic product, which 16
    # nodes resolve to 8.77e-15 here; the warning is the quartic Kerr bound of
    # cartesian_nodes(3) = 24, which this set is deliberately below.
    luna = RectMode(50e-6, 30e-6, "He", 1.0)
    modes = [replace(luna, n=n, m=m) for n, m in ((1, 1), (2, 1), (1, 2), (3, 2))]
    with pytest.warns(UserWarning, match="24 nodes per axis"):
        rule = transverse_quadrature(modes, n_r=16, n_theta=16)
    matrices = mode_matrix(modes, rule)
    assert_allclose(
        matrices.projection @ matrices.synthesis, np.eye(4), rtol=0, atol=1e-12
    )
    mode = RectMode(50e-6, 12.5e-6, "He", 1.0)
    pair = ModeCollection.with_polarisations([mode])
    assert len(pair) == 2
    assert pair[0].principal_component() == "x" and pair[1].principal_component() == "y"
    omega = np.asarray([2 * np.pi * C / 800e-9])
    # Measured 2.707284e-6 at 800 nm (n_x 0.9998959095351 + 2.5546e-6i,
    # n_y 0.9998959507144 + 5.2616e-6i); the bound is just below it so that a
    # future degeneracy regression cannot pass.
    assert abs(pair[0].neff(omega)[0] - pair[1].neff(omega)[0]) > 2e-6
    assert mode.orthogonal_polarisation.orthogonal_polarisation.polarisation == "x"
    assert mode.z_constant and mode.radius_is_constant and mode.scale_invariant
    assert mode.azimuthal_order == 1


def test_rectangular_tapers_and_parameter_freezing():
    path = Linear(50e-6, 25e-6, 0.1)
    one = RectMode(path, 12.5e-6)
    square = RectMode(path, path)
    assert not one.scale_invariant and square.scale_invariant
    assert not one.z_constant and not one.radius_is_constant
    for mode, factor in ((one, 0.5), (square, 0.25)):
        # Exact only because these widths are binary-exact: 25e-6 is half 50e-6
        # to the last bit and 0.5**2 is 0.25, so the ratios carry no rounding.
        # A general taper would need a rounding-level bound here.
        assert_allclose(mode.aeff(0.1) / mode.aeff(), factor, rtol=0, atol=0)
        assert_allclose(
            mode.normalisation(0.1) / mode.normalisation(), factor, rtol=0, atol=0
        )
        values = {
            name: float(p(np.asarray([0.1]))[0])
            for name, p in mode.parameter_paths.items()
        }
        frozen = mode.at_parameters(values)
        assert frozen.z_constant
        omega = np.asarray([2 * np.pi * C / 800e-9])
        assert_allclose(frozen.neff(omega), mode.neff(omega, 0.1), rtol=0, atol=0)
    assert one.at_parameters({}).half_width_y == one.half_width_y
    with pytest.raises(ModeError, match="unknown rectangular"):
        one.at_parameters({"radius": 1.0})
    with pytest.raises(ModeError, match="half-widths"):
        one.at_parameters({"half_width_y": -1.0})


def test_rectangular_fem_checked_values_from_luna_test_file():
    # Every input stated (``docs/dev/testing.md``); the wall is optmat's silver,
    # deliberately solprop's own data against Luna's FEM-checked numbers.
    mode = RectMode(
        50e-6,
        12.5e-6,
        n=1,
        m=1,
        polarisation="x",
        loss=True,
        cladding=silver(model="table/Ag"),
    )
    omega = 2 * np.pi * C / np.asarray([210e-9, 800e-9])
    # Luna test_rect_modes.jl: "checked with FEM using JCMwave". The original
    # 1% loss bounds survive the published metal data and filled-core convention.
    assert_allclose(
        db_per_m(mode, omega),
        [3.908368640803528, 0.44317059469197245],
        rtol=0.01,
        atol=0,
    )
    # The quoted beta precision is numerical, not an independent FEM error bar.
    # Dataset and interpolation changes move the real index by below 1e-9.
    assert_allclose(
        beta(mode, omega), [29919650.305427298, 7852915.234685494], rtol=1e-9, atol=0
    )
    assert_allclose(mode.aeff(), 1.111111111111111e-9, rtol=2e-15, atol=0)
    # Luna's own ZDW bound is 1 nm, on its own argon index (Börzsönyi's fit, supplied
    # explicitly) at its room temperature, 293.15 K, solprop's default: 563.76 nm.
    # optmat's argon index sits 1.09 nm longer, outside Luna's bound; with it the
    # comparison would measure the index data, not the guide.
    filled = replace(
        mode,
        gas="Ar",
        pressure=5.0,
        susceptibility=borzsonyi("Ar"),
        temperature=293.15,
    )
    with pytest.warns(MaterialRangeWarning, match="borzsonyi"):
        wavelength = zdw(filled, wavelength_range=(400e-9, 800e-9))
    assert abs(wavelength - 563.5e-9) < 1e-9


@pytest.mark.parametrize(
    "keywords,message",
    [
        ({"n": 0}, "indices"),
        ({"m": 0}, "indices"),
        ({"half_width_x": 0.0}, "half-widths"),
        ({"half_width_y": float("nan")}, "half-widths"),
        ({"loss": -1.0}, "loss scale"),
        ({"temperature": 0.0}, "temperature"),
    ],
)
def test_rectangular_constructor_refusals(keywords, message):
    values = dict(keywords)
    values.setdefault("half_width_x", 50e-6)
    values.setdefault("half_width_y", 12.5e-6)
    with pytest.raises(ModeError, match=message):
        RectMode(**values)


def test_rectangular_remaining_validation_and_mixture_paths():
    mode = RectMode(50e-6, 12.5e-6)
    object.__setattr__(mode, "polarisation", "invalid")
    with pytest.raises(ModeError, match="polarisation"):
        mode.__post_init__()
    mode = RectMode(50e-6, 12.5e-6)
    for values in ([], [0.0], [np.nan]):
        with pytest.raises(ModeError, match="omega"):
            mode.neff(np.asarray(values))
    with pytest.raises(ModeError, match="points"):
        mode.field(np.asarray([[np.nan, 0.0]]))
    # Qualify the production shape guard beyond the test session's type hook.
    with pytest.raises(ModeError, match="points"):
        inspect.unwrap(RectMode.field)(mode, np.zeros(2))
    assert_allclose(mode.field(np.asarray([[1.0, 1.0]])), 0.0, rtol=0, atol=0)
    with pytest.raises(ModeError, match="contrast"):
        RectMode(50e-6, 12.5e-6, cladding=1.0).neff(np.asarray([2e15]))
    terms = mode._terms(np.asarray([2e15]), 0.0)
    with pytest.raises(ModeError, match="non-finite"):
        mode._model_values(
            replace(terms, dwall=np.asarray([complex(np.nan)])), 50e-6, 12.5e-6
        )
    mixture = Mixture.fill({"He": 0.25, "Ar": 0.75}, 2.0)
    filled = mode.with_mixture(mixture)
    assert isinstance(filled, RectMode)
    assert np.all(np.isfinite(filled.neff(np.asarray([2e15]))))
    assert filled.at_parameters({"pressure": 1.0}).mixture is not mixture


def test_rectangular_luna_witness_at_matched_material_indices():
    witness = json.loads(
        (Path(__file__).parents[1] / "data/luna_rectangular.json").read_text()
    )
    assert len(witness["cases"]) == 240
    for case in witness["cases"]:
        mode = VacuumContrastMode(
            50e-6,
            12.5e-6,
            n=case["n"],
            m=case["m"],
            polarisation=case["polarisation"],
            cladding=complex(*case["wall"]),
            loss=True,
        )
        omega = np.asarray([2 * np.pi * C / case["wavelength"]])
        terms = replace(
            mode._terms(omega, 0.0),
            chi=np.asarray([complex(*case["core"]) ** 2 - 1]),
        )
        value = 1 + mode._model_values(terms, 50e-6, 12.5e-6)[0]
        # Identical material samples isolate the algebra from gas and metal data;
        # expm1/log1p versus sqrt differs by a few rounded complex operations.
        assert_allclose(value, complex(*case["neff"]), rtol=1e-15, atol=0)


def test_rectangular_filled_contrast_size_and_cartesian_resolution():
    mode = RectMode(50e-6, 12.5e-6, "Ar", 10.0, cladding=1.45)
    vacuum = VacuumContrastMode(50e-6, 12.5e-6, "Ar", 10.0, cladding=1.45)
    omega = np.asarray([2 * np.pi * C / 800e-9])
    change = np.imag(mode.neff(omega)) / np.imag(vacuum.neff(omega)) - 1
    # O(chi/(epsilon_wall-1)) predicts a few parts per thousand here. This
    # range guards the convention, not a claimed physical-model error bar.
    assert 0.002 < change[0] < 0.003
    mode = RectMode(50e-6, 12.5e-6, n=3, m=2)
    errors = []
    for count in (20, 22, 24):
        # Anything below cartesian_nodes(3) = 24 warns, which is the point of
        # the first two rows; 22 is the measured threshold the bound rounds up.
        context = (
            nullcontext()
            if count >= cartesian_nodes(3)
            else pytest.warns(UserWarning, match="24 nodes per axis")
        )
        with context:
            rule = transverse_quadrature([mode], n_r=count, n_theta=count)
        intensity = np.sum(mode.field(rule.nodes(mode.dimlimits())) ** 2, axis=1)
        integral = rule.weights(mode.dimlimits()) @ intensity**2
        errors.append(abs(integral / (9 * 50e-6 * 12.5e-6 / 16) - 1))
    # Fixed requested 1e-12 quartic accuracy: twenty nodes are insufficient,
    # twenty-two are the true threshold (1.99e-13) and twenty-four resolve the
    # trigonometric product to reduction rounding.
    assert errors[0] > 1e-12
    assert errors[1] < 1e-12
    assert errors[2] < 1e-12


def test_rectangular_loss_floor_gain_and_refractivity(caplog):
    mode = RectMode(50e-6, 12.5e-6)
    omega = np.asarray([2 * np.pi * C / 800e-9])
    base = mode.neff(omega)
    for specification, factor in ((False, 0.0), (0.25, 0.25)):
        changed = replace(mode, loss=specification)
        assert_allclose(np.real(changed.neff(omega)), np.real(base), rtol=0, atol=0)
        assert_allclose(
            np.imag(changed.neff(omega)), factor * np.imag(base), rtol=0, atol=0
        )
    absolute = replace(mode, loss=lambda values: np.full_like(values, 2.0))
    assert "no 'derivative'" in caplog.text
    assert_allclose(np.imag(absolute.neff(omega)), C / omega, rtol=0, atol=0)
    assert_allclose(
        np.imag(absolute.dneff_domega(omega)), -C / omega**2, rtol=2e-15, atol=0
    )  # Constant loss has an exact zero slope; only quotient rounding remains.
    far = np.asarray([2 * np.pi * C / 50e-6])
    # A 2 um guide at 50 um is far outside the first-order model, which is the point:
    # the floor and the clamp are what is left there, and the mode says so once.
    floor = RectMode(1e-6, 1e-6, cladding=0.5)
    with pytest.warns(GrazingIncidenceWarning, match="first-order wall-pair"):
        assert_allclose(np.real(floor.neff(far)), 1e-3, rtol=1e-12, atol=0)
    assert_allclose(floor.dneff_domega(far), 0, rtol=0, atol=0)
    gain = RectMode(1e-6, 1e-6, cladding=0.05 + 0.01j)
    with pytest.warns(GrazingIncidenceWarning, match="first-order wall-pair"):
        assert_allclose(np.imag(gain.neff(far)), 0, rtol=0, atol=0)
    assert_allclose(np.imag(gain.dneff_domega(far)), 0, rtol=0, atol=0)
    assert "clamped at zero" in caplog.text


def test_rectangular_refractivity_is_cancellation_free_against_mpmath():
    # The large-core limit is what the rationalised root exists for: with a
    # 10 cm guide the argument is -8e-12, where sqrt(1 + z) - 1 and (on this
    # NumPy build, whose complex log1p forms 1 + z) expm1(log1p(z)/2) both keep
    # about five digits. mpmath at 50 digits is the independent reference; the
    # model is re-evaluated there from the wall permittivity so that neither the
    # argument nor its root comes from production.
    half_width = 0.1
    omega = np.asarray([2 * np.pi * C / 800e-9])
    value = complex(
        RectMode(half_width, half_width, cladding=1.45).neff_minus_one(omega)[0]
    )
    with mp.workdps(50):
        frequency = mp.mpf(float(omega[0]))
        wall = mp.mpf("1.45") ** 2
        root = mp.sqrt(wall - 1)  # An evacuated core: chi is exactly zero.
        decrement = mp.mpf(0)
        for coefficient in (wall / root, 1 / root):  # TM-like x, TE-like y.
            scale = mp.mpf(C) / (frequency * mp.mpf(half_width))
            decrement += (mp.pi * scale / 2) ** 2 * (1 - 1j * scale * coefficient) ** 2
        argument = -decrement
        reference = complex(mp.sqrt(1 + argument) - 1)
        double = complex(argument)
    # Measured 2.02e-16 relative, against 5.63e-6 for either naive spelling,
    # which the bound below therefore sees.
    assert abs(value - reference) < 4 * EPS * abs(reference)
    naive = np.sqrt(1 + double) - 1
    assert abs(naive - reference) > 1e-8 * abs(reference)
    assert abs(np.expm1(0.5 * np.log1p(double)) - reference) > 1e-8 * abs(reference)


def test_rectangular_rejects_competing_mixture_declaration():
    with pytest.raises(ModeError, match="mixture carries"):
        RectMode(50e-6, 12.5e-6, "Ar", mixture=Mixture.fill({"He": 1.0}, 1.0))


def test_a_rectangular_guide_past_its_first_order_ratios_warns_once():
    """n lambda / 4b = 0.2 at 800 nm on a 1 um half-width in y.

    Both wall pairs are checked and the worst named; a second evaluation is silent.
    """
    mode = RectMode(50e-6, 1e-6, cladding=1.45)
    omega = np.array([2 * math.pi * C / 800e-9])
    with pytest.warns(GrazingIncidenceWarning, match=r"p lambda / 4a = 0\.2 at 800"):
        mode.neff(omega)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GrazingIncidenceWarning)
        mode.neff(omega)


def test_a_rectangular_guide_within_its_first_order_ratios_is_silent():
    """On 50 by 12.5 um both pairs' ratios stay below 0.1 from 400 nm to 2 um."""
    omega = 2 * math.pi * C / np.linspace(400e-9, 2e-6, 64)
    for polarisation in ("x", "y"):
        mode = RectMode(50e-6, 12.5e-6, cladding=1.45, polarisation=polarisation)
        with warnings.catch_warnings():
            warnings.simplefilter("error", GrazingIncidenceWarning)
            mode.neff(omega)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the table contract and the two fixed-geometry hollow modes."""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bounds import assert_within
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import IndexTable, MarcatiliMode, TabulatedHollowMode, TabulatedMode
from solprop.output.schema import serialise
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.mixtures import Mixture
from solprop.profiles import Constant, Linear, PressureGradient

EPS = np.finfo(float).eps


def simple_table(*, area=None, policy: Literal["log", "linear"] = "log"):
    return IndexTable(
        [1e15, 2e15, 3e15, 4e15],
        [1.1 + 1e-8j] * 4,
        area=area,
        loss_interpolation=policy,
    )


@given(st.integers(4, 60), st.booleans())
def test_sorted_inputs_are_canonical(count, reverse):
    omega = np.linspace(1e15, 4e15, count)
    values = 1.0 + omega * 1e-18 + 1j * omega * 1e-23
    area = omega * 1e-25
    order = slice(None, None, -1 if reverse else 1)
    table = IndexTable(omega[order], values[order], area=area[order])
    np.testing.assert_array_equal(table.omega, omega)
    np.testing.assert_array_equal(table.neff_values, values)
    np.testing.assert_array_equal(table.area, area)
    assert serialise(table) == serialise(IndexTable(omega, values, area))
    wavelength = IndexTable.from_wavelength(
        2 * np.pi * C / omega[order], values[order], area=area[order]
    )
    # One reciprocal conversion loses at most two float64 rounding errors.
    np.testing.assert_allclose(wavelength.omega, omega, rtol=2 * EPS, atol=0)


def test_arrays_are_owned_and_readonly():
    omega = np.linspace(1e15, 4e15, 4)
    index = np.full(4, 1.1 + 1e-8j)
    area = np.full(4, 1e-10)
    table = IndexTable(omega, index, area)
    for source, stored in zip(
        (omega, index, area), (table.omega, table.neff_values, table.area), strict=True
    ):
        assert stored is not None
        assert source.flags.writeable
        assert not stored.flags.writeable
        assert not np.shares_memory(source, stored)


@pytest.mark.parametrize("policy", ["linear", "log"])
def test_table_knots_loss_conversion_and_accessors(policy):
    wavelength = np.linspace(400e-9, 1000e-9, 12)
    loss = np.linspace(0.01, 1.0, 12)
    table = IndexTable.from_loss(
        wavelength,
        np.full(12, 1.45),
        loss,
        area=np.full(12, 1e-10),
        loss_interpolation=policy,
    )
    omega = 2 * np.pi * C / wavelength
    # Conversions and exp(log(x)) have a short chain of rounded operations.
    np.testing.assert_allclose(
        table.alpha(omega), loss * np.log(10) / 10, rtol=32 * EPS, atol=0
    )
    np.testing.assert_allclose(
        table.neff(table.omega), table.neff_values, rtol=4 * EPS, atol=0
    )
    np.testing.assert_allclose(
        table.beta(omega), omega * 1.45 / C, rtol=4 * EPS, atol=0
    )
    np.testing.assert_allclose(
        table.dbeta_domega(omega), 1.45 / C, rtol=4 * EPS, atol=0
    )
    np.testing.assert_allclose(table.effective_area(omega), 1e-10, rtol=4 * EPS, atol=0)
    assert "12 rows" in repr(table)


@pytest.mark.parametrize("policy", ["log", "linear"])
def test_derivative_matches_richardson_between_knots(policy):
    omega = np.linspace(1e15, 4e15, 20)
    index = 1.2 + 0.02 * np.sin(omega / 1e15) + 1j * np.exp(omega / 1e15) * 1e-5
    table = IndexTable(omega, index, loss_interpolation=policy)
    query = (omega[2:-3] + 2 * omega[3:-2]) / 3
    step = query * 1e-3
    wide = (table.neff(query + step) - table.neff(query - step)) / (2 * step)
    half = (table.neff(query + step / 2) - table.neff(query - step / 2)) / step
    difference = (4 * half - wide) / 3
    # Richardson cancels the cubic truncation; remaining log-loss h^4 and
    # float64 subtraction error are below the 1e-9 relative target.
    np.testing.assert_allclose(table.dneff_domega(query), difference, rtol=1e-9, atol=0)


@pytest.mark.parametrize(
    "omega,index,area,policy,message",
    [
        ([1, 2, 3], [1j] * 3, None, "log", "at least 4"),
        ([1, 2, 2, 4], [1j] * 4, None, "log", "duplicate"),
        ([0, 2, 3, 4], [1j] * 4, None, "log", "positive finite"),
        ([1, 2, 3, np.inf], [1j] * 4, None, "log", "positive finite"),
        ([[1, 2], [3, 4]], [1j] * 4, None, "log", "one-dimensional"),
        ([1, 2, 3, 4], [1j] * 3, None, "log", "same shape"),
        ([1, 2, 3, 4], [1j, 1j, np.nan, 1j], None, "log", "must be finite"),
        ([1, 2, 3, 4], [-1j] * 4, None, "linear", "gain"),
        ([1, 2, 3, 4], [1j, 0j, 1j, 0j], None, "log", r"rows \[1, 3\].*linear"),
        ([1, 2, 3, 4], [1j] * 4, [-1] * 4, "log", "area.*positive finite"),
        ([1, 2, 3, 4], [1j] * 4, [np.nan] * 4, "log", "area.*positive finite"),
        ([1, 2, 3, 4], [1j] * 4, [1] * 3, "log", "omega and area"),
    ],
)
def test_invalid_table_is_refused(omega, index, area, policy, message):
    with pytest.raises(ModeError, match=message):
        IndexTable(omega, index, area, policy)


@pytest.mark.parametrize(
    "index,loss,message",
    [
        ([1] * 3, [1] * 4, "same shape"),
        ([1] * 4, [1] * 3, "same shape"),
        ([1j] * 4, [1] * 4, "real arrays"),
        ([1] * 4, [1j] * 4, "real arrays"),
        ([1] * 4, [-1] * 4, "non-negative"),
        ([1] * 4, [np.inf] * 4, "finite"),
    ],
)
def test_invalid_loss_columns_are_refused(index, loss, message):
    with pytest.raises(ModeError, match=message):
        IndexTable.from_loss([1e-6, 2e-6, 3e-6, 4e-6], index, loss)


def test_zero_loss_empty_queries_and_support_errors():
    table = IndexTable([1e15, 2e15, 3e15, 4e15], [1.0] * 4, loss_interpolation="linear")
    np.testing.assert_array_equal(table.alpha(table.omega), np.zeros(4))
    assert table.neff(np.array([])).size == 0
    with pytest.raises(ModeError, match="no effective-area column"):
        table.effective_area([2e15])
    for values in ([0.0], [np.nan], [[2e15]]):
        with pytest.raises(ModeError, match="positive finite"):
            table.neff(values)
    for values in ([0.9e15], [4.1e15]):
        with pytest.raises(
            ModeError,
            match=r"requested.*rad/s.*nm.*outside table.*rad/s.*nm.*wavelength_limits",
        ):
            table.neff(values)


@pytest.mark.parametrize("cls", [TabulatedMode, TabulatedHollowMode])
def test_table_modes_keep_the_capillary_zero_frequency_message(cls):
    # Handing the whole grid instead of its active band is the likely user
    # error on this path; the parent's diagnostic names the DC bin.
    mode = cls(15e-6, table=simple_table())
    with pytest.raises(ModeError, match="no mode at zero frequency"):
        mode.neff(np.array([0.0, 2e15]))


@pytest.mark.parametrize("end", [0, 1])
@pytest.mark.parametrize("cls", [TabulatedMode, TabulatedHollowMode])
def test_reference_wavelength_at_a_table_end_constructs_and_evaluates(cls, end):
    # 2 pi C / (2 pi C / omega) is not omega for about five per cent of
    # doubles, so an end wavelength converts a few ulp outside the table.
    table = simple_table(area=np.full(4, 1e-10))
    mode = cls(15e-6, table=table, reference_wavelength=table.wavelength_range[end])
    assert mode.aeff() == pytest.approx(1e-10, rel=4 * EPS, abs=0)


def test_linear_loss_clamp_is_logged_once_and_area_is_not_clamped(caplog):
    table = IndexTable(
        [1, 2, 3, 4],
        1 + 1j * np.array([100.0, 1.0, 1.0, 1.0]),
        loss_interpolation="linear",
    )
    record = serialise(table)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            value, derivative = table.refractivity([2.5])
            assert np.imag(value)[0] == np.imag(derivative)[0] == 0
    assert len(caplog.records) == 1
    assert "clamped" in caplog.text
    assert serialise(table) == record
    with_area = IndexTable([1, 2, 3, 4], [1j] * 4, [100.0, 1.0, 1.0, 1.0])
    with pytest.raises(ModeError, match="interpolated area must be positive"):
        with_area.effective_area([2.5])


@dataclass(frozen=True)
class AbsoluteLoss:
    def __call__(self, omega):
        return omega * 1e-15

    def derivative(self, omega):
        return np.full_like(omega, 1e-15)


def marcatili_errors(count, *, kind="HE", loss=True, gradient=False, mixture=False):
    omega = 2 * np.pi * C / np.linspace(300e-9, 2e-6, count)
    family = {"kind": kind, "n": 1 if kind == "HE" else 0}
    vacuum = MarcatiliMode(15e-6, cladding=1.45, **family)
    table = IndexTable(omega, vacuum.neff(omega))
    pressure = PressureGradient.two_point(1.0, 10.0, 3.0) if gradient else 10.0
    if mixture:
        # Two populations sum their susceptibilities before the parent's
        # rationalised root; the rule must be exact on that path as well.
        populations = Mixture.fill({"Ar": 0.8, "Ne": 0.2}, pressure=pressure)
        model = TabulatedHollowMode(
            15e-6, mixture=populations, table=table, loss=loss, **family
        )
        exact = MarcatiliMode(
            15e-6, mixture=populations, cladding=1.45, loss=loss, **family
        )
    else:
        model = TabulatedHollowMode(
            15e-6, "Ar", pressure, table=table, loss=loss, **family
        )
        exact = MarcatiliMode(15e-6, "Ar", pressure, cladding=1.45, loss=loss, **family)
    query = 2 * np.pi * C / np.linspace(500e-9, 1500e-9, 501)
    errors = []
    # optmat's argon is qualified over the mandated 2 um window (SEP 3, Phase 4a),
    # so no range diagnostic is raised, and one would fail the test; both sides use
    # the identical gas expression.
    with nullcontext():
        for z in (0.0, 0.5, 1.0) if gradient else (0.0,):
            # At the knots the spline returns the tabulated value to rounding,
            # so the two routes differ only by the roundings of the complex
            # assembly n_eff**2 = n_v**2 + chi_g: allow four eps, no more.
            np.testing.assert_allclose(
                model.neff(omega, z), exact.neff(omega, z), rtol=4 * EPS, atol=0
            )
            errors.append(
                [
                    float(np.max(abs(model.neff(query, z) - exact.neff(query, z)))),
                    float(
                        np.max(
                            abs(
                                model.dneff_domega(query, z)
                                - exact.dneff_domega(query, z)
                            )
                        )
                    ),
                ]
            )
            fixed = model.at_parameters(
                {
                    name: float(path(np.array([z]))[0])
                    for name, path in model.parameter_paths.items()
                }
            )
            # The same evaluation at the sampled parameters: the one-evaluation
            # budget of 16 eps (accuracy.md section 7), normwise.
            assert_within(fixed.neff(query), model.neff(query, z), 16 * EPS)
    assert model.radius_is_constant
    varying = "pressure" if mixture else "density"
    assert set(model.parameter_paths) == {"core_radius", varying}
    return np.max(errors, axis=0)


@pytest.mark.parametrize("kind", ["HE", "TE", "TM"])
@pytest.mark.parametrize("loss", [True, False, 2.0, AbsoluteLoss()])
@pytest.mark.parametrize("gradient", [False, True])
def test_marcatili_exactness_and_convergence(kind, loss, gradient):
    errors = np.array(
        [
            marcatili_errors(n, kind=kind, loss=loss, gradient=gradient)
            for n in (50, 100, 200)
        ]
    )
    ratios = errors[:-1] / errors[1:]
    # Cubic interpolation is h^4 for values, h^3 for slopes, with
    # a 20% allowance for the finite 50/100/200-knot asymptotic ladder.
    np.testing.assert_allclose(
        ratios, np.broadcast_to([16.0, 8.0], ratios.shape), rtol=0.2, atol=0
    )


def test_marcatili_exactness_and_convergence_on_a_mixture():
    # Two populations sum their susceptibilities before the parent's root; the
    # rule has to be exact on that path as well as on a single gas's.
    errors = np.array([marcatili_errors(n, mixture=True) for n in (50, 100, 200)])
    ratios = errors[:-1] / errors[1:]
    np.testing.assert_allclose(
        ratios, np.broadcast_to([16.0, 8.0], ratios.shape), rtol=0.2, atol=0
    )


@pytest.mark.parametrize("cls", [TabulatedMode, TabulatedHollowMode])
def test_mode_area_profiles_and_polarisation(cls):
    table = simple_table(area=np.full(4, 1e-10))
    mode = cls(15e-6, table=table, reference_wavelength=800e-9)
    assert mode.aeff() == pytest.approx(
        1e-10, rel=4 * EPS, abs=0
    )  # Constant cubic, rounding only.
    # The mode reads the table's area; its scaling may cost a few roundings.
    np.testing.assert_allclose(
        mode.effective_area(np.array([2e15])),
        table.effective_area([2e15]),
        rtol=4 * EPS,
        atol=0,
    )
    assert mode.orthogonal_polarisation.table is table
    assert mode.rotated(0.4).reference_wavelength == 800e-9
    assert mode.z_constant
    assert cls.__name__ in repr(mode)
    no_area = cls(15e-6, table=simple_table())
    assert no_area.aeff() == MarcatiliMode(15e-6).aeff()
    with pytest.raises(ModeError, match="no effective-area column"):
        no_area.effective_area(np.array([2e15]))


@pytest.mark.parametrize(
    "keywords,message",
    [
        ({"model": "reduced"}, 'model="reduced"'),
        ({"radius": Linear(15e-6, 16e-6, 1.0)}, "tapered radius"),
        ({"cladding": 1.45}, "table carries the wall"),
        ({"reference_wavelength": 800e-9}, "area.*say which"),
        ({"table": simple_table(area=[1.0] * 4)}, "area.*say which"),
        (
            {"table": simple_table(area=[1.0] * 4), "reference_wavelength": 10e-6},
            "reference_wavelength.*inside",
        ),
        (
            {"table": simple_table(area=[1.0] * 4), "reference_wavelength": np.nan},
            "reference_wavelength.*inside",
        ),
    ],
)
def test_mode_refusals(keywords, message):
    for cls in (TabulatedMode, TabulatedHollowMode):
        with pytest.raises(ModeError, match=message):
            cls(**({"radius": 15e-6, "table": simple_table()} | keywords))


def test_filled_mode_refuses_varying_material_and_has_no_attribution():
    table = simple_table()
    with pytest.raises(
        ModeError, match="table holds one fill; a gradient needs TabulatedHollowMode"
    ):
        TabulatedMode(15e-6, "Ar", Linear(1.0, 2.0, 1.0), table=table)
    mixture = Mixture.fill({"Ar": 0.8, "Ne": 0.2}, pressure=Linear(1.0, 2.0, 1.0))
    with pytest.raises(ModeError, match="table holds one fill"):
        TabulatedMode(15e-6, mixture=mixture, table=table)
    mode = TabulatedMode(15e-6, "Ar", Constant(10.0), table=table)
    np.testing.assert_array_equal(mode.absorption_by_species(np.array([2e15])), [[0.0]])
    mixed = TabulatedMode(
        15e-6, mixture=Mixture.fill({"Ar": 0.8, "Ne": 0.2}, pressure=1.0), table=table
    )
    np.testing.assert_array_equal(
        mixed.absorption_by_species(np.array([2e15])), np.zeros((2, 1))
    )
    assert "Mixture" in repr(mixed)


def test_table_modes_never_evaluate_silica(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a table mode evaluated its unused cladding")

    monkeypatch.setattr(GlassIndex, "permittivity", forbidden)
    for cls in (TabulatedHollowMode, TabulatedMode):
        mode = cls(15e-6, table=simple_table())
        assert np.isfinite(mode.neff(np.array([2e15]))).all()


def test_invalid_policy_reaches_production_validation():
    # Match the capillary literal-refusal tests: production has no type hook.
    table = simple_table()
    object.__setattr__(table, "loss_interpolation", "wrong")
    with pytest.raises(ModeError, match=r"log.*linear"):
        table.__post_init__()

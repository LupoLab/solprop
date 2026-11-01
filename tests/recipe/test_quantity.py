# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Every comparison here is exact (class P, a structural contract): a quantity string
# converts to the one double nearest its exact value, which is the double of the literal
# a script writes, and writing a double back reads back to the same double.

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from solprop.exceptions import SolpropError
from solprop.recipe import _quantity
from solprop.recipe._errors import RecipeError
from solprop.recipe._quantity import (
    Dimension,
    Unit,
    beta_unit,
    format_quantity,
    parse_quantity,
    phase_unit,
)

METRE = Unit("m")
SECOND = Unit("s")
JOULE = Unit("J")
BAR = Unit("bar")
KELVIN = Unit("K")
RADIAN = Unit("rad")

PREFIXES = {
    "q": -30,
    "r": -27,
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "µ": -6,
    "μ": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
    "R": 27,
    "Q": 30,
}


def same_double(a, b):
    """Whether two doubles are identical, the sign of a zero included."""
    return a == b and math.copysign(1, a) == math.copysign(1, b)


# --- reading ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "unit", "expected"),
    [
        # The values of the format's example recipe, as a script writes them.
        ("125 um", METRE, 125e-6),
        ("1 m", METRE, 1.0),
        ("0 bar", BAR, 0.0),
        ("1.2 bar", BAR, 1.2),
        ("293.15 K", KELVIN, 293.15),
        ("800 nm", METRE, 800e-9),
        ("7.5 fs", SECOND, 7.5e-15),
        ("275 uJ", JOULE, 275e-6),
        ("100 nm", METRE, 100e-9),
        ("1500 nm", METRE, 1500e-9),
        ("200 fs", SECOND, 200e-15),
    ],
)
def test_parse_quantity_example_values_give_the_script_literal(text, unit, expected):
    assert same_double(parse_quantity(text, unit), expected)


@pytest.mark.parametrize(("prefix", "exponent"), PREFIXES.items())
def test_parse_quantity_every_prefix_shifts_the_exponent_exactly(prefix, exponent):
    # float() of the decimal literal is what a script's literal 1.5e<exponent> gives.
    assert same_double(
        parse_quantity(f"1.5 {prefix}m", METRE), float(f"1.5e{exponent}")
    )
    assert same_double(
        parse_quantity(f"2.25 {prefix}s", SECOND), float(f"2.25e{exponent}")
    )


@pytest.mark.parametrize(
    ("text", "unit", "expected"),
    [
        ("3 m", METRE, 3.0),
        ("3 s", SECOND, 3.0),
        ("5 g", Unit("kg"), 5e-3),
        ("5 kg", Unit("kg"), 5.0),
        ("2.5 THz", Unit("Hz"), 2.5e12),
        ("2 mJ", JOULE, 2e-3),
        ("1.5 GW", Unit("W"), 1.5e9),
        ("101325 Pa", Unit("Pa"), 101325.0),
        ("1.5 bar", Unit("Pa"), 150000.0),
        ("120 kPa", BAR, 1.2),
        ("250 mbar", BAR, 0.25),
        ("77 K", KELVIN, 77.0),
        ("1.5 mrad", RADIAN, 1.5e-3),
        ("1 J/(m*s)", Unit("W/m"), 1.0),
        ("1 J·s^-1", Unit("W"), 1.0),
        ("1 J*s^-1", Unit("W"), 1.0),
        ("2e14 W/cm^2", Unit("W/m^2"), 2e18),
        ("100 fs^2/mm", Unit("s^2/m"), 1e-25),
        ("-20 fs^2/mm", Unit("s^2/m"), -2e-26),
        ("30 fs^3/mm", Unit("s^3/m"), 3e-41),
        ("1.2 1/mm", Unit("1/m"), 1200.0),
        ("5 rad/fs", Unit("rad/s"), 5e15),
    ],
)
def test_parse_quantity_each_unit_converts_to_the_literal(text, unit, expected):
    assert same_double(parse_quantity(text, unit), expected)


@pytest.mark.parametrize("text", ["5 um", "5 µm", "5 μm", "5um", "  5   um  "])
def test_parse_quantity_micro_spellings_and_spacing_agree(text):
    assert same_double(parse_quantity(text, METRE), 5e-6)


@pytest.mark.parametrize(
    "text", ["1.5e-3 m", "1.5E-3 m", "+1.5e-3 m", ".0015 m", "0.0015m", "1.5 mm"]
)
def test_parse_quantity_number_forms(text):
    assert same_double(parse_quantity(text, METRE), 1.5e-3)


def test_parse_quantity_keeps_the_sign_of_zero():
    assert same_double(parse_quantity("-0 nm", METRE), -0.0)
    assert same_double(parse_quantity("0 nm", METRE), 0.0)


@pytest.mark.parametrize(
    ("value", "expected"), [(1.2, 1.2), (3, 3.0), (-0.0, -0.0), (1e-300, 1e-300)]
)
def test_parse_quantity_bare_number_is_in_the_keyword_unit(value, expected):
    result = parse_quantity(value, BAR)
    assert type(result) is float
    assert same_double(result, expected)


def test_parse_quantity_degrees_round_once():
    # pi/180 has no finite decimal; the exact product is rounded once, so 180 deg is
    # the double nearest pi and 90 deg half of it, where math.radians rounds twice.
    assert parse_quantity("180 deg", RADIAN) == math.pi
    assert parse_quantity("90 deg", RADIAN) == math.pi / 2
    assert parse_quantity("-45 deg", RADIAN) == -math.pi / 4


def test_parse_quantity_dimensionless_string_without_unit():
    assert parse_quantity("0.25", Unit("1")) == 0.25
    assert parse_quantity(2, Unit("1")) == 2.0


# --- refusals ---------------------------------------------------------------------


def test_parse_quantity_wrong_dimension_names_both():
    with pytest.raises(RecipeError, match=r"length \(m\).*pressure") as caught:
        parse_quantity("125 um", BAR, path="pressure")
    assert caught.value.path == "pressure"
    assert str(caught.value).startswith("pressure: ")


@pytest.mark.parametrize(
    ("text", "unit"),
    [
        ("1 Hz", Unit("rad/s")),  # a factor of 2 pi apart: angle is a dimension
        ("1 rad", Unit("1")),
        ("1 s^2", Unit("s^2/m")),
        ("1 J", Unit("W")),
    ],
)
def test_parse_quantity_refuses_other_dimensions(text, unit):
    with pytest.raises(RecipeError, match="but the value is"):
        parse_quantity(text, unit)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("800", "no unit, which is ambiguous"),
        ("1 W/m/s", "two solidi"),
        ("1 J/m*s", "product after a solidus is ambiguous"),
        ("1 m s", r"separate the factors with \*"),
        ("1 (m)", "only a denominator"),
        ("1 W/(m", "parentheses do not match"),
        ("1 W/((m))", "parentheses do not match"),
    ],
)
def test_parse_quantity_refuses_ambiguous_strings(text, match):
    with pytest.raises(RecipeError, match=match):
        parse_quantity(text, Unit("m"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("abc", "not a quantity"),
        ("", "not a quantity"),
        ("1.2.3 m", "separate the factors"),
        ("nan m", "not a quantity"),
        ("inf m", "not a quantity"),
        ("1 xyz", "unknown unit 'xyz'"),
        ("1 metre", "unknown unit 'metre'"),
        ("1 mm^", "not a unit factor"),
        ("1 m^0", "not a unit factor"),
        ("1 m^1.5", "not a unit factor"),
        ("1 m*", "not a unit factor"),
        ("1 m²", "unknown unit 'm²'"),
        ("1 /m", "unknown unit|not a unit factor"),
        ("1 mdeg", "did you mean 'deg'"),
        ("1 deg^2", "deg stands alone"),
        ("1 deg/s", "deg stands alone"),
    ],
)
def test_parse_quantity_refuses_unreadable_strings(text, match):
    with pytest.raises(RecipeError, match=match):
        parse_quantity(text, RADIAN if "deg" in text else METRE)


@pytest.mark.parametrize("text", ["20 degC", "20 °C", "20 C", "68 degF", "68 F", "5 ℃"])
def test_parse_quantity_temperature_is_kelvin_only(text):
    with pytest.raises(RecipeError, match="temperature is in K only"):
        parse_quantity(text, KELVIN)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "boolean"),
        (float("nan"), "not a finite number"),
        (float("inf"), "not a finite number"),
        (10**400, "too large"),
        ([1.0], "not list"),
        (None, "not NoneType"),
        ("1e400 m", "too large"),
        ("1e-400 m", "too small"),
        ("1e999999999999999999999 m", "out of range"),
        ("1e-999999999999999999999 m", "out of range"),
    ],
)
def test_parse_quantity_refuses_values_that_are_not_finite_quantities(value, match):
    with pytest.raises(RecipeError, match=match):
        parse_quantity(value, METRE)


def test_parse_quantity_refuses_a_unit_with_two_readings(monkeypatch):
    # No symbol in the table has two readings yet; adding the year, ``a``, would make
    # ``Pa`` both the pascal and the peta-year, which must be refused, not chosen.
    monkeypatch.setitem(_quantity._SYMBOLS, "a", (Dimension(time=1), 0))
    with pytest.raises(RecipeError, match="'Pa' has more than one reading"):
        parse_quantity("1 Pa", Unit("Pa"))


def test_recipe_error_is_a_solprop_value_error():
    with pytest.raises(SolpropError):
        parse_quantity("1 xyz", METRE)
    with pytest.raises(ValueError, match="unknown unit"):
        parse_quantity("1 xyz", METRE)


# --- units --------------------------------------------------------------------------


def test_unit_refuses_degrees_and_unreadable_text():
    with pytest.raises(RecipeError, match="decimal multiple of an SI unit"):
        Unit("deg")
    with pytest.raises(RecipeError, match="unknown unit"):
        Unit("furlong")
    with pytest.raises(RecipeError, match="empty"):
        Unit(" ")


def test_unit_dimension_and_equality():
    assert Unit("bar").dimension == Unit("Pa").dimension
    assert Unit("W/cm^2").dimension == Dimension(time=-3, mass=1)
    assert Unit("1").dimension == Dimension()
    assert Unit("m") == Unit("m")
    assert Unit("bar") != Unit("Pa")


@pytest.mark.parametrize(
    ("dimension", "text"),
    [
        (Dimension(), "dimensionless"),
        (Dimension(length=1), "length (m)"),
        (Dimension(time=-1, angle=1), "angular frequency (s^-1 rad)"),
        (Dimension(length=-1, time=2), "m^-1 s^2"),
        (Dimension(temperature=1), "temperature (K)"),
    ],
)
def test_dimension_text(dimension, text):
    assert str(dimension) == text


def test_dimension_arithmetic():
    length = Dimension(length=1)
    assert length * length == Dimension(length=2)
    assert length**-2 == Dimension(length=-2)


def test_element_units():
    assert [phase_unit(n).text for n in range(4)] == ["rad", "s", "s^2", "s^3"]
    assert [beta_unit(n).text for n in range(4)] == ["1/m", "s/m", "s^2/m", "s^3/m"]
    assert beta_unit(2).dimension == Dimension(length=-1, time=2)
    assert same_double(parse_quantity("-20 fs^2/mm", beta_unit(2)), -2e-26)
    for element_unit in (phase_unit, beta_unit):
        with pytest.raises(RecipeError, match="at least 0"):
            element_unit(-1)


# --- writing back -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "display", "text"),
    [
        (125e-6, METRE, "um", "125 um"),
        (1.0, METRE, "m", "1 m"),
        (800e-9, METRE, "nm", "800 nm"),
        (7.5e-15, SECOND, "fs", "7.5 fs"),
        (275e-6, JOULE, "uJ", "275 uJ"),
        (1.2, BAR, "bar", "1.2 bar"),
        (1.2, BAR, "kPa", "120 kPa"),
        (0.0, BAR, "bar", "0 bar"),
        (-0.0, METRE, "um", "-0 um"),
        (293.15, KELVIN, "K", "293.15 K"),
        (0.1, METRE, "mm", "100 mm"),
        (2e18, Unit("W/m^2"), "W/cm^2", "200000000000000 W/cm^2"),
        (1e-25, Unit("s^2/m"), "fs^2/mm", "100 fs^2/mm"),
        (1e300, METRE, "qm", "1e330 qm"),
        (5e-324, METRE, "Qm", "5e-354 Qm"),
        (1.5e22, METRE, "m", "1.5e22 m"),
        (1.25e-5, METRE, "m", "1.25e-5 m"),
        (math.pi, RADIAN, "deg", "180 deg"),
    ],
)
def test_format_quantity_writes_the_shortest_text(value, unit, display, text):
    assert format_quantity(value, unit, display) == text
    assert same_double(parse_quantity(text, unit), value)


def test_format_quantity_refuses_other_dimensions_and_non_finite_values():
    with pytest.raises(RecipeError, match="cannot write a length"):
        format_quantity(1.0, METRE, "fs")
    with pytest.raises(RecipeError, match="not a finite number"):
        format_quantity(float("inf"), METRE, "m")
    with pytest.raises(RecipeError, match="unknown unit"):
        format_quantity(1.0, METRE, "furlong")


ROUND_TRIPS = [
    (METRE, "um"),
    (METRE, "nm"),
    (METRE, "qm"),
    (METRE, "Qm"),
    (SECOND, "fs"),
    (JOULE, "uJ"),
    (BAR, "bar"),
    (BAR, "kPa"),
    (Unit("Pa"), "mbar"),
    (KELVIN, "K"),
    (Unit("s^2/m"), "fs^2/mm"),
    (Unit("W/m^2"), "W/cm^2"),
    (RADIAN, "deg"),
    (RADIAN, "mrad"),
    (Unit("1"), "1"),
]


@settings(max_examples=400)
@given(
    value=st.floats(allow_nan=False, allow_infinity=False),
    case=st.sampled_from(ROUND_TRIPS),
)
def test_format_quantity_reads_back_to_the_same_double(value, case):
    unit, display = case
    assert same_double(
        parse_quantity(format_quantity(value, unit, display), unit), value
    )

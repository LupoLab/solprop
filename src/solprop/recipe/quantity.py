# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quantities in a recipe: numbers with units, read and written back exactly.

A recipe value is either a bare number, taken in the keyword's own unit, or a quantity
string such as ``"125 um"``, ``"1.2 bar"`` or ``"2e14 W/cm^2"``, converted to that unit.
The keyword's unit is the one its docstring states, so pressures stay in bar
(``docs/dev/conventions.md``) and a bare ``1.2`` for a pressure means 1.2 bar.

**The unit grammar.** A unit is a product of factors, with at most one solidus: a
numerator of factors joined by ``*`` or ``·`` (or the single factor ``1``), then
optionally ``/`` and one factor or a parenthesised product, as the SI brochure writes
it (``W/cm^2``, ``fs^2/mm``, ``J/(m*s)``). ``a/b/c`` and ``a/b*c`` are refused as
ambiguous. A factor is an optional SI prefix (``q`` to ``Q``; ``u``, ``µ`` and ``μ`` all
read as micro), a symbol and an optional integer exponent ``^n`` or ``^-n``. The
symbols are ``m``, ``s``, ``g``, ``Hz``, ``J``, ``W``, ``Pa``, ``bar``, ``K``, ``rad``
and ``deg``. Temperature is in kelvin only: there is no Celsius or Fahrenheit. A
whitespace between the number and the unit is optional (``"800nm"`` is read).

**Angles are a dimension.** Plane angle is tracked as a fifth base dimension beside
length, time, mass and temperature, so an angular frequency in rad/s is never confused
with a frequency in Hz (a factor of 2π). A consequence is that ``rad`` is written only
where the keyword's unit is an angle.

**Exact conversion.** The number is read as a decimal, multiplied by the ratio of the
two units' factors exactly in :mod:`decimal`, and rounded once to the nearest binary64
value, so ``"800 nm"`` gives the double of the literal ``800e-9`` that a script writes.
Every unit but ``deg`` is a decimal multiple of an SI unit, so the ratio only shifts
the decimal exponent. ``deg`` multiplies by π/180, which has no finite decimal; it is
carried to 70 significant digits, so the one rounding to binary64 gives the double
nearest the exact angle unless the exact value lies within 10⁻⁷⁰ (relative) of a tie,
which no recipe value does. ``deg`` therefore stands alone: it takes no prefix, no
exponent and no other factor.

**Writing back.** :func:`format_quantity` writes a double in a display unit with the
fewest digits that read back to the same double. For a decimal display unit this is
the shortest round-tripping representation of the double (Python's :func:`repr`) with
its decimal point shifted, which is always exact; for ``deg`` it is the fewest of up
to 17 significant digits that read back, which always suffice.
"""

import decimal
import difflib
import functools
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from solprop.recipe.errors import RecipeError

__all__ = [
    "Dimension",
    "ElementUnits",
    "Unit",
    "beta_unit",
    "format_quantity",
    "parse_quantity",
    "phase_unit",
]


@dataclass(frozen=True)
class Dimension:
    """A physical dimension as integer exponents of five base dimensions.

    Attributes
    ----------
    length : int
        Exponent of length (m).
    time : int
        Exponent of time (s).
    mass : int
        Exponent of mass (kg).
    temperature : int
        Exponent of thermodynamic temperature (K).
    angle : int
        Exponent of plane angle (rad), kept apart so that rad/s and Hz differ.

    Examples
    --------
    >>> pressure = Dimension(length=-1, time=-2, mass=1)
    >>> str(pressure)
    'pressure (kg m^-1 s^-2)'
    >>> str(Dimension(length=1) * Dimension(length=1))
    'area (m^2)'
    """

    length: int = 0
    time: int = 0
    mass: int = 0
    temperature: int = 0
    angle: int = 0

    def __mul__(
        self,
        other: "Dimension",  # noqa: UP037 - the hook reads it in the class body
    ) -> "Dimension":  # noqa: UP037
        """Return the dimension of a product.

        Parameters
        ----------
        other : Dimension
            The other factor's dimension.

        Returns
        -------
        Dimension
            The sum of the exponents.
        """
        return type(self)(
            self.length + other.length,
            self.time + other.time,
            self.mass + other.mass,
            self.temperature + other.temperature,
            self.angle + other.angle,
        )

    def __pow__(self, exponent: int) -> "Dimension":  # noqa: UP037
        """Return the dimension raised to an integer power.

        Parameters
        ----------
        exponent : int
            The power.

        Returns
        -------
        Dimension
            The exponents multiplied by ``exponent``.
        """
        return type(self)(
            self.length * exponent,
            self.time * exponent,
            self.mass * exponent,
            self.temperature * exponent,
            self.angle * exponent,
        )

    def __str__(self) -> str:
        """Return the dimension's name, where it has one, and its base units.

        Returns
        -------
        str
            For example ``'length (m)'``, ``'kg m^-1 s^-2'`` or ``'dimensionless'``.
        """
        exponents = (self.mass, self.length, self.time, self.temperature, self.angle)
        parts = [
            symbol if power == 1 else f"{symbol}^{power}"
            for symbol, power in zip(_BASE_SYMBOLS, exponents, strict=True)
            if power
        ]
        if not parts:
            return "dimensionless"
        base = " ".join(parts)
        name = _DIMENSION_NAMES.get(self)
        return f"{name} ({base})" if name else base


_BASE_SYMBOLS = ("kg", "m", "s", "K", "rad")
_LENGTH = Dimension(length=1)
_TIME = Dimension(time=1)
_MASS = Dimension(mass=1)
_ENERGY = Dimension(length=2, time=-2, mass=1)
_POWER = Dimension(length=2, time=-3, mass=1)
_PRESSURE = Dimension(length=-1, time=-2, mass=1)

_DIMENSION_NAMES = {
    _LENGTH: "length",
    _TIME: "time",
    _MASS: "mass",
    Dimension(temperature=1): "temperature",
    Dimension(angle=1): "angle",
    Dimension(length=2): "area",
    Dimension(time=-1): "frequency",
    Dimension(time=-1, angle=1): "angular frequency",
    _ENERGY: "energy",
    _POWER: "power",
    _PRESSURE: "pressure",
    Dimension(time=-3, mass=1): "intensity",
}


@dataclass(frozen=True)
class _Scale:
    """What a unit text means: its dimension and its factor to the coherent SI unit.

    The factor is ``coefficient * 10**exponent10``; the coefficient is 1 for every unit
    but ``deg``.
    """

    dimension: Dimension
    exponent10: int
    coefficient: Decimal = Decimal(1)


# π to 70 significant digits; see the module docstring for why that is enough.
_PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816")
_DEGREE = decimal.Context(prec=70).divide(_PI, 180)

# symbol: (dimension, decimal exponent of its factor to the coherent SI unit). The
# coherent SI unit of mass is the kilogram, so the gram is 10^-3.
_SYMBOLS: dict[str, tuple[Dimension, int]] = {
    "m": (_LENGTH, 0),
    "s": (_TIME, 0),
    "g": (_MASS, -3),
    "Hz": (Dimension(time=-1), 0),
    "J": (_ENERGY, 0),
    "W": (_POWER, 0),
    "Pa": (_PRESSURE, 0),
    "bar": (_PRESSURE, 5),
    "K": (Dimension(temperature=1), 0),
    "rad": (Dimension(angle=1), 0),
}
_DEGREE_SYMBOL = "deg"

_PREFIXES: dict[str, int] = {
    "q": -30,
    "r": -27,
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "µ": -6,  # U+00B5 MICRO SIGN
    "μ": -6,  # U+03BC GREEK SMALL LETTER MU
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

_REFUSED_TEMPERATURES = frozenset({"degC", "°C", "C", "degF", "°F", "F", "℃", "℉"})

_FACTOR = re.compile(r"(?P<body>[^\W\d_]+)(?:\^(?P<power>[+-]?[1-9]\d*))?")
_MULTIPLY = re.compile(r"[*·⋅]")
_QUANTITY = re.compile(
    r"\s*(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>.*?)\s*",
    re.DOTALL,
)


def _refuse_unit(text: str, reason: str, path: str | None) -> RecipeError:
    """Return the error for a unit text that cannot be read.

    Parameters
    ----------
    text : str
        The unit text.
    reason : str
        Why it cannot be read.
    path : str or None
        The parameter path.

    Returns
    -------
    RecipeError
        The error, to raise.
    """
    return RecipeError(f"cannot read the unit {text!r}: {reason}", path=path)


def _parse_symbol(body: str, text: str, path: str | None) -> tuple[Dimension, int]:
    """Split a factor's letters into an optional prefix and a symbol.

    Parameters
    ----------
    body : str
        The letters of one factor, such as ``"fs"``.
    text : str
        The whole unit text, for the error.
    path : str or None
        The parameter path, for the error.

    Returns
    -------
    tuple of Dimension and int
        The symbol's dimension and the decimal exponent of the prefixed symbol's
        factor to the coherent SI unit.

    Raises
    ------
    RecipeError
        If ``body`` has no reading, or more than one.
    """
    readings: list[tuple[Dimension, int]] = []
    if body in _SYMBOLS:
        readings.append(_SYMBOLS[body])
    for prefix, exponent10 in _PREFIXES.items():
        rest = body.removeprefix(prefix)
        if rest != body and rest in _SYMBOLS:
            dimension, base10 = _SYMBOLS[rest]
            readings.append((dimension, base10 + exponent10))
    if len(readings) == 1:
        return readings[0]
    if readings:
        raise _refuse_unit(text, f"{body!r} has more than one reading", path)
    known = [*_SYMBOLS, _DEGREE_SYMBOL]
    nearest = difflib.get_close_matches(body, known, n=1)
    hint = f"; did you mean {nearest[0]!r}?" if nearest else ""
    raise _refuse_unit(
        text, f"unknown unit {body!r} (known: {', '.join(known)}){hint}", path
    )


def _parse_factors(
    product: str, text: str, path: str | None
) -> tuple[Dimension, int, int]:
    """Read a product of factors joined by ``*``.

    Parameters
    ----------
    product : str
        The product, such as ``"fs^2"`` or ``"J*s"``.
    text : str
        The whole unit text, for the error.
    path : str or None
        The parameter path, for the error.

    Returns
    -------
    tuple of Dimension, int and int
        The product's dimension, the decimal exponent of its factor and the number
        of factors.

    Raises
    ------
    RecipeError
        If a factor cannot be read, is a refused temperature unit or is ``deg``.
    """
    dimension = Dimension()
    exponent10 = 0
    factors = product.split("*")
    for factor in factors:
        if factor in _REFUSED_TEMPERATURES:
            raise _refuse_unit(text, "a temperature is in K only", path)
        match = _FACTOR.fullmatch(factor)
        if match is None:
            raise _refuse_unit(text, f"{factor!r} is not a unit factor", path)
        body, power = match["body"], int(match["power"] or 1)
        if body == _DEGREE_SYMBOL:
            raise _refuse_unit(text, "deg stands alone, with no prefix or power", path)
        symbol_dimension, symbol10 = _parse_symbol(body, text, path)
        dimension = dimension * symbol_dimension**power
        exponent10 += symbol10 * power
    return dimension, exponent10, len(factors)


def _parse_unit(text: str, path: str | None = None) -> _Scale:
    """Read a unit text into its dimension and factor (see the module docstring).

    Parameters
    ----------
    text : str
        The unit text.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    _Scale
        The dimension and the factor to the coherent SI unit.

    Raises
    ------
    RecipeError
        If the text is empty, ambiguous or cannot be read.
    """
    if text == _DEGREE_SYMBOL:
        return _Scale(Dimension(angle=1), 0, _DEGREE)
    compact = re.sub(r"\s*([*/()·⋅])\s*", r"\1", text.strip())
    compact = _MULTIPLY.sub("*", compact)
    if not compact:
        raise _refuse_unit(text, "it is empty", path)
    if re.search(r"\s", compact):
        raise _refuse_unit(text, "separate the factors with * or ·", path)
    numerator, slash, denominator = compact.partition("/")
    if "/" in denominator:
        raise _refuse_unit(
            text, "it has two solidi; write one, with a parenthesised denominator", path
        )
    if "(" in numerator or ")" in numerator:
        raise _refuse_unit(text, "only a denominator may be parenthesised", path)
    if numerator == "1" and slash:
        dimension, exponent10 = Dimension(), 0
    elif numerator == "1":
        return _Scale(Dimension(), 0)
    else:
        dimension, exponent10, _ = _parse_factors(numerator, text, path)
    if not slash:
        return _Scale(dimension, exponent10)
    grouped = denominator.startswith("(") and denominator.endswith(")")
    inner = denominator[1:-1] if grouped else denominator
    if "(" in inner or ")" in inner:
        raise _refuse_unit(text, "its parentheses do not match", path)
    under, under10, count = _parse_factors(inner, text, path)
    if count > 1 and not grouped:
        raise _refuse_unit(
            text,
            "a product after a solidus is ambiguous; parenthesise it, as in J/(m*s)",
            path,
        )
    return _Scale(dimension * under**-1, exponent10 - under10)


@dataclass(frozen=True)
class Unit:
    """The unit a keyword, or an element of one, is given in.

    A bare number for the keyword is taken in this unit, and every quantity string is
    converted to it. It is a decimal multiple of an SI unit, usually the coherent one
    (``"m"``, ``"s^2/m"``) but ``"bar"`` for a pressure.

    Attributes
    ----------
    text : str
        The unit in the grammar of the module docstring, in ASCII, or ``"1"`` for a
        dimensionless keyword. Two units are equal when their texts are.

    Raises
    ------
    RecipeError
        If ``text`` cannot be read, or is ``deg`` (a keyword's unit is ``rad``).

    Examples
    --------
    >>> Unit("bar").dimension
    Dimension(length=-1, time=-2, mass=1, temperature=0, angle=0)
    >>> str(Unit("s^2/m").dimension)
    'm^-1 s^2'
    """

    text: str

    def __post_init__(self) -> None:
        """Read the unit text, refusing one that is not a decimal multiple of SI."""
        if _unit_scale(self.text).coefficient != 1:
            raise RecipeError(
                f"a keyword's unit must be a decimal multiple of an SI unit, "
                f"not {self.text!r}"
            )

    @property
    def dimension(self) -> Dimension:
        """The unit's dimension.

        Returns
        -------
        Dimension
            The exponents of the five base dimensions.
        """
        return _unit_scale(self.text).dimension


@functools.cache
def _unit_scale(text: str) -> _Scale:
    """Read a keyword's unit text once (the texts are a small fixed set).

    Parameters
    ----------
    text : str
        The unit text.

    Returns
    -------
    _Scale
        As :func:`_parse_unit` returns it.
    """
    return _parse_unit(text)


type ElementUnits = Callable[[int], Unit]
"""The units of an array keyword whose elements differ, as a function of the index."""


def phase_unit(order: int) -> Unit:
    """Return the unit of element ``order`` of a ``phases`` list.

    The spectral phase is the Taylor series ``Σ φ_n (ω - ω₀)ⁿ / n!``, so ``φ_0`` is an
    angle and ``φ_n`` for ``n ≥ 1`` is in sⁿ (the radian of the phase is not carried
    into the higher orders, as is usual for dispersion coefficients).

    Parameters
    ----------
    order : int
        The Taylor order ``n``, at least 0.

    Returns
    -------
    Unit
        ``rad`` for order 0, ``s`` for order 1 and ``s^n`` after.

    Raises
    ------
    RecipeError
        If ``order`` is negative.

    Examples
    --------
    >>> [phase_unit(n).text for n in range(4)]
    ['rad', 's', 's^2', 's^3']
    """
    if order < 0:
        raise RecipeError(f"a Taylor order is at least 0, not {order}")
    if order == 0:
        return Unit("rad")
    return Unit("s" if order == 1 else f"s^{order}")


def beta_unit(order: int) -> Unit:
    """Return the unit of element ``order`` of a ``betas`` list.

    The propagation constant's Taylor coefficients ``β_n = ∂ⁿβ/∂ωⁿ`` are in sⁿ/m.

    Parameters
    ----------
    order : int
        The Taylor order ``n``, at least 0.

    Returns
    -------
    Unit
        ``1/m`` for order 0, ``s/m`` for order 1 and ``s^n/m`` after.

    Raises
    ------
    RecipeError
        If ``order`` is negative.

    Examples
    --------
    >>> [beta_unit(n).text for n in range(4)]
    ['1/m', 's/m', 's^2/m', 's^3/m']
    """
    if order < 0:
        raise RecipeError(f"a Taylor order is at least 0, not {order}")
    if order == 0:
        return Unit("1/m")
    return Unit("s/m" if order == 1 else f"s^{order}/m")


def _exact_context(*operands: Decimal) -> decimal.Context:
    """Return a context in which products and shifts of ``operands`` are exact.

    A product has at most as many digits as its factors together, and the exponent
    range is the largest :mod:`decimal` allows, so a rounding can only come from a
    bug; ``Inexact`` is trapped to make one loud.

    Parameters
    ----------
    *operands : Decimal
        The numbers to multiply or shift.

    Returns
    -------
    decimal.Context
        The context.
    """
    digits = sum(len(operand.as_tuple().digits) for operand in operands)
    return decimal.Context(
        prec=digits + 2,
        Emax=decimal.MAX_EMAX,
        Emin=decimal.MIN_EMIN,
        traps=[decimal.InvalidOperation, decimal.Overflow, decimal.Inexact],
    )


def _to_double(
    number: str, scale: _Scale, unit: Unit, path: str | None, text: str
) -> float:
    """Convert a decimal number from one unit to another exactly, then round once.

    Parameters
    ----------
    number : str
        The decimal number, in the unit ``scale`` describes.
    scale : _Scale
        The unit the number is in.
    unit : Unit
        The keyword's unit, of the same dimension.
    path : str or None
        The parameter path, for the error.
    text : str
        The quantity string, for the error.

    Returns
    -------
    float
        The double nearest the exact value in ``unit``.

    Raises
    ------
    RecipeError
        If the exponent is beyond :mod:`decimal`'s range, or the value overflows a
        double or underflows one to zero.
    """
    shift = scale.exponent10 - _unit_scale(unit.text).exponent10
    try:
        mantissa = Decimal(number)
        context = _exact_context(mantissa, scale.coefficient)
        exact = context.multiply(mantissa, scale.coefficient).scaleb(shift, context)
    except decimal.DecimalException:
        raise RecipeError(f"{text!r} is out of range", path=path) from None
    # float() of a Decimal goes through its exact decimal string and the correctly
    # rounded string-to-double conversion: the one rounding.
    value = float(exact)
    if not math.isfinite(value):
        raise RecipeError(f"{text!r} is too large for a double", path=path)
    if value == 0 and not exact.is_zero():
        raise RecipeError(f"{text!r} is too small for a double", path=path)
    return value


def parse_quantity(value: object, unit: Unit, *, path: str | None = None) -> float:
    """Read a recipe value given as a number or a quantity string, in ``unit``.

    Parameters
    ----------
    value : object
        An ``int`` or ``float``, taken in ``unit``, or a string of a decimal number and
        a unit (``"125 um"``, ``"2e14 W/cm^2"``), whose dimension must be the unit's.
        A string with no unit is read only for a dimensionless ``unit``.
    unit : Unit
        The keyword's unit.
    path : str or None, optional
        The parameter path, for the error message.

    Returns
    -------
    float
        The value in ``unit``: the double nearest the exact decimal value.

    Raises
    ------
    RecipeError
        If ``value`` is a boolean, not finite, not a number or quantity string, or of
        another dimension, if its unit cannot be read or has two readings, or if the
        value overflows or underflows a double.

    Examples
    --------
    >>> parse_quantity("800 nm", Unit("m")) == 800e-9
    True
    >>> parse_quantity("120 kPa", Unit("bar"))
    1.2
    >>> parse_quantity(1.2, Unit("bar"))
    1.2
    >>> parse_quantity("2e14 W/cm^2", Unit("W/m^2"))
    2e+18
    """
    if isinstance(value, bool):
        raise RecipeError(f"a boolean is not a quantity in {unit.text}", path=path)
    if isinstance(value, int | float):
        try:
            number = float(value)
        except OverflowError:
            raise RecipeError(f"{value} is too large for a double", path=path) from None
        if not math.isfinite(number):
            raise RecipeError(f"{value!r} is not a finite number", path=path)
        return number
    if not isinstance(value, str):
        raise RecipeError(
            f"expected a number in {unit.text} or a quantity string, "
            f"not {type(value).__name__}",
            path=path,
        )
    match = _QUANTITY.fullmatch(value)
    if match is None:
        raise RecipeError(
            f"{value!r} is not a quantity: write a number and a unit, such as "
            f"'800 nm', or a bare number in {unit.text}",
            path=path,
        )
    unit_text = match["unit"]
    if not unit_text:
        if unit.dimension == Dimension():
            return _to_double(
                match["number"], _Scale(Dimension(), 0), unit, path, value
            )
        raise RecipeError(
            f"{value!r} has no unit, which is ambiguous in a string: write the unit, "
            f"or a bare number in {unit.text}",
            path=path,
        )
    scale = _parse_unit(unit_text, path)
    if scale.dimension != unit.dimension:
        raise RecipeError(
            f"{value!r} is a {scale.dimension}, but the value is a {unit.dimension}",
            path=path,
        )
    return _to_double(match["number"], scale, unit, path, value)


def _decimal_text(number: Decimal) -> str:
    """Write a decimal as Python writes a float: plain or in e-notation.

    Parameters
    ----------
    number : Decimal
        The number.

    Returns
    -------
    str
        Plain digits for a decimal exponent from -4 to 15, as :func:`repr` writes a
        float, and e-notation otherwise; no trailing zeros, and ``-0`` for a negative
        zero.
    """
    if number.is_zero():
        return "-0" if number.is_signed() else "0"
    number = number.normalize(_exact_context(number))
    if -4 <= number.adjusted() < 16:
        return format(number, "f")
    return format(number, "e").replace("e+", "e")


def _reads_back(text: str, unit: Unit, value: float) -> bool:
    """Tell whether a quantity string reads back to exactly a value.

    Parameters
    ----------
    text : str
        The quantity string.
    unit : Unit
        The keyword's unit.
    value : float
        The value it should read back to.

    Returns
    -------
    bool
        Whether it does, the sign of a zero included.
    """
    try:
        parsed = parse_quantity(text, unit)
    except RecipeError:
        return False
    return parsed == value and math.copysign(1, parsed) == math.copysign(1, value)


def format_quantity(value: float, unit: Unit, display: str) -> str:
    """Write a value in ``unit`` as a quantity string in the display unit.

    The number has the fewest significant digits that read back, through
    :func:`parse_quantity`, to exactly ``value``, the sign of a zero included.

    Parameters
    ----------
    value : float
        The value in ``unit``; finite.
    unit : Unit
        The keyword's unit.
    display : str
        The unit to write in, of the same dimension, in ASCII by convention (``"um"``).

    Returns
    -------
    str
        A quantity string such as ``"125 um"``.

    Raises
    ------
    RecipeError
        If ``value`` is not finite, or ``display`` cannot be read or is of another
        dimension.

    Examples
    --------
    >>> format_quantity(125e-6, Unit("m"), "um")
    '125 um'
    >>> format_quantity(7.5e-15, Unit("s"), "fs")
    '7.5 fs'
    >>> format_quantity(1.2, Unit("bar"), "kPa")
    '120 kPa'
    """
    if not math.isfinite(value):
        raise RecipeError(f"{value!r} is not a finite number")
    value = float(value)
    scale = _parse_unit(display)
    if scale.dimension != unit.dimension:
        raise RecipeError(
            f"cannot write a {unit.dimension} in {display!r}, a {scale.dimension}"
        )
    shortest = Decimal(repr(value))
    if scale.coefficient == 1:
        shift = _unit_scale(unit.text).exponent10 - scale.exponent10
        shifted = shortest.scaleb(shift, _exact_context(shortest))
        if scale.dimension == Dimension() and display.strip() == "1":
            return _decimal_text(shifted)  # a dimensionless number has no unit
        return f"{_decimal_text(shifted)} {display}"
    # deg: search the significant digits of value / coefficient for the fewest that
    # read back. Seventeen always do: their rounding error is below half a unit in
    # the last place of any double, and the factor is carried to 70 digits.
    exact = Decimal(value).scaleb(_unit_scale(unit.text).exponent10)
    quotient = decimal.Context(prec=40).divide(exact, scale.coefficient)
    for digits in range(1, 18):
        rounded = decimal.Context(prec=digits).plus(quotient)
        text = f"{_decimal_text(rounded)} {display}"
        if _reads_back(text, unit, value):
            return text
    return format_quantity(value, unit, unit.text)  # pragma: no cover - 17 digits do

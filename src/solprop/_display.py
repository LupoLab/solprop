# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Format objects for a person reading them at a prompt.

The package's printing rule, in one place: ``repr(x)`` is one line and never holds an
array, so that it is safe in a log record or an error message; ``str(x)`` is a short
aligned block in the units people quote (nm, fs, uJ, um, bar); and IPython and Jupyter
show that block when the object is the last expression of a cell, through
:meth:`Displayed._repr_pretty_`.

Printing never evaluates physics, never warns and never touches a file it does not
already own: a block is built from numbers the object already holds.

This module imports nothing from the rest of the package, and NumPy only where an array
is actually summarised, so that any module may use it.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "Displayed",
    "block",
    "engineering",
    "line",
    "phase_text",
    "summarise",
    "tail",
]

_PREFIXES = {
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "u",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
}
"""SI prefixes by decimal exponent. ``u`` stands for micro: the package's text is ASCII
wherever a terminal, a log file or an HDF5 attribute may have to carry it."""

_WIDTH = 88
"""Longest line of a block, the package's line length."""


@runtime_checkable
class _Printer(Protocol):
    """The part of IPython's pretty printer that :class:`Displayed` uses."""

    def text(self, value: str) -> None:
        """Write text.

        Parameters
        ----------
        value : str
            The text.
        """
        ...


def engineering(value: float | None, unit: str = "", digits: int = 4) -> str:
    """Return a number with an SI prefix, as a person would write it.

    Parameters
    ----------
    value : float or None
        The number, in the SI unit named by `unit`.
    unit : str, optional
        The unit's symbol, without a prefix: ``"m"``, ``"s"``, ``"J"``, ``"W"``.
        With an empty unit no prefix is used and the number is written plainly.
    digits : int, optional
        Significant figures.

    Returns
    -------
    str
        For example ``"800 nm"``, ``"30 fs"``, ``"1.5 m"``; ``"none"`` for ``None``
        and ``"nan"``, ``"inf"`` or ``"-inf"`` for a value that is not finite.

    Examples
    --------
    >>> from solprop._display import engineering
    >>> engineering(800e-9, "m"), engineering(1.5, "m"), engineering(275e-6, "J")
    ('800 nm', '1.5 m', '275 uJ')
    >>> engineering(float("nan"), "s"), engineering(None, "s"), engineering(0.0, "W")
    ('nan', 'none', '0 W')
    """
    if value is None:
        return "none"
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if not unit:
        return f"{number:.{digits}g}"
    if number == 0.0:
        return f"0 {unit}"
    exponent = 3 * math.floor(math.log10(abs(number)) / 3.0)
    exponent = min(max(exponent, min(_PREFIXES)), max(_PREFIXES))
    scaled = number / 10.0**exponent
    # Rounding to the requested figures can carry 999.96 over to 1000.
    if abs(float(f"{scaled:.{digits}g}")) >= 1000.0 and exponent < max(_PREFIXES):
        exponent += 3
        scaled = number / 10.0**exponent
    return f"{scaled:.{digits}g} {_PREFIXES[exponent]}{unit}"


def summarise(values: object, unit: str = "") -> str:
    """Return a one-phrase summary of an array, never its contents.

    Parameters
    ----------
    values : array_like
        The array.
    unit : str, optional
        SI unit of the values, for the range; empty to give the shape alone.

    Returns
    -------
    str
        For example ``"<201 values, 0 m to 1.5 m>"`` or ``"<2x1x8192 complex>"``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop._display import summarise
    >>> summarise(np.linspace(0.0, 1.5, 201), "m")
    '<201 values, 0 m to 1.5 m>'
    >>> summarise(np.zeros((2, 1, 8), dtype=complex))
    '<2x1x8 complex>'
    """
    import numpy as np  # noqa: PLC0415 - kept out of the import of modules that only format scalars

    array = np.asarray(values)
    if array.ndim != 1 or not unit or array.size == 0 or np.iscomplexobj(array):
        shape = "x".join(str(size) for size in array.shape) or "scalar"
        kind = "complex" if np.iscomplexobj(array) else "values"
        return f"<{shape} {kind}>"
    low, high = float(np.min(array)), float(np.max(array))
    return (
        f"<{array.size} values, {engineering(low, unit)} to {engineering(high, unit)}>"
    )


_PHASE_NAMES = ("CEP", "delay", "GDD", "TOD", "FOD")
"""What the first Taylor coefficients of a spectral phase are called."""


def phase_text(coefficients: Iterable[float]) -> str:
    """Return Taylor spectral-phase coefficients as people quote them.

    Parameters
    ----------
    coefficients : iterable of float
        ``c_0, c_1, ...`` in ``s**n``; a tuple or an array.

    Returns
    -------
    str
        For example ``"CEP 0.3 rad, GDD 250 fs^2"``: zero coefficients are left out,
        order ``n`` is in ``fs^n``, and ``"none"`` stands for no phase at all.

    Examples
    --------
    >>> from solprop._display import phase_text
    >>> phase_text((0.3, 0.0, 250e-30, -4000e-45))
    'CEP 0.3 rad, GDD 250 fs^2, TOD -4000 fs^3'
    >>> phase_text(())
    'none'
    """
    parts = []
    for order, value in enumerate(coefficients):
        if float(value) == 0.0:
            continue
        name = _PHASE_NAMES[order] if order < len(_PHASE_NAMES) else f"order {order}"
        unit = "rad" if order == 0 else ("fs" if order == 1 else f"fs^{order}")
        parts.append(f"{name} {float(value) * 1e15**order:.4g} {unit}")
    return ", ".join(parts) or "none"


def tail(text: str, width: int = 68) -> str:
    """Return the end of a long text, which for a path is the part that matters.

    Parameters
    ----------
    text : str
        The text, typically a file path.
    width : int, optional
        Longest result, in characters.

    Returns
    -------
    str
        The text itself, or an ellipsis and its last ``width - 3`` characters.

    Examples
    --------
    >>> from solprop._display import tail
    >>> tail("/a/very/long/path/to/run.h5", 12)
    '...to/run.h5'
    """
    return text if len(text) <= width else "..." + text[-(width - 3) :]


def line(name: str, fields: Iterable[tuple[str, object]]) -> str:
    """Return a one-line ``Name(key=value, ...)`` description.

    Parameters
    ----------
    name : str
        The class name.
    fields : iterable of tuple
        ``(key, value)`` pairs; a value is written with ``str``, and a pair whose key
        is empty is written as the value alone.

    Returns
    -------
    str
        The line.

    Examples
    --------
    >>> from solprop._display import line
    >>> line("Grid", [("n_t", 8192), ("", "field")])
    'Grid(n_t=8192, field)'
    """
    parts = [f"{key}={value}" if key else str(value) for key, value in fields]
    return f"{name}({', '.join(parts)})"


def block(title: str, rows: Sequence[tuple[str, object]]) -> str:
    """Return an aligned multi-line description.

    Parameters
    ----------
    title : str
        The first line: the class and its headline.
    rows : sequence of tuple
        ``(key, value)`` pairs, one line each. A value holding newlines is indented
        under its key; a line longer than the package's 88 columns is cut with an
        ellipsis, so that a block is always safe to paste into a page.

    Returns
    -------
    str
        The block, without a trailing newline.

    Examples
    --------
    >>> from solprop._display import block
    >>> print(block("Pulse", [("energy", "275 uJ"), ("duration", "7.5 fs")]))
    Pulse
      energy:   275 uJ
      duration: 7.5 fs
    """
    width = max((len(key) for key, _ in rows), default=0) + 1
    lines = [title]
    for key, value in rows:
        first, *rest = str(value).split("\n")
        lines.append(f"  {key + ':':<{width}} {first}")
        lines.extend(f"  {'':<{width}} {extra}" for extra in rest)
    return "\n".join(
        text if len(text) <= _WIDTH else text[: _WIDTH - 3] + "..." for text in lines
    )


class Displayed:
    """Give a class the package's printing: a block for ``str`` and for IPython.

    A class provides ``_describe()`` returning its title and its rows, and its own
    one-line ``__repr__``; this mixin turns the first into ``__str__`` and into the
    text IPython and Jupyter show for the last expression of a cell. The plain Python
    prompt, log records and error messages keep the one-line ``repr``.
    """

    def _describe(self) -> tuple[str, Sequence[tuple[str, object]]]:
        """Return the title and the rows of the block.

        Returns
        -------
        tuple
            The first line, and ``(key, value)`` pairs for the rest.
        """
        return repr(self), ()

    def __str__(self) -> str:
        """Return the aligned block.

        Returns
        -------
        str
            The title and one line per row.
        """
        return block(*self._describe())

    def _repr_pretty_(self, printer: _Printer, cycle: bool) -> None:
        """Write the block for IPython's pretty printer.

        Parameters
        ----------
        printer : object
            IPython's printer.
        cycle : bool
            Whether the object was met again while printing itself.
        """
        printer.text(repr(self) if cycle else str(self))

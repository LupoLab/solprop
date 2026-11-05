# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a scan's results page draws, from the kernel's ``scan_overview`` reply."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from solprop_gui.model.display import scaled_unit

__all__ = [
    "ScanLines",
    "profile_points",
    "result_title",
    "scan_lines",
    "shown_results",
    "variable_unit",
]

_FIRST: Final = ("energy", "peak_power", "peak_intensity", "fwhm_t")
"""The results a scan shows first, where it recorded them, in order."""

_TITLES: Final = {
    "fwhm_t": "Duration",
    "fwhm_t_outer": "Outer duration",
    "omega0": "Centre frequency",
    "peak_power_total": "Peak power, all modes",
}
"""What a result is called where its name does not read as it stands."""

_UNITS: Final = {
    "energy": "J",
    "pressure": "bar",
    "radius": "m",
    "wavelength": "m",
    "fwhm": "s",
    "width": "s",
    "length": "m",
    "peak_power": "W",
}
"""The unit a scan's variable is in, by its keyword, where it has one."""

_MOST_LINES: Final = 8
"""The most values of a second variable drawn as lines of their own."""


def variable_unit(variable: Mapping[str, object]) -> str:
    """Return the unit a scan variable's values are in.

    Parameters
    ----------
    variable : Mapping
        Its ``name``, and the ``unit`` its values were written in, if any.

    Returns
    -------
    str
        The unit; a keyword's own where the values are numbers.

    Examples
    --------
    >>> variable_unit({"name": "pressure[0]", "unit": "bar"})
    'bar'
    >>> variable_unit({"name": "energy"}), variable_unit({"name": "gamma"})
    ('J', '')
    """
    written = str(variable.get("unit") or "")
    return written or _UNITS.get(str(variable["name"]).split("[", 1)[0], "")


def result_title(name: str) -> str:
    """Return a scan result's plain name.

    Parameters
    ----------
    name : str
        Its name in the file, as ``"stats/energy"``.

    Returns
    -------
    str
        What the page calls it.

    Examples
    --------
    >>> result_title("stats/fwhm_t"), result_title("stats/peak_power")
    ('Duration', 'Peak power')
    """
    short = name.rsplit("/", 1)[-1]
    return _TITLES.get(short, short.replace("_", " ").capitalize())


def _varies(values: object) -> bool:
    """Return whether a result changes over the scan's points.

    Parameters
    ----------
    values : object
        Its values.

    Returns
    -------
    bool
        Whether its finite values move by more than rounding.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size) and np.ptp(finite) > 1e-12 * float(np.abs(finite).max())


def shown_results(results: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Return the results worth offering, the most read first.

    A result the scan's variables do not move (a core radius scanned over pressure) is
    left out, unless nothing moves.

    Parameters
    ----------
    results : Mapping
        The reply's ``results``, each with its ``values``.

    Returns
    -------
    list of str
        Their names.

    Examples
    --------
    >>> shown_results({"stats/core_radius": {"values": [1.0, 1.0]},
    ...                "stats/energy": {"values": [1.0, 2.0]}})
    ['stats/energy']
    >>> shown_results({"stats/a": {"values": [1.0, 1.0]}})
    ['stats/a']
    """
    moving = [name for name, entry in results.items() if _varies(entry["values"])]
    names = moving or list(results)

    def order(name: str) -> int:
        """Return where a result comes in the list.

        Parameters
        ----------
        name : str
            The result.

        Returns
        -------
        int
            Its place among the first, or after them.
        """
        short = name.rsplit("/", 1)[-1]
        return _FIRST.index(short) if short in _FIRST else len(_FIRST)

    return sorted(names, key=order)


@dataclass(frozen=True)
class ScanLines:
    """A scan's result as lines over its first variable.

    Attributes
    ----------
    x : numpy.ndarray
        The first variable's values.
    xlabel : str
        Its name.
    xunit : str
        Its unit, or ``""``.
    lines : list of tuple
        Each line's name (the second variable's value, or ``""``) and its values.
    unit : str
        The result's unit.
    failed : numpy.ndarray
        The first variable's values where a point failed.
    """

    x: np.ndarray
    xlabel: str
    xunit: str
    lines: list[tuple[str, np.ndarray]]
    unit: str
    failed: np.ndarray


def scan_lines(
    scan: Mapping[str, object], name: str, labels: Mapping[str, str] | None = None
) -> ScanLines:
    """Lay a result out as lines over the scan's first variable, one per second value.

    A third variable and beyond are read at their first value; a second with many
    values gives its first few lines.

    Parameters
    ----------
    scan : Mapping
        The reply: ``variables``, ``status`` and ``results``.
    name : str
        The result.
    labels : Mapping or None, optional
        The variables' names to show, by keyword.

    Returns
    -------
    ScanLines
        The lines.

    Examples
    --------
    >>> scan = {"variables": [{"name": "pressure", "values": [1.0, 2.0]},
    ...                       {"name": "energy", "values": [1e-6, 2e-6]}],
    ...         "status": [["done", "done"], ["failed", "done"]],
    ...         "results": {"stats/x": {"units": "J", "values": [[1, 2], [3, 4]]}}}
    >>> drawn = scan_lines(scan, "stats/x")
    >>> [title for title, _y in drawn.lines], drawn.lines[1][1].tolist()
    (['energy 1 µJ', 'energy 2 µJ'], [2.0, 4.0])
    >>> drawn.failed.tolist(), drawn.xunit
    ([2.0], 'bar')
    """
    variables: Sequence[Mapping[str, object]] = scan["variables"]  # pyright: ignore[reportAssignmentType]
    results: Mapping[str, Mapping[str, object]] = scan["results"]  # pyright: ignore[reportAssignmentType]
    named = labels or {}
    first = variables[0]
    x = np.asarray(first["values"], dtype=float)
    values = np.asarray(results[name]["values"], dtype=float).reshape(x.size, -1)
    status = np.asarray(scan["status"], dtype=object).reshape(x.size, -1)
    lines: list[tuple[str, np.ndarray]] = [("", values[:, 0])]
    if len(variables) > 1:
        second = variables[1]
        key = str(second["name"])
        seconds = np.asarray(second["values"], dtype=float)
        unit, scale = scaled_unit(seconds, variable_unit(second))
        rest = values.reshape(x.size, seconds.size, -1)[:, :, 0]
        lines = [
            (
                f"{named.get(key, key)} {value * scale:.4g} {unit}".rstrip(),
                rest[:, place],
            )
            for place, value in enumerate(seconds[:_MOST_LINES])
        ]
        status = status.reshape(x.size, seconds.size, -1)[:, :, 0]
    failed = x[(status == "failed").any(axis=1)]
    key = str(first["name"])
    return ScanLines(
        x,
        named.get(key, key),
        variable_unit(first),
        lines,
        str(results[name].get("units") or ""),
        failed,
    )


def profile_points(build: Mapping[str, object]) -> list[str]:
    """Return the scannable points of the profiles a recipe gives as lists.

    Parameters
    ----------
    build : Mapping
        The recipe's ``[build]`` table.

    Returns
    -------
    list of str
        ``name[i]`` for each value of a list of numbers or quantities.

    Examples
    --------
    >>> profile_points({"pressure": ["1.2 bar", "0 bar"], "gas": "He", "betas": [0, 1]})
    ['pressure[0]', 'pressure[1]', 'betas[0]', 'betas[1]']
    >>> profile_points({"pressure": [[0, "1 bar"], [1, "2 bar"]]})
    []
    """
    found: list[str] = []
    for name, value in build.items():
        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(item, int | float | str) and not isinstance(item, bool)
                for item in value
            )
        ):
            found.extend(f"{name}[{index}]" for index in range(len(value)))
    return found

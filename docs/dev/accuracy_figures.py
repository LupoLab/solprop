# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The figures of ``accuracy.md``, drawn from the stored convergence records.

Every ladder on that page emits its data as one JSON file under
``tests/data/convergence/``, written by the test that runs it and committed like a
witness (``docs/dev/testing.md`` section 5). This module reads those files and draws
them, and does nothing else: it runs no study, imports no test module and never touches
``solprop`` itself, so a documentation build is a read of files the suite maintains.

It is imported by the ``plot`` directives of ``accuracy.md`` as ``accuracy_figures``.
That works without any ``conf.py`` entry because matplotlib's plot directive executes
inline content with the page's own directory as the working directory and first on
``sys.path``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "backend_equivalence",
    "ionisation_knots",
    "load",
    "m2_mode2_along_z",
    "m2_order",
    "m2_quadrature",
    "m2_tolerance",
    "phi_quadrature",
    "plasma_trapezoid",
    "quadrature_alarm",
    "soliton_n6",
    "spatial_accuracy",
    "tableau_order",
]

RECORDS = Path(__file__).resolve().parents[2] / "tests" / "data" / "convergence"

# Luna is one colour throughout the page and solprop another, so that a reader who has
# seen one figure can read the next without the legend.
SOLPROP = "C0"
LUNA = "C3"

SIZE = (6.4, 3.6)


def load(study: str) -> dict:
    """Return the stored record of one study.

    Parameters
    ----------
    study : str
        The record's name, without the extension.

    Returns
    -------
    dict
        The record as it is stored, schema 1.
    """
    return json.loads((RECORDS / f"{study}.json").read_text())


def series(record: dict, name: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one series' axis values and one of its columns, sorted along the axis.

    Parameters
    ----------
    record : dict
        A record, as :func:`load` returns it.
    name : str
        Which series to take.
    column : str
        Which metric to take from it.

    Returns
    -------
    tuple of numpy.ndarray
        The axis values and the metric, with the rungs where the metric is absent --
        an order has no value on the coarsest rung of a ladder -- dropped from both.
    """
    rows = [row for row in record["rows"] if row["series"] == name]
    rows.sort(key=lambda row: _sortable(row["axis"]))
    kept = [row for row in rows if row[column] is not None]
    axis = np.array([_sortable(row["axis"]) for row in kept], dtype=float)
    return axis, np.array([float(row[column]) for row in kept])


def _sortable(axis: object) -> float:
    """Return an axis value as a number, so a ladder plots in its own order."""
    if isinstance(axis, int | float):
        return float(axis)
    try:
        return float(str(axis))
    except ValueError:
        return float("nan")


def names(record: dict) -> list[str]:
    """Return the record's series, in the order they were first written."""
    seen = []
    for row in record["rows"]:
        if row["series"] not in seen:
            seen.append(row["series"])
    return seen


def _slope(ax: Axes, x: np.ndarray, y: np.ndarray, order: float, label: str) -> None:
    """Draw a reference power law through the first point of a ladder."""
    reference = y[0] * (x[0] / x) ** order
    ax.plot(x, reference, ls=":", lw=1.2, color="0.5", label=label)


def _ticks(ax: Axes, record: dict, column: str) -> None:
    """Label the axis with the rungs themselves.

    A ladder at 8000, 16000, 32000 and 64000 steps on a base-2 logarithmic axis reads
    as 2^13 to 2^16, which is four numbers none of the rungs is. The rungs are what the
    record holds and what the tables quote, so they are what the axis says.
    """
    values = sorted(
        {row["axis"] for row in record["rows"] if isinstance(row["axis"], int)}
    )
    del column
    if values:
        ax.set_xticks(values, [f"{value:d}" for value in values], fontsize=8)
        ax.minorticks_off()


def _bound(ax: Axes, value: float, label: str) -> None:
    """Draw the bound a study is asserted against."""
    ax.axhline(value, ls="--", lw=1.2, color="0.4", label=label)


def _finish(ax: Axes, record: dict, ylabel: str, *, legend: bool = True) -> Axes:
    """Label the axes from the record itself and tidy the frame."""
    axis = record["axis"]
    unit = "" if axis["unit"] in {"1", ""} else f" ({axis['unit']})"
    ax.set_xlabel(f"{axis['label']}{unit}")
    ax.set_ylabel(ylabel)
    ax.set_title(record["title"], fontsize=10)
    if legend:
        ax.legend(frameon=False, fontsize=8)
    return ax


def _axes(size: tuple[float, float] = SIZE) -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=size, layout="constrained")
    return figure, ax


# --------------------------------------------------------------- the integrators


def tableau_order() -> None:
    """Draw the global order ladder of the two tableaux."""
    record = load("tableau_order")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        steps, error = series(record, name, "error")
        ax.plot(steps, error, "o-", lw=1.6, ms=4, color=f"C{index}", label=name)
    steps, error = series(record, "DOPRI5 solution", "error")
    _slope(ax, steps, error, 5.0, "fifth order")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _ticks(ax, record, "error")
    _finish(ax, record, "error at z = 1")


def phi_quadrature() -> None:
    """Draw the cumulative operator against an independent quadrature of it."""
    record = load("phi_quadrature")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        pair, deviation = series(record, name, "deviation")
        ax.plot(pair, deviation, "o", ms=4, color=f"C{index}", label=name)
    # Each interval's own bound, the same for both representations.
    pair, bound = series(record, names(record)[0], "bound")
    ax.plot(pair, bound, ls="--", lw=1.2, color="0.4", label="asserted bound")
    ax.set_yscale("log")
    _finish(ax, record, "deviation from quad_vec (relative)")


def ionisation_knots() -> None:
    """Draw the rate table's interpolation error against its knot count."""
    record = load("ionisation_knots")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        _, knots = series(record, name, "knots")
        _, error = series(record, name, "error")
        ax.plot(knots, error, "o-", lw=1.6, ms=4, color=f"C{index}", label=name)
        _slope(ax, knots, error, 4.0, "fourth order")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, record, "worst error in log rate")
    # The record's axis is the interval count; the page speaks of knots, which is the
    # same ladder plus one, and is what the curve above is drawn against.
    ax.set_xticks(knots, [f"{value:.0f}" for value in knots], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("knots")


def plasma_trapezoid() -> None:
    """Draw the plasma column's two second-order ladders."""
    record = load("plasma_trapezoid")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        samples, deviation = series(record, name, "deviation")
        ax.plot(samples, deviation, "o-", lw=1.6, ms=4, color=f"C{index}", label=name)
    samples, deviation = series(record, names(record)[0], "deviation")
    _slope(ax, samples, deviation, 2.0, "second order")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _ticks(ax, record, "deviation")
    _finish(ax, record, "deviation (relative)")


# ------------------------------------------------------- the transverse quadrature


def quadrature_alarm() -> None:
    """Draw the quadrature alarm against the radial order on a three-mode set."""
    record = load("quadrature_alarm")
    bound = record["columns"][0]["bound"]
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        order, alarm = series(record, name, "alarm")
        ax.plot(order, alarm, "o-", lw=1.6, ms=4, color=f"C{index}", label=name)
    _bound(ax, bound, "the 1e-6 alarm threshold")
    ax.set_yscale("log")
    _ticks(ax, record, "alarm")
    _finish(ax, record, "embedded pair's disagreement")


def m2_quadrature() -> None:
    """Draw the second mode's exit energy against the radial order, on both codes."""
    record = load("m2_quadrature")
    _, ax = _axes()
    for name in names(record):
        order, energy = series(record, name, "mode2_energy")
        colour = LUNA if name.startswith("Luna") else SOLPROP
        ax.plot(order, 1e6 * energy, "o-", lw=1.6, ms=5, color=colour, label=name)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, None)
    _ticks(ax, record, "mode2_energy")
    _finish(ax, record, "second mode at the exit (µJ)")


# ------------------------------------------------------------------- milestone M1


def soliton_n6() -> None:
    """Draw the N = 6 recurrence ladder, with the polarisation taper off and on."""
    record = load("soliton_n6")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        steps, error = series(record, name, "error")
        ax.plot(steps, error, "o-", lw=1.6, ms=4, color=f"C{index}", label=name)
    steps, error = series(record, names(record)[0], "error")
    _slope(ax, steps, error, 5.0, "fifth order")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _ticks(ax, record, "error")
    _finish(ax, record, "recurrence error (relative)")


# ------------------------------------------------------------------- milestone M2


def m2_order() -> None:
    """Draw the M2 order study, one switch at a time."""
    record = load("m2_order")
    _, ax = _axes()
    for index, name in enumerate(names(record)):
        steps, difference = series(record, name, "difference")
        _, order = series(record, name, "order")
        label = f"{name} — order {order[0]:.2f}" if order.size else name
        ax.plot(steps, difference, "o-", lw=1.6, ms=4, color=f"C{index}", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _ticks(ax, record, "difference")
    _finish(ax, record, "successive difference (relative)")


def m2_tolerance() -> None:
    """Draw the second mode's exit energy against the step tolerance, on both codes."""
    record = load("m2_tolerance")
    _, ax = _axes()
    for name in names(record):
        rtol, energy = series(record, name, "mode2_energy")
        colour = LUNA if name.startswith("Luna") else SOLPROP
        ax.plot(rtol, 1e6 * energy, "o-", lw=1.6, ms=5, color=colour, label=name)
    converged = series(record, "Luna, full case", "mode2_energy")[1].min()
    ax.axhline(
        1e6 * converged, ls="--", lw=1.2, color="0.4", label="Luna at rtol = 1e-9"
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0.0, None)
    _finish(ax, record, "second mode at the exit (µJ)")


def m2_mode2_along_z() -> None:
    """Draw the second mode's energy along the fibre, both codes, plasma on and off."""
    record = load("m2_mode2_along_z")
    _, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True, layout="constrained")
    for ax, switch in zip(axes, ("off", "on"), strict=True):
        for name in names(record):
            if not name.endswith(switch):
                continue
            _, position = series(record, name, "z")
            _, energy = series(record, name, "energy")
            colour = LUNA if name.startswith("Luna") else SOLPROP
            ax.plot(position, 1e6 * energy, lw=1.6, color=colour, label=name)
        ax.set_xlabel("z (m)")
        ax.set_title(f"plasma {switch}", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("second mode (µJ)")


# ------------------------------------------------------------ backend equivalence


def backend_equivalence() -> None:
    """Draw each rung's derived bound beside what it measures."""
    record = load("backend_equivalence")
    rows = sorted(record["rows"], key=lambda row: float(row["order"]))
    labels = [f"{row['series']}: {row['axis']}" for row in rows]
    position = np.arange(len(rows))
    # The polarisation ladders add three backends and analytic controls. Preserve
    # one readable line per observation instead of overprinting the fixed-height plot.
    _, ax = _axes((12.0, max(3.2, 0.22 * len(rows))))
    positive = [row["deviation"] for row in rows if row["deviation"] > 0]
    zero_position = min(positive) / 10
    measured = [row["deviation"] or zero_position for row in rows]
    ax.hlines(
        position,
        measured,
        [row["bound"] for row in rows],
        color="0.75",
        lw=2.0,
    )
    ax.plot(
        [row["bound"] for row in rows], position, "|", ms=12, color="0.4", label="bound"
    )
    ax.plot(
        measured,
        position,
        "o",
        ms=5,
        color=SOLPROP,
        label="measured",
    )
    ax.set_yticks(position, labels, fontsize=8)
    for index, row in enumerate(rows):
        if row["deviation"] == 0:
            ax.annotate(
                "0",
                (zero_position, index),
                xytext=(-10, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("relative deviation (exact zeros are labelled at the left)")
    ax.set_title(record["title"], fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def spatial_accuracy() -> None:
    """Draw the largest measured fraction of the bound for each spatial series."""
    record = load("spatial_accuracy")
    grouped: dict[str, dict[str, float]] = {}
    for row in record["rows"]:
        backend, label = row["series"].split(": ", 1)
        ratio = float(row["deviation"]) / float(row["bound"])
        group = grouped.setdefault(label, {})
        group[backend] = max(group.get(backend, 0.0), ratio)
    labels = sorted(grouped)
    backends = sorted({backend for group in grouped.values() for backend in group})
    positive = [
        value for group in grouped.values() for value in group.values() if value > 0
    ]
    zero_position = min(positive) / 4
    _, ax = _axes((10, max(4, 0.3 * len(labels))))
    for offset, backend in enumerate(backends):
        positions = [
            index for index, label in enumerate(labels) if backend in grouped[label]
        ]
        values = [grouped[labels[index]][backend] for index in positions]
        ax.scatter(
            [value or zero_position for value in values],
            positions,
            marker=("o", "s", "^")[offset % 3],
            label=backend,
            s=20,
        )
        for value, position in zip(values, positions, strict=True):
            if value == 0:
                ax.annotate(
                    "0",
                    (zero_position, position),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    fontsize=7,
                    va="center",
                )
    ax.axvline(1.0, color="0.3", linestyle="--", label="bound")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Largest deviation / predeclared bound in each series")
    ax.legend(frameon=False, fontsize=8)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The numbers `docs/dev/accuracy.md` quotes from the convergence records match them.

The page's figures are drawn from the records (`tests/test_accuracy_figures.py`), but
its tables and prose quote numbers by hand, and a regenerated record does not rewrite
them.
`QUOTES` lists every number the page quotes from a record: the passage of the page it
stands in, the number as printed, and how it is read from `tests/data/convergence/`.
Each passage must be on the page, and each number must equal the record's value to the
precision printed -- within half a unit of its last printed digit. So a record that
moves a quoted number fails here, and so does an edit that changes a quoted passage
without changing its entry.

Passages are compared with runs of whitespace collapsed, so a line break in the page
does not matter. A number the page quotes from a record that is not listed here is not
checked: add it with the passage. Numbers from witnesses, tests' own assertions or
other measurements stay prose (`docs/dev/accuracy.md` section 9).
"""

import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import convergence_records as cr

PAGE = Path(__file__).resolve().parents[1] / "docs" / "dev" / "accuracy.md"

EPSILON = sys.float_info.epsilon


@functools.cache
def _record(study: str) -> dict[tuple[str, str], dict[str, object]]:
    """Return a record's rows keyed by ``(series, axis)``, the axis as a string."""
    return {
        (str(row["series"]), str(row["axis"])): row for row in cr.load(study)["rows"]
    }


def at(study: str, series: str, axis: object, column: str) -> float:
    """Return one cell of a record."""
    value = _record(study)[series, str(axis)][column]
    assert isinstance(value, int | float), (study, series, axis, column)
    return float(value)


def column(study: str, series: str, name: str) -> list[float]:
    """Return one column of a series, every rung."""
    return [
        float(row[name])  # pyright: ignore[reportArgumentType]
        for (stored, _), row in _record(study).items()
        if stored == series
    ]


@dataclass(frozen=True)
class Quote:
    """One number the page quotes, where it stands and how the record gives it.

    Attributes
    ----------
    passage : str
        Text of the page containing the number, whitespace collapsed.
    printed : str
        The number as the page prints it; a substring of `passage`.
    value : callable
        Returns the record's value of the number.
    """

    passage: str
    printed: str
    value: Callable[[], float]


ALARM = ("quadrature_alarm", "three HE_1m modes")
EQUIVALENCE = "backend_equivalence"
NUMBA = "numba_cpu against numpy_ref"
NATIVE = "native against numpy_ref"
CUDA = "native_cuda against numpy_ref"
M2 = "m2_witness_metrics"
M2_REDUCED = "reduced case"
M4 = "m4_witness_metrics"
TOLERANCE = "m2_tolerance"
LUNA_FULL = "Luna, full case"
SOLPROP_FULL = "solprop, full case"

_ALARMS = (
    "**2.365e-1, 3.316e-2, 1.451e-4, 1.777e-8, 4.224e-15** at n_r = 8, 16, 24, 32, 48"
)
_N6 = "9.2e-6, 6.1e-8, 8.9e-9, 4.7e-9 at 8000, 16000, 32000, 64000 steps: order 7.2"
# The page's own multiplication sign and en dashes, which a passage must match.
_VUV = (
    "1.43× in the fundamental, 1.30× over both modes; arrival 4.730 fs against 4.622 fs"  # noqa: RUF001
)
_LUNA_FALLS = "Luna's second mode falls from 3.341 to 2.543, 2.101 and 1.893 µJ"
_CUDA_RUNGS = "9.56e-22, 7.62e-18, 5.42e-17, 6.10e-16, 9.31e-16 and 2.73e-17"
_LADDER = (
    "one linear application",
    "one step",
    "eight steps",
    "a full adaptive run",
    "a capillary with an absorber and a kink",
    "the analytic field family",
)


def _move(series: str) -> float:
    """Return the full M2 case's relative move in mode-2 energy over the rtol ladder."""
    energies = column(TOLERANCE, series, "mode2_energy")
    return (max(energies) - min(energies)) / at(
        TOLERANCE, series, "1e-9", "mode2_energy"
    )


def _vuv_cost() -> float:
    """Return the reduced M2 VUV band's error at an alarm of 1.0e-1 (n_r = 8)."""
    coarse = at("m2_alarm_cost", "reduced M2 case", 8, "vuv_energy")
    converged = at("m2_alarm_cost", "reduced M2 case", 48, "vuv_energy")
    return (coarse - converged) / converged


def _luna_fall() -> float:
    """Return Luna's mode-2 energy at its default tolerance over that at 1e-9."""
    return at(TOLERANCE, LUNA_FULL, "1e-6", "mode2_energy") / at(
        TOLERANCE, LUNA_FULL, "1e-9", "mode2_energy"
    )


QUOTES = (
    # Section 1: the cumulative operator and the plasma trapezoid.
    Quote(
        "**4.6e-11** and **1.4e-10** over twenty random intervals",
        "4.6e-11",
        lambda: max(column("phi_quadrature", "expansion", "deviation")),
    ),
    Quote(
        "**4.6e-11** and **1.4e-10** over twenty random intervals",
        "1.4e-10",
        lambda: max(column("phi_quadrature", "z table", "deviation")),
    ),
    Quote(
        "the measured floor (2.8e-8 here)",
        "2.8e-8",
        lambda: max(column("phi_quadrature", "expansion", "bound")),
    ),
    Quote(
        "9.9e-5 at 8193 samples",
        "9.9e-5",
        lambda: at("plasma_trapezoid", "energy identity", 8193, "deviation"),
    ),
    # Section 2: the transverse quadrature.
    Quote(_ALARMS, "2.365e-1", lambda: at(*ALARM, 8, "alarm")),
    Quote(_ALARMS, "3.316e-2", lambda: at(*ALARM, 16, "alarm")),
    Quote(_ALARMS, "1.451e-4", lambda: at(*ALARM, 24, "alarm")),
    Quote(_ALARMS, "1.777e-8", lambda: at(*ALARM, 32, "alarm")),
    Quote(_ALARMS, "4.224e-15", lambda: at(*ALARM, 48, "alarm")),
    Quote(
        "what an alarm of 1.0e-1 costs",
        "1.0e-1",
        lambda: at("m2_alarm_cost", "reduced M2 case", 8, "alarm"),
    ),
    Quote("the M2 VUV band wrong by 6.6e-5", "6.6e-5", _vuv_cost),
    # Section 4: milestone M1, the N = 6 ladder.
    Quote(
        _N6, "9.2e-6", lambda: at("soliton_n6", "wide band, taper off", 8000, "error")
    ),
    Quote(
        _N6, "6.1e-8", lambda: at("soliton_n6", "wide band, taper off", 16000, "error")
    ),
    Quote(
        _N6, "8.9e-9", lambda: at("soliton_n6", "wide band, taper off", 32000, "error")
    ),
    Quote(
        _N6, "4.7e-9", lambda: at("soliton_n6", "wide band, taper off", 64000, "error")
    ),
    Quote(_N6, "7.2", lambda: at("soliton_n6", "wide band, taper off", 16000, "order")),
    Quote(
        "floors at 6e-8 on the wide band",
        "6e-8",
        lambda: at("soliton_n6", "wide band, taper on", 32000, "error"),
    ),
    # Section 5: milestone M2, the order study, the tolerance ladder and the witness.
    Quote(
        "| constant | off | 6.21 |",
        "6.21",
        lambda: at("m2_order", "constant fill, no plasma", 500, "order"),
    ),
    Quote(
        "| constant | on | 5.95 |",
        "5.95",
        lambda: at("m2_order", "constant fill, plasma", 500, "order"),
    ),
    Quote(
        "| vacuum gradient | off | 1.42 |",
        "1.42",
        lambda: at("m2_order", "vacuum gradient, no plasma", 500, "order"),
    ),
    Quote(
        "| vacuum gradient | on | **1.51** |",
        "1.51",
        lambda: at("m2_order", "vacuum gradient, plasma", 500, "order"),
    ),
    Quote(
        "move the reduced case's shape by 6.7e-6",
        "6.7e-6",
        lambda: at(M2, M2_REDUCED, "two decades of tolerance, relative move", "value"),
    ),
    Quote(
        "second-mode exit energy by 2.6e-6",
        "2.6e-6",
        lambda: _move(SOLPROP_FULL),
    ),
    Quote("while solprop's moves by 2.6e-6", "2.6e-6", lambda: _move(SOLPROP_FULL)),
    Quote(
        "| fundamental, shape | 5.7e-2 |",
        "5.7e-2",
        lambda: at(M2, M2_REDUCED, "fundamental, shape", "value"),
    ),
    Quote(
        "| fundamental, content | 1.02783 |",
        "1.02783",
        lambda: at(M2, M2_REDUCED, "fundamental, content", "value"),
    ),
    Quote(
        _VUV, "1.43", lambda: at(M2, M2_REDUCED, "VUV band in the fundamental", "value")
    ),
    Quote(
        _VUV, "1.30", lambda: at(M2, M2_REDUCED, "VUV band over both modes", "value")
    ),
    Quote(_VUV, "4.730", lambda: at(M2, M2_REDUCED, "VUV arrival, solprop", "value")),
    Quote(_VUV, "4.622", lambda: at(M2, M2_REDUCED, "VUV arrival, Luna", "value")),
    Quote(
        "| second mode, energy | 1.107 |",
        "1.107",
        lambda: at(M2, M2_REDUCED, "second mode, content", "value"),
    ),
    Quote(
        _LUNA_FALLS,
        "3.341",
        lambda: 1e6 * at(TOLERANCE, LUNA_FULL, "1e-6", "mode2_energy"),
    ),
    Quote(
        _LUNA_FALLS,
        "2.543",
        lambda: 1e6 * at(TOLERANCE, LUNA_FULL, "1e-7", "mode2_energy"),
    ),
    Quote(
        _LUNA_FALLS,
        "2.101",
        lambda: 1e6 * at(TOLERANCE, LUNA_FULL, "1e-8", "mode2_energy"),
    ),
    Quote(
        _LUNA_FALLS,
        "1.893",
        lambda: 1e6 * at(TOLERANCE, LUNA_FULL, "1e-9", "mode2_energy"),
    ),
    Quote("Luna's falls by a factor of 1.76 onto it", "1.76", _luna_fall),
    # Section 6: milestone M4.
    Quote(
        "| exit energy, ours over Luna's | 1.00001 |",
        "1.00001",
        lambda: at(M4, "reduced case", "exit energy, ours over Luna's", "value"),
    ),
    Quote(
        "| fundamental, content | 1.00001 |",
        "1.00001",
        lambda: at(M4, "reduced case", "fundamental, content", "value"),
    ),
    Quote(
        "| fundamental, shape | 0.03833 |",
        "0.03833",
        lambda: at(M4, "reduced case", "fundamental, shape", "value"),
    ),
    # Section 7: backend equivalence, `numba_cpu` and Kokkos against the reference.
    *(
        Quote(f"| {row} | {bound} ε | {measured} |", printed, value)
        for row, bound, measured, printed, value in (
            (
                "one linear application",
                32,
                "4.8e-16 (not bitwise: the contraction above)",
                "4.8e-16",
                lambda: at(EQUIVALENCE, NUMBA, "one linear application", "deviation"),
            ),
            (
                "one step",
                16,
                "5.0e-17",
                "5.0e-17",
                lambda: at(EQUIVALENCE, NUMBA, "one step", "deviation"),
            ),
            (
                "eight steps",
                128,
                "2.0e-16",
                "2.0e-16",
                lambda: at(EQUIVALENCE, NUMBA, "eight steps", "deviation"),
            ),
            (
                "a full adaptive soliton run",
                1600,
                "6.2e-16",
                "6.2e-16",
                lambda: at(EQUIVALENCE, NUMBA, "a full adaptive run", "deviation"),
            ),
            (
                "a capillary with an absorber and a kink",
                3200,
                "1.4e-15",
                "1.4e-15",
                lambda: at(
                    EQUIVALENCE,
                    NUMBA,
                    "a capillary with an absorber and a kink",
                    "deviation",
                ),
            ),
        )
    ),
    *(
        # The bounds in units of epsilon, which the table prints and the record holds
        # in absolute terms.
        Quote(f"| {row} | {bound} ε |", str(bound), value)
        for row, bound, value in (
            (
                "one linear application",
                32,
                lambda: (
                    at(EQUIVALENCE, NUMBA, "one linear application", "bound") / EPSILON
                ),
            ),
            (
                "one step",
                16,
                lambda: at(EQUIVALENCE, NUMBA, "one step", "bound") / EPSILON,
            ),
            (
                "eight steps",
                128,
                lambda: at(EQUIVALENCE, NUMBA, "eight steps", "bound") / EPSILON,
            ),
            (
                "a full adaptive soliton run",
                1600,
                lambda: (
                    at(EQUIVALENCE, NUMBA, "a full adaptive run", "bound") / EPSILON
                ),
            ),
            (
                "a capillary with an absorber and a kink",
                3200,
                lambda: (
                    at(
                        EQUIVALENCE,
                        NUMBA,
                        "a capillary with an absorber and a kink",
                        "bound",
                    )
                    / EPSILON
                ),
            ),
        )
    ),
    Quote(
        "| the analytic field family | 1e-13 | 4.6e-16 |",
        "1e-13",
        lambda: at(EQUIVALENCE, NUMBA, "the analytic field family", "bound"),
    ),
    Quote(
        "| the analytic field family | 1e-13 | 4.6e-16 |",
        "4.6e-16",
        lambda: at(EQUIVALENCE, NUMBA, "the analytic field family", "deviation"),
    ),
    Quote(
        "| the reduced M2 case | 1e-10 | 3.5e-14, at identical step counts |",
        "1e-10",
        lambda: at(EQUIVALENCE, NUMBA, "the reduced M2 case", "bound"),
    ),
    Quote(
        "| the reduced M2 case | 1e-10 | 3.5e-14, at identical step counts |",
        "3.5e-14",
        lambda: at(EQUIVALENCE, NUMBA, "the reduced M2 case", "deviation"),
    ),
    Quote(
        "| the reduced M2 case, native CPU | 1e-10 | 3.5e-14, at identical step",
        "3.5e-14",
        lambda: at(EQUIVALENCE, NATIVE, "the reduced M2 case", "deviation"),
    ),
    Quote(
        "| the reduced M2 case, native CUDA | 1e-10 | 5.5534e-13, at identical step",
        "5.5534e-13",
        lambda: at(EQUIVALENCE, CUDA, "the reduced M2 case", "deviation"),
    ),
    *(
        Quote(
            _CUDA_RUNGS,
            printed,
            functools.partial(at, EQUIVALENCE, CUDA, rung, "deviation"),
        )
        for printed, rung in zip(
            ("9.56e-22", "7.62e-18", "5.42e-17", "6.10e-16", "9.31e-16", "2.73e-17"),
            _LADDER,
            strict=True,
        )
    ),
    # Section 10: free space.
    Quote(
        "fails at 256 nodes (3.06e-11 against",
        "3.06e-11",
        lambda: at(
            "spatial_accuracy", "numpy_ref: raw QDHT Gram (1)", 256, "deviation"
        ),
    ),
)
"""Every number the page quotes from a record."""


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def half_unit(printed: str) -> float:
    """Return half a unit in the last printed digit of a number as printed.

    Examples
    --------
    >>> half_unit("4.224e-15"), half_unit("0.890"), half_unit("32")
    (5e-19, 0.0005, 0.5)
    """
    mantissa, _, exponent = printed.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 0.5 * 10.0 ** (int(exponent or 0) - decimals)


def _id(quote: Quote) -> str:
    return f"{quote.printed} in {quote.passage[:40]}"


@pytest.mark.parametrize("quote", QUOTES, ids=[_id(quote) for quote in QUOTES])
def test_the_page_quotes_the_record(quote):
    page = _collapsed(PAGE.read_text())
    passage = _collapsed(quote.passage)
    assert quote.printed in passage
    assert passage in page, f"the passage is no longer on the page: {passage!r}"
    value = quote.value()
    tolerance = half_unit(quote.printed) * (1.0 + 1e-9)
    assert abs(value - float(quote.printed)) <= tolerance, (
        f"the page prints {quote.printed} where the record gives {value:.6g}"
    )


def test_a_drifted_number_fails():
    """The comparison is to the printed precision, not to the number's leading digit."""
    assert abs(4.264e-15 - at(*ALARM, 48, "alarm")) > half_unit("4.264e-15")
    assert abs(1.776e-8 - at(*ALARM, 32, "alarm")) > half_unit("1.776e-8")

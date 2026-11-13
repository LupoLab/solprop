# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEPs 18 and 27's field plasma column: its rows of the case catalogue.

**What is compared.** A field grid's plasma column -- the ionisation rate, the
running integrals of the fraction and of the free current, the loss current -- on
every route that carries one: the mode-averaged helium capillary (and its analytic
signal), the two-mode capillary, a helium/argon mixture, argon on the radial, Cartesian
and line free-space routes and the hybrid planar argon gap (the cases of
``plasma_column_cases.py``; the three spatial routes run in node chunks that leave a
remainder, so the band column meets a partial block), each under both running-integral
rules, SEP 18's band rule and the trapezoid. Native holds the free current as SEP 27's
held current; the private gated form of the original equation
(``tests/gated_source.py``) is a benchmark both backends also run. On native against
``numpy_ref``:

- ``derivative``: one nonlinear evaluation, the first stage's derivative at the
  launch, ``N(a_0)``;
- ``steps``: one, eight or ten fixed steps over the case, a row each (one step over
  the whole case is not the first of ten), with every statistic of the case's
  specification read at their end for the held current -- the electron density, the
  peak ionisation rate and SEP 18's integration diagnostics.

Every field comparison is peak-normalised (``metric="peak"``), as the bound counts
roundings per sample of the peak.

**The bound is a rounding budget fixed before any run** (``plasma_column_cases.bound``,
the equivalence ladders' bound with the band rule's wider sums): per evaluation,
``32 eps ceil(log2 n_to)`` for the transforms' depth and ``gamma(12 (2K + 1) n_to)``
for the plasma column's sums, which the band rule widens by its ``2K + 1`` taps per
integrand (one tap for the trapezoid); seven evaluations a step, added. A device build
takes the equivalence ladders' predeclared envelope where it is the larger
(``bounds.ladder_bound``).

**Statistics** (:func:`held_to_the_field`) read the same plane through the same rule.
The electron density and the peak ionisation rate are held relative to their value, to
a thousand field bounds; SEP 18's diagnostics absolutely, to a thousandth of their
strictest threshold (``bounds.INTEGRATION_AGREEMENT``: the mixture's argon barely
ionises, and its work defect is rounding noise that one ulp of a platform's
exponential moves by tens of per cent).

**The guard.** Each row's term is the plasma: ``optmat_rows.without_plasma`` zeroes
every population's density, and each guarded rung's reference must then move by more
than ``parity_ladder.DETECTION`` (a thousand) of that rung's bounds. The weak helium
launch of the capillaries has a negligible new population, so the old tests kept it
as an edge case and also seeded ten per cent pre-ionisation on every route, so that
the free-current propagation cannot pass on zeros: the pre-ionised rows carry the
guard. A cold row whose plasma cannot reach its rung says so, with what was measured.

**How to read a row.** Every row is a :func:`plasma_row` call, for example::

    plasma_row(
        name=f"column {name} {rule} pre-ionised {STEP_NAMES[steps]}",
        route=case.route,
        stepper=case.stepper,
        features=f"{rule} rule, pre-ionised",
        why=f"Pre-ionised by a tenth, {case.what} under the {rule} rule is ...",
        build=partial(
            built_column, name, rule, preionisation=PREIONISED, steps=steps
        ),
        rungs={"steps": over_steps(steps)},
        every_statistic=held_to_the_field(density_is_live=True),
    )

``name`` makes the test id (``plasma column radial band pre-ionised ten steps``);
``route``, ``stepper`` and ``features`` are the axes the row covers. ``build`` makes the
problem: here a case of ``plasma_column_cases.py`` under one rule, a tenth of its gas
ionised at the launch, marched in `steps` fixed steps. ``rungs`` names what is
compared, each with its bound: here the field after the march, within that many
steps' budget. Every statistic is then held by :func:`held_to_the_field`, the
reference's electron density required to be positive. ``why`` is the claim a failure
quotes.
"""

import dataclasses
import functools
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from backends.optmat_rows import without_plasma
from backends.plasma_column_cases import CASES, bound, column_case, options_for
from bounds import INTEGRATION_AGREEMENT
from gated_source import _gated_source
from parity_ladder import Bound, Built, Detection, Row, Statistic, StatisticRule
from solprop.stats import PLASMA_INTEGRATION_STATISTICS

# --- bounds ---------------------------------------------------------------------------

COLD_MIXTURE_DENSITY = 1e-6
"""Relative agreement of the cold-start mixture's electron density.

Without pre-ionisation the mixture's argon barely ionises, so its share of the density
rests on a near-threshold rate that platform-level rounding moves: after one step NumPy
on Linux x86-64 differs from both compiled backends by 4.4e-8 (they agree exactly with
each other, and everything agrees on Apple silicon and after eight steps). The bound
keeps a factor of about 20 over that measurement.
"""


def one_evaluation(built: Built) -> float:
    """Bound one evaluation of the stage source: the ladder's budget for one call.

    The established backend ladder accounts for scans, transforms and products; this
    independent budget is fixed before the native source is evaluated.
    """
    return bound(built.problem, built.grid, "native", 0)


def over_steps(steps: int) -> Bound:
    """Bound `steps` fixed steps: seven evaluations a step, their budgets added."""
    return lambda built: bound(built.problem, built.grid, "native", steps)


def held_to_the_field(
    *, density_at_least: float = 0.0, density_is_live: bool = False
) -> StatisticRule:
    """Return the rule a column row holds its statistics to, from its field bound.

    The electron density and the peak ionisation rate relative to each value, to a
    thousand field bounds (the density to `density_at_least` where that is larger);
    SEP 18's diagnostics absolutely, to ``INTEGRATION_AGREEMENT``. `density_is_live`
    requires the reference's electron density to be positive.
    """

    def _statistic(built: Built, name: str, field: float) -> Statistic:
        if name in PLASMA_INTEGRATION_STATISTICS:
            return Statistic(INTEGRATION_AGREEMENT, absolute=True)
        if name == "electron_density":
            return Statistic(
                max(1e3 * field, density_at_least),
                elementwise=True,
                positive=density_is_live,
            )
        return Statistic(1e3 * field, elementwise=True)

    return _statistic


# --- what a row builds and switches off -----------------------------------------------

RULES = ("band", "trapezoid")
"""The running-integral rules: SEP 18's band rule and the trapezoid."""

PREIONISED = 0.1
"""The fraction of the gas a pre-ionised row starts ionised: a free current at once."""

STEP_NAMES = {1: "one step", 8: "eight steps", 10: "ten steps"}
"""How a row's name says how many fixed steps it marches."""


@dataclasses.dataclass(frozen=True)
class _Case:
    """How a case of ``plasma_column_cases.py`` reads in a row: axes and a phrase."""

    route: str
    stepper: str
    what: str


ROUTES = {
    "mode_averaged": _Case(
        "capillary (mode-averaged)", "adams", "the mode-averaged helium capillary"
    ),
    "analytic": _Case(
        "capillary (mode-averaged)",
        "adams",
        "the mode-averaged helium capillary without its harmonic, whose Kerr term "
        "reads the analytic signal and the plasma its real part,",
    ),
    "multimode": _Case(
        "capillary (multimode)", "adams", "the two-mode helium capillary"
    ),
    "mixture": _Case(
        "capillary (mode-averaged)",
        "adams",
        "the capillary filled half with helium and half with argon, two populations,",
    ),
    "radial": _Case("radial", "rk", "argon on the radial route"),
    "free": _Case(
        "cartesian", "rk", "argon on the Cartesian route, in node chunks of seven,"
    ),
    "line": _Case("line", "rk", "argon on the line at 1 mJ, in node chunks of seven,"),
    "planar": _Case(
        "planar", "rk", "the hybrid planar argon gap, in node chunks of five,"
    ),
}
"""Each case's route, stepper and how a claim names it."""

HELIUM = ("mode_averaged", "analytic", "multimode")
"""The helium capillaries, whose launch frees no electron without pre-ionisation."""


def built_column(
    name: str,
    rule: str,
    *,
    preionisation: float | None = None,
    steps: int | None = None,
) -> Built:
    """Return ``plasma_column_cases.column_case`` as a built row, with its chunks."""
    grid, problem, run_request = column_case(
        name, rule, preionisation=preionisation, steps=steps
    )
    return Built(problem, run_request, grid, options=options_for(name))


def plasma_row(
    *,
    name: str,
    route: str,
    stepper: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a row of the field plasma column, guarded by switching the plasma off.

    Its id is ``plasma {name}``; it is compared peak-normalised, and it is medium when
    it marches. The planar case's build warns that its temporal absorber is weak
    against the walk-off over its original 0.1 m, which no row marches (the planar
    family's collar note, as in its own parity tests). `fields` are the rest of
    :class:`parity_ladder.Row`.
    """
    walk_off = (
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    )
    return Row(
        id=f"plasma {name}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": "plasma",
            "features": features,
        },
        build=build,
        switch_off=without_plasma,
        metric="peak",
        medium="steps" in fields["rungs"],
        quiet=(walk_off,) if route == "planar" else (),
        **fields,
    )


# --- the held current, one evaluation -------------------------------------------------

HELIUM_FREES_NOTHING_AT_THE_LAUNCH = (
    "The helium launch frees no electron at the launch either: switching the plasma "
    "off moves the reference's derivative by at most 7.8e-4 bounds (measured), "
    "rounding. Kept as the old test held it, the column's whole stage source "
    "assembled beside the Kerr term on a population that frees nothing; the "
    "pre-ionised helium marches carry the plasma's guard."
)
"""Why a cold helium capillary's evaluation is unguarded."""


def held_current_rows() -> Iterator[Row]:
    """Yield one evaluation of the held current on every route, under both rules.

    The cases as they are, without pre-ionisation. Measured against bounds of 2.1e-12 to
    9.8e-11: at most 9e-4 of the bound (2.5e-14, the mixture's band rule), the others
    2.9e-16 to 9.3e-15. Native's band leak, its resident buffers and its ledger are
    ``test_native_held_plasma.py``'s.
    """
    for name in CASES:
        case = ROUTES[name]
        for rule in RULES:
            yield plasma_row(
                name=f"held current {name} {rule}",
                route=case.route,
                stepper="none",
                features=f"{rule} rule, held current",
                why=f"On {case.what} native assembles the held current's stage source "
                f"under the {rule} rule -- its scans, transforms and products -- as "
                "the reference does, in one evaluation at the launch.",
                build=partial(built_column, name, rule),
                rungs={"derivative": one_evaluation},
                unguarded={"derivative": HELIUM_FREES_NOTHING_AT_THE_LAUNCH}
                if name in HELIUM
                else {},
            )


# --- the plasma column over fixed steps -----------------------------------------------


def preionised_column_rows() -> Iterator[Row]:
    """Yield the held current's march on every route, under both rules, pre-ionised.

    One step and ten fixed steps, a tenth of the gas ionised at the launch, so that the
    free-current propagation cannot pass on zeros. Measured: the field at most 0.8 % of
    its bound (1.7e-12 against 2.2e-10, the mixture's ten trapezoid steps), the density
    and peak rate at most 4.8e-7 of theirs, the diagnostics at most 5.7e-13 (the
    undershoot) against their 1e-7.
    """
    for name in CASES:
        case = ROUTES[name]
        for rule in RULES:
            for steps in (1, 10):
                yield plasma_row(
                    name=f"column {name} {rule} pre-ionised {STEP_NAMES[steps]}",
                    route=case.route,
                    stepper=case.stepper,
                    features=f"{rule} rule, pre-ionised",
                    why=f"Pre-ionised by a tenth, {case.what} under the {rule} rule "
                    f"is the reference's after {STEP_NAMES[steps]}, and so are its "
                    "electron density, peak ionisation rate and integration "
                    "diagnostics.",
                    build=partial(
                        built_column,
                        name,
                        rule,
                        preionisation=PREIONISED,
                        steps=steps,
                    ),
                    rungs={"steps": over_steps(steps)},
                    every_statistic=held_to_the_field(density_is_live=True),
                )


HELIUM_FREES_NOTHING = (
    "Without pre-ionisation the helium launch frees no electron (the reference's "
    "electron density is exactly zero after the run), so switching the plasma off "
    "moves the field by at most 7.2e-6 bounds (measured), rounding. Kept as the old "
    "test's edge case: the column runs on a population that frees nothing, and "
    "native's electron density must be exactly zero too (it is held elementwise); the "
    "pre-ionised rows carry the plasma's guard."
)
"""Why a cold helium capillary's march is unguarded."""

ARGON_BARELY_IONISES = (
    "Without pre-ionisation the mixture's argon frees 3.1e16 electrons per m^3, about "
    "3e-9 of the gas, so switching the plasma off moves the field by 6.3e-9 of its "
    "peak (measured): {moved} of this row's bounds. Kept as the old test's edge case: "
    "its electron density, held elementwise to the cold mixture's own floor, still "
    "sees the plasma (a native without it fails the row, measured), and the "
    "pre-ionised mixture rows carry the plasma's guard."
)
"""Why a cold mixture's march is guarded by less than a thousand bounds, or not."""

PLANAR_COLD_BAND_WINDOW = Detection(
    threshold=100,
    reason="the cold planar gap frees 2.9e22 electrons per m^3 over its 1e-7 m, which "
    "move the field by 8.7e-7 of its peak (measured), against the band rule's "
    "ten-step budget of 6.9e-9: 126 bounds. Its one step clears 1,259 bounds and the "
    "trapezoid's ten 1,099; the pre-ionised rows clear 8e4 or more.",
)
"""The cold planar gap's ten band steps: a guard of a hundred bounds, and why."""


def cold(case: _Case, rule: str, steps: int) -> str:
    """Return a cold row's claim: its march and statistics, without pre-ionisation."""
    return (
        f"Without pre-ionisation, {case.what} under the {rule} rule is the "
        f"reference's after {STEP_NAMES[steps]}, and so are its electron density, peak "
        "ionisation rate and integration diagnostics."
    )


def cold_column_rows() -> Iterator[Row]:
    """Yield the held current's march on every route, under both rules, cold.

    One step and ten fixed steps without pre-ionisation: the weak helium launch, the
    barely ionised mixture and argon on the spatial routes. Measured: the field at
    most 1.7e-5 of its bound (3.7e-15), the density and peak
    rate at most 4.8e-7 of theirs, the diagnostics at most 3.4e-14 (the work defect)
    against their 1e-7.
    """
    for name in HELIUM:
        case = ROUTES[name]
        for rule in RULES:
            for steps in (1, 10):
                yield plasma_row(
                    name=f"column {name} {rule} cold {STEP_NAMES[steps]}",
                    route=case.route,
                    stepper=case.stepper,
                    features=f"{rule} rule, cold",
                    why=cold(case, rule, steps),
                    build=partial(
                        built_column, name, rule, preionisation=0.0, steps=steps
                    ),
                    rungs={"steps": over_steps(steps)},
                    unguarded={"steps": HELIUM_FREES_NOTHING},
                    every_statistic=held_to_the_field(),
                )
    mixture = ROUTES["mixture"]
    # The plasma moves the field by the same 6.3e-9 of its peak on all four rows;
    # `moved` is that in each row's bounds (measured), below a thousand on all four.
    for rule, steps, moved in (
        ("band", 1, 32.6),
        ("band", 10, 3.3),
        ("trapezoid", 10, 28.8),
    ):
        yield plasma_row(
            name=f"column mixture {rule} cold {STEP_NAMES[steps]}",
            route=mixture.route,
            stepper=mixture.stepper,
            features=f"{rule} rule, cold",
            why=cold(mixture, rule, steps),
            build=partial(
                built_column, "mixture", rule, preionisation=0.0, steps=steps
            ),
            rungs={"steps": over_steps(steps)},
            unguarded={"steps": ARGON_BARELY_IONISES.format(moved=moved)},
            every_statistic=held_to_the_field(density_at_least=COLD_MIXTURE_DENSITY),
        )
    yield plasma_row(
        name="column mixture trapezoid cold one step",
        route=mixture.route,
        stepper=mixture.stepper,
        features="trapezoid rule, cold",
        why=cold(mixture, "trapezoid", 1),
        build=partial(built_column, "mixture", "trapezoid", preionisation=0.0, steps=1),
        rungs={"steps": over_steps(1)},
        detection={
            "steps": Detection(
                threshold=100, reason=ARGON_BARELY_IONISES.format(moved=286)
            )
        },
        every_statistic=held_to_the_field(density_at_least=COLD_MIXTURE_DENSITY),
    )
    for name in ("radial", "free", "line", "planar"):
        case = ROUTES[name]
        for rule in RULES:
            for steps in (1, 10):
                yield plasma_row(
                    name=f"column {name} {rule} cold {STEP_NAMES[steps]}",
                    route=case.route,
                    stepper=case.stepper,
                    features=f"{rule} rule, cold",
                    why=cold(case, rule, steps),
                    build=partial(
                        built_column, name, rule, preionisation=0.0, steps=steps
                    ),
                    rungs={"steps": over_steps(steps)},
                    detection={"steps": PLANAR_COLD_BAND_WINDOW}
                    if (name, rule, steps) == ("planar", "band", 10)
                    else {},
                    every_statistic=held_to_the_field(),
                )


# --- the private gated benchmark ------------------------------------------------------


def gated_rows() -> Iterator[Row]:
    """Yield the private gated form on every route, under both rules, pre-ionised.

    Pre-ionised, so the free current, and with it the gated integral, is live from the
    first sample rather than a rounding-level population. Every backend of the row,
    the guard's included, is built under the private switch. That native runs the
    gated form at all, rather than its held current, and that it lands nothing outside
    the band, are ``test_native_held_plasma.py``'s.
    """
    for name in CASES:
        case = ROUTES[name]
        for rule in RULES:
            yield plasma_row(
                name=f"gated {name} {rule} one step",
                route=case.route,
                stepper=case.stepper,
                features=f"{rule} rule, pre-ionised, gated benchmark",
                why=f"On {case.what} the private gated form under the {rule} rule is "
                "the reference's in one evaluation at the launch and after one "
                "fixed step.",
                bound_note="The gated form replaces the held current's two "
                "transforms and products by one filter and one trapezoid scan of the "
                "same current rows and a product with the window: at most "
                "gamma(2 width n_to + 2) of the integral's magnitudes, inside the "
                "ladder's per-call gamma(12 width n_to), so the held form's budget, "
                "fixed before the evaluation, holds it (docs/dev/testing.md section "
                "3). Measured at most 1.2e-3 of the bound (the evaluation).",
                build=partial(
                    built_column, name, rule, preionisation=PREIONISED, steps=1
                ),
                rungs={"derivative": one_evaluation, "steps": over_steps(1)},
                context=_gated_source,
            )
            yield plasma_row(
                name=f"gated {name} {rule} eight steps",
                route=case.route,
                stepper=case.stepper,
                features=f"{rule} rule, pre-ionised, gated benchmark",
                why=f"On {case.what} the private gated form under the {rule} rule is "
                "the reference's after eight fixed steps.",
                bound_note="The band-rule ladder's bound at seven source evaluations "
                "a step, fixed before the march, as for the one evaluation. Measured "
                "at most 1.5 % of it (2.6e-12, the mixture's trapezoid).",
                build=partial(
                    built_column, name, rule, preionisation=PREIONISED, steps=8
                ),
                rungs={"steps": over_steps(8)},
                context=_gated_source,
            )


# --- every row ------------------------------------------------------------------------


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every plasma column row."""
    return (
        *held_current_rows(),
        *preionised_column_rows(),
        *cold_column_rows(),
        *gated_rows(),
    )

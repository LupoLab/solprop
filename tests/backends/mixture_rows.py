# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The mixture's channels and the envelope plasma host route: their catalogue rows.

**What is compared.** Native against ``numpy_ref`` on fixed schedules of the
two-population mixture case (``tests/mixture_cases.py``: two synthetic species, each
with its Kerr term and an analytic rate, on a field grid) and of the envelope plasma
host route:

- the mixture's ladder, its first step and its first eight steps of ``dz_init``: the
  field (``steps``) and every species statistic at the launch (``launch``) and at the
  end (from ``test_mixture_ladder.py``, whose adaptive run stays there);
- three independent density profiles along z, an inert third population among them, so
  that the operator and the prefactor are z tables: sixteen steps (from
  ``test_mixture_profiles.py``);
- a pre-ionisation that follows a path along z, stepped or smooth, on the field case's
  first eight steps and on an envelope case's four: the field and the statistics at the
  end (from ``test_preionisation_paths.py``);
- an independent Raman oscillator per population, one cached kernel: sixteen steps
  (from ``test_numpy_mixture_raman.py``, which keeps the kernel's refresh count);
- argon's cycle-averaged envelope plasma, elliptically polarised, at the launch's
  amplitude and ten times it: one step and eight (from
  ``test_envelope_plasma_host.py``).

**The bounds are the old tests', fixed before native runs.** The mixture's ladder
budget is ``64 eps`` per fine sample, channel, stage and step, ``64 n_to 2 7 n eps``
(:func:`ladder_budget`); the profiles add the tables' Clenshaw evaluation,
``16 (d + 1)^2`` per stage (:func:`profiles_budget`); the Raman oscillators count four
response components (:func:`raman_budget`); the envelope paths take the envelope
geometry ladder's ``128 n_to 7 4 eps`` (:func:`envelope_budget`); the host route's one
step ``128 n_to 7 eps`` (:func:`host_budget`), and its two-plane comparison an ad hoc
``2e-12`` of the peak (:data:`HOST_PEAK`, #807).

**Where the old form could not be kept exactly.** ``species_ionisation`` was held to
``rtol=b, atol=b``: the rows hold it absolutely to ``b``, the stricter
(``geometry_rows.held_as_the_mixture``). The envelope paths and the host route's one
step were held to ``assert_allclose(rtol=b, atol=b max|ref|)``, which admits
``b (max|ref| + |ref|)`` at an element: the rows hold the stricter peak-normalised
``b max|ref|`` alone. The host route's two planes were held to ``rtol=2e-10, atol=2e-12
max|ref|``: the row holds the stricter ``2e-12`` of the peak alone.

**The guard.** Each row's term is switched off on the translated problem: the field
channels' ionisation (``geometry_rows.without_ionisation``), the density profiles frozen
at the launch (:func:`frozen_profiles`), the paths held at their launch value
(:func:`constant_paths`), the Raman oscillators (``optmat_rows.without_raman``) or the
envelope plasma (``optmat_rows.without_plasma``). Each guarded rung's reference must
then move by more than ``parity_ladder.DETECTION`` (a thousand) of its bounds.

**How to read a row.** Every row is a :func:`mixture_row` call, for example::

    mixture_row(
        name="mixture preionisation field stepped-smooth",
        route="capillary (mode-averaged)",
        response="kerr, plasma",
        features="two populations, pre-ionisation paths along z",
        why="A pre-ionisation that follows a path along z is read per stage ...",
        build=partial(field_paths, "stepped-smooth"),
        switch_off=constant_paths(field_case, FIELD_PATHS["stepped-smooth"]),
        rungs={"steps": ladder_budget(FIELD_STEPS)},
        window=FIELD_STEPS,
        every_statistic=held_as_the_mixture,
    )

``name`` makes the test id; ``route``, ``response`` and ``features`` are the axes the
row covers. ``build`` makes the problem: here the field case with its two paths and its
statistics. ``rungs`` names what is compared, with its bound: the field after the
ladder's first eight steps (``window``), and every statistic there by its rule.
``switch_off`` removes the row's term for the guard; ``why`` is the claim a failure
quotes.
"""

import dataclasses
import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends.envelope_host_cases import argon_host_case
from backends.geometry_rows import held_as_the_mixture, without_ionisation
from backends.host_cases_mixture_threads import NAMES as LADDER_NAMES
from backends.optmat_rows import without_plasma, without_raman
from backends.preionisation_cases import (
    ENVELOPE_PATHS,
    FIELD_PATHS,
    FIELD_STEPS,
    envelope_case,
    field_case,
)
from backends.preionisation_cases import NAMES as PATH_NAMES
from mixture_cases import independent_profiles, mixture_translation, raman_mixture
from parity_ladder import Built, Row, Statistic
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.problem import PropagationProblem

EPS = float(np.finfo(float).eps)
"""The machine epsilon."""

ENVELOPE_PATH_NAMES = (
    "species_ionisation",
    "species_electron_density",
    "electron_density",
)
"""The envelope path case's statistics."""

HOST_PEAK = 2e-12
"""The host route's two planes' bound, of the reference's peak: ad hoc (#807).

The old test held both planes to ``assert_allclose(rtol=2e-10, atol=2e-12 max|ref|)``,
saying only that the source's discrepancy is below 2e-10 and perturbs the field
weakly over 8 um; the row holds its absolute part alone, the stricter."""

HOST_STEPS = 8
"""The host route's fixed steps of 1 um."""

# --- the cases ------------------------------------------------------------------------


def with_statistics(built: Built, names: tuple[str, ...]) -> Built:
    """Return the build whose run reads the statistics `names`."""
    return dataclasses.replace(
        built, run_request=built.run_request.for_run(stats=StatsSpec(names=names))
    )


def ladder() -> Built:
    """Return the mixture case on its fixed ladder of ``dz_init`` steps, read fully.

    A hundred steps of 10 um over the millimetre, of which a row takes the first few
    (``window``); the statistics are ``host_cases_mixture_threads.NAMES``.
    """
    _, grid, _, _, (problem, run_request) = mixture_translation()
    stepper = run_request.stepper
    run_request = run_request.with_stepper(
        dataclasses.replace(
            stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, stepper.dz_init),
        )
    )
    return with_statistics(Built(problem, run_request, grid), LADDER_NAMES)


def profiles(*, frozen: bool = False) -> Built:
    """Return the three independent profiles' case, sixteen fixed steps.

    `frozen` holds every density at its launch value (``mixture_cases``).
    """
    _, grid, _, problem, run_request = independent_profiles(frozen=frozen)
    return Built(problem, run_request, grid)


def field_paths(name: str) -> Built:
    """Return the field case with the paths `name`, its statistics read."""
    problem, run_request, grid = field_case(FIELD_PATHS[name])
    return with_statistics(Built(problem, run_request, grid), PATH_NAMES)


def envelope_paths(name: str) -> Built:
    """Return the envelope case with the paths `name`, its three statistics read."""
    problem, run_request, grid = envelope_case(ENVELOPE_PATHS[name])
    assert problem.flags.n_channels == 2
    return with_statistics(Built(problem, run_request, grid), ENVELOPE_PATH_NAMES)


def raman_components() -> Built:
    """Return the Raman mixture on sixteen fixed steps over its millimetre."""
    grid, problem, run_request = raman_mixture()
    steps = 16
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
        )
    )
    return Built(problem, run_request, grid)


def host(amplitude: float) -> Built:
    """Return the argon host case at `amplitude` times its launch, eight steps of 1 um.

    Saved at 1 um and at 8 um.
    """
    problem, run_request, grid = argon_host_case()
    problem = dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial, field0=problem.initial.field0 * amplitude
        ),
    )
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=FixedSchedule.uniform(0.0, HOST_STEPS * 1e-6, 1e-6),
        )
    )
    return Built(problem, run_request, grid)


def ionises_a_fifth(built: Built) -> bool:
    """Return whether the launch's own pulse ionises argon past a fifth.

    The old test's witness that the run makes substantial new electrons, beside the
    prescribed density's: the response's fraction at the window's end, on the launch
    synthesised to the physical field.
    """
    problem, grid = built.problem, built.grid
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    modal = np.fft.ifft(problem.initial.field0, axis=-1)
    physical = (problem.transform.synthesis @ modal.reshape(1, -1)).reshape(
        2, grid.n_to
    )
    response = EnvelopePlasmaResponse(
        "Ar", assembly.channels[0].rate, preionisation=0.1
    )
    return bool(response.ionised_fraction(physical, dt=grid.dt_fine)[-1] > 0.2)


# --- the bounds -----------------------------------------------------------------------


def ladder_budget(steps: int) -> Callable[[Built], float]:
    """Return the ladder's budget over `steps`: ``64 n_to 2 7 steps eps``.

    Seven stages a step, each with two FFTs, a cubic evaluation and two accumulated
    current integrals per channel: 64 eps per fine sample, per channel, per stage.
    """

    def bound(built: Built) -> float:
        return 64 * built.grid.n_to * 2 * 7 * steps * EPS

    return bound


def profiles_budget(built: Built) -> float:
    """Return the profiles' budget: ``(16 (d + 1)^2 + 64 n_to 2) 7 16 eps``.

    The two-channel ladder's 64 eps a sample beside the Clenshaw evaluation of the
    tables of degree ``d`` (the operator's and the prefactor's largest), ``8 (d +
    1)^2`` eps each, at each of sixteen seven-stage steps.
    """
    problem = built.problem
    prefactor = problem.responses.prefactor
    assert prefactor is not None
    degree = max(int(problem.linear.phi_degrees.max()), int(prefactor.degrees.max()))
    return (16 * (degree + 1) ** 2 + 64 * built.grid.n_to * 2) * 7 * 16 * EPS


def raman_budget(built: Built) -> float:
    """Return the Raman mixture's budget: ``64 n_to 4 7 16 eps``.

    Two component kernels, each with one convolution and two plasma exposures: the
    channel ladder's 64 eps a sample summed over the four response components, at each
    of sixteen seven-stage steps.
    """
    return 64 * built.grid.n_to * 4 * 7 * 16 * EPS


def envelope_budget(built: Built) -> float:
    """Return the envelope paths' budget: ``128 n_to 7 4 eps``.

    The envelope geometry ladder's (``geometry_rows``): the polynomial rate's integrals
    converge to rounding, and transforms, two-channel scans and projections contribute
    under 128 eps a sample a stage, over four steps.
    """
    return 128 * built.grid.n_to * 7 * 4 * EPS


def host_budget(built: Built) -> float:
    """Return the host route's one-step budget: ``128 n_to 7 eps``.

    One seven-stage step of temporal transforms and modal projection sums; the source's
    angular error is bounded apart by its independent witnesses.
    """
    return 128 * built.grid.n_to * 7 * EPS


def host_peak(built: Built) -> float:
    """Return :data:`HOST_PEAK`, the host route's two planes' ad hoc bound."""
    del built
    return HOST_PEAK


def held_relative(built: Built, name: str, bound: float) -> Statistic:
    """Hold a statistic relative at every element to the rung's bound."""
    del built, name
    return Statistic(bound, elementwise=True)


# --- the switch-offs ------------------------------------------------------------------


def frozen_profiles(problem: PropagationProblem) -> PropagationProblem:
    """Return the profiles' case rebuilt with every density held at its launch value."""
    rebuilt = profiles(frozen=True).problem
    assert rebuilt.shape == problem.shape
    return rebuilt


def constant_paths(
    build: Callable[[Any], tuple[PropagationProblem, Any, Any]], paths: tuple[Any, ...]
) -> Callable[[PropagationProblem], PropagationProblem]:
    """Return a switch-off rebuilding a path case with each path at its launch value."""

    def switch(problem: PropagationProblem) -> PropagationProblem:
        held = tuple(float(path(np.array([0.0]))[0]) for path in paths)
        rebuilt = build(held)[0]
        assert rebuilt.shape == problem.shape
        return rebuilt

    return switch


# --- the rows -------------------------------------------------------------------------


def mixture_row(
    *,
    name: str,
    route: str,
    stepper: str = "rk",
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a mixture row; it marches, so it is medium.

    Its id is `name`; `fields` are the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=name,
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium=True,
        **fields,
    )


def _ladder_rows() -> tuple[Row, ...]:
    """Return the mixture ladder's two rows: its first step and its first eight."""
    return tuple(
        mixture_row(
            name=f"mixture ladder {label}",
            route="capillary (mode-averaged)",
            response="kerr, plasma",
            features="two synthetic populations, every species statistic",
            why="Two independent plasma populations, each with its Kerr term, are "
            f"the reference's over the ladder's {label}, and so is every species "
            "statistic at the launch and there.",
            bound_note="64 eps a fine sample, channel, stage and step. The fractions "
            "were held to rtol = atol = the budget and are now held absolutely to it.",
            build=ladder,
            switch_off=without_ionisation,
            rungs={
                "launch": ladder_budget(steps),
                "steps": ladder_budget(steps),
            },
            window=steps,
            every_statistic=held_as_the_mixture,
        )
        for label, steps in (("first step", 1), ("first eight steps", 8))
    )


def _path_rows() -> tuple[Row, ...]:
    """Return the pre-ionisation paths' rows: the field case and the envelope case."""
    fields = tuple(
        mixture_row(
            name=f"mixture preionisation field {name}",
            route="capillary (mode-averaged)",
            response="kerr, plasma",
            features="two populations, pre-ionisation paths along z",
            why="A pre-ionisation that follows a path along z is read per stage as "
            "the reference reads it: the field and every species statistic after the "
            "ladder's first eight steps.",
            bound_note="The mixture ladder's budget, unchanged: a path changes only "
            "the per-stage scalar each backend reads from the same table. The "
            "fractions are held absolutely to it.",
            build=partial(field_paths, name),
            switch_off=constant_paths(field_case, FIELD_PATHS[name]),
            rungs={"steps": ladder_budget(FIELD_STEPS)},
            window=FIELD_STEPS,
            every_statistic=held_as_the_mixture,
        )
        for name in sorted(FIELD_PATHS)
    )
    envelopes = tuple(
        mixture_row(
            name=f"mixture preionisation envelope {name}",
            route="capillary (envelope)",
            response="envelope plasma",
            features="two populations, pre-ionisation paths along z",
            why="On the envelope route a pre-ionisation path along z is read per "
            "stage as the reference reads it: the field and the populations' "
            "statistics after four fixed steps.",
            bound_note="The envelope geometry ladder's 128 eps a sample a stage, over "
            "four steps. The field was held to rtol = b, atol = b max|ref|; the row "
            "holds the stricter b max|ref| alone.",
            build=partial(envelope_paths, name),
            switch_off=constant_paths(envelope_case, ENVELOPE_PATHS[name]),
            rungs={"steps": envelope_budget},
            every_statistic=held_relative,
            metric="peak",
        )
        for name in sorted(ENVELOPE_PATHS)
    )
    return (*fields, *envelopes)


def _host_rows() -> tuple[Row, ...]:
    """Return the envelope plasma host route's rows."""
    planes = tuple(
        mixture_row(
            name=f"mixture envelope host argon amplitude {amplitude:g}",
            route="capillary (envelope)",
            response="envelope plasma",
            features="cycle-averaged ADK argon, ellipticity 0.35, two planes",
            why="Argon's cycle-averaged envelope plasma, elliptically polarised, is "
            f"the reference's after one step and after eight at {amplitude:g} times "
            "the launch.",
            bound_note="Ad hoc (#807): the old test held both planes to rtol = 2e-10, "
            "atol = 2e-12 max|ref|; the row holds the stricter 2e-12 of the peak "
            "alone.",
            build=partial(host, amplitude),
            switch_off=without_plasma,
            rungs={
                "one_step": host_peak,
                "steps": host_peak,
            },
            metric="peak",
            expect={"": ionises_a_fifth} if amplitude > 1 else {},
        )
        for amplitude in (1.0, 10.0)
    )
    return (
        mixture_row(
            name="mixture envelope host argon one step",
            route="capillary (envelope)",
            response="envelope plasma",
            features="cycle-averaged ADK argon, ellipticity 0.35",
            why="Native executes the envelope plasma host route's sources as the "
            "reference does over one step.",
            bound_note="One seven-stage step's transforms and projection sums, 128 eps "
            "a sample a stage. Held to rtol = b, atol = b max|ref| before; the row "
            "holds the stricter b max|ref| alone.",
            build=partial(host, 1.0),
            switch_off=without_plasma,
            rungs={"one_step": host_budget},
            metric="peak",
        ),
        *planes,
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the mixture and the envelope plasma host route."""
    return (
        *_ladder_rows(),
        mixture_row(
            name="mixture independent profiles",
            route="capillary (pressure gradient)",
            response="kerr, plasma",
            features="three independent density profiles, an inert population, "
            "z tables",
            why="Three populations whose densities vary independently along z, one "
            "inert, are the reference's over sixteen fixed steps.",
            bound_note="The two-channel ladder's 64 eps a sample beside the tables' "
            "Clenshaw evaluation, 8 (d + 1)^2 eps each, at every stage.",
            build=profiles,
            switch_off=frozen_profiles,
            rungs={"steps": profiles_budget},
        ),
        *_path_rows(),
        mixture_row(
            name="mixture raman components",
            route="capillary (mode-averaged)",
            response="kerr, raman, plasma",
            features="an independent Raman oscillator per population, one kernel",
            why="One Raman convolution carrying each population's own oscillator is "
            "the reference's over sixteen fixed steps.",
            bound_note="The channel ladder's 64 eps a sample summed over four response "
            "components.",
            build=raman_components,
            switch_off=without_raman,
            rungs={"steps": raman_budget},
            vacuous="The two oscillators move the reference's sixteen steps by 43 "
            "bounds (measured), under the least guard of a hundred: at these "
            "amplitudes the Raman share of the run is 1.4e-7 of the field, and the "
            "kernel's own refresh count is what test_numpy_mixture_raman.py holds.",
        ),
        *_host_rows(),
    )

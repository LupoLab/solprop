# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 3's optmat materials: their rows of the case catalogue.

**What is compared.** A run under :class:`~solprop.physdata.selection.MaterialSelection`
reaches the backends as the arrays any other run carries -- index and wall tables,
carrier Kerr coefficients, rate tables, Raman rows with their decay paths -- and no
backend reads optmat. Each row is one of the fifteen runs of ``optmat_parity_cases.py``
(the handoffs of SEP 3's Phases 1b, 1c and 1d), built through the driver a user would
call, ten fixed steps over a short guide, and it climbs one ladder (``AGENTS.md``,
backends) on native against ``numpy_ref``:

- ``linear``: one step of the launch scaled by ``parity_ladder.LINEAR_SCALE``, so that
  every pass of the step runs and the cubic and delayed terms are ``1e-40`` of the
  linear one: what the rung compares is the operator built from optmat's indices;
- ``derivative``: one nonlinear evaluation, the first stage's derivative at the
  launch, ``N(a_0)``;
- ``one_step`` and ``steps``: one step, and eight steps (``window=8``) with every
  statistic of the case's specification read at their end;
- ``run``: the complete short run, the driver's own ``Simulation.run`` through the
  output layer, with its saved planes, per-plane statistics and periodic rows (the
  assembled tensor bundle runs at the seam).

Every comparison is peak-normalised (``metric="peak"``, ``max |ours - theirs| / max
|theirs|``), as the bounds count roundings per sample of the peak.

**The bounds are rounding budgets written before any measurement.** One stage of a
capillary case is the pipeline of ``tests/backends/test_vector_raman_routes.py`` --
transforms of length ``n_to``, the node synthesis and projection over ``n_nodes``
nodes, the plasma's cumulative scan along ``n_to`` -- whose rounding that suite bounds
at ``256 eps`` per fine sample and node. The Raman kernel adds a sum of ``n_rows``
damped rows at every sample of its doubled axis, one multiply and one add per row,
relative to the sum of the rows' magnitudes: ``n_rows`` more roundings per sample. So
one stage's two backends are within ``e_stage = 256 eps (n_to + n_nodes + n_rows)`` of
each other, peak-normalised (:func:`one_stage`). A rung of ``k`` stages accumulates
``k e_stage``, carried by at most ``exp(Phi_k)`` through the nonlinear half, with
``Phi_k = z_k ||N(a_0)|| / ||a_0||`` measured on the reference at the launch, the growth
factor of ``tests/general_parity.py`` (:func:`growth`). A step is seven stages. These
are the bounds SEP 3's cost study recorded beside its measurements.

**Statistics** (:func:`statistic`). A statistic is a functional of the field. Every
statistic quadratic in it (energy, powers, intensities, widths read off a power trace)
moves by at most ``FIELD_STATISTIC`` times the field's relative deviation: two for the
square, and two again for a width's crossing of a half maximum. A plasma statistic
moves through the rate: by ``2 S`` with ``S = d ln W / d ln F`` taken where the rate
is a thousandth of the run's peak rate (``bounds.rate_log_sensitivity``), which bounds
``S`` over the part of the pulse that ionises, since ``S`` falls as the field rises.
SEP 18's run diagnostics are held absolutely (``bounds.INTEGRATION_AGREEMENT``).

**The guard.** Each row's term is switched off on the translated problem -- the
plasma populations' densities zeroed (the same derivative, to the bit, as the case
rebuilt without plasma, the old guard), a Raman leaf's rows zeroed, the Kerr
coefficient zeroed, the silver wall's operator replaced by silica's -- and each guarded
rung's reference must then move by more than ``parity_ladder.DETECTION`` (a thousand)
of that rung's bounds: a backend that dropped the term misses the reference by a
thousand bounds. Every guarded rung clears it by 1.8e5 bounds or more (measured), so
none lowers it. Where the term cannot reach a rung -- the ``linear`` rung of a
nonlinear term, the launch's derivative of the wall -- the rung is unguarded and says
why it is kept.

**How to read a row.** Every row is an :func:`optmat_row` call, for example::

    optmat_row(
        case="ar_field",
        route="capillary",
        stepper="adams",
        response="plasma",
        features="field grid",
        why="Argon's optmat rate on a field grid: ...",
        switch_off=without_plasma,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
    )

``case`` names the run of ``optmat_parity_cases.py`` the row builds and makes its test
id (``optmat ar_field``); ``route``, ``stepper``, ``response`` and ``features`` are the
axes it covers. ``rungs`` names what is compared, each with its bound: the operator's
step (seven stages, no growth), one stage, then ``k`` steps grown by ``exp(Phi_k)``;
``window=8`` cuts the ``steps`` rung to the schedule's first eight steps, whose bound
``grown_over(8)`` is. ``switch_off`` removes the row's term for the guard;
``unguarded`` names the rungs it cannot reach. ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from bounds import INTEGRATION_AGREEMENT, rate_log_sensitivity
from optmat_parity_cases import build_case, variant
from parity_ladder import Bound, Built, Row, Statistic, reference
from solprop import stats
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.glasses import OptmatGlass
from solprop.translate.ionisation import DEFAULT_RTOL
from solprop.translate.problem import PropagationProblem

# --- bounds ---------------------------------------------------------------------------

SAMPLE_CONSTANT = 256
"""Roundings per fine sample and node of one stage (``test_vector_raman_routes``)."""

STAGES = 7
"""Stages of one Dormand-Prince step."""

FIELD_STATISTIC = 4.0
"""Relative sensitivity of a field statistic to the field's relative deviation."""

PLASMA_STATISTICS = frozenset(
    {
        "electron_density",
        "peak_ionisation_rate",
        "ionisation",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_ionisation_energy",
    }
)
"""Statistics that move through an ionisation rate."""


def one_stage(built: Built) -> float:
    """Return one stage's budget, ``e_stage = 256 eps (n_to + n_nodes + n_rows)``.

    ``n_rows`` counts the Raman rows the problem carries, summed over its leaves.
    """
    problem = built.problem
    rows = sum(term.n_terms for term in problem.responses.raman_terms)
    return (
        SAMPLE_CONSTANT
        * float(np.finfo(float).eps)
        * (problem.grids.n_to + problem.transform.n_nodes + rows)
    )


def one_step_of_the_operator(built: Built) -> float:
    """Bound the linear rung: a step's seven stages, with no nonlinear growth."""
    return STAGES * one_stage(built)


def growth(built: Built, length: float) -> float:
    """Return ``exp(Phi)``, ``Phi = length ||N(a_0)|| / ||a_0||`` on the reference."""
    launch = built.problem.initial.field0
    derivative = reference(built, "derivative")
    return math.exp(length * float(np.linalg.norm(derivative) / np.linalg.norm(launch)))


def grown_over(steps: int) -> Bound:
    """Bound the schedule's first `steps` steps: their stages, grown by ``exp(Phi)``."""

    def _bound(built: Built) -> float:
        schedule = built.run_request.stepper.schedule
        assert schedule is not None
        length = float(schedule.edges[steps])
        return steps * STAGES * one_stage(built) * growth(built, length)

    return _bound


def grown_over_the_schedule(built: Built) -> float:
    """Bound the complete run: every step of the schedule, grown over its length."""
    schedule = built.run_request.stepper.schedule
    assert schedule is not None
    length = float(schedule.edges[-1])
    return len(schedule.steps) * STAGES * one_stage(built) * growth(built, length)


def rate_sensitivity(built: Built, peak_rate: float) -> float:
    """Return the largest ``d ln W / d ln F`` of the row's rates where they ionise.

    Taken where each rate is a thousandth of the run's peak rate; one when the row has
    no plasma or did not ionise.
    """
    simulation = built.simulation
    if simulation is None or not (peak_rate > 0.0):
        return 1.0
    rates = [
        response.rate
        for response in simulation.responses
        if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse)
    ]
    sensitivities = [1.0]
    for rate in rates:
        # optmat's rate declares its threshold and maximum field (SEP 3, Phase 1d).
        assert isinstance(rate, OptmatRate)
        low, high = float(rate.threshold), float(rate.maximum_field)
        target = 1e-3 * peak_rate
        if float(rate(high)) <= target or float(rate(low)) >= target:
            continue
        sensitivities.append(rate_log_sensitivity(rate, target, (low, high)))
    return max(sensitivities)


def statistic(built: Built, name: str, field: float) -> Statistic:
    """Return a statistic's bound from its rung's field bound ``field``.

    A plasma statistic's peak rate is the reference's complete run's.
    """
    base = name.removeprefix("rows/")
    if base in stats.PLASMA_INTEGRATION_STATISTICS:
        return Statistic(INTEGRATION_AGREEMENT, absolute=True)
    if base in PLASMA_STATISTICS:
        peaks = [
            float(np.nanmax(values))
            for key, values in reference(built, "run").statistics.items()
            if key.removeprefix("rows/").endswith("peak_ionisation_rate")
            and np.size(values)
        ]
        sensitivity = rate_sensitivity(built, max(peaks, default=0.0))
        return Statistic(2.0 * sensitivity * field)
    return Statistic(FIELD_STATISTIC * field)


# --- what a row builds, switches off and expects ------------------------------------


def built_case(name: str) -> Built:
    """Return one case of ``optmat_parity_cases.py`` as a built row."""
    case = build_case(name)
    return Built(
        case.problem,
        case.run_request,
        case.grid,
        modes=case.modes,
        simulation=case.simulation,
    )


def _frozen(array: np.ndarray) -> np.ndarray:
    """Return `array` read-only, as every array crossing the seam is."""
    array.flags.writeable = False
    return array


def _scaled(table: Any, factor: float) -> Any:
    """Return a Chebyshev table with its every coefficient multiplied by `factor`."""
    return dataclasses.replace(table, coefficients=_frozen(table.coefficients * factor))


def with_plasma_density(
    problem: PropagationProblem, factor: float
) -> PropagationProblem:
    """Return the problem with every plasma population's density scaled by `factor`.

    The density multiplies the whole plasma response (its free electrons, currents and
    ionisation loss), field or envelope.
    """
    responses = problem.responses
    channels = tuple(
        dataclasses.replace(channel, density=_scaled(channel.density, factor))
        for channel in responses.plasma_channels
    )
    envelope = responses.envelope_plasma
    if envelope is not None:
        envelope = dataclasses.replace(
            envelope,
            channels=tuple(
                dataclasses.replace(channel, density=_scaled(channel.density, factor))
                for channel in envelope.channels
            ),
        )
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            responses, plasma_channels=channels, envelope_plasma=envelope
        ),
    )


def without_plasma(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with no plasma: every population's density zeroed."""
    return with_plasma_density(problem, 0.0)


def with_raman_rows(problem: PropagationProblem, factor: float) -> PropagationProblem:
    """Return the problem with every Raman leaf's rows scaled by `factor`."""
    responses = problem.responses
    leaves = tuple(
        dataclasses.replace(leaf, undamped=_frozen(leaf.undamped * factor))
        for leaf in responses.raman_terms
    )
    return dataclasses.replace(
        problem, responses=dataclasses.replace(responses, raman_terms=leaves)
    )


def without_raman(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with every Raman leaf's rows zeroed."""
    return with_raman_rows(problem, 0.0)


def with_kerr(problem: PropagationProblem, factor: float) -> PropagationProblem:
    """Return the problem with its compiled Kerr coefficient scaled by `factor`.

    The mixture's populations' carrier coefficients are summed into the term's density
    path, which the coefficient multiplies.
    """
    responses = problem.responses
    kerr = responses.kerr_coefficients
    assert kerr is not None
    scaled = dataclasses.replace(kerr, coefficient=kerr.coefficient * factor)
    return dataclasses.replace(
        problem, responses=dataclasses.replace(responses, kerr_coefficients=scaled)
    )


def without_kerr(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with its Kerr term zeroed."""
    return with_kerr(problem, 0.0)


@functools.cache
def _silica_walled() -> PropagationProblem:
    """Return the silver-walled case rebuilt with optmat's silica, its default wall."""
    return variant("silver_wall", cladding=None).problem


def with_a_silica_wall(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the operator of the same capillary walled in silica.

    A wall is not a field of the problem but enters it through the guide's index and
    loss, its linear operator: the capillary is rebuilt with optmat's silica, its
    default cladding, and its operator replaces the silver wall's.
    """
    silica = _silica_walled()
    assert silica.shape == problem.shape
    return dataclasses.replace(problem, linear=silica.linear)


THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE = (
    "At 1e-20 of the launch the row's nonlinear term is 1e-40 of the linear one and "
    "moves this rung by nothing (measured: exactly zero, or 4e-34 bounds); it is kept "
    "as the ladder's one check of the linear operator built from optmat's indices on "
    "its own, which moves the scaled launch over the step by 1.3e5 to 4.6e8 of these "
    "bounds (measured on every row), so a backend that dropped the operator fails it."
)
"""Why a nonlinear term's row keeps its ``linear`` rung unguarded."""


def _leaf(built: Built) -> Any:
    """Return the row's one Raman leaf."""
    (leaf,) = built.problem.responses.raman_terms
    return leaf


def _decay_rates_at_the_end(built: Built) -> np.ndarray:
    """Return the leaf's decay rates on the run's last plane, one per row group."""
    end = built.run_request.stepper.save_planes[-1]
    return np.array([float(table(end)) for table in _leaf(built).decay_rates])


def _mixes_isotropic_and_rank_two_rows(built: Built) -> bool:
    """The leaf holds isotropic ``(1, 0)`` and rank-two ``(1/4, 3/4)`` rows."""
    weights = {tuple(row) for row in _leaf(built).tensor_weights}
    return weights == {(1.0, 0.0), (0.25, 0.75)}


def leaf_of(rows: int, *, decays_vanish_at_the_end: bool) -> Callable[[Built], bool]:
    """Return the predicate: the leaf of Phase 1c's handoff, as the parity run carries.

    `rows` rows, isotropic and rank-two rows mixed, and the decay paths at the run's
    end: exactly zero with the leaf's density there (N2's and O2's groups at exact
    vacuum), or all positive (H2's Dicke cap, finite at vacuum, and the floor's
    small rates).
    """

    def the_leaf_is_the_handoff(built: Built) -> bool:
        leaf = _leaf(built)
        rates = _decay_rates_at_the_end(built)
        if decays_vanish_at_the_end:
            end = built.run_request.stepper.save_planes[-1]
            at_the_end = (
                leaf.density is not None
                and float(leaf.density(end)) == 0.0
                and bool(np.all(rates == 0.0))
            )
        else:
            at_the_end = bool(np.all(rates > 0.0))
        return (
            leaf.n_terms == rows
            and _mixes_isotropic_and_rank_two_rows(built)
            and at_the_end
        )

    the_leaf_is_the_handoff.__name__ = (
        f"a_leaf_of_{rows}_rows_mixed_whose_decay_rates_"
        + ("vanish" if decays_vanish_at_the_end else "stay_positive")
        + "_at_the_end"
    )
    return the_leaf_is_the_handoff


def the_peak_rate_is_the_held_rate_at_every_plane(built: Built) -> bool:
    """Every plane's peak rate on the reference is the held rate ``W(E_m)``.

    The field exceeds the model's maximum field at each of them, and the run held the
    rate there; the table is within its build tolerance (``DEFAULT_RTOL``) of direct
    evaluation.
    """
    assert built.simulation is not None
    (rate,) = [
        response.rate
        for response in built.simulation.responses
        if isinstance(response, PlasmaResponse)
    ]
    assert isinstance(rate, OptmatRate)
    held = float(rate(rate.maximum_field))
    peaks = reference(built, "run").statistics["peak_ionisation_rate"]
    return float(rate(2.0 * rate.maximum_field)) == held and bool(
        np.all(np.abs(peaks - held) <= DEFAULT_RTOL * abs(held))
    )


def one_kerr_term_per_population_each_its_own(built: Built) -> bool:
    """Helium and argon each carry a Kerr term at the carrier, and they differ."""
    assert built.simulation is not None
    terms = {
        response.species: response.chi3
        for response in built.simulation.responses
        if isinstance(response, KerrResponse)
    }
    return set(terms) == {"He", "Ar"} and terms["He"] != terms["Ar"]


def walled_by_optmats_silica(modes: tuple[Any, ...]) -> bool:
    """Every mode is walled by optmat's recommended silica, its default cladding."""
    return all(isinstance(mode.cladding, OptmatGlass) for mode in modes) and {
        mode.cladding.source for mode in modes
    } == {"fit/SiO2"}


def walled_by_optmats_silver_table(modes: tuple[Any, ...]) -> bool:
    """Every mode is walled by optmat's silver table."""
    return {mode.cladding.source for mode in modes} == {"table/Ag"}


def one_leaf_without_a_decay_path(leaves: tuple[Any, ...]) -> bool:
    """One leaf, the bundle's sampled rows, with no decay path or a zero one."""
    (leaf,) = leaves
    return leaf.decay_rates == () or all(
        float(np.max(np.abs(table.coefficients))) == 0.0 for table in leaf.decay_rates
    )


def every_row_starts_at_a_non_zero_sample(built: Built) -> bool:
    """Every row of the tensor bundle, isotropic and rank-two, starts non-zero."""
    return bool(np.all(_leaf(built).undamped[:, 0] != 0.0))


def optmat_row(
    *,
    case: str,
    route: str,
    stepper: str,
    response: str,
    features: str,
    switch_off: Callable[[PropagationProblem], PropagationProblem],
    **fields: Any,
) -> Row:
    """Return the row of one optmat case, compared peak-normalised.

    Its id is ``optmat {case}``. Every statistic of the case's specification is read at
    the end of the ``steps`` and ``run`` rungs, bounded by :func:`statistic`. Medium
    tier: ten fixed steps, a second or two with the translation. The cases' builds warn
    on purpose where a gas fit leaves its qualified band or a Raman response its
    validated densities, and ``optmat_parity_cases.build_case`` quiets them; their runs
    warn nowhere but at argon's held rate. `fields` are the rest of
    :class:`parity_ladder.Row`.
    """
    return Row(
        id=f"optmat {case}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=functools.partial(built_case, case),
        switch_off=switch_off,
        metric="peak",
        every_statistic=statistic,
        medium=True,
        **fields,
    )


# --- optmat's ionisation rates --------------------------------------------------------


def plasma_rows() -> Iterator[Row]:
    """Yield argon and nitrogen under ``MaterialSelection()``, field and envelope.

    And argon launched above its rate model's maximum field, so that the table's hold
    is inside the run (the handoff of Phase 1d). On an envelope grid the reference's
    plasma is the cycle quadrature on the host, the largest share of these rows' time.

    Re-measured for the catalogue against field bounds of 2.2e-11 to 3.5e-8 and
    statistic bounds of 5.7e-9 to 7.4e-7: the worst field rung 6.1e-13 (``ar_hold``'s
    complete run), the others at most 4.9e-15 (the first stage's derivative on
    ``ar_envelope``), and the worst statistic 1.1e-13 (``ar_hold``'s peak power), the
    others at most 4.9e-14 (``quadrature_error``); the old modules quoted 2.1e-15 and
    6.1e-14. The plasma's share of the first stage's derivative is 1.0 to 1.13 of its
    peak (re-measured; the old guard's docstring quoted 1e-4 to 1e-1), against stage
    bounds of 2.2e-11 to 9.2e-11.
    """
    for grid in ("field", "envelope"):
        yield optmat_row(
            case=f"ar_{grid}",
            route="capillary",
            stepper="adams",
            response="plasma",
            features=f"{grid} grid",
            why=f"Argon's optmat ionisation rate on a {grid} grid: the plasma's "
            "ionisation, currents and loss are the reference's in one evaluation, "
            "one step, eight steps with every statistic and the driver's complete run.",
            switch_off=without_plasma,
            rungs={
                "linear": one_step_of_the_operator,
                "derivative": one_stage,
                "one_step": grown_over(1),
                "steps": grown_over(8),
                "run": grown_over_the_schedule,
            },
            window=8,
            unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        )
    yield optmat_row(
        case="ar_hold",
        route="capillary",
        stepper="adams",
        response="plasma",
        features="field grid, the rate table's hold",
        why="Argon launched at 1.06e11 V/m, above its rate model's maximum field of "
        "8.6e10 V/m: the rate table holds its last value inside the run, at every "
        "plane, and the run that holds it is the reference's.",
        bound_note="The family's largest departures, still eight orders inside: "
        "6.1e-13 on the complete run against 3.5e-8 and 1.1e-13 on its peak power "
        "against 1.4e-7 (measured). Over the other cases' 40 um steps this field "
        "ionises past what a fixed step holds, so its ten steps cover 40 um.",
        switch_off=without_plasma,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={"": the_peak_rate_is_the_held_rate_at_every_plane},
        # The driver says the run reached the rate's maximum field: the row's point.
        quiet=("ignore::solprop.exceptions.MaterialRangeWarning",),
    )
    for grid in ("field", "envelope"):
        yield optmat_row(
            case=f"n2_{grid}",
            route="capillary",
            stepper="adams",
            response="plasma",
            features=f"{grid} grid, gradient, beside a Raman leaf",
            why=f"Nitrogen on a gradient from 1.5 to 0.3 bar on a {grid} grid, its "
            "optmat rate beside its 72-row Raman leaf: the plasma is the reference's "
            "at every rung.",
            switch_off=without_plasma,
            rungs={
                "linear": one_step_of_the_operator,
                "derivative": one_stage,
                "one_step": grown_over(1),
                "steps": grown_over(8),
                "run": grown_over_the_schedule,
            },
            window=8,
            unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        )


# --- optmat's gas Raman responses ----------------------------------------------------


def raman_rows() -> Iterator[Row]:
    """Yield the gas Raman leaves of SEP 3's Phase 1c handoff.

    Dry air's 126-row composition leaf on a pressure gradient to the vacuum floor and,
    on density paths to exact vacuum, with N2's and O2's decay rates zero on the last
    plane; hydrogen at constant density and under the reference-density shift with its
    Dicke cap finite at vacuum; nitrogen's 72 isotropic and rank-two rows on two
    polarisations. The Raman rows enter the bounds as one more rounding per row and
    sample.

    Re-measured for the catalogue against field bounds of 2.6e-11 to 6.9e-9 and
    statistic bounds of 3.0e-9 to 2.7e-8: the worst field rung 6.0e-16 (the first
    stage's derivative on ``air_vacuum``) and the worst statistic 5.0e-14
    (``quadrature_error`` of ``air_floor``'s complete run); the old module quoted
    7.1e-16 and 2.7e-14. The leaf's share of the first stage's derivative is 0.16 to
    0.43 of its peak.
    """
    yield optmat_row(
        case="air_floor",
        route="capillary",
        stepper="adams",
        response="Raman",
        features="field grid, gradient to the vacuum floor, composition leaf",
        why="Dry air's 126-row composition leaf, its amplitude the summed density of "
        "three labels, on a pressure gradient from 1.5 bar to the vacuum floor: the "
        "leaf's response is the reference's at every rung.",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={"": leaf_of(126, decays_vanish_at_the_end=False)},
    )
    yield optmat_row(
        case="air_vacuum",
        route="capillary",
        stepper="adams",
        response="Raman",
        features="field grid, density paths to exact vacuum, the z table",
        why="The same leaf with every population falling linearly to exact vacuum, "
        "where N2's and O2's decay rates are zero on the last plane, is the "
        "reference's at every rung; its three independent density paths take the z "
        "table, which the native CPU build evaluates natively (SEP 29 section 6.6).",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={"": leaf_of(126, decays_vanish_at_the_end=True)},
    )
    yield optmat_row(
        case="h2_constant",
        route="capillary",
        stepper="adams",
        response="Raman",
        features="field grid, constant density",
        why="Hydrogen at constant density, its shifted lines exact: its Raman "
        "response is the reference's at every rung.",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
    )
    yield optmat_row(
        case="h2_reference",
        route="capillary",
        stepper="adams",
        response="Raman",
        features="field grid, density path to exact vacuum, reference-density shift",
        why="Hydrogen falling from 2 bar to exact vacuum under the reference-density "
        "shift, its Dicke cap finite at vacuum: its Raman response is the "
        "reference's at every rung.",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={"": leaf_of(4, decays_vanish_at_the_end=False)},
    )
    for grid in ("field", "envelope"):
        yield optmat_row(
            case="n2_vector" if grid == "field" else "n2_vector_envelope",
            route="capillary",
            stepper="adams",
            response="Raman",
            features=f"{grid} grid, two polarisations, density path to exact vacuum",
            why="Nitrogen falling to exact vacuum on two polarisations of HE11 on a "
            f"{grid} grid: the 72-row leaf's isotropic and rank-two rows contract as "
            "a tensor, its decay rates zero on the last plane, as the reference's at "
            "every rung.",
            switch_off=without_raman,
            rungs={
                "linear": one_step_of_the_operator,
                "derivative": one_stage,
                "one_step": grown_over(1),
                "steps": grown_over(8),
                "run": grown_over_the_schedule,
            },
            window=8,
            unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
            expect={"": leaf_of(72, decays_vanish_at_the_end=True)},
        )


# --- optmat's Kerr, wall and silica materials -----------------------------------------


def material_rows() -> Iterator[Row]:
    """Yield the Kerr mixture, the silver wall and the silica bundles (SEP 3, Phase 2).

    Every capillary case under the selection is walled by optmat's recommended silica,
    its default cladding there.

    Re-measured for the catalogue against field bounds of 4.4e-11 to 8.3e-9 and
    statistic bounds of 1.0e-8 to 3.3e-8: the worst field rung 3.7e-16 (the complete
    runs of ``he_ar_kerr`` and ``silver_wall``) and the worst statistic 2.9e-14
    (``quadrature_error``); the old module quoted 7.6e-16 (on the retired
    ``numba_cpu``) and 2.7e-14.
    """
    yield optmat_row(
        case="he_ar_kerr",
        route="capillary",
        stepper="adams",
        response="Kerr",
        features="field grid, mixture",
        why="A helium/argon mixture with each population's carrier Kerr coefficient "
        "(Phase 1b) is the reference's at every rung.",
        switch_off=without_kerr,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={
            "": one_kerr_term_per_population_each_its_own,
            "problem.responses.labels": ("He", "Ar"),
            "modes": walled_by_optmats_silica,
        },
    )
    yield optmat_row(
        case="silver_wall",
        route="capillary",
        stepper="adams",
        response="wall",
        features="field grid, metal wall",
        why="An argon capillary walled by optmat's silver table: the guide's index "
        "and loss from the table are the reference's at every rung.",
        switch_off=with_a_silica_wall,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={
            "derivative": "The wall acts through the linear operator, which the "
            "launch's derivative does not apply (measured: the wall moves it by "
            "exactly zero); the rung is kept as the Kerr evaluation on the "
            "silver-walled guide.",
        },
        expect={"modes": walled_by_optmats_silver_table},
    )
    yield optmat_row(
        case="gnlse_bundle",
        route="gnlse",
        stepper="rk",
        response="Raman",
        features="envelope grid, the silica bundle",
        why="The GNLSE with fused silica's bundle, whose sampled row has no decay "
        "path, is the reference's at every rung.",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={"problem.responses.raman_terms": one_leaf_without_a_decay_path},
    )
    yield optmat_row(
        case="bundle_vector",
        route="single mode, two polarisations (assembled)",
        stepper="rk",
        response="Raman",
        features="field grid, the silica tensor bundle",
        why="The silica tensor bundle on two polarisations, whose isotropic and "
        "rank-two rows both start at a non-zero sample, is the reference's at every "
        "rung, its complete run at the seam.",
        switch_off=without_raman,
        rungs={
            "linear": one_step_of_the_operator,
            "derivative": one_stage,
            "one_step": grown_over(1),
            "steps": grown_over(8),
            "run": grown_over_the_schedule,
        },
        window=8,
        unguarded={"linear": THE_LINEAR_RUNG_IS_THE_OPERATOR_ALONE},
        expect={
            "problem.responses.raman_terms": one_leaf_without_a_decay_path,
            "": every_row_starts_at_a_non_zero_sample,
        },
    )


# --- every row ------------------------------------------------------------------------


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every optmat row."""
    return (*plasma_rows(), *raman_rows(), *material_rows())

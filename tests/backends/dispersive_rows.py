# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response: its rows of the case catalogue.

**What is compared.** solprop-native evaluates ``responses.dispersive_kerr`` on every
route. On the fused node path (the multimode capillary, the mode-averaged and radial
routes) it applies the spectral multipliers on the modal columns, forms the products at
the nodes under the collar, projects them and lands them in the stage's own
polarisation spectrum; it finishes the embedded rule's share when the step is scored,
filters a lone symmetric response on the stage's own transforms and rebuilds the
symmetric mixture filter when the densities change. On the Cartesian free-space and
line routes and the hybrid planar route it runs chunk by chunk
(``SeparableDispersive``), on the general modal route in the route's synthesis
(``GeneralDispersive``). ``numpy_ref``, which applies the multipliers at every node, is
the oracle. The problems are those of ``dispersive_cases.py`` (the fused path) and
``dispersive_route_cases.py`` (the other routes), for both evaluators.

**The bounds are rounding budgets, derived before any run** from the transforms and
node reductions of one compiled evaluation, counted for the reference's node-by-node
arithmetic too (each case module's ``evaluation_bound`` and its docstring): native's
response takes the same transforms per channel (fewer when a lone symmetric response is
the stage's own polarisation) and the same sums over the nodes, the slab rule and the
modes. One step is its evaluations' budgets added (``bounds.step_evaluations``: seven
for a Runge-Kutta step), the ten-step window ten of those. A row's ``bound_note`` says
what was measured against its bound and anything added to it.

**The guard.** Each row's term is the response. :func:`without_response` zeroes every
channel's static coefficient, and each guarded rung's reference must then move by more
than ``parity_ladder.DETECTION`` (a thousand) of that rung's bounds, or by the rung's
own lowered ``detection``, which says why that is enough.

**How to read a row.** Every row is a :func:`dispersive_row` call, for example::

    dispersive_row(
        name=f"capillary beside Raman thg={thg}",
        evaluator=evaluator,
        route="capillary",
        stepper="none",
        features="Raman",
        why="Beside nitrogen's Raman response, the dispersive response must not "
        "filter the drive: ...",
        build=partial(capillary, evaluator, gas="N2", raman=True, thg=thg),
        rungs={"projected": capillary_evaluation},
        metadata=dispersive_path(alone=False, in_place=False, borrowed=False),
        expect={"problem.flags.has_raman": True},
        medium=True,
    )

``name`` and ``evaluator`` make the test id (``dispersive capillary beside Raman
thg=True symmetric``); ``route``, ``stepper`` and ``features`` are the axes the row
covers. ``build`` makes the problem: here the three-mode argon capillary of
``dispersive_cases.capillary`` filled with nitrogen, its Raman response on. ``rungs``
names what is compared, each with its bound: here one evaluation's projected
polarisation on both quadrature rules, within the fused path's one-evaluation budget.
``metadata`` is what native must report about the path its response took (not alone,
so neither in place nor on borrowed buffers), and ``expect`` what the built problem
must hold. ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import numpy as np

from backends.dispersive_cases import (
    GRID_CASES,
    LENGTH,
    STEPS,
    averaged,
    capillary,
    falling,
    radial,
)
from backends.dispersive_cases import evaluation_bound as capillary_bound
from backends.dispersive_route_cases import (
    ARGON,
    ARGON_FILL,
    CORE_CHI3,
    EVALUATORS,
    FREE_CASES,
    HELIUM,
    STRANDS,
    free_grid,
    free_problem,
    general_problem,
    planar_problem,
    raman,
    strand,
)
from backends.dispersive_route_cases import evaluation_bound as route_bound
from bounds import dispersive_device_bound, step_evaluations
from parity_ladder import Bound, Built, Detection, Row, embedded
from solprop.backends.session import ExecutionOptions
from solprop.driver import Simulation
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import Mixture
from solprop.spatial import SpatialMedium
from solprop.stepper import RungeKutta
from solprop.translate.problem import PropagationProblem
from solprop.translate.responses import KerrForm
from solprop.translate.stage import scalars

# --- bounds ---------------------------------------------------------------------------


def capillary_evaluation(built: Built) -> float:
    """One evaluation on the fused path; on a device build its predeclared envelope."""
    return dispersive_device_bound(capillary_bound(built.problem), built.problem)


def route_evaluation(built: Built) -> float:
    """One evaluation on a spatial or general route; a device build's envelope."""
    return dispersive_device_bound(route_bound(built.problem), built.problem)


def route_evaluation_cpu(built: Built) -> float:
    """One evaluation on a spatial or general route, the CPU count on every build."""
    return route_bound(built.problem)


def one_step_of(evaluation: Bound) -> Bound:
    """Bound one step: every evaluation's rounding, added (``step_evaluations``)."""
    return lambda built: step_evaluations(built.run_request) * evaluation(built)


def window_of(evaluation: Bound) -> Bound:
    """Bound the replayed window of :data:`STEPS` fixed steps."""

    def _bound(built: Built) -> float:
        schedule = built.run_request.stepper.schedule
        assert schedule is not None and len(schedule.steps) == STEPS
        return STEPS * step_evaluations(built.run_request) * evaluation(built)

    return _bound


def alarm_of(evaluation: Bound) -> Bound:
    """Bound the quadrature alarm after the window: the window's and an evaluation's."""
    window = window_of(evaluation)
    return lambda built: window(built) + evaluation(built)


# --- what a row builds, switches off and expects ------------------------------------


def without_response(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the dispersive response's every coefficient zeroed."""
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    silent = dataclasses.replace(leaf, coefficients=(0.0,) * len(leaf.coefficients))
    responses = dataclasses.replace(problem.responses, dispersive_kerr=silent)
    return dataclasses.replace(problem, responses=responses)


def built_from(
    make: Callable[[], Any], options: ExecutionOptions | None = None
) -> Callable[[], Built]:
    """Return a factory of the built row that `make` makes, with `options` if given.

    `make` returns a driver's :class:`Simulation`, an assembled
    ``(problem, run_request, grid)`` or a :class:`Built`.
    """

    def _build() -> Built:
        made = make()
        if isinstance(made, Simulation):
            made = Built(made.translate(), made._run_request(), made.grid)
        elif not isinstance(made, Built):
            made = Built(*made)
        return made if options is None else dataclasses.replace(made, options=options)

    return _build


def dispersive_row(
    *,
    name: str,
    evaluator: str,
    route: str,
    stepper: str,
    features: str,
    build: Callable[[], Any],
    options: ExecutionOptions | None = None,
    medium: bool | None = None,
    **fields: Any,
) -> Row:
    """Return a row of the dispersive response, guarded by switching it off.

    Its id is ``dispersive {name} {evaluator}``. `build` and `options` are as
    :func:`built_from` takes them; the row is medium when it marches its window unless
    `medium` says otherwise. `fields` are the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=f"dispersive {name} {evaluator}",
        axes={
            "route": route,
            "stepper": stepper,
            "features": features,
            "response": f"dispersive {evaluator}",
        },
        build=built_from(build, options),
        switch_off=without_response,
        medium="steps" in fields["rungs"] if medium is None else medium,
        **fields,
    )


def dispersive_path(
    *, alone: bool, in_place: bool, borrowed: bool
) -> dict[str, object]:
    """Return the native ``dispersive_path`` metadata a row expects.

    `alone`: the response is the stage's only nonlinear term; `in_place`: it filters
    the stage's own spectrum rather than a field of its own; `borrowed`: its buffers
    are the stage's transforms.
    """
    return {
        "dispersive_path": {"alone": alone, "in_place": in_place, "borrowed": borrowed}
    }


def _densities(built: Built, z: float) -> tuple[float, ...]:
    return tuple(scalars(built.problem, 0.0, z).dispersive_kerr_densities)


def helium_falls_from_four_times_argon_to_a_quarter(built: Built) -> bool:
    """The densities' ratio is 4 at the launch and 1/4 at the end, to 1 %."""
    (he0, ar0), (he1, ar1) = _densities(built, 0.0), _densities(built, LENGTH)
    return abs(he0 / ar0 / 4.0 - 1.0) <= 1e-2 and abs(he1 / ar1 / 0.25 - 1.0) <= 1e-2


def helium_reaches_zero_and_argon_does_not(built: Built) -> bool:
    """Helium's density is exactly zero at the end of the guide; argon's is not."""
    helium, argon = _densities(built, LENGTH)
    return helium == 0.0 and argon > 0.0


def has_two_field_components(built: Built) -> bool:
    """The state carries two transverse field components."""
    return built.problem.transform.n_pol * built.problem.shape[1] == 2


def holds_its_own_plasma_exactly_on_an_envelope_grid(
    envelope: bool,
) -> Callable[[object], bool]:
    """Return the predicate: an envelope plasma is held exactly on an envelope grid."""

    def holds_its_own_plasma(plasma: object) -> bool:
        return (plasma is not None) == envelope

    return holds_its_own_plasma


# --- the multimode capillary ----------------------------------------------------------


def multimode_capillary_rows(evaluator: str) -> Iterator[Row]:
    """Yield the response alone on the three-mode argon capillary, 33 nodes.

    Every grid case (field or envelope, the full cubic or the conserving one): one
    evaluation under the in-stage absorber, one Runge-Kutta step and ten steps of the
    default integrator under the split-step absorber; then node chunks and the
    Runge-Kutta window.
    """
    lone_symmetric = evaluator == "symmetric"
    for case, grid in GRID_CASES.items():
        # A lone symmetric response projects through the stage's own transforms
        # (borrowed) unless its products are a field grid's analytic signals.
        borrowed = lone_symmetric and case != "field conserving"
        yield dispersive_row(
            name=f"capillary {case}",
            evaluator=evaluator,
            route="capillary",
            stepper="none",
            features=case,
            why="Alone on the multimode capillary under the in-stage absorber, one "
            "evaluation of the response projects onto both quadrature rules as the "
            "reference's does.",
            bound_note="Alone, the response holds its own filtered field beside the "
            "bare one the collar reads. Measured 3.0-10.5 eps on both rules, under "
            "bounds of 1064-1998 eps (re-measured for the catalogue; the old test "
            "quoted 2.3-4.3 eps).",
            build=partial(capillary, evaluator, **grid),
            rungs={"projected": capillary_evaluation},
            metadata=dispersive_path(alone=True, in_place=False, borrowed=borrowed),
            expect={"problem.transform.n_nodes": 33},
        )
        yield dispersive_row(
            name=f"capillary {case} one step",
            evaluator=evaluator,
            route="capillary",
            stepper="rk",
            features=case,
            why="One Runge-Kutta step on the multimode capillary, seven evaluations "
            "through the whole stage, is the reference's step; under the split-step "
            "absorber a lone symmetric response filters the stage's own spectrum in "
            "place.",
            bound_note="Measured 1.1-2.2 eps, under bounds of 7448-13986 eps, seven "
            "evaluations' budgets (re-measured; the old test quoted 1.2-1.5 eps).",
            build=partial(
                capillary,
                evaluator,
                stepped=True,
                n_save=None,
                integrator=RungeKutta(),
                save_planes=[LENGTH / STEPS],
                **grid,
            ),
            rungs={"one_step": one_step_of(capillary_evaluation)},
            metadata=dispersive_path(
                alone=True, in_place=lone_symmetric, borrowed=borrowed
            ),
        )
        # The quadrature alarm is the statistic the embedded rule exists for; the
        # field grids' cases carry it.
        alarm = {"quadrature_error": alarm_of(capillary_evaluation)}
        yield dispersive_row(
            name=f"capillary {case} window",
            evaluator=evaluator,
            route="capillary",
            stepper="adams",
            features=case,
            why="Ten fixed steps of the default integrator (the resident Adams "
            "engine) on the multimode capillary are the reference's, and on a field "
            "grid so is the quadrature alarm after them.",
            bound_note="Measured 6.7-12.9 eps, under bounds of 155 000-292 000 eps "
            "(re-measured; the old test quoted 5.8-6.7 eps). The response's share of "
            "the embedded projection is finished when the step is scored; without it "
            "the alarm would be near one. Both alarms agree to the window's and an "
            "evaluation's bounds, added: 0.1-0.3 eps measured.",
            build=partial(capillary, evaluator, stepped=True, **grid),
            rungs={"steps": window_of(capillary_evaluation)},
            statistics=alarm if case.startswith("field") else {},
        )
    # Five chunks of eight nodes, the last of one; one case beside Raman, whose column
    # the response then runs beside.
    for case, beside_raman in (
        ("field full", False),
        ("field conserving", False),
        ("field conserving", True),
        ("envelope full", False),
    ):
        nitrogen_and_raman = {"gas": "N2", "raman": True} if beside_raman else {}
        yield dispersive_row(
            name=f"capillary {case} chunks of eight"
            + (" beside Raman" if beside_raman else ""),
            evaluator=evaluator,
            route="capillary",
            stepper="none",
            features=f"{case}, node chunks",
            why="In node chunks of eight the response's products accumulate as in one "
            "chunk: it packs each chunk's columns of the scaled projections once a "
            "stage.",
            build=partial(
                capillary, evaluator, **nitrogen_and_raman, **GRID_CASES[case]
            ),
            options=ExecutionOptions(node_chunk=8),
            rungs={"projected": capillary_evaluation},
        )
    yield dispersive_row(
        name="capillary Runge-Kutta window",
        evaluator=evaluator,
        route="capillary",
        stepper="rk",
        features="field full",
        why="The Runge-Kutta march over ten steps, seven stages a step with the first "
        "recycled, is the reference's.",
        build=partial(capillary, evaluator, stepped=True, integrator=RungeKutta()),
        rungs={"steps": window_of(capillary_evaluation)},
        # Runge-Kutta is the request's integrator when it names none.
        expect={"run_request.stepper.integrator": None},
    )


# --- the mode-averaged, two-component and radial routes ------------------------------


def other_fused_route_rows(evaluator: str) -> Iterator[Row]:
    """Yield one evaluation and ten steps on the fused path's other routes."""
    for case, grid in GRID_CASES.items():
        yield dispersive_row(
            name=f"mode-averaged {case}",
            evaluator=evaluator,
            route="mode-averaged",
            stepper="adams",
            features=case,
            why="On the mode-averaged route, one node, the response's evaluation and "
            "ten steps are the reference's.",
            build=partial(averaged, evaluator, **grid),
            rungs={
                "projected": capillary_evaluation,
                "steps": window_of(capillary_evaluation),
            },
            expect={"problem.transform.n_nodes": 1},
        )
    for case in ("field full", "field conserving", "envelope full"):
        yield dispersive_row(
            name=f"two-component {case}",
            evaluator=evaluator,
            route="two-component",
            stepper="adams",
            features=case,
            why="The isotropic Kleinman contraction of an elliptical launch, two field "
            "components on the quadrature, is the reference's, for one evaluation and "
            "ten steps.",
            build=partial(averaged, evaluator, vector=True, **GRID_CASES[case]),
            rungs={
                "projected": capillary_evaluation,
                "steps": window_of(capillary_evaluation),
            },
            expect={"": has_two_field_components},
        )
    for envelope in (False, True):
        yield dispersive_row(
            name=f"radial {'envelope' if envelope else 'field'}",
            evaluator=evaluator,
            route="radial",
            stepper="adams",
            features="",
            why="On the Hankel route, sixteen nodes, the response's evaluation and ten "
            "steps are the reference's.",
            build=partial(radial, evaluator, envelope=envelope, thg=envelope),
            rungs={
                "projected": capillary_evaluation,
                "steps": window_of(capillary_evaluation),
            },
            expect={"problem.transform.n_nodes": 16},
        )


# --- mixtures -------------------------------------------------------------------------


def mixture_rows(evaluator: str) -> Iterator[Row]:
    """Yield helium and argon on the capillary: constant, varying, one member at zero.

    **A member at exactly zero density.** The retired Numba stage built the mixture
    filter from densities ``(3e25, 0, 1e24)`` and a vacuum's ``(0, 0, 0)``, where the
    filter is one and the output multipliers exactly zero. Native's rows are the C++
    test's (``DispersiveMixture.TheRowsAreTheReferenceArithmetic``, a zero factor and a
    zero total); here they are reached by a real run whose partial pressures fall to
    zero along the guide. This is weaker than the original: the mixture filter's rows,
    and the rule that the same densities leave them unrebuilt, are not read directly;
    the source they make is. A native stage's input row after a vacuum is not visible,
    so the exact unit filter stays on the C++ test, and a vacuum's exact zero source on
    ``test_native_dispersive_kerr.py``.
    """
    why = (
        "Helium and argon are one product under the mixture filter, rebuilt natively "
        "from the channels' densities by the reference's arithmetic, or the "
        "polynomial's E^3, shared and weighed by both channels' output multipliers."
    )
    for case in ("field full", "envelope conserving"):
        helium_and_argon = partial(
            capillary, evaluator, gas={"He": 0.6, "Ar": 0.4}, **GRID_CASES[case]
        )
        yield dispersive_row(
            name=f"capillary {case} helium and argon",
            evaluator=evaluator,
            route="capillary",
            stepper="none",
            features=f"{case}, mixture",
            why=why,
            build=helium_and_argon,
            rungs={"projected": capillary_evaluation},
            expect={"problem.responses.dispersive_kerr.n_channels": 2},
            medium=True,  # as the old test, which also ran the window below
        )
        yield dispersive_row(
            name=f"capillary {case} helium and argon window",
            evaluator=evaluator,
            route="capillary",
            stepper="adams",
            features=f"{case}, mixture",
            why=why,
            build=partial(helium_and_argon, stepped=True),
            rungs={"steps": window_of(capillary_evaluation)},
        )
    yield dispersive_row(
        name="capillary varying composition",
        evaluator=evaluator,
        route="capillary",
        stepper="adams",
        features="mixture along z",
        why="With helium falling and argon rising along the guide, the mixture filter "
        "is rebuilt each stage as the reference's is, over ten steps.",
        build=partial(falling, evaluator, 0.5, 2.0),
        rungs={"steps": window_of(capillary_evaluation)},
        expect={"": helium_falls_from_four_times_argon_to_a_quarter},
    )
    yield dispersive_row(
        name="capillary a member at zero density",
        evaluator=evaluator,
        route="capillary",
        stepper="adams",
        features="a zero member",
        why="Where helium's partial pressure reaches an exact zero and argon carries "
        "on alone, the source there and the march down to it are the reference's.",
        bound_note="The route modules' CPU count of one evaluation on every build "
        "(``dispersive_route_cases.evaluation_bound``), as the original test held it, "
        "and ten steps of it for the window.",
        build=partial(falling, evaluator, 0.0, 2.0),
        rungs={
            "source": route_evaluation_cpu,
            "steps": window_of(route_evaluation_cpu),
        },
        source_z=LENGTH,
        expect={"": helium_reaches_zero_and_argon_does_not},
    )


def leg_bound_rows() -> Iterator[Row]:
    """Yield krypton and xenon under the polynomial: one leg bound, one group.

    The band starts at 180 nm and the pulse is 5 uJ.
    """
    one_group = {
        "problem.responses.dispersive_kerr.n_channels": 2,
        "problem.responses.dispersive_kerr.groups": ((0, 1),),
    }
    why = (
        "Krypton and xenon share one leg bound, so the polynomial forms one group: one "
        "D E and one D^2 E with their coefficient tables added, density-weighted, an "
        "exact rewriting of the reference's per-species sum."
    )
    for case in ("field full", "field conserving", "envelope full"):
        krypton_and_xenon = partial(
            capillary,
            "polynomial",
            gas={"Kr": 0.5, "Xe": 0.5},
            wavelength_limits=(180e-9, 4e-6),
            energy=5e-6,
            **GRID_CASES[case],
        )
        yield dispersive_row(
            name=f"capillary {case} krypton and xenon",
            evaluator="polynomial",
            route="capillary",
            stepper="none",
            features=f"{case}, leg bound",
            why=why,
            build=krypton_and_xenon,
            rungs={"projected": capillary_evaluation},
            expect=one_group,
            medium=True,  # as the old test, which also ran the window below
        )
        yield dispersive_row(
            name=f"capillary {case} krypton and xenon window",
            evaluator="polynomial",
            route="capillary",
            stepper="adams",
            features=f"{case}, leg bound",
            why=why,
            build=partial(krypton_and_xenon, stepped=True),
            rungs={"steps": window_of(capillary_evaluation)},
        )
    yield dispersive_row(
        name="general krypton and xenon",
        evaluator="polynomial",
        route="general",
        stepper="adams",
        features="real, leg bound",
        why="Krypton and xenon in one region of the general modal route share a leg "
        "bound: one group, one cube, and the reference's source.",
        build=partial(
            _general, "polynomial", "real", gas={"Kr": 0.5, "Xe": 0.5}, core=None
        ),
        rungs={"source": route_evaluation},
        masked_launch=True,
        expect=one_group,
    )


# --- beside Raman and plasma ----------------------------------------------------------


def beside_rows(evaluator: str) -> Iterator[Row]:
    """Yield the response beside nitrogen's Raman response and beside plasma.

    Beside another term the response must leave that term's drive alone: Raman and the
    plasma read the bare field, never the response's filtered one.

    **Beside plasma** the comparison is the whole stage source, which holds the
    envelope currents on both backends (the engine assembles them into the projected
    spectrum, the reference beside it). The retired comparison of both rules'
    projected spectra beside plasma is not carried: an envelope's currents are
    assembled apart from those spectra on both backends, and under the carrier policy
    native's projected spectra there differ from the reference's by 1.78 (field) and
    1.81 (envelope), relative, so its floor made a bound of about 180 % that could not
    see the response (issue #789). Both rules are held without plasma by the
    capillary rows.
    """
    why = (
        "Beside nitrogen's Raman response, the dispersive response must not filter "
        "the drive: the Raman response reads the bare field, as in the reference."
    )
    for thg in (True, False):
        nitrogen_and_raman = partial(
            capillary, evaluator, gas="N2", raman=True, thg=thg
        )
        yield dispersive_row(
            name=f"capillary beside Raman thg={thg}",
            evaluator=evaluator,
            route="capillary",
            stepper="none",
            features="Raman",
            why=why,
            build=nitrogen_and_raman,
            rungs={"projected": capillary_evaluation},
            metadata=dispersive_path(alone=False, in_place=False, borrowed=False),
            expect={"problem.flags.has_raman": True},
            medium=True,  # as the old test, which also ran the window below
        )
        yield dispersive_row(
            name=f"capillary beside Raman thg={thg} window",
            evaluator=evaluator,
            route="capillary",
            stepper="adams",
            features="Raman",
            why=why,
            build=partial(nitrogen_and_raman, stepped=True),
            rungs={"steps": window_of(capillary_evaluation)},
        )
    for envelope in (False, True):
        grid = "envelope" if envelope else "field"
        plasma_case = {
            "plasma": True,
            "thg": False,
            "envelope": envelope,
            "energy": 60e-6,
        }
        why = (
            "Beside the "
            + ("envelope's own plasma current streams" if envelope else "plasma column")
            + ", the dispersive response must not filter the field the plasma reads."
        )
        # The floor is the same problem under the carrier policy, without the
        # response: its native disagreement is added to the bound, and its reference,
        # which differs from the row's by the response alone, is the guard's.
        yield dispersive_row(
            name=f"capillary {grid} beside plasma",
            evaluator=evaluator,
            route="capillary",
            stepper="none",
            features=f"{grid} plasma",
            why=why,
            bound_note="The plasma column and the prefactor's band edges already "
            "disagree with the reference's, so the bound adds the carrier-policy "
            "source's disagreement on this backend: 149 eps (field) and 124 eps "
            "(envelope) re-measured, where the old test quoted 268 and 178. The "
            "dispersive sources sit within 0.3 eps of it. A plasma current the "
            "response filtered would be wrong by the multiplier's departure from one, "
            "about 1e-3, far above either.",
            build=partial(capillary, evaluator, **plasma_case),
            floor=built_from(partial(capillary, "carrier-kerr", **plasma_case)),
            rungs={"source": capillary_evaluation},
            expect={
                "problem.responses.envelope_plasma": (
                    holds_its_own_plasma_exactly_on_an_envelope_grid(envelope)
                ),
                "problem": embedded,
            },
            medium=True,  # as the old test, which also ran the window below
        )
        yield dispersive_row(
            name=f"capillary {grid} beside plasma window",
            evaluator=evaluator,
            route="capillary",
            stepper="adams",
            features=f"{grid} plasma",
            why=why,
            bound_note="The window's budget plus the carrier policy's own ten-step "
            "disagreement on this backend: 11.7 eps (field) and 7.7 eps (envelope). "
            "Measured 7.9-27.1 eps, under bounds of 155 000-292 000 eps (re-measured; "
            "the old test quoted 6.6-6.7 eps for the carrier policy and 6.8-19.5 eps "
            "for the response).",
            build=partial(capillary, evaluator, stepped=True, **plasma_case),
            floor=built_from(
                partial(capillary, "carrier-kerr", stepped=True, **plasma_case)
            ),
            rungs={"steps": window_of(capillary_evaluation)},
        )


# --- free space and the line ----------------------------------------------------------
#
# A source rung on the spatial and general routes evaluates the launch hard-masked to
# the active band on both backends (``masked_launch``), as the native state loads it:
# native masks what it loads and the reference's sample takes its field as given, so
# an unmasked launch would be two inputs. The hybrid planar launch has content outside
# the band, which moved the samples 3e-11 to 6e-11 apart; at the one input they meet
# to about 5 eps.


def _free(evaluator: str, case: str, *, line: bool = False, beside_raman: bool = False):
    """Return the free-space (or line) problem of one grid case, Raman beside or not."""
    grid = free_grid(**FREE_CASES[case])
    thg = FREE_CASES[case]["thg"]
    responses: list[object] = [DispersiveKerrResponse(ARGON, evaluator, thg=thg)]
    if beside_raman:
        responses.append(raman(thg))
    return Built(*free_problem(responses, grid, line=line), grid)


def _free_mixture(evaluator: str) -> Built:
    """Return helium and argon in the free-space medium: two channels."""
    grid = free_grid("field", True)
    medium = SpatialMedium(Mixture.fill({"He": 0.6, "Ar": 0.4}, 2.0))
    responses = [
        DispersiveKerrResponse(HELIUM, evaluator, species="He"),
        DispersiveKerrResponse(ARGON, evaluator, species="Ar"),
    ]
    return Built(*free_problem(responses, grid, medium=medium), grid)


def _free_runge_kutta(evaluator: str) -> Built:
    grid = free_grid("field", True)
    response = DispersiveKerrResponse(ARGON, evaluator)
    return Built(*free_problem([response], grid, adams=False), grid)


LINE_WINDOW_GUARD = Detection(
    threshold=100,
    reason="the whole Kerr effect of the line's beam over its 0.1 mm window is "
    "8.7e-9 of the field (measured, on every case and evaluator), against window "
    "budgets of 2.3e-11 to 5.4e-11, so this rung clears 161 to 381 bounds; the "
    "line's source rung clears a thousand by far",
)
"""The line's ten-step window: a guard of a hundred bounds, and why."""


def free_space_rows(evaluator: str) -> Iterator[Row]:
    """Yield the Cartesian free-space and line routes, run chunk by chunk."""
    lone_symmetric = evaluator == "symmetric"
    for line in (False, True):
        route = "line" if line else "cartesian"
        for case in FREE_CASES:
            yield dispersive_row(
                name=f"{route} {case}",
                evaluator=evaluator,
                route=route,
                stepper="adams",
                features=case,
                why=f"On the {route} route the response alone runs on each chunk's "
                "columns, which are its nodes, and gives the reference's source and "
                "ten steps.",
                bound_note="Measured 2.8-12.9 eps on the source, under bounds of "
                "704-3232 eps (re-measured; the old test quoted 2.2-4.6 eps).",
                build=partial(_free, evaluator, case, line=line),
                rungs={
                    "source": route_evaluation,
                    "steps": window_of(route_evaluation),
                },
                masked_launch=True,
                metadata=dispersive_path(
                    alone=True,
                    in_place=lone_symmetric,
                    borrowed=lone_symmetric and case != "field conserving",
                ),
                detection={"steps": LINE_WINDOW_GUARD} if line else {},
            )
    for case in ("field full", "field conserving"):
        yield dispersive_row(
            name=f"cartesian {case} beside Raman",
            evaluator=evaluator,
            route="cartesian",
            stepper="adams",
            features=f"{case}, Raman",
            why="In free space beside Raman the column runs beside the response, whose "
            "spectra are added to the chunk's, as in the reference.",
            bound_note="Measured 5.4-10.7 eps on the source, under bounds of 1728-3232 "
            "eps (re-measured; the old test quoted 3.0-4.4 eps).",
            build=partial(_free, evaluator, case, beside_raman=True),
            rungs={
                "source": route_evaluation,
                "steps": window_of(route_evaluation),
            },
            masked_launch=True,
            metadata=dispersive_path(alone=False, in_place=False, borrowed=False),
            expect={"problem.flags.has_raman": True},
        )
    yield dispersive_row(
        name="cartesian one step",
        evaluator=evaluator,
        route="cartesian",
        stepper="rk",
        features="field full",
        why="One Runge-Kutta step through the whole Cartesian stage, seven "
        "evaluations, is the reference's.",
        build=partial(_free_runge_kutta, evaluator),
        rungs={"one_step": one_step_of(route_evaluation)},
    )
    yield dispersive_row(
        name="cartesian helium and argon",
        evaluator=evaluator,
        route="cartesian",
        stepper="adams",
        features="mixture",
        why="Helium and argon in the free-space medium are two channels, each with its "
        "own leg bound: the symmetric evaluator's one product under the mixture "
        "filter, rebuilt natively from the channels' g^4, or the polynomial's two "
        "groups, as in the reference.",
        build=partial(_free_mixture, evaluator),
        rungs={
            "source": route_evaluation,
            "steps": window_of(route_evaluation),
        },
        masked_launch=True,
        expect={"problem.responses.dispersive_kerr.n_channels": 2},
    )
    yield dispersive_row(
        name="cartesian chunks of twenty",
        evaluator=evaluator,
        route="cartesian",
        stepper="adams",
        features="node chunks",
        why="In chunks of 20 columns, the last a partial one, each chunk's source "
        "lands where it belongs.",
        build=partial(_free, evaluator, "field full"),
        options=ExecutionOptions(node_chunk=20),
        rungs={"source": route_evaluation},
        masked_launch=True,
        expect={"problem.transform.n_nodes": lambda n_nodes: n_nodes % 20 != 0},
    )


# --- the hybrid planar route ----------------------------------------------------------


def _planar_blocks(evaluator: str) -> Built:
    """Return the planar route in blocks of three y rows, the last a partial one."""
    built = Built(*planar_problem([DispersiveKerrResponse(ARGON, evaluator)]))
    geometry = built.problem.transform.geometry
    assert geometry is not None
    chunk = ExecutionOptions(node_chunk=3 * geometry.shape[0])
    return dataclasses.replace(built, options=chunk)


def planar_rows(evaluator: str) -> Iterator[Row]:
    """Yield the hybrid planar route: TE1 and TE3 of an argon gap, four k_y."""
    lone_symmetric = evaluator == "symmetric"
    for thg in (True, False):
        response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
        yield dispersive_row(
            name=f"planar thg={thg}",
            evaluator=evaluator,
            route="planar",
            stepper="adams",
            features=f"thg={thg}",
            why="On the planar route the block's rows are filtered, synthesised in x, "
            "multiplied and projected back as in the reference, for one evaluation "
            "and ten steps.",
            bound_note="Measured 3.0-5.3 eps on the source, under bounds of "
            "1792-3488 eps (re-measured; the old test quoted 3.4-5.8 eps).",
            build=partial(planar_problem, [response]),
            rungs={
                "source": route_evaluation,
                "steps": window_of(route_evaluation),
            },
            masked_launch=True,
        )
        yield dispersive_row(
            name=f"planar thg={thg} beside Raman",
            evaluator=evaluator,
            route="planar",
            stepper="adams",
            features="Raman",
            why="On the planar route beside Raman the response's spectra are added to "
            "the block's and the Raman response reads the bare rows, as in the "
            "reference.",
            build=partial(planar_problem, [response, raman(thg)]),
            rungs={"source": route_evaluation},
            masked_launch=True,
        )
    response = DispersiveKerrResponse(ARGON, evaluator)
    yield dispersive_row(
        name="planar in-stage collar",
        evaluator=evaluator,
        route="planar",
        stepper="adams",
        features="in-stage collar",
        why="On the planar route the in-stage collar is a pass of its own, which packs "
        "the rows afresh, so a lone symmetric response filters in place beside it and "
        "agrees with the reference all the same.",
        build=partial(planar_problem, [response], absorbed=True),
        rungs={
            "source": route_evaluation,
            "steps": window_of(route_evaluation),
        },
        masked_launch=True,
        metadata={
            "dispersive_path": lambda path: (
                path is not None and path["in_place"] == lone_symmetric
            )
        },
        expect={"problem.responses.absorber_scheme": "nonlinear"},
    )
    yield dispersive_row(
        name="planar blocks of three rows",
        evaluator=evaluator,
        route="planar",
        stepper="adams",
        features="node chunks",
        why="In blocks of three y rows of four, the last a partial one, the response's "
        "products accumulate as in one block.",
        build=partial(_planar_blocks, evaluator),
        rungs={"source": route_evaluation},
        masked_launch=True,
    )
    yield dispersive_row(
        name="planar one step",
        evaluator=evaluator,
        route="planar",
        stepper="rk",
        features="full",
        why="One Runge-Kutta step through the whole planar stage is the reference's.",
        build=partial(planar_problem, [response], integrator=RungeKutta(), steps=1),
        rungs={"one_step": one_step_of(route_evaluation)},
        medium=True,  # as the old test was
    )


# --- the general modal route ----------------------------------------------------------


def _general(
    evaluator: str,
    form: KerrForm,
    *,
    gas: Mixture | dict[str, float] | None,
    core: float | None,
    adams: bool = True,
):
    """Return the general modal route on a strand of gas (a 1 bar fill) and core."""
    if isinstance(gas, dict):
        gas = Mixture.fill(gas, 1.0)
    return general_problem(strand(gas, core=core), evaluator, form, adams=adams)


def _cut_at_the_first_gas_node(evaluator: str) -> Built:
    """Return the general route in chunks cut at its first gas node.

    The first chunk then holds the solid core's nodes alone: the original's
    construction, an exact chunk both backends are prepared with.
    """
    built = Built(*general_problem(strand(ARGON_FILL), evaluator, "real"))
    tables = built.problem.transform.modal_tables
    assert tables is not None
    region = np.asarray(tables.region_index)
    first_gas = int(np.flatnonzero(region == 1)[0])
    assert first_gas >= 1
    assert np.all(region[:first_gas] != 1)
    chunk = ExecutionOptions(node_chunk=first_gas)
    return dataclasses.replace(built, options=chunk)


def general_rows(evaluator: str) -> Iterator[Row]:
    """Yield the general modal route: a 1.5 um strand's HE11 in a gas exterior, 2 nJ."""
    for form in ("real", "analytic"):
        for strand_name, (gas, core) in STRANDS.items():
            yield dispersive_row(
                name=f"general {strand_name} {form}",
                evaluator=evaluator,
                route="general",
                stepper="adams",
                features=f"{form}, {strand_name}",
                why="On the general modal route the multipliers act in synthesis and "
                "the products at the gas region's nodes: alone the response writes "
                "each chunk's source, beside the solid core's constant cubic it adds "
                "to the regional one, as in the reference.",
                bound_note="Measured 1.2-1.8 eps on the source, under bounds of "
                "560-1088 eps (re-measured; the old test quoted 0.5-1.6 eps).",
                build=partial(_general, evaluator, form, gas=gas, core=core),
                rungs={
                    "source": route_evaluation,
                    "steps": window_of(route_evaluation),
                },
                masked_launch=True,
                unguarded={
                    "steps": "the gas region moves the 2 nJ launch's ten steps by "
                    "1.7e-11 to 7.0e-11, the size of the bounds (1.8e-11 to 3.5e-11): "
                    "the window holds the general route's march, the source rung the "
                    "response (issue #789)"
                },
                metadata=dispersive_path(
                    alone=core is None, in_place=False, borrowed=False
                ),
                expect={
                    "problem.responses.dispersive_kerr.regions": (
                        lambda regions: regions is not None
                    )
                },
            )
    yield dispersive_row(
        name="general one step",
        evaluator=evaluator,
        route="general",
        stepper="rk",
        features="real, beside a solid core",
        why="One Runge-Kutta step on the general modal route, beside a solid core, is "
        "the reference's.",
        build=partial(
            _general, evaluator, "real", gas=ARGON_FILL, core=CORE_CHI3, adams=False
        ),
        rungs={"one_step": one_step_of(route_evaluation)},
    )
    yield dispersive_row(
        name="general a chunk without a gas node",
        evaluator=evaluator,
        route="general",
        stepper="adams",
        features="real, node chunks",
        why="A chunk of the solid core's nodes alone forms and lands no product, the "
        "core having no regional tensor, and the evaluation still agrees with the "
        "reference.",
        bound_note="The route's CPU count of one evaluation on every build, as the "
        "original test held it.",
        build=partial(_cut_at_the_first_gas_node, evaluator),
        rungs={"source": route_evaluation_cpu},
        masked_launch=True,
        quiet=("ignore::solprop.exceptions.GeneralModalBasisWarning",),
    )


# --- every row ------------------------------------------------------------------------


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every dispersive row."""
    found = [*leg_bound_rows()]
    for evaluator in EVALUATORS:
        found += [
            *multimode_capillary_rows(evaluator),
            *other_fused_route_rows(evaluator),
            *mixture_rows(evaluator),
            *beside_rows(evaluator),
            *free_space_rows(evaluator),
            *planar_rows(evaluator),
            *general_rows(evaluator),
        ]
    return tuple(found)

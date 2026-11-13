# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response: the catalogue's rows.

The cases of ``dispersive_cases.py`` (the fused path: multimode capillary,
mode-averaged, two-component, radial) and ``dispersive_route_cases.py`` (free space,
the line, the hybrid planar route, the general modal route), at the bounds those
modules derive from summation lengths before any run. The term each row switches off
for its guard is the response: every channel's static coefficient zeroed.
"""

import dataclasses
import functools
from collections.abc import Callable
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
from parity_ladder import Bound, Built, Row, embedded
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


def without_response(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the dispersive response's every coefficient zeroed."""
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    silent = dataclasses.replace(leaf, coefficients=(0.0,) * len(leaf.coefficients))
    responses = dataclasses.replace(problem.responses, dispersive_kerr=silent)
    return dataclasses.replace(problem, responses=responses)


# --- what the rows expect ------------------------------------------------------------

NODES = "problem.transform.n_nodes"
LEAF = "problem.responses.dispersive_kerr"
RAMAN = {"problem.flags.has_raman": True}
TWO = {f"{LEAF}.n_channels": 2}
ONE_GROUP = TWO | {f"{LEAF}.groups": ((0, 1),)}


def path(alone: bool, in_place: bool, borrowed: bool) -> dict[str, object]:
    """Return the native ``dispersive_path`` metadata a row expects."""
    return {
        "dispersive_path": {"alone": alone, "in_place": in_place, "borrowed": borrowed}
    }


def _densities(built: Built, z: float) -> tuple[float, ...]:
    return tuple(scalars(built.problem, 0.0, z).dispersive_kerr_densities)


def _ratios(built: Built) -> bool:
    """Helium falls from four times argon's density to a quarter of it, to 1 %."""
    (he0, ar0), (he1, ar1) = _densities(built, 0.0), _densities(built, LENGTH)
    return abs(he0 / ar0 / 4.0 - 1.0) <= 1e-2 and abs(he1 / ar1 / 0.25 - 1.0) <= 1e-2


def _two_components(built: Built) -> bool:
    return built.problem.transform.n_pol * built.problem.shape[1] == 2


def _held(envelope: bool, plasma: object) -> bool:
    """The envelope's own plasma is held exactly on an envelope grid."""
    return (plasma is not None) == envelope


def _zero_member(built: Built) -> bool:
    """Helium's density reaches an exact zero at the end; argon's does not."""
    helium, argon = _densities(built, LENGTH)
    return helium == 0.0 and argon > 0.0


# --- builders -------------------------------------------------------------------------


def _built(make: Callable[[], Any], options: ExecutionOptions | None) -> Built:
    """Build a row from a driver's run, an assembled triple or a built row."""
    made = make()
    if isinstance(made, Simulation):
        made = Built(made.translate(), made._run_request(), made.grid)
    elif not isinstance(made, Built):
        made = Built(*made)
    return made if options is None else dataclasses.replace(made, options=options)


def _free(evaluator: str, case: str, *, line: bool = False, beside: bool = False):
    """Return the free-space (or line) problem of one grid case, Raman beside or not."""
    grid = free_grid(**FREE_CASES[case])
    thg = FREE_CASES[case]["thg"]
    responses: list[object] = [DispersiveKerrResponse(ARGON, evaluator, thg=thg)]
    if beside:
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


def _general(
    evaluator: str,
    form: KerrForm,
    gas: Mixture | dict[str, float] | None,
    core: float | None,
    *,
    adams: bool = True,
):
    """Return the general modal route on a strand of gas (a 1 bar fill) and core."""
    if isinstance(gas, dict):
        gas = Mixture.fill(gas, 1.0)
    return general_problem(strand(gas, core=core), evaluator, form, adams=adams)


def _planar_blocks(evaluator: str) -> Built:
    """Return the planar route in blocks of three y rows, the last a partial one."""
    built = Built(*planar_problem([DispersiveKerrResponse(ARGON, evaluator)]))
    geometry = built.problem.transform.geometry
    assert geometry is not None
    chunk = ExecutionOptions(node_chunk=3 * geometry.shape[0])
    return dataclasses.replace(built, options=chunk)


def _without_gas_chunk(evaluator: str) -> Built:
    """Return the general route cut at its first gas node: a chunk of core nodes."""
    built = Built(*general_problem(strand(ARGON_FILL), evaluator, "real"))
    tables = built.problem.transform.modal_tables
    assert tables is not None
    region = np.asarray(tables.region_index)
    first_gas = int(np.flatnonzero(region == 1)[0])
    assert first_gas >= 1
    assert np.all(region[:first_gas] != 1)
    chunk = ExecutionOptions(node_chunk=first_gas)
    return dataclasses.replace(built, options=chunk)


def _add(
    rows: list[Row],
    evaluator: str,
    name: str,
    axes: str,
    make: Callable[[], Any],
    rungs: dict[str, Bound],
    *,
    options: ExecutionOptions | None = None,
    **fields: Any,
) -> None:
    """Add a row: ``axes`` is ``route/stepper/features``; medium if it marches."""
    route, stepper, features = axes.split("/")
    fields.setdefault("medium", "steps" in rungs)
    axis = {"route": route, "stepper": stepper, "features": features}
    rows.append(
        Row(
            id=f"dispersive {name} {evaluator}",
            axes=axis | {"response": f"dispersive {evaluator}"},
            build=partial(_built, make, options),
            rungs=rungs,
            switch_off=without_response,
            **fields,
        )
    )


# --- rows -----------------------------------------------------------------------------

PROJECTED = {"projected": capillary_evaluation}
WINDOW = {"steps": window_of(capillary_evaluation)}
STEP = {"one_step": one_step_of(capillary_evaluation)}
ALARM = {"quadrature_error": alarm_of(capillary_evaluation)}
SOURCE = {"source": route_evaluation}
ROUTE_WINDOW = {"steps": window_of(route_evaluation)}
ROUTE_STEP = {"one_step": one_step_of(route_evaluation)}
GENERAL_WINDOW = {
    "steps": "the gas region moves the 2 nJ launch's ten steps by 2e-11 to 7e-11, the "
    "bound's own size: the window holds the general route's march, the source rung "
    "the response"
}
"""Why the general route's window carries no guard of its own."""


def fused_rows(evaluator: str) -> list[Row]:
    """Return one evaluator's fused-path rows."""
    rows: list[Row] = []
    add = partial(_add, rows, evaluator)
    cap = partial(capillary, evaluator)
    lone = evaluator == "symmetric"
    for case, grid in GRID_CASES.items():
        borrowed = lone and case != "field conserving"
        stepped = partial(cap, stepped=True, **grid)
        rk = {"integrator": RungeKutta(), "save_planes": [LENGTH / STEPS]}
        add(f"capillary {case}", f"capillary/none/{case}", partial(cap, **grid),
            PROJECTED, metadata=path(True, False, borrowed),
            expect={NODES: 33})  # fmt: skip
        add(f"capillary {case} one step", f"capillary/rk/{case}",
            partial(stepped, n_save=None, **rk), STEP,
            metadata=path(True, lone, borrowed))  # fmt: skip
        add(f"capillary {case} window", f"capillary/adams/{case}", stepped, WINDOW,
            statistics=ALARM if case.startswith("field") else {})  # fmt: skip
        add(f"mode-averaged {case}", f"mode-averaged/adams/{case}",
            partial(averaged, evaluator, **grid), PROJECTED | WINDOW,
            expect={NODES: 1})  # fmt: skip
    chunked = [("field full", False), ("field conserving", False),
               ("field conserving", True), ("envelope full", False)]  # fmt: skip
    for case, beside in chunked:
        gas = {"gas": "N2", "raman": True} if beside else {}
        add(f"capillary {case} chunks of eight{' beside Raman' * beside}",
            f"capillary/none/{case}, node chunks",
            partial(cap, **gas, **GRID_CASES[case]),
            PROJECTED, options=ExecutionOptions(node_chunk=8))  # fmt: skip
    for case in ("field full", "field conserving", "envelope full"):
        add(f"two-component {case}", f"two-component/adams/{case}",
            partial(averaged, evaluator, vector=True, **GRID_CASES[case]),
            PROJECTED | WINDOW,
            expect={"": _two_components})  # fmt: skip
    for envelope in (False, True):
        add(f"radial {'envelope' if envelope else 'field'}", "radial/adams/",
            partial(radial, evaluator, envelope=envelope, thg=envelope),
            PROJECTED | WINDOW, expect={NODES: 16})  # fmt: skip
    for case in ("field full", "envelope conserving"):
        mixture = partial(cap, gas={"He": 0.6, "Ar": 0.4}, **GRID_CASES[case])
        add(f"capillary {case} helium and argon", f"capillary/none/{case}, mixture",
            mixture, PROJECTED, expect=TWO, medium=True)  # fmt: skip
        add(f"capillary {case} helium and argon window",
            f"capillary/adams/{case}, mixture", partial(mixture, stepped=True),
            WINDOW)  # fmt: skip
    add("capillary varying composition", "capillary/adams/mixture along z",
        partial(falling, evaluator, 0.5, 2.0), WINDOW,
        expect={"": _ratios})  # fmt: skip
    add("capillary a member at zero density", "capillary/adams/a zero member",
        partial(falling, evaluator, 0.0, 2.0),
        {"source": route_evaluation_cpu, "steps": window_of(route_evaluation_cpu)},
        source_z=LENGTH, expect={"": _zero_member})  # fmt: skip
    for thg in (True, False):
        nitrogen = partial(cap, gas="N2", raman=True, thg=thg)
        add(f"capillary beside Raman thg={thg}", "capillary/none/Raman", nitrogen,
            PROJECTED, metadata=path(False, False, False), expect=RAMAN,
            medium=True)  # fmt: skip
        add(f"capillary beside Raman thg={thg} window", "capillary/adams/Raman",
            partial(nitrogen, stepped=True), WINDOW)  # fmt: skip
    for envelope in (False, True):
        kind = {"plasma": True, "thg": False, "envelope": envelope, "energy": 60e-6}
        grid = "envelope" if envelope else "field"
        plasma = {
            "problem.responses.envelope_plasma": partial(_held, envelope),
            "problem": embedded,
        }
        carrier = partial(capillary, "carrier-kerr", **kind)
        add(f"capillary {grid} beside plasma", f"capillary/none/{grid} plasma",
            partial(cap, **kind), {"source": capillary_evaluation},
            floor=partial(_built, carrier, None), expect=plasma,
            medium=True)  # fmt: skip
        add(f"capillary {grid} beside plasma window", f"capillary/adams/{grid} plasma",
            partial(cap, stepped=True, **kind), WINDOW,
            floor=partial(_built, partial(carrier, stepped=True), None))  # fmt: skip
    add("capillary Runge-Kutta window", "capillary/rk/field full",
        partial(cap, stepped=True, integrator=RungeKutta()), WINDOW,
        expect={"run_request.stepper.integrator": None})  # fmt: skip
    return rows


def leg_bound_rows() -> list[Row]:
    """Return krypton and xenon under the polynomial: one leg bound, one group."""
    rows: list[Row] = []
    add = partial(_add, rows, "polynomial")
    band = {"wavelength_limits": (180e-9, 4e-6), "energy": 5e-6}
    for case in ("field full", "field conserving", "envelope full"):
        pair = partial(capillary, "polynomial", gas={"Kr": 0.5, "Xe": 0.5}, **band,
                       **GRID_CASES[case])  # fmt: skip
        add(f"capillary {case} krypton and xenon", f"capillary/none/{case}, leg bound",
            pair, PROJECTED, expect=ONE_GROUP, medium=True)  # fmt: skip
        add(f"capillary {case} krypton and xenon window",
            f"capillary/adams/{case}, leg bound", partial(pair, stepped=True),
            WINDOW)  # fmt: skip
    add("general krypton and xenon", "general/adams/real, leg bound",
        partial(_general, "polynomial", "real", {"Kr": 0.5, "Xe": 0.5}, None),
        SOURCE, expect=ONE_GROUP, masked_launch=True)  # fmt: skip
    return rows


def route_rows(evaluator: str) -> list[Row]:
    """Return one evaluator's free-space, line, planar and general modal rows."""
    rows: list[Row] = []
    add = partial(_add, rows, evaluator, masked_launch=True)
    lone = evaluator == "symmetric"
    for line in (False, True):
        route = "line" if line else "cartesian"
        for case in FREE_CASES:
            add(f"{route} {case}", f"{route}/adams/{case}",
                partial(_free, evaluator, case, line=line), SOURCE | ROUTE_WINDOW,
                metadata=path(True, lone, lone and case != "field conserving"),
            )  # fmt: skip
    for case in ("field full", "field conserving"):
        add(f"cartesian {case} beside Raman", f"cartesian/adams/{case}, Raman",
            partial(_free, evaluator, case, beside=True), SOURCE | ROUTE_WINDOW,
            metadata=path(False, False, False), expect=RAMAN)  # fmt: skip
    _add(rows, evaluator, "cartesian one step", "cartesian/rk/field full",
         partial(_free_runge_kutta, evaluator), ROUTE_STEP)  # fmt: skip
    add("cartesian helium and argon", "cartesian/adams/mixture",
        partial(_free_mixture, evaluator), SOURCE | ROUTE_WINDOW,
        expect=TWO)  # fmt: skip
    add("cartesian chunks of twenty", "cartesian/adams/node chunks",
        partial(_free, evaluator, "field full"), SOURCE,
        options=ExecutionOptions(node_chunk=20),
        expect={NODES: lambda n: n % 20 != 0})  # fmt: skip
    for thg in (True, False):
        response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
        add(f"planar thg={thg}", f"planar/adams/thg={thg}",
            partial(planar_problem, [response]), SOURCE | ROUTE_WINDOW)  # fmt: skip
        add(f"planar thg={thg} beside Raman", "planar/adams/Raman",
            partial(planar_problem, [response, raman(thg)]), SOURCE)  # fmt: skip
    response = DispersiveKerrResponse(ARGON, evaluator)
    add("planar in-stage collar", "planar/adams/in-stage collar",
        partial(planar_problem, [response], absorbed=True), SOURCE | ROUTE_WINDOW,
        metadata={"dispersive_path": lambda found: found and found["in_place"] == lone},
        expect={"problem.responses.absorber_scheme": "nonlinear"})  # fmt: skip
    add("planar blocks of three rows", "planar/adams/node chunks",
        partial(_planar_blocks, evaluator), SOURCE)  # fmt: skip
    _add(rows, evaluator, "planar one step", "planar/rk/full",
         partial(planar_problem, [response], integrator=RungeKutta(), steps=1),
         ROUTE_STEP, medium=True)  # fmt: skip
    for form in ("real", "analytic"):
        for name, (gas, core) in STRANDS.items():
            add(f"general {name} {form}", f"general/adams/{form}, {name}",
                partial(_general, evaluator, form, gas, core),
                SOURCE | ROUTE_WINDOW, metadata=path(core is None, False, False),
                expect={f"{LEAF}.regions": lambda found: found is not None},
                unguarded=GENERAL_WINDOW)  # fmt: skip
    _add(rows, evaluator, "general one step", "general/rk/real, beside a solid core",
         partial(_general, evaluator, "real", ARGON_FILL, CORE_CHI3, adams=False),
         ROUTE_STEP)  # fmt: skip
    add("general a chunk without a gas node", "general/adams/real, node chunks",
        partial(_without_gas_chunk, evaluator), {"source": route_evaluation_cpu},
        quiet=("ignore::solprop.exceptions.GeneralModalBasisWarning",))  # fmt: skip
    return rows


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every dispersive row."""
    found = [*leg_bound_rows()]
    for evaluator in EVALUATORS:
        found += fused_rows(evaluator) + route_rows(evaluator)
    return tuple(found)

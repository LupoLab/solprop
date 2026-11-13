# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response: the catalogue's rows.

The cases of ``dispersive_cases.py`` (the fused path: multimode capillary,
mode-averaged, two-component, radial) and ``dispersive_route_cases.py`` (free space,
the line, the hybrid planar route, the general modal route), each with the bounds
those modules derive from summation lengths before any run. The term each row switches
off for its guard is the response itself: every channel's static coefficient zeroed.
"""

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from backends.dispersive_cases import (
    GRID_CASES,
    LENGTH,
    POLICIES,
    STEPS,
    averaged,
    capillary,
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
from parity_ladder import Bound, Built, Row
from solprop.backends.session import ExecutionOptions
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import MaterialSelection, Mixture
from solprop.profiles import Linear
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.problem import PropagationProblem
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


# --- the guard ------------------------------------------------------------------------


def without_response(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the dispersive response's every coefficient zeroed."""
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    silent = dataclasses.replace(leaf, coefficients=(0.0,) * len(leaf.coefficients))
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, dispersive_kerr=silent),
    )


# --- builders -------------------------------------------------------------------------


def _from(simulation: Simulation, options: ExecutionOptions | None = None) -> Built:
    """Translate a driver's run."""
    return Built(
        simulation.translate(), simulation._run_request(), simulation.grid, options
    )


def _assembled(case: tuple[Any, ...]) -> Built:
    """Wrap a hand-assembled ``(problem, run_request, grid)``."""
    return Built(*case)


def falling_mixture(evaluator: str, helium_end: float, argon_end: float) -> Simulation:
    """Return helium from 2 bar and argon from 0.5 bar, linear along the guide."""
    fill = Mixture.partial_pressures(
        {"He": Linear(2.0, helium_end, LENGTH), "Ar": Linear(0.5, argon_end, LENGTH)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            75e-6,
            LENGTH,
            fill,
            None,
            wavelength=800e-9,
            wavelength_limits=(150e-9, 4e-6),
            trange=150e-15,
            fwhm=10e-15,
            energy=30e-6,
            modes=3,
            materials=MaterialSelection(electronic=POLICIES[evaluator]),
            plasma=False,
            raman=False,
            backend="numpy_ref",
            n_save=2,
            absorber_scheme="split_step",
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
            noise=None,
        )


# --- expectations -------------------------------------------------------------------


def nodes(count: int) -> Callable[[Built], None]:
    """Expect the quadrature to have `count` nodes."""

    def _expect(built: Built) -> None:
        assert built.problem.transform.n_nodes == count

    return _expect


def channels(count: int, groups: tuple[tuple[int, ...], ...] | None = None):
    """Expect the leaf to have `count` channels, and these polynomial groups."""

    def _expect(built: Built) -> None:
        leaf = built.problem.responses.dispersive_kerr
        assert leaf is not None and leaf.n_channels == count
        if groups is not None:
            assert leaf.groups() == groups

    return _expect


def _all(*checks: Callable[[Built], None]) -> Callable[[Built], None]:
    """Expect every check."""

    def _expect(built: Built) -> None:
        for one in checks:
            one(built)

    return _expect


def _has_raman(built: Built) -> None:
    assert built.problem.flags.has_raman


def _two_components(built: Built) -> None:
    assert built.problem.transform.n_pol * built.problem.shape[1] == 2


def _regions(built: Built) -> None:
    leaf = built.problem.responses.dispersive_kerr
    assert leaf is not None and leaf.regions is not None


def _plasma(envelope: bool) -> Callable[[Built], None]:
    """Expect the envelope's own plasma exactly on an envelope grid, and both rules."""

    def _expect(built: Built) -> None:
        problem = built.problem
        assert (problem.responses.envelope_plasma is not None) == envelope
        transform = problem.transform
        assert not np.array_equal(transform.projection, transform.projection_coarse)

    return _expect


def _density_ratios(built: Built) -> None:
    """Helium falls from four times argon's density to a quarter of it."""
    start = scalars(built.problem, 0.0, 0.0).dispersive_kerr_densities
    end = scalars(built.problem, 0.0, LENGTH).dispersive_kerr_densities
    assert abs(start[0] / start[1] / 4.0 - 1.0) <= 1e-2
    assert abs(end[0] / end[1] / 0.25 - 1.0) <= 1e-2


def _zero_member(built: Built) -> None:
    densities = scalars(built.problem, 0.0, LENGTH).dispersive_kerr_densities
    assert densities[0] == 0.0
    assert densities[1] > 0.0


def path(alone: bool, in_place: bool, borrowed: bool) -> dict[str, object]:
    """Return the native ``dispersive_path`` metadata a row expects."""
    return {
        "dispersive_path": {"alone": alone, "in_place": in_place, "borrowed": borrowed}
    }


# --- rows -----------------------------------------------------------------------------


def _row(
    identity: str,
    route: str,
    evaluator: str,
    stepper: str,
    features: str,
    build: Callable[[], Built],
    rungs: dict[str, Bound],
    **fields: Any,
) -> Row:
    """Return a dispersive row; medium when it marches a window or says so."""
    fields.setdefault("medium", "steps" in rungs)
    return Row(
        id=f"dispersive {identity}",
        axes={
            "route": route,
            "response": f"dispersive {evaluator}",
            "stepper": stepper,
            "features": features,
        },
        build=build,
        rungs=rungs,
        switch_off=without_response,
        **fields,
    )


def fused_rows() -> list[Row]:
    """Return the fused path's rows: capillary, mode-averaged, two-component, radial."""
    projected = {"projected": capillary_evaluation}
    window = {"steps": window_of(capillary_evaluation)}
    rows = []
    for evaluator in sorted(POLICIES):
        symmetric = evaluator == "symmetric"
        for case, grid in GRID_CASES.items():
            analytic = case == "field conserving"
            rows += [
                _row(
                    f"capillary {case} {evaluator}",
                    "capillary multimode",
                    evaluator,
                    "none",
                    f"{case}, in-stage collar",
                    lambda e=evaluator, g=grid: _from(capillary(e, **g)),
                    projected,
                    metadata=path(True, False, symmetric and not analytic),
                    expect=nodes(33),
                ),
                _row(
                    f"capillary {case} {evaluator} one Runge-Kutta step",
                    "capillary multimode",
                    evaluator,
                    "rk",
                    f"{case}, split-step collar",
                    lambda e=evaluator, g=grid: _from(
                        capillary(
                            e,
                            stepped=True,
                            integrator=RungeKutta(),
                            save_planes=[LENGTH / STEPS],
                            n_save=None,
                            **g,
                        )
                    ),
                    {"one_step": one_step_of(capillary_evaluation)},
                    metadata=path(True, symmetric, symmetric and not analytic),
                ),
                _row(
                    f"capillary {case} {evaluator} ten steps",
                    "capillary multimode",
                    evaluator,
                    "adams",
                    f"{case}, split-step collar",
                    lambda e=evaluator, g=grid: _from(capillary(e, stepped=True, **g)),
                    window,
                    statistics=(
                        {"quadrature_error": alarm_of(capillary_evaluation)}
                        if case.startswith("field")
                        else {}
                    ),
                ),
                _row(
                    f"mode-averaged {case} {evaluator}",
                    "mode-averaged",
                    evaluator,
                    "adams",
                    case,
                    lambda e=evaluator, g=grid: _assembled(averaged(e, **g)),
                    projected | window,
                    expect=nodes(1),
                ),
            ]
        for case, raman_on in (
            ("field full", False),
            ("field conserving", False),
            ("field conserving", True),
            ("envelope full", False),
        ):
            rows.append(
                _row(
                    f"capillary {case} {evaluator} chunks of eight"
                    + (" beside Raman" if raman_on else ""),
                    "capillary multimode",
                    evaluator,
                    "none",
                    f"{case}, node chunks" + (", Raman" if raman_on else ""),
                    lambda e=evaluator, c=case, r=raman_on: _from(
                        capillary(e, gas="N2" if r else "Ar", raman=r, **GRID_CASES[c]),
                        ExecutionOptions(node_chunk=8),
                    ),
                    projected,
                )
            )
        for case in ("field full", "field conserving", "envelope full"):
            rows.append(
                _row(
                    f"two-component {case} {evaluator}",
                    "mode-averaged two-component",
                    evaluator,
                    "adams",
                    case,
                    lambda e=evaluator, c=case: _assembled(
                        averaged(e, vector=True, **GRID_CASES[c])
                    ),
                    projected | window,
                    expect=_two_components,
                )
            )
        for envelope in (False, True):
            rows.append(
                _row(
                    f"radial {'envelope' if envelope else 'field'} {evaluator}",
                    "radial",
                    evaluator,
                    "adams",
                    "envelope full" if envelope else "field conserving",
                    lambda e=evaluator, v=envelope: _assembled(
                        radial(e, envelope=v, thg=v)
                    ),
                    projected | window,
                    expect=nodes(16),
                )
            )
        mixture = {"He": 0.6, "Ar": 0.4}
        for case in ("field full", "envelope conserving"):
            rows += [
                _row(
                    f"capillary {case} {evaluator} helium and argon",
                    "capillary multimode",
                    evaluator,
                    "none",
                    f"{case}, mixture",
                    lambda e=evaluator, c=case, m=mixture: _from(
                        capillary(e, gas=m, **GRID_CASES[c])
                    ),
                    projected,
                    expect=channels(2),
                    medium=True,
                ),
                _row(
                    f"capillary {case} {evaluator} helium and argon ten steps",
                    "capillary multimode",
                    evaluator,
                    "adams",
                    f"{case}, mixture",
                    lambda e=evaluator, c=case, m=mixture: _from(
                        capillary(e, gas=m, stepped=True, **GRID_CASES[c])
                    ),
                    window,
                ),
            ]
        rows += [
            _row(
                f"capillary {evaluator} varying composition",
                "capillary multimode",
                evaluator,
                "adams",
                "mixture along z",
                lambda e=evaluator: _from(falling_mixture(e, 0.5, 2.0)),
                window,
                expect=_density_ratios,
            ),
            _row(
                f"capillary {evaluator} a member at zero density",
                "capillary multimode",
                evaluator,
                "adams",
                "mixture to a zero member",
                lambda e=evaluator: _from(falling_mixture(e, 0.0, 2.0)),
                {
                    "source": route_evaluation_cpu,
                    "steps": window_of(route_evaluation_cpu),
                },
                source_z=LENGTH,
                expect=_zero_member,
            ),
        ]
        for thg in (True, False):
            rows += [
                _row(
                    f"capillary {evaluator} beside Raman thg={thg}",
                    "capillary multimode",
                    evaluator,
                    "none",
                    "Raman",
                    lambda e=evaluator, t=thg: _from(
                        capillary(e, gas="N2", raman=True, thg=t)
                    ),
                    projected,
                    metadata=path(False, False, False),
                    expect=_has_raman,
                    medium=True,
                ),
                _row(
                    f"capillary {evaluator} beside Raman thg={thg} ten steps",
                    "capillary multimode",
                    evaluator,
                    "adams",
                    "Raman",
                    lambda e=evaluator, t=thg: _from(
                        capillary(e, gas="N2", raman=True, thg=t, stepped=True)
                    ),
                    window,
                ),
            ]
        for envelope in (False, True):
            kind = {"plasma": True, "thg": False, "envelope": envelope, "energy": 60e-6}
            grid_name = "envelope" if envelope else "field"
            rows += [
                _row(
                    f"capillary {grid_name} {evaluator} beside plasma",
                    "capillary multimode",
                    evaluator,
                    "none",
                    "envelope plasma" if envelope else "plasma",
                    lambda e=evaluator, k=kind: _from(capillary(e, **k)),
                    {"source": capillary_evaluation},
                    floor=lambda k=kind: _from(capillary("carrier-kerr", **k)),
                    expect=_plasma(envelope),
                    medium=True,
                ),
                _row(
                    f"capillary {grid_name} {evaluator} beside plasma ten steps",
                    "capillary multimode",
                    evaluator,
                    "adams",
                    "envelope plasma" if envelope else "plasma",
                    lambda e=evaluator, k=kind: _from(capillary(e, stepped=True, **k)),
                    window,
                    floor=lambda k=kind: _from(
                        capillary("carrier-kerr", stepped=True, **k)
                    ),
                ),
            ]
        rows.append(
            _row(
                f"capillary {evaluator} Runge-Kutta ten steps",
                "capillary multimode",
                evaluator,
                "rk",
                "field full",
                lambda e=evaluator: _from(
                    capillary(e, stepped=True, integrator=RungeKutta())
                ),
                window,
                expect=_no_integrator,
            )
        )
    band = {"wavelength_limits": (180e-9, 4e-6), "energy": 5e-6}
    for case in ("field full", "field conserving", "envelope full"):
        rows += [
            _row(
                f"capillary {case} krypton and xenon",
                "capillary multimode",
                "polynomial",
                "none",
                f"{case}, one leg bound",
                lambda c=case: _from(
                    capillary(
                        "polynomial",
                        gas={"Kr": 0.5, "Xe": 0.5},
                        **band,
                        **GRID_CASES[c],
                    )
                ),
                projected,
                expect=channels(2, ((0, 1),)),
                medium=True,
            ),
            _row(
                f"capillary {case} krypton and xenon ten steps",
                "capillary multimode",
                "polynomial",
                "adams",
                f"{case}, one leg bound",
                lambda c=case: _from(
                    capillary(
                        "polynomial",
                        gas={"Kr": 0.5, "Xe": 0.5},
                        stepped=True,
                        **band,
                        **GRID_CASES[c],
                    )
                ),
                window,
            ),
        ]
    return rows


def _no_integrator(built: Built) -> None:
    """Expect the Runge-Kutta march: no multistep integrator."""
    assert built.run_request.stepper.integrator is None


def _chunk_split(built: Built) -> None:
    """Expect the free plane's columns to end in a partial chunk of twenty."""
    assert built.problem.transform.n_nodes % 20


def _general_without_gas_chunk(evaluator: str) -> Built:
    """Return the general route cut at its first gas node: a chunk of core nodes."""
    built = _assembled(general_problem(strand(ARGON_FILL), evaluator, "real"))
    tables = built.problem.transform.modal_tables
    assert tables is not None
    region = np.asarray(tables.region_index)
    first_gas = int(np.flatnonzero(region == 1)[0])
    assert first_gas >= 1
    assert np.all(region[:first_gas] != 1)
    return dataclasses.replace(built, options=ExecutionOptions(node_chunk=first_gas))


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


GENERAL_WINDOW = (
    "the gas region moves the 2 nJ launch's ten steps by 2e-11 to 7e-11, the bound's "
    "own size: the window holds the general route's march, the source rung the response"
)
"""Why the general route's window carries no guard of its own."""


def route_rows() -> list[Row]:
    """Return the free-space, line, planar and general modal rows."""
    source = {"source": route_evaluation}
    window = {"steps": window_of(route_evaluation)}
    step = {"one_step": one_step_of(route_evaluation)}
    masked = {"masked_launch": True}
    rows = []
    for evaluator in EVALUATORS:
        lone = evaluator == "symmetric"
        for line in (False, True):
            for case in FREE_CASES:
                rows.append(
                    _row(
                        f"{'line' if line else 'free'} {case} {evaluator}",
                        "line" if line else "cartesian",
                        evaluator,
                        "adams",
                        case,
                        lambda e=evaluator, c=case, v=line: _free(e, c, line=v),
                        source | window,
                        metadata=path(True, lone, lone and case != "field conserving"),
                        **masked,
                    )
                )
        for case in ("field full", "field conserving"):
            rows.append(
                _row(
                    f"free {case} {evaluator} beside Raman",
                    "cartesian",
                    evaluator,
                    "adams",
                    f"{case}, Raman",
                    lambda e=evaluator, c=case: _free(e, c, beside=True),
                    source | window,
                    metadata=path(False, False, False),
                    expect=_has_raman,
                    **masked,
                )
            )
        rows += [
            _row(
                f"free {evaluator} one Runge-Kutta step",
                "cartesian",
                evaluator,
                "rk",
                "field full",
                lambda e=evaluator: Built(
                    *free_problem(
                        [DispersiveKerrResponse(ARGON, e)],
                        free_grid("field", True),
                        adams=False,
                    ),
                    free_grid("field", True),
                ),
                step,
            ),
            _row(
                f"free {evaluator} helium and argon",
                "cartesian",
                evaluator,
                "adams",
                "mixture",
                lambda e=evaluator: _free_mixture(e),
                source | window,
                expect=channels(2),
                **masked,
            ),
            _row(
                f"free {evaluator} chunks of twenty",
                "cartesian",
                evaluator,
                "adams",
                "node chunks",
                lambda e=evaluator: dataclasses.replace(
                    _free(e, "field full"), options=ExecutionOptions(node_chunk=20)
                ),
                source,
                expect=_chunk_split,
                **masked,
            ),
        ]
        for thg in (True, False):
            rows += [
                _row(
                    f"planar {evaluator} thg={thg}",
                    "planar",
                    evaluator,
                    "adams",
                    "full" if thg else "conserving",
                    lambda e=evaluator, t=thg: _assembled(
                        planar_problem([DispersiveKerrResponse(ARGON, e, thg=t)])
                    ),
                    source | window,
                    **masked,
                ),
                _row(
                    f"planar {evaluator} thg={thg} beside Raman",
                    "planar",
                    evaluator,
                    "adams",
                    "Raman",
                    lambda e=evaluator, t=thg: _assembled(
                        planar_problem(
                            [DispersiveKerrResponse(ARGON, e, thg=t), raman(t)]
                        )
                    ),
                    source,
                    **masked,
                ),
            ]
        rows += [
            _row(
                f"planar {evaluator} in-stage collar",
                "planar",
                evaluator,
                "adams",
                "in-stage collar",
                lambda e=evaluator: _assembled(
                    planar_problem([DispersiveKerrResponse(ARGON, e)], absorbed=True)
                ),
                source | window,
                metadata={
                    "dispersive_path": lambda found, lone=lone: (
                        found is not None and found["in_place"] == lone
                    )
                },
                expect=_nonlinear_collar,
                **masked,
            ),
            _row(
                f"planar {evaluator} blocks of three rows",
                "planar",
                evaluator,
                "adams",
                "node chunks",
                lambda e=evaluator: _planar_blocks_of(e),
                source,
                **masked,
            ),
            _row(
                f"planar {evaluator} one Runge-Kutta step",
                "planar",
                evaluator,
                "rk",
                "full",
                lambda e=evaluator: _assembled(
                    planar_problem(
                        [DispersiveKerrResponse(ARGON, e)],
                        integrator=RungeKutta(),
                        steps=1,
                    )
                ),
                step,
                medium=True,
            ),
        ]
        for form in ("real", "analytic"):
            for name, (gas, core) in STRANDS.items():
                rows.append(
                    _row(
                        f"general {name} {form} {evaluator}",
                        "general",
                        evaluator,
                        "adams",
                        f"{form}, {name}",
                        lambda e=evaluator, f=form, g=gas, c=core: _assembled(
                            general_problem(strand(g, core=c), e, f)
                        ),
                        source | window,
                        metadata=path(core is None, False, False),
                        expect=_regions,
                        unguarded={"steps": GENERAL_WINDOW},
                        **masked,
                    )
                )
        rows += [
            _row(
                f"general {evaluator} one Runge-Kutta step",
                "general",
                evaluator,
                "rk",
                "real, beside a solid core",
                lambda e=evaluator: _assembled(
                    general_problem(
                        strand(ARGON_FILL, core=CORE_CHI3), e, "real", adams=False
                    )
                ),
                step,
            ),
            _row(
                f"general {evaluator} a chunk without a gas node",
                "general",
                evaluator,
                "adams",
                "real, node chunks",
                lambda e=evaluator: _general_without_gas_chunk(e),
                {"source": route_evaluation_cpu},
                quiet=("ignore::solprop.exceptions.GeneralModalBasisWarning",),
                **masked,
            ),
        ]
    rows.append(
        _row(
            "general krypton and xenon",
            "general",
            "polynomial",
            "adams",
            "real, one leg bound",
            lambda: _assembled(
                general_problem(
                    strand(Mixture.fill({"Kr": 0.5, "Xe": 0.5}, 1.0)),
                    "polynomial",
                    "real",
                )
            ),
            source,
            expect=channels(2, ((0, 1),)),
            **masked,
        )
    )
    return rows


def _planar_blocks_of(evaluator: str) -> Built:
    """Return the planar route in blocks of three y rows for one evaluator."""
    built = _assembled(planar_problem([DispersiveKerrResponse(ARGON, evaluator)]))
    geometry = built.problem.transform.geometry
    assert geometry is not None
    return dataclasses.replace(
        built, options=ExecutionOptions(node_chunk=3 * geometry.shape[0])
    )


def _nonlinear_collar(built: Built) -> None:
    assert built.problem.responses.absorber_scheme == "nonlinear"


ROWS = (*fused_rows(), *route_rows())
"""Every dispersive row."""

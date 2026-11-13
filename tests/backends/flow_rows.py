# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled linear flow on a fixed schedule: its rows of the case catalogue.

**What is compared.** Native against ``numpy_ref`` on the coupled-flow cases, the field
after each case's ten fixed steps (``steps``, or ``last_plane`` through the driver):

- ``flow marches ...`` rows: the eight cases of ``tests/flow_parity.py`` -- the scalar,
  TE/TM, TE and TM walls, the walled taper and pressure gradient, the lossless taper
  coupling and the split general block -- on the default march and on the Runge-Kutta
  march (from ``tests/driver/test_native_flow_marches.py``, which keeps the flow's
  actions against the reference's and the random-block formation);
- ``coupled flow ...`` rows: the strong cases of ``tests/coupled_flow_cases.py`` -- the
  scalar, TE/TM, TE and TM walls, the walled taper under a pressure gradient and the
  lossless taper coupling, a millijoule at ten bar so that the source is carried
  through the flow -- on the default march (from
  ``tests/driver/test_native_coupled_flow.py``, which keeps the split block's run: it
  declares no schedule and its controller takes 147 steps, an adaptive comparison,
  #809);
- ``moving guide ...`` rows: the Kerr-cored moving guides of
  ``tests/moving_guide_cases.py`` -- the kinked graded-index taper, the step-index
  HE11/HE12 taper and the same taper in a complex basis -- on the default march (from
  ``tests/driver/test_native_moving_guide.py``), and ``moving guide bound ...``: the
  step-index pair and complex basis at the retired source-bound tests' 10 nJ over
  0.1 mm (from ``tests/driver/test_native_moving_guide_bound.py``, which keeps their
  sources: the source rung's zero-length trial is refused by a coupled flow);
- through the driver, the last saved plane as the output gives it back
  (``last_plane``): the scalar wall, the TE/TM wall and a walled taper beside
  nitrogen's default plasma and Raman response, one row per response (``wall beside N2
  ...``, from ``tests/driver/test_wall_beside_gas_responses.py``, which keeps the
  energy statistic and the record); the input shot-noise draw through each coupled
  flow (``input noise ...``, from ``tests/driver/test_coupled_input_noise.py``); the
  TE/TM wall on a taper (``vector wall taper driver``, from
  ``tests/driver/test_coherent_capillary.py``); and the scalar wall and the lossless
  taper coupling whose ``auto`` runs ``test_native_coupled_flow.py`` and
  ``test_taper_coupling.py`` keep (``scalar wall driver``, ``taper coupling driver``);
  and the kinked taper through the entry point, its plane and physical energy
  (``moving guide entry point``, from the deleted
  ``tests/driver/test_native_moving_entry.py``).

**The bounds are the old tests', fixed before native runs.** A ``flow marches`` row is
bounded by ``flow_parity.ten_step_bound`` (``B_10``): the declared conditioning ``C``
carried over ten steps, ``exp(Phi)`` at the launch's nonlinear scale, the reference's
attenuation ``||a_0|| / ||a_L||`` and each step's diagonal half, source and flow
products. The attenuation reads the reference's own ten steps, which the bound runs
once (:func:`reference_steps`) through the same session as the old test did.

A ``coupled flow`` row is bounded by ``coupled_flow_cases.parity_bound``, which is
absolute and reads the reference's ten steps and its linear propagation (the launch
scaled down by a thousand and back up), run once (:func:`reference_and_linear`): these
rows compare peaks relative to the launch's peak (``Row.scale``), as the old test did,
and assert its nonlinear increment ``rho > 1e-5`` (``expect``). A ``moving guide``
row is bounded the same way by ``moving_guide_cases.moving_parity_bound``, which adds
each stage's combination of the maps; a ``moving guide bound`` row by ``B_10``, below
1e-9 where the retired tests held it (``expect``). A ``wall beside N2`` row is
bounded by ``wall_gas_cases.parity_bound``, ``B_10`` with the source split by response,
on the reference's last plane. The other driver rows hold their old tests' fixed gates,
1e-12 or 1e-11 of the plane's peak or of the reference's saved launch plane's
(``Row.scale``), which no derivation backs (#807), each with a ``bound_note``. The
kinked taper's schedule accumulates to an ulp past its millimetre; the run ends there,
as the old tests accepted (``parity_ladder.lands``).

**The guard.** A flow row's term is the coupled flow: :func:`without_flow` drops
``linear.coupled_flow``, so the channels propagate on their diagonal alone. A moving
guide's is its maps' motion: :func:`frozen_maps` holds every stage at the launch's
maps. A wall-beside-gas row's is its response, the plasma or the Raman one, rebuilt off
at the entry point as the old test did (:func:`without_the_response`); an input-noise
row's is the draw, rebuilt with no noise (:func:`without_the_draw`). The reference
must then move by more than ``parity_ladder.DETECTION`` (a thousand) of the row's
bound. The old marches test asked the same of native's run
without the flow; the old strong-case test guarded the flow not at all, only that the
source moved the field (``rho``).

**How to read a row.** Every row is a :func:`flow_row` call, for example::

    flow_row(
        name="flow marches te_tm runge_kutta",
        stepper="rk",
        features="the TE/TM wall's azimuthal block, one constant section",
        why="Ten fixed Runge-Kutta steps of the TE/TM wall ...",
        build=partial(marches_case, "te_tm", "runge_kutta"),
        switch_off=without_flow,
        rungs={"steps": marches_bound},
    )

``name`` makes the test id; ``stepper`` and ``features`` are the axes the row covers.
``build`` makes the problem: here the TE/TM wall on the Runge-Kutta march.
``rungs`` names what is compared, with its bound: the field after the ten steps,
inside ``B_10``. ``switch_off`` removes the row's term for the guard; ``why`` is the
claim a failure quotes.
"""

import contextlib
import dataclasses
import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pytest

import coupled_flow_cases
import flow_parity
import input_noise_cases
import moving_guide_cases
import wall_gas_cases
from modal_provider_cases import grin_grid
from parity_ladder import Built, Row, Statistic, reference
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.modes import CoherentWall
from solprop.stats import StatsSpec
from solprop.translate.problem import PropagationProblem
from solprop.translate.stage import scalars

QUIET = (
    *(
        f"ignore::{category.__module__}.{category.__name__}"
        for category in flow_parity.QUIET
    ),
    "ignore:the temporal absorber attenuates",
    # Noise-free launches into time windows kept tight for speed: the boundary monitor
    # warns (SEP 21), and the window's edge is not what these rows hold.
    "ignore::solprop.exceptions.BoundaryWarning",
    "ignore::solprop.exceptions.HostMarchWarning",
)
"""The warning filters of the old modules, which every row's run needs."""

MOVING_QUIET = (
    *QUIET,
    # The step-index guides warn on the active band's basis, which is expected here.
    "ignore::solprop.exceptions.GeneralModalBasisWarning",
)
"""The moving guides' filters: the flow's, and the basis warning of the step-index
taper (``tests/driver/test_native_moving_guide_bound.py``'s)."""

# --- the cases ------------------------------------------------------------------------


@functools.cache
def _translated(name: str) -> tuple[Any, PropagationProblem]:
    """Return a ``flow_parity`` case's simulation and its problem, built once."""
    simulation = flow_parity.build(name)
    return simulation, simulation.translate()


MARCHES = ("default", "runge_kutta")
"""The default march (the exponential Adams method, the split block's Runge-Kutta)
and the Runge-Kutta march."""


def marches_case(name: str, march: str) -> Built:
    """Return a ``flow_parity`` case on `march` (:data:`MARCHES`), ten fixed steps."""
    simulation, problem = _translated(name)
    run_request = simulation._run_request()
    if march == "runge_kutta":
        run_request = run_request.with_stepper(
            dataclasses.replace(run_request.stepper, integrator=None)
        )
    return Built(problem, run_request, simulation.grid)


def coupled_case(name: str) -> Built:
    """Return a strong case of ``coupled_flow_cases``: ten fixed steps, its march."""
    case = coupled_flow_cases.CASES[name]()
    assert case.run_request.stepper.schedule is not None
    return Built(case.problem, case.run_request, case.grid)


def moving_case(name: str) -> Built:
    """Return a Kerr-cored moving guide of ``moving_guide_cases.CASES``, ten steps."""
    case = moving_guide_cases.CASES[name]()
    return Built(case.problem, case.run_request, case.grid)


def bound_case(descriptors: tuple, **basis: Any) -> Built:
    """Return a step-index source-bound case, ten steps over its first 0.1 mm."""
    problem, run_request, grid = moving_guide_cases.bound_step_problem(
        descriptors, **basis
    )
    return Built(problem, run_request, grid)


def off_route_case(name: str) -> Built:
    """Return a ``flow_parity`` case on the exponential Adams march, ten fixed steps."""
    simulation, problem = _translated(name)
    run_request = coupled_flow_cases.on_adams(simulation._run_request())
    return Built(problem, run_request, simulation.grid)


def wide_case(modes: int) -> Built:
    """Return the lossless taper with `modes` channels, ten fixed steps of 2 um."""
    simulation = coupled_flow_cases.wide_simulation(modes)
    return Built(simulation.translate(), simulation._run_request(), simulation.grid)


@contextlib.contextmanager
def routed_off(route: str):
    """Hold native's flow off the eigenbasis route, every bin or every other one.

    ``coupled_flow_cases.off_route``, for the whole check: every backend the row
    constructs forms its flow under it (the reference's augmented exponential takes no
    route).
    """
    with pytest.MonkeyPatch.context() as patch:
        coupled_flow_cases.off_route(patch, route)
        yield


def moving_entry_case() -> Built:
    """Return the kinked graded-index taper through the entry point, ten steps.

    Observing the physical energy alone, the old test's statistic; no absorber and no
    window watch, as there.
    """
    simulation = moving_guide_cases.ten_step_simulation(
        moving_guide_cases.grin_taper(moving_guide_cases.kinked_radius(), kerr=True),
        grin_grid(),
        stats=StatsSpec(names=("physical_energy",)),
        **(moving_guide_cases.GRIN_PULSE | {"energy": moving_guide_cases.GRIN_ENERGY}),
    )
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber=None, boundary_check=None
    )
    return driver_case(simulation)


def driver_case(simulation: Any) -> Built:
    """Return a driver's row: the simulation's translation, request, grid and itself."""
    return Built(
        simulation.translate(),
        simulation._run_request(),
        simulation.grid,
        simulation=simulation,
    )


def wall_gas_case(name: str, **overrides: Any) -> Built:
    """Return a wall beside nitrogen's default plasma and Raman, ten fixed steps.

    Observing the energy alone: the default statistics read the plasma column, which
    the guard's problem without the plasma does not carry, and the field does not
    depend on what is observed.
    """
    return driver_case(
        wall_gas_cases.simulation(
            name, **({"stats": StatsSpec(names=("energy",))} | overrides)
        )
    )


def noise_case(route: str, noise: str | None = "input") -> Built:
    """Return an input-noise route's run on the reference, ten fixed steps."""
    build = input_noise_cases.ROUTES[route]
    return driver_case(build(noise, input_noise_cases.PULSE_ENERGY, "numpy_ref"))


def vector_wall_taper() -> Built:
    """Return the TE/TM wall on the qualification taper, a nanojoule, ten steps."""
    taper = coupled_flow_cases.taper()
    return driver_case(
        coupled_flow_cases.capillary(
            wall=coupled_flow_cases.VECTOR_WALL,
            modes=coupled_flow_cases.vector_modes(radius=taper),
            radius=taper,
        )
    )


def scalar_wall() -> Built:
    """Return the scalar wall of the qualification capillary, a nanojoule, ten steps."""
    return driver_case(coupled_flow_cases.capillary(wall=CoherentWall()))


def lossless_taper() -> Built:
    """Return the lossless taper coupling, no wall, a nanojoule, ten steps."""
    return driver_case(coupled_flow_cases.lossless())


# --- the bounds -----------------------------------------------------------------------


def reference_steps(built: Built) -> np.ndarray:
    """Return the reference's state after the built row's ten steps, run once.

    ``flow_parity.run``, the old test's own run; kept with the row's readings under a
    key that names no rung.
    """
    key = "reference ten steps"
    if key not in built.readings:
        built.readings[key] = flow_parity.run(
            NumpyRefBackend, built.problem, built.run_request, built.grid
        )
    return built.readings[key]


def marches_bound(built: Built) -> float:
    """Return ``flow_parity.ten_step_bound`` on the case's scales (``B_10``)."""
    scales = flow_parity.flow_scales(
        built.problem, built.run_request, built.grid, reference_steps(built)
    )
    return flow_parity.ten_step_bound(scales)


def launch_peak(built: Built) -> float:
    """Return the launch's peak modulus: the scale of the strong cases' bounds."""
    return float(np.max(np.abs(built.problem.initial.field0)))


def _as_case(built: Built) -> coupled_flow_cases.Case:
    """Return the built row as the coupled-flow harness's case, its length the end's."""
    length = float(built.run_request.stepper.z_end)
    return coupled_flow_cases.Case(built.problem, built.run_request, built.grid, length)


def reference_and_linear(built: Built) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference's ten steps and its linear propagation, run once.

    ``coupled_flow_cases.reference_and_linear``: the launch, and the launch scaled
    down by a thousand and back up, which is the same launch propagated linearly to a
    millionth of its nonlinear increment.
    """
    key = "reference and linear ten steps"
    if key not in built.readings:
        built.readings[key] = coupled_flow_cases.reference_and_linear(_as_case(built))
    return built.readings[key]


def increment(built: Built) -> float:
    """Return ``rho``, the nonlinear increment per step relative to the launch's peak.

    The reference's ten steps less its linear propagation, over the launch's peak and
    the step count.
    """
    expected, linear = reference_and_linear(built)
    scale = launch_peak(built)
    return float(np.max(np.abs(expected - linear))) / scale / coupled_flow_cases.STEPS


def coupled_bound(built: Built) -> float:
    """Return ``coupled_flow_cases.parity_bound`` relative to the launch's peak."""
    expected, linear = reference_and_linear(built)
    return coupled_flow_cases.parity_bound(
        _as_case(built), expected, linear
    ) / launch_peak(built)


def moving_bound(built: Built) -> float:
    """Return ``moving_guide_cases.moving_parity_bound`` relative to the launch's peak.

    The coupled flow's parity bound plus each stage's combination of the maps.
    """
    expected, linear = reference_and_linear(built)
    return moving_guide_cases.moving_parity_bound(
        _as_case(built), expected, linear
    ) / launch_peak(built)


def bound_below_a_billionth(built: Built) -> bool:
    """Return whether ``B_10`` is below 1e-9, where the retired tests held it."""
    return 0.0 < marches_bound(built) < 1e-9


def off_route_bound(built: Built) -> float:
    """Return ``B_10`` at the larger of the actions' and the augmented exponential's
    formation bounds (``flow_parity.adams_formation_bound``)."""
    scales = flow_parity.flow_scales(
        built.problem, built.run_request, built.grid, reference_steps(built)
    )
    formation = max(
        scales.formation,
        flow_parity.adams_formation_bound(built.problem, built.run_request, built.grid),
    )
    return flow_parity.ten_step_bound(dataclasses.replace(scales, formation=formation))


def routed(route: str) -> Callable[[Built], bool]:
    """Return a check that native holds the row's bins off the route, as declared.

    Every bin off it (``"off"``) or some (``"mixed"``), the Schur factor's lower
    triangle zero, the resident Adams engine stepping it, and ``B_10`` below 1e-8, as
    the old test asserted.
    """

    def check(built: Built) -> bool:
        from solprop_native import (  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
            NativeBackend,
            backend,
        )

        held = backend._host_flow(built.problem)
        assert held is not None
        off = not np.any(held.route) if route == "off" else not np.all(held.route)
        native = NativeBackend(built.problem, built.run_request, grid=built.grid)
        try:
            resident = type(native._resident).__name__ == "_ResidentAdams"
        finally:
            native.close()
        return (
            off
            and bool(np.any(held.route)) == (route == "mixed")
            and bool(np.all(np.tril(held.triangular, -1) == 0.0))
            and resident
            and 0.0 < off_route_bound(built) < 1e-8
        )

    return check


def twice_the_field_s(built: Built, name: str, bound: float) -> Statistic:
    """Return the physical energy's rule: relative, within twice the field's bound.

    An energy is a quadratic form of the state, so twice the state's relative bound.
    """
    del built
    assert name == "physical_energy"
    return Statistic(2.0 * bound, elementwise=True)


def exercised_below_a_millionth(built: Built) -> bool:
    """Return whether ``rho > 1e-5`` and the bound is under 1e-6 of the launch."""
    return exercises_the_nonlinear_half(built) and 0.0 < coupled_bound(built) < 1e-6


def wall_gas_bound(name: str) -> Callable[[Built], float]:
    """Return a wall-beside-gas case's ten-step bound on native, from the reference.

    ``wall_gas_cases.parity_bound``: ``B_10`` with the source split by response, read
    on the reference's last plane.
    """

    def bound(built: Built) -> float:
        return wall_gas_cases.parity_bound(
            name, "native", reference(built, "last_plane").field
        )

    return bound


def fixed(value: float) -> Callable[[Built], float]:
    """Return a fixed bound: an old test's own constant, underived (#807)."""

    def bound(built: Built) -> float:
        del built
        return value

    return bound


def launch_plane_peak(built: Built) -> float:
    """Return the peak modulus of the reference's saved launch plane.

    The scale of the entry points' ``auto`` checks: the plane at z = 0 as the output
    keeps it (the reference's complete run, :func:`parity_ladder.reference`), not the
    problem's launch.
    """
    return float(np.max(np.abs(reference(built, "run").field[0])))


def exercises_the_nonlinear_half(built: Built) -> bool:
    """Return whether the source moves the field through the flow: ``rho > 1e-5``."""
    return increment(built) > 1e-5


# --- the switch-offs ------------------------------------------------------------------


def without_flow(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its coupled flow: each channel on its diagonal."""
    assert problem.linear.coupled_flow is not None
    return dataclasses.replace(
        problem, linear=dataclasses.replace(problem.linear, coupled_flow=None)
    )


def frozen_maps(problem: PropagationProblem) -> PropagationProblem:
    """Return a moving guide's problem with its maps frozen at the launch.

    Every section's synthesis and projection terms are replaced by one constant term,
    the maps the launch's stage combines (its ``transform_weights``): a stage anywhere
    then reads the entrance's maps, exactly, and the guide's taper reaches the source no
    more. The flow and its connection are kept.
    """
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None and tables.projection is not None
    weights = np.asarray(scalars(problem, 0.0, 0.0).transform_weights)
    frozen = {}
    for name in ("synthesis", "projection"):
        terms = np.asarray(getattr(tables, name))
        flat = terms.reshape(-1, *terms.shape[2:])
        constant = np.zeros_like(terms)
        constant[:, 0] = np.tensordot(weights, flat, axes=1)
        frozen[name] = constant
    transform = dataclasses.replace(
        problem.transform, modal_tables=dataclasses.replace(tables, **frozen)
    )
    return dataclasses.replace(
        problem, route=dataclasses.replace(problem.route, transform=transform)
    )


def without_the_response(
    name: str, response: str
) -> Callable[[PropagationProblem], PropagationProblem]:
    """Return a switch-off rebuilding a wall-beside-gas case with `response` off.

    ``plasma=False`` or ``raman=False`` at the entry point, as the old test switched
    them.
    """

    def switch(problem: PropagationProblem) -> PropagationProblem:
        rebuilt = wall_gas_case(name, **{response: False}).problem
        assert rebuilt.shape == problem.shape
        return rebuilt

    return switch


def without_the_draw(route: str) -> Callable[[PropagationProblem], PropagationProblem]:
    """Return a switch-off rebuilding an input-noise route with no noise at all."""

    def switch(problem: PropagationProblem) -> PropagationProblem:
        rebuilt = noise_case(route, None).problem
        assert rebuilt.shape == problem.shape
        return rebuilt

    return switch


# --- the rows -------------------------------------------------------------------------

ROUTES = {
    "scalar": "capillary (coherent wall)",
    "te_tm": "capillary (coherent wall)",
    "taper": "capillary (coherent wall)",
    "gradient": "capillary (coherent wall)",
    "te": "capillary (coherent wall)",
    "tm": "capillary (coherent wall)",
    "lossless": "capillary (taper coupling)",
    "split": "general guide (split block)",
}
"""Each ``flow_parity`` case's route."""

FEATURES = {
    "scalar": "the scalar wall over three channels, one constant section",
    "te_tm": "the TE/TM wall's azimuthal block, one constant section",
    "taper": "the scalar wall on a taper, a frozen residual per section",
    "gradient": "the scalar wall under a pressure gradient, frozen residuals",
    "te": "the TE/TM wall beside TE_01 and TE_02, E_x and E_y resolved",
    "tm": "the TE/TM wall beside TM_01 and TM_02, E_x and E_y resolved",
    "lossless": "the lossless taper coupling, no wall",
    "split": "the split pair's declared rotation, a Kerr core",
}
"""Each ``flow_parity`` case's features."""


def flow_row(
    *,
    name: str,
    stepper: str,
    features: str,
    build: Callable[[], Built],
    route: str,
    response: str = "kerr",
    **fields: Any,
) -> Row:
    """Return a coupled-flow row; it marches, so it is medium.

    Unless it is slow by cost (``slow``, the reason). Its id is `name`; `fields` are
    the rest of :class:`parity_ladder.Row`.
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
        medium=not fields.get("slow"),
        quiet=QUIET,
        **fields,
    )


def _stepper(name: str, march: str) -> str:
    """Return the stepper axis of a case on a march: the split block's default is RK."""
    return "adams" if march == "default" and name != "split" else "rk"


def _marches_rows() -> tuple[Row, ...]:
    """Return the eight ``flow_parity`` cases on both marches."""
    return tuple(
        flow_row(
            name=f"flow marches {name} {march}",
            stepper=_stepper(name, march),
            route=ROUTES[name],
            features=FEATURES[name],
            why=f"Ten fixed steps of the {name} case's coupled flow on the {march} "
            "march are the reference's, the flow's actions formed natively.",
            bound_note="flow_parity's B_10, written before any measurement: the "
            "declared conditioning over ten steps, exp(Phi) at the launch and the "
            "reference's attenuation, times each step's diagonal half, source and "
            "flow products.",
            build=partial(marches_case, name, march),
            switch_off=without_flow,
            rungs={"steps": marches_bound},
            window=flow_parity.STEPS if name == "split" else None,
        )
        for name in flow_parity.CASES
        for march in MARCHES
    )


COUPLED = {
    "scalar_wall": (
        "capillary (coherent wall)",
        "the scalar wall over three channels, 1 mJ at 10 bar",
    ),
    "te_tm_wall": (
        "capillary (coherent wall)",
        "the TE/TM wall's azimuthal block over two families, 1 mJ at 10 bar",
    ),
    "te_wall": (
        "capillary (coherent wall)",
        "the TE/TM wall beside TE_01 and TE_02, 1 mJ at 10 bar",
    ),
    "tm_wall": (
        "capillary (coherent wall)",
        "the TE/TM wall beside TM_01 and TM_02, 1 mJ at 10 bar",
    ),
    "taper_gradient": (
        "capillary (coherent wall)",
        "the scalar wall on a taper under a pressure gradient, frozen residuals",
    ),
    "lossless_taper": (
        "capillary (taper coupling)",
        "the lossless taper coupling, no wall, 1 mJ at 10 bar",
    ),
}
"""Each strong case's route and features, by ``coupled_flow_cases.CASES`` name."""


def _coupled_rows() -> tuple[Row, ...]:
    """Return the strong cases of ``coupled_flow_cases``, ten steps each."""
    return tuple(
        flow_row(
            name=f"coupled flow {name}",
            stepper="adams",
            route=COUPLED[name][0],
            features=COUPLED[name][1],
            why=f"Ten fixed steps of the {name} case, launched strong enough that "
            "its source is carried through the flow, are the reference's.",
            bound_note="coupled_flow_cases.parity_bound, written before any "
            "measurement and relative to the launch's peak: the shared linear work, "
            "the flow's formation and carries, the source on each stage's "
            "nonlinear increment rho and the explicit residual, over ten steps that "
            "each amplify by 1 + 3 rho + r. The row asserts rho > 1e-5 (the old "
            "test's own check, `expect`).",
            build=partial(coupled_case, name),
            switch_off=without_flow,
            rungs={"steps": coupled_bound},
            metric="peak",
            scale=launch_peak,
            expect={"": exercises_the_nonlinear_half},
        )
        for name in COUPLED
    )


MOVING = {
    "grin_kinked": (
        "the kinked graded-index taper, two sections of degree four, 10 nJ",
        "",
    ),
    "step_taper": (
        "HE11 and HE12 on the 10 % step-index taper, its first 0.2 mm, 1 nJ",
        "slow by cost (#792): the step-index connection's translation takes tens of "
        "seconds, minutes on a busy machine (the old test's mark); 7 s on the Mac",
    ),
    "complex_basis": (
        "the circular HE11 pair beside the HE12 parity pair, its first 0.2 mm, 1 nJ",
        "slow by cost (#792): the step-index connection's translation takes tens of "
        "seconds, minutes on a busy machine (the old test's mark); 14 s on the Mac",
    ),
}
"""Each Kerr-cored moving guide's features and its slow-by-cost reason (none: medium),
by ``moving_guide_cases.CASES`` name."""

BOUND_CASES = {
    "pair": (moving_guide_cases.BOUND_PAIR, {}, "HE11 and HE12", "6.5 s"),
    "complex": (
        moving_guide_cases.BOUND_COMPLEX,
        moving_guide_cases.BOUND_COMPLEX_BLOCKS,
        "the circular HE11 pair beside the HE12 parity pair",
        "13 s",
    ),
}
"""Each source-bound step-index case: its descriptors, blocks and basis, what it is and
its translation's time on the Mac."""


def _moving_row(name: str) -> Row:
    """Return a Kerr-cored moving guide's row, slow by cost or medium."""
    features, slow = MOVING[name]
    return Row(
        id=f"moving guide {name}",
        axes={
            "route": "general guide (moving)",
            "stepper": "adams",
            "response": "kerr",
            "features": features,
        },
        why=f"Ten fixed steps of the {name} moving guide are the reference's: native "
        "combines the maps' Chebyshev terms per stage, as the reference does.",
        bound_note="moving_guide_cases.moving_parity_bound, written before any "
        "measurement and relative to the launch's peak: the coupled flow's parity "
        "bound plus each stage's combination of the maps, carried on the nonlinear "
        "increment rho. The row asserts rho > 1e-5 (the old test's own check, "
        "`expect`).",
        build=partial(moving_case, name),
        switch_off=frozen_maps,
        rungs={"steps": moving_bound},
        metric="peak",
        scale=launch_peak,
        expect={"": exercises_the_nonlinear_half},
        medium=not slow,
        slow=slow,
        quiet=MOVING_QUIET,
        window=None,
    )


def _bound_row(name: str) -> Row:
    """Return a source-bound step-index case's ten steps: slow by cost."""
    descriptors, basis, what, seconds = BOUND_CASES[name]
    return Row(
        id=f"moving guide bound {name} ten steps",
        axes={
            "route": "general guide (moving)",
            "stepper": "adams",
            "response": "kerr",
            "features": f"{what} on the step-index taper, its first 0.1 mm, 10 nJ",
        },
        why=f"Ten fixed steps of {what} on the step-index taper at the retired tests' "
        "energy are the reference's, within the flow's ten-step bound.",
        bound_note="flow_parity's B_10, written before any measurement; the row also "
        "asserts it is below 1e-9, where the retired tests held it (`expect`).",
        build=partial(bound_case, descriptors, **basis),
        switch_off=frozen_maps,
        rungs={"steps": marches_bound},
        expect={"": bound_below_a_billionth},
        slow="slow by cost (#792): the step-index connection's translation takes tens "
        f"of seconds (the old test's mark); {seconds} on the Mac",
        quiet=MOVING_QUIET,
        window=None,
    )


WALL_GAS = {
    "scalar": ("capillary (coherent wall)", "the scalar wall"),
    "te_tm": ("capillary (coherent wall)", "the TE/TM wall's azimuthal block"),
    "taper": ("capillary (coherent wall)", "the scalar wall on a taper"),
}
"""Each wall-beside-gas case's route and what it is, by ``wall_gas_cases.CASES`` name.

The old test marked the TE/TM wall and the taper slow; each of their parity tests,
both backends' runs and both responses switched off, took 2.8 to 6.6 s on the Mac, so
their rows are medium."""


def _wall_gas_rows() -> tuple[Row, ...]:
    """Return each wall beside nitrogen's plasma and Raman, one row per response."""
    return tuple(
        flow_row(
            name=f"wall beside N2 {name} {response}",
            stepper="adams",
            route=WALL_GAS[name][0],
            response="kerr, plasma, raman",
            features=f"{WALL_GAS[name][1]} beside N2's default plasma and Raman, "
            "30 uJ through prop_capillary's defaults",
            why=f"Ten fixed steps of {WALL_GAS[name][1]} beside nitrogen's own plasma "
            "and Raman response, which the entry point turns on by default, leave "
            f"the reference's last plane, the {response} acting.",
            bound_note="wall_gas_cases.parity_bound, written before any measurement: "
            "B_10 with the source split by response, each with its own share, "
            "rounding and Lipschitz factor, read on the reference's last plane.",
            build=partial(wall_gas_case, name),
            switch_off=without_the_response(name, response),
            rungs={"last_plane": wall_gas_bound(name)},
        )
        for name in wall_gas_cases.CASES
        for response in ("plasma", "raman")
    )


NOISE_ROUTES = {
    "capillary (coherent wall)": ("adams", "wall"),
    "capillary (taper coupling)": ("adams", "taper"),
    "general guide (moving)": ("rk", "moving"),
    "general guide (split block)": ("rk", "split"),
}
"""Each input-noise route's stepper and the short name of its row."""


def _noise_rows() -> tuple[Row, ...]:
    """Return the input draw carried through each coupled flow."""
    return tuple(
        flow_row(
            name=f"input noise {NOISE_ROUTES[route][1]}",
            stepper=NOISE_ROUTES[route][0],
            route=route,
            features="the input shot-noise draw in the entrance state, 1 pJ, "
            "through the driver",
            why="The input draw needs no backend path of its own: native carries it "
            "through the coupled flow to the reference's last plane.",
            bound_note="1e-12 of the plane's peak, a rounding budget fixed before "
            "the comparison but not derived (#807): ten steps of transforms, flow "
            "actions and stage sums, a few hundred ulps of the peak at most.",
            build=partial(noise_case, route),
            switch_off=without_the_draw(route),
            rungs={"last_plane": fixed(1e-12)},
            metric="peak",
        )
        for route in input_noise_cases.ROUTES
    )


def _entry_rows() -> tuple[Row, ...]:
    """Return the coupled flows' driver runs: the vector wall and the auto checks."""
    return (
        flow_row(
            name="vector wall taper driver",
            stepper="adams",
            route="capillary (coherent wall)",
            features="the TE/TM wall on a taper, a dense frozen-residual flow, 1 nJ",
            why="The TE/TM wall on a taper, a dense frozen-residual flow, leaves the "
            "reference's last plane through the entry point.",
            bound_note="1e-11 of the plane's peak, an underived gate (#807) above the "
            "flow's parity bounds, at most 5.5e-12 of the launch's peak after ten "
            "steps (from test_coherent_capillary).",
            build=vector_wall_taper,
            switch_off=without_flow,
            rungs={"last_plane": fixed(1e-11)},
            metric="peak",
        ),
        flow_row(
            name="scalar wall driver",
            stepper="adams",
            route="capillary (coherent wall)",
            features="the scalar wall, 1 nJ, through the entry point",
            why="The scalar wall's ten steps through the entry point leave the "
            "reference's last plane on native, which backend='auto' resolves to.",
            bound_note="1e-12 of the reference's saved launch plane's peak, an "
            "underived gate (#807): a millionth of the launch is generous for ten "
            "steps of a nanojoule, where the source is below rounding. The test of "
            "auto's resolution keeps test_native_coupled_flow.",
            build=scalar_wall,
            switch_off=without_flow,
            rungs={"last_plane": fixed(1e-12)},
            metric="peak",
            scale=launch_plane_peak,
        ),
        flow_row(
            name="taper coupling driver",
            stepper="adams",
            route="capillary (taper coupling)",
            features="the lossless taper coupling, no wall, 1 nJ",
            why="The lossless taper's coupling, the lowest channel's parameter "
            "expansion, leaves the reference's last plane on native, which "
            "backend='auto' resolves to.",
            bound_note="1e-12 of the reference's saved launch plane's peak, an "
            "underived gate (#807). The test of auto's resolution keeps "
            "test_taper_coupling.",
            build=lossless_taper,
            switch_off=without_flow,
            rungs={"last_plane": fixed(1e-12)},
            metric="peak",
            scale=launch_plane_peak,
        ),
    )


def _off_route_rows() -> tuple[Row, ...]:
    """Return ``flow_parity``'s eight cases with bins off the eigenbasis route."""
    return tuple(
        flow_row(
            name=f"flow off route {name} {route}",
            stepper="adams",
            route=ROUTES[name],
            features=f"{FEATURES[name]}; "
            + ("every bin" if route == "off" else "every other bin")
            + " off the eigenbasis route, the augmented exponential",
            why=f"Ten fixed Adams steps of the {name} case with "
            + ("every bin" if route == "off" else "every other bin")
            + " off the flow's eigenbasis route, formed by the augmented exponential, "
            "are the reference's.",
            bound_note="flow_parity's B_10 at the larger of the actions' and the "
            "augmented exponential's formation bounds (adams_formation_bound), "
            "written before any measurement; the row's `expect` holds the routing "
            "and B_10 below 1e-8, as the old test did.",
            build=partial(off_route_case, name),
            switch_off=without_flow,
            rungs={"steps": off_route_bound},
            expect={"": routed(route)},
            context=partial(routed_off, route),
            window=flow_parity.STEPS if name == "split" else None,
        )
        for name in flow_parity.CASES
        for route in coupled_flow_cases.OFF_ROUTES
    )


WIDE_SLOW = {
    17: "",
    24: "slow by cost (#792): the old test's mark, beside its 24-channel session; "
    "14 s on the Mac",
}
"""Each wide flow's slow-by-cost reason (none: medium)."""


def _wide_rows() -> tuple[Row, ...]:
    """Return the wide lossless tapers, past the 16 channels a host task holds."""
    return tuple(
        flow_row(
            name=f"flow wide {modes} channels",
            stepper="adams",
            route="capillary (taper coupling)",
            features=f"the lossless taper coupling over {modes} channels, the "
            "resident Adams engine, ten steps of 2 um",
            why=f"Ten fixed steps of a {modes}-channel coupled flow, wider than a host "
            "task's stack, on the resident Adams engine are the reference's.",
            bound_note="coupled_flow_cases.parity_bound relative to the launch's "
            "peak, written before any measurement; the row's `expect` holds rho > "
            "1e-5 and the bound below a millionth, as the old test did.",
            build=partial(wide_case, modes),
            switch_off=without_flow,
            rungs={"steps": coupled_bound},
            metric="peak",
            scale=launch_peak,
            expect={"": exercised_below_a_millionth},
            slow=WIDE_SLOW[modes],
        )
        for modes in coupled_flow_cases.WIDE
    )


def _moving_entry_row() -> Row:
    """Return the kinked taper through the entry point: its plane and its energy."""
    return Row(
        id="moving guide entry point",
        axes={
            "route": "general guide (moving)",
            "stepper": "adams",
            "response": "kerr",
            "features": "the kinked graded-index taper through from_waveguide, 10 nJ",
        },
        why="A moving guide's ten steps through the entry point leave the "
        "reference's last plane and physical energy on native.",
        bound_note="moving_guide_cases.moving_parity_bound relative to the launch's "
        "peak on the plane, written before any measurement; the energy, a quadratic "
        "form of it, within twice that, relative. The old test held the energy alone, "
        "with pytest.approx's default absolute 1e-12 J beside the relative bound: "
        "1e-4 of this 1e-8 J (#806). The row holds the relative bound only.",
        build=moving_entry_case,
        switch_off=frozen_maps,
        rungs={"last_plane": moving_bound},
        metric="peak",
        scale=launch_peak,
        every_statistic=twice_the_field_s,
        medium=True,
        quiet=MOVING_QUIET,
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the coupled linear flow on a fixed schedule."""
    return (
        *_marches_rows(),
        *_coupled_rows(),
        *(_moving_row(name) for name in MOVING),
        *(_bound_row(name) for name in BOUND_CASES),
        _moving_entry_row(),
        *_wall_gas_rows(),
        *_noise_rows(),
        *_entry_rows(),
        *_off_route_rows(),
        *_wide_rows(),
    )

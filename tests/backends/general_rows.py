# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route on a fixed schedule: its rows of the case catalogue.

**What is compared.** Native against ``numpy_ref`` on the general modal route's parity
cases (``tests/general_parity.py``): the degenerate HE11 parity pair (real and analytic
cubic), the same pair in a circular basis, a delayed silica tensor, the split TE01/TM01
block and a graded-index guide of two LP modes, each launched with random phases in
every channel and scaled to a nonlinear scale ``Phi = 0.5`` over the millimetre:

- ``source``: one evaluation of the whole stage source at the launch (from
  ``test_native_general_source_bound.py``);
- ``steps``: the field after the case's ten fixed Runge-Kutta steps (the same module);
- the pair under the split temporal collar, launched with content outside the active
  band, on either march (from ``test_native_general_split.py``, which keeps the
  out-of-band bins' exact zeros on both backends);
- a guide with no responding region, ten steps of its linear flow (the same module).

**The bounds are the old tests', fixed before native runs.**
``general_parity.source_bound`` counts one stage's roundings (synthesis, two transforms,
the contraction, the projection) times the projected sum's measured conditioning at the
launch; ``general_parity.ten_step_bound`` carries it through ten steps at the launch's
nonlinear scale and adds both backends' rounding of each step's linear half,
``exp(Phi) (1.645 Phi eps_src + 10 * 32 u)``; the linear guide's bound is that at
``Phi = 0``. Each is a function of the scaled launch, read from the case's own cache.
The schedule's ten steps of 0.1 mm end an ulp short of the millimetre, where the old
tests' runs ended: the rows observe its tenth edge (``window``), their field to the bit.

**The guard.** A source row's term is the route's responses: :func:`without_source`
rebuilds the case with none, so the reference's source is zero and its ten steps are the
linear flow alone (the old ten-step test's own check, now at every rung). A split-collar
row's term is the collar (``session_rows.without_collar``), and the linear guide's its
flow (``session_rows.without_modal_flow`` zeroes the constant exponent). Each guarded
rung's reference must move by more than ``parity_ladder.DETECTION`` (a thousand) of its
bounds.

**Not here.** The moving guide under the split collar (``slow``, an absolute bound read
from three more reference runs of the quiet launch) stays in
``test_native_general_split.py``, beside the coupled-flow parity of the moving guides
(``backends/flow_rows.py``'s ``moving guide ...`` rows). So does the entry point's run:
the saved plane in the output's own convention, which the ``last_plane`` rung reads
now, though the run has not moved yet.

**How to read a row.** Every row is a :func:`general_row` call, for example::

    general_row(
        name="general pair split collar runge_kutta",
        stepper="rk",
        response="kerr",
        features="split temporal collar, content outside the active band",
        why="Under the split collar the pair's ten steps are the reference's ...",
        build=partial(split_collar, "runge_kutta"),
        switch_off=without_collar,
        rungs={"steps": ten_steps("pair")},
    )

``name`` makes the test id; ``stepper``, ``response`` and ``features`` are the axes the
row covers. ``build`` makes the problem: here the pair under the split collar, its
launch loud outside the band. ``rungs`` names what is compared, with its bound: the
field after the ten steps, inside the route's ten-step bound. ``switch_off`` removes
the row's term for the guard; ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends.session_rows import without_collar, without_modal_flow
from general_parity import (
    LENGTH,
    STEPS,
    conditioning,
    launch,
    nonlinear_scale,
    problem_for,
    scaled_case,
    source_bound,
    ten_step_bound,
)
from modal_provider_cases import general_grid, split_guide
from parity_ladder import Built, Row
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.boundaries import absorber
from solprop.exceptions import GeneralModalBasisWarning
from solprop.stepper.multistep import Multistep
from solprop.translate.general import general_modal_route
from solprop.translate.problem import PropagationProblem

QUIET = (
    "ignore::solprop.exceptions.GeneralModalBasisWarning",
    # The launches are noise-free in windows kept tight for speed: the boundary monitor
    # warns, and the window's edge is not what these rows hold.
    "ignore::solprop.exceptions.BoundaryWarning",
    "ignore::solprop.exceptions.SolpropWarning",
)
"""The warning filters of the old modules, which every row's run needs."""

OUT_OF_BAND = 1e-3
"""The split collar's launch content outside the active band, relative to its peak
(#230's)."""

MARCHES = {"runge_kutta": None, "adams": Multistep(variable_order=False)}
"""Both marches on a fixed schedule; a fixed order keeps Adams one discrete march."""

# --- the cases ------------------------------------------------------------------------


@functools.cache
def _scaled(name: str) -> tuple[Any, Any, Any, np.ndarray]:
    """Return a parity case's grid, guide, route and scaled launch, built once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        return scaled_case(name)


def case(name: str, *, linear: bool = False, **options: Any) -> Built:
    """Return a parity case on its ten-step schedule, from its scaled launch.

    `linear` drops the route's responses, so that no source is evaluated at all;
    `options` go to ``general_parity.problem_for`` (a collar, a march).
    """
    grid, guide, route, state = _scaled(name)
    if linear:
        route = dataclasses.replace(route, responses=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        problem, run_request = problem_for(grid, guide, route, state, **options)
    return Built(problem, run_request, grid)


def with_content_out_of_band(problem: PropagationProblem) -> PropagationProblem:
    """Return `problem` launching into every bin outside the active band as well.

    At :data:`OUT_OF_BAND` of the launch's peak: content each backend must drop as it
    loads the launch.
    """
    active = np.asarray(problem.grids.active_mask)
    assert np.count_nonzero(~active) > 0
    launched = np.array(problem.initial.field0, dtype=complex)
    launched[..., ~active] = OUT_OF_BAND * np.max(np.abs(launched))
    return dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=launched)
    )


def split_collar(march: str) -> Built:
    """Return the pair under the split temporal collar, loud outside the active band.

    On `march` (:data:`MARCHES`).
    """
    grid = _scaled("pair")[0]
    built = case(
        "pair",
        absorber=absorber(grid),
        absorber_scheme="split_step",
        integrator=MARCHES[march],
    )
    return dataclasses.replace(built, problem=with_content_out_of_band(built.problem))


def linear_guide() -> Built:
    """Return the split block in its own coordinates, no responding region, ten steps.

    The launch is the two channels' random-phase spectrum at unit scale; no collar and
    the Runge-Kutta march.
    """
    grid = general_grid()
    guide = split_guide(grid, rotated=False)
    route = general_modal_route(
        guide, grid, z_max=LENGTH, omega_reference=float(grid.omega0)
    )
    problem, run_request = problem_for(
        grid, guide, route, launch(grid, 2), absorber=None, integrator=None
    )
    return Built(problem, run_request, grid)


# --- the bounds -----------------------------------------------------------------------


def _source_bound(name: str, built: Built) -> float:
    """Return ``general_parity.source_bound`` at the scaled launch's conditioning."""
    grid, _, _, state = _scaled(name)
    maps = _general_maps(built.problem)
    assert maps is not None
    return source_bound(maps, grid.n_to, conditioning(maps, state, grid))


def one_stage(name: str) -> Callable[[Built], float]:
    """Return a case's one-stage source bound (``general_parity.source_bound``)."""
    return partial(_source_bound, name)


def ten_steps(name: str) -> Callable[[Built], float]:
    """Return a case's ten-step bound at the scaled launch's nonlinear scale.

    ``general_parity.ten_step_bound``: the one-stage bound carried through ten steps,
    with both backends' rounding of each step's linear half.
    """

    def bound(built: Built) -> float:
        grid, _, _, state = _scaled(name)
        maps = _general_maps(built.problem)
        assert maps is not None
        return ten_step_bound(
            nonlinear_scale(maps, state, grid), _source_bound(name, built)
        )

    return bound


def linear_flow(built: Built) -> float:
    """Return the ten-step bound at ``Phi = 0``: both backends' linear halves alone."""
    del built
    return ten_step_bound(0.0, 0.0)


# --- the switch-offs ------------------------------------------------------------------


def without_source(name: str) -> Callable[[PropagationProblem], PropagationProblem]:
    """Return a switch-off rebuilding a case with its route's responses off."""

    def switch(problem: PropagationProblem) -> PropagationProblem:
        rebuilt = case(name, linear=True).problem
        assert rebuilt.shape == problem.shape
        return rebuilt

    return switch


# --- the rows -------------------------------------------------------------------------

CASES = {
    "pair": ("kerr", "the degenerate HE11 parity pair, real cubic"),
    "pair_analytic": ("kerr", "the HE11 parity pair, analytic cubic"),
    "circular": ("kerr", "the HE11 pair in a circular basis"),
    "raman": ("kerr, raman", "a delayed silica tensor"),
    "split": ("kerr", "the split TE01/TM01 block in its own coordinates"),
    "grin": ("kerr", "a graded-index guide of two LP modes on its own grid"),
}
"""Each parity case's response and features, by ``general_parity.CASES`` name."""


def general_row(
    *,
    name: str,
    stepper: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    route: str = "general guide (constant)",
    **fields: Any,
) -> Row:
    """Return a general route row; it marches, so it is medium.

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
        quiet=QUIET,
        window=STEPS,
        **fields,
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the general modal route on a fixed schedule."""
    sources = tuple(
        general_row(
            name=f"general {name} source and ten steps",
            stepper="rk",
            response=CASES[name][0],
            features=f"{CASES[name][1]}, random-phase launch at Phi = 0.5",
            why=f"On the general modal route's {name} case native's stage source at "
            "the launch and its ten fixed steps are the reference's, within the "
            "route's operation count.",
            bound_note="general_parity's bounds, written before any measurement: the "
            "one-stage count times the projected sum's measured conditioning, and its "
            "ten-step carry at the launch's nonlinear scale.",
            build=partial(case, name),
            switch_off=without_source(name),
            rungs={
                "source": one_stage(name),
                "steps": ten_steps(name),
            },
        )
        for name in CASES
    )
    collars = tuple(
        general_row(
            name=f"general pair split collar {march}",
            stepper="rk" if march == "runge_kutta" else "adams",
            response="kerr",
            features="split temporal collar, content outside the active band",
            why="Under the split temporal collar, with a launch holding content "
            "outside the active band, the pair's ten fixed steps are the reference's "
            "on every bin.",
            bound_note="The route's ten-step bound at the quiet launch's nonlinear "
            "scale; the out-of-band content is dropped at the load on both backends "
            "(the residual test holds those bins to exactly zero).",
            build=partial(split_collar, march),
            switch_off=without_collar,
            rungs={"steps": ten_steps("pair")},
        )
        for march in sorted(MARCHES)
    )
    return (
        *sources,
        *collars,
        general_row(
            name="general linear guide",
            stepper="rk",
            response="none",
            features="no responding region, no source pass",
            why="A general guide with no responding region steps its linear flow as "
            "the reference does.",
            bound_note="The ten-step bound at Phi = 0, both backends' linear halves "
            "alone (320 u). The retired test also made the source raise; native "
            "counts a stage's dispatch whether or not it has a source to form, so "
            "that half has no witness here.",
            build=linear_guide,
            switch_off=without_modal_flow,
            rungs={"steps": linear_flow},
        ),
    )

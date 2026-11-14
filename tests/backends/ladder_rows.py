# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The validation ladder's soliton, the Adams march's matched runs and the modal Raman.

**What is compared.** Native against ``numpy_ref`` on fixed schedules:

- the validation ladder's third-order soliton (``docs/dev/backends.md`` section 3.7),
  one step of a millimetre and eight (from ``test_backend_ladder.py``); the rows write
  their readings into the backend equivalence record (:func:`ladder_record`);
- the soliton's eight steps with the propagators tabulated and with them recomputed
  (from ``test_numpy_ref_backend.py``);
- the Adams march's source on the analytic field and on the envelope with its plasma:
  one evaluation a step from the anchor at the launch (``source``), and eight fixed
  steps at the method's top order (``steps``) (from ``test_compiled_multistep.py``,
  which keeps the propagator and the run's source contribution);
- SEP 35's carrier frame on a hydrogen window and on its gradient, replayed on the
  reference's own accepted steps at a fixed order (from ``test_carrier_frame.py``);
- the vector Raman tensor on modal routes: a mode whose polarisation direction depends
  on position, and two envelope plasma populations each with its own Raman kernel,
  their densities constant or falling along z (from ``test_vector_raman_routes.py``):
  one evaluation at the launch (``derivative``) and the two fixed steps.

**The bounds are the old tests', fixed before native runs.** The soliton's ``16 eps``
a step (``bounds.ladder_bound``, a device build's envelope where larger); the
propagator modes the eight-step rung's ``8 x 16 eps``; the Adams source one transform
round trip (``bounds.fft_bound``) and its matched run ``512 eps`` a step relative to the
field, as the carrier frame's windows; the modal Raman ``256 (n_to + n_nodes) 14 eps``
for the transforms, contraction and transverse reductions of fourteen stages.

**Where the old form could not be kept exactly.** The modal Raman was held to
``assert_allclose(rtol=0, atol=b max|ref|)``, which is the peak-normalised comparison
the rows make, now strict.

**The guard.** Each row's term is switched off on the translated problem: the Kerr
term (``optmat_rows.without_kerr``), every response (``planar_rows.without_responses``,
which keeps the temporal collar) or the Raman rows (``optmat_rows.without_raman``).
Each guarded rung's reference must then move by more than ``parity_ladder.DETECTION``
(a thousand) of its bounds.

**How to read a row.** Every row is a :func:`ladder_row` call, for example::

    ladder_row(
        name="ladder soliton eight steps",
        route="gnlse",
        stepper="rk",
        response="kerr",
        features="third-order soliton, eight steps of a millimetre",
        why="The third-order soliton's eight fixed steps are the reference's ...",
        build=partial(soliton, 8e-3, 1e-3),
        switch_off=without_kerr,
        rungs={"steps": soliton_budget(8)},
        record=partial(ladder_record, "eight steps", 2),
    )

``name`` makes the test id; ``route``, ``stepper``, ``response`` and ``features`` are
the axes the row covers. ``build`` makes the problem: here the soliton over eight fixed
steps. ``rungs`` names what is compared, with its bound. ``switch_off`` removes the
row's term for the guard, ``record`` writes the reading into the backend equivalence
record, and ``why`` is the claim a failure quotes.
"""

import functools
from collections.abc import Callable
from functools import partial
from typing import Any

from backends.adams_source_cases import (
    ROUTES,
    SOURCE_AT,
    SOURCE_STEPS,
    matched_run,
    one_evaluation,
)
from backends.carrier_frame_cases import WINDOW, capillary, gradient, matched_window
from backends.conftest import BETA2, GAMMA, GNLSE_SETTINGS, WAVELENGTH, soliton_field
from backends.host_cases_envelope_plasma import geometry_options
from backends.optmat_rows import without_kerr, without_raman
from backends.planar_rows import without_responses
from bounds import EPS, fft_bound, ladder_bound, record_backend
from convergence_records import record
from convergence_studies import BACKEND_EQUIVALENCE
from legacy_geometry import LEGACY_GRID
from parity_ladder import Built, Detection, Reading, Row
from solprop.backends.session import ExecutionOptions
from solprop.grid import EnvelopeGrid
from solprop.modes import ModeKind
from solprop.nonlinear.gnlse import problem_from
from solprop.stepper import FixedSchedule, PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.assembly import assemble
from vector_raman_cases import direction_translation, raman_populations

# --- the soliton ----------------------------------------------------------------------

NARROW_GRID = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)
"""The fibre's reference grid (``narrow_grid``): its carrier fixes the dispersion."""

SOLITON_GRID = EnvelopeGrid.from_limits(
    1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12, **LEGACY_GRID
)
"""The scaffold's envelope grid, 4096 bins (``soliton_grid``)."""

PROPAGATOR_LENGTH = 0.01
"""The propagator modes' run, m: eight steps of 1.25 mm."""


def soliton(length: float, step: float, **overrides: Any) -> Built:
    """Return the third-order soliton over `length` in fixed steps of `step`, m.

    The fixture ``soliton_problem`` of ``backends/conftest.py`` on the scaffold's grid:
    GNLSE dispersion and Kerr, saved at `length`, which is also the largest step;
    `overrides` add settings (a propagator mode).
    """
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=NARROW_GRID.omega0, gamma=GAMMA, shock=False
    )
    problem, run_request = assemble(
        SOLITON_GRID,
        fibre.mode,
        [fibre.kerr],
        soliton_field(SOLITON_GRID, 3),
        save_planes=[length],
        controller=PIController(rtol=1e-10, atol=1e-18),
        dz_max=length,
        dz_init=1e-4,
        schedule=FixedSchedule.uniform(0.0, length, step),
        **GNLSE_SETTINGS,
        **overrides,
    )
    return Built(problem, run_request, SOLITON_GRID)


def propagator_mode(mode: str) -> Built:
    """Return the soliton's eight steps over a centimetre, its propagators `mode`."""
    return soliton(PROPAGATOR_LENGTH, PROPAGATOR_LENGTH / 8, propagator_mode=mode)


def soliton_budget(steps: int) -> Callable[[Built], float]:
    """Return the ladder's ``16 eps`` a step over `steps`, a device's if larger.

    ``bounds.ladder_bound`` with seven transform round trips a step: a step is seven
    stages of a few complex products and two transforms each, 16 eps by Higham on
    ``log2(8192)``.
    """

    def bound(built: Built) -> float:
        return ladder_bound("native", steps * 16 * EPS, built.grid.n_to, steps * 7)

    return bound


def eight_step_rung(built: Built) -> float:
    """Return the eight-step rung's budget, ``8 x 16 eps`` (``REPEAT_BOUND``)."""
    del built
    return 8 * 16 * EPS


def ladder_record(axis: str, order: int, request: Any, readings: list[Reading]) -> None:
    """Write a soliton row's field reading into the backend equivalence record.

    The row of the validation ladder (``docs/dev/accuracy.md`` section 7) its old test
    wrote: ``native against numpy_ref`` (or the device build's name), its `axis` and
    `order`, the deviation with its bound.
    """
    (field,) = readings
    name = record_backend("native")
    record(
        BACKEND_EQUIVALENCE,
        request=request,
        backend=name,
        rows=[
            {
                "series": f"{name} against numpy_ref",
                "axis": axis,
                "deviation": field.measured,
                "bound": float(field.bound),
                "order": float(order),
            }
        ],
    )


# --- the Adams march and the carrier frame --------------------------------------------


def one_transform(built: Built) -> float:
    """Return one evaluation's budget, one transform round trip (``fft_bound``)."""
    return fft_bound(built.problem.grids.n_to)


def matched_steps(built: Built) -> float:
    """Return a matched run's budget, ``512 eps`` a step relative to the field.

    Two backends on one discrete method and history differ by rounding alone (SEP 25).
    """
    schedule = built.run_request.stepper.schedule
    assert schedule is not None
    return len(schedule.steps) * 512 * EPS


def carrier_window() -> Built:
    """Return the hydrogen window under the scalar frame, on its matched schedule.

    A weak collar, 2 uJ, a fixed order (``carrier_frame_cases``).
    """
    method = Multistep(variable_order=False, carrier_frame="scalar")
    simulation = capillary(
        method,
        WINDOW,
        rtol=1e-8,
        energy=2e-6,
        n_save=2,
        interpolate=False,
        collar="weak",
    )
    return matched_window(simulation)


def carrier_gradient() -> Built:
    """Return the hydrogen gradient under the scalar frame, on its matched schedule.

    The reference's march transports the compensation beside the shift.
    """
    method = Multistep(variable_order=False, carrier_frame="scalar")
    return matched_window(gradient(method, 1e-8), compensated=True)


# --- the modal vector Raman -----------------------------------------------------------

DIRECTIONS: dict[ModeKind, int] = {"HE": 2, "TE": 0, "TM": 0}
"""The position-dependent mode beside HE11, by kind, with its azimuthal order."""


def direction(kind: ModeKind) -> Built:
    """Return the two-mode Raman run with a `kind` mode, in chunks of five nodes."""
    problem, run_request, grid = direction_translation(kind, DIRECTIONS[kind])
    return Built(problem, run_request, grid, options=ExecutionOptions(node_chunk=5))


def populations(varying_density: bool) -> Built:
    """Return the two populations with their Raman kernels on the modal geometry."""
    problem, run_request, grid = raman_populations(varying_density)
    return Built(problem, run_request, grid, options=geometry_options("modal"))


def tensor_budget(built: Built) -> float:
    """Return the modal Raman's ``256 (n_to + n_nodes) 14 eps``.

    Temporal transforms, the Cartesian contraction and the transverse reductions per
    stage; fourteen stages cover the two-step run and the one evaluation.
    """
    problem = built.problem
    bound = 256 * (built.grid.n_to + problem.transform.n_nodes) * 14 * EPS
    assert 0 < bound < 1e-6
    return bound


# --- the rows -------------------------------------------------------------------------


def ladder_row(
    *,
    name: str,
    route: str,
    stepper: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    medium: bool = True,
    **fields: Any,
) -> Row:
    """Return a row of this module, medium unless it is one evaluation under a second.

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
        medium=medium,
        **fields,
    )


def _soliton_rows() -> tuple[Row, ...]:
    """Return the validation ladder's soliton rows and the propagator modes'."""
    ladder = tuple(
        ladder_row(
            name=f"ladder soliton {label}",
            route="gnlse",
            stepper="rk",
            response="kerr",
            features=f"third-order soliton, {label} of a millimetre",
            why=f"The third-order soliton's {label} through the whole pipeline are "
            "the reference's.",
            bound_note="16 eps a step by Higham on log2(8192) (the retired numba_cpu "
            "measured 2.4e-16 for one), a device build's envelope where larger.",
            build=partial(soliton, length, 1e-3),
            switch_off=without_kerr,
            rungs={"steps": soliton_budget(steps)},
            record=partial(ladder_record, label, order),
        )
        for label, steps, length, order in (
            ("one step", 1, 1e-3, 1),
            ("eight steps", 8, 8e-3, 2),
        )
    )
    modes = tuple(
        ladder_row(
            name=f"ladder soliton propagators {mode}",
            route="gnlse",
            stepper="rk",
            response="kerr",
            features=f"third-order soliton, propagators {mode}d",
            why=f"Asked to {mode} its propagators, native's one route steps the "
            "soliton as the reference does.",
            bound_note="The eight-step rung's 8 x 16 eps.",
            build=partial(propagator_mode, mode),
            switch_off=without_kerr,
            rungs={"steps": eight_step_rung},
        )
        for mode in ("tabulate", "recompute")
    )
    return (*ladder, *modes)


def _adams_rows() -> tuple[Row, ...]:
    """Return the Adams march's source rows and its carrier-frame windows."""
    sources = tuple(
        row
        for route in sorted(ROUTES)
        for row in (
            ladder_row(
                name=f"ladder adams source {route}",
                route="capillary (envelope)"
                if route == "envelope plasma"
                else "capillary (mode-averaged)",
                stepper="adams",
                response="kerr, plasma" if route == "envelope plasma" else "kerr",
                features=f"{route}, launch into the temporal collar, one evaluation "
                "a millimetre from the anchor",
                why=f"On the {route} column family the Adams march's untransported "
                "source a step from the anchor is the reference's.",
                bound_note="One transform round trip, fft_bound(n_to); the source is "
                "compared before any transport back to the anchor, where U(anchor, "
                "position) reaches 1e26 on the band's edges.",
                build=partial(one_evaluation, route),
                # The envelope plasma's one evaluation takes 1.6 s, the analytic
                # field's 0.1 s (measured).
                medium=route == "envelope plasma",
                switch_off=without_responses,
                rungs={"source": one_transform},
                masked_launch=True,
                source_z=SOURCE_AT,
            ),
            ladder_row(
                name=f"ladder adams matched run {route}",
                route="capillary (envelope)"
                if route == "envelope plasma"
                else "capillary (mode-averaged)",
                stepper="adams",
                response="kerr, plasma" if route == "envelope plasma" else "kerr",
                features=f"{route}, {SOURCE_STEPS} fixed steps at the top order",
                why=f"On the {route} column family the Adams march's eight fixed steps "
                "at its top order are the reference's.",
                bound_note="512 eps a step relative to the field, the matched-window "
                "bound. The source moves the field by 1.1e-5 of it over the run.",
                build=partial(matched_run, route),
                switch_off=without_responses,
                rungs={"steps": matched_steps},
            ),
        )
    )
    windows = tuple(
        ladder_row(
            name=f"ladder carrier frame matched {label}",
            route="capillary (mode-averaged)"
            if label == "window"
            else "capillary (pressure gradient)",
            stepper="adams",
            response="kerr",
            features=f"scalar carrier frame, 40 bar hydrogen {label}, the "
            "reference's accepted steps at a fixed order",
            why=f"Under the scalar carrier frame the hydrogen {label} replayed on "
            "the reference's accepted steps is the reference's.",
            bound_note="512 eps a step relative to the field: one discrete method "
            "and history (SEP 25).",
            build=build,
            switch_off=without_kerr,
            rungs={"steps": matched_steps},
        )
        for label, build in (("window", carrier_window), ("gradient", carrier_gradient))
    )
    return (*sources, *windows)


def _raman_rows() -> tuple[Row, ...]:
    """Return the modal vector Raman rows."""
    directions = tuple(
        ladder_row(
            name=f"ladder modal raman direction {kind}",
            route="capillary (multimode)",
            stepper="rk",
            response="raman",
            features=f"a {kind} mode beside HE11, position-dependent direction, "
            "partial node chunks",
            why=f"Beside HE11, a {kind} mode's Raman tensor acts on the synthesised, "
            "position-dependent polarisation direction as the reference's does.",
            bound_note="256 (n_to + n_nodes) eps a stage over fourteen stages. Held "
            "to assert_allclose(rtol=0, atol=b max|ref|) before: the same comparison.",
            build=partial(direction, kind),
            switch_off=without_raman,
            rungs={
                "derivative": tensor_budget,
                "steps": tensor_budget,
            },
            metric="peak",
        )
        for kind in sorted(DIRECTIONS)
    )
    mixtures = tuple(
        ladder_row(
            name=f"ladder modal raman populations {label}",
            route="capillary (envelope)",
            stepper="rk",
            response="raman, envelope plasma",
            features=f"two populations, a Raman kernel each, densities {label}",
            why="Two envelope plasma populations each carrying its own Raman kernel "
            f"({label} densities) are the reference's on the modal route.",
            bound_note="256 (n_to + n_nodes) eps a stage over fourteen stages. Held "
            "to assert_allclose(rtol=0, atol=b max|ref|) before: the same comparison.",
            build=partial(populations, varying),
            switch_off=without_raman,
            rungs={
                "derivative": tensor_budget,
                "steps": tensor_budget,
            },
            detection={
                "steps": Detection(
                    threshold,
                    f"The Raman kernels' share of the two fixed steps is {share} "
                    "bounds (measured), under a thousand; their share of the launch "
                    "derivative is 1.6e5 bounds.",
                )
            },
            metric="peak",
        )
        for label, varying, threshold, share in (
            ("constant", False, 500.0, 683),
            ("varying", True, 300.0, 410),
        )
    )
    return (*directions, *mixtures)


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of this module."""
    return (*_soliton_rows(), *_adams_rows(), *_raman_rows())

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Cartesian Raman tensor on fixed schedules: its rows of the case catalogue.

**What is compared.** Native against ``numpy_ref``, one evaluation at the launch
(``derivative``) and the field after the fixed schedule (``steps``), or (the complete
runs) the field and the run's statistics at its end:

- ``vector raman tails <geometry> <family>`` rows: hydrogen's vector Raman on a radial
  or Cartesian beam, real field, analytic field or envelope, two fixed steps
  (``vector_raman_cases.spatial_tails``), in partial chunks of three nodes, from
  ``test_vector_raman_routes.py``'s
  ``test_spatial_vector_raman_and_actual_width_tails``;
- ``vector raman populations <geometry> <density>`` rows: two envelope plasma
  populations each with its own Raman kernel on the radial and Cartesian geometries,
  their densities constant or (radial) falling along z
  (``vector_raman_cases.raman_populations``), from the same module's
  ``test_vector_raman_with_two_envelope_plasma_populations``; the Cartesian geometry
  refuses a varying density by name, which that test keeps;
- ``vector raman complete <family> <material> <steps>`` rows: a modal run with a
  delayed oblique weak probe, hydrogen or SF6, one fixed step or eight
  (``vector_raman_cases.vector_translation``), in partial chunks of five nodes, with
  the run's statistics at its end, from ``test_vector_raman.py``'s native instances of
  ``test_vector_complete_run_matches_reference_and_modal_routes``, which keeps the
  modal rows' route against the axis route, the buffer ledger and the first stage:
  that test reads native's first stage from a trial of an eighth of the run, which
  departs from the zero-length trial the ``derivative`` rung reads by up to 5e-14 of
  the peak, where the zero-length trial departs from the reference by at most
  7e-16 of it: that reading has no rung.

The modal routes' position-dependent direction and populations are
``backends/ladder_rows.py``'s ``ladder modal raman ...`` rows.

**The bounds are the old tests', fixed before native runs.** The tails and populations
``256 (n_to + n_nodes) 14 eps`` (``ladder_rows.tensor_budget``): the temporal
transforms, the Cartesian contraction and the transverse reductions of fourteen
stages. A complete run's ``per_rhs`` counts three convolution transform pairs at
``512 eps ceil(log2 2 n_to)`` and eight node reductions: ``2 x 7 x steps`` of it for
the run (``bounds.ladder_bound``'s CPU value,
a device build's envelope where larger), and ``112 per_rhs + 8 gamma(n_to n_nodes)``
relative for each statistic, elementwise (the old ``assert_allclose`` with ``rtol``
alone). The fields were held to ``assert_allclose(rtol=0, atol=b max|ref|)``: the
peak-normalised comparison the rows make, now strict.

**The guard.** A row's term is the Raman response (``optmat_rows.without_raman`` zeroes
every Raman leaf's rows): each guarded rung's reference must move by more than
``parity_ladder.DETECTION`` (a thousand) of its bound, or a rung's own measured
:class:`parity_ladder.Detection` (:func:`steps_guard`). Two fixed steps of 5 um
accumulate little Raman: the tails' and populations' ``steps`` share is 14 to 142
bounds, unguarded below a hundred, while the launch's evaluation carries their guard;
hydrogen's eight-step complete runs clear 767 and 922.

**How to read a row.** Every row is a :func:`raman_row` call, for example::

    raman_row(
        name="vector raman tails radial envelope",
        route="radial",
        response="kerr, raman",
        features="hydrogen's vector Raman on an elliptically polarised beam ...",
        why="On a radial beam, hydrogen's Raman tensor ...",
        build=partial(tails, "radial", "envelope"),
        rungs={"derivative": tensor_budget, "steps": tensor_budget},
    )

``name`` makes the test id; ``route``, ``response`` and ``features`` are the axes the
row covers. ``build`` makes the problem; ``rungs`` names what is compared, with its
bound; ``why`` is the claim a failure quotes. Every row is switched off by its Raman
response, compared peak-normalised and medium, as its old test was.
"""

import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any

from backends.host_cases_envelope_plasma import geometry_options
from backends.ladder_rows import tensor_budget
from backends.optmat_rows import without_raman
from bounds import EPS, gamma, ladder_bound
from parity_ladder import LEAST_DETECTION, Built, Detection, Row, Statistic
from solprop.backends.session import ExecutionOptions
from vector_raman_cases import (
    CHUNK,
    raman_populations,
    spatial_tails,
    vector_translation,
)

FAMILIES = {
    "field_real": "a real field, the third harmonic kept",
    "field_analytic": "an analytic field",
    "envelope": "an envelope",
}
"""Each field family, as a row's features name it."""

# --- the cases ------------------------------------------------------------------------


def tails(geometry: str, family: str) -> Built:
    """Return the spatial tails' run, in partial chunks of three nodes."""
    problem, run_request, grid = spatial_tails(geometry, family)
    return Built(problem, run_request, grid, options=ExecutionOptions(node_chunk=3))


def populations(geometry: str, varying_density: bool) -> Built:
    """Return the two Raman populations on a spatial geometry, in partial chunks."""
    problem, run_request, grid = raman_populations(varying_density, geometry)
    return Built(problem, run_request, grid, options=geometry_options(geometry))


def complete(family: str, material: str, steps: int) -> Built:
    """Return the modal vector run with its probe, in chunks of five nodes."""
    problem, run_request, grid, modes = vector_translation(
        family, material, steps=steps
    )
    return Built(problem, run_request, grid, options=CHUNK, modes=tuple(modes))


def partial_chunks_of_three(built: Built) -> bool:
    """Return whether the transform's nodes leave a partial chunk of three."""
    return built.problem.transform.n_nodes % 3 != 0


# --- the bounds -----------------------------------------------------------------------


def per_rhs(built: Built) -> float:
    """Return one evaluation's count: three convolution pairs and the reductions."""
    n_to = built.grid.n_to
    return 3 * 512 * EPS * math.ceil(math.log2(2 * n_to)) + 8 * gamma(
        built.problem.transform.n_nodes
    )


def evaluations(count: Callable[[Built], int]) -> Callable[[Built], float]:
    """Return a complete run's bound over `count` evaluations, both runs' errors.

    ``bounds.ladder_bound`` on native: ``2 count per_rhs``, or a device build's
    ``3 count`` transform round trips where larger.
    """

    def bound(built: Built) -> float:
        times = count(built)
        found = ladder_bound(
            "native", 2 * times * per_rhs(built), built.grid.n_to, 3 * times
        )
        assert found > 0
        return found

    return bound


def the_run_s_evaluations(built: Built) -> int:
    """Return seven stages a step over the schedule's steps."""
    schedule = built.run_request.stepper.schedule
    assert schedule is not None
    return 7 * len(schedule.steps)


def a_statistic_s(built: Built, name: str, bound: float) -> Statistic:
    """Return a statistic's rule: ``112 per_rhs + 8 gamma(n_to n_nodes)``, elementwise.

    A statistic adds a temporal reduction and a squared modulus to the run's per-step
    count, held relative at every element (the old ``assert_allclose`` with ``rtol``
    alone).
    """
    del name, bound
    return Statistic(
        112 * per_rhs(built)
        + 8 * gamma(built.grid.n_to * built.problem.transform.n_nodes),
        elementwise=True,
    )


# --- the guards' measured shares -----------------------------------------------------

TAILS_SHARE = {
    ("radial", "field_real"): 22.9,
    ("radial", "field_analytic"): 23.0,
    ("radial", "envelope"): 142.0,
    ("free", "field_real"): 22.6,
    ("free", "field_analytic"): 22.7,
    ("free", "envelope"): 129.0,
}
"""The Raman response's share of the tails' two fixed steps, in bounds (measured)."""

POPULATIONS_SHARE = {
    ("radial", "constant"): 23.1,
    ("radial", "varying"): 13.9,
    ("free", "constant"): 20.8,
}
"""The Raman kernels' share of the populations' two fixed steps, in bounds
(measured)."""

COMPLETE_SHARE = {
    ("field_real", "H2", 8): 767.0,
    ("field_analytic", "H2", 8): 767.0,
    ("envelope", "H2", 8): 922.0,
}
"""The Raman response's share of a complete run below a thousand bounds (measured);
every other complete run's clears it."""


def steps_guard(share: float | None, what: str) -> dict[str, Any]:
    """Return a row's ``steps`` guard below a thousand bounds, from its measured share.

    Lowered to the share's hundreds where it clears ``parity_ladder.LEAST_DETECTION``,
    unguarded below that (the launch's evaluation carries the row's
    guard); nothing where the share clears a thousand (`share` is ``None``).
    """
    if share is None:
        return {}
    reason = (
        f"{what} moves the reference's fixed steps by {share:g} bounds (measured): "
        "too short a run for the response to accumulate against the bound, while its "
        "share of the launch's evaluation clears a thousand (#806)."
    )
    if share < LEAST_DETECTION:
        return {"unguarded": {"steps": reason}}
    return {
        "detection": {"steps": Detection(100.0 * math.floor(share / 100.0), reason)}
    }


# --- the rows -------------------------------------------------------------------------


def raman_row(
    *,
    name: str,
    route: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a vector Raman row: Raman off for the guard, peaks compared, medium.

    Its id is `name`; `fields` are the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=name,
        axes={
            "route": route,
            "stepper": "rk",
            "response": response,
            "features": features,
        },
        build=build,
        switch_off=without_raman,
        metric="peak",
        medium=True,
        **fields,
    )


def _tails_rows() -> tuple[Row, ...]:
    """Return hydrogen's vector Raman on the radial and Cartesian beams."""
    return tuple(
        raman_row(
            name=f"vector raman tails {geometry} {family}",
            route=geometry,
            response="kerr, raman",
            features=f"hydrogen's vector Raman on an elliptically polarised beam, "
            f"{FAMILIES[family]}, two fixed steps, partial chunks of three nodes",
            why=f"On a {geometry} beam, hydrogen's Raman tensor acting on an "
            "elliptically polarised field, with the convolution's actual-width tails, "
            "is the reference's at the launch and over two fixed steps.",
            bound_note="256 (n_to + n_nodes) eps a stage over fourteen stages, "
            "written before the comparison.",
            build=partial(tails, geometry, family),
            rungs={
                "derivative": tensor_budget,
                "steps": tensor_budget,
            },
            expect={"": partial_chunks_of_three},
            **steps_guard(TAILS_SHARE[geometry, family], "The Raman response"),
        )
        for geometry in ("radial", "free")
        for family in FAMILIES
    )


def _population_rows() -> tuple[Row, ...]:
    """Return the two Raman populations on the spatial geometries."""
    return tuple(
        raman_row(
            name=f"vector raman populations {geometry} {label}",
            route=geometry,
            response="raman, envelope plasma",
            features=f"two populations, a Raman kernel each, densities {label}",
            why="Two envelope plasma populations each carrying its own Raman kernel "
            f"({label} densities) are the reference's on the {geometry} geometry.",
            bound_note="256 (n_to + n_nodes) eps a stage over fourteen stages, "
            "written before the comparison.",
            build=partial(populations, geometry, varying),
            rungs={
                "derivative": tensor_budget,
                "steps": tensor_budget,
            },
            **steps_guard(POPULATIONS_SHARE[geometry, label], "The Raman kernels"),
        )
        for geometry, label, varying in (
            ("radial", "constant", False),
            ("radial", "varying", True),
            ("free", "constant", False),
        )
    )


def _complete_rows() -> tuple[Row, ...]:
    """Return the modal vector runs with their probe, every family and material."""
    return tuple(
        raman_row(
            name=f"vector raman complete {family} {material} {steps}",
            route="capillary (multimode)",
            response="kerr, raman",
            features=f"{material}'s vector Raman, a delayed oblique weak probe, "
            f"{FAMILIES[family]}, {steps} fixed step{'s' if steps > 1 else ''}, "
            "partial chunks of five nodes",
            why=f"{material}'s Raman tensor on an elliptically polarised launch with a "
            "delayed oblique probe is the reference's after the run and in the run's "
            "statistics.",
            bound_note="Per evaluation three convolution transform pairs at 512 eps "
            "log2(2 n_to) and eight node reductions, 2 x 7 x steps of it for the run; "
            "each statistic 112 of it and a reduction over the nodes, relative at "
            "every element. Written before the comparison.",
            build=partial(complete, family, material, steps),
            rungs={"steps": evaluations(the_run_s_evaluations)},
            every_statistic=a_statistic_s,
            **steps_guard(
                COMPLETE_SHARE.get((family, material, steps)), "The Raman response"
            ),
        )
        for steps in (1, 8)
        for material in ("H2", "SF6")
        for family in FAMILIES
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every vector Raman row on the spatial geometries and the modal run."""
    return (*_tails_rows(), *_population_rows(), *_complete_rows())

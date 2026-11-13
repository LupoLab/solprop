# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The on-axis plasma statistics at the launch: their rows of the case catalogue.

**What is compared.** Native against ``numpy_ref`` on an ionising argon capillary at its
entrance (``tests/axis_plasma_cases.py``), one row per kind of on-axis row and scan: a
field grid's one-component scan of two rows (``rotated-x``, their magnitude), an
envelope's one-component scan of two rows (``envelope-rotated-x``, their ellipse), the
band rule's filtered exposure (``band``), and an envelope's cycle-averaged scan of one
and two modes (``envelope``, ``envelope-two-modes``). Each row reads the on-axis plasma
statistics the session observes at the launch, nothing stepped (``launch``), from
``tests/sessions/test_native_session_axis_plasma.py``'s
``test_the_plan_s_axis_row_is_the_host_scan_s_and_the_reference_s``. That test read
native's statistics plan evaluated on the loaded launch and the reference's host scan of
it; the session's launch observation is those evaluations, to the bit.

**The bound is the old test's, fixed before native runs.** The node scans' parity
tolerance, ``1e-10`` of each statistic's largest value (``PLASMA_RTOL``): a compiled
plasma column against the reference's multipass scan over the window, the tolerance
the backends' on-axis statistics are held to (``test_on_axis_statistics.py``). The rung
reads every statistic at once, and each is also compared on its own, peak-normalised
and positive somewhere, as the old test held them one by one.

**The guard.** A row's term is the ionisation: ``geometry_rows.without_ionisation``
scales every channel's rate by ``exp(-1000)``, a field channel's or an envelope
channel's, so the axis's density and rate vanish and the reference's statistics must
move by more than ``parity_ladder.DETECTION`` (a thousand) of the bound.

**How to read a row.** Every row is an :func:`axis_row` call, for example::

    axis_row(
        kind="band",
        route="capillary (multimode)",
        features="the band rule's filtered exposure, one mode",
        why="With the band rule's filtered exposure the axis's plasma statistics ...",
    )

``kind`` names the case (``axis_plasma_cases.ENTRANCE_KINDS``) and makes the test id;
``route`` and ``features`` are the axes the row covers; ``why`` is the claim a failure
quotes. The rung, bound, statistics and guard are the same on every row.
"""

import functools
from functools import partial
from typing import Any

import numpy as np

from axis_plasma_cases import entrance_problem
from backends.geometry_rows import without_ionisation
from parity_ladder import Built, Row, Statistic
from solprop.stats import ON_AXIS_PLASMA_STATISTICS, StatsSpec

PLASMA_RTOL = 1e-10
"""The node scans' parity tolerance, a compiled plasma column against the reference's
multipass scan over the window, which the backends' on-axis statistics are held to
(``test_on_axis_statistics.py``): the native row's scan against the host's."""

QUIET = ("ignore::solprop.exceptions.MaterialRangeWarning",)
"""The band reaches past argon's fitted range, which is not what these rows hold."""

# --- the cases ------------------------------------------------------------------------


def entrance(kind: str) -> Built:
    """Return a kind's capillary at its entrance, observing the on-axis statistics."""
    grid, modes, problem, request = entrance_problem(kind)
    run_request = request.for_run(stats=StatsSpec(names=ON_AXIS_PLASMA_STATISTICS))
    return Built(problem, run_request, grid, modes=tuple(modes))


def has_a_y_field_on_the_axis(built: Built) -> bool:
    """Return whether the modes carry a y field on the axis, on one resolved component.

    The ``rotated-x`` cases' premise, which the old test asserted.
    """
    # Optional wheel: absent from the base environment.
    from solprop_native._statistics import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _on_axis,
    )

    assert built.modes is not None
    return built.problem.transform.n_pol == 1 and bool(
        np.any(_on_axis(list(built.modes))[1])
    )


# --- the bounds -----------------------------------------------------------------------


def plasma_rtol(built: Built) -> float:
    """Return :data:`PLASMA_RTOL`, the node scans' parity tolerance."""
    del built
    return PLASMA_RTOL


def within_the_scans_tolerance(built: Built, name: str, bound: float) -> Statistic:
    """Return an on-axis statistic's rule: peak-normalised at the bound, positive.

    The old test's ``max |ours - theirs| <= 1e-10 max |theirs|``, the reference's value
    positive somewhere.
    """
    del built
    assert name in ON_AXIS_PLASMA_STATISTICS
    return Statistic(bound, positive=True)


# --- the rows -------------------------------------------------------------------------


def axis_row(*, kind: str, route: str, features: str, why: str, **fields: Any) -> Row:
    """Return a kind's launch row; one evaluation, so fast, as its old test was.

    `fields` are the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=f"axis plasma launch {kind}",
        axes={
            "route": route,
            "stepper": "none",
            "response": "kerr, plasma",
            "features": features,
        },
        why=why,
        bound_note="The node scans' parity tolerance, 1e-10 of each statistic's "
        "largest value, written before the comparison; the old test's.",
        build=partial(entrance, kind),
        switch_off=without_ionisation,
        rungs={"launch": plasma_rtol},
        metric="peak",
        every_statistic=within_the_scans_tolerance,
        expect={"problem.initial.noise0": None, **fields.pop("expect", {})},
        quiet=QUIET,
        **fields,
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return the on-axis plasma statistics at the launch, one row per kind."""
    return (
        axis_row(
            kind="rotated-x",
            route="capillary (multimode)",
            features="two rotated modes with a y field on the axis, a field grid "
            "resolving x alone: the one-component scan of both rows' magnitude",
            why="Where the modes carry a y field on the axis and the field grid "
            "resolves x alone, the axis's plasma statistics read both rows' magnitude "
            "as the reference's host scan does.",
            expect={"": has_a_y_field_on_the_axis},
        ),
        axis_row(
            kind="envelope-rotated-x",
            route="capillary (envelope)",
            features="two rotated modes with a y field on the axis, an envelope "
            "resolving x alone: the one-component scan of both rows' ellipse",
            why="Where the modes carry a y field on the axis and the envelope resolves "
            "x alone, the axis's plasma statistics read both rows' polarisation "
            "ellipse as the reference's host scan does.",
            expect={"": has_a_y_field_on_the_axis},
        ),
        axis_row(
            kind="band",
            route="capillary (multimode)",
            features="the band rule's filtered exposure, one mode",
            why="With the band rule's filtered exposure the axis's plasma statistics "
            "are the reference's host scan's.",
        ),
        axis_row(
            kind="envelope",
            route="capillary (envelope)",
            features="an envelope's cycle-averaged scan, one mode",
            why="An envelope's cycle-averaged scan of one mode gives the reference's "
            "axis plasma statistics.",
        ),
        axis_row(
            kind="envelope-two-modes",
            route="capillary (envelope)",
            features="an envelope's cycle-averaged scan, two modes",
            why="An envelope's cycle-averaged scan of two modes gives the reference's "
            "axis plasma statistics.",
        ),
    )

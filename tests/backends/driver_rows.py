# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs through the driver on a fixed schedule: their rows of the case catalogue.

**What is compared.** Native against ``numpy_ref`` on whole runs through the driver and
the output layer, read back as a test reads them:

- ``slab ...`` rows: the hollow planar (slab) guide's projected multimode launch,
  TE1/TE3/TE5 and the tapered TE1/TE3, ten fixed steps of 10 cm of argon
  (``tests/slab_cases.py``); the last saved plane (``last_plane``), from
  ``tests/driver/test_prop_slab.py``, which keeps each backend's launched energy and
  the warnings the route gives;
- ``legacy entry ...`` rows: the legacy entry points ``prop_capillary`` and
  ``prop_gnlse``, ten fixed steps at the Adams method's fixed top order
  (``tests/legacy_entry_cases.py``); the last saved plane (``last_plane``), from the
  native instance of ``tests/driver/test_coherent_capillary.py``'s
  ``test_the_legacy_entry_points_are_what_they_were``, which keeps each run's record and
  native's repeat of its own run.

**The bounds are the old tests', fixed before native runs.** A slab row is bounded by
``slab_cases.parity_bound``: the ten steps' seven DOPRI5 stages, each one transform
round trip each way and the modal synthesis and projection over the rule's nodes, the
errors adding. A legacy row is bounded by ``legacy_entry_cases.legacy_parity_bound``,
the same count over the march's evaluations a step (DOPRI5's seven, or the Adams
march's weights' absolute sum) and the plane's samples. Both are relative to the
plane's peak, as the old tests compared them.

**The guard.** A slab or legacy row's term is the Kerr response
(``optmat_rows.without_kerr``): the reference's last plane must move by more than
``parity_ladder.DETECTION`` (a thousand) of the row's bound without it.

**How to read a row.** Every row is a :func:`driver_row` call, for example::

    driver_row(
        name="slab projected multimode constant",
        route="slab guide (multimode envelope)",
        stepper="rk",
        response="kerr",
        features="TE1/TE3/TE5 projected from a 0.735a Gaussian, an inert collar",
        why="Ten fixed steps of the slab's projected multimode launch leave ...",
        build=partial(slab, False),
        switch_off=without_kerr,
        rungs={"last_plane": slab_bound},
        metric="peak",
    )

``name`` makes the test id; ``route``, ``stepper``, ``response`` and ``features`` are
the axes the row covers. ``build`` makes the problem: here the constant gap's
projected launch, run on the reference by its simulation. ``rungs`` names what is
compared, with its bound: the last saved plane, relative to its peak. ``switch_off``
removes the row's term for the guard; ``why`` is the claim a failure quotes.
"""

import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import legacy_entry_cases
import slab_cases
from backends.flow_rows import driver_case
from backends.optmat_rows import without_kerr
from parity_ladder import Built, Row, reference

SLAB_QUIET = (
    # The FFT padding reaches beyond the argon fit; the notice is not the subject.
    "ignore::solprop.exceptions.MaterialRangeWarning",
    # The modes' group-delay walk-off and a taper's omitted connection, which the
    # residual test asserts by name.
    "ignore:.*walking off:solprop.exceptions.SolpropWarning",
    "ignore::solprop.exceptions.AdiabaticTaperWarning",
)
"""The slab module's warning filters, and the two warnings its parity test caught."""

LEGACY_QUIET = (
    # The band reaches past helium's and silica's fitted ranges, and the bore is a
    # marginal case by design: test_coherent_capillary's module filters.
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore::solprop.exceptions.GrazingIncidenceWarning",
    "ignore:the temporal absorber attenuates",
)
"""The legacy entry points' module's warning filters."""

# --- the cases ------------------------------------------------------------------------


def slab(taper: bool) -> Built:
    """Return the slab's projected multimode run on the reference, ten fixed steps."""
    return driver_case(slab_cases.launched(*slab_cases.projected_multimode(taper)))


def legacy_entry(entry: str) -> Built:
    """Return a legacy entry point's ten-step run on the reference.

    The simulation ``prop_capillary`` or ``prop_gnlse`` builds and runs
    (``legacy_entry_cases.legacy_simulation``).
    """
    return driver_case(legacy_entry_cases.legacy_simulation(entry))


# --- the bounds -----------------------------------------------------------------------


def slab_bound(built: Built) -> float:
    """Return ``slab_cases.parity_bound`` over the run's own quadrature rule's nodes."""
    assert built.simulation is not None
    rule = built.simulation.settings.quadrature
    assert rule is not None
    return slab_cases.parity_bound(rule.n_nodes)


def legacy_bound(built: Built) -> float:
    """Return ``legacy_entry_cases.legacy_parity_bound`` over the plane's samples.

    The samples are the reference's last plane's, as the old test read them off the
    native plane of the same shape.
    """
    assert built.simulation is not None
    samples = reference(built, "last_plane").field.shape[-1]
    return legacy_entry_cases.legacy_parity_bound(samples, built.simulation)


# --- the rows -------------------------------------------------------------------------


def driver_row(
    *,
    name: str,
    route: str,
    stepper: str,
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a driver run's row; it marches, so it is medium unless it says slow.

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
        medium=not fields.get("slow"),
        **fields,
    )


def _slab_rows() -> tuple[Row, ...]:
    """Return the slab's projected multimode runs, the constant gap and the taper."""
    return tuple(
        driver_row(
            name=f"slab projected multimode {label}",
            route="slab guide (multimode envelope)",
            stepper="rk",
            response="kerr",
            features=f"{orders} projected from a 0.735a Gaussian, ten fixed steps of "
            "10 cm of argon, an inert collar",
            why=f"Ten fixed steps of the slab's projected {orders} launch leave the "
            "reference's last plane on native.",
            bound_note="slab_cases.parity_bound over the run's slab rule's nodes, "
            "relative to the plane's peak, written before the runs were compared "
            "(measured below 1e-15 of a 1.6e-10 bound).",
            build=partial(slab, taper),
            switch_off=without_kerr,
            rungs={"last_plane": slab_bound},
            metric="peak",
            quiet=SLAB_QUIET,
        )
        for label, taper, orders in (
            ("constant", False, "TE1/TE3/TE5"),
            ("tapered", True, "tapered TE1/TE3"),
        )
    )


LEGACY_VACUOUS = {
    "prop_capillary": "The Kerr term moves the reference's last plane by 12.1 bounds "
    "(measured), under the least guard of 100: 10 nJ in helium over 0.2 mm is nearly "
    "linear, so the bound, which counts the nonlinear term's re-evaluations, is far "
    "above what the term contributes (#806). Kept as the old test's comparison.",
    "prop_gnlse": "",
}
"""Each legacy entry point's reason its row is vacuous (none: guarded)."""


def _legacy_rows() -> tuple[Row, ...]:
    """Return the legacy entry points' ten-step runs."""
    return tuple(
        driver_row(
            name=f"legacy entry {entry}",
            route=route,
            stepper="adams",
            response="kerr",
            features=f"{entry}, ten fixed steps at the Adams method's fixed top order, "
            "no wall",
            why=f"Ten fixed steps of the legacy {entry} run leave the reference's last "
            "plane on native.",
            bound_note="legacy_entry_cases.legacy_parity_bound over the plane's "
            "samples, relative to the plane's peak, derived before the comparison was "
            "first run.",
            build=partial(legacy_entry, entry),
            switch_off=without_kerr,
            rungs={"last_plane": legacy_bound},
            metric="peak",
            vacuous=LEGACY_VACUOUS[entry],
            quiet=LEGACY_QUIET,
        )
        for entry, route in (
            ("prop_capillary", "capillary (multimode)"),
            ("prop_gnlse", "gnlse"),
        )
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the driver's runs on a fixed schedule."""
    return (*_slab_rows(), *_legacy_rows())

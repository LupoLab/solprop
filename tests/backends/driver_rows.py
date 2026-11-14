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
  native's repeat of its own run;
- ``rk replay ...`` rows: the GNLSE and coherent-wall capillary routes of
  ``capability_routes.ROUTES`` replaying 400 uniform steps on the Runge-Kutta march
  (``tests/rk_route_cases.py``); every saved plane (``run``), from
  ``tests/driver/test_native_rk_march.py``'s ``test_a_replay``, which keeps native's
  repeat of its own run, its march and its steps.

**The bounds are the old tests', fixed before native runs.** A slab row is bounded by
``slab_cases.parity_bound``: the ten steps' seven DOPRI5 stages, each one transform
round trip each way and the modal synthesis and projection over the rule's nodes, the
errors adding. A legacy row is bounded by ``legacy_entry_cases.legacy_parity_bound``,
the same count over the march's evaluations a step (DOPRI5's seven, or the Adams
march's weights' absolute sum) and the plane's samples. Both are relative to the
plane's peak, as the old tests compared them. A replay row holds every saved plane to a
hundred times the controller's relative tolerance of their common peak, the old test's
gate for the adaptive runs it shares its helper with, which no derivation backs for a
replay (#807).

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
import rk_route_cases
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


def replay(route: str) -> Built:
    """Return a route replaying 400 uniform steps on the Runge-Kutta march."""
    return driver_case(rk_route_cases.replayed(route, 400))


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


def a_hundred_tolerances(built: Built) -> float:
    """Return a hundred times the request's controller's relative tolerance."""
    controller = built.run_request.stepper.controller
    assert controller is not None
    return 100.0 * float(controller.rtol)


# --- the rows -------------------------------------------------------------------------


def driver_row(
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
    """Return a driver run's row; it marches, so it is medium unless it says otherwise.

    A row keeps its old test's tier: ``medium=False`` for one whose test was fast.
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


REPLAY_QUIET = (
    # The test's own filter: the window's edge is not what these rows hold.
    "ignore::solprop.exceptions.BoundaryWarning",
)
"""The replays' warning filter; a host march's warning stays an error, as it was."""


REPLAY_VACUOUS = {
    "gnlse": "The Kerr term moves the reference's planes by 10.8 bounds (measured), "
    "under the least guard of 100: the gate is a hundred controller tolerances, 1e-5 "
    "of the peak, against a nonlinear phase of 2e-4 rad over the route's 0.2 mm "
    "(#806, #807). Kept as the old test's comparison.",
    "capillary (coherent wall)": "The Kerr term moves the reference's planes by "
    "0.069 bounds and the coherent wall's flow by 0.99 (measured), under the least "
    "guard of 100: the gate is a hundred controller tolerances, 1e-4 of the peak "
    "(#806, #807). Kept as the old test's comparison.",
}
"""Each replayed route's reason its row is vacuous: no term clears the gate's guard."""


REPLAY_MEDIUM = {"gnlse": False, "capillary (coherent wall)": True}
"""Each replay's tier, its old test's: the GNLSE replay was fast, the wall's medium
(#792 re-tiers on CI times)."""


def _replay_rows() -> tuple[Row, ...]:
    """Return the GNLSE and coherent-wall routes replayed on the Runge-Kutta march."""
    return tuple(
        driver_row(
            name=f"rk replay {route}",
            route=route,
            stepper="rk",
            response="kerr",
            features="400 uniform fixed steps landing on the route's save planes",
            why=f"The {route} route's 400 replayed Runge-Kutta steps leave the "
            "reference's saved planes on native.",
            bound_note="A hundred times the controller's relative tolerance of the "
            "planes' common peak, the old test's gate for its adaptive runs, which no "
            "derivation backs for a replay (#807).",
            build=partial(replay, route),
            switch_off=without_kerr,
            rungs={"run": a_hundred_tolerances},
            metric="peak",
            vacuous=REPLAY_VACUOUS[route],
            medium=REPLAY_MEDIUM[route],
            quiet=REPLAY_QUIET,
        )
        for route in REPLAY_VACUOUS
    )


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the driver's runs on a fixed schedule."""
    return (*_slab_rows(), *_legacy_rows(), *_replay_rows())

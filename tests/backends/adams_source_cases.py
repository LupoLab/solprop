# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march's source on the stage pipeline's two other column families.

Shared by ``test_compiled_multistep.py`` and the catalogue's march rows
(``backends/ladder_rows.py``): a helium capillary on the analytic field (no harmonic)
and on the envelope with its plasma, its launch reaching into the temporal collar; one
evaluation of the source a step from the anchor (:func:`one_evaluation`), and a matched
run of fixed steps at the method's top order (:func:`matched_run`).
"""

import dataclasses

import numpy as np

from backends.matched import fixed_order
from parity_ladder import Built
from solprop.driver.simulation import Simulation
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule

ROUTES = {
    "analytic field": {"thg": False, "plasma": False},
    "envelope plasma": {"envelope": True, "plasma": True},
}
"""The stage pipeline's two other column families, for one evaluation of the source."""


SOURCE_STEPS = 8
"""The fixed steps of :func:`matched_run`."""

SOURCE_LENGTH = SOURCE_STEPS * 5e-6
"""Its length, m: steps inside the in-stage collar's stability limit of 7e-6 m."""

SOURCE_AT = 1e-3
"""Where the one evaluation is taken, m, a step from the anchor at the launch."""


def source_case(route, length, **settings):
    """Return `route`'s helium capillary of `length` and its edged launch.

    The window's own grid, whose collar is the original evaluation's. The launch is the
    pulse plus half of it rolled two fifths of the window along, which reaches into the
    temporal collar.
    """
    limits = (0.01, 800e-9, (200e-9, 2e-6), 100e-15)
    grid = (
        EnvelopeGrid.from_limits(*limits, thg=False)
        if route.get("envelope")
        else FieldGrid.from_limits(*limits)
    )
    simulation = Simulation.from_capillary(
        150e-6,
        length,
        "He",
        1.0,
        wavelength=800e-9,
        fwhm=10e-15,
        energy=5e-5,
        grid=grid,
        n_save=2,
        noise=None,
        integrator=Multistep(formulation="interaction"),
        **settings,
        **route,
    )
    problem = simulation.translate()
    pulse = np.fft.ifft(problem.initial.field0, axis=-1)
    edge = np.roll(pulse, pulse.shape[-1] * 2 // 5, axis=-1)
    return simulation, problem, np.fft.fft(pulse + 0.5 * edge, axis=-1)


def _launched(simulation: Simulation, problem, lab, run_request) -> Built:
    """Return the build of `problem` launching `lab`, the simulation's own context."""
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=lab)
    )
    return Built(
        problem,
        run_request,
        simulation.grid,
        modes=simulation.modes,
        simulation=simulation,
    )


def one_evaluation(route: str) -> Built:
    """Return `route`'s centimetre capillary launching its edged pulse.

    The source is evaluated once at :data:`SOURCE_AT` (the row's ``source_z``), on the
    launch hard-masked to the active band, as the native state loads it.
    """
    simulation, problem, lab = source_case(ROUTES[route], 0.01)
    return _launched(simulation, problem, lab, simulation._run_request())


def matched_run(route: str) -> Built:
    """Return `route`'s capillary over :data:`SOURCE_STEPS` fixed steps of 5 um.

    At the Adams method's top order (``matched.fixed_order``), so that the two backends
    run one discrete method and history.
    """
    simulation, problem, lab = source_case(
        ROUTES[route],
        SOURCE_LENGTH,
        schedule=FixedSchedule.uniform(0.0, SOURCE_LENGTH, 5e-6),
    )
    return _launched(simulation, problem, lab, fixed_order(simulation._run_request()))

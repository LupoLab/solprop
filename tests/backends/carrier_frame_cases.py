# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 35's hydrogen capillaries under the carrier frame, and a matched window.

Shared by ``test_carrier_frame.py`` and the catalogue's march rows
(``backends/ladder_rows.py``): the dense-gas capillary and its pressure gradient, and
:func:`matched_window`, a window replayed on the reference's own accepted steps.
"""

import dataclasses
import functools
import warnings

import numpy as np

from parity_ladder import Built
from sessions.runs import run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import SolpropWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.stepper.multistep import Multistep

THG = {"field": None, "envelope": False, "envelope_thg": True}
"""The third-harmonic choice per grid: the field grid's default (on), off, on."""


@functools.cache
def capillary(
    integrator: Multistep | RungeKutta,
    length: float = 2e-3,
    *,
    n_save: int = 5,
    interpolate: bool = True,
    rtol: float = 1e-8,
    grid_kind: str = "field",
    energy: float = 1e-7,
    modes: int = 1,
    frame_offset: float = 0.0,
    collar: str = "spectral",
) -> Simulation:
    """Return a 16 um, 40 bar hydrogen capillary on a 200 fs grid: kappa near -5.5e3/m.

    The dense gas of SEP 35's cases C and D, reduced: Kerr only (no Raman, no plasma),
    so that a short window runs in seconds. `frame_offset` moves a fixed custom frame's
    reference frequency by that fraction of the grid's. `collar` is ``"spectral"`` (no
    temporal collar: a manufactured source replaces the whole source, the collar's
    term with it), ``"weak"`` (a 1 m reference length, so that accuracy sets a short
    window's step) or ``"default"`` (the window's own in-stage collar, which then binds
    the step: 1.4 um at PEC5/6 on 3 mm).
    """
    limits = (200e-9, 2e-6)
    grid = (
        FieldGrid.from_limits(length, 800e-9, limits, 200e-15)
        if grid_kind == "field"
        else EnvelopeGrid.from_limits(
            length, 800e-9, limits, 200e-15, thg=grid_kind == "envelope_thg"
        )
    )
    collars = {
        "spectral": absorber(grid, temporal=False),
        "weak": absorber(grid, boundary_length=1.0),
        "default": None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        simulation = Simulation.from_capillary(
            16e-6,
            length,
            "H2",
            40.0,
            grid=grid,
            wavelength=800e-9,
            fwhm=30e-15,
            energy=energy,
            modes=modes,
            rtol=rtol,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=n_save,
            integrator=integrator,
            raman=False,
            plasma=False,
            thg=THG[grid_kind],
            interpolate_planes=interpolate,
            absorber=collars[collar],
        )
    if frame_offset:
        simulation.frame = Frame(
            grid.omega0 * (1.0 + frame_offset), subtract_beta0=False, beta1=None
        )
    return simulation


WINDOW = 3e-3
"""A short physical window, m: a strong 2 uJ pulse broadening in 40 bar of hydrogen."""


@functools.cache
def gradient(integrator, rtol: float) -> Simulation:
    """Return a 3 mm hydrogen gradient, 40 to 20 bar, weak collar: varying operator."""
    grid = FieldGrid.from_limits(WINDOW, 800e-9, (200e-9, 2e-6), 200e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        return Simulation.from_capillary(
            16e-6,
            WINDOW,
            "H2",
            (40.0, 20.0),
            grid=grid,
            wavelength=800e-9,
            fwhm=30e-15,
            energy=2e-6,
            modes=1,
            rtol=rtol,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=2,
            integrator=integrator,
            raman=False,
            plasma=False,
            absorber=absorber(grid, boundary_length=1.0),
        )


def matched_window(simulation: Simulation, *, compensated: bool = False) -> Built:
    """Return a window replayed on the reference's own accepted steps, fixed.

    The reference marches the simulation adaptively through its session; its accepted
    steps, the last stretched to end on :data:`WINDOW`, become the build's fixed
    schedule, so that two backends run one discrete method and history (SEP 25). The
    run is saved at the schedule's end, within an ulp or two of the window's, where
    the old tests' runs made their last observation.
    `compensated` asserts the reference's march transports the gradient's
    compensation (section 4.5).
    """
    problem = simulation.translate()
    run_request = simulation._run_request()
    backend = NumpyRefBackend(
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        result = run(backend, problem, run_request, record={})
        attempts = result.joined[0]
        march = backend._adams
        assert march is not None
        assert not compensated or march._engine.compensated
    finally:
        backend.close()
    steps = np.asarray(attempts.dz)[np.asarray(attempts.accepted, dtype=bool)].copy()
    steps[-1] = WINDOW - float(np.sum(steps[:-1]))
    schedule = FixedSchedule(steps)
    # The schedule's end is a rounding from the window's, and the run's last
    # observation is there: the end is the plane.
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=schedule,
            save_planes=np.array([float(schedule.edges[-1])]),
        )
    )
    return Built(
        problem,
        run_request,
        simulation.grid,
        modes=simulation.modes,
        simulation=simulation,
    )

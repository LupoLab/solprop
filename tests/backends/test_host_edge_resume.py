# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference judges a resumed launch's safety ramp on its clean columns (SEP 40).

A transverse edge's safety ramp judges the launch without its ``input`` noise. A fresh
load has both parts and forms the clean columns there; a resumed state already carries
its draw, and a run with no record of the monitor judges it as a launch, so the
reference takes the draw away from the state it holds. Both judge the same columns.
"""

import warnings
from typing import Any, cast

import numpy as np
import pytest

from solprop.backends.registry import select
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryWarning
from solprop.grid import EnvelopeGrid, LineGrid
from solprop.noise import NoiseRealisation
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.transverse_monitor import TransverseEdge

LENGTH = 2e-3
"""The run's length, m."""


def _line() -> Simulation:
    """Return a linear line-route run on the reference, its collar outside the beam."""
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=0.0),
        save_planes=[LENGTH / 2, LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH * 1e-3,
        backend="numpy_ref",
        spatial_absorber=SpatialAbsorber(kappa=8e4),
    )
    return Simulation.from_free(
        EnvelopeGrid.from_limits(LENGTH, 800e-9, (700e-9, 900e-9), 200e-15),
        LineGrid(300e-6, 96, collar_fraction=1.0),
        SpatialMedium(1.0),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9), 40e-6),
        settings=settings,
        noise="input",
    )


def _judged(simulation: Simulation, state: np.ndarray, draw, *, resume: bool) -> None:
    """Load `state` with `draw` and attach a fresh monitor, which judges the launch."""
    problem = simulation.translate()
    backend = cast(
        "Any",
        select(
            "numpy_ref",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        ),
    )
    try:
        backend.set_initial(state, draw, resume=resume)
        monitor = simulation._boundary_monitor()
        assert monitor is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            backend.attach_monitor(monitor)
    finally:
        backend.close()


def test_a_resume_with_no_monitor_record_judges_the_launch_without_its_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = TransverseEdge._judge_ramp_band
    judged: list[np.ndarray] = []

    def seen(self: Any, columns: np.ndarray, *args: Any, **kwargs: Any) -> None:
        judged.append(np.array(columns))
        judge(self, columns, *args, **kwargs)

    monkeypatch.setattr(TransverseEdge, "_judge_ramp_band", seen)
    simulation = _line()
    problem = simulation.translate()
    assert problem.transverse is not None
    rng = np.random.default_rng(113)
    shape = problem.shape[1:]
    # Inside the active band, so the mask the load applies leaves the draw whole.
    spectrum = 1e-9 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    spectrum = spectrum * problem.grids.active_mask
    draw = NoiseRealisation(spectrum=spectrum, seed=None, model="input")
    field0 = np.asarray(problem.initial.field0)
    _judged(simulation, field0, draw, resume=False)
    # The resumed state carries the draw it was launched with.
    _judged(simulation, field0 + spectrum[np.newaxis], draw, resume=True)
    assert len(judged) == 2
    fresh, resumed = judged
    assert np.max(fresh) > 0.0
    # A column's own rounding is that of adding the draw and taking it away again.
    np.testing.assert_allclose(resumed, fresh, rtol=1e-10, atol=1e-30)

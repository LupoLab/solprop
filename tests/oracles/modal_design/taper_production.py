# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the taper note's cases through the production route.

The taper note's case -- a hollow capillary tapering from 125 to 62.5 micrometres
radius over 1 or 10 cm between 5 mm straight ends, at 800 nm, launched in HE_11 --
is propagated by ``prop_capillary`` (``numpy_ref``) with a few channels, under one of
the note's two boundaries:

* **Robin**: ``wall=CoherentWall()``, the condensed outgoing boundary with the HE
  cladding term of a constant ``cladding=1.45``, which is the prototype's ``Wall``
  impedance exactly; the taper connection is carried by the frozen residual flow;
* **lossless**: ``loss=False`` and no wall.

The production run is linear: the fill is at zero pressure, so the Kerr term the
entry point requires has no density to act on, and the collar is ``1e9`` m long, so
its rates are negligible (below 1e-9 of the launch over the 11 cm case). Being linear,
every frequency bin propagates on its own, and the grid is built so that the carrier
is exactly a bin (``grid_for``); the modal fractions are read at that bin as
``|b_n(L)|^2 / |b_1(0)|^2``, with ``b`` the stored true-power amplitude.

The full campaign against the prototype's mapped-coordinate FEM was run by hand; its
small case's receipt is ``tests/data/taper_note.json``, which
``tests/driver/test_taper_note.py`` compares the production route with.
"""

import dataclasses
import itertools
import time
import warnings

import numpy as np

from modal_design.taper import Taper
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import ConvergenceError
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall
from solprop.profiles import FromCallable, PiecewiseLinear
from solprop.stepper import FixedSchedule, RungeKutta

C = 299792458.0
COUNTS = (8, 12, 20)


@dataclasses.dataclass(frozen=True)
class Case:
    """One taper-note case: the geometry, the boundary and the carrier."""

    length: float
    smooth: bool
    robin: bool
    wavelength: float = 800e-9
    counts: tuple[int, ...] = COUNTS
    per_straight: int = 50
    """Fixed steps per straight end on the coarser schedule."""
    scale: float = 1.0
    """Radii scaled by this and lengths by its square: the paraxial similarity.

    In ``x = r/a`` and ``z / (k a^2)`` the lossless equation is unchanged by it, so a
    scaled case is the note's conversion at a size a ten-step test can run; the Robin
    impedance ``k a / v`` scales with the radius and is not invariant.
    """

    @property
    def name(self):
        parts = [
            f"{round(self.length * 100)}cm",
            "smooth" if self.smooth else "kinked",
            "robin" if self.robin else "lossless",
        ]
        if self.wavelength != 800e-9:
            parts.append(f"{round(self.wavelength * 1e9)}nm")
        if self.scale != 1.0:
            parts.insert(0, f"scaled{self.scale:g}")
        return "-".join(parts)

    @property
    def taper(self):
        return Taper(
            length=self.length * self.scale**2,
            smooth=self.smooth,
            loss=self.robin,
            initial_radius=125e-6 * self.scale,
            final_radius=62.5e-6 * self.scale,
            straight=5e-3 * self.scale**2,
            wavelength=self.wavelength,
        )


CASES = {
    case.name: case
    for case in (
        # The note's campaign: the kinked taper, both boundaries, both lengths.
        Case(0.01, smooth=False, robin=True),
        Case(0.01, smooth=False, robin=False),
        Case(0.1, smooth=False, robin=True),
        Case(0.1, smooth=False, robin=False),
        # Side study: smoothed ends.
        Case(0.01, smooth=True, robin=True),
        Case(0.1, smooth=True, robin=True),
        Case(0.01, smooth=True, robin=False),
        # Side study: one shorter wavelength.
        Case(0.01, smooth=False, robin=True, wavelength=400e-9, counts=(8, 12, 20, 30)),
        Case(0.01, smooth=False, robin=False, wavelength=400e-9, counts=(8, 20, 30)),
        # The pinned regression's case: the 1 cm Robin taper at a quarter of the
        # radius, 1.25 mm long, four channels and twelve steps on the coarse schedule
        # (tests/driver/test_taper_note.py).
        Case(0.01, smooth=False, robin=True, counts=(4,), per_straight=3, scale=0.25),
        # Its lossless twin, which the taper coupling's power conservation test runs
        # (tests/driver/test_taper_coupling.py).
        Case(0.01, smooth=False, robin=False, counts=(4,), per_straight=3, scale=0.25),
    )
}


def profile(case):
    """Return the production radius profile of a case, with its analytic slope."""
    taper = case.taper
    events = np.array(taper.events, dtype=float)
    if not case.smooth:
        a0, a1 = taper.initial_radius, taper.final_radius
        return PiecewiseLinear.from_stages(events, [a0, a0, a1, a1])

    def value(z):
        return np.array([taper.geometry(float(point))[0] for point in np.atleast_1d(z)])

    def slope(z):
        return np.array([taper.geometry(float(point))[1] for point in np.atleast_1d(z)])

    return FromCallable(
        value, (0.0, float(events[-1])), kinks=events[1:-1].copy(), slope=slope
    )


_RECORDED = {"fft_size": "pow2", "collar_fraction": 0.05, "collar_placement": "inside"}
"""The grid geometry the study was recorded on: a power-of-two transform whose window
is the request, with the temporal collar inside it. :func:`grid_for` sizes the window
so that the carrier is a bin, which a window widened past the request undoes."""


def grid_for(wavelength, length):
    """Return a field grid on which the carrier is exactly a frequency bin.

    ``d_omega = 2 pi / (n_to dt_fine)``; choosing ``dt_fine = K T / n_to`` with ``T``
    the carrier's period makes the carrier bin ``K``. ``K`` is the largest integer
    that keeps ``dt_fine`` at or below what the band demands.
    """
    limits = (0.625 * wavelength, 1.25 * wavelength)
    default = FieldGrid.from_limits(length, wavelength, limits, 200e-15, **_RECORDED)
    period = wavelength / C
    size = default.n_to
    bins = int(np.floor(default.dt_fine * size / period))
    step = bins * period / size
    grid = FieldGrid.from_limits(
        length, wavelength, limits, size * step, dt=step, **_RECORDED
    )
    carrier = int(np.argmin(abs(grid.omega - 2 * np.pi / period)))
    if abs(grid.omega[carrier] * period / (2 * np.pi) - 1) > 1e-12:
        raise RuntimeError("the carrier is not a bin of the constructed grid")
    if not grid.inband_mask[carrier]:
        raise RuntimeError("the carrier bin is outside the scored band")
    return grid, carrier


def schedule(case, per_straight):
    """Return a fixed schedule landing on every kink, ``per_straight`` per straight."""
    events = case.taper.events
    pieces = []
    for left, right in itertools.pairwise(events):
        count = max(1, round(per_straight * (right - left) / case.taper.straight))
        pieces.append(np.diff(np.linspace(left, right, count + 1)))
    steps = np.concatenate(pieces)
    # The schedule must reach the end exactly; linspace's segments do, their sum may
    # not by an ulp, and the replay refuses a schedule that stops short.
    steps[-1] += events[-1] - float(np.sum(steps))
    return FixedSchedule(steps)


def production(case, count, per_straight, validity=None, **settings):
    """Run one production case and return its fractions at the carrier bin.

    The study's receipts were recorded with the Runge-Kutta march on fixed schedules,
    so that is the march here. `settings` override the entry point's, as
    ``integrator=`` or ``schedule=None`` with an ``rtol`` for the controller's steps
    under the fixed schedule's longest.
    """
    length = case.taper.events[-1]
    grid, carrier = grid_for(case.wavelength, length)
    wall = None
    if case.robin:
        wall = (
            CoherentWall()
            if validity is None
            else CoherentWall(validity_limit=validity)
        )
    fixed = schedule(case, per_straight)
    step = float(np.max(fixed.steps))
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = Simulation.from_capillary(
            profile(case),
            length,
            "He",
            0.0,
            wavelength=case.wavelength,
            grid=grid,
            absorber=absorber(grid, boundary_length=1e9),
            fwhm=30e-15,
            energy=1e-9,
            modes=count,
            n_save=2,
            backend="numpy_ref",
            wall=wall,
            loss=case.robin,
            cladding=1.45,
            plasma=False,
            raman=False,
            **(
                {
                    "noise": None,
                    **{
                        "schedule": fixed,
                        "dz_max": step,
                        "dz_init": step,
                        "integrator": RungeKutta(),
                    }
                    | settings,
                }
            ),
        )
        record = simulation.metadata().modal or {}
        try:
            handler = simulation.run(progress="none")
        except ConvergenceError as error:
            # A replay whose error norm turns non-finite now stops by name instead of
            # writing NaN into the output; either way the level diverged.
            handler, failure = None, str(error)
    seconds = time.perf_counter() - start
    if handler is None:
        return {
            "count": count,
            "steps": len(fixed.steps),
            "largest_step": step,
            "validity_limit": validity,
            "power": float("nan"),
            "fractions": np.full(count, np.nan),
            "seconds": seconds,
            "flow": None,
            "warnings": sorted({type(w.message).__name__ for w in caught}),
            "error": failure,
        }
    planes = handler.planes()
    launch = planes[0].field[:, 0, carrier]
    final = planes[-1].field[:, 0, carrier]
    if np.max(abs(launch[1:])) != 0.0:
        raise RuntimeError("the launch is not HE_11 alone")
    fractions = abs(final) ** 2 / abs(launch[0]) ** 2
    flow = record.get("flow") if isinstance(record, dict) else None
    return {
        "count": count,
        "steps": len(fixed.steps),
        "largest_step": step,
        "validity_limit": validity,
        "power": float(np.sum(fractions)),
        "fractions": fractions,
        "seconds": seconds,
        "flow": flow,
        "warnings": sorted({type(w.message).__name__ for w in caught}),
    }

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the taper note's cases through the production route against the mapped FEM.

The taper note's case -- a hollow capillary tapering from 125 to 62.5 micrometres
radius over 1 or 10 cm between 5 mm straight ends, at 800 nm, launched in HE_11 --
is propagated by ``prop_capillary`` (``numpy_ref``) with N = 8, 12 and 20 channels, and
compared with the prototype's mapped-coordinate radial FEM (``taper.spatial``), which is
re-run here and checked against the prototype's recorded receipt at 800 nm.

The two boundaries are the note's:

* **Robin**: ``wall=CoherentWall()``, the condensed outgoing boundary with the HE
  cladding term of a constant ``cladding=1.45``, which is the prototype's ``Wall``
  impedance exactly; the taper connection is carried by the frozen residual flow;
* **lossless**: ``loss=False`` and no wall, the production route for a lossless
  multimode taper, which was **adiabatic** when this study was written: it carried no
  connection, so it was measured as what it was (the route now carries the taper
  coupling by default, ``docs/guide/waveguides/modes.md``).

The production run is linear: the fill is at zero pressure, so the Kerr term the
entry point requires has no density to act on, and the collar is ``1e9`` m long, so
its rates are negligible (below 1e-9 of the launch over the 11 cm case). Being linear,
every frequency bin propagates on its own, and the grid is built so that the carrier
is exactly a bin (``grid_for``); the modal fractions are read at that bin as
``|b_n(L)|^2 / |b_1(0)|^2``, with ``b`` the stored true-power amplitude.

Targets, stated before any result:

* launched-power error ``|P_prod - P_FEM| <= 2e-4``, with ``P_FEM`` the FEM's final
  power over its own launched power (its launch interpolation deficit, 1.3e-6 at 512
  elements, is its discretisation, not the model's);
* for every FEM fraction above ``1e-3``, the relative error ``<= 1 %``;
* for every FEM fraction at or below ``1e-3``, the absolute error ``<= 1e-5``: the
  1 % relative target evaluated at the 1e-3 threshold, so that the two targets meet
  continuously there. It is a bound derived from the stated targets, not fitted to a
  result.

The FEM's own error is estimated from its ladder, which refines the radial and the z
grid independently (second order in each): ``|f(512, 2048) - f(256, 2048)| / 3`` and
``|f(512, 8192) - f(512, 4096)| / 3``. The production's own z error is the difference
between the last two of a ladder of fixed schedules, each twice as fine as the one
before; a level that is not passive, or whose replay stops with a ``ConvergenceError``,
is recorded as diverged and skipped (``production_ladder``). Conversion is not required
to be monotonic in the transition length: kink interference oscillates.

Run by hand, never by a lane; each case is bounded (under eight minutes on a loaded
laptop). With ``PYTHONPATH=tests/oracles``::

    uv run --no-sync python -m modal_design.taper_production DIR --case 1cm-kinked-robin
    uv run --no-sync python -m modal_design.taper_production DIR --all
    uv run --no-sync python -m modal_design.taper_production DIR --summarise

Each case writes ``DIR/<case>.json``; ``--summarise`` prints the table from them.
"""

import argparse
import dataclasses
import datetime
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from modal_design.taper import Taper, modal, spatial
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import ConvergenceError, ModeError
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall
from solprop.profiles import FromCallable, PiecewiseLinear
from solprop.stepper import FixedSchedule, RungeKutta

ROOT = Path(__file__).resolve().parents[2]
WALL_RECEIPT = Path(__file__).parent / "results" / "wall-receipt.json"
C = 299792458.0
COUNTS = (8, 12, 20)
LADDER = ((128, 2048), (256, 2048), (512, 2048), (512, 4096), (512, 8192))
POWER_TARGET = 2e-4
RELATIVE_TARGET = 1e-2
STRONG = 1e-3
WEAK_BOUND = RELATIVE_TARGET * STRONG
RELAXED_VALIDITY = 0.4
"""The validity limit a refused block is retried under, and recorded as such.

The ratios the default 0.25 refuses are the **collar's**: the active band reaches half
the lowest in-band frequency, where ``u_20 / (k a)`` at 62.5 micrometres is 0.32 to
0.37, while at the scored carrier it is 0.126. Collar bins are unscored and carry no
connection, and in a linear run no bin feeds another, so the relaxation
changes no compared number; that it was needed is itself a result.
"""


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


def diverged(result):
    """Return whether a run's fractions are not a passive answer at all."""
    fractions = np.asarray(result["fractions"])
    return bool(not np.all(np.isfinite(fractions)) or result["power"] > 1.0 + 1e-6)


def production_ladder(case, count):
    """Run a case on doubling schedules until two successive ones are passive.

    A block the default validity limit refuses is retried under
    :data:`RELAXED_VALIDITY`, and the refusal is kept. A schedule far above the
    advisory step can make the explicit residual grow without bound (the advisory step
    is advisory only); such a level is recorded as diverged and the ladder goes one
    level finer, at most three times.
    """
    refusal = None
    validity = None
    try:
        levels = [production(case, count, case.per_straight)]
    except ModeError as error:
        if "admits" not in str(error):
            raise
        refusal = str(error)
        validity = RELAXED_VALIDITY
        levels = [production(case, count, case.per_straight, validity)]
    per_straight = case.per_straight
    while len(levels) < 2 or diverged(levels[-2]) or diverged(levels[-1]):
        if len(levels) == 4:
            raise RuntimeError(f"{case.name} N={count} diverged on every schedule")
        per_straight *= 2
        levels.append(production(case, count, per_straight, validity))
    coarse, fine = levels[-2], levels[-1]
    fine["refusal"] = refusal
    fine["diverged"] = [
        {"steps": level["steps"], "power": level["power"], "error": level.get("error")}
        for level in levels[:-2]
    ]
    fine["coarse"] = {
        key: coarse[key] for key in ("steps", "power", "fractions", "seconds")
    }
    fine["z_difference"] = float(
        np.max(abs(np.asarray(fine["fractions"]) - coarse["fractions"]))
    )
    return fine


def reference(case):
    """Run the FEM ladder and the prototype's own modal solve of one case."""
    taper = case.taper
    # The FEM projects onto the first 20 orders, as the prototype did, or onto as many
    # as the largest retained set; a longer projection leaves the first 20 unchanged.
    projected = max(20, *case.counts)
    ladder = [
        spatial(taper, elements, steps, count=projected) for elements, steps in LADDER
    ]
    finest = ladder[-1]
    radial = abs(ladder[2]["fractions"] - ladder[1]["fractions"]) / 3
    longitudinal = abs(ladder[4]["fractions"] - ladder[3]["fractions"]) / 3
    prototype = []
    for count in case.counts:
        field, nfev = modal(taper, count)
        prototype.append({"count": count, "fractions": abs(field) ** 2, "nfev": nfev})
    return {
        "ladder": ladder,
        "power": finest["power"] / finest["launch_power"],
        "fractions": finest["fractions"] / finest["launch_power"],
        "error_estimate": radial + longitudinal,
        "power_error_estimate": (
            abs(ladder[2]["power"] - ladder[1]["power"])
            + abs(ladder[4]["power"] - ladder[3]["power"])
        )
        / 3,
        "prototype": prototype,
        "prototype_difference": prototype_difference(taper, finest),
    }


def prototype_difference(taper, finest):
    """Return the largest departure from the prototype receipt's finest FEM row."""
    if taper.wavelength != 800e-9 or not WALL_RECEIPT.exists():
        return None
    receipt = json.loads(WALL_RECEIPT.read_text())["artifacts"]["taper-full.json"]
    for row in receipt["results"]:
        stored = row["taper"]
        if (
            stored["length"] == taper.length
            and stored["smooth"] == taper.smooth
            and stored["loss"] == taper.loss
        ):
            spatial_row = row["spatial"][-1]
            stored_fractions = np.asarray(spatial_row["fractions"])
            return float(
                max(
                    np.max(abs(stored_fractions - finest["fractions"])),
                    abs(spatial_row["power"] - finest["power"]),
                )
            )
    return None


def compare(fractions, power, fem):
    """Return the three measures against the FEM, over the retained channels."""
    count = len(fractions)
    target = fem["fractions"][:count]
    strong = target > STRONG
    relative = abs(fractions[strong] - target[strong]) / target[strong]
    absolute = abs(fractions[~strong] - target[~strong])
    return {
        "power_error": float(abs(power - fem["power"])),
        "max_relative_strong": float(np.max(relative)) if relative.size else 0.0,
        "max_absolute_weak": float(np.max(absolute)) if absolute.size else 0.0,
        "strong_channels": int(np.sum(strong)),
        "fem_beyond_retained": float(np.sum(fem["fractions"][count:])),
        "fem_error_estimate_strong": float(
            np.max(fem["error_estimate"][:count][strong] / target[strong])
        ),
        "fem_error_estimate_weak": float(
            np.max(fem["error_estimate"][:count][~strong], initial=0.0)
        ),
    }


def verdict(measures):
    """Return whether one row meets the three stated targets."""
    return bool(
        measures["power_error"] <= POWER_TARGET
        and measures["max_relative_strong"] <= RELATIVE_TARGET
        and measures["max_absolute_weak"] <= WEAK_BOUND
    )


def run_case(case):
    """Run one case end to end and return its receipt."""
    start = time.perf_counter()
    fem = reference(case)
    rows = []
    for count in case.counts:
        result = production_ladder(case, count)
        measures = compare(np.asarray(result["fractions"]), result["power"], fem)
        prototype = next(row for row in fem["prototype"] if row["count"] == count)
        measures["prototype_difference"] = float(
            np.max(abs(np.asarray(result["fractions"]) - prototype["fractions"]))
        )
        prototype_measures = compare(
            prototype["fractions"], float(np.sum(prototype["fractions"])), fem
        )
        rows.append(
            {
                "production": result,
                "measures": measures,
                "meets_targets": verdict(measures),
                "prototype_measures": prototype_measures,
            }
        )
        print(
            f"{case.name} N={count}: power {measures['power_error']:.3e}, "
            f"strong {measures['max_relative_strong']:.3e}, "
            f"weak {measures['max_absolute_weak']:.3e}, "
            f"z {result['z_difference']:.1e}, {result['seconds']:.0f} s",
            flush=True,
        )
    return {
        "case": dataclasses.asdict(case) | {"name": case.name},
        "targets": {
            "power": POWER_TARGET,
            "relative_strong": RELATIVE_TARGET,
            "strong_threshold": STRONG,
            "absolute_weak": WEAK_BOUND,
        },
        "fem": fem,
        "rows": rows,
        "seconds": time.perf_counter() - start,
        "provenance": provenance(),
    }


def provenance():
    """Return what produced a receipt, for the record rather than for a key."""
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    return {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "head": head,
        "tracked_changes": dirty.splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "load_average": os.getloadavg(),
        "threads": {
            key: value
            for key, value in os.environ.items()
            if "THREAD" in key or key.startswith("OMP_")
        },
    }


def encode(value):
    """Return a JSON-serialisable copy, arrays as lists."""
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarise(directory):
    """Print the comparison table from every receipt in a directory."""
    print(
        "| Case | N | Power error (<= 2e-4) | Max strong relative (<= 1 %) "
        "| Max weak absolute (<= 1e-5) | Production z difference "
        "| Prototype difference | Meets |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for path in sorted(Path(directory).glob("*.json")):
        receipt = json.loads(path.read_text())
        for row in receipt["rows"]:
            measures = row["measures"]
            print(
                f"| {receipt['case']['name']} | {row['production']['count']} "
                f"| {measures['power_error']:.2e} "
                f"| {100 * measures['max_relative_strong']:.3f} % "
                f"| {measures['max_absolute_weak']:.2e} "
                f"| {row['production']['z_difference']:.1e} "
                f"| {measures['prototype_difference']:.1e} "
                f"| {'yes' if row['meets_targets'] else 'no'} |"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="directory for the receipts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", action="append", choices=sorted(CASES))
    group.add_argument("--all", action="store_true")
    group.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
    if args.summarise:
        summarise(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name in sorted(CASES) if args.all else args.case:
        target = args.output / f"{name}.json"
        if target.exists():
            parser.error(f"refusing to overwrite {target}")
        receipt = run_case(CASES[name])
        target.write_text(json.dumps(encode(receipt), indent=1) + "\n")
        print(f"wrote {target} ({receipt['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

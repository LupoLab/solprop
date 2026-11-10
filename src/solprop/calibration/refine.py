# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 2: refine the workspace settings for one simulation (SEP 46 section 6.6).

The machine profile fits the native backend to the machine; a costly campaign can
then fit it to its own simulation. :func:`refine` builds fresh workspaces from the
simulation's translated problem in the calling process, under the process's record
and under each candidate's, and never touches the simulation's state, output or
callbacks. The candidates are the operation-grain moves of the simulation's own work
sizes (:mod:`solprop.calibration.grain`), each timed by trial steps of the
simulation's own operators at positions spread along its length, interleaved with the
base record and admitted by §6.5's rule; the winners are validated together. Speed
alone never admits a candidate: every candidate's trial steps must end finite and
within ten times the run's relative tolerance of the base's (section 12 item 4).

The result is stored under the simulation's execution descriptor
(``_profile.descriptor``), and every later run or scan point with an equal descriptor
applies it while the machine profile it refined still holds.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from solprop.backends.session import Start
from solprop.calibration import _native, grain, selection
from solprop.exceptions import CalibrationError

if TYPE_CHECKING:
    from solprop.driver import Simulation

__all__ = ["ACCURACY", "POSITIONS", "ROUNDS", "RefinementReport", "refine"]

ROUNDS = 5
"""Interleaved rounds per comparison, as at the standard level."""

POSITIONS = 3
"""Trial-step windows along the simulation's length: its start, a third and two."""

ACCURACY = 10.0
"""How many times the run's relative tolerance a candidate's fields may differ by."""


@dataclass(frozen=True)
class RefinementReport:
    """What a refinement measured and kept.

    Attributes
    ----------
    status : str
        ``"published"`` (a refinement was stored), ``"kept"`` (the machine profile
        stays: no candidate won and was accurate), or ``"inconclusive"`` (too noisy,
        too little time, or no machine profile to refine).
    descriptor : str
        The simulation's execution descriptor.
    changes : Mapping[str, object]
        The workspace settings stored over the process's record.
    evidence : Mapping[str, object]
        The noise, each candidate's verdict and accuracy, and the validation.
    reason : str
        Why.
    seconds : float
        The wall time taken.
    path : pathlib.Path or None
        The refinement file, if one was stored.
    """

    status: Literal["published", "kept", "inconclusive"]
    descriptor: str
    changes: Mapping[str, object] = field(default_factory=dict[str, object])
    evidence: Mapping[str, object] = field(default_factory=dict[str, object])
    reason: str = ""
    seconds: float = 0.0
    path: Path | None = None

    def describe(self) -> str:
        """Return the report as lines for a terminal.

        Returns
        -------
        str
            The status, reason, changes and file.
        """
        lines = [
            f"refinement: {self.status} in {self.seconds:.1f} s ({self.descriptor})",
            f"  {self.reason}",
        ]
        lines += [f"  {name}: {value}" for name, value in self.changes.items()]
        if self.path is not None:
            lines.append(f"  refinement: {self.path}")
        return "\n".join(lines) + "\n"


class _Windows:
    """One record's backend on the simulation, and its trial steps along it.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation.
    record : Mapping[str, int or bool] or None
        A native execution record, or ``None`` for the process's own.
    """

    def __init__(
        self, simulation: Simulation, record: Mapping[str, int | bool] | None
    ) -> None:
        from solprop_native._execution import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            ExecutionRecord,
        )
        from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        problem = simulation.translate()
        self.backend = NativeBackend(
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            cpu_execution=None
            if record is None
            else ExecutionRecord.from_native(record),
        )
        self.probe = self.backend.timing_probe(Start.entrance(problem))
        start = float(problem.initial.z)
        end = float(max(problem.stepper.save_planes, default=start))
        self.dz = float(problem.stepper.dz_init)
        self.positions = (
            [start]
            if problem.stepper.schedule is not None
            else [
                start + (end - start) * index / POSITIONS for index in range(POSITIONS)
            ]
        )
        self.low_storage = problem.stepper.schedule is not None

    def fields(self) -> list[np.ndarray]:
        """Return the field each window's trial step ends in, on the host.

        A low-storage march advances its state with each step, so it has no fields
        to compare and returns none.

        Returns
        -------
        list of numpy.ndarray
            One per window.
        """
        found: list[np.ndarray] = []
        for z in self.positions:
            if self.low_storage:
                return []
            self.probe.trial(z, self.dz)
            found.append(self.probe.field())
        return found

    def time(self, repeats: int) -> float:
        """Return the seconds per trial step over the windows, after a warm-up.

        Parameters
        ----------
        repeats : int
            Trial steps timed in each window.

        Returns
        -------
        float
            Seconds per step.
        """
        steps: list[Callable[[], object]] = (
            [self.probe.trial]
            if self.low_storage
            else [(lambda z=z: self.probe.trial(z, self.dz)) for z in self.positions]
        )
        for step in steps:
            step()
            step()
        begun = time.perf_counter()
        for step in steps:
            for _ in range(repeats):
                step()
        return (time.perf_counter() - begun) / (repeats * len(steps))

    def close(self) -> None:
        """Close the backend."""
        self.backend.close()


def _accurate(
    found: list[np.ndarray], reference: list[np.ndarray], rtol: float
) -> bool:
    """Return whether `found` is finite and within ``ACCURACY * rtol`` of `reference`.

    Parameters
    ----------
    found, reference : list of numpy.ndarray
        The windows' trial-step fields under a candidate and under the base.
    rtol : float
        The run's relative tolerance.

    Returns
    -------
    bool
        Whether every window agrees, in the relative L2 norm.
    """
    for mine, theirs in zip(found, reference, strict=True):
        if not np.all(np.isfinite(mine)):
            return False
        scale = float(np.linalg.norm(theirs)) or 1.0
        if float(np.linalg.norm(mine - theirs)) > ACCURACY * rtol * scale:
            return False
    return True


def refine(
    simulation: Simulation,
    budget: float = 60.0,
    *,
    progress: Callable[[str], object] | None = None,
) -> RefinementReport:
    """Refine the native workspace settings for `simulation` (SEP 46 stage 2).

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation; it is translated if it is not already, and otherwise left
        alone.
    budget : float, optional
        The deadline, in seconds.
    progress : callable or None, optional
        Called with a line of text as each stage starts.

    Returns
    -------
    RefinementReport
        What was kept and whether it was stored.

    Raises
    ------
    CalibrationError
        For a budget that is not positive, or a simulation not on the native backend.
    """
    started = time.monotonic()
    report = _refine(simulation, budget, progress)
    return dataclasses.replace(report, seconds=time.monotonic() - started)


def _candidates(
    problem: object,
    sizing: Mapping[str, object],
    base: Mapping[str, int | bool],
    threads: int,
) -> list[grain.Candidate]:
    """Return the grain candidates that change this simulation's widths.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem, for its shape.
    sizing : Mapping[str, object]
        Its workspace's ``execution_sizing`` under the base record.
    base : Mapping[str, int or bool]
        The process's record.
    threads : int
        The pool's width.

    Returns
    -------
    list of solprop.calibration.grain.Candidate
        At most twelve.
    """
    shape = cast(tuple[int, ...], getattr(problem, "shape"))  # noqa: B009 - typed loosely
    operations = grain.operations(sizing)
    compact = ["simulation"] if math.prod(shape[:-1]) == 1 else []
    return [
        candidate
        for candidate in grain.proposals(
            base, {"simulation": operations}, threads, compact, limit=12
        )
        if candidate.setting == "compact"
        or grain.affected(candidate.record(base), base, operations, threads)
    ]


def _rounds(
    base: _Windows, windows: Mapping[str, _Windows], repeats: int, until: float
) -> dict[str, list[float | None]]:
    """Time interleaved rounds of every candidate's windows against the base's.

    Parameters
    ----------
    base : _Windows
        The process's record's windows.
    windows : Mapping[str, _Windows]
        Each candidate's.
    repeats : int
        Trial steps per window.
    until : float
        The monotonic time after which no round starts.

    Returns
    -------
    dict of str to list
        ``again``, the base's A/A ratio, and each candidate's ratio to the base, per
        round.
    """
    generator = random.Random(5000)
    ratios: dict[str, list[float | None]] = {"again": []}
    ratios |= {name: [] for name in windows}
    for _ in range(ROUNDS):
        if time.monotonic() > until:
            break
        slots = [("base", base), ("again", base), *windows.items()]
        generator.shuffle(slots)
        seconds = {name: window.time(repeats) for name, window in slots}
        for name in ratios:
            ratios[name].append(seconds[name] / seconds["base"])
    return ratios


def _together(
    combined: _Windows, base: _Windows, repeats: int, noise: float
) -> selection.Validation:
    """Validate the admitted settings together against the base on fresh rounds.

    Parameters
    ----------
    combined, base : _Windows
        The combination's windows and the base's.
    repeats : int
        Trial steps per window.
    noise : float
        The session's A/A noise.

    Returns
    -------
    solprop.calibration.selection.Validation
        The verdict.
    """
    ratios: list[float | None] = []
    for index in range(ROUNDS):
        pair = [("combined", combined), ("base", base)]
        if index % 2:
            pair.reverse()
        timed = {name: window.time(repeats) for name, window in pair}
        ratios.append(timed["combined"] / timed["base"])
    return selection.validate(
        {"simulation": (ratios, [1.0] * len(ratios))}, noise, ROUNDS
    )


def _refine(
    simulation: Simulation,
    budget: float,
    progress: Callable[[str], object] | None,
) -> RefinementReport:
    """Carry out :func:`refine`; its report's seconds are set by the caller.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation.
    budget : float
        The deadline, in seconds.
    progress : callable or None
        Called with a line of text as each stage starts.

    Returns
    -------
    RefinementReport
        The report.

    Raises
    ------
    CalibrationError
        For a budget that is not positive, or a simulation not on the native backend.
    """
    if not budget > 0:
        raise CalibrationError(f"a refinement's budget must be positive, got {budget}")
    if simulation.settings.backend not in ("native", "auto"):
        raise CalibrationError(
            "a refinement fits the native backend; this simulation runs on "
            f"{simulation.settings.backend!r}"
        )
    say = progress or (lambda _: None)
    deadline = time.monotonic() + budget
    import solprop_native as native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    if native.PROFILE.fingerprint is None:
        return RefinementReport(
            "inconclusive", "", reason="SOLPROP_CALIBRATION=off: no profile to refine"
        )
    problem = simulation.translate()
    base_record = cast(dict[str, int | bool], native.EXECUTION.native())
    opened = [_Windows(simulation, None)]
    base = opened[0]
    try:
        sizing = cast(dict[str, object], base.backend.metadata()["execution_sizing"])
        key = native._profile.descriptor(  # pyright: ignore[reportPrivateUsage]
            # The store's key keeps the four-axis shape it has always been written
            # with, so that refinements recorded before the batch axis left the
            # problem (SEP 47 section 6.2) are found again.
            shape=(1, *problem.shape),
            grid_kind=problem.grid_kind,
            integrator=type(problem.stepper.integrator).__name__,
            sizing=sizing,
            process=native.PROCESS,
            fingerprint=native.PROFILE.fingerprint.digest,
        )
        candidates = _candidates(problem, sizing, base_record, native.PROCESS.threads)
        evidence: dict[str, object] = {"candidates": [c.name for c in candidates]}
        if not candidates:
            return RefinementReport(
                "kept", key, evidence=evidence, reason="no candidate changes a width"
            )
        say(f"building {len(candidates)} candidate workspaces")
        windows = {
            c.name: _Windows(simulation, c.record(base_record)) for c in candidates
        }
        opened += list(windows.values())
        rtol = float(problem.stepper.controller.rtol)
        reference = base.fields()
        accurate = {
            name: _accurate(window.fields(), reference, rtol)
            for name, window in windows.items()
        }
        repeats = max(2, min(500, round(0.1 / max(base.time(1), 1e-6))))
        say("timing interleaved rounds")
        ratios = _rounds(base, windows, repeats, deadline - 0.2 * budget)
        ones = [1.0] * len(ratios["again"])
        noise = selection.noise(ones, ratios["again"], ROUNDS)
        if noise is None or noise > selection.NOISE_LIMIT:
            return RefinementReport(
                "inconclusive",
                key,
                evidence=evidence | {"noise": noise},
                reason="too noisy or too little time to decide",
            )
        verdicts = {
            name: selection.admit(ratios[name], ones, noise, ROUNDS) for name in windows
        }
        evidence |= {
            "noise": noise,
            "accurate": accurate,
            "verdicts": {name: verdict.record() for name, verdict in verdicts.items()},
        }
        best = grain.best([c for c in candidates if accurate[c.name]], verdicts)
        changes: dict[str, int | bool] = {}
        for candidate in best:
            changes |= candidate.changes
        reason = "no accurate candidate won"
        if best:
            say(f"validating {len(best)} settings together")
            combined = _Windows(simulation, {**base_record, **changes})
            opened.append(combined)
            validation = _together(combined, base, repeats, noise)
            evidence["validation"] = validation.record()
            reason = (
                "the combination is inaccurate"
                if not _accurate(combined.fields(), reference, rtol)
                else "the combination's validation failed"
                if not validation.passed
                else ""
            )
        if reason:
            return RefinementReport("kept", key, evidence=evidence, reason=reason)
        store = _native.store.Store(_native.store.cache_directory())
        path = store.publish_refinement(
            native.PROFILE.fingerprint.digest,
            key,
            {
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "base": base_record,
                "workspace": changes,
                "evidence": evidence,
            },
        )
        return RefinementReport(
            "published",
            key,
            changes=changes,
            evidence=evidence,
            reason="admitted: " + ", ".join(candidate.name for candidate in best),
            path=path,
        )
    finally:
        for window in opened:
            window.close()

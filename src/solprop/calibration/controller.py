# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calibration controller: children, rounds, a deadline and publication (SEP 46).

The controller never loads the native extension (section 6.2). It starts one child
per candidate process setting (:mod:`solprop.calibration.child`), each pinned to the
allocation being calibrated, plus one child on the built-in rules, and times the
calibration problems' windows in interleaved rounds: in each round every child once,
in a fresh order, and the built-in child twice for the A/A noise. The selection rule
(:mod:`solprop.calibration.selection`) admits a candidate only if it wins every round
by more than the noise; the winner is then validated against the built-in child on
fresh rounds, window by window. Everything runs under one monotonic deadline per
level (section 6.9): a child that overruns is terminated, a stage that cannot finish
its rounds is skipped and reported, and nothing partial is published. Publication
writes one complete generation, keeping the other allocation classes' rows, and
swaps ``current`` (section 6.8).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from solprop import _process
from solprop.calibration import _native, selection
from solprop.exceptions import CalibrationError

if TYPE_CHECKING:
    from solprop_native._allocation import (  # pyright: ignore[reportMissingImports]
        AllocationClass,
    )
    from solprop_native._store import (  # pyright: ignore[reportMissingImports]
        Generation,
        Row,
        Store,
    )
    from solprop_native._topology import (  # pyright: ignore[reportMissingImports]
        Topology,
    )

__all__ = [
    "LEVELS",
    "CalibrationReport",
    "Candidate",
    "Level",
    "builtin_wait",
    "calibrate",
    "candidates",
    "nested",
]

type LevelName = Literal["quick", "standard", "thorough"]
type Scores = dict[str, list[float | None]]


@dataclass(frozen=True)
class Level:
    """One calibration level's budget and depth.

    Attributes
    ----------
    name : str
        ``"quick"``, ``"standard"`` or ``"thorough"``.
    budget : float
        Its deadline, in seconds, for all of its work.
    rounds : int
        Interleaved rounds per comparison, and the fewest that decide.
    window : float
        The seconds each timed window aims at.
    problems : tuple of str
        The calibration problems timed.
    widths : int
        The team widths tried besides the built-in rule's.
    reserve : float
        The share of the budget kept for validation and publication.
    """

    name: LevelName
    budget: float
    rounds: int
    window: float
    problems: tuple[str, ...]
    widths: int
    reserve: float = 0.25


_ALL = ("gnlse", "capillary", "averaged", "radial", "line")

LEVELS: dict[str, Level] = {
    "quick": Level("quick", 20.0, 3, 0.1, ("gnlse", "averaged", "line"), 1),
    "standard": Level("standard", 60.0, 5, 0.15, _ALL, 2),
    "thorough": Level("thorough", 300.0, 7, 0.4, _ALL, 2),
}
"""The levels: about 20 s, a minute and five minutes (SEP 46 section 6.9)."""

ORDER: dict[str, int] = {"quick": 0, "standard": 1, "thorough": 2, "device": 3}
"""Levels by depth: a higher level replaces a lower one for the same allocation."""

_CLEARED = (
    "SOLPROP_THREADS",
    "SOLPROP_NATIVE_THREADS",
    "SOLPROP_NATIVE_KERNEL_THREADS",
    "SOLPROP_NATIVE_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
    "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES",
    "SOLPROP_FFTW_WISDOM",
    "OMP_WAIT_POLICY",
    "GOMP_SPINCOUNT",
    "KMP_BLOCKTIME",
    "SOLPROP_NATIVE_WAIT",
    "SOLPROP_NATIVE_WAIT_REQUEST",
    "OMP_NUM_THREADS",
)
"""The user's thread, wait and wisdom settings, which a child never inherits."""


@dataclass(frozen=True)
class Candidate:
    """One candidate's process settings.

    Attributes
    ----------
    threads : int or None
        The pool width, or ``None`` for the built-in rule.
    wait : str or None
        ``"spin"`` or ``"passive"``, or ``None`` for the built-in rule.
    """

    threads: int | None = None
    wait: str | None = None

    @property
    def name(self) -> str:
        """``"built-in"``, or the wait and width, such as ``"spin/6"``.

        Returns
        -------
        str
            The candidate's name in reports and scores.
        """
        if self.threads is None and self.wait is None:
            return "built-in"
        return f"{self.wait or 'default'}/{self.threads or 'default'}"

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        """Return a child's environment, which never calibrates.

        `base` without the user's thread, wait and wisdom settings, with this
        candidate's, and with FFT plans at estimate, so that no stage outside the
        FFT stage spends its time planning (SEP 46 section 6.3).

        Parameters
        ----------
        base : Mapping[str, str]
            The controller's environment.

        Returns
        -------
        dict of str to str
            The child's.
        """
        environment = {k: v for k, v in base.items() if k not in _CLEARED}
        environment |= {
            "SOLPROP_CALIBRATION": "off",
            "SOLPROP_FFTW_PLAN": "estimate",
            "PYTHONUNBUFFERED": "1",
        }
        if self.threads is not None:
            environment["SOLPROP_NATIVE_THREADS"] = str(self.threads)
        if self.wait is not None:
            environment["SOLPROP_NATIVE_WAIT_REQUEST"] = self.wait
        return environment


@dataclass(frozen=True)
class CalibrationReport:
    """What a calibration measured, chose and published.

    Attributes
    ----------
    level : str
        The level run.
    status : str
        ``"published"``, ``"inconclusive"`` (nothing published: a busy or noisy
        machine, a failed child, or too little time), or ``"held"`` (another process
        is calibrating this allocation).
    fingerprint : str
        The machine and build's store key.
    allocation : Mapping[str, object]
        The allocation class calibrated.
    chosen : Mapping[str, object]
        Each setting chosen (the process's width and wait), the built-in values
        where none replaced them.
    builtin : Mapping[str, object]
        The built-in rules' values.
    evidence : Mapping[str, object]
        The noise, every candidate's verdict, the validation and the probes.
    reason : str
        Why the status is what it is.
    seconds : float
        The wall time taken.
    path : pathlib.Path or None
        The published generation, if any.
    """

    level: str
    status: Literal["published", "inconclusive", "held"]
    fingerprint: str
    allocation: Mapping[str, object]
    chosen: Mapping[str, object] = field(default_factory=dict[str, object])
    builtin: Mapping[str, object] = field(default_factory=dict[str, object])
    evidence: Mapping[str, object] = field(default_factory=dict[str, object])
    reason: str = ""
    seconds: float = 0.0
    path: Path | None = None

    def describe(self) -> str:
        """Return the report as lines for a terminal.

        Returns
        -------
        str
            The status, the reason, each setting beside its built-in value, and the
            profile written.
        """
        lines = [
            f"calibration ({self.level}): {self.status} in {self.seconds:.1f} s",
            f"  {self.reason}",
        ]
        for name, value in self.chosen.items():
            built = self.builtin.get(name)
            note = "" if value == built else f" (built-in {built})"
            lines.append(f"  {name}: {value}{note}")
        for item in cast(list[dict[str, object]], self.evidence.get("nested", [])):
            allocation = cast(dict[str, object], item["allocation"])
            lines.append(
                f"  nested {allocation['cores']} cores: {item['status']}, "
                f"{item['reason']}"
            )
        if self.path is not None:
            lines.append(f"  profile: {self.path}")
        return "\n".join(lines) + "\n"


class _Deadline:
    """One monotonic deadline for all of a level's work.

    Parameters
    ----------
    seconds : float
        The budget from now.
    """

    def __init__(self, seconds: float) -> None:
        self.start = time.monotonic()
        self.end = self.start + seconds

    def remaining(self) -> float:
        """Return the seconds left, never negative.

        Returns
        -------
        float
            Seconds.
        """
        return max(0.0, self.end - time.monotonic())

    def elapsed(self) -> float:
        """Return the seconds since the start.

        Returns
        -------
        float
            Seconds.
        """
        return time.monotonic() - self.start

    def epoch(self) -> float:
        """Return the deadline in seconds since the epoch, for a claim file.

        Returns
        -------
        float
            Seconds since the epoch.
        """
        return time.time() + self.remaining()


class _ChildError(CalibrationError):
    """A child exited, overran the deadline or reported an error."""


class _Child:
    """A calibration child process and its reply queue.

    Parameters
    ----------
    candidate : Candidate
        Its process settings.
    cpus : sequence of int
        The CPUs it pins itself to; empty for none.
    environ : Mapping[str, str]
        The controller's environment, from which the child's is made.
    """

    def __init__(
        self, candidate: Candidate, cpus: Sequence[int], environ: Mapping[str, str]
    ) -> None:
        self.candidate = candidate
        self.report: dict[str, object] = {}
        self.errors: list[str] = []
        self.replies: queue.Queue[str | None] = queue.Queue()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "solprop.calibration.child",
                ",".join(str(cpu) for cpu in cpus),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=candidate.environment(environ),
        )
        self.readers = (
            threading.Thread(target=self._read, daemon=True),
            threading.Thread(target=self._drain, daemon=True),
        )
        for reader in self.readers:
            reader.start()

    def _read(self) -> None:
        """Queue each reply line, then ``None`` at the end of the child's output."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def _drain(self) -> None:
        """Keep the child's last 20 lines of standard error, for a failure's report."""
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.errors.append(line.rstrip())
            del self.errors[:-20]

    def send(self, request: Mapping[str, object]) -> None:
        """Write one request.

        Parameters
        ----------
        request : Mapping[str, object]
            The request.

        Raises
        ------
        _ChildError
            When the child's input is closed or broken.
        """
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as error:  # a broken or a closed pipe
            raise _ChildError(f"{self.candidate.name}: {error}") from error

    def receive(self, timeout: float) -> object:
        """Return one reply's value, waiting at most `timeout` seconds.

        Parameters
        ----------
        timeout : float
            Seconds to wait.

        Returns
        -------
        object
            The reply's value.

        Raises
        ------
        _ChildError
            When the child overran, exited or reported an error.
        """
        try:
            line = self.replies.get(timeout=max(timeout, 0.0))
        except queue.Empty as error:
            raise _ChildError(f"{self.candidate.name} overran the deadline") from error
        if line is None:
            tail = "; ".join(self.errors[-3:])
            raise _ChildError(f"{self.candidate.name} exited: {tail}")
        reply = json.loads(line)
        if "error" in reply:
            raise _ChildError(f"{self.candidate.name}: {reply['error']}")
        return reply["ok"]

    def request(self, request: Mapping[str, object], timeout: float) -> object:
        """Send one request and return its reply's value.

        Parameters
        ----------
        request : Mapping[str, object]
            The request.
        timeout : float
            Seconds to wait for its reply.

        Returns
        -------
        object
            The reply's value.
        """
        self.send(request)
        return self.receive(timeout)

    def stop(self) -> None:
        """Ask the child to exit, then terminate and reap it and close its pipes."""
        with contextlib.suppress(_ChildError, OSError, ValueError):
            self.send({"op": "exit"})
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=2.0)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - survived SIGTERM
                self.process.kill()
                self.process.wait()
        # The readers end at the end of the child's output, which its exit closes.
        for reader in self.readers:
            reader.join(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            with contextlib.suppress(OSError, AttributeError):  # every pipe was opened
                stream.close()  # pyright: ignore[reportOptionalMemberAccess]


def builtin_wait(budget: int, top_tier: int | None) -> str:
    """Return the built-in rule's wait for a pool of `budget` threads (ADR-071).

    Idle threads spin unless the team is wider than the chip's fastest kind of core.

    Parameters
    ----------
    budget : int
        The pool's width.
    top_tier : int or None
        The fastest kind's cores on a chip with several, else ``None``.

    Returns
    -------
    str
        ``"spin"`` or ``"passive"``.

    Examples
    --------
    >>> builtin_wait(6, 6), builtin_wait(18, 6), builtin_wait(36, None)
    ('spin', 'passive', 'spin')
    """
    return "passive" if top_tier is not None and budget > top_tier else "spin"


def candidates(
    level: Level,
    allocation: AllocationClass,
    budget: int,
    top_tier: int | None = None,
) -> list[Candidate]:
    """Return a level's candidate process settings for an allocation.

    The other wait at the built-in width, then narrower widths at their built-in
    wait (the fastest kind's cores on a chip with several, one socket's on several
    sockets, one L3 group's, half the cores), then the other wait at the first of
    them; at most :data:`~solprop.calibration.selection.MAX_CANDIDATES`.

    Parameters
    ----------
    level : Level
        The level, whose ``widths`` bounds the narrower widths tried.
    allocation : AllocationClass
        The allocation being calibrated.
    budget : int
        The built-in rule's width for it (``thread_budget``).
    top_tier : int or None, optional
        The chip's fastest kind's cores, where it has several.

    Returns
    -------
    list of Candidate
        The candidates, none equal to the built-in rule.
    """
    narrower: list[int] = []
    if len(allocation.kinds) > 1:
        narrower.append(allocation.kinds[0])
    if allocation.sockets > 1:
        narrower.append(allocation.cores // allocation.sockets)
    if allocation.l3_groups > allocation.sockets:
        narrower.append(allocation.cores // allocation.l3_groups)
    narrower.append(budget // 2)
    widths = sorted({width for width in narrower if 1 <= width < budget}, reverse=True)
    widths = widths[: level.widths]

    def other(width: int) -> str:
        """Return the wait the built-in rule does not choose at `width`.

        Parameters
        ----------
        width : int
            A pool's width.

        Returns
        -------
        str
            ``"spin"`` or ``"passive"``.
        """
        return "spin" if builtin_wait(width, top_tier) == "passive" else "passive"

    found = [Candidate(threads=budget, wait=other(budget))]
    found += [
        Candidate(threads=width, wait=builtin_wait(width, top_tier)) for width in widths
    ]
    if widths:
        found.append(Candidate(threads=widths[0], wait=other(widths[0])))
    return found[: selection.MAX_CANDIDATES]


def nested(topology: Topology, cpus: Sequence[int]) -> list[tuple[int, ...]]:
    """Return the nested allocations a calibration of `cpus` also measures.

    One socket's CPUs, one NUMA node's and one L3 group's, where `cpus` spans more
    than one (SEP 46 section 6.4): one calibration of a whole node then covers most
    smaller jobs on it. Equal subsets are listed once, at most two.

    Parameters
    ----------
    topology : solprop_native._topology.Topology
        The machine's.
    cpus : sequence of int
        The allocation's CPUs.

    Returns
    -------
    list of tuple of int
        The first group's CPUs for each kind of group spanned.
    """
    found: list[tuple[int, ...]] = []
    for groups in (topology.socket, topology.numa, topology.l3):
        held = [cpu for cpu in cpus if cpu in groups]
        if len({groups[cpu] for cpu in held}) < 2:
            continue
        first = groups[held[0]]
        subset = tuple(cpu for cpu in held if groups[cpu] == first)
        if subset not in found:
            found.append(subset)
    return found[:2]


def _geomean(values: Sequence[float]) -> float:
    """Return the geometric mean of positive `values`.

    Parameters
    ----------
    values : sequence of float
        Positive values.

    Returns
    -------
    float
        Their geometric mean.
    """
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _telemetry(path: Path = Path("/proc/stat")) -> dict[str, float] | None:
    """Return the host's running tasks and stolen time so far, or ``None``.

    Linux only (``/proc/stat``); elsewhere load is unknown and only the A/A noise
    decides (SEP 46 section 6.5).

    Parameters
    ----------
    path : pathlib.Path, optional
        The kernel's statistics file.

    Returns
    -------
    dict of str to float or None
        ``total`` and ``steal`` ticks and ``running`` tasks; ``None`` where unknown.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "cpu" and len(fields) > 8:
            ticks = [float(value) for value in fields[1:]]
            values["total"] = sum(ticks)
            values["steal"] = ticks[7]
        elif fields and fields[0] == "procs_running":
            values["running"] = float(fields[1])
    return values


def _busy(
    before: dict[str, float] | None, after: dict[str, float] | None, idle: int
) -> str:
    """Return why the host was too busy to calibrate, or ``""``.

    Work not belonging to the calibration beyond the CPUs outside its allocation, or
    more than a tenth of the time stolen by a hypervisor.

    Parameters
    ----------
    before, after : dict or None
        :func:`_telemetry` before and after the measurements.
    idle : int
        The CPUs outside the allocation.

    Returns
    -------
    str
        The reason, empty when the host was quiet enough or its load is unknown.
    """
    if before is None or after is None:
        return ""
    # One running task is this controller, another the child being timed.
    foreign = min(before.get("running", 0.0), after.get("running", 0.0)) - 2.0
    if foreign > idle:
        return f"{foreign:.0f} other tasks running for {idle} idle CPUs"
    total = after.get("total", 0.0) - before.get("total", 0.0)
    stolen = after.get("steal", 0.0) - before.get("steal", 0.0)
    if total > 0 and stolen / total > 0.1:
        return f"{100 * stolen / total:.0f} % of the time stolen by the hypervisor"
    return ""


@dataclass(frozen=True)
class _Plan:
    """What every allocation's session shares: level, store, deadline, progress."""

    level: Level
    environ: Mapping[str, str]
    store: Store
    digest: str
    deadline: _Deadline
    say: Callable[[str], object]


@dataclass(frozen=True)
class _Bench:
    """One allocation's prepared children and the repeats of each window."""

    builtin: _Child
    rivals: tuple[_Child, ...]
    repeats: Mapping[str, int]

    def time(self, plan: _Plan, child: _Child) -> list[float]:
        """Return `child`'s seconds per step in each problem's window.

        Parameters
        ----------
        plan : _Plan
            The calibration's plan, for its problems and deadline.
        child : _Child
            The child to time.

        Returns
        -------
        list of float
            One value per problem.
        """
        seconds: list[float] = []
        for problem in plan.level.problems:
            reply = child.request(
                {"op": "time", "problem": problem, "repeats": self.repeats[problem]},
                plan.deadline.remaining(),
            )
            seconds.append(float(cast(float, reply)))
        return seconds


def _start(
    plan: _Plan, cpus: Sequence[int], allocation: AllocationClass, opened: list[_Child]
) -> _Bench:
    """Start and prepare the built-in child and the candidates; size the windows.

    Every child started is appended to `opened`, so that the caller reaps it.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    cpus : sequence of int
        The allocation's CPUs.
    allocation : AllocationClass
        Its class.
    opened : list of _Child
        Filled with the children started.

    Returns
    -------
    _Bench
        The prepared children and each window's repeats.
    """
    builtin = _Child(Candidate(), cpus, plan.environ)
    opened.append(builtin)
    width = _native_budget(cpus)
    for candidate in candidates(
        plan.level, allocation, width, _process.top_tier_cores()
    ):
        opened.append(_Child(candidate, cpus, plan.environ))
    for child in opened:
        child.send(
            {"op": "prepare", "problems": dict.fromkeys(plan.level.problems, 1.0)}
        )
    for child in opened:
        child.report = cast(dict[str, object], child.receive(plan.deadline.remaining()))
    steps = cast(dict[str, dict[str, float]], builtin.report["problems"])
    repeats = {
        name: max(3, min(2000, round(plan.level.window / max(info["step"], 1e-6))))
        for name, info in steps.items()
    }
    return _Bench(builtin, tuple(opened[1:]), repeats)


def _rounds(plan: _Plan, bench: _Bench, until: float) -> Scores:
    """Time interleaved rounds; return each slot's score in each round.

    A slot's score is the geometric mean of its seconds per step over the problems,
    so that the ratio of two scores is the geometric mean of their windows' ratios.
    ``"again"`` is the built-in child timed a second time in the same round.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    bench : _Bench
        The prepared children.
    until : float
        The seconds of the deadline kept back; no round starts within them.

    Returns
    -------
    dict of str to list
        Each slot's score per round, ``None`` for a failed one.
    """
    slots = [("built-in", bench.builtin), ("again", bench.builtin)]
    slots += [(child.candidate.name, child) for child in bench.rivals]
    scores: Scores = {name: [] for name, _ in slots}
    generator = random.Random(1)
    for _ in range(plan.level.rounds):
        if plan.deadline.remaining() <= until:
            break
        order = list(slots)
        generator.shuffle(order)
        for name, child in order:
            try:
                scores[name].append(_geomean(bench.time(plan, child)))
            except _ChildError:
                scores[name].append(None)
    return scores


def _validate(
    plan: _Plan, bench: _Bench, winner: _Child, noise: float
) -> selection.Validation:
    """Validate the winner against the built-in child on fresh rounds, by window.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    bench : _Bench
        The prepared children.
    winner : _Child
        The admitted candidate.
    noise : float
        The session's A/A noise.

    Returns
    -------
    solprop.calibration.selection.Validation
        The verdict.
    """
    names = plan.level.problems
    combined: dict[str, list[float | None]] = {name: [] for name in names}
    table: dict[str, list[float | None]] = {name: [] for name in names}
    generator = random.Random(1000)
    for _ in range(plan.level.rounds):
        pair = [(combined, winner), (table, bench.builtin)]
        generator.shuffle(pair)
        for target, child in pair:
            try:
                seconds: list[float | None] = list(bench.time(plan, child))
            except _ChildError:
                seconds = [None] * len(names)
            for name, value in zip(names, seconds, strict=True):
                target[name].append(value)
    return selection.validate(
        {name: (combined[name], table[name]) for name in names},
        noise,
        plan.level.rounds,
    )


def _settings(child: _Child) -> dict[str, object]:
    """Return a child's process settings as it reported them: width and wait.

    Parameters
    ----------
    child : _Child
        A prepared child.

    Returns
    -------
    dict
        ``threads`` and ``wait``.
    """
    process = cast(dict[str, object], child.report["process"])
    return {"threads": process["threads"], "wait": process["wait"]}


def _decide(
    plan: _Plan, bench: _Bench, scores: Scores, noise: float
) -> tuple[dict[str, object], dict[str, object], str]:
    """Choose and validate a candidate; return the settings, evidence and reason.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    bench : _Bench
        The prepared children.
    scores : dict
        Each slot's score per round.
    noise : float
        The session's A/A noise.

    Returns
    -------
    tuple
        The process settings chosen, the evidence, and the reason.
    """
    winner_name, verdicts = selection.choose(
        {child.candidate.name: scores[child.candidate.name] for child in bench.rivals},
        scores["built-in"],
        noise,
        plan.level.rounds,
    )
    evidence: dict[str, object] = {
        "noise": noise,
        "verdicts": {name: verdict.record() for name, verdict in verdicts.items()},
    }
    winner = next((c for c in bench.rivals if c.candidate.name == winner_name), None)
    if winner is not None:
        plan.say(f"validating {winner_name}")
        validation = _validate(plan, bench, winner, noise)
        evidence["validation"] = validation.record()
        if not validation.passed:
            winner = None
    regions: dict[str, object] = {}
    for child in [bench.builtin] + ([winner] if winner is not None else []):
        with contextlib.suppress(_ChildError):
            regions[child.candidate.name] = child.request(
                {"op": "regions", "elements": 1000, "repeats": 200},
                plan.deadline.remaining(),
            )
    evidence["regions"] = regions
    if winner is None:
        return (
            _settings(bench.builtin),
            evidence,
            "the built-in rules kept: no candidate won every round and its validation",
        )
    return (
        _settings(winner),
        evidence,
        f"{winner_name} won every round and its validation",
    )


def _publish(
    store: Store, fingerprint: str, row: Row, allocation: AllocationClass
) -> Path:
    """Publish `row` in a new generation, keeping the other classes' rows.

    A row of the same class from a higher level is kept instead.

    Parameters
    ----------
    store : solprop_native._store.Store
        The user's store.
    fingerprint : str
        The machine and build's key.
    row : solprop_native._store.Row
        The new row.
    allocation : AllocationClass
        Its class.

    Returns
    -------
    pathlib.Path
        The generation file published.
    """
    previous: Generation | None = store.current(fingerprint)
    rows: list[Row] = []
    for kept in () if previous is None else previous.rows:
        try:
            same = (
                _native.allocation.AllocationClass.read(kept.allocation) == allocation
            )
        except ValueError:
            continue
        if not same:
            rows.append(kept)
        elif ORDER.get(kept.level, -1) > ORDER[row.level]:
            row = kept
    rows.append(row)
    return store.publish(
        _native.store.Generation(
            fingerprint=fingerprint,
            name=_native.store.generation_name(),
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rows=tuple(rows),
            wisdom={} if previous is None else previous.wisdom,
        )
    )


def _calibrate_allocation(
    plan: _Plan, cpus: Sequence[int], allocation: AllocationClass, until: float
) -> CalibrationReport:
    """Calibrate one allocation class; publish its row unless inconclusive.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    cpus : sequence of int
        The allocation's CPUs.
    allocation : AllocationClass
        Its class.
    until : float
        The seconds of the deadline kept back for validation and publication.

    Returns
    -------
    CalibrationReport
        The class's report.
    """
    base = CalibrationReport(
        level=plan.level.name,
        status="inconclusive",
        fingerprint=plan.digest,
        allocation=allocation.record(),
    )
    claim = plan.store.claim(plan.digest, allocation.key(), plan.deadline.epoch())
    if claim is None:
        return dataclasses.replace(
            base,
            status="held",
            reason="another process is calibrating this allocation",
            seconds=plan.deadline.elapsed(),
        )
    opened: list[_Child] = []
    try:
        before = _telemetry()
        bench = _start(plan, cpus, allocation, opened)
        plan.say(f"timing interleaved rounds on {allocation.cores} cores")
        scores = _rounds(plan, bench, until)
        noise = selection.noise(scores["built-in"], scores["again"], plan.level.rounds)
        evidence: dict[str, object] = {
            "rounds": plan.level.rounds,
            "problems": bench.builtin.report["problems"],
            "repeats": dict(bench.repeats),
            "scores": scores,
        }
        builtin = _settings(bench.builtin)
        busy = _busy(before, _telemetry(), max(0, (os.cpu_count() or 1) - len(cpus)))
        if noise is None or noise > selection.NOISE_LIMIT or busy:
            plan.store.record_inconclusive(plan.digest, allocation.key())
            return dataclasses.replace(
                base,
                reason=busy
                or (
                    "too few rounds within the budget"
                    if noise is None
                    else f"A/A noise {noise:.3f} above {selection.NOISE_LIMIT}"
                ),
                builtin=builtin,
                evidence=evidence,
                seconds=plan.deadline.elapsed(),
            )
        chosen, decision, reason = _decide(plan, bench, scores, noise)
        evidence |= decision | {"reason": reason}
        row = _native.store.Row(
            allocation=allocation.record(),
            level=plan.level.name,
            process=chosen,
            workspace={},
            fitted_under=chosen,
            evidence=evidence,
        )
        return CalibrationReport(
            level=plan.level.name,
            status="published",
            fingerprint=plan.digest,
            allocation=allocation.record(),
            chosen=chosen,
            builtin=builtin,
            evidence=evidence,
            reason=reason,
            seconds=plan.deadline.elapsed(),
            path=_publish(plan.store, plan.digest, row, allocation),
        )
    except _ChildError as error:
        return dataclasses.replace(
            base, reason=str(error), seconds=plan.deadline.elapsed()
        )
    finally:
        for child in opened:
            child.stop()
        claim.release()


def calibrate(
    level: str = "standard",
    budget: float | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cpus: Sequence[int] | None = None,
    progress: Callable[[str], object] | None = None,
) -> CalibrationReport:
    """Calibrate this machine's native backend for the process's allocation.

    At the standard and thorough levels the nested allocations it can pin children
    to (one socket, one NUMA node, one L3 group) are calibrated too, while the
    deadline allows; each gets its own row (SEP 46 section 6.4).

    Parameters
    ----------
    level : str, optional
        ``"quick"`` (about 20 s), ``"standard"`` (about a minute) or ``"thorough"``
        (about five minutes).
    budget : float or None, optional
        The deadline in seconds, the level's when ``None``.
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.
    cpus : sequence of int or None, optional
        The allocation's CPUs, the process's affinity mask when ``None``.
    progress : callable or None, optional
        Called with a line of text as each stage starts.

    Returns
    -------
    CalibrationReport
        The allocation's own report; the nested allocations' are in its evidence
        (``nested``).

    Raises
    ------
    CalibrationError
        For an unknown level or a budget that is not positive.
    """
    if level not in LEVELS:
        raise CalibrationError(
            f"unknown calibration level {level!r}: quick, standard or thorough"
        )
    chosen = LEVELS[level]
    seconds = chosen.budget if budget is None else float(budget)
    if not seconds > 0:
        raise CalibrationError(f"a calibration's budget must be positive, got {budget}")
    environ = dict(os.environ if environ is None else environ)
    from solprop import calibration  # noqa: PLC0415 - the package imports this module

    found = calibration.choice(environ | {"SOLPROP_CALIBRATION": "use"})
    if found.fingerprint is None:  # pragma: no cover - "use" always fingerprints
        raise CalibrationError("no fingerprint for this machine")
    plan = _Plan(
        level=chosen,
        environ=environ,
        store=_native.store.Store(calibration.store_directory(environ)),
        digest=found.fingerprint.digest,
        deadline=_Deadline(seconds),
        say=progress or (lambda _: None),
    )
    machine = _native.topology.read_topology()
    held = tuple(_process.allowed_cpus() if cpus is None else cpus)
    allocation = _native.allocation.classify(machine, held, _process.cgroup_cpus())
    plan.say(f"{chosen.name} calibration of {allocation.cores} cores")
    reserve = seconds * chosen.reserve
    report = _calibrate_allocation(plan, held, allocation, reserve)
    # A nested allocation is its own row: a whole node too noisy to decide still
    # gives each socket's.
    if chosen.name == "quick" or report.status == "held":
        return report
    reports: list[dict[str, object]] = []
    for subset in nested(machine, held):
        sub = _native.allocation.classify(machine, subset, None)
        if plan.deadline.remaining() < seconds / 3:
            reports.append(
                {
                    "allocation": sub.record(),
                    "status": "skipped",
                    "reason": "too little of the budget left",
                }
            )
            continue
        plan.say(f"nested allocation of {sub.cores} cores")
        inner = _calibrate_allocation(plan, subset, sub, reserve / 2)
        reports.append(
            {
                "allocation": sub.record(),
                "status": inner.status,
                "chosen": dict(inner.chosen),
                "reason": inner.reason,
            }
        )
    return dataclasses.replace(
        report,
        evidence=dict(report.evidence) | {"nested": reports},
        seconds=plan.deadline.elapsed(),
    )


def _native_budget(cpus: Sequence[int]) -> int:
    """Return the built-in rule's width for an allocation of `cpus`, as native reads it.

    The smallest of the CPUs, their physical cores, the fastest kind's cores and the
    control groups' quota (``thread_budget`` without ``SOLPROP_THREADS``).

    Parameters
    ----------
    cpus : sequence of int
        The allocation's CPUs; empty for the whole machine.

    Returns
    -------
    int
        Threads.
    """
    limits = [len(cpus) or (os.cpu_count() or 1)]
    for limit in (
        _process.physical_cores(cpus) if cpus else None,
        _process.top_tier_cores(),
        _process.cgroup_cpus(),
    ):
        if limit is not None:
            limits.append(limit)
    return max(1, min(limits))

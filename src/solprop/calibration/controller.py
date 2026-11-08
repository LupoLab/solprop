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
from solprop.calibration import _native, grain, selection
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
    "ORDER",
    "POLICY",
    "CalibrationReport",
    "Candidate",
    "Level",
    "LevelName",
    "builtin_wait",
    "calibrate",
    "candidates",
    "nested",
]

type LevelName = Literal["quick", "standard", "thorough"]
"""A measuring level's name."""
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
    grain : int
        The most grain candidates timed.
    fulls : bool
        Whether the grain moves the classes' full sizes as well as their thresholds.
    reserve : float
        The share of the budget kept for validation and publication.
    planning : float
        The share of the time left after the waiting and team that the grain stage
        leaves to FFT planning, which follows it; none at the quick level, which keeps
        the plan policy.
    """

    name: LevelName
    budget: float
    rounds: int
    window: float
    problems: tuple[str, ...]
    widths: int
    grain: int
    fulls: bool = False
    reserve: float = 0.2
    planning: float = 0.5


_ALL = ("gnlse", "capillary", "averaged", "radial", "line")
_LARGE = ("gnlse_large", "capillary_large", "radial_large", "line_large")
"""Production-sized problems, which no narrower team or grain move may slow down: a
setting that wins only on small problems loses on large ones (SEP 46 Part II)."""

LEVELS: dict[str, Level] = {
    # Quick: the wait, the team and the coarse grain (medium width, compact team).
    # Quick has no production-sized problem, so it tries no narrower team.
    "quick": Level(
        "quick", 20.0, 3, 0.08, ("gnlse", "averaged", "line"), 0, 4, planning=0.0
    ),
    "standard": Level("standard", 90.0, 5, 0.08, _ALL + _LARGE, 2, 10, reserve=0.15),
    "thorough": Level("thorough", 300.0, 7, 0.3, _ALL + _LARGE, 2, 40, fulls=True),
}
"""The levels: about 20 s, a minute and a half and five minutes (SEP 46 section 6.9)."""

ORDER: dict[str, int] = {"quick": 0, "standard": 1, "thorough": 2}
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
        FFT stage spends its time planning (SEP 46 section 6.3). NumPy's and SciPy's
        BLAS threads go idle at once (:func:`solprop._process.child_environment`),
        so that they do not spin on the CPUs being timed.

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
        return _process.child_environment(environment)


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
        ``total`` and ``steal`` ticks and ``running`` tasks, and each CPU's ticks
        (``cpuN.total``) and busy ticks (``cpuN.busy``, neither idle nor waiting);
        ``None`` where unknown.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and len(fields) > 8:
            ticks = [float(value) for value in fields[1:]]
            if fields[0] == "cpu":
                values["total"] = sum(ticks)
                values["steal"] = ticks[7]
            else:
                values[f"{fields[0]}.total"] = sum(ticks)
                values[f"{fields[0]}.busy"] = sum(ticks) - ticks[3] - ticks[4]
        elif fields and fields[0] == "procs_running":
            values["running"] = float(fields[1])
    return values


def _busy(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    idle: int,
    siblings: Sequence[int] = (),
) -> str:
    """Return why the host was too busy to calibrate, or ``""``.

    Work not belonging to the calibration beyond the CPUs outside its allocation, an
    SMT sibling of its cores more than half busy (another task sharing a core it
    times), or more than a tenth of the time stolen by a hypervisor.

    Parameters
    ----------
    before, after : dict or None
        :func:`_telemetry` before and after the measurements.
    idle : int
        The CPUs outside the allocation.
    siblings : sequence of int, optional
        The CPUs outside the allocation that share a core with it.

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
    for cpu in siblings:
        ticks = after.get(f"cpu{cpu}.total", 0.0) - before.get(f"cpu{cpu}.total", 0.0)
        busy = after.get(f"cpu{cpu}.busy", 0.0) - before.get(f"cpu{cpu}.busy", 0.0)
        if ticks > 0 and busy / ticks > 0.5:
            return (
                f"CPU {cpu}, an SMT sibling of the allocation's cores, was "
                f"{100 * busy / ticks:.0f} % busy with other work"
            )
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
    repeats: Mapping[str, int] = field(default_factory=dict[str, int])


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


def _rounds(
    plan: _Plan, bench: _Bench, until: float
) -> tuple[Scores, dict[str, dict[str, list[float | None]]]]:
    """Time interleaved rounds; return each slot's score and seconds in each round.

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
    tuple
        Each slot's score per round, ``None`` for a failed one; and each slot's
        seconds per step on each problem per round.
    """
    slots = [("built-in", bench.builtin), ("again", bench.builtin)]
    slots += [(child.candidate.name, child) for child in bench.rivals]
    scores: Scores = {name: [] for name, _ in slots}
    names = plan.level.problems
    seconds = {
        slot: {name: list[float | None]() for name in names} for slot, _ in slots
    }
    generator = random.Random(1)
    for _ in range(plan.level.rounds):
        if plan.deadline.remaining() <= until:
            break
        order = list(slots)
        generator.shuffle(order)
        for name, child in order:
            try:
                timed = list(bench.time(plan, child))
            except _ChildError:
                scores[name].append(None)
                for problem in names:
                    seconds[name][problem].append(None)
                continue
            scores[name].append(_geomean(timed))
            for problem, value in zip(names, timed, strict=True):
                seconds[name][problem].append(value)
    return scores, seconds


def _window_ratios(
    candidate: Mapping[str, Sequence[float | None]],
    base: Mapping[str, Sequence[float | None]],
) -> dict[str, list[float | None]]:
    """Return each problem's per-round ratio of a candidate's seconds to a base's.

    Parameters
    ----------
    candidate, base : Mapping[str, sequence of float or None]
        Seconds per round by problem, ``None`` for a failed round.

    Returns
    -------
    dict of str to list
        The ratios, ``None`` where either failed.
    """
    return {
        problem: [
            None if new is None or old is None or old <= 0 else new / old
            for new, old in zip(times, base[problem], strict=True)
        ]
        for problem, times in candidate.items()
    }


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
    plan: _Plan,
    bench: _Bench,
    scores: Scores,
    noise: float,
    seconds: Mapping[str, Mapping[str, Sequence[float | None]]] | None = None,
) -> tuple[_Child, dict[str, object], str]:
    """Choose and validate a candidate; return its child, the evidence and why.

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
    seconds : Mapping or None, optional
        Each slot's seconds per problem per round (:func:`_rounds`), so that no
        problem may be slower under the winner.

    Returns
    -------
    tuple
        The child whose process settings were chosen (the built-in one when none
        replaced them), the evidence, and the reason.
    """
    windows = (
        {}
        if seconds is None
        else {
            child.candidate.name: _window_ratios(
                seconds[child.candidate.name], seconds["built-in"]
            )
            for child in bench.rivals
        }
    )
    winner_name, verdicts = selection.choose(
        {child.candidate.name: scores[child.candidate.name] for child in bench.rivals},
        scores["built-in"],
        noise,
        plan.level.rounds,
        windows,
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
            bench.builtin,
            evidence,
            "the built-in rules kept: no candidate won every round and its validation",
        )
    return winner, evidence, f"{winner_name} won every round and its validation"


@dataclass(frozen=True)
class _Setup:
    """What one window runs under: a workspace record and an FFT planning.

    Attributes
    ----------
    record : Mapping[str, int or bool] or None
        A native execution record, ``None`` for the child's own.
    planning : Mapping[str, object] or None
        An FFT planning, ``None`` for the plan policy.
    """

    record: Mapping[str, int | bool] | None = None
    planning: Mapping[str, object] | None = None


def _compare_round(
    plan: _Plan,
    child: _Child,
    base: _Setup,
    candidates: Sequence[tuple[str, _Setup, tuple[str, ...]]],
    generator: random.Random,
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    """Time one round of candidate setups against a base setup.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan, with each window's repeats.
    child : _Child
        The child whose process settings were chosen.
    base : _Setup
        The base setup.
    candidates : sequence of tuple
        Each candidate's name, setup and the problems it changes.
    generator : random.Random
        Shuffles the round's order.

    Returns
    -------
    tuple
        ``again``, the base's A/A ratio over the problems, and each candidate's ratio
        to it over its problems, ``None`` for a failed window; and each candidate's
        ratio on each of its problems.
    """
    problems = tuple(sorted({name for _, _, names in candidates for name in names}))
    slots = [("base", base, problems), ("again", base, problems), *candidates]
    generator.shuffle(slots)
    timed: dict[str, dict[str, float]] = {}
    failed: set[str] = set()
    for name, setup, names in slots:
        timed[name] = {}
        for problem in names:
            try:
                reply = child.request(
                    {
                        "op": "time",
                        "problem": problem,
                        "repeats": plan.repeats[problem],
                        "record": setup.record,
                        "planning": setup.planning,
                    },
                    plan.deadline.remaining(),
                )
            except _ChildError:
                failed.add(name)
                break
            timed[name][problem] = float(cast(float, reply))
    ratios: dict[str, float | None] = {}
    windows: dict[str, dict[str, float | None]] = {}
    for name, _, names in slots:
        if name == "base":
            continue
        if name in failed or "base" in failed:
            ratios[name] = None
            windows[name] = dict.fromkeys(names)
            continue
        each = {p: timed[name][p] / timed["base"][p] for p in names}
        ratios[name] = _geomean(list(each.values()))
        windows[name] = dict(each)
    return ratios, windows


def _session_of(
    plan: _Plan,
    child: _Child,
    base: _Setup,
    candidates: Sequence[tuple[str, _Setup, tuple[str, ...]]],
    until: float,
    seed: int,
) -> tuple[float | None, dict[str, selection.Verdict]]:
    """Time interleaved rounds of candidate setups; return the noise and verdicts.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan.
    child : _Child
        The child whose process settings were chosen.
    base : _Setup
        The base setup.
    candidates : sequence of tuple
        Each candidate's name, setup and the problems it changes.
    until : float
        The seconds of the deadline kept back; no round starts within them.
    seed : int
        Seeds the rounds' order.

    Returns
    -------
    tuple
        The session's A/A noise (``None`` with too few rounds), and each
        candidate's verdict, empty when the noise is too high to decide.
    """
    generator = random.Random(seed)
    ratios: dict[str, list[float | None]] = {"again": []}
    ratios |= {name: [] for name, _, _ in candidates}
    windows: dict[str, dict[str, list[float | None]]] = {
        name: {problem: [] for problem in names} for name, _, names in candidates
    }
    for _ in range(plan.level.rounds):
        if plan.deadline.remaining() <= until:
            break
        round_ratios, round_windows = _compare_round(
            plan, child, base, candidates, generator
        )
        for name, value in round_ratios.items():
            ratios[name].append(value)
        for name, each in round_windows.items():
            if name in windows:
                for problem, value in each.items():
                    windows[name][problem].append(value)
    ones = [1.0] * len(ratios["again"])
    noise = selection.noise(ones, ratios["again"], plan.level.rounds)
    if noise is None or noise > selection.NOISE_LIMIT:
        return noise, {}
    return noise, {
        name: selection.admit(
            ratios[name], ones, noise, plan.level.rounds, windows[name]
        )
        for name, _, _ in candidates
    }


def _grain_candidates(
    plan: _Plan, child: _Child, sizing: Mapping[str, Mapping[str, object]]
) -> list[tuple[grain.Candidate, tuple[str, ...]]]:
    """Return the grain candidates and the problems each changes.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan, for its level's limits.
    child : _Child
        The child whose process settings were chosen.
    sizing : Mapping[str, Mapping[str, object]]
        Each problem's ``execution_sizing`` under the child's own record.

    Returns
    -------
    list of tuple
        Each candidate, with the problems whose operations (or compact team) it
        changes; a candidate that changes none is left out.
    """
    base = cast(dict[str, int | bool], child.report["execution"])
    capacity = int(cast(int, _settings(child)["threads"]))
    shapes = cast(dict[str, dict[str, list[int]]], child.report["problems"])
    compact = [
        name for name, info in shapes.items() if math.prod(info["shape"][:-1]) == 1
    ]
    found = {name: grain.operations(sizes) for name, sizes in sizing.items()}
    proposed: list[tuple[grain.Candidate, tuple[str, ...]]] = []
    for candidate in grain.proposals(
        base, found, capacity, compact, limit=plan.level.grain, fulls=plan.level.fulls
    ):
        names = tuple(
            name
            for name, operations in found.items()
            if grain.affected(candidate.record(base), base, operations, capacity)
            or (candidate.setting == "compact" and name in compact)
        )
        if names:
            proposed.append((candidate, names))
    return proposed


def _grain(
    plan: _Plan, child: _Child, until: float
) -> tuple[dict[str, int | bool], dict[str, object]]:
    """Fit the operation grain in the chosen process (SEP 46 phase 2).

    The child's operations, as its workspaces size them, give each operation class's
    candidates (:func:`solprop.calibration.grain.proposals`); each is timed on the
    problems it changes, interleaved with the child's own record, and admitted by
    §6.5's rule; the best of each setting are combined and validated together on
    fresh rounds of every problem. A failed validation drops the grain.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan, with each window's repeats.
    child : _Child
        The child whose process settings were chosen.
    until : float
        The seconds of the deadline kept back for publication.

    Returns
    -------
    tuple
        The workspace settings admitted (empty when none), and the evidence.
    """
    try:
        sizing = cast(
            dict[str, dict[str, object]],
            child.request({"op": "sizing", "record": None}, plan.deadline.remaining()),
        )
    except _ChildError as error:
        return {}, {"reason": str(error)}
    proposed = _grain_candidates(plan, child, sizing)
    evidence: dict[str, object] = {"candidates": [c.name for c, _ in proposed]}
    if not proposed:
        return {}, evidence | {"reason": "no candidate changes an operation's width"}
    plan.say(f"timing {len(proposed)} grain candidates")
    base = cast(dict[str, int | bool], child.report["execution"])
    noise, verdicts = _session_of(
        plan,
        child,
        _Setup(),
        [(c.name, _Setup(record=c.record(base)), names) for c, names in proposed],
        until,
        2000,
    )
    if noise is None or not verdicts:
        return {}, evidence | {"reason": "inconclusive grain noise", "noise": noise}
    evidence |= {
        "noise": noise,
        "verdicts": {name: verdict.record() for name, verdict in verdicts.items()},
    }
    best = grain.best([candidate for candidate, _ in proposed], verdicts)
    if not best:
        return {}, evidence | {"reason": "no grain candidate won every round"}
    changes: dict[str, int | bool] = {}
    for candidate in best:
        changes |= candidate.changes
    plan.say(f"validating {len(best)} grain settings")
    validation = _validate_setups(
        plan, child, _Setup(record={**base, **changes}), _Setup(), noise
    )
    evidence["validation"] = validation.record()
    if not validation.passed:
        return {}, evidence | {"reason": "the combined grain failed its validation"}
    admitted = ", ".join(candidate.name for candidate in best)
    return changes, evidence | {"reason": f"admitted: {admitted}"}


def _validate_setups(
    plan: _Plan, child: _Child, combined: _Setup, table: _Setup, noise: float
) -> selection.Validation:
    """Validate a combined setup against the base on fresh rounds of every window.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan, with each window's repeats.
    child : _Child
        The child.
    combined : _Setup
        The admitted settings together.
    table : _Setup
        The base they replace.
    noise : float
        The session's A/A noise.

    Returns
    -------
    solprop.calibration.selection.Validation
        The verdict.
    """
    names = plan.level.problems
    times: dict[str, dict[str, list[float | None]]] = {
        "combined": {name: [] for name in names},
        "table": {name: [] for name in names},
    }
    generator = random.Random(3000)
    for _ in range(plan.level.rounds):
        pair = [("combined", combined), ("table", table)]
        generator.shuffle(pair)
        for which, setup in pair:
            for name in names:
                try:
                    reply = child.request(
                        {
                            "op": "time",
                            "problem": name,
                            "repeats": plan.repeats[name],
                            "record": setup.record,
                            "planning": setup.planning,
                        },
                        plan.deadline.remaining(),
                    )
                    times[which][name].append(float(cast(float, reply)))
                except _ChildError:
                    times[which][name].append(None)
    return selection.validate(
        {name: (times["combined"][name], times["table"][name]) for name in names},
        noise,
        plan.level.rounds,
    )


POLICY: dict[str, str] = {
    "rows": "measure",
    "threaded_rows": "estimate",
    "planes": "measure",
}
"""The plan policy's effort for each group (``fftw_seed_effort``)."""


def _planning_candidates(
    level: Level, measured: Mapping[str, object], patient: Mapping[str, object] | None
) -> list[tuple[str, str, dict[str, object]]]:
    """Return each group's candidate plannings.

    Parameters
    ----------
    level : Level
        The level: patient plans only at thorough, and only for serial rows.
    measured, patient : Mapping[str, object]
        Captures at measure and at patient effort (``patient`` ``None`` below
        thorough): ``wisdom`` and ``groups`` by plan key.

    Returns
    -------
    list of tuple
        Each candidate's name, group and planning: the policy's efforts with one
        group's changed, and the wisdom of every group it plans beyond estimate.
    """
    groups = cast(dict[str, str], measured["groups"])
    wisdom = cast(dict[str, str], measured["wisdom"])

    def planning(
        efforts: Mapping[str, str], keys: Mapping[str, str]
    ) -> dict[str, object]:
        """Return a profile planning with `efforts` and the wisdom of `keys` they use.

        Parameters
        ----------
        efforts : Mapping[str, str]
            Each group's effort.
        keys : Mapping[str, str]
            Wisdom by plan key; that of a group planned at estimate is left out.

        Returns
        -------
        dict
            The planning.
        """
        return {
            "mode": "profile",
            "efforts": dict(efforts),
            "wisdom": {
                key: text
                for key, text in keys.items()
                if efforts[groups.get(key, "rows")] != "estimate"
            },
        }

    present = set(groups.values())
    found: list[tuple[str, str, dict[str, object]]] = []
    if "rows" in present:
        found.append(
            ("rows estimate", "rows", planning(POLICY | {"rows": "estimate"}, wisdom))
        )
        if patient is not None and level.fulls:
            found.append(
                (
                    "rows patient",
                    "rows",
                    planning(
                        POLICY | {"rows": "patient"},
                        wisdom | cast(dict[str, str], patient["wisdom"]),
                    ),
                )
            )
    if "threaded_rows" in present:
        found.append(
            (
                "threaded_rows measure",
                "threaded_rows",
                planning(POLICY | {"threaded_rows": "measure"}, wisdom),
            )
        )
    if "planes" in present:
        found.append(
            (
                "planes estimate",
                "planes",
                planning(POLICY | {"planes": "estimate"}, wisdom),
            )
        )
    return found


_CAPTURE_SHARE = 0.5
"""The share of the FFT stage's time left that each capture may take."""

_CAPTURE_MARGIN = 30.0
"""Seconds past its budget a capture's child may take before it counts as failed:
FFTW's time limit is per plan, and checked between its trials."""


def _planning(
    plan: _Plan, child: _Child, record: Mapping[str, int | bool], until: float
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    """Choose each FFT plan group's effort in the chosen setup (SEP 46 phase 3).

    The child plans every transform alone at measure (and at patient, thorough),
    from no wisdom within the level's time limit; the policy's efforts with that
    wisdom are the base, and each group's other effort a candidate, timed in the
    step on the problems holding that group's transforms and admitted by §6.5's rule;
    the winners are validated together. Quick leaves the plan policy.

    Parameters
    ----------
    plan : _Plan
        The calibration's plan, with each window's repeats.
    child : _Child
        The child whose process settings were chosen.
    record : Mapping[str, int or bool]
        The workspace record chosen, under which the transforms' threads are set.
    until : float
        The seconds of the deadline kept back for publication.

    Returns
    -------
    tuple
        The row's planning (``efforts``), the wisdom by plan key, and the evidence.
    """
    if plan.level.name == "quick":
        return {}, {}, {"reason": "the plan policy (quick)"}
    # Most measured plans take milliseconds; the limit bounds the odd slow one.
    limit = 20.0 if plan.level.fulls else 0.25

    def capture(effort: str, share: float) -> dict[str, object]:
        """Capture every problem's plans at `effort` within `share` of the time left.

        Each capture takes its share of the stage's time, so that the timing in the
        step still has the rest (SEP 46 section 6.7).

        Parameters
        ----------
        effort : str
            ``"measure"`` or ``"patient"``.
        share : float
            The share of the stage's time left that the capture may take.

        Returns
        -------
        dict
            The child's capture (:meth:`solprop.calibration.child._Session.capture`).
        """
        budget = max(0.0, plan.deadline.remaining() - until) * share
        return cast(
            dict[str, object],
            child.request(
                {
                    "op": "capture",
                    "effort": effort,
                    "limit": limit,
                    "budget": budget,
                    "record": dict(record),
                },
                min(plan.deadline.remaining(), budget + _CAPTURE_MARGIN),
            ),
        )

    try:
        plan.say("planning the transforms at measure")
        measured = capture("measure", _CAPTURE_SHARE)
        # A capture cut short keeps what it planned: the problems it skipped are not
        # timed, and plan as a run without a profile does.
        if not measured["groups"]:
            return {}, {}, {"reason": "no transform planned within the stage's share"}
        patient = None
        if plan.level.fulls:
            plan.say("planning the serial transforms at patient")
            patient = capture("patient", _CAPTURE_SHARE)
            if patient.get("skipped"):
                patient = None  # incomplete: patient is not a candidate
    except _ChildError as error:
        return {}, {}, {"reason": str(error)}
    candidates = _planning_candidates(plan.level, measured, patient)
    groups = cast(dict[str, str], measured["groups"])
    contains = cast(dict[str, list[str]], measured["problems"])
    evidence: dict[str, object] = {
        "plans": len(groups),
        "skipped": list(cast(list[str], measured.get("skipped", []))),
        "planning_seconds": measured["seconds"],
        "candidates": [name for name, _, _ in candidates],
    }
    timed = [
        (
            name,
            _Setup(record=record, planning=planning),
            tuple(
                p
                for p, keys in contains.items()
                if any(groups[k] == group for k in keys)
            ),
        )
        for name, group, planning in candidates
    ]
    timed = [item for item in timed if item[2]]
    chosen = dict(POLICY)
    wisdom: dict[str, str] = {}
    # Every candidate, with all the wisdom it would publish, is compared with the
    # built-in plan policy as a run without a profile plans: stored wisdom planned
    # alone can run slower in the step than plans measured in place (on 16 cores of
    # an AMD EPYC 9B45 one case ran 17 % slower), so it is never assumed neutral.
    builtin = _Setup(record=record, planning=None)
    if timed:
        plan.say(f"timing {len(timed)} FFT planning candidates")
        noise, verdicts = _session_of(plan, child, builtin, timed, until, 4000)
        evidence["noise"] = noise
        evidence["verdicts"] = {
            name: verdict.record() for name, verdict in verdicts.items()
        }
        admitted = {
            group: (name, planning)
            for name, group, planning in candidates
            if name in verdicts and verdicts[name].admitted
        }
        if admitted and noise is not None:
            for _, planning in admitted.values():
                efforts = cast(dict[str, str], planning["efforts"])
                chosen |= {g: e for g, e in efforts.items() if e != POLICY[g]}
            combined = {
                "mode": "profile",
                "efforts": chosen,
                "wisdom": {
                    key: text
                    for key, text in (
                        cast(dict[str, str], measured["wisdom"])
                        | (cast(dict[str, str], patient["wisdom"]) if patient else {})
                    ).items()
                    if chosen[groups.get(key, "rows")] != "estimate"
                    and (chosen[groups.get(key, "rows")] != "patient" or patient)
                },
            }
            validation = _validate_setups(
                plan, child, _Setup(record=record, planning=combined), builtin, noise
            )
            evidence["validation"] = validation.record()
            if validation.passed:
                wisdom = cast(dict[str, str], combined["wisdom"])
            else:
                chosen = dict(POLICY)
    if chosen == POLICY:
        evidence["reason"] = (
            "the plan policy: no effort won every round and its validation"
        )
        return {}, {}, evidence
    evidence["reason"] = "admitted: " + ", ".join(
        f"{g} {e}" for g, e in chosen.items() if e != POLICY[g]
    )
    return {"efforts": chosen}, wisdom, evidence


def _publish(
    store: Store,
    fingerprint: str,
    row: Row,
    allocation: AllocationClass,
    wisdom: Mapping[str, str] | None = None,
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
    wisdom : Mapping[str, str] or None, optional
        Its plans' wisdom by plan key, added to the generation's.

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
            # The plans' wisdom accumulates: another allocation's plans keep theirs.
            wisdom={**({} if previous is None else previous.wisdom), **(wisdom or {})},
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
        plan = dataclasses.replace(plan, repeats=bench.repeats)
        plan.say(f"timing interleaved rounds on {allocation.cores} cores")
        scores, seconds = _rounds(plan, bench, until)
        noise = selection.noise(scores["built-in"], scores["again"], plan.level.rounds)
        evidence: dict[str, object] = {
            "rounds": plan.level.rounds,
            "problems": bench.builtin.report["problems"],
            "repeats": dict(bench.repeats),
            "scores": scores,
        }
        builtin = _settings(bench.builtin)
        busy = _busy(
            before,
            _telemetry(),
            max(0, (os.cpu_count() or 1) - len(cpus)),
            _siblings(cpus),
        )
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
        final, decision, reason = _decide(plan, bench, scores, noise, seconds)
        chosen = _settings(final)
        # The grain stage leaves FFT planning its share of the time left, so that a
        # long grain stage cannot leave the planning stage none (SEP 46 section 6.9).
        left = max(0.0, plan.deadline.remaining() - until)
        workspace, grain_evidence = _grain(
            plan, final, until + left * plan.level.planning
        )
        table = cast(dict[str, int | bool], final.report["execution"])
        planning, wisdom, planning_evidence = _planning(
            plan, final, {**table, **workspace}, until
        )
        reason = (
            f"{reason}; grain: {grain_evidence.get('reason')}; "
            f"FFT planning: {planning_evidence.get('reason')}"
        )
        evidence |= decision | {
            "reason": reason,
            "grain": grain_evidence,
            "planning": planning_evidence,
        }
        row = _native.store.Row(
            allocation=allocation.record(),
            level=plan.level.name,
            process=chosen,
            workspace=workspace,
            fitted_under=chosen,
            evidence=evidence,
            planning=planning,
        )
        return CalibrationReport(
            level=plan.level.name,
            status="published",
            fingerprint=plan.digest,
            allocation=allocation.record(),
            chosen=chosen | workspace,
            builtin=builtin | {name: table[name] for name in workspace},
            evidence=evidence,
            reason=reason,
            seconds=plan.deadline.elapsed(),
            path=_publish(plan.store, plan.digest, row, allocation, wisdom),
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
        ``"quick"`` (about 20 s), ``"standard"`` (about 90 s) or ``"thorough"``
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


def _siblings(cpus: Sequence[int]) -> list[int]:
    """Return the CPUs outside `cpus` that share a physical core with one of them.

    Parameters
    ----------
    cpus : sequence of int
        The allocation's CPUs.

    Returns
    -------
    list of int
        Their SMT siblings, ascending.
    """
    machine = _native.topology.read_topology()
    cores = {machine.core[cpu] for cpu in cpus if cpu in machine.core}
    held = set(cpus)
    return sorted(
        cpu for cpu in machine.cpus if machine.core[cpu] in cores and cpu not in held
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

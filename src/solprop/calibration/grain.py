# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Operation-grain candidates from the work sizes the calibration problems run (SEP 46).

Each CPU operation class threads by its work: serially below its threshold, on the
medium width below its full size, on its pool's capacity from there
(``_execution.ExecutionRecord``). A workspace's ``execution_sizing`` records every
operation it runs, with its class, work and width. A threshold or full size matters
only where it falls between two work sizes the problems actually run, so the
candidates of a class are its one-step moves past the nearest of them: thread one
more size, serialise one more, widen one more to the full pool, narrow one more to
the medium width. At most four a class (section 6.5); a candidate that changes no
operation's width is left out. The medium width and the compact team take a short
ladder of their own. Each candidate is then timed on the problems it changes, and
§6.5's rule decides (:mod:`solprop.calibration.selection`).

Every function here is pure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solprop.calibration.selection import Verdict

__all__ = [
    "TIMED",
    "Candidate",
    "Operation",
    "affected",
    "best",
    "operations",
    "proposals",
    "width",
]

type Record = Mapping[str, int | bool]
"""A native execution record (``ExecutionRecord.native()``)."""

TIMED = ("stage", "fft", "response", "raman", "copy", "dense", "modal")
"""The classes a trial-step window runs, in the order their candidates are tried.

The exponential Adams trial's passes and a coupled flow's matrix functions run in
no calibration window: their thresholds keep the built-in table's.
"""


@dataclass(frozen=True)
class Operation:
    """One operation a problem runs, as its workspace's sizing records it.

    Attributes
    ----------
    pool : str
        ``"kernels"``, ``"fft"``, ``"raman_fft"`` or ``"raman_tail_fft"``.
    kind : str
        Its class: ``"stage"``, ``"response"``, ``"dense"``, ``"fft"``, ``"raman"``,
        ``"copy"``, ``"flow"``, ``"modal"`` or ``"adams"``.
    elements : int
        Its work.
    samples : int
        A response's fine samples per column; zero otherwise.
    """

    pool: str
    kind: str
    elements: int
    samples: int = 0


def operations(sizing: Mapping[str, object]) -> list[Operation]:
    """Return the operations a workspace's ``execution_sizing`` records.

    Parameters
    ----------
    sizing : Mapping[str, object]
        The sizing, keyed ``pool.operation.elements[.samples.N]``.

    Returns
    -------
    list of Operation
        One per key, in key order; keys of another form are left out.

    Examples
    --------
    >>> found = operations({"kernels.stage.4096": {}, "fft.fft.66.samples.33": {}})
    >>> [(operation.kind, operation.elements, operation.samples) for operation in found]
    [('fft', 66, 33), ('stage', 4096, 0)]
    """
    found: list[Operation] = []
    for key in sorted(sizing):
        parts = key.split(".")
        if len(parts) == 3 and parts[2].isdigit():
            found.append(Operation(parts[0], parts[1], int(parts[2])))
        elif len(parts) == 5 and parts[3] == "samples" and parts[2].isdigit():
            found.append(Operation(parts[0], parts[1], int(parts[2]), int(parts[4])))
    return found


def width(record: Record, operation: Operation, capacity: int) -> int:
    """Return the width `record` gives `operation` on a pool of `capacity` threads.

    The native rule (``operation_threads``), without the compact cap, the stable team
    or an explicit pool: one below the threshold, the full pool for a long or wide
    response, the medium width below the full size, the full pool from it.

    Parameters
    ----------
    record : Mapping[str, int or bool]
        A native execution record.
    operation : Operation
        The operation.
    capacity : int
        The pool's width.

    Returns
    -------
    int
        Threads.

    Examples
    --------
    >>> table = {"stage_threshold": 100, "stage_full": 1000, "medium_threads": 4}
    >>> [width(table, Operation("kernels", "stage", n), 18) for n in (50, 500, 5000)]
    [1, 4, 18]
    """
    threshold = int(record[f"{operation.kind}_threshold"])
    full = int(record[f"{operation.kind}_full"])
    if operation.elements < threshold:
        return 1
    if (
        operation.kind == "response"
        and operation.samples
        and (
            operation.samples >= int(record["response_full_samples"])
            or operation.elements // operation.samples
            >= int(record["response_full_columns"])
        )
    ):
        return capacity
    if operation.elements < full:
        return min(capacity, int(record["medium_threads"]))
    return capacity


def affected(
    record: Record, base: Record, found: Iterable[Operation], capacity: int
) -> bool:
    """Return whether `record` gives any of `found` another width than `base` does.

    Parameters
    ----------
    record, base : Mapping[str, int or bool]
        Two native execution records.
    found : iterable of Operation
        A problem's operations.
    capacity : int
        The pool's width.

    Returns
    -------
    bool
        Whether timing the problem can tell the two apart.
    """
    return any(
        width(record, operation, capacity) != width(base, operation, capacity)
        for operation in found
    )


@dataclass(frozen=True)
class Candidate:
    """One grain candidate: a setting's group and the record's changed fields.

    Attributes
    ----------
    setting : str
        What it moves: an operation class, ``"medium"`` or ``"compact"``.
    name : str
        A short description, such as ``"stage threshold 2048"``.
    changes : Mapping[str, int]
        The fields it sets over the base record.
    """

    setting: str
    name: str
    changes: Mapping[str, int]

    def record(self, base: Record) -> dict[str, int | bool]:
        """Return `base` with this candidate's changes.

        Parameters
        ----------
        base : Mapping[str, int or bool]
            A native execution record.

        Returns
        -------
        dict
            The candidate's record.
        """
        return {**base, **self.changes}


def _class_moves(
    kind: str, sizes: list[int], record: Record, *, fulls: bool
) -> list[Candidate]:
    """Return a class's one-step moves past its nearest work sizes.

    Parameters
    ----------
    kind : str
        The operation class.
    sizes : list of int
        Its work sizes in the problems, ascending.
    record : Mapping[str, int or bool]
        The base record.
    fulls : bool
        Whether to move its full size as well as its threshold.

    Returns
    -------
    list of Candidate
        At most four.
    """
    threshold = int(record[f"{kind}_threshold"])
    full = int(record[f"{kind}_full"])
    serial = [size for size in sizes if size < threshold]
    threaded = [size for size in sizes if size >= threshold]
    medium = [size for size in threaded if size < full]
    whole = [size for size in threaded if size >= full]
    moves: list[tuple[str, int, int]] = []
    if serial:
        moves.append(("threshold", max(serial), max(full, *serial)))
    if threaded:
        lowered = min(threaded) + 1
        moves.append(("threshold", lowered, max(full, lowered)))
    if fulls and medium:
        moves.append(("full", threshold, max(medium)))
    if fulls and whole and min(whole) + 1 > threshold:
        moves.append(("full", threshold, min(whole) + 1))
    # Each move puts a different size on the other side of a boundary, so no two are
    # alike.
    return [
        Candidate(
            kind,
            f"{kind} {field} {new_threshold if field == 'threshold' else new_full}",
            {f"{kind}_threshold": new_threshold, f"{kind}_full": new_full},
        )
        for field, new_threshold, new_full in moves
    ]


def proposals(
    record: Record,
    problems: Mapping[str, list[Operation]],
    capacity: int,
    compact: Iterable[str] = (),
    *,
    limit: int | None = None,
    fulls: bool = True,
) -> list[Candidate]:
    """Return the grain candidates for problems running `problems`' operations.

    Parameters
    ----------
    record : Mapping[str, int or bool]
        The base record: the process's, under which the problems were sized.
    problems : Mapping[str, list of Operation]
        Each problem's operations.
    capacity : int
        The pool's width.
    compact : iterable of str
        The problems that run on a compact team.
    limit : int or None, optional
        The most candidates; the medium widths and compact teams come first, then
        the classes in :data:`TIMED` order.
    fulls : bool, optional
        Whether to move the classes' full sizes as well as their thresholds.

    Returns
    -------
    list of Candidate
        The medium widths, the compact teams, then each timed class's moves; each
        changes some operation's width (or a compact problem's team).

    Examples
    --------
    >>> base = {"stage_threshold": 100, "stage_full": 1000, "medium_threads": 4}
    >>> found = proposals(base, {"p": [Operation("kernels", "stage", 500)]}, 18)
    >>> [candidate.name for candidate in found]
    ['medium 2', 'medium 8', 'stage threshold 501', 'stage full 500']
    """
    sizes: dict[str, set[int]] = {}
    for found in problems.values():
        for operation in found:
            sizes.setdefault(operation.kind, set()).add(operation.elements)
    candidates: list[Candidate] = []
    medium = int(record["medium_threads"])
    for value in (2, 8):
        if value != medium and value < capacity:
            candidate = Candidate(
                "medium", f"medium {value}", {"medium_threads": value}
            )
            if any(
                affected(candidate.record(record), record, found, capacity)
                for found in problems.values()
            ):
                candidates.append(candidate)
    teams = list(compact)
    current = int(record.get("compact_threads", capacity))
    if teams:
        for value in (4, 16):
            if (
                value != current
                and value <= capacity
                and min(value, capacity) != min(current, capacity)
            ):
                candidates.append(
                    Candidate("compact", f"compact {value}", {"compact_threads": value})
                )
    for kind in TIMED:
        if kind in sizes and f"{kind}_threshold" in record:
            candidates += _class_moves(kind, sorted(sizes[kind]), record, fulls=fulls)
    return candidates if limit is None else candidates[:limit]


def best(
    candidates: Iterable[Candidate], verdicts: Mapping[str, Verdict]
) -> list[Candidate]:
    """Return each setting's admitted candidate with the lowest score.

    Parameters
    ----------
    candidates : iterable of Candidate
        The candidates timed.
    verdicts : Mapping[str, Verdict]
        Each one's verdict, by name.

    Returns
    -------
    list of Candidate
        At most one a setting, in the order the settings first appear.
    """
    chosen: dict[str, tuple[float, Candidate]] = {}
    for candidate in candidates:
        verdict = verdicts[candidate.name]
        if not verdict.admitted or verdict.score is None:
            continue
        held = chosen.get(candidate.setting)
        if held is None or verdict.score < held[0]:
            chosen[candidate.setting] = (verdict.score, candidate)
    return [candidate for _, candidate in chosen.values()]

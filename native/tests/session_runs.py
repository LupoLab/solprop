# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Drive a native session or a native group to its end and keep what it observed.

The group tests compare each member of a native group session
(:class:`solprop_native.group.NativeGroupSession`, started through solprop's
:func:`~solprop.backends.sessions.start_group`) with the same member run alone through
the native session (:func:`~solprop.backends.sessions.start_session`): the oracle that
replaced the transitional lockstep batch in solprop SEP 47's phase 5.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from solprop.backends.events import ObservationReady, join_attempts
from solprop.backends.export import ExportRecipe
from solprop.backends.session import Start
from solprop.backends.sessions import start_group, start_session

IDENTITY = ExportRecipe(power=False, optics=False)
"""The state as the session holds it, in the internal FFT convention."""


class _Recorder:
    """Keep one trajectory's observations, each with the attempts and rows since the
    previous one, assembled from every event's chunk (exactly-once delivery)."""

    def __init__(self) -> None:
        self.planes: list[dict[str, Any]] = []
        self._chunks: list[tuple[Any, Any]] = []

    def take(self, export: Any, events: Sequence[Any]) -> None:
        """Record a batch's events, exporting each observation's field."""
        for event in events:
            history = getattr(event, "history", None)
            if history is not None:
                self._chunks.append((history, getattr(event, "step_stats", None)))
            if not isinstance(event, ObservationReady):
                continue
            history, rows = join_attempts(self._chunks)
            self._chunks = []
            self.planes.append(
                {
                    "z": float(event.z),
                    "attempts": {
                        "z": np.asarray(history.z).copy(),
                        "dz": np.asarray(history.dz).copy(),
                        "accepted": np.asarray(history.accepted).copy(),
                        "error_norm": np.asarray(history.error_norm).copy(),
                    },
                    "field": np.array(export(event.ticket, IDENTITY)),
                    "stats": {
                        name: np.asarray(value) for name, value in event.stats.items()
                    },
                    "row_z": np.asarray(rows.z).copy(),
                    "rows": {
                        name: np.asarray(value).copy()
                        for name, value in rows.values.items()
                    },
                }
            )


class Finished(NamedTuple):
    """What one trajectory's session observed, and its execution record."""

    planes: list[dict[str, Any]]
    """Each observation, as :func:`alone` returns them."""
    record: Mapping[str, object]
    """The session's :meth:`~solprop.backends.session.RunSession.execution_record`,
    read before it closed."""


def finish(
    owner: Any,
    problem: Any,
    request: Any,
    start: Start | None = None,
    *,
    owns: bool = True,
) -> Finished:
    """Run one trajectory through its backend's session to its end.

    Parameters
    ----------
    owner : solprop.backends.protocol.Backend
        The backend, native or the reference, built for `problem` and `request`.
    problem, request
        The translated problem and its run's request.
    start : solprop.backends.session.Start or None, optional
        Where it starts; the problem's entrance by default.
    owns : bool, optional
        Whether the session closes `owner`; a caller that reads the backend's
        metadata or ledger afterwards passes ``False`` and closes it.

    Returns
    -------
    Finished
        Its observations, the launch plane first, and its execution record.
    """
    session = start_session(
        owner,
        problem,
        request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
    )
    recorder = _Recorder()
    try:
        while not session.finished:
            batch = session.advance()
            recorder.take(session.export, batch.events)
            session.acknowledge(batch.receipt)
        record = dict(session.execution_record())
    finally:
        session.close()
    return Finished(recorder.planes, record)


def alone(owner: Any, problem: Any, request: Any, start: Start) -> list[dict[str, Any]]:
    """Run one member alone through the native session; return its observations.

    The session owns `owner` and closes it.
    """
    return finish(owner, problem, request, start).planes


def steps(planes: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Return the accepted and the rejected attempts of a recorded run."""
    accepted = np.concatenate([plane["attempts"]["accepted"] for plane in planes])
    return int(np.count_nonzero(accepted)), int(
        accepted.size - np.count_nonzero(accepted)
    )


def grouped(
    owners: Sequence[Any],
    problems: Sequence[Any],
    requests: Sequence[Any],
    starts: Sequence[Start],
) -> list[list[dict[str, Any]]]:
    """Run members as one native group; return each member's observations.

    The group owns `owners` and closes them.
    """
    group = start_group(owners, problems, requests, starts)
    recorders = [_Recorder() for _ in range(group.n_members)]
    try:
        assert type(group).__name__ == "NativeGroupSession"
        while not group.finished:
            batch = group.advance()
            for index, events in enumerate(batch.members):
                recorders[index].take(functools.partial(group.export, index), events)
            group.acknowledge(batch.receipt)
    finally:
        group.close()
    return [recorder.planes for recorder in recorders]


def one_thread() -> bool:
    """Return whether the native backend runs one thread, as the default lane sets it.

    The members of a group are bitwise their lone sessions there, where the
    reductions' order is fixed.
    """
    import os

    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


def assert_same(left: Any, right: Any, what: str = "") -> None:
    """Assert two recorded values agree bitwise, nested dicts and lists included."""
    if isinstance(left, dict):
        mine, theirs = cast("dict[str, Any]", left), cast("dict[str, Any]", right)
        assert mine.keys() == theirs.keys(), what
        for key in mine:
            assert_same(mine[key], theirs[key], f"{what}.{key}")
        return
    if isinstance(left, list):
        mine_list, theirs_list = cast("list[Any]", left), cast("list[Any]", right)
        assert len(mine_list) == len(theirs_list), what
        for index, (one, other) in enumerate(zip(mine_list, theirs_list, strict=True)):
            assert_same(one, other, f"{what}[{index}]")
        return
    assert np.asarray(left).tobytes() == np.asarray(right).tobytes(), what

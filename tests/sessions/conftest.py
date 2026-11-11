# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures of the lifecycle conformance suite: each executor, and sessions that close.

``executor`` runs a test once per session implementation (``sessions.executors``);
``scripted`` is the scripted executor alone, for the outcomes only a script can choose.
``start`` and ``start_scripted`` start sessions of them and close every one at
teardown, so a failing test leaks no backend.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from sessions.executors import EXECUTORS, Executor, Started, entrance
from solprop.backends.session import RunSession
from solprop.translate.problem import PropagationProblem, Translation
from solprop.translate.run_request import RunRequest

Starter = Callable[..., Started]


def _available(name: str) -> Executor:
    """Return an executor, skipping a test whose executor is not installed.

    Parameters
    ----------
    name : str
        The executor's name.

    Returns
    -------
    sessions.executors.Executor
        The executor.
    """
    chosen = EXECUTORS[name]
    if chosen.requires is not None:
        pytest.importorskip(
            chosen.requires, reason=f"{chosen.requires} is not installed"
        )
    if not chosen.available:
        pytest.skip(f"the native session refuses the {name} executor on a device build")
    return chosen


@pytest.fixture(params=sorted(EXECUTORS))
def executor(request: pytest.FixtureRequest) -> Executor:
    """Return each executor in turn."""
    return _available(request.param)


@pytest.fixture(params=["scripted", "scripted-host"])
def scripted(request: pytest.FixtureRequest) -> Executor:
    """Return each scripted executor: on the bridge's driver and on the reference's."""
    return _available(request.param)


def _starter(chosen: Executor) -> Iterator[Starter]:
    """Yield a function that starts sessions of an executor; close them afterwards.

    Parameters
    ----------
    chosen : sessions.executors.Executor
        The executor.

    Yields
    ------
    callable
        ``(problem, request=None, *, stop=None, listener=None, **script) -> Started``,
        where `problem` is a translation, whose request is used unless one is given,
        or a bare problem beside its request.
    """
    opened: list[RunSession] = []

    def start(
        problem: PropagationProblem | Translation,
        request: RunRequest | None = None,
        **controls: Any,
    ) -> Started:
        # A translation brings its own request, which an explicit one replaces.
        if isinstance(problem, Translation):
            problem, translated = problem
            request = translated if request is None else request
        if request is None:
            msg = "a bare problem starts with its run's request"
            raise TypeError(msg)
        started = chosen.start(problem, request, entrance(problem), **controls)
        opened.append(started.session)
        return started

    yield start
    for session in opened:
        session.close()


@pytest.fixture
def start(executor: Executor) -> Iterator[Starter]:
    """Start sessions of the executor under test."""
    yield from _starter(executor)


@pytest.fixture
def start_scripted(scripted: Executor) -> Iterator[Starter]:
    """Start sessions of the scripted executor."""
    yield from _starter(scripted)

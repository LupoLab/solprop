# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's refusals around its session, dense output, probe and engines.

What a caller can misuse once the session owns the schedule (SEP 52): a backend closed
under an open session, a start whose plane index lies behind it, a probe's start of
the wrong shape, a dense output asked for before it holds a step, an engine's stages
before a trial; and the engines' own members of the protocol (``reading``, the
low-storage ``extend``), which the session does not call.
"""

import numpy as np
import pytest

from builders import load_reference
from sessions.executors import entrance, low_storage, soliton, window
from solprop.backends.events import Outcome, Terminal
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.dense import AdamsDense, RungeKuttaDense
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stepper.multistep import Multistep


def _backend(translation):
    problem, request = translation
    return problem, request, NumpyRefBackend(problem, request, grid=window())


def test_a_backend_closed_under_its_session_ends_the_run_failed():
    problem, request, backend = _backend(soliton())
    session = backend.prepare(problem, request).start(entrance(problem), owns=False)
    launch = session.advance()
    session.acknowledge(launch.receipt)
    backend.close()
    batch = session.advance()
    (terminal,) = batch.events
    assert isinstance(terminal, Terminal)
    assert terminal.outcome is Outcome.FAILED
    assert isinstance(terminal.error, BackendError)
    assert "closed" in str(terminal.error)
    session.close()


def test_a_start_whose_plane_lies_behind_it_is_refused():
    problem, request, backend = _backend(soliton())
    try:
        # The state at the second plane, with the first plane named as the next.
        z = float(request.stepper.save_planes[1])
        start = Start(problem.initial.field0, z, plane_index=0, plane_pending=False)
        session = backend.prepare(problem, request).start(start, owns=False)
        (terminal,) = session.advance().events
        assert isinstance(terminal, Terminal)
        assert isinstance(terminal.error, BackendError)
        assert "cannot propagate backwards" in str(terminal.error)
        session.close()
    finally:
        backend.close()


def test_a_probe_refuses_a_start_of_another_shape():
    problem, _request, backend = _backend(soliton())
    try:
        field = np.zeros((1, 1, problem.shape[-1] + 2), dtype=complex)
        with pytest.raises(BackendError, match="initial state has shape"):
            backend.timing_probe(Start(field, 0.0))
    finally:
        backend.close()


@pytest.mark.parametrize("integrator", [None, Multistep()], ids=["rk", "adams"])
def test_a_dense_output_refuses_what_it_does_not_hold(integrator):
    _problem, _request, backend = _backend(
        soliton(integrator=integrator, interpolate=True)
    )
    try:
        load_reference(backend)
        dense = backend.method(None).dense
        assert isinstance(dense, AdamsDense if integrator else RungeKuttaDense)
        # No accepted step yet: nothing to interpolate, and nothing held to restore.
        with pytest.raises(BackendError, match=r"accepted .*step is held"):
            dense.observe(0.001)
        with pytest.raises(BackendError, match="no dense observation holds"):
            dense.restore()
    finally:
        backend.close()


def test_an_engine_has_no_stages_before_a_trial():
    _problem, request, backend = _backend(soliton())
    try:
        engine = NumpyRungeKutta(backend.workspace, request.stepper.tableau)
        with pytest.raises(
            BackendError, match="take a trial before reading its stages"
        ):
            engine.stages  # noqa: B018 - the property's refusal is the subject
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["runge_kutta", "adams", "low_storage"])
def test_each_engine_reads_the_committed_state_as_its_workspace_does(kind):
    """The engine protocol's ``reading`` is the workspace's (SEP 52 section 5.2)."""
    translation = soliton(integrator=Multistep() if kind == "adams" else None)
    if kind == "low_storage":
        translation = low_storage(*translation)
    _problem, _request, backend = _backend(translation)
    try:
        load_reference(backend)
        monitor = BoundaryMonitor.build(window(), None, BoundaryCheck())
        backend.workspace.bind_monitor(monitor)
        march = backend.method(None).march
        engine = march.engine if not isinstance(march, HostAdamsMarch) else None
        if isinstance(march, HostAdamsMarch):
            engine = march._engine  # pyright: ignore[reportPrivateUsage]
        assert engine is not None
        expected = backend.workspace.reading()
        found = engine.reading()  # pyright: ignore[reportAttributeAccessIssue]
        assert found.total == expected.total
        np.testing.assert_array_equal(found.regions, expected.regions)
        if isinstance(march, HostLowStorageMarch):
            # The low-storage step holds no dense extension: interpolation is refused
            # at translation, and the engine says so if asked.
            with pytest.raises(BackendError, match="holds no dense extension"):
                engine.extend(0.0, 1e-3)  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        backend.close()

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service loop over a session, end to end, when a consumer fails (SEP 47 5.3).

A service failure cannot silently acknowledge an event or continue the run; the
primary failure is preserved and the state it leaves is checkpointed from the pending
batch's own export, so the checkpoint and the plane share one field and no plane is
stored twice (SEP 47 sections 5.3 and 5.4). These run through
:func:`solprop.driver.run.run_problem` on the NumPy reference.
"""

import numpy as np
import pytest

from sessions.executors import PLANES, entrance, fibre, soliton, window
from solprop.backends.driver_session import DriverSession
from solprop.backends.registry import select
from solprop.backends.session import StopToken
from solprop.driver.run import run_problem
from solprop.driver.simulation import metadata_for
from solprop.exceptions import StopRequested
from solprop.log import NullReporter
from solprop.output import CallbackOutput, MemoryOutput
from solprop.output.schema import Output, SavedPlane

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

FAILING = 2
"""The failing plane's index among the saved planes: the second save plane, after the
launch plane and the first."""


class ConsumerError(RuntimeError):
    """What the failing consumer raises."""


def _run(output: Output, progress: object = "none") -> None:
    """Run the soliton on the reference into `output`.

    Parameters
    ----------
    output : solprop.output.schema.Output
        Where the planes go.
    progress : object, optional
        The reporter.
    """
    problem = soliton()
    backend = select("numpy_ref", problem, grid=window(), modes=[fibre().mode])
    try:
        run_problem(
            problem,
            backend,
            output,
            metadata=metadata_for(problem, window(), modes=[fibre().mode]),
            progress=progress,  # pyright: ignore[reportArgumentType]
            start=entrance(problem),
        )
    finally:
        backend.close()


class _FailingStore(MemoryOutput):
    """A plane store that keeps what it is offered, and may fail on one plane.

    Parameters
    ----------
    fails : bool
        Whether its consumer raises on plane :data:`FAILING`, before it is stored.
    """

    def __init__(self, *, fails: bool = True) -> None:
        super().__init__()
        self.offered: list[SavedPlane] = []
        self._fails = fails

    def save(self, plane: SavedPlane) -> None:
        self.offered.append(plane)
        if self._fails and len(self.offered) == FAILING + 1:
            msg = f"the consumer failed at z = {plane.z}"
            raise ConsumerError(msg)
        super().save(plane)


def test_a_consumer_failure_checkpoints_the_pending_plane():
    """SEP 47 sections 5.3 and 5.4: the run raises the consumer's error, and its
    terminal checkpoint is the pending plane, with that plane's field and attempts.
    """
    output = _FailingStore()
    with pytest.raises(ConsumerError):
        _run(output)
    pending = output.offered[-1]
    assert pending.z == PLANES[FAILING - 1]
    assert output.n_saved == FAILING
    stored = output["z"].tolist()
    assert stored == [0.0, PLANES[0]]
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None
    assert checkpoint.plane.z == pending.z
    assert pending.field is not None and checkpoint.plane.field is not None
    assert checkpoint.plane.field.tobytes() == pending.field.tobytes()
    # Not stored, so the checkpoint carries the plane's attempts for the resume.
    assert checkpoint.plane.step_history.n_steps == pending.step_history.n_steps > 0
    np.testing.assert_array_equal(
        checkpoint.plane.step_history.z, pending.step_history.z
    )
    assert checkpoint.next_save_index == FAILING
    assert output.outcome is not None and output.outcome.status == "failed"


class _FailingReporter(NullReporter):
    """A reporter that raises at one plane, after the plane was committed."""

    def __init__(self) -> None:
        self.planes: list[float] = []

    def plane(self, z, summary) -> None:
        del summary
        self.planes.append(z)
        if len(self.planes) == FAILING:
            msg = f"the reporter failed at z = {z}"
            raise ConsumerError(msg)


def test_a_failure_after_a_commit_saves_no_plane_twice():
    """SEP 47 section 5.3: the plane's transaction precedes the reporter; when the
    reporter fails, the checkpoint shares the committed plane's field and carries none
    of its attempts again.
    """
    output = _FailingStore(fails=False)
    with pytest.raises(ConsumerError):
        _run(output, progress=_FailingReporter())
    stored = output["z"].tolist()
    assert stored == [0.0, *PLANES[:FAILING]]
    assert [plane.z for plane in output.offered] == stored
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None
    assert checkpoint.plane.z == stored[-1]
    saved = output.offered[-1].field
    assert saved is not None and checkpoint.plane.field is not None
    assert checkpoint.plane.field.tobytes() == saved.tobytes()
    assert checkpoint.plane.step_history.n_steps == 0
    assert checkpoint.plane.step_stats.n_rows == 0


def test_a_callback_consumer_failure_propagates_and_delivers_each_plane_once():
    """SEP 47 section 5.3: a consumer's exception stops the run as it was raised, and
    no plane is delivered twice. A callback output keeps nothing, so no checkpoint.
    """
    delivered: list[float] = []

    def receive(plane: SavedPlane) -> None:
        delivered.append(plane.z)
        if len(delivered) == FAILING + 1:
            msg = f"the consumer failed at z = {plane.z}"
            raise ConsumerError(msg)

    with pytest.raises(ConsumerError):
        _run(CallbackOutput(receive))
    assert delivered == [0.0, *PLANES[:FAILING]]


def _yield_every_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every advance of the run yield after one attempt."""
    advance = DriverSession.advance

    def bounded(self, *, work_limit=1, deadline=1e6):
        del work_limit, deadline
        return advance(self, work_limit=1, deadline=1e6)

    monkeypatch.setattr(DriverSession, "advance", bounded)


class _StoppingReporter(NullReporter):
    """A reporter that hears every attempt and sets a stop at one of them."""

    def __init__(self, at: int) -> None:
        self.heard: list[tuple[float, float, bool]] = []
        self.token = StopToken()
        self._at = at

    def step(self, z, dz, accepted) -> None:
        self.heard.append((z, dz, accepted))
        if len(self.heard) == self._at:
            self.token.set()


@pytest.mark.parametrize("yields", [False, True], ids=["whole planes", "every attempt"])
def test_assembled_planes_and_checkpoints_do_not_depend_on_chunking(
    monkeypatch, yields
):
    """SEP 47 sections 5.3 and 5.5: the service loop assembles each plane's and the
    terminal checkpoint's attempts from chunks, whatever the yields deliver them in.

    The reporter hears each attempt as its chunk is delivered, and sets a stop at the
    seventh, which falls between planes. Each stored plane holds the attempts since
    the previous one and the checkpoint those since the last stored plane, each heard
    attempt exactly once, whether each advance runs to a barrier or yields after every
    attempt. Yielding after every attempt, the run stops just after the seventh, with
    attempts since the last plane; running to its barriers, it hears the seventh only
    with the next plane, and stops there, the checkpoint holding none.
    """
    if yields:
        _yield_every_attempt(monkeypatch)
    reporter = _StoppingReporter(at=7)
    output = _FailingStore(fails=False)
    problem = soliton()
    backend = select("numpy_ref", problem, grid=window(), modes=[fibre().mode])
    try:
        run_problem(
            problem,
            backend,
            output,
            metadata=metadata_for(problem, window(), modes=[fibre().mode]),
            progress=reporter,
            start=entrance(problem),
            stop=reporter.token,
        )
    except StopRequested:
        pass
    finally:
        backend.close()
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None
    records = [plane.step_history for plane in output.offered]
    records.append(checkpoint.plane.step_history)
    assert records[0].n_steps == 0  # the launch plane
    # A record stores each step's start and length; its end is the attempt heard.
    np.testing.assert_allclose(
        np.concatenate([record.z + record.h for record in records]),
        [z for z, _, _ in reporter.heard],
        rtol=1e-14,
    )
    np.testing.assert_array_equal(
        np.concatenate([record.accepted for record in records]),
        [accepted for _, _, accepted in reporter.heard],
    )
    assert (checkpoint.plane.step_history.n_steps > 0) is yields
    assert checkpoint.plane.z == reporter.heard[-1][0]
    if not yields:
        assert checkpoint.plane.z == output.offered[-1].z

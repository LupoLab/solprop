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
from solprop.backends.registry import select
from solprop.driver.run import run_problem
from solprop.driver.simulation import metadata_for
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

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run stopped by an exception or a signal writes a state belonging to its position.

An accepted step moves the state and its position in one commit, so an exception
raised as the step is being recorded -- a progress reporter's, Ctrl-C's -- leaves a
terminal checkpoint whose field is the field at its position. SIGTERM, a scheduler's
warning before it kills the process, stops the run at once outside a commit and as
soon as one ends inside it; the run writes its checkpoint and the march's state, and
``resume=True`` continues it.
"""

import signal
import threading

import numpy as np
import pytest

from driver.test_resume import N_SAVE, CrashError
from driver.test_resume_history import RTOL, assert_accurate_run, soliton, tight_run
from march_bounds import TIGHTER
from solprop._interrupts import committing, handled
from solprop.exceptions import RunTerminated
from solprop.log import NullReporter
from solprop.output.hdf5 import open_file, resume

needs_default_sigterm = pytest.mark.skipif(
    signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL,
    reason="SIGTERM already has a handler in this process",
)


class AfterStep(NullReporter):
    """Act once, on the first step accepted between two planes past the first."""

    def __init__(self, planes, action):
        self.planes = planes
        self.action = action
        self.done = False

    def step(self, z, dz, accepted):
        if (
            accepted
            and not self.done
            and z > self.planes[1]
            and not any(np.isclose(z, one) for one in self.planes)
        ):
            self.done = True
            self.action()


def planes_of(build):
    """Return the run's planes, the entrance first."""
    return [0.0, *build().settings.save_planes.tolist()]


def raise_crash():
    raise CrashError("the reporter failed")


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_an_exception_as_a_step_is_recorded_leaves_a_consistent_checkpoint(
    build_simulation, tmp_path, backend
):
    # The step's state and position are committed before it is recorded: the
    # checkpoint is the accepted step's, and the resumed run is the uninterrupted one
    # to its accuracy envelope (an adaptive pair, SEP 25 section 4.4).
    build = soliton(build_simulation, backend)
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reporter = AfterStep(planes_of(build), raise_crash)
    with pytest.raises(CrashError):
        build().run(part, progress=reporter)
    assert reporter.done
    state = resume(part)
    assert state.checkpoint is not None and state.march is not None
    assert state.march.z == state.checkpoint.plane.z
    build().run(part, resume=True, progress="none")
    assert_accurate_run(
        whole,
        part,
        tight_run(
            soliton(build_simulation, backend, rtol=RTOL / TIGHTER),
            tmp_path / "tight.h5",
        ),
    )


@pytest.mark.medium
@needs_default_sigterm
def test_sigterm_writes_the_run_s_state_and_a_resume_continues_it(
    build_simulation, tmp_path
):
    build = soliton(build_simulation, "numpy_ref")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reporter = AfterStep(planes_of(build), lambda: signal.raise_signal(signal.SIGTERM))
    with pytest.raises(RunTerminated) as stopped:
        build().run(part, progress=reporter)
    assert stopped.value.code == 128 + signal.SIGTERM
    # The handler is the run's alone.
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    state = resume(part)
    assert state.outcome is not None and state.outcome.status == "failed"
    assert "terminated by SIGTERM" in state.outcome.message
    assert state.checkpoint is not None and state.march is not None
    build().run(part, resume=True, progress="none")
    assert_accurate_run(
        whole,
        part,
        tight_run(
            soliton(build_simulation, "numpy_ref", rtol=RTOL / TIGHTER),
            tmp_path / "tight.h5",
        ),
    )
    with open_file(part) as stored:
        assert np.asarray(stored["z"]).size == N_SAVE


@needs_default_sigterm
def test_a_signal_inside_a_commit_waits_for_its_end():
    events = []
    with handled():
        with pytest.raises(RunTerminated), committing():
            signal.raise_signal(signal.SIGTERM)
            events.append("the commit finished")
        events.append("raised after it")
        with pytest.raises(RunTerminated):
            signal.raise_signal(signal.SIGTERM)
            events.append("not reached")
        # A commit written while stopping drops a second signal.
        with committing(deliver=False):
            signal.raise_signal(signal.SIGTERM)
    assert events == ["the commit finished", "raised after it"]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_an_application_s_own_handler_is_kept():
    def own(signal_number, frame):
        pass

    previous = signal.signal(signal.SIGTERM, own)
    try:
        with handled():
            assert signal.getsignal(signal.SIGTERM) is own
        assert signal.getsignal(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, previous)


@needs_default_sigterm
def test_a_run_outside_the_main_thread_installs_no_handler():
    # Python delivers signals to the main thread only, and only it may set a handler:
    # a run in a worker thread leaves the process's handlers as they were.
    seen = []

    def work():
        with handled():
            seen.append(
                (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
            )

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert seen == [(signal.SIG_DFL, signal.getsignal(signal.SIGINT))]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

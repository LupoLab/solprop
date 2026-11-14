# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march stops where the Python march would (solprop SEP 39 7.5).

A native call runs many attempts and returns at an event or its bound; whatever stops
the run -- a native failure at the startup sample, a trial or an accept, the floor, the
budget, a signal or a stop flag during the call, a progress reporter's exception as the
call's steps are recorded -- leaves the run at the call's last accepted step, its
state, position, history and ledger together, and a terminal checkpoint that a resume
continues.

Every native run marches in the native execution session (SEP 47), whose advance is
the native call here, read through what the run persists: the file's outcome, ledger,
checkpoint and method history. Where a stop is a decision -- the budget, the floor --
the native march is held to the reference's independent Adams march on the same
problem.
"""

import dataclasses
import signal
import sys
import threading
from typing import Any, cast

import numpy as np
import pytest

from driver.test_interrupts import AfterStep, needs_default_sigterm, planes_of
from driver.test_resume import CrashError
from driver.test_resume_history import (
    RTOL,
    assert_accurate_run,
    coupled,
    soliton,
    tight_run,
)
from march_bounds import TIGHTER
from solprop.backends import available, sessions
from solprop.backends.session import DEFAULT_WORK_LIMIT
from solprop.cli import main
from solprop.exceptions import ConvergenceError, ResponseBudgetError, RunTerminated
from solprop.log import NullReporter
from solprop.output.hdf5 import open_file, resume
from solprop.recipe import Recipe
from solprop.stepper.multistep import Multistep

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    # The fixtures, borrowed from test_resume_history, launch noise-free into time
    # windows kept tight for speed: the boundary monitor warns at the launch (SEP 21),
    # and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

backend_module = pytest.importorskip("solprop_native.backend")
NativeAdamsMarch = backend_module._NativeAdamsMarch
NativeSession = pytest.importorskip("solprop_native.session").NativeSession


def with_fault(monkeypatch, stage: str, after: int, category: str) -> list[Any]:
    """Arm a native failure (or, for ``"sigint"`` and ``"sigterm"``, a signal raised in
    the native call) when the run's march is first configured, before a session leases
    the state; return the marches it was armed on."""
    armed: list[Any] = []
    configure = NativeAdamsMarch._configure

    def arming(self):
        configure(self)
        if not armed:
            self._backend._native_state.adams_march_fault(stage, after, category)
            armed.append(self)

    monkeypatch.setattr(NativeAdamsMarch, "_configure", arming)
    return armed


def assert_consistent_stop(part) -> None:
    """Assert the stopped run's checkpoint and march state belong to one position."""
    state = resume(part)
    assert state.checkpoint is not None and state.march is not None
    assert state.march.z == state.checkpoint.plane.z
    assert state.march.positions[-1] == state.march.z


def ledger_of(part) -> dict[str, Any]:
    """Return the call ledger the run's outcome records."""
    with open_file(part) as stored:
        assert stored.outcome is not None
        return dict(stored.outcome.logical_calls)


@pytest.mark.parametrize(
    "fault",
    [
        ("startup", "backend", RuntimeError),
        ("trial", "convergence", ConvergenceError),
        ("trial", "value", ValueError),
        ("accept", "backend", RuntimeError),
    ],
    ids=lambda fault: f"{fault[0]}-{fault[1]}-{fault[2].__name__}",
)
def test_a_native_failure_raises_its_class_at_the_last_accepted_step(
    build_simulation, tmp_path, monkeypatch, fault
):
    stage, category, raised = fault
    build = soliton(build_simulation, "native")
    part = tmp_path / "part.h5"
    armed = with_fault(monkeypatch, stage, 0 if stage == "startup" else 30, category)
    with pytest.raises(raised, match="an injected native failure"):
        build().run(part, progress="none")
    assert len(armed) == 1
    ledger = ledger_of(part)
    if stage == "startup":
        # The startup sample was charged before it failed, and nothing was accepted:
        # the checkpoint is the entrance, with no history to keep.
        assert (ledger["startup"], ledger["accepted_steps"]) == (1, 0)
        state = resume(part)
        assert state.checkpoint is not None and state.checkpoint.plane.z == 0.0
        assert state.march is None
        return
    # The steps committed before the failure stand, with their history and orders.
    assert ledger["accepted_steps"] + ledger["rejected_steps"] >= 30
    orders = cast("list[list[int]]", ledger["orders"])
    assert sum(count for _, count in orders) == ledger["accepted_steps"]
    assert_consistent_stop(part)
    state = resume(part)
    assert state.march is not None
    assert state.march.z == state.outcome.z  # pyright: ignore[reportOptionalMemberAccess]


SOURCE_PATHS = {
    "elementwise": (lambda build_simulation: soliton(build_simulation, "native"), 30),
    "fused": (
        lambda build_simulation: soliton(
            build_simulation, "native", integrator=Multistep(formulation="interaction")
        ),
        30,
    ),
    "coupled": (lambda _: coupled("native"), 2),
}
"""The native trial's three source evaluations -- the exponential march's elementwise
trial, the interaction formulation's fused trial and a coherent wall's coupled flow --
each with the trials before its failure, within its run."""


# The coherent wall's warnings, as the resume tests that run it take them.
@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates a packet")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("path", SOURCE_PATHS)
def test_a_failure_inside_a_trial_s_source_leaves_the_accepted_state(
    build_simulation, tmp_path, monkeypatch, path
):
    # Issue #129: a trial evaluates its source with its prediction standing in the
    # state. A failure there must leave the accepted state, checkpointed and
    # snapshotted, as the same failure before the prediction does: the two runs make
    # the same attempts, and a prediction left behind differs by a step's change.
    make, after = SOURCE_PATHS[path]
    build = make(build_simulation)
    stops = {}
    for stage in ("trial", "trial_source"):
        part = tmp_path / f"{stage}.h5"
        with monkeypatch.context() as patch:
            with_fault(patch, stage, after, "backend")
            with pytest.raises(RuntimeError, match="an injected native failure"):
                build().run(part, progress="none")
        assert_consistent_stop(part)
        stops[stage] = resume(part)
    before, inside = stops["trial"], stops["trial_source"]
    assert before.checkpoint is not None and inside.checkpoint is not None
    assert before.march is not None and inside.march is not None
    assert inside.checkpoint.plane.z == before.checkpoint.plane.z > 0.0
    for found, accepted in (
        (inside.checkpoint.plane.field, before.checkpoint.plane.field),
        (inside.march.state, before.march.state),
    ):
        scale = np.linalg.norm(accepted)
        assert np.linalg.norm(np.asarray(found) - accepted) <= 1e-12 * scale


@pytest.mark.medium
def test_a_run_stopped_by_a_native_failure_continues_from_its_checkpoint(
    build_simulation, tmp_path, monkeypatch, caplog
):
    build = soliton(build_simulation, "native")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    with monkeypatch.context() as patch:
        with_fault(patch, "trial", 40, "backend")
        with pytest.raises(RuntimeError, match="an injected native failure"):
            build().run(part, progress="none")
    assert_consistent_stop(part)
    build().run(part, resume=True, progress="none")
    assert_accurate_run(
        whole,
        part,
        tight_run(
            soliton(build_simulation, "native", rtol=RTOL / TIGHTER),
            tmp_path / "tight.h5",
        ),
    )


@pytest.mark.medium
def test_an_exception_as_native_steps_are_recorded_leaves_a_consistent_checkpoint(
    build_simulation, tmp_path
):
    # The progress reporter runs after the call's commit: the backend has adopted the
    # call's state by then, so the checkpoint is its last accepted step's.
    build = soliton(build_simulation, "native")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")

    def crash():
        raise CrashError("the reporter failed")

    reporter = AfterStep(planes_of(build), crash)
    with pytest.raises(CrashError):
        build().run(part, progress=reporter)
    assert reporter.done
    assert_consistent_stop(part)
    build().run(part, resume=True, progress="none")
    assert_accurate_run(
        whole,
        part,
        tight_run(
            soliton(build_simulation, "native", rtol=RTOL / TIGHTER),
            tmp_path / "tight.h5",
        ),
    )


@pytest.mark.medium
@needs_default_sigterm
def test_sigterm_during_a_native_call_waits_for_its_commit(
    build_simulation, tmp_path, monkeypatch
):
    # The signal arrives while the native march runs, at its 41st trial: it is held
    # until the call's state and rows are adopted, then stops the run there.
    build = soliton(build_simulation, "native")
    calls = calling(monkeypatch)
    with_fault(monkeypatch, "trial", 40, "sigterm")
    part = tmp_path / "part.h5"
    with pytest.raises(RunTerminated):
        build().run(part, progress="none")
    assert calls.signalled(40)
    state = resume(part)
    assert state.checkpoint is not None and state.march is not None
    # The call ran to its end before the signal was raised: the checkpoint is where
    # it returned, past where it began.
    assert state.march.z == state.checkpoint.plane.z == calls.ends[-1]
    assert state.checkpoint.plane.z > calls.starts[-1]


FORMS = {
    "exponential": Multistep(),
    "interaction": Multistep(formulation="interaction"),
}
"""The two forms of the native march."""

needs_default_sigint = pytest.mark.skipif(
    signal.getsignal(signal.SIGINT) is not signal.default_int_handler,
    reason="something else handles SIGINT here",
)


ONE_PLANE_LENGTH = 0.2
"""The one-plane run's length, m, with an N = 10 soliton: about a thousand steps."""


def one_plane(build_simulation, form: str):
    """Return an N = 10 soliton saved only at its end, with no boundary monitor: only a
    native call's own bound returns the march to Python before the end of the run."""

    def unmonitored():
        simulation = build_simulation(
            order=10,
            length=ONE_PLANE_LENGTH,
            n_save=2,
            backend="native",
            rtol=RTOL,
            integrator=FORMS[form],
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=None
        )
        return simulation

    return unmonitored


class Calls:
    """Each native call a run made: its attempts and where it began and returned."""

    def __init__(self) -> None:
        self.attempts: list[int] = []
        self.starts: list[float] = []
        self.ends: list[float] = []

    def signalled(self, trials: int) -> bool:
        """Return whether the run's last call made the attempt after `trials` attempts,
        the one a signal armed after `trials` trials arrived in."""
        return sum(self.attempts[:-1]) <= trials < sum(self.attempts)


def calling(monkeypatch) -> Calls:
    """Record each native session advance, the run's native call, as it returns or
    raises."""
    calls = Calls()
    advance = NativeSession.advance

    def counted(self, **bounds):
        before = self.status()
        calls.starts.append(before.z)
        try:
            return advance(self, **bounds)
        finally:
            after = self.status()
            calls.attempts.append(after.attempts - before.attempts)
            calls.ends.append(after.z)

    monkeypatch.setattr(NativeSession, "advance", counted)
    return calls


TRIALS = 300
"""The trials before the signal: past the first native call of the one-plane run,
whose bound is 256 attempts."""


def assert_stopped_at_its_call(part, calls: Calls) -> None:
    """Assert the run made no native call after the one the signal arrived in, and
    stopped where that call returned."""
    assert calls.signalled(TRIALS)
    assert len(calls.attempts) >= 3
    assert all(found <= DEFAULT_WORK_LIMIT for found in calls.attempts)
    assert_consistent_stop(part)
    with open_file(part) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "failed"
        assert 0.0 < stored.outcome.z < ONE_PLANE_LENGTH
        assert stored.outcome.z == calls.ends[-1]
        assert stored.n_saved == 1
        taken = cast("int", stored.outcome.logical_calls["accepted_steps"])
    assert taken <= sum(calls.attempts)


@pytest.mark.parametrize("form", FORMS)
def test_a_flag_set_during_a_native_call_stops_the_march_as_it_returns(
    build_simulation, tmp_path, monkeypatch, form
):
    # A front end's flag is set while the native call holds no Python lock; here a
    # signal handler sets it, which Python runs as the call returns, as it would see
    # a flag another thread set during the call.
    flag = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: flag.set())
    try:
        calls = calling(monkeypatch)
        with_fault(monkeypatch, "trial", TRIALS, "sigint")
        part = tmp_path / "part.h5"
        one_plane(build_simulation, form)().run(part, progress="none", stop=flag)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert flag.is_set()
    assert_stopped_at_its_call(part, calls)


@pytest.mark.medium
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(
    ("number", "raised"),
    [
        pytest.param(
            signal.SIGTERM, RunTerminated, marks=needs_default_sigterm, id="sigterm"
        ),
        pytest.param(
            signal.SIGINT, KeyboardInterrupt, marks=needs_default_sigint, id="sigint"
        ),
    ],
)
def test_a_signal_during_a_one_plane_march_stops_it_as_the_native_call_returns(
    build_simulation, tmp_path, monkeypatch, form, number, raised
):
    # Python's handler runs when the call's bound brings the main thread back to the
    # interpreter, so a signal needs no plane to reach the run.
    build = one_plane(build_simulation, form)
    part = tmp_path / "part.h5"
    with monkeypatch.context() as patch:
        calls = calling(patch)
        with_fault(patch, "trial", TRIALS, signal.Signals(number).name.lower())
        with pytest.raises(raised):
            build().run(part, progress="none")
    assert_stopped_at_its_call(part, calls)
    build().run(part, resume=True, progress="none")
    with open_file(part) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "complete"
        assert stored.outcome.z == ONE_PLANE_LENGTH


ONE_PLANE_RECIPE = f"""\
[recipe]
format = 1
route = "gnlse"

[build]
gamma = 0.11
length = "{ONE_PLANE_LENGTH} m"
betas = [0, 0, "-0.01183 ps^2/m"]
wavelength = "835 nm"
wavelength_limits = ["500 nm", "2 um"]
trange = "1 ps"
width = "100 fs"
peak_power = "1075 W"
shape = "sech"
n_save = 2
noise = "none"
backend = "native"
integrator = {{type = "Multistep"}}
"""
"""The N = 10 soliton of :func:`one_plane`, as a recipe."""


@needs_default_sigterm
def test_the_command_line_stops_a_one_plane_march_on_its_first_signal(
    tmp_path, monkeypatch, capsys
):
    # `solprop run` maps the signal onto the run's flag, so here the signal has nothing
    # but the flag to reach the march by: unpolled, the run finished its plane.
    recipe, part = tmp_path / "soliton.toml", tmp_path / "part.h5"
    recipe.write_text(ONE_PLANE_RECIPE, encoding="utf-8")
    with monkeypatch.context() as patch:
        calls = calling(patch)
        with_fault(patch, "trial", TRIALS, "sigterm")
        status = main(["run", str(recipe), "-o", str(part), "--progress", "none"])
    assert status == 128 + signal.SIGTERM
    assert "solprop resume" in capsys.readouterr().err
    assert_stopped_at_its_call(part, calls)
    assert main(["resume", str(part), "--progress", "none"]) == 0
    with open_file(part) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "complete"
        assert stored.outcome.z == ONE_PLANE_LENGTH


def two_marches(make, monkeypatch, part) -> dict[str, Any]:
    """Run the simulation to `part` on the native march and on the reference; return
    each run's outcome, as the file records it: the error's class and message, its
    position and its ledger.

    `make` takes a backend's name and returns the simulation's builder. The native
    march runs in the native session; the reference's is its own independent Adams
    march (SEP 47 section 6.6)."""
    found = {}
    original = NativeAdamsMarch.__init__
    for backend in ("native", "numpy_ref"):
        marches: list[Any] = []
        served: list[str] = []

        def recording(self, *args, marches=marches, **keywords):
            original(self, *args, **keywords)
            marches.append(self)

        def serving(*args, served=served, start=sessions.start_session, **keywords):
            session = start(*args, **keywords)
            served.append(type(session).__name__)
            return session

        with monkeypatch.context() as patch:
            patch.setattr(NativeAdamsMarch, "__init__", recording)
            patch.setattr(sessions, "start_session", serving)
            try:
                make(backend)().run(part)
                error = None
            except (ConvergenceError, ResponseBudgetError) as stopped:
                # The message names the backend; the rest is the march's.
                error = (type(stopped), str(stopped).replace(backend, "BACKEND"))
        if backend == "native":
            assert len(marches) == 1
            assert served == ["NativeSession"]
        else:
            assert marches == []
            assert served == ["HostSession"]
        ledger = ledger_of(part)
        with open_file(part) as stored:
            assert stored.outcome is not None
            z = stored.outcome.z
        found[backend] = {
            "error": error,
            "z": z,
            "ledger": ledger["startup"] + ledger["predictor"] + ledger["corrected"],
            "steps": (ledger["accepted_steps"], ledger["rejected_steps"]),
        }
    return found


def assert_same_stop(native: dict[str, Any], reference: dict[str, Any]) -> None:
    """Assert the two marches stopped alike: the same error and calls, and positions a
    rounding apart (SEP 25 asserts no bits).

    The reference forms each norm in its own order, so each proposal moves by the
    norm's relative rounding, a few units in the last place of the field's sums, and
    the steps' sum by at most their count times that: under 1e-10 of the position over
    these fewer than a hundred steps (measured 9.4e-12 on the budget case)."""
    assert native["error"] == reference["error"]
    assert (native["ledger"], native["steps"]) == (
        reference["ledger"],
        reference["steps"],
    )
    assert native["z"] == pytest.approx(reference["z"], rel=1e-10, abs=0.0)


def test_the_budget_stops_both_marches_at_the_same_step(
    build_simulation, tmp_path, monkeypatch
):
    def make(backend):
        return soliton(build_simulation, backend, integrator=Multistep(max_calls=60))

    found = two_marches(make, monkeypatch, tmp_path / "b.h5")
    native, reference = found["native"], found["numpy_ref"]
    assert native["error"] is not None and native["error"][0] is ResponseBudgetError
    # The native march and the reference's independent one take the same decisions
    # on this soliton: the same calls stop the run at the same step, with the same
    # message.
    assert_same_stop(native, reference)


def test_the_floor_stops_both_marches_at_the_same_step(
    build_simulation, tmp_path, monkeypatch
):
    def make(backend):
        # At this tolerance the steps want to be shorter than this floor.
        tight = soliton(build_simulation, backend, rtol=1e-12)

        def build():
            simulation = tight()
            simulation.settings = dataclasses.replace(
                simulation.settings, dz_min=5e-6, dz_init=5e-6
            )
            return simulation

        return build

    found = two_marches(make, monkeypatch, tmp_path / "f.h5")
    native, reference = found["native"], found["numpy_ref"]
    assert native["error"] is not None and native["error"][0] is ConvergenceError
    assert "below the floor" in native["error"][1]
    assert_same_stop(native, reference)


def test_both_marches_take_the_same_steps(build_simulation, tmp_path, monkeypatch):
    def make(backend):
        return soliton(build_simulation, backend)

    found = two_marches(make, monkeypatch, tmp_path / "s.h5")
    assert found["native"]["error"] is None
    assert_same_stop(found["native"], found["numpy_ref"])


@needs_default_sigterm
def test_a_signal_too_late_to_stop_the_run_says_so(tmp_path, monkeypatch, capsys):
    # A signal after the run's last poll stops nothing: the file is complete, so the
    # command mustn't exit as stopped, or tell anyone to resume it.
    recipe, output = tmp_path / "soliton.toml", tmp_path / "run.h5"
    recipe.write_text(ONE_PLANE_RECIPE, encoding="utf-8")
    run = Recipe.run

    def signalled(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(Recipe, "run", signalled)
    status = main(["run", str(recipe), "-o", str(output), "--progress", "none"])
    assert status == 0
    error = capsys.readouterr().err
    assert "too late" in error
    assert "solprop resume" not in error
    with open_file(output) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "complete"


class FailingInTheCheckpoint(NullReporter):
    """A reporter that counts what it hears, and raises if it is called while the run
    handles its failure, as a checkpoint is written."""

    def __init__(self) -> None:
        self.heard = 0

    def step(self, z, dz, accepted) -> None:
        if isinstance(sys.exc_info()[1], RunTerminated):
            raise CrashError("the reporter failed as the checkpoint was written")
        self.heard += 1


@pytest.mark.medium
@needs_default_sigterm
def test_a_reporter_cannot_lose_the_failure_s_checkpoint(
    build_simulation, tmp_path, monkeypatch
):
    # The signal is raised as its advance returns, before the batch reached the
    # service loop: its attempts go into the checkpoint without the reporter hearing
    # them, so a reporter that would raise there costs nothing.
    build = soliton(build_simulation, "native")
    with_fault(monkeypatch, "trial", 40, "sigterm")
    reporter = FailingInTheCheckpoint()
    part = tmp_path / "part.h5"
    with pytest.raises(RunTerminated) as stopped:
        build().run(part, progress=reporter)
    assert not getattr(stopped.value, "__notes__", None)
    assert_consistent_stop(part)
    state = resume(part)
    assert state.checkpoint is not None
    history = state.checkpoint.plane.step_history
    assert history.n_steps > 0
    ledger = ledger_of(part)
    # Every attempt is stored; the held batch's alone were not reported.
    assert reporter.heard < ledger["accepted_steps"] + ledger["rejected_steps"]

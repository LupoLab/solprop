# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march stops where the Python march would (solprop SEP 39 7.5).

A native call runs many attempts and returns at an event or its bound; whatever stops
the run -- a native failure at the startup sample, a trial or an accept, the floor, the
budget, a signal or a stop flag during the call, a progress reporter's exception as the
call's steps are recorded -- leaves the backend at the call's last accepted step, its
state, position, history and ledger together, and a terminal checkpoint that a resume
continues.
"""

import dataclasses
import signal
import threading
from typing import Any, cast

import pytest

from driver.test_interrupts import AfterStep, needs_default_sigterm, planes_of
from driver.test_resume import CrashError
from driver.test_resume_history import RTOL, assert_accurate_run, soliton, tight_run
from march_bounds import TIGHTER
from solprop.backends import available
from solprop.cli import main
from solprop.exceptions import ConvergenceError, ResponseBudgetError, RunTerminated
from solprop.output.hdf5 import open_file, resume
from solprop.recipe import Recipe
from solprop.stepper.multistep import Multistep

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

backend_module = pytest.importorskip("solprop_native.backend")
NativeAdamsMarch = backend_module._NativeAdamsMarch


def with_fault(monkeypatch, stage: str, after: int, category: str) -> list[Any]:
    """Arm a native failure when the run's march is first configured; return the
    marches it was armed on."""
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


@pytest.mark.parametrize(
    ("stage", "category", "raised"),
    [
        ("startup", "backend", RuntimeError),
        ("trial", "convergence", ConvergenceError),
        ("trial", "value", ValueError),
        ("accept", "backend", RuntimeError),
    ],
)
def test_a_native_failure_raises_its_class_at_the_last_accepted_step(
    build_simulation, tmp_path, monkeypatch, stage, category, raised
):
    build = soliton(build_simulation, "native")
    armed = with_fault(monkeypatch, stage, 0 if stage == "startup" else 30, category)
    with pytest.raises(raised, match="an injected native failure"):
        build().run(tmp_path / "part.h5", progress="none")
    (march,) = armed
    assert march.native
    ledger = march.ledger
    if stage == "startup":
        # The startup sample was charged before it failed, and nothing was accepted:
        # the checkpoint is the entrance, with no history to keep.
        assert (ledger.startup, ledger.accepted_steps) == (1, 0)
        assert march._backend._z == 0.0
        state = resume(tmp_path / "part.h5")
        assert state.checkpoint is not None and state.checkpoint.plane.z == 0.0
        assert state.march is None
        return
    # The steps committed before the failure stand, with their history.
    assert ledger.accepted_steps + ledger.rejected_steps >= 30
    assert march._history and march._history[-1].z == march._backend._z
    assert len(ledger.orders) == ledger.accepted_steps
    assert_consistent_stop(tmp_path / "part.h5")


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
    # The signal arrives while the native march runs: it is held until the call's
    # state and rows are adopted, then stops the run there.
    build = soliton(build_simulation, "native")
    planes = planes_of(build)
    run = NativeAdamsMarch._native_run
    seen: list[float] = []

    def signalled(self, state, bounds):
        if not seen and state["z"] > planes[1]:
            seen.append(state["z"])
            signal.raise_signal(signal.SIGTERM)
        return run(self, state, bounds)

    monkeypatch.setattr(NativeAdamsMarch, "_native_run", signalled)
    part = tmp_path / "part.h5"
    with pytest.raises(RunTerminated):
        build().run(part, progress="none")
    assert seen
    state = resume(part)
    assert state.checkpoint is not None and state.march is not None
    # The call ran to its event before the signal was raised.
    assert state.march.z == state.checkpoint.plane.z > seen[0]


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


def calling(monkeypatch, at_call: int, act) -> list[int]:
    """Run `act` as native call number `at_call` begins; return each call's attempts."""
    run = NativeAdamsMarch._native_run
    attempts: list[int] = []

    def counted(self, state, bounds):
        if len(attempts) + 1 == at_call:
            act()
        result = run(self, state, bounds)
        attempts.append(len(result["rows"]))
        return result

    monkeypatch.setattr(NativeAdamsMarch, "_native_run", counted)
    return attempts


def assert_stopped_at_its_call(part, attempts: list[int], at_call: int) -> None:
    """Assert the run made no native call after `at_call`, and stopped at its end."""
    assert len(attempts) == at_call
    assert all(found <= NativeAdamsMarch.MAX_ATTEMPTS for found in attempts)
    assert_consistent_stop(part)
    with open_file(part) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "failed"
        assert 0.0 < stored.outcome.z < ONE_PLANE_LENGTH
        assert stored.n_saved == 1
        taken = cast("int", stored.outcome.logical_calls["accepted_steps"])
    assert taken <= sum(attempts)


@pytest.mark.parametrize("form", FORMS)
def test_a_flag_set_during_a_native_call_stops_the_march_as_it_returns(
    build_simulation, tmp_path, monkeypatch, form
):
    flag = threading.Event()
    attempts = calling(monkeypatch, 3, flag.set)
    part = tmp_path / "part.h5"
    one_plane(build_simulation, form)().run(part, progress="none", stop=flag)
    assert_stopped_at_its_call(part, attempts, 3)


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
        attempts = calling(patch, 3, lambda: signal.raise_signal(number))
        with pytest.raises(raised):
            build().run(part, progress="none")
    assert_stopped_at_its_call(part, attempts, 3)
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
        attempts = calling(patch, 3, lambda: signal.raise_signal(signal.SIGTERM))
        status = main(["run", str(recipe), "-o", str(part), "--progress", "none"])
    assert status == 128 + signal.SIGTERM
    assert "solprop resume" in capsys.readouterr().err
    assert_stopped_at_its_call(part, attempts, 3)
    assert main(["resume", str(part), "--progress", "none"]) == 0
    with open_file(part) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "complete"
        assert stored.outcome.z == ONE_PLANE_LENGTH


def two_marches(build, monkeypatch, run) -> dict[str, Any]:
    """Run `run(simulation)` on the native march and on the host one; return each
    march's outcome: the error's class and message, its position and its ledger."""
    found = {}
    original = NativeAdamsMarch.__init__
    for march in ("native", "host"):
        marches: list[Any] = []

        def recording(self, *args, marches=marches, **keywords):
            original(self, *args, **keywords)
            marches.append(self)

        with monkeypatch.context() as patch:
            patch.setattr(backend_module, "_ADAMS_MARCH", march)
            patch.setattr(NativeAdamsMarch, "__init__", recording)
            try:
                run(build())
                error = None
            except (ConvergenceError, ResponseBudgetError) as stopped:
                error = (type(stopped), str(stopped))
        (taken,) = marches
        assert taken.native == (march == "native")
        found[march] = {
            "error": error,
            "z": taken._backend._z,
            "ledger": taken.ledger.total,
            "steps": (taken.ledger.accepted_steps, taken.ledger.rejected_steps),
        }
    return found


def assert_same_stop(native: dict[str, Any], host: dict[str, Any]) -> None:
    """Assert the two marches stopped alike: the same error and calls, and positions a
    rounding apart (measured equal; SEP 25 asserts no bits)."""
    assert native["error"] == host["error"]
    assert (native["ledger"], native["steps"]) == (host["ledger"], host["steps"])
    assert native["z"] == pytest.approx(host["z"], rel=1e-12, abs=0.0)


def test_the_budget_stops_both_marches_at_the_same_step(
    build_simulation, tmp_path, monkeypatch
):
    build = soliton(build_simulation, "native", integrator=Multistep(max_calls=60))
    found = two_marches(
        build, monkeypatch, lambda simulation: simulation.run(tmp_path / "b.h5")
    )
    native, host = found["native"], found["host"]
    assert native["error"] is not None and native["error"][0] is ResponseBudgetError
    # Within one build both marches round alike (SEP 39 4.9): the same calls stop the
    # run at the same step, with the same message.
    assert_same_stop(native, host)


def test_the_floor_stops_both_marches_at_the_same_step(
    build_simulation, tmp_path, monkeypatch
):
    # At this tolerance the steps want to be shorter than this floor.
    tight = soliton(build_simulation, "native", rtol=1e-12)

    def build():
        simulation = tight()
        simulation.settings = dataclasses.replace(
            simulation.settings, dz_min=5e-6, dz_init=5e-6
        )
        return simulation

    found = two_marches(
        build, monkeypatch, lambda simulation: simulation.run(tmp_path / "f.h5")
    )
    native, host = found["native"], found["host"]
    assert native["error"] is not None and native["error"][0] is ConvergenceError
    assert "below the floor" in native["error"][1]
    assert_same_stop(native, host)


def test_both_marches_take_the_same_steps(build_simulation, tmp_path, monkeypatch):
    build = soliton(build_simulation, "native")
    found = two_marches(
        build, monkeypatch, lambda simulation: simulation.run(tmp_path / "s.h5")
    )
    assert found["native"]["error"] is None
    assert_same_stop(found["native"], found["host"])


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

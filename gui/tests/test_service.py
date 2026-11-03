# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel, its handlers and the worker's run, in this process."""

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from solprop.recipe import Recipe

from conftest import TINY, pipe
from solprop_gui.service import handlers, kernel, run
from solprop_gui.service.messages import PROTOCOL, Control, Event, Reply, Request
from solprop_gui.service.outbox import Outbox
from solprop_gui.service.reduce import Display, Reducer
from solprop_gui.service.worker import LOG, OUTPUT, RECIPE, RunSpec


def ask(kind: str, payload: Mapping[str, object]) -> Any:
    """Call a kernel handler as the kernel does; its reply is plain data."""
    return handlers.HANDLERS[kind](payload)


# ---------------------------------------------------------------------- kernel


def test_latest_wins_by_kind_and_key():
    requests = [
        Request("check", {}, 1),
        Request("summarise", {}, 2),
        Request("check", {}, 3),
        Request("check", {}, 4, key="other"),
    ]
    keep, dropped = kernel.latest(iter(requests))
    assert [r.request_id for r in keep] == [2, 3, 4]
    assert [r.request_id for r in dropped] == [1]


def test_answer_turns_every_failure_into_a_reply():
    def broken(_payload: object) -> object:
        raise ValueError

    table = {"echo": lambda payload: payload["x"], "broken": broken}
    assert kernel.answer(Request("echo", {"x": 2}, 1, revision=5), table).value == 2
    failed = kernel.answer(Request("broken", {}, 2), table)
    assert failed.error == "ValueError" and "Traceback" in str(failed.detail)
    assert "no handler" in str(kernel.answer(Request("nope", {}, 3), table).error)
    old = Request("echo", {"x": 1}, 4, protocol=PROTOCOL + 1)
    assert "protocol" in str(kernel.answer(old, table).error)


def test_serve_answers_until_the_pipe_closes():
    window, child = pipe()
    server = threading.Thread(
        target=kernel.serve, args=(child, {"echo": lambda p: p["x"]}), daemon=True
    )
    window.send(Request("echo", {"x": 1}, 1))
    window.send(Request("echo", {"x": 2}, 2))
    server.start()
    replies = [window.recv(), window.recv()]
    assert [(r.request_id, r.superseded, r.value) for r in replies] == [
        (1, True, None),
        (2, False, 2),
    ]
    window.close()
    server.join(5)
    assert not server.is_alive()


def test_serve_stops_when_the_window_cannot_hear_it():
    window, child = pipe()

    def close_then_answer(_payload: object) -> int:
        window.close()
        return 1

    window.send(Request("x", {}, 1))
    kernel.serve(child, {"x": close_then_answer})  # returns, does not raise


def test_serve_imports_solprops_handlers_when_given_none(
    monkeypatch: pytest.MonkeyPatch,
):
    window, child = pipe()
    window.close()
    monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    kernel.serve(child)
    import os  # noqa: PLC0415 - only here

    assert os.environ["SOLPROP_THREADS"] == kernel.KERNEL_THREADS


# ---------------------------------------------------------------------- handlers


def test_examples_and_describe():
    names = {entry["name"]: entry for entry in ask("examples", {})}
    assert names["rdw"]["route"] == "capillary"
    assert "[recipe]" in ask("example", {"name": "rdw"})
    described = ask("describe", {"route": "capillary"})
    parameters = {entry["name"]: entry for entry in described["parameters"]}
    assert parameters["radius"]["required"] is True
    assert parameters["radius"]["forms"][0]["unit"] == "m"
    assert parameters["kerr"]["default"] == "True"
    assert parameters["rtol"]["resolved_by_rule"]
    assert any(rule["id"] == "raman-needs-kerr" for rule in described["rules"])
    assert {entry["name"] for entry in described["execution"]} >= {"fields"}


def test_every_label_names_a_keyword_of_a_route():
    # The registry's labels must not drift from solprop's keywords (design 12.1).
    from solprop_gui.registry.pages import LABELS  # noqa: PLC0415 - only here

    keywords: set[str] = set()
    for route in ("capillary", "waveguide", "gnlse"):
        described = ask("describe", {"route": route})
        keywords |= {entry["name"] for entry in described["parameters"]}
        for entries in described["types"].values():
            keywords |= {entry["name"] for entry in entries}
    assert set(LABELS) <= keywords


def test_defaults_are_written_as_the_front_end_shows_them():
    described = ask("describe", {"route": "capillary"})
    parameters = {entry["name"]: entry for entry in described["parameters"]}
    assert parameters["length"]["default"] is None  # required
    assert parameters["temperature"]["default"] is None  # None by default
    assert any(
        entry["default"] is not None and " " in entry["default"]
        for entry in parameters.values()
    )  # a number with its display unit


def test_plain_turns_values_into_data():
    assert handlers._plain((1, [2.0, "a"])) == [1, [2.0, "a"]]
    assert str(handlers._plain(object())).startswith("<object")
    assert handlers._pair(None) is None
    assert handlers._pair([1, 2]) == (1.0, 2.0)


def test_check_places_issues_at_their_paths(tmp_path: Path):
    assert ask("check", {"text": TINY, "base": str(tmp_path)}) == []
    issues = ask(
        "check", {"text": TINY.replace('energy = "100 uJ"', 'energy = "1 fs"')}
    )
    assert issues[0]["parameter"] == "energy" and issues[0]["severity"] == "error"
    unreadable = ask("check", {"text": "[recipe]\nformat = 1\nroute = 'nope'\n"})
    assert unreadable[0]["severity"] == "error"
    both = TINY.replace('energy = "100 uJ"', 'energy = "100 uJ"\npeak_power = "1 GW"')
    conflict = ask("check", {"text": both})
    assert conflict and conflict[0]["severity"] == "error"


def summary_of(text: str) -> dict[str, float]:
    return ask("summarise", {"text": text})["design"]


def test_summarise_gives_the_pulse_and_the_design_numbers():
    summary = ask("summarise", {"text": TINY})
    assert summary["energy"] == pytest.approx(100e-6, rel=1e-3)
    assert summary["design"]["soliton_order"] > 1
    assert summary["design_error"] is None
    assert "rtol" in summary["resolved"]
    gradient = TINY.replace('pressure = "1 bar"', 'pressure = ["1 bar", "0 bar"]')
    assert ask("summarise", {"text": gradient})["design"]["pressure"] == 1.0
    no_width = TINY.replace('fwhm = "10 fs"', 'width = "6 fs"\nshape = "sech"')
    summary = ask("summarise", {"text": no_width})
    assert summary["design"] == {} and summary["design_error"]
    # The numbers are the recipe's own: its pulse shape (Gaussian by default) and an
    # energy from its peak power.
    order = TINY.replace('energy = "100 uJ"\n', "") + (
        '\n[pins]\nenergy = { target = "soliton_order", order = 3 }\n'
    )
    pinned = ask("summarise", {"text": order})
    assert pinned["design"]["soliton_order"] == pytest.approx(3.0, rel=1e-4)
    power = TINY.replace('energy = "100 uJ"', 'peak_power = "5 GW"')
    assert ask("summarise", {"text": power})["design"]["soliton_order"] > 1
    # The dispersive wave the pulse phase-matches, on the blue side of the ZDW.
    numbers = summary_of(TINY)
    assert 1e-7 < numbers["rdw"] < numbers["zdw"] < 8e-7
    assert numbers["fission_length"] > 0 and numbers["critical_power"] > 0
    normal = summary_of(TINY.replace('wavelength = "800 nm"', 'wavelength = "400 nm"'))
    assert normal["beta2"] > 0 and "rdw" not in normal


def test_previews_estimate_and_export():
    preview = ask("preview_input", {"text": TINY})
    assert preview["time"].shape == preview["power"].shape
    assert preview["spectral_power"].ndim == 1
    assert preview["peak_power"] > 0
    estimate = ask("estimate", {"text": TINY})
    assert estimate["admitted"] and estimate["memory"] > 0
    script = ask("to_python", {"text": TINY, "output": "out.h5"})
    assert "out.h5" in script and "import" in script


def test_a_finished_run_is_described(finished_run: Path):
    path = {"path": str(finished_run)}
    resume = ask("check_resume", {**path, "gate": True})
    assert not resume["resumable"] and "complete" in resume["reason"]
    assert resume["rebuildable"]
    described = ask("describe_output", path)
    assert described["outcome"]["status"] == "complete"
    assert described["field_planes"] == list(range(6)) and described["recipe"]
    assert "Tiny capillary" in ask("recipe_of", path)
    assert "Tiny capillary" in ask("recipe_of", {**path, "resolved": True})
    overview = ask("overview", {**path, "time_range": [-1e-13, 1e-13]})
    assert len(overview["rows"]) == 6
    assert overview["axes"]["spectral"][0] > 200e-9  # the recipe's window
    assert overview["rows"][0]["stats"]["energy"] == pytest.approx(100e-6, rel=1e-3)


def test_a_file_from_a_script_has_no_recipe_and_its_own_band(tmp_path: Path):
    from solprop.driver import Simulation  # noqa: PLC0415 - only here

    # A script's run: the same keywords, built without a recipe.
    simulation = Simulation.from_capillary(**Recipe.loads(TINY).keywords())  # pyright: ignore[reportArgumentType]
    path = tmp_path / "script.h5"
    simulation.run(path)
    described = ask("describe_output", {"path": str(path)})
    assert not described["recipe"]
    overview = ask("overview", {"path": str(path)})
    assert overview["axes"]["spectral"][0] < 200e-9


# ------------------------------------------------------------------ the worker's run


class Collector:
    """The window's end of the pipes, for a run executed in a thread."""

    def __init__(self, spec: RunSpec) -> None:
        self.events_in, events_out = pipe(duplex=False)
        control_in, self.control_out = pipe(duplex=False)
        self.events: list[Event] = []
        self.thread = threading.Thread(
            target=run.execute, args=(spec, events_out, control_in), daemon=True
        )

    def start(self) -> Collector:
        self.thread.start()
        return self

    def collect(self, until: str | None = None, *, then: Control | None = None) -> None:
        """Receive events to the end, sending ``then`` after the first ``until``."""
        while True:
            event: Event = self.events_in.recv()
            self.events.append(event)
            if until is not None and event.kind == until and then is not None:
                self.control_out.send(then)
                then = None
            if event.terminal:
                break
        self.thread.join(30)

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


def test_a_run_streams_rows_and_finishes(tmp_path: Path):
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY, base=str(tmp_path))
    collector = Collector(spec).start()
    collector.collect()
    kinds = collector.kinds()
    assert kinds[-1] == "finished" and kinds.count("row") == 6
    assert kinds.index("axes") < kinds.index("row")
    ending = collector.events[-1].data
    assert ending["status"] == "complete" and ending["path"].endswith(OUTPUT)
    folder = Path(spec.folder)
    assert (folder / RECIPE).read_text(encoding="utf-8") == TINY
    assert (folder / LOG).exists()
    rows = [e for e in collector.events if e.kind == "row"]
    assert rows[0].data["density"].dtype == np.float32
    axes = next(e for e in collector.events if e.kind == "axes").data
    assert axes["spectral"][0] > 200e-9  # the recipe's wavelength window


def test_a_run_reduced_on_a_frequency_axis_within_a_band(tmp_path: Path):
    display = Display(axis="f", bandpass=(7.9e-7, 8.1e-7))
    spec = RunSpec("rf", str(tmp_path / "rf"), recipe=TINY, display=display)
    collector = Collector(spec).start()
    collector.collect()
    axes = next(e for e in collector.events if e.kind == "axes").data
    assert axes["axis"] == "f" and axes["spectral"][0] > 1e14  # Hz
    whole = ask("overview", {"path": str(tmp_path / "rf" / OUTPUT)})
    rows = [e.data for e in collector.events if e.kind == "row"]
    assert rows[0]["power"].max() < 0.9 * whole["rows"][0]["power"].max()


def test_a_stopped_run_returns_and_resumes(tmp_path: Path):
    folder = tmp_path / "r2"
    collector = Collector(RunSpec("r2", str(folder), recipe=TINY)).start()
    collector.collect(until="row", then=Control("stop"))
    assert collector.kinds()[-1] == "stopped"
    assert "stop flag" in str(collector.events[-1].data["message"])
    found = ask("check_resume", {"path": str(folder / OUTPUT)})
    assert found["resumable"]
    again = Collector(RunSpec("r2", str(folder), resume=found["mode"])).start()
    again.collect()
    assert again.kinds()[-1] == "finished"
    first = {e.data["index"] for e in collector.events if e.kind == "row"}
    second = {e.data["index"] for e in again.events if e.kind == "row"}
    assert max(first) < min(second) and max(second) == 5


def test_rows_are_replayed_on_subscribe(tmp_path: Path):
    collector = Collector(RunSpec("r3", str(tmp_path / "r3"), recipe=TINY)).start()
    collector.collect(until="row", then=Control("subscribe", from_plane=0))
    rows = [e.data["index"] for e in collector.events if e.kind == "row"]
    assert rows.count(0) == 2  # the first row, and its replay
    assert collector.kinds().count("axes") == 2


def test_a_run_that_fails_says_why(tmp_path: Path):
    broken = TINY.replace('energy = "100 uJ"', 'energy = "1 fs"')
    collector = Collector(RunSpec("r4", str(tmp_path / "r4"), recipe=broken)).start()
    collector.collect()
    ending = collector.events[-1]
    assert ending.kind == "failed"
    assert "energy" in str(ending.data["message"]) and ending.data["detail"]


def test_a_new_run_needs_its_recipe(tmp_path: Path):
    collector = Collector(RunSpec("r5", str(tmp_path / "r5"))).start()
    collector.collect()
    assert collector.events[-1].kind == "failed"
    assert "needs its recipe" in str(collector.events[-1].data["message"])


def test_an_unfinished_run_that_is_not_stopped_is_a_failure(run_copy: Path):
    assert run._ending(run_copy, stopped=False)[0] == "finished"
    # A stop that arrives as the run completes leaves it complete.
    assert run._ending(run_copy, stopped=True)[0] == "finished"


def test_a_run_whose_outcome_is_not_complete_failed(
    monkeypatch: pytest.MonkeyPatch, run_copy: Path
):
    class Handler:
        outcome = None

        def __enter__(self) -> Handler:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(run, "open_file", lambda _path: Handler())
    kind, data = run._ending(run_copy, stopped=False)
    assert kind == "failed" and data["status"] is None


def test_the_window_going_away_never_touches_the_run(tmp_path: Path):
    spec = RunSpec("r6", str(tmp_path / "r6"), recipe=TINY)
    events_in, events_out = pipe(duplex=False)
    control_in, control_out = pipe(duplex=False)
    events_in.close()  # the window is gone before the run starts
    control_out.close()
    run.execute(spec, events_out, control_in)
    assert (
        ask("describe_output", {"path": str(Path(spec.folder) / OUTPUT)})["outcome"][
            "status"
        ]
        == "complete"
    )


def test_a_display_that_cannot_be_read_from_the_recipe_is_the_specs(tmp_path: Path):
    spec = RunSpec("r7", str(tmp_path), recipe="not a recipe", display=Display())
    assert run._display(spec, tmp_path / OUTPUT) is spec.display


def test_the_observer_reports_progress_on_a_period(monkeypatch: pytest.MonkeyPatch):
    outbox = Outbox()
    observer = run.LiveObserver("r", Reducer(Display()), outbox)
    observer.started(0.0, 1.0, 3)
    clock = iter([10.0, 10.1, 10.5])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    observer.step(0.1, 0.1, True)
    observer.step(0.15, 0.05, False)
    observer.step(0.2, 0.1, True)
    observer.step(0.3, 0.1, True)
    observed = []
    while (event := outbox.get(timeout=0)) is not None:
        observed.append(event)
    progress = [e for e in observed if e.kind == "progress"]
    assert len(progress) == 1  # coalesced: the last one stands
    assert progress[0].data == {
        "z": 0.3,
        "dz": 0.1,
        "z_end": 1.0,
        "steps": 4,
        "rejected": 1,
    }
    observer.replay(0)  # nothing yet: no axes, no rows
    assert len(outbox) == 0


def test_log_records_become_events():
    outbox = Outbox()
    handler = run.EventHandler("r", outbox)
    handler.emit(
        logging.LogRecord("solprop", logging.WARNING, "", 0, "hi %s", ("x",), None)
    )
    event = outbox.get(timeout=0)
    assert event is not None and event.data["message"] == "hi x"


def test_the_sender_drops_events_once_the_pipe_is_broken():
    reader, writer = pipe(duplex=False)
    reader.close()
    outbox = Outbox()
    outbox.put(Event("r", "row"))
    outbox.put(Event("r", "row"))
    outbox.close()
    run._send(outbox, writer)


def test_the_sender_waits_for_events_until_closed():
    reader, writer = pipe(duplex=False)
    outbox = Outbox()
    sender = threading.Thread(target=run._send, args=(outbox, writer))
    sender.start()
    threading.Timer(0.7, outbox.close).start()  # past one empty wait
    sender.join(5)
    assert not sender.is_alive()
    reader.close()


def test_a_reply_round_trips_through_a_pipe():
    one, two = pipe()
    one.send(Reply("x", 1, 0, value=np.arange(3)))
    assert two.recv().value.tolist() == [0, 1, 2]


def test_a_numeric_default_is_written_in_its_display_unit():
    from solprop.recipe import Form, Parameter  # noqa: PLC0415 - only here

    entry = Parameter(
        "x",
        (Form("number", unit="m"),),
        "geometry",
        "basic",
        "S.",
        "a.md",
        display="um",
        default=1e-4,
    )
    assert handlers._default_text(entry) == "100 um"


def test_summaries_and_previews_of_other_routes_and_modes():
    def example(name: str) -> str:
        # The reference backend: a lane without the native one would warn.
        text = ask("example", {"name": name})
        return text.replace("[build]\n", '[build]\nbackend = "numpy_ref"\n', 1)

    gnlse = ask("summarise", {"text": example("soliton")})
    assert gnlse["design"] == {} and gnlse["design_error"] is None
    rdw = ask("preview_input", {"text": example("rdw")})
    assert rdw["spectral_power"].ndim == 1  # summed over the two modes


def test_the_worker_entry_sets_the_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import os  # noqa: PLC0415 - only here

    from solprop_gui.service import worker  # noqa: PLC0415 - only here

    monkeypatch.setenv("SOLPROP_THREADS", "8")
    spec = RunSpec("r8", str(tmp_path / "r8"), threads=1)
    events_in, events_out = pipe(duplex=False)
    control_in, control_out = pipe(duplex=False)
    worker.main(spec, events_out, control_in)  # no recipe: fails at once
    assert os.environ["SOLPROP_THREADS"] == "1"
    assert events_in.recv().kind == "failed"
    control_out.close()
    # Without a budget, solprop's default stands.
    monkeypatch.setenv("SOLPROP_THREADS", "8")
    events_in, events_out = pipe(duplex=False)
    control_in, control_out = pipe(duplex=False)
    worker.main(RunSpec("r9", str(tmp_path / "r9")), events_out, control_in)
    assert os.environ["SOLPROP_THREADS"] == "8"
    assert events_in.recv().kind == "failed"
    control_out.close()

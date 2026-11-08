# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run lifecycle end to end, through the real kernel and workers, without Qt.

Edit, check, save, export, run, stop, reopen, resume and compare: the first
milestone's path through the service (design 16).
"""

import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from solprop.service.client import KernelClient
from solprop.service.launch import prepare, start_direct
from solprop.service.messages import Event, Reply
from solprop.service.record import read_events, request_stop
from solprop.service.worker import OUTPUT, RunSpec

from conftest import TINY
from solprop_gui.model.recipe import RecipeModel
from solprop_gui.model.workspace import list_runs, new_run_folder

TIMEOUT = 120.0


class Kernel:
    """A kernel client whose replies are waited for in order."""

    def __init__(self) -> None:
        self.replies: queue.Queue[Reply] = queue.Queue()
        self.client = KernelClient(self.replies.put)

    def ask(self, kind: str, payload: dict[str, object], revision: int = 0) -> object:
        request = self.client.request(kind, payload, revision=revision)
        while True:
            reply = self.replies.get(timeout=TIMEOUT)
            if reply.request_id == request:
                assert reply.ok, reply.detail or reply.error
                return reply.value


def run_until(spec: RunSpec, stop_after_rows: int | None = None) -> list[Event]:
    """Run a spec as a job, following its record, stopping it after some rows."""
    folder = prepare(spec)
    start_direct(folder)
    seen: list[Event] = []
    offset = 0
    deadline = time.monotonic() + TIMEOUT
    while not any(event.terminal for event in seen):
        assert time.monotonic() < deadline, "the run did not end"
        events, offset = read_events(folder, offset)
        seen.extend(events)
        rows = sum(e.kind == "row" for e in seen)
        if stop_after_rows is not None and rows >= stop_after_rows:
            request_stop(folder)
            stop_after_rows = None
        time.sleep(0.05)
    return seen


@pytest.fixture(scope="module")
def kernel():
    kernel = Kernel()
    yield kernel
    kernel.client.close()


def test_the_whole_lifecycle(kernel: Kernel, tmp_path: Path):
    longer = TINY.replace('length = "5 cm"', 'length = "15 cm"').replace(
        "n_save = 6", "n_save = 16"
    )
    # Edit and check: a wrong value is refused at its path, a good one accepted.
    model = RecipeModel(longer)
    model.set("energy", "1 fs")
    kernel.client.revision = model.revision
    issues = kernel.ask("check", {"text": model.to_text()}, model.revision)
    assert issues and issues[0]["parameter"] == "energy"  # pyright: ignore[reportIndexIssue]
    model.set("energy", "120 uJ")
    kernel.client.revision = model.revision
    assert kernel.ask("check", {"text": model.to_text()}, model.revision) == []

    # Save, and export: the script runs on its own and makes the same run.
    saved = model.save(tmp_path / "setup.toml")
    assert RecipeModel.load(saved).get("energy") == "120 uJ"
    script = str(
        kernel.ask("to_python", {"text": model.to_text(), "output": "script.h5"})
    )
    (tmp_path / "run.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, "run.py"], cwd=tmp_path, check=True, timeout=600)

    # Run, and stop part-way.
    workspace = tmp_path / "workspace"
    folder = new_run_folder(workspace, model.title)
    spec = RunSpec(folder.name, str(folder), recipe=model.to_text(), base=str(tmp_path))
    events = run_until(spec, stop_after_rows=3)
    assert events[-1].kind == "stopped"
    output = str(folder / OUTPUT)

    # Reopen: solprop says it can continue, and gives its setup back.
    resume = kernel.ask("check_resume", {"path": output, "gate": True})
    assert resume["resumable"] and resume["rebuildable"]  # pyright: ignore[reportIndexIssue]
    setup = RecipeModel(
        str(kernel.ask("recipe_of", {"path": output, "resolved": True}))
    )
    assert setup.route == "capillary" and setup.title == model.title

    # Resume to the end.
    resumed = run_until(RunSpec(folder.name, str(folder), resume=resume["mode"]))  # pyright: ignore[reportIndexIssue]
    assert resumed[-1].kind == "finished"
    described = kernel.ask("describe_output", {"path": output})
    assert described["outcome"]["status"] == "complete"  # pyright: ignore[reportIndexIssue]
    assert [run.folder for run in list_runs(workspace)] == [folder]

    # Compare: the run and the exported script's run end on the same spectrum.
    window = {"wavelength_range": [200e-9, 2e-6]}  # the script's file has no recipe
    ours = kernel.ask("overview", {"path": output, **window})
    theirs = kernel.ask("overview", {"path": str(tmp_path / "script.h5"), **window})
    last_ours = np.asarray(ours["rows"][-1]["density"]["wavelength"])  # pyright: ignore[reportIndexIssue]
    last_theirs = np.asarray(theirs["rows"][-1]["density"]["wavelength"])  # pyright: ignore[reportIndexIssue]
    assert len(ours["rows"]) == len(theirs["rows"]) == 16  # pyright: ignore[reportIndexIssue, reportArgumentType]
    # A resume agrees with an unbroken run to the integrator's tolerance, not bitwise.
    np.testing.assert_allclose(
        last_ours, last_theirs, rtol=0, atol=1e-2 * last_ours.max()
    )

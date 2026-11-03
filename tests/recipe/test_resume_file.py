# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resuming from a file alone, and what a file allows (SEP 41, S2 and S17)."""

import dataclasses
import json
import threading

import h5py
import numpy as np
import pytest

from solprop.cli import main
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError, OutputError
from solprop.log import NullReporter
from solprop.output import SCHEMA_VERSION, resumability
from solprop.output import resumable as resumable_module
from solprop.output._writer_lock import WriterLock
from solprop.output.hdf5 import open_file
from solprop.recipe import Recipe, RecipeError, resume_file
from solprop.stepper.multistep import Multistep

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]


class CrashError(RuntimeError):
    """What a crashed run raises, standing in for a machine going away."""


class StopAt(NullReporter):
    """Set a flag at the first saved plane."""

    def __init__(self, flag):
        self.flag = flag

    def plane(self, z, summary):
        self.flag.set()


class CrashAt(NullReporter):
    """Raise at the second saved plane."""

    def __init__(self):
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen == 2:
            raise CrashError


def _soliton(**changes):
    recipe = Recipe.example("soliton")
    for key, value in changes.items():
        recipe.set(key, value)
    return recipe


def _stopped(path, recipe=None):
    flag = threading.Event()
    (recipe or _soliton()).run(path, progress=StopAt(flag), stop=flag)


def _last_field(path):
    with open_file(path) as handler:
        return np.asarray(handler["field"][-1]), handler.outcome


def test_a_stopped_file_says_where_it_continues_and_resumes_from_itself(tmp_path):
    path = tmp_path / "stopped.h5"
    _stopped(path)
    answer = resumability(path, gate=True)
    assert answer.resumable
    assert answer.mode is True
    assert answer.z is not None and 0 < answer.z < 1.33
    assert answer.recipe == "stored" and answer.rebuildable
    assert answer.comparison is not None and answer.comparison.continues
    resume_file(path, progress="none")
    field, outcome = _last_field(path)
    assert outcome is not None and outcome.status == "complete"
    _soliton().run(tmp_path / "whole.h5", progress="none")
    whole, _ = _last_field(tmp_path / "whole.h5")
    # An adaptive run resumed agrees with the uninterrupted one to the controller's
    # tolerance, here rtol 1e-7 on a soliton (class T).
    assert np.max(np.abs(field - whole)) <= 1e-5 * np.max(np.abs(whole))
    done = resumability(path)
    assert not done.resumable
    assert done.reason is not None and "complete" in done.reason


def test_the_command_resumes_a_file(tmp_path):
    path = tmp_path / "stopped.h5"
    _stopped(path)
    assert main(["resume", str(path), "--progress", "none"]) == 0
    assert _last_field(path)[1].status == "complete"  # type: ignore[union-attr]


def test_a_file_from_a_script_continues_only_from_its_script(tmp_path):
    simulation = _soliton().build()
    simulation._recipe = None
    flag = threading.Event()
    simulation.run(tmp_path / "script.h5", progress=StopAt(flag), stop=flag)
    answer = resumability(tmp_path / "script.h5", gate=True)
    assert answer.resumable
    assert answer.recipe == "none" and not answer.rebuildable
    assert answer.comparison is None
    with pytest.raises(RecipeError, match="written from a script"):
        resume_file(tmp_path / "script.h5")


def test_a_newer_recipe_format_is_named(tmp_path):
    path = tmp_path / "newer.h5"
    _stopped(path)
    with h5py.File(path, "r+") as file:
        file["recipe"].attrs["format"] = 99
    answer = resumability(path)
    assert answer.recipe == "newer" and answer.recipe_format == 99
    assert answer.resumable


def test_a_live_writer_is_named(tmp_path):
    path = tmp_path / "live.h5"
    _stopped(path)
    lock = WriterLock(path)
    lock.acquire()
    try:
        answer = resumability(path)
    finally:
        lock.release()
    assert not answer.resumable
    assert answer.reason == "a writer still holds the file"


def test_a_file_of_another_schema_is_named(tmp_path):
    path = tmp_path / "old.h5"
    _stopped(path)
    with h5py.File(path, "r+") as file:
        file.attrs["schema_version"] = 1
    answer = resumability(path)
    assert not answer.resumable
    assert answer.reason is not None and "output schema 1" in answer.reason


def test_a_failure_before_any_plane_leaves_nothing_to_continue(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as file:
        file.attrs["schema_version"] = SCHEMA_VERSION
    answer = resumability(path)
    assert not answer.resumable
    assert answer.reason is not None and "no state" in answer.reason


def test_a_file_of_an_earlier_build_s_flat_orders_is_named(tmp_path):
    # An earlier development build of this schema wrote a Multistep run's orders one
    # per step: the file is said not to continue, for that reason, before a resume.
    path = tmp_path / "flat.h5"
    simulation = Simulation.from_gnlse(
        0.11, 0.05, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=100e-15,
        peak_power=9.8, shape="sech", n_save=4, integrator=Multistep(),
        backend="numpy_ref",
    )  # fmt: skip
    with pytest.raises(CrashError):
        simulation.run(path, progress=CrashAt())
    assert resumability(path).resumable
    with h5py.File(path, "r+") as file:
        dataset = file["meta/outcome"]
        assert isinstance(dataset, h5py.Dataset)
        outcome = json.loads(dataset[()])
        runs = outcome["logical_calls"]["orders"]
        flat = [order for order, count in runs for _ in range(count)]
        outcome["logical_calls"]["orders"] = flat
        dataset[()] = json.dumps(outcome)
    answer = resumability(path)
    assert not answer.resumable
    assert answer.reason is not None
    assert "flat form an earlier development build wrote" in answer.reason


def test_a_forced_file_needs_force(tmp_path):
    path = tmp_path / "forced.h5"
    _stopped(path)
    changed = _soliton(gamma=0.12)
    flag = threading.Event()
    changed.build().run(path, resume="force", progress=StopAt(flag), stop=flag)
    answer = resumability(path)
    assert answer.resumable and answer.mode == "force"


def test_an_interpolated_last_plane_needs_interpolated(tmp_path):
    path = tmp_path / "dense.h5"
    recipe = _soliton(interpolate_planes=True)
    with pytest.raises(CrashError):
        recipe.run(path, progress=CrashAt())
    answer = resumability(path)
    assert answer.resumable and answer.mode == "interpolated"
    resume_file(path, "interpolated", progress="none")


def test_the_gate_shows_a_moved_record(tmp_path):
    path = tmp_path / "moved.h5"
    _stopped(path)
    # Edit the stored resolved recipe, as a later solprop might rebuild it differently.
    with h5py.File(path, "r+") as file:
        group = file["recipe"]
        assert isinstance(group, h5py.Group)
        stored = group["resolved"]
        assert isinstance(stored, h5py.Dataset)
        text = bytes(stored[()]).decode()
        del group["resolved"]
        group.create_dataset(
            "resolved",
            data=text.replace('gamma = "0.11 W^-1/m"', 'gamma = "0.12 W^-1/m"'),
            dtype=h5py.string_dtype(),
        )
    answer = resumability(path, gate=True)
    assert answer.comparison is not None
    assert not answer.comparison.continues
    assert "gamma" in answer.comparison.summary()


def test_a_received_file_does_not_choose_its_own_budget(tmp_path, monkeypatch):
    path = tmp_path / "budget.h5"
    recipe = _soliton()
    document = recipe.to_dict()
    document["execution"] = {"memory_budget": "none"}
    _stopped(path, Recipe.from_dict(document))
    seen = {}
    original = Simulation.run

    def run(self, output, **keywords):
        seen["budget"] = keywords.get("memory_budget")
        return original(self, output, **keywords)

    monkeypatch.setattr(Simulation, "run", run)
    resume_file(path, progress="none")
    assert isinstance(seen["budget"], int)
    resume_file(tmp_path / "budget.h5", memory_budget="warn:1 TiB", progress="none")
    assert seen["budget"] == "warn:1 TiB"


def test_a_file_that_is_not_hdf5_is_refused(tmp_path):
    path = tmp_path / "notes.h5"
    path.write_text("not an HDF5 file")
    with pytest.raises(OutputError, match=r"cannot read .*notes\.h5"):
        resumability(path)


def test_a_last_plane_without_a_field_and_no_checkpoint_cannot_continue(
    tmp_path, monkeypatch
):
    path = tmp_path / "fieldless.h5"
    _stopped(path)
    original = resumable_module.resume

    def fieldless(where):
        state = original(where)
        plane = dataclasses.replace(state.plane, field=None, noise_phase=None)
        return dataclasses.replace(state, plane=plane, checkpoint=None)

    monkeypatch.setattr(resumable_module, "resume", fieldless)
    answer = resumability(path)
    assert not answer.resumable
    assert answer.reason == (
        "its last plane kept no field and no terminal checkpoint follows it"
    )


def test_the_gate_reports_a_rebuild_that_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "refused.h5"
    _stopped(path)

    def refuse(self):
        msg = "the fill is no longer available"
        raise DriverError(msg)

    monkeypatch.setattr(Recipe, "build", refuse)
    answer = resumability(path, gate=True)
    assert answer.resumable
    assert answer.comparison is None
    assert answer.gate_error is not None
    assert "the fill is no longer available" in answer.gate_error


def test_a_node_budget_given_to_the_resume_reaches_the_rebuild(tmp_path, monkeypatch):
    path = tmp_path / "capillary.h5"
    _stopped(path, Recipe.example("overview"))
    seen = []
    original = Recipe.build

    def build(self):
        seen.append(self.to_dict()["build"].get("node_budget"))
        return original(self)

    monkeypatch.setattr(Recipe, "build", build)
    resume_file(path, node_budget=2**26, progress="none")
    assert seen == [2**26]
    assert _last_field(path)[1].status == "complete"  # type: ignore[union-attr]

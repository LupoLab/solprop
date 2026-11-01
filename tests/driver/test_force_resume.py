# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``resume="force"`` on a capillary: what it continues, what it refuses (SEP 42).

A short argon capillary on ``numpy_ref``, stopped part-way through real files. A
forced pressure change continues and marks the file; a change the stored arrays cannot
hold -- plasma off, another mode count, noise added -- is refused before anything is
computed or written; and the readers respect the mark.
"""

import json

import h5py
import pytest

from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import OutputError, ProcessingError, SolpropWarning
from solprop.grid import FieldGrid
from solprop.log import NullReporter
from solprop.output import open_file
from solprop.output.hdf5 import HDF5Output, _dataset, resume
from solprop.processing import field_plane

LENGTH = 4e-4  # m
N_SAVE = 5

pytestmark = [
    # A short field grid and a tight window: what the window holds is not tested here.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates a packet"),
]


class CrashError(RuntimeError):
    """What an interrupted run raises."""


class Interrupt(NullReporter):
    """Stop the run once it has seen `after` planes."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise CrashError(f"stopped after plane {self.seen}")


def capillary(**changes):
    """A short argon capillary, plasma on, four planes."""
    options = {"pressure": 1.0, "radius": 50e-6, "plasma": True, "noise": None}
    options |= changes
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    return Simulation.from_capillary(
        options.pop("radius"),
        LENGTH,
        "Ar",
        options.pop("pressure"),
        wavelength=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-6,
        n_r=8,
        n_save=N_SAVE,
        backend="numpy_ref",
        **options,
    )


@pytest.fixture
def stopped(tmp_path):
    """A capillary file stopped after two planes."""
    path = tmp_path / "run.h5"
    with pytest.raises(CrashError):
        capillary().run(path, progress=Interrupt(2))
    return path


def snapshot(path):
    """Return what a refusal must leave unchanged: the planes and the outcome."""
    with h5py.File(path, "r") as file:
        return int(_dataset(file, "z").shape[0]), _dataset(file, "meta/outcome")[()]


@pytest.mark.parametrize("change", ["pressure", "radius"])
def test_a_forced_change_continues_and_marks_the_file(stopped, change):
    before = resume(stopped).n_saved
    changed = capillary(
        **{"pressure": 1.2} if change == "pressure" else {"radius": 55e-6}
    )
    with pytest.raises(OutputError, match='pass resume="force"'):
        changed.run(stopped, resume=True)
    with pytest.warns(SolpropWarning, match="forcing a resume"):
        handler = changed.run(stopped, resume="force", progress="none")
    assert handler.n_saved == N_SAVE
    outcome = resume(stopped).outcome
    assert outcome is not None and outcome.status == "complete"
    [entry] = outcome.resumes
    assert entry["forced"] is True
    assert entry["from_plane"] == before - 1
    # The entry is the grouped text, bounded however many paths moved.
    assert len(json.dumps(entry)) < 4096


def test_the_readers_respect_a_forced_file(stopped):
    with pytest.warns(SolpropWarning, match="forcing a resume"):
        capillary(pressure=1.2).run(stopped, resume="force", progress="none")
    with pytest.warns(
        SolpropWarning, match="was continued with resume='force'"
    ) as caught:
        reader = open_file(stopped)
    assert len(caught) == 1
    with reader:
        forced_at = reader.outcome.forced_positions[0]  # pyright: ignore[reportOptionalMemberAccess] - a finished run has an outcome
        positions = list(reader["z"])
        earlier = max(i for i, z in enumerate(positions) if z <= forced_at)
        assert field_plane(reader, earlier).geometry.z == positions[earlier]
        with pytest.raises(ProcessingError, match="beyond the forced resume"):
            field_plane(reader, -1)
        with pytest.raises(ProcessingError, match="beyond the forced resume"):
            field_plane(reader, -1, source=capillary(pressure=1.2))
    # An internal reader stays quiet.
    with HDF5Output(stopped, readonly=True) as quiet:
        assert quiet.n_saved == N_SAVE
    # The original script now finds the file finished and is warned, not continued.
    with pytest.warns(SolpropWarning, match="was continued with resume='force'"):
        capillary().run(stopped, resume=True)


def test_force_refuses_statistics_the_file_cannot_hold(stopped):
    before = snapshot(stopped)
    with pytest.raises(OutputError, match="its statistics do not fit") as refused:
        capillary(plasma=False).run(stopped, resume="force")
    assert "nothing was computed or written" in str(refused.value)
    assert snapshot(stopped) == before
    assert not resume(stopped).outcome.forced_positions  # pyright: ignore[reportOptionalMemberAccess] - a stopped run has an outcome


@pytest.mark.parametrize(
    "change", [{"modes": 2}, {"noise": "input", "seed": 3}], ids=["modes", "noise"]
)
def test_force_does_not_cross_the_layout(stopped, change):
    before = snapshot(stopped)
    with pytest.raises(OutputError, match="the layout differs"):
        capillary(**change).run(stopped, resume="force")
    assert snapshot(stopped) == before

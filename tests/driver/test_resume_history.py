# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A resumed Adams run continues its history rather than starting it afresh.

The output keeps the march's state -- the accepted state as the backend held it, the
history's rows, the order and the proposed step -- at every save plane and terminal
checkpoint (``MarchSnapshot``). A resume at that position continues from it, so on
every engine a run stopped and resumed continues the uninterrupted run's march; where no
snapshot fits it starts the history afresh, as before, and says so.

These runs are adaptive, so a resumed run is not promised the uninterrupted run's steps
or bits: rounding may move a controller decision (SEP 25 section 4.4). Each run is held
to an accuracy envelope instead, against the same case at a hundredth of its tolerance
(``tests/march_bounds.py``), and the continuation itself is witnessed by the snapshot
being taken up rather than set aside.
"""

import dataclasses
import inspect
import logging
import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import solprop.driver.run as run_module
from backends.test_exponential_varying import EXPONENTIAL, gradient
from driver.test_resume import LENGTH, N_SAVE, CrashError, run_until
from flow_parity import LENGTH as FLOW_LENGTH
from flow_parity import build as flow_case
from march_bounds import TIGHTER, assert_in_envelope
from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import OutputError, SolpropWarning
from solprop.log import NullReporter
from solprop.output import HDF5Output
from solprop.output.hdf5 import open_file, resume
from solprop.output.schema import MarchSnapshot, SavedPlane, _example_metadata
from solprop.stepper import Multistep, RungeKutta
from solprop.stepper.multistep import ComparisonMultistep

INTERACTION = Multistep(formulation="interaction")

RTOL = 1e-6
"""The cases' relative tolerance; their references take a hundredth of it."""

AFRESH = "the Adams history starts afresh"
"""What a resume logs when it sets the march's snapshot aside."""


def soliton(build_simulation, backend, resume_history=None, rtol=RTOL, **keywords):
    """Return the soliton fibre, adaptive, on `backend`, on the Adams march."""
    # The GNLSE's default is the Runge-Kutta march (SEP 37); this file is about the
    # Adams history, so the soliton takes Multistep() unless a test names another.
    keywords.setdefault("integrator", Multistep())

    def build():
        simulation = build_simulation(
            length=LENGTH,
            n_save=N_SAVE,
            backend=backend,
            rtol=rtol,
            **{"integrator": Multistep(), **(keywords)},
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, resume_history=resume_history
        )
        return simulation

    return build


def varying(backend, rtol=RTOL):
    """Return a helium capillary along a pressure gradient: a compensated march."""

    def build():
        return gradient(integrator=EXPONENTIAL, rtol=rtol, backend=backend)

    return build


def coupled(backend, rtol=RTOL):
    """Return a coherent wall along a gradient: a frozen-residual coupled flow."""

    def build():
        return flow_case(
            "gradient",
            schedule=None,
            dz_max=FLOW_LENGTH,
            rtol=rtol,
            integrator=Multistep(),
            backend=backend,
            save_planes=[FLOW_LENGTH / 4, FLOW_LENGTH / 2, FLOW_LENGTH],
            n_save=None,
        )

    return build


def fields(path):
    """Return the planes' positions and fields stored in a closed output file."""
    with open_file(path) as stored:
        return np.asarray(stored["z"]), np.asarray(stored["field"])


def assert_accurate_run(whole, part, reference):
    """Assert a resumed file is the uninterrupted run's to its accuracy envelope.

    The save positions are prescribed, so they agree exactly; the fields are each
    held to four times the uninterrupted run's own departure from `reference`, the
    case at a hundredth of the tolerance (``march_bounds.assert_in_envelope``).
    """
    z_whole, field_whole = fields(whole)
    z_part, field_part = fields(part)
    _, field_reference = fields(reference)
    assert np.array_equal(z_whole, z_part)
    assert_in_envelope([field_part], field_reference, field_whole, what="the planes")


def tight_run(build_tight, path):
    """Run the reference at :data:`march_bounds.TIGHTER` the tolerance into `path`."""
    build_tight().run(path, progress="none")
    return path


CASES = {
    "soliton-numpy_ref": lambda build, rtol=RTOL: soliton(
        build, "numpy_ref", rtol=rtol
    ),
    "soliton-numba_cpu": lambda build, rtol=RTOL: soliton(
        build, "numba_cpu", rtol=rtol
    ),
    "interaction-numba_cpu": lambda build, rtol=RTOL: soliton(
        build, "numba_cpu", integrator=INTERACTION, rtol=rtol
    ),
    "varying-numpy_ref": lambda _, rtol=RTOL: varying("numpy_ref", rtol),
    "varying-numba_cpu": lambda _, rtol=RTOL: varying("numba_cpu", rtol),
    "coupled-numpy_ref": lambda _, rtol=RTOL: coupled("numpy_ref", rtol),
}


@pytest.mark.medium
@pytest.mark.parametrize("case", sorted(CASES))
# The coupled case's short window and marginal wall, as its own parity cases run them.
@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates a packet")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_resumed_run_is_the_uninterrupted_one(
    build_simulation, tmp_path, case, caplog
):
    build = CASES[case](build_simulation)
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        CASES[case](build_simulation, rtol=RTOL / TIGHTER), tmp_path / "tight.h5"
    )
    run_until(build(), part, after=2)
    state = resume(part)
    assert state.march is not None
    assert state.march.z == state.plane.z
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        build().run(part, resume=True, progress="none")
    # The snapshot was taken up: the history continued rather than restarting.
    assert AFRESH not in caplog.text
    assert_accurate_run(whole, part, reference)


@pytest.mark.medium
@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
def test_a_run_stopped_between_planes_continues_from_its_checkpoint(
    build_simulation, tmp_path, monkeypatch, backend_class, caplog
):
    # A failing source evaluation -- a TDSE provider gone, a budget spent -- stops the
    # march at its last accepted step, between save planes; the run writes a terminal
    # checkpoint there and the march's state beside it.
    name = "numpy_ref" if backend_class is NumpyRefBackend else "numba_cpu"
    build = soliton(build_simulation, name)
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        soliton(build_simulation, name, rtol=RTOL / TIGHTER), tmp_path / "tight.h5"
    )
    original = backend_class._adams_lab_source
    planes = [0.0, *build().settings.save_planes.tolist()]
    positions = []

    def failing(self, field, position, *args, **keywords):
        # The first call after a step accepted strictly between two planes past the
        # first: the run stops there, with no plane to resume from.
        last = positions[-1] if positions else 0.0
        if last > planes[1] and not any(np.isclose(last, one) for one in planes):
            raise CrashError("the source failed")
        positions.append(position)
        return original(self, field, position, *args, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(backend_class, "_adams_lab_source", failing)
        with pytest.raises(CrashError):
            build().run(part, progress="none")
    state = resume(part)
    assert state.checkpoint is not None
    assert state.march is not None and state.march.z == state.checkpoint.plane.z
    with open_file(whole) as first:
        assert state.march.z not in np.asarray(first["z"]).tolist()
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        build().run(part, resume=True, progress="none")
    assert AFRESH not in caplog.text
    assert_accurate_run(whole, part, reference)


@pytest.mark.medium
def test_without_the_history_a_resume_starts_afresh(build_simulation, tmp_path):
    build = soliton(build_simulation, "numpy_ref", resume_history=False)
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    reference = tight_run(
        soliton(
            build_simulation, "numpy_ref", resume_history=False, rtol=RTOL / TIGHTER
        ),
        tmp_path / "tight.h5",
    )
    run_until(build(), part, after=2)
    assert resume(part).march is None
    build().run(part, resume=True, progress="none")
    found, truth = fields(part)[1], fields(whole)[1]
    # A fresh history at order one is a different trajectory, as accurate as the first.
    assert not np.array_equal(found, truth)
    assert_accurate_run(whole, part, reference)


@pytest.mark.medium
def test_a_snapshot_over_the_default_size_is_kept_only_on_request(
    build_simulation, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(run_module, "MARCH_SNAPSHOT_AUTO_BYTES", 1)
    path = tmp_path / "run.h5"
    with caplog.at_level(logging.INFO, logger="solprop.driver.run"):
        soliton(build_simulation, "numpy_ref")().run(path, progress="none")
    assert resume(path).march is None
    assert "resume_history=True keeps it" in caplog.text
    asked = tmp_path / "asked.h5"
    soliton(build_simulation, "numpy_ref", resume_history=True)().run(
        asked, progress="none"
    )
    assert resume(asked).march is not None


@pytest.mark.medium
def test_the_snapshot_is_rewritten_in_place(build_simulation, tmp_path):
    # HDF5 does not reclaim a deleted dataset's space: a snapshot recreated at every
    # plane would grow the file by the history's size each time.
    path = tmp_path / "run.h5"
    offsets = []

    class Watch(NullReporter):
        def plane(self, z, summary):
            with h5py.File(path, "r") as file:
                offsets.append(file["march_snapshot/rows/store"].id.get_offset())

    soliton(build_simulation, "numpy_ref")().run(path, progress=Watch())
    assert len(offsets) == N_SAVE - 1
    assert len(set(offsets)) == 1


def test_a_snapshot_from_another_run_is_not_continued(build_simulation):
    simulation = build_simulation(
        length=LENGTH, n_save=N_SAVE, backend="numpy_ref", integrator=Multistep()
    )
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[0]))
        snapshot = backend.march_snapshot()
        assert snapshot is not None
        backend.set_initial(problem.initial.field0, None)
        assert "resumes at z = 0.0 m" in str(backend.restore_march(snapshot))
        backend.set_initial(problem.initial.field0, None, z=snapshot.z)
        assert backend.restore_march(snapshot) is None
        moved = snapshot.__class__(
            **{**vars(snapshot), "method": repr(INTERACTION)},
        )
        backend.set_initial(problem.initial.field0, None, z=snapshot.z)
        assert "taken by" in str(backend.restore_march(moved))
    finally:
        backend.close()
    runge_kutta = build_simulation(
        length=LENGTH, n_save=N_SAVE, backend="numpy_ref", integrator=RungeKutta()
    )
    problem = runge_kutta.translate()
    backend = NumpyRefBackend(problem, grid=runge_kutta.grid)
    try:
        assert backend.march_snapshot() is None
        assert "Runge-Kutta" in str(backend.restore_march(snapshot))
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.skipif("native" not in available(), reason="needs solprop-native")
def test_native_keeps_its_history_resident_and_continues_it(
    build_simulation, tmp_path, caplog
):
    # The resident engine exports its rows into the file and installs them on the
    # resume (SEP 29 section 6.6); `test_native_resume_history` checks them exactly.
    build = soliton(build_simulation, "native")
    path = tmp_path / "run.h5"
    run_until(build(), path, after=2)
    march = resume(path).march
    assert march is not None and set(march.rows) == {"store"}
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        build().run(path, resume=True, progress="none")
    assert AFRESH not in caplog.text
    with open_file(path) as stored:
        assert np.asarray(stored["z"]).size == N_SAVE


def march(**changes):
    """Return a small hand-built march snapshot at z = 0.25 m, with `changes`."""
    fields = {
        "z": 0.25,
        "state": np.ones((1, 1, 1, 4), complex),
        "proposal": 0.1,
        "method": "M()",
        "order": 2,
        "rejections": 0,
        "last_rejected": False,
        "positions": np.array([0.2, 0.25]),
        "identities": ("accepted", "predicted"),
        "slots": np.array([0, 1]),
        "rows": {"store": np.zeros((4, 1, 1, 1, 4), complex)},
    }
    return MarchSnapshot(**{**fields, **changes})


def saved_with(path, *snapshots):
    """Write one plane at z = 0.25 m, then each snapshot in turn, into `path`."""
    metadata = _example_metadata()
    handler = HDF5Output(path)
    handler.open(metadata)
    try:
        handler.save(
            SavedPlane(z=0.25, field=np.ones((1, 1, metadata.grid.n_omega), complex))
        )
        for one in snapshots:
            handler.save_march(one)
    finally:
        handler.close()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        # Positions and slots of different lengths are refused by the runtime shape
        # check before validation; the identities are a tuple, which it cannot see.
        ({"identities": ("accepted",)}, "disagree in number"),
        ({"positions": np.array([0.2, np.nan])}, "must be finite"),
        ({"state": np.full((1, 1, 1, 4), np.inf, complex)}, "must be finite"),
        ({"z": math.inf}, "must be finite"),
        ({"proposal": math.nan}, "must be finite"),
    ],
)
def test_an_inconsistent_snapshot_is_refused(changes, message):
    with pytest.raises(OutputError, match=message):
        march(**changes)


def test_a_snapshot_cannot_be_saved_before_the_run_opens(tmp_path):
    with pytest.raises(OutputError, match="before open"):
        HDF5Output(tmp_path / "run.h5").save_march(march())


def test_a_rewritten_snapshot_holds_only_its_own_rows(tmp_path):
    # A compensated march's rows, then a plain one's: the rewrite in place must not
    # leave the first snapshot's sampled states and rates behind for a resume to read.
    path = tmp_path / "run.h5"
    store = np.zeros((4, 1, 1, 1, 4), complex)
    compensated = {"store": store, "states": store + 1, "rates": store + 2}
    saved_with(path, march(rows=compensated), march(rows={"store": store + 3}))
    snapshot = resume(path).march
    assert snapshot is not None
    assert set(snapshot.rows) == {"store"}
    assert np.array_equal(snapshot.rows["store"], store + 3)


def test_a_snapshot_taken_elsewhere_is_not_returned(tmp_path):
    # A snapshot not at the resume position -- the plane after it was saved, the
    # snapshot's own rewrite was not -- describes another state.
    path = tmp_path / "run.h5"
    saved_with(path, march(z=0.2, positions=np.array([0.15, 0.2])))
    assert resume(path).march is None


@pytest.mark.parametrize(
    ("damage", "missing"),
    [
        (lambda group: group.attrs.__delitem__("method"), "method"),
        (lambda group: group.__delitem__("slots"), "slots"),
        (lambda group: group.attrs.__setitem__("identities", "not json"), "Expecting"),
    ],
)
def test_an_unreadable_snapshot_is_ignored_with_a_warning(
    tmp_path, caplog, damage, missing
):
    path = tmp_path / "run.h5"
    saved_with(path, march())
    with h5py.File(path, "a") as file:
        damage(file["march_snapshot"])
    with caplog.at_level(logging.WARNING, logger="solprop.output.hdf5"):
        state = resume(path)
    assert state.march is None
    assert state.plane.z == 0.25
    assert "ignoring the unreadable march snapshot" in caplog.text
    assert missing in caplog.text


@pytest.mark.medium
def test_a_kept_history_is_set_aside_on_request(build_simulation, tmp_path, caplog):
    # The interrupted run kept its snapshot; the resume asks for a fresh history.
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    soliton(build_simulation, "numpy_ref")().run(whole, progress="none")
    run_until(soliton(build_simulation, "numpy_ref")(), part, after=2)
    assert resume(part).march is not None
    afresh = soliton(build_simulation, "numpy_ref", resume_history=False)()
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        afresh.run(part, resume=True, progress="none")
    assert "the Adams history starts afresh: resume_history=False" in caplog.text
    found, truth = fields(part)[1], fields(whole)[1]
    # Not the uninterrupted run: the history was not continued.
    assert not np.array_equal(found, truth)


@pytest.mark.medium
def test_rows_that_do_not_fit_the_store_start_the_history_afresh(
    build_simulation, tmp_path, caplog
):
    # A snapshot whose rows are not the engine's shape -- written by another build, or
    # damaged -- is refused by the backend, which says why; the run goes on afresh.
    build = soliton(build_simulation, "numpy_ref")
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    run_until(build(), part, after=2)
    reference = tight_run(
        soliton(build_simulation, "numpy_ref", rtol=RTOL / TIGHTER),
        tmp_path / "tight.h5",
    )
    with h5py.File(part, "a") as file:
        rows = file["march_snapshot/rows"]
        assert isinstance(rows, h5py.Group)
        store = np.asarray(rows["store"])
        del rows["store"]
        rows["store"] = store[:-1]
    assert resume(part).march is not None
    with pytest.warns(
        SolpropWarning, match="the Adams history starts afresh: the snapshot's 'store'"
    ):
        build().run(part, resume=True, progress="none")
    found, truth = fields(part)[1], fields(whole)[1]
    assert not np.array_equal(found, truth)
    # As in the resume without the history: a fresh history at order one is a
    # different trajectory, held to the same accuracy envelope.
    assert_accurate_run(whole, part, reference)


def test_rows_another_history_keeps_are_refused_by_name(build_simulation):
    # A plain history keeps only its samples; a snapshot carrying a compensated
    # history's states is another march's, and nothing of it is installed.
    simulation = build_simulation(
        length=LENGTH, n_save=N_SAVE, backend="numpy_ref", integrator=Multistep()
    )
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[0]))
        snapshot = backend.march_snapshot()
        assert snapshot is not None
        rows = {**snapshot.rows, "states": snapshot.rows["store"]}
        backend.set_initial(problem.initial.field0, None, z=snapshot.z)
        reason = backend.restore_march(dataclasses.replace(snapshot, rows=rows))
        assert reason == (
            "the numpy_ref backend cannot continue this Adams history: its engine "
            "keeps no ['states'] rows"
        )
        assert backend.march_snapshot() is None
    finally:
        backend.close()


def test_the_general_path_refuses_a_snapshot_it_cannot_install(build_simulation):
    # Called directly, a march that keeps no history store says so rather than
    # installing rows it has nowhere to put.
    simulation = build_simulation(
        length=LENGTH, n_save=N_SAVE, backend="numpy_ref", integrator=Multistep()
    )
    problem = simulation.translate()
    general = build_simulation(
        length=LENGTH,
        n_save=N_SAVE,
        backend="numpy_ref",
        integrator=ComparisonMultistep(order=3),
    )
    backends = [
        NumpyRefBackend(problem, grid=simulation.grid),
        NumpyRefBackend(general.translate(), grid=general.grid),
    ]
    try:
        backends[0].set_initial(problem.initial.field0, None)
        backends[0].propagate_to(float(problem.stepper.save_planes[0]))
        snapshot = backends[0].march_snapshot()
        assert snapshot is not None
        backends[1].set_initial(problem.initial.field0, None, z=snapshot.z)
        assert "general Adams path" in str(backends[1].restore_march(snapshot))
    finally:
        for backend in backends:
            backend.close()


def test_a_backend_without_a_march_resumes_without_one(build_simulation, recwarn):
    # The driver continues a history only on a backend that offers the hooks; one
    # without them (any `Backend`, not only a step driver) simply loads the plane.
    simulation = build_simulation(
        length=LENGTH, n_save=N_SAVE, backend="numpy_ref", integrator=Multistep()
    )
    # Unwrapped: the runtime type check would refuse a stand-in for either argument.
    continue_march = inspect.unwrap(Simulation._continue_march)
    continue_march(simulation, SimpleNamespace(), SimpleNamespace(march=march()))
    assert not recwarn.list

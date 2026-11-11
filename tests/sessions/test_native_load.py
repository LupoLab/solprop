# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend's own load, Adams record and timing probe (SEP 47 phase 5).

The native backend loads a start (``solprop_native._load``), restores a resumed Adams
history into its own record (``_NativeAdamsMarch``, no longer an ``AdamsMarch``) and
times single trials (``solprop_native._probe``) without solprop's legacy step driver.
These are held to what the reference does with the same inputs: its load
(``HostDriver.set_initial``), its refusals and its probe's trial.
"""

from typing import Any, cast

import numpy as np
import pytest

from sessions.executors import (
    _soliton_context,  # pyright: ignore[reportPrivateUsage]
    edged,
    low_storage,
    soliton,
)
from solprop.backends.multistep import CallLedger
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.monitor import BoundaryCheck
from solprop.stepper.multistep import Multistep

native_load = pytest.importorskip("solprop_native._load")


def _pair(problem: Any, request: Any, context: dict[str, Any]) -> tuple[Any, Any]:
    """Return a native and a reference backend for the same run."""
    native: Any = select("native", problem, request, **context)
    return native, select("numpy_ref", problem, request, **context)


def test_an_input_draw_is_added_at_the_launch_alone():
    # The input model's draw rides in the launched state once, as the reference loads
    # it; a resume at the launch takes none, and the clean columns are the launch's.
    check = BoundaryCheck(transverse_budget=10.0)
    problem, request, context = edged(check, noise="input")
    native, reference = _pair(problem, request, context)
    try:
        field = np.asarray(problem.initial.field0)
        z = float(problem.initial.z)
        # The load rule itself, whose record holds the field the state is given; the
        # backend's kept record drops it once the state holds it.
        loaded = native_load.load(
            problem, request.stepper, field, None, z=z, resume=False
        )
        kept = native._load(field, None, z=z, resume=False)
        assert kept.field.size == 0
        assert np.array_equal(kept.floor, loaded.floor)
        reference.set_initial(field, None, z=float(problem.initial.z))
        host = cast("Any", reference)
        assert loaded.floor is not None and loaded.noise is None
        assert np.array_equal(loaded.floor, host._input_floor)
        expected = np.asarray(field)[np.newaxis] + loaded.floor
        assert np.array_equal(loaded.field, expected.reshape(-1))
        assert loaded.clean_columns is not None
        assert np.array_equal(loaded.clean_columns, host._clean_columns)
        resumed = native_load.load(
            problem, request.stepper, field, None, z=z, resume=True
        )
        assert np.array_equal(resumed.field, np.ascontiguousarray(field).reshape(-1))
        assert resumed.clean_columns is None
    finally:
        native.close()
        reference.close()


def test_a_replay_resumes_at_its_edges_alone():
    problem, request = low_storage(*soliton())
    native: Any = select("native", problem, request, **_soliton_context())
    try:
        schedule = request.stepper.schedule
        assert schedule is not None
        field = np.asarray(problem.initial.field0)
        edge = float(schedule.edges[3])
        assert native._load(field, None, z=edge, resume=True).cursor == 3
        between = 0.5 * (edge + float(schedule.edges[4]))
        with pytest.raises(BackendError, match="one of its edges"):
            native._load(field, None, z=between, resume=True)
        with pytest.raises(BackendError, match="operator table"):
            native._load(field, None, z=-1.0, resume=True)
        with pytest.raises(BackendError, match="initial state has shape"):
            native._load(field[..., :-1], None, z=0.0, resume=False)
    finally:
        native.close()


def test_a_resumed_history_is_refused_where_it_cannot_continue():
    # The record installs nothing it refuses, as the reference's march refuses it.
    problem, request = soliton(integrator=Multistep())
    native: Any = select("native", problem, request, **_soliton_context())
    try:
        march = native._adams
        assert march is not None and type(march).__mro__[1] is object
        march.ledger.startup = 3
        with pytest.raises(BackendError, match="atomic"):
            native._restore(
                Start(field=None, z=0.0, ledgers={"logical": {}, "atomic": {"x": 1}}),
                native_load.load(
                    problem,
                    request.stepper,
                    np.asarray(problem.initial.field0),
                    None,
                    z=0.0,
                    resume=False,
                ),
            )
        record = CallLedger(startup=2, accepted_steps=5, orders=[1] * 5).to_record()
        march.restore_ledger(record)
        assert (march.ledger.startup, march.ledger.accepted_steps) == (2, 5)
        state = march.start_state(0.25, 1e-4, 0)
        assert state["ledger"]["startup"] == 2
        assert state["history"] == [] and state["z"] == 0.25
    finally:
        native.close()


@pytest.mark.parametrize("integrator", [None, Multistep()], ids=["rk", "adams"])
def test_the_probe_commits_nothing_and_its_trial_meets_the_reference(integrator):
    problem, request = soliton(integrator=integrator)
    native, reference = _pair(problem, request, _soliton_context())
    try:
        probe = native.timing_probe(Start.entrance(problem))
        loaded = native._native_state.field_to_host().copy()
        dz = float(request.stepper.dz_init)
        probe.trial()
        whole = probe.field()
        # The readback loaded the start again, exactly: nothing was committed.
        assert np.array_equal(native._native_state.field_to_host(), loaded)
        probe.trial(float(problem.initial.z), 0.5 * dz)
        half = probe.field()
        assert not np.array_equal(half, whole)
        with pytest.raises(BackendError, match="take a trial"):
            probe.field()
        expected = reference.timing_probe(Start.entrance(problem))
        expected.trial()
        oracle = expected.field()
        # One Dormand-Prince trial of the soliton from its launch: the two backends'
        # arithmetic agrees to a few roundings of the peak (SEP 25); measured 2.7e-19
        # of the peak on an Intel Xeon Gold 6240.
        scale = float(np.max(np.abs(oracle)))
        np.testing.assert_allclose(whole, oracle, rtol=0.0, atol=64e-16 * scale)
    finally:
        native.close()
        reference.close()


def test_a_low_storage_probe_counts_its_evaluations_and_has_no_field():
    problem, request = low_storage(*soliton())
    native: Any = select("native", problem, request, **_soliton_context())
    try:
        probe = native.timing_probe(Start.entrance(problem))
        for _ in range(3):
            probe.trial()
        # Four evaluations a step (SEP 24 section 5), physical ones.
        assert native.metadata()["source_evaluations"] == 12
        with pytest.raises(BackendError, match="no trial field"):
            probe.field()
        # The next load counts from zero, as a fresh backend's.
        native.timing_probe(Start.entrance(problem))
        assert native.metadata()["source_evaluations"] == 0
    finally:
        native.close()

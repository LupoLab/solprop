# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The TDSE binding through a real provider worker (SEP 49 section 7.2).

The worker serves Pyroraptor's adapter around the manufactured haCC provider of
``hacc_cases`` (``tests/source_adapters.py``), which this process never imports as its
provider. The same provider served in this process (``source_channel``) is the direct
call the worker must reproduce bit for bit.
"""

import json
import sys
import threading
import warnings
from typing import Any, cast

import numpy as np
import pytest

import source_channel
from driver.test_tdse_hacc_binding import _descriptor, _fields
from driver.test_tdse_hacc_run import _simulation
from hacc_cases import FakeHaccProvider, alpha_law, preset_basis
from solprop.driver import tdse
from solprop.exceptions import (
    BackendError,
    DriverError,
    StopRequested,
    TDSEDomainWarning,
    TDSEUnqualifiedModelWarning,
)
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output import open_file, resume

DEADLINE = 10.0
"""A test-harness bound on a cancellation; the production code has none."""


@pytest.fixture
def workers(monkeypatch):
    """Serve the manufactured provider in real workers; return the workers started."""
    monkeypatch.setattr(tdse, "_ADAPTER", "source_adapters:manufactured_tdse")
    started = []
    original = tdse._open_provider

    def opened(**options):
        worker = original(**options)
        started.append(worker)
        return worker

    monkeypatch.setattr(tdse, "_open_provider", opened)
    return started


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))


def _bind(descriptor, execution=None, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return tdse.prepare_binding(
            descriptor,
            TDSEExecution(max_batch_rows=2) if execution is None else execution,
            3,
            **options,
        )


def _evaluate_once(descriptor):
    binding = _bind(descriptor)
    binding.reserve(1)
    result = binding.evaluate(_fields(3), z=0.1, purpose="startup", density=2.0)
    binding.record_trial(accepted=True)
    binding.release_reservation()
    binding.close()
    return result, binding


def _plain(record):
    return json.loads(json.dumps(record, allow_nan=False))


@pytest.mark.medium
def test_evaluations_through_the_worker_equal_the_provider_called_directly(workers):
    descriptor = _descriptor(preset_basis(1))
    provider_modules = ("pyroraptor", "source_adapters")
    with pytest.MonkeyPatch.context() as isolated:
        # Another test of this process may have loaded a provider module (a fake
        # one, in-process): set them aside, so that what follows shows what this
        # run imports, whatever ran before it.
        for name in [n for n in sys.modules if n.split(".")[0] in provider_modules]:
            isolated.delitem(sys.modules, name)
        through, worker_binding = _evaluate_once(descriptor)
        # The provider lived in the worker alone.
        loaded = [n for n in sys.modules if n.split(".")[0] in provider_modules]
        assert not loaded
    with pytest.MonkeyPatch.context() as patch:
        fake = source_channel.install(patch, FakeHaccProvider())
        fake.response = alpha_law(1.02, lambda5=2.0**-200)
        direct, direct_binding = _evaluate_once(descriptor)
    for name in ("total_current", "first_order_current", "valid"):
        np.testing.assert_array_equal(getattr(through, name), getattr(direct, name))
    assert through.monitors_json == direct.monitors_json
    assert through.outcomes_json == direct.outcomes_json
    assert worker_binding.ledger == direct_binding.ledger
    assert _plain(worker_binding.metadata()) == _plain(direct_binding.metadata())
    assert len(workers) == 1 and not workers[0].alive


@pytest.mark.medium
def test_the_handshake_refuses_another_execution_space_by_name(monkeypatch, workers):
    monkeypatch.setattr(tdse, "_ADAPTER", "source_adapters:cuda_tdse")
    with pytest.raises(BackendError, match="native build uses"):
        _bind(_descriptor(preset_basis(1)))
    assert len(workers) == 1 and not workers[0].alive


@pytest.mark.medium
def test_the_handshake_records_another_provider_revision(monkeypatch, workers):
    monkeypatch.setattr(tdse, "_ADAPTER", "source_adapters:stale_tdse")
    binding = _bind(_descriptor(preset_basis(1)))
    try:
        build = cast("Any", binding.metadata())["build"]
        assert build["native"]["source_revision"] == "old0000"
    finally:
        binding.close()


@pytest.mark.medium
def test_a_failure_in_preparation_is_named_and_ends_the_worker(workers):
    with pytest.raises(BackendError, match="no complement option") as raised:
        _bind(
            _descriptor(preset_basis(1)),
            TDSEExecution(max_batch_rows=2, complement="vendor_blas"),
        )
    assert "raised in the TDSE provider worker" in "\n".join(raised.value.__notes__)
    assert len(workers) == 1 and not workers[0].alive


@pytest.mark.medium
def test_the_transport_copies_are_in_the_ledger_and_admitted_before_start(workers):
    descriptor = _descriptor(preset_basis(1))
    execution = TDSEExecution(max_batch_rows=2)
    estimate = tdse.estimate_binding_memory(descriptor, execution, 3)
    binding = _bind(descriptor, execution)
    ledger = binding.memory_ledger()
    window = descriptor.atomic_window
    assert window is not None
    copies = 8 * 2 * window.samples * (2 + 20)
    assert ledger["transport.solprop"] == ledger["transport.worker"] == copies
    assert estimate["estimate.transport"] == 2 * copies
    assert any(name.startswith("model.") for name in ledger)
    binding.close()
    limit = sum(estimate.values()) - 1
    with pytest.raises(DriverError, match="TDSE persistent data"):
        _bind(descriptor, TDSEExecution(max_batch_rows=2, memory_limit=limit))
    assert len(workers) == 1


@pytest.mark.medium
def test_a_stop_during_an_atomic_call_kills_the_worker_and_keeps_the_charge(
    monkeypatch, workers
):
    monkeypatch.setenv("MANUFACTURED_HANG_AFTER", "0")
    flag = threading.Event()
    binding = _bind(_descriptor(preset_basis(1)), stop=flag)
    binding.reserve(1)
    threading.Timer(0.3, flag.set).start()
    with pytest.raises(StopRequested):
        binding.evaluate(_fields(3), z=0.1, purpose="startup")
    assert binding.ledger.attempted == 1 and binding.ledger.failed_calls == 0
    assert not workers[0].alive
    assert binding.metadata()["failures"] == []
    binding.close()


@pytest.mark.medium
def test_after_a_provider_exception_the_retry_gets_a_fresh_worker(
    monkeypatch, workers, tmp_path
):
    """A worker is never used again after a failure (section 6.5)."""
    monkeypatch.setenv("MANUFACTURED_FAIL_ONCE", str(tmp_path / "failed-once"))
    binding = _bind(_descriptor(preset_basis(1)))
    binding.reserve(1)
    with pytest.raises(BackendError, match=r"manufactured provider failure \[native\]"):
        binding.evaluate(_fields(3), z=0.1, purpose="startup")
    assert not workers[0].alive
    binding.reserve(1)
    result = binding.evaluate(_fields(3), z=0.1, purpose="startup")
    assert np.all(np.isfinite(result.total_current))
    assert len(workers) == 2 and workers[1].alive
    assert binding.ledger.attempted == 3 and binding.ledger.failed_calls == 1
    binding.close()
    assert not workers[1].alive


class _Watch:
    """Pass standard error on; set a flag when the worker says its call hangs."""

    def __init__(self, stream, flag):
        self._stream, self._flag = stream, flag

    def write(self, text):
        if "MANUFACTURED CALL HANGS" in text:
            self._flag.set()
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()


@pytest.mark.slow
def test_a_run_stopped_during_an_atomic_call_checkpoints_its_last_step(
    monkeypatch, workers, tmp_path
):
    """numpy_ref: the stop ends the run as stopped, the call charged (section 6.7)."""
    monkeypatch.setenv("MANUFACTURED_HANG_AFTER", "12")
    flag = threading.Event()
    monkeypatch.setattr(sys, "stderr", _Watch(sys.stderr, flag))
    path = tmp_path / "stopped.h5"
    simulation = _simulation(backend="numpy_ref")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEDomainWarning)
        simulation.run(path, progress="none", stop=flag)
    stopped = resume(path)
    assert stopped.outcome is not None
    # As any stopped run: ended before its length, with a terminal checkpoint.
    assert stopped.outcome.status == "failed"
    assert "StopRequested" in stopped.outcome.message
    calls = stopped.outcome.atomic_calls
    # Twelve calls returned; the thirteenth was charged and never returned.
    assert calls["attempted"] == 13
    assert calls["failed_calls"] == 0
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is not None
        assert stored.terminal_checkpoint.plane.z > 0
    assert not workers[0].alive

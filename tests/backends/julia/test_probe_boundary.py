# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check timing-probe ownership and transactions around a scripted engine."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from backends.julia import test_session_boundary as boundary_tests
from backends.julia.cases import cpu_case
from solprop.backends.julia import _probe
from solprop.backends.julia._fft_cache import FFTCache, FFTFingerprint
from solprop.backends.julia._probe import JuliaTimingProbe
from solprop.backends.session import SessionError, Start
from solprop.exceptions import BackendError

boundary = boundary_tests.boundary
SHAPE = boundary_tests.SHAPE


@pytest.fixture
def probe_boundary(boundary, monkeypatch):
    calls = []
    module = boundary.engine.module

    def create(*arguments):
        _, _, field, z, noise, resume, _ = arguments
        calls.append(("create", z, noise, resume))
        return SimpleNamespace(field=field.copy(), ready=False, closed=False)

    def trial(probe, z, dz):
        if probe.closed:
            raise RuntimeError("closed trial")
        calls.append(("trial", z, dz))
        probe.ready = True

    def copy(output, probe):
        if not probe.ready:
            raise RuntimeError("take a trial before reading its field")
        output[...] = probe.field
        probe.ready = False

    def close(probe):
        calls.append(("close",))
        probe.closed = True

    monkeypatch.setattr(module, "start_trial_probe", create, raising=False)
    monkeypatch.setattr(module, "trial_probe!", trial, raising=False)
    monkeypatch.setattr(module, "copy_trial_probe_field!", copy, raising=False)
    monkeypatch.setattr(module, "close_trial_probe!", close, raising=False)
    monkeypatch.setattr(boundary.owner, "_request", cpu_case().request, raising=False)
    monkeypatch.setattr(boundary.owner, "prepare", lambda *_: boundary.prepared)
    return boundary, calls


def test_probe_copies_start_and_returns_detached_read_once_fields(probe_boundary):
    boundary, calls = probe_boundary
    field = np.ones(SHAPE, dtype=np.complex128)
    start = Start(field, 0.0, monitor={"unused": object()})
    probe = boundary.owner.timing_probe(start)
    assert start.field is None
    field[...] = 7
    probe.trial()
    result = probe.field()
    np.testing.assert_array_equal(result, np.ones(SHAPE))
    assert result.flags.owndata and result.flags.c_contiguous
    with pytest.raises(BackendError, match="take a trial"):
        probe.field()
    probe.trial(z=0.1, dz=0.002)
    probe.field()[...] = 19
    np.testing.assert_array_equal(result, np.ones(SHAPE))
    assert calls[:2] == [("create", 0.0, None, False), ("trial", None, None)]
    assert calls[2] == ("trial", 0.1, 0.002)
    boundary.owner.close()
    assert calls[-1] == ("close",)
    with pytest.raises(SessionError, match="closed"):
        probe.trial()


def test_failed_probe_construction_preserves_start(probe_boundary, monkeypatch):
    boundary, _ = probe_boundary
    start = Start(np.ones(SHAPE, dtype=np.complex128), 0.0)
    field = start.field

    def fail(*_):
        raise RuntimeError("scripted construction failure")

    monkeypatch.setattr(boundary.engine.module, "start_trial_probe", fail)
    count = len(boundary.runtime.owners._entries)
    with pytest.raises(BackendError, match="scripted construction failure"):
        boundary.owner.timing_probe(start)
    assert start.field is field and len(boundary.runtime.owners._entries) == count
    assert not boundary.owner._probes


@pytest.mark.parametrize("invalid", ["consumed", "dtype", "shape", "nan", "z"])
def test_invalid_probe_start_refuses_before_engine_entry(probe_boundary, invalid):
    boundary, calls = probe_boundary
    start = Start(np.ones(SHAPE, dtype=np.complex128), 0.0)
    if invalid == "consumed":
        start.take_field()
    elif invalid == "dtype":
        start.field = np.ones(SHAPE, dtype=np.complex64)
    elif invalid == "shape":
        start.field = np.ones((1, 1, 3), dtype=np.complex128)
    elif invalid == "nan":
        assert start.field is not None
        start.field[...] = np.nan
    else:
        start.z = np.inf
    original = start.field
    with pytest.raises((SessionError, BackendError), match=r"used|finite"):
        boundary.owner.timing_probe(start)
    assert start.field is original and not calls


def test_worker_thread_cannot_enter_or_close_probe(probe_boundary):
    boundary, calls = probe_boundary
    probe = JuliaTimingProbe(
        boundary.prepared, Start(np.ones(SHAPE, dtype=np.complex128), 0.0)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        for operation in (probe.trial, probe.field, probe.memory_ledger, probe.close):
            with pytest.raises(BackendError, match="main thread"):
                executor.submit(operation).result()
    assert len(calls) == 1
    probe.close()
    probe.close()
    assert probe.memory_ledger() == {}
    assert len(calls) == 2


@pytest.mark.parametrize("interrupted", [False, True])
def test_planning_failure_survives_probe_cleanup_failure(
    probe_boundary, monkeypatch, tmp_path, interrupted
):
    boundary, _ = probe_boundary
    boundary.owner._fft_cache = FFTCache(
        FFTFingerprint("f" * 64, "{}"), tmp_path / "wisdom.json", (), False, "measure"
    )
    monkeypatch.setattr(_probe, "refresh_fft_cache", lambda cache: cache)
    boundary.engine.module.prepare_fft_planning = lambda _: object()
    field = np.ones(SHAPE, dtype=np.complex128)
    start = Start(field, 0.0)
    initial_owners = len(boundary.runtime.owners._entries)
    published = []
    failure = (
        KeyboardInterrupt("primary planning interruption")
        if interrupted
        else MemoryError("primary planning copy failure")
    )

    def fail_copy(*_):
        raise failure

    def fail_close(*_):
        raise RuntimeError("secondary Julia close failure")

    monkeypatch.setattr(_probe, "read_planning", fail_copy)
    monkeypatch.setattr(
        _probe, "publish_fft_cache", lambda *args: published.append(args)
    )
    monkeypatch.setattr(boundary.engine.module, "close_trial_probe!", fail_close)
    with pytest.raises(type(failure)) as caught:
        boundary.owner.timing_probe(start)
    assert caught.value is failure
    assert any("secondary Julia close failure" in note for note in failure.__notes__)
    assert start.field is field and not boundary.owner._probes
    np.testing.assert_array_equal(field, np.ones(SHAPE))
    assert len(boundary.runtime.owners._entries) == initial_owners
    assert published == []

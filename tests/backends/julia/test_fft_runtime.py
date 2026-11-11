# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check loaded-library fingerprint admission and successful-start publication."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backends.julia import test_session_boundary as boundary_tests
from solprop.backends.julia import _fft_cache, _fft_runtime, _session
from solprop.backends.julia._fft_cache import FFTCache, FFTFingerprint, forget_fft_cache
from solprop.backends.julia._fft_runtime import FFTPlanRecord, verified_identity
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._source import file_digest
from solprop.backends.session import Start
from solprop.exceptions import BackendError

boundary = boundary_tests.boundary


@pytest.fixture
def evidence(setup_record, tmp_path):
    project = dict(setup_record.projects)["cpu"]
    project.mkdir()
    project.joinpath("Manifest.toml").write_text(
        '[[deps.FFTW]]\nversion = "1.10.0"\ngit-tree-sha1 = "'
        + "a" * 40
        + '"\n[[deps.FFTW_jll]]\nversion = "3.3.11+0"\nuuid = "fftw-fixture"\n'
        + 'git-tree-sha1 = "'
        + "b" * 40
        + '"\n',
        encoding="utf-8",
    )
    artifact = setup_record.depot / "artifacts" / ("c" * 40)
    artifact.mkdir(parents=True)
    for name in ("double.so", "single.so"):
        artifact.joinpath(name).write_bytes(name.encode("utf-8"))
    inventory = tmp_path / "inventory.toml"
    inventory.write_text(
        '[[artifacts]]\nname = "FFTW"\npackage_uuid = "fftw-fixture"\n'
        + 'tree_hash = "'
        + "c" * 40
        + '"\npath = "artifacts/'
        + "c" * 40
        + '"\nfiles = ["double.so", "single.so"]\n',
        encoding="utf-8",
    )
    setup = replace(
        setup_record, inventories=(("cpu", inventory, file_digest(inventory)),)
    )
    loaded = {
        "julia_version": "1.13.3",
        "julia_machine": "aarch64-apple-darwin",
        "julia_cpu_name": "fixture-cpu",
        "provider": "fftw",
        "fftw_package_version": "1.10.0",
        "fftw_library_version": "3.3.11",
        "library_f64": str(artifact / "double.so"),
        "library_f32": str(artifact / "single.so"),
    }
    return setup, loaded


def test_fingerprint_hashes_actual_selected_bytes_and_runtime_facts(evidence):
    setup, loaded = evidence
    actual = verified_identity(setup, loaded)
    assert actual.cpu_name == loaded["julia_cpu_name"] != setup.cpu_name
    assert actual.cpu_target == setup.cpu_target
    assert dict(actual.libraries) == {
        "float64": file_digest(Path(loaded["library_f64"])),
        "float32": file_digest(Path(loaded["library_f32"])),
    }
    Path(loaded["library_f32"]).write_bytes(b"changed actual library bytes")
    assert actual.libraries != verified_identity(setup, loaded).libraries
    preferences = dict(setup.projects)["cpu"] / "LocalPreferences.toml"
    preferences.write_text('[FFTW]\nprovider = "fftw"\n', encoding="utf-8")
    assert (
        actual.preferences_digest != verified_identity(setup, loaded).preferences_digest
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("julia_version", "1.14.0"),
        ("provider", "mkl"),
        ("fftw_package_version", "1.9.0"),
        ("library_f32", "/foreign/single.so"),
        ("library_f64", "relative/double.so"),
    ],
)
def test_mismatched_loaded_runtime_or_artifact_is_refused(evidence, field, value):
    setup, loaded = evidence
    with pytest.raises(BackendError, match=r"Julia|loaded"):
        verified_identity(setup, {**loaded, field: value})


def test_existing_unlisted_file_is_not_admitted(evidence):
    setup, loaded = evidence
    foreign = Path(loaded["library_f64"]).with_name("unlisted.so")
    foreign.write_bytes(b"same artifact directory is insufficient")
    with pytest.raises(BackendError, match="selected artifact files"):
        verified_identity(setup, {**loaded, "library_f64": str(foreign)})


def test_missing_native_metadata_forces_process_only_cache(evidence, monkeypatch):
    setup, loaded = evidence
    runtime = JuliaRuntime(setup, 1, None, None)
    monkeypatch.setattr(_fft_runtime, "_runtime_facts", lambda _: loaded)

    def missing():
        raise BackendError("no native metadata distribution")

    monkeypatch.setattr(
        _fft_runtime,
        "_native",
        SimpleNamespace(topology=SimpleNamespace(read_topology=missing)),
    )
    monkeypatch.setenv("SOLPROP_FFTW_WISDOM", "/must-not-use-incomplete-topology")
    monkeypatch.setenv("SOLPROP_CALIBRATION", "auto")
    found = _fft_runtime.load_planning_cache(runtime)
    assert found.path is None and not found.file_supplied and found.offered == ()
    forget_fft_cache()


@pytest.fixture
def planned(boundary, monkeypatch, tmp_path):
    cache = FFTCache(
        FFTFingerprint("f" * 64, "{}"), tmp_path / "wisdom.json", (), False, "measure"
    )
    boundary.owner._fft_cache = cache
    boundary.engine.module.prepare_fft_planning = lambda _: object()
    facts = FFTPlanRecord((("transform", "new"),), (("transform", "new"),), (), ())
    monkeypatch.setattr(_session, "read_planning", lambda *_: facts)
    published = []

    def publish(cache, measurements):
        published.append(dict(measurements))
        return True

    monkeypatch.setattr(_session, "publish_fft_cache", publish)
    return boundary, published


def test_successful_start_publishes_only_after_consumption(planned, monkeypatch):
    boundary, published = planned
    start = Start(np.ones(boundary_tests.SHAPE, dtype=np.complex128), 0.0)

    def publish(cache, measured):
        assert start.field is None
        published.append(dict(measured))
        return True

    monkeypatch.setattr(_session, "publish_fft_cache", publish)
    with boundary.prepared.start(start, owns=False) as session:
        assert published == [{"transform": "new"}]
        assert session.execution_record()["fft_plans_measured"] == 1
        assert session.execution_record()["fft_cache_written"] is True
        assert session._fft_cache is None and session._planning_record is None


def test_failed_planning_copy_preserves_start_and_does_not_publish(
    planned, monkeypatch
):
    boundary, published = planned
    field = np.ones(boundary_tests.SHAPE, dtype=np.complex128)
    start = Start(field, 0.0)
    initial_owners = len(boundary.runtime.owners._entries)

    def fail(*_):
        raise MemoryError("interrupted planning copy")

    monkeypatch.setattr(_session, "read_planning", fail)
    with pytest.raises(MemoryError, match="interrupted planning"):
        boundary.prepared.start(start, owns=False)
    assert start.field is field and published == []
    assert boundary.engine.sessions[-1].closed
    assert len(boundary.runtime.owners._entries) == initial_owners


def test_cache_write_memory_failure_keeps_a_successful_reachable_session(
    planned, monkeypatch, tmp_path, caplog
):
    boundary, _ = planned
    assert boundary.owner._fft_cache is not None
    boundary.owner._fft_cache = replace(
        boundary.owner._fft_cache, path=tmp_path / "wisdom.json"
    )

    def fail(*_):
        raise MemoryError("optional cache write")

    monkeypatch.setattr(_fft_cache, "_replace", fail)
    monkeypatch.setattr(_session, "publish_fft_cache", _fft_cache.publish_fft_cache)
    start = Start(np.ones(boundary_tests.SHAPE, dtype=np.complex128), 0.0)
    with boundary.prepared.start(start, owns=False) as session:
        assert start.field is None and not boundary.engine.sessions[-1].closed
        assert session.execution_record()["fft_cache_written"] is False
        assert session in boundary.owner._sessions
    assert "cannot store FFTW plans" in caplog.text
    assert boundary.engine.sessions[-1].closed


def test_unexpected_postconstruction_failure_closes_engine_and_restores_start(
    planned, monkeypatch
):
    boundary, _ = planned
    field = np.ones(boundary_tests.SHAPE, dtype=np.complex128)
    start = Start(field, 0.0)
    initial_owners = len(boundary.runtime.owners._entries)

    def fail(*_):
        raise RuntimeError("unexpected publication defect")

    monkeypatch.setattr(_session, "publish_fft_cache", fail)
    with pytest.raises(RuntimeError, match="publication defect"):
        boundary.prepared.start(start, owns=False)
    assert start.field is field and boundary.engine.sessions[-1].closed
    assert len(boundary.runtime.owners._entries) == initial_owners

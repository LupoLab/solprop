# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify detached Julia FFT identity and atomic cache policy without an engine."""

import inspect
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from solprop import calibration
from solprop.backends.julia import _fft_cache as cache
from solprop.exceptions import BackendError


@pytest.fixture
def identity():
    return cache.FFTEngineIdentity(
        engine_digest="a" * 64,
        julia_version="1.13.3",
        julia_binary_digest="b" * 64,
        cpu_target="generic",
        cpu_name="fixture-cpu",
        machine="aarch64-apple-darwin",
        provider="fftw",
        fftw_package_version="1.10.0",
        fftw_package_tree="c" * 40,
        fftw_jll_version="3.3.11+0",
        fftw_jll_tree="d" * 40,
        fftw_library_version="3.3.11",
        artifacts=(("fftw", "e" * 40),),
        libraries=(("float32", "f" * 64), ("float64", "1" * 64)),
        preferences_digest="2" * 64,
    )


@pytest.fixture
def policy(tmp_path, monkeypatch):
    # Keep this pure cache test independent of an installed native distribution.
    monkeypatch.setattr(cache, "store_directory", lambda _: tmp_path)
    cache.forget_fft_cache()
    yield {"SOLPROP_CACHE_DIR": str(tmp_path)}
    cache.forget_fft_cache()


def _load(identity, policy):
    return cache.load_fft_cache(identity, {"cores": [[0, 1], [2, 3]]}, environ=policy)


def _record(found, plans):
    return {
        "format": 1,
        "fingerprint": found.fingerprint.digest,
        "identity": found.fingerprint.canonical,
        "plans": plans,
    }


def _write(found, record):
    assert found.path is not None
    found.path.parent.mkdir(parents=True, exist_ok=True)
    found.path.write_text(json.dumps(record), encoding="utf-8")


def test_later_start_refreshes_process_and_replaced_file_offers(identity, policy):
    first = _load(identity, policy)
    assert not first.offered
    assert cache.publish_fft_cache(first, {"temporal": "first successful plan"})
    second = cache.refresh_fft_cache(first)
    assert dict(second.offered) == {"temporal": "first successful plan"}
    assert not first.offered and second.file_supplied
    _write(first, _record(first, {"spatial": "external plan", "temporal": "older"}))
    third = cache.refresh_fft_cache(first)
    assert dict(third.offered) == {
        "spatial": "external plan",
        "temporal": "first successful plan",
    }
    assert third.fingerprint is first.fingerprint
    assert third.path == first.path and third.effort == first.effort
    cache.forget_fft_cache()
    assert first.path is not None
    first.path.unlink()
    assert not cache.refresh_fft_cache(third).offered


def test_process_only_refresh_preserves_its_bound_policy(identity, policy):
    first = _load(identity, {**policy, "SOLPROP_CALIBRATION": "off"})
    assert first.path is None
    assert not cache.publish_fft_cache(first, {"temporal": "measured"})
    assert cache.refresh_fft_cache(first).offered == (("temporal", "measured"),)
    cache.forget_fft_cache(first.fingerprint.digest)
    assert not cache.refresh_fft_cache(first).offered


def test_calibration_clear_removes_only_this_machines_julia_plans(
    identity, policy, monkeypatch, tmp_path
):
    topology = {"cores": [[0, 1], [2, 3]]}
    monkeypatch.setattr(
        calibration._native.topology,
        "read_topology",
        lambda: SimpleNamespace(record=lambda: topology),
    )
    monkeypatch.setattr(
        calibration, "choice", lambda _: SimpleNamespace(fingerprint=None)
    )
    environment = {**policy, "SOLPROP_FFTW_WISDOM": str(tmp_path / "named")}
    default, named = _load(identity, policy), _load(identity, environment)
    foreign = cache.load_fft_cache(identity, {"cores": [[8]]}, environ=policy)
    for found in (default, named, foreign):
        assert cache.publish_fft_cache(found, {"temporal": "plan"})
    unrelated = tmp_path / "named.julia-unrelated"
    unrelated.write_text("{}", encoding="utf-8")
    malformed = tmp_path / "named.julia-malformed"
    malformed.write_text("{broken", encoding="utf-8")
    assert calibration.clear(environment) == 2
    assert default.path is not None and not default.path.exists()
    assert named.path is not None and not named.path.exists()
    assert foreign.path is not None and foreign.path.is_file()
    assert unrelated.is_file() and malformed.is_file()
    assert not cache.refresh_fft_cache(default).offered
    remaining = set(tmp_path.rglob("*"))
    assert calibration.clear(environment) == 0
    assert set(tmp_path.rglob("*")) == remaining
    assert not list(tmp_path.rglob("*.lock.lock"))


def test_clear_without_topology_still_forgets_process_measurements(
    identity, policy, monkeypatch
):
    def absent():
        raise BackendError("topology unavailable")

    first = _load(identity, {**policy, "SOLPROP_CALIBRATION": "off"})
    cache.publish_fft_cache(first, {"temporal": "plan"})
    monkeypatch.setattr(calibration._native.topology, "read_topology", absent)
    assert cache.clear_fft_cache(policy) == 0
    assert not cache.refresh_fft_cache(first).offered


def test_every_engine_field_and_topology_affect_identity(identity):
    before = cache.fingerprint(identity, {"cores": [1], "model": "A"})
    for name, value in asdict(identity).items():
        if name == "artifacts":
            changed = (("fftw", "0" * 40),)
        elif name == "libraries":
            changed = (("float32", "0" * 64), ("float64", "1" * 64))
        elif name.endswith("digest") or name.endswith("tree"):
            changed = "0" * len(value)
        else:
            changed = f"{value}-different"
        assert (
            cache.fingerprint(
                replace(identity, **{name: changed}), {"cores": [1], "model": "A"}
            )
            != before
        )
    assert cache.fingerprint(identity, {"cores": [2], "model": "A"}) != before


def test_identity_is_canonical_copied_and_validated(identity):
    topology = {"nested": {"cores": [1, 2]}, "model": "A"}
    found = cache.fingerprint(identity, topology)
    assert found == cache.fingerprint(
        identity, {"model": "A", "nested": {"cores": [1, 2]}}
    )
    topology["nested"]["cores"].append(3)
    assert "[1,2]" in found.canonical
    assert found != cache.fingerprint(identity, topology)
    assert replace(identity, libraries=tuple(reversed(identity.libraries))) == identity
    for field, value in (
        ("engine_digest", "Z" * 64),
        ("julia_version", ""),
        ("fftw_package_tree", "a" * 39),
        ("artifacts", ()),
        ("artifacts", (("x", "a" * 40), ("x", "b" * 40))),
        ("artifacts", (("", "a" * 40),)),
        ("libraries", (("float64", "a" * 64),)),
    ):
        with pytest.raises(BackendError, match="julia FFT"):
            replace(identity, **{field: value})
    for bad in ({"x": float("nan")}, {"x": object()}, {"x": float("inf")}):
        with pytest.raises(BackendError, match="finite JSON"):
            cache.fingerprint(identity, bad)


def test_default_off_override_and_effort_policy(
    identity, policy, tmp_path, monkeypatch
):
    found = _load(identity, policy)
    assert (
        found.path
        == tmp_path / "machines" / f"julia-{found.fingerprint.digest}" / "wisdom.json"
    )
    assert found.offered == () and not found.file_supplied
    assert found.effort == "measure"
    assert _load(identity, policy | {"SOLPROP_CALIBRATION": " off "}).path is None
    monkeypatch.chdir(tmp_path)
    for mode in ("off", "invalid"):
        overridden = _load(
            identity,
            policy
            | {
                "SOLPROP_FFTW_WISDOM": " local-plans ",
                "SOLPROP_CALIBRATION": mode,
            },
        )
        assert (
            overridden.path
            == tmp_path / f"local-plans.julia-{found.fingerprint.digest}"
        )
    for effort in ("estimate", "measure", "patient", ""):
        assert _load(identity, policy | {"SOLPROP_FFTW_PLAN": effort}).effort == (
            effort or "measure"
        )
    for effort in ("MEASURE", " ", "exhaustive"):
        with pytest.raises(BackendError, match="SOLPROP_FFTW_PLAN"):
            _load(identity, policy | {"SOLPROP_FFTW_PLAN": effort})


def test_invalid_path_or_mode_warns_and_keeps_process_only(identity, policy, caplog):
    for extra in ({"SOLPROP_FFTW_WISDOM": "/"}, {"SOLPROP_CALIBRATION": "invalid"}):
        assert _load(identity, policy | extra).path is None
    assert "cannot keep FFTW plans" in caplog.text


def test_roundtrip_detached_process_precedence_and_forget(identity, policy):
    found = _load(identity, policy)
    plans = {"one": "wisdom-one"}
    assert cache.publish_fft_cache(found, plans)
    plans["one"] = "caller-mutated"
    assert found.offered == ()
    loaded = _load(identity, policy)
    assert loaded.file_supplied and loaded.offered == (("one", "wisdom-one"),)
    assert not cache.publish_fft_cache(found, {"one": "wisdom-one"})
    assert found.path is not None
    before = found.path.stat()
    assert not cache.publish_fft_cache(found, {})
    assert not cache.publish_fft_cache(found, {"one": "wisdom-one"})
    assert found.path.stat().st_mtime_ns == before.st_mtime_ns
    _write(found, _record(found, {"one": "external", "two": "external-two"}))
    assert dict(_load(identity, policy).offered) == {
        "one": "wisdom-one",
        "two": "external-two",
    }
    assert (
        _load(identity, policy | {"SOLPROP_CALIBRATION": "off"}).offered
        == loaded.offered
    )
    cache.forget_fft_cache(found.fingerprint.digest)
    assert _load(identity, policy | {"SOLPROP_CALIBRATION": "off"}).offered == ()
    assert dict(_load(identity, policy).offered) == {
        "one": "external",
        "two": "external-two",
    }
    assert loaded.offered == (("one", "wisdom-one"),)


def test_process_only_is_fingerprint_specific(identity, policy):
    off = policy | {"SOLPROP_CALIBRATION": "off"}
    found = _load(identity, off)
    assert not cache.publish_fft_cache(found, {"one": "measured"})
    assert _load(identity, off).offered == (("one", "measured"),)
    assert _load(replace(identity, cpu_target="native"), off).offered == ()
    cache.forget_fft_cache("unrelated")
    assert _load(identity, off).offered
    cache.forget_fft_cache()
    assert _load(identity, off).offered == ()


@pytest.mark.parametrize("bad", [None, [], 7, {"format": True}, {"format": 2}])
def test_incompatible_records_warn(identity, policy, caplog, bad):
    found = _load(identity, policy)
    _write(found, bad)
    assert _load(identity, policy).offered == ()
    assert "ignoring FFTW plans" in caplog.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", "foreign"),
        ("identity", "different provider"),
        ("plans", []),
        ("plans", {"key": 1}),
    ],
)
def test_foreign_identity_and_invalid_plans_warn(
    identity, policy, caplog, field, value
):
    found = _load(identity, policy)
    _write(found, _record(found, {}) | {field: value})
    assert _load(identity, policy).offered == ()
    assert "ignoring FFTW plans" in caplog.text


def test_bad_json_nonutf8_and_unreadable_file_warn(
    identity, policy, caplog, monkeypatch
):
    found = _load(identity, policy)
    assert found.path is not None
    found.path.parent.mkdir(parents=True)
    for contents in (b"not JSON", b"\xff"):
        found.path.write_bytes(contents)
        assert not _load(identity, policy).offered

    def fail_read(*args, **kwargs):
        raise PermissionError("no read")

    monkeypatch.setattr(Path, "read_text", fail_read)
    assert not _load(identity, policy).offered
    assert "cannot read FFTW plans" in caplog.text


def test_external_atomic_replacement_same_mtime_size_is_seen(identity, policy):
    found = _load(identity, policy)
    _write(found, _record(found, {"a": "first"}))
    assert _load(identity, policy).offered == (("a", "first"),)
    assert found.path is not None
    before = found.path.stat()
    other = found.path.with_suffix(".new")
    other.write_text(json.dumps(_record(found, {"a": "other"})), encoding="utf-8")
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert other.stat().st_size == before.st_size
    other.replace(found.path)
    assert _load(identity, policy).offered == (("a", "other"),)


def test_publication_failure_cleans_temp_and_remembers(
    identity, policy, caplog, monkeypatch
):
    found = _load(identity, policy)

    def fail_replace(*args, **kwargs):
        raise PermissionError("no replace")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        assert not cache.publish_fft_cache(found, {"a": "first"})
    assert "cannot store FFTW plans" in caplog.text
    assert found.path is not None and not found.path.exists()
    assert list(found.path.parent.iterdir()) == [
        found.path.with_name(f"{found.path.name}.lock")
    ]
    assert _load(identity, policy).offered == (("a", "first"),)
    assert cache.publish_fft_cache(found, {"a": "first"})
    with pytest.raises(BackendError, match="text pairs"):
        # Exercise the production boundary independently of the optional test hook.
        inspect.unwrap(cache.publish_fft_cache)(found, {"a": 3})


@pytest.mark.skipif(
    sys.platform == "win32", reason="cross-process writer flock is POSIX"
)
def test_concurrent_writers_merge_without_lost_measurements(identity, policy, tmp_path):
    found = _load(identity, policy)
    assert found.path is not None
    payload = tmp_path / "identity.json"
    payload.write_text(json.dumps(asdict(identity)), encoding="utf-8")
    program = textwrap.dedent("""
        import json, sys
        from pathlib import Path
        from solprop.backends.julia._fft_cache import (
            FFTEngineIdentity, FFTCache, FFTFingerprint, publish_fft_cache,
        )
        raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        for name in ("artifacts", "libraries"):
            raw[name] = tuple(map(tuple, raw[name]))
        identity = FFTEngineIdentity(**raw)
        record = FFTCache(FFTFingerprint(sys.argv[2], sys.argv[3]),
                          Path(sys.argv[4]), (), False, "measure")
        assert publish_fft_cache(record, {sys.argv[5]: sys.argv[5]})
    """)
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                str(payload),
                found.fingerprint.digest,
                found.fingerprint.canonical,
                str(found.path),
                str(index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(4)
    ]
    for child in children:
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stdout + stderr
    assert dict(_load(identity, policy).offered) == {
        str(index): str(index) for index in range(4)
    }


def test_import_and_process_cache_need_no_execution_runtime(tmp_path):
    program = textwrap.dedent("""
        import sys
        class RefuseEngines:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'solprop_native', 'juliacall'}:
                    raise AssertionError('execution runtime import: ' + fullname)
        sys.meta_path.insert(0, RefuseEngines())
        from solprop.backends.julia._fft_cache import forget_fft_cache
        forget_fft_cache()
    """)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

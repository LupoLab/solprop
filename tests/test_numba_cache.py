# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The keyed numba disk cache: a data file is only ever used for the key that wrote it.

numba's own cache can cross-wire an entry when two processes save different keys of one
function at once (see :mod:`solprop._numba_cache`), after which every process asking
for one key runs the other's code. The in-process tests build that cross-wired state
directly; the medium one races real processes into an empty cache.

These are cache-validity tests, and their exact comparisons are structural contracts
(SEP 25): the dependency digests are cache keys, and each kernel's result is a scale
times one (plus zero), exact in any arithmetic, so it names which variant's compiled
code ran -- a cached kernel against the variant it was saved for, never two different
compiled variants or thread counts compared for rounding.
"""

import json
import os
import subprocess
import sys
import textwrap
import time

import numba
import numpy as np
import pytest
from numba.core import config

from solprop import _numba_cache
from solprop._numba_cache import (
    _constant,
    _dependency_digest,
    _frozen_value,
    _KeyedCacheFile,
    _replacing,
    njit,
)


def _scaled(scale):
    """Return a closure variant: the code and argument types of all, one constant."""

    def kernel(out, samples):
        for index in range(samples.shape[0]):
            out[index] = scale * samples[index]

    return njit(cache=True)(kernel)


def _run(dispatcher, dtype: type[np.inexact] = np.float64):
    out = np.zeros(3, dtype)
    dispatcher(out, np.ones(3, dtype))
    return out[0], sum(dispatcher._cache_hits.values())


@pytest.fixture
def cache_directory(tmp_path, monkeypatch):
    """Point numba's disk cache at a fresh directory for this test."""
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    return tmp_path


def test_a_file_holding_another_keys_code_is_a_miss_not_a_wrong_kernel(
    cache_directory,
):
    """The state numba's index race leaves behind is refused rather than run."""
    assert _run(_scaled(2.0)) == (2.0, 0)
    assert _run(_scaled(3.0)) == (3.0, 0)
    files = sorted(cache_directory.rglob("*.nbc"))
    assert len(files) == 2
    assert not list(cache_directory.rglob("*.nbi"))
    # Each variant is read back from its own file.
    assert _run(_scaled(2.0)) == (2.0, 1)
    assert _run(_scaled(3.0)) == (3.0, 1)
    # Cross-wire them: each file now holds the other variant's compilation.
    first, second = (path.read_bytes() for path in files)
    files[0].write_bytes(second)
    files[1].write_bytes(first)
    assert _run(_scaled(2.0)) == (2.0, 0)
    assert _run(_scaled(3.0)) == (3.0, 0)
    # The misses recompiled and rewrote both files, which are read back again.
    assert _run(_scaled(2.0)) == (2.0, 1)
    assert _run(_scaled(3.0)) == (3.0, 1)


def test_signatures_of_one_function_keep_separate_files(cache_directory):
    assert _run(_scaled(2.0), np.float64) == (2.0, 0)
    assert _run(_scaled(2.0), np.complex128) == (2.0, 0)
    assert len(list(cache_directory.rglob("*.nbc"))) == 2
    assert _run(_scaled(2.0), np.complex128) == (2.0, 1)


def test_a_changed_source_stamp_is_a_miss(cache_directory):
    """A file written from another version of the source is not read back."""
    assert _run(_scaled(2.0)) == (2.0, 0)
    dispatcher = _scaled(2.0)
    stamp = dispatcher._cache._cache_file._source_stamp
    dispatcher._cache._cache_file._source_stamp = (stamp[0] + 1.0, stamp[1])
    assert _run(dispatcher) == (2.0, 0)
    # Rewritten in place under the new stamp, not added beside the old file.
    assert len(list(cache_directory.rglob("*.nbc"))) == 1
    assert _run(_scaled(2.0)) == (2.0, 0)


def test_an_eager_signature_compiles_into_the_keyed_cache(cache_directory):
    def double(value):
        return 2.0 * value

    compiled = njit("float64(float64)", cache=True)(double)
    assert compiled(1.5) == 3.0
    assert len(list(cache_directory.rglob("*.nbc"))) == 1
    again = njit(["float64(float64)"], cache=True)(double)
    assert sum(again._cache_hits.values()) == 1
    with pytest.raises(TypeError):
        again(np.ones(2))


def test_the_flush_numba_offers_leaves_the_files(cache_directory):
    _run(_scaled(2.0))
    _scaled(2.0)._cache.flush()
    assert _run(_scaled(2.0)) == (2.0, 1)


_BODY = """
from numba import njit

@njit(inline="always")
def body(value):
    return value * {factor}
"""

_CONSTANT = """
FACTOR = {factor}
"""

_KERNEL = """
from solprop._numba_cache import njit
from {module}_dependency import {name}

@njit(cache=True)
def kernel(value):
    return {call}
"""


def _import_fresh(directory, kind, factor, monkeypatch):
    """Write a dependency module and a kernel module that uses it, and import both.

    The kernel module's text never changes, so numba's source stamp cannot see the
    dependency change; its size does, so Python's bytecode cache cannot hide it.
    """
    module = f"cached_{kind}"
    source = _BODY if kind == "body" else _CONSTANT
    call = "body(value)" if kind == "body" else "value * FACTOR"
    name = "body" if kind == "body" else "FACTOR"
    (directory / f"{module}_dependency.py").write_text(source.format(factor=factor))
    kernel_path = directory / f"{module}.py"
    if not kernel_path.exists():
        kernel_path.write_text(_KERNEL.format(module=module, name=name, call=call))
    monkeypatch.syspath_prepend(str(directory))
    for stale in (module, f"{module}_dependency"):
        sys.modules.pop(stale, None)
    imported = __import__(module)
    return imported.kernel


@pytest.mark.parametrize("kind", ["body", "constant"])
def test_an_edited_dependency_in_another_file_is_a_miss(
    kind, cache_directory, tmp_path, monkeypatch
):
    """An inlined body or an imported constant changes; the kernel's file does not."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    sources = tmp_path / "sources"
    sources.mkdir()

    def run(factor):
        kernel = _import_fresh(sources, kind, factor, monkeypatch)
        return kernel(1.0), sum(kernel._cache_hits.values())

    assert run(2.0) == (2.0, 0)
    assert run(2.0) == (2.0, 1)
    assert run(50.0) == (50.0, 0)
    assert run(50.0) == (50.0, 1)
    assert len(list(cache_directory.rglob("*.nbc"))) == 1


def _shared(value):
    return value


_SHARED = njit()(_shared)


def _rich(value):
    # A module (not digested), a tuple constant and a nested code object, in a
    # dependency, where the walk digests the code.
    nested = lambda item: item in (1, 2)  # noqa: E731 - a nested code object
    return _SHARED(np.floor(value)), nested


_RICH = njit()(_rich)


def _outer(value):
    # Reaches ``_SHARED`` directly and through ``_RICH``: walked once.
    return _RICH(value), _SHARED(value)


def test_the_dependency_digest_follows_bodies_cells_and_constants():
    outer = _dependency_digest(_outer)
    assert outer == _dependency_digest(_outer)
    assert outer != _dependency_digest(_rich)

    def body(value):
        return value

    def other(value):
        return -value

    def make(inner, scale):
        def kernel(value):
            return inner(value) * scale

        return kernel

    first = _dependency_digest(make(njit()(body), 2.0))
    assert _dependency_digest(make(njit()(body), 2.0)) == first
    assert _dependency_digest(make(njit()(other), 2.0)) != first
    assert _dependency_digest(make(njit()(body), 3.0)) != first


_DIGEST = """
from solprop._numba_cache import _dependency_digest

def body(value):
    labels = lambda name: name in {"alpha", "beta", "gamma", "delta"}
    return labels, value

print(_dependency_digest(body))
"""


def test_the_dependency_digest_is_the_same_under_any_string_hashing():
    """A frozenset constant iterates in hash order, which every process salts anew."""
    digests = {
        subprocess.run(
            [sys.executable, "-c", _DIGEST],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for seed in (1, 2, 3)
    }
    assert len(digests) == 1


def test_only_values_numba_freezes_are_digested():
    array = np.arange(3.0)
    assert _frozen_value(array) == _frozen_value(array.copy())
    assert _frozen_value(array) != _frozen_value(array + 1.0)
    assert _frozen_value((1, 2.5, np.float64(3.0))) == ("1", "2.5", "np.float64(3.0)")
    # A module, a class or a function has no frozen value, nor does a tuple holding one.
    assert _frozen_value(np) is None
    assert _frozen_value(int) is None
    assert _frozen_value((1, np)) is None
    assert _constant(frozenset({"b", "a"})) == ("frozenset", ["'a'", "'b'"])


def test_a_failed_write_leaves_neither_the_file_nor_its_temporary(tmp_path):
    target = tmp_path / "entry.nbc"
    with pytest.raises(RuntimeError, match="interrupted"), _replacing(target) as stream:
        stream.write(b"partial")
        raise RuntimeError("interrupted")
    assert list(tmp_path.iterdir()) == []


# A key in numba's form: (signature, target, (bytecode digest, closure digest)).
KEY = ("(float64,)", "cpu", ("bytecode", "closure"))


def _refused(path, _target):
    """Stand in for a replace that Windows refuses while a reader holds the file."""
    raise PermissionError(13, "in use", str(path))


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_a_refused_replace_loses_the_write_only_on_windows(
    tmp_path, monkeypatch, platform
):
    cache = _KeyedCacheFile(str(tmp_path), "kernel-1.py314", (0.0, 1), _refused)
    monkeypatch.setattr(_numba_cache.Path, "replace", _refused)
    monkeypatch.setattr(_numba_cache.sys, "platform", platform)
    if platform == "win32":
        cache.save(KEY, {"payload": 1})
        assert cache.load(KEY) is None
    else:
        with pytest.raises(PermissionError):
            cache.save(KEY, {"payload": 1})
    assert list(tmp_path.iterdir()) == []


def test_without_jit_the_function_is_returned_as_numba_returns_it(monkeypatch):
    def kernel(value):
        return value

    monkeypatch.setattr(config, "DISABLE_JIT", True)
    assert njit(cache=True, nogil=True)(kernel) is kernel
    assert numba.njit(cache=True)(kernel) is kernel


_RACER = """
import json, pathlib, sys, time
import numpy as np
from solprop._numba_cache import njit

OFFSET = 0.0

@njit(inline="always")
def scaled(value, scale):
    return scale * value + OFFSET

def make(scale):
    def kernel(out, samples):
        for index in range(samples.shape[0]):
            out[index] = scaled(samples[index], scale)
    return njit(cache=True)(kernel)

def run(variant):
    dtype = np.float64 if variant % 2 else np.complex128
    out = np.zeros(3, dtype)
    compiled = make(float(variant))
    compiled(out, np.ones(3, dtype))
    return float(out[0].real), sum(compiled._cache_hits.values())

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    if sys.argv[1] == "check":
        print(json.dumps([run(variant) for variant in range(int(sys.argv[2]))]))
    else:
        (here / f"ready-{sys.argv[1]}").touch()
        while not (here / "go").exists():
            time.sleep(0.001)
        run(int(sys.argv[1]))
"""


@pytest.mark.medium
def test_variants_saved_by_racing_processes_each_load_their_own_code(tmp_path):
    """Eight processes save eight keys of one function at the same instant.

    The kernel inlines a body and reads a constant, and every process salts its string
    hashing differently, so the header's dependency digest is compared across processes
    too.

    Under numba's own cache, twelve such processes cross-wired entries in three
    trials of four; here every key is read back, and is its own.
    """
    n_variants = 8
    script = tmp_path / "racer.py"
    script.write_text(textwrap.dedent(_RACER))
    environment = {"NUMBA_CACHE_DIR": str(tmp_path / "cache")}
    racers = [
        subprocess.Popen(
            [sys.executable, str(script), str(variant)],
            env={**os.environ, **environment, "PYTHONHASHSEED": str(variant)},
        )
        for variant in range(n_variants)
    ]
    deadline = time.monotonic() + 120.0
    while len(list(tmp_path.glob("ready-*"))) < n_variants:
        assert time.monotonic() < deadline, "the racers never became ready"
        assert all(racer.poll() is None for racer in racers)
        time.sleep(0.01)
    (tmp_path / "go").touch()
    assert [racer.wait(timeout=120) for racer in racers] == [0] * n_variants
    checked = subprocess.run(
        [sys.executable, str(script), "check", str(n_variants)],
        env={**os.environ, **environment, "PYTHONHASHSEED": str(n_variants)},
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = [tuple(entry) for entry in json.loads(checked.stdout)]
    assert loaded == [(float(variant), 1) for variant in range(n_variants)]

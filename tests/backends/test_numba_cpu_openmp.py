# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How pyFFTW's OpenMP threads wait: the decision, and an import that leaves no trace.

The policy is decided once, before pyFFTW's runtime loads, so the decision is a pure
function tested here over its whole table. The import itself is checked in a fresh
interpreter, because in this one pyFFTW was loaded long ago.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from solprop.backends.numba_cpu import openmp


@pytest.mark.parametrize(
    ("platform", "environ", "loaded", "expected"),
    [
        ("linux", {}, False, "passive"),
        ("darwin", {}, False, "runtime default"),
        ("win32", {}, False, "runtime default"),
        ("linux", {}, True, "inherited"),
        ("linux", {"OMP_WAIT_POLICY": "active"}, True, "inherited"),
        (
            "linux",
            {"OMP_WAIT_POLICY": "active"},
            False,
            "environment: OMP_WAIT_POLICY=active",
        ),
        ("linux", {"GOMP_SPINCOUNT": "0"}, False, "environment: GOMP_SPINCOUNT=0"),
        (
            "darwin",
            {"OMP_WAIT_POLICY": "passive", "GOMP_SPINCOUNT": "10"},
            False,
            "environment: OMP_WAIT_POLICY=passive, GOMP_SPINCOUNT=10",
        ),
    ],
)
def test_the_policy_is_set_only_on_linux_when_nothing_else_decides_it(
    platform, environ, loaded, expected
):
    """A user's variable or an earlier import wins; only Linux's default is changed."""
    assert openmp.wait_policy(platform, environ, loaded=loaded) == expected


NATIVE_INSTALLED = importlib.util.find_spec("solprop_native") is not None


@pytest.mark.parametrize("given", [None, "active"])
def test_the_import_restores_the_environment_it_found(given):
    """The variable is pyFFTW's alone: a runtime loaded afterwards reads the user's.

    On macOS with solprop-native installed, the native package is imported first and
    leaves its own passive default in the environment, which pyFFTW's runtime reads.
    """
    environ = {
        key: value
        for key, value in os.environ.items()
        if key not in openmp.VARIABLES and key != "PYTHONWARNINGS"
    }
    if given is not None:
        environ["OMP_WAIT_POLICY"] = given
    expected = dict(environ)
    if openmp.native_first(sys.platform, installed=NATIVE_INSTALLED, loaded=False):
        expected.setdefault("OMP_WAIT_POLICY", "passive")
    script = (
        "import json, os, sys\n"
        "from solprop.backends.numba_cpu import openmp\n"
        "print(json.dumps([os.environ.get('OMP_WAIT_POLICY'), openmp.WAIT_POLICY,"
        " 'pyfftw' in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environ,
        capture_output=True,
        text=True,
        check=True,
    )
    after, policy, imported = json.loads(result.stdout.splitlines()[-1])
    assert after == expected.get("OMP_WAIT_POLICY")
    assert imported
    assert policy == openmp.wait_policy(sys.platform, expected, loaded=False)


@pytest.mark.parametrize(
    ("platform", "installed", "loaded", "expected"),
    [
        ("darwin", True, False, True),
        ("darwin", False, False, False),
        ("darwin", True, True, False),
        ("linux", True, False, False),
        ("win32", True, False, False),
    ],
)
def test_the_native_backend_loads_before_pyfftw_only_on_macos(
    platform, installed, loaded, expected
):
    """Only macOS refuses the native import after pyFFTW's; too late is left alone."""
    assert openmp.native_first(platform, installed=installed, loaded=loaded) == expected


@pytest.mark.skipif(
    sys.platform != "darwin" or not NATIVE_INSTALLED,
    reason="macOS refuses the native import after pyFFTW; needs solprop-native",
)
def test_a_numba_cpu_import_leaves_the_native_backend_importable():
    """pyFFTW's import loads the native backend first, which then imports cleanly.

    Before, ``numba_cpu`` imported pyFFTW alone, and a later default (native) run in
    the same process failed with the native package's import-order refusal.
    """
    code = (
        "import json, sys\n"
        "from solprop.backends.numba_cpu import openmp\n"
        "loaded = 'solprop_native' in sys.modules\n"
        "import solprop_native\n"
        "print(json.dumps([openmp.NATIVE_FIRST, loaded]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={
            key: value for key, value in os.environ.items() if key != "PYTHONWARNINGS"
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [True, True]


@pytest.mark.parametrize(
    ("platform", "environ", "loaded", "expected"),
    [
        ("linux", {}, False, "none"),
        ("linux", {"OMP_PROC_BIND": "false"}, False, "none"),
        ("linux", {"OMP_PROC_BIND": " FALSE "}, False, "none"),
        ("linux", {"OMP_PROC_BIND": "close"}, False, "unbound"),
        ("linux", {"OMP_PLACES": "cores"}, False, "unbound"),
        ("darwin", {"OMP_PROC_BIND": "close"}, False, "none"),
        (
            "linux",
            {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"},
            True,
            "inherited: OMP_PROC_BIND=spread, OMP_PLACES=cores",
        ),
    ],
)
def test_binding_is_stopped_on_linux_for_numba_cpus_runtimes(
    platform, environ, loaded, expected
):
    """``OMP_PLACES`` alone binds under GNU OpenMP; only Linux is changed."""
    assert openmp.binding_policy(platform, environ, loaded=loaded) == expected


def test_binding_is_stopped_for_pyfftws_import_and_restored():
    """pyFFTW's runtime loads unbound; the user's variables come back afterwards."""
    code = (
        "import json, os\n"
        "from solprop.backends.numba_cpu import openmp\n"
        "print(json.dumps([openmp.BINDING, os.environ.get('OMP_PROC_BIND')]))\n"
    )
    environ = {
        key: value
        for key, value in os.environ.items()
        if key not in openmp.BINDING_VARIABLES and key != "PYTHONWARNINGS"
    }
    environ["OMP_PROC_BIND"] = "close"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=environ,
    )
    binding, after = json.loads(result.stdout.strip().splitlines()[-1])
    assert after == "close"
    assert binding == ("unbound" if sys.platform == "linux" else "none")


def test_a_shared_gnu_runtime_is_recognised_by_name(tmp_path):
    """Only ``libgomp.so``, not a wheel's hashed copy, is Numba's layer's runtime."""
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 0 08:01 1 /lib/x86_64-linux-gnu/libgomp-e985bcbb.so.1.0.0\n\n",
        encoding="utf-8",
    )
    assert not openmp._gnu_openmp_loaded(maps)
    maps.write_text(
        "7f00-7f01 r-xp 0 08:01 1 /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0\n",
        encoding="utf-8",
    )
    assert openmp._gnu_openmp_loaded(maps)
    assert not openmp._gnu_openmp_loaded(tmp_path / "absent")


class _Parallel:
    def __init__(self, started):
        self._is_initialized = started


class _Config:
    def __init__(self, layer):
        self.THREADING_LAYER = layer


@pytest.fixture
def numba_state(monkeypatch):
    """Numba's launch state and layer, replaced for one test."""

    def apply(*, started=False, layer="default", gnu=False):
        config = _Config(layer)
        monkeypatch.setattr(
            "numba.np.ufunc.parallel", _Parallel(started), raising=False
        )
        monkeypatch.setitem(sys.modules, "numba.np.ufunc.parallel", _Parallel(started))
        monkeypatch.setattr("numba.core.config", config, raising=False)
        monkeypatch.setitem(sys.modules, "numba.core.config", config)
        monkeypatch.setattr(openmp, "_gnu_openmp_loaded", lambda: gnu)
        for name in (*openmp.VARIABLES, *openmp.BINDING_VARIABLES):
            monkeypatch.delenv(name, raising=False)
        return config

    return apply


@pytest.mark.parametrize(
    ("state", "platform", "expected"),
    [
        ({"started": True}, "linux", "started"),
        ({}, "darwin", "runtime default"),
        ({"layer": "tbb"}, "linux", "environment"),
        ({"gnu": True}, "linux", "workqueue"),
        ({}, "linux", "passive"),
    ],
)
def test_numbas_threads_start_passive_or_on_the_workqueue_layer(
    numba_state, monkeypatch, state, platform, expected
):
    """Passive and unbound when it can be set; workqueue when the runtime is loaded."""
    config = numba_state(**state)
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    with openmp.numba_launch(platform) as launch:
        inside = (os.environ.get("OMP_WAIT_POLICY"), os.environ.get("OMP_PROC_BIND"))
    assert launch == expected
    assert os.environ.get("OMP_WAIT_POLICY") is None
    assert os.environ["OMP_PROC_BIND"] == "close"
    if expected == "passive":
        assert inside == ("passive", "false")
    if expected == "environment":
        assert inside == (None, "false")
    if expected == "workqueue":
        assert config.THREADING_LAYER == "workqueue"


def test_a_users_wait_policy_is_kept_at_numbas_launch(numba_state, monkeypatch):
    """An explicit wait policy is the user's choice: only binding is stopped."""
    numba_state()
    monkeypatch.setenv("OMP_WAIT_POLICY", "active")
    with openmp.numba_launch("linux") as launch:
        inside = os.environ["OMP_WAIT_POLICY"]
    assert (launch, inside) == ("environment", "active")

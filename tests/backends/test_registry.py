# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Selecting a backend by name, and the three errors that can come back.

``docs/dev/backends.md`` section 1: an unknown name is a :class:`ValueError` listing
what is usable, and a known name whose dependencies are missing is an
:class:`ImportError` naming the extra to install. The two are different questions --
"what did I mean?" and "what do I install?" -- and one error type for both would answer
neither. A name that was real until its backend was deleted is the third question --
"where did it go?" -- and answers it with a
:class:`~solprop.exceptions.BackendError` saying it has been removed and what to use.
"""

import ctypes
import importlib.machinery
import importlib.util
import logging
import shutil
import sys
import types
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from solprop.backends import Backend, available, registry, resolve_name, select
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import BACKENDS, REMOVED_BACKENDS, _construct
from solprop.driver import SimulationSettings
from solprop.exceptions import BackendError
from solprop.stepper import PIController, RungeKutta

pytestmark = pytest.mark.real_auto

LENGTH = 0.01  # m


def test_the_reference_backend_is_always_available():
    """It needs nothing beyond the package's own dependencies, so it is always there."""
    assert "numpy_ref" in available()
    assert set(available()) <= set(BACKENDS)


def test_automatic_selection_probes_without_importing_or_caching(monkeypatch):
    """Installation can change; probing must not initialize a native runtime."""
    probes = []
    present = False
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        if not name.startswith("solprop_native"):
            # Coverage looks up its source package when it sees sys.path change.
            return real_find_spec(name, *args)
        probes.append(name)
        return importlib.machinery.ModuleSpec(name, None) if present else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    before = sys.modules.get("solprop_native")
    with pytest.raises(BackendError, match="not installed"):
        resolve_name()
    present = True
    assert resolve_name() == "native"
    assert resolve_name("numpy_ref") == "numpy_ref"
    assert resolve_name("unregistered") == "unregistered"
    present = False
    with pytest.raises(BackendError, match="not installed"):
        resolve_name()
    # Each "auto" asks whether the package is installed; when it is, its CPU-baseline
    # check is looked up too (this stand-in has no files, so there is none to run).
    assert probes == ["solprop_native"] * 4
    assert sys.modules.get("solprop_native") is before


def test_auto_constructs_only_the_resolved_backend(
    linear_problem, narrow_grid, monkeypatch
):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            importlib.machinery.ModuleSpec(name, None)
            if name == "solprop_native"
            else real_find_spec(name)
        ),
    )
    calls = []
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    result = NumpyRefBackend(problem, grid=narrow_grid)

    def construct(name, problem, **context):
        calls.append(name)
        return result

    monkeypatch.setattr(registry, "_construct", construct)
    try:
        assert select("auto", problem) is result
        assert calls == ["native"]
    finally:
        result.close()


@pytest.mark.parametrize("error", [ImportError, BackendError])
def test_auto_does_not_hide_an_installed_native_failure(
    linear_problem, narrow_grid, monkeypatch, caplog, error
):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, None),
    )
    calls = []

    def construct(name, problem, **context):
        calls.append(name)
        raise error("installed backend is unusable")

    monkeypatch.setattr(registry, "_construct", construct)
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with (
        caplog.at_level(logging.INFO, logger=registry.__name__),
        pytest.raises(error, match="installed backend is unusable"),
    ):
        select("auto", problem)
    assert calls == ["native"]
    assert not caplog.records


def test_availability_follows_the_installed_extras():
    """Detection is by find_spec, so it tracks the environment rather than a cache."""
    installed = importlib.util.find_spec("solprop_native") is not None
    assert ("native" in available()) is installed


def test_select_builds_the_reference_backend(linear_problem, narrow_grid, caplog):
    """The registry returns something that satisfies the seam."""
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with caplog.at_level(logging.INFO, logger=registry.__name__):
        backend = select("numpy_ref", problem, grid=narrow_grid)
    assert isinstance(backend, NumpyRefBackend)
    assert isinstance(backend, Backend)
    assert (backend.name, backend.max_batch) == ("numpy_ref", 1)
    backend.close()
    assert caplog.messages == ["Using numpy_ref backend"]


def test_select_builds_the_installed_native_backend(linear_problem, narrow_grid):
    """The external package is imported only when its registered name is selected."""
    if importlib.util.find_spec("solprop_native") is None:
        pytest.skip("the optional solprop-native package is not installed")
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    backend = select("native", problem, grid=narrow_grid)
    assert (backend.name, backend.max_batch) == ("native", 1)
    backend.close()


def test_optional_constructor_receives_the_original_contract_without_fallback(
    linear_problem, narrow_grid, monkeypatch
):
    package = ModuleType("solprop_native")
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    received = []

    def constructor(actual, **options):
        received.append((actual, options))
        raise BackendError("optional runtime allocation refused")

    monkeypatch.setattr(package, "NativeBackend", constructor, raising=False)
    monkeypatch.setitem(sys.modules, "solprop_native", package)
    with pytest.raises(BackendError, match="optional runtime allocation refused"):
        _construct("native", problem, grid=narrow_grid, modes=None, profiles=None)
    assert len(received) == 1
    assert received[0][0] is problem
    assert received[0][1] == {
        "grid": narrow_grid,
        "modes": None,
        "profiles": None,
        "options": None,
    }


def test_an_unknown_name_lists_what_is_usable(linear_problem, narrow_grid):
    """The first tier: a name this package has never heard of."""
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(ValueError, match="not a known backend; usable backends are"):
        select("numpy-ref", problem)


def test_the_table_holds_the_backends_that_exist():
    """The in-tree CUDA backend is gone, and mlx was retired in SEP 47."""
    assert sorted(BACKENDS) == ["native", "numpy_ref"]


# The tombstones: a removed backend's name is refused wherever a request is validated,
# and never resolves to another backend.

REMOVED = {
    "numba_cpu": "numba_cpu backend was removed in SEP 43",
    "mlx": "mlx backend was retired in SEP 47",
}
"""Each removed backend, and what its refusal says happened to it."""


def assert_names_the_replacements(refused, name):
    """The refusal says the backend is gone and names both backends to use instead."""
    message = str(refused.value)
    assert REMOVED[name] in message
    assert "backend='native'" in message
    assert "backend='numpy_ref'" in message


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_the_removed_backend_is_only_a_tombstone(name):
    assert name in REMOVED_BACKENDS
    assert name not in BACKENDS
    assert name not in available()
    assert name not in available("single")
    assert not set(REMOVED_BACKENDS) & set(BACKENDS)


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_resolving_the_removed_backend_names_the_replacements(name):
    with pytest.raises(BackendError) as refused:
        resolve_name(name)
    assert_names_the_replacements(refused, name)


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_selecting_the_removed_backend_names_the_replacements(
    linear_problem, narrow_grid, monkeypatch, name
):
    def forbidden(*args, **kwargs):
        raise AssertionError("no backend is constructed for a removed name")

    monkeypatch.setattr(registry, "_construct", forbidden)
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(BackendError) as refused:
        select(name, problem, grid=narrow_grid)
    assert_names_the_replacements(refused, name)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("name", sorted(REMOVED))
def test_settings_naming_the_removed_backend_name_the_replacements(name, precision):
    with pytest.raises(BackendError) as refused:
        SimulationSettings(
            controller=PIController(rtol=1e-5, atol=1e-14),
            save_planes=[1e-3],
            dz_max=1e-3,
            dz_init=1e-3,
            backend=name,
            precision=precision,
            integrator=RungeKutta(),
        )
    assert_names_the_replacements(refused, name)


def test_a_known_name_without_its_extra_names_the_extra(
    linear_problem, narrow_grid, monkeypatch
):
    """The second tier: a name that is real and an install that is not complete."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "solprop_native" else real_find_spec(name),
    )
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(
        ImportError,
        match=r"install solprop, or build solprop-native from native/",
    ):
        select("native", problem)


def test_construct_refuses_a_registry_entry_without_an_implementation(
    linear_problem, narrow_grid
):
    """The private dispatcher stays exhaustive when the registry grows."""
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(AssertionError, match="unhandled backend registry entry"):
        _construct("unregistered", problem, grid=None, modes=None, profiles=None)


def test_optional_backend_constructor_forwards_the_context(
    monkeypatch, linear_problem, narrow_grid
):
    """Test import/dispatch wiring without claiming execution of native Kokkos."""
    calls = []
    module = types.ModuleType("solprop_native")

    def construct(problem, **context):
        calls.append((problem, context))
        return NumpyRefBackend(problem, **context)

    monkeypatch.setattr(module, "NativeBackend", construct, raising=False)
    monkeypatch.setitem(sys.modules, "solprop_native", module)
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    backend = _construct("native", problem, grid=narrow_grid, modes=None, profiles=None)
    try:
        assert calls == [
            (
                problem,
                {"grid": narrow_grid, "modes": None, "profiles": None, "options": None},
            )
        ]
        assert isinstance(backend, Backend)
    finally:
        backend.close()


# Automatic selection has no fallback (SEP 43 section 5): an installed distribution
# build whose baseline this processor is below is refused, and so is a missing package.

NATIVE_CPU_CHECK = (
    Path(__file__).parents[2] / "native" / "src" / "solprop_native" / "_cpu.py"
)
# What the installed package's check reports, as a stand-in _cpu.py the probe loads.
SHORTFALL_CHECK = """
class CpuBaselineError(ImportError):
    def __init__(self, baseline, missing):
        self.baseline = baseline
        self.missing_features = tuple(missing)
        super().__init__(f"requires an {baseline} processor")

def baseline_shortfall(package_dir):
    record = (package_dir / "_cpu_baseline.txt").read_text(encoding="utf-8").strip()
    if record == "met":
        return None
    return CpuBaselineError(record, ["avx2", "fma"])
"""


def install_native(
    monkeypatch, tmp_path, baseline, check: str | Path = SHORTFALL_CHECK
):
    """Make an installed solprop_native, files only, that find_spec reports."""
    package = tmp_path / "solprop_native"
    package.mkdir()
    if isinstance(check, Path):
        shutil.copyfile(check, package / "_cpu.py")
    else:
        (package / "_cpu.py").write_text(check, encoding="utf-8")
    (package / "_cpu_baseline.txt").write_text(f"{baseline}\n", encoding="utf-8")
    spec = importlib.machinery.ModuleSpec("solprop_native", None, is_package=True)
    spec.submodule_search_locations = [str(tmp_path / "elsewhere"), str(package)]
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: spec if name == "solprop_native" else real_find_spec(name),
    )
    return package


def hide_native(monkeypatch):
    """Make solprop-native absent, whatever this lane has installed."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "solprop_native" else real_find_spec(name),
    )


def test_auto_refuses_without_the_native_package(monkeypatch):
    hide_native(monkeypatch)
    with pytest.raises(BackendError, match="is not installed") as refused:
        resolve_name()
    assert "pip install solprop" in str(refused.value)
    assert "uv sync" in str(refused.value)
    assert 'backend="numpy_ref"' in str(refused.value)


def test_settings_on_auto_refuse_without_the_native_package(monkeypatch):
    hide_native(monkeypatch)
    with pytest.raises(BackendError, match="solprop-native is not installed"):
        SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1e-3],
            dz_max=1e-3,
            dz_init=1e-3,
        )


@pytest.mark.parametrize("name", ["numpy_ref", "native"])
def test_a_named_backend_is_not_refused_at_resolution(monkeypatch, name):
    hide_native(monkeypatch)
    assert resolve_name(name) == name


def test_auto_refuses_below_the_native_baseline_without_importing_it(
    monkeypatch, tmp_path
):
    install_native(monkeypatch, tmp_path, "x86-64-v3")
    before = sys.modules.get("solprop_native")
    with pytest.raises(BackendError, match="requires an x86-64-v3") as refused:
        resolve_name()
    assert isinstance(refused.value.__cause__, ImportError)
    assert sys.modules.get("solprop_native") is before


def test_auto_keeps_native_on_a_processor_that_meets_the_baseline(
    monkeypatch, tmp_path
):
    install_native(monkeypatch, tmp_path, "met")
    assert resolve_name() == "native"


def test_explicit_native_still_raises_the_baseline_refusal(
    linear_problem, narrow_grid, monkeypatch, tmp_path
):
    """No fallback for a named backend: the package's import refusal surfaces."""
    install_native(monkeypatch, tmp_path, "x86-64-v3")
    calls = []

    def construct(name, problem, **context):
        calls.append(name)
        raise ImportError(
            "this solprop-native build requires an x86-64-v3 processor, and this "
            "one lacks avx2, fma"
        )

    monkeypatch.setattr(registry, "_construct", construct)
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with (
        warnings.catch_warnings(),
        pytest.raises(ImportError, match="requires an x86-64-v3 processor"),
    ):
        warnings.simplefilter("error")
        select("native", problem)
    assert calls == ["native"]


@pytest.mark.parametrize("error", [ImportError, RuntimeError, BackendError])
def test_auto_raises_other_native_failures_above_the_baseline(
    linear_problem, narrow_grid, monkeypatch, tmp_path, error
):
    """A refusal at import, a stale build or a broken extension still raise."""
    install_native(monkeypatch, tmp_path, "met")
    calls = []

    def construct(name, problem, **context):
        calls.append(name)
        raise error("installed backend is unusable")

    monkeypatch.setattr(registry, "_construct", construct)
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(error, match="installed backend is unusable"):
        select("auto", problem)
    assert calls == ["native"]


def test_auto_runs_the_native_package_s_own_check(monkeypatch, tmp_path):
    """The real _cpu.py, loaded by path, on a stand-in Windows Sandy Bridge."""
    sandy_bridge = {14, 17, 36, 37, 38, 39}  # PF_* constants: no AVX2 (40)

    class Kernel32:
        @staticmethod
        def IsProcessorFeaturePresent(constant):  # noqa: N802 - the Windows name
            return int(constant in sandy_bridge)

    monkeypatch.setattr(ctypes, "WinDLL", lambda name: Kernel32(), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    install_native(monkeypatch, tmp_path, "x86-64-v3", check=NATIVE_CPU_CHECK)
    with pytest.raises(
        BackendError, match="lacks abm, avx2, bmi1, bmi2, f16c, fma, movbe"
    ):
        resolve_name()
    sandy_bridge.add(40)
    assert resolve_name() == "native"


def test_a_build_without_a_check_or_record_keeps_native(monkeypatch, tmp_path):
    """A local or editable build records no baseline; a stand-in has no files."""
    package = install_native(monkeypatch, tmp_path, "unused", check=NATIVE_CPU_CHECK)
    (package / "_cpu_baseline.txt").unlink()
    assert resolve_name() == "native"
    (package / "_cpu.py").unlink()
    assert resolve_name() == "native"


# SOLPROP_REQUIRE_DEVICE (SEP 43 section 5): checked at selection, on the actual build.


def fake_build(monkeypatch, backend, count):
    """Make ``solprop_native`` report a build, without a device to run it on."""
    fake: Any = types.ModuleType("solprop_native")
    fake.__version__ = "0.3.0+cuda"
    fake.build_info = lambda: types.SimpleNamespace(
        device_backend=backend, device_count=count
    )
    spec = importlib.machinery.ModuleSpec("solprop_native", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: spec if name == "solprop_native" else real_find_spec(name),
    )
    monkeypatch.setitem(sys.modules, "solprop_native", fake)


@pytest.mark.parametrize(
    ("value", "backend", "count", "message"),
    [
        ("cuda", "none", 0, "device backend 'none'"),
        ("hip", "cuda", 1, "device backend 'cuda'"),
        ("any", "none", 0, "device backend 'none'"),
        ("cuda", "cuda", 0, "sees no device"),
        ("gpu", "cuda", 1, "use one of any, cuda, hip"),
    ],
)
def test_a_required_device_refuses_what_does_not_meet_it(
    monkeypatch, value, backend, count, message
):
    fake_build(monkeypatch, backend, count)
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, value)
    with pytest.raises(BackendError, match=message):
        registry._check_required_device("native")


@pytest.mark.parametrize(
    ("value", "backend"),
    [("cuda", "cuda"), ("hip", "hip"), ("any", "hip"), ("", "none")],
)
def test_a_required_device_passes_what_meets_it(monkeypatch, value, backend):
    fake_build(monkeypatch, backend, 1 if backend != "none" else 0)
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, value)
    registry._check_required_device("native")


@pytest.mark.parametrize("name", ["numpy_ref"])
def test_a_required_device_refuses_the_other_backends_by_name(monkeypatch, name):
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "any")
    with pytest.raises(BackendError, match=f"this run is on {name}"):
        registry._check_required_device(name)


def test_a_required_device_is_refused_before_anything_is_constructed(
    linear_problem, narrow_grid, monkeypatch
):
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "cuda")
    built = []
    monkeypatch.setattr(registry, "_construct", lambda *a, **k: built.append(a))
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    with pytest.raises(BackendError, match="this run is on numpy_ref"):
        select("numpy_ref", problem, grid=narrow_grid)
    assert not built


def test_a_required_device_leaves_a_missing_package_to_the_selection(monkeypatch):
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "cuda")
    monkeypatch.setattr(registry, "_satisfied", lambda backend: False)
    assert registry._check_required_device("native") is None


def test_a_required_device_reports_a_build_that_will_not_load(monkeypatch):
    """A CUDA build with no visible device raises from Kokkos as the package loads."""
    spec = importlib.machinery.ModuleSpec("solprop_native", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: spec if name == "solprop_native" else real_find_spec(name),
    )
    monkeypatch.setitem(sys.modules, "solprop_native", None)
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "cuda")
    with pytest.raises(BackendError, match="did not load"):
        registry._check_required_device("native")

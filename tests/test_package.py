# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import pkgutil
import subprocess
import sys
import typing

import numpy as np
import pytest
from jaxtyping import TypeCheckError

import solprop
from runtime_types import requires_typecheck
from solprop import backends, exceptions
from solprop.physdata.sellmeier import _wavelength_extremes

HEAVY_MODULES = (
    "numba",
    "matplotlib",
    "h5py",
    "pyfftw",
    "tqdm",
)


def test_version_is_a_string():
    assert isinstance(solprop.__version__, str)
    assert solprop.__version__


def test_package_logger_has_null_handler():
    handlers = logging.getLogger("solprop").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_top_level_import_does_not_load_heavy_modules():
    code = (
        "import sys, solprop; "
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_no_removed_device_backend_survives_in_the_package_or_its_imports():
    # The in-tree device backend has been removed. The check is by substring rather
    # than by the deleted name so that it also catches a reintroduction under a
    # neighbouring spelling, and the import runs in a subprocess because what matters
    # is what importing the seam pulls in -- no device runtime, ever. The subprocess
    # measures what the import *adds*, because an interpreter may already carry an
    # unrelated redirector module installed by a wheel this package never asked for.
    on_disk = [module.name for module in pkgutil.iter_modules(backends.__path__)]
    assert on_disk and not [name for name in on_disk if "cuda" in name.lower()]
    assert not [name for name in dir(backends) if "cuda" in name.lower()]
    code = (
        "import sys; before = set(sys.modules); import solprop.backends; "
        "print(sorted(m for m in set(sys.modules) - before if 'cuda' in m.lower()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_every_submodule_is_an_attribute_loaded_on_first_use():
    """PEP 562: ``dir`` and tab completion see the submodules; importing stays cheap."""
    # A module whose name starts with an underscore is implementation (the display
    # formatter), not surface, exactly as the structure figures treat it:
    # it is neither an attribute nor offered by completion.
    on_disk = sorted(
        module.name
        for module in pkgutil.iter_modules(solprop.__path__)
        if not module.name.startswith("_")
    )
    assert sorted(solprop._SUBMODULES) == on_disk
    assert "_display" not in dir(solprop)
    assert set(on_disk) <= set(dir(solprop))
    # The package's own imports are implementation, not surface: completion should
    # offer the modules and the public names, not `importlib` or `logging`.
    assert {"importlib", "logging", "Final", "ModuleType"}.isdisjoint(dir(solprop))
    assert {"__version__", "configure_logging", "SolpropError"} <= set(dir(solprop))
    code = (
        "import sys, solprop; before = 'solprop.plotting' in sys.modules; "
        "module = solprop.plotting; print(before, module.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "False solprop.plotting"
    assert solprop.grid is sys.modules["solprop.grid"]
    with pytest.raises(AttributeError, match="no attribute 'banana'"):
        solprop.banana  # noqa: B018 - the lookup is the test


def test_exception_hierarchy_roots_at_solprop_error():
    for name in exceptions.__all__:
        cls = getattr(exceptions, name)
        if isinstance(cls, typing.TypeAliasType):
            # ParameterPath, the type of an error's parameter, is not an error.
            continue
        if cls is exceptions.RunTerminated:
            # A stop, not an error: an `except Exception` must not swallow it.
            assert issubclass(cls, SystemExit) and not issubclass(cls, Exception)
            continue
        root = (
            exceptions.SolpropWarning
            if name.endswith("Warning")
            else exceptions.SolpropError
        )
        assert issubclass(cls, root)
    assert issubclass(exceptions.GridError, ValueError)
    assert issubclass(exceptions.BackendError, RuntimeError)


def test_warning_hierarchy_roots_at_solprop_warning():
    assert issubclass(exceptions.SolpropWarning, UserWarning)
    assert issubclass(exceptions.MaterialRangeWarning, exceptions.SolpropWarning)
    assert not issubclass(exceptions.SolpropWarning, exceptions.SolpropError)


@requires_typecheck
def test_runtime_shape_checking_is_installed():
    # The jaxtyping import hook in conftest.py is what enforces every array annotation
    # in the package. It is easy for it to be silently inactive -- if it is scoped to
    # `import solprop`, nothing but exceptions and log gets instrumented, because the
    # top level deliberately imports almost nothing. This test fails loudly instead.
    assert _wavelength_extremes(np.ones(3)) == (pytest.approx(1.883652e9),) * 2
    with pytest.raises(TypeCheckError):
        _wavelength_extremes(np.ones(3, dtype=np.int32))


def test_the_runtime_imports_without_posix_only_modules():
    # Windows has neither fcntl nor resource; a module that imports either at load
    # time would make single runs and scans unusable there. Blocking both reproduces
    # that on any platform.
    code = (
        "import sys; sys.modules['fcntl'] = None; sys.modules['resource'] = None; "
        "import solprop, solprop.sol, solprop.scan, solprop.output.scan, "
        "solprop.processing.scans, solprop.driver.executors, "
        "solprop.backends.numba_cpu"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

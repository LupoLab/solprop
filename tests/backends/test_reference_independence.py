# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference runs on its own host machinery, never on the native bridge's.

SEP 47 section 6.6 and the maintainer's decision at acceptance on the transition: the
reference session is built independently in phase 2 and shares only stateless
numerical modules with the inherited driver, which remains solely as the native
bridge. These checks hold the reference package (``solprop.backends.numpy_ref``) and
its host package (``solprop.backends.host``) to that: no import of a legacy control
module or class, at module level or inside a function; no legacy class among their
module globals or in a reference backend's ancestry; the reference's marches and its
group are the host copies; and the run layer serves a reference backend with the host
session.
"""

import ast
import importlib
import pkgutil
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import solprop.backends.host
import solprop.backends.numpy_ref
from scripted_driver import ScriptedDriver
from solprop.backends import batch, driver, low_storage, multistep, multistep_general
from solprop.backends.driver_session import DriverSession
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.group import HostGroup
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.session import HostSession
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.backends.sessions import start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep
from solprop.translate.run_request import describe_run

PACKAGES = (solprop.backends.numpy_ref, solprop.backends.host)
"""The reference backend and its host machinery."""

LEGACY_MODULES = (
    "solprop.backends.driver",
    "solprop.backends.batch",
    "solprop.backends.multistep_general",
)
"""The native bridge's control modules, which hold nothing else."""

LEGACY_NAMES = {
    "solprop.backends.multistep": {"AdamsMarch"},
    "solprop.backends.low_storage": {"LowStorageMarch"},
}
"""The bridge's classes in modules whose other contents are shared numerical code."""

LEGACY_CLASSES = (
    driver.StepDriver,
    multistep.AdamsMarch,
    multistep_general.GeneralPath,
    low_storage.LowStorageMarch,
    batch.LockstepRunner,
)
"""The bridge's control classes."""


def _modules() -> list[str]:
    """Return the dotted name of every module of the two packages."""
    names = []
    for package in PACKAGES:
        names.append(package.__name__)
        names.extend(
            info.name
            for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
        )
    return sorted(names)


MODULES = _modules()


def _imports(name: str) -> list[tuple[str, set[str]]]:
    """Return every import a module's source makes, function bodies included.

    Parameters
    ----------
    name : str
        The module.

    Returns
    -------
    list
        ``(module, names)`` per import: the imported module and the names taken from
        it, empty for a plain ``import``.
    """
    module = importlib.import_module(name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, set()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            taken = {alias.name for alias in node.names}
            found.append((node.module, taken))
            # `from solprop.backends import driver` imports the module itself.
            found.extend((f"{node.module}.{one}", set()) for one in taken)
    return found


def test_the_packages_are_found():
    assert "solprop.backends.numpy_ref.backend" in MODULES
    assert "solprop.backends.host.driver" in MODULES
    assert "solprop.backends.host.session" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_a_reference_module_imports_no_legacy_control(name):
    for module, taken in _imports(name):
        assert module not in LEGACY_MODULES, (name, module)
        assert not taken & LEGACY_NAMES.get(module, set()), (name, module, taken)


@pytest.mark.parametrize("name", MODULES)
def test_a_reference_module_holds_no_legacy_control(name):
    module = importlib.import_module(name)
    for key, value in vars(module).items():
        if isinstance(value, ModuleType):
            assert value.__name__ not in LEGACY_MODULES, (name, key)
        else:
            assert not any(value is legacy for legacy in LEGACY_CLASSES), (name, key)


@pytest.mark.parametrize(
    "backend", [NumpyRefBackend, NumpySingleBackend, NumpyBatchBackend]
)
def test_a_reference_backend_inherits_only_the_host_driver(backend):
    assert issubclass(backend, HostDriver)
    assert not set(backend.__mro__) & set(LEGACY_CLASSES)


def test_the_reference_marches_and_groups_with_the_host_copies(
    soliton_problem, narrow_grid
):
    length = 1e-3
    schedule = FixedSchedule.uniform(0.0, length, length / 4, save_planes=[length])
    marches = {
        "_adams": (Multistep(), HostAdamsMarch),
        "_low_storage": (LowStorageRungeKutta(), HostLowStorageMarch),
    }
    for attribute, (integrator, march) in marches.items():
        problem = soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            schedule=schedule,
            dz_max=length / 4,
            dz_init=length / 4,
            integrator=integrator,
        )
        with closing(NumpyRefBackend(problem, grid=narrow_grid)) as backend:
            assert type(getattr(backend, attribute)) is march
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=length / 4,
        dz_init=length / 4,
    )
    member = NumpyRefBackend(problem, grid=narrow_grid)
    member.set_initial(problem.initial.field0, None)
    with closing(NumpyBatchBackend([member])) as group:
        assert isinstance(group, HostGroup)


def test_the_run_layer_serves_the_reference_with_the_host_session(
    soliton_problem, narrow_grid
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    request = describe_run(problem)
    reference = NumpyRefBackend(problem, grid=narrow_grid)
    with start_session(reference, problem, request, None) as session:
        assert type(session) is HostSession
    scripted = cast("Any", ScriptedDriver(problem, grid=narrow_grid))
    with start_session(scripted, problem, request, None) as session:
        assert type(session) is DriverSession


def test_the_host_session_refuses_a_bridge_driver(soliton_problem, narrow_grid):
    problem = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    scripted = cast("Any", ScriptedDriver(problem, grid=narrow_grid))
    try:
        with pytest.raises(BackendError, match="host driver"):
            HostSession(scripted, problem, describe_run(problem), None)
    finally:
        scripted.close()

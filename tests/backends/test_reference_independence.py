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
session and a reference group with the host group session. The bridge's transitional
sessions (:mod:`solprop.backends.driver_session`) are legacy control too: the
reference composes the parts every Python session shares
(:mod:`solprop.backends.session_core`) instead.
"""

import ast
import importlib
import os
import pkgutil
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import solprop.backends.host
import solprop.backends.numpy_ref
from scripted_driver import ScriptedDriver
from solprop.backends import batch, driver, driver_session, low_storage, multistep
from solprop.backends.driver_session import DriverSession
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.group import HostGroup
from solprop.backends.host.group_session import HostGroupSession
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.session import HostSession
from solprop.backends.multistep import HostHistory
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.backends.session import Start
from solprop.backends.sessions import start_group, start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep
from solprop.stepper.multistep import ComparisonMultistep

PACKAGES = (solprop.backends.numpy_ref, solprop.backends.host)
"""The reference backend and its host machinery."""

LEGACY_MODULES = (
    "solprop.backends.driver",
    "solprop.backends.batch",
    "solprop.backends.driver_session",
)
"""The native bridge's control modules and transitional sessions, which hold nothing
else."""

LEGACY_NAMES = {
    "solprop.backends.multistep": {"AdamsMarch"},
    "solprop.backends.low_storage": {"LowStorageMarch"},
}
"""The bridge's classes in modules whose other contents are shared numerical code."""

LEGACY_CLASS_NAMES = frozenset(
    {
        "StepDriver",
        "AdamsMarch",
        "GeneralPath",
        "LowStorageMarch",
        "LockstepRunner",
        "DriverSession",
        "DriverPrepared",
        "DriverGroupSession",
        "DriverPreparedGroup",
    }
)
"""The bridge's control classes by name: no identifier of the reference may be one,
whatever route reached it -- an import, an alias, or a module attribute."""

LEGACY_CLASSES = (
    driver.StepDriver,
    multistep.AdamsMarch,
    low_storage.LowStorageMarch,
    batch.LockstepRunner,
    driver_session.DriverSession,
    driver_session.DriverPrepared,
    driver_session.DriverGroupSession,
    driver_session.DriverPreparedGroup,
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


def _tree(name: str) -> ast.Module:
    """Return a module's parsed source.

    Parameters
    ----------
    name : str
        The module.

    Returns
    -------
    ast.Module
        Its syntax tree.
    """
    module = importlib.import_module(name)
    assert module.__file__ is not None
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


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
    found: list[tuple[str, set[str]]] = []
    for node in ast.walk(_tree(name)):
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
    assert "solprop.backends.host.group_session" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_a_reference_module_imports_no_legacy_control(name):
    for module, taken in _imports(name):
        assert module not in LEGACY_MODULES, (name, module)
        assert not taken & LEGACY_NAMES.get(module, set()), (name, module, taken)


@pytest.mark.parametrize("name", MODULES)
def test_a_reference_module_names_no_legacy_control_by_any_route(name):
    # `from solprop.backends import multistep` then `multistep.AdamsMarch`, an alias
    # (`import solprop.backends.low_storage as ls`; `ls.LowStorageMarch`) or an
    # imported name: every route names the class, as an identifier or an attribute.
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.Name):
            assert node.id not in LEGACY_CLASS_NAMES, (name, node.id, node.lineno)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in LEGACY_CLASS_NAMES, (name, node.attr, node.lineno)
        elif isinstance(node, ast.alias):
            assert node.name.rsplit(".", 1)[-1] not in LEGACY_CLASS_NAMES, (
                name,
                node.name,
            )


def test_the_route_check_sees_a_module_attribute():
    # The check above on a snippet that reaches a legacy class through its module.
    tree = ast.parse(
        "from solprop.backends import multistep as m\nmarch = m.AdamsMarch\n"
    )
    attributes = [
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    ]
    assert set(attributes) & LEGACY_CLASS_NAMES == {"AdamsMarch"}


def test_importing_the_reference_loads_no_legacy_driver():
    # A fresh interpreter, without the native package: the reference, its session and
    # the run layer's door load neither the inherited driver nor the bridge's batch.
    code = (
        "import sys\n"
        "import solprop.backends.numpy_ref, solprop.backends.numpy_ref.batch\n"
        "import solprop.backends.numpy_ref.single, solprop.backends.host.session\n"
        "import solprop.backends.host.adams, solprop.backends.host.group\n"
        "import solprop.backends.host.group_session\n"
        "import solprop.backends.sessions\n"
        f"watched = {{*{list(LEGACY_MODULES)!r}, 'solprop_native'}}\n"
        "print(','.join(sorted(set(sys.modules) & watched)))\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "SOLPROP_CALIBRATION": "off"},
    )
    assert found.stdout.strip() == "", found.stdout


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
        problem, run_request = soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            schedule=schedule,
            dz_max=length / 4,
            dz_init=length / 4,
            integrator=integrator,
        )
        with closing(
            NumpyRefBackend(problem, run_request, grid=narrow_grid)
        ) as backend:
            assert type(getattr(backend, attribute)) is march
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=length / 4,
        dz_init=length / 4,
    )
    member = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    member.set_initial(problem.initial.field0, None)
    with closing(NumpyBatchBackend([member])) as group:
        assert isinstance(group, HostGroup)


def test_the_run_layer_serves_the_reference_with_the_host_session(
    soliton_problem, narrow_grid
):
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    request = run_request
    reference = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    with start_session(reference, problem, request, Start.entrance(problem)) as session:
        assert type(session) is HostSession
    scripted = cast("Any", ScriptedDriver(problem, run_request, grid=narrow_grid))
    with start_session(scripted, problem, request, Start.entrance(problem)) as session:
        assert type(session) is DriverSession


def test_the_run_layer_serves_a_reference_group_with_the_host_group_session(
    soliton_problem, narrow_grid
):
    length = 1e-3
    schedule = FixedSchedule.uniform(0.0, length, length / 4, save_planes=[length])
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=length / 4,
        dz_init=length / 4,
    )
    members = [NumpyRefBackend(problem, run_request, grid=narrow_grid)]
    with start_group(
        members, [problem], [run_request], [Start.entrance(problem)]
    ) as group:
        assert type(group) is HostGroupSession
        assert isinstance(getattr(group, "_runner"), NumpyBatchBackend)  # noqa: B009


def test_the_host_session_refuses_a_bridge_driver(soliton_problem, narrow_grid):
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    scripted = cast("Any", ScriptedDriver(problem, run_request, grid=narrow_grid))
    try:
        with pytest.raises(BackendError, match="host driver"):
            HostSession(scripted, problem, run_request, Start.entrance(problem))
    finally:
        scripted.close()


def test_an_injected_adams_engine_is_gated_and_checked_as_the_bridge_s(
    soliton_problem, narrow_grid
):
    # The bridge's march takes a resident engine on the fused and exponential paths
    # only; the reference's takes an injected one there alone, and names what it lacks.
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    with closing(NumpyRefBackend(problem, run_request, grid=narrow_grid)) as backend:
        with pytest.raises(BackendError, match="exponential path lacks"):
            HostAdamsMarch(backend, Multistep(), engine=object())
        general = HostAdamsMarch(backend, ComparisonMultistep(order=3), engine=object())
        assert isinstance(general._engine, HostHistory)

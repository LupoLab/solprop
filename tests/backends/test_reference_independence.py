# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference runs on its own host machinery; the legacy control is gone.

SEP 47 section 6.6 and the maintainer's decision at acceptance on the transition: the
reference session is built independently and shares only stateless numerical modules
with the native backend, and phase 5 deletes the step driver the native backend
inherited (``solprop.backends.driver``), its Adams and low-storage marches and the
bridge's sessions. These checks hold the reference package
(``solprop.backends.numpy_ref``) and its host package (``solprop.backends.host``) to
that -- no import of a legacy control module or class, at module level or inside a
function, by any route -- and the tree to the deletion: no legacy module imports and no
legacy class remains in the shared modules. The reference's marches and its group are
the host copies, and the run layer serves a reference backend with the host session and
a reference group with the host group session.
"""

import ast
import importlib
import importlib.util
import os
import pkgutil
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest

import solprop.backends.host
import solprop.backends.numpy_ref
from solprop.backends import low_storage
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.group import HostGroup
from solprop.backends.host.group_session import HostGroupSession
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.session import HostSession
from solprop.backends.numpy_ref import NumpyRefBackend, multistep
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.session import Start
from solprop.backends.sessions import start_group, start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep

PACKAGES = (solprop.backends.numpy_ref, solprop.backends.host)
"""The reference backend and its host machinery."""

LEGACY_MODULES = (
    "solprop.backends.driver",
    "solprop.backends.batch",
    "solprop.backends.driver_session",
)
"""The native bridge's control module and the transitional sessions and lockstep
runner deleted in SEP 47 phase 5, which held nothing else: none may come back into the
reference."""

LEGACY_NAMES = {
    "solprop.backends.numpy_ref.multistep": {"AdamsMarch"},
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


def test_the_legacy_control_is_deleted():
    # SEP 47 phase 5: the modules are gone and the shared modules hold no legacy class.
    for name in LEGACY_MODULES:
        assert importlib.util.find_spec(name) is None, name
    for module in (low_storage, multistep):
        assert not set(vars(module)) & LEGACY_CLASS_NAMES, module.__name__


def test_importing_the_reference_loads_no_legacy_driver():
    # A fresh interpreter, without the native package: the reference, its session and
    # the run layer's door load neither the inherited driver nor the bridge's batch.
    code = (
        "import sys\n"
        "import solprop.backends.numpy_ref, solprop.backends.numpy_ref.batch\n"
        "import solprop.backends.host.session\n"
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
            legacy = isinstance(value, type) and value.__name__ in LEGACY_CLASS_NAMES
            assert not legacy, (name, key)


@pytest.mark.parametrize("backend", [NumpyRefBackend, NumpyBatchBackend])
def test_a_reference_backend_inherits_only_the_host_driver(backend):
    assert issubclass(backend, HostDriver)
    assert not {one.__name__ for one in backend.__mro__} & LEGACY_CLASS_NAMES


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


def test_an_injected_adams_engine_is_gated_and_checked_as_the_bridge_s(
    soliton_problem, narrow_grid
):
    # The reference's march takes an injected engine and names what it lacks.
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    with closing(NumpyRefBackend(problem, run_request, grid=narrow_grid)) as backend:
        with pytest.raises(BackendError, match="exponential path lacks"):
            HostAdamsMarch(backend, Multistep(), engine=object())
        with pytest.raises(BackendError, match="fused path lacks"):
            HostAdamsMarch(
                backend, Multistep(formulation="interaction"), engine=object()
            )

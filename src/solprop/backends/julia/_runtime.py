# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Start one verified Julia stack per process, with lazy device loading.

Julia's runtime cannot be unloaded. Failed initialization or extension loading is
therefore a process-lifetime failure, and a fork after initialization is refused.
The accepted SIGTERM limitation is described in SEP 48 section 5.4.
"""

from __future__ import annotations

import importlib
import os
import signal
import sys
import threading
from ctypes.util import dllist
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solprop import _process
from solprop.backends.julia._identity import SetupRecord
from solprop.backends.julia._setup import select_setup
from solprop.exceptions import BackendError

__all__ = ["JuliaRuntime", "get_runtime"]

_DEVICE_IMPORTS = {
    "cuda": "import CUDA",
    "rocm": "import AMDGPU",
    "metal": "import Metal",
}


@dataclass(frozen=True)
class _ProcessIdentity:
    """Pin content identity and physical locations used by the running loader."""

    setup_id: str
    executable: Path
    depot: Path
    engine: Path
    projects: tuple[tuple[str, Path], ...]


def _process_identity(record: SetupRecord) -> _ProcessIdentity:
    """Resolve the installation locations before the runtime fixes its load paths.

    Parameters
    ----------
    record : SetupRecord
        Verified installation.

    Returns
    -------
    _ProcessIdentity
        Content identity and resolved runtime, source and environment paths.
    """
    return _ProcessIdentity(
        record.setup_id,
        record.julia_executable.resolve(),
        record.depot.resolve(),
        record.engine.resolve(),
        tuple((device, path.resolve()) for device, path in record.projects),
    )


@dataclass(eq=False)
class JuliaRuntime:
    """Retain the process's Julia runtime and its immutable installation identity.

    Attributes
    ----------
    setup : SetupRecord
        The verified installation fixed at first initialization.
    threads : int
        Julia's default-pool width, fixed within the process thread budget.
    namespace : object
        An isolated Julia module used only for package loading and bridge access.
    engine : object
        The opaque ``SolpropJulia`` module; bridge calls are dynamically typed.
    devices : set of str
        Device packages already loaded successfully into this runtime.
    """

    setup: SetupRecord
    threads: int
    # JuliaCall synthesizes these module attributes at runtime; keep its untyped
    # boundary here rather than pretending that they are Python protocol objects.
    namespace: Any
    engine: Any
    devices: set[str] = field(default_factory=lambda: {"cpu"})
    _identity: _ProcessIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Pin physical paths as well as content identity before later selections."""
        self._identity = _process_identity(self.setup)

    def check(self) -> None:
        """Refuse a forked or poisoned process before another engine operation.

        Raises
        ------
        BackendError
            If initialization or extension loading failed, or this is a fork child.
        """
        if _STATE.pid != os.getpid():
            raise BackendError(
                "the julia runtime cannot run after fork; use a spawned fresh process"
            )
        if _STATE.failure is not None:
            raise BackendError(
                "the julia runtime failed during initialization or device loading; "
                "start a fresh process"
            ) from _STATE.failure


@dataclass
class _State:
    """Serialize startup and retain its process-lifetime success or failure."""

    pid: int = field(default_factory=os.getpid)
    lock: threading.Lock = field(default_factory=threading.Lock)
    attempted: bool = False
    failure: BaseException | None = None
    runtime: JuliaRuntime | None = None


_STATE = _State()


def get_runtime(*, setup_id: str | None = None, device: str = "cpu") -> JuliaRuntime:
    """Select and verify a setup before starting Julia or loading a device extension.

    Parameters
    ----------
    setup_id : str or None, optional
        The setup's published identity, or unique compatible selection.
    device : str, optional
        ``cpu``, ``cuda``, ``rocm`` or ``metal``. The whole installed stack is checked.

    Returns
    -------
    JuliaRuntime
        The process's single runtime, with the requested device package loaded.

    Raises
    ------
    BackendError
        For an incomplete installation, incompatible process identity, foreign Julia
        initialization, worker-thread first initialization, fork, or failed startup.
    """
    # Check before acquiring an inherited lock: the parent may have held it when
    # fork occurred. A process that only imported this module has no Julia state.
    if _STATE.pid != os.getpid():
        if _STATE.attempted:
            raise BackendError(
                "the julia runtime cannot run after fork; use a spawned fresh process"
            )
        _STATE.pid = os.getpid()
        _STATE.lock = threading.Lock()
    with _STATE.lock:
        if _STATE.failure is not None:
            raise BackendError(
                "the julia runtime previously failed; start a fresh process"
            ) from _STATE.failure
        record = select_setup(setup_id=setup_id, device=device)
        runtime = _STATE.runtime
        if runtime is None:
            runtime = _initialize(record)
            _STATE.runtime = runtime
        elif runtime._identity != _process_identity(record):
            raise BackendError(
                f"the julia process uses setup {runtime.setup.setup_id}; "
                f"setup {record.setup_id} or its environment location "
                "requires a fresh process"
            )
        if _process.thread_budget().threads < runtime.threads:
            raise BackendError(
                "the julia thread pool exceeds the current process budget; "
                "start a fresh process with the smaller budget"
            )
        if device not in runtime.devices:
            # select_setup checked all inventories before any extension import.
            # Import failure can leave partially initialized modules and cannot be
            # rolled back by deleting entries from Python's sys.modules.
            try:
                runtime.namespace.seval(_DEVICE_IMPORTS[device])
            except BaseException as error:
                _STATE.failure = error
                if not isinstance(error, Exception):
                    raise
                raise BackendError(
                    f"the julia {device} extension failed to load; "
                    "start a fresh process"
                ) from error
            runtime.devices.add(device)
        return runtime


def _initialize(record: SetupRecord) -> JuliaRuntime:
    """Start the verified stack on the main thread, poisoning any partial startup.

    Parameters
    ----------
    record : SetupRecord
        Installation whose sources and artifacts have passed preflight.

    Returns
    -------
    JuliaRuntime
        Loaded CPU engine with its stop setter compiled.

    Raises
    ------
    BackendError
        If startup is not allowed or the loaded runtime differs from its setup.
    """
    if threading.current_thread() is not threading.main_thread():
        raise BackendError(
            "first julia preparation must run on Python's main thread before workers"
        )
    if "juliacall" in sys.modules:
        raise BackendError(
            "juliacall was imported outside solprop's verified setup; "
            "start a fresh process"
        )
    if any(
        Path(path).name.lower().startswith(("libjulia.", "libjulia-"))
        for path in dllist()
    ):
        raise BackendError(
            "libjulia was loaded outside solprop's verified setup; "
            "start a fresh process"
        )
    if any(name.startswith("juliacall-") for name in sys._xoptions):
        raise BackendError(
            "Python -X juliacall options override the verified runtime; "
            "start a fresh process without them"
        )
    if any(os.pathsep in str(path) for _, path in record.projects):
        raise BackendError(
            "the julia setup path contains the load-path separator; "
            "prepare it in a different depot"
        )
    threads = _process.thread_budget().threads
    previous_sigint = signal.getsignal(signal.SIGINT)
    _configure(record, threads)
    _STATE.attempted = True
    try:
        try:
            bridge = importlib.import_module("juliacall")
        finally:
            # Julia needs its installed GC handlers and signal mask. Restoring
            # that mask duplicates macOS SIGINT delivery; preserve only Python's
            # actual handler. Julia's fatal SIGTERM listener remains installed.
            signal.signal(signal.SIGINT, previous_sigint)
        if bridge.__version__ != record.bridge_version:
            raise BackendError("the loaded juliacall version differs from its setup")
        namespace = bridge.newmodule("SolpropRuntime")
        _verify_runtime(namespace, record, threads)
        namespace.seval(
            "import Pkg; Pkg.offline(true); import PythonCall; import SolpropJulia"
        )
        if namespace.seval("string(VERSION)") != record.julia_version:
            raise BackendError("the loaded Julia runtime differs from its setup")
        pythoncall_version = namespace.seval("string(Base.pkgversion(PythonCall))")
        if pythoncall_version != record.bridge_version:
            raise BackendError("the loaded PythonCall version differs from its setup")
        if not namespace.seval(
            "Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing"
        ):
            raise BackendError("the julia PythonCall extension did not activate")
        runtime = JuliaRuntime(record, threads, namespace, namespace.SolpropJulia)
        # Compile the leaf now; a stop listener must never enter the JIT or Julia's
        # foreign-thread adoption path when a different thread requests a stop.
        runtime.engine.stop_setter_address()
        return runtime
    except BaseException as error:
        _STATE.failure = error
        if not isinstance(error, Exception):
            raise
        raise BackendError(
            "the julia runtime failed to start; "
            "start a fresh process after fixing setup"
        ) from error


def _verify_runtime(namespace: Any, record: SetupRecord, threads: int) -> None:
    """Check Julia's actual environment before importing any engine code.

    Parameters
    ----------
    namespace : object
        JuliaCall's dynamically generated module.
    record : SetupRecord
        The installation selected before startup.
    threads : int
        Expected positive default-pool width; the interactive pool is disabled.

    Raises
    ------
    BackendError
        If any effective path or thread-pool width differs from the requested value.
    """
    # JuliaCall's dynamically generated module is the confined untyped boundary.
    load_path = tuple(str(value) for value in namespace.seval("collect(LOAD_PATH)"))
    expected = (*(str(path) for _, path in record.projects), "@stdlib")
    depot_path = tuple(str(value) for value in namespace.seval("collect(DEPOT_PATH)"))
    active = Path(str(namespace.seval("Base.active_project()"))).resolve()
    if (
        load_path != expected
        or depot_path != (str(record.depot),)
        or active != (record.projects[0][1] / "Project.toml").resolve()
        or int(namespace.seval("Base.Threads.nthreads(:default)")) != threads
        or int(namespace.seval("Base.Threads.nthreads(:interactive)")) != 0
    ):
        raise BackendError(
            "the loaded julia environment or thread pools differ from setup"
        )


def _configure(record: SetupRecord, threads: int) -> None:
    """Fix the bridge and loader environment without initializing Julia.

    Parameters
    ----------
    record : SetupRecord
        Installation whose complete stack has been verified.
    threads : int
        Positive width within solprop's process thread budget.
    """
    # A foreign sysimage/library, interactive extension or JuliaPkg default could
    # bypass the verified source stack or download on import. Replace that ambient
    # configuration; these values remain fixed for subsequent extension loading.
    for name in tuple(os.environ):
        if name.startswith("PYTHON_JULIACALL_"):
            del os.environ[name]
    os.environ.update(
        PYTHON_JULIACALL_EXE=str(record.julia_executable),
        PYTHON_JULIACALL_PROJECT=str(record.projects[0][1]),
        PYTHON_JULIACALL_INIT="yes",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        PYTHON_JULIACALL_STARTUP_FILE="no",
        PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION="no",
        PYTHON_JULIACALL_THREADS=f"{threads},0",
        JULIA_NUM_THREADS=f"{threads},0",
        JULIA_CPU_TARGET=record.cpu_target,
        JULIA_DEPOT_PATH=str(record.depot),
        JULIA_LOAD_PATH=os.pathsep.join(
            [str(path) for _, path in record.projects] + ["@stdlib"]
        ),
        JULIA_PKG_OFFLINE="true",
        JULIA_CONDAPKG_BACKEND="Null",
        # Julia can launch a precompilation child that embeds Python and imports
        # PythonCall's depot-local pysrc. Keep that verified package tree immutable;
        # this does not change the initialized host's sys.dont_write_bytecode.
        PYTHONDONTWRITEBYTECODE="1",
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load the source checkout's private Julia runtime only on explicit selection."""

import functools
import os
import platform
import sys
from pathlib import Path
from typing import Any

from solprop.exceptions import BackendError


def _blas_provider() -> str:
    """Prefer the measured system BLAS on macOS; retain a portable CPU default."""
    default = "accelerate" if sys.platform == "darwin" else "openblas"
    return os.environ.get("SOLPROP_JULIA_BLAS", default)


def thread_limit(name: str, default: int = 1) -> int:
    """Read a positive thread-pool limit from an environment variable.

    Parameters
    ----------
    name : str
        Environment variable to read.
    default : int, optional
        Limit when the variable is absent.

    Returns
    -------
    int
        Positive thread count.

    Raises
    ------
    BackendError
        If the value is not a positive integer.
    """
    try:
        value = int(os.environ.get(name, str(default)))
        if value < 1:
            raise ValueError(name)
    except ValueError as exc:
        raise BackendError(f"{name} must be a positive integer") from exc
    return value


def call_julia(function: Any, *args: object, nogil: bool = False) -> Any:
    """Translate bridge failures at the dynamic JuliaCall boundary.

    Parameters
    ----------
    function : object
        JuliaCall function wrapper.
    *args : object
        Arguments accepted by the Julia function.
    nogil : bool, optional
        Release the GIL only for pure Julia numerical work.

    Returns
    -------
    object
        Converted result or opaque Julia handle.

    Raises
    ------
    BackendError
        If Julia or the bridge raises an exception; the cause is retained.
    """
    try:
        return function._jl_call_nogil(*args) if nogil else function(*args)
    except Exception as exc:
        raise BackendError(f"julia execution failed: {exc}") from exc


# JuliaCall exposes dynamically compiled Julia modules, so Any is confined to its
# boundary. Numerical arrays are converted to owned NumPy arrays by the adapter.
@functools.cache
def runtime() -> Any:
    """Load the experimental Julia module from this source checkout.

    Returns
    -------
    object
        JuliaCall's dynamically compiled module wrapper.

    Raises
    ------
    BackendError
        If the isolated runtime or Julia project has not been prepared.
    """
    root = Path(__file__).resolve().parents[4]
    if "juliacall" in sys.modules:
        raise BackendError(
            "julia requires a fresh Python process: JuliaCall was already initialized"
        )
    executable = root / ".julia-runtime/julia-1.13.2/bin/julia"
    project = root / "julia"
    if not executable.is_file() or not (project / "Manifest.toml").is_file():
        raise BackendError(
            "julia requires a source checkout with its private Julia 1.13.2 runtime; "
            "follow julia/README.md"
        )
    settings = {
        "PYTHON_JULIACALL_EXE": str(executable),
        "PYTHON_JULIACALL_PROJECT": str(project),
        "PYTHON_JULIACALL_STARTUP_FILE": "no",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "PYTHON_JULIACALL_THREADS": str(thread_limit("SOLPROP_JULIA_THREADS")),
        "JULIA_DEPOT_PATH": str(root / ".julia-depot"),
        "JULIA_LOAD_PATH": "@:@stdlib",
        "JULIA_CONDAPKG_BACKEND": "Null",
    }
    # The experiment deliberately owns one Julia runtime per Python process.
    for name, value in settings.items():
        if name in os.environ and os.environ[name] != value:
            raise BackendError(f"julia private runtime conflicts with {name}")
        os.environ[name] = value
    import juliacall  # noqa: PLC0415 - Julia must never initialize on package import

    module = juliacall.newmodule("SolpropBackend")
    provider = _blas_provider()
    if provider == "accelerate":
        if sys.platform != "darwin":
            raise BackendError("Julia Accelerate BLAS requires macOS")
        call_julia(module.seval, "using AppleAccelerate")
        library = str(call_julia(module.AppleAccelerate.BLAS.get_config))
        if "Accelerate" not in library:
            raise BackendError("Julia Accelerate BLAS forwarding is unavailable")
        call_julia(
            module.AppleAccelerate.set_num_threads,
            thread_limit("SOLPROP_BLAS_THREADS"),
        )
    elif provider == "mkl":
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise BackendError("Julia MKL is qualified only on Linux x86_64")
        call_julia(module.seval, "using MKL")
        if not call_julia(module.MKL.MKL_jll.is_available):
            raise BackendError("Julia MKL libraries are unavailable")
    elif provider != "openblas":
        raise BackendError("SOLPROP_JULIA_BLAS must be openblas, accelerate or mkl")
    call_julia(module.seval, "using SolpropJulia")
    return module.SolpropJulia

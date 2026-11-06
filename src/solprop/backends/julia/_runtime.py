# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load the source checkout's private Julia runtime only on explicit selection."""

import functools
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from solprop.exceptions import BackendError

VERSION = "1.13.3"
"""The Julia release the experiment's manifest is resolved with."""


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

    ``SOLPROP_JULIA_EXE`` names the Julia 1.13 executable (default: the one
    ``julia/bootstrap.py`` downloads) and ``SOLPROP_JULIA_DEPOT`` its package depot
    (default ``.julia-depot``). ``SOLPROP_JULIA_DEVICE=cuda`` loads CUDA.jl first, so
    that the engine's CUDA extension is available.

    Returns
    -------
    object
        JuliaCall's dynamically compiled module wrapper.

    Raises
    ------
    BackendError
        If the runtime or Julia project has not been prepared.
    """
    root = Path(__file__).resolve().parents[4]
    if "juliacall" in sys.modules:
        raise BackendError(
            "julia requires a fresh Python process: JuliaCall was already initialized"
        )
    executable = Path(
        os.environ.get(
            "SOLPROP_JULIA_EXE", str(root / f".julia-runtime/julia-{VERSION}/bin/julia")
        )
    )
    project = root / "julia"
    if not executable.is_file() or not (project / "Manifest.toml").is_file():
        raise BackendError(
            "julia requires a source checkout with a prepared Julia 1.13 runtime; "
            "follow julia/README.md"
        )
    settings = {
        "PYTHON_JULIACALL_EXE": str(executable),
        "PYTHON_JULIACALL_PROJECT": str(project),
        "PYTHON_JULIACALL_STARTUP_FILE": "no",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "PYTHON_JULIACALL_THREADS": str(thread_limit("SOLPROP_JULIA_THREADS")),
        "JULIA_DEPOT_PATH": os.environ.get(
            "SOLPROP_JULIA_DEPOT", str(root / ".julia-depot")
        ),
        "JULIA_CONDAPKG_BACKEND": "Null",
    }
    device = os.environ.get("SOLPROP_JULIA_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise BackendError("SOLPROP_JULIA_DEVICE must be cpu or cuda")
    # The CUDA packages live in a second environment stacked on the project's, so a
    # CPU run never resolves or loads them.
    settings["JULIA_LOAD_PATH"] = (
        f"@:{project / 'cuda'}:@stdlib" if device == "cuda" else "@:@stdlib"
    )
    # The experiment deliberately owns one Julia runtime per Python process.
    for name, value in settings.items():
        if name in os.environ and os.environ[name] != value:
            raise BackendError(f"julia runtime conflicts with {name}")
        os.environ[name] = value
    import juliacall  # noqa: PLC0415 - Julia must never initialize on package import

    module = juliacall.newmodule("SolpropBackend")
    if device == "cuda":
        call_julia(module.seval, "using CUDA")
    digest = source_digest()
    call_julia(module.seval, "using SolpropJulia")
    module.SolpropJulia.record_digest(digest)
    return module.SolpropJulia


def source_digest() -> str:
    """Return a SHA-256 over the Julia sources, the manifest and this adapter.

    The experiment's counterpart of a native build's source digest: it names exactly
    what a timing ran, and the engine refuses to run if its loaded module's digest
    differs from the files on disk.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    root = Path(__file__).resolve().parents[4]
    paths = sorted(
        [
            *(root / "julia/src").rglob("*.jl"),
            *(root / "julia/ext").rglob("*.jl"),
            root / "julia/Project.toml",
            root / "julia/Manifest.toml",
            *Path(__file__).parent.glob("*.py"),
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()

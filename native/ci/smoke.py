# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke-test an installed solprop-native distribution wheel.

Run it in isolated mode, from outside the source tree, so that it imports the
installed package and nothing from a checkout::

    python -I native/ci/smoke.py [--version VERSION]

It checks that the package imports and passes ``check_build``, that it is the
distribution build, that its version is the expected one, and that the libraries it
loaded (FFT, BLAS, OpenMP) are the copies inside the installed wheel, not ones from
the build machine or the system. It then runs ``python -m solprop_native`` and checks
that the build record it prints is JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import solprop_native


def bundled(library: str, package: Path) -> bool:
    """Return whether a loaded library lies in the installed package or its repair dir.

    auditwheel and delvewheel copy libraries to ``solprop_native.libs`` beside the
    package, delocate to ``solprop_native/.dylibs``, and the build copies its OpenBLAS
    into the package itself; a static library is reported as the extension.
    """
    path = Path(library).resolve()
    roots = (package, package.parent / "solprop_native.libs")
    return any(path.is_relative_to(root) for root in roots)


def main() -> None:
    """Run the checks; exit non-zero with a message on the first failure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="the version the wheel must carry")
    arguments = parser.parse_args()

    solprop_native.check_build()
    info = solprop_native.build_info()
    if (
        arguments.version is not None
        and solprop_native.__version__ != arguments.version
    ):
        sys.exit(f"version {solprop_native.__version__}, expected {arguments.version}")
    if info.build_profile != "distribution":
        sys.exit(f"build profile {info.build_profile!r}, expected 'distribution'")
    package = Path(solprop_native.__file__).resolve().parent
    loaded = {
        "fft_library": info.fft_library,
        "blas_library": info.blas_library,
        "openmp_library": info.openmp_library,
    }
    for name, library in loaded.items():
        # Accelerate is the operating system's; an empty record is a static library
        # whose symbols the extension does not export (FFTW on Windows).
        if (
            library
            and "Accelerate.framework" not in library
            and not bundled(library, package)
        ):
            sys.exit(f"{name} {library} is not the wheel's own copy (under {package})")
    if info.blas_backend == "OpenBLAS" and info.blas_threads is None:
        sys.exit("the bundled OpenBLAS reports no thread count")
    # Every library the wheel bundles travels with its licence.
    licences = {path.name for path in package.glob("LICENSE-*.txt")}
    required = {"LICENSE-fftw.txt"}
    if info.blas_backend == "OpenBLAS":
        required.add("LICENSE-openblas.txt")
    if info.openmp_enabled and not licences & {
        "LICENSE-libomp.txt",
        "LICENSE-libgomp.txt",
    }:
        required.add("LICENSE-libomp.txt or LICENSE-libgomp.txt")
    if missing := sorted(required - licences):
        sys.exit(f"the wheel lacks {', '.join(missing)}")

    record = subprocess.run(
        [sys.executable, "-I", "-m", "solprop_native"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    json.loads(record)
    summary = {
        "version": solprop_native.__version__,
        "cpu_baseline": info.cpu_baseline,
        "execution_space": info.execution_space,
        "blas": f"{info.blas_backend} ({info.blas_threading}, {info.blas_threads})",
        "blas_thread_policy": info.blas_thread_policy,
        "fft_backend": info.fft_backend,
        **loaded,
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

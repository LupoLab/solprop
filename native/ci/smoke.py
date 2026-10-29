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
the build machine or the system. On Arm it checks that both precisions of the bundled
FFTW carry the NEON codelets. It then runs ``python -m solprop_native`` and checks
that the build record it prints is JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
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


def fftw_versions(fft_library: str) -> dict[str, str]:
    """Return the version strings of the bundled FFTW's double and single libraries.

    FFTW names its codelet family in the version (``fftw-3.3.11-neon``). The single
    precision library lies beside the double one under the repair's naming.
    """
    single = sorted(
        {
            path.resolve()
            for path in Path(fft_library).parent.glob("libfftw3f[-.]*")
            if path.suffix not in {".a", ".la"}
        }
    )
    if len(single) != 1:
        sys.exit(f"expected one single-precision FFTW beside {fft_library}: {single}")
    versions = {}
    for symbol, library in (
        ("fftw_version", fft_library),
        ("fftwf_version", str(single[0])),
    ):
        handle = ctypes.CDLL(library)
        address = ctypes.addressof(ctypes.c_char.in_dll(handle, symbol))
        versions[symbol] = ctypes.string_at(address).decode()
    return versions


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
    # Compiling with -mcpu does not enable FFTW's NEON codelets; the recipe must.
    fftw = {}
    if platform.machine() in {"arm64", "aarch64"}:
        fftw = fftw_versions(info.fft_library)
        if scalar := [name for name, version in fftw.items() if "-neon" not in version]:
            sys.exit(f"the bundled FFTW lacks NEON codelets: {fftw} ({scalar})")
    if info.blas_backend == "OpenBLAS" and info.blas_threads is None:
        sys.exit("the bundled OpenBLAS reports no thread count")
    # Every library the wheel bundles travels with its licence.
    licences = {path.name for path in package.glob("LICENSE-*.txt")}
    required = {"LICENSE-fftw.txt"}
    if info.blas_backend == "OpenBLAS":
        required.add("LICENSE-openblas.txt")
    if info.vector_math.startswith("SLEEF"):
        required.add("LICENSE-sleef.txt")
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
        **fftw,
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

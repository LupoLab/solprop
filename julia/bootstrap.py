# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare the experiment's Julia runtime and instantiate its committed manifest.

``SOLPROP_JULIA_EXE`` names an existing Julia 1.13 (for example juliaup's); without
it the checksummed official release is downloaded into ``.julia-runtime/``. The
packages go to ``SOLPROP_JULIA_DEPOT``, by default ``.julia-depot/``.
"""

import hashlib
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.13.3"
# Digests published by Julia's official versions.json for these release archives.
RELEASES = {
    ("darwin", "arm64"): (
        "mac/aarch64",
        "macaarch64",
        "b9f0e8450fbcb392aa3e8b43fa1fc09993665a31bf705d7ef8084f0ac4ad3765",
    ),
    ("linux", "x86_64"): (
        "linux/x64",
        "linux-x86_64",
        "5c61cb4b40332cb5a3504e7779e2f5dac093caa4a1116ded19f01413f85c63e4",
    ),
    ("linux", "aarch64"): (
        "linux/aarch64",
        "linux-aarch64",
        "9ceb053bee6dfb8189bd40afd6925de3a9b7ff8fac8d01a5fcdd84af171ddf03",
    ),
}
ROOT = Path(__file__).resolve().parents[1]


def executable() -> Path:
    """Return the Julia executable, downloading the checksummed release if needed."""
    if named := os.environ.get("SOLPROP_JULIA_EXE"):
        return Path(named)
    release = RELEASES.get((sys.platform, platform.machine()))
    if release is None:
        raise RuntimeError("no Julia release is recorded for this platform")
    directory, suffix, expected = release
    private = ROOT / ".julia-runtime"
    binary = private / f"julia-{VERSION}/bin/julia"
    if binary.is_file():
        return binary
    private.mkdir(exist_ok=True)
    name = f"julia-{VERSION}-{suffix}.tar.gz"
    archive = private / name
    if not archive.is_file():
        url = f"https://julialang-s3.julialang.org/bin/{directory}/1.13/{name}"
        urllib.request.urlretrieve(url, archive)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError("Julia archive SHA256 does not match the release")
    with tarfile.open(archive) as bundle:
        bundle.extractall(private, filter="data")
    return binary


def main() -> None:
    """Instantiate and precompile the Julia project."""
    env = os.environ | {
        "JULIA_DEPOT_PATH": os.environ.get(
            "SOLPROP_JULIA_DEPOT", str(ROOT / ".julia-depot")
        ),
        "JULIA_LOAD_PATH": "@:@stdlib",
        "JULIA_CONDAPKG_BACKEND": "Null",
        "JULIA_PYTHONCALL_EXE": sys.executable,
    }
    # --cuda (--rocm) also prepares the GPU environment, which adds CUDA.jl (AMDGPU.jl)
    # to the engine's.
    projects = ["julia"] + [
        f"julia/{gpu}"
        for gpu in ("cuda", "rocm")
        if f"--{gpu}" in sys.argv[1:] or (gpu == "cuda" and "--cuda-local" in sys.argv)
    ]
    # --cuda-local: CUDA.jl takes the system toolkit rather than its own, needed where
    # native is built against a system CUDA in the same process, whose cuBLAS would
    # otherwise meet CUDA.jl's own (an undefined cublasLt symbol, CUDA 13.1 on an H200).
    if "--cuda-local" in sys.argv[1:]:
        (ROOT / "julia/cuda/LocalPreferences.toml").write_text(
            '[CUDA_Runtime_jll]\nlocal = "true"\n', encoding="utf-8"
        )
    for project in projects:
        subprocess.run(
            [
                str(executable()),
                "--startup-file=no",
                f"--project={ROOT / project}",
                "-e",
                "using Pkg; Pkg.instantiate(); Pkg.precompile()",
            ],
            env=env,
            check=True,
        )
    sys.stdout.write("Prepared Julia and julia/Manifest.toml\n")


if __name__ == "__main__":
    main()

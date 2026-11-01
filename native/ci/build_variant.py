# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the extension for a second CPU baseline, for merge_variant.py to add to a wheel.

cibuildwheel runs this as before-build, in the environment of the wheel's own build::

    python ci/build_variant.py PACKAGE --baseline B --output DIR --build-dir DIR

--cmake-arg adds to CMAKE_ARGS, after what is there, so that it wins. The build makes
its own OpenBLAS unless one is named here, so SOLPROP_NATIVE_OPENBLAS_ROOT, which may
name the tree this very build fills for the main build, is not passed on.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Build one wheel for the baseline into an emptied output directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--cmake-arg", action="append", default=[])
    arguments = parser.parse_args()

    shutil.rmtree(arguments.output, ignore_errors=True)
    environment = dict(os.environ)
    environment.pop("SOLPROP_NATIVE_OPENBLAS_ROOT", None)
    environment["CMAKE_ARGS"] = " ".join(
        [environment.get("CMAKE_ARGS", ""), *arguments.cmake_arg]
    ).strip()
    settings = {
        "cmake.define.SOLPROP_NATIVE_PROFILE": "DISTRIBUTION",
        "cmake.define.SOLPROP_NATIVE_CPU_BASELINE": arguments.baseline,
        "build-dir": arguments.build_dir,
    }
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps"]
    command += ["--wheel-dir", str(arguments.output)]
    for key, value in settings.items():
        command += ["--config-settings", f"{key}={value}"]
    subprocess.run([*command, str(arguments.package)], check=True, env=environment)


if __name__ == "__main__":
    main()

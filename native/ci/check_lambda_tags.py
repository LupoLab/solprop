#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fail on a device lambda that two CUDA object files instantiate (#815).

nvcc gives an extended ``__host__ __device__`` lambda a wrapper whose copy, call and
delete functions are static members of ``__nv_hdl_helper``, in an anonymous namespace:
each object file holds its own, set where that file constructs the lambda. The host code
that copies the wrapper (Kokkos's dispatch) is an inline template with external
linkage, so when two files instantiate the same lambda the link keeps one file's copy
of it, which reads that file's members. A run that constructed the lambda only in the
other file then calls a null function: the radial route's device GEMM, instantiated
alike in three files, crashed so in a Release build on an NVIDIA A40. A kernel in a
shared header whose views can match between files takes a named functor instead.

``python native/ci/check_lambda_tags.py OBJECT...`` lists each object file's helper
symbols with ``nm`` and exits 1, naming every lambda's tag that more than one file
holds, unless ``ALLOWED`` lists it with the reason it never runs. It also fails when it
finds no helper symbol at all, so that a change in nvcc's naming cannot make it pass by
finding nothing.

Standard library only; it needs GNU ``nm`` (binutils) and object files that a CUDA
build compiled, at any optimisation level: the members exist whether or not a link
happens to keep the wrong copy. It sees only the kernels the build's device target
instantiates: a build for compute capability 9.0 compiles no ``device_dense_gemm``, so
the lambda behind #815 shows only in a build for 8.6 (``AMPERE86``). The CUDA
device-compile lane builds both and runs this check over each.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

__all__ = ["ALLOWED", "HELPER", "helper_tags", "main", "shared_tags"]

HELPER = re.compile(r"__nv_hdl_helper<(.*)>::fp_copier$")
"""A demangled extended lambda's copy-function static; its group is the lambda's tag."""
ALLOWED: tuple[str, ...] = (
    # Kokkos-FFT scales a transform's output in a lambda unless the normalisation is
    # none; the backend passes `KokkosFFT::Normalization::none` to every transform and
    # scales explicitly, so the lambda is instantiated but never constructed.
    r"KokkosFFT::Impl::normalize_impl<",
)
"""Shared tags allowed, each a pattern searched for in the tag."""


def helper_tags(path: Path) -> set[str]:
    """Return the tags of the extended lambdas whose helper members `path` holds."""
    listing = subprocess.run(
        ["nm", "--demangle", "--format=posix", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tags: set[str] = set()
    for line in listing.splitlines():
        # POSIX format: name, type, value, size; the demangled name holds spaces.
        name = line.rsplit(" ", 3)[0] if line.count(" ") >= 3 else line
        if match := HELPER.search(name):
            tags.add(match[1])
    return tags


def shared_tags(objects: list[Path]) -> tuple[int, dict[str, list[Path]]]:
    """Return the helper count, and each tag several of `objects` hold with them."""
    holders: dict[str, list[Path]] = {}
    count = 0
    for path in objects:
        for tag in helper_tags(path):
            holders.setdefault(tag, []).append(path)
            count += 1
    return count, {tag: paths for tag, paths in holders.items() if len(paths) > 1}


def main(argv: list[str] | None = None) -> int:
    """Check the object files named; return 1 on a lambda two of them hold."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("check_lambda_tags: name the object files", file=sys.stderr)
        return 2
    count, shared = shared_tags([Path(arg) for arg in args])
    if not count:
        print("check_lambda_tags: no extended lambda helper found", file=sys.stderr)
        return 1
    failed = False
    for tag, paths in sorted(shared.items()):
        files = ", ".join(sorted(path.name for path in paths))
        if any(re.search(pattern, tag) for pattern in ALLOWED):
            print(f"allowed in several files ({files}): {tag}")
        else:
            print(f"lambda in several files ({files}): {tag}", file=sys.stderr)
            failed = True
    if not failed:
        print(f"{count} extended lambda helpers in {len(args)} files, none shared")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

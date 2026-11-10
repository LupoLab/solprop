#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fail on a fused multiply-add in a strict file's code in a built wheel (solprop #135).

The Adams march (``march::``), the boundary watch (``watch::``), the Runge-Kutta
march (``rk::``), the native execution session (``session::``), its statistics
plan's finishing (``statistics::``) and its dense-output weights (``dense::``) are
compiled without contraction, so that their
decisions round as solprop's Python does (ADR-060, SEP 45, SEP 47).
``python native/ci/check_strict_fma.py WHEEL`` unpacks the wheel, disassembles each
extension in it with ``objdump`` and exits 1, naming every function of those
namespaces that holds a fused multiply-add instruction.
It also fails when it finds none of the namespaces' functions, so that a change in
symbol names or visibility cannot make it pass by finding nothing.

Standard library only; it needs GNU ``objdump`` and reads x86-64 and AArch64 code.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

__all__ = ["STRICT", "fused_functions", "main"]

STRICT = re.compile(
    r"^[0-9a-f]+ <solprop_native::(march|watch|rk|session|statistics|dense)::"
)
"""A function label in ``objdump -C`` output from one of the strict files."""

LABEL = re.compile(r"^[0-9a-f]+ <.*>:$")
FUSED = re.compile(
    # x86-64: the FMA3 and AVX-512 forms (vfmadd231sd, vfnmsub132pd, vfmaddsub213pd).
    r"\tv?fn?m(?:add|sub)(?:sub|add)?\d{3}[ps][sdh]\b"
    # AArch64: the scalar forms and the vector multiply-accumulates.
    r"|\tfn?m(?:add|sub)\b|\tfml[as]\b"
)


def fused_functions(library: Path) -> tuple[int, dict[str, int]]:
    """Return the strict functions' count and, for each that fuses, its FMA count."""
    listing = subprocess.run(
        ["objdump", "-d", "-C", "--no-show-raw-insn", str(library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    strict = 0
    fused: dict[str, int] = {}
    name: str | None = None
    for line in listing.splitlines():
        if LABEL.match(line):
            name = line if STRICT.match(line) else None
            strict += name is not None
        elif name is not None and FUSED.search(line):
            fused[name] = fused.get(name, 0) + 1
    return strict, fused


def main(argv: list[str] | None = None) -> int:
    """Check every extension in the wheel named; return 1 on a fused strict function."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("check_strict_fma: name one wheel", file=sys.stderr)
        return 2
    failed = False
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(args[0]) as wheel:
            wheel.extractall(temporary)
        libraries = sorted(Path(temporary).rglob("_core*.so"))
        if not libraries:
            print(
                "check_strict_fma: the wheel holds no _core extension", file=sys.stderr
            )
            return 1
        for library in libraries:
            strict, fused = fused_functions(library)
            where = library.relative_to(temporary)
            if not strict:
                print(f"{where}: no strict function found", file=sys.stderr)
                failed = True
            for name, count in sorted(fused.items()):
                print(
                    f"{where}: {count} fused multiply-add(s) in {name}", file=sys.stderr
                )
                failed = True
            if strict and not fused:
                print(f"{where}: {strict} strict functions, none fused")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

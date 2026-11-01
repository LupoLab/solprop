# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Add a second build of the extension to a wheel, before the wheel is repaired.

The variant's extension goes in beside the wheel's as ``_core_v2.<abi tag>.<suffix>``.
``_cpu_baseline.txt`` becomes the variant's baseline, the wheel's floor, and
``_cpu_variants.txt`` lists both builds, best first (solprop_native/_cpu.py). The wheel
is rewritten in place, with its RECORD::

    python ci/merge_variant.py WHEEL VARIANT_DIR --best x86-64-v3 --variant x86-64-v2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

EXTENSION = re.compile(r"^solprop_native/_core\.[^/]+\.(so|pyd)$")


def record_line(name: str, data: bytes) -> str:
    """Return a wheel RECORD line for a file."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def extension_of(wheel: zipfile.ZipFile) -> str:
    """Return the name of the one extension in a wheel."""
    names = [name for name in wheel.namelist() if EXTENSION.match(name)]
    if len(names) != 1:
        raise SystemExit(f"expected one extension in {wheel.filename}: {names}")
    return names[0]


def main() -> None:
    """Rewrite the wheel with both builds and their records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--best", required=True)
    parser.add_argument("--variant", required=True)
    arguments = parser.parse_args()

    variants = sorted(arguments.variant_dir.glob("*.whl"))
    if len(variants) != 1:
        raise SystemExit(f"expected one wheel in {arguments.variant_dir}: {variants}")
    with zipfile.ZipFile(variants[0]) as source:
        data = source.read(extension_of(source))

    additions: dict[str, bytes] = {}
    with zipfile.ZipFile(arguments.wheel) as wheel:
        name = extension_of(wheel)
        additions[name.replace("/_core.", "/_core_v2.", 1)] = data
        additions["solprop_native/_cpu_baseline.txt"] = f"{arguments.variant}\n".encode()
        additions["solprop_native/_cpu_variants.txt"] = (
            f"{arguments.best} _core\n{arguments.variant} _core_v2\n".encode()
        )
        record = next(n for n in wheel.namelist() if n.endswith(".dist-info/RECORD"))
        kept = [
            line
            for line in wheel.read(record).decode().splitlines()
            if line.split(",", 1)[0] not in additions
        ]
        kept += [record_line(path, content) for path, content in additions.items()]
        with tempfile.TemporaryDirectory() as directory:
            merged = Path(directory) / arguments.wheel.name
            with zipfile.ZipFile(merged, "w", zipfile.ZIP_DEFLATED) as out:
                for item in wheel.infolist():
                    if item.filename not in additions and item.filename != record:
                        out.writestr(item, wheel.read(item.filename))
                for path, content in additions.items():
                    out.writestr(path, content)
                out.writestr(record, "\n".join(kept) + "\n")
            shutil.move(merged, arguments.wheel)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record Luna's metal interpolants and raw tables without changing its checkout."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
WAVELENGTHS = [
    0.01,
    0.017,
    0.018,
    0.12,
    0.126,
    0.127,
    0.12782,
    0.13,
    0.14,
    0.149,
    0.15,
    0.151,
    0.2,
    0.205,
    0.206,
    0.2066,
    0.21,
    0.3,
    0.4,
    0.5,
    0.5635,
    0.6,
    0.8,
    1.0,
    1.5,
    1.69,
    1.7,
    1.71,
    1.9,
    1.99,
    2.0,
    2.01,
    5.0,
    10.0,
    12.39,
    12.4,
    12.41,
    20.0,
    100.0,
    190.0,
]
PROGRAM = r"""
using Luna
const LUNA_DIR = get(ENV, "LUNA_DIR", joinpath(homedir(), ".julia/dev/Luna"))
@assert realpath(pathof(Luna)) == realpath(joinpath(LUNA_DIR, "src", "Luna.jl"))
println("VERSION\t",VERSION)
wavelengths = parse.(Float64, split(read(stdin,String)))
for name in (:Ag,:Al)
    table=Luna.PhysData.data_metal(name)
    for row in eachrow(table)
        println("RAW\t",name,"\t",join(row,'\t'))
    end
    lookup=Luna.PhysData.lookup_metal(name)
    for wl in wavelengths
        value=lookup(wl)
        println("INDEX\t",name,"\t",wl,"\t",real(value),"\t",imag(value))
    end
end
"""


def main():
    """Generate forty-wavelength witnesses and retain source table rows for audit."""
    child = subprocess.run(
        ["julia", "--startup-file=no", "-e", PROGRAM],
        input=" ".join(map(str, WAVELENGTHS)),
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stderr.write(child.stderr)
    if child.returncode:
        return child.returncode
    result = {name: {"raw": [], "index": []} for name in ("Ag", "Al")}
    version = ""
    for line in child.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "VERSION":
            version = parts[1]
        else:
            result[parts[1]][parts[0].lower()].append(list(map(float, parts[2:])))

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(LUNA), *args], text=True, capture_output=True, check=True
        ).stdout.strip()

    output = {
        "provenance": {
            "luna_path": str(LUNA),
            "luna_commit": git("rev-parse", "HEAD"),
            "luna_branch": git("branch", "--show-current"),
            "julia_version": version,
            "generator": Path(__file__).name,
            "generated": datetime.now(UTC).isoformat(),
        },
        "metals": result,
    }
    Path(__file__).with_name("luna_metals.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    sys.stdout.write("wrote forty wavelengths per metal and the Luna source tables\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

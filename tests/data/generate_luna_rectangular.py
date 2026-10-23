# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate rectangular-mode witnesses using the read-only Luna checkout."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
PROGRAM = r"""
using Luna
const LUNA_DIR = get(ENV, "LUNA_DIR", joinpath(homedir(), ".julia/dev/Luna"))
@assert realpath(pathof(Luna)) == realpath(joinpath(LUNA_DIR, "src", "Luna.jl"))
println("VERSION\t",VERSION)
for material in (:constant,:SiO2,:Ag,:Al), pressure in (0.,5.,10.), pol in (:x,:y),
    (n,m) in ((1,1),(3,2)), wl in (210e-9,400e-9,563.5e-9,800e-9,1030e-9)
    omega=2pi*Luna.PhysData.c/wl
    gas=Luna.PhysData.ref_index_fun(:Ar,pressure,Luna.PhysData.roomtemp)
    core=(omega;z)->gas(2pi*Luna.PhysData.c/omega)
    material_fun=material==:constant ? wl->0.05+3.5im :
        Luna.PhysData.ref_index_fun(material)
    wall=(omega;z)->material_fun(2pi*Luna.PhysData.c/omega)
    mode=Luna.RectModes.RectMode(z->50e-6,z->12.5e-6,n,m,pol,core,wall)
    value=Luna.Modes.neff(mode,omega)
    nc=core(omega,z=0); ns=wall(omega,z=0)
    println("ROW\t",join((material,pressure,pol,n,m,wl,real(nc),imag(nc),
            real(ns),imag(ns),real(value),imag(value)),'\t'))
end
"""


def main():
    """Record local material indices beside each independent Luna mode result."""
    child = subprocess.run(
        ["julia", "--startup-file=no", "-e", PROGRAM],
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stderr.write(child.stderr)
    if child.returncode:
        return child.returncode
    rows = []
    version = ""
    for line in child.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "VERSION":
            version = parts[1]
            continue
        rows.append(
            {
                "cladding": parts[1],
                "pressure": float(parts[2]),
                "polarisation": parts[3],
                "n": int(parts[4]),
                "m": int(parts[5]),
                "wavelength": float(parts[6]),
                "core": [float(parts[7]), float(parts[8])],
                "wall": [float(parts[9]), float(parts[10])],
                "neff": [float(parts[11]), float(parts[12])],
            }
        )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(LUNA), *args], text=True, capture_output=True, check=True
        ).stdout.strip()

    test = LUNA / "test/test_rect_modes.jl"
    data = {
        "provenance": {
            "luna_path": str(LUNA),
            "luna_commit": git("rev-parse", "HEAD"),
            "luna_branch": git("branch", "--show-current"),
            "julia_version": version,
            "generator": Path(__file__).name,
            "generated": datetime.now(UTC).isoformat(),
            "fem_source": str(test),
            "fem_source_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
            "fem_note": "Luna's test file, checked with FEM using JCMwave",
        },
        "cases": rows,
    }
    Path(__file__).with_name("luna_rectangular.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    sys.stdout.write(f"wrote {len(rows)} rectangular cases\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

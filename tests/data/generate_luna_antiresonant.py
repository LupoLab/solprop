# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate Vincetti witnesses through a read-only Julia child outside Luna.

Run with ``uv run --no-sync python tests/data/generate_luna_antiresonant.py``. The child
uses ``--startup-file=no`` and absolute Luna source paths. No package resolution or
checkout changes occur. Loss is sampled through Julia's alpha accessor and
``Modes.dB_per_m``, never ``imag(neff)``, which carries twice the loss in Luna's
Vincetti mode. Default-silica reference cases construct a fresh lookup per call, because
Luna's cached spline lookup returns a stale bracket after construction; an explicitly
labelled stale block preserves that first-call defect. Stored results are witnesses, not
physical truth.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_antiresonant.json")
GEOMETRIES = [
    (15e-6, 180e-9, 9e-6, 6),
    (5e-6, 80e-9, 1e-6, 10),
    ((10e-6 + 2.5e-6) / math.sin(math.pi / 8) - 10e-6, 1e-6, 10e-6, 8),
]
PROGRAM = r"""
const LUNA_DIR = get(ENV, "LUNA_DIR", joinpath(homedir(), ".julia/dev/Luna"))
pushfirst!(LOAD_PATH, LUNA_DIR)
using Luna
import Luna.PhysData: wlfreq
println("VERSION\t", VERSION)
for line in eachline(stdin)
    fields = split(line, '\t')
    key, clad = fields[1:2]
    a, t, r, pressure = parse.(Float64, fields[3:6])
    tubes = parse(Int, fields[7])
    wavelengths = parse.(Float64, split(fields[8], ','))
    cladn = clad == "silica" ? ((w; z=0) -> PhysData.ref_index(:SiO2, wlfreq(w))) :
                              ((w; z=0) -> parse(Float64, clad))
    for (i, wavelength) in enumerate(wavelengths)
        mode = Antiresonant.VincettiMode(a, :ArB, pressure;
            wallthickness=t, # codespell:ignore wallthickness
            tube_radius=r, Ntubes=tubes, cladn=cladn, T=294.)
        w = wlfreq(wavelength)
        tube = real(cladn(w))
        b2 = i <= 10 ? Modes.dispersion(mode, 2, w) : 0.0
        row = [wavelength, real(Antiresonant.neff_real(mode, w))-1,
               Modes.ALPHA(mode, w), Modes.dB_per_m(mode, w),
               Antiresonant.Aeff(mode, w), Antiresonant.Rco_eff(mode, w),
               Antiresonant.Δneff(mode, w), Antiresonant.CL(mode, w), tube,
               real(mode.m.coren(w; z=0)), b2,
               Antiresonant.getδ(a, r, tubes),
               Antiresonant.getRco(r, tubes, Antiresonant.getδ(a, r, tubes)),
               Antiresonant.getr_ext(a, tubes, Antiresonant.getδ(a, r, tubes))]
        println(key, "\t", join(string.(real.(row)), ','))
    end
end
for wavelength in [450e-9, 300e-9]
    stale = Antiresonant.VincettiMode(15e-6, :ArB, 1.;
            wallthickness=180e-9, # codespell:ignore wallthickness
        tube_radius=12e-6, Ntubes=6, T=294.)
    w = wlfreq(wavelength)
    # Capture the first spectral call before any other observer can move the finder.
    stale_index = Antiresonant.neff_real(stale, w)
    stale_tube = real(stale.cladn(w; z=0))
    correct = Antiresonant.VincettiMode(15e-6, :ArB, 1.;
            wallthickness=180e-9, # codespell:ignore wallthickness
        tube_radius=12e-6, Ntubes=6, T=294.,
        cladn=(w; z=0) -> PhysData.ref_index(:SiO2, wlfreq(w)))
    println("STALE\t", join(string.(real.([wavelength, stale_index-1,
        Antiresonant.neff_real(correct,w)-1,
        Modes.ALPHA(stale,w), Modes.ALPHA(correct,w),
        stale_tube, real(correct.cladn(w; z=0))])), ','))
end
"""
COLUMNS = [
    "wavelength",
    "refractivity",
    "alpha",
    "db_per_m",
    "effective_area",
    "effective_radius",
    "correction",
    "confinement_loss",
    "tube_index",
    "gas_index",
    "beta2",
    "gap",
    "core_radius_roundtrip",
    "tube_radius_roundtrip",
]


def main():
    """Write tabulated samples with the Julia and source identities."""
    cases = {}
    lines = []
    wavelengths = np.geomspace(250e-9, 2e-6, 60)
    # Spread the ten finite-difference samples over the complete band.
    order = list(np.linspace(0, 59, 10, dtype=int))
    order += [index for index in range(60) if index not in order]
    wavelength_string = ",".join(map(repr, wavelengths[order].tolist()))
    for index, geometry in enumerate(GEOMETRIES):
        for pressure in (0.0, 1.0, 30.0):
            for cladding in ("silica", "1.44" if index == 2 else "1.45"):
                key = f"g{index}-p{pressure:g}-{cladding}"
                radius, wall, outer, tubes = geometry
                cases[key] = {
                    "geometry": geometry,
                    "pressure_bar": pressure,
                    "cladding": cladding,
                    "samples": [],
                }
                lines.append(
                    f"{key}\t{cladding}\t{radius}\t{wall}\t{outer}\t{pressure}\t{tubes}\t{wavelength_string}"
                )
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", PROGRAM.replace("ALPHA", chr(945))],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=OUTPUT.parent.resolve(),
    )
    sys.stderr.write(result.stderr)
    if result.returncode:
        return result.returncode
    stale = []
    version = None
    for line in result.stdout.splitlines():
        key, data = line.split("\t")
        if key == "VERSION":
            version = data
        elif key == "STALE":
            stale.append(list(map(float, data.split(","))))
        else:
            cases[key]["samples"].append(list(map(float, data.split(","))))

    def git(*args):
        return subprocess.run(
            ["git", "--git-dir", str(LUNA / ".git"), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    provenance = {
        "generated": datetime.now(UTC).isoformat(),
        "julia_version": version,
        "luna_path": str(LUNA),
        "luna_commit": git("rev-parse", "HEAD"),
        "luna_branch": git("branch", "--show-current"),
        "temperature_k": 294.0,
        "gas": "ArB",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "luna_source_sha256": hashlib.sha256(
            (LUNA / "src/Antiresonant.jl").read_bytes()
        ).hexdigest(),
        "loss_note": "Modes.alpha and dB_per_m; no imaginary-neff witness.",
    }
    OUTPUT.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "columns": COLUMNS,
                "cases": cases,
                "stale_columns": [
                    "wavelength",
                    "stale_refractivity",
                    "correct_refractivity",
                    "stale_alpha",
                    "correct_alpha",
                    "stale_tube",
                    "correct_tube",
                ],
                "stale": stale,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

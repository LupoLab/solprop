# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate Luna's design-tool and offline-optics numbers through a read-only child.

Run with ``uv run --no-sync python tests/data/generate_luna_tools_optics.py``. The Julia
child uses ``--startup-file=no`` and the absolute Luna source path; nothing in the
checkout is resolved, written or changed. Stored results are regression witnesses, not
physical truth: they say what Luna computes, including where Luna is wrong.

What is stored: ``Tools.params``/``capillary_params``/``λRDW``/``pressureRDW`` on Luna's
own test cases; the glass indices behind ``prop_material``; the transfer function of
``prop_material`` for a millimetre of fused silica; and the single-reflection transfer
function of every ``lookup_mirror`` type inside its flat band. Luna's transfer functions
act on FFT-convention spectra, so they are the complex conjugates of solprop's.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_tools_optics.json")

MIRRORS = {
    "PC70": (450e-9, 1200e-9),
    "HD59": (1000e-9, 1060e-9),
    "PC147": (650e-9, 1350e-9),
    "PC1611": (855e-9, 1195e-9),
    "PC1821": (805e-9, 1345e-9),
    "HD120": (900e-9, 1200e-9),
    "ThorlabsUMC": (650e-9, 1050e-9),
}
GLASSES = ("SiO2", "BK7", "CaF2", "BaF2", "MgF2")
GLASS_WAVELENGTHS = (400e-9, 800e-9, 1500e-9)
N_SAMPLES = 41

PROGRAM = r"""
const LUNA_DIR = get(ENV, "LUNA_DIR", joinpath(homedir(), ".julia/dev/Luna"))
pushfirst!(LOAD_PATH, LUNA_DIR)
using Luna
import Luna.PhysData: wlfreq
println("VERSION\t", VERSION)

m = Capillary.MarcatiliMode(125e-6, :HeB, 0.4, model=:reduced)
p = Tools.params(300e-6, 10e-15, 800e-9, m, :HeB, P=0.4)
println("params\t", join(string.([p.N, p.Lfiss, p.zdw, p.P0/p.Pcr, p.β2, p.γ, p.n2,
                                   p.Aeff, p.Lloss, p.I0, p.P0]), ','))
q = Tools.capillary_params(6e-9, 20e-15, 800e-9, 14e-6, :Kr, P=15.0)
println("capillary\t", join(string.([q.N, q.Lfiss, q.zdw, q.β2, q.γ]), ','))
rdw = Tools.λRDW(125e-6, :He, 1.0, 800e-9)
println("rdw\t", rdw, ",", Tools.pressureRDW(125e-6, :He, rdw, 800e-9))
println("pzdw\t", Tools.pressureZDW(125e-6, :Ar, 500e-9))

for line in eachline(stdin)
    fields = split(line, '\t')
    if fields[1] == "glass"
        glass = Symbol(fields[2])
        values = [real(PhysData.ref_index(glass, parse(Float64, w)))
                  for w in split(fields[3], ',')]
        println("glass\t", fields[2], "\t", join(string.(values), ','))
    elseif fields[1] == "material"
        wavelengths = parse.(Float64, split(fields[2], ','))
        ω = wlfreq.(wavelengths)
        Eω = ones(ComplexF64, length(ω))
        Fields.prop_material!(Eω, ω, :SiO2, 1e-3, 800e-9)
        println("material\t", join(string.(real.(Eω)), ','), "\t",
                join(string.(imag.(Eω)), ','))
    elseif fields[1] == "mirror"
        t = PhysData.lookup_mirror(Symbol(fields[2]))
        values = t.(parse.(Float64, split(fields[3], ',')))
        println("mirror\t", fields[2], "\t", join(string.(real.(values)), ','), "\t",
                join(string.(imag.(values)), ','))
    end
end
"""  # noqa: RUF001 - Luna's field names are Greek letters


def samples(limits: tuple[float, float]) -> list[float]:
    """Return evenly spaced wavelengths strictly inside a band."""
    low, high = limits
    step = (high - low) / (N_SAMPLES + 1)
    return [low + step * (index + 1) for index in range(N_SAMPLES)]


def main() -> int:
    """Run the child and store what it prints."""
    requests = [
        "\t".join(("glass", glass, ",".join(repr(w) for w in GLASS_WAVELENGTHS)))
        for glass in GLASSES
    ]
    material_wavelengths = samples((500e-9, 1100e-9))
    requests.append(
        "\t".join(("material", ",".join(repr(w) for w in material_wavelengths)))
    )
    mirror_wavelengths = {name: samples(band) for name, band in MIRRORS.items()}
    requests.extend(
        "\t".join(("mirror", name, ",".join(repr(w) for w in wavelengths)))
        for name, wavelengths in mirror_wavelengths.items()
    )
    child = subprocess.run(
        ["julia", "--startup-file=no", "-e", PROGRAM],
        input="\n".join(requests) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    record: dict[str, object] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "luna_commit": subprocess.run(
            ["git", "-C", str(LUNA), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "program_sha256": hashlib.sha256(PROGRAM.encode()).hexdigest(),
        "glass_wavelengths": list(GLASS_WAVELENGTHS),
        "material": {"medium": "SiO2", "thickness": 1e-3, "reference": 800e-9},
        "glasses": {},
        "mirrors": {},
    }
    glasses: dict[str, list[float]] = {}
    mirrors: dict[str, dict[str, list[float]]] = {}
    for line in child.stdout.splitlines():
        fields = line.split("\t")
        key = fields[0]
        if key == "VERSION":
            record["julia"] = fields[1]
        elif key in ("params", "capillary", "rdw"):
            record[key] = [float(value) for value in fields[1].split(",")]
        elif key == "pzdw":
            record[key] = float(fields[1])
        elif key == "glass":
            glasses[fields[1]] = [float(value) for value in fields[2].split(",")]
        elif key == "material":
            record["material"] = {
                "medium": "SiO2",
                "thickness": 1e-3,
                "reference": 800e-9,
                "wavelength": material_wavelengths,
                "real": [float(value) for value in fields[1].split(",")],
                "imag": [float(value) for value in fields[2].split(",")],
            }
        elif key == "mirror":
            mirrors[fields[1]] = {
                "wavelength": mirror_wavelengths[fields[1]],
                "real": [float(value) for value in fields[2].split(",")],
                "imag": [float(value) for value in fields[3].split(",")],
            }
    record["glasses"] = glasses
    record["mirrors"] = mirrors
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

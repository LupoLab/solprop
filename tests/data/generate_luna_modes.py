# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna values that the capillary-mode tests compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_modes.py

It writes ``tests/data/luna_modes.json`` with a provenance block naming the Luna
commit, branch and Julia version. The test suite only reads that file: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout.

The stored numbers are regression witnesses, not truth. Two differences are expected
and are asserted explicitly in ``tests/modes/test_luna_modes.py`` rather than tolerated:

* the **cladding data**. Luna's default silica index is an unreferenced 400-row table
  with an unphysical seam in ``k``; solprop uses Malitson plus Franta's tabulated
  absorption. Every case is therefore generated twice, once with a **constant** cladding
  index -- where the two implementations must agree to rounding, because nothing differs
  -- and once with each side's own default, where the difference is the data and is
  measured.
* the **helium index**. Luna's default helium is the Ermolov fit divided by the wrong
  reference density (1 bar instead of 1 atm); its capillary tests use ``:HeB``, the
  Boerzsoenyi fit, which the solprop tests supply explicitly as a susceptibility
  (``tests/modes/witness_materials.py``). The cases name the source explicitly.

The quadrature block records Gauss and Gauss-Kronrod nodes and weights straight from
``QuadGK``, so that solprop's own Kronrod extension -- computed by Laurie's algorithm
rather than shipped as a table -- is checked against an independent implementation.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_modes.json")

TEMPERATURE = 294.0

# solprop (gas, source) -> the Luna symbol implementing the same model.
LUNA_SYMBOL = {
    ("He", "borzsonyi"): "HeB",
    ("He", "ermolov"): "He",
    ("Ne", "borzsonyi"): "Ne",
    ("Ar", "borzsonyi"): "ArB",
    ("Kr", "borzsonyi"): "Kr",
}

# Cladding specifications: a Julia expression for a constant index, or "SiO2" for each
# side's own default silica model.
CONSTANT_CLADDING = "1.45"
DEFAULT_CLADDING = "SiO2"

WAVELENGTHS = [
    250e-9,
    400e-9,
    532e-9,
    800e-9,
    1030e-9,
    1500e-9,
    2000e-9,
]


def index_case(  # noqa: PLR0913 - one row of a case table, kept flat on purpose
    key,
    gas,
    pressure,
    radius,
    *,
    source="borzsonyi",
    kind="HE",
    n=1,
    m=1,
    model="full",
    loss=True,
    cladding=CONSTANT_CLADDING,
):
    """Describe one stored effective-index case.

    Parameters
    ----------
    key : str
        Name the case is stored under.
    gas : str
        Fill species.
    pressure : float
        Absolute pressure in bar.
    radius : float
        Core radius in m.
    source : str, optional
        Which published index model to use.
    kind : str, optional
        ``"HE"``, ``"TE"`` or ``"TM"``.
    n, m : int, optional
        Azimuthal and radial indices.
    model : str, optional
        ``"full"`` or ``"reduced"``.
    loss : bool, optional
        Whether the mode carries its loss.
    cladding : str, optional
        A constant index as a Julia literal, or ``"SiO2"`` for each side's default.

    Returns
    -------
    tuple
        ``(key, spec)``.
    """
    return key, {
        "gas": gas,
        "source": source,
        "pressure_bar": pressure,
        "radius_m": radius,
        "kind": kind,
        "n": n,
        "m": m,
        "model": model,
        "loss": loss,
        "cladding": cladding,
    }


INDEX_CASES = dict(
    [
        index_case("he-full-loss-const", "He", 1.0, 125e-6),
        index_case("he-full-noloss-const", "He", 1.0, 125e-6, loss=False),
        index_case("he-reduced-loss-const", "He", 1.0, 125e-6, model="reduced"),
        index_case(
            "he-reduced-noloss-const", "He", 1.0, 125e-6, model="reduced", loss=False
        ),
        index_case("ar-full-loss-const", "Ar", 2.0, 50e-6),
        index_case("ne-full-loss-const", "Ne", 5.0, 100e-6),
        index_case("kr-reduced-loss-const", "Kr", 0.5, 200e-6, model="reduced"),
        index_case("he-he12-const", "He", 1.0, 125e-6, m=2),
        index_case("he-he31-const", "He", 1.0, 125e-6, n=3),
        index_case("he-te01-const", "He", 1.0, 125e-6, kind="TE", n=0),
        index_case("he-tm01-const", "He", 1.0, 125e-6, kind="TM", n=0),
        index_case("he-full-loss-silica", "He", 1.0, 125e-6, cladding=DEFAULT_CLADDING),
        index_case(
            "he-reduced-loss-silica",
            "He",
            1.0,
            125e-6,
            model="reduced",
            cladding=DEFAULT_CLADDING,
        ),
        index_case("ar-full-loss-silica", "Ar", 2.0, 50e-6, cladding=DEFAULT_CLADDING),
    ]
)

ZDW_CASES = {
    # key: gas, source, pressure in bar, radius in m, model, cladding
    "zdw-he-0.4bar-125um": (
        "He",
        "borzsonyi",
        0.4,
        125e-6,
        "reduced",
        CONSTANT_CLADDING,
    ),
    "zdw-he-5.9bar-75um": ("He", "borzsonyi", 5.9, 75e-6, "reduced", CONSTANT_CLADDING),
    "zdw-ne-1bar-100um": ("Ne", "borzsonyi", 1.0, 100e-6, "reduced", CONSTANT_CLADDING),
    "zdw-ar-2bar-50um": ("Ar", "borzsonyi", 2.0, 50e-6, "reduced", CONSTANT_CLADDING),
    "zdw-ar-0.5bar-125um": ("Ar", "borzsonyi", 0.5, 125e-6, "full", CONSTANT_CLADDING),
    "zdw-he-0.4bar-125um-silica": (
        "He",
        "borzsonyi",
        0.4,
        125e-6,
        "reduced",
        DEFAULT_CLADDING,
    ),
}

AREA_CASES = {
    # key: radius in m, kind, n, m
    "aeff-he11-75um": (75e-6, "HE", 1, 1),
    "aeff-he11-125um": (125e-6, "HE", 1, 1),
    "aeff-he12-125um": (125e-6, "HE", 1, 2),
    "aeff-he31-125um": (125e-6, "HE", 3, 1),
    "aeff-te01-125um": (125e-6, "TE", 0, 1),
    "aeff-tm01-125um": (125e-6, "TM", 0, 1),
}

QUADRATURE_CASES = {"gauss-8": 8, "gauss-16": 16, "gauss-24": 24}

KRONROD_CASES = {"kronrod-7": 7, "kronrod-10": 10, "kronrod-16": 16}

RULE_CASES = {
    # key: n_r, n_theta, kronrod, full, radius in m
    "radial-24": (24, 16, False, False, 100e-6),
    "radial-kronrod-24": (24, 16, True, False, 100e-6),
    "polar-16x8": (16, 8, False, True, 100e-6),
    "polar-kronrod-16x8": (16, 8, True, True, 100e-6),
}

JULIA_PROGRAM = r"""
# A plain text protocol rather than JSON.jl: this must run against whatever Julia
# environment the machine has, without installing anything into it.
#   in :  I<TAB>key<TAB>symbol<TAB>P<TAB>a<TAB>kind<TAB>n<TAB>m<TAB>model
#           <TAB>loss<TAB>clad<TAB>lam1,...
#         Z<TAB>key<TAB>symbol<TAB>P<TAB>a<TAB>model<TAB>clad
#         A<TAB>key<TAB>a<TAB>kind<TAB>n<TAB>m
#         G<TAB>key<TAB>n
#         K<TAB>key<TAB>n
#   out:  NE|ZD|AE|GQ|KR<TAB>key<TAB>comma separated values, or "failed"
using Luna
import Luna.PhysData: wlfreq
# QuadGK is reached through Luna's own module rather than being loaded separately:
# it is a dependency of Luna, not of the default environment, and nothing here may
# resolve or write into the read-only checkout.
const QuadGK = Luna.Modes.QuadGK

# `clad` is either "SiO2", meaning Luna's own default silica model, or a literal
# constant refractive index.
function makemode(sym, P, a, kind, n, m, model, loss, clad)
    if clad == "SiO2"
        return Capillary.MarcatiliMode(a, sym, P; n=n, m=m, kind=kind, T=TEMP,
                                       model=model, loss=loss)
    else
        v = parse(ComplexF64, clad)
        cladn = (w; z=0.0) -> v
        return Capillary.MarcatiliMode(a, sym, P, cladn; n=n, m=m, kind=kind, T=TEMP,
                                       model=model, loss=loss)
    end
end

const TEMP = TEMPERATURE_PLACEHOLDER

for line in eachline(stdin)
    isempty(strip(line)) && continue
    f = split(line, '\t')
    kind, key = f[1], f[2]
    if kind == "I"
        sym = Symbol(f[3])
        P = parse(Float64, f[4]); a = parse(Float64, f[5])
        mkind = Symbol(f[6]); n = parse(Int, f[7]); m = parse(Int, f[8])
        model = Symbol(f[9]); loss = f[10] == "true"
        clad = f[11]
        lams = [parse(Float64, x) for x in split(f[12], ',')]
        mode = makemode(sym, P, a, mkind, n, m, model, loss, clad)
        out = Float64[]
        for l in lams
            v = Capillary.neff(mode, wlfreq(l))
            push!(out, real(v)); push!(out, imag(v))
        end
        println("NE\t", key, "\t", join(string.(out), ","))
    elseif kind == "Z"
        sym = Symbol(f[3])
        P = parse(Float64, f[4]); a = parse(Float64, f[5])
        model = Symbol(f[6]); clad = f[7]
        mode = makemode(sym, P, a, :HE, 1, 1, model, true, clad)
        z = Capillary.zdw(mode)
        println("ZD\t", key, "\t", z === missing ? "failed" : string(z))
    elseif kind == "A"
        a = parse(Float64, f[3]); mkind = Symbol(f[4])
        n = parse(Int, f[5]); m = parse(Int, f[6])
        mode = Capillary.MarcatiliMode(a, :He, 0.0; n=n, m=m, kind=mkind)
        println("AE\t", key, "\t", string(Capillary.Aeff(mode)), ",",
                string(Modes.N(mode)))
    elseif kind == "G"
        n = parse(Int, f[3])
        x, w = QuadGK.gauss(n)
        println("GQ\t", key, "\t", join(string.(vcat(x, w)), ","))
    elseif kind == "K"
        n = parse(Int, f[3])
        xk, wk, gw = QuadGK.kronrod(n)
        println("KR\t", key, "\t", join(string.(vcat(xk, wk, gw)), ","))
    elseif kind == "R"
        nr = parse(Int, f[3]); nt = parse(Int, f[4])
        kron = f[5] == "true"; full = f[6] == "true"; a = parse(Float64, f[7])
        q = Modes.transverse_quadrature(:polar, full; nr=nr, nθ=nt, kronrod=kron)
        dl = (:polar, (0.0, 0.0), (a, 2π))
        xs = Modes.quadrature_nodes(q, dl)
        w = Modes.quadrature_weights(q, dl)
        wc = Modes.quadrature_weights(q, dl; coarse=true)
        flat = Float64[]
        for x in xs
            push!(flat, x[1]); push!(flat, x[2])
        end
        println("RU\t", key, "\t", join(string.(vcat(flat, w, wc)), ","))
    end
end
"""


def git(*args: str) -> str:
    """Run a read-only git command in the Luna checkout and return its output.

    Parameters
    ----------
    *args : str
        Arguments after ``git``.

    Returns
    -------
    str
        Standard output, stripped.
    """
    return subprocess.run(
        ["git", "-C", str(LUNA), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def encode() -> str:
    """Build the tab-separated request sent to Luna.

    Returns
    -------
    str
        The whole request, one command per line.
    """
    lines = []
    wavelengths = ",".join(repr(value) for value in WAVELENGTHS)
    for key, spec in INDEX_CASES.items():
        symbol = LUNA_SYMBOL[spec["gas"], spec["source"]]
        lines.append(
            f"I\t{key}\t{symbol}\t{spec['pressure_bar']!r}\t{spec['radius_m']!r}\t"
            f"{spec['kind']}\t{spec['n']}\t{spec['m']}\t{spec['model']}\t"
            f"{str(spec['loss']).lower()}\t{spec['cladding']}\t{wavelengths}"
        )
    for key, (gas, source, pressure, radius, model, clad) in ZDW_CASES.items():
        symbol = LUNA_SYMBOL[gas, source]
        lines.append(f"Z\t{key}\t{symbol}\t{pressure!r}\t{radius!r}\t{model}\t{clad}")
    for key, (radius, kind, n, m) in AREA_CASES.items():
        lines.append(f"A\t{key}\t{radius!r}\t{kind}\t{n}\t{m}")
    for key, count in QUADRATURE_CASES.items():
        lines.append(f"G\t{key}\t{count}")
    for key, count in KRONROD_CASES.items():
        lines.append(f"K\t{key}\t{count}")
    for key, (n_r, n_theta, kronrod, full, radius) in RULE_CASES.items():
        lines.append(
            f"R\t{key}\t{n_r}\t{n_theta}\t{str(kronrod).lower()}\t"
            f"{str(full).lower()}\t{radius!r}"
        )
    return "\n".join(lines) + "\n"


def decode(text: str) -> dict[str, dict[str, Any]]:
    """Parse Luna's replies into one mapping per section.

    Parameters
    ----------
    text : str
        Luna's standard output.

    Returns
    -------
    dict
        Section tag to ``{key: value}``, values being lists of floats or ``None``.

    Raises
    ------
    RuntimeError
        If Luna produced no output at all.
    """
    sections: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        tag, key, payload = line.split("\t")
        entry = sections.setdefault(tag, {})
        entry[key] = (
            None
            if payload == "failed"
            else [float(value) for value in payload.split(",")]
        )
    if not sections:
        raise RuntimeError("Luna produced no output")
    return sections


def main() -> int:
    """Run Luna and write the reference file.

    Returns
    -------
    int
        A process exit code.
    """
    program = JULIA_PROGRAM.replace("TEMPERATURE_PLACEHOLDER", repr(TEMPERATURE))
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", program],
        input=encode(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    sections = decode(result.stdout)
    payload = {
        "provenance": {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "luna_path": str(LUNA),
            "luna_commit": git("rev-parse", "HEAD"),
            "luna_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "luna_describe": git("log", "-1", "--format=%h %s"),
            "julia_version": subprocess.run(
                ["julia", "--startup-file=no", "--version"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "temperature_k": TEMPERATURE,
            "constant_cladding": CONSTANT_CLADDING,
            "note": (
                "Regression witnesses, not truth. Generated by "
                "tests/data/generate_luna_modes.py; the documented differences are "
                "named in tests/modes/test_luna_modes.py."
            ),
        },
        "wavelengths": WAVELENGTHS,
        "index_cases": INDEX_CASES,
        "zdw_cases": {
            key: {
                "gas": value[0],
                "source": value[1],
                "pressure_bar": value[2],
                "radius_m": value[3],
                "model": value[4],
                "cladding": value[5],
            }
            for key, value in ZDW_CASES.items()
        },
        "area_cases": {
            key: {"radius_m": value[0], "kind": value[1], "n": value[2], "m": value[3]}
            for key, value in AREA_CASES.items()
        },
        "rule_cases": {
            key: {
                "n_r": value[0],
                "n_theta": value[1],
                "kronrod": value[2],
                "full": value[3],
                "radius_m": value[4],
            }
            for key, value in RULE_CASES.items()
        },
        "neff": sections["NE"],
        "zdw": sections["ZD"],
        "area": sections["AE"],
        "gauss": sections["GQ"],
        "kronrod": sections["KR"],
        "rules": sections["RU"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    commit = payload["provenance"]["luna_commit"]
    sys.stdout.write(f"wrote {OUTPUT} from Luna {commit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

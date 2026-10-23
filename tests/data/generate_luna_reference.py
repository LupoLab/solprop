# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna values that the physdata tests compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_reference.py

It writes ``tests/data/luna_physdata.json`` with a provenance block naming the Luna
commit, branch and Julia version. The test suite only reads that file: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout.

The stored numbers are regression witnesses, not truth. Where solprop deliberately
differs -- the corrected He et al. expression for CH4, N2O and SF6, and the reference
states of the Ermolov helium and PR #425 krypton fits -- the comparison test names the
difference and its size (see ``tests/physdata/test_luna_reference.py``).
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from solprop.physdata import Gas

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_physdata.json")

# solprop (gas, source) -> the Luna symbol implementing the same model on `modal-fixed`.
# Luna has no equivalent for the Bideau-Mehu models or for the krypton fit, which live
# only in the unmerged PR #425.
LUNA_SYMBOL = {
    (Gas.Air, "borzsonyi"): "Air",
    (Gas.He, "ermolov"): "He",
    (Gas.He, "borzsonyi"): "HeB",
    (Gas.Ne, "borzsonyi"): "Ne",
    (Gas.Ar, "kingston"): "Ar",
    (Gas.Ar, "borzsonyi"): "ArB",
    (Gas.Kr, "borzsonyi"): "Kr",
    (Gas.Xe, "borzsonyi"): "Xe",
    (Gas.N2, "borzsonyi"): "N2",
    (Gas.H2, "peck"): "H2",
    (Gas.D2, "peck"): "D2",
    (Gas.O2, "zhang"): "O2",
    (Gas.CH4, "quanfu-he"): "CH4",
    (Gas.SF6, "quanfu-he"): "SF6",
    (Gas.N2O, "quanfu-he"): "N2O",
}

# The validity range of each model, in m: the published one, as solprop's former
# registry carried it (dc4ca0d, before SEP 3 removed it); the stored wavelengths were
# sampled inside these, and test_luna_reference.py supplies the same laws.
VALIDITY = {
    (Gas.Air, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.He, "ermolov"): (90e-9, 1000e-9),
    (Gas.He, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.Ne, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.Ar, "kingston"): (110e-9, 1000e-9),
    (Gas.Ar, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.Kr, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.Xe, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.N2, "borzsonyi"): (400e-9, 1000e-9),
    (Gas.H2, "peck"): (168.0e-9, 1694.5e-9),
    (Gas.D2, "peck"): (168.0e-9, 1694.5e-9),
    (Gas.O2, "zhang"): (400e-9, 1800e-9),
    (Gas.CH4, "quanfu-he"): (307e-9, 725e-9),
    (Gas.SF6, "quanfu-he"): (264e-9, 725e-9),
    (Gas.N2O, "quanfu-he"): (307e-9, 725e-9),
}

PRESSURE_BAR = 1.0
TEMPERATURE = 294.0
DENSITY_STATES = [
    (p, t) for t in (200.0, 294.0, 400.0) for p in (0.01, 0.1, 1.0, 5.0, 40.0)
]

JULIA_PROGRAM = r"""
# A plain text protocol rather than JSON.jl: this must run against whatever Julia
# environment the machine has, without installing anything into it.
#   in :  I<TAB>key<TAB>symbol<TAB>pressure<TAB>temperature<TAB>lam1,lam2,...
#         D<TAB>key<TAB>symbol<TAB>pressure<TAB>temperature
#         S<TAB>key<TAB>symbol
#   out:  N|B1|D|X3|N2|IP<TAB>key<TAB>comma separated values, or "nothing"
using Luna

num(x) = x === nothing ? "nothing" : string(x)

for line in eachline(stdin)
    isempty(strip(line)) && continue
    f = split(line, '\t')
    kind, key = f[1], f[2]
    sym = Symbol(f[3])
    if kind == "I"
        p, T = parse(Float64, f[4]), parse(Float64, f[5])
        lams = [parse(Float64, x) for x in split(f[6], ',')]
        n = [real(PhysData.ref_index(sym, l, p, T)) for l in lams]
        b1 = [PhysData.dispersion(1, sym, l, p, T) for l in lams]
        println("N\t", key, "\t", join(num.(n), ","))
        println("B1\t", key, "\t", join(num.(b1), ","))
    elseif kind == "D"
        p, T = parse(Float64, f[4]), parse(Float64, f[5])
        d = try PhysData.density(sym, p, T) catch; nothing end
        println("D\t", key, "\t", num(d))
    elseif kind == "S"
        x3 = try real(PhysData.χ3(sym, 1.0, 294.0)) catch; nothing end
        n2 = try real(PhysData.n2(sym, 1.0, 294.0)) catch; nothing end
        ip = try PhysData.ionisation_potential(sym, unit=:SI) catch; nothing end
        println("X3\t", key, "\t", num(x3))
        println("N2\t", key, "\t", num(n2))
        println("IP\t", key, "\t", num(ip))
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


def build_query() -> dict[str, dict[str, dict[str, Any]]]:
    """Build the request sent to Luna, inside each model's published validity range.

    Returns
    -------
    dict
        The query, with one entry per comparable model, per density state and per gas.
    """
    index = {}
    for (gas, source), symbol in LUNA_SYMBOL.items():
        shortest, longest = VALIDITY[gas, source]
        index[f"{gas.value}-{source}"] = {
            "symbol": symbol,
            # Ten wavelengths, inset from the ends so that neither implementation is
            # asked for a value exactly at a stated bound.
            "wavelengths": np.geomspace(shortest * 1.02, longest * 0.98, 10).tolist(),
            "pressure": PRESSURE_BAR,
            "temperature": TEMPERATURE,
        }
    # Density and the scalar data do not depend on the index model, so any Luna symbol
    # for the gas will do. Krypton's solprop default has no Luna equivalent at all.
    per_gas = {}
    for (gas, _), symbol in LUNA_SYMBOL.items():
        per_gas.setdefault(gas, symbol)
    density = {
        f"{gas.value}-{p}-{t}": {"symbol": symbol, "pressure": p, "temperature": t}
        for gas, symbol in per_gas.items()
        for p, t in DENSITY_STATES
    }
    scalars = {
        gas.value: {"symbol": symbol}
        for gas, symbol in sorted(per_gas.items(), key=lambda item: item[0].value)
    }
    return {"index": index, "density": density, "scalars": scalars}


def encode(query: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Encode the query in the line protocol the Julia program reads.

    Parameters
    ----------
    query : dict
        The output of :func:`build_query`.

    Returns
    -------
    str
        One request per line.
    """
    lines = []
    for key, spec in query["index"].items():
        wavelengths = ",".join(repr(float(value)) for value in spec["wavelengths"])
        lines.append(
            f"I\t{key}\t{spec['symbol']}\t{spec['pressure']}\t"
            f"{spec['temperature']}\t{wavelengths}"
        )
    for key, spec in query["density"].items():
        lines.append(
            f"D\t{key}\t{spec['symbol']}\t{spec['pressure']}\t{spec['temperature']}"
        )
    lines.extend(
        f"S\t{key}\t{spec['symbol']}" for key, spec in query["scalars"].items()
    )
    return "\n".join(lines) + "\n"


def decode(text: str) -> dict[str, dict[str, Any]]:
    """Decode the Julia program's output.

    Parameters
    ----------
    text : str
        Standard output of the Julia program.

    Returns
    -------
    dict
        Sections keyed by the protocol tag, each mapping a key to a value or a list.

    Raises
    ------
    RuntimeError
        If Luna reported a missing value where one is expected.
    """
    sections: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        tag, key, values = line.split("\t")
        parsed = [None if v == "nothing" else float(v) for v in values.split(",")]
        section = sections.setdefault(tag, {})
        section[key] = parsed if len(parsed) > 1 else parsed[0]
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
    query = build_query()
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", JULIA_PROGRAM],
        input=encode(query),
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
            "pressure_bar": PRESSURE_BAR,
            "temperature_k": TEMPERATURE,
            "note": (
                "Regression witnesses, not truth. Generated by "
                "tests/data/generate_luna_reference.py; see that file and "
                "tests/physdata/test_luna_reference.py for the documented differences."
            ),
        },
        "wavelengths": {k: v["wavelengths"] for k, v in query["index"].items()},
        "refractive_index": sections["N"],
        "beta1": sections["B1"],
        "density": sections["D"],
        "chi3": sections["X3"],
        "n2": sections["N2"],
        "ionisation_potential": sections["IP"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    written = payload["provenance"]["luna_commit"]
    sys.stdout.write(f"wrote {OUTPUT} from Luna {written}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

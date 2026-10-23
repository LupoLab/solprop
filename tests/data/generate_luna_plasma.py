# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna plasma polarisations the plasma tests compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_plasma.py

It writes ``tests/data/luna_plasma.json`` with a provenance block naming the Luna
commit, branch and Julia version. The test suite only reads that file: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout.

The stored numbers are regression witnesses, not truth. Three things about them matter:

* **The ionisation potential is given to Luna, not looked up by it.** Luna converts its
  hartree table with a hard-coded ``27.21138602 eV``, so its potential differs from
  solprop's by 1.7e-8. Passing solprop's value in joules removes that difference from
  the comparison and leaves ``e^2/m_e`` as the only constant to account for, which the
  test rescales exactly. The value is recorded in the file's parameters and the test
  supplies that recorded value as explicit species data; solprop's own value is now
  optmat's first ionisation energy (SEP 3, Phase 4b), so a regeneration records that.
* **Most of the cases use an analytic rate**, an eighth power with a threshold, spelled
  identically on both sides. Those cases carry no ionisation model at all, so what they
  compare is the plasma arithmetic. One case uses Luna's own direct PPT, which is where
  the ionisation model's CODATA sensitivity shows up instead. solprop has no PPT of its
  own any more (SEP 3, Phase 4b): the test supplies Luna's recorded rate.
* **Every case is run twice**, the second time with a zero potential, which is exactly
  the free-electron current. The loss current is the difference, and each half is linear
  in one constant, so both can be rescaled rather than tolerated.

The pre-ionised case is a **documented divergence**: Luna adds the pre-ionised fraction
outside the exponential, so its fraction exceeds one and its ``1 - f`` goes negative
``tests/nonlinear/test_luna_plasma.py`` asserts the exact
relation between the two rather than merely tolerating it.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from solprop.physdata import Gas, ionisation_potential

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_plasma.json")

GAS = Gas.Ar
WAVELENGTH = 800e-9
N_SAMPLES = 1025

# A carrier-resolved 10 fs pulse at 5e10 V/m, which is 3.3e14 W/cm^2 and ionises argon
# appreciably. The analytic rate's parameters are chosen so that the ionisation-loss
# current is a fifth of the total rather than a rounding of it: the test reconstructs it
# as a difference of two runs, and a term that is a millionth of the total would lose
# every digit of that difference.
AMPLITUDE = 5e10
DURATION = 10e-15
SPAN = 3.0
RATE_SCALE = 2e14
RATE_THRESHOLD = 1e10
PREIONFRAC = 0.2

# The index whose field sample is set to exactly zero, so that one case exercises the
# guard on dividing by the field in the middle of a pulse rather than only in its wings.
ZEROED_INDEX = 500

# The elliptical case: a second component in quadrature, so the magnitude is never a
# rescaling of either component and the vector kernel is genuinely exercised.
ELLIPTICITY = 0.4


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


def signals() -> dict[str, list[float]]:
    """Return the deterministic time axis and fields both sides work from.

    Returns
    -------
    dict
        The time axis in s and the field components in V/m, as plain lists.
    """
    t = np.linspace(-SPAN * DURATION, SPAN * DURATION, N_SAMPLES)
    carrier = 2.0 * np.pi * 299792458.0 / WAVELENGTH
    envelope = AMPLITUDE * np.exp(-((t / DURATION) ** 2))
    field = envelope * np.cos(carrier * t)
    zeroed = field.copy()
    zeroed[ZEROED_INDEX] = 0.0
    return {
        "t": t.tolist(),
        "field": field.tolist(),
        "zeroed": zeroed.tolist(),
        "field_x": field.tolist(),
        "field_y": (ELLIPTICITY * envelope * np.sin(carrier * t)).tolist(),
    }


def julia_vector(values: list[float]) -> str:
    """Render a sequence of floats as a Julia vector literal, bit for bit.

    Parameters
    ----------
    values : list of float
        The samples.

    Returns
    -------
    str
        A Julia array literal whose parsed values are the same doubles.
    """
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def julia_program(arrays: dict[str, list[float]]) -> str:
    """Build the Julia program, with the parameters interpolated as literals.

    Parameters
    ----------
    arrays : dict
        The output of :func:`signals`.

    Returns
    -------
    str
        The complete program: a header of constants and the sidecar source.
    """
    header = f"""
const GAS = :{GAS.value}
const WAVELENGTH = {WAVELENGTH!r}
const IONPOT = {ionisation_potential(GAS)!r}
const PREIONFRAC = {PREIONFRAC!r}
const RATE_SCALE = {RATE_SCALE!r}
const RATE_REFERENCE = {AMPLITUDE!r}
const RATE_THRESHOLD = {RATE_THRESHOLD!r}
const TAXIS = {julia_vector(arrays["t"])}
const DT = {float(arrays["t"][1] - arrays["t"][0])!r}
const FIELD = {julia_vector(arrays["field"])}
const ZEROED = {julia_vector(arrays["zeroed"])}
const FIELD_X = {julia_vector(arrays["field_x"])}
const FIELD_Y = {julia_vector(arrays["field_y"])}
"""
    return header + Path(__file__).with_name("luna_plasma.jl").read_text()


def decode(output: str) -> dict[str, list[float]]:
    """Parse the Julia program's tagged output into arrays.

    Parameters
    ----------
    output : str
        Standard output of the Julia program.

    Returns
    -------
    dict
        Tag to the list of values it carried.
    """
    emitted: dict[str, list[float]] = {}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        tag, values = line.split("\t", 1)
        emitted[tag] = [float(value) for value in values.split(",")]
    return emitted


def conditioning(emitted: dict[str, list[float]]) -> dict[str, float]:
    """Return how large the loss current is beside the free one, per case.

    The test reconstructs the loss part as a difference of two runs, which costs a
    relative accuracy of about the reciprocal of this number. A case where it is tiny
    would compare nothing.

    Parameters
    ----------
    emitted : dict
        The decoded Luna output.

    Returns
    -------
    dict
        Case name to the ratio of the peak loss polarisation to the peak free one.
    """
    ratios = {}
    for case in ("SCALAR", "PREIONISED", "ZEROS", "PPT"):
        full = np.asarray(emitted[f"{case}_P"])
        free = np.asarray(emitted[f"{case}_FREE_P"])
        ratios[case] = float(np.max(np.abs(full - free)) / np.max(np.abs(free)))
    return ratios


def main() -> int:
    """Run Luna and write the reference file.

    Returns
    -------
    int
        A process exit code.
    """
    arrays = signals()
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", julia_program(arrays)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    emitted = decode(result.stdout)
    if "CONSTANTS" not in emitted:
        sys.stderr.write("Luna emitted nothing; is the sidecar program in place?\n")
        return 1

    ratios = conditioning(emitted)
    payload: dict[str, Any] = {
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
            "loss_over_free": ratios,
            "note": (
                "Regression witnesses, not truth. Luna is given solprop's ionisation "
                "potential in joules, so only e^2/m_e needs rescaling; the pre-ionised "
                "case diverges by design. Generated by "
                "tests/data/generate_luna_plasma.py; see that file and "
                "tests/nonlinear/test_luna_plasma.py for the differences."
            ),
        },
        "parameters": {
            "gas": GAS.value,
            "wavelength": WAVELENGTH,
            "ionisation_potential": ionisation_potential(GAS),
            "preionfrac": PREIONFRAC,
            "rate_scale": RATE_SCALE,
            "rate_reference": AMPLITUDE,
            "rate_threshold": RATE_THRESHOLD,
            "zeroed_index": ZEROED_INDEX,
            "dt": float(arrays["t"][1] - arrays["t"][0]),
        },
        "signals": arrays,
        "luna": emitted,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    written = payload["provenance"]["luna_commit"]
    sys.stdout.write(f"wrote {OUTPUT} from Luna {written}\n")
    sys.stdout.write(f"loss over free: {ratios}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

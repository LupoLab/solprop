# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate Luna's per-mode energy along z for the full M2 case, with two switches.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_rdw_switches.py

It writes ``tests/data/luna_rdw_switches.json`` (the energies) and
``tests/data/luna_rdw_switches_provenance.json``. The test suite reads only those files:
it never runs Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment rather than
against the checkout's own project, so no manifest there is touched.

**Why this exists.** The first comparison left the full four-mode case's second mode
at 0.564 of Luna's energy at the exit while the third and fourth agreed to 8 % and
17 %, and it was not clear whether that is a phase-sensitive exit value or a real
divergence. The two codes' mode-2 energy along z answers it, and if they do diverge
the question is which term does it -- hence the plasma and the waveguide loss, switched
independently, giving a two-by-two ladder.

Two further ladders rule out the two things that are neither code's physics: the
**transverse quadrature order**, at 32, 64 and 128 on both sides, and the **step
tolerance**, over two decades on both sides. Luna's tolerance rows go through
``prop_capillary_args`` and ``Luna.run`` because its interface cannot set one.

**The reference row is a check, not data.** Luna's statistics were never trimmed out of
``luna_rdw_full.h5``: it already carries ``stats/energy`` per mode at every one of the
4365 accepted steps, which is finer than the 201 save planes. The ``REFERENCE`` row here
is the same run again, so that the three switched rows can be believed only once it
reproduces what is already stored.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solprop.constants import ROOM_TEMPERATURE

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
HERE = Path(__file__).parent
EMITTED = HERE / "luna_rdw_switches.json"
PROVENANCE = HERE / "luna_rdw_switches_provenance.json"

# Luna's published example, verbatim, as `generate_luna_rdw.py` runs it.
LAMBDA0 = 800e-9
TAU_FWHM = 7.5e-15
ENERGY = 275e-6
FULL = {
    "radius": 125e-6,
    "length": 1.5,
    "p_in": 0.8,
    "p_out": 0.0,
    "modes": 4,
    "lambda_min": 90e-9,
    "lambda_max": 4e-6,
    "trange": 400e-15,
    "n_save": 201,
}
SWITCHES = ("REFERENCE", "NOPLASMA", "NOLOSS", "NEITHER")

# Two more ladders on the reference row, so that the quadrature and the stepping are
# ruled out on Luna's side by measurement rather than by inspection, as they were on
# solprop's. The orders are the same integers on both sides: each code rounds `nr` up to
# the Kronrod extension of the `ceil((nr - 1) / 2)`-point Gauss rule, so the realised
# node counts match, and each row emits the count it actually used.
QUADRATURE_ORDERS = (32, 64, 128)
# Three decades below Luna's own default, which is the tolerance every published Luna
# result carries because its interface cannot set one. The third decade is there
# because the second did not settle Luna's higher modes.
TOLERANCE_EXPONENTS = (6, 7, 8, 9)
ATOL = 1e-10
LADDERS = (
    tuple(f"NR{nr}" for nr in QUADRATURE_ORDERS)
    + tuple(f"RTOL{exponent}" for exponent in TOLERANCE_EXPONENTS)
    + tuple(f"NOPLASMA_RTOL{exponent}" for exponent in TOLERANCE_EXPONENTS)
)

# `Luna.Utils._native_fftw_threads[]`, encoded numerically in the Julia program because
# its `emit` is numbers-only.
FFTW_POOL_NAMES = {0: "none", 1: "native", 2: "julia-task", 3: "undetermined"}


def git(*arguments: str) -> str:
    """Return the output of a read-only git command in the Luna checkout."""
    return subprocess.run(
        ["git", "-C", str(LUNA), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def julia_program() -> str:
    """Return the Julia program: the case constants, then the sidecar."""
    lines = [
        "const GAS = :He",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const TAU_FWHM = {TAU_FWHM!r}",
        f"const ENERGY = {ENERGY!r}",
        f"const TEMPERATURE = {ROOM_TEMPERATURE!r}",
        f"const RADIUS = {FULL['radius']!r}",
        f"const LENGTH = {FULL['length']!r}",
        f"const P_IN = {FULL['p_in']!r}",
        f"const P_OUT = {FULL['p_out']!r}",
        f"const MODES = {FULL['modes']!r}",
        f"const LAMBDA_MIN = {FULL['lambda_min']!r}",
        f"const LAMBDA_MAX = {FULL['lambda_max']!r}",
        f"const TRANGE = {FULL['trange']!r}",
        f"const N_SAVE = {FULL['n_save']!r}",
        f"const QUADRATURE_ORDERS = {list(QUADRATURE_ORDERS)!r}",
        f"const TOLERANCE_EXPONENTS = {list(TOLERANCE_EXPONENTS)!r}",
        f"const ATOL = {ATOL!r}",
    ]
    return "\n".join(lines) + "\n" + (HERE / "luna_rdw_switches.jl").read_text()


def decode(stdout: str) -> dict[str, list[float]]:
    """Return the tagged, tab-separated lines Julia emitted as a flat mapping."""
    emitted: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        tag, values = line.split("\t", 1)
        emitted[tag] = [float(value) for value in values.split(",")]
    return emitted


def digest(path: Path) -> str:
    """Return the SHA-256 of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    """Run Luna's four switch settings and store the energies they emit."""
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", julia_program()],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr)
        return result.returncode
    emitted = decode(result.stdout)
    if "CONSTANTS" not in emitted:
        sys.stderr.write("Luna emitted nothing; is the sidecar program in place?\n")
        return 1
    EMITTED.write_text(json.dumps(emitted, indent=1, sort_keys=True) + "\n")

    payload: dict[str, Any] = {
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
        "temperature_k": ROOM_TEMPERATURE,
        "shotnoise": False,
        "lambda0": LAMBDA0,
        "tau_fwhm": TAU_FWHM,
        "energy": ENERGY,
        "full": FULL,
        "switches": {
            "REFERENCE": {"plasma": True, "loss": True},
            "NOPLASMA": {"plasma": False, "loss": True},
            "NOLOSS": {"plasma": True, "loss": False},
            "NEITHER": {"plasma": False, "loss": False},
        },
        "quadrature_orders": list(QUADRATURE_ORDERS),
        "tolerance_exponents": list(TOLERANCE_EXPONENTS),
        "atol": ATOL,
        "runtime_s": {tag: emitted[f"{tag}_RUNTIME"][0] for tag in SWITCHES + LADDERS},
        "steps": {tag: int(emitted[f"{tag}_STEPS"][0]) for tag in SWITCHES + LADDERS},
        "transverse_points": {
            tag: int(emitted[f"{tag}_POINTS"][0]) for tag in SWITCHES + LADDERS
        },
        "julia_threads": int(emitted["THREADS"][0]),
        "blas_threads": int(emitted["THREADS"][1]),
        "fftw_threads": int(emitted["THREADS"][2]),
        "fftw_pool": FFTW_POOL_NAMES[int(emitted["THREADS"][3])],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (EMITTED,)
        },
        "note": (
            "Energies are evaluated from the saved spectra with Fields.energyfuncs, "
            "the functional Stats.energy uses, so the REFERENCE row is comparable with "
            "the untrimmed stats/energy already in luna_rdw_full.h5. The "
            "temperature is solprop's ROOM_TEMPERATURE not Luna's roomtemp, as "
            "generate_luna_rdw.py already does. julia_threads, blas_threads, "
            "fftw_threads and fftw_pool are read inside the Julia program rather "
            "than assumed."
        ),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(payload["runtime_s"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

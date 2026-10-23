# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna results the two modal-overlap oracles compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_overlap.py

It writes ``tests/data/luna_overlap.json`` (the emitted values) and
``tests/data/luna_overlap_provenance.json``. The test suite reads only those files: it
never runs Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment rather than
against the checkout's own project, so no manifest there is touched.

This is a **new** pair rather than an extension of ``generate_luna_rdw.py``: that
generator's two HDF5 witnesses are 3 MB of committed data and its full case costs 105 s,
and nothing about the overlap oracles should make regenerating those witnesses
necessary.

What the two oracles are, and why they do not compare the two codes with each other:

* **The projection of a known field.** A field that is the fundamental alone has a Kerr
  polarisation ``ehat_1(r)^3 f(t)``, which factorises, so the drive projected into mode
  p is ``T_p f(t)`` with ``T_p = int ehat_1^3 ehat_p dA``. Every later step of Luna's
  ``_finish_modal!`` is the same for every mode, so the ratio ``nl[:, p] / nl[:, 1]`` is
  exactly ``T_p / T_1`` -- a pure Bessel number, 0.4785423 for the second mode
  Luna's own transform is asked for it directly.
* **The undepleted-pump two-mode solution.** With a constant fill, no plasma, a weak
  pump in the fundamental and a weak seed in the second mode, the first-order solution
  of the coupled envelope equations is a closed form whose only content is that same
  overlap ratio and the phase mismatch. Each code is measured against it rather than
  against the other.

Everything the comparison needs from Luna is emitted here rather than assumed: the mode
normalisations, the effective areas, the Bessel zeros, the profiles at a few radii, the
discrete Gram matrix of the basis on Luna's own quadrature, the ratios at three radial
orders with and without the Kronrod extension -- so that "it does not move with the
quadrature order" is a measurement on Luna's side too -- and, for the second oracle, the
input and exit spectra of both modes for a Kerr run and a linear one, from which the
Python side reconstructs Luna's own linear propagation without modelling any of it.
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
EMITTED = HERE / "luna_overlap.json"
PROVENANCE = HERE / "luna_overlap_provenance.json"

# The M2 geometry, at a constant fill: the overlap tensor does not depend on the gas at
# all -- the profile is J_0(u_m r/a) and the normalisation is (pi/2) a^2 J_1(u_m)^2 eps0
# c -- but keeping the radius and the fill of the case under investigation means the
# numbers below can be read beside the RDW witnesses' without a change of geometry.
RADIUS = 125e-6
GAS = "He"
PRESSURE = 1.0
LAMBDA0 = 800e-9

# The grid the projection is evaluated on. Only its frequency axis matters: the ratio is
# frequency-independent by construction, and the length enters nothing.
LENGTH = 0.03
LAMBDA_MIN = 200e-9
LAMBDA_MAX = 4e-6
TRANGE = 400e-15

# The spectrum put in the fundamental. Any shape gives the same ratio; a Gaussian a
# tenth of the carrier wide keeps the drive well above rounding across a wide band, so
# the *spread* of the ratio over that band is a meaningful witness.
BANDWIDTH_FRACTION = 0.1
RATIO_FLOOR = 1e-8

REDUCED_MODES = 2
FULL_MODES = 4
PROFILE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)

# Luna's own default is nr = 64 (`Interface.jl`), which with the Kronrod extension it
# also defaults to becomes 65 nodes. The two higher orders are the ladder.
QUADRATURE_ORDERS = (64, 128, 256)
REFERENCE_NR = 512

# The undepleted-pump case of the second oracle. Its length is set by the phase
# mismatch: beta_1 - beta_2 is 100.6 rad/m for these two modes at 800 nm, so 2 cm is
# about two radians -- large enough that the mismatch factor is well away from one and
# far enough from pi that the linear transfer's phase can be read off unambiguously.
# The energy sets the nonlinear phase to 2.2e-3, so the second-order term the
# first-order solution drops is of that size; the seed is an amplitude fraction of the
# pump, and it is what makes the second mode's *linear* transfer measurable, without
# which the mode it drives has no reference to propagate against.
UNDEPLETED = {
    "length": 0.02,
    "pressure": 1.0,
    "fwhm": 30e-15,
    "energy": 3e-8,
    "trange": 600e-15,
    "lambda_min": 400e-9,
    "lambda_max": 2e-6,
    "seed": 1e-3,
    "nr": 64,
    "rtol": 1e-11,
    "atol": 1e-14,
}

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
    omega0 = 2.0 * 3.141592653589793 * 299792458.0 / LAMBDA0
    lines = [
        f"const GAS = :{GAS}",
        f"const RADIUS = {RADIUS!r}",
        f"const PRESSURE = {PRESSURE!r}",
        f"const TEMPERATURE = {ROOM_TEMPERATURE!r}",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const LENGTH = {LENGTH!r}",
        f"const LAMBDA_MIN = {LAMBDA_MIN!r}",
        f"const LAMBDA_MAX = {LAMBDA_MAX!r}",
        f"const TRANGE = {TRANGE!r}",
        f"const OMEGA0 = {omega0!r}",
        f"const BANDWIDTH = {BANDWIDTH_FRACTION * omega0!r}",
        f"const RATIO_FLOOR = {RATIO_FLOOR!r}",
        f"const REDUCED_MODES = {REDUCED_MODES!r}",
        f"const FULL_MODES = {FULL_MODES!r}",
        f"const PROFILE_FRACTIONS = {list(PROFILE_FRACTIONS)!r}",
        f"const QUADRATURE_ORDERS = {list(QUADRATURE_ORDERS)!r}",
        f"const REFERENCE_NR = {REFERENCE_NR!r}",
    ]
    lines += [
        f"const UNDEPLETED_{key.upper()} = {value!r}"
        for key, value in UNDEPLETED.items()
    ]
    header = "\n".join(lines) + "\n"
    return header + (HERE / "luna_overlap.jl").read_text()


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
    """Run Luna's side of both oracles and store what it emitted."""
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
        "radius": RADIUS,
        "gas": GAS,
        "pressure_bar": PRESSURE,
        "lambda0": LAMBDA0,
        "reduced_modes": REDUCED_MODES,
        "full_modes": FULL_MODES,
        "quadrature_orders": list(QUADRATURE_ORDERS),
        "reference_nr": REFERENCE_NR,
        "profile_fractions": list(PROFILE_FRACTIONS),
        "undepleted": UNDEPLETED,
        "julia_threads": int(emitted["THREADS"][0]),
        "blas_threads": int(emitted["THREADS"][1]),
        "fftw_threads": int(emitted["THREADS"][2]),
        "fftw_pool": FFTW_POOL_NAMES[int(emitted["THREADS"][3])],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (EMITTED,)
        },
        "note": (
            "The temperature is solprop's ROOM_TEMPERATURE rather than "
            "Luna's roomtemp, as tests/data/generate_luna_rdw.py already does, so that "
            "0.29 % of density is out of the comparison. Nothing here depends on the "
            "gas: the HE_1m profile and its normalisation are geometry alone. "
            "julia_threads, blas_threads, fftw_threads and fftw_pool are read inside "
            "the Julia program rather than assumed."
        ),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(payload["files"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

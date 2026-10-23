# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate split and analytic two-channel propagation witnesses from read-only Luna."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_mixture_propagation.json")
PARAMETERS = {
    "length": 1e-3,
    "wavelength": 800e-9,
    "lambda_min": 250e-9,
    "lambda_max": 3e-6,
    "trange": 200e-15,
    "radius": 30e-6,
    "fwhm": 30e-15,
    "energy": 30e-9,
    "temperature": 300.0,
    "rate_reference": 3e10,
    "rate_scales": [2e13, 7e13],
    "potentials": [2e-18, 3e-18],
    "steps": [16, 32, 64],
    "rtol": 1e-9,
    "atol": 1e-14,
}


def main():
    """Record bounds first, then preserve Julia output and provenance."""
    # Fixed schedules eliminate controller drift. DOPRI5 has global order five:
    # the finest-grid error estimate is |u_32-u_64|/(2**5-1). Cross-code comparison
    # permits the sum of the independently measured estimates and a 64-eps budget
    # per fine sample, channel, stage and step. Its coefficient is fixed here before
    # any field is generated; the test additionally checks refinement consistency.
    bounds = {
        "order": 5,
        "richardson_divisor": 31,
        "roundoff_coefficient": 64,
        "roundoff_formula": "64 * n_to * n_channels * 7 * n_steps * eps",
        "split_formula": "64 * n_to * 2 * 7 * n_steps * eps",
        "integrator_rtol": PARAMETERS["rtol"],
        "integrator_atol": PARAMETERS["atol"],
        "eps": float(np.finfo(float).eps),
    }
    evidence = Path("build/mixture-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "luna-mixture-propagation-bounds.json").write_text(
        json.dumps(bounds, indent=2) + "\n"
    )
    header = "\n".join(
        f"const {key.upper()} = {value!r}" for key, value in PARAMETERS.items()
    )
    header += "\nconst CACHE_PATH = " + json.dumps(
        str((evidence / "julia-scratch").resolve())
    )
    program = (
        header
        + "\n"
        + Path(__file__).with_name("luna_mixture_propagation.jl").read_text()
    )
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", program],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "JULIA_SCRATCH_TRACK_ACCESS": "0",
            "MPLCONFIGDIR": str((evidence / "matplotlib").resolve()),
        },
    )
    (evidence / "luna-mixture-propagation-stdout.log").write_text(result.stdout)
    (evidence / "luna-mixture-propagation-stderr.log").write_text(result.stderr)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    emitted = {}
    for line in result.stdout.splitlines():
        if "\t" in line:
            tag, values = line.split("\t", 1)
            emitted[tag] = [float(value) for value in values.split(",")]
    commit = subprocess.run(
        ["git", "-C", str(LUNA), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    version = subprocess.run(
        ["julia", "--startup-file=no", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {
        "provenance": {
            "luna_path": str(LUNA),
            "luna_commit": commit,
            "julia_version": version,
            "generated": datetime.now(UTC).isoformat(),
            "note": (
                "Regression witnesses, not truth. Constants, optical operator, "
                "initial spectrum and mode-average prefactor are emitted for "
                "matched-constant arithmetic comparisons."
            ),
        },
        "parameters": PARAMETERS,
        "bounds": bounds,
        "luna": emitted,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(f"Wrote {OUTPUT} from Luna {commit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

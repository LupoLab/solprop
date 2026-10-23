# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate mixture susceptibility witnesses from the read-only Luna checkout.

Run with ``uv run python tests/data/generate_luna_mixture.py``. The sidecar emits
its constants, species susceptibilities and actual partial densities, so differences
in reference states or CoolProp versions are recorded rather than hidden in a bound.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))


def main() -> int:
    """Record the Luna identity and the emitted mixture data."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(LUNA), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    before = git("status", "--porcelain")
    sidecar = Path(__file__).with_name("luna_mixture.jl")
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", sidecar.read_text()],
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stderr.write(result.stderr)
    if result.returncode:
        return result.returncode
    emitted = {}
    for line in result.stdout.splitlines():
        if "\t" in line:
            tag, values = line.split("\t", 1)
            emitted[tag] = [float(value) for value in values.split(",")]
    if before != git("status", "--porcelain"):
        raise RuntimeError("Luna checkout status changed during witness generation")
    output = {
        "provenance": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "julia": subprocess.run(
                ["julia", "--startup-file=no", "--version"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "checkout_status": before,
        },
        "definition": (
            "Tuple/tuple ref_index_fun; N2/O2/Ar partial pressures in bar; "
            "per-particle gamma in m^3; wavelengths in m; T=300 K"
        ),
        "emitted": emitted,
    }
    Path(__file__).with_name("luna_mixture.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

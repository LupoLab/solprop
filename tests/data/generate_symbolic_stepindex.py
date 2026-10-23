# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate 40-digit step-index eigenvalues and implicit derivatives with mpmath.

Run with ``uv run python tests/data/generate_symbolic_stepindex.py``. Double precision
roots supply initial guesses only; mpmath solves and differentiates the original
characteristic product independently, without importing the production derivative.
The separate cutoff/root-count tests qualify completeness of the double root scan.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mpmath as mp

from solprop.constants import C
from solprop.modes.stepindex import _Kind, _roots

OUTPUT = Path(__file__).with_name("symbolic_stepindex.json")
FAMILIES: tuple[tuple[_Kind, int], ...] = (
    ("HE", 1),
    ("HE", 2),
    ("HE", 3),
    ("EH", 1),
    ("TE", 0),
    ("TM", 0),
)
FIBRES = (
    ("issue320", 2e-6, 1.47, 1.45, (500e-9, 800e-9, 1000e-9)),
    ("strand", 1e-6, 1.45, 1.0, (500e-9, 1030e-9, 1500e-9)),
    ("NA", 4e-6, 1.45 + 0.12**2 / (2 * 1.45), 1.45, (500e-9, 800e-9, 1030e-9)),
)


@dataclass(frozen=True)
class Case:
    fibre: str
    radius: float
    core: float
    cladding: float
    wavelength: float
    kind: _Kind
    n: int
    m: int
    material_slope: float = 0.0


def characteristic(index, omega, materials, n, kind):
    """Evaluate the textbook product with high-precision Bessel functions."""
    radius, core, cladding = materials
    u = radius * omega / mp.mpf(str(C)) * mp.sqrt(core**2 - index**2)
    w = radius * omega / mp.mpf(str(C)) * mp.sqrt(index**2 - cladding**2)
    j = mp.besselj(n, u)
    k = mp.besselk(n, w)
    jp = (mp.besselj(n - 1, u) - mp.besselj(n + 1, u)) / 2
    kp = -(mp.besselk(n - 1, w) + mp.besselk(n + 1, w)) / 2
    first, second = u * w**2 * jp * k, u**2 * w * j * kp
    ratio = (cladding / core) ** 2
    if kind == "TE":
        return first + second
    if kind == "TM":
        return first + ratio * second
    return (first + second) * (first + ratio * second) - n**2 * (u**2 + w**2) * (
        w**2 + ratio * u**2
    ) * j**2 * k**2


def witness_row(case: Case, seed: float):
    """Refine an eigenpair and differentiate it at fixed index and frequency."""
    a, nc, nl, wavelength_mp = (
        mp.mpf(str(x)) for x in (case.radius, case.core, case.cladding, case.wavelength)
    )
    frequency = 2 * mp.pi * mp.mpf(str(C)) / wavelength_mp
    slope = mp.mpf(str(case.material_slope))

    def equation(index, omega):
        return characteristic(
            index, omega, (a, nc + slope * (omega - frequency), nl), case.n, case.kind
        )

    guess = mp.sqrt(nc**2 - (mp.mpf(str(seed)) * mp.mpf(str(C)) / (a * frequency)) ** 2)
    root = mp.findroot(
        lambda index: equation(index, frequency),
        guess,
        solver="newton",
        tol=mp.mpf("1e-37"),
    )
    derivative = -mp.diff(lambda omega: equation(root, omega), frequency) / mp.diff(
        lambda index: equation(index, frequency), root
    )
    return asdict(case) | {
        "neff": float(mp.re(root)),
        "derivative": float(mp.re(derivative)),
        "residual": str(abs(equation(root, frequency))),
    }


def main():
    """Write the witness with precision, source and case provenance."""
    mp.mp.dps = 40
    rows = []
    for name, radius, core, cladding, wavelengths in FIBRES:
        for wavelength in wavelengths:
            v = 2 * math.pi * radius / wavelength * math.sqrt(core**2 - cladding**2)
            for kind, n in FAMILIES:
                for m, seed in enumerate(
                    _roots(v, (cladding / core) ** 2, n, kind)[:2], start=1
                ):
                    rows.append(
                        witness_row(
                            Case(name, radius, core, cladding, wavelength, kind, n, m),
                            seed,
                        )
                    )
    seed = _roots(
        2 * math.pi / 1.03 * math.sqrt(1.45**2 - 1), (1 / 1.45) ** 2, 1, "HE"
    )[0]
    rows.append(
        witness_row(
            Case("linear core", 1e-6, 1.45, 1.0, 1030e-9, "HE", 1, 1, 1e-17), seed
        )
    )
    OUTPUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "generator": Path(__file__).name,
                    "mpmath": mp.__version__,
                    "digits": 40,
                    "generated": datetime.now(UTC).isoformat(),
                    "reference": "Snyder and Love, chapter 12",
                },
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(f"wrote {len(rows)} cases to {OUTPUT}\n")


if __name__ == "__main__":
    main()

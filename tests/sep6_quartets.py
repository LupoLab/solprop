# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reproduce SEP 6's material tables from optmat's public models (SEP 6, Phase 0).

``uv run python tests/sep6_quartets.py`` writes ``tests/data/sep6_quartets.json``;
``tests/test_sep6_quartets.py`` recomputes every table and compares. Everything is an
evaluation of optmat's :class:`optmat.gas.nonlinear.HyperpolarizabilityModel` at the
pinned optmat: its parameters, declared range and largest fitted ``W2`` and, for the
unit check, its own ``gamma3``. Nothing is fitted or transcribed here.

The tables are those of SEP 6, Part I:

- ``si_adapter``: solprop's SI evaluation of the polynomial against optmat's ``gamma3``
  for the Kerr, third-harmonic and pump-probe processes (section 4.2's Taylor factor and
  section 4.5's unit conversions);
- ``survey``: section 4.7's random-quartet comparison of the symmetric and output-only
  constructions with the polynomial;
- ``fwm``: section 4.7's degenerate-pump quartets;
- ``harmonic``: section 4.8's shortest pumps;
- ``leg_bounds``: section 4.10's bound ``sqrt(3 W2_max / 4)``;
- ``demonstration``: section 4.7's demonstration quartets, unbounded and bounded.

Ratios are dimensionless, so the polynomial is evaluated per unit ``gamma0``, with
optmat's coefficients in its own units (``A`` in 1e-10 cm^2, ``B`` and ``B'`` in
1e-20 cm^4) and wavenumbers in cm^-1, except in ``si_adapter``.
"""

import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import optmat
from optmat.exceptions import ModelRangeWarning
from optmat.gas.gases import Gas
from optmat.gas.nonlinear import AU_HYPERPOLARIZABILITY, hyperpolarizability_models

from solprop.constants import EPS0

GASES = (
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "H2",
    "D2",
    "N2",
    "O2",
    "CO2",
    "CH4",
    "SF6",
    "CO",
    "N2O",
    "O3",
)
RECORD = Path(__file__).parent / "data" / "sep6_quartets.json"


def _model(gas: str):
    return hyperpolarizability_models()[Gas(gas)]


def _coefficients(gas: str) -> tuple[float, float, float, float]:
    """Return ``A, B, B'`` in cm units and the largest fitted ``W2`` in cm^-2."""
    model = _model(gas)
    _, a, b, b_prime = model.parameters
    return (
        a * 1e-10,
        b * 1e-20,
        b_prime * 1e-20,
        float(str(model.record["max_w2_fitted_cm-2"])),
    )


def _polynomial(gas: str, legs: np.ndarray) -> np.ndarray:
    """Return ``gamma / gamma0`` of quartets ``legs[..., 4]`` in cm^-1."""
    a, b, b_prime, _ = _coefficients(gas)
    s2 = (legs**2).sum(-1)
    s4 = (legs**4).sum(-1)
    return 1 + a * s2 + b * s2**2 + b_prime * s4


def _kerr(gas: str, nu: np.ndarray | float) -> np.ndarray:
    nu = np.asarray(nu, float)
    return _polynomial(gas, np.stack([nu, nu, nu, nu], axis=-1))


def _symmetric(gas: str, legs: np.ndarray) -> np.ndarray:
    return np.prod(_kerr(gas, np.abs(legs)) ** 0.25, axis=-1)


def _cap(gas: str) -> float:
    """Return section 4.10's leg bound ``sqrt(3 W2_max / 4)`` in cm^-1."""
    return float(np.sqrt(0.75 * _coefficients(gas)[3]))


def si_adapter() -> dict[str, float]:
    """Return, per gas, the largest relative difference from optmat's ``gamma3``.

    solprop's own SI evaluation: ``gamma3 = gamma0 * au / (6 eps0)`` times the
    polynomial with ``A`` in m^2 (1e-10 cm^2 = 1e-14 m^2), ``B``, ``B'`` in m^4
    (1e-20 cm^4 = 1e-28 m^4) and vacuum wavenumbers in m^-1.
    """
    out = {}
    for gas in GASES:
        model = _model(gas)
        gamma0, a, b, b_prime = model.parameters
        low = model.wavelength_range[0]
        wavelengths = np.array([max(low, 400e-9), 800e-9, 1030e-9, 2000e-9])
        worst = 0.0
        for process in ("kerr", "thg"):
            nu = 1.0 / wavelengths
            legs = {
                "kerr": np.stack([nu, nu, nu, nu], -1),
                "thg": np.stack([3 * nu, nu, nu, nu], -1),
            }[process]
            s2 = (legs**2).sum(-1)
            s4 = (legs**4).sum(-1)
            ours = (
                gamma0
                * AU_HYPERPOLARIZABILITY
                / (6 * EPS0)
                * (1 + a * 1e-14 * s2 + b * 1e-28 * s2**2 + b_prime * 1e-28 * s4)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ModelRangeWarning)
                theirs = model.gamma3(process, wavelengths)
            worst = max(worst, float(np.max(np.abs(ours / theirs - 1))))
        probe = 1.0 / 600e-9
        pump = 1.0 / 1030e-9
        legs = np.array([probe, probe, pump, pump])
        ours_pp = _polynomial(gas, legs * 1e-2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModelRangeWarning)
            theirs_pp = model.gamma_au("pump_probe", 1030e-9, 600e-9) / gamma0
        worst = max(worst, abs(float(ours_pp / theirs_pp) - 1))
        out[gas] = worst
    return out


def survey(gas: str) -> dict[str, float]:
    """Return section 4.7's survey row: 300 000 draws, ``default_rng(4)``."""
    low = 700.0 if gas == "O3" else 400.0
    high = 2000.0
    w2_max = _coefficients(gas)[3]
    rng = np.random.default_rng(4)
    nu = 1e7 / rng.uniform(low, high, size=(300_000, 3))
    out_nu = nu[:, 0] + nu[:, 1] - nu[:, 2]
    legs = np.column_stack([out_nu, nu])
    with np.errstate(divide="ignore"):
        out_nm = 1e7 / out_nu
    keep = (
        (out_nu > 0) & (out_nm >= low) & (out_nm <= high) & ((legs**2).sum(1) <= w2_max)
    )
    legs = legs[keep]
    poly = _polynomial(gas, legs)
    symmetric = 100 * np.abs(_symmetric(gas, legs) / poly - 1)
    output = 100 * np.abs(_kerr(gas, legs[:, 0]) / poly - 1)
    return {
        "quartets": int(keep.sum()),
        "diagonal_ratio": float(_kerr(gas, 1e7 / low) / _kerr(gas, 1e7 / high)),
        "symmetric_median": float(np.median(symmetric)),
        "symmetric_p95": float(np.percentile(symmetric, 95)),
        "symmetric_max": float(symmetric.max()),
        "output_median": float(np.median(output)),
        "output_p95": float(np.percentile(output, 95)),
        "output_max": float(output.max()),
    }


def _row(
    gas: str, legs_nm: tuple[float, ...], signs: tuple[int, ...]
) -> dict[str, float]:
    """Return one quartet's comparisons; legs given output first, in nm."""
    nu = np.array([1e7 / x for x in legs_nm])
    poly = float(_polynomial(gas, nu))
    cap = _cap(gas)
    bounded = float(_polynomial(gas, np.minimum(nu, cap)))
    signed = nu * np.array(signs)
    return {
        "polynomial_over_kerr_800": poly / float(_kerr(gas, 1e7 / 800)),
        "polynomial_over_kerr_pump": poly / float(_kerr(gas, nu[1])),
        "symmetric_error_pct": 100 * (float(_symmetric(gas, nu)) / poly - 1),
        "output_only_error_pct": 100 * (float(_kerr(gas, nu[0])) / poly - 1),
        "carrier_400_error_pct": 100 * (float(_kerr(gas, 1e7 / 400)) / poly - 1),
        "carrier_800_error_pct": 100 * (float(_kerr(gas, 1e7 / 800)) / poly - 1),
        "s2_over_max": float((nu**2).sum() / _coefficients(gas)[3]),
        "bounded_over_polynomial": bounded / poly,
        "largest_pair_sum_cm": float(
            max(abs(signed[i] + signed[j]) for i in range(4) for j in range(i + 1, 4))
        ),
    }


def _fwm(pump: float, idler: float) -> tuple[float, ...]:
    signal = 1.0 / (2.0 / pump - 1.0 / idler)
    return (signal, pump, pump, idler)


FWM = (
    ("He", 800, 2000),
    ("Ar", 800, 2000),
    ("Xe", 800, 2000),
    ("N2", 800, 2000),
    ("Ar", 1030, 2000),
    ("Xe", 1030, 2000),
    ("Ar", 515, 2000),
    ("Ar", 343, 1030),
    ("Xe", 343, 1030),
)
DEMONSTRATION = {
    "Ar 400+400-800": ("Ar", _fwm(400, 800), (-1, 1, 1, -1)),
    "Ar SPM 266.7": ("Ar", (800 / 3,) * 4, (-1, 1, 1, -1)),
    "Ar XPM 266.7 by 400": ("Ar", (800 / 3, 800 / 3, 400, 400), (-1, 1, 1, -1)),
    "He THG 800": ("He", (800 / 3, 800, 800, 800), (-1, 1, 1, 1)),
    "He 250+250-1500": ("He", _fwm(250, 1500), (-1, 1, 1, -1)),
    "He 200+200-600": ("He", _fwm(200, 600), (-1, 1, 1, -1)),
    "He XPM 120 by 800": ("He", (120, 120, 800, 800), (-1, 1, 1, -1)),
    "He SPM 120": ("He", (120,) * 4, (-1, 1, 1, -1)),
    "He SPM 80": ("He", (80,) * 4, (-1, 1, 1, -1)),
}


def harmonic(gas: str) -> dict[str, float]:
    """Return section 4.8's shortest pumps, nm."""
    w2_max = _coefficients(gas)[3]
    low = 700.0 if gas == "O3" else 400.0
    return {
        "thg_within_w2": 1e7 / np.sqrt(w2_max / 12),
        "harmonic_leg_within_kerr_w2": 3e7 / np.sqrt(w2_max / 4),
        "harmonic_leg_within_declared": 3 * low,
    }


def tables() -> dict[str, object]:
    """Return every table, with the identities of the models evaluated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ModelRangeWarning)
        return {
            "si_adapter": si_adapter(),
            "survey": {gas: survey(gas) for gas in GASES},
            "fwm": {
                f"{gas} {p}/{i}": _row(gas, _fwm(p, i), (-1, 1, 1, -1))
                for gas, p, i in FWM
            },
            "harmonic": {gas: harmonic(gas) for gas in GASES},
            "leg_bounds_nm": {gas: 1e7 / _cap(gas) for gas in GASES},
            "demonstration": {
                name: _row(gas, legs, signs)
                for name, (gas, legs, signs) in DEMONSTRATION.items()
            },
            "models": {gas: _model(gas).identity.digest for gas in GASES},
        }


def main() -> None:
    """Write the record with the commit and versions it was made at."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    record = {
        "solprop_commit": commit,
        "optmat": optmat.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        **tables(),
    }
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

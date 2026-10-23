# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Derive the Marcatili effective index and its frequency derivatives symbolically.

Run once, by hand:

.. code-block:: bash

    uv run python tests/data/generate_symbolic_marcatili.py

It writes ``tests/data/symbolic_marcatili.json``: for each of six cases, the
propagation constant and its first three frequency derivatives, the power loss
coefficient and the zero-dispersion wavelength, all obtained by differentiating the
closed-form effective index with sympy rather than by any finite difference. The test
suite only reads that file, so sympy is needed to regenerate it and not to run the
tests; it is a dev dependency only because the type checker and the doctest
collection import this script.

This is a port of Luna's ``test/symbolic_marcatilli.py``, which is where the reference
numbers in Luna's own ``test_capillary.jl`` come from. It differs in three ways. It
covers the reduced *and* the full model rather than the full model alone; it emits the
loss of both; and it derives each case twice, once from solprop's own Sellmeier
coefficients and reference density and once from the constants Luna hard-codes, so that
the same file both validates solprop's analytic derivative and shows exactly how much
of any disagreement with Luna's published numbers is the input data rather than the
mathematics.

What this witnesses is the *calculus*: the inputs are shared with solprop, and what
sympy supplies independently is the differentiation. The chain rule through the
material models is separately checked against finite differences, and the algebra of
the model against a contour integral, in ``tests/modes/test_capillary.py``.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import scipy.optimize
import sympy

from solprop.constants import C
from solprop.physdata import Gas, density
from solprop.physdata.sellmeier import PoleSumSusceptibility, ReferenceState

OUTPUT = Path(__file__).with_name("symbolic_marcatili.json")

RADIUS = 50e-6
"""Core radius in m, as in Luna's script."""

U_NM = 2.4048255576957724
"""The HE11 Bessel zero, as in Luna's script."""

WAVELENGTH = 800e-9
"""Where the derivatives are evaluated, m."""

PRESSURE_BAR = 2.0
TEMPERATURE = 294.0

CLADDINGS = {
    "silica-like": complex(1.45, 0.0),
    "metal-like": complex(0.036759, 5.5698),
}
"""The two constant cladding indices Luna's script uses."""

# Luna divides the Boerzsoenyi argon coefficients by its own 1 bar, 0 C density and
# evaluates the fill density with CoolProp. Both are transcribed here exactly as they
# appear in Luna's script, so that the "luna" cases reproduce its published numbers.
LUNA_REFERENCE_DENSITY = 2.6541047057884805e25
LUNA_FILL_DENSITY = 4.933761614599279e25
LUNA_POLES = ((20332.29e-8, 206.12e-6), (34458.31e-8, 8.066e-3))

ZDW_BRACKET = (1e14, 1e16)
"""Angular frequencies in rad/s to bracket the zero-dispersion root in, as in Luna."""

MAX_ORDER = 3
"""Highest frequency derivative of beta emitted."""


def solprop_inputs() -> dict[str, Any]:
    """Return the argon pole sum and fill density solprop itself would use.

    Returns
    -------
    dict
        ``poles`` as ``(strength, resonance)`` pairs with the strength already at the
        reference state, ``reference_density`` in m^-3 and ``fill_density`` in m^-3.
    """
    # The Boerzsoenyi argon fit at its 1 bar, 0 C reference state, as the mode tests
    # supply it (tests/modes/witness_materials.py; solprop has no gas registry since
    # SEP 3's Phase 4b). This script needs its coefficients rather than its values, so
    # that sympy can differentiate the same expression solprop evaluates; the
    # reference and fill densities are solprop's, optmat's equation of state.
    model = PoleSumSusceptibility(
        gas=Gas.Ar,
        source="borzsonyi",
        reference_state=ReferenceState(pressure_bar=1.0, temperature=273.15),
        wavelength_range=(400e-9, 1000e-9),
        citation="A. Börzsönyi et al., Appl. Opt. 47, 4856 (2008)",
        b=tuple(strength for strength, _ in LUNA_POLES),
        c=tuple(resonance for _, resonance in LUNA_POLES),
    )
    return {
        "poles": [
            [strength, resonance]
            for strength, resonance in zip(model.b, model.c, strict=True)
        ],
        "reference_density": model.reference_density,
        "fill_density": density(Gas.Ar, PRESSURE_BAR, TEMPERATURE),
    }


def luna_inputs() -> dict[str, Any]:
    """Return the constants Luna's script hard-codes.

    Returns
    -------
    dict
        The same shape as :func:`solprop_inputs`.
    """
    return {
        "poles": [list(pole) for pole in LUNA_POLES],
        "reference_density": LUNA_REFERENCE_DENSITY,
        "fill_density": LUNA_FILL_DENSITY,
    }


def effective_index(omega: sympy.Symbol, model: str, cladding: complex, inputs: dict):
    """Build the symbolic complex effective index of one case.

    Parameters
    ----------
    omega : sympy.Symbol
        The angular frequency, real.
    model : str
        ``"full"`` or ``"reduced"``.
    cladding : complex
        Constant cladding refractive index.
    inputs : dict
        As returned by :func:`solprop_inputs`.

    Returns
    -------
    sympy.Expr
        ``n_eff(omega)``, complex valued.
    """
    # Wavelength in microns, which is what every published Sellmeier expression uses.
    wavelength = 2e6 * sympy.pi * C / omega
    scale = sympy.Rational(1) * inputs["fill_density"] / inputs["reference_density"]
    susceptibility = scale * sum(
        strength * wavelength**2 / (wavelength**2 - resonance)
        for strength, resonance in inputs["poles"]
    )
    permittivity_clad = sympy.sympify(cladding) ** 2
    # The HE cladding term v_n.
    term = (permittivity_clad + 1) / (2 * sympy.sqrt(permittivity_clad - 1))
    inverse_ka = C / (omega * RADIUS)
    if model == "full":
        return sympy.sqrt(
            1
            + susceptibility
            - (U_NM * inverse_ka) ** 2 * (1 - sympy.I * term * inverse_ka) ** 2
        )
    return (
        1
        + susceptibility / 2
        - (U_NM * inverse_ka) ** 2 / 2
        + sympy.I * (U_NM * inverse_ka) ** 2 * term * inverse_ka
    )


def evaluate(model: str, cladding: complex, inputs: dict) -> dict[str, Any]:
    """Differentiate one case and evaluate it at the reference wavelength.

    Parameters
    ----------
    model : str
        ``"full"`` or ``"reduced"``.
    cladding : complex
        Constant cladding refractive index.
    inputs : dict
        As returned by :func:`solprop_inputs`.

    Returns
    -------
    dict
        ``beta`` (orders 0 to MAX_ORDER, SI), ``alpha`` in 1/m and ``zdw`` in m.
    """
    omega = sympy.Symbol("omega", positive=True)
    index = effective_index(omega, model, cladding, inputs)
    propagation = (omega / C * index).as_real_imag()[0]
    attenuation = 2 * (omega / C * index).as_real_imag()[1]
    point = 2 * sympy.pi * C / WAVELENGTH
    orders = [
        float(sympy.N(sympy.diff(propagation, omega, order).subs(omega, point), 30))
        for order in range(MAX_ORDER + 1)
    ]
    dispersion = sympy.lambdify(omega, sympy.diff(propagation, omega, 2))
    root = scipy.optimize.brentq(dispersion, *ZDW_BRACKET)
    return {
        "beta": orders,
        "alpha": float(sympy.N(attenuation.subs(omega, point), 30)),
        "zdw": float(2 * sympy.pi * C / root),
    }


def main() -> int:
    """Derive every case and write the reference file.

    Returns
    -------
    int
        A process exit code.
    """
    sources = {"solprop": solprop_inputs(), "luna": luna_inputs()}
    cases: dict[str, Any] = {}
    for source, inputs in sources.items():
        for model in ("reduced", "full"):
            for cladding_name, cladding in CLADDINGS.items():
                key = f"{source}-{model}-{cladding_name}"
                sys.stdout.write(f"deriving {key}\n")
                cases[key] = {
                    "source": source,
                    "model": model,
                    "cladding": [cladding.real, cladding.imag],
                    **evaluate(model, cladding, inputs),
                }
    payload = {
        "provenance": {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "generated_by": "tests/data/generate_symbolic_marcatili.py",
            "derived_with": f"sympy {sympy.__version__}",
            "python": sys.version.split()[0],
            "after": (
                "Luna test/symbolic_marcatilli.py on the modal-fixed checkout, "
                "extended to the reduced model and to both losses"
            ),
            "radius_m": RADIUS,
            "u_nm": U_NM,
            "wavelength_m": WAVELENGTH,
            "pressure_bar": PRESSURE_BAR,
            "temperature_k": TEMPERATURE,
            "gas": "Ar",
            "index_source": "borzsonyi",
            "note": (
                "Symbolic derivatives, not finite differences. The 'solprop' cases "
                "share their inputs with the package and so test the calculus; the "
                "'luna' cases use the constants Luna hard-codes and reproduce its "
                "published numbers."
            ),
        },
        "inputs": sources,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(f"wrote {OUTPUT} with {len(cases)} cases\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

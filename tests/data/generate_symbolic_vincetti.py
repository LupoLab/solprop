# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate an independent symbolic calculus witness for the Vincetti fits.

Run with ``uv run --no-sync python tests/data/generate_symbolic_vincetti.py``.
No solprop mode formula is imported. Sympy differentiates the complete empirical
expression with respect to dimensionless frequency and the two material indices.
A Malitson tube and a weak analytic dispersive complex gas exercise every chain
rule, including the tube-index-dependent exponents. Samples include resonance
centres and offsets of the dispersion width, in addition to 200 nm--2 um.

The expression is solprop's form of the fits: dispersion width 0.03, the printed ``3 +
10 lambda/a`` in the effective radius, the normalised frequency ``(2t/lambda) sqrt(n_t^2
- n_g^2)`` wherever it enters (the minimum loss keeps its ``n_t^2 - 1`` factors), the
waveguide decrement divided by the gas index once, and the area with the paper's
``1/n_g`` prefactor. The gas symbol is real in the algebra and complex in the numbers,
exactly as the implementation carries an absorbing species.
"""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.optimize
import sympy as sp

OUTPUT = Path(__file__).with_name("symbolic_vincetti.json")
GEOMETRIES = [
    (15e-6, 180e-9, 9e-6, 6),
    (5e-6, 80e-9, 1e-6, 10),
    ((10e-6 + 2.5e-6) / math.sin(math.pi / 8) - 10e-6, 1e-6, 10e-6, 8),
]
LIGHT_SPEED = 299792458
BESSEL_ZERO = 2.4048255576957728


def symbolic_expressions(geometry):
    """Build the expressions independently of the production arithmetic."""
    x, tube, gas = sp.symbols("x tube gas", positive=True)
    radius, wall, outer, count = geometry
    wavelength = 2 * sp.pi * LIGHT_SPEED / (x * 1e15)
    ratio = wall / outer
    frequency = 2 * wall / wavelength * sp.sqrt(tube**2 - gas**2)
    cutoffs = []
    loss = 0
    dispersion = 0
    for nu in range(1, 9):
        for mu in range(1, 9):
            if nu == 1:
                he = (
                    abs(0.21 + 0.175 * mu - 0.1 / (mu - 0.35) ** 2)
                    * ratio ** (0.55 + 0.005 * sp.sqrt(tube**4 - 1))
                    + 0.04 * sp.sqrt(mu) * ratio
                )
                eh = (
                    0.73 + 0.57 * (mu**0.8 + 1.5) / 4 - 0.04 / (mu - 0.35)
                ) * ratio ** (0.5 - (tube - 1) / (10 * (mu + 0.5) ** 0.1))
            else:
                he = 0.3 / tube**0.3 * (2 / nu) ** 1.2 * abs(mu - 0.8) * ratio + nu - 1
                eh = (
                    (
                        11.5
                        / (nu**1.2 * (7.75 - nu))
                        * (0.34 + mu / 4 * (tube / 1.2) ** 1.15)
                        / (mu + 0.2 / tube) ** 0.15
                        * ratio
                        ** (
                            0.75
                            + 0.06 / tube**1.15
                            + 0.1 * sp.sqrt(1.44 / tube) * (nu - 2)
                        )
                    )
                    + nu
                    - 1
                )
            amplitude = 2000 * sp.exp(-0.05 * abs(mu - 1) ** 2.6)
            for cutoff in (he, eh):
                loss += amplitude * 0.003**2 / (0.003**2 + (frequency - cutoff) ** 2)
                if mu == 1:
                    dispersion += (
                        amplitude
                        * (cutoff**2 - frequency**2)
                        / ((frequency**2 - cutoff**2) ** 2 + (0.03 * frequency) ** 2)
                    )
            if mu == 1:
                cutoffs.append(he)
    minimum = (
        3e-4
        * wavelength**4.5
        / radius**4
        * (1 - ratio) ** -12
        * sp.sqrt(tube**2 - 1)
        / (wall * sp.sqrt(outer))
        * sp.exp(2 * wavelength / (outer * (tube**2 - 1)))
    )
    gap = 2 * ((radius + outer) * sp.sin(sp.pi / count) - outer)
    effective_radius = (1.027 + 0.001 * (frequency + 2 / frequency**4)) * sp.sqrt(
        radius**2
        + count
        / sp.pi
        * 3
        / 64
        * outer**2
        * (1 + (3 + 10 * wavelength / radius) * gap / outer)
    )
    shift = 4.5e-7 * (1 - ratio) ** -4 * (wavelength / radius) ** 2 * dispersion
    decrement = (
        0.5 * (BESSEL_ZERO * LIGHT_SPEED / (x * 1e15 * effective_radius)) ** 2 / gas
    )
    refractivity = (
        gas
        - 1
        - decrement
        + shift
        + sp.I * LIGHT_SPEED * sp.log(10) / (20 * x * 1e15) * minimum * loss
    )
    derivative = [sp.diff(refractivity, variable) for variable in (x, tube, gas)]
    return (
        (x, tube, gas),
        (refractivity, *derivative, decrement - shift),
        frequency,
        cutoffs,
    )


def material_values(x):
    """Evaluate symbolic material models and their exact frequency slopes."""
    variable = sp.Symbol("x", positive=True)
    wavelength_um = 2 * sp.pi * LIGHT_SPEED / (variable * 1e9)
    tube = sp.sqrt(
        1
        + sum(
            b * wavelength_um**2 / (wavelength_um**2 - c**2)
            for b, c in [
                (0.6961663, 0.0684043),
                (0.4079426, 0.1162414),
                (0.8974794, 9.896161),
            ]
        )
    )
    gas = 1 + 0.0003 * (variable / 2) ** 2 + sp.I * 1e-8 * variable / 2
    return sp.lambdify(
        variable, [tube, sp.diff(tube, variable), gas, sp.diff(gas, variable)], "numpy"
    )(x)


def complex_columns(values):
    """Encode complex arrays as real/imaginary pairs."""
    array = np.asarray(values)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def main():
    """Write the symbolic witness and source provenance."""
    cases = []
    for geometry in GEOMETRIES:
        variables, expressions, frequency, cutoffs = symbolic_expressions(geometry)
        evaluate = sp.lambdify(variables, expressions, "numpy", cse=True)
        detunings = sp.lambdify(
            variables, [frequency - cutoff for cutoff in cutoffs], "numpy", cse=True
        )
        wavelengths = np.geomspace(200e-9, 2e-6, 40).tolist()
        resonance_samples = []
        for branch in (1, 2):
            for offset in (-0.03, 0.0, 0.03):

                def residual(
                    wavelength, branch=branch, offset=offset, detunings=detunings
                ):
                    point = 2 * math.pi * LIGHT_SPEED / wavelength / 1e15
                    tube, _, gas, _ = material_values(point)
                    # The filled frequency is complex for the absorbing gas; the
                    # resonance is centred on its real part.
                    return float(np.real(detunings(point, tube, gas)[branch]) - offset)

                if residual(200e-9) * residual(2e-6) < 0:
                    root = scipy.optimize.brentq(residual, 200e-9, 2e-6, xtol=1e-22)
                    wavelengths.append(root)
                    resonance_samples.append(root)
        wavelengths = np.array(sorted(wavelengths))
        x = 2 * math.pi * LIGHT_SPEED / wavelengths / 1e15
        tube, dtube, gas, dgas = material_values(x)
        values, dx, dn, dg, denominator = evaluate(x, tube, gas)
        derivative = (dx + dn * dtube + dg * dgas) / 1e15
        area = (
            0.48
            / (8 * math.pi * np.real(gas))
            * (BESSEL_ZERO * wavelengths) ** 2
            / np.real(denominator)
        )
        cases.append(
            {
                "geometry": geometry,
                "wavelength": wavelengths.tolist(),
                "resonance_wavelength": resonance_samples,
                "tube_index": np.asarray(tube).tolist(),
                "dtube_domega": (np.asarray(dtube) / 1e15).tolist(),
                "gas_index": complex_columns(gas),
                "dgas_domega": complex_columns(dgas / 1e15),
                "refractivity": complex_columns(values),
                "derivative": complex_columns(derivative),
                "effective_area": area.tolist(),
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "generated": datetime.now(UTC).isoformat(),
                    "generator_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "sympy_version": sp.__version__,
                    "light_speed": LIGHT_SPEED,
                    "bessel_zero": BESSEL_ZERO,
                    "conventions": {
                        "dispersion_width": 0.03,
                        "radius_coefficient": 10,
                        "single_power_decrement": True,
                        "area_over_gas_index": True,
                        "filled_frequency": True,
                    },
                    "note": (
                        "Independent symbolic differentiation; shared empirical fits "
                        "are not independently validated physics."
                    ),
                },
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

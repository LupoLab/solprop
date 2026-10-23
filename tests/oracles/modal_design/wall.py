# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare scalar wall condensation with Robin roots and a radial FEM reference.

All radial matrices use x=r/a, with inner product integral x f* g dx.
The FEM uses continuous linear elements and the natural Robin boundary term;
it does not contain a Bessel basis or the condensation formula.
"""

import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import linalg, optimize, special

from modal_design.evidence import run


@dataclass(frozen=True)
class Wall:
    radius: float = 150e-6
    wavelength: float = 6e-6
    core_index: float = 1.0
    clad_index: complex = 1.3
    waist: float = 80e-6

    @property
    def k(self):
        return 2 * np.pi * self.core_index / self.wavelength

    @property
    def eta(self):
        relative = self.clad_index / self.core_index
        nu = (relative**2 + 1) / (2 * np.sqrt(relative**2 - 1 + 0j))
        return self.k * self.radius / nu


def robin_roots(wall, count):
    zeros = special.jn_zeros(0, count)
    roots = []
    for zero in zeros:

        def residual(pair):
            q = pair[0] + 1j * pair[1]
            value = q * special.jv(1, q) + 1j * wall.eta * special.jv(0, q)
            return [value.real, value.imag]

        initial = zero * (1 - 1j / wall.eta)
        result = optimize.root(residual, [initial.real, initial.imag], tol=1e-11)
        value = result.x[0] + 1j * result.x[1]
        scaled = np.linalg.norm(residual(result.x)) / (abs(wall.eta) + abs(value))
        if scaled > 2e-12:
            raise RuntimeError(
                f"Robin root failed: {result.message}; residual={scaled}"
            )
        roots.append(value)
    return np.array(roots)


def condensed(wall, count, *, regularised=True):
    zeros = special.jn_zeros(0, count)
    boundary = -np.sqrt(2) * np.sign(special.jv(1, zeros)) * zeros
    denominator = (2 * count if regularised else 0) - 1j * wall.eta
    return -np.diag(zeros**2) + np.outer(boundary, boundary) / denominator


def radial_fem(wall, elements):
    nodes = np.linspace(0, 1, elements + 1)
    mass = np.zeros((elements + 1, elements + 1))
    stiffness = np.zeros_like(mass)
    points, weights = special.roots_legendre(3)
    shape = np.array([(1 - points) / 2, (1 + points) / 2])
    for index, (left, right) in enumerate(pairwise(nodes)):
        width = right - left
        radius = left + width * (points + 1) / 2
        local_mass = (shape * (weights * radius * width / 2)) @ shape.T
        local_stiffness = (left + right) / (2 * width) * np.array([[1, -1], [-1, 1]])
        mass[index : index + 2, index : index + 2] += local_mass
        stiffness[index : index + 2, index : index + 2] += local_stiffness
    operator = -stiffness.astype(complex)
    operator[-1, -1] += 1j * wall.eta
    return nodes, mass, operator


def gaussian(wall, radius):
    # Normalisation is on the entire incoming plane, not on its truncated bore.
    return (
        2
        * wall.radius
        / wall.waist
        * np.exp(-((wall.radius * radius / wall.waist) ** 2))
    )


def bessel_basis(count, radius):
    zeros = special.jn_zeros(0, count)
    return (
        np.sqrt(2)
        * special.jv(0, radius[:, None] * zeros)
        / np.abs(special.jv(1, zeros))
    )


def modal_launch(wall, count):
    points, weights = special.roots_legendre(max(256, 4 * count))
    radius = (points + 1) / 2
    return bessel_basis(count, radius).T @ (
        weights * radius / 2 * gaussian(wall, radius)
    )


def evolve(values, vectors, initial, wall, distance, *, square_root=False):
    delta = values / wall.radius**2
    beta_shift = (
        delta / (np.sqrt(wall.k**2 + delta) + wall.k)
        if square_root
        else delta / (2 * wall.k)
    )
    coefficients = linalg.solve(vectors, initial)
    field = vectors @ (
        np.exp(1j * beta_shift[:, None] * distance) * coefficients[:, None]
    )
    derivative = vectors @ (
        1j
        * beta_shift[:, None]
        * np.exp(1j * beta_shift[:, None] * distance)
        * coefficients[:, None]
    )
    return field, derivative


def metrics(field, derivative, metric):
    power = np.real(np.sum(field.conj() * (metric @ field), axis=0))
    slope = 2 * np.real(np.sum(field.conj() * (metric @ derivative), axis=0))
    return power, -slope / power


def wall_case(wall, counts, elements):
    roots = robin_roots(wall, 3)
    length = 1 / (-np.imag(roots[0] ** 2) / (wall.k * wall.radius**2))
    distance = np.linspace(0, length, 201)
    references = []
    for resolution in elements:
        start = time.perf_counter()
        radius, mass, operator = radial_fem(wall, resolution)
        values, vectors = linalg.eig(operator, mass)
        field, derivative = evolve(
            values, vectors, gaussian(wall, radius), wall, distance
        )
        power, rate = metrics(field, derivative, mass)
        references.append((radius, mass, field, power, rate))
        exact = -(roots**2)
        matched = np.array(
            [values[np.argmin(abs(values - target))] for target in exact]
        )
        if not np.all(np.isfinite(field)):
            raise RuntimeError("nonfinite FEM propagation")
        print(
            f"FEM {wall.wavelength:g}, {resolution}: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )
    radius, mass, field_ref, power_ref, rate_ref = references[-1]
    rows = []
    for count in counts:
        for model in ("condensed", "unregularised", "diagonal"):
            regularised = model == "condensed"
            operator = condensed(wall, count, regularised=regularised)
            if model == "diagonal":
                zeros = special.jn_zeros(0, count)
                operator = np.diag(-(zeros**2) + 2j * zeros**2 / wall.eta)
            values, vectors = linalg.eig(operator)
            initial = modal_launch(wall, count)
            field, derivative = evolve(values, vectors, initial, wall, distance)
            power, rate = metrics(field, derivative, np.eye(count))
            reconstructed = bessel_basis(count, radius) @ field
            difference = reconstructed - field_ref
            field_error = np.sqrt(
                np.maximum(
                    0, np.real(np.sum(difference.conj() * (mass @ difference), axis=0))
                )
                / power_ref
            )
            matched = np.array(
                [values[np.argmin(abs(values - target))] for target in -(roots**2)]
            )
            matrix = wall.k**2 * np.eye(count) + operator / wall.radius**2
            beta = np.asarray(linalg.sqrtm(matrix), dtype=np.complex128)
            metric = (beta + beta.conj().T) / (2 * wall.k)
            flux_residual = 1j * (metric @ beta - beta.conj().T @ metric) + (
                matrix - matrix.conj().T
            ) / (2j * wall.k)
            rows.append(
                {
                    "count": count,
                    "model": model,
                    "regularised": regularised,
                    "root_relative_error": abs(matched + roots**2) / abs(roots**2),
                    "captured_launch": float(np.vdot(initial, initial).real),
                    "max_power_error": float(np.max(abs(power - power_ref))),
                    "rms_rate_relative_error": float(
                        np.sqrt(np.mean((rate - rate_ref) ** 2)) / np.mean(rate_ref)
                    ),
                    "rms_rate_relative_error_after_1mm": float(
                        np.sqrt(
                            np.mean(
                                (rate[distance >= 1e-3] - rate_ref[distance >= 1e-3])
                                ** 2
                            )
                        )
                        / np.mean(rate_ref[distance >= 1e-3])
                    ),
                    "max_field_relative_error": float(np.max(field_error)),
                    "eigenvector_condition": float(np.linalg.cond(vectors)),
                    "min_flux_eigenvalue": float(linalg.eigvalsh(metric)[0]),
                    "flux_identity_scaled": float(
                        linalg.norm(flux_residual)
                        / max(1, linalg.norm(matrix) / wall.k)
                    ),
                    "power": power,
                    "loss_rate": rate,
                }
            )
    return {
        "parameters": wall,
        "length": length,
        "distance": distance,
        "exact_roots": roots,
        "physical_bore_capture": 1 - np.exp(-2 * (wall.radius / wall.waist) ** 2),
        "fem_levels": elements,
        "fem_power_differences": [
            float(np.max(abs(item[3] - power_ref))) for item in references[:-1]
        ],
        "fem_rate_differences": [
            float(np.sqrt(np.mean((item[4] - rate_ref) ** 2)) / np.mean(rate_ref))
            for item in references[:-1]
        ],
        "reference_power": power_ref,
        "reference_rate": rate_ref,
        "rows": rows,
    }


def study(full):
    """Reproduce the scalar wall campaign without normalising away launch loss."""
    wavelengths = (0.8e-6, 3e-6, 6e-6, 12e-6) if full else (6e-6,)
    indices = (1.3, 1.45) if full else (1.3,)
    return [
        wall_case(
            Wall(wavelength=wavelength, clad_index=index), (10, 20, 40), (128, 256, 512)
        )
        for wavelength in wavelengths
        for index in indices
    ]


if __name__ == "__main__":
    run(study)

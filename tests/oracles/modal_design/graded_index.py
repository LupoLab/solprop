# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare radial variational and layer-matched GRIN eigenproblems.

The TE0/TM0 sectors are full vector Maxwell sectors, not a qualification of
hybrid GRIN families. LP group spacing is a separate weak-guidance witness.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import linalg, optimize, special

from modal_design.evidence import run


@dataclass(frozen=True)
class Grin:
    radius: float = 25e-6
    wavelength: float = 1.03e-6
    centre: float = 1.45
    contrast: float = 0.005
    quartic: float = 0.0

    @property
    def wave_number(self):
        return 2 * np.pi * self.radius / self.wavelength

    def permittivity(self, x):
        shape = (1 - self.quartic) * np.minimum(x, 1) ** 2 + self.quartic * np.minimum(
            x, 1
        ) ** 4
        return self.centre**2 * (1 - 2 * self.contrast * shape)


def variational(profile, count, family, *, extent=2.0, order=1):
    nodes = np.linspace(0, extent, count + 1)
    mass = np.zeros((count + 1, count + 1))
    operator = np.zeros_like(mass)
    roots, weights = special.roots_legendre(8)
    shape = np.array([(1 - roots) / 2, (1 + roots) / 2])
    for panel, (left, right) in enumerate(pairwise(nodes)):
        x = left + (right - left) * (roots + 1) / 2
        measure = weights * (right - left) / 2
        eps = profile.permittivity(x)
        derivative = np.repeat(
            np.array([[-1.0], [1.0]]) / (right - left), x.size, axis=1
        )
        if family in ("TE", "TM"):
            gradient = derivative + shape / x
            inverse = 1 / eps if family == "TM" else np.ones_like(x)
            local_mass = (shape * (measure * x * inverse)) @ shape.T
            local_operator = (
                -(gradient * (measure * x * inverse)) @ gradient.T
                + (shape * (measure * x * inverse * eps * profile.wave_number**2))
                @ shape.T
            )
        else:
            local_mass = (shape * (measure * x)) @ shape.T
            local_operator = (
                -(derivative * (measure * x)) @ derivative.T
                + (
                    shape
                    * (measure * (x * eps * profile.wave_number**2 - order**2 / x))
                )
                @ shape.T
            )
        mass[panel : panel + 2, panel : panel + 2] += local_mass
        operator[panel : panel + 2, panel : panel + 2] += local_operator
    start = 0 if family == "LP" and order == 0 else 1
    values, vectors = linalg.eigh(
        operator[start:-1, start:-1],
        mass[start:-1, start:-1],
        subset_by_index=(count - start - 4, count - start - 1),
    )
    values, vectors = values[::-1], vectors[:, ::-1]
    return np.sqrt(values), vectors, nodes[start:-1], mass[start:-1, start:-1]


def layer_frame(profile, beta, eps, radius, family, *, decaying=False):
    q2 = eps * profile.wave_number**2 - beta**2
    q = np.sqrt(q2 + 0j)
    inverse = 1 / eps if family == "TM" else 1.0
    if decaying:
        decay = np.sqrt(-q2)
        return np.array(
            [
                radius * special.kv(1, decay * radius),
                -inverse * decay * special.kv(0, decay * radius),
            ]
        )
    if q2 < 0:
        decay = np.sqrt(-q2)
        # J and Y at imaginary argument become nearly parallel. I/K keep the
        # independent growing/decaying solutions; column scaling limits overflow.
        return np.array(
            [
                [
                    radius * special.iv(1, decay * radius),
                    radius * special.kv(1, decay * radius),
                ],
                [
                    inverse * decay * special.iv(0, decay * radius),
                    -inverse * decay * special.kv(0, decay * radius),
                ],
            ]
        )
    return np.array(
        [
            [radius * special.jv(1, q * radius), radius * special.yv(1, q * radius)],
            [
                inverse * q * special.jv(0, q * radius),
                inverse * q * special.yv(0, q * radius),
            ],
        ]
    )


def layer_residual(profile, beta, layers, family):
    edges = np.linspace(0, 1, layers + 1)
    eps = profile.permittivity((edges[:-1] + edges[1:]) / 2)
    field = layer_frame(profile, beta, eps[0], edges[1], family)[:, 0]
    for index in range(1, layers):
        previous = layer_frame(profile, beta, eps[index], edges[index], family)
        following = layer_frame(profile, beta, eps[index], edges[index + 1], family)
        scale = np.linalg.norm(previous, axis=0)
        previous, following = previous / scale, following / scale
        field = following @ linalg.solve(previous, field)
        field /= linalg.norm(field)
    exterior = layer_frame(
        profile, beta, profile.permittivity(1.0), 1.0, family, decaying=True
    )
    # J/Y columns share a constant real or imaginary phase within each region.
    result = field[0] * exterior[1] - field[1] * exterior[0]
    return float(result.real)


def layer_root(profile, seed, layers, family):
    lower = profile.wave_number * np.sqrt(profile.permittivity(1.0))
    upper = profile.wave_number * profile.centre
    samples = np.linspace(
        max(lower + 1e-8, seed - 0.15), min(upper - 1e-8, seed + 0.15), 61
    )
    values = [layer_residual(profile, beta, layers, family) for beta in samples]
    intervals = [
        (left, right)
        for left, right, a, b in zip(
            samples[:-1], samples[1:], values[:-1], values[1:], strict=True
        )
        if a * b < 0
    ]
    if not intervals:
        raise RuntimeError(f"no layered root near {seed}")
    left, right = min(intervals, key=lambda pair: abs(sum(pair) / 2 - seed))
    return optimize.brentq(
        lambda beta: layer_residual(profile, beta, layers, family),
        left,
        right,
        xtol=5e-13,
    )


def layer_profile(profile, beta, layers, family, points):
    """Reconstruct matched layers without borrowing the variational eigenvector."""
    edges = np.linspace(0, 1, layers + 1)
    eps = profile.permittivity((edges[:-1] + edges[1:]) / 2)
    result = np.zeros(points.size, complex)
    first = points <= edges[1]
    q = np.sqrt(eps[0] * profile.wave_number**2 - beta**2)
    result[first] = special.jv(1, q * points[first])
    field = layer_frame(profile, beta, eps[0], edges[1], family)[:, 0]
    for index in range(1, layers):
        frame = layer_frame(profile, beta, eps[index], edges[index], family)
        scale = np.linalg.norm(frame, axis=0)
        coefficients = linalg.solve(frame / scale, field) / scale
        selected = (points > edges[index]) & (points <= edges[index + 1])
        for position in np.flatnonzero(selected):
            result[position] = (
                layer_frame(profile, beta, eps[index], points[position], family)
                @ coefficients
            )[0] / points[position]
        field = (
            layer_frame(profile, beta, eps[index], edges[index + 1], family)
            @ coefficients
        )
    outer = points > 1
    decay = np.sqrt(beta**2 - profile.permittivity(1.0) * profile.wave_number**2)
    result[outer] = (
        field[0] * special.kv(1, decay * points[outer]) / special.kv(1, decay)
    )
    return result


def study(full):
    """Refine finite-core parabolic/non-parabolic TE/TM and LP mode groups."""
    levels = (96, 192, 384) if full else (64, 128, 256)
    rows = []
    for quartic in (0.0, 0.25):
        profile = Grin(quartic=quartic)
        for family in ("TE", "TM"):
            for count in levels:
                values, vectors, points, mass = variational(profile, count, family)
                layered = layer_root(profile, values[0], count, family)
                reference = layer_profile(profile, layered, count, family, points)
                reference /= np.sqrt(np.vdot(reference, mass @ reference).real)
                overlap = abs(np.vdot(vectors[:, 0], mass @ reference))
                rows.append(
                    {
                        "quartic": quartic,
                        "family": family,
                        "resolution": count,
                        "fem_beta_radius": values,
                        "layer_beta_radius": layered,
                        "relative_difference": abs(values[0] - layered) / layered,
                        "profile_overlap_defect": abs(1 - overlap),
                    }
                )
        for order in (0, 1, 2):
            values, _, _, _ = variational(profile, levels[-1], "LP", order=order)
            oscillator = (
                2
                * profile.wave_number
                * profile.centre
                * np.sqrt(2 * profile.contrast)
                * (2 * np.arange(4) + order + 1)
            )
            ideal = np.sqrt((profile.wave_number * profile.centre) ** 2 - oscillator)
            rows.append(
                {
                    "quartic": quartic,
                    "family": "LP",
                    "order": order,
                    "beta_radius": values,
                    "ideal_parabolic_beta_radius": ideal,
                    "finite_core_difference": values - ideal,
                }
            )
        if full:
            for extent in (1.5, 2.0, 3.0):
                values, _, _, _ = variational(
                    profile, int(192 * extent), "TE", extent=extent
                )
                rows.append(
                    {
                        "quartic": quartic,
                        "family": "TE_domain",
                        "extent": extent,
                        "beta_radius": values,
                    }
                )
    return rows


if __name__ == "__main__":
    run(study)

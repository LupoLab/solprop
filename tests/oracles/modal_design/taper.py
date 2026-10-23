# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare local Bessel coupling with a mapped-coordinate radial FEM evolution."""

from dataclasses import dataclass, replace

import numpy as np
from scipy import integrate, sparse, special
from scipy.sparse.linalg import spsolve

from modal_design.evidence import run
from modal_design.wall import Wall, bessel_basis, condensed, radial_fem


@dataclass(frozen=True)
class Taper:
    length: float = 0.01
    smooth: bool = False
    loss: bool = False
    initial_radius: float = 125e-6
    final_radius: float = 62.5e-6
    straight: float = 0.005
    # The taper note's carrier. A field of the case rather than a literal so that the
    # production study can ask the same reference at a shorter wavelength; the default
    # keeps every recorded receipt's case exactly as it was.
    wavelength: float = 800e-9

    @property
    def events(self):
        return (
            0,
            self.straight,
            self.straight + self.length,
            2 * self.straight + self.length,
        )

    def geometry(self, z):
        fraction = np.clip((z - self.straight) / self.length, 0, 1)
        active = self.straight < z < self.straight + self.length
        shape = 3 * fraction**2 - 2 * fraction**3 if self.smooth else fraction
        slope = (
            (6 * fraction - 6 * fraction**2 if self.smooth else 1) / self.length
            if active
            else 0.0
        )
        difference = self.final_radius - self.initial_radius
        return self.initial_radius + difference * shape, difference * slope


def coupling(count):
    zeros = special.jn_zeros(0, count)
    signed = zeros * np.sign(special.jv(1, zeros))
    denominator = zeros[:, None] ** 2 - zeros[None, :] ** 2
    np.fill_diagonal(denominator, 1)
    connection = 2 * np.outer(signed, signed) / denominator
    np.fill_diagonal(connection, 0)
    return connection


def modal(taper, count, tolerance=1e-10):
    wall = Wall(
        radius=taper.initial_radius, wavelength=taper.wavelength, clad_index=1.45
    )
    connection = coupling(count)
    diagonal = -np.diag(special.jn_zeros(0, count) ** 2)
    field = np.eye(count, dtype=complex)[:, 0]
    evaluations = 0

    def rhs(z, state):
        radius, derivative = taper.geometry(z)
        laplacian = (
            condensed(replace(wall, radius=radius), count) if taper.loss else diagonal
        )
        return (
            1j / (2 * wall.k * radius**2) * laplacian - derivative / radius * connection
        ) @ state

    for left, right in zip(taper.events[:-1], taper.events[1:], strict=True):
        solution = integrate.solve_ivp(
            rhs,
            (left, right),
            field,
            method="DOP853",
            rtol=tolerance,
            atol=tolerance / 100,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        field = solution.y[:, -1]
        evaluations += solution.nfev
    return field, evaluations


def dilation(elements):
    result = np.zeros((elements + 1, elements + 1))
    points, weights = special.roots_legendre(3)
    shape = np.array([(1 - points) / 2, (1 + points) / 2])
    derivative = np.repeat(np.array([[-elements], [elements]]), 3, axis=1)
    for index in range(elements):
        radius = (index + (points + 1) / 2) / elements
        local = (shape * (weights * radius / (2 * elements))) @ (
            shape + radius * derivative
        ).T
        result[index : index + 2, index : index + 2] += local
    return result


def spatial(taper, elements, steps, count=20):
    wall = Wall(
        radius=taper.initial_radius, wavelength=taper.wavelength, clad_index=1.45
    )
    radius, mass, operator = radial_fem(wall, elements)
    laplacian = operator.real
    advective = dilation(elements)
    wall_term = np.zeros_like(operator)
    wall_term[-1, -1] = 1.0
    if not taper.loss:
        radius = radius[:-1]
        mass, laplacian, advective, wall_term = [
            matrix[:-1, :-1] for matrix in (mass, laplacian, advective, wall_term)
        ]
    mass, laplacian, advective, wall_term = map(
        sparse.csc_matrix, (mass, laplacian, advective, wall_term)
    )
    field = bessel_basis(1, radius)[:, 0].astype(complex)
    launch = float(np.vdot(field, mass @ field).real)
    total = taper.events[-1]
    for left, right in zip(taper.events[:-1], taper.events[1:], strict=True):
        count_steps = int(np.ceil(steps * (right - left) / total))
        step = (right - left) / count_steps
        for index in range(count_steps):
            position = left + (index + 0.5) * step
            physical_radius, slope = taper.geometry(position)
            eta = wall.eta * physical_radius / wall.radius
            local = (
                1j
                / (2 * wall.k * physical_radius**2)
                * (laplacian + 1j * eta * wall_term)
                + slope / physical_radius * advective
            )
            field = spsolve(mass - step / 2 * local, (mass + step / 2 * local) @ field)
    projected = bessel_basis(count, radius).T @ (mass @ field)
    return {
        "elements": elements,
        "steps": steps,
        "launch_power": launch,
        "power": float(np.vdot(field, mass @ field).real),
        "fractions": abs(projected) ** 2,
    }


def study(full):
    """Refine smooth/kinked, lossless/Robin taper conversion independently."""
    rows = []
    cases = [
        Taper(length=length, smooth=smooth, loss=loss)
        for length in ((0.01, 0.1) if full else (0.01,))
        for smooth in ((False, True) if full else (False,))
        for loss in (False, True)
    ]
    for taper in cases:
        modes = []
        for count in (8, 12, 20):
            field, nfev = modal(taper, count)
            modes.append(
                {
                    "count": count,
                    "power": float(np.vdot(field, field).real),
                    "fractions": abs(field) ** 2,
                    "nfev": nfev,
                }
            )
        spatial_rows = [
            spatial(taper, elements, steps)
            for elements, steps in (
                (128, 2048),
                (256, 2048),
                (512, 2048),
                (512, 4096),
                (512, 8192),
            )
        ]
        rows.append({"taper": taper, "modal": modes, "spatial": spatial_rows})
    return rows


if __name__ == "__main__":
    run(study)

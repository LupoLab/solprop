# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Construct full step-index fields from longitudinal Maxwell boundary data.

Only propagation constants are taken from the existing solver. Transverse fields
are recovered from the curl equations, independently of its Table 12-3 profiles.
H is scaled by the vacuum impedance during the boundary solve for conditioning.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, special

from modal_design.evidence import run
from solprop.constants import EPS0, C
from solprop.modes import StepIndexMode

IMPEDANCE = 1 / (EPS0 * C)


@dataclass(frozen=True)
class VectorMode:
    radius: float
    wavelength: float
    core: float
    clad: float
    beta: float
    order: int
    coefficients: np.ndarray
    boundary_residual: float


def radial_data(mode, radius, inside):
    k = 2 * np.pi / mode.wavelength
    index = mode.core if inside else mode.clad
    transverse = k**2 * index**2 - mode.beta**2
    scale = np.sqrt(abs(transverse))
    function = special.jv if inside else special.kv
    derivative = special.jvp if inside else special.kvp
    normalisation = function(mode.order, scale * mode.radius)
    return (
        transverse,
        function(mode.order, scale * radius) / normalisation,
        scale * derivative(mode.order, scale * radius) / normalisation,
    )


def fields(mode, radius, angle, *, inside, parity="helical"):
    transverse, radial, slope = radial_data(mode, radius, inside)
    index = mode.core if inside else mode.clad
    k = 2 * np.pi / mode.wavelength
    electric_z, magnetic_z = mode.coefficients[:2] if inside else mode.coefficients[2:]
    order = mode.order
    if parity == "helical" or order == 0:
        angular_e = angular_h = np.exp(1j * order * angle)
        angular_de = angular_dh = 1j * order * angular_e
    elif parity == "even":
        angular_e, angular_h = np.cos(order * angle), 1j * np.sin(order * angle)
        angular_de, angular_dh = (
            -order * np.sin(order * angle),
            1j * order * np.cos(order * angle),
        )
    else:
        angular_e, angular_h = np.sin(order * angle), -1j * np.cos(order * angle)
        angular_de, angular_dh = (
            order * np.cos(order * angle),
            1j * order * np.sin(order * angle),
        )
    ez = electric_z * radial * angular_e
    hz = magnetic_z * radial * angular_h
    dez = electric_z * slope * angular_e
    dhz = magnetic_z * slope * angular_h
    # Gauss nodes avoid r=0; the limiting axis is checked by radial refinement.
    theta_e = electric_z * radial * angular_de / radius
    theta_h = magnetic_z * radial * angular_dh / radius
    er = 1j / transverse * (mode.beta * dez + k * theta_h)
    ep = 1j / transverse * (mode.beta * theta_e - k * dhz)
    hr = 1j / transverse * (mode.beta * dhz - k * index**2 * theta_e)
    hp = 1j / transverse * (mode.beta * theta_h + k * index**2 * dez)
    cosine, sine = np.cos(angle), np.sin(angle)
    electric = np.array([er * cosine - ep * sine, er * sine + ep * cosine, ez]).T
    magnetic = (
        np.array([hr * cosine - hp * sine, hr * sine + hp * cosine, hz]).T / IMPEDANCE
    )
    return electric, magnetic


def solve_boundary(source):
    wavelength, radius = source.profile_wavelength, float(source.radius)
    k = 2 * np.pi / wavelength
    beta = k * float(np.real(source.neff(np.array([C * k]), 0.0)[0]))
    shell = VectorMode(
        radius,
        wavelength,
        float(source.core),
        float(source.cladding),
        beta,
        source.n,
        np.ones(4),
        0.0,
    )
    columns = []
    for column in range(4):
        coefficients = np.eye(4)[column]
        temporary = VectorMode(
            radius,
            wavelength,
            shell.core,
            shell.clad,
            beta,
            source.n,
            coefficients,
            0.0,
        )
        inner_e, inner_h = fields(
            temporary, np.array([radius]), np.zeros(1), inside=True
        )
        outer_e, outer_h = fields(
            temporary, np.array([radius]), np.zeros(1), inside=False
        )
        columns.append(
            [
                inner_e[0, 2] - outer_e[0, 2],
                IMPEDANCE * (inner_h[0, 2] - outer_h[0, 2]),
                inner_e[0, 1] - outer_e[0, 1],
                IMPEDANCE * (inner_h[0, 1] - outer_h[0, 1]),
            ]
        )
    boundary = np.array(columns).T
    boundary /= np.maximum(
        np.linalg.norm(boundary, axis=1, keepdims=True), np.finfo(float).tiny
    )
    _, singular, adjoint = linalg.svd(boundary)
    coefficients = adjoint[-1].conj()
    return VectorMode(
        radius,
        wavelength,
        shell.core,
        shell.clad,
        beta,
        source.n,
        coefficients,
        float(singular[-1] / singular[0]),
    )


def quadrature(radius, extent, radial_order=96, angles=24):
    roots, weights = special.roots_legendre(radial_order)
    radii, radial_weights = [], []
    for left, right in ((0, radius), (radius, extent)):
        r = left + (right - left) * (roots + 1) / 2
        radii.extend(r)
        radial_weights.extend(weights * (right - left) / 2 * r)
    angle = np.arange(angles) * 2 * np.pi / angles
    return (
        np.repeat(radii, angles),
        np.tile(angle, 2 * radial_order),
        np.repeat(radial_weights, angles) * 2 * np.pi / angles,
    )


def sampled(mode, radius, angle, parity="even"):
    electric = np.empty((radius.size, 3), complex)
    magnetic = np.empty_like(electric)
    for inside in (True, False):
        selected = (radius < mode.radius) == inside
        electric[selected], magnetic[selected] = fields(
            mode, radius[selected], angle[selected], inside=inside, parity=parity
        )
    return electric, magnetic


def power(electric, magnetic, weights):
    return (
        np.sum(
            weights
            * (
                electric[:, 0] * magnetic[:, 1].conj()
                - electric[:, 1] * magnetic[:, 0].conj()
            )
        )
        / 2
    )


def vector_case(core, clad, radius, wavelength, *, radial_order=96, descriptors=None):
    descriptors = descriptors or [
        ("HE", 1, 1),
        ("HE", 1, 2),
        ("HE", 2, 1),
        ("EH", 1, 1),
        ("TE", 0, 1),
        ("TM", 0, 1),
    ]
    modes = []
    for kind, order, radial in descriptors:
        source = StepIndexMode(
            radius,
            core=core,
            cladding=clad,
            profile_wavelength=wavelength,
            kind=kind,
            n=order,
            m=radial,
        )
        mode = solve_boundary(source)
        for parity in ("even", "odd") if order else ("even",):
            modes.append((f"{kind}{order}{radial}-{parity}", mode, parity))
    k = 2 * np.pi / wavelength
    slowest = min(np.sqrt(mode.beta**2 - k**2 * clad**2) for _, mode, _ in modes)
    extent = radius + 24 / slowest
    points, angles, weights = quadrature(radius, extent, radial_order)
    electric, magnetic, rows = [], [], []
    for label, mode, parity in modes:
        e, h = sampled(mode, points, angles, parity)
        norm = float(power(e, h, weights).real)
        if norm <= 0:
            raise RuntimeError(f"nonpositive flux {label}: {norm}")
        e, h = e / np.sqrt(norm), h / np.sqrt(norm)
        electric.append(e)
        magnetic.append(h)
        index = np.where(points < radius, core, clad)
        n2 = np.where(points < radius, 2.6e-20, 0)
        chi3 = 4 / 3 * index**2 * EPS0 * C * n2
        intensity = np.sum(abs(e) ** 2, axis=1)
        bilinear = np.sum(e * e, axis=1)
        nonlinear = (
            EPS0
            * chi3[:, None]
            / 4
            * (2 * intensity[:, None] * e + bilinear[:, None] * e.conj())
        )
        gamma = (C * k / 4 * np.sum(weights[:, None] * e.conj() * nonlinear)).real
        scalar_area = np.sum(weights * intensity) ** 2 / np.sum(weights * intensity**2)
        # Boundary tests include normal displacement/induction, not only the solve rows.
        theta = np.arange(24) * 2 * np.pi / 24
        boundary_r = np.full(theta.size, radius)
        ei, hi = fields(mode, boundary_r, theta, inside=True, parity=parity)
        eo, ho = fields(mode, boundary_r, theta, inside=False, parity=parity)
        radial_vector = np.stack(
            (np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=1
        )
        d_residual = np.max(
            abs(np.sum((core**2 * ei - clad**2 * eo) * radial_vector, axis=1))
        ) / max(np.max(abs(core**2 * ei)), np.finfo(float).tiny)
        b_residual = np.max(abs(np.sum((hi - ho) * radial_vector, axis=1))) / max(
            np.max(abs(hi)), np.finfo(float).tiny
        )
        rows.append(
            {
                "label": label,
                "beta": mode.beta,
                "boundary": mode.boundary_residual,
                "normal_d": d_residual,
                "normal_b": b_residual,
                "gamma": gamma,
                "scalar_gamma": k * 2.6e-20 / scalar_area,
                "longitudinal_fraction": np.sum(weights * abs(e[:, 2]) ** 2)
                / np.sum(weights * intensity),
            }
        )
    overlap = np.array([[power(e, h, weights) for h in magnetic] for e in electric])
    gram = (overlap + overlap.conj().T) / 2
    return {
        "core": core,
        "clad": clad,
        "radius": radius,
        "wavelength": wavelength,
        "extent": extent,
        "radial_order": radial_order,
        "rows": rows,
        "flux_gram_error": float(np.max(abs(gram - np.eye(len(rows))))),
        "gram": gram,
    }


def study(full):
    """Check full vector boundary fields, parity, power and Kerr coefficients."""
    parameters = [(1.45, 1.0, 1.5e-6, 835e-9), (1.45, 1.449, 25e-6, 835e-9)]
    levels = (48, 96, 192) if full else (96,)
    rows = [
        vector_case(*case, radial_order=order)
        for case in parameters
        for order in levels
    ]
    if full:
        rows.extend(
            vector_case(
                1.45,
                1.0,
                250e-9,
                835e-9,
                radial_order=order,
                descriptors=[("HE", 1, 1)],
            )
            for order in levels
        )
    return rows


if __name__ == "__main__":
    run(study)

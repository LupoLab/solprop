# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound same-field spatial observations and certify each width's conditioning.

Transform envelopes precede cancellation; no propagated-field error enters these
comparisons. The width proof uses the represented sample perturbation, positive energy,
centroid movement, interpolation slopes and stable threshold membership. Optional
outer rings may enter the radius set only when their perturbed values stay below the
peak of the guaranteed prefix. It is the same conditional derivation independently
reviewed with the standalone spatial observation leaf.
"""

import math

import numpy as np

from bounds import gamma
from solprop import fourier
from solprop.constants import EPS0, C
from solprop.spatial import _fluence_rings
from solprop.translate.spatial import axial_admittance


def _cartesian_profile(axes, fluence, center, perturbation):
    """Bound bilinear rings under the certified centroid displacement."""
    x, y = axes
    dc, error, radius = perturbation
    eps = np.finfo(float).eps
    samples = fluence.reshape(len(y), len(x))
    rings = _fluence_rings(x, y, *center)
    assert rings is not None
    guaranteed_limit = (
        min(
            center[0] - x[0] - dc[0],
            x[-1] - center[0] - dc[0],
            center[1] - y[0] - dc[1],
            y[-1] - center[1] - dc[1],
        )
        - 512 * eps * radius
    )
    assert guaranteed_limit > 2 * rings.radii[1]
    limit = min(
        center[0] - x[0], x[-1] - center[0], center[1] - y[0], y[-1] - center[1]
    )
    spacing = rings.radii[1]
    positions = spacing * np.arange(
        math.floor((limit + np.max(dc) + 512 * eps * radius) / spacing) + 1
    )
    profile = np.empty(len(positions))
    for index, distance in enumerate(positions):
        h = np.clip(
            (center[0] + distance * rings.cosines - x[0]) / (x[1] - x[0]),
            0,
            len(x) - 1,
        )
        v = np.clip(
            (center[1] + distance * rings.sines - y[0]) / (y[1] - y[0]),
            0,
            len(y) - 1,
        )
        ix = np.minimum(h.astype(int), len(x) - 2)
        iy = np.minimum(v.astype(int), len(y) - 2)
        fx, fy = h - ix, v - iy
        profile[index] = np.mean(
            (1 - fy) * ((1 - fx) * samples[iy, ix] + fx * samples[iy, ix + 1])
            + fy * ((1 - fx) * samples[iy + 1, ix] + fx * samples[iy + 1, ix + 1])
        )
    lips = np.array(
        [
            np.max(np.abs(np.diff(samples, axis=1))) / (x[1] - x[0]),
            np.max(np.abs(np.diff(samples, axis=0))) / (y[1] - y[0]),
        ]
    )
    profile_error = (
        error + float(lips @ (dc + 512 * eps * radius)) + 512 * eps * np.max(fluence)
    )
    return positions, profile, profile_error, guaranteed_limit


def _threshold_bounds(positions, profile, profile_error, guaranteed_limit, radius):
    """Certify threshold membership and propagate interpolation sensitivity."""
    eps = np.finfo(float).eps
    bounds = []
    for fraction, scale in ((math.exp(-2), 1.0), (0.5, 2.0)):
        peak = float(np.max(profile))
        level = fraction * peak
        # Peak is 1-Lipschitz; both a node and its threshold can change.
        margin = float(np.min(np.abs(profile - level)))
        assert margin > (1 + fraction) * profile_error
        assert profile[0] > level + (1 + fraction) * profile_error
        index = int(np.flatnonzero(profile < level)[0])
        # Even if a last radius appears/disappears at floor(limit/spacing),
        # the actual crossing and all preceding bins exist in both evaluations.
        assert positions[index] < guaranteed_limit
        slope = float(profile[index - 1] - profile[index])
        assert slope > 4 * profile_error
        bound = (
            scale
            * (positions[index] - positions[index - 1])
            * (3 + fraction)
            * profile_error
            / (slope - 2 * profile_error)
        )
        bounds.append(bound + 512 * eps * radius)
    return bounds


def _width_bounds(route, geometry, fluence, axis_fluence, error):
    """Certify positive mass and unchanged width crossings before comparison."""
    # Inputs are the same represented fluence samples used by the reference.
    # |delta F_i| and |delta F_axis| <= error cover both complete evaluations.
    # All margins below are checked before comparing Julia and NumPy widths.
    fluence = np.asarray(fluence)
    error = float(error)
    eps = np.finfo(float).eps
    x = geometry.axes[0]
    y = np.array([0.0]) if route == "radial" else geometry.axes[1]
    coords = np.column_stack((np.tile(x, len(y)), np.repeat(y, len(x))))
    if route == "radial":
        coords = np.column_stack((x, np.zeros(len(x))))
    weights = geometry.spatial_weights if route == "radial" else np.ones(fluence.size)
    total = float(np.sum(weights * fluence))
    total_error = float(np.sum(weights)) * error
    assert error > 0 and 2 * total_error < total
    radius = max(np.max(np.linalg.norm(coords, axis=1)), 1e-10)
    if route == "radial":
        center = np.zeros(2)
        dc = np.zeros(2)
    else:
        center = np.sum(coords * fluence[:, None], axis=0) / total
        dc = np.sum(np.abs(coords - center) * error, axis=0) / (total - total_error)
        dc += 512 * eps * radius
    r2 = np.sum((coords - center) ** 2, axis=1)
    m2 = float(np.sum(weights * fluence * r2) / total)
    dm2 = float(
        np.sum(weights * np.abs(r2 - m2)) * error / (total - total_error)
        + np.dot(dc, dc)
    )
    dm2 += 512 * eps * radius**2
    multiplier = 4 if route == "line" else 2
    assert m2 > 2 * dm2
    moment_bound = (
        multiplier
        * dm2
        / (math.sqrt(multiplier * (m2 - dm2)) + math.sqrt(multiplier * m2))
    )
    moment_bound += 512 * eps * radius
    if route == "radial":
        positions = np.r_[0.0, x]
        profile = np.r_[axis_fluence, fluence]
        profile_error = error + 512 * eps * max(np.max(fluence), axis_fluence)
        guaranteed_limit = np.inf
    elif route == "line":
        spacing = (x[1] - x[0]) / 2
        limit = min(center[0] - x[0], x[-1] - center[0])
        guaranteed_limit = limit - dc[0] - 512 * eps * radius
        assert guaranteed_limit > 2 * spacing
        positions = spacing * np.arange(
            math.floor((limit + dc[0] + 512 * eps * radius) / spacing) + 1
        )
        profile = 0.5 * (
            np.interp(center[0] - positions, x, fluence)
            + np.interp(center[0] + positions, x, fluence)
        )
        lip = np.max(np.abs(np.diff(fluence))) / (x[1] - x[0])
        profile_error = (
            error + lip * (dc[0] + 512 * eps * radius) + 512 * eps * np.max(fluence)
        )
    else:
        positions, profile, profile_error, guaranteed_limit = _cartesian_profile(
            (x, y), fluence, center, (dc, error, radius)
        )
    if route != "radial":
        # The radius-count floor may add/drop an outer ring. Evaluate through
        # the largest possible count, then certify every optional outer ring
        # is below the guaranteed prefix peak by a positive perturbation gap.
        guaranteed = positions < guaranteed_limit
        prefix_peak = float(np.max(profile[guaranteed]))
        optional = profile[~guaranteed]
        assert optional.size == 0 or np.max(optional) < prefix_peak - 2 * profile_error
    return [
        moment_bound,
        *_threshold_bounds(positions, profile, profile_error, guaranteed_limit, radius),
    ]


def spatial_observation_bounds(problem, grid, field, z):
    """Return separate SI-unit budgets for a stable scalar vacuum spatial fixture."""
    route, geometry = problem.route, problem.transform.geometry
    assert geometry is not None
    coefficients = route.flux.at(z)
    active = coefficients.active
    # The complex square root is uniformly conditioned on these vacuum fixtures.
    assert np.all(coefficients.k_squared.imag == 0)
    assert np.max(coefficients.kperp_sq) < 0.5 * np.min(
        coefficients.k_squared.real[active]
    )
    admittance = axial_admittance(coefficients, grid.omega, slice(None))
    magnitudes = abs(field[:, 0, :]).T
    magnetic = abs(admittance.T) * magnitudes
    n_frequency, columns = magnitudes.shape
    if route.kind == "radial":
        synthesis = problem.transform.synthesis / grid.pad_scale
        electric_envelope = magnitudes @ abs(synthesis.T)
        magnetic_envelope = magnetic @ abs(synthesis.T)
        electric = synthesis @ field[:, 0, :]
        closure = synthesis @ (np.conj(admittance) * field[:, 0, :])
    else:
        electric_envelope = np.repeat(
            magnitudes.sum(axis=1)[:, None] / columns, columns, axis=1
        )
        magnetic_envelope = np.repeat(
            magnetic.sum(axis=1)[:, None] / columns, columns, axis=1
        )
        shape = (*reversed(geometry.shape), n_frequency)
        electric = np.fft.ifft2(field[:, 0, :].reshape(shape), axes=(0, 1)).reshape(
            columns, n_frequency
        )
        closure = np.fft.ifft2(
            (np.conj(admittance) * field[:, 0, :]).reshape(shape), axes=(0, 1)
        ).reshape(columns, n_frequency)
    axis_e = magnitudes @ abs(geometry.on_axis)
    axis_h = magnetic @ abs(geometry.on_axis)
    factor = 0.5 * EPS0 * C
    parseval = fourier.energy_weights(n_frequency, grid.dt, kind=grid.kind)
    density = fourier.spectral_density(
        np.ones(n_frequency, complex), grid.dt, kind=grid.kind
    )
    temporal = np.ones(n_frequency)
    if grid.kind == "field":
        temporal[1:-1] = 2
    # Conditional on the qualified componentwise gamma(32*n) FFT model used by
    # the modal observation witnesses, against absolute input sums, 256 times
    # the total dimensions covers both full electric/H pipelines,
    # independent root/conjugate products, axis maps and subsequent reductions.
    # Maxima are 1-Lipschitz. This is deliberately independent of observed error.
    allowance = gamma(256 * (grid.n_t + n_frequency + columns))
    flux_envelope = factor * np.sum(
        parseval[:, None] * electric_envelope * magnetic_envelope, axis=0
    )
    energy_envelope = factor * np.sum(
        geometry.spectral_weights[:, None]
        * parseval
        * abs(admittance)
        * abs(field[:, 0, :]) ** 2
    )
    peak_envelope = (
        factor
        * (temporal @ electric_envelope)
        * (temporal @ magnetic_envelope)
        / grid.n_t**2
    )
    axis_peak = factor * (temporal @ axis_e) * (temporal @ axis_h) / grid.n_t**2
    bounds = {
        "energy": allowance * energy_envelope,
        "peak_intensity": allowance * np.max(peak_envelope),
        "on_axis_intensity": allowance * axis_peak,
        "on_axis_spectrum": allowance * factor * density * axis_e * axis_h,
    }
    fluence = factor * np.sum(np.real(electric * np.conj(closure)) * parseval, axis=1)
    axis_electric = geometry.on_axis @ field[:, 0, :]
    axis_magnetic = geometry.on_axis @ (np.conj(admittance) * field[:, 0, :])
    axis_fluence = (
        factor
        * np.sum(np.real(axis_electric * np.conj(axis_magnetic)) * density)
        * grid.d_omega
    )
    # Both the physical-node and independent axis fluences must fit the common
    # perturbation used in the width inequalities; their units are J/m^2.
    axis_envelope = factor * np.sum(density * axis_e * axis_h) * grid.d_omega
    error = allowance * max(float(np.max(flux_envelope)), float(axis_envelope))
    assert np.all(fluence >= 0) and axis_fluence >= 0
    widths = _width_bounds(route.kind, geometry, fluence, axis_fluence, error)
    bounds["beam_radius"] = np.asarray(widths[:2])
    bounds["fwhm_r"] = widths[2]
    return bounds

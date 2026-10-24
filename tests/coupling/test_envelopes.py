# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The oracles behind the recorded coupling envelopes (SEP 31, section 4.3).

Each recorded :class:`~solprop.coupling.envelopes.PairEnvelope` is recomputed here
from an independent reference and held to its bound:

- the capillary envelope: the scalar model's captured shares (``couple`` on sampled
  planes, through the closed-form Hankel transforms) against the overlap of the
  capillary's own :meth:`~solprop.modes.MarcatiliMode.field` with its
  :meth:`~solprop.modes.MarcatiliMode.normalisation`, both transverse components,
  on an independent polar Gauss--Legendre quadrature;
- the step-index envelope: the weak-guidance LP01 overlap the scalar model uses
  against the full-vector flux overlap of the exact bound HE11 fields
  (:meth:`~solprop.modes.StepIndexMode.electromagnetic_field`).

The numerical error of each reference is estimated by refining its quadrature.
"""

import math
from typing import cast

import numpy as np
import pytest

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.coupling import (
    CAPILLARY_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    CouplingModel,
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    ScalarModeBasis,
    couple,
)
from solprop.coupling.models import radial_rule
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.modes import MarcatiliMode, StepIndexMode
from solprop.spatial import SpatialMedium

RADIUS = 75e-6
PRESSURE = 3.0
WAVELENGTH = 800e-9


def polar_rule(outer, breaks, n_theta=32, panel=None):
    """Return an independent polar rule: Gauss--Legendre in r, uniform in theta."""
    panel = outer / 16 if panel is None else panel
    radii, area = radial_rule(breaks, outer, panel=panel)
    theta = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    r = np.repeat(radii, n_theta)
    t = np.tile(theta, radii.size)
    w = np.repeat(area, n_theta) / n_theta
    return r, t, w


def capillary_simulation(radius, modes, grid_kwargs):
    """Return a short argon capillary simulation on the tests' envelope grid."""
    return Simulation.from_capillary(
        radius,
        1e-4,
        "Ar",
        PRESSURE,
        modes=modes,
        n_save=2,
        noise=None,
        envelope=True,
        energy=1e-6,
        fwhm=30e-15,
        **grid_kwargs,
    )


GRID = {
    "lambda0": WAVELENGTH,
    "wavelength_limits": (700e-9, 920e-9),
    "trange": 200e-15,
}


def gaussian_field(grid, waist, *, shift=0.0, tilt=0.0):
    """Return an x-polarised Gaussian of unit-peak spectrum, displaced and tilted."""
    spectral_width = 30e-15 / (2 * math.sqrt(math.log(2)))

    def field(x, y, omega):
        k = omega / C
        spatial = np.exp(-((x - shift) ** 2 + y**2) / waist**2)
        phase = np.exp(1j * np.outer(x, k) * tilt)
        spectral = np.exp(-(((omega - grid.omega0) * spectral_width) ** 2) / 2)
        return 1e-9 * spatial[:, None] * phase * spectral[None, :]

    return field


def reference_shares(modes, grid, waist, shift, tilt, index):
    """Return each HE1m's captured share by the Marcatili field on a polar rule.

    The overlap in the common measure ``M0 = n eps0 c dA / 2`` of the analytic
    Gaussian with each mode's vacuum-normalised field ``field / sqrt(N)``, divided
    by the Gaussian's whole-plane norm (closed form), at the peak bin.
    """
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    omega = grid.omega[k_bin]
    k = omega / C
    r, t, w = polar_rule(RADIUS, [], panel=RADIUS / 24)
    x, y = r * np.cos(t), r * np.sin(t)
    gaussian = np.exp(-((x - shift) ** 2 + y**2) / waist**2) * np.exp(1j * k * tilt * x)
    norm = 0.5 * index * EPS0 * C * math.pi * waist**2 / 2
    shares = []
    for mode in modes:
        profile = mode.field(np.column_stack((r, t)))[:, 0] / math.sqrt(
            mode.normalisation()
        )
        overlap = 0.5 * index * EPS0 * C * np.sum(w * profile * gaussian)
        self_norm = 0.5 * index * EPS0 * C * np.sum(w * profile * profile)
        shares.append(abs(overlap) ** 2 / self_norm / norm)
    return np.array(shares), k_bin


@pytest.fixture(scope="module")
def capillary_port():
    """A 75 um argon capillary's port with eight HE1m modes."""
    simulation = capillary_simulation(RADIUS, 8, GRID)
    return simulation.input_port()


@pytest.mark.parametrize(
    ("waist", "shift", "tilt"),
    [
        (0.3, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.7, 0.1, 0.0),
        (1.0, 0.0, 5e-4),
        (0.5, 0.2, 1e-3),
    ],
)
def test_the_capillary_envelope_bounds_the_scalar_model_against_marcatili(
    capillary_port, waist, shift, tilt
):
    """Scalar shares of a Gaussian against Marcatili's own fields, within the bound.

    The beam is sampled on a Cartesian plane (so displacement and tilt are kept);
    the reference integrates the analytic beam against ``MarcatiliMode.field``.
    The comparison is at the peak bin, in the common measure, share by share.
    """
    port = capillary_port
    grid = port.grid
    index = float(port.reference_index[np.argmin(np.abs(grid.omega - grid.omega0))])
    medium = SpatialMedium("Ar", PRESSURE)
    space = FreeGrid(4 * RADIUS, 128, 4 * RADIUS, 128)
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        gaussian_field(grid, waist * RADIUS, shift=shift * RADIUS, tilt=tilt),
        space=space,
        medium=medium,
    )
    result = couple(
        plane, port, model=CouplingModel("scalar_paraxial", model_error=0.1)
    )
    modes = [
        MarcatiliMode(RADIUS, "Ar", PRESSURE, m=m) for m in range(1, port.shape[0] + 1)
    ]
    expected, k_bin = reference_shares(
        modes, grid, waist * RADIUS, shift * RADIUS, tilt, index
    )
    profiles = port.profiles
    assert profiles is not None
    # Captured share per mode in the common metric: G_mm |b_m|^2 / incident.
    b = result.launch.samples[:, 0, k_bin]
    g = index / profiles.effective_index[:, k_bin]
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    shares = weights[k_bin] * g * np.abs(b) ** 2 / result.report.incident[k_bin]
    # The bound covers the envelope; the plane's own representation error of the
    # sampled beam (its numerical item) is added, as the admission adds it.
    numerical = next(
        item.value for item in result.report.items if item.name == "numerical"
    )
    assert numerical is not None
    allowance = CAPILLARY_ENVELOPE.bound + 4 * numerical + 1e-9
    assert np.max(np.abs(shares - expected)) <= allowance


@pytest.mark.parametrize("ratio", [0.5, 0.8, 1.25, 2.0])
def test_the_capillary_envelope_bounds_a_change_of_radius(ratio):
    """A capillary's HE1m coefficients into another's against Marcatili's fields.

    The source carries a coherent mixture of its first three modes; the reference
    overlaps ``MarcatiliMode.field`` of both capillaries on a polar rule broken at
    both radii, and the scalar model's direct overlap is compared share by share.
    """
    source_radius = ratio * RADIUS
    source = capillary_simulation(source_radius, 3, GRID)
    target = capillary_simulation(RADIUS, 6, GRID)
    port = target.input_port()
    grid = port.grid
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    profiles = ScalarModeBasis.from_modes(source.modes, grid)
    coefficients = np.zeros((3, 1, grid.n_omega), dtype=complex)
    coefficients[:, 0, k_bin] = [1.0, 0.4j, -0.2]
    plane = FieldPlane.from_samples(
        coefficients,
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("modal", "0" * 64, profiles.labels),
        profiles=profiles,
    )
    result = couple(
        plane, port, model=CouplingModel("scalar_paraxial", model_error=0.1)
    )
    assert port.reference_index is not None
    index = float(port.reference_index[k_bin])
    outer = max(source_radius, RADIUS)
    r, t, w = polar_rule(outer, [min(source_radius, RADIUS)], panel=outer / 32)
    points = np.column_stack((r, t))
    field = np.zeros(r.size, dtype=complex)
    for mode, value in zip(
        cast("list[MarcatiliMode]", source.modes),
        coefficients[:, 0, k_bin],
        strict=True,
    ):
        n_eff = float(np.real(mode.neff(grid.omega[k_bin : k_bin + 1]))[0])
        profile = mode.field(points)[:, 0] * (r <= source_radius)
        field += value / math.sqrt(n_eff) * profile / math.sqrt(mode.normalisation())
    expected = []
    for mode in cast("list[MarcatiliMode]", target.modes):
        profile = (
            mode.field(points)[:, 0] * (r <= RADIUS) / math.sqrt(mode.normalisation())
        )
        n_eff = float(np.real(mode.neff(grid.omega[k_bin : k_bin + 1]))[0])
        amplitude = np.sum(w * profile * field) / np.sum(w * profile * profile)
        expected.append(math.sqrt(n_eff) * amplitude * index)
    # The target's vacuum-normalised amplitude in M0 is overlap / self-norm; the
    # scalar model's b = sqrt(n_eff) a carries the factor n_ref of G^-1 P.
    launched = result.launch.samples[:, 0, k_bin]
    scale = float(np.max(np.abs(launched)))
    expected_b = np.array(expected) / index
    assert (
        np.max(np.abs(np.abs(launched) ** 2 - np.abs(expected_b) ** 2))
        <= (CAPILLARY_ENVELOPE.bound + 1e-12) * scale**2
    )


def lp01_overlap(source, target):
    """Return the weak-guidance captured share |integral v_A v_B dA|^2."""
    outer = max(source.outer(), target.outer())
    r, w = radial_rule([source.radius, target.radius], outer, panel=source.radius / 8)
    return float(np.sum(w * source.values(r)[0] * target.values(r)[0]) ** 2)


def vector_overlap(first, second, omega, refine=1):
    """Return |F_AB|^2 of two unit-power HE11 fields on a polar rule."""
    radii = sorted({first._radius(0.0), second._radius(0.0)})
    outer = max(first.dimlimits()[2][0], second.dimlimits()[2][0])
    r, t, w = polar_rule(
        outer, radii, n_theta=16 * refine, panel=radii[0] / (8 * refine)
    )
    xy = np.column_stack((r * np.cos(t), r * np.sin(t)))
    ea, ha = first.electromagnetic_field(xy, omega)
    eb, hb = second.electromagnetic_field(xy, omega)
    cross = (ea[:, 0] * hb[:, 1].conj() - ea[:, 1] * hb[:, 0].conj()) + (
        eb[:, 0].conj() * ha[:, 1] - eb[:, 1].conj() * ha[:, 0]
    )
    return float(abs(0.25 * np.sum(w * cross)) ** 2)


def step_index_cases():
    """Yield the envelope's sampled pairs: contrast, wavelength, V_target, ratio.

    With constant indices the overlaps depend on the wavelength only through V, so
    one wavelength samples the whole range; both ends of every range are included.
    """
    wavelength = 1064e-9
    for contrast in (0.001, 0.005, 0.01):
        for v_target in (1.5, 1.8, 2.1, 2.4):
            for ratio in (0.7, 0.85, 1.0, 1.2, 1.43):
                if 1.5 <= ratio * v_target <= 2.4:
                    yield contrast, wavelength, v_target, ratio


def fibre(radius, cladding, core, wavelength):
    """Return an HE11 even step-index mode of the given materials."""
    return StepIndexMode(
        radius, core=core, cladding=cladding, profile_wavelength=wavelength
    )


@pytest.mark.medium
def test_the_step_index_envelope_bounds_lp01_against_the_vector_he11_overlap():
    """Weak-guidance LP01 overlap against the exact HE11 flux overlap, within bound.

    Each pair shares its materials and wavelength and differs in core radius;
    the vector reference's error is estimated by doubling its quadrature.
    """
    core = 1.45
    worst = 0.0
    numerical = 0.0
    for contrast, wavelength, v_target, ratio in step_index_cases():
        cladding = core * math.sqrt(1.0 - 2.0 * contrast)
        aperture = math.sqrt(core**2 - cladding**2)
        k = 2 * math.pi / wavelength
        radius_target = v_target / (k * aperture)
        radius_source = ratio * radius_target
        source = fibre(radius_source, cladding, core, wavelength)
        target = fibre(radius_target, cladding, core, wavelength)
        grid = EnvelopeGrid.from_limits(
            1e-3, wavelength, (0.95 * wavelength, 1.05 * wavelength), 200e-15
        )
        scalar = lp01_overlap(
            ScalarModeBasis.from_modes([source], grid),
            ScalarModeBasis.from_modes([target], grid),
        )
        omega = 2 * math.pi * C / wavelength
        vector = vector_overlap(source, target, omega)
        refined = vector_overlap(source, target, omega, refine=2)
        numerical = max(numerical, abs(vector - refined))
        worst = max(worst, abs(scalar - vector))
    assert numerical <= STEP_INDEX_ENVELOPE.numerical
    assert worst + numerical <= STEP_INDEX_ENVELOPE.bound

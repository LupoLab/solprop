# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The oracles behind the recorded coupling envelopes (SEP 31, section 4.3).

Each recorded :class:`~solprop.coupling.envelopes.PairEnvelope` is recomputed here
from an independent reference and held to its bound:

- the capillary envelope: the scalar J0 profiles against Marcatili's leaky HE1m core
  field :math:`J_0(u_m(1 - i\delta)r/a)`, the physics the scalar model omits, on an
  independent polar Gauss--Legendre rule over the envelope's declared range. Beside
  it, two consistency checks hold the implementation (``couple`` on sampled planes,
  through the closed-form Hankel transforms) to the overlap of
  :meth:`~solprop.modes.MarcatiliMode.field` by quadrature. ``MarcatiliMode.field``
  of an HE1m mode *is* the scalar J0 profile, so those checks bound the quadrature
  and the code, not the model;
- the step-index envelope: the weak-guidance LP01 overlap the scalar model uses
  against the full-vector flux overlap of the exact bound HE11 fields
  (:meth:`~solprop.modes.StepIndexMode.electromagnetic_field`).

The numerical error of each reference is estimated by refining its quadrature.
"""

import math
from typing import cast

import numpy as np
import pytest
from scipy.special import jn_zeros, jv

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
from solprop.exceptions import CouplingError
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
def test_the_scalar_capillary_overlap_matches_marcatili_fields_by_quadrature(
    capillary_port, waist, shift, tilt
):
    """Scalar shares of a Gaussian against Marcatili's own fields: a consistency check.

    The beam is sampled on a Cartesian plane (so displacement and tilt are kept);
    the reference integrates the analytic beam against ``MarcatiliMode.field``,
    which for HE1m is the same J0 profile, so this checks the closed-form transforms
    and the fit against an independent quadrature, not the scalar model's physics
    (the leaky-field oracle below bounds that). The comparison is at the peak bin,
    in the common measure, share by share.
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
    # The same profile on both sides: the difference is quadrature and the plane's
    # own representation error of the sampled beam (its numerical item); measured
    # 1.8e-14 of the incident energy, the reference rule's rounding.
    numerical = next(
        item.value for item in result.report.items if item.name == "numerical"
    )
    assert numerical is not None
    allowance = 1e-12 + 4 * numerical + 1e-9
    assert np.max(np.abs(shares - expected)) <= allowance


@pytest.mark.parametrize("ratio", [0.5, 0.8, 1.25, 2.0])
def test_a_scalar_change_of_radius_matches_marcatili_fields_by_quadrature(ratio):
    """A capillary's HE1m coefficients into another's: a consistency check.

    As above, ``MarcatiliMode.field`` is the scalar profile, so the comparison bounds
    the direct overlap's quadrature, not the model.

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
        <= 1e-12 * scale**2
    )


# ------------------------------------------------------ the leaky-field oracle

HE_ZEROS = jn_zeros(0, 8)
"""The first eight J0 zeros, the HE1m core wavenumbers u_m."""

LEAKY_WALL_TERM = 1.477
"""|nu_n| of silica (index 1.45): (eps + 1)/(2 sqrt(eps - 1)), sizing k a only."""


def polar_disc(radius, n_r, n_theta):
    """Return a Gauss--Legendre (r) by uniform (theta) rule on a disc, with weights."""
    x, weights = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * radius * (x + 1)
    radial = 0.5 * radius * weights * r
    theta = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    return (
        np.repeat(r, n_theta),
        np.tile(theta, n_r),
        np.repeat(radial, n_theta) * 2 * np.pi / n_theta,
    )


def leaky_profiles(radius, delta, count, r):
    """Return Marcatili's HE1m core fields J0(u (1 - i delta) r / a), zero outside."""
    return np.array(
        [
            jv(0, u * (1 - 1j * delta) * r / radius) * (r <= radius)
            for u in HE_ZEROS[:count]
        ]
    )


def projected_shares(profiles, field, weights, field_norm):
    """Return the per-mode shares and the least-squares captured share of a field."""
    gram = (profiles.conj() * weights) @ profiles.T
    overlaps = (profiles.conj() * weights) @ field
    per_mode = np.abs(overlaps) ** 2 / np.real(np.diag(gram)) / field_norm
    total = np.real(overlaps.conj() @ np.linalg.solve(gram, overlaps)) / field_norm
    return per_mode, total


def discrepancy(scalar, leaky):
    """Return the larger of the summed per-mode and the total share changes."""
    return max(
        float(np.sum(np.abs(leaky[0] - scalar[0]))), float(abs(leaky[1] - scalar[1]))
    )


def beam_discrepancy(delta, beam, n_r=100, n_theta=32):
    """Return a Gaussian beam's share discrepancy, leaky against scalar, a = 1.

    ``beam`` is (HE1m count, waist/a, displacement/a, tilt angle times k a).
    """
    count, waist, shift, tilt_ka = beam
    r, t, w = polar_disc(1.0, n_r, n_theta)
    x, y = r * np.cos(t), r * np.sin(t)
    field = np.exp(-((x - shift) ** 2 + y**2) / waist**2) * np.exp(1j * tilt_ka * x)
    norm = math.pi * waist**2 / 2
    return discrepancy(
        projected_shares(leaky_profiles(1.0, 0.0, count, r), field, w, norm),
        projected_shares(leaky_profiles(1.0, delta, count, r), field, w, norm),
    )


def radius_discrepancy(delta, ratio, n_r=100, n_theta=32):
    """Return a coherent three-mode exit's discrepancy into eight modes at a ratio.

    ``delta`` is the larger of the two capillaries' parameters; it scales as 1/a.
    """
    r, _, w = polar_disc(max(1.0, ratio), n_r, n_theta)
    shares = []
    for scale in (0.0, delta):
        source_delta, target_delta = (
            (scale, scale * ratio) if ratio < 1 else (scale / ratio, scale)
        )
        source = leaky_profiles(ratio, source_delta, 3, r)
        source = source / np.sqrt(np.sum(w * np.abs(source) ** 2, axis=1))[:, None]
        field = source.sum(axis=0)
        norm = float(np.sum(w * np.abs(field) ** 2))
        shares.append(
            projected_shares(leaky_profiles(1.0, target_delta, 8, r), field, w, norm)
        )
    return discrepancy(*shares)


BEAMS = [
    (count, waist, shift, tilt)
    for count in (1, 2, 4, 8)
    for waist in (0.3, 0.5, 0.7, 1.0)
    for shift in (0.0, 0.1, 0.2)
    for tilt in (0.0, 0.59)
]
"""The envelope's beams: HE1m count, waist/a, displacement/a, tilt angle times k a."""

RATIOS = (0.5, 0.8, 1.0, 1.25, 2.0)


def worst_leaky_discrepancy(delta, **rule):
    """Return the largest discrepancy over the envelope's beams and radius ratios."""
    beams = max(beam_discrepancy(delta, case, **rule) for case in BEAMS)
    radii = max(radius_discrepancy(delta, ratio, **rule) for ratio in RATIOS)
    return beams, radii


@pytest.mark.medium
def test_the_capillary_envelope_bounds_the_leaky_core_field():
    """The scalar J0 model against Marcatili's leaky HE1m field, over the range.

    At the range's top, delta = 1e-2 (a 37.5 um silica capillary at 1.6 um, or
    18.8 um at 800 nm), the largest discrepancy is 2.9e-3 (equal radii: the leaky
    modes are not orthogonal) and 1.8e-3 for the beams; the recorded bound is twice
    the largest. Halving delta halves the beam cases (first order, from the tilt and
    displacement) and quarters the radius cases, which is what lets the allowance
    scale linearly with delta. The quadrature's numerical error is estimated by
    refining the rule.
    """
    top = CAPILLARY_ENVELOPE.ranges["leaky"][1]
    beams, radii = worst_leaky_discrepancy(top)
    measured = max(beams, radii)
    assert measured == pytest.approx(2.9e-3, rel=0.05)
    assert 2 * measured <= CAPILLARY_ENVELOPE.bound
    fine_beams, fine_radii = worst_leaky_discrepancy(top, n_r=200, n_theta=48)
    assert abs(max(fine_beams, fine_radii) - measured) <= CAPILLARY_ENVELOPE.numerical
    half_beams, half_radii = worst_leaky_discrepancy(0.5 * top)
    # At most linear below the top: the allowance's scaling is conservative there.
    assert half_beams <= 0.55 * beams
    assert half_radii <= 0.3 * radii
    assert CAPILLARY_ENVELOPE.allowance({"leaky": 0.5 * top}) >= 2 * max(
        half_beams, half_radii
    )


def test_a_capillary_beyond_the_leaky_range_is_refused():
    """A capillary beyond the leaky range is refused, not extrapolated."""
    port = capillary_simulation(12e-6, 1, GRID).input_port()
    profiles = port.profiles
    assert profiles is not None
    leaky = profiles.leaky(port.grid.omega)
    assert leaky is not None
    centre = int(np.argmin(np.abs(port.grid.omega - port.grid.omega0)))
    # A 12 um silica capillary at 800 nm: 1.475 c/(omega a) = 1.57e-2.
    assert leaky[centre] == pytest.approx(1.57e-2, rel=0.01)
    plane = FieldPlane.from_analytic(
        port.grid,
        PlaneGeometry(0.0),
        gaussian_field(port.grid, 6e-6),
        space=FreeGrid(60e-6, 64, 60e-6, 64),
        medium=SpatialMedium("Ar", PRESSURE),
    )
    with pytest.raises(CouplingError, match="leaky"):
        couple(plane, port, model=CouplingModel("scalar_paraxial", model_error=0.2))


def test_the_capillary_envelope_item_scales_with_the_pair(capillary_port):
    """The reported envelope item is the bound scaled by the energy-weighted delta."""
    port = capillary_port
    plane = FieldPlane.from_analytic(
        port.grid,
        PlaneGeometry(0.0),
        gaussian_field(port.grid, 0.5 * RADIUS),
        space=FreeGrid(4 * RADIUS, 64, 4 * RADIUS, 64),
        medium=SpatialMedium("Ar", PRESSURE),
    )
    report = couple(
        plane, port, model=CouplingModel("scalar_paraxial", model_error=0.1)
    ).report
    profiles = port.profiles
    assert profiles is not None
    delta = profiles.leaky(port.grid.omega)
    assert delta is not None
    weighted = float(np.sum(delta * report.incident) / np.sum(report.incident))
    # A 75 um silica capillary at 800 nm: 1.475 c/(omega a) = 2.5e-3.
    assert weighted == pytest.approx(2.5e-3, rel=0.02)
    item = next(item for item in report.items if item.name == "envelope")
    # The same weighted sum, formed in another order: rounding only.
    assert item.value == pytest.approx(
        CAPILLARY_ENVELOPE.allowance({"leaky": weighted}), rel=1e-12
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

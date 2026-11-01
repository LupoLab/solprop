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
  (:meth:`~solprop.modes.StepIndexMode.electromagnetic_field`);
- the beam-into-fibre envelope: an incident field's scalar LP01 overlap against its
  full-vector flux coupling into the exact HE11 pair, the field's H from its own
  plane-wave spectrum, maximised over every field inside the widest admitted cone
  and sampled by Gaussian beams;
- the vector model's step-index facet envelope: the flux pairing with the incident
  field against an independent full-vector mode-matching solution of the entrance
  facet (``facet_mode_matching``), maximised over every field inside that cone.

The numerical error of each reference is estimated by refining its quadrature.
"""

import itertools
import math
from typing import cast

import numpy as np
import pytest
from scipy.special import jn_zeros, jv

from facet_mode_matching import (
    Y_VACUUM,
    Beam,
    StepIndexFacet,
    hybrid_eigenvalue,
    mode_shares,
    ring_rule,
)
from modal_provider_cases import make_provider
from solprop import fourier
from solprop.constants import EPS0, MU0, C
from solprop.coupling import (
    CAPILLARY_ENVELOPE,
    STEP_INDEX_BEAM_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    VECTOR_STEP_INDEX_ENVELOPE,
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
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, StepIndexMode
from solprop.modes.guide import ModalGuide
from solprop.modes.stepindex import _cutoff
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
    "wavelength": WAVELENGTH,
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


# ------------------------------------------------- the beam-into-fibre oracle

BEAM_CORE = 1.45
BEAM_WAVELENGTH = 1064e-9
WIDEST_CONE = math.acos(1.0 - 1.0 / 8.0)
"""The cone of a budget approaching one: every admitted cone lies inside it."""


def gauss_panels(edges, panel, nodes=16):
    """Return Gauss--Legendre nodes and weights on panels at most `panel` long."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    points, weights = [], []
    for low, high in itertools.pairwise(edges):
        bounds = np.linspace(low, high, max(1, math.ceil((high - low) / panel)) + 1)
        for left, right in itertools.pairwise(bounds):
            half = 0.5 * (right - left)
            points.append(left + half * (x + 1))
            weights.append(half * w)
    return np.concatenate(points), np.concatenate(weights)


class BeamFibreReference:
    r"""The full-vector coupling of incident fields into a fibre's exact HE11 pair.

    A field in the core index is its transverse plane-wave spectrum
    :math:`\hat E_t(\mathbf k) = \int E_t e^{i\mathbf k\cdot\mathbf r}dA` inside the
    widest admitted cone; each plane wave's :math:`E_z = -\mathbf k_t\cdot
    \hat E_t/k_z` and :math:`\mathbf H = \mathbf k\times\mathbf E/(\omega\mu_0)`
    (the convention of ``electromagnetic_field``, :math:`e^{i\omega t - i\mathbf
    k\cdot\mathbf r}`), so :math:`\hat H_t = Y(\mathbf k)\hat E_t`. A mode's
    amplitude is the flux pairing :math:`\frac14\int(\mathbf E\times\mathbf h^* +
    \mathbf e^*\times\mathbf H)\cdot\hat z\,dA`, taken by Parseval in the Fourier
    plane. The HE11 fields are split exactly into azimuthal orders 0 and 2 and
    transformed by Gauss--Legendre Hankel sums broken at the core radius; the
    LP01 profile the model uses is transformed the same way (the model itself uses
    its closed form). The k plane is a Gauss--Legendre (radius) by uniform (angle)
    rule.
    """

    def __init__(self, contrast, v, refine=1):
        cladding = BEAM_CORE * math.sqrt(1 - 2 * contrast)
        self.k0 = 2 * math.pi / BEAM_WAVELENGTH
        self.omega = self.k0 * C
        radius = v / (self.k0 * math.sqrt(BEAM_CORE**2 - cladding**2))
        self.radius = radius
        modes = {
            parity: StepIndexMode(
                radius,
                core=BEAM_CORE,
                cladding=cladding,
                profile_wavelength=BEAM_WAVELENGTH,
                parity=parity,
            )
            for parity in ("even", "odd")
        }
        grid = EnvelopeGrid.from_limits(
            1e-3,
            BEAM_WAVELENGTH,
            (0.95 * BEAM_WAVELENGTH, 1.05 * BEAM_WAVELENGTH),
            200e-15,
        )
        lp01 = ScalarModeBasis.from_modes([modes["even"]], grid)
        kappa_star = BEAM_CORE * self.k0 * math.sin(WIDEST_CONE)
        outer = modes["even"].dimlimits()[2][0]
        panel = min(radius / 2, 2 * math.pi / kappa_star) / refine
        self.r, self.r_weights = gauss_panels([0.0, radius, outer], panel)
        self.kappa, kappa_weights = gauss_panels(
            [0.0, kappa_star], kappa_star / (24 * refine)
        )
        n_psi = 64 * refine
        psi = 2 * np.pi * (np.arange(n_psi) + 0.5) / n_psi
        self.n_psi = n_psi
        kappa = np.repeat(self.kappa, n_psi)
        angle = np.tile(psi, self.kappa.size)
        self.angle = angle
        self.weights = (
            np.repeat(kappa_weights * self.kappa, n_psi) / n_psi / (2 * np.pi)
        )
        self.kx, self.ky = kappa * np.cos(angle), kappa * np.sin(angle)
        kz = np.sqrt((BEAM_CORE * self.k0) ** 2 - kappa**2)
        scale = EPS0 * C**2 / (self.omega * kz)  # 1/(omega mu0 kz)
        kx, ky = self.kx, self.ky
        self.admittance = np.empty((kappa.size, 2, 2))
        self.admittance[:, 0, 0] = -kx * ky * scale
        self.admittance[:, 0, 1] = -(ky**2 + kz**2) * scale
        self.admittance[:, 1, 0] = (kz**2 + kx**2) * scale
        self.admittance[:, 1, 1] = kx * ky * scale
        self._bessel = {}
        self.decomposition = 0.0
        self.coupling = {name: self._coupling(mode) for name, mode in modes.items()}
        self.lp01 = (
            2 * np.pi * np.repeat(self._hankel(lp01.values(self.r)[0], 0), n_psi)
        )

    def _hankel(self, values, order):
        if order not in self._bessel:
            self._bessel[order] = jv(order, np.outer(self.kappa, self.r)) * (
                self.r * self.r_weights
            )
        return self._bessel[order] @ values

    def _coupling(self, mode):
        """Return c(k) with a(f) = sum_k w_k c(k) . E_t(k), shape (n_k, 2)."""
        r = self.r
        samples = {}
        for phi in (0.0, math.pi / 4, math.pi / 2, 0.37):
            xy = np.column_stack((r * math.cos(phi), r * math.sin(phi)))
            samples[phi] = mode.electromagnetic_field(xy, self.omega)
        # An HE11 field is exactly f0(r) + f2(r) cos 2phi in one Cartesian
        # component and g2(r) sin 2phi in the other (cos and sin swap with parity).
        even = mode.parity == "even"
        cosine = {("e", 0): even, ("e", 1): not even, ("h", 0): not even}
        cosine["h", 1] = even
        spectra = {}
        for (field, component), is_cosine in cosine.items():
            index = 0 if field == "e" else 1
            at = {phi: values[index][:, component] for phi, values in samples.items()}
            if is_cosine:
                zeroth = 0.5 * (at[0.0] + at[math.pi / 2])
                second = 0.5 * (at[0.0] - at[math.pi / 2])
                trig, angular = math.cos(2 * 0.37), np.cos(2 * self.angle)
            else:
                zeroth = np.zeros_like(at[0.0])
                second = at[math.pi / 4]
                trig, angular = math.sin(2 * 0.37), np.sin(2 * self.angle)
            # The split is exact: check it at a fourth angle.
            residual = np.max(np.abs(zeroth + second * trig - at[0.37]))
            peak = np.max(np.abs(samples[0.0][index]))
            self.decomposition = max(self.decomposition, float(residual / peak))
            # f(r) cos(n phi) transforms to 2 pi i^n cos(n psi) H_n[f].
            spectra[field, component] = (
                2
                * np.pi
                * (
                    np.repeat(self._hankel(zeroth, 0), self.n_psi)
                    - np.repeat(self._hankel(second, 2), self.n_psi) * angular
                )
            )
        first = np.stack((np.conj(spectra["h", 1]), -np.conj(spectra["h", 0])), axis=1)
        dual = np.stack((-np.conj(spectra["e", 1]), np.conj(spectra["e", 0])), axis=1)
        return 0.25 * (first + np.einsum("kji,kj->ki", self.admittance, dual))

    def worst_field(self):
        """Return the largest |captured discrepancy| / incident over all fields.

        In the M0 inner product the scalar capture is sum |<v_p, f>|^2 over the
        x and y LP01 channels and the vector capture sum |<q_m, f>|^2 over the
        HE11 pair; the largest modulus of their difference's eigenvalues on the
        span of the four vectors bounds every field in the cone.
        """
        root = np.sqrt(self.weights)
        zero = np.zeros_like(self.lp01)
        scalar = [
            np.concatenate((self.lp01 * root, zero)),
            np.concatenate((zero, self.lp01 * root)),
        ]
        factor = math.sqrt(2 / (BEAM_CORE * EPS0 * C))
        vector = [
            (factor * np.conj(self.coupling[name]) * root[:, None]).T.ravel()
            for name in ("even", "odd")
        ]
        basis, _ = np.linalg.qr(np.column_stack(scalar + vector))
        s = basis.conj().T @ np.column_stack(scalar)
        q = basis.conj().T @ np.column_stack(vector)
        difference = s @ s.conj().T - q @ q.conj().T
        return float(np.max(np.abs(np.linalg.eigvalsh(difference))))

    def gaussian(self, waist, shift=(0.0, 0.0), tilt=(0.0, 0.0)):
        """Return an x-polarised Gaussian's transverse spectrum, shape (n_k, 2)."""
        qx, qy = (BEAM_CORE * self.k0 * angle for angle in tilt)
        dx, dy = self.kx - qx, self.ky - qy
        spectrum = (
            math.pi
            * waist**2
            * np.exp(-(dx**2 + dy**2) * waist**2 / 4)
            * np.exp(1j * (dx * shift[0] + dy * shift[1]))
        )
        return np.stack((spectrum, np.zeros_like(spectrum)), axis=1)

    def captured(self, field):
        """Return the scalar and vector captures per channel over M0 incident."""
        half_admittance = 0.5 * BEAM_CORE * EPS0 * C
        w = self.weights
        incident = half_admittance * np.sum(w * np.sum(np.abs(field) ** 2, axis=1))
        scalar = [
            half_admittance * abs(np.sum(w * self.lp01 * field[:, p])) ** 2
            for p in (0, 1)
        ]
        vector = [
            abs(np.sum(w[:, None] * self.coupling[name] * field)) ** 2
            for name in ("even", "odd")
        ]
        return np.array(scalar) / incident, np.array(vector) / incident


@pytest.mark.medium
def test_the_beam_fibre_envelope_bounds_every_field_in_the_cone():
    """The scalar LP01 capture against the vector HE11 pair, maximised over fields.

    The largest discrepancy over every field inside the widest admitted cone is
    0.467 Delta at V 1.5 and falls with V (0.284 Delta at 2.4, where the cone's
    edge truncating the HE11 spectrum makes it conservative); at V 1.5 it is
    linear in Delta from 5e-4 to 1e-2 (the overlaps depend on V and Delta
    alone, so not on the core index or the wavelength). The recorded bound is
    at least twice it at every contrast, through the allowance's linear scaling;
    the reference's numerical error is estimated by refining every rule.
    """
    top = STEP_INDEX_BEAM_ENVELOPE.ranges["contrast"][1]
    worst = {v: BeamFibreReference(top, v) for v in (1.5, 1.8, 2.1, 2.4)}
    largest = {v: reference.worst_field() for v, reference in worst.items()}
    for reference in worst.values():
        # The azimuthal split of the exact fields is exact to rounding.
        assert reference.decomposition < 1e-12
    assert largest[1.5] == pytest.approx(4.67e-3, rel=0.01)
    assert largest[2.4] == pytest.approx(2.84e-3, rel=0.01)
    assert max(largest, key=largest.__getitem__) == 1.5
    refined = BeamFibreReference(top, 1.5, refine=2).worst_field()
    numerical = abs(refined - largest[1.5])
    assert numerical <= STEP_INDEX_BEAM_ENVELOPE.numerical
    assert 2 * largest[1.5] + numerical <= STEP_INDEX_BEAM_ENVELOPE.bound
    for contrast in (5e-4, 1e-3, 5e-3):
        low = BeamFibreReference(contrast, 1.5).worst_field()
        assert low / contrast == pytest.approx(largest[1.5] / top, rel=0.01)
        assert 2 * low <= STEP_INDEX_BEAM_ENVELOPE.allowance({"contrast": contrast})


@pytest.mark.parametrize("v", [1.5, 2.4])
def test_gaussian_beams_lie_within_the_beam_fibre_envelope(v):
    """Gaussian beams of every waist, offset and tilt stay inside the field bound.

    Waists 0.5a to 2.6a, displaced to 0.5a and tilted to half the acceptance
    angle sqrt(2 Delta) along x and along y, x-polarised, at contrast 0.01: each
    beam's discrepancy (summed over both polarisation channels) is below the
    largest over all fields, and the worst reaches 3.3e-3 at V 1.5 (a centred
    beam at 0.8a, whose waist mismatch projects onto the vector correction).
    """
    contrast = STEP_INDEX_BEAM_ENVELOPE.ranges["contrast"][1]
    reference = BeamFibreReference(contrast, v)
    bound = reference.worst_field()
    radius = reference.radius
    acceptance = math.sqrt(2 * contrast)
    worst = 0.0
    for waist in (0.5, 0.8, 1.1, 1.4, 1.8, 2.2, 2.6):
        for shift, tilt in (
            ((0.0, 0.0), (0.0, 0.0)),
            ((0.25, 0.0), (0.0, 0.0)),
            ((0.5, 0.0), (0.0, 0.0)),
            ((0.0, 0.5), (0.0, 0.0)),
            ((0.0, 0.0), (0.25, 0.0)),
            ((0.0, 0.0), (0.5, 0.0)),
            ((0.0, 0.0), (0.0, 0.5)),
            ((0.5, 0.0), (0.0, 0.5)),
        ):
            field = reference.gaussian(
                waist * radius,
                (shift[0] * radius, shift[1] * radius),
                (tilt[0] * acceptance, tilt[1] * acceptance),
            )
            scalar, vector = reference.captured(field)
            discrepancy = float(np.sum(np.abs(scalar - vector)))
            # Per channel the bound is the same eigenvalue problem's: rounding.
            assert discrepancy <= bound + 1e-12
            worst = max(worst, discrepancy)
    if v == 1.5:
        assert worst == pytest.approx(3.3e-3, rel=0.02)


def test_the_beam_fibre_reference_takes_the_overlap_couple_computes():
    """The reference's scalar capture is the share ``couple`` reports for a beam.

    A centred Gaussian (waist a) into a fibre of contrast 0.001 and V 2 at
    1064 nm, sampled on a radial plane in the core index: ``couple``'s captured
    share at the profile bin equals the reference's scalar capture of the same
    beam, which lies wholly inside the cone.
    """
    contrast, v = 1e-3, 2.0
    reference = BeamFibreReference(contrast, v)
    radius = reference.radius
    scalar, _ = reference.captured(reference.gaussian(radius))
    cladding = BEAM_CORE * math.sqrt(1 - 2 * contrast)
    fibre = StepIndexMode(
        radius, core=BEAM_CORE, cladding=cladding, profile_wavelength=BEAM_WAVELENGTH
    )
    simulation = Simulation.from_gnlse(
        None,
        1e-3,
        fibre=fibre,
        n2=2.6e-20,
        fwhm=100e-15,
        energy=1e-12,
        n_save=2,
        wavelength=BEAM_WAVELENGTH,
        wavelength_limits=(900e-9, 1300e-9),
        trange=400e-15,
        noise=None,
    )
    port = simulation.input_port()
    grid = port.grid
    width = 100e-15 / (2 * math.sqrt(math.log(2)))

    def field(r, omega):
        spectral = np.exp(-(((omega - grid.omega0) * width) ** 2) / 2)
        return np.exp(-((r / radius) ** 2))[:, None] * spectral[None, :]

    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        field,
        space=RadialGrid(30 * radius, 256),
        index=np.asarray(port.reference_index),
    )
    report = couple(
        plane, port, model=CouplingModel("scalar_paraxial", model_error=0.1)
    ).report
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    captured = report.captured
    assert captured is not None
    # Two quadratures of one overlap: the plane's radial spectral sum with the
    # closed-form LP01 transform, and the reference's k-plane rule.
    assert captured[k_bin] / report.incident[k_bin] == pytest.approx(
        scalar[0], rel=1e-8
    )
    assert scalar[1] == 0.0


FACET_WAVELENGTH = 1e-6
FACET_OMEGA = 2 * math.pi * C / FACET_WAVELENGTH
FACET_CORE = 1.45


def facet_fibre(contrast, v):
    """Return (radius, (input, core, cladding) indices) of a matched-input fibre."""
    cladding = FACET_CORE * math.sqrt(1.0 - 2.0 * contrast)
    k = FACET_OMEGA / C
    radius = v / (k * math.sqrt(FACET_CORE**2 - cladding**2))
    return radius, (FACET_CORE, FACET_CORE, cladding)


def test_the_facet_reference_solves_the_exact_vector_eigenvalue():
    """The reference's own HE11 converges to the exact vector eigenvalue equation.

    Snyder and Love's equation, solved directly, and the package's step-index
    solver agree to rounding; the reference reaches them as 1/N in its basis
    size, as a smooth basis meeting a jump in E_r does, without using either.
    """
    radius, indices = facet_fibre(0.01, 2.0)
    k = FACET_OMEGA / C
    u = hybrid_eigenvalue(1, 2.0, indices[1], indices[2])
    exact = math.sqrt(indices[1] ** 2 - (u / (k * radius)) ** 2)
    package = StepIndexMode(
        radius,
        core=indices[1],
        cladding=indices[2],
        profile_wavelength=FACET_WAVELENGTH,
    ).neff(np.array([FACET_OMEGA]))[0]
    assert abs(float(np.real(package)) - exact) < 1e-12
    errors = []
    for count in (50, 100, 200):
        facet = StepIndexFacet(1, radius, indices, FACET_OMEGA, 8 * radius, count)
        assert facet.orthonormality < 1e-12
        guided = facet.guided()
        assert guided.size == 1
        errors.append(abs(float(np.real(facet.beta[guided[0]])) / k - exact))
    # First order in 1/N once asymptotic: the last doubling about halves it.
    assert errors[0] > errors[1]
    assert errors[2] < 0.6 * errors[1]
    assert errors[1] < 2e-6


def test_the_facet_reference_meets_fresnel_and_conserves_power():
    """A uniform guide is a planar interface: every basis field meets Fresnel.

    With the core and cladding of one index the guide's modes are the cylinder's
    own fields, its admittance is diagonal, and each field's transmission is the
    modal Fresnel coefficient 2 Y_0 / (Y_0 + Y_1) of its TE or TM admittance. On a
    real step-index facet the incident power equals the reflected plus the
    transmitted power to rounding.
    """
    radius = 2e-6
    facet = StepIndexFacet(1, radius, (1.45, 1.5, 1.5), FACET_OMEGA, 8 * radius, 60)
    k = FACET_OMEGA / C
    basis = facet.basis
    live = basis.k < 0.9 * k * 1.45
    beta_in = np.sqrt((k * 1.45) ** 2 - basis.k[live] ** 2)
    beta_out = np.sqrt((k * 1.5) ** 2 - basis.k[live] ** 2)
    te = basis.is_te[live]
    y_in = np.where(
        te, beta_in / (FACET_OMEGA * MU0), FACET_OMEGA * EPS0 * 1.45**2 / beta_in
    )
    y_out = np.where(
        te, beta_out / (FACET_OMEGA * MU0), FACET_OMEGA * EPS0 * 1.5**2 / beta_out
    )
    np.testing.assert_allclose(np.real(np.diag(facet.y_fibre))[live], y_out, rtol=1e-9)
    for j in np.flatnonzero(live)[:: max(1, int(np.sum(live)) // 8)]:
        unit = np.zeros(basis.k.size, dtype=complex)
        unit[j] = 1.0
        transmitted = 2 * np.linalg.solve(
            facet.y_input + facet.y_fibre, facet.y_input @ unit
        )
        position = int(np.flatnonzero(np.flatnonzero(live) == j)[0])
        expected = 2 * y_in[position] / (y_in[position] + y_out[position])
        assert abs(transmitted[j] - expected) < 1e-9
    radius, indices = facet_fibre(0.03, 2.0)
    step = StepIndexFacet(1, radius, indices, FACET_OMEGA, 8 * radius, 60)
    radii, weights, theta = ring_rule((0.0, radius, 8 * radius), radius / 4)
    r, t = radii[:, None], theta[None, :]
    ex = Beam(1.0, (0.3, 0.0), 0.2).field(
        r * np.cos(t), r * np.sin(t), radius, k * 0.35
    )
    coefficients = step.basis.project(
        radii, weights, theta, ex * np.cos(t), -ex * np.sin(t)
    )
    incident, reflected, transmitted_power = step.amplitudes(
        coefficients, int(step.guided()[0])
    )[3]
    assert reflected > 0.0
    assert abs(incident - reflected - transmitted_power) < 1e-10 * incident


def test_the_facet_reference_guided_field_is_the_package_he11():
    """The reference's HE11 has the radial profile of the package's exact field.

    The order-one mode is circularly polarised, so its |e_t|^2 is the azimuthal
    mean of the even HE11's from
    :meth:`~solprop.modes.StepIndexMode.electromagnetic_field`. Both are unit
    power; the reference's profile converges as 1/N (1e-2 at N = 100).
    """
    radius, indices = facet_fibre(0.01, 2.0)
    facet = StepIndexFacet(1, radius, indices, FACET_OMEGA, 8 * radius, 100)
    u, _ = facet.unit_mode(int(facet.guided()[0]))
    probe = radius * np.array([0.1, 0.4, 0.7, 1.3, 1.8, 2.5])
    er, ep, _ = facet.basis.fields(probe)
    reference = np.abs(er @ u) ** 2 + np.abs(ep @ u) ** 2
    mode = StepIndexMode(
        radius,
        core=indices[1],
        cladding=indices[2],
        profile_wavelength=FACET_WAVELENGTH,
    )
    theta = 2 * np.pi * (np.arange(64) + 0.5) / 64
    xy = np.column_stack(
        (
            np.repeat(probe, 64) * np.tile(np.cos(theta), 6),
            np.repeat(probe, 64) * np.tile(np.sin(theta), 6),
        )
    )
    electric, _ = mode.electromagnetic_field(xy, FACET_OMEGA)
    package = np.mean(
        (np.abs(electric[:, 0]) ** 2 + np.abs(electric[:, 1]) ** 2).reshape(6, 64),
        axis=1,
    )
    np.testing.assert_allclose(reference, package, rtol=1e-2)


@pytest.mark.parametrize(("contrast", "v"), [(0.003, 2.0), (0.003, 4.0), (0.01, 2.0)])
def test_the_facet_correction_is_the_local_fresnel_estimate(contrast, v):
    """At weak guidance the omitted facet term is the local Fresnel estimate.

    To leading order the admittance difference D = Y_f - Y_0 is the local index
    step Y_vac (n(r) - n_co) and Y_0 + Y_f is 2 Y_vac n_co, so the identity
    a_exact - a_model = -<u_i, D (Y_0 + Y_f)^-1 D e_m>/4 becomes
    -(Y_vac (n_cl - n_co)^2 / (8 n_co)) times the cladding overlap of beam and
    mode: second order in the step, and negative. The rigorous solution agrees to
    a few per cent, the size of the next order.
    """
    radius, indices = facet_fibre(contrast, v)
    radii, weights, theta = ring_rule((0.0, radius, 8 * radius), radius / 4)
    r, t = radii[:, None], theta[None, :]
    ex = Beam(1.0).field(r * np.cos(t), r * np.sin(t), radius, 0.0)
    incident = (
        (
            0.5
            * FACET_CORE
            * Y_VACUUM
            * np.sum((radii * weights)[:, None] * np.abs(ex) ** 2)
        )
        * 2
        * np.pi
        / theta.size
    )
    facet = StepIndexFacet(1, radius, indices, FACET_OMEGA, 8 * radius, 100)
    coefficients = facet.basis.project(
        radii, weights, theta, ex * np.cos(t), -ex * np.sin(t)
    )
    mode = int(facet.guided()[0])
    u, _ = facet.unit_mode(mode)
    exact, model_exact_h, _, _ = facet.amplitudes(coefficients, mode)
    step = indices[2] - indices[1]
    local = -(Y_VACUUM * step**2 / (8 * FACET_CORE)) * (
        u.conj() @ facet.cladding_gram @ coefficients
    )
    rigorous = (abs(exact) ** 2 - abs(model_exact_h) ** 2) / incident
    estimate = 2 * float(np.real(np.conj(model_exact_h) * local)) / incident
    assert rigorous < 0.0
    assert rigorous == pytest.approx(estimate, rel=0.05)


def guided_count(order, v, contrast):
    """Return the guided modes of one azimuthal order from the exact cutoffs.

    The package's exact vector cutoffs (TE0m and TM0m for order 0, HE and EH
    otherwise) at the facet fibre's squared index ratio 1 - 2 contrast: the
    count the reference's own eigenproblem must find.
    """
    ratio = 1.0 - 2.0 * contrast
    kinds = ("TE", "TM") if order == 0 else ("HE", "EH")
    return sum(
        sum(
            1
            for _ in itertools.takewhile(
                lambda m, kind=kind: _cutoff(kind, order, m, ratio) < v,
                itertools.count(1),
            )
        )
        for kind in kinds
    )


def facet_worst_field(contrast, v, order, outer=30.0, count=300):
    """Return one order's largest omitted facet share over every field in the cone.

    A 30a cylinder and 300 TE and 300 TM fields resolve the guided modes of the
    recorded sweep away from the narrow bands just above the LP0m-like cutoffs
    (an 8a cylinder misses HE12 at V = 3.93); the reference's guided count is
    held to the exact cutoffs' first.
    """
    radius, indices = facet_fibre(contrast, v)
    facet = StepIndexFacet(order, radius, indices, FACET_OMEGA, outer * radius, count)
    assert facet.guided().size == guided_count(order, v, contrast)
    return facet.worst_field(math.sin(WIDEST_CONE))


def fresnel_ceiling(contrast):
    """Return the omitted share of a field wholly in the cladding at normal incidence.

    A mode carried entirely by the cladding meets a planar step from the core
    index: the model's pairing over-launches it by 1/(1 - R) and the exact
    transmission gives 1 - R, R the normal-incidence Fresnel reflectance, so the
    omitted share is R(2 - R)/(1 - R), about Delta^2/2. Modes approaching cutoff
    approach it.
    """
    ratio = math.sqrt(1.0 - 2.0 * contrast)
    reflectance = ((1.0 - ratio) / (1.0 + ratio)) ** 2
    return reflectance * (2.0 - reflectance) / (1.0 - reflectance)


FACET_LARGEST = 1.371e-3
"""The largest omitted facet share of the recorded sweep at contrast 0.05: every
field in the cone, order one (HE11, EH11 and HE12 guided) at V = 3.905."""


@pytest.mark.medium
def test_the_vector_step_index_envelope_bounds_the_omitted_facet():
    """The vector model against the full facet solution, over every field in the cone.

    The facet term is the exact guided share minus the model's with the same
    incident H. Summed over one azimuthal order's guided modes it is a Hermitian
    form of the incident field, so its largest eigenvalue modulus inside the
    widest admitted cone bounds every incident field, of any shape and
    polarisation (orders do not mix). At contrast 0.05 the recorded sweep over V
    1.5 to 8, with points just above every cutoff, is largest at V = 3.905
    (1.371e-3, HE12 just resolved above its cutoff) and is 7.4e-4 at V = 1.5
    (HE11 alone); a guided mode wholly in the cladding approaches the
    normal-incidence Fresnel ceiling, 1.39e-3, which no resolved case exceeds.
    The bound is twice that ceiling, numerical error included. Widening the
    cylinder estimates the numerical error, and halving the contrast quarters the
    term: the allowance scales with the squared contrast.
    """
    envelope = VECTOR_STEP_INDEX_ENVELOPE
    top = envelope.ranges["contrast"][1]
    ceiling = fresnel_ceiling(top)
    worst = facet_worst_field(top, 3.905, 1)
    assert worst == pytest.approx(FACET_LARGEST, rel=0.01)
    assert worst <= ceiling
    single = facet_worst_field(top, 1.5, 1)
    assert single == pytest.approx(7.38e-4, rel=0.01)
    wider = facet_worst_field(top, 3.905, 1, outer=45.0, count=450)
    assert abs(wider - worst) <= envelope.numerical
    assert wider <= ceiling
    assert 2 * ceiling + envelope.numerical <= envelope.bound
    half = facet_worst_field(0.5 * top, 3.905, 1)
    assert half / worst == pytest.approx(0.25, rel=0.1)
    assert envelope.allowance({"contrast_squared": (0.5 * top) ** 2}) >= 2 * half


@pytest.mark.medium
def test_a_wide_beam_near_cutoff_lies_within_the_facet_envelope():
    """A centred beam much wider than the core, just above HE12's cutoff.

    An x-polarised Gaussian of waist 10a at V = 3.92 and contrast 0.05, in a 30a
    cylinder: summed over the three guided modes of orders 1 and -1, the model
    over-launches by 1.11e-3 (most of the beam lies in the cladding, where
    the index steps), below the largest over every field and half the bound.
    """
    top = VECTOR_STEP_INDEX_ENVELOPE.ranges["contrast"][1]
    radius, indices = facet_fibre(top, 3.92)
    shares = mode_shares(
        Beam(10.0), radius, indices, FACET_OMEGA, which=None, outer=30.0, count=300
    )
    assert shares.facet < 0.0
    assert abs(shares.facet) == pytest.approx(1.11e-3, rel=0.02)
    assert abs(shares.facet) <= FACET_LARGEST
    assert 2 * abs(shares.facet) <= VECTOR_STEP_INDEX_ENVELOPE.bound


def facet_port(contrast, v_centre):
    """Return a general modal port of a step-index HE11 pair, V at 835 nm given.

    48 azimuthal nodes: with the provider's default nine a narrow guide's node rule
    aliases a sampled beam's oblique columns into a spurious apparent gain.
    """
    wavelength = 835e-9
    radius, (_, core, cladding) = facet_fibre(contrast, v_centre)
    radius *= wavelength / FACET_WAVELENGTH
    grid = FieldGrid.from_limits(1e-3, wavelength, (815e-9, 860e-9), 60e-15)
    provider = make_provider(
        radius=radius,
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        indices=(core, cladding),
    )
    active = grid.omega[grid.active_mask]
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * active[0], 1.01 * active[-1], 17),
        nodes=provider.node_rule(0.99 * active[0], n_r=48, n_theta=48),
        degenerate_blocks=((0, 1),),
    )
    simulation = Simulation.from_waveguide(
        (),
        1e-4,
        provider=guide,
        grid=grid,
        wavelength=wavelength,
        energy=1e-12,
        fwhm=20e-15,
        shape="gauss",
        backend="numpy_ref",
        dz_max=5e-5,
        dz_init=5e-5,
        n_save=2,
        noise=None,
    )
    return simulation.input_port(), radius, (core, core, cladding)


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning")
def test_vector_paraxial_launches_the_facet_reference_within_its_items():
    """``couple`` into a step-index guide against the facet solution, bin by bin.

    One Cartesian plane carries a different beam in each of four bins, so one
    coupling samples V from 1.8 to 2.4. In each bin the launched share equals
    the reference's own evaluation of the model to the guide's tabulation
    (measured to 4e-6); the incident beam's H closure, the model's other omission,
    is within the angular item; and the facet the model omits is within the
    envelope item. The envelope item is the allowance at the plane's weighted
    squared contrast.
    """
    port, radius, indices = facet_port(0.01, 2.4)
    grid = port.grid
    aperture = math.sqrt(indices[1] ** 2 - indices[2] ** 2)
    frequency = np.asarray(port.normalised_frequency)
    active = np.flatnonzero(grid.active_mask)
    beams = {
        int(active[np.argmin(np.abs(frequency[active] - v))]): beam
        for v, beam in (
            (1.8, Beam(2.0)),
            (2.0, Beam(1.5, (0.5, 0.0), 0.3)),
            (2.2, Beam(1.0, (0.3, 0.0))),
            (2.4, Beam(2.0, (0.25, 0.0), 0.2)),
        )
    }

    def field(x, y, omega):
        samples = np.zeros((x.size, omega.size), dtype=complex)
        for k, beam in beams.items():
            j = int(np.argmin(np.abs(omega - grid.omega[k])))
            samples[:, j] = beam.field(x, y, radius, omega[j] / C * aperture)
        return samples

    lowest = min(grid.omega[k] for k in beams)
    # The corner column, sqrt(2) pi / spacing, stays below 0.9 n k at the lowest
    # bin: no column is evanescent in any bin.
    spacing = 0.8 * 2 * math.pi * C / lowest / indices[0]
    half = 8 * radius
    count = 2 * math.ceil(half / spacing)
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        field,
        space=FreeGrid(half, count, half, count),
        index=np.asarray(port.reference_index),
    )
    report = couple(
        plane, port, model=CouplingModel("vector_paraxial", model_error=0.5)
    ).report
    items = {(item.name, item.scope): item for item in report.items}
    envelope = items["envelope", "model"]
    angular = cast("float", items["angular", "field"].value)
    contrast = np.asarray(port.contrast)
    weighted = float(np.sum(contrast**2 * report.incident) / np.sum(report.incident))
    assert envelope.counted
    assert VECTOR_STEP_INDEX_ENVELOPE.identity in envelope.note
    assert envelope.value == pytest.approx(
        VECTOR_STEP_INDEX_ENVELOPE.allowance({"contrast_squared": weighted}), rel=1e-12
    )
    for k, beam in beams.items():
        shares = mode_shares(beam, radius, indices, grid.omega[k], outer=12.0)
        launched = report.launched[k] / report.incident[k]
        # The guide's tables interpolate E/H between 17 knots: the measured
        # difference from the reference's own model evaluation is 4e-6.
        assert abs(launched - shares.model) < 1e-5
        assert abs(shares.model_exact_h - shares.model) <= angular
        assert abs(shares.facet) <= cast("float", envelope.value)
        assert abs(launched - shares.exact) <= cast("float", envelope.value) + angular

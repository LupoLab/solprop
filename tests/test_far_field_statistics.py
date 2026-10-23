# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The far-field aperture statistics' definitions and reference pass (SEP 23).

Section 7 of the proposal, items 1 and 3 to 7 and the request-level refusals of item
12, on saved planes through :func:`solprop.stats.collect` and on the public pieces of
:mod:`solprop.spatial`. The statistics that need the run's operator (items 8 and 9) and
the backend doors are in ``tests/backends/test_far_field_statistics.py``.
"""

import contextlib
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import integrate, special

from bounds import EPS, assert_within, gamma
from solprop.constants import EPS0, C
from solprop.exceptions import ApertureEdgeWarning, OutputError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.output.schema import SavedPlane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    EDGE_MOMENTS,
    ApertureTables,
    SpatialFlux,
    SpatialMedium,
    aperture_profile,
    aperture_scale,
    aperture_statistics,
    check_aperture_profile,
    mask_plane_radius,
    quadrature_half_width,
)
from solprop.stats import (
    MAX_APERTURES,
    MAX_REGIONS,
    ApertureSet,
    StatsSpec,
    check_aperture_choices,
    collect,
)
from solprop.translate.spatial import aperture_tables

DISTANCE = 0.25
"""Exit face to mask plane, m."""


@pytest.fixture(scope="module")
def grid():
    # Every bin of this envelope grid is at positive frequency, above 1.0e15 rad/s.
    return EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 900e-9), 100e-15, thg=False)


def gaussian(grid, space, waist, *, energy=1e-6):
    """A waist-plane Gaussian beam of 1/e^2 intensity radius `waist`, in J."""
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy)
    return SpatialPulse(pulse, waist).build(grid, space, medium=SpatialMedium())


def axial(grid, space):
    """The saved-plane admittance of a vacuum plane, chunk by chunk."""
    flux = SpatialFlux.resolve(grid, space, SpatialMedium())
    return lambda columns: (flux.admittance(columns), None)


def obliquity(grid, space):
    """The vacuum admittance c k_z / omega of every column and bin, independently.

    ``sqrt(1 - (c kappa / omega)**2)`` on active propagating bins at positive
    frequency, and zero on every other bin, which is no forward power port
    (SEP 34 section 4.2).
    """
    columns = np.arange(space.n_columns)
    if isinstance(space, LineGrid):
        kappa_sq = space.k_x[columns] ** 2
    else:
        kappa_sq = (
            space.k_x[columns % space.n_x] ** 2 + space.k_y[columns // space.n_x] ** 2
        )
    omega = grid.omega[None, :]
    square = 1 - kappa_sq[:, None] * (C / np.where(omega > 0, omega, 1.0)) ** 2
    live = grid.active_mask[None, :] & (omega > 0) & (square > 0)
    return np.where(live, np.sqrt(np.where(live, square, 1.0)), 0.0)


def flat(grid, space, field):
    """Divide a spectrum by the root of its admittance, zero where there is none.

    The axial-flux measure weighs ``|E|^2`` by the obliquity; this field's measured
    density is then the plain ``|E|^2`` the paraxial closed forms integrate, so the
    aperture geometry is tested free of the ~1/(k w)^2 obliquity.
    """
    eta = obliquity(grid, space)
    root = np.sqrt(np.where(eta > 0, eta, 1.0))
    return np.where(eta > 0, 1 / root, 0.0)[:, None, :] * field


def share(part, whole):
    """Each bin's share of the beam, NaN on bins that carry no flux at all."""
    return np.divide(
        part,
        whole,
        out=np.full(np.broadcast(part, whole).shape, np.nan),
        where=whole > 0,
    )


def plane(field):
    return SavedPlane(z=0.0, field=field)


def far_field(spec, grid, space, field):
    flux = SpatialFlux.resolve(grid, space, SpatialMedium())
    return collect(spec, grid, plane(field), flux=flux)


def hole(diameter, *, centre=(0.0, 0.0), **options):
    return ApertureSet(
        centres=[list(centre)], diameters=[diameter], distances=[DISTANCE], **options
    )


def unit_spacing(space):
    """The transverse-wavevector lattice spacings, rad/m."""
    return (2 * np.pi / (space.n_x * (space.x[1] - space.x[0])),) + (
        (2 * np.pi / (space.n_y * (space.y[1] - space.y[0])),)
        if space.n_y > 1
        else (1.0,)
    )


# ------------------------------------------------------------------- item 1: profiles


def test_each_profile_takes_its_closed_form_values():
    radius, width = 1.0, 0.125  # binary-exact, so the edge arguments are exact
    # Hard-edge membership is exact (a hard-mask contract).
    assert aperture_profile(
        np.array([0.0, radius, np.nextafter(radius, 2), 3.0]), radius, 0.0, "hard"
    ).tolist() == [1.0, 1.0, 0.0, 0.0]
    # The raised cosine is exactly one inside r - w, one half at r and zero outside
    # r + w, by construction rather than by rounding.
    assert aperture_profile(
        np.array([0.0, 0.875, 1.0, 1.125, 3.0]), radius, width, "rcos"
    ).tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]
    tanh = aperture_profile(np.array([0.0, radius, 1.625]), radius, width, "tanh")
    assert tanh[1] == 0.5  # tanh(0) is zero exactly
    # One tanh, one subtraction and a product with 1/w: a few eps absolutely.
    assert abs(tanh[0] - 0.5 * (1 + math.tanh(8.0))) <= 4 * EPS
    tail = 1 / (1 + math.exp(10.0))  # 0.5 (1 - tanh 5), the exponential tail
    assert abs(tanh[2] - tail) <= 4 * EPS
    assert abs(tanh[2] / math.exp(-10.0) - 1) <= math.exp(-10.0) + 1e-9
    gauss = aperture_profile(np.array([0.0, radius, 10.0]), radius, 4.0, "supergauss")
    assert gauss[0] == 1.0 and gauss[2] == 0.0  # exp(-0) and an underflow, exactly
    assert abs(gauss[1] - math.exp(-1.0)) <= 2 * EPS


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    radii=st.lists(st.floats(0.0, 3.0), min_size=2, max_size=40),
    width=st.floats(0.01, 0.5),
)
def test_the_raised_cosine_is_compact_and_monotone(radii, width):
    rho = np.sort(np.asarray(radii))
    values = aperture_profile(rho, 1.0, width, "rcos")
    assert np.all((values >= 0) & (values <= 1))
    # A sine is monotone on [-pi/2, pi/2]; its rounding may not be, by an ulp.
    assert np.all(np.diff(values) <= 2 * EPS)
    # Outside the edge, with a margin for the argument's rounding, exactly 1 and 0.
    margin = 8 * EPS * (1 + 1 / width)
    assert np.all(values[rho <= (1 - width) * (1 - margin)] == 1.0)
    assert np.all(values[rho >= (1 + width) * (1 + margin)] == 0.0)


@pytest.mark.parametrize("kind", ["tanh", "rcos"])
def test_the_symmetric_edges_sum_to_one(kind):
    radius, width = 2e-3, 1.7e-4
    offset = np.linspace(0.0, radius, 257)
    total = aperture_profile(radius + offset, radius, width, kind) + aperture_profile(
        radius - offset, radius, width, kind
    )
    # Each argument (rho - r)/w carries eps (r + 2|u|)/w of rounding through a slope
    # of at most pi/4; each profile adds an eps, and the sum one more.
    bound = 4 * EPS * (1 + (radius + 2 * offset) / width)
    assert np.all(np.abs(total - 1) <= bound)


def test_the_edge_moments_match_quadrature():
    tanh, _ = integrate.quad(
        lambda t: t * special.expit(-2 * t), 0, np.inf, epsabs=0, epsrel=1e-13
    )
    rcos, _ = integrate.quad(
        lambda t: t * 0.5 * (1 - math.sin(math.pi * t / 2)),
        0,
        1,
        epsabs=0,
        epsrel=1e-13,
    )
    # SEP 23 section 7 item 1: m_2 against SciPy quadrature to 1e-12.
    assert abs(tanh / EDGE_MOMENTS["tanh"] - 1) < 1e-12
    assert abs(rcos / EDGE_MOMENTS["rcos"] - 1) < 1e-12
    assert EDGE_MOMENTS["hard"] == 0.0


@pytest.mark.parametrize("kind", ["tanh", "rcos"])
def test_the_corrected_diameter_restores_the_hard_area(kind):
    radius, width = 1.0, 0.1
    corrected = float(quadrature_half_width(radius, width, kind))
    area, _ = integrate.quad(
        lambda rho: (
            float(aperture_profile(rho, corrected, width, kind)) * 2 * np.pi * rho
        ),
        0,
        radius + 40 * width,
        points=[corrected - width, corrected, corrected + width],
        epsabs=0,
        epsrel=1e-13,
        limit=200,
    )
    # Exact for rcos and to the tanh tail (1e-11 at w = D/20); SEP 23 section 4.5's
    # drafting check found residuals below 2e-10 at w = r/10.
    assert abs(area / (np.pi * radius**2) - 1) < 2e-10
    # Without the correction the area is pi (r^2 + sigma^2), sigma^2 = 4 m_2 w^2.
    plain, _ = integrate.quad(
        lambda rho: float(aperture_profile(rho, radius, width, kind)) * 2 * np.pi * rho,
        0,
        radius + 40 * width,
        points=[radius - width, radius, radius + width],
        epsabs=0,
        epsrel=1e-13,
        limit=200,
    )
    sigma2 = 4 * EDGE_MOMENTS[kind] * width**2
    assert abs(plain / (np.pi * (radius**2 + sigma2)) - 1) < 2e-10


def test_profile_and_correction_refusals():
    # The type checker refuses a misspelt profile in the test session; the named
    # check is what a caller outside it meets.
    with pytest.raises(OutputError, match="unknown aperture profile"):
        check_aperture_profile("square")
    with pytest.raises(OutputError, match="unknown aperture weighting"):
        check_aperture_choices("tanh", "intensity")
    check_aperture_choices("hard", "amplitude")
    with pytest.raises(OutputError, match="half-width"):
        aperture_profile(1.0, 0.0, 0.1, "tanh")
    with pytest.raises(OutputError, match="order of at least 2"):
        aperture_profile(1.0, 1.0, 1.5, "supergauss")
    with pytest.raises(OutputError, match="positive edge width"):
        aperture_profile(1.0, 1.0, -0.1, "rcos")
    with pytest.raises(OutputError, match="symmetric edge"):
        quadrature_half_width(1.0, 0.1, "supergauss")
    with pytest.raises(OutputError, match="4 m_2 w"):
        quadrature_half_width(0.1, 1.0, "tanh")
    assert quadrature_half_width(np.ones(2), 0.3, "hard").tolist() == [1.0, 1.0]


def test_the_scale_table_and_the_radius():
    omega = np.array([-2.0, 0.0, 3.0e15, 6.0e15])
    table = aperture_scale(np.array([0.25, 0.5]), omega)
    # Zero at and below zero frequency is a structural contract.
    assert np.all(table[:, :2] == 0.0)
    assert_within(table[:, 2:], np.outer([0.25, 0.5], C / omega[2:]), 2 * EPS)
    # The stored axis is the negated physical wavevector: a column stored at
    # -X/s images onto the centre X exactly (exact products of powers of two).
    radius = mask_plane_radius(np.array([2.0]), np.array([-0.5]), np.zeros(1), (1, 0))
    assert radius.tolist() == [[0.0]]


def test_tables_validate_their_shapes_and_derived_constants():
    shared = {
        "k_x": np.zeros(2),
        "k_y": np.zeros(1),
        "weights": np.ones(2),
        "on_axis": np.ones(2, complex),
        "regions": np.empty((0, 4)),
        "centres": np.zeros((1, 2)),
        "half_widths": np.ones(1),
        "edges": np.full(1, 0.25),
        "distances": np.ones(1),
        "scale": np.ones((1, 3)),
    }
    tables = ApertureTables(**shared, profile="tanh", exponent=1)
    assert (tables.n_columns, tables.n_apertures, tables.n_regions) == (2, 1, 0)
    assert tables.profile_code == 1
    assert tables.inverse_widths.tolist() == [4.0]
    assert ApertureTables(**shared).inverse_widths.tolist() == [0.0]
    with pytest.raises(OutputError, match="disagree in shape"):
        ApertureTables(**{**shared, "weights": np.ones(3)})
    with pytest.raises(OutputError, match="exponent of 1 or 2"):
        ApertureTables(**shared, exponent=3)


# ------------------------------------------------------ item 3: every weight is one


def test_a_hole_enclosing_every_image_is_the_beam(grid):
    space = FreeGrid(3e-4, 32, 2.5e-4, 24, window_factor=0.1)
    field = gaussian(grid, space, 60e-6)
    rng = np.random.default_rng(23)
    field = field * (1 + 0.2 * rng.standard_normal(field.shape))
    k_max = math.hypot(np.max(np.abs(space.k_x)), np.max(np.abs(space.k_y)))
    assert k_max < np.min(grid.omega) / C  # every column propagates at every bin
    diameter = 0.1  # wider than every image, s_max k_max = 0.016 m
    spec = StatsSpec(
        names=("beam_spectrum", "aperture_spectrum", "aperture_on_axis_spectrum"),
        apertures=hole(diameter),
    )
    values = far_field(spec, grid, space, field)
    tables = aperture_tables(spec, grid, space)
    assert tables is not None
    columns = np.arange(space.n_columns)
    k_x, k_y = space.k_x[columns % space.n_x], space.k_y[columns // space.n_x]
    radius = mask_plane_radius(tables.scale[0], k_x, k_y, (0.0, 0.0))
    # Every weight is exactly one (hard-edge membership and the paraxial disc).
    assert np.all(aperture_profile(radius, diameter / 2, 0.0, "hard") == 1.0)
    assert np.all(radius < DISTANCE)
    # The same sum of positive terms on both sides: the summation bound.
    assert_within(
        values["aperture_spectrum"][0],
        values["beam_spectrum"],
        gamma(2 * (space.n_columns + 8)),
    )
    axis = collect(
        StatsSpec(names=("on_axis_spectrum",)),
        grid,
        plane(field),
        flux=SpatialFlux.resolve(grid, space, SpatialMedium()),
    )
    amplitude = np.einsum("r,rpw->pw", space.on_axis, field)
    magnitude = np.einsum("r,rpw->pw", np.abs(space.on_axis), np.abs(field))
    # A chunked complex sum against one contraction: each amplitude within
    # gamma(2n) of its magnitude sum, and the density quadratic in it.
    error = gamma(2 * (space.n_columns + 8)) * magnitude
    factor = 0.5 * EPS0 * C * grid.dt**2 / (2 * np.pi)
    bound = factor * np.sum(2 * np.abs(amplitude) * error + error**2, axis=0)
    assert_within(
        values["aperture_on_axis_spectrum"][0],
        axis["on_axis_spectrum"],
        bound + 4 * EPS * np.abs(axis["on_axis_spectrum"]),
        elementwise=True,
    )


# -------------------------------------------------- items 2 and 4: Gaussian far field


@pytest.fixture(scope="module")
def far_beam(grid):
    waist = 50e-6
    # A box of forty waists sampled at a quarter waist: the Gaussian's truncation
    # (exp(-400)) and aliasing (exp(-79)) are below rounding. The mask-plane lattice
    # is then pi w0 / L = 0.079 of the far-field radius at every frequency.
    space = FreeGrid(20 * waist, 160, 20 * waist, 160, window_factor=0)
    return space, waist, flat(grid, space, gaussian(grid, space, waist))


def far_radius(grid, waist):
    """The Gaussian's far-field 1/e^2 intensity radius at each bin, m."""
    return 2 * DISTANCE * C / (grid.omega * waist)


def mask_spacing(grid, space):
    """The lattice spacing in the mask plane at each bin, m (x and y alike)."""
    return DISTANCE * C / grid.omega * unit_spacing(space)[0]


def edge_fraction(grid, space, field, diameter):
    """Each bin's share of the beam within one lattice diagonal of a hard edge."""
    columns = np.arange(space.n_columns)
    k_x, k_y = space.k_x[columns % space.n_x], space.k_y[columns // space.n_x]
    scale = DISTANCE * C / grid.omega
    radius = np.hypot(scale[None, :] * k_x[:, None], scale[None, :] * k_y[:, None])
    near = np.abs(radius - diameter / 2) <= math.sqrt(2) * mask_spacing(grid, space)
    density = (
        space.spectral_weights[:, None]
        * obliquity(grid, space)
        * np.sum(np.abs(field) ** 2, axis=1)
    )
    return share(np.sum(density * near, axis=0), np.sum(density, axis=0))


def test_a_centred_hard_hole_collects_the_gaussian_far_field_fraction(grid, far_beam):
    space, waist, field = far_beam
    diameter = 2 * far_radius(grid, waist)[grid.n_omega // 2]
    spec = StatsSpec(
        names=("beam_spectrum", "aperture_spectrum"), apertures=hole(diameter)
    )
    values = far_field(spec, grid, space, field)
    fraction = share(values["aperture_spectrum"][0], values["beam_spectrum"])
    expected = 1 - np.exp(-(diameter**2) / (2 * far_radius(grid, waist) ** 2))
    # Exact for the paraxial map: the hard edge's staircase, bounded by the energy
    # within one lattice diagonal of the edge, plus rounding.
    bound = edge_fraction(grid, space, field, diameter) + gamma(8 * space.n_columns)
    live = values["beam_spectrum"] > 0  # the bins with a forward measure
    assert np.all(np.abs(fraction - expected)[live] <= bound[live])
    assert np.max(expected) - np.min(expected) > 0.1  # the hole is chromatic


@pytest.mark.parametrize("weighting", ["amplitude", "quadrature"])
def test_a_smooth_hole_matches_the_profile_integral(grid, far_beam, weighting):
    space, waist, field = far_beam
    radii = far_radius(grid, waist)
    spacing = mask_spacing(grid, space)
    reference = radii[int(np.argmin(np.abs(grid.omega - grid.omega0)))]
    # An off-axis hole whose edge crosses the beam: the widest edge each weighting
    # admits (D/4, D/8). Off axis, because a tanh profile is not flat at its own
    # centre -- W'(0) = -sech^2(r/w)/(2w) -- so W(|x - X|) has a cone there whose
    # lattice error falls only as dx_m^3; here the cone sits where the beam is below
    # exp(-18) of its peak at every bin compared.
    offset, diameter = 5 * reference, 8 * reference
    width = diameter / (4 if weighting == "amplitude" else 8)
    # Amplitude tanh at r/w = 2 warns, naming rcos, and still builds.
    with (
        pytest.warns(ApertureEdgeWarning, match="rcos")
        if weighting == "amplitude"
        else contextlib.nullcontext()
    ):
        apertures = hole(
            diameter,
            centre=(offset, 0.0),
            edges=[width],
            profile="tanh",
            weighting=weighting,
        )
    spec = StatsSpec(names=("beam_spectrum", "aperture_spectrum"), apertures=apertures)
    values = far_field(spec, grid, space, field)
    fraction = share(values["aperture_spectrum"][0], values["beam_spectrum"])
    half = float(
        quadrature_half_width(diameter / 2, width, "tanh")
        if weighting == "quadrature"
        else diameter / 2
    )
    power = 1 if weighting == "quadrature" else 2
    compared = np.flatnonzero(
        (width >= 4 * spacing) & (offset >= 3 * radii) & (values["beam_spectrum"] > 0)
    )
    assert compared.size > grid.n_omega // 2
    for index in compared:
        w_ff = radii[index]
        # The Gaussian's intensity averaged around a circle of radius rho about the
        # hole's centre: its closed angular integral, a scaled Bessel I0.
        expected, _ = integrate.quad(
            lambda rho, w_ff=w_ff: (
                float(aperture_profile(rho, half, width, "tanh")) ** power
                * 4
                * rho
                / w_ff**2
                * math.exp(-2 * (offset - rho) ** 2 / w_ff**2)
                * special.i0e(4 * offset * rho / w_ff**2)
            ),
            0,
            offset + 12 * w_ff,
            points=[half - width, half, half + width, offset],
            epsabs=1e-15,
            epsrel=1e-13,
            limit=400,
        )
        assert expected > 1e-3  # the edge is in the beam
        # The smooth edge aliases at exp(-pi^2 w / dx_m) < exp(-120); the cone at
        # the centre contributes below exp(-18) (dx_m/w_ff)^3 = 8e-12; the
        # Gaussian's own aliasing and truncation are below rounding; quad 1e-13;
        # rounding gamma(8 n) = 5e-11. 2e-10 is their sum with a margin.
        assert abs(fraction[index] - expected) < 2e-10, index


def test_a_gaussian_mask_passes_its_closed_form_share(grid, far_beam):
    space, waist, field = far_beam
    radii = far_radius(grid, waist)
    radius = radii[int(np.argmin(np.abs(grid.omega - grid.omega0)))]
    mask = hole(2 * radius, edges=[2.0], profile="supergauss")
    spec = StatsSpec(
        names=("beam_spectrum", "aperture_spectrum", "aperture_on_axis_spectrum"),
        apertures=mask,
    )
    values = far_field(spec, grid, space, field)
    fraction = share(values["aperture_spectrum"][0], values["beam_spectrum"])
    # A Gaussian amplitude mask exp(-(rho/r)^2) on a Gaussian beam passes the share
    # r^2 / (r^2 + w_ff^2) of its energy: a smooth integrand, so the lattice sum's
    # aliasing is below rounding, which gamma(8 n) bounds.
    expected = radius**2 / (radius**2 + radii**2)
    live = values["beam_spectrum"] > 0
    assert np.all(np.abs(fraction[live] - expected[live]) < gamma(8 * space.n_columns))
    alone = far_field(
        StatsSpec(names=("aperture_on_axis_spectrum",), apertures=mask),
        grid,
        space,
        field,
    )
    assert_within(
        alone["aperture_on_axis_spectrum"],
        values["aperture_on_axis_spectrum"],
        gamma(4 * space.n_columns),
    )


def test_a_tilted_beam_lands_at_its_angle_at_every_frequency(grid):
    waist, angle = 60e-6, 20e-3
    # Ten waists sampled at about a tenth of one: the tilt and the beam's spectrum
    # stay inside the lattice at every in-band bin.
    space = FreeGrid(3e-4, 96, 3e-4, 64, window_factor=0)
    x, y = np.meshgrid(space.x, space.y)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    profile = np.exp(-(x**2 + y**2) / waist**2)
    # The crossing-beam lesson's +x beam: internal phase exp(-i kappa x), with the
    # achromatic kappa = omega sin(theta) / c.
    tilt = np.exp(-1j * x[..., None] * grid.omega * np.sin(angle) / C)
    physical = profile[..., None] * tilt * temporal[0, 0]
    field = np.fft.fft2(physical, axes=(0, 1)).reshape(-1, 1, grid.n_omega)
    image = DISTANCE * np.sin(angle)
    band = grid.inband_mask
    radii = 2 * DISTANCE * C / (grid.omega * waist)
    diameter = 6 * np.max(radii[band])  # three spot radii at the lowest frequency
    holes = ApertureSet(
        centres=[[image, 0.0], [-image, 0.0]],
        diameters=[diameter, diameter],
        distances=[DISTANCE, DISTANCE],
        labels=("towards +x", "mirror"),
    )
    spec = StatsSpec(names=("beam_spectrum", "aperture_spectrum"), apertures=holes)
    values = far_field(spec, grid, space, field)
    collected = values["aperture_spectrum"][:, band] / values["beam_spectrum"][band]
    diagonal = math.sqrt(2) * mask_spacing(grid, space)[band]
    # The hole at +X takes all but the energy beyond three spot radii less a lattice
    # diagonal (0.44 of one), exp(-2 x 2.56^2) < 1e-5, at every in-band frequency.
    assert np.all(collected[0] > 1 - 1e-5)
    # The mirror, 2X away, takes only the tail beyond its edge less a diagonal, and
    # the transform's rounding floor: |E|^2 at eps^2 of the peak on its ~1e3 columns.
    tail = np.exp(-2 * ((2 * image - diameter / 2 - diagonal) / radii[band]) ** 2)
    assert np.all(collected[1] <= tail + 1e-20)


def test_a_fixed_wavevector_moves_as_one_over_omega(grid):
    waist = 300e-6
    space = FreeGrid(1.5e-3, 96, 1.5e-3, 32, window_factor=0)
    kappa = -space.k_x[-24]  # a lattice wavevector, physical +x side
    assert kappa > 0
    x, y = np.meshgrid(space.x, space.y)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    physical = (np.exp(-(x**2 + y**2) / waist**2) * np.exp(-1j * kappa * x))[
        ..., None
    ] * temporal[0, 0]
    field = np.fft.fft2(physical, axes=(0, 1)).reshape(-1, 1, grid.n_omega)
    band = np.flatnonzero(grid.inband_mask)
    high = band[np.argmax(grid.omega[band])]
    low = band[np.argmin(grid.omega[band])]
    images = DISTANCE * C * kappa / grid.omega[[high, low]]  # 1.4 mm and 1.8 mm
    holes = ApertureSet(
        centres=[[images[0], 0.0], [images[1], 0.0]],
        diameters=[3e-4, 3e-4],
        distances=[DISTANCE, DISTANCE],
    )
    spec = StatsSpec(
        names=("beam_spectrum", "aperture_spectrum", "region_spectrum"),
        apertures=holes,
        spectral_regions=np.array([[0.0, np.inf, -np.inf, np.inf]]),
    )
    values = far_field(spec, grid, space, field)
    fraction = share(values["aperture_spectrum"], values["beam_spectrum"])
    # The image of one wavevector lies at z c kappa / omega: the hole placed for the
    # highest in-band frequency collects it there and not at the lowest, and the
    # other way round.
    assert fraction[0, high] > 0.5 > fraction[1, high]
    assert fraction[1, low] > 0.4 > fraction[0, low]
    # A region selects the same wavevector at every frequency: the beam's half plane.
    in_band = grid.inband_mask
    assert_within(
        values["region_spectrum"][0, in_band] / values["beam_spectrum"][in_band],
        np.ones(np.sum(in_band)),
        1e-12,
    )


# ------------------------------------------------- item 5: quadrature unbiasedness


def test_quadrature_weighting_collects_the_hard_area():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (780e-9, 820e-9), 40e-15, thg=False)
    # 128 lattice cells either side of the axis: the disc of 40 and its tanh tail to
    # 15 widths (exp(-30)) fit inside.
    space = FreeGrid(1e-3, 256, 1e-3, 256, window_factor=0)
    field = flat(grid, space, np.ones((space.n_columns, 1, grid.n_omega), complex))
    # The carrier's bin: an active one, where the plane has a forward measure.
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    scale = DISTANCE * C / grid.omega[index]
    delta_x, delta_y = unit_spacing(space)
    spacing = scale * delta_x  # the mask-plane lattice at the chosen bin
    radius, width = 40 * spacing, 4 * spacing

    def area(apertures=None, tables=None):
        if tables is None:
            spec = StatsSpec(names=("aperture_spectrum",), apertures=apertures)
            tables = aperture_tables(spec, grid, space)
        assert tables is not None
        values = aperture_statistics(
            ("beam_spectrum", "aperture_spectrum"),
            grid,
            field,
            tables,
            axial=axial(grid, space),
        )
        ratio = values["aperture_spectrum"][0, index] / values["beam_spectrum"][index]
        return ratio * space.n_columns * scale**2 * delta_x * delta_y

    hard_area = np.pi * radius**2
    options = {"edges": [width], "profile": "tanh"}
    corrected = area(hole(2 * radius, weighting="quadrature", **options))
    # The tanh tail's residual at w = D/20 (1e-11), the smooth edge's lattice
    # aliasing exp(-pi^2 w/dx_m) over its 250 cells of circumference (1e-15) and
    # rounding gamma(4 n): 1e-9 with a margin.
    assert abs(corrected / hard_area - 1) < 1e-9
    sigma2 = 4 * EDGE_MOMENTS["tanh"] * width**2
    tables = aperture_tables(
        StatsSpec(names=("aperture_spectrum",), apertures=hole(2 * radius, **options)),
        grid,
        space,
    )
    assert tables is not None
    plain = ApertureTables(
        **{
            name: getattr(tables, name)
            for name in (
                "k_x",
                "k_y",
                "weights",
                "on_axis",
                "regions",
                "centres",
                "half_widths",
                "edges",
                "distances",
                "scale",
            )
        },
        profile="tanh",
        exponent=1,
    )
    # The profile used once at D, without the correction, collects pi sigma^2 more.
    assert abs(area(tables=plain) / (hard_area + np.pi * sigma2) - 1) < 1e-9
    amplitude = area(hole(2 * radius, **options))
    exact, _ = integrate.quad(
        lambda rho: (
            float(aperture_profile(rho, radius, width, "tanh")) ** 2 * 2 * np.pi * rho
        ),
        0,
        radius + 40 * width,
        points=[radius],
        epsabs=0,
        epsrel=1e-13,
        limit=200,
    )
    assert abs(amplitude / exact - 1) < 1e-9
    # Under W^2 the first-order deficit is the edge length times w/2; the next order
    # is w/r = 0.1 of it.
    deficit = hard_area - amplitude
    assert abs(deficit / (2 * np.pi * radius * width / 2) - 1) < 0.1


# ------------------------------------------------------ item 6: Parseval and end bins


def test_the_beam_spectrum_integrates_to_the_energy(grid):
    space = FreeGrid(3e-4, 32, 3e-4, 24, window_factor=0.1)
    field = gaussian(grid, space, 60e-6)
    spec = StatsSpec(names=("energy", "beam_spectrum"))
    values = far_field(spec, grid, space, field)
    terms = space.n_columns * grid.n_omega
    # Two orders of a sum of m positive terms with their roundings: 2 gamma_m.
    assert_within(
        np.sum(values["beam_spectrum"]) * grid.d_omega,
        values["energy"],
        2 * gamma(terms + 16),
    )


def test_occupied_end_bins_carry_no_axial_flux():
    # The end bins' analytic-signal weight differs from the interior's, so a plain
    # spectral density over every bin would overcount them; the axial-flux measure
    # admits no zero-frequency or inactive bin as a forward power port (SEP 34
    # section 4.3), so occupied ends leave both statistics alone and the beam
    # spectrum integrates to the energy.
    grid = FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 60e-15)
    space = LineGrid(2e-4, 16, window_factor=0)
    rng = np.random.default_rng(5)
    field = rng.standard_normal((space.n_columns, 1, grid.n_omega)) + 0j
    assert not grid.active_mask[0] and not grid.active_mask[-1]
    spec = StatsSpec(names=("energy", "beam_spectrum"))
    values = far_field(spec, grid, space, field)
    inner = field.copy()
    inner[:, :, [0, -1]] = 0
    cleared = far_field(spec, grid, space, inner)
    assert values["energy"] == cleared["energy"]
    assert np.array_equal(values["beam_spectrum"], cleared["beam_spectrum"])
    # Two orders of a sum of m positive terms with their roundings: 2 gamma_m.
    assert_within(
        np.sum(values["beam_spectrum"]) * grid.d_omega,
        values["energy"],
        2 * gamma(space.n_columns * grid.n_omega + 16),
    )


# -------------------------------------------------------------- item 7: regions


def test_quadrants_tile_the_beam(grid):
    space = FreeGrid(3e-4, 32, 2.5e-4, 24, window_factor=0.1)
    rng = np.random.default_rng(7)
    field = gaussian(grid, space, 60e-6) * (1 + rng.standard_normal((1, 1, 1)))
    quadrants = np.array(
        [
            [-np.inf, 0.0, -np.inf, 0.0],
            [0.0, np.inf, -np.inf, 0.0],
            [-np.inf, 0.0, 0.0, np.inf],
            [0.0, np.inf, 0.0, np.inf],
        ]
    )
    spec = StatsSpec(
        names=("beam_spectrum", "region_spectrum"), spectral_regions=quadrants
    )
    values = far_field(spec, grid, space, field)
    assert_within(
        np.sum(values["region_spectrum"], axis=0),
        values["beam_spectrum"],
        2 * gamma(space.n_columns + 16),
    )


def single_column(space, grid, column):
    field = np.zeros((space.n_columns, 1, grid.n_omega), dtype=complex)
    field[column, 0, :] = 1.0
    return field


def test_region_membership_is_exact_at_zero_nyquist_and_an_edge(grid):
    space = FreeGrid(3e-4, 16, 2.5e-4, 8, window_factor=0.1)  # even: a Nyquist column
    nyquist = space.n_x // 2
    assert space.k_x[nyquist] < 0  # stored -pi/dx, physical +pi/dx
    edge = -space.k_x[13]  # a physical wavevector on the lattice, 3 cells out
    regions = np.array(
        [
            [-np.inf, 0.0, -np.inf, np.inf],
            [0.0, edge, -np.inf, np.inf],
            [edge, -space.k_x[nyquist], -np.inf, np.inf],
            [-space.k_x[nyquist], np.inf, -np.inf, np.inf],
        ]
    )
    spec = StatsSpec(
        names=("beam_spectrum", "region_spectrum"), spectral_regions=regions
    )
    # Column 0 carries kappa = 0, which belongs to [0, ...); column 13 lies on the
    # edge `edge`, which belongs to the region it opens; the Nyquist column to the
    # last, non-negative region; column 15 (stored -dk) to the second and column 1
    # (stored +dk) to the first.
    for column, member in ((0, 1), (13, 2), (nyquist, 3), (15, 1), (1, 0)):
        values = far_field(spec, grid, space, single_column(space, grid, column))
        others = np.delete(values["region_spectrum"], member, axis=0)
        assert np.all(others == 0.0), column  # hard-mask membership: exact zeros
        assert_within(values["region_spectrum"][member], values["beam_spectrum"], EPS)


# ---------------------------------------------------------------- the line


def test_a_slit_on_a_line_and_its_refusals(grid):
    space = LineGrid(4e-4, 64, window_factor=0)
    field = gaussian(grid, space, 60e-6)
    k_max = np.max(np.abs(space.k_x))
    slit = ApertureSet(
        centres=[[0.0, 0.0]],
        diameters=[0.1],
        distances=[DISTANCE],
        edges=[0.002],
        profile="tanh",
        weighting="quadrature",
    )
    spec = StatsSpec(
        names=("beam_spectrum", "aperture_spectrum", "region_spectrum"),
        apertures=slit,
        spectral_regions=np.array([[-np.inf, np.inf, -1.0, 1.0]]),
    )
    tables = aperture_tables(spec, grid, space)
    assert tables is not None
    assert tables.half_widths.tolist() == [0.05]  # straight edges: D_q = D
    values = far_field(spec, grid, space, field)
    # Every image lies at most s_max k_max = 0.018 m out, 16 edge widths inside the
    # slit's edge, where the tanh weight differs from one by exp(-32) = 1.3e-14.
    assert DISTANCE * C / np.min(grid.omega) * k_max < 0.018
    assert_within(values["aperture_spectrum"][0], values["beam_spectrum"], 1e-12)
    assert_within(values["region_spectrum"][0], values["beam_spectrum"], gamma(256))
    off_axis = ApertureSet(centres=[[0.0, 1e-3]], diameters=[1e-3], distances=[0.25])
    with pytest.raises(OutputError, match="y centre must be zero"):
        aperture_tables(
            StatsSpec(names=("aperture_spectrum",), apertures=off_axis), grid, space
        )
    with pytest.raises(OutputError, match="would select nothing"):
        aperture_tables(
            StatsSpec(
                names=("region_spectrum",),
                spectral_regions=np.array([[-1.0, 1.0, 0.5, 1.0]]),
            ),
            grid,
            space,
        )


# ----------------------------------------------------- item 12: request refusals


def apertures(**changes):
    options = {
        "centres": [[1e-3, 0.0]],
        "diameters": [1e-3],
        "distances": [0.25],
        "edges": [1e-4],
        "profile": "tanh",
    }
    options.update(changes)
    return ApertureSet(**options)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"diameters": []}, "1 to 16"),
        ({"diameters": [1e-3] * (MAX_APERTURES + 1)}, "1 to 16"),
        ({"diameters": np.array([[1e-3], [1e-3, 2e-3]], dtype=object)}, "numeric"),
        ({"diameters": np.ones((1, 1))}, "a flat vector"),
        ({"centres": [[1e-3, 0.0, 0.0]]}, "centres must be finite"),
        ({"distances": [np.nan]}, "distances must be finite"),
        ({"centres": np.array([["a", "b"]], dtype=object)}, "must be numeric"),
        ({"diameters": [-1e-3]}, "positive"),
        ({"profile": "hard"}, "silently unused"),
        ({"edges": None}, "needs edges"),
        ({"edges": [3e-4]}, "D/4"),
        ({"edges": [2e-4], "weighting": "quadrature"}, "D/8"),
        ({"edges": [0.0], "profile": "rcos"}, "D/4"),
        ({"edges": [1.0], "profile": "supergauss"}, "at least 2"),
        (
            {"edges": [4.0], "profile": "supergauss", "weighting": "quadrature"},
            "symmetric",
        ),
        ({"centres": [[0.2, 0.0]], "diameters": [0.1], "edges": [0.01]}, "paraxial"),
        (
            {
                "centres": [[0.2, 0.0]],
                "diameters": [0.09],
                "edges": [0.02],
                "profile": "rcos",
            },
            "paraxial",
        ),
        ({"labels": ("a", "b")}, "unique non-empty"),
        ({"labels": ("",)}, "unique non-empty"),
    ],
)
def test_an_aperture_set_refuses_by_name(changes, message):
    with pytest.raises(OutputError, match=message):
        apertures(**changes)


def test_a_wide_amplitude_tanh_edge_warns_once_naming_rcos():
    # r = 5e-4 m: r/w = 2 and 3.9 warn, in one warning naming both apertures; the set
    # is still built as asked.
    with pytest.warns(ApertureEdgeWarning, match=r"apertures \[0, 2\].*rcos") as caught:
        wide = apertures(
            centres=[[1e-3, 0.0]] * 3,
            diameters=[1e-3] * 3,
            distances=[0.25] * 3,
            edges=[2.5e-4, 1e-4, 5e-4 / 3.9],
        )
    assert len(caught) == 1
    assert wide.n_apertures == 3 and wide.profile == "tanh"


@pytest.mark.parametrize(
    "changes",
    [
        {"edges": [1.25e-4]},  # r/w = 4 exactly
        {"edges": [1.25e-4], "weighting": "quadrature"},
        {"edges": [2.5e-4], "profile": "rcos"},
        {"edges": [2.0], "profile": "supergauss"},
        {"edges": None, "profile": "hard"},
    ],
)
def test_other_edges_build_without_a_warning(changes):
    # Warnings are errors in the test session: building is the assertion.
    assert apertures(**changes).n_apertures == 1


def test_an_aperture_set_owns_its_tables_and_compares_by_content():
    centres = np.array([[1e-3, 0.0]])
    first = apertures(centres=centres)
    centres[0, 0] = 2e-3
    assert np.asarray(first.centres)[0, 0] == 1e-3
    assert not np.asarray(first.centres).flags.writeable
    assert first.labels == ("0",) and first.n_apertures == 1
    assert first.same_as(apertures()) and not first.same_as(None)
    assert not first.same_as(apertures(edges=[5e-5]))
    assert not first.same_as(apertures(profile="hard", edges=None))
    hard = apertures(profile="hard", edges=None)
    assert hard.same_as(apertures(profile="hard", edges=None))
    assert first != apertures()  # identity-compared, like StatsSpec


def test_a_spec_refuses_a_missing_or_malformed_table():
    with pytest.raises(OutputError, match="aperture_on_axis_spectrum"):
        StatsSpec(names=("aperture_on_axis_spectrum",))
    with pytest.raises(OutputError, match="region_spectrum"):
        StatsSpec(names=("region_spectrum",))
    with pytest.raises(OutputError, match="at most 16"):
        StatsSpec(spectral_regions=np.tile([0.0, 1.0, 0.0, 1.0], (MAX_REGIONS + 1, 1)))
    with pytest.raises(OutputError, match="NaN"):
        StatsSpec(spectral_regions=np.array([[np.nan, 1.0, 0.0, 1.0]]))
    with pytest.raises(OutputError, match=r"rows \[0\]"):
        StatsSpec(spectral_regions=np.array([[0.0, 1.0, 1.0, 1.0]]))
    with pytest.raises(OutputError, match="between save planes"):
        StatsSpec(names=("beam_spectrum",), periodic=("beam_spectrum",))
    spec = StatsSpec(names=("energy",), apertures=apertures())
    assert (spec.n_apertures, spec.n_regions, StatsSpec().n_apertures) == (1, 0, 0)
    assert spec.same_tables(StatsSpec(apertures=apertures()))
    assert not spec.same_tables(StatsSpec())
    assert not StatsSpec().same_tables(spec)


def test_collect_refuses_the_operator_statistics_and_radial_grids(grid):
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0.1)
    field = gaussian(grid, space, 60e-6)
    for name in ("absorbed_power",):
        with pytest.raises(OutputError, match=f"{name}.*linear operator"):
            far_field(StatsSpec(names=(name,)), grid, space, field)
    radial = RadialGrid(3e-4, 16)
    with pytest.raises(OutputError, match="radial grid"):
        collect(
            StatsSpec(names=("beam_spectrum",)),
            grid,
            plane(np.zeros((16, 1, grid.n_omega), complex)),
            flux=SpatialFlux.resolve(grid, radial, SpatialMedium()),
        )
    with pytest.raises(OutputError, match="requires spatial geometry"):
        collect(StatsSpec(names=("beam_spectrum",)), grid, plane(field))
    with pytest.raises(OutputError, match="not a spatial saved-field statistic"):
        far_field(StatsSpec(names=("omega0",)), grid, space, field)


def test_the_pass_refuses_what_its_tables_cannot_answer(grid):
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0.1)
    field = gaussian(grid, space, 60e-6)
    tables = aperture_tables(StatsSpec(names=("beam_spectrum",)), grid, space)
    assert tables is not None
    for names, message in (
        (("energy",), "not far-field statistics"),
        (("aperture_spectrum",), "need apertures"),
        (("region_spectrum",), "needs spectral_regions"),
        (("absorbed_power",), "linear operator"),
    ):
        with pytest.raises(OutputError, match=message):
            aperture_statistics(names, grid, field, tables, axial=axial(grid, space))
    with pytest.raises(OutputError, match="matching its tables"):
        aperture_statistics(
            ("beam_spectrum",), grid, field[:-1], tables, axial=axial(grid, space)
        )
    with pytest.raises(OutputError, match="at least one column"):
        aperture_statistics(
            ("beam_spectrum",), grid, field, tables, axial=axial(grid, space), chunk=0
        )
    assert aperture_tables(StatsSpec(names=("energy",)), grid, space) is None
    # Chunked or whole, the beam spectrum agrees to the summation bound.
    whole = aperture_statistics(
        ("beam_spectrum",), grid, field, tables, axial=axial(grid, space)
    )
    chunked = aperture_statistics(
        ("beam_spectrum",), grid, field, tables, axial=axial(grid, space), chunk=5
    )
    assert_within(
        chunked["beam_spectrum"], whole["beam_spectrum"], gamma(2 * space.n_columns)
    )

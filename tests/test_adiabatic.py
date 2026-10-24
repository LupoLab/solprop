# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the adiabatic path certificate against an independent scalar-wave witness.

Each profile is given once as a wave-number square ``q^2 = k0^2 (n0^2 + rho(z))``:
to the certificate as a density path ``rho`` with susceptibility one, and to the
witness (``tests/oracles/scalar_wave.py``) as ``q`` and ``q'/q``. The witness solves
the full scalar equation, so the certificate's bounds must hold against it within
the witness's own refinement estimate.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from bounds import EPS
from oracles.scalar_wave import (
    Profile,
    Scattering,
    epstein_reflectance,
    fresnel_step,
    scatter,
)
from solprop.adiabatic import (
    AdiabaticAccuracy,
    AdiabaticAdmission,
    certify_radial_path,
    certify_wave_path,
    field_error_bound,
    reflection_bound,
)
from solprop.constants import C
from solprop.exceptions import MaterialError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.linear import Frame
from solprop.profiles import FromCallable, Linear, PiecewiseLinear, ZProfile
from solprop.spatial import SpatialDispersion, SpatialMedium

WAVELENGTH = 800e-9
OMEGA = 2 * math.pi * C / WAVELENGTH
K0 = OMEGA / C
UNBOUNDED = AdiabaticAccuracy(field_error=1e300, reflected_energy=1.0)
"""Budgets nothing exceeds, so a test reads the enclosure itself."""

# The certificate's own arithmetic: a few dozen rounded operations on each term,
# conditioned by S/s = 1 far from cutoff, with a safety factor of four.
CERTIFICATE_ROUNDING = 256 * EPS


@dataclass(frozen=True)
class Case:
    """One profile in both forms.

    Attributes
    ----------
    path : ZProfile
        ``rho(z)``, dimensionless.
    witness : Profile
        The same ``q`` for the scalar-wave solver.
    base : float
        ``n0^2``.
    """

    path: ZProfile
    witness: Profile
    base: float = 1.0

    def certify(self, accuracy: AdiabaticAccuracy = UNBOUNDED) -> AdiabaticAdmission:
        return certify_wave_path(
            np.array([OMEGA]),
            np.array([self.base + 0j]),
            np.array([[1.0 + 0j]]),
            [self.path],
            np.zeros(1),
            (self.witness.start, self.witness.end),
            accuracy,
        )


def _case(  # noqa: PLR0913 - the profile and its declarations
    rho, slope, start, end, *, base=1.0, jumps=(), kinks=(), turning=()
):
    """Build a smooth or jumping profile from ``rho`` and its derivative."""

    def q(z):
        return K0 * math.sqrt(base + rho(z))

    def log_slope(z):
        return slope(z) / (2.0 * (base + rho(z)))

    path = FromCallable(
        np.vectorize(rho, otypes=[float]),
        (start, end),
        np.array(kinks, dtype=float),
        jumps=np.array(jumps, dtype=float),
        turning=np.array(turning, dtype=float),
    )
    return Case(
        path, Profile(q, log_slope, start, end, tuple(jumps), tuple(kinks)), base
    )


def tanh_ramp(contrast, width, half, base=1.0):
    """``rho = contrast (1 + tanh(z/width))/2`` on ``[-half, half]``, Epstein's."""
    return _case(
        lambda z: contrast * 0.5 * (1.0 + math.tanh(z / width)),
        lambda z: contrast * 0.5 / (width * math.cosh(z / width) ** 2),
        -half,
        half,
        base=base,
    )


def step(contrast, half, base=1.0):
    """An abrupt step of ``rho`` from zero to `contrast` at ``z = 0``."""
    return _case(
        lambda z: 0.0 if z < 0.0 else contrast,
        lambda z: 0.0,
        -half,
        half,
        base=base,
        jumps=(0.0,),
    )


def gaussian_bump(contrast, sigma, half):
    """``rho = contrast exp(-z^2 / (2 sigma^2))``, turning at its peak."""
    return _case(
        lambda z: contrast * math.exp(-(z * z) / (2 * sigma * sigma)),
        lambda z: -contrast * z / sigma**2 * math.exp(-(z * z) / (2 * sigma * sigma)),
        -half,
        half,
        turning=(0.0,),
    )


def grating(contrast, period, count):
    """``rho = contrast sin^2(pi z / period)`` over `count` periods, turning at each
    half period."""
    end = count * period
    return _case(
        lambda z: contrast * math.sin(math.pi * z / period) ** 2,
        lambda z: contrast * math.pi / period * math.sin(2 * math.pi * z / period),
        0.0,
        end,
        turning=tuple(np.arange(1, 2 * count) * period / 2),
    )


def linear_ramp(contrast, length, half):
    """A piecewise-linear ramp, two kinks, given as a resolved profile."""
    witness = Profile(
        lambda z: K0 * math.sqrt(1.0 + min(max(z, 0.0), length) / length * contrast),
        lambda z: (
            contrast / length / (2.0 * (1.0 + z / length * contrast))
            if 0.0 <= z < length
            else 0.0
        ),
        -half,
        length + half,
        kinks=(0.0, length),
    )
    path = PiecewiseLinear.from_stages([0.0, length], [0.0, contrast])
    return Case(path, witness)


GAS = 1.0008**2 - 1.0
"""The square contrast of a 3 bar gas index step, ``n = 1 -> 1.0008``."""
GLASS = 1.5**2 - 1.0
"""The square contrast of ``n = 1 -> 1.5``."""


def assert_bounds_hold(admission: AdiabaticAdmission, wave: Scattering) -> None:
    """The certified bounds enclose the witness within its refinement estimate."""
    # The witness's error is its refinement estimate, doubled; the certificate's
    # rounding enlarges its bound by CERTIFICATE_ROUNDING relative at most.
    assert wave.flux_residual < 1e-12
    reflection = admission.reflection * (1 + CERTIFICATE_ROUNDING)
    assert wave.reflectance <= reflection + 2 * wave.reflectance_error
    forward = admission.field_error * (1 + CERTIFICATE_ROUNDING)
    assert wave.forward_error <= forward + 2 * wave.forward_ratio_error
    # The passive endpoint also obeys the tighter B* = Lambda tanh(Lambda).
    tighter = admission.variation * math.tanh(admission.variation)
    assert wave.forward_error <= tighter * (1 + CERTIFICATE_ROUNDING) + (
        2 * wave.forward_ratio_error
    )


# ----------------------------------------------------------------- the A1 witnesses

# Review A1 at 800 nm, fully specified here. "Gas" is n = 1 -> 1.0008 (argon near
# 3 bar); a ramp is the Epstein layer rho = contrast (1 + tanh(z/w))/2 on [-6w, 6w]
# (gas, w = 1 mm / 12) or [-8w, 8w] (glass, w = 0.5 um); the bump is a Gaussian of
# sigma = 0.1 mm on [-0.6, 0.6] mm. The review's printed reflectances for the smooth
# gas rows (1e-24) were below its solver's own resolution; only the bounds and
# the refined witness are claimed here.
A1 = {
    "gas tanh ramp, 1 mm": lambda: tanh_ramp(GAS, 1e-3 / 12, 0.5e-3),
    "gas abrupt step": lambda: step(GAS, 1e-6),
    "gas Gaussian bump, 0.2 mm": lambda: gaussian_bump(GAS, 1e-4, 0.6e-3),
    "glass tanh ramp, 0.5 um": lambda: tanh_ramp(GLASS, 0.5e-6, 4e-6),
    "glass abrupt step": lambda: step(GLASS, 1e-6),
}


@pytest.mark.medium
@pytest.mark.parametrize("name", list(A1))
def test_the_review_profiles_sit_inside_both_bounds(name):
    case = A1[name]()
    admission = case.certify()
    wave = scatter(case.witness, rtol=1e-9)
    assert_bounds_hold(admission, wave)
    # Every A1 path is monotone on each declared piece (the bump turns once), so
    # the enclosure is the exact variation of ln q: a quarter of the log ratios of
    # q^2 between the turning points and the ends, jumps included.
    knots = np.array([case.witness.start, 0.0, case.witness.end])
    values = case.path(knots)
    values[1] = case.path(np.array([0.0]))[0]
    expected = 0.25 * float(np.sum(np.abs(np.diff(np.log1p(values)))))
    assert admission.exact
    assert admission.variation == pytest.approx(expected, rel=CERTIFICATE_ROUNDING)


@pytest.mark.parametrize("contrast", [GAS, GLASS])
def test_an_abrupt_step_attains_the_reflection_bound(contrast):
    case = step(contrast, 1e-6)
    admission = case.certify()
    wave = scatter(case.witness)
    exact_reflection, exact_forward = fresnel_step(K0, K0 * math.sqrt(1 + contrast))
    # A step's reflectance is the Fresnel value, which is tanh^2(Lambda) exactly:
    # the bound is attained, within the certificate's and the closed form's
    # rounding.
    assert admission.reflection == pytest.approx(
        exact_reflection, rel=CERTIFICATE_ROUNDING
    )
    # The witness's jump transfer is exact; its integration of the flat pieces is
    # a phase only, so it reproduces the closed form to rounding.
    assert wave.reflectance == pytest.approx(exact_reflection, rel=64 * EPS)
    assert wave.forward_error == pytest.approx(exact_forward, rel=1e-12)
    assert_bounds_hold(admission, wave)


def test_the_witness_reproduces_an_epstein_layer():
    # An independent check of the witness itself: the glass ramp is an Epstein
    # layer in q^2 (logistic scale s = w/2), truncated at +-8w.
    width, half = 0.5e-6, 4e-6
    case = tanh_ramp(GLASS, width, half)
    wave = scatter(case.witness)
    exact = epstein_reflectance(K0, K0 * math.sqrt(1 + GLASS), width / 2)
    # Truncation leaves a jump in q'/q of at most contrast sech^2(8)/(w (1+contrast))
    # at each end, reflecting an amplitude of at most that over (4 q_min) each; its
    # interference with the layer's own amplitude bounds the change of R.
    tail = GLASS / math.cosh(half / width) ** 2 / (width * (1 + GLASS))
    stray = 2 * tail / (4 * K0)
    envelope = 2 * math.sqrt(exact) * stray + stray**2
    assert abs(wave.reflectance - exact) <= envelope + 2 * wave.reflectance_error


def test_a_nonmonotone_bump_counts_both_flanks():
    case = gaussian_bump(GAS, 1e-4, 0.6e-3)
    admission = case.certify()
    ends = case.path(np.array([-0.6e-3]))[0]
    # Up and down: both flanks' log ratios, though the ends nearly agree.
    expected = 0.5 * (math.log1p(GAS) - math.log1p(ends))
    assert admission.variation == pytest.approx(expected, rel=CERTIFICATE_ROUNDING)
    assert admission.variation > 1.9 * 0.25 * math.log1p(GAS)


def test_repeated_oscillations_grow_the_bound_and_a_real_reflection():
    # A Bragg grating (period lambda/2): each half period adds |Delta ln q|, so the
    # variation grows with the count, and so does the true coherent reflection.
    period, contrast = WAVELENGTH / 2, 2e-4
    few, many = grating(contrast, period, 40), grating(contrast, period, 400)
    small, large = few.certify(), many.certify()
    per_half = 0.25 * math.log(1 + contrast)
    for admission, count in ((small, 40), (large, 400)):
        assert admission.variation == pytest.approx(
            2 * count * per_half, rel=CERTIFICATE_ROUNDING * count
        )
    wave = scatter(many.witness)
    assert_bounds_hold(large, wave)
    # The grating genuinely reflects more than the default budget allows, and the
    # default refuses it.
    assert wave.reflectance > 1e-4
    with pytest.raises(MaterialError, match="exceeds its adiabatic budget"):
        many.certify(AdiabaticAccuracy())
    few.certify(AdiabaticAccuracy(field_error=2e-5, reflected_energy=2e-5))


def test_a_kinked_ramp_is_bounded_and_exact():
    case = linear_ramp(GLASS, 1e-6, 1e-6)
    admission = case.certify()
    wave = scatter(case.witness)
    assert_bounds_hold(admission, wave)
    assert admission.exact
    assert admission.variation == pytest.approx(
        0.25 * math.log(1 + GLASS), rel=CERTIFICATE_ROUNDING
    )
    # The smooth-with-kinks ramp reflects less than the step it is bounded by.
    assert wave.reflectance < admission.reflection


# ---------------------------------------------------------------------- admission


def test_the_default_budget_admits_a_gas_step_and_refuses_a_glass_one():
    admitted = step(GAS, 1e-6).certify(AdiabaticAccuracy())
    assert admitted.field_error < 1e-6
    with pytest.raises(MaterialError, match=r"field error 0\.0414 > 0\.0001"):
        step(GLASS, 1e-6).certify(AdiabaticAccuracy())


def test_a_tightened_budget_refuses_what_the_default_admits():
    case = step(GAS, 1e-6)
    case.certify(AdiabaticAccuracy())
    with pytest.raises(MaterialError, match=r"reflected energy 1\.6e-07 > 1e-08"):
        case.certify(AdiabaticAccuracy(field_error=1.0, reflected_energy=1e-8))
    with pytest.raises(MaterialError, match="800 nm"):
        case.certify(AdiabaticAccuracy(field_error=1e-9))


@pytest.mark.parametrize("value", [0.0, -1e-4, math.inf, math.nan, True])
def test_the_budgets_must_be_finite_and_positive(value):
    with pytest.raises(MaterialError, match="finite and positive"):
        AdiabaticAccuracy(field_error=value)
    with pytest.raises(MaterialError, match="finite and positive"):
        AdiabaticAccuracy(reflected_energy=value)


@pytest.mark.parametrize(
    ("omega", "kperp_sq", "z_span", "match"),
    [
        ([OMEGA], [0.0], (1e-3, 0.0), "finite z0 < z1"),
        ([OMEGA], [0.0], (0.0, math.inf), "finite z0 < z1"),
        ([-OMEGA], [0.0], (0.0, 1e-3), "finite positive frequencies"),
        ([math.nan], [0.0], (0.0, 1e-3), "finite positive frequencies"),
        ([OMEGA], [-1.0], (0.0, 1e-3), "finite and >= 0"),
        ([OMEGA], [math.inf], (0.0, 1e-3), "finite and >= 0"),
    ],
)
def test_the_wave_certificate_refuses_an_invalid_input(omega, kperp_sq, z_span, match):
    with pytest.raises(MaterialError, match=match):
        certify_wave_path(
            np.array(omega),
            np.ones(1, dtype=complex),
            np.ones((1, 1), dtype=complex),
            [Linear(0.0, 1e-3, 1e-3)],
            np.array(kperp_sq),
            z_span,
        )


def test_the_bounds_are_stable_for_tiny_variation():
    # tanh and expm1 keep full relative accuracy where 1 - cosh would not.
    assert reflection_bound(1e-12) == pytest.approx(1e-24, rel=4 * EPS)
    assert field_error_bound(1e-12) == pytest.approx(1e-24, rel=4 * EPS)


def test_a_column_crossing_cutoff_refuses():
    # q^2 = k0^2 (1 + rho) - k_perp^2 changes sign when k_perp^2 sits between the
    # two ends' k^2.
    path = Linear(0.0, 1e-3, 1e-3)
    between = K0**2 * (1 + 0.5e-3)
    with pytest.raises(MaterialError, match="crosses q = 0"):
        certify_wave_path(
            np.array([OMEGA]),
            np.ones(1, dtype=complex),
            np.ones((1, 1), dtype=complex),
            [path],
            np.array([0.0, between]),
            (0.0, 1e-3),
        )


def test_a_column_nonpropagating_everywhere_is_outside_the_certificate():
    path = Linear(0.0, 1e-3, 1e-3)
    admission = certify_wave_path(
        np.array([OMEGA]),
        np.ones(1, dtype=complex),
        np.ones((1, 1), dtype=complex),
        [path],
        np.array([0.0, 4 * K0**2]),
        (0.0, 1e-3),
    )
    assert admission.support.tolist() == [[True], [False]]


def test_a_near_cutoff_column_needs_the_largest_variation():
    # Near cutoff q varies much more than the bulk index (section 4.5).
    path = Linear(0.0, 1e-3, 1e-3)
    near = K0**2 * (1 - 1e-3)
    admission = certify_wave_path(
        np.array([OMEGA]),
        np.ones(1, dtype=complex),
        np.ones((1, 1), dtype=complex),
        [path],
        np.array([0.0, near]),
        (0.0, 1e-3),
        UNBOUNDED,
    )
    assert admission.worst_kperp_sq == near
    # s runs from 1e-3 k0^2 to 2e-3 k0^2: Lambda = ln(2)/4.
    assert admission.variation == pytest.approx(math.log(2) / 4, rel=1e-9)


def test_a_varying_imaginary_index_refuses():
    with pytest.raises(MaterialError, match="nonconstant absorbing path"):
        certify_wave_path(
            np.array([OMEGA]),
            np.ones(1, dtype=complex),
            np.array([[1.0 + 1e-6j]]),
            [Linear(0.0, 1e-3, 1e-3)],
            np.zeros(1),
            (0.0, 1e-3),
        )
    # Uniform loss is not a varying path.
    admission = certify_wave_path(
        np.array([OMEGA]),
        np.array([2.25 + 1e-6j]),
        np.ones((1, 1), dtype=complex),
        [Linear(1e-3, 1e-3, 1e-3)],
        np.zeros(1),
        (0.0, 1e-3),
    )
    assert admission.variation == 0.0


def test_an_unresolved_callable_refuses():
    path = FromCallable(lambda z: 1e-3 * z, (0.0, 1.0))
    with pytest.raises(MaterialError, match="no declared variation"):
        certify_wave_path(
            np.array([OMEGA]),
            np.ones(1, dtype=complex),
            np.ones((1, 1), dtype=complex),
            [path],
            np.zeros(1),
            (0.0, 1.0),
        )


# ------------------------------------------------------------------ spatial media


def _radial(medium, length=1e-3, radius=6e-4, points=16):
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = RadialGrid(radius, points, window_factor=0.0)
    return SpatialDispersion(medium, grid, Frame.comoving(grid)), space


def test_an_argon_gradient_is_admitted_with_its_exact_variation():
    length = 1e-3
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, length))
    dispersion, space = _radial(medium, length)
    admission = certify_radial_path(dispersion, space.k_r**2, (0.0, length))
    assert admission.exact
    assert admission.field_error < 1e-7
    # Independently: the worst column's Lambda is half the log ratio of q at the
    # ends, from the medium's own index.
    omega = np.array([admission.worst_omega])
    kperp_sq = admission.worst_kperp_sq
    assert kperp_sq is not None
    ends = [
        np.sqrt((omega * np.real(medium.index(omega, z)) / C) ** 2 - kperp_sq)[0]
        for z in (0.0, length)
    ]
    # The index's square root and the certificate's affine form round differently,
    # each within a few ulps of k^2; the log of a ratio near one keeps that
    # absolute error, relative to Lambda ~ 1e-4 it is ~ 1e-12.
    assert admission.variation == pytest.approx(
        0.5 * abs(math.log(ends[1] / ends[0])), rel=1e-9
    )
    assert admission.identity["material"] == "Ar"


def test_a_uniform_medium_is_admitted_with_zero_variation():
    dispersion, space = _radial(SpatialMedium(1.5 + 1e-6j))
    admission = certify_radial_path(dispersion, space.k_r**2, (0.0, 1e-3))
    assert admission.variation == 0.0
    assert admission.worst_omega is None


def test_a_custom_pressure_profile_refuses_without_declared_variation():
    medium = SpatialMedium(
        "Ar", pressure=FromCallable(lambda z: 1.0 + 0.5 * z / 1e-3, (0.0, 1e-3))
    )
    dispersion, space = _radial(medium)
    with pytest.raises(MaterialError, match="no declared variation"):
        certify_radial_path(dispersion, space.k_r**2, (0.0, 1e-3))


def test_a_fine_radial_grid_refuses_its_near_cutoff_columns():
    # 0.16 um radial spacing reaches k_perp near k in the band: those columns'
    # q varies far more than the gas index does.
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, 1e-3))
    dispersion, space = _radial(medium, radius=1e-5, points=64)
    with pytest.raises(MaterialError, match="exceeds its adiabatic budget"):
        certify_radial_path(dispersion, space.k_r**2, (0.0, 1e-3))

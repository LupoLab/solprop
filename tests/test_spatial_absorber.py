# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transverse boundary's specification and design formulas (SEP 40 section 4).

The constants are phase 0's calibration on the discrete operators; these tests hold
the formulas to the numbers recorded in the SEP's Part II.
"""

import math

import numpy as np
import pytest

from solprop.boundaries import SpatialAbsorber as Exported
from solprop.constants import C
from solprop.exceptions import GridError
from solprop.grid import FieldGrid
from solprop.spatial import SpatialMedium
from solprop.spatial_absorber import (
    DEFAULT_ATTENUATION,
    SpatialAbsorber,
    SpatialAbsorberDesign,
    collar_attenuation,
    collar_profile,
    collar_strength,
    crossing_limit,
    reflection_threshold,
    rms_wavenumber,
    safety_ramp_strength,
)

K800 = 2 * math.pi / 800e-9


def _suite_band() -> np.ndarray:
    """The suite's ``spatial_absorber`` case: argon at 1 bar over its requested band."""
    grid = FieldGrid.from_limits(0.08, 800e-9, (500e-9, 1.6e-6), 250e-15)
    omega = grid.omega[grid.inband_mask & grid.active_mask]
    return np.real(SpatialMedium("Ar", 1.0).index(omega)) * omega / C


def test_the_profile_is_the_planck_window_s_complement() -> None:
    s = np.linspace(-0.5, 1.5, 201)
    p = collar_profile(s)
    assert np.all(p[s <= 0] == 0.0)
    assert np.all(p[s >= 1] == 1.0)
    assert np.all(np.diff(p) >= 0.0)
    assert collar_profile(np.array([0.5]))[0] == pytest.approx(0.5, abs=1e-15)


def test_the_root_exceeds_the_weak_loss_formula_as_measured() -> None:
    # Section 4.2: the exact-symbol root is 1.06 times the weak-loss formula at
    # kappa_p w = 20 for the Planck profile.
    width = 300e-6
    kappa_p = 20 / width
    k_z = math.sqrt(K800**2 - kappa_p**2)
    formula = DEFAULT_ATTENUATION * kappa_p / (0.5 * k_z * width)
    alpha0 = collar_strength(kappa_p, np.array([K800]), width)
    assert alpha0 / formula == pytest.approx(1.06, abs=0.005)
    assert collar_attenuation(kappa_p, K800, alpha0, width) == pytest.approx(
        DEFAULT_ATTENUATION, rel=1e-9
    )


@pytest.mark.parametrize(
    ("fraction", "alpha0"), [(1.0, 698), (1.5, 436), (2.0, 318), (2.6, 242)]
)
def test_the_suite_case_s_designs_reproduce(fraction: float, alpha0: float) -> None:
    # Part II, "the adopted rule and the direct check": designed on the requested band
    # at kappa_p = 3 sqrt(2) / w0, w0 = 100 um, collar width f x 300 um.
    kappa_p = 3 * math.sqrt(2) / 100e-6
    assert collar_strength(kappa_p, _suite_band(), fraction * 300e-6) == pytest.approx(
        alpha0, abs=1.0
    )


def test_an_angle_is_solved_per_frequency_not_at_the_extremes() -> None:
    # Section 4.2: pairing the bluest kappa_p with the reddest k_z over-strengthens a
    # declared angle by about omega_max / omega_min.
    k = _suite_band()
    sin_theta = 25e-3
    kappa_p = k * sin_theta  # n ~ 1
    per_frequency = collar_strength(kappa_p, k, 300e-6)
    paired = collar_strength(float(kappa_p.max()), np.array([k.min()]), 300e-6)
    assert paired / per_frequency == pytest.approx(k.max() / k.min(), rel=0.05)


def test_the_threshold_follows_the_measured_law() -> None:
    k = np.array([K800])
    alpha0, width = 698.0, 300e-6
    kappa_lo = reflection_threshold(alpha0, width, k)
    k_z = math.sqrt(K800**2 - kappa_lo**2)
    assert kappa_lo * width == pytest.approx(
        7.36 * (k_z * alpha0 * width**2) ** 0.182, rel=1e-12
    )


def test_the_suite_band_opens_between_two_and_two_point_six() -> None:
    # The band opens (kappa_lo <= kappa_p) only at the wide designs; accuracy does
    # not follow the opening (section 4.4), but the numbers are the SEP's.
    k = _suite_band()
    kappa_p = 3 * math.sqrt(2) / 100e-6
    ratios = []
    for fraction in (2.0, 2.6):
        width = fraction * 300e-6
        ratios.append(
            reflection_threshold(collar_strength(kappa_p, k, width), width, k)
        )
    assert ratios[0] > kappa_p > ratios[1]


def test_the_safety_ramp_s_strength_and_its_floor() -> None:
    k = np.array([K800])
    alpha = safety_ramp_strength(6e5, k, 300e-6)
    k_z = math.sqrt(K800**2 - 6e5**2)
    assert alpha == pytest.approx(2 * (DEFAULT_ATTENUATION + 2) * 6e5 / (k_z * 300e-6))
    # Beyond cutoff the floor 0.44 k keeps it finite.
    beyond = safety_ramp_strength(1.2 * K800, k, 300e-6)
    assert beyond == pytest.approx(
        2 * (DEFAULT_ATTENUATION + 2) * 1.2 * K800 / (0.44 * K800 * 300e-6)
    )


def test_the_ramp_s_strength_does_not_depend_on_the_run_length() -> None:
    # Section 4.4: nothing in the design takes L; the length is the collar's width or
    # the requested size.
    k = _suite_band()
    assert safety_ramp_strength(6e5, k, 300e-6) == safety_ramp_strength(6e5, k, 300e-6)
    kappa_p = 3 * math.sqrt(2) / 100e-6
    assert collar_strength(kappa_p, k, 300e-6) == collar_strength(kappa_p, k, 300e-6)


def test_the_crossing_limit() -> None:
    k = np.array([K800])
    kappa_p, width = 1e5, 300e-6
    k_z = math.sqrt(K800**2 - kappa_p**2)
    assert crossing_limit(kappa_p, k, width) == pytest.approx(
        0.5 * width * k_z / kappa_p
    )
    assert crossing_limit(kappa_p, k, width, 0.15) == pytest.approx(
        0.15 * width * k_z / kappa_p
    )


def test_a_gaussian_s_rms_wavenumber() -> None:
    # For exp(-r^2/w0^2), kappa_rms w0 = sqrt(2) on |kappa_perp|.
    w0 = 100e-6
    kappa = np.linspace(0, 20 / w0, 20001)
    power = kappa * np.exp(-((kappa * w0) ** 2) / 2)  # 2 pi kappa |E(kappa)|^2
    assert rms_wavenumber(power, kappa) * w0 == pytest.approx(math.sqrt(2), rel=1e-6)
    assert rms_wavenumber(np.zeros(3), np.ones(3)) == 0.0


def test_a_band_past_cutoff_is_refused() -> None:
    with pytest.raises(GridError, match="strictly between"):
        collar_strength(1.1 * K800, np.array([K800]), 300e-6)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"kappa": 1e5, "angle": 0.1}, "not both"),
        ({"kappa": -1.0}, "kappa"),
        ({"angle": 2.0}, "angle"),
        ({"discard_above": 0.0}, "discard_above"),
        ({"attenuation": 0.0}, "attenuation"),
        ({"safety_fraction": 1.0}, "safety_fraction"),
        ({"kappa": 1e5, "real_space": False}, "real-space collar"),
        ({"kappa": 2e5, "discard_above": 1e5}, "below"),
    ],
)
def test_the_specification_refuses_what_it_cannot_mean(
    arguments: dict[str, object], match: str
) -> None:
    with pytest.raises(GridError, match=match):
        SpatialAbsorber(**arguments)  # pyright: ignore[reportArgumentType]


def test_the_specification_s_band_source_and_export() -> None:
    assert Exported is SpatialAbsorber
    assert SpatialAbsorber().band_source == "launch"
    assert SpatialAbsorber(angle=0.02).band_source == "angle"
    assert SpatialAbsorber(kappa=2e5).band_source == "kappa"
    assert SpatialAbsorber(real_space=False).band_source == "launch"


def test_a_design_freezes_its_rates() -> None:
    design = SpatialAbsorberDesign(
        SpatialAbsorber(real_space=False), np.array([0.0, 1.0]), 1.0, (9.0,)
    )
    assert not design.has_collar
    assert not design.ramp_rate.flags.writeable
    with pytest.raises(GridError, match="nonnegative"):
        SpatialAbsorberDesign(SpatialAbsorber(), np.array([-1.0]), 1.0, (9.0,))

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Analytic Jones identities and the Stokes ellipse orientation regression."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.exceptions import PolarisationError
from solprop.polarisation import (
    cartesian,
    ellipse,
    ellipticity,
    horizontal,
    linear_polariser,
    rotate,
    rotation,
    stokes,
    waveplate,
)

# Two 2x2 products, the Stokes quadratic forms and elementary functions require
# fewer than 32 rounded operations per component. Absolute error handles exact zeros.
ROUNDING = 32 * np.finfo(float).eps


@pytest.mark.parametrize("angle", [0.0, math.pi / 7, -math.pi / 3, math.pi / 2])
def test_linear_ellipse_recovers_orientation(angle):
    field = np.array([math.cos(angle), math.sin(angle)])
    parameters = stokes(field)
    result = ellipse(parameters)
    assert_allclose(
        parameters,
        [1, math.cos(2 * angle), math.sin(2 * angle), 0],
        rtol=ROUNDING,
        atol=ROUNDING,
    )
    assert result.angle == pytest.approx(angle, rel=ROUNDING, abs=ROUNDING)
    assert result.major == pytest.approx(1.0, rel=ROUNDING)
    # The minor semiaxis takes a square root of a cancellation of unit Stokes terms.
    assert result.minor <= math.sqrt(ROUNDING)


@pytest.mark.parametrize("handedness", [-1, 1])
def test_circular_stokes_and_ellipse(handedness):
    field = np.array([1, handedness * 1j]) / math.sqrt(2)
    parameters = stokes(field, normalise=True)
    assert_allclose(parameters, [1, 0, 0, handedness], rtol=0, atol=ROUNDING)
    result = ellipse(parameters)
    assert result.angle == 0
    assert result.handedness == handedness
    assert result.major == result.minor
    assert ellipticity(parameters) == 1
    assert_allclose(cartesian(parameters), [0, 0, handedness], rtol=0, atol=ROUNDING)


def test_elliptical_axes_recover_a_rotated_jones_state():
    angle = math.pi / 5
    field = rotation(-angle) @ np.array([2, 1j])
    result = ellipse(stokes(field))
    assert_allclose(
        [result.major, result.minor, result.angle],
        [2, 1, angle],
        rtol=ROUNDING,
        atol=ROUNDING,
    )
    assert result.handedness == 1
    assert ellipticity(stokes(field)) == pytest.approx(0.5, rel=ROUNDING)


@pytest.mark.parametrize("phase", [0, math.pi / 2, math.pi, -math.pi / 3])
def test_waveplate_is_unitary_and_preserves_intensity(phase):
    plate = waveplate(phase)
    assert_allclose(plate.conj().T @ plate, np.eye(2), rtol=0, atol=ROUNDING)
    field = np.array([1, 1j])
    assert stokes(plate @ field)[0] == pytest.approx(stokes(field)[0], rel=ROUNDING)


def test_rotated_polariser_obeys_malus_law():
    angle = math.pi / 3
    polariser = rotate(linear_polariser(), angle)
    assert_allclose(polariser @ polariser, polariser, rtol=ROUNDING, atol=ROUNDING)
    assert stokes(polariser @ horizontal())[0] == pytest.approx(
        math.cos(angle) ** 2, rel=ROUNDING
    )


def test_waveplate_phase_sign_is_the_optics_sign():
    # Fast axis x: a quarter-wave plate delays the y component of light linear at
    # +45° by a quarter period, which is (x + i y)/sqrt(2), positive V, in the
    # optics convention where a delay is a positive phase (conventions.md §2).
    field = waveplate(math.pi / 2) @ np.array([1, 1])
    assert_allclose(stokes(field, normalise=True), [1, 0, 0, 1], rtol=0, atol=ROUNDING)
    assert_allclose(field / field[0], [1, 1j], rtol=0, atol=ROUNDING)


def test_zero_field_has_unnormalised_stokes_but_no_normalised_coordinates():
    assert_allclose(stokes([0, 0]), np.zeros(4), rtol=0, atol=0)
    result = ellipse([0, 0, 0, 0])
    assert (result.major, result.minor, result.angle, result.handedness) == (0, 0, 0, 0)
    for function in (cartesian, ellipticity):
        with pytest.raises(PolarisationError, match="zero intensity"):
            function([0, 0, 0, 0])
    with pytest.raises(PolarisationError, match="zero intensity"):
        stokes([0, 0], normalise=True)


@pytest.mark.parametrize("value", [[], [1], [1, 2, 3], [np.nan, 1], [np.inf, 0]])
def test_stokes_rejects_invalid_jones_vectors(value):
    with pytest.raises(PolarisationError, match="Jones"):
        stokes(value)


@pytest.mark.parametrize(
    "value", [[1, 2, 0, 0], [-1, 0, 0, 0], [1, 0, 0], [1, np.nan, 0, 0], [1, 0, 0, 1j]]
)
def test_ellipse_rejects_invalid_stokes_vectors(value):
    with pytest.raises(PolarisationError, match="Stokes"):
        ellipse(value)


@pytest.mark.parametrize("function", [rotation, waveplate])
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_optics_reject_nonfinite_angles(function, value):
    with pytest.raises(PolarisationError, match="finite"):
        function(value)


def test_rotate_rejects_a_nonmatrix():
    with pytest.raises(PolarisationError, match="shape"):
        rotate([1, 0], 0.0)


def test_normalised_stokes_avoids_squaring_large_fields():
    large = np.finfo(float).max / 4
    assert_allclose(
        stokes([large, 1j * large], normalise=True), [1, 0, 0, 1], rtol=0, atol=ROUNDING
    )
    with pytest.raises(PolarisationError, match="overflow"):
        stokes([large, large])


def test_linear_ellipticity_does_not_divide_by_zero():
    assert ellipticity([1, 1, 0, 0]) == 0


def test_unpolarised_stokes_has_no_preferred_orientation():
    result = ellipse([2, 0, 0, 0])
    assert (result.major, result.minor, result.angle, result.handedness) == (1, 1, 0, 0)

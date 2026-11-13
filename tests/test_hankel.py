# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from hankel_oracles import (
    construction_error,
    gaussian_bound,
    gram_bound,
    round_trip_bounds,
)
from solprop import hankel
from solprop.exceptions import GridError
from solprop.hankel import QDHT
from spatial_records import record_metric


@pytest.mark.parametrize("size", [16, 64, 256])
def test_raw_hankel_gram_has_independently_bounded_truncation(size, request):
    transform = QDHT(1e-3, size)
    truncation, floating = gram_bound(transform)
    bound = truncation + floating
    assert np.all(bound > 0)
    defect = np.abs(transform.matrix @ transform.matrix - np.eye(size))
    assert np.all(defect <= bound), (size, np.max(defect), np.max(bound))
    # Preserve the original failed exact-orthogonality assertion as an
    # explicit diagnostic. Raw finite QDHT truncation must not be corrected away.
    assert np.max(defect) > size * np.finfo(float).eps
    record_metric(request, "raw QDHT Gram (1)", size, np.max(defect), np.max(bound))


@pytest.mark.parametrize("size", [32, 64, 128])
def test_hankel_gaussian_matches_continuous_transform(size, request):
    transform = QDHT(1e-3, size)
    waist = transform.radius / 6
    reference, bound = gaussian_bound(transform, waist)
    assert np.all(bound > 0)
    actual = transform.forward(np.exp(-((transform.r / waist) ** 2)))
    assert np.all(np.abs(actual - reference) <= bound)
    record_metric(
        request,
        "Gaussian Hankel transform (m²)",
        size,
        np.max(np.abs(actual - reference)),
        np.max(bound),
    )


@pytest.mark.parametrize("size", [16, 64])
def test_hankel_roundtrip_and_weighted_parseval_for_vector_complex_data(size, request):
    transform = QDHT(7e-4, size)
    rng = np.random.default_rng(310)
    values = rng.normal(size=(2, 3, size)) + 1j * rng.normal(size=(2, 3, size))
    spectrum = transform.forward(values)
    bound, energy_bound = round_trip_bounds(transform, values, spectrum)
    actual = transform.inverse(spectrum)
    assert np.all(np.abs(actual - values) <= bound)
    radial = np.sum(np.abs(values) ** 2 * transform.radial_weights, axis=-1)
    spectral = np.sum(np.abs(spectrum) ** 2 * transform.spectral_weights, axis=-1)
    assert np.all(np.abs(radial - spectral) <= energy_bound)
    record_metric(
        request,
        "complex QDHT round trip (1)",
        size,
        np.max(np.abs(actual - values)),
        np.max(bound),
    )
    record_metric(
        request,
        "weighted Parseval (m²)",
        size,
        np.max(np.abs(radial - spectral)),
        np.max(energy_bound),
    )


@settings(max_examples=60, deadline=None)
@given(
    size=st.integers(min_value=2, max_value=160),
    radius=st.floats(min_value=1e-6, max_value=1e-1),
    complex_values=st.booleans(),
    batch=st.lists(st.integers(min_value=1, max_value=3), max_size=2).map(tuple),
    exponent=st.integers(min_value=-60, max_value=60),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_hankel_round_trip_and_parseval_hold_for_any_size_radius_and_scale(
    size, radius, complex_values, batch, exponent, seed
):
    transform = QDHT(radius, size)
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(*batch, size))
    if complex_values:
        values = values + 1j * rng.normal(size=(*batch, size))
    values *= 10.0**exponent
    spectrum = transform.forward(values)
    # The derived bounds of the guide's "Finite-transform error": the contour
    # truncation b_N, construction and GEMM rounding, all fixed from |values| before
    # the transform is measured. They are linear (field) and quadratic (energy) in the
    # samples, so the scale 10**exponent, kept inside the normal range with the radius
    # weights, does not loosen them; the radius sets the units only.
    bound, energy_bound = round_trip_bounds(transform, values, spectrum)
    actual = transform.inverse(spectrum)
    assert actual.dtype == values.dtype  # real stays real, complex complex
    assert np.all(np.abs(actual - values) <= bound)
    radial = np.sum(np.abs(values) ** 2 * transform.radial_weights, axis=-1)
    spectral = np.sum(np.abs(spectrum) ** 2 * transform.spectral_weights, axis=-1)
    assert np.all(np.abs(radial - spectral) <= energy_bound)
    assert np.all(radial > 0.0)


def test_hankel_arrays_are_immutable_and_scalings_have_physical_units():
    small = QDHT(1e-3, 8)
    large = QDHT(2e-3, 8)
    for name in (
        "r",
        "k_r",
        "j1",
        "zeros",
        "matrix",
        "radial_weights",
        "spectral_weights",
    ):
        assert not getattr(small, name).flags.writeable
    with pytest.raises(FrozenInstanceError):
        small.__setattr__("radius", 2e-3)
    # Radius changes involve powers of two and are exact binary scalings.
    assert_allclose(large.matrix, small.matrix, rtol=0, atol=0)
    assert_allclose(large.r, 2 * small.r, rtol=0, atol=0)
    assert_allclose(large.k_r, small.k_r / 2, rtol=0, atol=0)
    assert_allclose(large.radial_weights, 4 * small.radial_weights, rtol=0, atol=0)
    assert_allclose(large.spectral_weights, small.spectral_weights / 4, rtol=0, atol=0)
    assert small.forward_scale * small.inverse_scale == 1
    assert np.all(construction_error(small) > 0)


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf, 1e-300, 1e300, 1e-153])
def test_hankel_rejects_invalid_radius(radius):
    with pytest.raises(GridError, match="radius"):
        QDHT(radius, 8)


@pytest.mark.parametrize("size", [0, 1, -3, True])
def test_hankel_rejects_invalid_size(size):
    with pytest.raises(GridError, match="n_r"):
        QDHT(1e-3, size)


def test_hankel_rejects_a_nonfinite_special_function_result(monkeypatch):
    original = hankel.jv

    def bad_bessel(order, argument):
        return (
            np.full_like(argument, np.nan) if order == 0 else original(order, argument)
        )

    monkeypatch.setattr(hankel, "jv", bad_bessel)
    with pytest.raises(GridError, match="unrepresentable radial axes"):
        QDHT(1e-3, 8)


@pytest.mark.parametrize("method", ["forward", "inverse"])
@pytest.mark.parametrize(
    "values", [[], [0.0] * 7, [[0.0] * 9], [np.nan] * 8, [np.inf] * 8, ["x"] * 8]
)
def test_hankel_rejects_invalid_samples(method, values):
    with pytest.raises(GridError, match="Hankel samples"):
        getattr(QDHT(1e-3, 8), method)(values)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import construction_error, gamma, gaussian_bound, gram_bound
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
    truncation, floating = gram_bound(transform)
    gram = truncation + floating
    rng = np.random.default_rng(310)
    values = rng.normal(size=(2, 3, size)) + 1j * rng.normal(size=(2, 3, size))
    scaled = np.abs(values) / transform.j1
    # Two real GEMMs for complex input, with one division and two scalings;
    # gamma(N+8) covers application roundoff independently of construction.
    absolute = np.abs(transform.matrix)
    product = absolute @ absolute
    application = gamma(size + 8) * (2 + gamma(size + 8)) * product
    bound = (scaled @ (gram + application)) * transform.j1
    spectrum = transform.forward(values)
    actual = transform.inverse(spectrum)
    assert np.all(np.abs(actual - values) <= bound)
    radial = np.sum(np.abs(values) ** 2 * transform.radial_weights, axis=-1)
    spectral = np.sum(np.abs(spectrum) ** 2 * transform.spectral_weights, axis=-1)
    energy_bound = 4 * np.pi * transform.forward_scale / transform.zeros[-1] * np.sum(
        scaled * (scaled @ (gram + application)), axis=-1
    ) + gamma(size + 12) * (radial + spectral)
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

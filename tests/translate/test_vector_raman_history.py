# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check long-delay coherence, taper bias and zero-padding on production kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.backends.numpy_ref.passes import raman_polarisation, tensor_raman_spectrum
from solprop.constants import EPS0
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.raman import (
    DampedOscillator,
    RamanResponse,
    RamanTensorTerm,
    undamped_dephasing,
)
from solprop.profiles import Constant
from solprop.translate.raman import kernel_taper
from solprop.translate.responses import raman_coefficients
from vector_raman_tensor_oracle import convolution_bound


def response():
    return RamanResponse(
        tensor_terms=(
            RamanTensorTerm(DampedOscillator(1.0, 2e13, undamped_dephasing()), "rank2"),
        ),
        thg=False,
    )


def grid(window):
    return EnvelopeGrid.from_limits(
        0.1, 800e-9, (600e-9, 1.2e-6), window, dt=0.5e-15, thg=False
    )


def compiled_polarisation(field, temporal):
    leaf = raman_coefficients(
        response(), None, temporal, Constant(1.0), n_pol=2, span=0.1, tol=1e-12
    )
    assert leaf is not None
    spectrum = tensor_raman_spectrum((leaf,), (0.0,), (1.0,), temporal.dt_fine)
    return raman_polarisation(
        field[None], spectrum, kind="envelope", thg=False, density=1.0
    )[0]


def test_delayed_orthogonal_probe_exposes_taper_bias_and_window_extension():
    short, long = grid(100e-15), grid(400e-15)
    assert short.dt_fine == long.dt_fine
    taper = kernel_taper(short)
    candidates = np.flatnonzero((taper > 0.25) & (taper < 0.75))
    index = int(candidates[len(candidates) // 2])
    delay = index * short.dt_fine
    expected = -0.25 * short.dt_fine * np.sin(2e13 * delay) * 1e-3 * EPS0
    assert abs(expected) > np.finfo(float).tiny
    outputs = []
    for temporal in (short, long):
        field = np.zeros((2, temporal.n_to), complex)
        field[0, 0] = 1.0
        field[1, index] = 1e-3
        actual = compiled_polarisation(field, temporal)
        hv, hr = response().tensor_impulse(
            np.arange(temporal.n_to) * temporal.dt_fine, 1.0
        )
        bound = EPS0 * convolution_bound(
            field, hv, hr, temporal.dt_fine, envelope=True, fft=True
        )
        assert_allclose(
            actual[1, index],
            expected * kernel_taper(temporal)[index],
            rtol=0,
            atol=bound,
        )
        outputs.append(actual[1, index])
    # Padding protects causality, not the long-delay tensor amplitude. Extending
    # the window moves this same physical lag back into the untapered region.
    assert kernel_taper(long)[index] == 1.0
    assert abs(outputs[0] / outputs[1]) < 0.75


@pytest.mark.parametrize("count", [17, 64, 97])
def test_last_sample_vector_excitation_has_no_wrapped_response(count):
    field = np.zeros((2, count), complex)
    field[:, -1] = (1.0, 0.4 + 0.3j)
    dt = 1e-15
    actual = response().evaluate(field, 1.0, kind="envelope", dt=dt)
    hv, hr = response().tensor_impulse(np.arange(count) * dt, 1.0)
    bound = EPS0 * convolution_bound(field, hv, hr, dt, envelope=True, fft=True)
    assert_allclose(actual, 0.0, rtol=0, atol=bound)

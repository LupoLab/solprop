# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Limiting-case and convergence checks of an oxygen-like molecular Raman response.

Driven by solprop's former oxygen constants (``former_raman.FORMER_O2``, from its
registry at dc4ca0d) through the supplied molecular route: a spin-free rotor with
no even states is the case these exercise. The checks of those constants against
their printed sources went with the registry (SEP 3, Phase 4b); optmat's O2 is
qualified in optmat.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from former_raman import FORMER_O2, former_response
from solprop.constants import EPS0, K_B
from solprop.nonlinear.raman import rotational_nonrigid


def test_oxygen_rotor_uses_only_the_allowed_odd_mechanical_rotation_states():
    parameters = FORMER_O2.rotational
    assert parameters is not None
    lines = rotational_nonrigid(parameters, temperature=295.0)
    even = lines.quantum_numbers % 2 == 0
    np.testing.assert_array_equal(lines.populations[even], 0.0)
    np.testing.assert_array_equal(lines.amplitudes[even[:-2]], 0.0)
    assert np.all(lines.populations[~even] > 0.0)
    assert np.all(lines.amplitudes[~even[:-2]] > 0.0)
    assert np.all(lines.frequencies > 0.0)
    assert float(lines.populations.sum()) == pytest.approx(1.0, abs=2e-15)


def test_oxygen_rotational_truncation_converges_at_room_temperature():
    parameters = FORMER_O2.rotational
    assert parameters is not None
    integrals = []
    for limit in (50, 80, 100):
        lines = rotational_nonrigid(parameters, temperature=295.0, max_j=limit)
        integrals.append(float(np.sum(lines.amplitudes / lines.frequencies)))
    # The default 50-state cutoff has a small Boltzmann tail for O2.
    # Refining to 80 then 100 resolves it without fitting a susceptibility.
    assert abs(integrals[0] / integrals[1] - 1.0) < 5e-8
    assert abs(integrals[1] / integrals[2] - 1.0) < 1e-12


def test_oxygen_response_is_causal_and_its_sampling_converges():
    response = former_response(FORMER_O2)
    density = 6e5 / (K_B * 295.0)
    np.testing.assert_array_equal(response.impulse(np.array([-1e-12, 0.0]), density), 0)
    # At the peak of a 10 fs Gaussian envelope the past intensity drive is
    # exp(-4 ln(2) t^2 / fwhm^2)/2. Independent adaptive integration is in fs
    # with a normalised kernel, avoiding tiny SI integrands and FFT reuse.
    scale = float(
        np.max(np.abs(response.impulse(np.linspace(0, 100e-15, 1001), density)))
    )
    assert scale > 0
    integral, error = quad(
        lambda delay_fs: (
            response.impulse(np.array([delay_fs * 1e-15]), density)[0]
            / scale
            * math.exp(-4 * math.log(2) * (delay_fs / 10.0) ** 2)
            / 2
        ),
        0.0,
        100.0,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    assert error < 1e-10
    errors = []
    for dt_fs in (1.0, 0.5, 0.25):
        time_fs = np.arange(-256.0, 256.0, dt_fs)
        field = np.exp(-2 * math.log(2) * (time_fs / 10.0) ** 2).astype(complex)
        polarisation = response.evaluate(
            field[None, :], density, kind="envelope", dt=dt_fs * 1e-15
        )
        actual = polarisation[0, len(time_fs) // 2].real / (
            EPS0 * density * scale * 1e-15
        )
        errors.append(abs(actual - integral))
    # The causal kernel is zero but has a nonzero right derivative at the
    # endpoint: the trapezoidal convolution has second-order error.
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1
    assert errors[-1] / abs(integral) < 2e-3

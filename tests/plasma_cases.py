# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The analytic plasma cases, shared by the response tests and the backend's.

:meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate` is the oracle for every
backend (``docs/dev/backends.md`` section 2), and the cases that pin it -- a constant
field and rate with its closed form, the Drude limit of a pre-ionised gas, and the
energy identity -- have to say the same thing about the reference backend's fused
passes as they do about the response. They live here so that there is one statement of
each closed form rather than two that can drift apart.

Checking the plasma polarisation only against itself -- the fused column against the
multipass one, the batched form against the columnwise one, and a scalar propagation
against a vector one at 0 and 45 degrees -- is not enough: none of these is an oracle.
"""

import numpy as np

from solprop.constants import E_CHARGE, M_E

CHARGE_MASS_RATIO = E_CHARGE**2 / M_E

DENSITY = 2.5e25  # argon at about one bar, m^-3

# A pulse strong enough to ionise argon appreciably: 5e10 V/m is 3.3e14 W/cm^2.
AMPLITUDE = 5e10
DURATION = 10e-15
CARRIER = 2.35e15  # 800 nm


def constant_rate(value):
    """A rate equal to ``value`` wherever the field is non-zero, and zero at zero field.

    The zero at zero field is what the loss term needs, and what the constructor checks.
    """

    def rate(field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(magnitude > 0.0, value, 0.0)

    return rate


def power_rate(scale=2e14, reference=AMPLITUDE, threshold=1e10):
    """A steep analytic rate with a threshold, standing in for a real one.

    Eighth power of the field, so it turns on over about the same range as a tunnelling
    rate does, but it carries no atomic units at all -- which is what lets the stored
    Luna witness compare the plasma arithmetic and nothing else.
    """

    def rate(field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(
            magnitude >= threshold, scale * (magnitude / reference) ** 8, 0.0
        )

    return rate


def zero_rate(field_amplitude):
    """A rate that never ionises anything, for the pre-ionised Drude limit."""
    return np.zeros_like(np.asarray(field_amplitude, dtype=float))


def pulse(n_samples=2**13 + 1, span=6.0, amplitude=AMPLITUDE):
    """A carrier-resolved Gaussian pulse and its time step, in V/m and s."""
    t = np.linspace(-span * DURATION, span * DURATION, n_samples)
    field = amplitude * np.exp(-((t / DURATION) ** 2)) * np.cos(CARRIER * t)
    return t, float(t[1] - t[0]), field


def constant_field_reference(t, field, rate, potential, preionisation):
    """The exact polarisation per atom for a constant field and a constant rate."""
    fraction = 1.0 - (1.0 - preionisation) * np.exp(-rate * t)
    free = (
        CHARGE_MASS_RATIO
        * field
        * (
            preionisation * t**2 / 2.0
            + (1.0 - preionisation)
            * (t**2 / 2.0 - (t - (1.0 - np.exp(-rate * t)) / rate) / rate)
        )
    )
    loss = potential * (1.0 - preionisation) * (1.0 - np.exp(-rate * t)) / field
    return fraction, free + loss


def gaussian_pulse_and_derivative(n_samples):
    """A pulse that starts and ends at zero, with its analytic time derivative."""
    t = np.linspace(-6.0 * DURATION, 6.0 * DURATION, n_samples)
    envelope = np.exp(-((t / DURATION) ** 2))
    field = AMPLITUDE * envelope * np.cos(CARRIER * t)
    derivative = (
        AMPLITUDE
        * envelope
        * (-2.0 * t / DURATION**2 * np.cos(CARRIER * t) - CARRIER * np.sin(CARRIER * t))
    )
    return t, float(t[1] - t[0]), field, derivative

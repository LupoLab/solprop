# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent tensor experiments; no production response/translation/backend imports.

The orientation quadrature supplies the rank-four oracle. Cartesian and circular
direct sums and the three-drive FFT candidate use separate contractions. All
experiments are dimensionless; physical scalar kernels are checked separately.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate, special

from solprop.stepper.schedule import CONCLUSIVE_FACTOR, ORDER_MARGIN, observed_order

BUDGETS = json.loads(
    Path(__file__)
    .with_name("data")
    .joinpath("vector_raman_oracle/budgets.json")
    .read_text()
)
EPS = np.finfo(float).eps
JONES = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)


def gamma(count):
    """Bound a floating-point reduction without assuming cancellation is small."""
    return count * EPS / (1 - count * EPS)


def causal_sum(kernel, drive, dt):
    """Sum past samples explicitly; math.fsum is separate from FFT arithmetic."""
    count = len(kernel)
    result = np.empty(count, dtype=np.result_type(drive, float))
    for at in range(count):
        terms = kernel[: at + 1] * drive[at::-1]
        real = math.fsum(float(one) for one in terms.real)
        result[at] = dt * (
            complex(real, math.fsum(float(one) for one in terms.imag))
            if np.iscomplexobj(drive)
            else real
        )
    return result


def quadratic(field, *, envelope):
    """Form a real symmetric history; its longitudinal Cartesian component is zero."""
    product = field[:, None, :] * field[None, :, :].conj()
    return product.real * (0.5 if envelope else 1)


def orientation_tensor():
    """Integrate the covariance of n_i n_j over the three-dimensional sphere.

    The factor 45/4 normalizes the longitudinal element to one. The quadrature
    does not insert a Cartesian Kronecker-delta identity into its integrand.
    """
    cosine, weights = special.roots_legendre(6)
    azimuth = np.arange(16) * (2 * np.pi / 16)
    directions = np.array(
        [
            [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
            for z in cosine
            for phi in azimuth
        ]
    )
    weights = np.repeat(weights / 32, 16)
    dyads = directions[:, :, None] * directions[:, None, :]
    mean = np.einsum("a,aij->ij", weights, dyads)
    covariance = np.einsum("a,aij,akl->ijkl", weights, dyads, dyads)
    covariance -= np.einsum("ij,kl->ijkl", mean, mean)
    return 45 / 4 * covariance


def orientation_direct(field, vibration, rotation, dt, *, envelope):
    """Contract quadrature-derived tensor components with direct history sums."""
    count = field.shape[1]
    history = np.zeros((3, 3, count))
    history[:2, :2] = quadratic(field, envelope=envelope)
    tensor = orientation_tensor()
    induced = np.zeros_like(history)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                for ell in range(2):
                    induced[i, j] += tensor[i, j, k, ell] * causal_sum(
                        rotation, history[k, ell], dt
                    )
    isotropic = causal_sum(vibration, np.trace(history, axis1=0, axis2=1), dt)
    for component in range(3):
        induced[component, component] += isotropic
    return np.einsum("ijt,jt->it", induced[:2, :2], field)


def cartesian_direct(field, vibration, rotation, dt, *, envelope):
    """Evaluate the physical channel contraction before any effective-kernel sum."""
    history = quadratic(field, envelope=envelope)
    trace = history[0, 0] + history[1, 1]
    isotropic = causal_sum(vibration, trace, dt)
    matrix = np.empty_like(history)
    for i in range(2):
        for j in range(2):
            drive = 3 * history[i, j] - (trace if i == j else 0)
            matrix[i, j] = 0.5 * causal_sum(rotation, drive, dt)
    matrix[0, 0] += isotropic
    matrix[1, 1] += isotropic
    return np.einsum("ijt,jt->it", matrix, field)


def circular_direct(circular, vibration, rotation, dt):
    """Evaluate both helicities of Eq. (32a) in scalar-kernel units without noise."""
    total = np.sum(circular.real**2 + circular.imag**2, axis=0)
    common = causal_sum(vibration / 2 + rotation / 8, total, dt)
    result = np.empty_like(circular)
    for p in range(2):
        other = 1 - p
        coherence = causal_sum(rotation, circular[p] * circular[other].conj(), dt)
        result[p] = common * circular[p] + 0.75 * coherence * circular[other]
    return result


def three_drive_fft(field, vibration, rotation, dt, *, envelope):
    """Prototype the three-drive FFT plan independently of direct sums."""
    history = quadratic(field, envelope=envelope)
    drives = np.array(
        [
            history[0, 0] + history[1, 1],
            history[0, 0] - history[1, 1],
            2 * history[0, 1],
        ]
    )
    count = field.shape[1]
    kernels = np.array([vibration + rotation / 4, 3 * rotation / 4])
    spectra = np.fft.rfft(kernels, n=2 * count, axis=-1) * dt
    transformed = np.fft.rfft(drives, n=2 * count, axis=-1)
    convolved = np.fft.irfft(transformed * spectra[[0, 1, 1]], n=2 * count, axis=-1)[
        :, :count
    ]
    trace, difference, cross = convolved
    return np.array(
        [
            (trace + difference) * field[0] + cross * field[1],
            cross * field[0] + (trace - difference) * field[1],
        ]
    )


def convolution_bound(field, vibration, rotation, dt, *, envelope, fft=False):
    """Use absolute summands, including matrix multiplication and optional FFT error."""
    history = quadratic(field, envelope=envelope)
    trace_abs = np.sum(np.abs(history), axis=(0, 1))
    magnitude = np.abs(vibration) + 2 * np.abs(rotation)
    scale = np.max(causal_sum(magnitude, trace_abs, dt)) * np.max(np.abs(field))
    count = field.shape[1]
    bound = gamma(32 * count) * scale
    if fft:
        # Three drive combinations, two kernel combinations and the final tensor
        # product are covered by the factor 8 multiplying the FFT's L1 bound.
        transform = 64 * EPS * math.ceil(math.log2(2 * count))
        scale_fft = dt * np.sum(magnitude) * np.max(trace_abs) * np.max(np.abs(field))
        bound += 8 * transform * scale_fft
    return max(float(bound), np.finfo(float).tiny)


def smooth_pulse(time):
    """Return a compact smooth envelope with varying ellipticity and relative phase."""
    window = np.where((time > 0) & (time < 8), np.sin(np.pi * time / 8) ** 2, 0)
    return np.array(
        [
            window * np.exp(0.2j * time),
            0.7 * window * np.exp(1j * (0.4 * time + 0.1 * time**2)),
        ]
    )


def oscillator_reference(time, *, rtol, atol):
    """Solve two independently driven tensor oscillators in dimensionless time.

    A exp(-gamma*t) sin(omega*t) solves y''+2 gamma y'+(omega²+gamma²)y
    = A omega drive. The decay is not silently substituted for a shifted frequency.
    """

    def rhs(at, state):
        envelope = smooth_pulse(np.array([at]))[:, 0]
        x, y = envelope
        drive = np.array(
            [abs(x) ** 2 / 2, abs(y) ** 2 / 2, (x * y.conjugate()).real / 2]
        )
        values = state.reshape(2, 2, 3)
        out = np.empty_like(values)
        for channel, (amplitude, omega, decay) in enumerate(
            ((0.7, 2.3, 0.17), (1.1, 0.9, 0.03))
        ):
            out[channel, 0] = values[channel, 1]
            out[channel, 1] = (
                amplitude * omega * drive
                - 2 * decay * values[channel, 1]
                - (omega**2 + decay**2) * values[channel, 0]
            )
        return out.ravel()

    solution = integrate.solve_ivp(
        rhs,
        (0, float(time[-1])),
        np.zeros(12),
        method="DOP853",
        t_eval=time,
        rtol=rtol,
        atol=atol,
        max_step=0.1,
    )
    assert solution.success, solution.message
    vibration, rotation = solution.y.reshape(2, 2, 3, len(time))[:, 0]
    scalar = vibration[0] + vibration[1]
    x, y = smooth_pulse(time)
    return np.array(
        [
            (scalar + rotation[0] - rotation[1] / 2) * x + 1.5 * rotation[2] * y,
            1.5 * rotation[2] * x + (scalar + rotation[1] - rotation[0] / 2) * y,
        ]
    )


def oscillator_experiment():
    """Record three temporal resolutions and a tighter independent ODE solve."""
    errors, rows, solutions = [], [], []
    for count in BUDGETS["temporal_counts"]:
        time = np.linspace(0, 12, count)
        vibration = 0.7 * np.sin(2.3 * time) * np.exp(-0.17 * time)
        rotation = 1.1 * np.sin(0.9 * time) * np.exp(-0.03 * time)
        field = smooth_pulse(time)
        actual = cartesian_direct(field, vibration, rotation, time[1], envelope=True)
        reference = oscillator_reference(
            time, rtol=BUDGETS["oscillator_rtol"], atol=BUDGETS["oscillator_atol"]
        )
        scale = float(np.max(np.abs(reference)))
        error = float(np.max(np.abs(actual - reference)) / scale)
        errors.append(error)
        rows.append({"count": count, "relative_error": error})
        solutions.append((time, reference, scale))
    time, reference, scale = solutions[-1]
    refined = oscillator_reference(
        time, rtol=BUDGETS["oscillator_rtol"] / 8, atol=BUDGETS["oscillator_atol"] / 8
    )
    repeat = float(np.max(np.abs(reference - refined)) / scale)
    orders = [observed_order(errors[i], errors[i + 1]) for i in range(2)]
    floor = max(repeat, 128 * EPS)
    conclusive = min(errors) > CONCLUSIVE_FACTOR * floor
    passed = conclusive and all(
        BUDGETS["temporal_order_lower"] < order <= 2 + ORDER_MARGIN for order in orders
    )
    return {
        "rows": rows,
        "orders": orders,
        "ode_repeat_error": repeat,
        "floor": floor,
        "order_pass": passed,
    }


def algebra_experiments():
    """Return absolute residuals and predeclared bounds for full complex histories."""
    rng = np.random.default_rng(4001)
    time = np.arange(97) * 0.07
    field = rng.normal(size=(2, 97)) + 1j * rng.normal(size=(2, 97))
    vibration = np.sin(1.7 * time) * np.exp(-0.2 * time)
    rotation = np.sin(0.7 * time) * np.exp(-0.03 * time)
    reference = cartesian_direct(field, vibration, rotation, 0.07, envelope=True)
    oriented = orientation_direct(field, vibration, rotation, 0.07, envelope=True)
    fft = three_drive_fft(field, vibration, rotation, 0.07, envelope=True)
    circular = JONES @ circular_direct(
        JONES.conj().T @ field, vibration, rotation, 0.07
    )
    angle = 0.731
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = cartesian_direct(rotate @ field, vibration, rotation, 0.07, envelope=True)
    return {
        "orientation_error": float(np.max(np.abs(oriented - reference))),
        "circular_error": float(np.max(np.abs(circular - reference))),
        "rotation_error": float(np.max(np.abs(rotated - rotate @ reference))),
        "fft_error": float(np.max(np.abs(fft - reference))),
        "direct_bound": convolution_bound(
            field, vibration, rotation, 0.07, envelope=True
        ),
        "fft_bound": convolution_bound(
            field, vibration, rotation, 0.07, envelope=True, fft=True
        ),
    }

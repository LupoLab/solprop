# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Causal-current, band assembly and averaging experiments independent of backends."""

import numpy as np
from scipy import integrate

from envelope_oracle_angular import order_record, population


def spectral_factors(omega, mask, *, carrier=None):
    """Internal FFT equivalent-polarisation multipliers, without a UPPE factor."""
    supported = mask & (omega > 0)
    drude = np.zeros(omega.size, dtype=np.float64)
    loss = np.zeros(omega.size, dtype=np.complex128)
    divisor = omega[supported] if carrier is None else carrier
    drude[supported] = -1 / divisor**2
    loss[supported] = -1j / divisor
    return drude, loss


def spectral_experiments():
    omega = np.array([-1.0, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    mask = (omega >= 0.75) & (omega <= 1.5)
    drude, loss = spectral_factors(omega, mask)
    frozen_drude, frozen_loss = spectral_factors(omega, mask, carrier=1.0)
    # Prescribed density and current: an algebraic source test.
    prefactor = -1j * omega
    exact_phase = prefactor * drude
    exact_loss = prefactor * loss
    # Relabel the same absolute bins. The atomic rate's wavelength remains fixed.
    relabelled = []
    for carrier in (0.8, 1.0, 1.2):
        detuning = omega - carrier
        other, _ = spectral_factors(carrier + detuning, mask)
        relabelled.append(float(np.max(np.abs(other - drude))))
    return {
        "omega": omega.tolist(),
        "drude": drude.tolist(),
        "loss_imaginary": loss.imag.tolist(),
        "uppe_drude_imaginary": exact_phase.imag.tolist(),
        "uppe_loss_real": exact_loss.real.tolist(),
        "shock_false_drude_imaginary": (-1j * frozen_drude).imag.tolist(),
        "shock_false_loss_real": (-1j * frozen_loss).real.tolist(),
        "relabel_error": max(relabelled),
        "inactive_exact_zero": bool(
            np.all(drude[~mask] == 0) and np.all(loss[~mask] == 0)
        ),
    }


def causal_case(half_window, padding, samples_per_unit, carrier):
    """Double causal integration vs separately windowed drive/current FFT sources.

    Padding zeros AFTER a fixed physical window and collar. Window extension
    changes the physical data and the collar width. They are separate controls.
    """
    count = int(2 * half_window * samples_per_unit)
    time = np.arange(count) / samples_per_unit - half_window
    envelope = np.exp(-(time**2) / 2)
    density = 0.1 + 0.9 / (1 + np.exp(-2 * time))
    drive = density * envelope * np.cos(carrier * time + 0.2)
    current = integrate.cumulative_trapezoid(drive, x=time, initial=0)
    polarisation = integrate.cumulative_trapezoid(current, x=time, initial=0)
    # C2 collar, with inner half of the time window untouched.
    collar_coordinate = np.clip((np.abs(time) / half_window - 0.5) / 0.5, 0, 1)
    collar = 1 - (
        10 * collar_coordinate**3 - 15 * collar_coordinate**4 + 6 * collar_coordinate**5
    )
    omega = 2 * np.pi * np.fft.fftfreq(count * padding, d=1 / samples_per_unit)
    band = (omega > carrier / 2) & (omega < 1.5 * carrier)
    drude_factor, loss_factor = spectral_factors(omega, band)
    source = drude_factor * np.fft.fft(collar * drive, n=count * padding)
    direct = np.fft.fft(collar * polarisation, n=count * padding)
    bare = np.fft.fft(polarisation, n=count * padding)
    loss_source = loss_factor * np.fft.fft(collar * drive, n=count * padding)
    integrated_loss = np.fft.fft(collar * current, n=count * padding)
    scale = np.linalg.norm(source[band])
    return {
        "half_window": half_window,
        "padding": padding,
        "samples_per_unit": samples_per_unit,
        "carrier": carrier,
        "collared_relative_error": float(
            np.linalg.norm((direct - source)[band]) / scale
        ),
        "uncollared_relative_error": float(
            np.linalg.norm((bare - source)[band]) / scale
        ),
        "loss_integral_relative_error": float(
            np.linalg.norm((integrated_loss - loss_source)[band])
            / np.linalg.norm(loss_source[band])
        ),
        "residual_current": float(current[-1]),
        "end_polarisation": float(polarisation[-1]),
        "drive_at_boundary": float(max(abs(drive[0]), abs(drive[-1]))),
    }


def causal_experiments():
    windows = [
        causal_case(width, padding, 4096, carrier)
        for carrier in (2, 12)
        for width in (4, 8, 16)
        for padding in (1, 2, 4)
    ]
    # Small residual current and a remote collar leave the trapezoid error dominant.
    levels = [causal_case(16, 2, count, 12) for count in (128, 256, 512)]
    return {
        "windows": windows,
        "temporal": order_record(
            [row["collared_relative_error"] for row in levels], 2, 1e-12
        ),
        "levels": levels,
    }


def carrier_envelope_case(cycles, depletion, ratio, samples_per_cycle):
    """Compare field-current fundamental with cycle moments for W=k E^4.

    Time is in carrier radians. k sets peak rate / carrier frequency;
    cycles sets inverse relative bandwidth independently of k.
    """
    width = 2 * np.pi * cycles
    count = int(12 * cycles * samples_per_cycle)
    time = (np.arange(count) - count / 2) * 2 * np.pi / samples_per_cycle
    amplitude = np.exp(-(time**2) / (2 * width**2))
    envelope = np.array([amplitude, 1j * ratio * amplitude])
    field = (envelope * np.exp(-1j * time)).real
    square = np.sum(field**2, axis=0)
    rate = depletion * square**2
    fraction, survival = population(rate, 2 * np.pi / samples_per_cycle, 0)
    averaged_rate = depletion * amplitude**4 * (3 + 2 * ratio**2 + 3 * ratio**4) / 8
    averaged_fraction, averaged_survival = population(
        averaged_rate, 2 * np.pi / samples_per_cycle, 0
    )
    field_loss = depletion * survival * field * square
    envelope_loss = (
        depletion
        * averaged_survival
        * np.array(
            [
                amplitude**3 * (3 + ratio**2),
                1j * ratio * amplitude**3 * (1 + 3 * ratio**2),
            ]
        )
        / 4
    )
    frequency = 2 * np.pi * np.fft.fftfreq(count, d=2 * np.pi / samples_per_cycle)
    band = (frequency > 0.5) & (frequency < 1.5)
    # Conjugate optics envelopes at the FFT interface, then remodulate internally.
    loss_actual = 2 * np.fft.fft(field_loss, axis=-1)[:, band]
    loss_expected = np.fft.fft(envelope_loss.conj() * np.exp(1j * time), axis=-1)[
        :, band
    ]
    drive_actual = 2 * np.fft.fft(fraction * field, axis=-1)[:, band]
    drive_expected = np.fft.fft(
        (averaged_fraction * envelope).conj() * np.exp(1j * time), axis=-1
    )[:, band]
    components = []
    for component in range(2):
        denominator = np.linalg.norm(loss_expected[component])
        components.append(
            0.0
            if denominator == 0
            else float(
                np.linalg.norm(loss_actual[component] - loss_expected[component])
                / denominator
            )
        )
    return {
        "cycles_sigma": cycles,
        "bandwidth_over_carrier": 1 / width,
        "peak_instantaneous_rate_over_carrier": depletion,
        "ellipticity": ratio,
        "samples_per_cycle": samples_per_cycle,
        "loss_component_relative_errors": components,
        "drude_relative_error": float(
            np.linalg.norm(drive_actual - drive_expected)
            / np.linalg.norm(drive_expected)
        ),
        "population_max_error": float(np.max(np.abs(fraction - averaged_fraction))),
        "loss_work_error": float(
            abs(
                integrate.trapezoid(np.sum(field * field_loss, axis=0), x=time)
                - fraction[-1]
            )
        ),
    }


def carrier_envelope_experiments():
    # Independent sweeps; coupled refinement could hide which approximation fails.
    rows = [
        carrier_envelope_case(cycles, depletion, ratio, 128)
        for cycles in (0.25, 2, 8)
        for depletion in (1e-3, 0.05, 0.5)
        for ratio in (0, 0.35, 1)
    ]
    temporal = [carrier_envelope_case(2, 0.05, 0.35, count) for count in (32, 64, 128)]
    return {
        "cases": rows,
        "work_temporal": order_record(
            [row["loss_work_error"] for row in temporal], 2, 64 * np.finfo(float).eps
        ),
        "levels": temporal,
    }

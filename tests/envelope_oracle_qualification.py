# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare production currents with independent carrier integration and exact work.

The synthetic W = k omega0 (E/Eref)^4 law isolates envelope averaging from atomic
rate accuracy. SI fields, times, current units and the internal FFT convention
are retained. The oracle budgets remain fixed and their source files are unchanged.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import integrate

from envelope_oracle_angular import order_record
from mixture_cases import AnalyticRate
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.envelope_plasma import make_source_column
from solprop.constants import CHARGE_MASS_RATIO, C
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.physdata import ionisation_potential
from solprop.translate.envelope_plasma import build_cycle_integrals
from solprop.translate.ionisation import build_rate_table


@dataclass(frozen=True)
class InstantaneousPolynomial(AnalyticRate):
    convention: str = "instantaneous-isotropic"


@cache
def compiled_cycle(depletion):
    omega = 2 * np.pi * C / 800e-9
    rate = InstantaneousPolynomial(
        scale=depletion * omega,
        reference=1e10,
        exponent=4,
        threshold=1.0,
        maximum=1e30,
    )
    table = build_rate_table(rate, (1.0, 1e11), rtol=1e-7)
    cycle = build_cycle_integrals(EnvelopePlasmaResponse("Ar", table), None)
    arrays = [np.zeros((4, 512)) for _ in range(2)]
    for level, rule in enumerate(cycle.rules):
        arrays[0][level, : len(rule.nodes)] = rule.nodes
        arrays[1][level, : len(rule.weights)] = rule.weights
    for array in arrays:
        array.setflags(write=False)
    return RateArrays.of(cycle.rate), tuple(arrays)


@cache
def source_kernel():
    return make_source_column(2, parallel=False)


def source_window(radians, window_sigmas, width, collared):
    window = np.ones(radians.size)
    if collared:
        coordinate = np.clip(4 * np.abs(radians) / (window_sigmas * width) - 1, 0, 1)
        window -= 10 * coordinate**3 - 15 * coordinate**4 + 6 * coordinate**5
    window.setflags(write=False)
    return window


def carrier_comparison(
    cycles, depletion, ratio, samples_per_cycle, *, window_settings=(12, 1, False)
):
    omega = 2 * np.pi * C / 800e-9
    width = 2 * np.pi * cycles
    window_sigmas, padding, collared = window_settings
    count = int(window_sigmas * cycles * samples_per_cycle)
    radians = (np.arange(count) - count / 2) * 2 * np.pi / samples_per_cycle
    dt = 2 * np.pi / (omega * samples_per_cycle)
    time = radians / omega
    amplitude = 1e10 * np.exp(-(radians**2) / (2 * width**2))
    # Raw forward FFTs represent the internal +i omega t carrier. The optical
    # Jones vector is (1, +i q); only this existing boundary conjugates it.
    envelope = np.asarray([amplitude, -1j * ratio * amplitude])
    field = (envelope * np.exp(1j * radians)).real
    square = np.sum(field**2, axis=0)
    rate = depletion * omega * (square / 1e20) ** 2
    exposure = integrate.cumulative_trapezoid(rate, dx=dt, initial=0)
    fraction, survival = -np.expm1(-exposure), np.exp(-exposure)
    potential, density = ionisation_potential("Ar"), 1e24
    # W/E^2 = k omega E^2/Eref^4 extends continuously through real-field zero.
    real_loss = (
        potential * density * survival * depletion * omega / 1e40 * field * square
    )
    real_drive = CHARGE_MASS_RATIO * density * fraction * field
    source = np.zeros((2, 2, count), dtype=np.complex128)
    histories, status = np.empty((5, 1, count)), np.zeros(1, dtype=np.int64)
    window = source_window(radians, window_sigmas, width, collared)
    table, rules = compiled_cycle(depletion)
    source_kernel()(
        envelope,
        (source, histories, status),
        table,
        rules,
        (density, 0.0, CHARGE_MASS_RATIO, potential, dt),
        window,
    )
    if np.any(status):
        raise AssertionError(
            f"production angular refinement failed at {status.tolist()}"
        )
    transform_size = count * padding
    frequency = (
        2 * np.pi * np.fft.fftfreq(transform_size, d=2 * np.pi / samples_per_cycle)
    )
    band = (frequency > 0.5) & (frequency < 1.5)
    spectra = np.fft.fft(source * np.exp(1j * radians), n=transform_size, axis=-1)[
        :, :, band
    ]
    carrier_spectra = (
        2
        * np.fft.fft(
            np.array([real_drive, real_loss]) * window, n=transform_size, axis=-1
        )[:, :, band]
    )
    real_current = integrate.cumulative_trapezoid(real_drive, dx=dt, initial=0)
    current_actual = (
        2 * np.fft.fft(real_current * window, n=transform_size, axis=-1)[:, band]
    )
    current_expected = spectra[0] / (1j * omega * frequency[band])
    relative = []
    for current in range(2):
        components = []
        for component in range(2):
            norm = np.linalg.norm(spectra[current, component])
            components.append(
                0.0
                if norm == 0
                else float(
                    np.linalg.norm(
                        carrier_spectra[current, component]
                        - spectra[current, component]
                    )
                    / norm
                )
            )
        relative.append(components)
    # Exact polynomial moments and analytic Gaussian exposure are independent of
    # the source kernel, rate interpolation and angular quadrature.
    coefficient = (3 + 2 * ratio**2 + 3 * ratio**4) / 8
    average_rate = depletion * omega * (amplitude / 1e10) ** 4 * coefficient
    average_exposure = integrate.cumulative_trapezoid(average_rate, dx=dt, initial=0)
    analytic_fraction = -np.expm1(-average_exposure)
    analytic_survival = np.exp(-average_exposure)
    exact_loss = (
        potential
        * density
        * depletion
        * omega
        / 4e40
        * analytic_survival
        * window
        * np.array(
            [
                amplitude**3 * (3 + ratio**2),
                -1j * ratio * amplitude**3 * (1 + 3 * ratio**2),
            ]
        )
    )
    source_error = float(
        np.linalg.norm(source[1] - exact_loss) / np.linalg.norm(exact_loss)
    )
    work = integrate.trapezoid(
        np.sum((envelope.conj() * source[1]).real, axis=0) / 2, x=time
    )
    work_target = potential * density * histories[3, 0, -1]
    real_work = integrate.trapezoid(np.sum(field * real_loss, axis=0), x=time)
    return {
        "cycles_sigma": cycles,
        "peak_rate_over_carrier": depletion,
        "ratio": ratio,
        "samples_per_cycle": samples_per_cycle,
        "bandwidth_over_carrier": 1 / width,
        "window_sigmas": window_sigmas,
        "padding": padding,
        "collared": collared,
        "field_boundary_over_peak": float(max(amplitude[0], amplitude[-1]) / 1e10),
        "drude_component_relative_errors": relative[0],
        "loss_component_relative_errors": relative[1],
        "causal_current_relative_error": float(
            np.linalg.norm(current_actual - current_expected)
            / np.linalg.norm(current_expected)
        ),
        "residual_current_over_peak": float(
            np.linalg.norm(real_current[:, -1])
            / np.max(np.linalg.norm(real_current, axis=0))
        ),
        "production_loss_relative_error": source_error,
        "production_population_error": float(
            np.max(np.abs(histories[3, 0] - analytic_fraction))
        ),
        "envelope_work_relative_error": float(abs(work - work_target) / work_target),
        "carrier_work_relative_error": float(
            abs(real_work - potential * density * fraction[-1])
            / (potential * density * fraction[-1])
        ),
    }


def work_convergence(ratio):
    levels = [
        carrier_comparison(2, 0.05, ratio, resolution) for resolution in (16, 32, 64)
    ]
    # Polynomial moments/table log interpolation resolve to rounding. The
    # independently declared rounding floor must be separated from O(dt^2).
    floor = 64 * np.finfo(float).eps
    return {
        "levels": levels,
        "envelope": order_record(
            [one["envelope_work_relative_error"] for one in levels], 2, floor
        ),
        "carrier": order_record(
            [one["carrier_work_relative_error"] for one in levels], 2, floor
        ),
    }


def controlled_cases():
    return [
        carrier_comparison(cycles, depletion, ratio, 64)
        for cycles in (0.25, 2, 8)
        for depletion in (1e-3, 0.05, 0.5)
        for ratio in (0.0, 0.35, 1.0)
    ]


def window_cases():
    """Vary physical extent, post-window zero padding and source collar separately."""
    return [
        carrier_comparison(
            cycles, 0.05, 0.35, 128, window_settings=(extent, padding, collar)
        )
        for cycles in (0.25, 8)
        for extent in (6, 12, 24)
        for padding in (1, 2, 4)
        for collar in (False, True)
    ]


def mixture_work_convergence():
    """Isolate summed ionisation work from dispersive energy exchange."""
    rows = []
    omega = 2 * np.pi * C / 800e-9
    for samples in (16, 32, 64):
        radians = (np.arange(24 * samples) - 12 * samples) * 2 * np.pi / samples
        dt = 2 * np.pi / (omega * samples)
        amplitude = 1e10 * np.exp(-(radians**2) / (2 * (4 * np.pi) ** 2))
        field = np.asarray([amplitude, -0.35j * amplitude])
        loss = np.zeros_like(field)
        target, inventories = 0.0, []
        for species, density, initial, depletion in (
            ("Ar", 3e23, 0.1, 0.05),
            ("He", 7e23, 0.35, 0.08),
        ):
            potential = ionisation_potential(species)
            sources = np.zeros((2, *field.shape), dtype=np.complex128)
            histories = np.empty((5, 1, radians.size))
            status = np.zeros(1, dtype=np.int64)
            table, rules = compiled_cycle(depletion)
            window = np.ones(radians.size)
            window.setflags(write=False)
            source_kernel()(
                field,
                (sources, histories, status),
                table,
                rules,
                (density, initial, CHARGE_MASS_RATIO, potential, dt),
                window,
            )
            if np.any(status):
                raise AssertionError("mixture angular refinement failed")
            loss += sources[1]
            newly_ionised = density * (histories[3, 0, -1] - initial)
            inventories.append(float(potential * newly_ionised))
            target += potential * newly_ionised
        work = integrate.trapezoid(
            np.sum((field.conj() * loss).real, axis=0) / 2, dx=dt
        )
        rows.append(
            {
                "samples_per_cycle": samples,
                "work_J_per_m3": float(work),
                "species_inventory_change_J_per_m3": inventories,
                "relative_error": float(abs(work - target) / target),
            }
        )
    return {
        "levels": rows,
        "order": order_record(
            [one["relative_error"] for one in rows], 2, 64 * np.finfo(float).eps
        ),
    }

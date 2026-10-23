# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Separate oscillator integration, carrier averaging and pump-sideband reductions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from envelope_oracle_angular import order_record
from solprop.backends.registry import select
from solprop.constants import EPS0, C
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, RamanTensorTerm
from solprop.physdata.raman_types import Dephasing, Linewidth
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from vector_raman_tensor_oracle import (
    BUDGETS,
    JONES,
    orientation_tensor,
    oscillator_reference,
    smooth_pulse,
)


def oscillator_response():
    """Use the independently driven oracle oscillator pair without changing units."""
    return RamanResponse(
        tensor_terms=tuple(
            RamanTensorTerm(
                DampedOscillator(amplitude, omega, Dephasing(Linewidth(decay / np.pi))),
                "isotropic" if role == "isotropic" else "rank2",
            )
            for amplitude, omega, decay, role in (
                (0.7, 2.3, 0.17, "isotropic"),
                (1.1, 0.9, 0.03, "rank2"),
            )
        )
    )


@pytest.mark.slow
def test_production_tensor_converges_to_independent_driven_oscillators():
    errors = []
    for count in BUDGETS["temporal_counts"]:
        time = np.linspace(0, 12, count)
        reference = oscillator_reference(
            time, rtol=BUDGETS["oscillator_rtol"], atol=BUDGETS["oscillator_atol"]
        )
        actual = (
            oscillator_response().evaluate(
                smooth_pulse(time), 1.0, kind="envelope", dt=time[1]
            )
            / EPS0
        )
        errors.append(float(np.max(abs(actual - reference)) / np.max(abs(reference))))
    # The oracle fixed both ODE tolerances before production comparisons. The
    # independently repeated solve bounded its normalized error by 2e-12 here.
    assert order_record(errors, 2, 2e-12)["status"] == "pass"


@pytest.mark.slow
@pytest.mark.parametrize("helicity", [-1, 1])
def test_resolved_carrier_and_envelope_agree_at_three_temporal_resolutions(helicity):
    # The carrier is far above both oscillator resonances. Project only the
    # fundamental sideband; third harmonics and carrier averaging are separate
    # physics, with the accepted 1% model budget independent of FFT rounding.
    carrier = 60.0
    for count in (4096, 8192, 16384):
        time = np.arange(count) * (16 / count)
        envelope = smooth_pulse(time)
        envelope[1] *= helicity
        field = np.real(envelope * np.exp(1j * carrier * time))
        response = oscillator_response()
        resolved = response.evaluate(field, 1.0, kind="field", dt=time[1]) / EPS0
        expected = response.evaluate(envelope, 1.0, kind="envelope", dt=time[1]) / EPS0
        frequency = 2 * np.pi * np.fft.fftfreq(count, time[1])
        positive_band = (frequency > carrier - 15) & (frequency < carrier + 15)
        analytic = 2 * np.fft.ifft(np.fft.fft(resolved) * positive_band)
        recovered = analytic * np.exp(-1j * carrier * time)
        relative = np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
        assert relative < BUDGETS["carrier_averaging_relative_target"]


def frequency_response(omega, amplitude, resonance, decay):
    """Integrate a causal damped sine against exp(-i omega t) analytically."""
    return amplitude * resonance / ((decay + 1j * omega) ** 2 + resonance**2)


def linearised_sidebands(pump, stokes, anti_stokes, omega, *, decay=None):
    """Linearise orientation-averaged dipoles before evaluating any convolution.

    Both sidebands are retained: the positive-frequency quadratic coefficient
    contains anti-Stokes conjugates, even when its launched amplitude is zero.
    """
    tensor = orientation_tensor()[:2, :2, :2, :2]

    def susceptibility(q, frequency):
        hv = frequency_response(frequency, 0.7, 2.3, 0.17 if decay is None else decay)
        hr = frequency_response(frequency, 1.1, 0.9, 0.03 if decay is None else decay)
        return hv * np.trace(q) * np.eye(2) + hr * np.einsum("ijkl,kl->ij", tensor, q)

    equilibrium = susceptibility(np.real(np.outer(pump, pump.conj())) / 2, 0)
    positive = (
        np.outer(pump, anti_stokes.conj())
        + np.outer(stokes, pump.conj())
        + np.outer(pump.conj(), stokes)
        + np.outer(anti_stokes.conj(), pump)
    ) / 4
    negative = positive.conj()
    return (
        equilibrium @ stokes + susceptibility(positive, omega) @ pump,
        equilibrium @ anti_stokes + susceptibility(negative, -omega) @ pump,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "state", ["co-linear", "cross-linear", "co-circular", "cross-circular"]
)
def test_weak_pump_stokes_linearisation_retains_generated_anti_stokes(state):
    pump = np.array([1, 0], complex) if "linear" in state else JONES[:, 0]
    stokes = (
        pump
        if state.startswith("co-")
        else (np.array([0, 1], complex) if "linear" in state else JONES[:, 1])
    )
    anti_stokes = np.zeros(2, complex)
    omega = 0.9
    period = 2 * np.pi / omega
    # Twenty decay times remove the switch-on transient before projecting four
    # full periods. Central differencing removes even perturbation powers;
    # epsilon^2 bounds the remaining cubic correction independently of dt.
    dt = period / 512
    count = 100 * 512
    time = np.arange(count) * dt
    epsilon = 1e-4
    perturbation = stokes[:, None] * np.exp(1j * omega * time)
    response = oscillator_response()
    derivative = (
        response.evaluate(
            pump[:, None] + epsilon * perturbation, 1.0, kind="envelope", dt=dt
        )
        - response.evaluate(
            pump[:, None] - epsilon * perturbation, 1.0, kind="envelope", dt=dt
        )
    ) / (2 * epsilon * EPS0)
    sample = slice(-4 * 512, None)
    actual = (
        np.mean(derivative[:, sample] * np.exp(-1j * omega * time[sample]), axis=1),
        np.mean(derivative[:, sample] * np.exp(1j * omega * time[sample]), axis=1),
    )
    expected = linearised_sidebands(pump, stokes, anti_stokes, omega)
    # Euler-Maclaurin: the sine kernel has h(0)=0; trapezoidal error starts
    # with dt^2 h'(0)/12. Add projection/cancellation rounding and the bounded
    # cubic perturbation. This is separate from the accepted gain target.
    absolute = 4 * dt**2 * (0.7 * 2.3 + 1.1 * 0.9) / 12 + 1e-6
    assert absolute < BUDGETS["propagation_gain_relative_target"]
    for got, reference in zip(actual, expected, strict=True):
        assert_allclose(got, reference, rtol=0, atol=absolute)
    if state == "cross-circular":
        # Opposite helicities cancel this particular generated sideband by the
        # orientation contraction; the other three launches retain it.
        assert np.linalg.norm(expected[1]) < 128 * np.finfo(float).eps
    else:
        assert np.linalg.norm(expected[1]) > 0.1


def gain_experiment(state, backend_name="numpy_ref", *, steps=8, ffac=16):
    """Compare a full run with an independently linearised Stokes/anti-Stokes system."""
    # Synthetic fast-dephasing lines put the centre of this finite window in
    # steady state. Material uncertainties play no role in this normalization
    # and propagation check. No Kerr, dispersion, shock, loss or noise is added.
    length, scale, amplitude, decay = 1e-3, 1e14, 1e6, 0.4
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 8e-12, dt=4e-15, ffac=ffac
    )
    pump = np.array([1, 0], complex) if "linear" in state else JONES[:, 0]
    stokes = (
        pump
        if state.startswith("co-")
        else (np.array([0, 1], complex) if "linear" in state else JONES[:, 1])
    )
    index = round(0.9 * scale * grid.n_t * grid.dt / (2 * np.pi))
    omega = -2 * np.pi * index / (grid.n_t * grid.dt)
    seed = 1e-5
    temporal = pump[:, None] + seed * stokes[:, None] * np.exp(1j * omega * grid.t)
    response = RamanResponse(
        tensor_terms=(
            RamanTensorTerm(
                DampedOscillator(
                    0.7 * amplitude,
                    2.3 * scale,
                    Dephasing(Linewidth(decay * scale / np.pi)),
                ),
                "isotropic",
            ),
            RamanTensorTerm(
                DampedOscillator(
                    1.1 * amplitude,
                    0.9 * scale,
                    Dephasing(Linewidth(decay * scale / np.pi)),
                ),
                "rank2",
            ),
        )
    )
    mode = SpecifiedMode.from_taylor([0.0, 0.0], omega_ref=grid.omega0, aeff=1.0)
    problem = assemble(
        grid,
        mode,
        [response],
        np.fft.fft(temporal, axis=-1)[None],
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        controller=PIController(rtol=1e-10, atol=1e-16),
        save_planes=[length],
        dz_init=length / steps,
        dz_max=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
    )
    # Synthesis/projection of a unit-area square mode gives this coefficient
    # from the SI UPPE prefactor and |E|^2 = 2 P/(eps0 c Aeff).
    coefficient = grid.omega0 * amplitude / (EPS0 * C**2 * scale)
    q = np.real(np.outer(pump, pump.conj())) / 2
    tensor = orientation_tensor()[:2, :2, :2, :2]
    matrix0 = frequency_response(0, 0.7, 2.3, decay) * np.trace(q) * np.eye(2)
    matrix0 += frequency_response(0, 1.1, 0.9, decay) * np.einsum(
        "ijkl,kl->ij", tensor, q
    )
    pump_phase = coefficient * float(np.real(pump.conj() @ matrix0 @ pump))
    generator = np.empty((4, 4), complex)
    for column in range(4):
        basis = np.eye(4, dtype=complex)[:, column]
        positive, negative = linearised_sidebands(
            pump, basis[:2], basis[2:].conj(), omega / scale, decay=decay
        )
        generator[:, column] = np.concatenate(
            [-1j * coefficient * positive, 1j * coefficient * negative.conj()]
        )
    generator += np.diag([1j * pump_phase] * 2 + [-1j * pump_phase] * 2)
    expected = expm(length * generator) @ np.concatenate([seed * stokes, np.zeros(2)])
    owner = select(backend_name, problem, grid=grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        owner.propagate_to(length)
        final = np.fft.ifft(owner.observe().require_field()[0, 0], axis=-1)
    finally:
        owner.close()
    # Observe well inside the flat collar, >100 dephasing times after turn-on.
    centre = abs(grid.t) < 1e-12
    design = np.column_stack(
        [
            np.ones(np.count_nonzero(centre)),
            np.exp(1j * omega * grid.t[centre]),
            np.exp(-1j * omega * grid.t[centre]),
        ]
    )
    fitted = np.linalg.lstsq(design, final[:, centre].T, rcond=None)[0]
    actual = np.concatenate(
        [
            fitted[1] * np.exp(1j * pump_phase * length),
            fitted[2].conj() * np.exp(-1j * pump_phase * length),
        ]
    )
    error = np.linalg.norm(actual - expected) / seed
    return {
        "state": state,
        "backend": backend_name,
        "n_to": grid.n_to,
        "steps": steps,
        "sideband_relative_error": float(error),
        "pump_fractional_change": float(abs(np.linalg.norm(fitted[0]) ** 2 - 1)),
        "stokes_amplitude_ratio": float(np.linalg.norm(actual[:2]) / seed),
        "anti_stokes_amplitude_over_seed": float(np.linalg.norm(actual[2:]) / seed),
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "state", ["co-linear", "cross-linear", "co-circular", "cross-circular"]
)
def test_complete_undepleted_gain_matches_independent_sideband_evolution(state):
    record = gain_experiment(state)
    assert (
        record["sideband_relative_error"]
        < BUDGETS["propagation_numerical_relative_target"]
    )
    assert (
        record["pump_fractional_change"]
        < BUDGETS["undepleted_pump_fractional_change_max"]
    )

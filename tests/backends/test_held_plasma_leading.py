# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recover the occupied leading taper's offset and held tail, including DC data."""

import math

import numpy as np
from scipy.integrate import quad, quad_vec
from tests.backends.test_held_plasma import EPS, grid_case

from solprop.backends.numpy_ref.passes import held_plasma_spectrum


def _recover(grid, current):
    """Restore the ordinary continuation from the source and its zero modes."""
    count, dt = grid.n_to, grid.dt_fine
    collar = grid.t_collar
    gate, slope = grid.t_window_fine, collar.window_derivative(grid.t_fine)
    edge = np.fft.rfft(slope * current)
    omega = 2 * np.pi * np.fft.rfftfreq(count, dt)
    active = omega > 0
    assembled = held_plasma_spectrum(
        gate * current, slope * current, omega, np.arange(omega.size), active
    )
    spectrum = 1j * omega * assembled
    # Nonzero optical frequencies cannot determine a constant or a secular
    # trend. Supply the zero-mode bookkeeping dictated by the continuation,
    # rather than trying to infer it from the active optical band.
    spectrum[0] = np.sum(gate * current)
    primitive = np.zeros_like(edge)
    primitive[1:] = edge[1:] / (1j * omega[1:])
    origin = np.fft.irfft(primitive, n=count)[0]
    held_tail = -dt * edge[0].real
    recovered = (
        np.fft.irfft(spectrum, n=count) + origin + held_tail * np.arange(count) / count
    )
    return recovered, held_tail, edge


def test_preionised_leading_pulse_recovers_the_offset_and_tail(record_property):
    grid = grid_case()
    collar, count, dt = grid.t_collar, grid.n_to, grid.dt_fine
    period = count * dt
    left, right = collar.zero_low, collar.one_low
    width = right - left
    centre, sigma, initial = (left + right) / 2, width / 8, 0.2
    # In units e^2/m_e = 1, a fixed pre-ionised fraction and E = dG/dt
    # give J = f0 (G-G(t0)). There is no new population or loss current.
    start = math.exp(-(((grid.t_fine[0] - centre) / sigma) ** 2))

    def current_at(times):
        return initial * (np.exp(-(((times - centre) / sigma) ** 2)) - start)

    def edge_at(position):
        times = np.asarray([position * period])
        return period * collar.window_derivative(times)[0] * current_at(times[0])

    points = np.asarray([left, centre, right]) / period
    c_minus, c_error = quad(edge_at, -0.5, right / period, points=points, epsabs=1e-13)
    total, total_error = quad(edge_at, -0.5, 0.5, points=points, epsabs=1e-13)
    bins = np.arange(1, count // 2 + 1)
    coefficients, coefficient_error = quad_vec(
        lambda position: (
            edge_at(position) * np.exp(-2j * np.pi * bins * (position + 0.5))
        ),
        -0.5,
        0.5,
        points=points,
        epsabs=1e-13,
        epsrel=1e-13,
        norm="max",
        limit=10000,
    )

    def edge_variation(position):
        # Analytic second derivative of the Planck step, from g'=g(1-g)a.
        # Both tapers contribute, including the constant subtracted at t0.
        times = np.asarray([position * period])
        t = times[0]
        if left < t < right:
            lo, hi, direction = left, right, 1.0
        elif collar.one_high < t < collar.zero_high:
            lo, hi, direction = collar.one_high, collar.zero_high, -1.0
        else:
            return 0.0
        gate = collar.window(times)[0]
        slope = collar.window_derivative(times)[0]
        a = direction * (hi - lo) * ((t - lo) ** -2 + (t - hi) ** -2)
        da = -2 * direction * (hi - lo) * ((t - lo) ** -3 + (t - hi) ** -3)
        second = slope * (1 - 2 * gate) * a + gate * (1 - gate) * da
        current = current_at(t)
        derivative = (
            -2
            * initial
            * (t - centre)
            / sigma**2
            * math.exp(-(((t - centre) / sigma) ** 2))
        )
        return period**2 * abs(second * current + slope * derivative)

    variation, variation_error = quad(
        edge_variation,
        -0.5,
        0.5,
        points=[
            left / period,
            centre / period,
            right / period,
            collar.one_high / period,
        ],
        epsabs=1e-12,
    )
    # A C1 product h=g'J has |h_k| <= ||h'||_1/(T |omega_k|).
    # Thus the omitted primitive modes k>M have total magnitude at most
    # T||h'||_1/(2 pi^2 M); the difference F(t)-F(t0) costs twice this.
    variation += variation_error
    tail_bound = variation / (2 * math.pi**2 * (count // 2))
    current = current_at(grid.t_fine)
    recovered, held_tail, edge = _recover(grid, current)
    # Independent sampling errors of g'J, quadrature error and the explicit
    # truncated-series bound. No cancellation-based certificate is used here.
    aliases = abs(dt * edge[1:] - coefficients) + coefficient_error
    primitive_bound = 4 * np.sum(aliases / (2 * np.pi * bins))
    dc_bound = abs(dt * edge[0].real - total) + total_error
    allowance = 2 * tail_bound + primitive_bound + dc_bound + c_error
    allowance += 64 * EPS * count * max(abs(current))
    plateau = (grid.t_fine >= collar.one_low) & (grid.t_fine <= collar.one_high)
    assert 0 < allowance < abs(c_minus) / 10
    assert np.max(abs(recovered[plateau] - (current[plateau] - c_minus))) <= allowance
    assert abs(held_tail + total) <= dc_bound + 8 * EPS * abs(total)
    assert abs(recovered[-1] - held_tail) <= allowance
    record_property("c_minus", c_minus)
    record_property("held_tail", held_tail)
    record_property("rho_minus", abs(c_minus) / max(abs(current)))

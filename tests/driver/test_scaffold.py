# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The fixed-step interaction-picture scaffold on the N-soliton ladder (P1)."""

import math

import numpy as np
import pytest

from bounds import UNIT_ROUNDOFF, assert_within, propagation_growth
from march_bounds import LINEAR_STEP, MARGIN, evaluation_rounding
from solprop.driver.scaffold import propagate_fixed_rk4
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse

BETA2 = -11.83e-27  # s^2/m
GAMMA = 0.11  # 1/(W m)
TAU0 = 100e-15  # s, natural sech width
WAVELENGTH = 835e-9
DISPERSION_LENGTH = TAU0**2 / abs(BETA2)
SOLITON_PERIOD = math.pi / 2 * DISPERSION_LENGTH
# The N-soliton returns to its shape after one period with the global phase +pi/4 in
# the optics convention; the internal FFT convention is its conjugate.
RECURRENCE_PHASE = -math.pi / 4


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12)


@pytest.fixture(scope="module")
def problem(grid):
    return problem_from(
        [0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False
    )


def soliton(grid, order):
    peak = order**2 * abs(BETA2) / (GAMMA * TAU0**2)
    return SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)[
        0, 0
    ]


def recurrence_error(final, initial, phase):
    return float(
        np.linalg.norm(final - np.exp(1j * phase) * initial) / np.linalg.norm(initial)
    )


def realised_phase(final, initial):
    peak_bin = int(np.argmax(np.abs(initial)))
    return float(np.angle(final[peak_bin] / initial[peak_bin]))


def test_the_fundamental_soliton_is_stationary_with_its_phase(grid, problem):
    initial = soliton(grid, 1)
    z, spectra = propagate_fixed_rk4(problem, grid, initial, DISPERSION_LENGTH, 200)
    # Measured 8e-11 with these parameters; the bound leaves an order of magnitude.
    assert recurrence_error(spectra[-1], initial, -0.5) < 1e-9
    assert realised_phase(spectra[-1], initial) == pytest.approx(-0.5, abs=1e-9)
    assert z.tolist() == [0.0, DISPERSION_LENGTH]
    assert spectra.shape == (2, grid.n_omega)


def test_saving_planes_does_not_change_the_steps(grid, problem):
    initial = soliton(grid, 1)
    z, spectra = propagate_fixed_rk4(
        problem, grid, initial, DISPERSION_LENGTH, 40, n_save=4
    )
    _, direct = propagate_fixed_rk4(problem, grid, initial, DISPERSION_LENGTH, 40)
    np.testing.assert_allclose(z, np.linspace(0.0, DISPERSION_LENGTH, 5))
    # The same forty RK4 steps either way, run twice: a matched march, within its
    # rounding budget (``tests/march_bounds.py``: the linear half, and four
    # evaluations a step carrying the fundamental soliton's ``Phi = 1`` over the
    # dispersion length).
    steps = 40
    bound = (
        MARGIN
        * propagation_growth(1.0, 1.0)
        * steps
        * (LINEAR_STEP * UNIT_ROUNDOFF + 4 * evaluation_rounding(grid.n_t) / steps)
    )
    assert_within(spectra[-1], direct[-1], bound)
    # The entrance is the launch masked, no arithmetic: exact.
    assert np.array_equal(spectra[0], initial * grid.active_mask)


def test_the_state_is_masked_to_the_active_band_once(grid, problem):
    initial = soliton(grid, 1) + 1.0
    _, spectra = propagate_fixed_rk4(problem, grid, initial, DISPERSION_LENGTH, 2)
    assert np.all(spectra[:, ~grid.active_mask] == 0.0)


@pytest.mark.slow
def test_the_third_order_soliton_recurs_at_fourth_order(grid, problem):
    initial = soliton(grid, 3)
    coarse = propagate_fixed_rk4(problem, grid, initial, SOLITON_PERIOD, 2000)[1][-1]
    fine = propagate_fixed_rk4(problem, grid, initial, SOLITON_PERIOD, 4000)[1][-1]
    error_coarse = recurrence_error(coarse, initial, RECURRENCE_PHASE)
    error_fine = recurrence_error(fine, initial, RECURRENCE_PHASE)
    # Measured 1.3e-6 and 9.7e-8, an observed order of 3.8; the scheme is fourth order
    # once the step resolves the nonlinear phase. Assert order 3 with margin.
    assert error_fine < 1e-6
    assert math.log2(error_coarse / error_fine) > 3.0
    assert realised_phase(fine, initial) == pytest.approx(RECURRENCE_PHASE, abs=1e-6)


@pytest.mark.medium
def test_the_third_order_soliton_recurs_over_one_period(grid, problem):
    """One period at a few hundred steps: the scaffold recurs, phase and all.

    Milestone M1's problem on the scaffold at N = 6 and 16000 steps takes 12.6 s and
    was measured at 1.2e-5 (5.2e-7 at 32000); no lane measures it now. What a lane needs
    from this file every time is that the fixed-step integrator still reproduces a
    recurrence it was not told about, which N = 3 does in a second. The ladder that
    reads the scaffold's *order* is the slow test above, at 2000 and 4000 steps.
    """
    initial = soliton(grid, 3)
    final = propagate_fixed_rk4(problem, grid, initial, SOLITON_PERIOD, 400)[1][-1]
    # Measured 4.930e-4 at 400 steps, and the phase to 1.26e-4. The slow ladder above
    # measures 1.3e-6 at 2000 steps, and 400 steps is 5^4 = 625 coarser, so 4.9e-4 is
    # the fourth-order scheme's own truncation error and the bound leaves a factor of
    # four on it. tests/backends/test_numpy_ref_soliton.py asserts the same number from
    # the other side, as the gap between this integrator and the reference backend.
    assert recurrence_error(final, initial, RECURRENCE_PHASE) < 2e-3
    assert realised_phase(final, initial) == pytest.approx(RECURRENCE_PHASE, abs=2e-3)


def test_the_scaffold_refuses_a_field_grid(problem):
    field = FieldGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12)
    with pytest.raises(ValueError, match="envelope grids only"):
        propagate_fixed_rk4(problem, field, np.zeros(field.n_omega, complex), 1.0, 10)


def test_the_scaffold_rejects_a_spectrum_of_the_wrong_shape(grid, problem):
    with pytest.raises(ValueError, match="expected"):
        propagate_fixed_rk4(problem, grid, np.zeros(3, complex), 1.0, 10)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"z_end": 0.0}, "z_end must be positive"),
        ({"z_end": math.inf}, "z_end must be positive"),
        ({"n_steps": 0}, "n_steps must be at least 1"),
        ({"n_save": 0}, "n_save must be at least 1"),
        ({"n_steps": 10, "n_save": 3}, "must be a multiple of n_save"),
    ],
)
def test_the_scaffold_rejects_a_bad_request(grid, problem, overrides, message):
    request = {"z_end": 1.0, "n_steps": 10, "n_save": 1} | overrides
    with pytest.raises(ValueError, match=message):
        propagate_fixed_rk4(problem, grid, soliton(grid, 1), **request)

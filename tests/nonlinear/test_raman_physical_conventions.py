# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent angular and spectral checks of the molecular Raman conventions.

The line-strength formulas are checked on solprop's former molecular constants
(``former_raman``, from its registry at dc4ca0d), supplied as a caller's data: the
conventions are the formulas', and any physical set of constants exercises them.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import gammaln, lpmv, roots_legendre

from former_raman import (
    FORMER_CH4,
    FORMER_D2,
    FORMER_H2,
    FORMER_N2,
    FORMER_O2,
    FORMER_SF6,
)
from solprop.constants import AMAGAT, EPS0, HBAR
from solprop.nonlinear.raman import Dephasing, rotational_nonrigid, vibrational
from solprop.physdata import Linewidth, MolecularRaman


def gas_id(parameters: MolecularRaman) -> str:
    """Name a parametrised set of molecular constants after its gas."""
    return str(getattr(parameters.gas, "value", parameters.gas))


@pytest.mark.parametrize(
    "gas",
    [FORMER_N2, FORMER_O2, FORMER_H2, FORMER_D2, FORMER_CH4, FORMER_SF6],
    ids=gas_id,
)
def test_vibrational_strength_from_quantum_oscillator_impulse(gas):
    # Independent impulse evolution under H_int = -alpha'(r) r E_real^2 / 2.
    # In oscillator units q=(a+a†)/sqrt(2), the impulse is exp(i epsilon q/2).
    # It is evaluated as a matrix exponential, not from the classical line
    # prefactor. Six levels suffice: the displacement is only epsilon=1e-3.
    parameters = gas.vibrational
    assert parameters is not None
    annihilation = np.diag(np.sqrt(np.arange(1, 6)), 1)
    position = (annihilation + annihilation.T) / math.sqrt(2)
    impulse = 1e-3
    ground = np.zeros(6, dtype=complex)
    ground[0] = 1
    state = expm(0.5j * impulse * position) @ ground
    phase_time = 1.3
    state *= np.exp(-1j * (np.arange(6) + 0.5) * phase_time)
    displacement = np.vdot(state, position @ state).real
    alpha_si = 4 * math.pi * EPS0 * parameters.polarisability_derivative
    # Restore q_scale^2/hbar = 1/(mu Omega) and divide induced polarization
    # by epsilon_0 rho E and the impulse's integrated real-field-square drive.
    expected = (
        alpha_si**2
        * displacement
        / (EPS0 * parameters.reduced_mass * parameters.frequency * impulse)
    )
    actual = vibrational(parameters).amplitude * math.sin(phase_time)
    # Matrix exponential and phase evolution at tiny displacement: allow 1e-12
    # relative roundoff, far below the former factor-two amplitude defect.
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("lower_j", [0, 1, 2, 5, 8, 15])
@pytest.mark.parametrize("gas", [FORMER_N2, FORMER_O2], ids=gas_id)
def test_rotational_strength_from_magnetic_sublevel_quadrature(gas, lower_j):
    # First-order response to V = -Delta_alpha_SI E^2 cos(theta)^2 / 2.
    # Integrate normalised spherical harmonics, rather than restating the
    # closed-form line-strength coefficient in rotational_nonrigid.
    parameters = gas.rotational
    assert parameters is not None
    lines = rotational_nonrigid(parameters, temperature=294.0)
    x, weights = roots_legendre(64)
    upper_j = lower_j + 2
    angular_strength = 0.0
    for magnetic_m in range(lower_j + 1):
        harmonics = []
        for quantum_j in (lower_j, upper_j):
            normalisation = math.sqrt(
                (2 * quantum_j + 1)
                / (4 * math.pi)
                * math.exp(
                    gammaln(quantum_j - magnetic_m + 1)
                    - gammaln(quantum_j + magnetic_m + 1)
                )
            )
            harmonics.append(normalisation * lpmv(magnetic_m, quantum_j, x))
        matrix_element = (
            2 * math.pi * np.dot(weights, x**2 * harmonics[0] * harmonics[1])
        )
        angular_strength += (1 if magnetic_m == 0 else 2) * matrix_element**2
    population_difference = lines.populations[lower_j] / (2 * lower_j + 1)
    population_difference -= lines.populations[upper_j] / (2 * upper_j + 1)
    alpha_si = 4 * math.pi * EPS0 * parameters.polarisability_anisotropy
    expected = alpha_si**2 / (EPS0 * HBAR) * population_difference * angular_strength
    # The integrands are polynomials below the quadrature's exact degree.
    # Allow 1e-11 for associated-Legendre evaluation and sublevel summation.
    assert lines.amplitudes[lower_j] == pytest.approx(expected, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("relative_detuning", [0.0, 0.5, 1.0, 2.0])
def test_linewidth_is_the_fwhm_of_the_fourier_transformed_coherence(relative_detuning):
    width = 3.0e9
    decay = Dephasing(Linewidth(constant=width))
    # Dimensionless integration time u = pi * width * t. A Fourier transform
    # with cyclic detuning delta_nu uses cos(2 pi delta_nu t), not cos(delta_nu t).
    spectral_line, _error = quad(
        lambda u: (
            decay.damping(np.array([u / (math.pi * width)]), AMAGAT)[0]
            * math.cos(2 * relative_detuning * u)
        ),
        0.0,
        40.0,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    expected = 1 / (1 + (2 * relative_detuning) ** 2)
    # Tail < exp(-40); quadrature budget 1e-12. At +/-width/2 the
    # spectral intensity is one half, so the full width is exactly width.
    assert spectral_line == pytest.approx(expected, rel=0.0, abs=3e-12)

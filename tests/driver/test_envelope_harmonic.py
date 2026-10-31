# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The third harmonic an envelope grid generates agrees with the field grid's.

Ten microns of a hydrogen-filled capillary is far inside the coherence length of the
true mismatch ``beta(3 omega0) - 3 beta(omega0)``, about 4e3 rad/m, so the harmonic
grows as ``L**2`` and its energy has a closed-form weak-conversion estimate. An envelope
frame that measured its delay from the carrier would add a mismatch of
``2 beta_1 omega0``, about 1.2e7 rad/m, and generate three orders of magnitude less.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, jn_zeros

from solprop.constants import EPS0, C
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.physdata.equation_of_state import density
from solprop.physdata.nonlinear_data import gamma3
from solprop.processing.fields import get_spectrum
from solprop.stepper import RungeKutta

# The window reaches beyond the hydrogen index fit; the extrapolated bins are not the
# ones the harmonic is measured in.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
)

RADIUS = 20e-6
LENGTH = 10e-6
PRESSURE = 5.0
WAVELENGTH = 1030e-9
FWHM = 40e-15
ENERGY = 4e-6
OMEGA0 = 2.0 * np.pi * C / WAVELENGTH


def harmonic_fraction(kind: str) -> float:
    """Return the energy fraction within 2.6-3.4 omega0 after the short capillary."""
    limits, window = (250e-9, 4e-6), 1.2e-12
    if kind == "field":
        grid = FieldGrid.from_limits(LENGTH, WAVELENGTH, limits, window)
    else:
        grid = EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, limits, window, thg=True)
    simulation = Simulation.from_capillary(
        RADIUS,
        LENGTH,
        "H2",
        PRESSURE,
        grid=grid,
        wavelength=WAVELENGTH,
        fwhm=FWHM,
        energy=ENERGY,
        loss=False,
        plasma=False,
        raman=False,
        noise=None,
        integrator=RungeKutta(),
        rtol=1e-8,
        atol=1e-10,
        n_save=2,
        backend="numba_cpu",
        thg=True,
    )
    output = simulation.run(progress="none")
    try:
        omega, spectrum = get_spectrum(output, plane=-1, band="all")
        density_ = (np.abs(np.asarray(spectrum)) ** 2).reshape(-1, omega.size).sum(0)
    finally:
        output.close()
    weights = density_ * np.gradient(omega)
    band = (omega > 2.6 * OMEGA0) & (omega < 3.4 * OMEGA0)
    return float(np.sum(weights[band]) / np.sum(weights))


def weak_conversion_fraction() -> float:
    """Return the undepleted, phase-matched estimate of the harmonic energy fraction.

    With ``E = Re[F(r) a(t) exp(-i omega0 t)]``, ``F = J0(u01 r / radius)`` the
    frequency-independent HE11 profile and ``P = eps0 chi3 E**3``, the harmonic
    amplitude grows as ``(3 omega0 / 2c) (chi3 / 4) a**3 L``, projected on ``F``. For a
    Gaussian pulse of peak ``a0`` the energy ratio is then
    ``(3 omega0 chi3 L / 8c)**2 (int F**4 / int F**2)**2 a0**4 / sqrt(3)``.
    """
    chi3 = gamma3("H2", wavelength=WAVELENGTH) * density("H2", PRESSURE)
    u01 = jn_zeros(0, 1)[0]
    f2 = quad(lambda r: j0(u01 * r / RADIUS) ** 2 * r, 0.0, RADIUS)[0]
    f4 = quad(lambda r: j0(u01 * r / RADIUS) ** 4 * r, 0.0, RADIUS)[0]
    duration = FWHM * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    peak_sq = ENERGY / (0.5 * EPS0 * C * 2.0 * np.pi * f2 * duration)
    coupling = 3.0 * OMEGA0 * chi3 * LENGTH / (8.0 * C)
    return float(coupling**2 * (f4 / f2) ** 2 * peak_sq**2 / np.sqrt(3.0))


def test_the_envelope_harmonic_matches_the_field_grid_and_the_estimate():
    field = harmonic_fraction("field")
    envelope = harmonic_fraction("envelope")
    estimate = weak_conversion_fraction()
    # Measured: the two grids agree to 1e-4 and the estimate to 0.3 %, the difference
    # being the dispersion and self-phase modulation the estimate leaves out. The
    # defect this guards against was a factor of 4e3 low.
    assert envelope == pytest.approx(field, rel=1e-2)
    assert field == pytest.approx(estimate, rel=2e-2)

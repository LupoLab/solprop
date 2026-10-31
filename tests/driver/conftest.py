# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures for the driver: the GNLSE soliton the M1 tests are measured on.

The constants and the grids are those of ``tests/backends/conftest.py``, restated here
because a sibling conftest is not importable; the M1 tests reuse the numbers measured
against the backend directly in ``tests/backends/test_numpy_ref_soliton.py``, so the two
files describe the same problem and must stay in step.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from selected_backend import selected_backend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid

BETA2 = -11.83e-27  # s^2/m, anomalous
GAMMA = 0.11  # 1/(W m)
TAU0 = 100e-15  # s, the natural sech width
WAVELENGTH = 835e-9  # m
DISPERSION_LENGTH = TAU0**2 / abs(BETA2)
SOLITON_PERIOD = math.pi / 2 * DISPERSION_LENGTH
# The N-soliton returns to its shape after one period with the global phase +pi/4 in the
# optics convention, which is the one an output file stores; the internal FFT convention
# the backend works in carries its conjugate.
RECURRENCE_PHASE = math.pi / 4
# Short enough that the loop tests cost a few hundred steps, long enough that the pulse
# actually evolves: a tenth of the dispersion length.
SHORT_LENGTH = DISPERSION_LENGTH / 10


@pytest.fixture(scope="session")
def test_backend_name():
    """Backend selected for M1 by ``SOLPROP_TEST_BACKEND`` (``"auto"`` by default)."""
    name = selected_backend()
    if name not in available():
        pytest.fail(f"SOLPROP_TEST_BACKEND={name!r} is not available")
    return name


@dataclass(frozen=True)
class Soliton:
    """The fibre, the scales and the two metrics the soliton tests read off them."""

    beta2: float
    gamma: float
    tau0: float
    wavelength: float
    dispersion_length: float
    period: float
    recurrence_phase: float
    short_length: float

    def peak_power(self, order):
        """Return the peak power of the N-soliton of the given order, in W."""
        return order**2 * abs(self.beta2) / (self.gamma * self.tau0**2)

    def error(self, final, start, phase=None):
        """Return the phase-resolved error as a fraction of the input's norm."""
        turn = self.recurrence_phase if phase is None else phase
        return float(
            np.linalg.norm(final - np.exp(1j * turn) * start) / np.linalg.norm(start)
        )

    def phase(self, final, start):
        """Return the global phase actually realised, read at the brightest bin."""
        peak = int(np.argmax(np.abs(start)))
        return float(np.angle(final[peak] / start[peak]))


@pytest.fixture(scope="session")
def soliton():
    """The soliton case of ``tests/backends/conftest.py``, with its two metrics."""
    return Soliton(
        beta2=BETA2,
        gamma=GAMMA,
        tau0=TAU0,
        wavelength=WAVELENGTH,
        dispersion_length=DISPERSION_LENGTH,
        period=SOLITON_PERIOD,
        recurrence_phase=RECURRENCE_PHASE,
        short_length=SHORT_LENGTH,
    )


@pytest.fixture(scope="session")
def narrow_grid():
    """The 1024-bin, 1 ps envelope grid of the cheap backend tests."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)


@pytest.fixture(scope="session")
def soliton_grid():
    """The 4096-bin, 4 ps envelope grid the N = 3 recurrence is measured on."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12)


@pytest.fixture(scope="session")
def wide_grid():
    """A band that holds the compressed N = 6 pulse: 350 nm to 3 um, 8192 bins."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (350e-9, 3e-6), 4e-12)


@pytest.fixture
def build_simulation(soliton):
    """Return ``(**keywords) -> Simulation`` for the soliton fibre through from_gnlse.

    The defaults are a short, cheap run of the N = 3 soliton with four saved planes;
    every keyword of :meth:`solprop.driver.Simulation.from_gnlse` overrides them.
    """

    def build(*, order=3, length=SHORT_LENGTH, gamma=GAMMA, **keywords):
        settings = {
            "wavelength": WAVELENGTH,
            "wavelength_limits": (500e-9, 2e-6),
            "trange": 1e-12,
            "width": TAU0,
            "peak_power": soliton.peak_power(order),
            "shape": "sech",
            "shock": False,
            "rtol": 1e-8,
            "atol": 1e-18,
            "n_save": 4,
        } | keywords
        return Simulation.from_gnlse(
            gamma, length, [0.0, 0.0, BETA2], **{"noise": None, **settings}
        )

    return build

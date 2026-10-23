# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Guard per-integral reuse of a step-index Maxwell solution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import _stepindex_vector as vector
from solprop.modes.stepindex import StepIndexMode
from solprop.profiles import Linear

OMEGA = 2 * np.pi * C / 835e-9


def _fresh_power(mode, omega, z, start):
    """Integrate with an independent mode solve at every radial quadrature point."""
    radius = mode._radius(z)
    angles = np.arange(2 * mode.n + 3) * 2 * np.pi / (2 * mode.n + 3)

    def integrand(x):
        points = radius * x * np.column_stack((np.cos(angles), np.sin(angles)))
        electric, magnetic = vector._fields(mode, points, omega, z)
        return float(
            np.mean(
                np.real(
                    electric[:, 0] * magnetic[:, 1].conj()
                    - electric[:, 1] * magnetic[:, 0].conj()
                )
            )
            * np.pi
            * x
        )

    # Preserve the historical full-domain and separately certified tail rules; the
    # tolerances belong to the unchanged quadrature.
    if start < 1:
        core, core_error = quad(
            integrand, start, 1.0, epsabs=0.0, epsrel=2e-11, limit=150
        )
        tail, tail_error = quad(
            integrand,
            1.0,
            np.inf,
            epsabs=1e-15 * abs(core),
            epsrel=2e-11,
            limit=150,
        )
        return radius**2 * (core + tail), radius**2 * (core_error + tail_error)
    value, error = quad(integrand, start, np.inf, epsabs=0.0, epsrel=2e-11, limit=150)
    return radius**2 * value, radius**2 * error


@pytest.mark.parametrize(
    ("kind", "n", "m", "parity", "start"),
    [
        ("HE", 1, 1, "even", 0.0),
        ("HE", 1, 2, "odd", 0.0),
        ("HE", 1, 2, "even", 3.0),
        ("EH", 1, 1, "even", 0.5),
        ("TE", 0, 1, "even", 0.0),
        ("TM", 0, 1, "even", 3.0),
    ],
)
def test_power_reuses_one_solve_and_matches_fresh_quadrature(
    monkeypatch, kind, n, m, parity, start
):
    mode = StepIndexMode(
        Linear(4e-6, 3.6e-6, 1e-3),
        core=1.45,
        cladding=1.40,
        profile_wavelength=835e-9,
        kind=kind,
        n=n,
        m=m,
        parity=parity,
    )
    original = StepIndexMode._eigenpairs
    calls = []

    def counted(self, omega, z):
        calls.append((float(omega[0]), z))
        return original(self, omega, z)

    monkeypatch.setattr(StepIndexMode, "_eigenpairs", counted)
    actual = vector._power(mode, OMEGA, 0.5e-3, start)
    assert calls == [(OMEGA, 0.5e-3)]
    expected = _fresh_power(mode, OMEGA, 0.5e-3, start)
    assert len(calls) > 2
    # Both are adaptive quadratures certified to epsrel 2e-11 of the same integral;
    # the fresh solves and the reused one may differ by rounding, which can move
    # quad's subdivision, so the values are held to the sum of the two certified
    # errors (4e-11) and the error estimates, a work diagnostic, to a factor of four.
    assert_allclose(actual[0], expected[0], rtol=4e-11, atol=0)
    assert expected[1] / 4 <= actual[1] <= 4 * expected[1]


def test_each_new_frequency_or_position_resolves_its_own_material_and_mode(monkeypatch):
    mode = StepIndexMode(
        Linear(4e-6, 3.6e-6, 1e-3),
        core=1.45,
        cladding=1.40,
        profile_wavelength=835e-9,
    )
    original = StepIndexMode._eigenpairs
    calls = []

    def counted(self, omega, z):
        calls.append((float(omega[0]), z))
        return original(self, omega, z)

    monkeypatch.setattr(StepIndexMode, "_eigenpairs", counted)
    points = ((OMEGA, 0.0), (0.9 * OMEGA, 0.0), (OMEGA, 0.5e-3))
    powers = [vector._power(mode, omega, z) for omega, z in points]
    assert calls == list(points)
    assert len({power for power, _ in powers}) == len(points)
    # The existing exact-key power cache remains valid for an identical query.
    assert vector._power(mode, *points[0]) == powers[0]
    assert calls == list(points)


def test_attenuating_mode_is_refused_before_quadrature(monkeypatch):
    mode = StepIndexMode(
        4e-6, core=1.45, cladding=1.40, profile_wavelength=835e-9, loss=1.0
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified loss must be rejected before integration")

    monkeypatch.setattr(vector, "quad", forbidden)
    with pytest.raises(ModeError, match="attenuating"):
        vector._power(mode, OMEGA, 0.0)

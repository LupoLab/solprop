# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian cutoff certificates from an independent decimal oracle."""

import numpy as np
import pytest

from cartesian_oracles import longitudinal_square
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.linear import Frame
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.translate.free import free_operators
from solprop.translate.spatial import separable_values
from spatial_records import record_metric


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("index", [1.0, 1 + 1e-12j, 1 - 1e-12j])
def test_cartesian_square_and_prefactor_meet_the_predeclared_certificate(
    kind, index, request
):
    constructor = FieldGrid.from_limits if kind == "field" else EnvelopeGrid.from_limits
    grid = constructor(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-7, 9, 4e-7, 12, window_factor=0)
    dispersion = SpatialDispersion(SpatialMedium(index), grid, Frame.comoving(grid))
    high, low, transverse_low = dispersion.cartesian_squares(space)
    actual = (high[None] - space.kperp_sq[:, None]) + (
        low[None] - transverse_low[:, None]
    )
    # This expression and its constants were derived in the cutoff contract before
    # the first coefficient measurement. Decimal's 100-digit oracle adds 10^-90 S.
    epsilon = np.finfo(float).eps
    scale = np.abs(high)[None] + space.kperp_sq[:, None]
    bound = (
        8 * epsilon * np.abs(actual)
        + (64 * epsilon**2 + 1e-90) * scale
        + 32 * np.nextafter(0.0, 1.0)
    )
    expected = longitudinal_square(space, grid, index)
    active = grid.active_mask
    defect = np.abs(actual[:, active] - expected[:, active])
    assert np.all(defect <= bound[:, active])
    # The real-part certificate governs the sign even with material loss/gain;
    # its scale must not be inflated by the imaginary longitudinal component.
    real_bound = (
        8 * epsilon * np.abs(np.real(actual))
        + (64 * epsilon**2 + 1e-90) * scale
        + 32 * np.nextafter(0.0, 1.0)
    )
    assert np.all(
        np.abs(np.real(actual[:, active] - expected[:, active]))
        <= real_bound[:, active]
    )
    # The certificate's purpose: the sign of every active bin is decided (structural).
    assert np.array_equal(
        np.real(actual[:, active]) > 0, np.real(expected[:, active]) > 0
    )
    # Compare the most cancellation-sensitive active bin, with its absolute bound.
    condition = np.where(active, np.abs(np.real(actual)), np.inf)
    column, frequency = np.unravel_index(np.argmin(condition), condition.shape)
    _, data = free_operators(space, dispersion, n_pol=2, z_max=grid.length)
    operator, prefactor = separable_values(data)
    # The translated record and the dispersion object form the same quantities from the
    # same squares: a root, the frame's subtraction and a quotient, a few roundings of
    # the longitudinal wavenumber (the operator, with its cancellation against the
    # frame) or of the value itself (the prefactor). Fixed at 16 eps of each.
    wavenumber = float(np.sqrt(np.max(np.abs(high))))
    np.testing.assert_allclose(
        operator[0, :, 0],
        dispersion.operator(space),
        rtol=0,
        atol=16 * epsilon * wavenumber,
    )
    np.testing.assert_allclose(
        prefactor[:, 0],
        dispersion.prefactor(space) * grid.crop_scale,
        rtol=16 * epsilon,
        atol=0,
    )
    if index == 1.0:
        positive = (np.real(actual) > bound) & active
        # Inverse-root sensitivity plus division/root rounding, derived before
        # measuring the coefficient; the original Kerr absolute gate is separate.
        relative_bound = (
            bound[positive] / (np.real(actual)[positive] - bound[positive])
            + 16 * epsilon
        )
        exact_prefactor = np.broadcast_to(data.prefactor_numerator, actual.shape)[
            positive
        ] / np.sqrt(np.real(expected)[positive])
        relative_error = np.abs(prefactor[:, 0][positive] / exact_prefactor - 1)
        assert np.all(relative_error < relative_bound)

    record_metric(
        request,
        "Cartesian corrected longitudinal square (m^-2)",
        f"{kind}, index={index}",
        abs(actual[column, frequency] - expected[column, frequency]),
        bound[column, frequency],
    )


def test_cartesian_launch_uses_the_corrected_exact_linear_operator():
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    # Every column propagates (k_perp below pi/dx = 4.7e5 m^-1, far under k): a
    # launch is normalised in axial flux, and refuses a beam whose evanescent
    # columns carry its field (SEP 34). The small-k_perp operator is the one whose
    # cancellation the corrected form avoids.
    space = FreeGrid(3e-5, 9, 4e-5, 12, window_factor=0)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-11)
    distance = 1e-4
    medium = SpatialMedium()
    start = SpatialPulse(temporal, 8e-6).build(grid, space, medium=medium)
    launch = SpatialPulse(temporal, 8e-6, propz=distance).build(
        grid, space, medium=medium
    )
    dispersion = SpatialDispersion(medium, grid, Frame.comoving(grid))
    expected = start * np.exp(distance * dispersion.operator(space))[:, None]
    # The launch and the direct expression perform the same scalar operations: a
    # complex exponential (a few ulps of its unit modulus, plus the rounding of its
    # argument, distance times the operator) and one product. Each is then
    # normalised at its own plane: the lossless flow keeps the energy, so the two
    # factors differ by the rounding of their two sums of n positive terms.
    operator = dispersion.operator(space)
    eps = np.finfo(float).eps
    phase_error = 8 * eps * distance * float(np.max(np.abs(operator)))
    terms = space.n_columns * grid.n_omega
    normalisation = terms * eps / (1 - terms * eps)
    np.testing.assert_allclose(
        launch,
        expected,
        rtol=0,
        atol=(8 * eps + phase_error + normalisation) * float(np.max(np.abs(start))),
    )

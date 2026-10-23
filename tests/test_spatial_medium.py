# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bulk-index, frame, loss/gain and evanescent-branch oracles."""

import dataclasses
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import gamma
from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame
from solprop.physdata.glasses import fused_silica
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import gamma as susceptibility_for
from solprop.profiles import Linear
from solprop.spatial import EVANESCENT_RATE, SpatialDispersion, SpatialMedium


@pytest.fixture(params=["field", "envelope"])
def grid(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    return constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)


@pytest.mark.parametrize("index", [1.5 + 2e-6j, 1.5 - 2e-6j, 1 + 0j])
def test_bulk_loss_gain_and_frame_at_zero_transverse_frequency(grid, index):
    medium = SpatialMedium(index)
    frame = Frame.comoving(grid)
    model = SpatialDispersion(medium, grid, frame)
    expected = -index.imag * grid.omega / C * grid.active_mask
    # k_z=k for k_perp=0; the comoving nondispersive real phase cancels.
    absolute = gamma(64) * float(grid.omega.max()) / C
    assert_allclose(model.operator(np.zeros(1))[0], expected, rtol=0, atol=absolute)
    prefactor = -0.5j * grid.omega / (EPS0 * C * index.real) * grid.active_mask
    assert_allclose(model.prefactor(np.zeros(1))[0], prefactor, rtol=gamma(32), atol=0)
    fixed = dataclasses.replace(frame, beta0=30.0, beta1=2e-9)
    assert SpatialDispersion(medium, grid, fixed).frame_quantities() == (
        30.0 if frame.subtract_beta0 else 0.0,
        2e-9,
    )


def test_evanescent_prefactor_and_attenuation_cap_have_the_literal_limits(grid):
    model = SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid))
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    wave = grid.omega[index] / C
    # Use deficits much larger than one ULP of k^2; the sub-cap and capped
    # cases stay well resolved, and the cutoff itself is the same squared wave.
    deficits = np.array([0.0, 100.0**2, 1000.0**2])
    transverse = wave**2 + deficits
    expected_loss = -np.minimum(np.sqrt(deficits), EVANESCENT_RATE)
    phase = model.operator(transverse)[:, index]
    # Square-root conditioning: delta sqrt(s) <= sqrt(delta s) at the cutoff.
    absolute = np.sqrt(gamma(32) * wave**2)
    assert_allclose(np.real(phase), expected_loss, rtol=0, atol=absolute)
    assert_allclose(
        model.prefactor(transverse)[:, index],
        -0.5j * grid.omega[index],
        rtol=gamma(8),
        atol=0,
    )
    assert np.all(model.operator(transverse)[:, ~grid.active_mask] == 0)
    assert np.all(model.prefactor(transverse)[:, ~grid.active_mask] == 0)


@pytest.mark.parametrize("material", ["Ar", "silica"])
def test_bulk_material_derivative_uses_the_existing_analytic_models(material):
    model = fused_silica() if material == "silica" else material
    medium = SpatialMedium(model)
    omega = 2 * np.pi * C / np.array([700e-9, 800e-9, 950e-9])
    if material == "silica":
        expected = model.index(omega)
        derivative = model.dindex_domega(omega)
    else:
        density = float(medium.density(np.array([0.0]))[0])
        susceptibility = medium.susceptibility
        assert susceptibility is not None
        expected = np.sqrt(1 + density * susceptibility.susceptibility(omega))
        derivative = (
            density * susceptibility.dsusceptibility_domega(omega) / (2 * expected)
        )
    # This is interface wiring to separately qualified material expressions.
    assert_allclose(medium.index(omega), expected, rtol=gamma(8), atol=0)
    assert_allclose(medium.dindex_domega(omega), derivative, rtol=gamma(8), atol=0)
    frozen = medium.at_density(0.0 if material == "Ar" else 1.0)
    assert frozen.density.is_constant
    if material == "Ar":
        np.testing.assert_array_equal(frozen.index(omega), np.ones(3))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"material": -1}, "index"),
        ({"material": 1 + np.inf * 1j}, "index"),
        ({"temperature": 0}, "temperature"),
        ({"temperature": np.nan}, "temperature"),
        ({"pressure": 2}, "pressure"),
        ({"material": "Ar", "pressure": -1}, "pressure"),
        ({"material": "Ar", "pressure": Linear(1, -1, 1)}, "pressure"),
    ],
)
def test_spatial_medium_refuses_invalid_configuration(options, match):
    with pytest.raises(MaterialError, match=match):
        SpatialMedium(**options)


def test_spatial_medium_refuses_a_temperature_other_than_its_mixtures():
    mixture = Mixture.fill({"Ar": 1.0}, 2.0, temperature=300.0)
    # Not given, or given as the mixture's own: the mixture's temperature.
    assert SpatialMedium(mixture).temperature == 300.0
    assert SpatialMedium(mixture, temperature=300.0).temperature == 300.0
    assert SpatialMedium("Ar").temperature == ROOM_TEMPERATURE
    assert SpatialMedium("Ar", temperature=250.0).temperature == 250.0
    with pytest.raises(MaterialError, match=r"own temperature \(300 K\).*omit"):
        SpatialMedium(mixture, temperature=ROOM_TEMPERATURE)


@pytest.mark.parametrize(
    "omega,z",
    [(np.array([0.0]), 0.0), (np.array([np.nan]), 0.0), (np.array([1.0]), np.inf)],
)
def test_spatial_material_evaluation_rejects_invalid_points(omega, z):
    medium = SpatialMedium()
    for evaluate in (medium.index, medium.dindex_domega):
        with pytest.raises(MaterialError, match="frequencies"):
            evaluate(omega, z)


@pytest.mark.parametrize("density", [-1.0, np.nan, np.inf, 0.0])
def test_nongas_density_cannot_change_its_normalisation(density):
    with pytest.raises(MaterialError, match="density"):
        SpatialMedium().at_density(density)


@pytest.mark.parametrize("transverse", [np.array([-1.0]), np.array([np.inf])])
def test_spatial_operator_rejects_invalid_transverse_squares(grid, transverse):
    model = SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid))
    for evaluate in (model.operator, model.prefactor):
        with pytest.raises(MaterialError, match="transverse"):
            evaluate(transverse)


@pytest.mark.parametrize("cap", [0.0, np.nan, np.inf])
def test_spatial_dispersion_rejects_invalid_clamp(grid, cap):
    with pytest.raises(MaterialError, match="evanescent_rate"):
        SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid), cap)


def test_spatial_dispersion_refuses_frame_outside_grid(grid):
    with pytest.raises(MaterialError, match="reference"):
        SpatialDispersion(
            SpatialMedium(),
            grid,
            dataclasses.replace(Frame.comoving(grid), omega0=10 * grid.omega.max()),
        )


def test_nonfinite_material_index_is_rejected_at_the_spatial_boundary(monkeypatch):
    glass = fused_silica()
    monkeypatch.setattr(
        type(glass), "index", lambda self, omega: np.full(omega.shape, np.nan)
    )
    with pytest.raises(MaterialError, match="nonfinite refractive index"):
        SpatialMedium(glass).index(np.array([2 * np.pi * C / 800e-9]))


def test_explicit_gas_susceptibility_is_preserved():
    susceptibility = susceptibility_for("Ar")
    medium = SpatialMedium("Ar", susceptibility=susceptibility)
    assert medium.susceptibility is susceptibility

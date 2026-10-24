# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the total-degree candidate independently of the production seam."""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.constants import C
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.physdata import (
    Component,
    Mixture,
    Species,
    TabulatedSusceptibility,
    species,
)
from solprop.profiles import Constant, Linear, Scaled
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.translate import radial
from solprop.translate.free import free_operators
from solprop.translate.linop import build_phi, delta_phi
from solprop.translate.radial import radial_operators

EPS = np.finfo(float).eps


@pytest.fixture
def optical_population():
    omega = np.array([1e14, 1e17])
    table = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.zeros(2),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic constant susceptibility",
    )
    return Species("test gas", table)


def test_independent_radial_profiles_use_the_general_table(optical_population):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    components = tuple(
        Component(
            optical_population,
            Linear((i + 1) * 1e24, (i + 2) * 1e24, grid.length),
            str(i),
        )
        for i in range(3)
    )
    medium = SpatialMedium(Mixture.of(*components))
    dispersion = SpatialDispersion(medium, grid, Frame.comoving(grid))
    # Positive roots stay far from cutoff. Bound 256 rounded optical-scale
    # operations over the span, and keep the independent quadrature budget below it.
    wave = float(np.max(grid.omega[grid.active_mask])) / C
    bound = grid.length * 256 * EPS * wave
    operator, prefactor = radial_operators(
        space, dispersion, n_pol=1, z_max=grid.length
    )
    assert operator.phi_representation == "table"
    assert not prefactor.budget_exceeded
    # For this synthetic nondispersive material, both squared roots are affine in
    # z. Integrate sqrt(A+Bz) analytically using its rationalised divided difference.
    # Direct quadrature of the difference of optical-scale roots chases roundoff;
    # the closed form is independent of the table and has the same fixed bound.
    omega = grid.omega[grid.active_mask]
    wave_numbers = omega / C
    index0, index1 = np.sqrt(1 + 2e-30 * np.array([6e24, 9e24]))
    first = np.sqrt((wave_numbers * index0) ** 2 - space.k_r[:, None] ** 2)
    last = np.sqrt((wave_numbers * index1) ** 2 - space.k_r[:, None] ** 2)
    integral = (
        2 * grid.length / 3 * (first**2 + first * last + last**2) / (first + last)
    )
    index_integral = (
        2
        * grid.length
        / 3
        * (index0**2 + index0 * index1 + index1**2)
        / (index0 + index1)
    )
    # The adiabatic amplitude term of SEP 34 section 4.4 is the endpoint log-ratio
    # of the same closed-form roots, every column propagating far from cutoff. Its
    # table is resolved to a relative 1e-9 of itself (see tests/translate/
    # test_radial.py), absolutely far below the phase bound for a ratio of 1e-6.
    amplitude = -0.5 * np.log(np.real(last) / np.real(first))
    reference = np.zeros((space.n_r, grid.n_omega), dtype=complex)
    reference[:, grid.active_mask] = amplitude - 1j * (
        integral - wave_numbers * index_integral
    )
    assert_allclose(
        delta_phi(operator, 0.0, grid.length)[0, :, 0],
        reference,
        atol=bound + 1e-9 * float(np.max(np.abs(amplitude))),
        rtol=0,
    )


def test_cartesian_media_refuse_changing_composition_even_at_constant_total(
    optical_population,
):
    grid = EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    mixture = Mixture.of(
        Component(optical_population, Linear(0.0, 1e24, grid.length), "first"),
        Component(optical_population, Linear(1e24, 0.0, grid.length), "second"),
    )
    assert_allclose(
        mixture.total_density(np.array([0.0, grid.length])),
        [1e24, 1e24],
        rtol=0,
        atol=0,
    )
    dispersion = SpatialDispersion(SpatialMedium(mixture), grid, Frame.comoving(grid))
    with pytest.raises(TranslationError, match="constant media"):
        free_operators(
            FreeGrid(3e-4, 9, 4e-4, 12), dispersion, n_pol=1, z_max=grid.length
        )


def test_split_and_reordered_populations_preserve_the_cumulative_operator():
    length = 0.2
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    gas = species("He")
    path = Linear(1e26, 2e25, length)
    populations = (
        Component(gas, Scaled(path, 0.5), "left"),
        Component(gas, Scaled(path, 0.5), "right"),
        Component(gas, Constant(0.0), "zero"),
    )
    mixtures = (
        Mixture.of(Component(gas, path)),
        Mixture.of(*populations),
        Mixture.of(*reversed(populations)),
    )
    # The frame subtracts three optical-scale quantities. Allow 256 epsilon for
    # model and projection arithmetic, plus both 1e-12 table targets at that
    # scale, integrated over the full span. Every population count shares the
    # rationalised root.
    wave = float(np.max(grid.omega[grid.active_mask])) / C
    bound = length * wave * (256 * EPS + 2e-12)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        operators = [
            build_phi(
                MarcatiliMode(125e-6, mixture=mix, cladding=1.45),
                grid,
                Frame.comoving(grid),
                z_max=length,
            )
            for mix in mixtures
        ]
    for z in (0.0, 0.037, length):
        reference = delta_phi(operators[0], 0.0, z)
        for operator in operators[1:]:
            assert_allclose(delta_phi(operator, 0.0, z), reference, atol=bound, rtol=0)


@pytest.mark.parametrize("failed_build", [1, 2])
def test_independent_radial_tables_refuse_either_unresolved_part(
    optical_population, monkeypatch, failed_build
):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = RadialGrid(6e-4, 16, window_factor=0.0)
    mixture = Mixture.of(
        *(
            Component(optical_population, Linear(1e24, 2e24, grid.length), str(i))
            for i in range(3)
        )
    )
    model = SpatialDispersion(SpatialMedium(mixture), grid, Frame.comoving(grid))
    original = radial.build
    count = 0

    def unresolved(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        return (
            replace(result, budget_exceeded=True) if count == failed_build else result
        )

    monkeypatch.setattr(radial, "build", unresolved)
    with pytest.raises(TranslationError, match="interval budget"):
        radial_operators(space, model, n_pol=1, z_max=grid.length)

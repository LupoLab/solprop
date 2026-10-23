# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Constant Cartesian translation and its bounded-workspace public guards."""

import dataclasses
import warnings

import numpy as np
import pytest

from hankel_oracles import gamma
from solprop.exceptions import GrazingIncidenceWarning, SolpropWarning, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.linear import Frame
from solprop.profiles import Linear
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.translate.free import _warn_grazing, free_operators, free_transform
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.spatial import separable_values


@pytest.fixture
def free_inputs():
    time = EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 9, 4e-4, 12)
    return space, time, SpatialDispersion(SpatialMedium(), time, Frame.comoving(time))


def test_free_transform_budget_and_separable_storage(free_inputs):
    space, time, dispersion = free_inputs
    column_bytes = 20 * 16 * 2 * time.n_to
    transform = free_transform(
        space, time, n_pol=2, z_max=time.length, node_budget=7 * column_bytes
    )
    assert transform.node_chunk == 7
    assert transform.synthesis.size == transform.projection.size == 0
    with pytest.raises(TranslationError, match="node_budget"):
        free_transform(
            space, time, n_pol=2, z_max=time.length, node_budget=column_bytes - 1
        )
    linear, coefficients = free_operators(space, dispersion, n_pol=2, z_max=time.length)
    expected = sum(
        getattr(coefficients, item.name).nbytes
        for item in dataclasses.fields(coefficients)
        if isinstance(getattr(coefficients, item.name), np.ndarray)
    )
    assert linear.nbytes == expected
    full, prefactor = separable_values(coefficients)
    chunk, selected = separable_values(coefficients, slice(2, 7))
    # A chunk is the same elementwise evaluation on fewer columns; a vector loop may
    # round its tail differently, so it is held to a few roundings of the values.
    np.testing.assert_allclose(
        chunk, full[:, 2:7], rtol=0, atol=gamma(4) * np.max(np.abs(full))
    )
    np.testing.assert_allclose(
        selected, prefactor[2:7], rtol=0, atol=gamma(4) * np.max(np.abs(prefactor))
    )
    # Six scalar products/additions integrate this constant operator.
    bound = gamma(16) * np.max(np.abs(full))
    measured = delta_phi(linear, 0.001, 0.003)
    assert np.max(np.abs(measured - full * 0.002)) < bound
    # The unitary part is one product per element on the constant path.
    np.testing.assert_allclose(
        unitary_phase(linear, 0.002),
        np.imag(full) * 0.002,
        rtol=0,
        atol=gamma(2) * 0.002 * np.max(np.abs(full)),
    )


@pytest.mark.parametrize(
    "name,value",
    [("n_pol", 0), ("n_pol", 3), ("z_max", 0.0), ("z_max", np.inf), ("margin", -0.1)],
)
def test_free_translation_controls_are_validated(free_inputs, name, value):
    space, time, dispersion = free_inputs
    controls = {"n_pol": 1, "z_max": time.length, "margin": 0.01}
    controls[name] = value
    for factory, argument in ((free_transform, time), (free_operators, dispersion)):
        with pytest.raises(TranslationError, match=r"polarisation|span"):
            factory(space, argument, **controls)


def test_free_operator_names_its_varying_medium_refusal(free_inputs):
    space, time, _ = free_inputs
    medium = SpatialMedium("Ar", pressure=Linear(1.0, 2.0, time.length))
    with pytest.raises(TranslationError, match="constant media"):
        free_operators(
            space,
            SpatialDispersion(medium, time, Frame.comoving(time)),
            n_pol=1,
            z_max=time.length,
        )


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_grazing_correction_resolves_the_original_microscopic_grid(kind):
    constructor = FieldGrid.from_limits if kind == "field" else EnvelopeGrid.from_limits
    time = constructor(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-7, 9, 4e-7, 12, window_factor=0)
    dispersion = SpatialDispersion(SpatialMedium(), time, Frame.comoving(time))
    expected = dispersion.shared_values()
    linear, data = free_operators(space, dispersion, n_pol=1, z_max=time.length)
    assert linear.separable is data
    np.testing.assert_array_equal(data.k_squared, expected[0])
    assert np.any(data.k_squared_low != 0)
    assert np.any(data.kperp_sq_low != 0)
    np.testing.assert_array_equal(
        data.prefactor_numerator, expected[2] * time.crop_scale
    )
    np.testing.assert_array_equal(data.prefactor_clamped, expected[3] * time.crop_scale)


@pytest.mark.parametrize("side", [-1, 0, 1])
@pytest.mark.parametrize("relative_offset", [1.0, 1024.0])
def test_grazing_diagnostic_respects_the_prederived_roundoff_envelope(
    free_inputs, side, relative_offset
):
    space, time, dispersion = free_inputs
    _, original = free_operators(space, dispersion, n_pol=1, z_max=time.length)
    transverse = float(np.unique(space.kperp_sq)[1])
    # At equal high parts Bq is 128*epsilon²*r. One epsilon²*r is
    # unresolved; 1024*epsilon²*r is resolved, before evaluating the warning.
    residual = side * relative_offset * np.finfo(float).eps ** 2 * transverse
    data = dataclasses.replace(
        original,
        k_squared=np.full(time.n_omega, transverse, complex),
        k_squared_low=np.full(time.n_omega, residual, complex),
        kperp_sq_low=np.zeros(space.n_columns),
    )
    expected_warning = side == 0 or relative_offset == 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GrazingIncidenceWarning)
        _warn_grazing(space, time, data)
    assert len(caught) == int(expected_warning)
    if expected_warning:
        assert caught[0].category is GrazingIncidenceWarning
        assert issubclass(GrazingIncidenceWarning, SolpropWarning)
        message = str(caught[0].message)
        assert "omega bin" in message and "ix=" in message and "iy=" in message
        assert "m^-2" in message and "does not measure" in message
        assert "Continuing" in message and "split-square" in message


def test_grazing_diagnostic_ignores_inactive_frequency_bins(free_inputs):
    space, time, dispersion = free_inputs
    _, original = free_operators(space, dispersion, n_pol=1, z_max=time.length)
    square = original.k_squared.copy()
    square[~time.active_mask] = space.kperp_sq[1]
    data = dataclasses.replace(original, k_squared=square)
    _warn_grazing(space, time, data)


def test_grazing_search_checks_all_tied_high_parts(free_inputs):
    space, time, dispersion = free_inputs
    _, original = free_operators(space, dispersion, n_pol=1, z_max=time.length)
    high, counts = np.unique(space.kperp_sq, return_counts=True)
    transverse = high[np.flatnonzero(counts >= 3)[0]]
    columns = np.flatnonzero(space.kperp_sq == transverse)
    offset = 1024 * np.finfo(float).eps ** 2 * transverse
    low = np.full(space.n_columns, offset)
    low[columns[1]] = 0
    data = dataclasses.replace(
        original,
        k_squared=np.full(time.n_omega, transverse, complex),
        k_squared_low=np.zeros(time.n_omega, complex),
        kperp_sq_low=low,
    )
    with pytest.warns(GrazingIncidenceWarning, match=f"column {columns[1]} "):
        _warn_grazing(space, time, data)

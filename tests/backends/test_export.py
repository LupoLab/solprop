# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference export of a session's field (SEP 47 section 5.4).

:func:`solprop.backends.export.apply_recipe` fills a destination with the field
converted by a recipe. These tests pin its refusals and its in-place conversion, which
the session lifecycle suite reaches only through whole runs.
"""

import numpy as np
import pytest

from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.exceptions import BackendError, GridError
from solprop.fourier import power_amplitude, spectrum_to_optics

POWER = ExportRecipe(power=True, optics=False)
STORED = ExportRecipe(power=True, optics=True)
OPTICS = ExportRecipe(power=False, optics=True)
IDENTITY = ExportRecipe(power=False, optics=False)


def state() -> np.ndarray:
    """Return a small modal field: two modes, one polarisation, four bins."""
    rng = np.random.default_rng(47)
    return rng.standard_normal((2, 1, 4)) + 1j * rng.standard_normal((2, 1, 4))


def test_a_true_power_export_refuses_a_missing_weight():
    with pytest.raises(GridError, match="needs the route's weight"):
        apply_recipe(state(), POWER, weight=None)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_true_power_export_refuses_a_weight_not_positive_and_finite(bad):
    weight = np.full((2, 1, 4), 2.0)
    weight[1, 0, 2] = bad
    with pytest.raises(GridError, match="positive and finite"):
        apply_recipe(state(), POWER, weight=weight)


def test_an_identity_export_of_a_field_into_itself_leaves_it_unchanged():
    field = state()
    before = field.copy()
    assert apply_recipe(field, IDENTITY, weight=None, out=field) is field
    np.testing.assert_array_equal(field, before)


def test_a_unit_weight_export_into_itself_converts_in_place_without_scaling():
    field = state()
    expected = spectrum_to_optics(field.copy(), kind="envelope")
    out = apply_recipe(field, STORED, weight=np.ones((2, 1, 4)), out=field)
    assert out is field
    np.testing.assert_array_equal(field, expected)


def test_an_export_into_itself_converts_with_the_two_step_bits():
    field = state()
    weight = np.full((2, 1, 4), 3.0)
    expected = spectrum_to_optics(
        power_amplitude(field.copy(), weight), kind="envelope"
    )
    apply_recipe(field, STORED, weight=weight, out=field)
    np.testing.assert_array_equal(field, expected)


def test_a_destination_overlapping_the_state_is_refused():
    field = state()
    # A writeable, C-contiguous complex128 view of the field that is not the field.
    view = field[:]
    assert view is not field
    with pytest.raises(BackendError, match="must not overlap"):
        apply_recipe(field, OPTICS, weight=None, out=view)


def test_a_destination_of_the_wrong_shape_or_kind_is_refused():
    field = state()
    with pytest.raises(BackendError, match="has shape"):
        apply_recipe(field, OPTICS, weight=None, out=np.empty((2, 1, 3), complex))
    with pytest.raises(BackendError, match="C-contiguous complex128"):
        apply_recipe(field, OPTICS, weight=None, out=np.empty((2, 1, 4), np.complex64))

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The export recipe's reference pass (SEP 47 section 5.4).

:func:`~solprop.backends.export.apply_recipe` is the one conversion a saved field
makes on its way to a destination. Its bits are those of today's two steps, the
true-power weight (``fourier.power_amplitude``) and then the optics convention
(``fourier.spectrum_to_optics``), with signed zeros and every sign of value; that is a
structural contract of lossless storage, so the equalities here are exact.


It also pins the refusals and the in-place conversion, which the session lifecycle
suite reaches only through whole runs.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from solprop import _process, fourier
from solprop.backends.export import ExportRecipe, apply_recipe, destination_refusal
from solprop.exceptions import BackendError, GridError

SHAPE = (3, 2, 8)

RECIPES = [
    ExportRecipe(power=power, optics=optics)
    for power in (False, True)
    for optics in (False, True)
]


def _field() -> np.ndarray:
    """Return a field with signed zeros in both parts and values of every sign."""
    rng = np.random.default_rng(346)
    field = rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)
    parts = field.view(np.float64).reshape(*SHAPE, 2)
    parts[..., ::3, 0] = -0.0
    parts[..., 1::4, 1] = 0.0
    parts[..., 2::5, 1] = -0.0
    return field


def _weight(*, unit: bool) -> np.ndarray:
    rng = np.random.default_rng(47)
    return np.ones(SHAPE) if unit else 0.5 + rng.random(SHAPE)


def _two_steps(field: np.ndarray, recipe: ExportRecipe, weight: np.ndarray):
    """Return today's conversion in two steps, each into a new array."""
    converted = fourier.power_amplitude(field, weight) if recipe.power else field.copy()
    if recipe.optics:
        converted = fourier.spectrum_to_optics(converted, kind="envelope")
    return converted


@pytest.mark.parametrize("unit", [True, False], ids=["unit weight", "weighted"])
@pytest.mark.parametrize("recipe", RECIPES, ids=str)
@pytest.mark.parametrize("threads", [False, True], ids=["one call", "threads"])
def test_the_pass_has_the_bits_of_the_two_steps(recipe, unit, threads, monkeypatch):
    if threads:
        monkeypatch.setattr(_process, "PARALLEL_BYTES", 0)
        monkeypatch.setattr(
            _process, "thread_budget", lambda: SimpleNamespace(threads=4)
        )
    field = _field()
    weight = _weight(unit=unit)
    expected = _two_steps(field, recipe, weight)
    original = field.tobytes()
    fresh = apply_recipe(field, recipe, weight=weight)
    assert fresh.tobytes() == expected.tobytes()
    assert not np.shares_memory(fresh, field)
    out = np.full(SHAPE, np.nan + 1j * np.nan)
    assert apply_recipe(field, recipe, weight=weight, out=out) is out
    assert out.tobytes() == expected.tobytes()
    assert field.tobytes() == original  # the state is only read
    in_place = field.copy()
    assert apply_recipe(in_place, recipe, weight=weight, out=in_place) is in_place
    assert in_place.tobytes() == expected.tobytes()


def test_a_spatial_recipe_needs_no_weight():
    field = _field()
    recipe = ExportRecipe(power=False, optics=True)
    expected = fourier.spectrum_to_optics(field, kind="envelope")
    assert apply_recipe(field, recipe, weight=None).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("weight", "match"),
    [
        (None, "needs the route's weight"),
        (np.zeros(SHAPE), "positive and finite"),
        (np.full(SHAPE, np.inf), "positive and finite"),
    ],
    ids=["missing", "zero", "infinite"],
)
def test_an_unusable_weight_is_refused(weight, match):
    with pytest.raises(GridError, match=match):
        apply_recipe(_field(), ExportRecipe(power=True, optics=True), weight=weight)


def test_an_unfit_destination_is_refused_by_what_is_wrong():
    field = _field()
    readonly = np.empty(SHAPE, complex)
    readonly.setflags(write=False)
    unfit = [
        ("has shape", np.empty((3, 2, 9), complex)),
        ("has dtype complex64", np.empty(SHAPE, np.complex64)),
        ("C-contiguous", np.empty((3, 2, 16), complex)[..., ::2]),
        ("read-only", readonly),
        ("overlap", field[...]),
    ]
    for named, out in unfit:
        with pytest.raises(BackendError, match=named):
            apply_recipe(
                field, ExportRecipe(power=False, optics=True), weight=None, out=out
            )
    # A conversion in place is refused where its array is no destination either.
    strided = np.empty((3, 2, 16), complex)[..., ::2]
    with pytest.raises(BackendError, match="C-contiguous"):
        apply_recipe(
            strided, ExportRecipe(power=False, optics=True), weight=None, out=strided
        )


def test_a_fit_destination_has_no_refusal():
    assert destination_refusal(np.empty(SHAPE, complex), SHAPE) is None
    assert destination_refusal(np.empty((1, *SHAPE), complex)[0], SHAPE) is None


POWER = ExportRecipe(power=True, optics=False)
STORED = ExportRecipe(power=True, optics=True)
OPTICS = ExportRecipe(power=False, optics=True)
IDENTITY = ExportRecipe(power=False, optics=False)


def state() -> np.ndarray:
    """Return a small modal field: two modes, one polarisation, four bins."""
    rng = np.random.default_rng(47)
    return rng.standard_normal((2, 1, 4)) + 1j * rng.standard_normal((2, 1, 4))


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
    expected = fourier.spectrum_to_optics(field.copy(), kind="envelope")
    out = apply_recipe(field, STORED, weight=np.ones((2, 1, 4)), out=field)
    assert out is field
    np.testing.assert_array_equal(field, expected)


def test_a_destination_overlapping_the_state_is_refused():
    field = state()
    # A writeable, C-contiguous complex128 view of the field that is not the field.
    view = field[:]
    assert view is not field
    with pytest.raises(BackendError, match="must not overlap"):
        apply_recipe(field, OPTICS, weight=None, out=view)

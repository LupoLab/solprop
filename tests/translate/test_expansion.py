# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop.exceptions import TranslationError
from solprop.profiles import Constant, FromCallable, Linear, PressureGradient
from solprop.translate.expansion import (
    MAX_PARAMETERS,
    RANGE_SAMPLES,
    ParameterExpansion,
    _mapping_noise,
    _weight_tables,
    build_expansion,
)

# The mirror-extension FFT of length 2n loses O(log2(2n) * eps) and the linear
# combination over n_terms loses Higham's n * eps; at the degrees these tests reach that
# is well under 1e-14. Rounded up once and reused, as tests/translate/test_chebyshev.py
# does with its own round-off atol.
ROUND_OFF_ATOL = 5e-14


def one_path(power=1):
    """Return ``(paths, value_at)`` for ``rho -> [1, rho, ..., rho**power]``."""
    paths = {"density": Linear(1.0, 3.0, 1.0)}

    def value_at(values):
        return np.array([values["density"] ** k for k in range(power + 1)])

    return paths, value_at


# --------------------------------------------------------------- one parameter


@pytest.mark.parametrize("power", [0, 1, 5])
def test_a_polynomial_in_the_parameter_is_reproduced_to_rounding(power):
    """The tail test sees a polynomial exactly, so the expansion is the polynomial."""
    paths, value_at = one_path(power)
    expansion = build_expansion(value_at, paths, 0.0, 1.0)
    assert expansion is not None
    assert expansion.degrees.tolist() == [power]
    assert expansion.n_terms == power + 1
    for z in (0.0, 0.13, 0.5, 1.0):
        density = float(paths["density"](np.array([z]))[0])
        np.testing.assert_allclose(
            expansion(z),
            [density**k for k in range(power + 1)],
            rtol=0.0,
            atol=ROUND_OFF_ATOL * max(1.0, density**power),
        )


def test_a_constant_value_collapses_to_one_term():
    paths = {"density": Linear(1.0, 3.0, 1.0)}
    expansion = build_expansion(lambda values: np.array([2.5]), paths, 0.0, 1.0)
    assert expansion is not None
    assert expansion.n_terms == 1
    assert float(expansion(0.7).item()) == 2.5


def test_the_integrated_weights_make_the_expansion_cumulative():
    """With ``integrate`` the value at z is the integral of the quantity from z0."""
    paths = {"density": Linear(0.0, 2.0, 1.0)}
    expansion = build_expansion(
        lambda values: np.array([values["density"]]), paths, 0.0, 1.0, integrate=True
    )
    assert expansion is not None
    # rho(z) = 2z, so int_0^z rho = z**2.
    for z in (0.0, 0.25, 1.0):
        assert float(expansion(z).item()) == pytest.approx(z**2, abs=ROUND_OFF_ATOL)
    assert float(expansion.delta(0.25, 0.75).item()) == pytest.approx(
        0.75**2 - 0.25**2, abs=ROUND_OFF_ATOL
    )


def test_the_value_at_the_start_of_a_cumulative_expansion_is_exactly_zero():
    paths = {"density": Linear(1.0, 2.0, 1.0)}
    expansion = build_expansion(
        lambda values: np.array([values["density"] ** 2]),
        paths,
        0.0,
        1.0,
        integrate=True,
    )
    assert expansion is not None
    assert float(expansion(0.0).item()) == 0.0


def test_a_declared_kink_is_an_edge_of_the_scalar_tables():
    fill = PressureGradient.from_stages([0.0, 0.4, 1.0], [5.0, 5.0, 0.5])
    paths = {"density": fill}
    expansion = build_expansion(
        lambda values: np.array([values["density"] ** 3]),
        paths,
        0.0,
        1.0,
        kinks=fill.breakpoints,
        integrate=True,
    )
    assert expansion is not None
    assert expansion.weights.hard_kinks.tolist() == [0.4]


def test_the_expansion_reports_its_span_and_its_size():
    paths, value_at = one_path()
    expansion = build_expansion(value_at, paths, 0.0, 1.0)
    assert expansion is not None
    assert expansion.z_span == (0.0, 1.0)
    assert expansion.value_shape == (2,)
    assert not expansion.budget_exceeded
    assert expansion.nbytes == (
        expansion.terms.nbytes + expansion.weights.coefficients.nbytes
    )


# -------------------------------------------------------------- two parameters


def test_a_tensor_product_resolves_each_axis_to_its_own_degree():
    """The multi-index is the tensor product, flattened in C order."""
    paths = {"core_radius": Linear(1.0, 2.0, 1.0), "density": Linear(1.0, 3.0, 1.0)}

    def value_at(values):
        return np.array([values["core_radius"] ** 3 * values["density"]])

    expansion = build_expansion(value_at, paths, 0.0, 1.0)
    assert expansion is not None
    assert expansion.parameters == ("core_radius", "density")
    assert expansion.degrees.tolist() == [3, 1]
    assert expansion.n_terms == 8
    for z in (0.0, 0.3, 1.0):
        radius = float(paths["core_radius"](np.array([z]))[0])
        density = float(paths["density"](np.array([z]))[0])
        assert float(expansion(z).item()) == pytest.approx(
            radius**3 * density, abs=1e-13
        )


def test_more_parameters_than_the_tensor_grid_carries_are_refused():
    paths = {f"p{index}": Linear(1.0, 2.0, 1.0) for index in range(MAX_PARAMETERS + 1)}
    assert build_expansion(lambda values: np.array([1.0]), paths, 0.0, 1.0) is None


# --------------------------------------------------------- when it cannot be done


def test_a_kink_in_the_parameter_is_refused_rather_than_smeared():
    """A quantity that is not analytic in the parameter has no truncation to reach."""
    paths = {"density": Linear(0.0, 2.0, 1.0)}

    def value_at(values):
        return np.array([abs(values["density"] - 1.0)])

    assert build_expansion(value_at, paths, 0.0, 1.0, tol=1e-12) is None


def test_the_refusal_is_logged_with_the_parameter_that_did_not_resolve(caplog):
    paths = {"density": Linear(0.0, 2.0, 1.0)}
    with caplog.at_level("DEBUG", logger="solprop.translate.expansion"):
        build_expansion(
            lambda values: np.array([abs(values["density"] - 1.0)]), paths, 0.0, 1.0
        )
    assert "density" in caplog.text
    assert "falling back to the z table" in caplog.text


def test_a_build_with_no_paths_is_refused():
    with pytest.raises(TranslationError, match="at least one varying path"):
        build_expansion(lambda values: np.array([1.0]), {}, 0.0, 1.0)


def test_a_constant_path_is_refused_rather_than_given_a_dimension():
    with pytest.raises(TranslationError, match="does not vary"):
        build_expansion(
            lambda values: np.array([values["density"]]),
            {"density": Constant(2.0)},
            0.0,
            1.0,
        )


def test_a_reversed_range_is_refused():
    with pytest.raises(TranslationError, match="need z1 > z0"):
        build_expansion(
            lambda values: np.array([values["density"]]),
            {"density": Linear(1.0, 2.0, 1.0)},
            1.0,
            0.0,
        )


def test_a_path_that_leaves_its_bracketed_range_is_refused():
    """A Chebyshev series must never be extrapolated, so a bad bracket is an error.

    The range is bracketed by sampling, which a measured profile can defeat. Here the
    corner drives the adaptive split of the scalar tables down to its own width floor,
    far below the sweep's spacing, and a needle beside the corner is then sampled by the
    split although the sweep never saw it.
    """
    centre = 0.5 + 0.5 / (RANGE_SAMPLES - 1)

    def spiky(z):
        z = np.asarray(z, dtype=float)
        needle = np.exp(-(((z - centre - 3e-5) / 1e-5) ** 2))
        return 1.0 + z + 3.0 * np.abs(z - centre) + needle

    paths = {"density": FromCallable(spiky, (0.0, 1.0), kinks=np.empty(0))}
    with pytest.raises(TranslationError, match="would be extrapolated"):
        build_expansion(lambda values: np.array([values["density"]]), paths, 0.0, 1.0)


# -------------------------------------------------------------- the group itself


def usable_parts():
    """Return a valid one-parameter expansion, whose pieces are spoiled one by one."""
    paths, value_at = one_path()
    expansion = build_expansion(value_at, paths, 0.0, 1.0)
    assert expansion is not None
    return expansion


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ranges", np.zeros((2, 2)), "need ranges of shape"),
        ("degrees", np.array([1, 1]), "need ranges of shape"),
        ("ranges", np.array([[3.0, 1.0]]), "must be increasing"),
        ("terms", np.zeros((3, 2)), "terms has"),
    ],
)
def test_the_group_refuses_pieces_that_do_not_agree(field, value, message):
    expansion = usable_parts()
    parts = {
        "parameters": expansion.parameters,
        "ranges": expansion.ranges,
        "degrees": expansion.degrees,
        "terms": expansion.terms,
        "weights": expansion.weights,
        field: value,
    }
    with pytest.raises(TranslationError, match=message):
        ParameterExpansion(**parts)


def test_the_group_refuses_weights_of_the_wrong_value_shape():
    expansion = usable_parts()
    paths, value_at = one_path(3)
    other = build_expansion(value_at, paths, 0.0, 1.0)
    assert other is not None
    with pytest.raises(TranslationError, match="weights has value shape"):
        ParameterExpansion(
            expansion.parameters,
            expansion.ranges,
            expansion.degrees,
            expansion.terms,
            other.weights,
        )


def test_a_path_that_is_not_finite_over_the_range_is_refused():
    """The range cannot be bracketed from values that are not there."""

    def broken(z):
        return np.where(np.asarray(z) < 0.5, 1.0, np.inf)

    paths = {"density": FromCallable(broken, (0.0, 1.0), np.empty(0))}
    with pytest.raises(TranslationError, match="is not finite over"):
        build_expansion(lambda values: np.array([values["density"]]), paths, 0.0, 1.0)


def test_a_sampler_that_changes_shape_is_refused():
    paths, _ = one_path()
    shapes = iter([np.zeros(2), np.zeros(3)])

    def value_at(values):
        del values
        return next(shapes)

    with pytest.raises(TranslationError, match="having returned"):
        build_expansion(value_at, paths, 0.0, 1.0)


def test_a_sampler_that_returns_a_non_finite_value_is_refused():
    paths, _ = one_path()
    with pytest.raises(TranslationError, match="non-finite value"):
        build_expansion(lambda values: np.array([np.inf]), paths, 0.0, 1.0)


def test_the_build_reports_what_it_cost_at_debug_level(caplog):
    paths, value_at = one_path(3)
    with caplog.at_level("DEBUG", logger="solprop.translate.expansion"):
        build_expansion(value_at, paths, 0.0, 1.0)
    assert "expansion in ('density',)" in caplog.text
    assert "4 terms" in caplog.text


def test_a_narrow_range_is_resolved_at_its_mapping_noise() -> None:
    # A path varying by 1e-5 of its value maps onto [-1, 1] with about 2e5 eps of
    # rounding noise, far above the requested 1e-12: without the mapping floor the split
    # tiles the start of the range until it exhausts its budget.
    paths = {
        "density": FromCallable(
            lambda z: 1.0 + 1e-5 * np.sqrt(1.0 + z), (0.0, 1.0), kinks=np.empty(0)
        )
    }
    ranges = np.array([[1.0 + 1e-5, 1.0 + 1e-5 * np.sqrt(2.0)]])
    floor = _mapping_noise(ranges)
    assert 1e-11 < floor < 1e-9
    table = _weight_tables(
        paths,
        ("density",),
        ranges,
        np.array([2]),
        0.0,
        1.0,
        tol=1e-12,
        kinks=(),
        n_max=64,
        max_intervals=512,
        integrate=False,
    )
    assert not table.budget_exceeded
    z = np.linspace(0.0, 1.0, 201)
    x = (2.0 * paths["density"](z) - ranges[0, 0] - ranges[0, 1]) / np.ptp(ranges[0])
    expected = np.stack([np.ones_like(x), x, 2.0 * x**2 - 1.0], axis=1)
    # Accepted at the floor, the table is within it of the products (aliasing factor
    # two), plus the sampled paths' own noise, the floor's sizing.
    values = np.stack([table(float(one)) for one in z])
    assert np.max(np.abs(values - expected)) <= 4.0 * floor


def test_the_mapping_floor_stays_below_the_default_tolerance_for_wide_ranges() -> None:
    # A 0.7 % density range (a short window of a gentle gradient) gives about 300 eps of
    # noise: below 1e-12, so such tables are built exactly as before.
    ranges = np.array([[1.9626e25, 1.9758e25]])
    assert _mapping_noise(ranges) < 1e-12
    assert _mapping_noise(np.array([[2.0, 2.0]])) == 0.0

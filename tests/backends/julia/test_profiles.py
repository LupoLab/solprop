# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify independent profile statistics without native or Julia evaluation."""

import gc
import json
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from backends.julia.cases import cpu_case
from runtime_types import unchecked
from solprop.backends.julia import _profiles
from solprop.backends.julia._descriptors import TableTransfer
from solprop.backends.julia._profiles import lower_statistic_profile
from solprop.exceptions import BackendError, TranslationError
from solprop.profiles import Constant, Linear, PiecewiseLinear, PressureGradient
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.linop import DEFAULT_MARGIN, profile_table


@pytest.fixture
def problem():
    return cpu_case().problem


def _evaluate(value, positions):
    """Evaluate transferred polynomials independently with NumPy Chebyshev."""
    if isinstance(value, float):
        return np.full_like(positions, value)
    actual = np.empty_like(positions)
    for index, position in enumerate(positions):
        interval = np.clip(
            np.searchsorted(value.edges, position, side="right") - 1,
            0,
            len(value.edges) - 2,
        )
        lo, hi = value.edges[interval : interval + 2]
        coordinate = (2 * position - lo - hi) / (hi - lo)
        actual[index] = np.polynomial.chebyshev.chebval(
            coordinate, value.coefficients[0, :, interval]
        )
    return actual


def _check_set(profile, start, end, edges):
    knots = profile.breakpoints
    middles = (edges[:-1] + edges[1:]) / 2
    return np.unique(
        np.concatenate(
            (
                np.linspace(start, end, 4097),
                knots[(start <= knots) & (knots <= end)],
                edges[(start <= edges) & (edges <= end)],
                middles[(start <= middles) & (middles <= end)],
            )
        )
    )


def _with_density(problem, table):
    return replace(
        problem,
        linear=replace(problem.linear, beta0=None, beta1=None),
        responses=replace(problem.responses, density=table),
    )


def _constant_table(value, *, degree=0, singleton=False):
    shape = (2, degree + 1, 1) if singleton else (2, degree + 1)
    coefficients = np.zeros(shape, dtype=float)
    coefficients[:, 0] = value
    return PiecewiseChebyshev(
        coefficients,
        np.array([0.0, 0.007, 0.03]),
        np.full(2, degree, dtype=np.intp),
        np.empty(0),
    )


@pytest.mark.parametrize(
    ("name", "profile"),
    [
        ("density", Linear(1e24, 1.8e24, 0.014)),
        ("pressure", PressureGradient.two_point(0.02, 1.0, 3.0)),
        (
            "core_radius",
            PiecewiseLinear(
                np.array([0.0, 0.0013, 0.003]), np.array([125e-6, 119e-6, 122e-6])
            ),
        ),
    ],
)
def test_distinct_profiles_keep_units_knots_and_measured_error(problem, name, profile):
    result = lower_statistic_profile(problem, name, profile, 0.02)
    assert result.source == "lowered profile"
    assert isinstance(result.value, TableTransfer)
    value = result.value
    assert value.edges[0] == 0.0
    assert value.edges[-1] == 0.02 * (1 + DEFAULT_MARGIN)
    for knot in profile.breakpoints:
        if 0.0 < knot < value.edges[-1]:
            assert knot in value.edges
    assert result.relative_error <= 1e-12
    check = _check_set(profile, 0.0, 0.02, value.edges)
    assert result.check_points == len(check)
    reference = profile(check)
    actual = _evaluate(value, check)
    # The record is a measured dense-set error. Independent Clenshaw evaluation
    # adds a gamma_(8*n_deg) budget scaled by the absolute coefficient sum.
    scale = float(np.max(np.abs(reference)))
    rounding = (
        8
        * value.coefficients.shape[1]
        * np.finfo(float).eps
        * float(np.max(np.sum(np.abs(value.coefficients), axis=1)))
    )
    assert rounding > 0.0
    assert np.max(np.abs(actual - reference)) <= result.max_error + rounding
    assert result.relative_error == result.max_error / scale
    assert json.loads(json.dumps(result.record(), allow_nan=False)) == result.record()
    # A radius ending before the run holds its end; pressure remains in bar.
    independent_positions = np.array([0.0, 0.0047, 0.0169, 0.02, value.edges[-1]])
    np.testing.assert_allclose(
        _evaluate(value, independent_positions),
        profile(independent_positions),
        rtol=1e-12,
        atol=rounding,
    )


def test_measurement_reuses_exact_scalar_table_and_copies_its_storage(problem):
    profile = Linear(2.0, 3.0, 0.02)
    candidate = profile_table(profile, 0.0, 0.03)
    original = _with_density(problem, candidate)
    result = lower_statistic_profile(original, "pressure", profile, 0.02)
    assert result.source == "responses.density"
    assert isinstance(result.value, TableTransfer)
    value = result.value
    edges, lower, upper, packed = candidate.packed()
    assert np.array_equal(value.edges, edges)
    assert np.array_equal(value.coefficients, packed.transpose(2, 1, 0))
    assert (value.lower, value.upper) == (lower, upper)
    for owned, source in ((value.edges, edges), (value.coefficients, packed)):
        assert owned.flags.owndata
        assert owned.flags.f_contiguous
        assert not owned.flags.writeable
        assert not np.shares_memory(owned, source)
        with pytest.raises(ValueError, match="read-only"):
            owned.flat[0] = 0.0
    metadata = result.record()
    metadata["source"] = "changed"
    assert result.source == "responses.density"
    assert result.record()["source"] == "responses.density"
    with pytest.raises(FrozenInstanceError):
        result.__setattr__("source", "changed")


@pytest.mark.parametrize("value", [0.0, 2.5, -2.5])
@pytest.mark.parametrize("singleton", [False, True])
def test_only_structurally_exact_constant_series_collapse(problem, value, singleton):
    candidate = _constant_table(value, singleton=singleton)
    result = lower_statistic_profile(
        _with_density(problem, candidate), "density", Constant(value), 0.02
    )
    assert result.source == "responses.density"
    assert isinstance(result.value, float)
    assert result.value == value
    if value == 0.0:
        assert not np.signbit(result.value)
    assert result.max_error == result.relative_error == 0.0
    assert result.check_points > 4097


@pytest.mark.parametrize(("value", "degree"), [(2.5, 2), (-0.0, 0)])
def test_mathematically_constant_series_retain_nontrivial_representation(
    problem, value, degree
):
    candidate = _constant_table(value, degree=degree)
    result = lower_statistic_profile(
        _with_density(problem, candidate), "density", Constant(value), 0.02
    )
    assert result.source == "responses.density"
    assert isinstance(result.value, TableTransfer)
    # Coefficient bits, including negative zero, are structural storage data.
    packed = candidate.packed()[3].transpose(2, 1, 0)
    assert np.array_equal(
        result.value.coefficients.view(np.uint64), packed.view(np.uint64)
    )


@pytest.mark.parametrize("kind", ["complex", "vector", "short", "incompatible"])
def test_ineligible_candidates_do_not_replace_the_requested_profile(problem, kind):
    profile = Linear(2.0, 3.0, 0.02)
    candidate = profile_table(profile, 0.0, 0.03)
    if kind == "complex":
        candidate = replace(
            candidate, coefficients=candidate.coefficients.astype(complex)
        )
    elif kind == "vector":
        candidate = replace(
            candidate,
            coefficients=np.repeat(candidate.coefficients[..., None], 2, axis=2),
        )
    elif kind == "short":
        candidate = profile_table(profile, 0.0, 0.01)
    else:
        candidate = profile_table(Linear(5.0, 7.0, 0.02), 0.0, 0.03)
    result = lower_statistic_profile(
        _with_density(problem, candidate), "pressure", profile, 0.02
    )
    assert result.source == "lowered profile"
    positions = np.linspace(0.0, 0.02, 25)
    # Two-term linear interpolation and independent evaluation need only a few
    # rounding operations; this bound includes the fit's dense-set measurement.
    bound = result.max_error + 32 * np.finfo(float).eps * 3.0
    assert bound > 0.0
    np.testing.assert_allclose(
        _evaluate(result.value, positions), profile(positions), atol=bound, rtol=0.0
    )


def test_candidate_discrepancy_overflow_is_not_an_invalid_requested_profile(problem):
    largest = np.finfo(float).max
    problem = _with_density(problem, _constant_table(largest))
    problem = replace(
        problem, linear=replace(problem.linear, beta0=_constant_table(-largest))
    )
    result = lower_statistic_profile(problem, "density", Constant(largest), 0.02)
    assert result.source == "responses.density"
    assert result.value == largest
    assert result.max_error == result.relative_error == 0.0


@pytest.mark.parametrize("end", [0.01, 0.0])
def test_nonmarching_run_uses_existing_short_measurement_extension(problem, end):
    problem = replace(problem, initial=replace(problem.initial, z=0.01))
    result = lower_statistic_profile(problem, "pressure", Linear(2.0, 4.0, 0.02), end)
    assert isinstance(result.value, TableTransfer)
    measured_end = 0.01 + 1e-12
    assert result.value.edges[0] == 0.0
    assert result.value.edges[-1] == measured_end + measured_end * DEFAULT_MARGIN
    assert result.check_points >= 4097


def test_real_fit_retains_best_fallback_or_refuses_by_name(problem, monkeypatch):
    profile = PiecewiseLinear(
        np.array([0.0, 0.0013, 0.003]), np.array([125e-6, 119e-6, 122e-6])
    )
    monkeypatch.setattr(_profiles, "_FIT_TARGET", 0.0)
    result = lower_statistic_profile(problem, "core_radius", profile, 0.02)
    assert result.source == "lowered profile"
    assert 0.0 < result.relative_error <= 1e-12
    measured = []
    # Build the actual three candidates independently; no successful fit is mocked.
    for tolerance in (1e-12, 1e-13, 1e-14):
        table = profile_table(profile, 0.0, 0.021, tol=tolerance)
        check = _check_set(profile, 0.0, 0.02, table.breakpoints)
        exact = profile(check)
        discrepancy = np.max(np.abs(table.at_positions(check) - exact))
        measured.append(float(discrepancy / np.max(np.abs(exact))))
    assert result.relative_error == min(measured)
    monkeypatch.setattr(_profiles, "_PROFILE_TOLERANCE", 0.0)
    with pytest.raises(
        BackendError, match=r"julia core_radius statistic.*best measured"
    ):
        lower_statistic_profile(problem, "core_radius", profile, 0.02)


@dataclass(frozen=True, eq=False)
class _MalformedProfile(Constant):
    kind: str = "nan"

    @property
    def breakpoints(self):
        if self.kind == "knots-nan":
            return np.array([np.nan])
        if self.kind == "knots-shape":
            return np.zeros((1, 1))
        if self.kind == "knots-complex":
            return np.array([0.0j])
        return np.empty(0)

    def __call__(self, z):
        if self.kind == "nan":
            return np.full_like(z, np.nan)
        if self.kind == "infinity":
            return np.full_like(z, np.inf)
        if self.kind == "shape":
            return np.ones((len(z), 1))
        if self.kind == "complex":
            return np.ones(len(z), dtype=complex)
        if self.kind == "text":
            return np.full(len(z), "one")
        if self.kind == "evaluation-error":
            raise TranslationError("deliberate profile evaluation failure")
        return np.ones_like(z)


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "infinity",
        "shape",
        "complex",
        "text",
        "knots-nan",
        "knots-shape",
        "knots-complex",
        "evaluation-error",
    ],
)
def test_malformed_profiles_fail_by_statistic_name_with_original_cause(problem, kind):
    with pytest.raises(BackendError, match="julia pressure statistic") as caught:
        lower_statistic_profile(
            problem, "pressure", _MalformedProfile(1.0, kind=kind), 0.02
        )
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize(
    ("start", "end"),
    [(0.0, np.nan), (0.0, np.inf), (0.0, np.finfo(float).max)],
)
def test_invalid_measurement_spans_fail_before_sampling(problem, start, end):
    problem = replace(problem, initial=replace(problem.initial, z=start))
    with pytest.raises(BackendError, match=r"julia density statistic.*(finite|span)"):
        lower_statistic_profile(problem, "density", Constant(1.0), end)


def test_unknown_statistic_is_named_before_preparation(problem):
    with pytest.raises(BackendError, match=r"julia.*unsupported"):
        unchecked(lower_statistic_profile)(problem, "unsupported", Constant(1.0), 0.02)


def test_no_profile_or_problem_is_retained_and_repeat_calls_are_not_cached():
    problem = cpu_case().problem
    profile = Linear(2.0, 3.0, 0.02)
    problem_ref, profile_ref = weakref.ref(problem), weakref.ref(profile)
    result = lower_statistic_profile(problem, "pressure", profile, 0.02)
    second = lower_statistic_profile(problem, "pressure", profile, 0.02)
    assert second is not result
    assert isinstance(result.value, TableTransfer)
    assert isinstance(second.value, TableTransfer)
    assert not np.shares_memory(result.value.coefficients, second.value.coefficients)
    del problem, profile
    gc.collect()
    assert problem_ref() is None
    assert profile_ref() is None
    assert result.check_points >= 4097


def test_profile_import_and_evaluation_do_not_load_native_or_julia(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class RefuseEngines(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('solprop_native', 'juliacall'):
                    raise AssertionError('unexpected engine import: ' + fullname)
        sys.meta_path.insert(0, RefuseEngines())
        from backends.julia.cases import cpu_case
        from solprop.backends.julia._profiles import lower_statistic_profile
        from solprop.profiles import Linear
        lowered = lower_statistic_profile(cpu_case().problem, 'pressure',
                                          Linear(2., 3., .02), .02)
        assert lowered.check_points >= 4097
        assert 'solprop_native' not in sys.modules
        assert 'juliacall' not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@dataclass(frozen=True, eq=False)
class _ChangingProfile(Linear):
    # A custom callable can read mutable external data despite a frozen container.
    factor: np.ndarray = field(default_factory=lambda: np.ones(1))

    def __call__(self, z):
        return super().__call__(z) * self.factor[0]


def test_mutated_custom_callable_is_measured_again_without_hidden_cache(problem):
    profile = _ChangingProfile(2.0, 3.0, 0.02)
    first = lower_statistic_profile(problem, "pressure", profile, 0.02)
    profile.factor[0] = 2.0
    second = lower_statistic_profile(problem, "pressure", profile, 0.02)
    positions = np.array([0.0, 0.0037, 0.02])
    # Two independently fitted short linear series: fitting measurements plus a
    # 32-epsilon arithmetic allowance at the largest resulting magnitude.
    bound = 2 * first.max_error + second.max_error + 32 * np.finfo(float).eps * 6
    assert bound > 0.0
    np.testing.assert_allclose(
        _evaluate(second.value, positions),
        2 * _evaluate(first.value, positions),
        rtol=0.0,
        atol=bound,
    )


def test_shared_fit_arithmetic_failure_remains_a_named_cause(problem):
    # The finite samples' normalized coefficients are representable, but the
    # shared FFT intermediate currently overflows (#657). Do not swallow it.
    with pytest.raises(
        BackendError, match=r"julia density statistic.*overflow"
    ) as caught:
        lower_statistic_profile(
            problem, "density", Constant(np.finfo(float).max / 4), 0.02
        )
    assert isinstance(caught.value.__cause__, FloatingPointError)


def test_nonfinite_candidate_data_is_not_silently_used_or_replaced(problem):
    problem = _with_density(problem, _constant_table(np.nan))
    with pytest.raises(BackendError, match=r"julia pressure statistic.*finite"):
        lower_statistic_profile(problem, "pressure", Constant(2.5), 0.02)

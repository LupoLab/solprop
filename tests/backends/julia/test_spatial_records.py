# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check spatial preparation ownership, exact coefficients and early admission."""

import gc
import weakref
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from backends.julia.spatial_cases import spatial_case, spatial_collar
from backends.julia.spatial_observation_bounds import spatial_observation_bounds
from solprop.backends.julia._lower import lower_run
from solprop.backends.julia._spatial import (
    CartesianTransfer,
    RadialTransfer,
    lower_spatial,
)
from solprop.backends.julia.backend import JuliaBackend
from solprop.translate.precision import DOUBLE, SINGLE


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_spatial_lowering_owns_exact_representations_and_releases_problem(
    kind, geometry
):
    case, _ = spatial_case(kind, geometry, material=True)
    problem = case.problem
    lowered = lower_spatial(problem)
    assert lower_spatial(problem) is lowered
    assert problem.transform.geometry is not None
    np.testing.assert_array_equal(
        lowered.spatial_rate, problem.transform.geometry.absorber_rate
    )
    assert not np.shares_memory(
        lowered.spatial_rate, problem.transform.geometry.absorber_rate
    )
    if isinstance(lowered, RadialTransfer):
        np.testing.assert_array_equal(
            lowered.modal.synthesis, problem.transform.synthesis
        )
        assert lowered.modal.kerr is not None
    else:
        assert isinstance(lowered, CartesianTransfer)
        assert lowered.source.kerr is not None
        assert problem.linear.separable is not None
        for descriptor in fields(lowered.operator):
            actual = getattr(lowered.operator, descriptor.name)
            expected = getattr(problem.linear.separable, descriptor.name)
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
                assert not np.shares_memory(actual, expected)
        assert lowered.shape == problem.transform.geometry.shape

    def owned(value):
        if isinstance(value, np.ndarray):
            assert value.flags.owndata and value.flags.f_contiguous
            assert not value.flags.writeable
        elif is_dataclass(value):
            for descriptor in fields(value):
                owned(getattr(value, descriptor.name))
        elif isinstance(value, tuple):
            for item in value:
                owned(item)

    owned(lowered)
    reference = weakref.ref(problem)
    del case, problem
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("collar", [False, True])
def test_spatial_absorption_keeps_an_adams_source_without_material(geometry, collar):
    case, _ = spatial_case("envelope", geometry, method="adams")
    if collar:
        case = spatial_collar(case)
    lowered = lower_run(case.problem, case.request)
    assert lowered.source_absent is not collar


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("precision", [DOUBLE, SINGLE], ids=["double", "single"])
def test_internal_spatial_admission_is_lazy_at_both_precisions(
    geometry, precision, monkeypatch
):
    case, _ = spatial_case("field", geometry)

    def forbidden(**kwargs):
        pytest.fail("descriptor admission entered Julia")

    monkeypatch.setattr("solprop.backends.julia.backend.get_runtime", forbidden)
    problem = replace(case.problem, precision=precision)
    owner = JuliaBackend(problem, case.request)
    try:
        assert owner.metadata()["precision"] == precision.name
    finally:
        owner.close()


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_spatial_width_witnesses_satisfy_the_declared_conditioning(kind, geometry):
    case, grid = spatial_case(kind, geometry, resolved=True)
    bounds = spatial_observation_bounds(
        case.problem, grid, case.problem.initial.field0, 0.0
    )
    for name, bound in bounds.items():
        assert np.all(np.isfinite(bound))
        assert np.all(np.asarray(bound) >= 0), name
        assert np.any(np.asarray(bound) > 0), name

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preionisation as a path along z on the responses and components."""

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from rate_oracles import adk_oracle
from solprop.exceptions import MaterialError
from solprop.nonlinear import EnvelopePlasmaResponse, responses_for
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Mixture, species
from solprop.physdata.mixtures import Component, check_preionisation
from solprop.profiles import Constant, FromCallable, Linear, PiecewiseLinear

RAMP = PiecewiseLinear(np.array([0.0, 0.2, 0.5, 1.0]), np.array([0.0, 0.0, 0.3, 0.3]))
"""A fraction rising from nothing to 0.3 between two kinks, over a metre."""


def responses():
    """Return a builder for each response type, taking the preionisation."""
    return {
        # The trapezoid, so that a direct evaluation needs no grid's taps.
        "field": lambda value: PlasmaResponse(
            "Ar", adk_oracle("Ar"), preionisation=value, integration="trapezoid"
        ),
        "envelope": lambda value: EnvelopePlasmaResponse(
            "Ar", adk_oracle("Ar"), preionisation=value
        ),
        "component": lambda value: Component(
            species("Ar"), Constant(1e24), preionisation=value
        ),
    }


@pytest.mark.parametrize("kind", ["field", "envelope", "component"])
@pytest.mark.parametrize(
    "path",
    [RAMP, Linear(0.0, 1.0, 2.0), Constant(1.0), FromCallable(np.cos, (0.0, 1.5))],
    ids=["piecewise", "linear-to-one", "constant", "callable"],
)
def test_a_path_in_the_unit_interval_is_accepted_as_it_is(kind, path):
    assert responses()[kind](path).preionisation is path


@pytest.mark.parametrize("kind", ["field", "envelope", "component"])
@pytest.mark.parametrize(
    ("path", "where"),
    [
        (Linear(0.0, 1.2, 1.0), r"reaches 1\.0125 at z = 0\.84375 m"),
        (Linear(-0.1, 0.5, 1.0), "reaches -0.1 at z = 0 m"),
        (FromCallable(lambda z: np.full(np.shape(z), np.nan), (0.0, 1.0)), "nan"),
        (Constant(1.5), "reaches 1.5 at z = 0 m"),
    ],
    ids=["above", "below", "not-finite", "unbounded-span"],
)
def test_a_path_leaving_the_unit_interval_is_refused_where_it_does(kind, path, where):
    with pytest.raises(MaterialError, match=r"preionisation must be in \[0, 1\].*"):
        responses()[kind](path)
    with pytest.raises(MaterialError, match=where):
        check_preionisation(path)


def test_a_breakpoint_between_the_probes_is_sampled():
    # 65 probes over a metre miss 0.51; the breakpoint itself is sampled.
    spike = PiecewiseLinear(
        np.array([0.0, 0.505, 0.51, 0.515, 1.0]), np.array([0.1, 0.1, 1.1, 0.1, 0.1])
    )
    with pytest.raises(MaterialError, match=r"reaches 1\.1 at z = 0\.51 m"):
        check_preionisation(spike, "Ar")


@pytest.mark.parametrize("kind", ["field", "envelope", "component"])
@pytest.mark.parametrize("bad", ["0.1", [0.1], True])
def test_something_that_is_neither_a_number_nor_a_path_is_refused(kind, bad):
    # Under the test session's runtime hook the annotation refuses a string or a list
    # first; in production, and for a boolean (an int to the hook), the check does.
    with pytest.raises((MaterialError, TypeCheckError), match="preionisation"):
        responses()[kind](bad)


def test_the_check_names_the_owner_and_refuses_a_boolean():
    with pytest.raises(MaterialError, match="preionisation for N2 must be a number"):
        check_preionisation(True, "N2")
    with pytest.raises(MaterialError, match=r"must be in \[0, 1\] for N2, got 2"):
        check_preionisation(2, "N2")


def test_the_field_reference_refuses_a_path_and_takes_a_plane_by_replace():
    plasma = PlasmaResponse(
        "Ar", adk_oracle("Ar"), preionisation=RAMP, integration="trapezoid"
    )
    field = np.full((1, 8), 1e6)
    with pytest.raises(MaterialError, match=r"dataclasses\.replace"):
        plasma.ionised_fraction(field, dt=1e-16, kind="field")
    with pytest.raises(MaterialError, match=r"dataclasses\.replace"):
        plasma.evaluate(field, 1e24, dt=1e-16, kind="field")
    plane = float(RAMP(np.array([0.35]))[0])
    at_plane = dataclasses.replace(plasma, preionisation=plane)
    # 1 - (1 - f0) exp(0) is f0 to one rounding of the subtraction.
    np.testing.assert_allclose(
        at_plane.ionised_fraction(field, dt=1e-16, kind="field"),
        np.full(8, plane),
        rtol=2 * np.finfo(float).eps,
    )


def test_the_envelope_reference_refuses_a_path_and_takes_a_plane_by_replace():
    plasma = EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=RAMP)
    field = np.zeros((1, 3), complex)
    with pytest.raises(MaterialError, match=r"dataclasses\.replace"):
        plasma.sources(field, 1e24, dt=1e-16)
    with pytest.raises(MaterialError, match=r"dataclasses\.replace"):
        plasma.ionised_fraction(field, dt=1e-16)
    at_plane = dataclasses.replace(plasma, preionisation=0.15)
    assert at_plane.ionised_fraction(field, dt=1e-16).tolist() == [0.15] * 3


def test_a_mixture_carries_one_path_per_component_into_its_responses():
    mixture = Mixture.fill({"Ar": 0.5, "He": 0.5}, 1.0)
    helium = Linear(0.01, 0.02, 1.0)
    paths = Mixture(
        tuple(
            dataclasses.replace(
                component, preionisation=RAMP if component.label == "Ar" else helium
            )
            for component in mixture.components
        ),
        mixture.temperature,
        mixture.shared,
        mixture.fractions,
    )
    built = responses_for(paths, plasma=True, kerr=False, lambda0=800e-9)
    by_label = {
        one.species: one.preionisation
        for one in built
        if isinstance(one, PlasmaResponse)
    }
    assert by_label == {"Ar": RAMP, "He": helium}


def test_a_component_path_is_printed_and_recorded_as_a_path():
    number = Component(species("Ar"), Constant(1e24), preionisation=0.25)
    path = Component(species("Ar"), Constant(1e24), preionisation=RAMP)
    assert repr(number) == "Component(Ar, preionisation=0.25)"
    assert repr(path) == f"Component(Ar, preionisation={RAMP!r})"
    mixture = Mixture((path,))
    assert f"preionisation {RAMP!r}" in str(mixture._describe())
    record = component_records(mixture)[0]["preionisation"]
    assert record["kind"] == "PiecewiseLinear"
    assert record["breakpoints_m"] == [0.0, 0.2, 0.5, 1.0]
    np.testing.assert_allclose(record["values"], RAMP(np.array(record["z_m"])))
    assert component_records(Mixture((number,)))[0]["preionisation"] == 0.25


def component_records(mixture):
    """Return the serialised component records, typed for the checker."""
    return cast("list[dict[str, Any]]", mixture.serialise()["components"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float32(0.25), 0.25),
        (np.float64(0.3), 0.3),
        (np.int64(1), 1.0),
        (np.array(0.4), 0.4),
        (np.array(0, dtype=np.int32), 0.0),
    ],
    ids=["float32", "float64", "int64", "0-d-float", "0-d-int"],
)
def test_numpy_real_scalars_are_accepted_as_the_float_they_hold(value, expected):
    checked = check_preionisation(value, "Ar")
    assert type(checked) is float
    assert checked == expected


@pytest.mark.parametrize("bad", [np.bool_(True), np.array([0.3]), np.array(0.5j)])
def test_numpy_values_that_are_not_one_real_number_are_refused(bad):
    with pytest.raises(MaterialError, match="must be a number or a ZProfile"):
        check_preionisation(bad, "Ar")


def test_a_numpy_scalar_is_stored_as_a_python_float():
    for build in responses().values():
        stored = build(np.float64(0.2)).preionisation
        assert type(stored) is float
        assert stored == 0.2

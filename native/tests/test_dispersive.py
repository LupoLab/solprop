# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The adapter of solprop SEP 6's dispersive response.

The native response executes the layout the adapter packs (``_dispersive_payload``),
solprop ``numba_cpu``'s ``dispersive_terms_layout``: these tests hold the layout to
the leaf (the general modal route's regions among it), the field family to the sector
and one evaluation to solprop's reference. The full parity ladders are solprop's
``tests/backends/test_native_dispersive_kerr.py`` (the fused node path) and
``test_native_dispersive_routes.py`` (the Cartesian, line, planar and general modal
routes).
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from bounds import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import MaterialSelection
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import profile_table
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.responses import dispersive_terms

from solprop_native import NativeBackend
from solprop_native.backend import (
    _dispersive_payload,  # pyright: ignore[reportPrivateUsage]
    _native_problem,  # pyright: ignore[reportPrivateUsage]
)

EPS = float(np.finfo(float).eps)
LENGTH = 2e-3
POLICIES = {"symmetric": "symmetric-kerr", "polynomial": "gas-polynomial"}


def capillary(
    policy: str, gas: str | dict[str, float] = "Ar", *, envelope: bool, thg: bool
) -> KernelProblem:
    """Return a short three-mode capillary under a dispersive policy, translated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            75e-6,
            LENGTH,
            gas,
            2.0,
            lambda0=800e-9,
            wavelength_limits=(180e-9, 4e-6),
            trange=150e-15,
            fwhm=10e-15,
            energy=10e-6,
            modes=3,
            materials=MaterialSelection(electronic=policy),
            plasma=False,
            raman=False,
            backend="numpy_ref",
            n_save=2,
            envelope=envelope,
            thg=thg,
        )
        return simulation.translate()


def averaged(evaluator: str, *, thg: bool) -> tuple[KernelProblem, FieldGrid]:
    """Return a mode-averaged argon problem on a field grid, one node."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (150e-9, 4e-6), 150e-15)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=2e-6).build(grid)
    problem = assemble(
        grid,
        MarcatiliMode(75e-6, cladding=1.45, loss=False),
        [
            DispersiveKerrResponse(
                OptmatKerr("Ar").dispersive_model(), evaluator, thg=thg
            )
        ],
        pulse,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / 10,
        density=4.9e25,
        absorber_scheme="split_step",
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 10),
    )
    return problem, grid


def test_a_symmetric_mixture_is_one_product_under_its_filter() -> None:
    """Helium and argon: one group, one field and one product, the factor its own.

    The mixture's term names the factor after the channels and the unit factor; its
    rows stand at the first species' filter until the native stage rebuilds them from
    the channels' ``g^4``.
    """
    problem = capillary(
        "symmetric-kerr", {"He": 0.6, "Ar": 0.4}, envelope=True, thg=False
    )
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    assert leaf.n_channels == 2
    payload = cast(dict[str, Any], _dispersive_payload(problem))
    bins = leaf.positive.size
    assert payload["dispersive_symmetric"] is True
    assert payload["dispersive_form"] == 1
    assert payload["dispersive_inputs"].shape == (1, bins)
    # Copies, indices, masks and power-of-two scalings of a mask are exact; the
    # quartics are three products of the filters', within gamma(3) each way.
    np.testing.assert_array_equal(payload["dispersive_inputs"][0], leaf.filters[0])
    np.testing.assert_array_equal(payload["dispersive_term_rows"], [0])
    np.testing.assert_array_equal(payload["dispersive_term_channels"], [3])
    np.testing.assert_array_equal(payload["dispersive_group_offsets"], [0, 2])
    np.testing.assert_array_equal(payload["dispersive_group_members"], [0, 1])
    squares = leaf.filters * leaf.filters
    np.testing.assert_allclose(
        payload["dispersive_quartics"], squares * squares, rtol=2 * gamma(3), atol=0.0
    )
    positive = leaf.positive.astype(float)
    np.testing.assert_array_equal(payload["dispersive_output_scale"], positive / 4.0)
    np.testing.assert_array_equal(payload["dispersive_input_scale"], np.ones(bins))


def test_the_polynomial_groups_species_of_one_leg_bound() -> None:
    """Krypton and xenon share one leg bound: one cube, one group of two fields.

    Each channel weighs the shared ``E^3`` and ``T(E, E, D E)`` by its own output
    multipliers; the weighted pair carries its channels' factors already.
    """
    problem = capillary(
        "gas-polynomial", {"Kr": 0.5, "Xe": 0.5}, envelope=False, thg=False
    )
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    assert leaf.groups() == ((0, 1),)
    assert leaf.outputs is not None
    payload = cast(dict[str, Any], _dispersive_payload(problem))
    positive = leaf.positive.astype(float)
    # The field grid's conserving sector: the analytic family, Pi_+ / 8 on the output.
    assert payload["dispersive_form"] == 1
    # The masks, indices and power-of-two scalings are exact; a product of filters
    # or of a multiplier and the mask is within gamma(2) each way.
    np.testing.assert_array_equal(payload["dispersive_input_scale"], positive)
    np.testing.assert_allclose(
        payload["dispersive_inputs"],
        np.stack([leaf.filters[0], leaf.filters[0] * leaf.filters[0]]) * leaf.positive,
        rtol=2 * gamma(2),
        atol=0.0,
    )
    np.testing.assert_array_equal(payload["dispersive_term_rows"], [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(payload["dispersive_term_channels"], [0, 0, 1, 1, 2])
    np.testing.assert_allclose(
        payload["dispersive_term_multipliers"][2],
        leaf.outputs[1, 0] * positive / 8.0,
        rtol=2 * gamma(2),
        atol=0.0,
    )
    np.testing.assert_array_equal(
        payload["dispersive_term_multipliers"][4], positive / 8.0
    )
    np.testing.assert_array_equal(
        payload["dispersive_pair_constants"], np.asarray(leaf.products)
    )
    assert payload["dispersive_quartics"].size == 0


@pytest.mark.parametrize(
    ("envelope", "thg", "family", "form"),
    [
        (False, True, "field_real", 0),
        (False, False, "field_analytic", 1),
        (True, False, "envelope", 1),
        (True, True, "envelope", 2),
    ],
)
def test_the_sector_chooses_the_field_family(
    envelope: bool, thg: bool, family: str, form: int
) -> None:
    """A field grid's number-conserving sector is formed on analytic signals."""
    problem = capillary("gas-polynomial", envelope=envelope, thg=thg)
    backend = NativeBackend(problem)
    try:
        payload = _native_problem(problem, backend._terms)  # pyright: ignore[reportPrivateUsage]
    finally:
        backend.close()
    assert payload["family"] == family
    assert payload["dispersive_form"] == form
    assert payload["nonlinear"] is False


def test_a_problem_without_the_leaf_packs_nothing() -> None:
    problem = capillary("carrier-kerr", envelope=False, thg=True)
    assert _dispersive_payload(problem) == {"dispersive": False}


def general_leaf(evaluator: str) -> tuple[Any, FieldGrid]:
    """Return a two-region leaf: helium in region 3, argon and krypton in region 5."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 200e-15)
    table = profile_table(Constant(1e25), 0.0, 1.0)
    species = ("He", "Ar", "Kr")
    responses = [
        DispersiveKerrResponse(
            OptmatKerr(name).dispersive_model(), evaluator, species=name, thg=False
        )
        for name in species
    ]
    leaf = dispersive_terms(responses, list(species), [table] * 3, grid)
    return dataclasses.replace(leaf, regions=np.array([3, 5, 5], np.int32)), grid


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_the_general_route_lays_out_its_regions(evaluator: str) -> None:
    """Each group's key is its region; the polynomial forms one cube per region.

    The symmetric evaluator's argon and krypton are one mixture in region 5; the
    polynomial's three species have three leg bounds, so three groups, and two cubes
    (regions 3 and 5) precede their pairs. Every node's region is packed beside them,
    and the route's grid is a field grid whose number-conserving sector the native
    route forms on its own analytic signal.
    """
    leaf, _ = general_leaf(evaluator)
    region_index = np.array([5, 5, 3, 0, 3], dtype=np.int32)
    problem = SimpleNamespace(
        responses=SimpleNamespace(dispersive_kerr=leaf),
        transform=SimpleNamespace(
            modal_tables=SimpleNamespace(region_index=region_index)
        ),
        grid_kind="field",
    )
    payload = cast(dict[str, Any], _dispersive_payload(cast(KernelProblem, problem)))
    np.testing.assert_array_equal(payload["dispersive_node_region"], region_index)
    assert payload["dispersive_form"] == 1
    if evaluator == "symmetric":
        assert leaf.groups() == ((0,), (1, 2))
        np.testing.assert_array_equal(payload["dispersive_group_keys"], [3, 5])
        np.testing.assert_array_equal(payload["dispersive_term_rows"], [0, 1])
        np.testing.assert_array_equal(payload["dispersive_term_channels"], [0, 4])
        return
    assert leaf.groups() == ((0,), (1,), (2,))
    np.testing.assert_array_equal(payload["dispersive_group_keys"], [3, 5, 5])
    # Two cubes, by region in the order the groups name them; then each group's pair.
    np.testing.assert_array_equal(
        payload["dispersive_term_rows"], [0, 2, 3, 1, 4, 5, 1, 6, 7]
    )


def test_a_trial_without_the_densities_is_refused() -> None:
    """The native response weighs its channels from each stage's densities."""
    problem, grid = averaged("symmetric", thg=True)
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        stages = problem.stepper.tableau.n_stages
        terms = backend._terms.phi  # pyright: ignore[reportPrivateUsage]
        assert terms is not None
        n_phi = terms.shape[0]
        with pytest.raises(ValueError, match="dispersive densities"):
            backend._native_state.trial_step(  # pyright: ignore[reportPrivateUsage]
                1e-4,
                np.zeros((stages + 1) * n_phi),
                np.zeros(stages * n_phi),
                np.ones(stages),
                np.ones(stages),
                np.ones(stages),
                np.zeros(stages),
                np.zeros((stages, 0)),
            )
    finally:
        backend.close()


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("thg", [True, False])
def test_one_evaluation_matches_the_reference(evaluator: str, thg: bool) -> None:
    """The adapter end to end: the projected polarisation of one evaluation.

    The bound is solprop's (``test_numba_dispersive_kerr.evaluation_bound``) for one
    node: the transforms both backends take, ``8 eps log2 n_to`` each, doubled.
    """
    problem, grid = averaged(evaluator, thg=thg)
    transforms = 2 + (3 if evaluator == "symmetric" else 6)
    bound = 2 * transforms * 8 * EPS * np.ceil(np.log2(problem.grids.n_to))
    spectra: list[npt.NDArray[np.complex128]] = []
    for backend_class in (NativeBackend, NumpyRefBackend):
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend._adams_lab_source(problem.initial.field0, 0.0)  # pyright: ignore[reportPrivateUsage]
            if isinstance(backend, NativeBackend):
                spectra.append(backend._native_state.projected_to_host(False))  # pyright: ignore[reportPrivateUsage]
            else:
                projected = np.asarray(backend._last_projected)  # pyright: ignore[reportPrivateUsage]
                spectra.append(np.asarray(np.fft.rfft(projected, axis=-1)))
        finally:
            backend.close()
    ours, theirs = spectra
    scale = float(np.linalg.norm(theirs))
    assert scale > 0
    assert float(np.linalg.norm(ours - theirs)) / scale < bound

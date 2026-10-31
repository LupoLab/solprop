# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route's translation: the record, the transform and assembly.

Three leaves of kernel schema 12 and the ordinary diagonal operator become one problem
here; what the reference backend then does with it is
``tests/backends/test_general_modal.py``.
"""

import ast
import dataclasses
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from legacy_geometry import LEGACY_GRID
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    general_case,
    general_grid,
    general_guide,
)
from runtime_types import unchecked
from solprop.exceptions import ModeError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import StepIndexMode
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.material import IndexMaterial, MaterialKerr
from solprop.modes.quadrature import quadrature_rule
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata.glasses import bk7
from solprop.stepper import PIController
from solprop.translate import general
from solprop.translate.coupled import dense_metric, metric_at, orthonormal_metric
from solprop.translate.general import (
    check_noise_basis,
    general_modal_route,
    interpolated_gram,
    metric_departure,
    modal_identity,
    occupancy_weight,
)
from solprop.translate.modal import general_modal_transform
from solprop.translate.modal_tables import DEFAULT_GRAM_ATOL
from solprop.translate.problem import assemble, debug_summary
from solprop.translate.responses import (
    KerrForm,
    instantaneous_model_id,
    isotropic_kerr_tensor,
    regional_responses,
)


@pytest.fixture(scope="module")
def case():
    """One HE11 channel of a 1.5 um step-index strand, and its route."""
    return general_case(grid=legacy_grid())


def legacy_grid():
    """Return the shared grid as it was built when this module's numbers were taken."""
    return FieldGrid.from_limits(
        GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15, **LEGACY_GRID
    )


def problem_of(case, **overrides):
    """Assemble the case's general modal problem, with arguments replaced."""
    state = np.zeros((1, 1, case.grid.n_omega), dtype=complex)
    state[0, 0, int(np.argmin(np.abs(case.grid.omega - case.grid.omega0)))] = 1.0
    arguments = {
        "controller": PIController(rtol=1e-8, atol=1e-14),
        "save_planes": [GENERAL_Z_END],
        "dz_max": GENERAL_Z_END / 10,
        "dz_init": GENERAL_Z_END / 20,
        "general": case.route,
    } | overrides
    modes = overrides.pop("modes", list(case.guide.mode_collection))
    return assemble(
        case.grid,
        modes,
        overrides.pop("responses", []),
        state,
        **{
            name: value
            for name, value in arguments.items()
            if name not in ("modes", "responses")
        },
    )


# ------------------------------------------------------------------- the record


def test_the_route_carries_the_leaves_and_derives_the_identity_metric(case):
    """The metric is the identity, and its error row is the measured Gram defect."""
    route = case.route
    assert route.n_modes == 1
    metric = route.metric
    assert metric.kind == "identity"
    assert metric.diagonal is None and metric.dense is None
    assert np.all(metric.error == case.table.gram_defect)
    assert np.all(metric.minimum_eigenvalue == 1.0)
    np.testing.assert_array_equal(metric.omega, route.tables.omega)
    assert metric.gauge_id == route.tables.gauge_id
    assert metric.certificate_id == "sampled-gram"


def test_the_route_refuses_a_certificate_it_cannot_stand_behind(case):
    for defect, message in ((-1.0, "gram_defect"), (np.inf, "gram_defect")):
        with pytest.raises(TranslationError, match=message):
            replace(case.route, gram_defect=defect)
    with pytest.raises(TranslationError, match="certificate_id"):
        replace(case.route, certificate_id="")


def test_the_route_refuses_a_defect_it_will_not_call_the_identity(case):
    """A declared defect is gated, not merely recorded.

    The route's whole statement is `W = I`; on a continuous wave the resulting error
    in `da/dz` is `|1/W_00 - 1|`, which is the defect itself to first order, and
    nothing downstream reads the error row again. A route that assembled and ran at a
    defect of 0.5 would be wrong by about half and say nothing, so the budget is
    `gram_atol`, `DEFAULT_GRAM_ATOL` by default -- the same number `DenseModalTable`
    gates its own sampling with.
    """
    with pytest.raises(TranslationError, match="above its gram_atol"):
        replace(case.route, gram_defect=0.5)
    # It is a budget and not a constant: a caller who has looked at the number may
    # declare a wider one, exactly as the table's own gate allows.
    widened = replace(case.route, gram_defect=0.5, gram_atol=0.6)
    assert widened.gram_defect == 0.5
    assert np.all(widened.metric.error == 0.5)
    for atol in (0.0, -1.0, np.inf):
        with pytest.raises(TranslationError, match="gram_atol"):
            replace(case.route, gram_atol=atol)


def test_the_route_measures_its_departure_between_the_knots_and_gates_it(case):
    """The interpolated departure is measured, recorded and gated.

    The **knot** Gram defect alone is not enough to declare ``W = I`` on the identity
    route: what the interpolated basis does at the run's own bins can be three orders
    further from orthonormal on a sparsely sampled guide. It is measured, carried on the
    record and in the modal identity, and gated against the same `gram_atol` the knot
    defect is, because to first order that departure is what declaring the identity
    costs the run's `physical_energy`.
    """
    route = case.route
    # The bound, before the numbers: the occupied departure is a maximum over a
    # subset of the bins the whole-band departure maximises over, so it is at most
    # that, and it must be inside the route's own budget for the route to have
    # assembled at all. It is **not** bounded below by `gram_defect`: that is a
    # maximum over the knots across the whole sampled band, including the
    # long-wavelength end no run occupies, so the two are extrema over different
    # supports and either may be the larger.
    block = interpolated_gram(case.table, route.tables)
    assert 0.0 < route.occupied_departure <= metric_departure(block)
    assert route.occupied_departure <= route.gram_atol
    # And the record carries it, quantised like every other measured entry.
    labels = list(case.guide.mode_collection.channel_labels)
    identity = modal_identity(route, case.grid, channels=labels, has_noise=False)
    recorded = cast("float", entry(identity, "metric", "occupied_departure"))
    assert recorded == pytest.approx(route.occupied_departure, rel=1e-5)
    with pytest.raises(TranslationError, match="between its knots"):
        replace(route, occupied_departure=0.5)
    # A caller who has looked at the number may declare a wider budget, exactly as
    # the knot defect's own gate allows.
    assert replace(route, occupied_departure=0.5, gram_atol=0.6).gram_atol == 0.6
    for bad in (-1e-12, np.inf, np.nan):
        with pytest.raises(TranslationError, match="occupied_departure"):
            replace(route, occupied_departure=bad)


def test_the_occupied_support_is_where_the_light_is_and_not_the_whole_band(case):
    """The gate looks where a run puts light, not where the knots are sparsest.

    `metric_departure`'s default is the maximum over **every** bin, which is the
    right question for a nonunitary transform and the wrong one for interpolation
    error: that error lives at the long-wavelength end of the sampled band, and a
    run's active mask reaches there while its light never does. The three numbers
    are ordered, and the route gates the smallest of them.
    """
    block = interpolated_gram(case.table, case.route.tables)
    grid = case.grid
    everywhere = metric_departure(block)
    inband = metric_departure(block, occupied=grid.inband_mask)
    launch = np.zeros((1, 1, grid.n_omega), dtype=complex)
    launch[0, 0, int(np.argmin(np.abs(grid.omega - grid.omega0)))] = 1.0
    occupied, weight = occupancy_weight(grid, launch)
    weighted = metric_departure(block, occupied=occupied, weight=weight)
    assert inband < everywhere
    # Exactly, a weighted mean over bins inside the band is never above the band's
    # maximum. Computed, `sum(w d) / sum(w)` over `n` positive weights carries one
    # rounding per product, `n - 1` per sum and one for the quotient, so it can sit
    # up to `gamma(2 n + 1)` of the maximum above it: with the launch in one bin the
    # quotient is `fl(fl(w d) / w)`, which is `d` or an ulp either side of it,
    # whichever way the platform's last bits of `d` round.
    assert weight is not None
    count = int(np.count_nonzero(weight[occupied]))
    unit = np.finfo(float).eps / 2
    allowance = (2 * count + 1) * unit / (1 - (2 * count + 1) * unit)
    assert weighted <= inband * (1 + allowance)
    # A run with no declared launch falls back to the declared band, unweighted,
    # which is the conservative of the two: a maximum over a band is never below a
    # mean weighted inside it.
    assert occupancy_weight(grid, None) == (
        pytest.approx(grid.inband_mask),
        None,
    )


def test_a_departure_over_an_empty_or_unweighted_support_is_refused_by_name(case):
    """Nothing occupied and nothing launched are different, and both are refused."""
    block = np.broadcast_to(
        np.eye(2, dtype=complex)[:, :, None], (2, 2, case.grid.n_omega)
    ).copy()
    with pytest.raises(TranslationError, match="at least one occupied bin"):
        metric_departure(block, occupied=np.zeros(case.grid.n_omega, dtype=bool))
    with pytest.raises(TranslationError, match="occupied mask covers"):
        metric_departure(block, occupied=np.ones(3, dtype=bool))
    with pytest.raises(TranslationError, match="weights covers"):
        metric_departure(block, weight=np.ones(3))
    with pytest.raises(TranslationError, match="finite and non-negative"):
        metric_departure(block, weight=-np.ones(case.grid.n_omega))
    with pytest.raises(TranslationError, match="positive total weight"):
        metric_departure(block, weight=np.zeros(case.grid.n_omega))
    with pytest.raises(TranslationError, match="own frequency axis"):
        occupancy_weight(case.grid, np.zeros((1, 1, 3), dtype=complex))
    with pytest.raises(TranslationError, match="no energy inside"):
        occupancy_weight(case.grid, np.zeros((1, 1, case.grid.n_omega), dtype=complex))


def test_the_route_refuses_a_reference_with_no_execution_map(case):
    """A table sampled over a path of positions carries no map."""
    bare = replace(case.route.tables)
    object.__setattr__(bare, "synthesis", None)
    object.__setattr__(bare, "projection", None)
    with pytest.raises(TranslationError, match="carries neither"):
        replace(case.route, tables=bare)


def test_the_route_prints_its_extents_and_its_certificate(case):
    assert repr(case.route) == (
        "GeneralModalRoute(1 channel, 288 nodes, 1 response, gram 1.3e-06)"
    )
    block = str(case.route)
    assert "identity, defect" in block and "isotropic-kerr" in block
    linear = replace(case.route, responses=None)
    assert "0 responses" in repr(linear)
    assert "none (a linear guide)" in str(linear)


# ---------------------------------------------------------------- the transform


def test_the_general_transform_is_the_tables_and_checked_placeholders(case):
    transform = general_modal_transform(
        list(case.guide.mode_collection), case.route.tables, z_max=GENERAL_Z_END
    )
    assert transform.modal_tables is case.route.tables
    assert transform.n_modes == 1 and transform.n_pol == 1
    assert not transform.mode_averaged
    assert not np.any(transform.synthesis)
    assert transform.synthesis.shape == (1, 1)
    assert transform.taper_scale is None and transform.node_weights is None
    assert float(transform.aeff(0.0)) > 0.0


def test_the_general_transform_needs_the_guides_channels(case):
    with pytest.raises(TranslationError, match="channels in state order"):
        general_modal_transform([], case.route.tables, z_max=GENERAL_Z_END)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"synthesis": np.ones((1, 1))}, "zero placeholders"),
        (
            {
                "synthesis": np.zeros((2, 1)),
                "projection": np.zeros((1, 2)),
                "projection_coarse": np.zeros((1, 2)),
                "node_chunk": 2,
            },
            "zero placeholders",
        ),
        ({"node_chunk": 1, "node_weights": np.ones(1)}, "node_chunk is 1"),
    ],
)
def test_a_general_transform_refuses_a_static_matrix_that_carries_anything(
    case, change, message
):
    transform = general_modal_transform(
        list(case.guide.mode_collection), case.route.tables, z_max=GENERAL_Z_END
    )
    with pytest.raises(TranslationError, match=message):
        replace(transform, **change)


# ------------------------------------------------------------------- assembly


def test_the_assembled_problem_carries_the_three_leaves_and_validates(case):
    problem = problem_of(case)
    assert problem.transform.modal_tables is case.route.tables
    assert problem.linear.modal_metric is not None
    assert problem.linear.modal_metric.kind == "identity"
    assert problem.responses.regional_responses is case.route.responses
    assert problem.linear.coupled_flow is None
    assert problem.validate() is None
    assert not problem.flags.mode_averaged


def test_the_power_weight_of_a_poynting_basis_is_the_constant_one(case):
    """No power conversion is needed: the amplitude is already true-power."""
    problem = problem_of(case)
    assert problem.linear.power_weight_representation == "constant"
    weight = problem.linear.power_weight_constant
    assert weight is not None and np.all(weight == 1.0)


def test_the_prefactor_is_the_crop_scale_inside_the_band_and_zero_outside(case):
    """The -i omega is the record's own factor; what is left here is the grid."""
    problem = problem_of(case)
    assert problem.responses.prefactor_representation == "constant"
    prefactor = problem.responses.prefactor_constant
    assert prefactor is not None
    active = case.grid.active_mask
    assert np.all(prefactor[..., active] == case.grid.crop_scale)
    assert np.all(prefactor[..., ~active] == 0.0)
    assert np.all(np.imag(prefactor) == 0.0)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"quadrature": quadrature_rule("polar", full=False, n_r=4)},
            "quadrature belongs",
        ),
        ({"components": ("x", "y")}, "components belongs"),
        ({"density": 1.0}, "density belongs"),
        ({"prefactor_index": 1.0}, "prefactor_index belongs"),
        ({"shock": False}, "shock=False"),
    ],
)
def test_a_legacy_knob_beside_a_general_route_is_refused_by_name(
    case, override, message
):
    with pytest.raises(TranslationError, match=message):
        problem_of(case, **override)


def test_a_declared_response_beside_a_general_route_is_refused(case):
    state = np.zeros((1, 1, case.grid.n_omega), dtype=complex)
    state[0, 0, 30] = 1.0
    with pytest.raises(TranslationError, match="declared response"):
        assemble(
            case.grid,
            list(case.guide.mode_collection),
            [KerrResponse(chi3=GENERAL_CHI3)],
            state,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[GENERAL_Z_END],
            dz_max=GENERAL_Z_END / 10,
            dz_init=GENERAL_Z_END / 20,
            general=case.route,
        )


def test_an_envelope_grid_cannot_carry_a_general_modal_route(case):
    """A leaf is tabulated on the canonical coarse grid."""
    grid = EnvelopeGrid.from_limits(GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15)
    state = np.zeros((1, 1, grid.n_omega), dtype=complex)
    state[0, 0, 3] = 1.0
    with pytest.raises(TranslationError, match="field-grid route"):
        assemble(
            grid,
            list(case.guide.mode_collection),
            [],
            state,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[GENERAL_Z_END],
            dz_max=GENERAL_Z_END / 10,
            dz_init=GENERAL_Z_END / 20,
            general=case.route,
        )


def test_a_state_with_two_polarisations_is_refused(case):
    state = np.zeros((1, 2, case.grid.n_omega), dtype=complex)
    state[0, :, 30] = 1.0
    with pytest.raises(TranslationError, match="keeps n_pol = 1"):
        assemble(
            case.grid,
            list(case.guide.mode_collection),
            [],
            state,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[GENERAL_Z_END],
            dz_max=GENERAL_Z_END / 10,
            dz_init=GENERAL_Z_END / 20,
            general=case.route,
        )


def test_an_active_band_outside_the_guides_support_is_refused(case):
    """Outside its support the reference is zero, so a run must not reach there."""
    narrowed = replace(case.route.tables)
    mask = np.array(case.route.tables.valid)
    # The coarse bins sit at `crop_index` in the fine axis, so cut the support
    # above one that the run's active band reaches.
    cut = int(case.grid.crop_index[np.flatnonzero(case.grid.active_mask)[-1]])
    mask[:, :, cut:] = False
    mask.setflags(write=False)
    object.__setattr__(narrowed, "valid", mask)
    with pytest.raises(TranslationError, match="is not defined"):
        problem_of(case, general=replace(case.route, tables=narrowed))


# --------------------------------------------------------- the compiled response


def test_the_regional_record_is_the_isotropic_tensor_at_its_declared_region():
    tables = regional_responses([1e-22, 2e-22], [0, 2], n_to=8, dt_fine=1e-16)
    assert tables.n_responses == 2 and tables.n_components == 3 and not tables.n_terms
    np.testing.assert_array_equal(tables.response_region, [0, 2])
    assert_allclose(tables.chi3[0, 0], [1e-22, 2e-22], rtol=0.0, atol=0.0)
    assert_allclose(
        tables.instantaneous_tensor[0], isotropic_kerr_tensor(), rtol=0.0, atol=0.0
    )
    assert np.all(tables.temporal_weight == 1e-16)


def test_a_delayed_declaration_is_needed_for_every_region_or_for_none():
    """One record, one term axis: the sequences are checked against each other."""
    with pytest.raises(TranslationError, match="one delayed declaration per region"):
        regional_responses(
            [1e-22, 2e-22],
            [0, 2],
            raman=[SilicaTensorResponse()],
            n_to=8,
            dt_fine=1e-16,
        )


def test_a_susceptibility_without_a_region_is_refused():
    with pytest.raises(TranslationError, match="one chi3 per region"):
        regional_responses([1e-22, 2e-22], [0], n_to=8, dt_fine=1e-16)


def test_the_orthonormal_metric_records_the_defect_it_was_given():
    metric = orthonormal_metric(
        np.linspace(0.0, 2e15, 4),
        np.array([0.0, 1.0]),
        defect=3e-7,
        gauge_id="lab",
        certificate_id="sampled-gram",
    )
    assert metric.kind == "identity"
    assert metric.error.shape == (1, 4)
    assert np.all(metric.error == 3e-7)


def test_a_snapshot_table_evaluated_off_its_position_is_the_same_solve(case):
    """The record spans the run because the geometry does not move."""
    tables = case.route.tables
    np.testing.assert_array_equal(tables.electric[0], tables.electric[1])
    assert tables.frozen_rule
    assert float(tables.z_edges[-1]) >= GENERAL_Z_END
    with pytest.raises(ModeError, match="does not extrapolate"):
        case.table.evaluate(float(case.grid.omega0), 1e-6)


# ------------------------------------- the constructor a run reaches the route by


def test_the_route_constructor_builds_what_the_fixture_builds(case):
    """`general_modal_route` is one guide's leaves on the run's canonical grids."""
    route = case.route
    assert route.responses is not None
    assert route.responses.n_responses == 1
    assert route.responses.response_region.tolist() == [0]
    assert_allclose(
        route.responses.chi3.reshape(-1), [GENERAL_CHI3], rtol=0.0, atol=0.0
    )
    assert route.gram_defect == case.table.gram_defect
    assert_allclose(route.tables.omega, case.grid.omega, rtol=0.0, atol=0.0)
    assert_allclose(route.tables.omega_fine, case.grid.omega_fine, rtol=0.0, atol=0.0)
    # The profile error the record carries is the guide's own measurement, not a
    # number this call invented.
    sampling = case.guide.sampling
    assert sampling is not None
    assert sampling.profile_error > 0.0
    assert_allclose(
        route.tables.profile_error, sampling.profile_error, rtol=0.0, atol=0.0
    )


def test_a_region_declaring_n2_is_converted_at_the_reference_frequency():
    """`n2` becomes an absolute `chi3` against the region's own index."""
    grid = general_grid()
    guide = general_guide(grid=grid, chi3=None)
    core = guide.regions[0]
    declared = MaterialKerr(n2=2.6e-20)
    regions = (dataclasses.replace(core, responses=(declared,)), guide.regions[1])
    provider = guide.field_provider
    assert isinstance(provider, StepIndexProvider)
    guide = dataclasses.replace(
        guide,
        regions=regions,
        field_provider=dataclasses.replace(provider, regions=regions),
    )
    route = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
    )
    assert route.responses is not None
    assert_allclose(
        route.responses.chi3.reshape(-1),
        [declared.susceptibility(core.material, float(grid.omega0))],
        rtol=0.0,
        atol=0.0,
    )


DISPERSIVE_CORE = bk7()
"""A real dispersive core law for the conversion gate's fixture.

The step-index provider refuses an absorbing material, so fused silica's own law
cannot be the core of a solvable strand here; N-BK7 is extinction-free over the
whole band and its permittivity spans 1.82 % of itself across this grid's active
band, above the route's one per cent.
"""


def test_a_dispersive_region_refuses_an_n2_with_no_declared_wavelength():
    """The default conversion frequency is a guess when the index moves.

    A region's compiled leaf carries one `chi3`, so a declared `n2` is converted at
    one frequency. With a constant index the choice cannot matter and is allowed;
    with a law that moves across a run's factor-two active band it changes the
    executed coefficient by more than `N2_CONVERSION_RTOL`, and the route refuses
    rather than picking the run's carrier on the caller's behalf. Declaring `at=` --
    the wavelength the number was measured at -- is the caller saying where, and it
    is admitted.
    """
    grid = general_grid()
    indices = (DISPERSIVE_CORE, 1.0)
    # A dispersive core needs more knots than the constant-index fixture to
    # keep the table inside `PROVIDER_BETA_RTOL` at the held-out frequencies.
    knots = 17
    with pytest.raises(TranslationError, match="no measurement wavelength"):
        general_modal_route(
            general_guide(
                grid=grid,
                kerr=MaterialKerr(n2=2.6e-20),
                indices=indices,
                knots=knots,
            ),
            grid,
            z_max=GENERAL_Z_END,
            omega_reference=grid.omega0,
        )
    declared = MaterialKerr(n2=2.6e-20, at=1550e-9)
    route = general_modal_route(
        general_guide(grid=grid, kerr=declared, indices=indices, knots=knots),
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
    )
    assert route.responses is not None
    assert_allclose(
        route.responses.chi3.reshape(-1),
        [declared.susceptibility(IndexMaterial(DISPERSIVE_CORE), float(grid.omega0))],
        rtol=0.0,
        atol=0.0,
    )
    # A constant index moves by nothing, so the undeclared default is admitted.
    flat = general_guide(grid=grid, kerr=MaterialKerr(n2=2.6e-20))
    assert (
        general_modal_route(
            flat, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
        ).responses
        is not None
    )
    assert general.N2_CONVERSION_RTOL == 0.01


def test_a_declared_wavelength_outside_the_regions_index_model_is_refused():
    """`at=` is where the region's own index is read, so it has to be readable.

    The conversion `chi3 = 4 eps0 c n_0^2 n2 / 3` reads the **region's** index at
    the declared wavelength; N-BK7's law is quoted over a finite window and a
    number measured at 100 nm has no index there. The route refuses it naming the
    region and both intervals, rather than letting the material's own generic
    refusal surface from the middle of the leaf compile.
    """
    grid = general_grid()
    guide = general_guide(
        grid=grid,
        kerr=MaterialKerr(n2=2.6e-20, at=100e-9),
        indices=(DISPERSIVE_CORE, 1.0),
        knots=17,
    )
    with pytest.raises(TranslationError, match="own index model is valid only over"):
        general_modal_route(
            guide, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
        )


def test_the_real_cubic_is_refused_on_a_grid_that_folds_its_third_harmonic():
    """`n_to >= 2 n_t`, which is `ffac=6`, or the harmonic folds into the band.

    Both axes share one bin spacing, so a cubic of a field filling the coarse band
    reaches `3 n_t / 2` bins and folds about the fine Nyquist to `n_to - 3 n_t / 2`;
    that is at or above the coarse top exactly when `n_to >= 2 n_t`. A field grid built
    with `ffac=4` has no oversampling at all on this window, so the refusal fires, and
    the analytic-signal form -- which generates no harmonic -- is admitted on the same
    grid.
    """
    narrow = FieldGrid.from_limits(
        GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15, ffac=4.0
    )
    assert narrow.n_to < 2 * narrow.n_t
    guide = general_guide(grid=narrow)
    with pytest.raises(TranslationError, match="the harmonic folds about the fine"):
        general_modal_route(
            guide, narrow, z_max=GENERAL_Z_END, omega_reference=narrow.omega0
        )
    route = general_modal_route(
        guide,
        narrow,
        z_max=GENERAL_Z_END,
        omega_reference=narrow.omega0,
        kerr_form="analytic",
    )
    assert route.responses is not None
    assert route.responses.model_id == "isotropic-kerr"
    # A linear guide has no cubic at all, so the room is not asked about.
    assert (
        general_modal_route(
            general_guide(grid=narrow, chi3=None),
            narrow,
            z_max=GENERAL_Z_END,
            omega_reference=narrow.omega0,
        ).responses
        is None
    )


def test_an_unknown_kerr_form_is_refused_by_name():
    """The vocabulary is two words, and a third is not evaluated as one of them.

    Past the test session's runtime hook, which rejects the annotation before the
    body is reached; in production no hook runs and this check is the only one.
    """
    unknown = cast("KerrForm", "quartic")
    with pytest.raises(TranslationError, match="got form='quartic'"):
        unchecked(isotropic_kerr_tensor)(unknown)
    # The identity the record carries is built from the same vocabulary and makes
    # the same refusal, rather than the bare KeyError of a mapping lookup.
    with pytest.raises(TranslationError, match="got form='quartic'"):
        unchecked(instantaneous_model_id)(unknown, delayed=False)


def test_a_zero_n2_is_never_ambiguous():
    """Nothing to be wrong about: a vanishing coefficient converts to zero anywhere."""
    grid = general_grid()
    route = general_modal_route(
        general_guide(
            grid=grid, kerr=MaterialKerr(n2=0.0), indices=(bk7(), 1.0), knots=17
        ),
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
    )
    assert route.responses is not None
    assert float(route.responses.chi3.reshape(-1)[0]) == 0.0


def test_a_guide_whose_regions_are_all_linear_compiles_no_source():
    grid = general_grid()
    route = general_modal_route(
        general_guide(grid=grid, chi3=None),
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
    )
    assert route.responses is None


def test_a_guide_built_from_constructed_modes_cannot_be_propagated():
    """No sampling, no reference: `from_step_index` declares no resolution."""
    grid = general_grid()
    guide = ModalGuide.from_step_index(
        [StepIndexMode(1.5e-6, core=1.45, cladding=1.0, profile_wavelength=835e-9)],
        core=IndexMaterial(1.45),
        cladding=IndexMaterial(1.0),
    )
    assert guide.sampling is None
    with pytest.raises(TranslationError, match="declares no sampling"):
        general_modal_route(
            guide, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
        )


def test_the_route_constructor_refuses_an_envelope_grid(case):
    envelope = EnvelopeGrid.from_limits(GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15)
    with pytest.raises(TranslationError, match="propagates on a field grid"):
        general_modal_route(
            case.guide,
            envelope,
            z_max=GENERAL_Z_END,
            omega_reference=envelope.omega0,
        )


def test_a_guide_narrower_than_the_active_band_is_refused_with_both_bands():
    """The >=2x rule, with the measured example in the message."""
    grid = legacy_grid()
    # A guide solved over the run's declared window alone, which is the mistake
    # the message is written for: half the frequencies the run will reach.
    declared = grid.omega[grid.inband_mask]
    narrow = general_guide(
        grid=SimpleNamespace(
            omega=grid.omega, active_mask=grid.inband_mask, omega0=grid.omega0
        )
    )
    assert narrow.sampling is not None
    assert float(narrow.sampling.omega[0]) > float(declared[0]) * 0.98
    with pytest.raises(TranslationError) as raised:
        general_modal_route(
            narrow, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
        )
    message = str(raised.value)
    assert "a factor 2.19 in frequency" in message
    assert "755.9 to 1655.9 nm" in message
    assert "twice the band its run declares" in message


def test_noise_is_refused_in_a_basis_it_is_not_defined_in(case):
    """The one-photon-per-channel noise draw is defined only in an orthogonal basis.

    Past the constructor, which already pairs a diagonal metric with a diagonal
    array: what this guards is the one line that reads `kind`.
    """
    metric = case.route.metric
    spoiled = replace(metric)
    object.__setattr__(spoiled, "kind", "diagonal")
    assert check_noise_basis(spoiled, has_noise=False) is None
    assert check_noise_basis(None, has_noise=True) is None
    assert check_noise_basis(metric, has_noise=True) is None
    with pytest.raises(TranslationError, match="one photon per channel per bin"):
        check_noise_basis(spoiled, has_noise=True)


def test_the_assembly_logs_its_summary_at_debug(case, caplog):
    """The translation summary reaches a log, which is what it is for."""
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.problem"):
        problem_of(case)
    assert any(
        "route general modal guide" in record.getMessage() for record in caplog.records
    )


def test_the_debug_summary_carries_the_certificates_nothing_else_reads(case):
    """The Gram defect and the profile error reach a log."""
    text = debug_summary(problem_of(case))
    assert "route general modal guide" in text
    assert f"defect {case.route.gram_defect:.3e}" in text
    assert "certificate sampled-gram" in text
    sampling = case.guide.sampling
    assert sampling is not None
    assert f"profile error: {sampling.profile_error:.3e}" in text
    assert "guide support: 755.9 to 1655.9 nm" in text
    assert "region source: 1 region(s), isotropic-kerr-real, 3 components" in text
    assert "array(" not in text


def test_the_translation_layer_records_the_accessor_list_without_importing_output():
    """The accessor sets live in `solprop.stats`, beside the catalogue they select.

    `modal_identity` publishes the list a general modal file carries, and the layers of
    `docs/dev/architecture.md` section 3 put translation below output: reading the
    list from `solprop.stats` is what lets it do that without a translation module
    importing an output one.
    """
    source = Path(general.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "solprop.stats" in imported
    assert not [one for one in imported if one.startswith("solprop.output")]


# ------------------------------------------------ the declared change of basis


def entry(record, *keys):
    """Return a nested entry of a `JsonValue` mapping, past the static checker.

    The alias is a union, so a literal chain of subscripts is a type error on every
    step; the runtime shape is a mapping and the tests below assert what it holds.
    """
    value = record
    for key in keys:
        value = cast("dict", value)[key]
    return value


@pytest.fixture(scope="module")
def circular_route():
    """A two-channel guide declared in the circular basis, and its route."""
    grid = general_grid()
    guide = general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        circular=(0, 1),
        grid=grid,
    )
    route = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )
    return SimpleNamespace(grid=grid, guide=guide, route=route)


def test_a_declared_basis_gives_the_route_a_dense_metric_on_its_own_grids(
    circular_route,
):
    """`W' = U.H W U` is stored, not decorated: the leaf is the sampled Gram."""
    route, grid = circular_route.route, circular_route.grid
    metric = route.metric
    assert metric.kind == "dense"
    assert route.metric_tables is metric
    assert metric.gauge_id == route.tables.gauge_id
    assert metric.gauge_id.endswith("+circular(0,1)")
    assert_allclose(metric.omega, route.tables.omega, rtol=0, atol=0)
    assert_allclose(metric.z_edges, route.tables.z_edges, rtol=0, atol=0)
    assert metric.n_modes == 2
    # Outside the guide's band the metric is the identity, not a zero block: a
    # singular metric certifies no positive error or noise norm, and nothing lives
    # there anyway.
    block = metric_at(metric, 0.0)
    assert block is not None
    sampling = circular_route.guide.sampling
    assert sampling is not None
    outside = np.flatnonzero(grid.omega < sampling.omega_span[0])
    assert outside.size
    assert_allclose(block[:, :, outside[-1]], np.eye(2), rtol=0, atol=0)
    # And it is the measured Gram inside it.
    inside = int(np.argmin(abs(grid.omega - grid.omega0)))
    assert 0.0 < metric_departure(block[:, :, inside : inside + 1]) < DEFAULT_GRAM_ATOL


def test_a_guide_in_the_provider_s_own_coordinates_keeps_the_identity_metric():
    case = general_case()
    assert case.route.metric_tables is None
    assert case.route.metric.kind == "identity"


def test_the_route_refuses_a_metric_that_is_not_its_own(circular_route):
    route = circular_route.route
    metric = route.metric
    changes = [
        (
            {
                "metric_tables": orthonormal_metric(
                    route.tables.omega,
                    route.tables.z_edges,
                    defect=0.0,
                    gauge_id=route.tables.gauge_id,
                    certificate_id="sampled-gram",
                )
            },
            "is the dense metric of a declared change of basis",
        ),
        (
            {"metric_tables": dataclasses.replace(metric, gauge_id="elsewhere")},
            "describes the basis its reference is in",
        ),
    ]
    for override, message in changes:
        with pytest.raises(TranslationError, match=message):
            dataclasses.replace(route, **override)


def test_the_route_refuses_a_metric_on_another_grid_or_another_width(circular_route):
    route = circular_route.route
    metric = route.metric
    assert metric.dense is not None
    shifted = dense_metric(
        route.tables.omega + 1.0,
        route.tables.z_edges,
        metric.dense[0, 0],
        defect=0.0,
        gauge_id=route.tables.gauge_id,
        certificate_id="sampled-gram",
    )
    with pytest.raises(TranslationError, match="own omega"):
        dataclasses.replace(route, metric_tables=shifted)
    narrow = dense_metric(
        route.tables.omega,
        route.tables.z_edges,
        np.broadcast_to(
            np.eye(1, dtype=complex)[:, :, None], (1, 1, len(route.tables.omega))
        ).copy(),
        defect=0.0,
        gauge_id=route.tables.gauge_id,
        certificate_id="sampled-gram",
    )
    with pytest.raises(TranslationError, match="describes 1 channel"):
        dataclasses.replace(route, metric_tables=narrow)


def test_the_route_refuses_a_basis_too_far_from_orthonormal(circular_route):
    """The nonunitary refusal, at the translation layer, on the stored block."""
    route = circular_route.route
    metric = route.metric
    assert metric.dense is not None
    skewed = dense_metric(
        route.tables.omega,
        route.tables.z_edges,
        metric.dense[0, 0] * 1.5,
        defect=0.0,
        gauge_id=route.tables.gauge_id,
        certificate_id="sampled-gram",
    )
    with pytest.raises(TranslationError, match="from orthonormal over the run"):
        dataclasses.replace(route, metric_tables=skewed)


def test_the_identity_records_the_basis_by_four_independent_paths(circular_route):
    """A file in one basis cannot resume in another, and each path names itself."""
    route, grid = circular_route.route, circular_route.grid
    labels = list(circular_route.guide.mode_collection.channel_labels)
    declared = modal_identity(route, grid, channels=labels, has_noise=True)
    plain_case = general_case(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),)
    )
    plain = modal_identity(
        plain_case.route,
        plain_case.grid,
        channels=list(plain_case.guide.mode_collection.channel_labels),
        has_noise=True,
    )
    assert entry(declared, "metric", "kind") == "dense"
    assert entry(plain, "metric", "kind") == "identity"
    assert entry(declared, "metric", "gauge_id") != entry(plain, "metric", "gauge_id")
    assert entry(declared, "tables", "gauge_id") != entry(plain, "tables", "gauge_id")
    for record, expected in ((declared, ["left", "right"]), (plain, [None, None])):
        rows = cast("list[dict[str, object]]", record["channels"])
        assert [one["handedness"] for one in rows] == expected
    assert "basis" in cast("dict", entry(declared, "guide", "sampling"))
    assert "basis" not in cast("dict", entry(plain, "guide", "sampling"))
    # The draw is recorded as made in the reference basis whatever basis the run
    # executes in, because that is where it was made.
    assert declared["noise_reference_basis"] == "orthonormal true power"
    assert plain["noise_reference_basis"] == "orthonormal true power"


def test_a_declared_basis_assembles_and_the_noise_check_admits_it(circular_route):
    route, grid = circular_route.route, circular_route.grid
    check_noise_basis(route.metric, has_noise=True)
    with pytest.raises(TranslationError, match="basis with unequal channel norms"):
        check_noise_basis(
            dataclasses.replace(
                route.metric,
                kind="diagonal",
                dense=None,
                diagonal=np.ones((1, 1, 2, len(route.tables.omega))),
            ),
            has_noise=True,
        )
    state = np.zeros((2, 1, grid.n_omega), dtype=complex)
    state[1, 0, int(np.argmin(np.abs(grid.omega - grid.omega0)))] = 1.0
    problem = assemble(
        grid,
        list(circular_route.guide.mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[GENERAL_Z_END],
        dz_max=GENERAL_Z_END / 10,
        dz_init=GENERAL_Z_END / 20,
        general=route,
    )
    assert problem.linear.modal_metric is not None
    assert problem.linear.modal_metric.kind == "dense"
    assert problem.linear.modal_metric.n_modes == 2

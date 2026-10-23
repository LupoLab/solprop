# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The flat seam record of a general guide, against the table it is built from."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from bounds import EPS, assert_within
from modal_provider_cases import OMEGA, make_provider
from solprop.exceptions import ModeError
from solprop.modes.guide import StepIndexProvider
from solprop.modes.material import MaterialRegion
from solprop.profiles import Linear
from solprop.translate.modal_tables import (
    DenseModalTable,
    ModalTransformTables,
    modal_transform_tables,
    project_transverse_field,
    sample_modal_table,
)


@pytest.fixture(scope="module")
def reference():
    """Return a two-channel dense table and the flat record built from it."""
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=24)
    knots = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    table = sample_modal_table(
        provider, knots, np.array([0.0]), nodes, degenerate_blocks=((0, 1),)
    )
    # Offset so that no stored frequency lands on a knot: every point is one
    # the table interpolates.
    coarse = np.linspace(0.9951 * OMEGA, 1.0043 * OMEGA, 8)
    return table, coarse, modal_transform_tables(table, coarse, profile_error=1e-9)


def test_the_record_is_the_table_evaluated_on_the_run_grid(reference):
    table, coarse, record = reference
    # Every stored frequency is between two knots: the record is a resampling of
    # the table, and stores exactly what evaluate returns there (a structural
    # contract: the record holds those results, it computes no others).
    assert not set(coarse.tolist()) & set(table.omega.tolist())
    for index, frequency in enumerate(coarse):
        sample = table.evaluate(float(frequency), 0.0)
        for name in ("electric", "magnetic", "backward_electric", "backward_magnetic"):
            np.testing.assert_array_equal(
                getattr(record, name)[0, :, :, :, index], getattr(sample, name)
            )
        np.testing.assert_array_equal(
            record.reciprocal_gram[0, :, :, index], sample.reciprocal_gram
        )
    np.testing.assert_array_equal(record.source_frequency_factor, -1j * coarse)
    np.testing.assert_array_equal(record.omega, coarse)
    np.testing.assert_array_equal(record.omega_fine, coarse)
    np.testing.assert_array_equal(record.z_edges, table.z)
    np.testing.assert_array_equal(record.region_index, table.nodes.region_index)
    np.testing.assert_array_equal(record.weights[0], table.nodes.weights)
    np.testing.assert_array_equal(record.node_xy[0], table.nodes.xy)
    assert record.valid.shape == (1, 2, len(coarse)) and np.all(record.valid)
    assert record.factors is None
    assert record.model_id == table.samples[0][0].physical_id
    assert record.policy_id == "dense-knots-cubic-frequency"


def test_the_stored_grid_carries_the_tables_own_frequency_behaviour(reference):
    table, coarse, record = reference
    # A spline through the record's seven stored frequencies, evaluated where it
    # stored nothing, against the table's own interpolant there. Predeclared 1e-6
    # relative: the record resamples a five-knot cubic on a denser grid over the
    # same band, so the resampling error is far below the table's own.
    bound = 1e-6
    middle = float((coarse[3] + coarse[4]) / 2)
    weight = np.sqrt(table.nodes.weights)[:, None, None]
    for name in ("electric", "magnetic"):
        stored = np.moveaxis(getattr(record, name)[0], -1, 0)
        predicted = CubicSpline(coarse, stored, axis=0)(middle)
        direct = getattr(table.evaluate(middle, 0.0), name)
        assert_allclose(
            np.linalg.norm(weight * (predicted - direct))
            / np.linalg.norm(weight * direct),
            0.0,
            rtol=0,
            atol=bound,
        )


def test_the_coefficient_dual_recovers_the_amplitudes_it_synthesised(reference):
    _, coarse, record = reference
    # The dual is the inverse of the synthesis it is built beside: push a declared
    # amplitude vector through the stored E and H, contract the dual against the
    # six physical components, and the vector comes back. Predeclared 1e-12
    # absolute on amplitudes of order one: the inverse of a 2x2 metric at unit
    # condition and two quadrature sums.
    bound = 1e-12
    amplitudes = np.array([0.3 + 0.4j, -0.2 + 0.1j])
    for index in range(len(coarse)):
        field = np.concatenate(
            [
                record.electric[0, :, :, :, index] @ amplitudes,
                record.magnetic[0, :, :, :, index] @ amplitudes,
            ],
            axis=1,
        )
        recovered = np.einsum(
            "mqc,qc->m", record.coordinate_dual[0, :, :, :, index], field
        )
        assert_allclose(recovered, amplitudes, rtol=0, atol=bound)
    # It pairs with E and H, in that order, and the longitudinal columns are zero:
    # only the transverse components carry longitudinal flux.
    assert np.all(record.coordinate_dual[:, :, :, 2] == 0)
    assert np.all(record.coordinate_dual[:, :, :, 5] == 0)


def test_the_record_prints_and_names_its_extents(reference):
    _, _, record = reference
    assert repr(record) == (
        "ModalTransformTables(2 channels, 432 nodes, 8 frequencies, 1 z)"
    )
    assert "dense reference" in str(record)


def test_a_fine_grid_is_marked_valid_only_where_the_table_reaches(reference):
    """The fine grid owns the support mask, and neither execution map.

    Both maps are tabulated on the **coarse** grid (contract section 4): the state is a
    coarse spectrum, so a column of S that no coarse bin reaches could never be read.
    Only `valid` is as long as the fine grid.
    """
    table, coarse, _ = reference
    fine = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 9)
    record = modal_transform_tables(table, coarse, profile_error=1e-9, omega_fine=fine)
    inside = (table.omega[0] <= fine) & (fine <= table.omega[-1])
    np.testing.assert_array_equal(record.valid[0, 0], inside)
    assert not np.all(inside)
    assert record.valid.shape == (1, 2, len(fine))
    assert len(fine) != len(coarse)
    assert record.synthesis is not None and record.projection is not None
    assert record.synthesis.shape[-1] == len(coarse)
    assert record.projection.shape[-1] == len(coarse)


def test_a_record_without_reciprocal_partners_leaves_those_leaves_empty(reference):
    # A provider that certifies no reciprocal basis leaves the three optional
    # leaves empty together, and the record is still complete: a nonlinear
    # translation may need them, but the reference itself stands.
    table, coarse, _ = reference
    bare = DenseModalTable(
        table.omega,
        table.z,
        tuple(
            tuple(
                replace(
                    sample,
                    backward_electric=None,
                    backward_magnetic=None,
                    reciprocal_gram=None,
                )
                for sample in row
            )
            for row in table.samples
        ),
        table.nodes,
        table.degenerate_blocks,
    )
    record = modal_transform_tables(bare, coarse, profile_error=1e-9)
    assert record.backward_electric is None
    assert record.backward_magnetic is None
    assert record.reciprocal_gram is None
    assert record.coordinate_dual is not None
    assert "reciprocal:    no" in str(record)


def test_a_grid_outside_the_table_is_stored_as_an_explicit_zero(reference):
    """A canonical grid leaves the band; the record is zero there, never evaluated.

    A field grid starts at zero and runs past any guide's support, so the leaf a
    problem carries necessarily covers frequencies the table was never solved at.
    Contract section 4 requires an explicit zero there rather than an extrapolated
    profile, and the `valid` mask says which fine bins the table reached.
    """
    table, _, _ = reference
    wide = np.linspace(0.9 * OMEGA, 1.1 * OMEGA, 9)
    record = modal_transform_tables(table, wide, profile_error=1e-9)
    inside = (table.omega[0] <= wide) & (wide <= table.omega[-1])
    assert inside.any() and not inside.all()
    for name in ("electric", "magnetic", "coordinate_dual", "reciprocal_gram"):
        stored = getattr(record, name)
        assert np.all(stored[..., ~inside] == 0.0)
        assert np.any(stored[..., inside] != 0.0)
    assert record.projection is not None and record.synthesis is not None
    assert np.all(record.projection[..., ~inside] == 0.0)
    assert np.all(record.synthesis[..., ~inside] == 0.0)
    np.testing.assert_array_equal(record.valid[0, 0], inside)
    # Inside the band it is still the table's own evaluation, stored exactly.
    column = int(np.flatnonzero(inside)[0])
    sample = table.evaluate(float(wide[column]), 0.0)
    np.testing.assert_array_equal(record.electric[0, :, :, :, column], sample.electric)


def test_the_maps_are_the_reference_fields_and_the_reciprocal_projection(reference):
    """S is the stored profile; Q carries W^-1, the weights and the 1/4.

    This is a **wiring** test and not a physical one: it asserts that the constructor
    puts the fields and the projection functional of `_projection_map` in the axes the
    record declares, in the order a backend reads them. What the projection *is* --
    the source functional of a bound reciprocal basis,
    `da_m/dz = -i omega [W^-1]_mn int(e_n* . P) dA / 4`, with the `-i omega` held
    separately in `source_frequency_factor` -- is derived in
    `docs/guide/waveguides/general-modal.md` and measured against a hand-written
    quadrature in
    `tests/backends/test_general_modal.py`; here it is only checked to be where the
    contract says it is.
    """
    table, coarse, record = reference
    assert record.synthesis is not None and record.projection is not None
    assert record.synthesis.shape == (1, 1, 432, 3, 2, len(coarse))
    assert record.projection.shape == (1, 1, 2, 432, 3, len(coarse))
    for index in range(len(coarse)):
        sample = table.evaluate(float(coarse[index]), 0.0)
        # S is the stored electric profile on the coarse grid, unweighted.
        np.testing.assert_array_equal(
            record.synthesis[0, 0, :, :, :, index], sample.electric
        )
        expected = (
            np.einsum(
                "mn,qcn->mqc", np.linalg.inv(sample.flux_metric), sample.electric.conj()
            )
            * table.nodes.weights[None, :, None]
            / 4
        )
        # The same inverse-metric contraction formed by the test: a 2 x 2 inverse
        # and a two-term sum per element, held normwise to 16 eps.
        assert_within(
            record.projection[0, 0, :, :, :, index], expected, 16 * EPS, what="P"
        )


def test_a_snapshot_is_broadcast_across_the_positions_it_is_asked_for(reference):
    """A constant guide has one solve, and the record spans the run with it.

    The seam requires a leaf's positions to contain the run, and a snapshot table
    has no z interpolant at all, so the one solve is stored at every position it is
    asked for. One Chebyshev coefficient per interval is then exact.
    """
    table, coarse, _ = reference
    record = modal_transform_tables(
        table, coarse, profile_error=1e-9, z=[0.0, 1e-3, 2e-3]
    )
    assert record.synthesis is not None and record.projection is not None
    assert record.synthesis.shape[:2] == (2, 1)
    assert record.frozen_rule
    np.testing.assert_array_equal(record.electric[0], record.electric[2])
    np.testing.assert_array_equal(record.projection[0], record.projection[1])
    np.testing.assert_array_equal(record.z_edges, [0.0, 1e-3, 2e-3])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"factors": object()}, "factors must be None"),
        ({"gauge_id": ""}, "must be non-empty"),
        ({"model_id": ""}, "must be non-empty"),
        ({"policy_id": ""}, "must be non-empty"),
        ({"omega": np.array([2e15, 1e15, 3e15, 4e15])}, "increasing vector"),
        ({"z_edges": np.array([])}, "increasing vector"),
        ({"omega_fine": np.array([-1.0, 1e15, 2e15])}, "must be non-negative"),
        ({"electric": np.zeros((1, 4, 3, 2, 3), complex)}, "one frequency per"),
        (
            {"magnetic": np.zeros((1, 4, 3, 2, 8), complex)},
            "magnetic must have shape",
        ),
        ({"reciprocal_gram": None}, "supplied together"),
        ({"profile_error": np.full((1, 8), -1.0)}, "non-negative"),
        ({"region_index": np.full(432, -1, np.int32)}, "non-negative"),
        ({"synthesis": None}, "supplied together"),
        ({"synthesis": np.zeros((1, 1, 1, 1, 1, 1), complex)}, "synthesis must have"),
        ({"synthesis": np.zeros((2, 1, 1, 1, 1, 1), complex)}, "must carry 1 interval"),
        ({"projection": np.zeros((1, 1, 1, 1, 1, 1), complex)}, "projection must have"),
    ],
)
def test_the_record_refuses_what_it_cannot_carry(reference, change, message):
    _, _, record = reference
    with pytest.raises(ModeError, match=message):
        replace(record, **change)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("electric", np.zeros((1, 4, 2, 2, 8), complex), "electric must have shape"),
        ("electric", np.zeros((4, 3, 2, 8), complex), "electric must have shape"),
        ("valid", np.ones((1, 2, 3), bool), "valid must have shape"),
    ],
)
def test_the_record_refuses_an_extent_the_runtime_hook_would_have_caught(
    reference, name, value, message
):
    # Set past the constructor, as tests/modes/test_stepindex.py does for its own
    # fixed extents: the session's runtime hook refuses these shapes before
    # __post_init__ sees them, and the check has to stay reachable in production,
    # where no hook runs.
    _, _, record = reference
    copy = replace(record)
    object.__setattr__(copy, name, value)
    with pytest.raises(ModeError, match=message):
        copy.__post_init__()


def test_the_record_owns_its_arrays_instead_of_freezing_the_callers(reference):
    # np.ascontiguousarray returns the caller's own object when it is already
    # contiguous, so the record used to freeze the caller's array and could still
    # be rewritten through a view the caller already held. Both ends are checked.
    _, _, record = reference
    frequencies = np.array(record.omega, dtype=float)
    weights = np.array(record.weights, dtype=float)
    view = weights[0]
    rebuilt = replace(record, omega=frequencies, weights=weights)
    assert frequencies.flags.writeable and weights.flags.writeable
    assert rebuilt.omega is not frequencies and rebuilt.weights is not weights
    assert not rebuilt.omega.flags.writeable and not rebuilt.weights.flags.writeable
    view *= 2.0
    frequencies += 1.0
    np.testing.assert_array_equal(rebuilt.weights, record.weights)
    np.testing.assert_array_equal(rebuilt.omega, record.omega)


def test_the_record_refuses_a_weight_that_is_not_an_area(reference):
    _, _, record = reference
    with pytest.raises(ModeError, match="weights must be positive"):
        replace(record, weights=np.zeros_like(record.weights))
    with pytest.raises(ModeError, match="must be finite"):
        replace(record, node_xy=np.full_like(record.node_xy, np.nan))


def test_a_snapshot_declares_one_interval_and_a_path_declares_its_own(reference):
    _, coarse, record = reference
    assert record.profile_error.shape == (1, len(coarse))
    assert isinstance(record, ModalTransformTables)
    with pytest.raises(ModeError, match="shape"):
        replace(record, profile_error=np.zeros((2, len(coarse))))


@pytest.mark.medium
def test_a_record_over_a_path_stores_every_position_it_was_asked_for():
    # An interval per pair of stored positions once there is a path, while the
    # dense arrays stay indexed by the positions themselves.
    original = make_provider(radius=250e-9)
    path = Linear(250e-9, 250.0025e-9, 1.0)
    provider = StepIndexProvider(
        (replace(original.modes[0], radius=path),),
        (
            MaterialRegion("core", 0.0, path, original.regions[0].material),
            MaterialRegion("cladding", path, None, original.regions[1].material),
        ),
    )
    nodes = provider.node_rule(OMEGA, n_r=24)
    knots = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    positions = np.array([0.0, 0.25, 0.5, 1.0])
    table = sample_modal_table(provider, knots, positions, nodes)
    coarse = np.linspace(0.995 * OMEGA, 1.004 * OMEGA, 4)
    stored = np.array([0.1, 0.6, 0.9])
    record = modal_transform_tables(
        table, coarse, profile_error=1e-6, z=stored, policy_id="tapered"
    )
    np.testing.assert_array_equal(record.z_edges, stored)
    assert record.electric.shape[0] == len(stored)
    assert record.profile_error.shape == (len(stored) - 1, len(coarse))
    assert record.policy_id == "tapered"
    for index, position in enumerate(stored):
        sample = table.evaluate(float(coarse[2]), float(position))
        np.testing.assert_array_equal(
            record.electric[index, :, :, :, 2], sample.electric
        )
        np.testing.assert_array_equal(record.node_xy[index], table.nodes.xy)
    assert "3," in str(record)


def test_the_records_identity_is_scalars_and_tags_and_no_array(reference):
    """Resume compares this mapping, not megabytes of E and H field by field."""
    _, coarse, record = reference
    identity = record.identity
    assert identity["gauge_id"] == record.gauge_id
    assert identity["model_id"] == record.model_id
    assert identity["policy_id"] == record.policy_id
    assert identity["n_modes"] == 2
    assert identity["n_components"] == 3
    assert identity["n_omega"] == len(coarse)
    assert identity["n_omega_fine"] == len(coarse)
    assert identity["n_z"] == 1
    assert identity["n_regions"] == int(np.max(record.region_index)) + 1
    assert identity["omega_span"] == [float(coarse[0]), float(coarse[-1])]
    assert identity["z_span"] == [0.0, 0.0]
    assert identity["has_synthesis"] is True
    assert identity["has_projection"] is True
    assert identity["has_reciprocal"] is True
    assert identity["has_coordinate_dual"] is True
    assert identity["factors_kind"] is None
    assert identity["frozen_rule"] is True
    assert identity["profile_error_max"] == 1e-9
    # The record is eq=False, so nothing compares two of them by value; the
    # identity is what a resume may compare, and it holds no array at all.
    for value in identity.values():
        assert not isinstance(value, np.ndarray)
        assert isinstance(value, str | int | float | bool | list | type(None))
    assert record.identity == identity


def test_the_identity_moves_with_a_declared_id_and_with_an_extent(reference):
    _, coarse, record = reference
    assert replace(record, policy_id="other").identity != record.identity
    shorter = modal_transform_tables(
        reference[0], coarse[:4], profile_error=1e-9
    ).identity
    assert shorter["n_omega"] == 4
    assert shorter != record.identity


def test_one_rule_broadcast_across_positions_is_the_frozen_rule_case(reference):
    """A record over several positions built from one rule says so."""
    _, _, record = reference
    assert record.frozen_rule
    positions = np.array([0.0, 1e-5])
    stacked = replace(
        record,
        z_edges=positions,
        electric=np.repeat(record.electric, 2, axis=0),
        magnetic=np.repeat(record.magnetic, 2, axis=0),
        backward_electric=np.repeat(record.backward_electric, 2, axis=0),
        backward_magnetic=np.repeat(record.backward_magnetic, 2, axis=0),
        reciprocal_gram=np.repeat(record.reciprocal_gram, 2, axis=0),
        coordinate_dual=np.repeat(record.coordinate_dual, 2, axis=0),
        node_xy=np.repeat(record.node_xy, 2, axis=0),
        weights=np.repeat(record.weights, 2, axis=0),
        valid=np.repeat(record.valid, 2, axis=0),
    )
    assert stacked.frozen_rule
    assert stacked.identity["frozen_rule"] is True
    moved_xy = np.array(stacked.node_xy)
    moved_xy[1] *= 1.00001
    assert not replace(stacked, node_xy=moved_xy).frozen_rule
    moved_weights = np.array(stacked.weights)
    moved_weights[1] *= 1.00002
    assert not replace(stacked, weights=moved_weights).frozen_rule


def test_a_projected_launch_refuses_what_it_cannot_describe(reference):
    """The record binds its three arrays to one projection, and its share to [0, inf).

    A record assembled by hand with a captured flux or a residual of another length
    would describe two projections at once, and a negative or non-finite share would
    describe none.
    """
    _, _, record = reference
    good = project_transverse_field(
        record, np.einsum("qcmw,m->qcw", record.electric[0], np.array([1.0, 0.0]))
    )
    for name in ("captured", "residual"):
        with pytest.raises(ModeError, match=f"{name} covers its own frequency axis"):
            replace(good, **{name: getattr(good, name)[:-1]})
    for share in (-1e-12, np.inf, np.nan):
        with pytest.raises(ModeError, match="finite and non-negative"):
            replace(good, residual_share=share)
    # A share above one is **not** refused: on a basis that is not orthonormal the
    # projection can overshoot, and that is a measurement rather than an error.
    assert replace(good, residual_share=1.5).residual_share == 1.5


def test_projecting_a_field_refuses_a_record_or_a_shape_it_cannot_use(reference):
    """The map needs the dual, and the field must lie on the guide's own nodes."""
    _, _, record = reference
    nodes, components = record.electric.shape[1], record.electric.shape[2]
    bins = record.omega.size
    # The right **rank** and the right dtype, so the runtime type hook admits it
    # and the library's own check -- the one that stays reachable in production --
    # is what refuses it.
    with pytest.raises(ModeError, match="lies on this guide's nodes"):
        project_transverse_field(
            record, np.zeros((nodes, components, bins - 1), dtype=complex)
        )
    bare = replace(record)
    object.__setattr__(bare, "coordinate_dual", None)
    with pytest.raises(ModeError, match="carry no coordinate_dual"):
        project_transverse_field(
            bare, np.zeros((nodes, components, bins), dtype=complex)
        )
    # A field that is identically zero is not an error: it declares nothing, so
    # nothing is captured and nothing is missed, and every residual is undefined.
    empty = project_transverse_field(
        record, np.zeros((nodes, components, bins), dtype=complex)
    )
    assert empty.residual_share == 0.0
    assert np.all(np.isnan(empty.residual))
    assert not np.any(empty.amplitudes)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The four optional general-modal leaves of kernel schema 12.

`ModalMetricTables` and `CoupledFlowTables` (contract section 3) and
`RegionalResponseTables` (section 5); `ModalTransformTables` (section 4) has its own
module, `tests/translate/test_modal_transform_tables.py`.

Each record validates its own arrays and each of those checks is exercised here. The
tree walk `validate` performs reaches into them for dtype, finiteness and ownership
but relates nothing to the canonical state, so the last section of this module checks
the four leaves against a real problem's grids, channels and run (contract section 4).
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.exceptions import MaterialRangeWarning, TranslationError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import PIController
from solprop.translate.coupled import (
    CoupledFlowTables,
    ModalMetricTables,
    dense_metric,
    interval_count,
    metric_at,
)
from solprop.translate.modal_tables import ModalTransformTables
from solprop.translate.problem import _walk_arrays, assemble
from solprop.translate.responses import RegionalResponseTables

OMEGA = np.linspace(1.0e15, 2.0e15, 3)
N_OMEGA = len(OMEGA)


def override(builder, name, value):
    """Call one of the builders below with a single field replaced.

    A literal ``**{name: value}`` makes the static checker map the dictionary onto
    every keyword the builder has, including its integer counts, so the one field
    goes through here instead.
    """
    changes = {name: value}
    return builder(**changes)


def past_the_hook(record, **changes):
    """Re-run a record's validation with a value the session's hook would refuse.

    The rank of an array and the members of a ``Literal`` are refused by beartype
    before ``__post_init__`` sees them, and the record's own checks have to stay
    reachable in production, where no hook runs. The pattern is
    ``tests/translate/test_modal_transform_tables.py``'s.
    """
    copy = replace(record)
    for name, value in changes.items():
        object.__setattr__(copy, name, value)
    copy.__post_init__()


def identity_metric(omega=OMEGA, z_edges=(0.0,), **overrides):
    """An orthonormal metric over one frequency axis and one set of positions."""
    axis = np.asarray(omega, dtype=float)
    edges = np.asarray(z_edges, dtype=float)
    rows = (interval_count(len(edges)), len(axis))
    fields = {
        "kind": "identity",
        "omega": axis,
        "z_edges": edges,
        "diagonal": None,
        "dense": None,
        "error": np.zeros(rows),
        "minimum_eigenvalue": np.ones(rows),
        "maximum_eigenvalue": np.ones(rows),
        "gauge_id": "lab",
        "certificate_id": "orthonormal",
    } | overrides
    return ModalMetricTables(**fields)


def diagonal_metric(n_modes=2, n_edges=3, omega=OMEGA, **overrides):
    """A per-channel positive metric over a path of positions."""
    axis = np.asarray(omega, dtype=float)
    intervals = interval_count(n_edges)
    rows = (intervals, len(axis))
    fields = {
        "kind": "diagonal",
        "omega": axis,
        "z_edges": np.linspace(0.0, 1.0, n_edges),
        "diagonal": np.full((intervals, 2, n_modes, len(axis)), 0.5),
        "dense": None,
        "error": np.full(rows, 1e-12),
        "minimum_eigenvalue": np.full(rows, 0.5),
        "maximum_eigenvalue": np.full(rows, 0.5),
        "gauge_id": "lab",
        "certificate_id": "sampled",
    } | overrides
    return ModalMetricTables(**fields)


def dense_flow(n_modes=2, n_edges=2, omega=OMEGA, z_max=1.0, **overrides):
    """A dense generator over one section."""
    axis = np.asarray(omega, dtype=float)
    intervals = interval_count(n_edges)
    rows = (intervals, len(axis))
    fields = {
        "omega": axis,
        "z_edges": np.linspace(0.0, z_max, n_edges),
        "generator": np.zeros(
            (intervals, 2, n_modes, n_modes, len(axis)), dtype=complex
        ),
        "fixed_terms": None,
        "term_weights": None,
        "wall_vectors": None,
        "wall_weights": None,
        "connection": None,
        "generator_error": np.zeros(rows),
        "residual_norm_bound": np.zeros(rows),
        "inverse_condition_bound": np.ones(rows),
        "section_methods": ("constant_schur",) * intervals,
        "gauge_id": "lab",
        "certificate_id": "schur",
    } | overrides
    return CoupledFlowTables(**fields)


def factored_flow(n_modes=1, n_terms=2, **overrides):
    """A one-channel factored generator with a wall pair and a connection."""
    fields = {
        "omega": OMEGA,
        "z_edges": np.array([0.0, 1.0]),
        "generator": None,
        "fixed_terms": np.zeros((n_terms, n_modes, n_modes, 3), dtype=complex),
        "term_weights": np.zeros((1, 2, n_terms), dtype=complex),
        "wall_vectors": np.zeros((1, n_modes, 3), dtype=complex),
        "wall_weights": np.zeros((1, 2, 3), dtype=complex),
        "connection": np.zeros((1, 2, n_modes, n_modes, 3), dtype=complex),
        "generator_error": np.zeros((1, 3)),
        "residual_norm_bound": np.zeros((1, 3)),
        "inverse_condition_bound": np.ones((1, 3)),
        "section_methods": ("frozen_residual",),
        "gauge_id": "lab",
        "certificate_id": "factored",
    } | overrides
    return CoupledFlowTables(**fields)


def kerr_regions(n_responses=1, components=3, n_to=8, **overrides):
    """A Kerr-only regional record over one interval and ``n_to`` fine samples."""
    fields = {
        "response_region": np.arange(n_responses, dtype=np.int32),
        "chi3": np.full((1, 1, n_responses), 2.2e-52),
        "instantaneous_tensor": np.ones((n_responses, *(components,) * 4)),
        "delayed_tensor": np.zeros((n_responses, 0, *(components,) * 4)),
        "delayed_kernel": np.zeros((1, 1, n_responses, 0, 2 * n_to)),
        "kernel_valid_samples": np.zeros((n_responses, 0), dtype=np.int32),
        "temporal_weight": np.full(n_to, 1e-15),
        "model_id": "isotropic-kerr",
        "policy_id": "dense-nodes",
    } | overrides
    return RegionalResponseTables(**fields)


# --------------------------------------------------------------- the metric leaf


def test_an_identity_metric_owns_no_coefficient_array():
    metric = identity_metric()
    assert metric.kind == "identity"
    assert metric.diagonal is None
    assert metric.n_modes == 0
    assert metric.n_omega == 3


def test_a_diagonal_metric_reports_its_channels_and_freezes_its_array():
    metric = diagonal_metric(n_modes=4)
    assert metric.n_modes == 4
    assert metric.diagonal is not None
    assert not metric.diagonal.flags.writeable
    assert not metric.omega.flags.writeable


def test_a_metric_copies_rather_than_adopting_the_caller_s_array():
    weights = np.full((2, 2, 2, 3), 0.5)
    metric = diagonal_metric(diagonal=weights)
    weights[...] = 9.0
    assert metric.diagonal is not None
    assert float(np.max(metric.diagonal)) == 0.5


def dense_block(n_modes=2, n_omega=N_OMEGA, n_intervals=1, n_deg=1):
    """A Hermitian dense metric block, the identity plus a small off-diagonal."""
    single = np.eye(n_modes, dtype=complex) * (1.0 - 1e-6)
    if n_modes > 1:
        single[0, 1] = 1e-7 + 2e-8j
        single[1, 0] = np.conj(single[0, 1])
    return np.broadcast_to(
        single[:, :, None], (n_intervals, n_deg, n_modes, n_modes, n_omega)
    ).copy()


def test_a_dense_kind_without_its_array_is_refused_by_name():
    """Slice C3a2 admits `kind="dense"`, and only with the block that belongs to it."""
    with pytest.raises(TranslationError, match="present exactly for kind='dense'"):
        identity_metric(kind="dense")


def test_a_dense_array_without_its_kind_is_refused_by_name():
    with pytest.raises(TranslationError, match="present exactly for kind='dense'"):
        identity_metric(dense=dense_block())


def test_a_dense_metric_reports_its_channels_and_freezes_its_array():
    metric = identity_metric(
        kind="dense", z_edges=(0.0, 1.0), dense=dense_block(n_modes=3)
    )
    assert metric.kind == "dense"
    assert metric.n_modes == 3
    assert metric.dense is not None
    assert not metric.dense.flags.writeable


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dense": dense_block(n_omega=2)}, "n_modes, n_omega"),
        ({"dense": dense_block(n_intervals=2)}, "n_modes, n_omega"),
        ({"dense": dense_block(n_deg=0)}, "empty degree axis"),
    ],
)
def test_a_dense_metric_checks_its_extents(changes, message):
    with pytest.raises(TranslationError, match=message):
        identity_metric(kind="dense", z_edges=(0.0, 1.0), **changes)


def test_a_dense_metric_that_is_not_hermitian_is_refused_by_name():
    block = dense_block()
    block[0, 0, 0, 1] = 0.5
    with pytest.raises(TranslationError, match="must be Hermitian"):
        identity_metric(kind="dense", z_edges=(0.0, 1.0), dense=block)


def test_an_unknown_metric_kind_is_refused_by_name():
    with pytest.raises(TranslationError, match=r"modal_metric\.kind"):
        past_the_hook(identity_metric(), kind="orthogonal")


@pytest.mark.parametrize("field", ["gauge_id", "certificate_id"])
def test_a_metric_identity_must_be_non_empty(field):
    with pytest.raises(TranslationError, match="non-empty"):
        override(identity_metric, field, "")


@pytest.mark.parametrize(
    "omega",
    [
        np.array([]),
        np.array([1.0e15, np.inf, 2.0e15]),
        np.array([2.0e15, 1.0e15, 3.0e15]),
    ],
)
def test_a_metric_frequency_axis_must_be_a_finite_increasing_vector(omega):
    with pytest.raises(TranslationError, match="increasing vector"):
        identity_metric(omega=omega)


def test_a_metric_frequency_axis_must_be_one_dimensional():
    with pytest.raises(TranslationError, match="increasing vector"):
        past_the_hook(identity_metric(), omega=np.array([[1.0e15, 2.0e15]]))


def test_a_metric_position_axis_must_be_increasing():
    with pytest.raises(TranslationError, match="z_edges"):
        identity_metric(z_edges=np.array([1.0, 0.0]))


def test_a_metric_frequency_axis_must_be_non_negative():
    with pytest.raises(TranslationError, match="non-negative"):
        identity_metric(omega=np.array([-1.0e15, 1.0e15, 2.0e15]))


def test_a_metric_frequency_axis_may_start_at_zero():
    """The canonical coarse grid of a field problem does, and the leaf carries it."""
    metric = identity_metric(omega=np.array([0.0, 1.0e15]))
    assert float(metric.omega[0]) == 0.0


def test_a_dense_array_beside_an_identity_kind_is_refused():
    with pytest.raises(TranslationError, match=r"modal_metric\.dense"):
        identity_metric(dense=np.zeros((1, 1, 2, 2, 3), dtype=complex))


def test_a_diagonal_kind_without_its_array_is_refused():
    with pytest.raises(TranslationError, match="present exactly for"):
        identity_metric(kind="diagonal")


def test_a_diagonal_array_beside_an_identity_kind_is_refused():
    with pytest.raises(TranslationError, match="present exactly for"):
        identity_metric(diagonal=np.full((1, 1, 2, 3), 0.5))


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((5, 2, 2, 3), "n_intervals, n_deg"),
        ((2, 2, 2, 4), "one column per coarse frequency"),
        ((2, 2, 0, 3), "one column per coarse frequency"),
    ],
)
def test_a_diagonal_metric_checks_its_extents(shape, message):
    with pytest.raises(TranslationError, match=message):
        diagonal_metric(diagonal=np.full(shape, 0.5))


def test_a_diagonal_metric_must_have_four_axes():
    with pytest.raises(TranslationError, match="n_intervals, n_deg"):
        past_the_hook(diagonal_metric(), diagonal=np.full((2, 3), 0.5))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_a_diagonal_metric_must_be_finite_and_positive(value):
    with pytest.raises(TranslationError, match="finite and positive"):
        diagonal_metric(diagonal=np.full((2, 2, 2, 3), value))


def test_a_diagonal_metric_needs_a_degree():
    with pytest.raises(TranslationError, match="empty degree axis"):
        diagonal_metric(diagonal=np.full((2, 0, 2, 3), 0.5))


@pytest.mark.parametrize("field", ["error", "minimum_eigenvalue", "maximum_eigenvalue"])
def test_a_metric_certificate_row_must_have_one_entry_per_interval(field):
    with pytest.raises(TranslationError, match=f"modal_metric.{field}"):
        override(diagonal_metric, field, np.zeros((7, 3)))


@pytest.mark.parametrize("field", ["error", "minimum_eigenvalue", "maximum_eigenvalue"])
def test_a_metric_certificate_row_must_be_non_negative(field):
    with pytest.raises(TranslationError, match="non-negative"):
        override(diagonal_metric, field, np.full((2, 3), -1.0))


def test_a_metric_certificate_row_must_be_finite():
    with pytest.raises(TranslationError, match="must be finite"):
        diagonal_metric(error=np.full((2, 3), np.nan))


def test_a_metric_minimum_may_not_exceed_its_maximum():
    with pytest.raises(TranslationError, match="must not exceed"):
        diagonal_metric(minimum_eigenvalue=np.full((2, 3), 2.0))


def test_a_semidefinite_metric_is_stored_rather_than_refused():
    """A zero smallest eigenvalue is flux data; what it cannot do is certify a norm."""
    metric = diagonal_metric(minimum_eigenvalue=np.zeros((2, 3)))
    assert float(np.min(metric.minimum_eigenvalue)) == 0.0


def test_a_metric_prints_one_line_and_a_block():
    identity = identity_metric()
    assert "identity" in repr(identity) and "\n" not in repr(identity)
    assert "from the state" in str(identity)
    assert "snapshot at" in str(identity)
    path = diagonal_metric()
    assert "2 channels" in repr(path)
    assert "snapshot at" not in str(path)
    assert "channels" in str(path)
    assert "1 channel," in repr(diagonal_metric(n_modes=1))


# ----------------------------------------------------------- the coupled-flow leaf


def test_a_dense_flow_reports_its_representation_and_channels():
    flow = dense_flow()
    assert flow.representation == "dense"
    assert flow.n_modes == 2
    assert flow.generator is not None
    assert not flow.generator.flags.writeable


def test_a_factored_flow_reports_its_representation_and_channels():
    flow = factored_flow()
    assert flow.representation == "factored"
    assert flow.n_modes == 1
    for array in (
        flow.fixed_terms,
        flow.term_weights,
        flow.wall_vectors,
        flow.wall_weights,
        flow.connection,
    ):
        assert array is not None
        assert not array.flags.writeable


@pytest.mark.parametrize("field", ["gauge_id", "certificate_id"])
def test_a_flow_identity_must_be_non_empty(field):
    with pytest.raises(TranslationError, match="non-empty"):
        override(dense_flow, field, "")


def test_a_flow_frequency_axis_must_be_non_negative():
    with pytest.raises(TranslationError, match="non-negative"):
        dense_flow(omega=np.array([-1.0e15, 1.0e15, 2.0e15]))


def test_a_flow_position_axis_must_be_increasing():
    with pytest.raises(TranslationError, match=r"coupled_flow\.z_edges"):
        dense_flow(z_edges=np.array([1.0, 0.0]))


def test_the_two_flow_representations_may_not_be_mixed():
    with pytest.raises(TranslationError, match="exactly one complete representation"):
        dense_flow(fixed_terms=np.zeros((1, 2, 2, 3), dtype=complex))


def test_a_flow_without_either_representation_is_refused():
    with pytest.raises(TranslationError, match="either a dense generator"):
        dense_flow(generator=None)


def test_a_half_factored_flow_is_refused():
    with pytest.raises(TranslationError, match="either a dense generator"):
        factored_flow(term_weights=None)


@pytest.mark.parametrize(
    "generator",
    [
        np.zeros((1, 2, 2, 3, 3), dtype=complex),
        np.zeros((1, 2, 0, 0, 3), dtype=complex),
    ],
)
def test_a_dense_generator_needs_equal_non_empty_channel_axes(generator):
    with pytest.raises(TranslationError, match=r"coupled_flow\.generator"):
        dense_flow(generator=generator)


def test_a_dense_generator_must_have_five_axes():
    with pytest.raises(TranslationError, match=r"coupled_flow\.generator"):
        past_the_hook(dense_flow(), generator=np.zeros((1, 2, 2, 3), dtype=complex))


def test_a_dense_generator_must_be_finite():
    with pytest.raises(TranslationError, match="must be finite"):
        dense_flow(generator=np.full((1, 2, 2, 2, 3), np.nan, dtype=complex))


def test_a_dense_generator_counts_intervals_and_frequencies():
    with pytest.raises(TranslationError, match="interval"):
        dense_flow(generator=np.zeros((3, 2, 2, 2, 3), dtype=complex))


def test_factored_terms_count_their_frequencies():
    with pytest.raises(TranslationError, match=r"coupled_flow\.fixed_terms must have"):
        factored_flow(fixed_terms=np.zeros((2, 1, 1, 5), dtype=complex))


def test_factored_terms_need_at_least_one_term():
    with pytest.raises(TranslationError, match="at least one term"):
        factored_flow(
            fixed_terms=np.zeros((0, 1, 1, 3), dtype=complex),
            term_weights=np.zeros((1, 2, 0), dtype=complex),
        )


def test_factored_weights_must_match_the_term_count():
    with pytest.raises(TranslationError, match=r"coupled_flow\.term_weights"):
        factored_flow(term_weights=np.zeros((1, 2, 5), dtype=complex))


def test_factored_weights_must_have_three_axes():
    with pytest.raises(TranslationError, match=r"coupled_flow\.term_weights"):
        past_the_hook(factored_flow(), term_weights=np.zeros((1, 2), dtype=complex))


def test_factored_weights_must_be_finite():
    with pytest.raises(TranslationError, match="term_weights must be finite"):
        factored_flow(term_weights=np.full((1, 2, 2), np.nan, dtype=complex))


def test_a_wall_pair_is_supplied_together():
    with pytest.raises(TranslationError, match="supplied together"):
        factored_flow(wall_weights=None)


def test_wall_vectors_are_checked_against_the_channels():
    with pytest.raises(TranslationError, match=r"coupled_flow\.wall_vectors"):
        factored_flow(wall_vectors=np.zeros((1, 4, 3), dtype=complex))


def test_wall_weights_are_checked_against_the_frequencies():
    with pytest.raises(TranslationError, match=r"coupled_flow\.wall_weights must have"):
        factored_flow(wall_weights=np.zeros((1, 2, 5), dtype=complex))


def test_wall_weights_must_be_finite():
    with pytest.raises(TranslationError, match="wall_weights must be finite"):
        factored_flow(wall_weights=np.full((1, 2, 3), np.nan, dtype=complex))


def test_a_connection_is_checked_against_the_channels():
    with pytest.raises(TranslationError, match=r"coupled_flow\.connection must have"):
        factored_flow(connection=np.zeros((1, 2, 2, 2, 3), dtype=complex))


def test_a_connection_needs_equal_channel_axes():
    with pytest.raises(TranslationError, match=r"coupled_flow\.connection"):
        factored_flow(connection=np.zeros((1, 2, 1, 2, 3), dtype=complex))


@pytest.mark.parametrize(
    ("builder", "field", "value"),
    [
        (dense_flow, "generator", np.zeros((1, 0, 2, 2, 3), dtype=complex)),
        (factored_flow, "term_weights", np.zeros((1, 0, 2), dtype=complex)),
        (factored_flow, "wall_weights", np.zeros((1, 0, 3), dtype=complex)),
        (factored_flow, "connection", np.zeros((1, 0, 1, 1, 3), dtype=complex)),
    ],
)
def test_a_flow_coefficient_block_needs_a_degree(builder, field, value):
    """Contract section 5: the degree axis is validated, not read as a zero."""
    with pytest.raises(TranslationError, match="empty degree axis"):
        override(builder, field, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generator_error", -1.0),
        ("residual_norm_bound", -1.0),
        ("inverse_condition_bound", 0.5),
    ],
)
def test_a_flow_certificate_row_has_a_floor(field, value):
    with pytest.raises(TranslationError, match="must be at least"):
        override(dense_flow, field, np.full((1, 3), value))


def test_a_flow_certificate_row_is_checked_against_the_intervals():
    with pytest.raises(TranslationError, match=r"coupled_flow\.generator_error"):
        dense_flow(generator_error=np.zeros((4, 3)))


def test_a_flow_names_one_method_per_interval():
    with pytest.raises(TranslationError, match="one method per interval"):
        dense_flow(section_methods=("diagonal", "diagonal"))


def test_a_flow_refuses_an_unknown_section_method():
    with pytest.raises(TranslationError, match="unknown entries"):
        past_the_hook(dense_flow(), section_methods=("magnus",))


def test_a_flow_prints_one_line_and_a_block():
    flow = dense_flow()
    assert "dense" in repr(flow) and "\n" not in repr(flow)
    assert "1 section" in repr(flow)
    assert "wall" in str(flow) and "no" in str(flow)
    factored = factored_flow()
    assert "1 channel," in repr(factored)
    assert "frozen_residual" in str(factored)
    assert "connection" in str(factored)
    wide = dense_flow(
        n_edges=3,
        generator=np.zeros((2, 2, 2, 2, 3), dtype=complex),
        generator_error=np.zeros((2, 3)),
        residual_norm_bound=np.zeros((2, 3)),
        inverse_condition_bound=np.ones((2, 3)),
        section_methods=("diagonal", "cf4"),
    )
    assert "2 sections" in repr(wide)


# ------------------------------------------------------- the regional response leaf


def test_a_kerr_only_regional_record_reports_its_counts():
    tables = kerr_regions(n_responses=2)
    assert tables.n_responses == 2
    assert tables.n_components == 3
    assert tables.n_terms == 0
    assert not tables.chi3.flags.writeable
    assert not tables.response_region.flags.writeable
    assert not tables.instantaneous_tensor.flags.writeable
    assert not tables.delayed_tensor.flags.writeable
    assert not tables.delayed_kernel.flags.writeable
    assert not tables.kernel_valid_samples.flags.writeable
    assert not tables.temporal_weight.flags.writeable


@pytest.mark.parametrize("components", [1, 2, 3])
def test_a_regional_record_takes_one_two_or_three_components(components):
    assert kerr_regions(components=components).n_components == components


@pytest.mark.parametrize("field", ["model_id", "policy_id"])
def test_a_regional_identity_must_be_non_empty(field):
    with pytest.raises(TranslationError, match="non-empty"):
        override(kerr_regions, field, "")


@pytest.mark.parametrize(
    "regions", [np.zeros(0, dtype=np.int32), np.array([-1], dtype=np.int32)]
)
def test_a_regional_record_checks_its_region_indices(regions):
    with pytest.raises(TranslationError, match="response_region"):
        kerr_regions(response_region=regions)


def test_a_regional_region_index_must_be_one_dimensional():
    with pytest.raises(TranslationError, match="response_region"):
        past_the_hook(kerr_regions(), response_region=np.zeros((1, 1), dtype=np.int32))


@pytest.mark.parametrize(
    "tensor",
    [np.ones((2, 3, 3, 3, 3)), np.ones((1, 3, 3, 3, 2)), np.ones((1, 4, 4, 4, 4))],
)
def test_an_instantaneous_tensor_is_checked_against_the_counts(tensor):
    with pytest.raises(TranslationError, match="instantaneous_tensor must have"):
        kerr_regions(instantaneous_tensor=tensor)


def test_an_instantaneous_tensor_must_have_five_axes():
    with pytest.raises(TranslationError, match="instantaneous_tensor must have"):
        past_the_hook(kerr_regions(), instantaneous_tensor=np.ones((1, 3, 3, 3)))


def test_an_instantaneous_tensor_must_be_finite():
    with pytest.raises(TranslationError, match="instantaneous_tensor must be finite"):
        kerr_regions(instantaneous_tensor=np.full((1, 3, 3, 3, 3), np.nan))


def test_a_delayed_tensor_is_checked_against_the_components():
    with pytest.raises(TranslationError, match="delayed_tensor must have"):
        kerr_regions(delayed_tensor=np.zeros((1, 0, 2, 2, 2, 2)))


def test_a_delayed_tensor_must_have_six_axes():
    with pytest.raises(TranslationError, match="delayed_tensor must have"):
        past_the_hook(kerr_regions(), delayed_tensor=np.zeros((1, 0, 3, 3, 3)))


def test_a_delayed_regional_term_is_carried_with_its_kernel_and_support():
    """A regional term's tensor, kernel and support arrays must agree."""
    record = kerr_regions(
        delayed_tensor=np.zeros((1, 1, 3, 3, 3, 3)),
        delayed_kernel=np.zeros((1, 1, 1, 1, 16)),
        kernel_valid_samples=np.zeros((1, 1), dtype=np.int32),
    )
    assert record.n_terms == 1
    with pytest.raises(TranslationError, match="delayed_kernel must have shape"):
        kerr_regions(
            delayed_tensor=np.zeros((1, 1, 3, 3, 3, 3)),
            delayed_kernel=np.zeros((1, 1, 1, 2, 16)),
            kernel_valid_samples=np.zeros((1, 1), dtype=np.int32),
        )
    with pytest.raises(TranslationError, match="kernel_valid_samples must have shape"):
        kerr_regions(
            delayed_tensor=np.zeros((1, 1, 3, 3, 3, 3)),
            delayed_kernel=np.zeros((1, 1, 1, 1, 16)),
            kernel_valid_samples=np.zeros((1, 2), dtype=np.int32),
        )


@pytest.mark.parametrize("chi3", [np.zeros((0, 1, 1)), np.zeros((1, 1, 4))])
def test_a_regional_susceptibility_is_checked_against_the_responses(chi3):
    with pytest.raises(TranslationError, match="chi3 must have"):
        kerr_regions(chi3=chi3)


def test_a_regional_susceptibility_must_have_three_axes():
    with pytest.raises(TranslationError, match="chi3 must have"):
        past_the_hook(kerr_regions(), chi3=np.zeros((1, 1)))


def test_a_regional_susceptibility_needs_a_degree():
    """Contract section 5: the degree axis is validated, not read as a zero."""
    with pytest.raises(TranslationError, match="empty degree axis"):
        kerr_regions(chi3=np.zeros((1, 0, 1)))


def test_a_regional_susceptibility_must_be_finite():
    with pytest.raises(TranslationError, match="chi3 must be finite"):
        kerr_regions(chi3=np.full((1, 1, 1), np.nan))


@pytest.mark.parametrize("weight", [np.zeros(0), np.full(8, -1.0)])
def test_a_regional_temporal_weight_must_be_non_negative_seconds(weight):
    with pytest.raises(TranslationError, match="temporal_weight"):
        kerr_regions(temporal_weight=weight)


def test_a_regional_temporal_weight_must_be_one_dimensional():
    with pytest.raises(TranslationError, match="temporal_weight"):
        past_the_hook(kerr_regions(), temporal_weight=np.zeros((2, 2)))


def test_a_regional_kernel_is_checked_against_the_doubled_fine_axis():
    with pytest.raises(TranslationError, match="delayed_kernel must have"):
        kerr_regions(delayed_kernel=np.zeros((1, 1, 1, 0, 8)))


@pytest.mark.parametrize(
    "support", [np.zeros((2, 0), dtype=np.int32), np.zeros((1, 1), dtype=np.int32)]
)
def test_a_regional_support_count_is_checked_against_the_terms(support):
    with pytest.raises(TranslationError, match="kernel_valid_samples"):
        kerr_regions(kernel_valid_samples=support)


def test_a_regional_record_prints_one_line_and_a_block():
    one = kerr_regions()
    assert "1 response," in repr(one) and "\n" not in repr(one)
    assert "0 delayed terms" in repr(one)
    assert "none (Kerr only)" in str(one)
    assert "2 responses" in repr(kerr_regions(n_responses=2))


# ------------------------------------------------- the leaves against the state

Z_END = 1e-3
"""A millimetre: long enough to be a run, short enough to assemble in no time."""


@pytest.fixture(scope="module")
def strand_grid():
    """The smallest canonical field grid: 33 coarse bins and 65 fine ones."""
    return FieldGrid.from_limits(Z_END, 800e-9, (400e-9, 2e-6), 20e-15)


@pytest.fixture(scope="module")
def strand(strand_grid):
    """A one-channel field problem over a millimetre, to hang the leaves on.

    A **field** problem, because a general-modal leaf is tabulated on the canonical
    coarse grid itself and an envelope grid's is in FFT order about ``omega0``,
    which is not the increasing axis a tabulated leaf declares.
    """
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9).build(
        strand_grid
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            strand_grid,
            MarcatiliMode(125e-6, "He", 1.0),
            [KerrResponse(chi3=2.2e-52)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[Z_END],
            dz_max=1e-4,
            dz_init=1e-5,
        )


def modal_tables_on(problem, grid, n_nodes=4, n_modes=1, **overrides):
    """A one-channel E/H reference tabulated on a problem's own canonical grids.

    Hand-built rather than sampled from a provider: the canonical coarse grid runs from
    zero and out past the guide's active support, where the contract requires the
    reference to be explicitly zero rather than an evaluated profile, and the machinery
    that zero-fills a canonical tabulation belongs to the execution route, not to these
    records. What is under test here is the agreement between the record's extents and
    the state; a sampled table's physics is
    `tests/translate/test_modal_transform_tables.py`'s.
    """
    omega = overrides.pop("omega", problem.grids.omega)
    fine = overrides.pop("omega_fine", grid.omega_fine)
    z_edges = np.asarray(
        overrides.pop("z_edges", [0.0, problem.stepper.z_end]), dtype=float
    )
    positions = len(z_edges)
    field = np.zeros((positions, n_nodes, 3, n_modes, len(omega)), dtype=complex)
    fields = {
        "omega": omega,
        "omega_fine": fine,
        "z_edges": z_edges,
        "synthesis": None,
        "projection": None,
        "electric": field,
        "magnetic": field,
        "backward_electric": None,
        "backward_magnetic": None,
        "reciprocal_gram": None,
        "coordinate_dual": None,
        "node_xy": np.tile(
            np.stack([np.arange(n_nodes) * 1e-6, np.zeros(n_nodes)], axis=1),
            (positions, 1, 1),
        ),
        "weights": np.full((positions, n_nodes), 1e-12),
        "region_index": np.zeros(n_nodes, dtype=np.int32),
        "source_frequency_factor": -1j * np.asarray(omega, dtype=float),
        # Nothing is inside the support of an all-zero reference, which is what the
        # mask is for: a fine bin outside it is zero, not an invalid profile.
        "valid": np.zeros((positions, n_modes, len(fine)), dtype=bool),
        "profile_error": np.zeros((interval_count(positions), len(omega))),
        "factors": None,
        "gauge_id": "lab",
        "model_id": "hand-built-reference",
        "policy_id": "dense-nodes",
    } | overrides
    return ModalTransformTables(**fields)


def metric_on(problem, omega=None, **overrides):
    """An orthonormal metric on a problem's canonical grid, covering its run."""
    axis = problem.grids.omega if omega is None else omega
    return identity_metric(
        omega=axis, z_edges=[0.0, problem.stepper.z_end], **overrides
    )


def flow_on(problem, n_modes=1, omega=None, z_max=None, **overrides):
    """A dense flow on a problem's canonical grid, covering its run."""
    return dense_flow(
        n_modes=n_modes,
        omega=problem.grids.omega if omega is None else omega,
        z_max=problem.stepper.z_end if z_max is None else z_max,
        **overrides,
    )


def with_leaves(problem, **leaves):
    """Return the problem with general-modal leaves installed past the hook."""
    groups = {
        "coupled_flow": "linear",
        "modal_metric": "linear",
        "modal_tables": "transform",
        "regional_responses": "responses",
        "geometry": "transform",
    }
    changed = {}
    for name, value in leaves.items():
        group_name = groups[name]
        group = changed.get(group_name) or replace(getattr(problem, group_name))
        object.__setattr__(group, name, value)
        changed[group_name] = group
    return replace(problem, **changed)


@pytest.fixture(scope="module")
def furnished(strand, strand_grid):
    """The problem with all four leaves, each on the canonical grids."""
    return with_leaves(
        strand,
        modal_tables=modal_tables_on(strand, strand_grid),
        modal_metric=metric_on(strand),
        coupled_flow=flow_on(strand),
        regional_responses=kerr_regions(n_to=strand.grids.n_to),
    )


def test_four_leaves_on_the_canonical_grids_validate(furnished, strand, strand_grid):
    """Every axis of every leaf is the state's own, so the whole problem validates."""
    tables = furnished.transform.modal_tables
    metric = furnished.linear.modal_metric
    flow = furnished.linear.coupled_flow
    assert np.array_equal(tables.omega, strand.grids.omega)
    assert np.array_equal(tables.omega_fine, strand_grid.omega_fine)
    assert np.array_equal(metric.omega, strand.grids.omega)
    assert np.array_equal(flow.omega, strand.grids.omega)
    assert float(tables.z_edges[-1]) == strand.stepper.z_end
    assert furnished.validate() is None


def test_the_declared_index_leaves_cross_the_seam_as_int32(furnished):
    """The three leaves schema 12 admits `int32` for, and nothing else."""
    found = {
        name: values.dtype
        for name, values in _walk_arrays(furnished)
        if values.dtype == np.int32
    }
    assert set(found) == {
        "transform.modal_tables.region_index",
        "responses.regional_responses.response_region",
        "responses.regional_responses.kernel_valid_samples",
    }


def test_an_int32_array_at_an_undeclared_leaf_is_refused(furnished):
    """The dtype rule reaches every leaf, so `int32` is admitted by name only."""
    stepper = replace(furnished.stepper)
    planes = np.array([1], dtype=np.int32)
    planes.setflags(write=False)
    object.__setattr__(stepper, "save_planes", planes)
    spoiled = replace(furnished, stepper=stepper)
    with pytest.raises(TranslationError, match=r"stepper\.save_planes has dtype int32"):
        spoiled._check_arrays()


def test_a_modal_reference_beside_spatial_geometry_is_refused(strand, strand_grid):
    problem = with_leaves(
        strand, modal_tables=modal_tables_on(strand, strand_grid), geometry=object()
    )
    with pytest.raises(TranslationError, match="one or the other"):
        problem._check_modal_leaves()


def test_a_modal_reference_needs_a_single_polarisation(furnished):
    problem = replace(furnished)
    object.__setattr__(problem, "shape", (1, 1, 2, furnished.shape[-1]))
    with pytest.raises(TranslationError, match="keeps n_pol = 1"):
        problem._check_modal_leaves()


def test_a_modal_reference_counts_the_states_channels(furnished):
    problem = replace(furnished)
    object.__setattr__(problem, "shape", (1, 3, 1, furnished.shape[-1]))
    with pytest.raises(TranslationError, match="describes 1 channel"):
        problem._check_modal_leaves()


def test_a_metric_counts_the_states_channels(strand):
    problem = with_leaves(
        strand,
        modal_metric=diagonal_metric(n_modes=3, omega=strand.grids.omega, n_edges=2),
    )
    with pytest.raises(TranslationError, match="modal_metric describes 3"):
        problem._check_modal_leaves()


def test_a_coupled_flow_counts_the_states_channels(strand):
    problem = with_leaves(strand, coupled_flow=flow_on(strand, n_modes=2))
    with pytest.raises(TranslationError, match="couples 2 channel"):
        problem._check_modal_leaves()


def test_regional_responses_need_the_node_regions_they_index(strand, strand_grid):
    problem = with_leaves(strand, regional_responses=kerr_regions())
    with pytest.raises(TranslationError, match="which is absent"):
        problem._check_modal_leaves()
    far = with_leaves(
        strand,
        modal_tables=modal_tables_on(strand, strand_grid),
        regional_responses=kerr_regions(response_region=np.array([5], dtype=np.int32)),
    )
    with pytest.raises(TranslationError, match="names region 5"):
        far._check_modal_leaves()


def test_regional_responses_count_the_references_components(strand, strand_grid):
    """MAX_N_COMPONENTS is the ceiling; the reference's own count is the rule."""
    problem = with_leaves(
        strand,
        modal_tables=modal_tables_on(strand, strand_grid),
        regional_responses=kerr_regions(components=2),
    )
    with pytest.raises(TranslationError, match="2 Cartesian component"):
        problem._check_modal_leaves()


@pytest.mark.parametrize("leaf", ["modal_tables", "modal_metric", "coupled_flow"])
def test_a_leaf_on_another_frequency_grid_is_refused(strand, strand_grid, leaf):
    """One bin short of the canonical coarse grid is not the canonical coarse grid."""
    short = strand.grids.omega[:-1]
    builders = {
        "modal_tables": lambda: modal_tables_on(strand, strand_grid, omega=short),
        "modal_metric": lambda: metric_on(strand, omega=short),
        "coupled_flow": lambda: flow_on(strand, omega=short),
    }
    problem = with_leaves(strand, **{leaf: builders[leaf]()})
    with pytest.raises(TranslationError, match=r"is not grids\.omega"):
        problem._check_modal_leaves()


@pytest.mark.parametrize("leaf", ["modal_tables", "modal_metric", "coupled_flow"])
def test_a_leaf_that_stops_before_the_end_of_the_run_is_refused(
    strand, strand_grid, leaf
):
    """A z table that ends inside the run is refused, as every other one is."""
    half = [0.0, 0.5 * strand.stepper.z_end]
    builders = {
        "modal_tables": lambda: modal_tables_on(strand, strand_grid, z_edges=half),
        "modal_metric": lambda: identity_metric(omega=strand.grids.omega, z_edges=half),
        "coupled_flow": lambda: flow_on(strand, z_max=half[1]),
    }
    problem = with_leaves(strand, **{leaf: builders[leaf]()})
    with pytest.raises(TranslationError, match="does not contain the run"):
        problem._check_modal_leaves()


def test_a_reference_on_another_fine_grid_is_refused(strand, strand_grid):
    """S lives on the fine spectrum, which the coarse bins index through crop_index."""
    problem = with_leaves(
        strand,
        modal_tables=modal_tables_on(
            strand, strand_grid, omega_fine=strand_grid.omega_fine[:-1]
        ),
    )
    with pytest.raises(TranslationError, match="fine spectrum"):
        problem._check_modal_leaves()


def test_a_reference_whose_coarse_bins_sit_elsewhere_is_refused(strand, strand_grid):
    """The right length is not enough: the coarse bins must sit where they belong."""
    moved = strand_grid.omega_fine + 1.0
    problem = with_leaves(
        strand, modal_tables=modal_tables_on(strand, strand_grid, omega_fine=moved)
    )
    with pytest.raises(TranslationError, match="fine spectrum"):
        problem._check_modal_leaves()


# ----------------------------------------------------- the dense metric builder


def sampled_gram(n_modes=2, n_omega=N_OMEGA, defect=1e-6, off=1e-7):
    """A Hermitian sampled metric, the identity plus a declared departure."""
    block = np.broadcast_to(
        np.eye(n_modes, dtype=complex)[:, :, None], (n_modes, n_modes, n_omega)
    ).copy()
    block *= 1.0 - defect
    if n_modes > 1:
        block[0, 1] = off
        block[1, 0] = off
    return block


def test_a_dense_metric_carries_its_block_and_its_eigenvalues():
    metric = dense_metric(
        OMEGA,
        np.array([0.0, 1.0]),
        sampled_gram(),
        defect=1e-6,
        gauge_id="lab+circular(0,1)",
        certificate_id="sampled-gram",
    )
    assert metric.kind == "dense" and metric.n_modes == 2
    assert metric.dense is not None
    assert metric.dense.shape == (1, 1, 2, 2, len(OMEGA))
    # The eigenvalue rows are the stored block's own, which is what a controller
    # norm and a noise draw need certified.
    expected = np.linalg.eigvalsh(np.moveaxis(sampled_gram(), (0, 1), (-2, -1)))
    assert_allclose(metric.minimum_eigenvalue[0], expected.min(axis=-1), rtol=1e-14)
    assert_allclose(metric.maximum_eigenvalue[0], expected.max(axis=-1), rtol=1e-14)
    assert float(np.max(metric.error)) == 1e-6


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (np.zeros((2, 3, N_OMEGA), complex), "n_modes, n_modes"),
        (np.zeros((2, 2, 2), complex), "n_modes, n_modes"),
        (np.zeros((2, 2, len(OMEGA)), complex), "positive definite"),
    ],
)
def test_a_dense_metric_refuses_a_block_that_is_not_one(block, message):
    with pytest.raises(TranslationError, match=message):
        dense_metric(
            OMEGA,
            np.array([0.0, 1.0]),
            block,
            defect=0.0,
            gauge_id="lab",
            certificate_id="sampled-gram",
        )


def test_metric_at_evaluates_the_stored_block_and_passes_the_identity_through():
    metric = dense_metric(
        OMEGA,
        np.array([0.0, 1.0]),
        sampled_gram(),
        defect=1e-6,
        gauge_id="lab+circular(0,1)",
        certificate_id="sampled-gram",
    )
    evaluated = metric_at(metric, 0.5)
    assert evaluated is not None
    assert_allclose(evaluated, sampled_gram(), rtol=0, atol=0)
    assert metric_at(None, 0.5) is None
    assert metric_at(identity_metric(), 0.0) is None


def test_metric_at_refuses_a_diagonal_metric_by_name():
    with pytest.raises(TranslationError, match="no executed statistics path"):
        metric_at(diagonal_metric(n_modes=2), 0.0)

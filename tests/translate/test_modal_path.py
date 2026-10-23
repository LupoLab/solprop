# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving guide's path: its sections, its rule, its gauge, its degree and its maps.

The graded-index taper here is the cheap provider -- a one-dimensional element mesh,
milliseconds a solve -- so every door of the path is exercised in the default tier; the
step-index validations are in ``tests/driver/test_moving_guide.py``. Every bound is
stated before it is measured.
"""

import dataclasses

import numpy as np
import pytest
from numpy.polynomial import chebyshev as chebyshev_basis

from modal_provider_cases import grin_grid, grin_provider
from solprop.backends.numpy_ref.general_modal import GeneralMaps
from solprop.exceptions import BackendError, ModeError
from solprop.modes.connection import band_connection, modal_connection
from solprop.modes.guide import ModalGuide, ModalPath, ProviderChannelMode
from solprop.modes.vector import ModalNodeRule, flux_product
from solprop.profiles import Linear, PiecewiseLinear
from solprop.translate.modal_path import (
    INITIAL_PATH_DEGREE,
    MAX_PATH_DEGREE,
    in_frequency,
    path_rule,
    path_sections,
    path_snapshot,
    path_transform_tables,
    sample_modal_path,
    section_coefficients,
    transform_section,
    transform_weights,
)
from solprop.translate.modal_tables import (
    _projection_map,
    modal_transform_tables,
    sample_modal_table,
)

LENGTH = 1e-3
"""Length of every declared path here, m: a millimetre."""


def tapered(radius=None):
    """Return the graded-index sector on a taper, 25 um down to 22.5 um."""
    return dataclasses.replace(
        grin_provider(elements=32),
        radius=Linear(25e-6, 22.5e-6, LENGTH) if radius is None else radius,
    )


def kinked():
    """Return the same sector on a piecewise-linear taper with a corner at 0.4 mm."""
    return tapered(
        PiecewiseLinear(
            np.array([0.0, 4e-4, LENGTH]), np.array([25e-6, 24e-6, 22.5e-6])
        )
    )


def knots():
    """Return five knots over the graded-index grid's active band, in rad/s."""
    grid = grin_grid()
    active = grid.omega[grid.active_mask]
    return np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 5)


@pytest.fixture(scope="module")
def guide():
    """The graded-index taper as a guide, its path sampled at the entrance's rule."""
    provider = tapered()
    frequencies = knots()
    return ModalGuide.from_provider(
        provider,
        frequencies,
        nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=2),
    )


def test_the_sections_split_at_every_interior_kink_and_subdivide_equally():
    """The z domain's interior breakpoints, each interval divided `sections` times."""
    assert path_sections(tapered()).tolist() == [0.0, LENGTH]
    edges = path_sections(kinked(), 2)
    assert np.allclose(edges, [0.0, 2e-4, 4e-4, 7e-4, 1e-3], rtol=0, atol=1e-18)
    for bad in (0, -1, True):
        with pytest.raises(ModeError, match="positive integer"):
            path_sections(kinked(), bad)  # type: ignore[arg-type]


def test_the_rule_is_asked_for_again_at_every_position_as_the_entrance_was():
    """Its structure and its regions are the entrance's; its edges are the position's.

    The graded-index mesh scales with the core radius, so the rule at the exit is the
    entrance rule scaled by 0.9 exactly: every node moves with the interfaces.
    """
    provider = tapered()
    omega = float(knots()[0])
    entrance = provider.node_rule(omega, 0.0, n_r=2)
    exit_rule = path_rule(provider, entrance, LENGTH)
    assert np.array_equal(exit_rule.region_index, entrance.region_index)
    ratio = np.hypot(*exit_rule.xy.T) / np.hypot(*entrance.xy.T)
    np.testing.assert_allclose(ratio, 0.9, rtol=1e-14)
    unnamed = ModalNodeRule(
        entrance.xy, entrance.weights, entrance.region_index, {"rule": "hand made"}
    )
    with pytest.raises(ModeError, match="does not name its"):
        path_rule(provider, unnamed, LENGTH)


def test_a_rule_that_changes_structure_along_the_path_is_refused():
    """Node q is one moving point, so a rule of another size is a different node set."""
    provider = tapered()
    omega = float(knots()[0])
    entrance = provider.node_rule(omega, 0.0, n_r=2)
    wider = ModalNodeRule(
        entrance.xy,
        entrance.weights,
        entrance.region_index,
        {**entrance.identity, "n_r": 3},
    )
    with pytest.raises(ModeError, match="changes material"):
        path_rule(provider, wider, LENGTH)


class Flipped:
    """A provider whose own gauge flips the sign of its first channel somewhere.

    Every protocol member is spelled out, because the protocol is checked
    structurally and a delegating ``__getattr__`` is not seen by it.
    """

    def __init__(self, provider, *, z=np.inf, omega=np.inf):
        self.provider, self.z, self.omega = provider, z, omega

    def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
        return self.provider.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

    def solve(self, omega, z, *, nodes):
        sample = self.provider.solve(omega, z, nodes=nodes)
        if z <= self.z and omega <= self.omega:
            return sample
        sign = np.ones(sample.electric.shape[2])
        sign[0] = -1.0
        return dataclasses.replace(
            sample,
            electric=sample.electric * sign,
            magnetic=sample.magnetic * sign,
            backward_electric=sample.backward_electric * sign,
            backward_magnetic=sample.backward_magnetic * sign,
        )

    @property
    def regions(self):
        return self.provider.regions

    @property
    def z_breakpoints(self):
        return self.provider.z_breakpoints

    @property
    def omega_domain(self):
        return self.provider.omega_domain

    @property
    def z_domain(self):
        return self.provider.z_domain

    @property
    def identity(self):
        return self.provider.identity


def test_a_provider_whose_own_gauge_is_not_continuous_is_refused_by_name():
    """A sign that flips along z puts a connection in the maps the measured one lacks.

    The flip is placed between two sampled positions, so it is the continuity check
    along z that refuses it and not a solve.
    """
    provider = tapered()
    frequencies = knots()
    rule = provider.node_rule(float(frequencies[0]), 0.0, n_r=2)
    flipped = Flipped(provider, z=0.3 * LENGTH)
    with pytest.raises(ModeError, match="gauge is not continuous"):
        sample_modal_path(
            flipped,  # type: ignore[arg-type]
            frequencies,
            rule,
            np.array([0.0, LENGTH]),
            rtol=1.0,
            degree=2,
        )


def test_a_snapshot_keeps_the_provider_s_gauge_and_refuses_a_flip_across_the_band():
    """In frequency too: the knots are kept as solved, and a flip between is named."""
    provider = tapered()
    frequencies = knots()
    rule = provider.node_rule(float(frequencies[0]), 0.0, n_r=2)
    table = path_snapshot(provider, frequencies, 0.0, rule)
    aligned = sample_modal_table(provider, frequencies, np.array([0.0]), rule)
    for mine, theirs in zip(table.samples[0], aligned.samples[0], strict=True):
        assert np.array_equal(mine.electric, theirs.electric)

    with pytest.raises(ModeError, match="rad/s at z"):
        path_snapshot(
            Flipped(provider, omega=float(frequencies[2])),  # type: ignore[arg-type]
            frequencies,
            0.0,
            rule,
        )


def test_the_degree_search_is_measured_and_refused_where_it_cannot_resolve(guide):
    """The degree kept is one whose held-out departure was measured within budget.

    With the guide's own budget -- the frequency interpolation's measured error -- the
    graded-index taper resolves at the first degree tried, its held-out departure
    three or more orders under the budget. A budget below the round-off of the solves
    cannot be met by any degree and is refused at the ceiling, naming the numbers.
    """
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    path = sampling.path
    assert path.degree == INITIAL_PATH_DEGREE
    assert 0.0 < path.z_error < 1e-3 * path.rtol
    assert path.rtol == pytest.approx(sampling.profile_error)
    with pytest.raises(ModeError, match=f"exceed {MAX_PATH_DEGREE}"):
        sample_modal_path(
            guide.field_provider,
            sampling.omega,
            sampling.nodes,
            path.z_edges,
            rtol=1e-17,
        )


def test_the_channels_carry_their_index_and_area_along_the_path(guide):
    """The index at a sampled position is the provider's; between them it is held out.

    At a Chebyshev-Lobatto position the channel's index is the snapshot's own table
    there, to the rounding of the interpolant at its node (16 eps); half way between
    two the departure from a fresh solve is bounded by the path's own measured
    held-out error, because the index is the generator the check measured. Past the
    end the value is held, as a path's is. The area follows the core: smaller at the
    exit.
    """
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    path = sampling.path
    mode = guide.mode_collection.modes[0]
    assert isinstance(mode, ProviderChannelMode)
    assert not mode.z_constant
    assert mode.entrance().z_constant
    omega = np.array([float(sampling.omega[2])])
    positions = path.positions.reshape(-1)
    for position, table in zip(positions, mode.path_tables, strict=True):
        np.testing.assert_allclose(
            mode.neff(omega, float(position)),
            table.neff(omega),
            rtol=16 * np.finfo(float).eps,
            atol=0,
        )
    held = 0.5 * (positions[1] + positions[2])
    rule = path_rule(guide.field_provider, sampling.nodes, float(held))
    fresh = guide.field_provider.solve(float(omega[0]), float(held), nodes=rule)
    beta = float(np.real(1j * fresh.generator[0, 0]))
    measured = float(np.real(mode.neff(omega, float(held))[0])) * omega[0] / 299792458.0
    assert abs(measured - beta) <= path.z_error * abs(beta) + 1e-9 * abs(beta)
    assert mode.aeff(LENGTH) < mode.aeff(0.0)
    assert mode.aeff(2 * LENGTH) == mode.aeff(LENGTH)
    assert mode.dneff_domega(omega, LENGTH).shape == (1,)
    assert mode.neff_minus_one(omega, LENGTH).shape == (1,)
    area = mode.area_path
    assert area is not None
    assert float(area(np.array([LENGTH]))[0]) == pytest.approx(mode.aeff(LENGTH))
    assert mode.entrance().area_path is None


def test_a_path_record_refuses_what_it_cannot_describe():
    """Every field of the declaration and of the measurement is checked."""
    good = {
        "z_edges": np.array([0.0, LENGTH]),
        "degree": 4,
        "sections": 1,
        "rtol": 1e-6,
        "z_error": 1e-8,
        "gram_departure": 1e-9,
    }
    ModalPath(**good)
    for change, message in (
        ({"z_edges": np.array([0.0])}, "section edges"),
        ({"z_edges": np.array([LENGTH, 0.0])}, "section edges"),
        ({"degree": 0}, "degree is a positive integer"),
        ({"sections": True}, "sections is a positive integer"),
        ({"rtol": 0.0}, "rtol is a positive"),
        ({"z_error": -1.0}, "z_error is a non-negative"),
        ({"gram_departure": np.inf}, "gram_departure is a non-negative"),
    ):
        with pytest.raises(ModeError, match=message):
            ModalPath(**{**good, **change})
    path = ModalPath(**good)
    with pytest.raises(ModeError, match="one value per position"):
        path.interpolate(np.zeros(3), 0.0)
    values = np.arange(5.0)
    assert path.interpolate(values, 0.0) == 0.0
    assert path.interpolate(values, LENGTH) == 4.0
    assert path.section(-1.0) == 0 and path.section(2.0) == 0
    assert path.identity["degree"] == 4


def test_a_path_table_refuses_tables_out_of_place(guide):
    """One snapshot per Chebyshev-Lobatto position, in order, and a sane record."""
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    table = sample_modal_path(
        guide.field_provider,
        sampling.omega,
        sampling.nodes,
        sampling.path.z_edges,
        rtol=1.0,
        degree=1,
        measured=(1e-8, 1e-9),
    )
    assert table.n_intervals == 1 and len(table.tables) == 2
    assert len(table.section_tables(0)) == 2
    for change, message in (
        ({"z_edges": np.array([0.0])}, "at least two"),
        ({"degree": 0}, "positive integer"),
        ({"tables": table.tables[:1]}, "Chebyshev-Lobatto"),
        ({"z_error": -1.0}, "measured non-negative"),
    ):
        with pytest.raises(ModeError, match=message):
            dataclasses.replace(table, **change)
    with pytest.raises(ModeError, match="positive and finite"):
        sample_modal_path(
            guide.field_provider,
            sampling.omega,
            sampling.nodes,
            sampling.path.z_edges,
            rtol=0.0,
        )


@pytest.fixture(scope="module")
def moving_tables(guide):
    """The guide's moving record on the graded-index grid, and the path it came from."""
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    grid = grin_grid()
    path = sample_modal_path(
        guide.field_provider,
        sampling.omega,
        sampling.nodes,
        sampling.path.z_edges,
        rtol=sampling.path.rtol,
        degree=sampling.path.degree,
        measured=(sampling.path.z_error, sampling.path.gram_departure),
    )
    tables = path_transform_tables(
        path,
        grid.omega,
        omega_fine=grid.omega_fine,
        z_end=LENGTH,
        profile_error=sampling.profile_error,
    )
    return grid, path, tables


def test_the_maps_reproduce_each_sampled_snapshot_and_a_fresh_solve_between(
    guide, moving_tables
):
    """``S(z)`` and ``Q(z)`` from the stage weights, against the snapshots and a solve.

    Bounds, before the numbers. At a sampled position the weights reproduce that
    snapshot's own maps to the rounding of a sum of ``degree + 1`` terms, which is
    ``64 eps`` of the map's scale here. Half way between two positions the synthesis
    departs from a fresh solve's field by at most the path's measured held-out error
    plus the frequency interpolation's, relative to the map's largest entry, and the
    projection -- which carries the area weights of the moving rule and the inverse
    Gram -- by the same plus the fresh solve's Gram defect. The expanded basis is
    orthonormal there to the path's measured Gram departure.
    """
    grid, path, tables = moving_tables
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    column = int(np.argmin(abs(grid.omega - grid.omega0)))
    assert tables.synthesis is not None and tables.projection is not None
    terms = tables.synthesis.reshape(-1, *tables.synthesis.shape[2:])
    projection = tables.projection.reshape(-1, *tables.projection.shape[2:])
    scale = float(np.max(abs(terms[..., column])))
    for position, table in zip(path.positions[0], path.section_tables(0), strict=True):
        weights = transform_weights(tables, float(position))
        assert weights is not None
        synthesised = np.tensordot(weights, terms[..., column], axes=1)
        sample = table.evaluate(float(grid.omega[column]), float(position))
        assert (
            float(np.max(abs(synthesised - sample.electric)))
            < 64 * np.finfo(float).eps * scale * 16
        )
    held = float(0.5 * (path.positions[0, 1] + path.positions[0, 2]))
    rule = path_rule(guide.field_provider, sampling.nodes, held)
    fresh = path_snapshot(guide.field_provider, sampling.omega, held, rule)
    sample = fresh.evaluate(float(grid.omega[column]), held)
    weights = transform_weights(tables, held)
    assert weights is not None
    synthesised = np.tensordot(weights, terms[..., column], axes=1)
    projected = np.tensordot(weights, projection[..., column], axes=1)
    budget = sampling.path.z_error + sampling.profile_error
    assert float(np.max(abs(synthesised - sample.electric))) <= budget * scale
    expected = _projection_map(sample, rule.weights)
    assert float(np.max(abs(projected - expected))) <= (
        budget + fresh.gram_defect
    ) * float(np.max(abs(expected)))


def test_the_record_stores_its_sections_edges_and_one_region_per_node(moving_tables):
    """The flat record of a moving guide: its edges, its rule at each, its maps."""
    grid, path, tables = moving_tables
    assert tables.z_edges.tolist() == path.z_edges.tolist()
    assert tables.synthesis is not None
    assert tables.synthesis.shape[:2] == (1, path.degree + 1)
    assert not tables.frozen_rule
    assert np.array_equal(tables.region_index, path.tables[0].nodes.region_index)
    assert transform_section(tables, 0.5 * LENGTH) == 0
    assert transform_section(tables, 2 * LENGTH) == 0
    with pytest.raises(ModeError, match="run past its end"):
        path_transform_tables(path, grid.omega, z_end=2 * LENGTH, profile_error=0.0)


def test_the_weights_are_absent_for_maps_that_do_not_move(guide):
    """A snapshot's record, and no record at all, carry no transform weights."""
    grid = grin_grid()
    sampling = guide.sampling
    assert sampling is not None
    provider = grin_provider(elements=32)
    table = sample_modal_table(
        provider,
        sampling.omega,
        np.array([0.0]),
        provider.node_rule(float(sampling.omega[0]), n_r=2),
    )
    constant = modal_transform_tables(
        table, grid.omega, profile_error=0.0, z=[0.0, LENGTH]
    )
    assert transform_weights(constant, 0.5 * LENGTH) is None
    assert transform_weights(None, 0.5 * LENGTH) is None
    bare = dataclasses.replace(constant)
    object.__setattr__(bare, "synthesis", None)
    assert transform_weights(bare, 0.5 * LENGTH) is None


def test_the_section_coefficients_are_the_chebyshev_interpolant():
    """Samples at the Lobatto points become coefficients chebval reproduces."""
    x = -np.cos(np.pi * np.arange(5) / 4)
    values = (x**3 - 2 * x + 1j)[None, :]
    coefficients = section_coefficients(values)
    reproduced = chebyshev_basis.chebval(0.3, coefficients[0])
    assert abs(reproduced - (0.3**3 - 0.6 + 1j)) < 1e-14


def test_knot_values_are_carried_onto_the_run_s_bins_and_zero_outside():
    """The fields' own cubic in frequency, exactly zero outside the band."""
    knots_ = np.linspace(1.0, 2.0, 5)
    values = (knots_[:, None, None] ** 2) * np.ones((5, 2, 2))
    omega = np.array([0.5, 1.5, 2.5])
    inside = (omega >= 1.0) & (omega <= 2.0)
    carried = in_frequency(knots_, values.astype(complex), omega, inside)
    assert carried.shape == (2, 2, 3)
    assert carried[0, 0, 1] == pytest.approx(2.25)
    assert carried[0, 0, 0] == 0.0 and carried[0, 0, 2] == 0.0
    none = in_frequency(knots_, values.astype(complex), omega, np.zeros(3, bool))
    assert not np.any(none)


def test_a_band_connection_is_the_single_frequency_one_at_its_lowest_frequency():
    """One stencil for the band, built where modal_connection would build it there.

    At the band's lowest frequency the two build the same rules from the same
    request, so the matrices agree to the rounding of the batched build (16 eps of
    the largest entry); at the others the band's
    rule is the lowest frequency's, a superset of what each needs, and the two agree
    to the connection's own certificate.
    """
    provider = tapered()
    frequencies = knots()[:2]
    links = band_connection(provider, frequencies, 0.5 * LENGTH)
    alone = modal_connection(provider, float(frequencies[0]), 0.5 * LENGTH)
    np.testing.assert_allclose(
        links[0].matrix,
        alone.matrix,
        rtol=0,
        atol=16 * np.finfo(float).eps * float(np.max(abs(alone.matrix))),
    )
    other = modal_connection(provider, float(frequencies[1]), 0.5 * LENGTH)
    scale = float(np.max(abs(other.matrix)))
    assert float(np.max(abs(links[1].matrix - other.matrix))) < 1e-6 * scale
    with pytest.raises(ModeError, match="at least one frequency"):
        band_connection(provider, [], 0.5 * LENGTH)


def test_the_moving_guide_s_expanded_basis_is_orthonormal_between_positions(
    guide, moving_tables
):
    """The Gram departure the path measured is the one the expanded basis has."""
    _, path, _ = moving_tables
    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    assert sampling.path.gram_departure < 1e-3
    held = float(0.5 * (path.positions[0, 0] + path.positions[0, 1]))
    rule = path_rule(guide.field_provider, sampling.nodes, held)
    fresh = path_snapshot(guide.field_provider, sampling.omega, held, rule)
    sample = fresh.samples[0][0]
    metric = flux_product(sample.electric, sample.magnetic, rule.weights)
    assert float(np.max(abs(metric - np.eye(len(metric))))) < 1e-3


def test_an_entrance_rule_the_provider_would_not_build_again_is_refused():
    """A rule asked for at another position is not the entrance's, and is named."""
    provider = tapered()
    frequencies = knots()
    elsewhere = provider.node_rule(float(frequencies[0]), 0.5 * LENGTH, n_r=2)
    with pytest.raises(ModeError, match="returns a different rule"):
        ModalGuide.from_provider(provider, frequencies, nodes=elsewhere)


def test_a_provider_channel_carries_one_table_and_one_area_per_path_position(guide):
    """A moving channel's path rows and its path agree in count, or it is refused."""
    mode = guide.mode_collection.modes[0]
    assert isinstance(mode, ProviderChannelMode)
    with pytest.raises(ModeError, match="one index table and one area per position"):
        dataclasses.replace(mode, path_areas=mode.path_areas[:-1])
    with pytest.raises(ModeError, match="one index table and one area per position"):
        dataclasses.replace(mode.entrance(), path_tables=mode.path_tables)


class Unpaired(Flipped):
    """A provider that declares no reciprocal partners, as a provider may."""

    def solve(self, omega, z, *, nodes):
        sample = self.provider.solve(omega, z, nodes=nodes)
        return dataclasses.replace(
            sample,
            backward_electric=None,
            backward_magnetic=None,
            reciprocal_gram=None,
        )


def test_a_path_without_reciprocal_partners_stores_none_of_them():
    """The record's reciprocal group is optional on a moving guide as on a snapshot."""
    provider = Unpaired(tapered())
    frequencies = knots()
    rule = provider.node_rule(float(frequencies[0]), 0.0, n_r=2)
    path = sample_modal_path(
        provider,  # type: ignore[arg-type]
        frequencies,
        rule,
        np.array([0.0, LENGTH]),
        rtol=1.0,
        degree=1,
    )
    grid = grin_grid()
    tables = path_transform_tables(path, grid.omega, z_end=LENGTH, profile_error=0.0)
    assert tables.reciprocal_gram is None
    assert tables.backward_electric is None and tables.backward_magnetic is None
    assert tables.synthesis is not None and tables.synthesis.shape[:2] == (1, 2)


def test_the_reference_s_maps_refuse_weights_that_do_not_match_them(moving_tables):
    """Weights for maps that do not move, or none for maps that do, are refused."""
    _, _, tables = moving_tables
    assert tables.synthesis is not None and tables.projection is not None
    shape = tables.synthesis.shape
    constant = GeneralMaps(
        tables.synthesis[0, 0],
        tables.projection[0, 0],
        np.zeros((shape[2], 3, 3, 3, 3)),
        tables.source_frequency_factor,
        1,
    )
    assert constant.at(None) is constant
    with pytest.raises(BackendError, match="do not move"):
        constant.at(np.ones(1))
    moving = dataclasses.replace(
        constant,
        synthesis_terms=tables.synthesis.reshape(-1, *shape[2:]),
        projection_terms=tables.projection.reshape(-1, *tables.projection.shape[2:]),
    )
    assert moving.moving
    with pytest.raises(BackendError, match="carries none"):
        moving.at(None)
    weights = transform_weights(tables, 0.0)
    assert weights is not None
    evaluated = moving.at(weights)
    assert not evaluated.moving
    assert np.allclose(
        evaluated.synthesis,
        tables.synthesis[0, 0]
        - tables.synthesis[0, 1]
        + tables.synthesis[0, 2]
        - tables.synthesis[0, 3]
        + tables.synthesis[0, 4],
    )

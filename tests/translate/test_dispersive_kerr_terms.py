# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The dispersive electronic leaf of kernel schema 24 (SEP 6, phase 1).

``responses.dispersive_kerr`` carries each population's multipliers sampled on the
problem's fine spectral layout at absolute frequency, with its density; these tests hold
the leaf's shapes and flags to the grid, the refusals the translation layer makes on
its way, its validation, and the region notice it logs once per translation.
"""

import dataclasses
import logging

import numpy as np
import pytest

from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.physdata import Mixture
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.modal import BUFFERS_PER_NODE
from solprop.translate.problem import (
    SCHEMA_VERSION,
    _dispersive_chunk,
    assemble,
    dispersive_node_buffers,
)
from solprop.translate.responses import ResponseStage, compile_responses
from solprop.translate.stage import scalars

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

ARGON = OptmatKerr("Ar").dispersive_model()
HELIUM = OptmatKerr("He").dispersive_model()


@pytest.fixture(scope="module")
def field_grid():
    return FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)


@pytest.fixture(scope="module")
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)


@pytest.fixture(scope="module")
def thg_envelope_grid():
    return EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15, thg=True)


@pytest.fixture(scope="module")
def mode():
    return MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)


MIXTURE = Mixture.fill({"He": 0.5, "Ar": 0.5}, 1.0)


def compile_for(grid, mode, responses, **overrides):
    request = {"density": Constant(2.5e25), "z_max": 1.0} | overrides
    return compile_responses(responses, grid, mode, **request)


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("thg", [True, False])
def test_the_leaf_is_sampled_on_the_field_grids_one_sided_layout(
    field_grid, mode, evaluator, thg
):
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    compiled = compile_for(field_grid, mode, [response])
    leaf = compiled.dispersive_kerr
    assert leaf is not None
    assert compiled.kerr_coefficients is None
    assert compiled.response_order == (
        ResponseStage.DISPERSIVE_KERR,
        ResponseStage.ACCUMULATE,
    )
    n_omega_fine = field_grid.n_to // 2 + 1
    assert leaf.filters.shape == (1, n_omega_fine)
    assert leaf.sector == ("full" if thg else "conserving")
    assert not leaf.carrier
    assert not compiled.harmonic_carrier
    assert compiled.electronic_thg is thg
    assert not leaf.positive[0] and not leaf.positive[-1]
    assert leaf.coefficients == (response.coefficient,)
    expected = response.weights(field_grid.omega_fine, kind="field")
    np.testing.assert_array_equal(leaf.filters[0], expected.filter)
    if evaluator == "polynomial":
        assert leaf.outputs is not None and leaf.products is not None
        np.testing.assert_array_equal(leaf.outputs[0], expected.output)
        assert tuple(leaf.products[0]) == expected.products
    else:
        assert leaf.outputs is None and leaf.products is None


def test_the_envelope_full_cubic_needs_the_carrier_and_room(
    envelope_grid, thg_envelope_grid, mode
):
    full = DispersiveKerrResponse(HELIUM, "symmetric", thg=True)
    with pytest.raises(TranslationError, match="oversampled by at least 6"):
        compile_for(envelope_grid, mode, [full])
    compiled = compile_for(thg_envelope_grid, mode, [full])
    leaf = compiled.dispersive_kerr
    assert leaf is not None
    assert leaf.carrier and compiled.harmonic_carrier
    assert leaf.filters.shape == (1, thg_envelope_grid.n_to)
    conserving = compile_for(
        envelope_grid, mode, [DispersiveKerrResponse(HELIUM, "symmetric", thg=False)]
    )
    assert conserving.dispersive_kerr is not None
    assert not conserving.dispersive_kerr.carrier


def test_each_species_is_its_own_channel(field_grid, mode):
    responses = [
        DispersiveKerrResponse(ARGON, "polynomial", species="Ar"),
        DispersiveKerrResponse(HELIUM, "polynomial", species="He"),
    ]
    compiled = compile_for(field_grid, mode, responses, mixture=MIXTURE)
    leaf = compiled.dispersive_kerr
    assert leaf is not None
    assert leaf.labels == ("He", "Ar")
    assert leaf.coefficients == (responses[1].coefficient, responses[0].coefficient)
    assert not np.array_equal(leaf.filters[0], leaf.filters[1])


def test_a_population_takes_one_electronic_response(field_grid, mode):
    with pytest.raises(TranslationError, match="two electronic responses"):
        compile_for(
            field_grid,
            mode,
            [
                KerrResponse(1e-51, species="Ar"),
                DispersiveKerrResponse(ARGON, "polynomial", species="Ar"),
            ],
            mixture=MIXTURE,
        )
    mixed = compile_for(
        field_grid,
        mode,
        [
            KerrResponse(1e-52, species="He"),
            DispersiveKerrResponse(ARGON, "polynomial", species="Ar"),
        ],
        mixture=MIXTURE,
    )
    assert mixed.response_order == (
        ResponseStage.KERR,
        ResponseStage.DISPERSIVE_KERR,
        ResponseStage.ACCUMULATE,
    )


def test_the_electronic_responses_agree_on_thg_and_evaluator(field_grid, mode):
    with pytest.raises(TranslationError, match="must agree on thg"):
        compile_for(
            field_grid,
            mode,
            [
                KerrResponse(1e-52, thg=False, species="He"),
                DispersiveKerrResponse(ARGON, "polynomial", species="Ar"),
            ],
            mixture=MIXTURE,
        )
    with pytest.raises(TranslationError, match="agree on their evaluator"):
        compile_for(
            field_grid,
            mode,
            [
                DispersiveKerrResponse(HELIUM, "symmetric", species="He"),
                DispersiveKerrResponse(ARGON, "polynomial", species="Ar"),
            ],
            mixture=MIXTURE,
        )


@pytest.mark.parametrize("thg", [True, False])
def test_a_field_grid_raman_response_agrees_with_the_dispersive_one(
    field_grid, mode, thg
):
    """The drive form is the electronic response's harmonic sector, as for Kerr."""
    raman = response_for("SiO2", amplitude=1e-22, fraction=0.18, thg=thg)
    agreeing = compile_for(
        field_grid,
        mode,
        [DispersiveKerrResponse(ARGON, "symmetric", thg=thg), raman],
        density=Constant(1.0),
    )
    assert agreeing.raman_terms[0].thg is thg
    with pytest.raises(TranslationError, match="must agree"):
        compile_for(
            field_grid,
            mode,
            [DispersiveKerrResponse(ARGON, "symmetric", thg=not thg), raman],
            density=Constant(1.0),
        )


def problem_for(grid, mode, responses):
    start = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    return assemble(
        grid,
        mode,
        responses,
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=1e-3,
    )


def test_the_problem_carries_the_leaf_and_its_stage_row(field_grid, mode):
    problem = problem_for(
        field_grid, mode, [DispersiveKerrResponse(ARGON, "polynomial")]
    )
    assert problem.schema_version == SCHEMA_VERSION == 39
    assert problem.grids.carrier_2w0 is None
    bundle = scalars(problem, 0.0, 0.05)
    expected = float(mode.density(np.array([0.05]))[0])
    (density,) = bundle.dispersive_kerr_densities
    assert density == pytest.approx(expected, rel=1e-10)
    constant = problem_for(field_grid, mode, [KerrResponse(1e-51)])
    assert constant.responses.dispersive_kerr is None
    assert scalars(constant, 0.0, 0.05).dispersive_kerr_densities == ()


def test_validation_binds_the_leaf_to_the_grid(
    field_grid, envelope_grid, thg_envelope_grid, mode
):
    problem = problem_for(
        thg_envelope_grid, mode, [DispersiveKerrResponse(ARGON, "polynomial")]
    )
    assert problem.grids.carrier_2w0 is not None
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    spoiled = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, dispersive_kerr=dataclasses.replace(leaf, carrier=False)
        ),
    )
    with pytest.raises(TranslationError, match="carrier"):
        spoiled.validate()
    conserving = DispersiveKerrResponse(ARGON, "polynomial", thg=False)
    envelope_problem = problem_for(envelope_grid, mode, [conserving])
    field_leaf = compile_for(field_grid, mode, [conserving]).dispersive_kerr
    misplaced = dataclasses.replace(
        envelope_problem,
        responses=dataclasses.replace(
            envelope_problem.responses, dispersive_kerr=field_leaf
        ),
    )
    with pytest.raises(TranslationError, match="sampled on"):
        misplaced.validate()


def test_translation_logs_the_regions_once(caplog):
    """Section 4.10: a band beyond a gas's support is logged, never refused."""
    grid = FieldGrid.from_limits(0.1, 800e-9, (90e-9, 4e-6), 100e-15)
    helium = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    with caplog.at_level(logging.INFO, logger="solprop.translate.responses"):
        problem_for(grid, helium, [DispersiveKerrResponse(HELIUM, "symmetric")])
    notices = [
        record.getMessage()
        for record in caplog.records
        if "dispersive electronic response" in record.getMessage()
    ]
    assert len(notices) == 1
    assert "He: band" in notices[0] and "best effort" in notices[0]
    caplog.clear()
    narrow = FieldGrid.from_limits(0.1, 800e-9, (400e-9, 2e-6), 400e-15)
    with caplog.at_level(logging.INFO, logger="solprop.translate.responses"):
        problem_for(narrow, helium, [DispersiveKerrResponse(HELIUM, "symmetric")])
    assert not [
        record
        for record in caplog.records
        if "dispersive electronic response" in record.getMessage()
    ]


@pytest.mark.parametrize(
    ("responses", "extra"),
    [
        ([DispersiveKerrResponse(ARGON, "symmetric")], 0),
        ([DispersiveKerrResponse(ARGON, "polynomial")], 2),
        ([DispersiveKerrResponse(ARGON, "symmetric"), "raman"], 1),
        ([DispersiveKerrResponse(ARGON, "polynomial"), "raman"], 2),
    ],
)
def test_node_admission_counts_the_dispersive_node_buffers(
    field_grid, mode, responses, extra
):
    """The fused path's chunk holds the response's node buffers beside the pass's.

    A lone symmetric response of one species shares the node field buffer; beside
    another response it holds one of its own, and the polynomial two, whatever stands
    beside it. ``_raman_chunk`` already sizes the Raman histories, so the budget here
    is the dispersive admission's own.
    """
    raman = response_for("SiO2", amplitude=1e-22, fraction=0.18, thg=True)
    declared = [raman if response == "raman" else response for response in responses]
    start = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(field_grid)
    problem = assemble(
        field_grid,
        [mode],
        declared,
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=1e-3,
        quadrature=transverse_quadrature([mode], n_r=8),
        density=Constant(2.5e25),
    )
    transform = dataclasses.replace(
        problem.transform, node_chunk=problem.transform.n_nodes
    )
    assert transform.n_nodes > 2
    per_node = (BUFFERS_PER_NODE + extra) * field_grid.n_to * 16
    assert (
        _dispersive_chunk(
            transform, problem.responses, 1, field_grid.n_to, 2 * per_node
        ).node_chunk
        == 2
    )
    with pytest.raises(TranslationError, match="dispersive electronic response"):
        _dispersive_chunk(
            transform, problem.responses, 1, field_grid.n_to, per_node - 1
        )
    without = problem_for(field_grid, mode, [KerrResponse(1e-51)])
    assert (
        _dispersive_chunk(without.transform, without.responses, 1, field_grid.n_to, 1)
        is without.transform
    )


@pytest.mark.parametrize(("evaluator", "extra"), [("symmetric", 0), ("polynomial", 2)])
def test_node_admission_counts_a_symmetric_mixture_as_one_product(
    field_grid, mode, evaluator, extra
):
    """Helium and argon: under its mixture filter a symmetric mixture is one product.

    So, alone, it shares the node field buffer as one species does; the polynomial's
    two node buffers do not depend on its groups.
    """
    compiled = compile_for(
        field_grid,
        mode,
        [
            DispersiveKerrResponse(HELIUM, evaluator, species="He"),
            DispersiveKerrResponse(ARGON, evaluator, species="Ar"),
        ],
        mixture=MIXTURE,
    )
    leaf = compiled.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    start = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(field_grid)
    problem = assemble(
        field_grid,
        [mode],
        [KerrResponse(1e-51)],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=1e-3,
        quadrature=transverse_quadrature([mode], n_r=8),
    )
    transform = dataclasses.replace(
        problem.transform, node_chunk=problem.transform.n_nodes
    )
    per_node = (BUFFERS_PER_NODE + extra) * field_grid.n_to * 16
    admitted = _dispersive_chunk(transform, compiled, 1, field_grid.n_to, 2 * per_node)
    assert admitted.node_chunk == 2
    with pytest.raises(TranslationError, match="dispersive electronic response"):
        _dispersive_chunk(transform, compiled, 1, field_grid.n_to, per_node - 1)


@pytest.mark.parametrize(
    ("evaluator", "products"), [("symmetric", 1), ("polynomial", 3)]
)
def test_a_free_column_holds_the_products_in_time_and_frequency(evaluator, products):
    """A Cartesian column adds its products and their spectra, twice per product."""
    grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 8e-5
    )
    problem = assemble(
        grid,
        None,
        [DispersiveKerrResponse(ARGON, evaluator)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        density=Constant(2.5e25),
    )
    nodes = 0 if evaluator == "symmetric" else 2
    per_column = (20 + nodes + 2 * products) * grid.n_to * 16
    assert (
        _dispersive_chunk(
            problem.transform, problem.responses, 1, grid.n_to, 3 * per_column
        ).node_chunk
        == 3
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"evaluator": "cubic"}, "is not an evaluator"),
        ({"sector": "half"}, "is not a sector"),
        ({"labels": ()}, "unique channel labels"),
        ({"labels": ("Ar", "Ar")}, "unique channel labels"),
        ({"coefficients": ()}, "one coefficient and one density per channel"),
        ({"filters": "short"}, "filters has shape"),
        ({"outputs": None}, "exactly for the polynomial evaluator"),
        ({"outputs": "short"}, "outputs has the wrong shape"),
        ({"products": "short"}, "products has the wrong shape"),
        ({"regions": "negative"}, "non-negative int32 node region"),
    ],
)
def test_a_malformed_leaf_is_refused_by_field(field_grid, mode, change, match):
    leaf = compile_for(
        field_grid, mode, [DispersiveKerrResponse(ARGON, "polynomial")]
    ).dispersive_kerr
    assert leaf is not None
    assert leaf.outputs is not None
    assert leaf.products is not None
    # Shapes the runtime type hook admits (its axis names are independent per
    # field) but the leaf's own validation, which also runs in production, does not.
    malformed = {
        "filters": leaf.filters[:, :-1],
        "outputs": leaf.outputs[:, :, :-1],
        "products": np.concatenate([leaf.products, leaf.products]),
        "regions": np.full(1, -1, dtype=np.int32),
    }
    fields = {
        name: malformed[name] if value in ("short", "negative") else value
        for name, value in change.items()
    }
    with pytest.raises(TranslationError, match=match):
        dataclasses.replace(leaf, **fields)


def test_a_problem_without_an_electronic_term_has_no_harmonic_form(field_grid, mode):
    """The field grid's harmonic form is read off the electronic term; none, none."""
    assert compile_for(field_grid, mode, []).electronic_thg is None


def test_a_dispersive_response_beside_a_guides_own_channels_is_refused(
    field_grid, mode
):
    """A general modal guide's gas regions are its dispersive channels, alone."""
    response = DispersiveKerrResponse(ARGON, "symmetric")
    regional = compile_for(field_grid, mode, [response]).dispersive_kerr
    with pytest.raises(TranslationError, match="declared beside them"):
        compile_for(field_grid, mode, [response], regional_dispersive=regional)


def test_validation_binds_the_leaf_carrier_to_its_sector(thg_envelope_grid, mode):
    """The leaf's carrier flag follows its sector, beside a constant harmonic too.

    A constant harmonic Kerr term keeps the grid's harmonic carrier, so a full-sector
    leaf that drops its own flag passes the grid's check and is caught by the leaf's.
    """
    problem = problem_for(thg_envelope_grid, mode, [KerrResponse(1e-51, thg=True)])
    leaf = compile_for(
        thg_envelope_grid, mode, [DispersiveKerrResponse(ARGON, "polynomial")]
    ).dispersive_kerr
    assert leaf is not None and leaf.carrier
    spoiled = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, dispersive_kerr=dataclasses.replace(leaf, carrier=False)
        ),
    )
    with pytest.raises(TranslationError, match="full sector on an envelope grid"):
        spoiled.validate()


def test_validation_binds_the_leaf_to_its_route_and_populations(field_grid, mode):
    problem = problem_for(
        field_grid, mode, [DispersiveKerrResponse(ARGON, "symmetric", thg=False)]
    )
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None

    def spoiled(**change):
        return dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses, dispersive_kerr=dataclasses.replace(leaf, **change)
            ),
        )

    regions = np.zeros(1, dtype=np.int32)
    regions.setflags(write=False)
    with pytest.raises(TranslationError, match="exactly on the general modal route"):
        spoiled(regions=regions).validate()
    with pytest.raises(TranslationError, match="unknown populations"):
        spoiled(labels=("Xe",)).validate()


def test_a_problem_without_the_leaf_adds_no_node_buffers(field_grid, mode):
    compiled = compile_for(field_grid, mode, [KerrResponse(1e-51)])
    assert compiled.dispersive_kerr is None
    assert dispersive_node_buffers(compiled, spatial=False) == 0
    assert dispersive_node_buffers(compiled, spatial=True) == 0

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses

import numpy as np
import pytest

from solprop import fourier
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.modes.quadrature import mode_matrix, taper_scale
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.normalisation import mode_average_prefactor, sqrtw_to_vm
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import Linear, PressureGradient
from solprop.pulses import SechPulse
from solprop.stepper import PIController
from solprop.translate.modal import (
    BUFFERS_PER_NODE,
    DEFAULT_NODE_BUDGET,
    Transform,
    mode_averaged_transform,
    multimode_transform,
)
from solprop.translate.problem import assemble
from solprop.translate.responses import compile_responses

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

# The soliton parameters of tests/driver/test_scaffold.py.
BETA2 = -11.83e-27
GAMMA = 0.11
TAU0 = 100e-15
WAVELENGTH = 835e-9


@pytest.fixture(scope="module")
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12)


@pytest.fixture(scope="module")
def gnlse(envelope_grid):
    return problem_from(
        [0.0, 0.0, BETA2], omega_ref=envelope_grid.omega0, gamma=GAMMA, shock=False
    )


@pytest.fixture(scope="module")
def soliton(envelope_grid, gnlse):
    peak = abs(BETA2) / (GAMMA * TAU0**2)
    return SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(
        envelope_grid
    )


@pytest.fixture(scope="module")
def soliton_problem(envelope_grid, gnlse, soliton):
    return assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        soliton,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )


def scaffold_nonlinear(problem, grid, state):
    """The nonlinear term as `solprop.driver.scaffold` composes it, verbatim.

    Five lines of plain NumPy with no windows and no fused passes: the reference the
    compiled specification has to reproduce.
    """
    n_t, n_to = grid.n_t, grid.n_to
    to_field = sqrtw_to_vm(problem.aeff)
    prefactor = mode_average_prefactor(
        problem.mode, grid, shock=problem.shock, index=problem.prefactor_index
    )
    fine = fourier.oversample(state, n_to, kind="envelope")
    envelope = fourier.inverse(fine, kind="envelope") * to_field
    polarisation = problem.kerr.evaluate(envelope[np.newaxis], 1.0, kind="envelope")
    transformed = fourier.forward(polarisation[0], kind="envelope")
    return prefactor * fourier.crop(transformed, n_t, kind="envelope")


def compiled_nonlinear(problem, state, *, window=True):
    """Evaluate the compiled specification the way a backend's passes will.

    Pad, inverse transform, synthesis, the fused Kerr pass, projection, forward
    transform, crop and prefactor -- `docs/dev/architecture.md` section 8, without the
    fusion. Mode-averaged and scalar, which is all this reference covers.
    """
    grids, transform, responses = problem.grids, problem.transform, problem.responses
    n_omega_fine = fourier.n_omega_for(grids.n_to, problem.grid_kind)
    fine = np.zeros(n_omega_fine, dtype=complex)
    fine[grids.crop_index] = state
    time = fourier.inverse(fine, kind=problem.grid_kind)
    nodes = transform.synthesis @ time[np.newaxis]
    kerr = responses.kerr_coefficients
    density = float(responses.density(0.0))
    if kerr.analytic_signal:
        nodes = fourier.analytic_signal(nodes)
    if kerr.cubic_form == "real":
        polarisation = (
            kerr.coefficient * density * np.sum(nodes**2, axis=0, keepdims=True) * nodes
        )
    else:
        polarisation = (
            kerr.coefficient
            * density
            * (
                kerr.self_factor
                * np.sum(np.abs(nodes) ** 2, axis=0, keepdims=True)
                * nodes
                + kerr.conjugate_factor
                * np.sum(nodes**2, axis=0, keepdims=True)
                * np.conj(nodes)
            )
        )
    if kerr.take_real:
        polarisation = polarisation.real
    if window:
        polarisation = polarisation * grids.t_window_fine
    projected = transform.projection @ polarisation
    spectrum = fourier.forward(projected, kind=problem.grid_kind)
    # The soliton's fibre does not vary along z, so the prefactor is the array of the
    # fast path rather than a Clenshaw pass over its table.
    return responses.prefactor_constant[:, 0] * spectrum[..., grids.crop_index]


# ------------------------------------------------ the mode-averaged transform


def test_the_compiled_term_reproduces_the_scaffold(
    soliton_problem, envelope_grid, gnlse, soliton
):
    """The point of the translation layer: the same numbers, differently arranged."""
    state = soliton[0, 0] * envelope_grid.active_mask
    reference = scaffold_nonlinear(gnlse, envelope_grid, state)
    compiled = compiled_nonlinear(soliton_problem, state, window=False)[0]
    scale = float(np.max(np.abs(reference)))
    # The two differ only in where the pad scale and the 1/N of the transforms are
    # applied, so the gap is the round-off of one extra multiplication over a reduction
    # of length n_to: Higham's n*eps is 8192 * 2.2e-16 = 1.8e-12 relative to the largest
    # term, and the measured gap is far below it.
    assert float(np.max(np.abs(compiled - reference))) < 1e-15 * scale


def test_the_temporal_window_changes_nothing_for_a_pulse_inside_it(
    soliton_problem, soliton, envelope_grid
):
    """The compiled pass applies the collar; the scaffold does not, and here it cannot
    matter, because a 100 fs soliton in a 4 ps window has nothing in the collar."""
    state = soliton[0, 0] * envelope_grid.active_mask
    windowed = compiled_nonlinear(soliton_problem, state, window=True)
    bare = compiled_nonlinear(soliton_problem, state, window=False)
    assert float(np.max(np.abs(windowed - bare))) < 1e-15 * float(np.max(np.abs(bare)))


def test_the_synthesis_carries_the_conversion_and_the_pad_scale(
    soliton_problem, envelope_grid, gnlse
):
    expected = sqrtw_to_vm(gnlse.aeff) * envelope_grid.pad_scale
    assert float(soliton_problem.transform.synthesis[0, 0]) == expected
    assert soliton_problem.transform.projection.tolist() == [[1.0]]
    assert soliton_problem.transform.projection_coarse.tolist() == [[1.0]]


def test_a_constant_effective_area_needs_no_taper_table(soliton_problem, gnlse):
    assert soliton_problem.transform.taper_scale is None
    assert float(soliton_problem.transform.aeff(0.5)) == pytest.approx(gnlse.aeff)


def test_a_tapered_mode_averaged_run_tabulates_its_rescaling(envelope_grid):
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 0.8), "He", 1.0, cladding=1.45)
    transform = mode_averaged_transform(mode, envelope_grid, z_max=1.0)
    assert transform.taper_scale is not None
    for z in (0.0, 0.4, 0.8, 1.0):
        # sqrt(A_eff(0)/A_eff(z)) is a(0)/a(z) for a scale-invariant mode, which is
        # what modes.quadrature.taper_scale returns; the two mechanisms are one.
        assert float(transform.taper_scale(z)) == pytest.approx(
            taper_scale(mode, z), rel=1e-12
        )
        # And the rescaled synthesis is the conversion at that position.
        rescaled = float(transform.synthesis[0, 0]) * float(transform.taper_scale(z))
        assert rescaled == pytest.approx(
            sqrtw_to_vm(mode.aeff(z)) * envelope_grid.pad_scale, rel=1e-12
        )


def test_the_field_grid_kerr_form_reproduces_the_reference(envelope_grid):
    """On a field grid with the harmonic the cubic is the real one, not the analytic."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    kerr = KerrResponse(chi3=2.2e-52, thg=True)
    compiled = compile_responses(
        [kerr], grid, mode, density=mode.density, z_max=1.0
    ).kerr_coefficients
    assert compiled is not None
    rng = np.random.default_rng(11)
    field = rng.normal(size=(1, grid.n_to)) * 1e9
    density = float(mode.density(np.array([0.0]))[0])
    reference = kerr.evaluate(field, density, kind="field")
    got = (
        compiled.coefficient * density * np.sum(field**2, axis=0, keepdims=True) * field
    )
    # The same cubic through another arrangement of its four products (the sum is
    # over one component): 8 eps per element.
    np.testing.assert_allclose(got, reference, rtol=8 * np.finfo(float).eps, atol=0)


# ------------------------------------------------------- the multimode transform


@pytest.fixture(scope="module")
def mode_set():
    return [MarcatiliMode(125e-6, "He", 1.0, m=m, cladding=1.45) for m in (1, 2, 3)]


@pytest.fixture(scope="module")
def rule(mode_set):
    return transverse_quadrature(mode_set, n_r=12, radial_rule="gauss_kronrod")


def test_the_multimode_matrices_are_the_mode_module_s_own(
    mode_set, rule, envelope_grid
):
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x",)
    )
    matrices = mode_matrix(mode_set, rule, components=("x",))
    assert np.array_equal(
        transform.synthesis, matrices.synthesis * envelope_grid.pad_scale
    )
    assert np.array_equal(transform.projection, matrices.projection)
    assert np.array_equal(transform.projection_coarse, matrices.projection_coarse)
    assert transform.n_modes == 3
    assert transform.n_nodes == rule.n_nodes
    assert not transform.mode_averaged


def test_a_one_node_quadrature_is_an_ordinary_modal_transform(mode_set, envelope_grid):
    """The kind is stored, not inferred from the node count (kernel schema 26).

    Before it was stored, any transform with one node and no geometry read as
    mode-averaged, so a one-node rule had the effective-area conversion folded into its
    prefactor on top of its own weights.
    """
    two = transverse_quadrature(mode_set[:1], n_r=2)
    assert two.w_r is not None and two.w_r_coarse is not None
    one = dataclasses.replace(
        two, n_r=1, xi_r=two.xi_r[:1], w_r=two.w_r[:1], w_r_coarse=two.w_r_coarse[:1]
    )
    transform = multimode_transform(
        mode_set[:1], one, envelope_grid, z_max=1.0, components=("x",)
    )
    assert transform.n_nodes == 1
    assert transform.kind == "modal"
    assert not transform.mode_averaged
    assert transform.node_weights is not None


def test_the_mode_averaged_route_declares_its_kind(gnlse, envelope_grid):
    transform = mode_averaged_transform(gnlse.mode, envelope_grid, z_max=1.0)
    assert transform.kind == "mode_averaged"
    assert transform.mode_averaged


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kind": "radial"}, "kind must be 'mode_averaged' or 'modal'"),
        ({"kind": "mode_averaged"}, "a mode-averaged transform has one node"),
    ],
)
def test_a_kind_that_contradicts_the_transform_is_refused(
    mode_set, rule, envelope_grid, change, message
):
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x",)
    )
    with pytest.raises(TranslationError, match=message):
        dataclasses.replace(transform, **change)


def test_a_vector_run_carries_both_components(mode_set, rule, envelope_grid):
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x", "y")
    )
    assert transform.synthesis.shape == (2 * rule.n_nodes, 3)
    assert transform.n_nodes == rule.n_nodes


def test_a_taper_rescales_the_matrices_rather_than_rebuilding_them(rule, envelope_grid):
    modes = [
        MarcatiliMode(Linear(125e-6, 75e-6, 0.8), "He", 1.0, m=m, cladding=1.45)
        for m in (1, 2, 3)
    ]
    taper = transverse_quadrature(modes, n_r=12, radial_rule="gauss_kronrod")
    transform = multimode_transform(
        modes, taper, envelope_grid, z_max=1.0, components=("x",)
    )
    assert transform.taper_scale is not None
    for z in (0.2, 0.8, 1.0):
        scale = float(transform.taper_scale(z))
        rebuilt = mode_matrix(modes, taper, z, components=("x",))
        # ModeMatrices.rescaled is exact for a scale-invariant set: the normalised field
        # goes as 1/a at nodes placed at r = a xi and the weights as a^2. The gap is the
        # round-off of the rescaling against a fresh evaluation of the Bessel profiles.
        np.testing.assert_allclose(
            transform.synthesis * scale,
            rebuilt.synthesis * envelope_grid.pad_scale,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            transform.projection / scale, rebuilt.projection, rtol=1e-12
        )


def test_a_set_that_cannot_be_rescaled_is_refused(envelope_grid):
    """A cross-section that changes shape needs new matrices, which nothing carries."""

    class Squashed(MarcatiliMode):
        @property
        def scale_invariant(self):
            return False

    modes = [Squashed(Linear(125e-6, 75e-6, 0.8), "He", 1.0, cladding=1.45)]
    rule = transverse_quadrature(modes, n_r=8)
    with pytest.raises(TranslationError, match="not scale invariant"):
        multimode_transform(modes, rule, envelope_grid, z_max=1.0, components=("x",))


def test_an_empty_mode_set_is_refused(rule, envelope_grid):
    with pytest.raises(TranslationError, match="at least one mode"):
        multimode_transform([], rule, envelope_grid, z_max=1.0)


# ------------------------------------------------------------------------ node chunks


def test_a_small_problem_is_not_chunked(mode_set, rule, envelope_grid):
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x",)
    )
    assert transform.node_chunk == rule.n_nodes


def test_the_chunk_follows_the_memory_budget(mode_set, rule, envelope_grid):
    per_node = BUFFERS_PER_NODE * 1 * envelope_grid.n_to * 16
    transform = multimode_transform(
        mode_set,
        rule,
        envelope_grid,
        z_max=1.0,
        components=("x",),
        node_budget=4 * per_node,
    )
    assert transform.node_chunk == 4


def test_the_chunk_never_falls_below_one(mode_set, rule, envelope_grid):
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x",), node_budget=1
    )
    assert transform.node_chunk == 1


def test_a_budget_of_nothing_is_refused(mode_set, rule, envelope_grid):
    with pytest.raises(TranslationError, match="positive number of bytes"):
        multimode_transform(
            mode_set, rule, envelope_grid, z_max=1.0, components=("x",), node_budget=0
        )


def test_the_default_budget_is_stated_in_bytes():
    assert DEFAULT_NODE_BUDGET == 64 * 2**20


# ------------------------------------------------------------------------ validation


def test_a_projection_that_is_not_the_transpose_is_refused(soliton_problem):
    transform = soliton_problem.transform
    with pytest.raises(TranslationError, match="the transpose of synthesis"):
        Transform(
            synthesis=np.ones((2, 1)),
            projection=np.ones((1, 3)),
            projection_coarse=np.ones((1, 2)),
            taper_scale=None,
            node_chunk=1,
            aeff=transform.aeff,
            kind="modal",
        )


@pytest.mark.parametrize("chunk", [0, 3])
def test_a_chunk_outside_the_node_set_is_refused(soliton_problem, chunk):
    transform = soliton_problem.transform
    with pytest.raises(TranslationError, match="node_chunk must lie"):
        Transform(
            synthesis=np.ones((2, 1)),
            projection=np.ones((1, 2)),
            projection_coarse=np.ones((1, 2)),
            taper_scale=None,
            node_chunk=chunk,
            aeff=transform.aeff,
            kind="modal",
        )


@pytest.mark.parametrize(
    ("z_max", "margin", "message"),
    [(0.0, 0.05, "z_max must be positive"), (1.0, -1.0, "margin must be")],
)
def test_a_bad_span_is_refused(envelope_grid, gnlse, z_max, margin, message):
    with pytest.raises(TranslationError, match=message):
        mode_averaged_transform(gnlse.mode, envelope_grid, z_max=z_max, margin=margin)


def test_a_gradient_alone_leaves_the_transform_constant(envelope_grid):
    """A fill changes the index, not the field profile; only a taper moves the nodes."""
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(1.0, 2.0, 0.2), cladding=1.45
    )
    transform = mode_averaged_transform(mode, envelope_grid, z_max=1.0)
    assert transform.taper_scale is None


def test_the_transform_pair_inverts_on_an_orthonormal_set(mode_set, envelope_grid):
    """Projection times synthesis is the pad scale on the identity, and nothing else.

    The property the basis normalisation exists for: the propagated basis is
    ``e_m/sqrt(N_m)``, whose overlaps are ``2/(eps0 c)`` and not one, so the projection
    has to carry ``eps0 c/2`` to invert the synthesis. Without it every multimode run is
    wrong by that constant.
    """
    rule = transverse_quadrature(mode_set, n_r=24, radial_rule="gauss_kronrod")
    transform = multimode_transform(
        mode_set, rule, envelope_grid, z_max=1.0, components=("x",)
    )
    gram = transform.projection @ transform.synthesis / envelope_grid.pad_scale
    # The rule converges spectrally on the quadratic overlap of the Bessel profiles:
    # measured worst deviation from the identity 1.7e-8, 4.5e-13 and 8.9e-16 at n_r =
    # 12, 16 and 24, so by 24 it is the round-off of a 25-term sum (Higham's n * eps is
    # 5.6e-15) and not the rule at all.
    np.testing.assert_allclose(gram, np.eye(len(mode_set)), atol=1e-14)


def test_a_spatial_kind_must_match_its_geometry(envelope_grid):
    from solprop.grid import RadialGrid  # noqa: PLC0415
    from solprop.translate.radial import radial_transform  # noqa: PLC0415

    transform = radial_transform(RadialGrid(1e-3, 8), envelope_grid, n_pol=1, z_max=1.0)
    assert transform.kind == "radial"
    with pytest.raises(TranslationError, match="a radial geometry needs kind 'radial'"):
        dataclasses.replace(transform, kind="free")

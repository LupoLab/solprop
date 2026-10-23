# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The dispersive response's bodies and kernels against the reference's arithmetic.

``backends/numba_cpu/dispersive.py`` forms SEP 6's products at the nodes with the
bodies of ``backends/bodies.py`` and finishes them on the modal columns. Here each
piece is held to what it replaces in
:func:`solprop.nonlinear.responses.dispersive_cubic`: the closed-form scalar bodies to
the general contraction, the general contraction to the reference's ``_isotropic``,
the product kernels to the reference's sector groups, and the small spectral kernels
to plain NumPy. The module runs in the coverage lane's interpreted pass too, so its
arrays are small.

**The bounds.** A body and the reference form the same products in a different
association: a few roundings of numbers of the size of the result, ``8 eps``
relative. The kernels' spectral sums are one product per term, exact to ``4 eps``.
None is bitwise (SEP 25): a compiled kernel may contract its products. What stays
exact is structural: a cleared tail, a skipped region, an unchanged table, a binary
scaling.
"""

import cmath
import dataclasses

import numpy as np
import pytest

from solprop.backends.bodies import PLAIN
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.dispersive import (
    StageWeights,
    add_spectra,
    apply_window,
    combine_spectra,
    dispersive_form,
    dispersive_terms_layout,
    filter_spectra,
    make_dispersive_pairs,
    make_dispersive_self,
    mixture_rows,
    weigh_spectrum,
    weigh_terms,
)
from solprop.backends.numba_cpu.dispersive_general import make_general_dispersive
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    _isotropic,
    mixture_filter,
)
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

EPS = np.finfo(float).eps
FAMILIES = {"plain": PLAIN, "host": HOST}
FORMS = ("real", "conserving", "harmonic")
# Only an analytic signal carries the zero-frequency constant a shifted kernel removes.
SHIFTS = [(form, False) for form in FORMS] + [("conserving", True), ("harmonic", True)]


def random_fields(rng, shape, *, real):
    """Return three random fields of one shape, real or complex, of order one."""
    fields = [rng.standard_normal(shape) for _ in range(3)]
    if real:
        return fields
    return [one + 1j * rng.standard_normal(shape) for one in fields]


def conserving(x, y, z):
    """The reference's number-conserving group, as ``_product_spectrum`` forms it."""
    return (
        _isotropic(x, y, z.conj())
        + _isotropic(x, y.conj(), z)
        + _isotropic(x.conj(), y, z)
    )


def harmonic(x, y, z, carrier):
    """The reference's two harmonic groups, carried by ``exp(+-2 i omega0 t)``."""
    down = (
        _isotropic(x, y.conj(), z.conj())
        + _isotropic(x.conj(), y, z.conj())
        + _isotropic(x.conj(), y.conj(), z)
    )
    return carrier * _isotropic(x, y, z) + carrier.conj() * down


def reference(form, x, y, z, carrier):
    """The reference's product of three fields in a form."""
    if form == "real":
        return _isotropic(x, y, z)
    value = conserving(x, y, z)
    return value + harmonic(x, y, z, carrier) if form == "harmonic" else value


# --- the bodies ---------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_vector_bodies_are_the_reference_contraction(family):
    bodies = FAMILIES[family]
    rng = np.random.default_rng(3)
    x, y, z = random_fields(rng, (2, 1), real=False)
    carrier = cmath.exp(0.7j)
    got = np.array(
        bodies.dispersive_isotropic(
            x[0, 0], x[1, 0], y[0, 0], y[1, 0], z[0, 0], z[1, 0]
        )
    )
    np.testing.assert_allclose(got, _isotropic(x, y, z)[:, 0], rtol=8 * EPS)
    got = np.array(
        bodies.dispersive_conserving(
            x[0, 0], x[1, 0], y[0, 0], y[1, 0], z[0, 0], z[1, 0]
        )
    )
    np.testing.assert_allclose(got, conserving(x, y, z)[:, 0], rtol=8 * EPS)
    got = np.array(
        bodies.dispersive_harmonic(
            x[0, 0], x[1, 0], y[0, 0], y[1, 0], z[0, 0], z[1, 0], carrier
        )
    )
    np.testing.assert_allclose(
        got, harmonic(x, y, z, np.array(carrier))[:, 0], rtol=8 * EPS
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_scalar_bodies_are_the_one_component_contraction(family):
    """The closed forms of ``(u, u, u)``, ``(u, u, f)`` and ``(u, f, f)``."""
    bodies = FAMILIES[family]
    rng = np.random.default_rng(4)
    u, f, _ = random_fields(rng, (1, 1), real=False)
    a, b, _ = random_fields(rng, (1, 1), real=True)
    carrier = np.array(cmath.exp(-1.1j))
    scalar_u, scalar_f = complex(u[0, 0]), complex(f[0, 0])
    real_a, real_b = float(a[0, 0]), float(b[0, 0])
    # The real products are the reference's own grouping, to the bodies' 8 eps.
    expected = {
        "self": _isotropic(a, a, a)[0, 0],
        "pair": _isotropic(a, a, b)[0, 0],
        "cross": _isotropic(a, b, b)[0, 0],
    }
    got = {
        "self": bodies.dispersive_real_self(real_a),
        "pair": bodies.dispersive_real_pair(real_a, real_b),
        "cross": bodies.dispersive_real_cross(real_a, real_b),
    }
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, rel=8 * EPS), name
    cases = {
        "self": (bodies.dispersive_conserving_self(scalar_u), conserving(u, u, u)),
        "pair": (
            bodies.dispersive_conserving_pair(scalar_u, scalar_f),
            conserving(u, u, f),
        ),
        "cross": (
            bodies.dispersive_conserving_cross(scalar_u, scalar_f),
            conserving(u, f, f),
        ),
        "harmonic self": (
            bodies.dispersive_harmonic_self(scalar_u, complex(carrier)),
            harmonic(u, u, u, carrier),
        ),
        "harmonic pair": (
            bodies.dispersive_harmonic_pair(scalar_u, scalar_f, complex(carrier)),
            harmonic(u, u, f, carrier),
        ),
        "harmonic cross": (
            bodies.dispersive_harmonic_cross(scalar_u, scalar_f, complex(carrier)),
            harmonic(u, f, f, carrier),
        ),
    }
    for name, (got, expected) in cases.items():
        assert got == pytest.approx(complex(expected[0, 0]), rel=8 * EPS), name


def test_the_form_follows_the_grid_and_the_sector():
    assert dispersive_form("field", "full") == "real"
    assert dispersive_form("field", "conserving") == "conserving"
    assert dispersive_form("envelope", "conserving") == "conserving"
    assert dispersive_form("envelope", "full") == "harmonic"


# --- the product kernels -------------------------------------------------------------


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize(("form", "shifted"), SHIFTS)
def test_the_self_product_is_the_reference_s(form, shifted, n_pol):
    """Two nodes of eight samples, in place, minus a per-row constant when shifted."""
    rng = np.random.default_rng(5)
    field, _, _ = random_fields(rng, (2 * n_pol, 8), real=form == "real")
    carrier = np.exp(1j * np.linspace(0.0, 3.0, 8))
    offset = rng.standard_normal(2 * n_pol) + 1j * rng.standard_normal(2 * n_pol)
    source = field - offset[:, None] if shifted else field
    nodes = [source[node * n_pol : (node + 1) * n_pol] for node in range(2)]
    expected = np.concatenate([reference(form, u, u, u, carrier) for u in nodes])
    kernel = make_dispersive_self(form, n_pol, parallel=False, shifted=shifted)
    buffer = field.copy()
    kernel(buffer, buffer, carrier, offset if shifted else np.zeros(1, complex))
    np.testing.assert_allclose(
        buffer, expected, rtol=0, atol=16 * EPS * np.abs(expected).max()
    )


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize(("form", "shifted"), SHIFTS)
def test_the_polynomial_products_are_the_reference_s(form, shifted, n_pol):
    """``T(E, E, F)`` and ``c0 T(E, E, H) + c1 T(E, F, F)``, in place."""
    rng = np.random.default_rng(6)
    field, first, second = random_fields(rng, (2 * n_pol, 8), real=form == "real")
    carrier = np.exp(1j * np.linspace(0.0, 3.0, 8))
    offset = rng.standard_normal(2 * n_pol) + 1j * rng.standard_normal(2 * n_pol)
    bare = field - offset[:, None] if shifted else field
    c0, c1 = 0.75, -1.25
    pairs, weighted = [], []
    for node in range(2):
        rows = slice(node * n_pol, (node + 1) * n_pol)
        e, f, h = bare[rows], first[rows], second[rows]
        pairs.append(reference(form, e, e, f, carrier))
        weighted.append(
            c0 * reference(form, e, e, h, carrier)
            + c1 * reference(form, e, f, f, carrier)
        )
    kernel = make_dispersive_pairs(form, n_pol, parallel=False, shifted=shifted)
    derivative, curvature = first.copy(), second.copy()
    kernel(
        field,
        derivative,
        curvature,
        c0,
        c1,
        carrier,
        offset if shifted else np.zeros(1, complex),
    )
    for got, nodes in ((derivative, pairs), (curvature, weighted)):
        expected = np.concatenate(nodes)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=16 * EPS * np.abs(expected).max()
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_three_component_bodies_are_the_reference_contraction(family):
    """The general modal route's full vector field, real and analytic."""
    bodies = FAMILIES[family]
    rng = np.random.default_rng(12)
    x, y, z = random_fields(rng, (3, 1), real=False)
    arguments = [complex(one[c, 0]) for one in (x, y, z) for c in range(3)]
    got = np.array(bodies.dispersive_isotropic3(*arguments))
    np.testing.assert_allclose(got, _isotropic(x, y, z)[:, 0], rtol=8 * EPS)
    got = np.array(bodies.dispersive_conserving3(*arguments))
    np.testing.assert_allclose(got, conserving(x, y, z)[:, 0], rtol=8 * EPS)


@pytest.mark.parametrize(
    ("form", "shifted"), [("real", False), ("conserving", False), ("conserving", True)]
)
def test_the_general_products_are_the_reference_s_in_their_region(form, shifted):
    """Three nodes of three components, the middle one of another region: zero there.

    The cube and the polynomial's two derivative products of a chunk that starts at
    node one of the route.
    """
    rng = np.random.default_rng(13)
    real = form == "real"
    field, first, second = random_fields(rng, (9, 8), real=real)
    offset = rng.standard_normal(9) + 1j * rng.standard_normal(9)
    bare = field - offset[:, None] if shifted else field
    node_region = np.array([7, 1, 2, 1], dtype=np.int32)
    contract = _isotropic if real else conserving
    kernels = make_general_dispersive(form, parallel=False, shifted=shifted)
    dtype = float if real else complex
    cube = np.full((9, 8), np.nan, dtype=dtype)
    low, high = cube.copy(), cube.copy()
    dummy = offset if shifted else np.zeros(1, complex)
    kernels.cube(field, cube, node_region, 1, 1, 3, dummy)
    kernels.pairs(
        field, first, second, low, high, 0.75, -1.25, node_region, 1, 1, 3, dummy
    )
    for node in range(3):
        rows = slice(3 * node, 3 * node + 3)
        if node == 1:
            # Structural: a node outside the region is cleared, exactly.
            for got in (cube, low, high):
                np.testing.assert_array_equal(got[rows], 0.0)
            continue
        e, f, h = bare[rows], first[rows], second[rows]
        for got, expected in (
            (cube[rows], contract(e, e, e)),
            (low[rows], contract(e, e, f)),
            (high[rows], 0.75 * contract(e, e, h) - 1.25 * contract(e, f, f)),
        ):
            np.testing.assert_allclose(
                got, expected, rtol=0, atol=16 * EPS * np.abs(expected).max()
            )


def test_the_layout_groups_equal_leg_bounds_and_forms_a_cube_per_region():
    """Two species of one leg bound share a group; a region of its own, a cube."""
    leaf = small_problem("polynomial", envelope=False, thg=True)[
        0
    ].responses.dispersive_kerr
    assert leaf is not None
    assert leaf.outputs is not None and leaf.products is not None
    twice = dataclasses.replace(
        leaf,
        labels=("a", "b", "c"),
        coefficients=(*leaf.coefficients, 2.0, 3.0),
        filters=np.repeat(leaf.filters, 3, axis=0),
        outputs=np.repeat(leaf.outputs, 3, axis=0),
        products=np.repeat(leaf.products, 3, axis=0),
        densities=leaf.densities * 3,
    )
    terms = dispersive_terms_layout(twice, "field", analytic=False)
    assert terms.groups == ((0, 1, 2),) and terms.n_cubes == 1
    assert terms.n_fields == 2 and terms.n_products == 3
    regional = dispersive_terms_layout(twice, "field", analytic=False, keys=(4, 4, 9))
    assert regional.groups == ((0, 1), (2,)) and regional.keys == (4, 9)
    assert regional.cubes == (0, 1) and regional.n_products == 6
    assert regional.pair_products(1) == (4, 5)
    # Each channel's two output terms land on its region's cube and its group's pair.
    channel_rows = sorted(
        zip(regional.term_channels.tolist(), regional.term_rows.tolist(), strict=True)
    )
    assert channel_rows == [
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 4),
        (3, 3),
        (3, 5),
    ]


def symmetric_mixture_leaf(n_channels: int = 3):
    """Return a symmetric leaf of several channels, the first species' filter varied."""
    leaf = small_problem("symmetric", envelope=False, thg=True)[
        0
    ].responses.dispersive_kerr
    assert leaf is not None
    rise = np.linspace(1.0, 1.3, leaf.filters.shape[1])
    filters = np.vstack([leaf.filters * rise**channel for channel in range(n_channels)])
    return dataclasses.replace(
        leaf,
        labels=tuple("abcdefg"[:n_channels]),
        coefficients=tuple(
            leaf.coefficients[0] * (1 + channel) for channel in range(n_channels)
        ),
        filters=filters,
        densities=leaf.densities * n_channels,
    )


def test_the_symmetric_layout_takes_one_product_per_mixture():
    """A mixture is one group, one field and one product; a region of its own, two."""
    leaf = symmetric_mixture_leaf()
    terms = dispersive_terms_layout(leaf, "field", analytic=True)
    assert terms.groups == ((0, 1, 2),) and terms.mixtures == (0,)
    assert terms.n_fields == terms.n_products == 1
    # The mixture's coefficient is the factor after the unit one.
    assert terms.term_channels.tolist() == [leaf.n_channels + 1]
    np.testing.assert_array_equal(terms.input_scale, leaf.positive)
    regional = dispersive_terms_layout(leaf, "field", analytic=False, keys=(1, 2, 1))
    assert regional.groups == ((0, 2), (1,)) and regional.mixtures == (0,)
    assert regional.term_channels.tolist() == [leaf.n_channels + 1, 1]
    np.testing.assert_array_equal(regional.input_scale, 1.0)


@pytest.mark.parametrize("analytic", [True, False])
def test_the_stage_builds_the_mixture_filter_as_the_reference_does(analytic):
    """``mixture_rows`` is ``mixture_filter``'s arithmetic, to its rounding.

    The input row is ``g_mix`` (times ``Pi_+`` on the analytic family), the stage
    multiplier ``F g_mix`` times the sector's factor; a new composition rebuilds them,
    the same densities leave them, and a vacuum gives a unit filter and no output.
    The filter is a sum over the channels of positive powers and a root: 16 eps
    relative. Leaving a table and the vacuum's unit filter are structural, exact.
    """
    leaf = symmetric_mixture_leaf()
    terms = dispersive_terms_layout(leaf, "field", analytic=analytic)
    stage = StageWeights(leaf, terms)
    for densities in ((1e25, 2e25, 3e25), (3e25, 0.0, 1e24)):
        stage.update(densities)
        factors = np.array(leaf.coefficients) * np.array(densities)
        total, curve = mixture_filter(leaf.filters, factors)
        positive = leaf.positive.astype(float) if analytic else 1.0
        np.testing.assert_allclose(stage.inputs[0], curve * positive, rtol=16 * EPS)
        np.testing.assert_allclose(
            stage.multipliers[0], total * (curve * terms.output_scale), rtol=16 * EPS
        )
    before = stage.multipliers.copy()
    stage.inputs[:] = np.nan
    stage.update((3e25, 0.0, 1e24))
    assert np.isnan(stage.inputs).all()
    np.testing.assert_array_equal(stage.multipliers, before)
    stage.update((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(stage.inputs[0], terms.input_scale)
    np.testing.assert_array_equal(stage.multipliers[0], 0.0)


def test_the_mixture_kernel_is_the_reference_filter():
    """The kernel alone, on random positive tables and factors."""
    rng = np.random.default_rng(3)
    filters = 1.0 + rng.random((3, 17))
    factors = np.array([2.0, 0.5, 1.5, 9.0])
    squares = filters * filters
    inputs, outputs = np.empty(17), np.empty(17)
    scale = rng.random(17)
    total = mixture_rows(
        squares * squares,
        np.array([0, 2], dtype=np.int64),
        factors,
        np.ones(17),
        scale,
        inputs,
        outputs,
    )
    expected_total, expected = mixture_filter(filters[[0, 2]], factors[[0, 2]])
    # Sums of two positive powers and a root, and one product: 16 eps relative.
    assert total == pytest.approx(expected_total, rel=16 * EPS)
    np.testing.assert_allclose(inputs, expected, rtol=16 * EPS)
    np.testing.assert_allclose(outputs, expected * scale, rtol=16 * EPS)


# --- the spectral kernels ------------------------------------------------------------


def test_the_input_filter_weights_each_field_and_clears_the_tail():
    rng = np.random.default_rng(7)
    spectrum = rng.standard_normal((2, 10)) + 1j * rng.standard_normal((2, 10))
    multipliers = rng.standard_normal((3, 6))
    out = np.full((6, 10), np.nan + 0j)
    filter_spectra(spectrum, multipliers, out)
    # One real product per component: within the spectral kernels' 4 eps; the tail
    # beyond the multipliers is cleared, exactly.
    for field in range(3):
        block = out[2 * field : 2 * field + 2]
        np.testing.assert_allclose(
            block[:, :6], multipliers[field] * spectrum[:, :6], rtol=4 * EPS
        )
        np.testing.assert_array_equal(block[:, 6:], 0.0)
    # In place, as a lone response filters the stage's own spectrum.
    copy = spectrum.copy()
    filter_spectra(copy, multipliers[:1], copy)
    np.testing.assert_allclose(
        copy[:, :6], multipliers[0] * spectrum[:, :6], rtol=4 * EPS
    )


def test_the_terms_sharing_a_product_share_its_multiplier():
    """The polynomial's ``E^3`` of two channels: one multiplier, both weighted."""
    rng = np.random.default_rng(8)
    rows = np.array([0, 1, 0, 2], dtype=np.int64)
    channels = np.array([0, 0, 1, 1], dtype=np.int64)
    multipliers = rng.standard_normal((4, 5))
    factors = np.array([2.0, -0.5])
    out = np.full((3, 5), np.nan)
    weigh_terms(rows, channels, multipliers, factors, out)
    # Two binary scalings and a sum: one rounding, in 4 eps of the terms' size; a
    # lone binary scaling is exact.
    expected = 2.0 * multipliers[0] + (-0.5) * multipliers[2]
    scale = 2.0 * np.abs(multipliers[0]) + 0.5 * np.abs(multipliers[2])
    assert np.all(np.abs(out[0] - expected) <= 4 * EPS * scale)
    np.testing.assert_array_equal(out[1], 2.0 * multipliers[1])
    np.testing.assert_array_equal(out[2], -0.5 * multipliers[3])


@pytest.mark.parametrize("in_place", [False, True])
def test_the_products_sum_in_the_spectrum(in_place):
    """Three products on two columns, into the first block itself or not."""
    rng = np.random.default_rng(9)
    n_columns, n_bins = 2, 5
    spectra = rng.standard_normal((3 * n_columns, 7)) + 1j * rng.standard_normal(
        (3 * n_columns, 7)
    )
    weights = rng.standard_normal((3, n_bins))
    expected = sum(
        weights[product]
        * spectra[product * n_columns : (product + 1) * n_columns, :n_bins]
        for product in range(3)
    )
    if in_place:
        spectra = np.ascontiguousarray(spectra[:, :n_bins])
        out = spectra[:n_columns]
    else:
        out = np.empty((n_columns, n_bins), complex)
    combine_spectra(spectra, weights, out)
    np.testing.assert_allclose(
        out, expected, rtol=0, atol=4 * EPS * np.abs(expected).max()
    )


def test_the_products_add_to_the_stage_spectrum():
    """Three products on two columns, summed and then added once to what is there."""
    rng = np.random.default_rng(10)
    n_columns, n_bins = 2, 5
    spectra = rng.standard_normal((3 * n_columns, 7)) + 1j * rng.standard_normal(
        (3 * n_columns, 7)
    )
    weights = rng.standard_normal((3, n_bins))
    target = rng.standard_normal((n_columns, n_bins)) + 0j
    total = sum(
        weights[product]
        * spectra[product * n_columns : (product + 1) * n_columns, :n_bins]
        for product in range(3)
    )
    expected = target + total
    add_spectra(spectra, weights, target)
    np.testing.assert_allclose(
        target, expected, rtol=0, atol=4 * EPS * np.abs(expected).max()
    )


def test_the_collar_and_the_lone_multiplier_apply_in_place():
    rng = np.random.default_rng(11)
    source = rng.standard_normal((2, 6))
    window = np.linspace(0.0, 1.0, 6)
    applied = source.copy()
    apply_window(applied, window)
    # One product a sample, the spectral kernels' 4 eps.
    np.testing.assert_allclose(applied, window * source, rtol=4 * EPS)
    spectrum = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    weights = rng.standard_normal((1, 4))
    expected = weights[0] * spectrum
    weigh_spectrum(weights, spectrum)
    np.testing.assert_allclose(spectrum, expected, rtol=4 * EPS)


# --- the workspace, small enough to interpret ----------------------------------------


ARGON = OptmatKerr("Ar").dispersive_model()


def small_problem(evaluator, *, envelope, thg):
    """A one-node argon problem on a few hundred samples."""
    if envelope:
        grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (300e-9, 2e-6), 40e-15, thg=True)
    else:
        grid = FieldGrid.from_limits(1e-4, 800e-9, (300e-9, 2e-6), 40e-15)
    mode = MarcatiliMode(75e-6, cladding=1.45, loss=False)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=2e-6).build(grid)
    problem = assemble(
        grid,
        mode,
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg)],
        start,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        density=4.9e25,
        absorber_scheme="split_step",
        schedule=FixedSchedule.uniform(0.0, 1e-4, 1e-4),
    )
    return problem, grid


@pytest.mark.parametrize("evaluator", ["symmetric", "polynomial"])
@pytest.mark.parametrize(
    ("envelope", "thg"), [(False, True), (False, False), (True, True)]
)
def test_one_small_evaluation_is_the_reference_s(evaluator, envelope, thg):
    """The workspace end to end: filter, transforms, products, projection, collar.

    One node and one step of the Runge-Kutta march; under the split-step absorber, so
    a symmetric response filters the stage's spectrum in place. Bound: the evaluations
    of one step at ``64 eps log2(n_to)`` each, a few transforms per stage.
    """
    problem, grid = small_problem(evaluator, envelope=envelope, thg=thg)
    fields = []
    for backend_class in (NumbaCpuBackend, NumpyRefBackend):
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(1e-4)
            fields.append(backend.observe().require_field().copy())
        finally:
            backend.close()
    ours, theirs = fields
    measured = float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
    bound = 7 * 64 * EPS * np.log2(problem.grids.n_to)
    assert bound > 0
    assert measured < bound

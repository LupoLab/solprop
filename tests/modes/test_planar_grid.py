# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar grid: slab modes in x, a periodic collared y axis."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from solprop.constants import EPS0, C
from solprop.driver.simulation import _capillary_quadrature
from solprop.exceptions import GridError, ModeError
from solprop.grid import FreeGrid
from solprop.modes import (
    MarcatiliMode,
    ModeCollection,
    PlanarGrid,
    SlabMode,
    gaussian_coupling,
    gaussian_projection,
)
from solprop.modes.quadrature import mode_matrix, transverse_quadrature
from solprop.profiles import Linear

HALF_GAP = 63.5e-6


def slab_set(orders=(1, 3, 5), **fill):
    return [SlabMode(HALF_GAP, order=order, **fill) for order in orders]


def test_the_grid_lays_out_channels_fastest_and_nodes_x_fastest():
    modes = slab_set((1, 3))
    space = PlanarGrid(modes, 1e-3, 16, collar_fraction=0.2)
    assert space.n_p == 2
    assert space.n_columns == 32
    assert space.n_nodes == space.x.size * 16
    assert space.transverse_measure == "area"
    assert math.isclose(space.half_gap, HALF_GAP)
    # The y axis is FreeGrid's y axis exactly: the same helpers build the samples,
    # frequencies and collar, so this is identity of construction, not arithmetic.
    free = FreeGrid(1e-3, 16, 1e-3, 16, collar_fraction=0.2)
    assert_array_equal(space.y, free.y)
    assert_array_equal(space.k_y, free.k_y)
    # x = 0 is the ninth sample, where the x collar is exactly one.
    assert_array_equal(space.window_y, free.window.reshape(16, 16)[:, 8])
    assert math.isclose(space.dy, float(free.y[1] - free.y[0]))
    assert_array_equal(space.kperp_sq, np.repeat(space.k_y**2, 2))
    assert_array_equal(space.spectral_weights, np.full(32, space.dy / 16))
    assert np.all(np.diff(space.x) > 0)
    assert np.all(np.abs(space.x) < HALF_GAP)


def test_the_default_x_rule_is_the_modal_routes_default():
    """``max(32, 8 n_p, 6h + 6)`` with Kronrod, as ``from_waveguide`` builds it."""
    for orders in ((1,), (1, 3, 5), tuple(range(1, 10))):
        modes = slab_set(orders)
        space = PlanarGrid(modes, 1e-3, 8)
        rule = _capillary_quadrature(
            ModeCollection(modes), None, None, radial_rule="gauss_kronrod"
        )
        limits = modes[0].dimlimits(0.0)
        assert_array_equal(space.x, rule.nodes(limits)[:, 0])
        assert_array_equal(space.x_weights, rule.weights(limits))


def test_the_matrices_are_the_slab_rules_and_invert_on_the_set():
    modes = slab_set()
    space = PlanarGrid(modes, 1e-3, 8, n_x=40, x_rule="gauss")
    rule = transverse_quadrature(modes, n_r=40)
    matrices = mode_matrix(modes, rule, 0.0, components=("y",))
    assert_array_equal(space.synthesis, matrices.synthesis)
    assert_array_equal(space.projection, matrices.projection)
    assert_array_equal(space.projection_coarse, matrices.projection_coarse)
    # Orthonormal on the rule to its rounding (slice B's 1e-12 at the recorded count).
    assert_allclose(space.projection @ space.synthesis, np.eye(3), atol=1e-12)


def test_the_centre_and_on_axis_weights_reconstruct_the_mid_plane():
    modes = slab_set((1, 2, 3))
    space = PlanarGrid(modes, 1e-3, 8)
    # A normalised TE_p field at x = 0: cos(0) / sqrt(N) for odd p, sin(0) = 0 even.
    normalisation = math.sqrt(0.5 * EPS0 * C * HALF_GAP)
    assert_allclose(space.centre, [1 / normalisation, 0.0, 1 / normalisation])
    # A state uniform in y reconstructs the modal sum at the origin.
    amplitudes = np.array([2.0, 5.0, -1.0])
    columns = np.zeros((8, 3), complex)
    columns[0] = 8 * amplitudes
    on_axis = space.on_axis @ columns.ravel()
    assert_allclose(on_axis, space.centre @ amplitudes, rtol=1e-15)


def test_the_grid_prints_its_modes_and_axes():
    space = PlanarGrid(slab_set((1, 3)), 2e-3, 64)
    assert repr(space) == "PlanarGrid(TE1,TE3 x 33, gap=127 um, y=2 mm x 64)"
    text = str(space)
    assert "slab modes in x" in text
    assert "area: energy J, power W" in text


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        ([], "at least one slab mode"),
        ([MarcatiliMode(HALF_GAP)], "MarcatiliMode is not a SlabMode"),
        ([SlabMode(HALF_GAP), SlabMode(HALF_GAP)], "each slab channel once"),
        ([SlabMode(HALF_GAP), SlabMode(2 * HALF_GAP, order=3)], "shares one gap"),
        (
            [SlabMode(HALF_GAP), SlabMode(HALF_GAP, polarisation="tm")],
            "one polarisation family",
        ),
        (
            [SlabMode(Linear(HALF_GAP, 50e-6, 0.1))],
            "constant gap",
        ),
        (
            [SlabMode(HALF_GAP, "Ar", 1.0), SlabMode(HALF_GAP, "He", 1.0, order=3)],
            "shares one fill",
        ),
        (
            [SlabMode(HALF_GAP, "Ar", 1.0), SlabMode(HALF_GAP, "Ar", 2.0, order=3)],
            "shares one fill",
        ),
    ],
)
def test_a_set_the_hybrid_cannot_carry_is_refused_by_name(modes, match):
    with pytest.raises(ModeError, match=match):
        PlanarGrid(modes, 1e-3, 8)


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"n_y": 1}, "n_y must be an integer of at least 2"),
        ({"n_y": True}, "n_y must be an integer of at least 2"),
        ({"radius_y": 0.0}, "radius_y must be finite and positive"),
        ({"radius_y": math.inf}, "radius_y must be finite and positive"),
        ({"collar_fraction": -1.0}, "collar_fraction must be finite"),
        ({"n_x": 1}, "n_x must be an integer of at least 2"),
        ({"n_x": True}, "n_x must be an integer of at least 2"),
    ],
)
def test_an_invalid_axis_is_refused_by_name(keywords, match):
    arguments = {"radius_y": 1e-3, "n_y": 8} | keywords
    n_x = arguments.pop("n_x", None)
    window = arguments.pop("collar_fraction", 0.1)
    with pytest.raises(GridError, match=match):
        PlanarGrid(
            slab_set((1,)),
            arguments["radius_y"],
            arguments["n_y"],
            collar_fraction=window,
            n_x=n_x,
        )


def test_the_projection_keeps_each_orders_sign():
    modes = slab_set((1, 2, 3))
    centred = gaussian_projection(modes, 0.735 * HALF_GAP)
    assert centred[0] > 0
    # An even order's overlap with a centred beam cancels to the rule's rounding.
    assert abs(centred[1]) < 1e-12 * centred[0]
    offset = gaussian_projection(modes, 0.735 * HALF_GAP, offset=0.2 * HALF_GAP)
    # sin(pi x / a) is odd and the beam moves to positive x: a positive overlap.
    assert offset[1] > 0
    assert gaussian_projection(modes, 0.735 * HALF_GAP, offset=-0.2 * HALF_GAP)[1] < 0
    # Its squares over the beam's power are gaussian_coupling's shares, to the few
    # roundings of a square and a division over a separately formed power (8 eps).
    power = 0.5 * EPS0 * C * 0.735 * HALF_GAP * math.sqrt(math.pi / 2)
    shares = centred**2 / power
    assert_allclose(
        gaussian_coupling(modes, 0.735 * HALF_GAP),
        shares,
        rtol=0,
        atol=8 * np.finfo(float).eps * float(np.max(shares)),
    )

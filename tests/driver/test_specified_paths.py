# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A specified fibre whose dispersion and area both follow declared paths along z.

The closed form the accuracy is measured against is the walk-off
of a **linear** ``beta_1`` path: with a fixed frame the operator is

``D(omega, z) = -i (beta_1(z) - beta_1_frame) (omega - omega_0)``,

so the cumulative operator is a pure delay of

``tau(Z) = integral_0^Z (beta_1(z) - beta_1_frame) dz``,

which is ``g Z^2 / 2`` for ``beta_1(z) = beta_1_frame + g z`` -- exact, with no
discretisation anywhere, because ``D`` is degree one in ``z`` and the Chebyshev
integral of a degree-one interpolant is exact.
"""

import math
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop.backends.registry import resolve_name
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse
from solprop.driver.simulation import _gnlse_mode
from solprop.exceptions import ModeError, TranslationError
from solprop.grid import EnvelopeGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.modes.protocol import (
    GEOMETRY_PATH_NAMES,
    ChannelLabel,
    Mode,
    ParameterisedMode,
)
from solprop.nonlinear.gnlse import (
    ConstantDispersion,
    InterpolatedDispersion,
    SpecifiedMode,
    TaylorDispersion,
    _source_paths,
    as_dispersion_path,
)
from solprop.profiles import Linear, PiecewiseLinear
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta
from solprop.translate.linop import build_phi, delta_phi, parameter_paths
from solprop.translate.modal import (
    _DECLARED_PATHS,
    _radius_kinks,
    mode_averaged_transform,
)

LENGTH = 1e-3
"""A millimetre of fibre: the qualification scope's short window (AGENTS.md)."""

ENTRANCE_AREA = 1e-10
EXIT_AREA = 1.5e-10

BETA0 = 1.45 * 2 * np.pi * C / 835e-9 / C
"""``beta_0`` of a silica-like fibre at the reference, 1/m."""

BETA1_ENTRANCE = 1.47 / C
GRADIENT = 5.0e-8
"""``d beta_1 / dz`` in s/m^2: ``beta_1`` rises by one per cent over the millimetre.

A deliberately steep taper. The accuracy of the closed form below is limited not by
the interpolation -- there is none, the integrand is linear in ``z`` -- but by the
cancellation inside ``D``, which subtracts the frame's ``beta_0 + beta_1 (omega -
omega_0)`` from a ``beta`` of 1.85e7 1/m at the top of the grid. The absolute
round-off of the accumulated exponent is therefore about ``eps |beta| Z``, which
:func:`roundoff_floor` computes from the fibre itself rather than quoting, and the
walk-off has to stand well above it for the test to be about the mechanism.
"""

GRID = {
    "wavelength": 835e-9,
    "wavelength_limits": (600e-9, 1300e-9),
    "trange": 2e-12,
}
PULSE = {"fwhm": 200e-15, "energy": 1e-12, "shape": "gauss"}

LOSS_ENTRANCE = TaylorDispersion([1.0], 1.0)
LOSS_EXIT = TaylorDispersion([5.0e3], 1.0)
"""Two uniform attenuations in 1/m, for the cases where the loss alone moves.

A zero-degree Taylor expansion is a function of frequency that ignores it, which is
what a bend loss measured at two positions looks like.
"""


@pytest.fixture
def grid():
    """The millimetre grid every case here is built on."""
    return EnvelopeGrid.from_limits(LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)


def linear_walkoff(grid, positions=(0.0, LENGTH), areas=(ENTRANCE_AREA, EXIT_AREA)):
    """Return a fibre whose ``beta_1`` rises linearly from the entrance value."""
    nodes = np.asarray(positions, dtype=float)
    return SpecifiedMode.from_taylor_path(
        [[BETA0, BETA1_ENTRANCE + GRADIENT * z] for z in nodes],
        nodes,
        omega_ref=grid.omega0,
        aeff=list(areas),
    )


def fixed_frame(grid):
    """The frame the closed form is written in: both quantities pinned at entry."""
    return Frame(grid.omega0, subtract_beta0=True, beta0=BETA0, beta1=BETA1_ENTRANCE)


def roundoff_floor(grid, mode, length=LENGTH):
    """Return ``eps |beta| Z``, the round-off floor of the accumulated exponent.

    The operator subtracts the frame's ``beta_0 + beta_1 (omega - omega_0)`` from
    ``beta``, so what survives is known to the round-off of the largest number that
    entered the difference, times the distance it is accumulated over. Computed from
    the fibre and the grid rather than quoted, so that the bound and the docstrings
    cannot drift apart.
    """
    return float(
        np.finfo(float).eps * np.max(np.abs(mode.beta(grid.omega, length))) * length
    )


# ------------------------------------------------------------- the closed form


def test_a_linear_beta1_path_accumulates_its_exact_walk_off(grid):
    """``Phi`` is the exact ``-i g Z^2 / 2 (omega - omega_0)`` of the closed form.

    Predeclared bound, written before the run: 1e-9 relative on the accumulated
    delay, at four positions along the fibre and over the half of the active band
    where the detuning is largest. The bound is the cancellation inside ``D``, not
    an interpolation error -- there is none, the integrand is linear in ``z`` --
    and it is ``eps |beta| Z / |Phi|``, about 1.4e-13 at the band edge and rising
    towards the reference where ``Phi`` vanishes; 1e-9 leaves three decades.
    Measured 6.1e-13.
    """
    rtol = 1e-9
    mode = linear_walkoff(grid)
    table = build_phi(mode, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    assert table.phi_representation == "table"
    active = grid.active_mask
    detuning = grid.omega[active] - grid.omega0
    # Where the detuning is above half its extreme: the closed form passes through
    # zero at the reference, and a relative bound is meaningless there.
    outer = np.abs(detuning) > 0.5 * np.abs(detuning).max()
    worst = 0.0
    for fraction in (0.25, 0.5, 0.75, 1.0):
        distance = fraction * LENGTH
        accumulated = delta_phi(table, 0.0, distance)[0, 0, 0][active]
        expected = -1j * GRADIENT * distance**2 / 2.0 * detuning
        assert_allclose(accumulated[outer], expected[outer], rtol=rtol, atol=0.0)
        scale = np.max(np.abs(expected))
        worst = max(worst, float(np.max(np.abs(accumulated - expected)) / scale))
    assert worst < rtol


def test_a_kinked_beta1_path_lands_the_schedule_on_its_corner(grid):
    """Three declared positions: the middle one is a hard kink and the walk-off is
    the piecewise integral of the closed form.

    The same 1e-9 predeclared bound, for the same cancellation reason; measured
    2.9e-13. What the kink buys is that neither table interval straddles the
    corner, so each is still a degree-one integrand and the sum is exact rather
    than a fit across a cusp.
    """
    rtol = 1e-9
    corner = 0.4 * LENGTH
    nodes = np.array([0.0, corner, LENGTH])
    # beta_1 rises to the corner and returns to its entrance value by the end. The
    # corner sits at 0.4 L, so the fall is not the mirror image of the rise: it runs
    # over 0.6 L at two thirds of the slope. What the two legs are together is the
    # area under beta_1 - beta_1(0), a triangle of height GRADIENT * corner standing
    # on the whole base L.
    slopes = np.array([0.0, GRADIENT * corner, 0.0])
    mode = SpecifiedMode.from_taylor_path(
        np.column_stack([np.full(3, BETA0), BETA1_ENTRANCE + slopes]),
        nodes,
        omega_ref=grid.omega0,
        aeff=[ENTRANCE_AREA, EXIT_AREA, ENTRANCE_AREA],
    )
    assert corner in _radius_kinks(mode)
    table = build_phi(mode, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    assert corner in set(table.hard_kinks.tolist())
    active = grid.active_mask
    detuning = grid.omega[active] - grid.omega0
    # That triangle's area: half its height times its base, the two legs together.
    delay = 0.5 * GRADIENT * corner * LENGTH
    outer = np.abs(detuning) > 0.5 * np.abs(detuning).max()
    accumulated = delta_phi(table, 0.0, LENGTH)[0, 0, 0][active]
    expected = -1j * delay * detuning
    assert_allclose(accumulated[outer], expected[outer], rtol=rtol, atol=0.0)


def test_the_area_moves_with_the_dispersion(grid):
    mode = linear_walkoff(grid)
    assert not mode.z_constant
    assert not mode.dispersion_is_constant
    assert mode.aeff(0.0) == ENTRANCE_AREA
    assert mode.aeff(LENGTH) == pytest.approx(EXIT_AREA)
    assert mode.aeff(LENGTH / 2) == pytest.approx((ENTRANCE_AREA + EXIT_AREA) / 2)
    # The frame follows the mode, so a co-moving run sees no walk-off at all: the
    # operator of a pure beta_1 path is identically zero in its own frame. What is
    # left is the round-off of the cancellation, eps |beta| Z, computed from this
    # fibre: 4.10e-12, and the measured residual is 3.23e-12, just under it. The
    # assertion carries a factor of ten of headroom over that floor.
    floor = roundoff_floor(grid, mode)
    comoving = build_phi(mode, grid, Frame.comoving(grid), z_max=LENGTH, index=1.0)
    residual = delta_phi(comoving, 0.0, LENGTH)
    assert_allclose(residual, 0.0, atol=10.0 * floor)
    assert float(np.max(np.abs(residual))) < floor


# ------------------------------------------------ the constant path reduces exactly


def test_a_constant_dispersion_path_is_the_constant_route_bit_for_bit(grid):
    """One declared set at every position is the fibre that does not move.

    :class:`InterpolatedDispersion` reports itself constant when every position
    carries the same object, so the mode is ``z_constant``, takes ``constant_phi``
    and produces the identical table -- not merely a close one.

    The second half is the same statement beside a **varying** area: a constant
    dispersion path declares positions, but they are not kinks, and the kink walk
    must not invent them. Its corners and its taper table are the plain taper's.
    """
    value = TaylorDispersion([BETA0, BETA1_ENTRANCE, 2.0e-26], grid.omega0)
    derivative = TaylorDispersion([BETA0, BETA1_ENTRANCE, 2.0e-26], grid.omega0, 1)
    nodes = np.array([0.0, LENGTH])
    declared = SpecifiedMode(
        InterpolatedDispersion(nodes, (value, value)),
        InterpolatedDispersion(nodes, (derivative, derivative)),
        0.0,
        ENTRANCE_AREA,
    )
    plain = SpecifiedMode(value, derivative, 0.0, ENTRANCE_AREA)
    assert declared.z_constant and declared.dispersion_is_constant
    frame = fixed_frame(grid)
    one = build_phi(declared, grid, frame, z_max=LENGTH, index=1.0)
    two = build_phi(plain, grid, frame, z_max=LENGTH, index=1.0)
    assert one.phi_representation == two.phi_representation == "constant"
    np.testing.assert_array_equal(
        delta_phi(one, 0.0, LENGTH), delta_phi(two, 0.0, LENGTH)
    )
    # The same, with a taper beside the constant path. The path's positions are
    # 0 and LENGTH here; a declared position 37 % of the way along shows that a
    # constant path contributes no corner wherever it was sampled.
    taper = Linear(ENTRANCE_AREA, EXIT_AREA, LENGTH)
    inner = np.array([0.0, 0.37 * LENGTH, LENGTH])
    tapered = SpecifiedMode(
        InterpolatedDispersion(inner, (value, value, value)),
        InterpolatedDispersion(inner, (derivative, derivative, derivative)),
        0.0,
        taper,
    )
    plain_taper = SpecifiedMode(value, derivative, 0.0, taper)
    assert _radius_kinks(tapered) == _radius_kinks(plain_taper)
    assert tapered.parameters_own_z
    # And the transform's taper table is the plain taper's, interval for interval:
    # an invented corner splits it in three where two suffice.
    tapers = [
        mode_averaged_transform(one, grid, z_max=LENGTH).taper_scale
        for one in (tapered, plain_taper)
    ]
    assert tapers[0] is not None and tapers[1] is not None
    assert tapers[0].coefficients.shape == tapers[1].coefficients.shape
    np.testing.assert_array_equal(tapers[0].breakpoints, tapers[1].breakpoints)
    three = build_phi(tapered, grid, frame, z_max=LENGTH, index=1.0)
    four = build_phi(plain_taper, grid, frame, z_max=LENGTH, index=1.0)
    assert three.phi_representation == four.phi_representation == "expansion"
    np.testing.assert_array_equal(
        delta_phi(three, 0.0, LENGTH), delta_phi(four, 0.0, LENGTH)
    )


def test_the_area_only_fibre_keeps_the_parameter_expansion(grid):
    """The documented special case is untouched: a taper whose dispersion is fixed
    still declares one scalar parameter and still takes the expansion, which is what
    keeps the compiled and device routes open for it."""
    taper = SpecifiedMode.from_taylor(
        [BETA0, BETA1_ENTRANCE, 2.0e-26],
        omega_ref=grid.omega0,
        aeff=Linear(ENTRANCE_AREA, EXIT_AREA, LENGTH),
    )
    assert taper.dispersion_is_constant and taper.parameters_own_z
    assert sorted(parameter_paths([taper], 0.0, LENGTH) or {}) == ["effective_area"]
    table = build_phi(taper, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    assert table.phi_representation == "expansion"


def test_a_moving_dispersion_leaves_the_expansion_for_the_z_table(grid):
    mode = linear_walkoff(grid)
    assert not mode.parameters_own_z
    assert parameter_paths([mode], 0.0, LENGTH) is None
    with pytest.raises(ModeError, match="cannot be frozen at a parameter value"):
        mode.at_parameters({"effective_area": ENTRANCE_AREA})


# -------------------------------------------------------- the n_eff(omega, z) table


def test_an_index_path_carries_its_loss_and_its_dispersion(grid):
    omega = np.linspace(grid.omega[grid.active_mask].min() * 0.9, 4.0e15, 32)
    index = np.stack([np.full(32, 1.45 + 1e-9j), np.full(32, 1.46 + 2e-9j)])
    mode = SpecifiedMode.from_index_path(
        omega, index, [0.0, LENGTH], aeff=[ENTRANCE_AREA, EXIT_AREA]
    )
    probe = np.array([2.0e15])
    # beta = omega Re n / c and alpha = 2 omega Im n / c, interpolated in z.
    assert float(mode.beta(probe, LENGTH / 2)[0]) == pytest.approx(
        2.0e15 * 1.455 / C, rel=1e-12
    )
    assert float(mode.alpha(probe, LENGTH)[0]) == pytest.approx(
        2.0 * 2.0e15 * 2e-9 / C, rel=1e-12
    )
    assert not mode.z_constant


def test_an_index_path_that_misses_the_active_band_is_refused_by_name(grid):
    omega = np.linspace(1.0e15, 2.0e15, 8)
    index = np.stack([np.full(8, 1.45 + 0j), np.full(8, 1.46 + 0j)])
    mode = SpecifiedMode.from_index_path(
        omega, index, [0.0, LENGTH], aeff=[ENTRANCE_AREA, EXIT_AREA]
    )
    with pytest.raises(ModeError, match="outside the tabulated range"):
        build_phi(mode, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)


# ------------------------------------------------------------------- the refusals


def test_a_path_that_stops_before_the_run_does_is_refused_by_name(grid):
    mode = linear_walkoff(grid, positions=(0.0, LENGTH / 2))
    with pytest.raises(TranslationError, match="its dispersion over"):
        build_phi(mode, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    # And one that starts late is refused too: the entrance must be declared.
    late = linear_walkoff(grid, positions=(LENGTH / 4, LENGTH))
    with pytest.raises(TranslationError, match="does not cover the run"):
        build_phi(late, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)


def test_a_short_loss_or_derivative_path_is_refused_by_its_own_name(grid):
    """The value is not the only path that would be held over the rest of the fibre.

    A loss declared over half the fibre and held at its last value for the other
    half is the same silent, plausible, wrong answer as a short beta path -- it
    changes the transmission, and nothing says so. The derivative is the same
    statement about the frame. Each is refused under its own name, so the message
    says which declaration is short.
    """
    half = np.array([0.0, LENGTH / 2])
    whole = np.array([0.0, LENGTH])
    value = TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0)
    short_loss = InterpolatedDispersion(half, (LOSS_ENTRANCE, LOSS_EXIT))
    area = PiecewiseLinear(half, np.array([ENTRANCE_AREA, EXIT_AREA]))
    fibre = SpecifiedMode(value, value, short_loss, area)
    with pytest.raises(TranslationError, match="declares its loss over"):
        build_phi(fibre, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    # The derivative, declared short beside a value that covers the run: the paths
    # are sampled together, so the refusal names the derivative rather than the
    # value that happens to be checked first.
    moving = linear_walkoff(grid)
    short_derivative = SpecifiedMode(
        moving.dispersion,
        InterpolatedDispersion(
            half,
            (
                TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0, 1),
                TaylorDispersion([BETA0, 2 * BETA1_ENTRANCE], grid.omega0, 1),
            ),
        ),
        0.0,
        PiecewiseLinear(
            np.unique(np.concatenate((half, whole))),
            np.array([ENTRANCE_AREA, EXIT_AREA, ENTRANCE_AREA]),
        ),
    )
    with pytest.raises(TranslationError, match="declares its dispersion derivative"):
        build_phi(short_derivative, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)


def test_a_fibre_whose_loss_alone_moves_is_constructible(grid):
    """A constant beta with a declared loss path is a fibre, not a contradiction.

    The positions the area path is held to are the union over the spectral paths
    that **move**, so a constant value path -- which declares no position at all --
    does not make the comparison against an empty list.
    """
    value = TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0)
    nodes = np.array([0.0, LENGTH])
    fibre = SpecifiedMode(
        value,
        value,
        InterpolatedDispersion(nodes, (LOSS_ENTRANCE, LOSS_EXIT)),
        PiecewiseLinear(nodes, np.array([ENTRANCE_AREA, EXIT_AREA])),
    )
    assert not fibre.dispersion_is_constant
    assert not fibre.parameters_own_z
    probe = np.array([2.0e15])
    assert float(fibre.alpha(probe, 0.0)[0]) == pytest.approx(1.0)
    assert float(fibre.alpha(probe, LENGTH / 2)[0]) == pytest.approx(2500.5)
    table = build_phi(fibre, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    assert table.phi_representation == "table"


def test_a_kinked_loss_path_lands_its_corner_on_the_schedule(grid):
    """A corner declared in the loss alone is still a corner the tables split on.

    Not because the loss path is walked for kinks -- it is not -- but because the
    area path must be declared at the union of the moving paths' positions, and the
    area path is walked. That is what keeps "a declared position is a hard kink"
    true whichever path declares it.
    """
    corner = 0.3 * LENGTH
    nodes = np.array([0.0, corner, LENGTH])
    value = TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0)
    fibre = SpecifiedMode(
        value,
        value,
        InterpolatedDispersion(nodes, (LOSS_ENTRANCE, LOSS_EXIT, LOSS_ENTRANCE)),
        PiecewiseLinear(nodes, np.array([ENTRANCE_AREA, EXIT_AREA, ENTRANCE_AREA])),
    )
    assert corner in _radius_kinks(fibre)
    table = build_phi(fibre, grid, fixed_frame(grid), z_max=LENGTH, index=1.0)
    assert corner in set(table.hard_kinks.tolist())


def test_the_spectral_paths_must_be_declared_at_the_area_paths_positions(grid):
    """The union, not the value path alone: a loss corner the area does not carry
    is refused, because nothing would land on it."""
    value = TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0)
    nodes = np.array([0.0, LENGTH])
    with pytest.raises(ModeError, match="samples its dispersion and its area"):
        SpecifiedMode(
            value,
            value,
            InterpolatedDispersion(
                np.array([0.0, 0.3 * LENGTH, LENGTH]),
                (LOSS_ENTRANCE, LOSS_EXIT, LOSS_ENTRANCE),
            ),
            PiecewiseLinear(nodes, np.array([ENTRANCE_AREA, EXIT_AREA])),
        )


def test_a_z_profile_is_not_a_dispersion_path(grid):
    """A `ZProfile` satisfies the runtime-checkable protocol structurally -- its
    members are a superset -- so the boundary must refuse it by name, or a fibre
    built from one constructs and dies inside `beta` with a raw `TypeError`."""
    with pytest.raises(ModeError, match="is a ZProfile"):
        as_dispersion_path(Linear(1.0, 2.0, LENGTH))
    with pytest.raises(ModeError, match="is a ZProfile"):
        SpecifiedMode(
            Linear(BETA0, 2 * BETA0, LENGTH),
            TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0, 1),
            0.0,
            ENTRANCE_AREA,
        )


def test_a_dispersion_path_without_an_area_path_is_refused_by_name(grid):
    nodes = np.array([0.0, LENGTH])
    sets = (
        TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0),
        TaylorDispersion([BETA0, 2 * BETA1_ENTRANCE], grid.omega0),
    )
    derivatives = (
        TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0, 1),
        TaylorDispersion([BETA0, 2 * BETA1_ENTRANCE], grid.omega0, 1),
    )
    with pytest.raises(ModeError, match="must declare an effective-area path"):
        SpecifiedMode(
            InterpolatedDispersion(nodes, sets),
            InterpolatedDispersion(nodes, derivatives),
            0.0,
            ENTRANCE_AREA,
        )


def test_z_samples_that_do_not_match_are_refused_by_name(grid):
    # The two paths are declared at different positions.
    with pytest.raises(ModeError, match="samples its dispersion and its area"):
        SpecifiedMode.from_taylor_path(
            [[BETA0, BETA1_ENTRANCE], [BETA0, 2 * BETA1_ENTRANCE]],
            [0.0, LENGTH],
            omega_ref=grid.omega0,
            aeff=PiecewiseLinear(
                np.array([0.0, LENGTH / 2, LENGTH]),
                np.array([ENTRANCE_AREA, EXIT_AREA, ENTRANCE_AREA]),
            ),
        )
    # And a sequence of areas of the wrong length never reaches the mode.
    with pytest.raises(ModeError, match="samples its dispersion and its area"):
        SpecifiedMode.from_taylor_path(
            [[BETA0, BETA1_ENTRANCE], [BETA0, 2 * BETA1_ENTRANCE]],
            [0.0, LENGTH],
            omega_ref=grid.omega0,
            aeff=[ENTRANCE_AREA, EXIT_AREA, ENTRANCE_AREA],
        )


def test_the_value_and_the_derivative_must_agree_about_moving(grid):
    nodes = np.array([0.0, LENGTH])
    value = InterpolatedDispersion(
        nodes,
        (
            TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0),
            TaylorDispersion([BETA0, 2 * BETA1_ENTRANCE], grid.omega0),
        ),
    )
    with pytest.raises(ModeError, match="disagree"):
        SpecifiedMode(
            value,
            TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0, 1),
            0.0,
            PiecewiseLinear(nodes, np.array([ENTRANCE_AREA, EXIT_AREA])),
        )


@pytest.mark.parametrize(
    ("positions", "members", "message"),
    [
        ([0.0], (TaylorDispersion([1.0], 1.0),), "at least two declared positions"),
        (
            [1.0, 0.0],
            (TaylorDispersion([1.0], 1.0), TaylorDispersion([2.0], 1.0)),
            "finite and increasing",
        ),
        ([0.0, 1.0], (TaylorDispersion([1.0], 1.0),), "one set per position"),
    ],
)
def test_a_bad_dispersion_path_is_refused(positions, members, message):
    with pytest.raises(ModeError, match=message):
        InterpolatedDispersion(positions, members)


def test_a_declared_set_is_reproduced_exactly_at_its_own_position(grid):
    entrance = TaylorDispersion([BETA0, BETA1_ENTRANCE], grid.omega0)
    end = TaylorDispersion([BETA0, 2 * BETA1_ENTRANCE], grid.omega0)
    path = InterpolatedDispersion([0.0, LENGTH], (entrance, end))
    omega = np.linspace(1.0e15, 3.0e15, 16)
    np.testing.assert_array_equal(path(omega, 0.0), entrance(omega))
    np.testing.assert_array_equal(path(omega, LENGTH), end(omega))
    # Outside the declared range the value is held, as a scalar path's is.
    np.testing.assert_array_equal(path(omega, -1.0), entrance(omega))
    np.testing.assert_array_equal(path(omega, 2 * LENGTH), end(omega))


# --------------------------------------------------------- a moving guide, end to end


def test_a_tapered_capillary_reaches_prop_gnlse_with_both_paths():
    """``from_mode`` of a guide whose bore moves: the two paths come together by
    construction.

    Ten saved planes over a millimetre, which is the qualification scope; what is
    asserted is that the fibre carries both paths and that the run's own dispersion
    really changed between the entrance and the exit.
    """
    bore = Linear(15e-6, 12e-6, LENGTH)
    guide = MarcatiliMode(bore, cladding=1.45)
    fibre = cast("SpecifiedMode", _gnlse_mode(guide, 835e-9, needs_area=False))
    assert not fibre.dispersion_is_constant
    assert fibre.aeff(LENGTH) < fibre.aeff(0.0)
    # The bore's corners are the fibre's kinks, through the one mechanism.
    assert set(_radius_kinks(fibre)) >= set(bore.breakpoints.tolist())
    probe = np.array([2.0e15])
    assert float(fibre.beta(probe, LENGTH)[0]) != float(fibre.beta(probe, 0.0)[0])
    output = prop_gnlse(
        0.1,
        LENGTH,
        fibre=guide,
        shock=False,
        n_save=3,
        progress="none",
        noise=None,
        backend="numpy_ref",
        **GRID,
        **PULSE,
    )
    fields = output.fields("fft")
    assert fields.shape[0] == 3
    assert np.all(np.isfinite(fields))


def test_a_declared_taylor_path_runs_end_to_end(grid):
    """The other door: a fibre declared as a path rather than wrapped from a mode.

    `from_taylor_path` reaches the stepper, not only the translation layer, so the
    z table it takes is evaluated for every accepted step. Three saved planes over a
    millimetre, the qualification scope, on the reference (the native backend's runs
    of the table are below).
    """
    fibre = linear_walkoff(grid)
    output = prop_gnlse(
        0.1,
        LENGTH,
        fibre=fibre,
        shock=False,
        n_save=3,
        progress="none",
        noise=None,
        backend="numpy_ref",
        **GRID,
        **PULSE,
    )
    fields = output.fields("fft")
    assert fields.shape == (3, 1, 1, grid.n_omega)
    assert np.all(np.isfinite(fields))


@pytest.mark.medium
@pytest.mark.parametrize(
    "integrator",
    [FIXED_ORDER, RungeKutta(), LowStorageRungeKutta()],
    ids=["multistep", "runge_kutta", "low_storage"],
)
def test_a_moving_dispersion_runs_on_the_native_backend(grid, integrator):
    """The native CPU build evaluates the z table itself (SEP 29 section 6.6).

    ``backend="auto"`` prefers the native backend wherever it is installed, so a
    moving-dispersion fibre runs by default. Ten fixed steps over the millimetre on
    each march -- the exponential Adams method carries the varying operator's
    compensation, whose rate the derivative table supplies -- matched to the
    reference within the march's budget.
    """
    pytest.importorskip("solprop_native")
    assert resolve_name("auto") == "native"

    def simulation(backend):
        return Simulation.from_gnlse(
            0.1,
            LENGTH,
            fibre=linear_walkoff(grid),
            shock=False,
            n_save=3,
            noise=None,
            backend=backend,
            schedule=FixedSchedule(np.full(10, LENGTH / 10)),
            integrator=integrator,
            # No collar: ten fixed steps are refused beside the default one.
            absorber=False,
            **GRID,
            **PULSE,
        )

    native = simulation("auto")
    problem = native.translate()
    assert problem.linear.phi_representation == "table"
    got = native.run(progress="none").fields("fft")
    expected = simulation("numpy_ref").run(progress="none").fields("fft")
    bound = simulation_bound(native, backend="native")
    assert_matched(got, expected, bound, what="moving dispersion")


# ---------------------------------------------- the paths a source mode declares


class _StubMode:
    """A mode that moves, declaring whatever paths a test wants and nothing else.

    Every mode in the package that moves declares its parameters, so the branches
    of `_source_paths` and `_source_geometry` that a *bare* moving mode reaches --
    one that names its geometry paths by attribute, one that names none at all --
    need a double to reach them.
    """

    z_constant = False
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("stub", None, None, None, None, "stub")

    def __init__(self, **paths):
        for name, value in paths.items():
            setattr(self, name, value)

    def principal_component(self, z=0.0):
        del z

    def neff(self, omega, z=0.0):
        del z
        return np.ones_like(omega, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        del z
        return np.zeros_like(omega, dtype=complex)

    def dneff_domega(self, omega, z=0.0):
        del z
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        del z
        return ENTRANCE_AREA

    def dimlimits(self, z=0.0):
        del z
        return "polar", (0.0, 0.0), (1.0, 2 * np.pi)

    def field(self, points, z=0.0):
        del z
        return np.zeros((len(points), 2))


def test_a_bare_moving_mode_declares_its_geometry_by_attribute():
    # Not a ParameterisedMode, so the geometry names are what say where it moves.
    bore = Linear(15e-6, 10e-6, LENGTH)
    stub = cast("Mode", _StubMode(core_radius=bore))
    assert not isinstance(stub, ParameterisedMode)
    fibre = SpecifiedMode.from_mode(stub)
    assert not fibre.dispersion_is_constant
    assert np.array_equal(fibre.dispersion_path.breakpoints, bore.breakpoints)
    assert fibre.dispersion_path.z_span == bore.z_span


@pytest.mark.parametrize("name", GEOMETRY_PATH_NAMES)
def test_both_walks_read_the_same_declared_geometry_names(name):
    """The kink walk and the z-ownership walk must know the same names.

    `translate.modal._radius_kinks` reads them for the corners the tables and the
    schedule land on, and `nonlinear.gnlse._source_paths` for the paths a specified
    fibre inherits from its source; `gnlse` cannot import `translate.modal`, so the
    tuple lives in `modes.protocol` and both read it rather than restating it.
    """
    assert (*GEOMETRY_PATH_NAMES, "dispersion_path") == _DECLARED_PATHS
    bore = Linear(15e-6, 10e-6, LENGTH)
    stub = cast("Mode", _StubMode(**{name: bore}))
    assert _source_paths(stub) == (bore,)
    assert _radius_kinks(stub) == tuple(bore.breakpoints.tolist())


def test_a_moving_mode_that_declares_no_path_is_refused_by_name():
    stub = cast("Mode", _StubMode())
    with pytest.raises(ModeError, match="declares no path saying where"):
        SpecifiedMode.from_mode(stub)


def test_declared_paths_with_no_interval_in_common_are_refused_by_name():
    early = PiecewiseLinear(np.array([0.0, 1e-4]), np.array([15e-6, 12e-6]))
    late = PiecewiseLinear(np.array([5e-4, LENGTH]), np.array([1.0, 2.0]))
    stub = cast("Mode", _StubMode(core_radius=early, area_path=late))
    with pytest.raises(ModeError, match="no interval in common"):
        SpecifiedMode.from_mode(stub)


def test_a_source_that_does_not_move_declares_no_dispersion_positions(grid):
    source = SpecifiedMode.from_taylor([BETA0, BETA1_ENTRANCE], omega_ref=grid.omega0)
    # The fibre itself: a frequency-only dispersion, wrapped by ConstantDispersion,
    # covers any run and kinks nowhere.
    for path in (source.dispersion_path, source.derivative_path, source.loss_path):
        assert isinstance(path, ConstantDispersion)
        assert path.is_constant
        assert path.breakpoints.size == 0
        assert path.z_span == (0.0, math.inf)
    # And wrapping it keeps that, through the source's own z_constant.
    fibre = SpecifiedMode.from_mode(source)
    for path in (fibre.dispersion_path, fibre.derivative_path, fibre.loss_path):
        assert path.is_constant
        assert path.breakpoints.size == 0
        assert path.z_span == (0.0, math.inf)


def test_a_wrapped_moving_source_reports_the_same_positions_for_loss_and_beta():
    bore = Linear(15e-6, 12e-6, LENGTH)
    fibre = SpecifiedMode.from_mode(MarcatiliMode(bore, cladding=1.45))
    assert not fibre.loss_path.is_constant
    assert np.array_equal(
        fibre.loss_path.breakpoints, fibre.dispersion_path.breakpoints
    )
    assert fibre.loss_path.z_span == fibre.dispersion_path.z_span


def test_a_table_with_the_wrong_number_of_rows_is_refused(grid):
    # Three sets at two declared positions. The rank is right, so the runtime type
    # hook passes it and the library's own check is what refuses it -- which is the
    # check that has to stay reachable in production, where no hook runs.
    with pytest.raises(ModeError, match="one expansion set per declared position"):
        SpecifiedMode.from_taylor_path(
            [[BETA0, BETA1_ENTRANCE]] * 3,
            [0.0, LENGTH],
            omega_ref=grid.omega0,
            aeff=[ENTRANCE_AREA, EXIT_AREA],
        )
    omega = np.linspace(1.0e15, 2.0e15, 8)
    with pytest.raises(ModeError, match="one index row per declared position"):
        SpecifiedMode.from_index_path(
            omega,
            np.full((3, 8), 1.45 + 0j),
            [0.0, LENGTH],
            aeff=[ENTRANCE_AREA, EXIT_AREA],
        )

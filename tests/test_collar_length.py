# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The collar length, spectral-only collars and the custom absorber (SEP 22 5.4).

Every comparison here is structural (class P of ``docs/dev/testing.md`` section 3):
a default reproduces today's record and rates exactly, a zero user rate returns the
collar field for field, and the rates a custom absorber copies from its collar are the
collar's own arrays.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from solprop.boundaries import (
    MAX_ALPHA_LENGTH,
    TEMPORAL_ALPHA_LENGTH,
    Absorber,
    absorber,
    custom_absorber,
    temporal_collar_rate,
)
from solprop.driver import SimulationSettings
from solprop.driver.simulation import _refined_tdse_collar
from solprop.exceptions import DriverError, GridError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import SlabMode
from solprop.modes.slab import PlanarGrid
from solprop.nonlinear.tdse import TDSESampling
from solprop.output.schema import serialise
from solprop.stepper import PIController
from solprop.translate.free import free_transform
from solprop.translate.masks import temporal_rates
from solprop.translate.planar import planar_transform
from solprop.translate.radial import radial_transform
from solprop.translate.tdse import refine_response_grid


@pytest.fixture
def grid():
    return FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)


def test_an_absorber_at_its_defaults_records_its_defaults(grid):
    collar = absorber(grid)
    assert collar.collar_length is None and collar.temporal is True
    assert collar.in_stage_length == collar.reference_length
    record = serialise(collar)
    assert isinstance(record, dict)
    # Output schema 21: every field is recorded, the two SEP 22 added at their
    # defaults too.
    assert set(record) == {
        "__class__",
        "spectral_rate",
        "temporal_rate",
        "temporal_rate_fine",
        "band_mask",
        "reference_length",
        "collar_length",
        "temporal",
    }
    assert record["collar_length"] is None and record["temporal"] is True
    explicit = dataclasses.replace(collar, collar_length=collar.reference_length)
    assert explicit.collar_length is None
    assert serialise(explicit) == record


@pytest.mark.parametrize("length", [0.0, -1.0, np.inf, np.nan])
def test_a_collar_length_is_positive_and_finite(grid, length):
    with pytest.raises(GridError, match="collar_length must be positive and finite"):
        dataclasses.replace(absorber(grid), collar_length=length)


def test_the_two_ceilings_follow_their_own_lengths(grid):
    collar = absorber(grid)
    short = collar.reference_length / 1000
    # A collar whose in-stage length is far below the reference length: its temporal
    # rates reach 12.6 / ell_c, a thousand times the reference's ceiling, and validate.
    strong = dataclasses.replace(
        collar,
        temporal_rate=temporal_collar_rate(grid.t, grid.t_collar, short),
        temporal_rate_fine=temporal_collar_rate(grid.t_fine, grid.t_collar, short),
        collar_length=short,
    )
    assert strong.in_stage_length == short
    assert float(strong.temporal_rate.max()) == pytest.approx(
        TEMPORAL_ALPHA_LENGTH / short
    )
    assert (
        float(strong.temporal_rate.max()) > MAX_ALPHA_LENGTH / strong.reference_length
    )
    # The spectral rate keeps the reference's ceiling.
    with pytest.raises(GridError, match="spectral_rate must lie"):
        dataclasses.replace(
            strong,
            spectral_rate=np.full_like(collar.spectral_rate, 2 * MAX_ALPHA_LENGTH)
            / collar.reference_length,
        )
    # And the temporal rates are refused above their own length's ceiling.
    with pytest.raises(GridError, match="temporal_rate must lie"):
        dataclasses.replace(
            collar,
            temporal_rate=np.full(grid.n_t, 2 * MAX_ALPHA_LENGTH)
            / collar.reference_length,
            collar_length=2 * collar.reference_length,
        )


def test_a_spectral_only_collar_carries_no_temporal_rate(grid):
    collar = absorber(grid)
    spectral = absorber(grid, temporal=False)
    assert spectral.temporal is False
    assert not np.any(spectral.temporal_rate) and not np.any(
        spectral.temporal_rate_fine
    )
    assert spectral.temporal_rate.shape == (grid.n_t,)
    assert spectral.temporal_rate_fine.shape == (grid.n_to,)
    # The band mask and the spectral rate are the collar's, exactly.
    assert np.array_equal(spectral.spectral_rate, collar.spectral_rate)
    assert np.array_equal(spectral.band_mask, collar.band_mask)
    record = serialise(spectral)
    assert isinstance(record, dict) and record["temporal"] is False
    with pytest.raises(GridError, match="spectral-only collar"):
        dataclasses.replace(collar, temporal=False)


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_collar_of_zero_rates_crosses_the_seam_as_none(grid, scheme):
    hand_built = Absorber(
        absorber(grid).spectral_rate,
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        absorber(grid).reference_length,
    )
    for collar in (absorber(grid, temporal=False), hand_built):
        assert temporal_rates(collar, scheme=scheme) == (None, None)
    coarse, fine = temporal_rates(absorber(grid), scheme=scheme)
    assert coarse is not None and fine is not None


def test_a_zero_user_rate_returns_the_collar_field_for_field(grid):
    for collar in (
        absorber(grid),
        absorber(grid, temporal=False),
        absorber(grid, boundary_N=60),
    ):
        same = custom_absorber(grid, 0.0, collar=collar)
        assert serialise(same) == serialise(collar)
        assert same.spectral_rate is collar.spectral_rate
    assert serialise(custom_absorber(grid, 0.0)) == serialise(absorber(grid))


@pytest.mark.parametrize("kind", ["scalar", "array", "callable"])
def test_a_user_rate_moves_only_the_spectral_rate_and_its_length(grid, kind):
    collar = absorber(grid)
    strong = 1e4 * MAX_ALPHA_LENGTH / collar.reference_length
    rate = {
        "scalar": strong,
        "array": np.full(grid.n_omega, strong),
        "callable": lambda omega: np.full(omega.shape, strong),
    }[kind]
    custom = custom_absorber(grid, rate, collar=collar)
    # The collar's in-stage strength and its arrays are kept exactly.
    assert custom.temporal_rate is collar.temporal_rate
    assert custom.temporal_rate_fine is collar.temporal_rate_fine
    assert custom.band_mask is collar.band_mask
    assert custom.in_stage_length == collar.in_stage_length
    assert custom.collar_length == collar.reference_length
    # The reference length is the longest whose ceiling admits the sum.
    expected = collar.spectral_rate + np.where(collar.band_mask, strong, 0.0)
    assert np.array_equal(custom.spectral_rate, expected)
    assert custom.reference_length == MAX_ALPHA_LENGTH / float(np.max(expected))
    # Outside the band the operator is zero, so the user's rate is too.
    assert not np.any(custom.spectral_rate[~collar.band_mask])


def test_a_user_rate_that_fits_keeps_the_reference_length(grid):
    collar = absorber(grid)
    # The collar reaches its ceiling at the band edges, so a rate that fits is one
    # inside its flat region.
    inside = np.where(collar.spectral_rate == 0.0, 1.0, 0.0)
    custom = custom_absorber(grid, inside, collar=collar)
    assert custom.reference_length == collar.reference_length
    assert custom.collar_length is None
    stacked = custom_absorber(grid, 1e9, collar=custom_absorber(grid, 1e9))
    assert stacked.collar_length == collar.reference_length
    spectral = custom_absorber(grid, 1e9, collar=absorber(grid, temporal=False))
    assert spectral.temporal is False


@pytest.mark.parametrize(
    ("rate", "message"),
    [
        (-1.0, "finite and non-negative"),
        (np.nan, "finite and non-negative"),
        (np.ones(3), "shape"),
    ],
)
def test_a_user_rate_is_finite_non_negative_and_on_the_axis(grid, rate, message):
    with pytest.raises(GridError, match=message):
        custom_absorber(grid, rate)


def test_every_route_takes_the_spatial_collar_from_the_collar_length():
    time = EnvelopeGrid.from_limits(1e-3, 800e-9, (650e-9, 1000e-9), 300e-15)
    collar = absorber(time)
    shorter = dataclasses.replace(collar, collar_length=collar.reference_length / 4)
    planar = PlanarGrid([SlabMode(63.5e-6, "Ar", 1.0)], 1e-3, 8)
    routes: list[tuple[Any, Any]] = [
        (radial_transform, RadialGrid(300e-6, 16, window_factor=0.5)),
        (free_transform, FreeGrid(3e-4, 8, 4e-4, 6, window_factor=0.5)),
        (planar_transform, planar),
    ]
    for build, space in routes:
        default = build(space, time, n_pol=1, z_max=1e-3, absorber=collar).geometry
        moved = build(space, time, n_pol=1, z_max=1e-3, absorber=shorter).geometry
        assert default is not None and moved is not None
        window = default.window
        # The same expression on the same inputs: exact, by construction.
        assert np.array_equal(
            default.absorber_rate,
            TEMPORAL_ALPHA_LENGTH * (1 - window) / collar.reference_length,
        )
        assert np.array_equal(
            moved.absorber_rate,
            TEMPORAL_ALPHA_LENGTH * (1 - window) / shorter.in_stage_length,
        )
        assert np.any(moved.absorber_rate)


@pytest.mark.parametrize("kind", ["standard", "shorter", "spectral"])
def test_the_tdse_refinement_rebuilds_each_rate_from_its_own_length(grid, kind):
    refined = refine_response_grid(grid, TDSESampling(2))
    collar = absorber(grid)
    declared = {
        "standard": collar,
        "shorter": dataclasses.replace(
            collar,
            temporal_rate=temporal_collar_rate(grid.t, grid.t_collar, 0.01),
            temporal_rate_fine=temporal_collar_rate(grid.t_fine, grid.t_collar, 0.01),
            collar_length=0.01,
        ),
        "spectral": absorber(grid, temporal=False),
    }[kind]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 10,
        absorber=declared,
    )
    resolved = _refined_tdse_collar(settings, grid, refined).absorber
    assert resolved is not None
    expected = (
        np.zeros(refined.n_to)
        if kind == "spectral"
        else temporal_collar_rate(
            refined.t_fine, refined.t_collar, declared.in_stage_length
        )
    )
    # The same expression on the same axis: exact.
    assert np.array_equal(resolved.temporal_rate_fine, expected)
    assert resolved.temporal_rate is declared.temporal_rate
    assert resolved.collar_length == declared.collar_length
    assert resolved.temporal is declared.temporal
    custom = custom_absorber(grid, 1e9, collar=declared)
    with pytest.raises(DriverError, match="standard analytic collar"):
        _refined_tdse_collar(
            dataclasses.replace(settings, absorber=custom), grid, refined
        )


def test_a_spectral_only_collar_has_no_walk_off_to_check():
    from solprop.driver.simulation import _warn_capillary_walkoff  # noqa: PLC0415
    from solprop.exceptions import SolpropWarning  # noqa: PLC0415
    from solprop.modes import MarcatiliMode  # noqa: PLC0415

    # Twenty metres of a 125 um capillary: HE11 and HE12 drift by about a picosecond,
    # ten times the window, which the temporal collar is warned too weak for.
    long = FieldGrid.from_limits(20.0, 800e-9, (200e-9, 2e-6), 100e-15)
    modes = [MarcatiliMode(125e-6, m=m) for m in (1, 2)]
    with pytest.warns(SolpropWarning, match="walking off"):
        _warn_capillary_walkoff(modes, long, absorber(long))
    # A collar without a temporal part has nothing to warn about.
    _warn_capillary_walkoff(modes, long, absorber(long, temporal=False))

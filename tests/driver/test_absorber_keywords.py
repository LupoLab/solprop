# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The absorber keywords of the builders and interfaces (SEP 21 section 5).

``absorber=False`` switches the temporal and spectral collars off and keeps the hard
band mask and the collars' room, so the run shares the absorber-on run's grid;
``absorber_scheme`` is accepted everywhere a temporal collar can be; a scheme beside
``absorber=False``, or a keyword that contradicts the settings, is refused.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from solprop.boundaries import absorber, no_absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.interface import _with_boundary
from solprop.driver.simulation import _spatial_default
from solprop.exceptions import DriverError
from solprop.grid import FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.stepper import PIController
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.raman import kernel_taper


@pytest.fixture(scope="module")
def grid() -> FieldGrid:
    """Return a small field grid."""
    return FieldGrid.from_limits(0.1, 800e-9, (200e-9, 2e-6), 200e-15)


def _capillary(**keywords: Any) -> Simulation:
    return Simulation.from_capillary(
        100e-6, 0.1, "Ar", 0.5, wavelength=800e-9, fwhm=30e-15, energy=10e-6,
        wavelength_limits=(200e-9, 2e-6), trange=200e-15, noise=None, **keywords,
    )  # fmt: skip


def _gnlse(*, trange: float = 2e-12, **keywords: Any) -> Simulation:
    return Simulation.from_gnlse(
        0.1, 0.1, [0.0, 0.0, -1e-26], wavelength=1e-6, fwhm=100e-15, peak_power=1.0,
        trange=trange, wavelength_limits=(0.7e-6, 1.5e-6), noise=None, **keywords,
    )  # fmt: skip


def test_no_absorber_keeps_the_band_mask_and_the_reference_length(
    grid: FieldGrid,
) -> None:
    """Both soft collars are zero; the mask and in-stage length are the default's."""
    off, default = no_absorber(grid), absorber(grid)
    assert not np.any(off.spectral_rate)
    assert not np.any(off.temporal_rate)
    assert not np.any(off.temporal_rate_fine)
    assert np.array_equal(off.band_mask, default.band_mask)
    assert off.in_stage_length == default.in_stage_length
    assert off.temporal is False


def test_capillary_absorber_false_switches_both_collars_off() -> None:
    """``absorber=False`` on a capillary records zero rates and the default scheme."""
    boundary = _capillary(absorber=False).settings.absorber
    assert boundary is not None
    assert not np.any(boundary.spectral_rate)
    assert not np.any(boundary.temporal_rate)
    assert boundary.band_mask.any()


@pytest.mark.parametrize("build", [_capillary, _gnlse])
def test_a_scheme_beside_absorber_false_is_refused(
    build: Callable[..., Simulation],
) -> None:
    """There is no temporal collar for a scheme to apply."""
    with pytest.raises(DriverError, match="absorber_scheme"):
        build(absorber=False, absorber_scheme="split_step")


def test_gnlse_takes_the_absorber_keywords() -> None:
    """GNLSE takes the routes' default collar, ``False``, a collar and a scheme."""
    default = _gnlse().settings.absorber
    assert default is not None
    assert default.temporal
    off = _gnlse(absorber=False).settings.absorber
    assert off is not None
    assert not np.any(off.spectral_rate)
    simulation = _gnlse(absorber_scheme="split_step")
    assert simulation.settings.absorber_scheme == "split_step"
    collar = absorber(_gnlse().grid)
    assert _gnlse(absorber=collar).settings.absorber is collar


def test_waveguide_takes_the_scheme() -> None:
    """``from_waveguide`` passes the scheme to a hollow guide's settings."""
    simulation = Simulation.from_waveguide(
        [MarcatiliMode(100e-6, "Ar", 0.5, loss=False, cladding=1.45)], 0.1, "Ar", 0.5,
        wavelength=800e-9, fwhm=30e-15, energy=10e-6, wavelength_limits=(200e-9, 2e-6),
        trange=200e-15, noise=None, absorber_scheme="split_step",
    )  # fmt: skip
    assert simulation.settings.absorber_scheme == "split_step"


def _settings(grid: FieldGrid, **keywords: Any) -> SimulationSettings:
    return SimulationSettings(
        PIController(rtol=1e-6, atol=0.0), save_planes=[grid.length],
        dz_max=grid.length, dz_init=grid.length * 1e-3, **keywords,
    )  # fmt: skip


def test_spatial_absorber_false_keeps_the_lengths(grid: FieldGrid) -> None:
    """``False`` zeroes the settings' collar's rates and keeps its lengths."""
    collar = dataclasses.replace(absorber(grid), collar_length=0.02)
    settings = _with_boundary(_settings(grid, absorber=collar), grid, False, None)
    off = settings.absorber
    assert off is not None
    assert not np.any(off.spectral_rate)
    assert not np.any(off.temporal_rate_fine)
    assert off.in_stage_length == collar.in_stage_length
    assert off.reference_length == collar.reference_length
    # The settings carry none: the hard mask alone.
    bare = _with_boundary(_settings(grid), grid, False, None).absorber
    assert bare is not None
    assert not np.any(bare.spectral_rate)
    assert not np.any(bare.temporal_rate)
    assert bare.temporal is False
    assert bare.band_mask.any()


def test_spatial_keywords_refuse_a_contradiction(grid: FieldGrid) -> None:
    """A keyword may fill in the settings, never overrule them."""
    with pytest.raises(DriverError, match="already declare an absorber"):
        _with_boundary(
            _settings(grid, absorber=absorber(grid)), grid, absorber(grid), None
        )
    with pytest.raises(DriverError, match="absorber_scheme"):
        _with_boundary(
            _settings(grid, absorber_scheme="split_step"), grid, None, "nonlinear"
        )
    with pytest.raises(DriverError, match="absorber_scheme"):
        _with_boundary(_settings(grid), grid, False, "split_step")
    same = _with_boundary(_settings(grid), grid, absorber(grid), "split_step")
    assert same.absorber_scheme == "split_step"
    assert same.absorber is not None


@pytest.mark.parametrize("build", [_capillary, _gnlse])
def test_absorber_false_keeps_the_collar_s_room(
    build: Callable[..., Simulation],
) -> None:
    """The collars' rates are zero and their room is kept: one grid, absorber or none.

    The Raman tail taper, the held-plasma gate and every other reader of the collar
    keep their samples whatever the transform size rounds to (SEP 21, ADR-057).
    """
    with_collar, without = build().grid, build(absorber=False).grid
    assert without.n_t == with_collar.n_t
    assert without.t_collar == with_collar.t_collar
    np.testing.assert_array_equal(without.t, with_collar.t)


@pytest.mark.parametrize("trange", [2e-12, 1.998e-12, 1.5e-12])
def test_absorber_false_keeps_the_raman_taper(trange: float) -> None:
    """The response's tail taper is the absorber-on run's, many samples wide."""
    on = _gnlse(raman=True, trange=trange)
    off = _gnlse(raman=True, trange=trange, absorber=False)
    off.translate()
    taper = kernel_taper(off.grid)
    np.testing.assert_array_equal(taper, kernel_taper(on.grid))
    assert np.count_nonzero(taper < 1.0) > 10


def test_absorber_false_keeps_the_held_plasma_gate() -> None:
    """A field-plasma capillary's gate is the absorber-on run's."""
    on = _capillary(plasma=True).translate()
    off = _capillary(plasma=True, absorber=False).translate()
    assert plasma_source_record(off.grids) == plasma_source_record(on.grids)


def test_spatial_defaults_give_the_default_temporal_collar() -> None:
    """A spatial run whose settings declare no absorber gets the default temporal and
    spectral collars (SEP 21, D6); its transverse boundary is left to the settings'
    ``spatial_absorber``, where a real-space collar is asked for (SEP 40)."""
    grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 2e-6), 200e-15)
    settings = _spatial_default(_settings(grid), grid)
    assert settings.absorber is not None
    assert settings.absorber.temporal
    assert np.array_equal(settings.absorber.temporal_rate, absorber(grid).temporal_rate)
    assert settings.spatial_absorber is None
    given = _settings(grid, absorber=absorber(grid))
    assert _spatial_default(given, grid) is given


def test_the_modal_interfaces_take_the_boundary_check() -> None:
    """``boundary_check=None`` on ``prop_gnlse`` runs with nothing watched (SEP 21)."""
    from solprop.driver import prop_gnlse  # noqa: PLC0415
    from solprop.output.schema import PlaneStore  # noqa: PLC0415

    run = prop_gnlse(
        0.1, 0.01, [0.0, 0.0, -1e-26], wavelength=1e-6, fwhm=100e-15, peak_power=1.0,
        trange=2e-12, wavelength_limits=(0.7e-6, 1.5e-6), noise=None, n_save=2,
        progress="none", boundary_check=None, backend="numpy_ref",
    )  # fmt: skip
    assert isinstance(run, PlaneStore)
    assert run.outcome is not None and run.outcome.execution is not None
    assert "boundary_monitor" not in run.outcome.execution
    watched = prop_gnlse(
        0.1, 0.01, [0.0, 0.0, -1e-26], wavelength=1e-6, fwhm=100e-15, peak_power=1.0,
        trange=2e-12, wavelength_limits=(0.7e-6, 1.5e-6), noise=None, n_save=2,
        progress="none", backend="numpy_ref",
    )  # fmt: skip
    assert isinstance(watched, PlaneStore)
    assert watched.outcome is not None and watched.outcome.execution is not None
    assert "boundary_monitor" in watched.outcome.execution

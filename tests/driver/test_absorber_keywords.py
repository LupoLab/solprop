# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The absorber keywords of the builders and interfaces (SEP 21 section 5).

``absorber=False`` switches the temporal and spectral collars off and keeps the hard
band mask; ``absorber_scheme`` is accepted everywhere a temporal collar can be; a scheme
beside ``absorber=False``, or a keyword that contradicts the settings, is refused.
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
from solprop.exceptions import DriverError
from solprop.grid import FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.stepper import PIController


@pytest.fixture(scope="module")
def grid() -> FieldGrid:
    """Return a small field grid."""
    return FieldGrid.from_limits(0.1, 800e-9, (200e-9, 2e-6), 200e-15)


def _capillary(**keywords: Any) -> Simulation:
    return Simulation.from_capillary(
        100e-6, 0.1, "Ar", 0.5, lambda0=800e-9, fwhm=30e-15, energy=10e-6,
        wavelength_limits=(200e-9, 2e-6), trange=200e-15, noise=None, **keywords,
    )  # fmt: skip


def _gnlse(**keywords: Any) -> Simulation:
    return Simulation.from_gnlse(
        0.1, 0.1, [0.0, 0.0, -1e-26], lambda0=1e-6, fwhm=100e-15, peak_power=1.0,
        trange=2e-12, wavelength_limits=(0.7e-6, 1.5e-6), noise=None, **keywords,
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
    """GNLSE keeps no boundaries by default and takes a collar and a scheme."""
    assert _gnlse().settings.absorber is None
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
        lambda0=800e-9, fwhm=30e-15, energy=10e-6, wavelength_limits=(200e-9, 2e-6),
        trange=200e-15, noise=None, absorber_scheme="split_step",
    )  # fmt: skip
    assert simulation.settings.absorber_scheme == "split_step"


def _settings(grid: FieldGrid, **keywords: Any) -> SimulationSettings:
    return SimulationSettings(
        PIController(rtol=1e-6, atol=0.0), save_planes=[grid.length],
        dz_max=grid.length, dz_init=grid.length * 1e-3, **keywords,
    )  # fmt: skip


def test_spatial_absorber_false_keeps_the_lengths(grid: FieldGrid) -> None:
    """The real-space collar reads the in-stage length, which ``False`` keeps."""
    collar = dataclasses.replace(absorber(grid), collar_length=0.02)
    settings = _with_boundary(_settings(grid, absorber=collar), grid, False, None)
    off = settings.absorber
    assert off is not None
    assert not np.any(off.spectral_rate)
    assert not np.any(off.temporal_rate_fine)
    assert off.in_stage_length == collar.in_stage_length
    assert off.reference_length == collar.reference_length
    # Nothing to switch off: the settings carry no absorber, and keep none.
    assert _with_boundary(_settings(grid), grid, False, None).absorber is None


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

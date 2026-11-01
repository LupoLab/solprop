# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tiny builders for every propagation route, shared by the multistep and memory tests.

Each builder takes the noise model (``None``, ``"input"`` or ``"modified"``) and returns
an untranslated :class:`solprop.driver.Simulation` on a problem a fifth of a millimetre
long with steps of a tenth: small enough to translate and step in a test. They began
as SEP 16's capability audit, which asked which routes each integrator admits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import (
    CoherentWall,
    GradedIndexProvider,
    PlanarGrid,
    PowerLawPermittivity,
    SlabMode,
    StepIndexMode,
)
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial, MaterialKerr, MaterialRegion
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.profiles import Linear, PiecewiseLinear
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

LENGTH = 2e-4
"""Length of every probe, m: two steps of a tenth of a millimetre."""

STEP = LENGTH / 2
"""First and largest step, m."""

SEED = 16
"""Seed of every noise draw."""

# --------------------------------------------------------------------------- routes


def _run_keywords() -> dict[str, Any]:
    """Return the save plane and steps every waveguide constructor takes."""
    return {
        "save_planes": [LENGTH],
        "dz_max": STEP,
        "dz_init": STEP,
        "backend": "numpy_ref",
    }


def _spatial_settings() -> SimulationSettings:
    """Return the free-space and planar routes' settings."""
    return SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[LENGTH],
        dz_max=STEP,
        dz_init=STEP,
        backend="numpy_ref",
        # These coarse grids launch into the safety ramp's band, which the default
        # refuses (SEP 40 section 5.3), and their numbers predate the ramp; the
        # default is tested on adequate grids in tests/test_transverse_resolution.py.
        spatial_absorber=False,
    )


def gnlse(noise: str | None) -> Simulation:
    """Return a 1 kW-scale soliton in a fibre, the GNLSE envelope route."""
    return Simulation.from_gnlse(
        0.11,
        LENGTH,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
        **_run_keywords(),
    )


CAPILLARY_GRID = FieldGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15)
"""A small field grid for the capillary routes."""


def _capillary(noise: str | None, **overrides: Any) -> Simulation:
    """Return a helium-filled capillary on the shared small grid."""
    arguments = {
        "wavelength": 800e-9,
        "grid": CAPILLARY_GRID,
        "fwhm": 30e-15,
        "energy": 1e-5,
        "plasma": False,
        "raman": False,
        "noise": noise,
        "seed": None if noise is None else SEED,
        **_run_keywords(),
    } | overrides
    radius = arguments.pop("radius", 100e-6)
    pressure = arguments.pop("pressure", 1.0)
    return Simulation.from_capillary(
        radius, LENGTH, "He", pressure, **{"noise": None, **arguments}
    )


def capillary_averaged(noise: str | None) -> Simulation:
    """Return the mode-averaged capillary at constant pressure."""
    return _capillary(noise)


def capillary_envelope(noise: str | None) -> Simulation:
    """Return the mode-averaged capillary on an envelope grid."""
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15)
    return _capillary(noise, grid=grid, envelope=True)


def capillary_multimode(noise: str | None) -> Simulation:
    """Return three capillary modes at constant pressure."""
    return _capillary(noise, modes=3)


def capillary_gradient(noise: str | None) -> Simulation:
    """Return the mode-averaged capillary with a pressure gradient."""
    return _capillary(noise, pressure=(1.0, 0.1))


def capillary_wall(noise: str | None, **overrides: Any) -> Simulation:
    """Return three modes beside the coherent wall: a constant coupled flow."""
    return _capillary(noise, modes=3, wall=CoherentWall(), **overrides)


def capillary_taper(noise: str | None, **overrides: Any) -> Simulation:
    """Return three modes along a radius taper with its default taper coupling."""
    radius = PiecewiseLinear.from_stages(
        (0.0, 0.3 * LENGTH, 0.7 * LENGTH, LENGTH), [100e-6, 100e-6, 80e-6, 80e-6]
    )
    return _capillary(noise, modes=3, radius=radius, **overrides)


GENERAL_GRID = FieldGrid.from_limits(LENGTH, 835e-9, (815e-9, 860e-9), 60e-15)
"""The general modal route's grid: a narrow band, so the guide's table is cheap."""


def _step_index_guide() -> ModalGuide:
    """Return one HE11 channel of a 1.5 um strand with a Kerr core."""
    core, cladding = IndexMaterial(1.45), IndexMaterial(1.0)
    mode = StepIndexMode(
        1.5e-6,
        core=1.45,
        cladding=1.0,
        profile_wavelength=835e-9,
        kind="HE",
        n=1,
        m=1,
        parity="even",
    )
    regions = (
        MaterialRegion(
            "core", 0.0, mode.core_radius, core, (MaterialKerr(chi3=2.2e-22),)
        ),
        MaterialRegion("cladding", mode.core_radius, None, cladding),
    )
    guide = ModalGuide.from_step_index(
        (mode,), core=core, cladding=cladding, regions=regions
    )
    provider = guide.field_provider
    active = GENERAL_GRID.omega[GENERAL_GRID.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    return ModalGuide.from_provider(
        provider,  # pyright: ignore[reportArgumentType]
        np.linspace(0.99 * lo, 1.01 * hi, 9),
        nodes=provider.node_rule(0.99 * lo, n_r=16),  # pyright: ignore[reportOptionalMemberAccess]
    )


def general_constant(noise: str | None) -> Simulation:
    """Return the constant step-index guide on the general modal route."""
    return Simulation.from_waveguide(
        (),
        LENGTH,
        provider=_step_index_guide(),
        grid=GENERAL_GRID,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
        **_run_keywords(),
    )


GRIN_GRID = FieldGrid.from_limits(LENGTH, 1.03e-6, (1.0e-6, 1.06e-6), 100e-15)
"""The moving graded-index guide's grid."""


def general_moving(noise: str | None, **overrides: Any) -> Simulation:
    """Return a linearly tapering graded-index guide: moving maps, a coupled flow."""
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * np.sqrt(1.0 - 2.0 * 0.005)),
        radius=25e-6,
        family="LP",
        count=2,
        elements=32,
        extent=2.0,
        core_responses=(MaterialKerr(chi3=2.2e-22),),
    )
    provider = dataclasses.replace(provider, radius=Linear(25e-6, 24e-6, LENGTH))
    active = GRIN_GRID.omega[GRIN_GRID.active_mask]
    knots = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 13)
    guide = ModalGuide.from_provider(
        provider,
        knots,
        nodes=provider.node_rule(float(knots[0]), 0.0, n_r=2),
        sections=1,
    )
    arguments = {
        "wavelength": 1.03e-6,
        "fwhm": 60e-15,
        "energy": 1e-12,
        "noise": noise,
        "seed": None if noise is None else SEED,
        **_run_keywords(),
    } | overrides
    return Simulation.from_waveguide(
        (), LENGTH, provider=guide, grid=GRIN_GRID, **{"noise": None, **arguments}
    )


SPATIAL_GRID = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
"""The free-space routes' field grid."""


def _argon() -> tuple[SpatialMedium, list[object]]:
    """Return an argon medium and its Kerr response without the harmonic."""
    return SpatialMedium("Ar", 1.0), [
        KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)
    ]


def _spatial_pulse() -> SpatialPulse:
    """Return a 30 fs, 3 nJ Gaussian beam of 80 um waist."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )


def radial(noise: str | None) -> Simulation:
    """Return the radial free-space route, 32 Hankel nodes."""
    medium, responses = _argon()
    return Simulation.from_radial(
        SPATIAL_GRID,
        RadialGrid(6e-4, 32),
        medium,
        responses,
        _spatial_pulse(),
        settings=_spatial_settings(),
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
    )


def cartesian(noise: str | None) -> Simulation:
    """Return the Cartesian free-space route on a 12 x 16 plane."""
    medium, responses = _argon()
    return Simulation.from_free(
        SPATIAL_GRID,
        FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0),
        medium,
        responses,
        _spatial_pulse(),
        settings=_spatial_settings(),
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
    )


def line(noise: str | None) -> Simulation:
    """Return the line free-space route on 16 points."""
    medium, responses = _argon()
    return Simulation.from_free(
        SPATIAL_GRID,
        LineGrid(3e-4, 16, collar_fraction=0),
        medium,
        responses,
        _spatial_pulse(),
        settings=_spatial_settings(),
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
    )


class NoKeywordError(Exception):
    """The route's constructor offers no keyword for the variant's noise model."""


def planar(noise: str | None) -> Simulation:
    """Return the hybrid planar route: two slab modes of an argon gap, 8 y points."""
    if noise is not None:
        raise NoKeywordError("Simulation.from_planar takes no noise keyword")
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15)
    half_gap = 63.5e-6
    modes = [SlabMode(half_gap, "Ar", 1.0, order=p) for p in (1, 3)]
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * half_gap,
    )
    return Simulation.from_planar(
        grid,
        PlanarGrid(modes, 1e-3, 8),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        pulse,
        settings=_spatial_settings(),
    )


ROUTES: dict[str, Callable[[str | None], Simulation]] = {
    "gnlse": gnlse,
    "capillary (mode-averaged)": capillary_averaged,
    "capillary (envelope)": capillary_envelope,
    "capillary (multimode)": capillary_multimode,
    "capillary (pressure gradient)": capillary_gradient,
    "general guide (constant)": general_constant,
    "general guide (moving)": general_moving,
    "radial": radial,
    "cartesian": cartesian,
    "line": line,
    "planar": planar,
    "capillary (coherent wall)": capillary_wall,
    "capillary (taper coupling)": capillary_taper,
}
"""Each route's builder, which takes the noise model."""

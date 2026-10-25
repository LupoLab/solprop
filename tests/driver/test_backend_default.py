# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve backend defaults before scans or runs acquire native resources."""

import dataclasses
import importlib.machinery
import importlib.util
import pickle
import warnings
from functools import partial
from typing import TypedDict

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID
from solprop.backends.registry import select
from solprop.driver import (
    Simulation,
    SimulationSettings,
    prop_capillary,
    prop_free,
    prop_gnlse,
    prop_radial,
    prop_waveguide,
)
from solprop.exceptions import (
    MaterialRangeWarning,
)
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.output import MemoryOutput
from solprop.profiles import PiecewiseLinear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController


class _BackendChoice(TypedDict, total=False):
    backend: str


@pytest.fixture(params=[False, True])
def installation(request, monkeypatch):
    """Probe either installation state without importing an optional package."""
    real_find_spec = importlib.util.find_spec
    present = request.param
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            (importlib.machinery.ModuleSpec(name, None) if present else None)
            if name == "solprop_native"
            else real_find_spec(name)
        ),
    )
    return "native" if present else "numba_cpu"


def settings(**keywords):
    return SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.001],
        dz_init=0.001,
        dz_max=0.001,
        **keywords,
    )


def test_settings_keep_the_resolved_name_across_replacement_and_transfer(
    installation, monkeypatch
):
    original = settings()
    assert original.backend == installation
    # A worker receives the resolved choice even if its package availability differs.
    # An unavailable chosen backend is then a visible selection error, not a fallback.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert dataclasses.replace(original, dz_init=0.0005).backend == installation
    assert pickle.loads(pickle.dumps(original)).backend == installation
    assert settings(backend="numpy_ref").backend == "numpy_ref"


@pytest.mark.parametrize("kind", ["gnlse", "capillary", "waveguide", "radial", "free"])
@pytest.mark.parametrize("explicit", [False, True])
def test_public_interfaces_share_the_default_and_preserve_explicit_choices(
    installation, monkeypatch, narrow_grid, kind, explicit
):
    selected = []
    output = MemoryOutput()

    def run(simulation, *args, **kwargs):
        selected.append(simulation.settings.backend)
        return output

    monkeypatch.setattr(Simulation, "run", run)
    keywords: _BackendChoice = {"backend": "numpy_ref"} if explicit else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        if kind == "gnlse":
            result = prop_gnlse(
                0.11,
                0.001,
                [0.0, 0.0, -1.183e-26],
                lambda0=835e-9,
                grid=narrow_grid,
                width=100e-15,
                pulseshape="sech",
                peak_power=1.0,
                **keywords,
            )
        elif kind in {"capillary", "waveguide"}:
            interface = (
                partial(prop_capillary, 125e-6)
                if kind == "capillary"
                else partial(prop_waveguide, [MarcatiliMode(125e-6)])
            )
            result = interface(
                0.001,
                "He",
                1.0,
                lambda0=835e-9,
                grid=narrow_grid,
                fwhm=30e-15,
                energy=1e-9,
                plasma=False,
                raman=False,
                n_r=8,
                n_save=2,
                **keywords,
            )
        else:
            interface = (
                partial(prop_radial, narrow_grid, RadialGrid(6e-4, 8))
                if kind == "radial"
                else partial(prop_free, narrow_grid, FreeGrid(3e-4, 8, 3e-4, 8))
            )
            pulse = SpatialPulse(
                GaussianPulse(wavelength=835e-9, fwhm=30e-15, energy=1e-9), 60e-6
            )
            result = interface(
                SpatialMedium(),
                [],
                pulse,
                settings=settings(**keywords),
            )
    assert result is output
    assert selected == ["numpy_ref" if explicit else installation]


def test_auto_runs_a_z_table_taper_on_the_native_cpu_build():
    """The default call on a z-table problem takes the native backend, which runs it.

    A 1 m multimode capillary taper on the **adiabatic** route (the declared opt-out,
    ``taper_coupling=False``): its operator's parameter expansion over four channels
    does not resolve at the degree ceiling, so it takes the z table. ``"auto"`` picks
    the native backend and never falls back (the policy kept), and its CPU build
    evaluates the table natively (SEP 29 section 6.6), so the default call is
    admitted; the runs are compared with the reference in
    ``tests/driver/test_native_z_table.py``. The same taper by default carries the
    taper coupling, whose carrier is the lowest channel's expansion.
    """
    pytest.importorskip("solprop_native")
    taper = PiecewiseLinear(
        np.array([0.0, 0.5, 0.7, 1.0]), np.array([100e-6, 100e-6, 60e-6, 60e-6])
    )
    with warnings.catch_warnings():
        # The He index is extrapolated below its data at 90 nm, which is the case's
        # own warning; only the representations are under test (nothing propagates).
        warnings.simplefilter("ignore", MaterialRangeWarning)
        # A 2 mm dz_max is above the coupled flow's advisory step, which says so.
        warnings.filterwarnings("ignore", message=".*explicit residual is the limit")
        arguments = {
            "lambda0": 800e-9,
            "fwhm": 10e-15,
            "energy": 1e-9,
            # The former grid, on which the expansion does not resolve.
            "grid": FieldGrid.from_limits(
                1.0, 800e-9, (90e-9, 3e-6), 250e-15, **LEGACY_GRID
            ),
            "modes": 4,
            "dz_max": 2e-3,
            "n_save": 2,
        }
        coupled = Simulation.from_capillary(taper, 1.0, "He", 1.5, **arguments)
        assert coupled.translate().device_ready
        assert coupled.translate().linear.coupled_flow is not None
        simulation = Simulation.from_capillary(
            taper, 1.0, "He", 1.5, taper_coupling=False, **arguments
        )
        problem = simulation.translate()
    assert problem.linear.phi_representation == "table"
    backend = select(
        "auto",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        assert backend.name == "native"
    finally:
        backend.close()

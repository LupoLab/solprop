# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build bounded backend exercises from the registry's public source problems.

Import this module only after configuring the worker's scientific runtimes.
The vector Raman exercise uses, by design (SEP 30 Part II, phase 2), the current
optmat rotational hydrogen response with the propagation fixture's launch.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from benchmarks.detectors import far_field
from benchmarks.records import SuiteError
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import MolecularOptions, response_for
from solprop.physdata import MaterialSelection
from solprop.pulses import GaussianPulse
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec
from solprop.stepper import PIController

__all__ = [
    "dispersive",
    "envelope_plasma",
    "exercise",
    "fiber",
    "mixture_gradient",
    "plasma",
    "vector_raman",
]


def vector_raman() -> Simulation:
    """Keep the fixture's circular pump and oblique delayed probe, in SI units."""
    grid = EnvelopeGrid.from_limits(2e-4, 800e-9, (300e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-7).build(grid)
    delayed = pulse * np.exp(-1j * grid.omega * 20e-15)
    initial = np.concatenate(
        [pulse / np.sqrt(2) + 0.02 * delayed, 1j * pulse / np.sqrt(2) - 0.03 * delayed],
        axis=1,
    )
    response = response_for(
        "H2",
        MolecularOptions(vibration=False),
        selection=MaterialSelection(carrier_wavelength=800e-9),
        density=1e25,
    )
    return Simulation(
        grid,
        mode,
        [response],
        initial,
        settings=SimulationSettings(
            PIController(rtol=1e-6, atol=1e-10),
            density=1e25,
            quadrature=transverse_quadrature([mode], n_r=16),
            save_planes=[grid.length],
            dz_init=grid.length * 1e-4,
            dz_max=grid.length,
            stats=StatsSpec(names=("energy", "peak_power_total", "peak_intensity")),
            backend="numpy_ref",
        ),
    )


def envelope_plasma() -> Simulation:
    """Retain the envelope-plasma tutorial's complete eight-micrometre slab."""
    grid = EnvelopeGrid.from_limits(8e-6, 800e-9, (500e-9, 1.2e-6), 320e-15)
    return Simulation(
        grid,
        MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45, loss=False),
        [EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 800e-9))],
        GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-3),
        settings=SimulationSettings(
            PIController(rtol=1e-6, atol=1e-14),
            save_planes=[grid.length],
            dz_init=grid.length / 4,
            dz_max=grid.length / 4,
            ionisation_field_range=(1e7, 1e11),
            absorber=absorber(grid, boundary_length=1e-3),
            stats=StatsSpec(
                names=("energy", "species_ionisation", "peak_ionisation_rate")
            ),
            backend="numpy_ref",
        ),
    )


def plasma(name: str) -> Simulation:
    """Extract the small two-mode capillary used by plasma-integration parity."""
    if name not in ("plasma_band", "plasma_trapezoid", "ordinary_stats"):
        raise SuiteError(f"plasma exercise not implemented for {name}")
    simulation = Simulation.from_capillary(
        125e-6,
        1e-3,
        "He",
        1.0,
        lambda0=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_r=8,
        modes=2,
        thg=True,
        plasma_integration="band" if name == "plasma_band" else "trapezoid",
        absorber_scheme="split_step",
        backend="numpy_ref",
    )
    if name != "ordinary_stats":
        simulation.settings = replace(
            simulation.settings,
            stats=StatsSpec(
                names=(
                    "electron_density",
                    "peak_ionisation_rate",
                    *PLASMA_INTEGRATION_STATISTICS,
                )
            ),
        )
    return simulation


def dispersive() -> Simulation:
    """Extract the argon four-wave-mixing tutorial's gas-polynomial case."""
    wavelengths = np.array([400e-9, 800e-9, 1 / (2 / 400e-9 - 1 / 800e-9)])
    omega = 2 * math.pi * C / wavelengths

    def mismatch(pressure: float) -> float:
        mode = MarcatiliMode(50e-6, "Ar", pressure)
        excess = omega / C * np.real(mode.neff_minus_one(omega))
        return float(2 * excess[0] - excess[1] - excess[2])

    pressure = brentq(mismatch, 0.01, 5.0, xtol=1e-9)
    return Simulation.from_capillary(
        50e-6,
        0.5,
        "Ar",
        pressure,
        lambda0=400e-9,
        pulses=[
            GaussianPulse(wavelength=value, fwhm=30e-15, energy=10e-6)
            for value in (400e-9, 800e-9)
        ],
        wavelength_limits=(210e-9, 2e-6),
        trange=250e-15,
        materials=MaterialSelection(electronic="gas-polynomial"),
        backend="numpy_ref",
    )


def fiber(index: int = 0, *, noise: bool = False, collar: bool = True) -> Simulation:
    """Keep one member of the lockstep fixture, including its periodic statistics.

    ``collar=False`` builds the member without the temporal and spectral collars,
    for the batch case, whose fixed 0.1 mm steps the default collar would make
    unstable (SEP 21 refuses such a schedule).
    """
    return Simulation.from_gnlse(
        0.11 * (index + 1),
        1e-3,
        [0.0, 0.0, -1.183e-26 * (index + 1)],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=float(index + 1),
        pulseshape="sech",
        stats=StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=2),
        noise="modified" if noise else None,
        seed=index + 13 if noise else None,
        backend="numpy_ref",
        absorber=None if collar else False,
    )


def mixture_gradient() -> Simulation:
    """Use the manual's 80/20 He/Ar fill with the small capillary launch.

    The shared fill remains 3 to 0.2 bar over the original 1.5 m guide from
    ``guide/media/mixtures.md``. The radius, pulse and temporal grid are those
    of the capillary interface's short functional case; no pressure gradient
    is rescaled to the measured entrance window.
    """
    return Simulation.from_capillary(
        125e-6,
        1.5,
        {"He": 0.8, "Ar": 0.2},
        (3.0, 0.2),
        lambda0=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_r=8,
        backend="numpy_ref",
    )


def _dense_output() -> Simulation:
    beta2, gamma, width = -11.83e-27, 0.11, 100e-15
    return Simulation.from_gnlse(
        gamma,
        1.0,
        [0.0, 0.0, beta2],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=width,
        peak_power=4 * abs(beta2) / (gamma * width**2),
        pulseshape="sech",
        raman=False,
        shock=False,
        interpolate_planes=True,
        dz_init=1e-4,
        dz_max=1.0,
        backend="numpy_ref",
    )


def exercise(name: str) -> Simulation:
    """Select an explicit backend path without importing executable fixtures."""
    builders = {
        "vector_raman": vector_raman,
        "envelope_plasma": envelope_plasma,
        "dispersive_kerr": dispersive,
        "mixture_gradient": mixture_gradient,
        "dense_output": _dense_output,
        "far_field_stats": far_field,
    }
    if name in builders:
        return builders[name]()
    if name in ("plasma_band", "plasma_trapezoid", "ordinary_stats"):
        return plasma(name)
    if name in ("noise", "absorber_schemes"):
        simulation = fiber(noise=name == "noise")
        if name == "absorber_schemes":
            simulation.settings = replace(
                simulation.settings, absorber_scheme="split_step"
            )
        return simulation
    raise SuiteError(f"backend exercise not implemented for {name}")

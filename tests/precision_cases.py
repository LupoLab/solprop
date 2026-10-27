# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small fixed-schedule cases shared by the reduced-precision oracle tests.

The single-precision tests of SEP 33 and the mixed-precision tests of SEP 36 compare
each oracle with the binary64 reference on these problems: a soliton fibre, an ionising
field-grid capillary, a self-focusing Kerr beam on the spatial routes and a short
filament. Each takes the precision as its first argument.
"""

from contextlib import ExitStack

import numpy as np

from solprop.backends.protocol import Backend
from solprop.backends.registry import select
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.driver.simulation import SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.nonlinear.optmat_raman import gas_raman
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import gamma3
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import LowStorageRungeKutta, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule


def gnlse(  # noqa: PLR0913 - the case's own knobs
    precision: str,
    *,
    length: float = 0.05,
    steps: int | None = 10,
    gamma: float = 0.11,
    power: float = 100.0,
    raman: bool = True,
    integrator: RungeKutta | LowStorageRungeKutta | Multistep | None = None,
    beta0: float = 0.0,
    taper: bool = False,
    betas: list[float] | None = None,
    shock: bool = True,
    noise: str | None = None,
    seed: int | None = None,
    backend: str = "numpy_ref",
) -> Simulation:
    """Return a 1024-bin soliton fibre, replayed on a uniform schedule by default.

    `betas` replaces the dispersion (from ``beta0``), `shock` the self-steepening,
    `noise` and `seed` seed it with a noise model.
    """
    schedule = (
        None
        if steps is None
        else FixedSchedule.uniform(0.0, length, length / steps, save_planes=[length])
    )
    return Simulation.from_gnlse(
        gamma,
        length,
        [beta0, 0.0, -1.183e-26, 8.1e-41] if betas is None else betas,
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=50e-15,
        peak_power=power,
        pulseshape="sech",
        save_planes=[length],
        backend=backend,
        precision=precision,  # pyright: ignore[reportArgumentType]
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=seed,
        schedule=schedule,
        integrator=RungeKutta() if integrator is None else integrator,
        raman=raman,
        polarisation_taper=taper,
        shock=shock,
    )


def capillary(
    precision: str, gas: str | dict[str, float] = "Ar", **overrides
) -> Simulation:
    """Return a short field-grid capillary, ionising in argon, on 16 fixed steps.

    The split-step collar is unconditionally stable, so the steps need not stay inside
    the in-stage collar's limit, which a short capillary's strong collar lowers.
    """
    length = overrides.pop("length", 0.004)
    pressure = overrides.pop("pressure", 1.0)
    keywords = {
        "lambda0": 800e-9,
        "wavelength_limits": (150e-9, 3e-6),
        "trange": 300e-15,
        "fwhm": 10e-15,
        "energy": 60e-6,
        "n_save": 2,
        "backend": "numpy_ref",
        "precision": precision,
        "integrator": RungeKutta(),
        "schedule": FixedSchedule.uniform(0.0, length, length / 16),
        "absorber_scheme": "split_step",
    } | overrides
    return Simulation.from_capillary(
        50e-6, length, gas, pressure, **{"noise": None, **keywords}
    )


def loaded(simulation: Simulation, cleanup: ExitStack | None = None) -> Backend:
    """Return the simulation's backend with its launch loaded."""
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    if cleanup is not None:
        cleanup.callback(backend.close)
    backend.set_initial(problem.initial.field0, None)
    return backend


def finish(simulation: Simulation) -> np.ndarray:
    """Run to the last save plane and return the observed field."""
    backend = loaded(simulation)
    try:
        backend.propagate_to(float(simulation.settings.save_planes[-1]))
        field = backend.observe().field
        assert field is not None
        return field
    finally:
        backend.close()


def spatial_kerr(  # noqa: PLR0913 - the case's own knobs
    route: str,
    precision: str,
    *,
    steps: int = 10,
    absorbed: bool = False,
    energy: float = 3e-4,
    chi3: float | None = 1e-25,
    scheme: str = "nonlinear",
    window_factor: float = 0.1,
    integrator: RungeKutta | LowStorageRungeKutta | None = None,
) -> Simulation:
    """Return a self-focusing Kerr beam on the radial or Cartesian route.

    `absorbed` adds the grid's collars: the transverse window's in-stage rate beside
    the temporal one, whose scheme is `scheme`. ``chi3=None`` leaves the beam without a
    response, propagating linearly; ``window_factor=0`` removes the Cartesian grid's
    spatial collar. `integrator` replaces the default ``RungeKutta()``.
    """
    length = 0.1
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    spaces = {
        "radial": RadialGrid(2e-3, 32),
        "cartesian": FreeGrid(2e-3, 16, 2e-3, 16, window_factor),
        "line": LineGrid(2e-3, 16),
    }
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=energy), 300e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[length],
        dz_max=length,
        dz_init=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
        backend="numpy_ref",
        integrator=RungeKutta() if integrator is None else integrator,
        precision=precision,  # pyright: ignore[reportArgumentType]
        absorber=absorber(grid, boundary_length=100 * length) if absorbed else None,
        absorber_scheme=scheme,  # pyright: ignore[reportArgumentType]
    )
    constructor = Simulation.from_radial if route == "radial" else Simulation.from_free
    return constructor(
        grid,
        spaces[route],
        SpatialMedium(),
        [] if chi3 is None else [KerrResponse(chi3, thg=False)],
        pulse,
        settings=settings,
        noise=None,
    )


def plasma_beam(
    precision: str, *, route: str = "radial", gas: str = "Ar"
) -> Simulation:
    """Return a short filament on the radial (or Cartesian) route, ten steps.

    Nitrogen adds its molecular Raman response beside the plasma.
    """
    length = 1e-3
    grid = FieldGrid.from_limits(length, 800e-9, (350e-9, 2e-6), 150e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=2e-4), 100e-6
    )
    responses: list[object] = [
        KerrResponse(gamma3(gas, wavelength=800e-9), thg=False),
        PlasmaResponse(gas, OptmatRate(gas, 800e-9)),
    ]
    if gas == "N2":
        selection = MaterialSelection(carrier_wavelength=800e-9)
        responses.append(gas_raman(gas, selection=selection, thg=False))
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[length],
        dz_max=length,
        dz_init=length / 10,
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        backend="numpy_ref",
        integrator=RungeKutta(),
        ionisation_field_range=(1e7, 1e11),
        precision=precision,  # pyright: ignore[reportArgumentType]
    )
    if route == "cartesian":
        return Simulation.from_free(
            grid,
            FreeGrid(1e-3, 16, 1e-3, 16),
            SpatialMedium(gas, 1.0),
            responses,
            pulse,
            settings=settings,
        )
    return Simulation.from_radial(
        grid,
        RadialGrid(1e-3, 32),
        SpatialMedium(gas, 1.0),
        responses,
        pulse,
        settings=settings,
        noise=None,
    )


def envelope_plasma(precision: str, *, steps: int = 8) -> Simulation:
    """Return a 1 mm envelope-plasma capillary on 256 samples, `steps` uniform steps."""
    length = 0.001
    return capillary(
        precision,
        envelope=True,
        plasma=True,
        length=length,
        trange=150e-15,
        wavelength_limits=(300e-9, 2e-6),
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build declared cases after the worker has configured its scientific runtimes.

Each case is the interface call a user would write: no tolerance or integrator is
passed where the interface has a default (SEP 37), and every noisy launch fixes
:data:`~benchmarks.registry.SEED`. No gallery or test module executes here, and
no warning filter hides model warnings.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from benchmarks.records import Record, SuiteError
from benchmarks.registry import CASES, SEED

__all__ = ["build", "capillary", "gnlse", "integrator", "kerr", "seed", "spatial"]

if TYPE_CHECKING:
    from solprop.driver import Simulation, SimulationSettings
    from solprop.stepper import Multistep, RungeKutta
    from solprop.translate.precision import PrecisionName


def integrator(stepper: str) -> Multistep | RungeKutta | None:
    """Return the integrator a stepper override names; ``None`` keeps the default."""
    from solprop.stepper import Multistep, RungeKutta  # noqa: PLC0415

    return {
        "default": None,
        "multistep": Multistep(),
        "multistep-interaction": Multistep(formulation="interaction"),
        "rk": RungeKutta(),
    }[stepper]


def seed(precision: PrecisionName) -> int | None:
    """Return :data:`~benchmarks.registry.SEED` where the default noise draws a model.

    A reduced precision's default resolves to no noise until SEP 36 admits a model
    under it, and a seed beside that fallback is refused (SEP 37).
    """
    from solprop.driver.simulation import _resolve_noise  # noqa: PLC0415

    return SEED if _resolve_noise("default", None, precision)[0] is not None else None


def capillary(name: str, precision: PrecisionName = "double") -> Simulation:
    """Build a capillary case on its resolved or mode-averaged route."""
    from solprop.stats import default_spec  # noqa: PLC0415

    simulation = _resolved_capillary(name, precision)
    if name.endswith("_averaged"):
        # One resolved mode still integrates its transverse profile. The averaged
        # route uses its effective area and the matching ordinary statistics.
        simulation.settings = dataclasses.replace(
            simulation.settings,
            quadrature=None,
            components=None,
            stats=default_spec("mode_averaged"),
        )
    return simulation


def _resolved_capillary(name: str, precision: PrecisionName) -> Simulation:
    from solprop.driver import Simulation  # noqa: PLC0415 - worker-only imports

    if name.startswith("rdw") or name == "he_rdw_averaged":
        return Simulation.from_capillary(
            125e-6,
            1.5,
            "He",
            (0.8, 0.0),
            lambda0=800e-9,
            wavelength_limits=(90e-9, 4e-6),
            trange=400e-15,
            fwhm=7.5e-15,
            energy=275e-6,
            modes=4 if name == "rdw" else 1,
            seed=seed(precision),
            precision=precision,
        )
    if name == "ctc":
        return Simulation.from_capillary(
            16e-6,
            0.13,
            "H2",
            40.0,
            lambda0=515e-9,
            wavelength_limits=(220e-9, 4e-6),
            trange=2e-12,
            fwhm=400e-15,
            energy=10e-6,
            modes=3,
            loss=0.0,
            seed=seed(precision),
            precision=precision,
        )
    if name == "ctc_reduced":
        return Simulation.from_capillary(
            16e-6,
            0.1,
            "H2",
            40.0,
            lambda0=515e-9,
            wavelength_limits=(250e-9, 2e-6),
            trange=2e-12,
            fwhm=400e-15,
            energy=10e-6,
            modes=2,
            raman=True,
            temperature=294.0,
            seed=seed(precision),
            precision=precision,
        )
    if name.startswith("argon_blueshift"):
        return Simulation.from_capillary(
            13e-6,
            0.2,
            "Ar",
            1.7,
            lambda0=800e-9,
            fwhm=30e-15,
            energy=3e-6,
            loss=False,
            trange=400e-15,
            wavelength_limits=(160e-9, 4e-6),
            modes=1,
            envelope=name == "argon_blueshift_envelope",
            seed=seed(precision),
            precision=precision,
        )
    if name == "h2_raman_averaged":
        return Simulation.from_capillary(
            20e-6,
            0.4,
            "H2",
            10.0,
            lambda0=1030e-9,
            fwhm=40e-15,
            energy=3e-6,
            loss=False,
            plasma=False,
            trange=800e-15,
            wavelength_limits=(200e-9, 4e-6),
            modes=1,
            seed=seed(precision),
            precision=precision,
        )
    if name in ("h2_raman_6ps", "h2_raman_6ps_averaged"):
        return Simulation.from_capillary(
            20e-6,
            2.0,
            "H2",
            5.0,
            lambda0=1030e-9,
            fwhm=40e-15,
            energy=4e-6,
            loss=False,
            plasma=False,
            raman=True,
            trange=6e-12,
            wavelength_limits=(400e-9, 4e-6),
            modes=1,
            seed=seed(precision),
            precision=precision,
        )
    raise SuiteError(f"capillary builder not implemented for {name}")


def gnlse(name: str, precision: PrecisionName = "double") -> Simulation:
    """Extract the ideal soliton and the SEP 13 Raman/shock envelope inputs.

    Higher dispersion coefficients and the Raman/shock launch follow Dudley,
    Genty and Coen, Rev. Mod. Phys. 78, 1135 (2006),
    doi:10.1103/RevModPhys.78.1135, section III, as recorded in SEP 13.
    The ideal N=3 soliton uses the introduction's chosen scale coefficients.
    """
    import math  # noqa: PLC0415

    from solprop.driver import Simulation  # noqa: PLC0415

    if name == "gnlse_soliton":
        beta2, gamma, width = -11.83e-27, 0.11, 100e-15
        return Simulation.from_gnlse(
            gamma,
            math.pi / 2 * width**2 / abs(beta2),
            [0.0, 0.0, beta2],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=4e-12,
            width=width,
            peak_power=3**2 * abs(beta2) / (gamma * width**2),
            pulseshape="sech",
            raman=False,
            shock=False,
            seed=seed(precision),
            precision=precision,
        )
    if name == "gnlse_raman_shock":
        return Simulation.from_gnlse(
            0.11,
            0.15,
            [
                0.0,
                0.0,
                -11.830e-27,
                8.1038e-41,
                -9.5205e-56,
                2.0737e-70,
                -5.3943e-85,
                1.3486e-99,
                -2.5495e-114,
                3.0524e-129,
                -1.7140e-144,
            ],
            lambda0=835e-9,
            wavelength_limits=(400e-9, 2e-6),
            trange=12.5e-12,
            fwhm=50e-15 * 1.763,
            pulseshape="sech",
            peak_power=10e3,
            raman=True,
            shock=True,
            seed=seed(precision),
            precision=precision,
        )
    raise SuiteError(f"GNLSE builder not implemented for {name}")


def kerr(name: str, settings: SimulationSettings) -> Simulation:
    """Build SEP 28's full Cartesian envelope beam and material.

    The Sellmeier coefficients are inherited from the SEP 28 builder, citing
    Malitson, JOSA 55, 1205 (1965), doi:10.1364/JOSA.55.001205. This extraction
    preserves its declared UV extrapolation and records the resulting warning.
    """
    from solprop.constants import EPS0, MU0, C  # noqa: PLC0415
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.grid import EnvelopeGrid, FreeGrid  # noqa: PLC0415
    from solprop.nonlinear.responses import KerrResponse  # noqa: PLC0415
    from solprop.physdata.glasses import SellmeierGlass  # noqa: PLC0415
    from solprop.pulses import GaussianPulse, SpatialPulse  # noqa: PLC0415
    from solprop.spatial import SpatialMedium  # noqa: PLC0415

    size = CASES[name].dimensions[0]
    half_width, window = {
        192: (91.5e-6, 130e-15),
        384: (183e-6, 250e-15),
        768: (366e-6, 250e-15),
    }[size]
    grid = EnvelopeGrid.from_limits(
        CASES[name].length,
        260e-9,
        (143e-9, 600e-9),
        window,
        thg=False,
        fft_size="smooth",
    )
    space = FreeGrid(half_width * 1.1, size, half_width * 1.1, size, window_factor=0.0)
    glass = SellmeierGlass(
        name="malitson",
        source="Malitson 1965",
        wavelength_range=(0.21e-6, 3.71e-6),
        citation="Malitson, JOSA 55, 1205 (1965), doi:10.1364/JOSA.55.001205",
        poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
    )
    launch = SpatialPulse(
        GaussianPulse(wavelength=260e-9, fwhm=30e-15, energy=1e-7), 40e-6
    )
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium(glass),
        [KerrResponse(chi3=2.2e-22 * MU0 * EPS0 * C**2, thg=False)],
        launch,
        settings=settings,
    )


def spatial(name: str, settings: SimulationSettings) -> Simulation:
    """Extract the SEP 16 filament with its original physical resolution."""
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from solprop.constants import C  # noqa: PLC0415
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.grid import FieldGrid, FreeGrid, LineGrid, RadialGrid  # noqa: PLC0415
    from solprop.nonlinear.optmat_rates import OptmatRate  # noqa: PLC0415
    from solprop.nonlinear.responses import (  # noqa: PLC0415
        KerrResponse,
        PlasmaResponse,
    )
    from solprop.physdata import gamma3  # noqa: PLC0415
    from solprop.pulses import GaussianPulse, SpatialPulse  # noqa: PLC0415
    from solprop.spatial import SpatialMedium, critical_power  # noqa: PLC0415

    case = CASES[name]
    if name == "planar":
        return _planar(settings)
    plasma = "filament" in name
    grid = FieldGrid.from_limits(
        case.length,
        800e-9,
        (350e-9, 2e-6) if plasma else (500e-9, 1.6e-6),
        250e-15,
    )
    medium = SpatialMedium("Ar", 1.0)
    chi3 = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    fraction = 3.0 if plasma else 3000.0 if name == "line" else 0.9
    # The physical critical power at the fill's carrier index (SEP 34).
    carrier = 2 * math.pi * C / 800e-9
    index = float(np.real(medium.index(np.array([carrier]))[0]))
    power = fraction * critical_power(800e-9, chi3 * density, index=index)
    energy = power * 30e-15 * math.sqrt(math.pi / (4 * math.log(2)))
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), 100e-6
    )
    responses: list[KerrResponse | PlasmaResponse] = [KerrResponse(chi3, thg=False)]
    if plasma:
        responses.append(PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)))
    if plasma:
        settings = dataclasses.replace(settings, ionisation_field_range=(1e7, 1e11))
    if name.startswith("radial") or name == "spatial_absorber":
        return Simulation.from_radial(
            grid,
            RadialGrid(
                3e-4 if name == "spatial_absorber" else 1e-3, case.dimensions[0]
            ),
            medium,
            responses,
            pulse,
            settings=settings,
            seed=seed(settings.precision),
        )
    if name == "line" or name.startswith("cartesian_filament"):
        size = case.dimensions[0]
        space = (
            LineGrid(1e-3, size) if name == "line" else FreeGrid(6e-4, size, 6e-4, size)
        )
        return Simulation.from_free(
            grid, space, medium, responses, pulse, settings=settings
        )
    raise SuiteError(f"spatial builder not implemented for {name}")


def _planar(settings: SimulationSettings) -> Simulation:
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.grid import EnvelopeGrid  # noqa: PLC0415
    from solprop.modes.slab import SlabMode  # noqa: PLC0415
    from solprop.nonlinear.responses import KerrResponse  # noqa: PLC0415
    from solprop.physdata import gamma3  # noqa: PLC0415
    from solprop.pulses import GaussianPulse, PlanarPulse  # noqa: PLC0415
    from solprop.spatial import PlanarGrid  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(
        CASES["planar"].length, 800e-9, (650e-9, 1000e-9), 600e-15
    )
    half_gap = 63.5e-6
    modes = [SlabMode(half_gap, "Ar", 1.0, order=order) for order in (1, 3)]
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * half_gap,
    )
    return Simulation.from_planar(
        grid,
        PlanarGrid(modes, 1e-3, 32),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        pulse,
        settings=settings,
    )


def smoke() -> Simulation:
    """Build the runner's own short noiseless fibre."""
    from solprop.driver import Simulation  # noqa: PLC0415

    return Simulation.from_gnlse(
        0.11,
        CASES["smoke"].length,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(600e-9, 1200e-9),
        trange=200e-15,
        width=20e-15,
        peak_power=10.0,
        pulseshape="sech",
        raman=False,
        shock=False,
        n_save=2,
        noise=None,
    )


def build(configuration: Record) -> Simulation:
    """Build a case for the configured backend, stepper and observations.

    The tolerances are left to the interface. A stepper override replaces only the
    integrator. A case with a fixed step runs its declared schedule with the
    low-storage RK4IP march.
    """
    import numpy as np  # noqa: PLC0415

    from solprop.driver import SimulationSettings  # noqa: PLC0415
    from solprop.stepper import FixedSchedule, LowStorageRungeKutta  # noqa: PLC0415

    case = CASES[configuration["case"]]
    stop = float(configuration["stop"])
    # A shortened run (ab --length) saves its last plane where it stops.
    itinerary = np.linspace(0.0, stop, int(configuration["planes"]) + 1)[1:].tolist()
    fields = configuration["fields"]
    observed = {
        "save_planes": itinerary,
        "backend": configuration["backend"],
        "fields": fields,
        "field_planes": (len(itinerary),) if fields == "planes" else (),
    }
    chosen = integrator(configuration["stepper"])
    precision: PrecisionName = configuration.get("precision", "double")
    if case.builder in ("kerr", "spatial"):
        settings = SimulationSettings(
            precision=precision,
            save_planes=itinerary,
            dz_max=case.length,
            dz_init=case.length * 1e-4,
            auto_init=True,
            integrator=chosen,
            backend=configuration["backend"],
            fields=fields,
            field_planes=observed["field_planes"],
        )
        if case.step is not None:
            settings = dataclasses.replace(
                settings,
                integrator=LowStorageRungeKutta(scheme="rk4ip"),
                schedule=FixedSchedule(
                    np.diff(np.linspace(0.0, stop, max(1, round(stop / case.step)) + 1))
                ),
                propagator_mode="tabulate",
            )
        return (kerr if case.builder == "kerr" else spatial)(case.name, settings)
    if case.builder == "exercise":
        from benchmarks.exercises import exercise  # noqa: PLC0415

        simulation = exercise(case.name)
    elif case.builder == "advanced":
        from benchmarks.guides import advanced  # noqa: PLC0415

        options: dict[str, Any] = {"save_planes": itinerary}
        if chosen is not None:
            options["integrator"] = chosen
        simulation = advanced(case.name, case.length, options)
        if case.name == "grin_nonlinear":
            # Translated at construction (its launch is coupled): only the backend
            # and the field retention may still change.
            simulation.settings = dataclasses.replace(
                simulation.settings,
                backend=configuration["backend"],
                fields=fields,
                field_planes=observed["field_planes"],
            )
            return simulation
    elif case.builder == "capillary":
        simulation = capillary(case.name, precision)
    elif case.builder == "gnlse":
        simulation = gnlse(case.name, precision)
    elif case.builder == "smoke":
        simulation = smoke()
    else:
        raise SuiteError(f"unknown builder {case.builder}")
    if chosen is not None:
        observed["integrator"] = chosen
    simulation.settings = dataclasses.replace(simulation.settings, **observed)
    return simulation

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build declared cases after the worker has configured its scientific runtimes.

Definitions are extracted from the public sources recorded by the registry. No
gallery or test module executes here, and no warning filter hides model warnings.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from benchmarks.records import Record, SuiteError
from benchmarks.registry import ADAPTIVE, CASES, KERR, Case

__all__ = ["build", "capillary", "fixed_schedule", "gnlse", "kerr", "spatial"]

if TYPE_CHECKING:
    from solprop.driver import Simulation, SimulationSettings
    from solprop.stepper import FixedSchedule


def fixed_schedule(case: Case, configuration: Record) -> FixedSchedule:
    """Keep the exact measured step count and any unexecuted full-guide tail."""
    import numpy as np  # noqa: PLC0415 - worker-only imports

    from solprop.stepper import FixedSchedule  # noqa: PLC0415

    stop = float(configuration["stop"])
    edges = configuration.get("schedule", [])
    if edges:
        expected_steps = 4 if case.name == "envelope_plasma" else 10
        if len(edges) != expected_steps + 1 or edges[0] != 0 or edges[-1] > case.length:
            raise SuiteError(
                f"a fixed pilot schedule needs {expected_steps} "
                "entrance steps within the guide"
            )
    else:
        if case.step is None:
            raise SuiteError("fixed track requires a qualified schedule")
        # Freeze the integer count instead of allowing a rounded quotient to add
        # a step. The Kerr count includes its separate one-step lead-in.
        edges = np.linspace(0.0, stop, round(stop / case.step) + 1).tolist()
    lengths = np.diff(edges)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise SuiteError("fixed schedule edges must be finite and strictly increasing")
    if edges[-1] < case.length:
        corners = (5e-3,) if case.name == "moving_solid_guide" else ()
        tail = FixedSchedule.uniform(
            edges[-1],
            case.length,
            min(lengths),
            save_planes=[
                *(corner for corner in corners if corner > edges[-1]),
                case.length,
            ],
        )
        lengths = np.concatenate((lengths, tail.steps))
    return FixedSchedule(lengths)


def capillary(name: str) -> Simulation:
    """Build the original full guide, independently of the measured interval."""
    from solprop.stats import default_spec  # noqa: PLC0415

    simulation = _resolved_capillary(name)
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


def _resolved_capillary(name: str) -> Simulation:
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
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
        )
    if name in ("ctc", "ctc_reduced"):
        full = name == "ctc"
        # The full shape runs 0.3 m, where a noise-free run forms the continuum on
        # this 4 ps grid; the reduced case is the fundamental alone over 0.1 m on a
        # 2 ps grid, five times the pulse, which noise-free tests CtC-type propagation
        # without forming one. A 1 ps reduced window clipped the launch and seeded
        # the cascade.
        return Simulation.from_capillary(
            16e-6,
            0.3 if full else 0.1,
            "H2",
            40.0,
            lambda0=515e-9,
            wavelength_limits=(170e-9, 4e-6) if full else (250e-9, 2e-6),
            trange=4e-12 if full else 2e-12,
            fwhm=400e-15,
            energy=10e-6,
            modes=6 if full else 1,
            raman=True,
            temperature=294.0,
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
        )
    if name in ("argon_blueshift", "argon_blueshift_averaged"):
        return Simulation.from_capillary(
            13e-6,
            0.4,
            "Ar",
            1.7,
            lambda0=800e-9,
            fwhm=30e-15,
            energy=3e-6,
            loss=False,
            trange=400e-15,
            wavelength_limits=(300e-9, 4e-6),
            modes=1,
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
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
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
        )
    raise SuiteError(f"capillary builder not implemented for {name}")


def gnlse(name: str) -> Simulation:
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
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
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
            noise=None,
            rtol=1e-6,
            atol=1e-10,
            backend="numpy_ref",
        )
    raise SuiteError(f"GNLSE builder not implemented for {name}")


def kerr(name: str, settings: SimulationSettings) -> Simulation:
    """Build SEP 28's full Cartesian envelope beam and material.

    The Sellmeier coefficients are inherited from the SEP 28 builder, citing
    Malitson, JOSA 55, 1205 (1965), doi:10.1364/JOSA.55.001205. This extraction
    preserves its declared UV extrapolation and records the resulting warning.
    """
    from solprop.boundaries import no_absorber  # noqa: PLC0415
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
        collar_fraction=0.05,
        collar_placement="inside",
    )
    # The case's only track replays a fixed RK4IP schedule, whose 0.5 um steps the
    # default temporal collar would make unstable (SEP 21 refuses them), so the beam
    # keeps the geometry it was frozen with: no collars, the collar room inside the
    # window.
    settings = dataclasses.replace(
        settings, absorber=dataclasses.replace(no_absorber(grid), spatial=False)
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


def build(configuration: Record) -> Simulation:
    """Resolve the exact declared integrator and bounded observation itinerary."""
    import numpy as np  # noqa: PLC0415

    from solprop.driver import SimulationSettings  # noqa: PLC0415
    from solprop.stepper import (  # noqa: PLC0415
        LowStorageRungeKutta,
        Multistep,
        PIController,
        RungeKutta,
    )

    case = CASES[configuration["case"]]
    track = configuration["track"]
    if case.status != "implemented":
        raise SuiteError(f"{case.name}: {case.pending_reason}")
    start, stop = float(configuration["start"]), float(configuration["stop"])
    planes = np.linspace(start, stop, int(configuration["observations"]) + 1)[1:]
    # Translation sees the original guide endpoint; the benchmark loop only
    # visits the selected planes. No gradient or gas-flow law is rescaled.
    itinerary: list[float] = sorted(
        {*(float(value) for value in planes), case.length, *([start] if start else [])}
    )
    integrator = (
        Multistep()
        if track == ADAPTIVE[0]
        else LowStorageRungeKutta(scheme="rk4ip")
        if track == KERR
        else RungeKutta()
    )
    controller = PIController(
        rtol=configuration.get("reference_rtol") or configuration["rtol"] or 1e-6,
        atol=configuration["atol"] or 0.0,
    )
    schedule = fixed_schedule(case, configuration) if track not in ADAPTIVE else None
    settings = SimulationSettings(
        controller,
        save_planes=itinerary,
        dz_max=case.length,
        dz_init=case.length * 1e-4,
        auto_init=True,
        integrator=integrator,
        schedule=schedule,
        backend=configuration["backend"],
        fields=configuration["fields"],
        field_planes=(len(planes),) if configuration["fields"] == "planes" else (),
        propagator_mode="tabulate" if track == KERR else "auto",
        ionisation_field_range=(1e7, 1e11)
        if case.builder == "spatial" and "filament" in case.name
        else None,
    )
    if case.builder in ("capillary", "gnlse", "advanced", "exercise"):
        if case.builder == "exercise":
            from benchmarks.exercises import exercise  # noqa: PLC0415

            simulation = exercise(case.name)
        elif case.builder == "advanced":
            from benchmarks.guides import advanced  # noqa: PLC0415

            simulation = advanced(case.name)
        else:
            simulation = (
                capillary(case.name)
                if case.builder == "capillary"
                else gnlse(case.name)
            )
        simulation.settings = dataclasses.replace(
            simulation.settings,
            controller=controller,
            integrator=integrator,
            schedule=schedule,
            save_planes=itinerary,
            backend=configuration["backend"],
            fields=configuration["fields"],
            field_planes=(len(planes),) if configuration["fields"] == "planes" else (),
        )
        return simulation
    if case.builder == "kerr":
        return kerr(case.name, settings)
    if case.builder == "spatial":
        return spatial(case.name, settings)
    if case.builder == "smoke":
        from solprop.driver import Simulation  # noqa: PLC0415

        simulation = Simulation.from_gnlse(
            0.11,
            case.length,
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
            # The fixed track's 10 um steps are past the default collar's
            # stability limit, which SEP 21 refuses; the smoke case has none.
            absorber=False,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings,
            controller=controller,
            integrator=integrator,
            schedule=schedule,
            save_planes=itinerary,
            backend=configuration["backend"],
        )
        return simulation
    raise SuiteError(f"unknown builder {case.builder}")

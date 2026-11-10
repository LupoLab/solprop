# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share short binary64 numerical witnesses for the Julia session composition."""

from dataclasses import dataclass, replace

import numpy as np

from mixture_cases import AnalyticRate
from sessions.executors import soliton
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Translation, assemble
from solprop.translate.run_request import MethodKind, ReplaySpec


@dataclass(frozen=True)
class InstantaneousRate(AnalyticRate):
    """Declare the synthetic quadratic magnitude rate's physical convention."""

    convention: str = "instantaneous-isotropic"


def cpu_case(*, steps: int | None = 8, rtol=1e-8, dz_max=0.02):
    """Return a two-centimetre soliton without statistics."""
    original = soliton(planes=(0.01, 0.02))
    request = replace(
        original.request,
        error_control=replace(
            original.request.error_control,
            controller=PIController(rtol=rtol, atol=1e-16),
            dz_max=dz_max,
            dz_init=min(1e-3, dz_max),
        ),
        replay=None
        if steps is None
        else ReplaySpec(FixedSchedule(np.full(steps, 0.02 / steps))),
        observation=replace(
            original.request.observation, stats=StatsSpec(names=(), periodic=())
        ),
    )
    return Translation(original.problem, request)


def adams_case(*, method=None, steps: int | None = 8, rtol=1e-8, dz_max=0.02):
    """Return the same short soliton with resolved exponential Adams policy."""
    original = cpu_case(steps=steps, rtol=rtol, dz_max=dz_max)
    method = Multistep() if method is None else method
    request = replace(
        original.request,
        integration=replace(
            original.request.integration, kind=MethodKind.ADAMS, method=method
        ),
        error_control=replace(
            original.request.error_control, interval_cap=method.interval_cap
        ),
    )
    return Translation(original.problem, request)


def material_case(kind, response, *, steps=8):
    """Return an eight-micrometre scalar material witness without statistics."""
    length, wavelength = 8e-6, 800e-9
    grid_type = FieldGrid if kind == "field" else EnvelopeGrid
    grid = grid_type.from_limits(length, wavelength, (500e-9, 1.2e-6), 200e-15)
    omega = 2 * np.pi * C / wavelength
    mode = SpecifiedMode.from_taylor([omega / C, 1 / C], omega, aeff=1e-8)
    initial = GaussianPulse(wavelength=wavelength, fwhm=20e-15, energy=1e-5).build(grid)
    density = 1.0
    if response == "kerr":
        responses = [KerrResponse(chi3=1e-22, thg=False)]
    elif response == "raman":
        responses = [
            KerrResponse(chi3=1e-22, thg=False),
            response_for("SiO2", amplitude=1e-22, thg=False),
        ]
    else:
        # A declared quadratic law avoids transcribing atomic data. Both routes
        # consume its separately compiled instantaneous rate and own population.
        response_type = PlasmaResponse if kind == "field" else EnvelopePlasmaResponse
        responses = [
            response_type("Ar", InstantaneousRate(scale=1e13), preionisation=0.02)
        ]
        density = 1e24
    case = assemble(
        grid,
        mode,
        responses,
        initial,
        density=density,
        controller=PIController(rtol=1e-8, atol=1e-16),
        save_planes=[length],
        dz_init=length / steps,
        dz_max=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
        stats_spec=StatsSpec(names=(), periodic=()),
        ionisation_field_range=(1e7, 1e11) if response == "plasma" else None,
    )
    return case, grid, mode

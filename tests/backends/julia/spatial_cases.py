# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build seconds-long scalar spatial session witnesses."""

from dataclasses import replace

import numpy as np

from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import Translation, assemble
from solprop.translate.run_request import MethodKind


def spatial_case(
    kind,
    geometry,
    *,
    steps=8,
    method="rk",
    material=False,
    resolved=False,
):
    """Return an eight-micrometre scalar vacuum beam and its original grid."""
    length = 8e-6
    constructor = FieldGrid if kind == "field" else EnvelopeGrid
    grid = constructor.from_limits(length, 800e-9, (600e-9, 1.2e-6), 100e-15)
    fraction = 0.0
    if geometry == "radial":
        space = RadialGrid(4e-4, 16 if resolved else 8, collar_fraction=fraction)
    elif geometry == "line":
        space = LineGrid(4e-4, 24 if resolved else 8, collar_fraction=fraction)
    else:
        space = FreeGrid(
            4e-4,
            12 if resolved else 6,
            5e-4,
            10 if resolved else 4,
            collar_fraction=fraction,
        )
    pulse = SpatialPulse(
        GaussianPulse(
            wavelength=800e-9,
            fwhm=20e-15,
            energy=1e-5 if geometry == "line" else 1e-9,
        ),
        100e-6,
        jones=(1,),
    )
    medium = SpatialMedium()
    case = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)] if material else [],
        pulse.build(grid, space, medium=medium),
        spatial_grid=space,
        medium=medium,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length,
        dz_init=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
    )
    request = case.request.for_run(stats=StatsSpec(names=(), periodic=()))
    integration = request.integration
    if method == "adams":
        integration = replace(integration, kind=MethodKind.ADAMS, method=Multistep())
    elif method == "low_storage":
        integration = replace(
            integration, kind=MethodKind.LOW_STORAGE, method=LowStorageRungeKutta()
        )
    return Translation(case.problem, replace(request, integration=integration)), grid


def spatial_collar(case):
    """Attach a represented synthetic physical loss for source-ownership witnesses."""
    route = case.problem.route
    transform = route.transform
    assert transform.geometry is not None
    represented = replace(
        transform.geometry,
        absorber_rate=np.linspace(0.0, 1000.0, transform.n_nodes),
    )
    transform = replace(transform, geometry=represented)
    return Translation(
        replace(case.problem, route=replace(route, transform=transform)),
        case.request,
    )

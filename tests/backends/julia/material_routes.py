# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build short two-population witnesses with independently varying Raman kernels."""

from dataclasses import replace

import numpy as np

from backends.julia.cases import InstantaneousRate
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.modes.quadrature import transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.raman import DampedOscillator, RamanResponse
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Component,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
)
from solprop.physdata.raman_types import Dephasing, Linewidth
from solprop.profiles import Constant, Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

LENGTH = 1e-6
NAMES = (
    "energy",
    "species_ionisation",
    "species_electron_density",
    "electron_density",
    "peak_ionisation_rate",
)


def material_route(kind, geometry, *, gradient=False, taper=False):
    """Return a synthetic scalar mixture, its translation and observation context.

    All material numbers define test functions, not fitted atomic or molecular data.
    Two independently labelled densities, preionisations and linewidths exercise the
    actual translated records. The temporal window and length stay fixed as the march
    is refined; nonlinear propagation covers only one micrometre. This keeps the
    field carrier phase resolved by eight steps of every admitted march family.
    """
    constructor = FieldGrid if kind == "field" else EnvelopeGrid
    grid = constructor.from_limits(LENGTH, 800e-9, (500e-9, 1.2e-6), 160e-15)
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.zeros(2),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic constant susceptibility for numerical qualification",
    )
    components = tuple(
        Component(
            Species(
                f"synthetic-{index}",
                optical,
                gamma3=(index + 1) * 1e-46,
                ionisation=IonisationData((index + 1) * 1e-18),
            ),
            Linear((index + 1) * 1e24, (3 - index) * 1e24, LENGTH)
            if gradient
            else Constant((index + 1) * 1e24),
            f"population-{index}",
            preionisation=0.03 + 0.07 * index,
        )
        for index in range(2)
    )
    mixture = Mixture.of(*components)
    plasma = PlasmaResponse if kind == "field" else EnvelopePlasmaResponse
    responses = []
    for index, component in enumerate(components):
        assert component.species.gamma3 is not None
        responses.extend(
            (
                KerrResponse(
                    component.species.gamma3, thg=False, species=component.label
                ),
                RamanResponse(
                    models=(
                        DampedOscillator(
                            (index + 1) * 1e-32,
                            (index + 1) * 1e14,
                            Dephasing(
                                Linewidth(
                                    constant=(index + 1) * 1e12,
                                    density=(2 - index) * 1e13,
                                )
                            ),
                        ),
                    ),
                    thg=False,
                    species=component.label,
                ),
                plasma(
                    component.species,
                    InstantaneousRate(scale=(index + 1) * 1e13),
                    species=component.label,
                    preionisation=component.preionisation,
                ),
            )
        )
    pulse = GaussianPulse(
        wavelength=800e-9,
        fwhm=20e-15,
        energy=1e-4 if geometry == "line" else 1e-8,
    )
    space = None
    medium = None
    quadrature = None
    if geometry in ("averaged", "multimode"):
        radius = Linear(50e-6, 49.9e-6, LENGTH) if taper else 50e-6
        modes = [
            MarcatiliMode(radius, mixture=mixture, cladding=1.45, loss=False, m=index)
            for index in ((1, 2) if geometry == "multimode" else (1,))
        ]
        field = pulse.build(grid)
        if geometry == "multimode":
            quadrature = transverse_quadrature(modes, n_r=8)
            field = np.concatenate((0.8 * field, 0.6j * field), axis=0)
    else:
        assert not gradient and not taper
        modes = None
        if geometry == "radial":
            space = RadialGrid(4e-4, 8, collar_fraction=0.0)
        elif geometry == "cartesian":
            space = FreeGrid(4e-4, 6, 5e-4, 4, collar_fraction=0.0)
        else:
            assert geometry == "line"
            space = LineGrid(4e-4, 8, collar_fraction=0.0)
        medium = SpatialMedium(mixture)
        field = SpatialPulse(pulse, 100e-6, jones=(1,)).build(
            grid, space, medium=medium
        )
    case = assemble(
        grid,
        modes,
        responses,
        field,
        spatial_grid=space,
        medium=medium,
        quadrature=quadrature,
        taper_coupling=False,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / 8,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
        stats_spec=StatsSpec(names=NAMES, periodic=()),
        ionisation_field_range=(1e7, 1e11),
    )
    return case, grid, modes


def refined_request(request, steps):
    """Refine only the fixed march, preserving the scientific descriptor."""
    assert request.replay is not None
    return replace(
        request,
        replay=replace(
            request.replay,
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        ),
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic mixture data and analytic rates shared by channel qualification tests."""

from dataclasses import dataclass

import numpy as np

from legacy_geometry import LEGACY_GRID
from solprop.constants import C
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Component,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
)
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse
from solprop.stepper import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble


@dataclass(frozen=True)
class AnalyticRate:
    """Supply a clipped power law, with a constant-rate case for closed forms."""

    scale: float = 1e13
    exponent: int = 2
    reference: float = 3e10
    threshold: float = 1e7
    maximum: float = 1e16
    citation: str = "Synthetic rate for analytic channel tests"

    @property
    def parameters(self):
        return tuple(
            sorted(
                (
                    ("scale", self.scale),
                    ("exponent", self.exponent),
                    ("reference", self.reference),
                    ("threshold", self.threshold),
                    ("maximum", self.maximum),
                )
            )
        )

    def __call__(self, field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(
            magnitude >= self.threshold,
            np.minimum(
                self.maximum, self.scale * (magnitude / self.reference) ** self.exponent
            ),
            0.0,
        )


def mixture_case(
    *, exponent=2, kerr=True, density_scale=1.0, preionisation=None, schedule=None
):
    """Construct a small field problem with two independent synthetic populations.

    ``preionisation`` replaces the two components' initial fractions (numbers or paths
    along z) and ``schedule`` gives the run a fixed schedule.
    """
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.zeros(2),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic constant refractivity with no absorption",
    )
    components = tuple(
        Component(
            Species(
                f"synthetic {i}",
                optical,
                gamma3=(i + 1) * 1e-52,
                ionisation=IonisationData((i + 1) * 1e-18),
            ),
            Constant(density_scale * (i + 1) * 1e24),
            f"population-{i}",
            preionisation=0.03 + 0.07 * i
            if preionisation is None
            else preionisation[i],
        )
        for i in range(2)
    )
    mixture = Mixture.of(*components)
    # The held-current certificate requires a resolved collar at both edges; the
    # case is sized for the former geometry.
    grid = FieldGrid.from_limits(
        1e-3, 800e-9, (400e-9, 2e-6), 100e-15, ffac=4.0, **LEGACY_GRID
    )
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    responses = []
    for index, component in enumerate(components):
        if kerr:
            assert component.species.gamma3 is not None
            responses.append(
                KerrResponse(
                    component.species.gamma3, thg=False, species=component.label
                )
            )
        rate = build_rate_table(
            AnalyticRate(scale=(index + 1) * 1e13, exponent=exponent), (1e7, 1e11)
        )
        responses.append(
            PlasmaResponse(
                component.species,
                rate,
                preionisation=component.preionisation,
                species=component.label,
            )
        )
    initial = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9).build(grid)
    problem = assemble(
        grid,
        mode,
        responses,
        initial,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=1e-4,
        dz_init=1e-5,
        schedule=schedule,
    )
    return mixture, grid, mode, tuple(responses), problem

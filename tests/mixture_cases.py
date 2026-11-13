# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic mixture data and analytic rates shared by channel qualification tests."""

import dataclasses
from dataclasses import dataclass

import numpy as np

from legacy_geometry import LEGACY_GRID
from solprop.constants import C
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Component,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
)
from solprop.profiles import Constant, Linear
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import Flags
from solprop.translate.responses import compile_responses


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


def mixture_case(**options):
    """Construct a small field problem with two independent synthetic populations.

    ``preionisation`` replaces the two components' initial fractions (numbers or paths
    along z) and ``schedule`` gives the run a fixed schedule
    (:func:`mixture_translation`, whose problem this returns in place of the pair).
    """
    mixture, grid, mode, responses, (problem, _) = mixture_translation(**options)
    return mixture, grid, mode, responses, problem


def mixture_translation(
    *, exponent=2, kerr=True, density_scale=1.0, preionisation=None, schedule=None
):
    """Return :func:`mixture_case`'s parts with its problem's translation, the pair.

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
    translation = assemble(
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
    return mixture, grid, mode, tuple(responses), translation


def construct(mixture, responses, grid, original, original_request):
    """Assemble `mixture` in the mixture case's capillary on sixteen fixed steps.

    The launch, controller and steps are :func:`mixture_translation`'s (`original` and
    `original_request`); returns the mode, the problem and its run's request.
    """
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        original.initial.field0,
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=original_request.stepper.dz_max,
        dz_init=original_request.stepper.dz_init,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 16),
    )
    return mode, problem, run_request


def independent_profiles(*, frozen=False):
    """Return the mixture case with three independent density profiles along z.

    The two populations' densities fall and rise linearly and an inert third falls
    from 7e24 to 0.8e24 m^-3, so the operator and the prefactor are z tables; sixteen
    fixed steps (:func:`construct`). `frozen` holds every density at its launch value
    instead. Returns the mixture, the grid, the mode, the problem and its run's request.
    """
    mixture, grid, _, responses, (original, original_request) = mixture_translation()

    def density(start, end):
        return Constant(start) if frozen else Linear(start, end, grid.length)

    components = tuple(
        dataclasses.replace(
            component, density=density((index + 1) * 1e24, (4 - index) * 1e24)
        )
        for index, component in enumerate(mixture.components)
    )
    third = Component(
        dataclasses.replace(components[0].species, gamma3=0.0, ionisation=None),
        density(7e24, 0.8e24),
        "inert",
    )
    mixture = Mixture.of(*components, third)
    mode, problem, run_request = construct(
        mixture, responses, grid, original, original_request
    )
    return mixture, grid, mode, problem, run_request


def raman_mixture():
    """Return the mixture case with an independent Raman oscillator per population.

    Population ``i`` takes an undamped oscillator of amplitude ``(i + 1) 1e-37`` and
    rate ``(i + 1) 2e14 1/s``, compiled beside the case's own responses into one
    problem. Returns the grid, the problem and its run's request.
    """
    mixture, grid, mode, responses, (problem, run_request) = mixture_translation()
    extra = tuple(
        RamanResponse(
            models=(
                DampedOscillator((i + 1) * 1e-37, (i + 1) * 2e14, undamped_dephasing()),
            ),
            thg=False,
            species=label,
        )
        for i, label in enumerate(mixture.labels)
    )
    compiled = compile_responses(
        (*responses, *extra),
        grid,
        mode,
        density=mode.density,
        z_max=grid.length,
        mixture=mixture,
    )
    problem = dataclasses.replace(problem, responses=compiled)
    problem = dataclasses.replace(
        problem,
        flags=Flags.from_content(
            shape=problem.shape,
            linear=problem.linear,
            transform=problem.transform,
            responses=compiled,
            initial=problem.initial,
        ),
    )
    return grid, problem, run_request

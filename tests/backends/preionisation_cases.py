# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The two-population cases whose pre-ionisation follows a path along z.

Shared by ``test_preionisation_paths.py`` and the catalogue's mixture rows
(``backends/mixture_rows.py``): the mixture case's field problem on its fixed ladder of
``dz_init`` steps, and an envelope problem of the same two populations on four fixed
steps, each with a path per population -- a stepped ramp between two kinks on the
ladder's edges, or a smooth rise with no kink at all.
"""

from dataclasses import dataclass

import numpy as np

from mixture_cases import AnalyticRate, mixture_case, mixture_translation
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.profiles import FromCallable, PiecewiseLinear
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table

NAMES = (
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "electron_density",
    "peak_ionisation_rate",
)
"""The field case's statistics."""

FIELD_LENGTH = 1e-3
"""The mixture case's fibre, m; its fixed ladder takes steps of ``dz_init = 1e-5``."""

FIELD_STEPS = 8
"""The steps of the field ladder that are taken."""

FIELD_DZ = 1e-5
"""The field ladder's step, m."""


def stepped(length, low, high, start, end):
    """Return a fraction that ramps from `low` to `high` between two kinks, in m."""
    return PiecewiseLinear(
        np.array([0.0, start, end, length]), np.array([low, low, high, high])
    )


def smooth(length, low, high):
    """Return a fraction rising smoothly along the fibre, with no kink at all."""
    return FromCallable(
        lambda z: (
            low + (high - low) * np.sin(0.5 * np.pi * np.asarray(z) / length) ** 2
        ),
        (0.0, length),
    )


def field_case(preionisation):
    """Return the two-population field case on its fixed eight-step ladder."""
    _, grid, _, _, (problem, run_request) = mixture_translation(
        preionisation=preionisation,
        schedule=FixedSchedule.uniform(0.0, FIELD_LENGTH, FIELD_DZ),
    )
    return problem, run_request, grid


# Kinks on the ladder's edges (a fixed schedule must land on every one), inside the
# eight steps that are taken.
FIELD_PATHS = {
    "stepped-smooth": (
        stepped(FIELD_LENGTH, 0.02, 0.2, 3 * FIELD_DZ, 5 * FIELD_DZ),
        smooth(FIELD_LENGTH, 0.05, 0.3),
    ),
    "smooth-stepped": (
        smooth(FIELD_LENGTH, 0.01, 0.1),
        stepped(FIELD_LENGTH, 0.3, 0.05, 2 * FIELD_DZ, 6 * FIELD_DZ),
    ),
}
"""The field case's two paths, by name."""

ENVELOPE_LENGTH = 8e-6
"""The envelope case's length, m: four fixed steps."""


@dataclass(frozen=True)
class _InstantaneousRate(AnalyticRate):
    """The synthetic power law, declared an instantaneous isotropic rate."""

    convention: str = "instantaneous-isotropic"


def envelope_case(preionisation):
    """Return the two-population envelope case on a four-step fixed ladder."""
    mixture, _, _, _, _ = mixture_case(kerr=False, preionisation=preionisation)
    responses = tuple(
        EnvelopePlasmaResponse(
            component.species,
            build_rate_table(
                _InstantaneousRate(scale=(index + 1) * 1e13), (1e7, 1e11), rtol=1e-7
            ),
            preionisation=component.preionisation,
            species=component.label,
        )
        for index, component in enumerate(mixture.components)
    )
    grid = EnvelopeGrid.from_limits(ENVELOPE_LENGTH, 800e-9, (500e-9, 1.2e-6), 80e-15)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    initial = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7).build(grid)
    step = ENVELOPE_LENGTH / 4
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        initial,
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[ENVELOPE_LENGTH],
        dz_max=step,
        dz_init=step,
        schedule=FixedSchedule.uniform(0.0, ENVELOPE_LENGTH, step),
    )
    return problem, run_request, grid


ENVELOPE_PATHS = {
    "stepped-smooth": (
        stepped(ENVELOPE_LENGTH, 0.02, 0.2, ENVELOPE_LENGTH / 4, ENVELOPE_LENGTH / 2),
        smooth(ENVELOPE_LENGTH, 0.05, 0.3),
    ),
    "smooth-stepped": (
        smooth(ENVELOPE_LENGTH, 0.01, 0.1),
        stepped(
            ENVELOPE_LENGTH, 0.3, 0.05, ENVELOPE_LENGTH / 2, 3 * ENVELOPE_LENGTH / 4
        ),
    ),
}
"""The envelope case's two paths, by name."""

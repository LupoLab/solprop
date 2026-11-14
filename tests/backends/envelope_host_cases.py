# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope plasma host route's argon capillary, elliptically polarised.

Shared by ``test_envelope_plasma_host.py`` and the catalogue's mixture rows
(``backends/mixture_rows.py``).
"""

import numpy as np

from rate_oracles import adk_oracle
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import PIController
from solprop.translate.assembly import assemble


def argon_host_case():
    """Return argon in a 125 um capillary, 10 uJ in 20 fs, ellipticity 0.35.

    An envelope grid of 1 mm, cycle-averaged ADK plasma a tenth pre-ionised, steps of
    1 um saved at 1 um and 8 um. Returns the problem, its run's request and the grid.
    """
    grid = EnvelopeGrid.from_limits(0.001, 800e-9, (500e-9, 1.2e-6), 80e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-5).build(grid)
    field = pulse * np.array([1.0, 0.35j])[None, :, None] / np.sqrt(1 + 0.35**2)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    problem, run_request = assemble(
        grid,
        mode,
        [EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=0.1)],
        field,
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[1e-6, 8e-6],
        dz_max=1e-6,
        dz_init=1e-6,
        ionisation_field_range=(1e9, 1e11),
    )
    return problem, run_request, grid

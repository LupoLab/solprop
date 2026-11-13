# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An ionising argon capillary at its entrance, built for each kind of on-axis row.

The on-axis plasma statistics scan the row the mode profiles synthesise on the axis
(SEP 19, SEP 47 phase 4). These cases build one problem per kind of row and scan: a
field grid's one-component scan of two rows, an envelope's of two rows (their
ellipse), the band rule's filtered exposure, and an envelope's cycle-averaged scan of
one and two modes. ``tests/sessions/test_native_session_axis_plasma.py`` runs them.
"""

import math
import warnings
from typing import Any

import numpy as np

from rate_oracles import adk_oracle
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest


def entrance_problem(
    kind: str,
    schedule: FixedSchedule | None = None,
    save_planes: tuple[float, ...] = (0.02,),
) -> tuple[Any, list[MarcatiliMode], PropagationProblem, RunRequest]:
    """Return an argon capillary of one or two modes at its entrance, for a row kind.

    ``rotated-x`` carries modes with a y field on the axis on a transform resolving x
    alone, so a field grid's one-component scan reads the magnitude of both rows;
    ``envelope-rotated-x`` is the envelope form, whose scan reads both rows' ellipse.
    """
    envelope = kind.startswith("envelope")
    grid = (EnvelopeGrid if envelope else FieldGrid).from_limits(
        0.02, 800e-9, (300e-9, 3e-6), 100e-15
    )
    table = build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))

    def argon(m=1, phi=0.0):
        return MarcatiliMode(125e-6, "Ar", 1.0, m=m, cladding=1.45, loss=False, phi=phi)

    rotated = kind.endswith("rotated-x")
    modes = (
        [argon(phi=math.pi / 4), argon(2, phi=math.pi / 3)]
        if rotated
        else [argon(1), argon(2)]
        if kind.endswith("two-modes")
        else [argon()]
    )
    plasma = (
        EnvelopePlasmaResponse(Gas.Ar, table)
        if envelope
        else PlasmaResponse(Gas.Ar, table, integration="band")
        if kind == "band"
        else PlasmaResponse(Gas.Ar, table)
    )
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-4).build(grid)
    start = np.concatenate([built, 0.4j * built][: len(modes)], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, request = assemble(
            grid,
            modes,
            [KerrResponse(chi3=2.2e-52, thg=not envelope), plasma],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=list(save_planes),
            dz_max=0.002,
            dz_init=1e-4,
            schedule=schedule,
            quadrature=transverse_quadrature(
                modes, n_r=32, radial_rule="gauss_kronrod"
            ),
            components=("x",) if rotated else None,
        )
    return grid, modes, problem, request


ENTRANCE_KINDS = (
    "rotated-x",
    "envelope-rotated-x",
    "band",
    "envelope",
    "envelope-two-modes",
)

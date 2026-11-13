# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session gives the transverse quadrature alarm as the reference does.

A rule too coarse for the field is a statistic (``quadrature_error``) and one logged
warning a plane, never a controller input. The native session reads the largest
accepted step's score once an advance and gives the shared warning
(``solprop.translate.modal.warn_quadrature``, at ``QUADRATURE_ALARM``), once between
two observations; a rule that resolves the field gives none.
"""

import logging
import warnings
from typing import Any, cast

import numpy as np
import pytest

from sessions.executors import Started, drive, entrance
from solprop.backends.registry import select
from solprop.backends.sessions import start_session
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import PIController
from solprop.translate.assembly import assemble

pytest.importorskip("solprop_native.session")

LENGTH = 0.02
"""Two centimetres of the capillary: the alarm is scored at every accepted step."""


def _problem(n_r: int):
    """Return the three-mode Kerr run of ``test_numpy_ref_multimode.py`` on `n_r`."""
    grid = FieldGrid.from_limits(0.2, 800e-9, (300e-9, 3e-6), 100e-15)
    modes = [
        MarcatiliMode(125e-6, "He", 1.0, m=m, cladding=1.45, loss=False)
        for m in (1, 2, 3)
    ]
    rule = transverse_quadrature(modes, n_r=n_r, radial_rule="gauss_kronrod")
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=2e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, request = assemble(
            grid,
            modes,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            np.repeat(built, 3, axis=0),
            quadrature=rule,
            controller=PIController(rtol=1e-11, atol=1e-16),
            save_planes=[LENGTH / 2, LENGTH],
            dz_max=LENGTH / 8,
            dz_init=1e-4,
        )
    return grid, modes, problem, request


def _warnings(name: str, n_r: int, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Run the case on a backend; return the alarm warnings it logged."""
    grid, modes, problem, request = _problem(n_r)
    backend = select(name, problem, request, grid=grid, modes=modes)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="solprop.translate.modal"):
        session = start_session(backend, problem, request, entrance(problem))
        drive(Started(session))
    if name == "native":
        # The largest score the session drained for its alarm is in its record.
        record = cast("dict[str, Any]", session.execution_record()["native_session"])
        assert (record["quadrature_peak"] > 1e-6) == (n_r < 32)
    return [
        record.getMessage()
        for record in caplog.records
        if "transverse quadrature" in record.getMessage()
    ]


def test_a_rule_too_coarse_warns_once_a_plane_on_either_backend(caplog):
    # Nine radial nodes cannot integrate the quartic of three Bessel profiles.
    reference = _warnings("numpy_ref", 8, caplog)
    native = _warnings("native", 8, caplog)
    assert len(reference) >= 1
    assert len(native) == len(reference)
    for one in native:
        assert "above the 1e-06 alarm" in one


def test_a_rule_that_resolves_the_field_warns_on_neither(caplog):
    assert _warnings("numpy_ref", 32, caplog) == []
    assert _warnings("native", 32, caplog) == []

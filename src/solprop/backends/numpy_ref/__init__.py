# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference backend: the oracle.

Plain NumPy and ``scipy.fft`` through :mod:`solprop.fourier`, one named function per
pass, no fusion, no in-place tricks and no compiled code. It exists to be read and to be
right: every other backend is validated against the numbers it produces
(``docs/dev/backends.md`` section 2). It propagates a mode-averaged or multimode problem
on either grid with the Kerr, Raman and plasma terms, using interaction-picture DOPRI5
with the cumulative operator, the lagged PI controller, landing on save planes and hard
kinks, the shot-noise field and the hard band mask.

:mod:`solprop.backends.numpy_ref.passes` holds the passes;
:mod:`solprop.backends.numpy_ref.backend` holds the step loop and the seam.
"""

from solprop.backends.numpy_ref.backend import (
    QUADRATURE_ALARM,
    NumpyRefBackend,
)
from solprop.backends.numpy_ref.passes import (
    accept_step,
    accumulate,
    apply_hard_mask,
    apply_temporal_rate,
    error_norm,
    ionisation_rate,
    kerr_polarisation,
    nonlinear_polarisation,
    plasma_current,
    plasma_scan,
    project,
    raman_polarisation,
    raman_spectrum,
    stage_input,
    stage_output,
    synthesise,
    to_frequency,
    to_time,
)

__all__ = [
    "QUADRATURE_ALARM",
    "NumpyRefBackend",
    "accept_step",
    "accumulate",
    "apply_hard_mask",
    "apply_temporal_rate",
    "error_norm",
    "ionisation_rate",
    "kerr_polarisation",
    "nonlinear_polarisation",
    "plasma_current",
    "plasma_scan",
    "project",
    "raman_polarisation",
    "raman_spectrum",
    "stage_input",
    "stage_output",
    "synthesise",
    "to_frequency",
    "to_time",
]

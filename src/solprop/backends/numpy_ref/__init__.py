# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference backend: the oracle.

Plain NumPy and ``scipy.fft`` through :mod:`solprop.fourier`, one named function per
pass, no fusion, no in-place tricks and no compiled code. It exists to be read and to be
right: every other backend is validated against the numbers it produces
(``docs/dev/backends.md`` section 2). It runs every route and every march.

Reading order: :mod:`~solprop.backends.numpy_ref.passes` holds the formulas, one
function per pass of a stage; :mod:`~solprop.backends.numpy_ref.backend` the seam
(:class:`~solprop.backends.numpy_ref.backend.NumpyRefBackend`), which builds the
workspace (:mod:`~solprop.backends.numpy_ref.workspace`), the engines
(:mod:`~solprop.backends.numpy_ref.engines`,
:mod:`~solprop.backends.numpy_ref.multistep`) and the march of each run. The workspace
owns the stage source (:mod:`~solprop.backends.numpy_ref.source`), the statistics
(:mod:`~solprop.backends.numpy_ref.statistics`) and the coupled flow
(:mod:`~solprop.backends.numpy_ref.coupled_flow`); the Adams arithmetic and coefficients
are :mod:`~solprop.backends.numpy_ref.adams_kernels` and
:mod:`~solprop.backends.numpy_ref.adams_tables`, the general modal guide
:mod:`~solprop.backends.numpy_ref.general_modal`, the dense output
:mod:`~solprop.backends.numpy_ref.dense`.

The stateful control of a run is the reference's own: its session
(:mod:`~solprop.backends.numpy_ref.session`, composed of the parts in
:mod:`~solprop.backends.numpy_ref.session_parts`), which schedules a run and drives its
march, and the marches it drives (:mod:`~solprop.backends.numpy_ref.marches`): the
Runge-Kutta march (:mod:`~solprop.backends.numpy_ref.marches.runge_kutta`), the Adams
march (:mod:`~solprop.backends.numpy_ref.marches.adams`) and the low-storage march
(:mod:`~solprop.backends.numpy_ref.marches.low_storage`). They are the Python statement
of what native's ``session.cpp`` and compiled marches do, and native is traced against
them, but native runs in its own session and imports none of them. What the two backends
share is stateless: the controller cores (:mod:`solprop.backends.adams_core`,
:mod:`solprop.backends.rk_core`), the engine protocols
(:mod:`solprop.backends.engines`), the low-storage method
(:mod:`solprop.backends.low_storage`), the tableaux, stage tables, exponential
coefficients and call ledger, and the preparation refusals
(:func:`solprop.backends.session.check_preparation`).
"""

from solprop.backends.numpy_ref.backend import NumpyRefBackend

__all__ = ["NumpyRefBackend"]

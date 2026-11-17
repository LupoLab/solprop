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
workspace (:mod:`~solprop.backends.numpy_ref.workspace`) and the engines the marches of
:mod:`solprop.backends.host` drive (:mod:`~solprop.backends.numpy_ref.engines`,
:mod:`~solprop.backends.numpy_ref.multistep`). The workspace owns the stage source
(:mod:`~solprop.backends.numpy_ref.source`), the statistics
(:mod:`~solprop.backends.numpy_ref.statistics`) and the coupled flow
(:mod:`~solprop.backends.numpy_ref.coupled_flow`); the Adams arithmetic and coefficients
are :mod:`~solprop.backends.numpy_ref.adams_kernels` and
:mod:`~solprop.backends.numpy_ref.adams_tables`, the general modal guide
:mod:`~solprop.backends.numpy_ref.general_modal`, the dense output
:mod:`~solprop.backends.numpy_ref.dense`.
"""

from solprop.backends.numpy_ref.backend import NumpyRefBackend

__all__ = ["NumpyRefBackend"]

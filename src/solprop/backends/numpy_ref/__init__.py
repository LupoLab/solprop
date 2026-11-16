# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference backend: the oracle.

Plain NumPy and ``scipy.fft`` through :mod:`solprop.fourier`, one named function per
pass, no fusion, no in-place tricks and no compiled code. It exists to be read and to be
right: every other backend is validated against the numbers it produces
(``docs/dev/backends.md`` section 2). It runs every route and every march.

Reading order: :mod:`.passes` holds the formulas, one function per pass of a stage;
:mod:`.backend` the seam (:class:`NumpyRefBackend`), which builds the workspace
(:mod:`.workspace`) and the engines the marches of :mod:`solprop.backends.host` drive
(:mod:`.engines`, :mod:`.multistep`). The workspace owns the stage source
(:mod:`.source`), the statistics (:mod:`.statistics`) and the coupled flow
(:mod:`.coupled_flow`); the Adams arithmetic and coefficients are :mod:`.adams_kernels`
and :mod:`.adams_tables`, the general modal guide :mod:`.general_modal`, the dense
output :mod:`.dense`. ``construct`` is the registry's constructor.
"""

from collections.abc import Mapping, Sequence

from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = ["NumpyRefBackend", "construct"]


def construct(
    problem: PropagationProblem,
    request: RunRequest,
    *,
    grid: TimeGrid | None = None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    tdse_binding: TDSEBinding | None = None,
    options: ExecutionOptions | None = None,
) -> NumpyRefBackend:
    """Construct the reference for a problem, as the registry selects it.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem to step.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    grid : solprop.grid.TimeGrid or None, optional
        The grid the state lives on.
    modes : collections.abc.Sequence or None, optional
        The propagated modes.
    profiles : collections.abc.Mapping or None, optional
        The z-profiles keyed by statistic name.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned host response binding for a TDSE problem.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options.

    Returns
    -------
    NumpyRefBackend
        The reference.
    """
    return NumpyRefBackend(
        problem,
        request,
        grid=grid,
        modes=modes,
        profiles=profiles,
        tdse_binding=tdse_binding,
        options=options,
    )

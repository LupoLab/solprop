# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's scan of the field synthesised on the axis (SEP 19)."""

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.envelope_plasma import envelope_populations
from solprop.backends.populations import axis_row_matrix
from solprop.translate.problem import PropagationProblem

__all__ = ["axis_populations"]


def axis_populations(
    problem: PropagationProblem,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    z: float,
    axis: Float[np.ndarray, "2 n_modes"],
) -> tuple[Float[np.ndarray, " n_channels"], Float[np.ndarray, " n_channels"]]:
    """Scan the field synthesised on the axis: one row, every channel (SEP 19).

    The reference's, on the host: the axis is one row per saved plane, against every
    node of every stage; the native session scans the same row natively. The field
    is the same as the node fields' -- the same pad scale, taper and fine grid -- with
    both Cartesian components of the mode profiles, as for ``peak_intensity``; when the
    state carries the components itself, the profile's magnitude on the axis is the
    scalar synthesis. The tables, preionisation and running-integral rule are the node
    scan's.

    Parameters
    ----------
    problem : PropagationProblem
        The translated problem.
    field : numpy.ndarray
        One batch member's state, shape (n_modes, n_pol, n_omega).
    z : float
        Plane position in m.
    axis : numpy.ndarray
        The mode set's synthesis on the axis at `z`, taper included, rows x and y
        (``HostDriver._on_axis_synthesis``).

    Returns
    -------
    tuple of numpy.ndarray
        The final ionised fraction and the largest rate (cycle-averaged on an
        envelope grid) on the axis, one per channel.
    """
    grids = problem.grids
    modal_t = passes.to_time(
        field, grids.crop_index, grids.n_to, kind=problem.grid_kind
    )
    row = passes.synthesise(
        modal_t, axis_row_matrix(problem, axis), 1.0, 2 if field.shape[1] == 1 else 1
    )
    return _scan_axis_row(problem, row, z)


def _scan_axis_row(
    problem: PropagationProblem,
    row: Inexact[np.ndarray, "1 n_components n_to"],
    z: float,
) -> tuple[Float[np.ndarray, " n_channels"], Float[np.ndarray, " n_channels"]]:
    """Scan one row of the field in time: every channel, with the node scan's tables.

    - ``row``: the field on the axis on the fine time grid in V/m.
    - ``z``: plane position in m.
    - Returns: the final ionised fraction and the largest rate on the axis, one per
      channel.
    """
    responses, grids = problem.responses, problem.grids
    preionisation = tuple(
        float(channel.preionisation(z)) for channel in responses.population_channels
    )
    if responses.envelope_plasma is not None:
        fractions, peaks = envelope_populations(
            row, responses.envelope_plasma, preionisation, grids.dt_fine
        )
        return fractions[:, 0], peaks[:, 0]
    rates, population = passes.plasma_channel_fractions(
        row,
        responses.plasma_channels,
        preionisation,
        grids.dt_fine,
        responses.plasma_integral,
    )
    return population[:, 0, -1], rates[:, 0].max(axis=-1)

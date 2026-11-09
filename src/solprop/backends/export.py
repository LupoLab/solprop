# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Destination-based export of a session's field (SEP 47 section 5.4).

The output layer owns the definition of what a saved field means: true-power modal
amplitudes, or a spatial route's field measure, in the optics convention on disk
(ADR-017; ``docs/dev/conventions.md``). It states that definition once, as an
:class:`ExportRecipe`, and a session applies it while filling a destination: the
reference session with :func:`apply_recipe`, the native session in its own fused pass.
The definition does not move; only where it is evaluated does.

Every conversion step is exact or is the one multiplication today's two-step path
makes: a real square-root weight times the amplitude, then a conjugation and a
multiplication by ``(-1)**k``. Applied in that order, the fused pass is bitwise equal
to :func:`solprop.fourier.power_amplitude` followed by
:func:`solprop.fourier.spectrum_to_optics`, so lossless storage stays exact.

A destination is either an array the caller supplies -- a preallocated store row --
or a newly owned array the session allocates. Either way the result is detached: it is
never a view of a live workspace, and the session never writes to it again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float

from solprop.exceptions import BackendError, GridError

__all__ = ["ExportRecipe", "apply_recipe"]


@dataclass(frozen=True)
class ExportRecipe:
    """What conversion a field export applies on its way to a destination.

    Attributes
    ----------
    power : bool
        Multiply modal amplitudes by the square root of the route's true-power weight at
        the exported position (``fourier.power_amplitude``). False on spatial routes,
        whose field measure is already the stored quantity.
    optics : bool
        Convert from the internal FFT convention to the optics convention about the
        window centre, ``(-1)**k conj(.)`` (``fourier.spectrum_to_optics``). True for a
        plane store; False for a consumer that works in the internal convention, such as
        a plane observer or a terminal checkpoint record.

    Examples
    --------
    >>> from solprop.backends.export import ExportRecipe
    >>> ExportRecipe(power=True, optics=False)
    ExportRecipe(power=True, optics=False)
    """

    power: bool
    optics: bool


def apply_recipe(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    recipe: ExportRecipe,
    *,
    weight: Float[np.ndarray, "n_modes n_pol n_omega"] | None,
    out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Fill a destination with a field converted by a recipe, in one pass.

    The reference implementation of SEP 47's export. `state` is only read; the result
    is `out` when given, and otherwise a new array, so it never aliases `state`.

    Parameters
    ----------
    state : numpy.ndarray
        The committed field, complex128, internal FFT convention, internal amplitude
        units, shape ``(n_modes, n_pol, n_omega)``.
    recipe : ExportRecipe
        The conversion.
    weight : numpy.ndarray or None
        The true-power weight at the field's position, positive and finite, the shape
        of `state`; required when ``recipe.power`` and ignored otherwise.
    out : numpy.ndarray or None, optional
        The destination, complex128, C-contiguous, the shape of `state`, not
        overlapping it; a new array by default. `state` itself converts in place, for a
        session that owns the only reference to it; each step is elementwise, so the
        result is the same bits.

    Returns
    -------
    numpy.ndarray
        The converted field: `out` when given.

    Raises
    ------
    BackendError
        If `out` has the wrong shape, dtype or layout, or overlaps `state`.
    GridError
        If a weight is required and missing, or not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.export import ExportRecipe, apply_recipe
    >>> state = np.array([[[1 + 1j, 2 - 1j]]])
    >>> apply_recipe(state, ExportRecipe(power=True, optics=True),
    ...              weight=np.full((1, 1, 2), 4.0)).tolist()
    [[[(2-2j), (-4-2j)]]]
    """
    if out is None:
        out = np.empty_like(state, dtype=np.complex128, order="C")
    elif out is not state:
        _check_destination(state, out)
    if recipe.power:
        if weight is None:
            msg = "a true-power export needs the route's weight at its position"
            raise GridError(msg)
        if not np.all(np.isfinite(weight) & (weight > 0)):
            msg = "power weights must be positive and finite"
            raise GridError(msg)
        if not np.all(weight == 1.0):
            np.multiply(state, np.sqrt(weight), out=out)
        elif out is not state:
            np.copyto(out, state)
    elif out is not state:
        np.copyto(out, state)
    if recipe.optics:
        shift = np.ones(state.shape[-1])
        shift[1::2] = -1.0
        np.conj(out, out=out)
        np.multiply(shift, out, out=out)
    return out


def _check_destination(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    out: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> None:
    """Refuse a destination that cannot hold an export of `state`.

    Parameters
    ----------
    state : numpy.ndarray
        The field being exported.
    out : numpy.ndarray
        The proposed destination.

    Raises
    ------
    BackendError
        Naming the mismatch.
    """
    if out.shape != state.shape:
        msg = f"an export destination has shape {out.shape}, expected {state.shape}"
        raise BackendError(msg)
    if (
        out.dtype != np.complex128
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        msg = "an export destination must be writeable, C-contiguous complex128"
        raise BackendError(msg)
    if np.shares_memory(out, state):
        msg = "an export destination must not overlap the session's state"
        raise BackendError(msg)

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
never a view of a live workspace, and the session never writes to it again. A plane
store asks for its own room
(:meth:`solprop.output.schema.PlaneStore.field_destination`), so that a kept field is
written once, where it is kept.

The destination is always host memory. A session whose state lives on a device fills it
the same way: the recipe runs where the state is, into the device's own scratch, and one
copy crosses into the host destination; whether the conversion is better made before or
after that crossing is the device's choice to measure (SEP 47 section 5.4). No device
array crosses this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from solprop._process import parallel_rows
from solprop.exceptions import BackendError, GridError

if TYPE_CHECKING:
    from jaxtyping import Complex, Float

__all__ = ["ExportRecipe", "apply_recipe", "destination_refusal"]


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
        overlapping it; a new array by default. `state` itself converts in place,
        where the caller holds the only reference to it; each step is elementwise, so
        the result is the same bits.

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
    else:
        _check_destination(state, out)
    scale = None
    if recipe.power:
        if weight is None:
            msg = "a true-power export needs the route's weight at its position"
            raise GridError(msg)
        if not np.all(np.isfinite(weight) & (weight > 0)):
            msg = "power weights must be positive and finite"
            raise GridError(msg)
        if not np.all(weight == 1.0):
            scale = np.broadcast_to(weight, state.shape)
    if scale is None and not recipe.optics:
        if out is not state:
            np.copyto(out, state)
        return out
    n_omega = state.shape[-1]
    rows = state.reshape(-1, n_omega)
    parallel_rows(
        rows.shape[0],
        state.nbytes,
        partial(
            _convert_rows,
            rows,
            out.reshape(-1, n_omega),
            None if scale is None else scale.reshape(-1, n_omega),
            optics=recipe.optics,
        ),
    )
    return out


def _convert_rows(
    state: Complex[np.ndarray, "n_rows n_omega"],
    out: Complex[np.ndarray, "n_rows n_omega"],
    weight: Float[np.ndarray, "n_rows n_omega"] | None,
    selected: slice,
    *,
    optics: bool,
) -> None:
    """Convert some rows of a field into their rows of the destination.

    Each step is the one today's two-step path makes, so the bits are its own: the
    product with the weight's square root (``fourier.power_amplitude``), then the
    conjugate and the product with ``(-1)**k`` (``fourier.spectrum_to_optics``). Rows
    are independent, so slices may run in threads.

    - ``weight``: the weight's rows, or ``None`` for no amplitude conversion.
    - ``optics``: whether to convert to the optics convention.
    """
    source = state[selected]
    target = out[selected]
    if weight is not None:
        np.multiply(source, np.sqrt(weight[selected]), out=target)
        source = target
    if optics:
        shift = np.ones(state.shape[-1])
        shift[1::2] = -1.0
        # The conjugate read straight from the source: a copy first would be a pass.
        np.conj(source, out=target)
        np.multiply(shift, target, out=target)
    else:
        assert source is target  # an identity recipe returns before any rows


def _check_destination(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    out: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> None:
    """Refuse a destination that cannot hold an export of `state`."""
    refusal = destination_refusal(out, state.shape)
    if refusal is not None:
        raise BackendError(refusal)
    if out is not state and np.shares_memory(out, state):
        msg = "an export destination must not overlap the session's state"
        raise BackendError(msg)


def destination_refusal(out: np.ndarray, shape: tuple[int, ...]) -> str | None:
    """Return why an array cannot be an export destination, or ``None`` if it can.

    Every session validates a destination by this one rule before it writes to it
    (SEP 47 section 5.4), and names what is wrong: its shape, its dtype, its layout or
    that it is read-only. One whose elements overlap one another is not contiguous.

    Parameters
    ----------
    out : numpy.ndarray
        The proposed destination.
    shape : tuple of int
        The field's shape, ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    str or None
        The refusal, or ``None`` for a writeable, C-contiguous complex128 array of
        `shape`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.export import destination_refusal
    >>> destination_refusal(np.empty((1, 1, 4), complex), (1, 1, 4)) is None
    True
    >>> destination_refusal(np.empty((1, 1, 4), np.complex64), (1, 1, 4))
    'an export destination has dtype complex64, expected complex128'
    """
    if out.shape != shape:
        return f"an export destination has shape {out.shape}, expected {shape}"
    if out.dtype != np.complex128:
        return f"an export destination has dtype {out.dtype}, expected complex128"
    if not out.flags.c_contiguous:
        return "an export destination must be C-contiguous"
    if not out.flags.writeable:
        return "an export destination must be writeable; this one is read-only"
    return None

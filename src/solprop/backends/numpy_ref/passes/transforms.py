# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transform passes of a stage: between the modal spectrum and the node fields.

Passes 2, 3, 5 and 6 of ``docs/dev/architecture.md`` section 8, each one batched
operation over every modal column: :func:`to_time` pads the coarse spectrum onto the
fine grid and transforms it to time, :func:`synthesise` forms the field at a chunk of
quadrature nodes (one GEMM over the mode axis), :func:`project` takes a chunk's
polarisation back onto the modes (one GEMM, weights folded in), and
:func:`to_frequency` transforms the modal polarisation and crops it to the coarse grid.
The pad and crop scales are already inside the contract's matrices and prefactor, so
none of these applies a scale of its own (:mod:`solprop.backends.numpy_ref.passes`).
"""

import numpy as np
from jaxtyping import Complex, Float, Inexact, Int, Num

from solprop import fourier
from solprop.fourier import GridKind

__all__ = [
    "project",
    "synthesise",
    "to_frequency",
    "to_time",
]


def to_time(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
    crop_index: Int[np.ndarray, " n_crop"],
    n_to: int,
    *,
    kind: GridKind,
) -> Inexact[np.ndarray, "n_modes n_pol n_to"]:
    r"""Pad onto the fine grid and transform every modal column to time.

    Pass 2 of ``architecture.md`` section 8: one batched inverse transform of all
    ``n_modes * n_pol`` columns, whatever the quadrature does next. The coarse bins are
    scattered into the fine spectrum at ``crop_index`` with **no scale factor**: the pad
    scale ``n_to/n_t`` is inside ``transform.synthesis`` (``kernel-contract.md`` section
    4), so that every response sees the field in volts per metre whatever its degree in
    the field.

    The mode and polarisation axes are folded together for the transform, so the columns
    a scalar problem transforms are exactly the ones it transformed before the
    polarisation axis existed.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Coarse modal spectrum, complex128, shape ``(n_modes, n_pol, n_omega)``, in the
        square root of watts.
    crop_index : numpy.ndarray
        Where each coarse bin sits in the fine spectrum, ``numpy.intp``, shape
        ``(n_omega,)``.
    n_to : int
        Number of fine time samples, positive.
    kind : {'field', 'envelope'}
        Spectral layout, which selects the complex-to-real or complex-to-complex
        transform.

    Returns
    -------
    numpy.ndarray
        The modal field on the fine time grid, shape ``(n_modes, n_pol, n_to)``, in the
        square root of watts: real on a field grid, complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.transforms import to_time
    >>> spectrum = np.array([[[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]])
    >>> index = np.arange(3, dtype=np.intp)
    >>> to_time(spectrum, index, 8, kind="field")
    array([[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]])
    """
    n_modes, n_pol = spectrum.shape[:2]
    n_omega_fine = fourier.n_omega_for(n_to, kind)
    fine = np.zeros((n_modes * n_pol, n_omega_fine), dtype=np.complex128)
    fine[..., crop_index] = spectrum.reshape(n_modes * n_pol, -1)
    return fourier.inverse(fine, kind=kind).reshape(n_modes, n_pol, n_to)


def synthesise(
    modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
    synthesis: Float[np.ndarray, "n_rows n_modes_s"],
    taper: float,
    n_pol: int,
) -> Inexact[np.ndarray, "n_nodes_chunk n_pol_node n_to"]:
    r"""Synthesise the physical field on a chunk of quadrature nodes.

    Pass 3 of ``architecture.md`` section 8:
    ``E[i, p, t] = tau(z) sum_m S[i n_pol + p, m] a[m, t]``, one GEMM over the mode
    axis. `synthesis` is the chunk's rows of ``transform.synthesis``, which already
    carry the oversampling pad scale, and `taper` is ``transform.taper_scale(z)``, the
    exact rescaling a scale-invariant waveguide needs
    (:meth:`solprop.modes.quadrature.ModeMatrices.rescaled`).

    The transverse components are carried by the rows of `synthesis` **or** by the
    state's polarisation axis and never by both (``kernel-contract.md`` section 5), so
    the returned polarisation axis has ``n_pol`` times the state's length and one of the
    two factors is one.

    Parameters
    ----------
    modal_t : numpy.ndarray
        The modal field on the fine time grid from :func:`to_time`, shape
        ``(n_modes, n_pol, n_to)``, in the square root of watts.
    synthesis : numpy.ndarray
        The chunk's rows of the synthesis matrix, float64, shape
        ``(n_nodes_chunk * n_pol, n_modes)``, in V/(m sqrt(W)), already carrying the pad
        scale.
    taper : float
        The rescaling at this position, ``transform.taper_scale(z)``, or 1.0 for a
        waveguide whose cross-section does not move.
    n_pol : int
        Transverse components the transform resolves at each node, 1 or 2.

    Returns
    -------
    numpy.ndarray
        The field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol_node, n_to)``, in
        V/m: real on a field grid, complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.transforms import synthesise
    >>> modal = np.ones((1, 1, 4))
    >>> synthesise(modal, np.array([[2.0], [3.0]]), 1.0, 1).shape
    (2, 1, 4)
    """
    scaled = synthesis * taper
    nodes = np.tensordot(scaled, modal_t, axes=([1], [0]))
    n_rows, n_pol_state, n_to = nodes.shape
    return nodes.reshape(n_rows // n_pol, n_pol * n_pol_state, n_to)


def project(
    polarisation_t: Num[np.ndarray, "n_nodes_chunk n_pol_node n_to"],
    projection: Float[np.ndarray, "n_modes n_columns"],
    taper: float,
    n_pol: int,
) -> Num[np.ndarray, "n_modes n_pol n_to"]:
    r"""Project a chunk's polarisation back onto the modes.

    Pass 5 of ``architecture.md`` section 8:
    ``P[m, t] = sum_{i, p} W[m, i n_pol + p] P[i, p, t] / tau(z)``, one GEMM with the
    quadrature weights already folded into `projection`. The caller accumulates over
    chunks, which is what makes the space-domain buffers bounded by the chunk rather
    than by the node count (``architecture.md`` section 13).

    Parameters
    ----------
    polarisation_t : numpy.ndarray
        The polarisation at the chunk's nodes, shape
        ``(n_nodes_chunk, n_pol_node, n_to)``, in C/m^2.
    projection : numpy.ndarray
        The chunk's columns of the projection matrix, float64, shape
        ``(n_modes, n_nodes_chunk * n_pol)``, weights folded in.
    taper : float
        The rescaling at this position, ``transform.taper_scale(z)``, or 1.0. The
        projection is divided by it, as the synthesis is multiplied by it.
    n_pol : int
        Transverse components the transform resolves at each node, 1 or 2.

    Returns
    -------
    numpy.ndarray
        The modal polarisation on the fine time grid, shape
        ``(n_modes, n_pol, n_to)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.transforms import project
    >>> nodes = np.ones((2, 1, 4))
    >>> project(nodes, np.array([[1.0, 3.0]]), 1.0, 1).shape
    (1, 1, 4)
    """
    n_chunk, n_pol_node, n_to = polarisation_t.shape
    rows = polarisation_t.reshape(n_chunk * n_pol, n_pol_node // n_pol, n_to)
    scaled = projection / taper
    return np.tensordot(scaled, rows, axes=([1], [0]))


def to_frequency(
    polarisation_t: Num[np.ndarray, "n_modes n_pol n_to"],
    crop_index: Int[np.ndarray, " n_crop"],
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Transform the modal polarisation to frequency and crop to the coarse grid.

    Pass 6 of ``architecture.md`` section 8: one batched forward transform of all
    ``n_modes * n_pol`` columns. The crop takes the coarse bins out of the fine spectrum
    at ``crop_index`` with **no scale factor**: the crop scale ``n_t/n_to`` is inside
    ``responses.prefactor`` (``kernel-contract.md`` section 5), which
    :func:`~solprop.backends.numpy_ref.passes.stepping.stage_output` applies next.

    Parameters
    ----------
    polarisation_t : numpy.ndarray
        The modal polarisation on the fine time grid, accumulated over the chunks by
        :func:`project`, shape ``(n_modes, n_pol, n_to)``, in C/m^2.
    crop_index : numpy.ndarray
        Where each coarse bin sits in the fine spectrum, ``numpy.intp``, shape
        ``(n_omega,)``.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        The modal polarisation on the coarse grid, complex128, shape
        ``(n_modes, n_pol, n_omega)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.transforms import to_frequency
    >>> samples = np.ones((1, 1, 8))
    >>> index = np.arange(3, dtype=np.intp)
    >>> to_frequency(samples, index, kind="field").real
    array([[[8., 0., 0.]]])
    """
    n_modes, n_pol = polarisation_t.shape[:2]
    flat = polarisation_t.reshape(n_modes * n_pol, -1)
    spectrum = fourier.forward(flat, kind=kind)[..., crop_index]
    return spectrum.reshape(n_modes, n_pol, -1)

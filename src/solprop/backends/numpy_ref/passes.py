# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The passes of one interaction-picture stage, one function each.

``docs/dev/architecture.md`` section 8 lists the anatomy of a stage as seven numbered
passes over the data, and ``docs/dev/backends.md`` section 2 says the reference backend
writes each of them as a named function with its formula in the docstring. That is this
module. Nothing here is fused, nothing is written in place and nothing allocates a
buffer it keeps: every function takes arrays and returns a new one, so a reader can
follow one pass without holding the rest of the step in mind. The optimised backends
fuse these passes and are validated against the numbers this module produces.

The sequence is: pad, inverse transform, conversion to volts per metre, the responses,
the temporal window, the forward transform, the crop, the prefactor and the spectral
window.

**Scales.** The oversampling **pad** scale is already inside ``transform.synthesis`` and
the **crop** scale is already inside ``responses.prefactor``
(``docs/dev/kernel-contract.md`` sections 4 and 5), so :func:`to_time` and
:func:`to_frequency` scatter and gather through ``grids.crop_index`` and apply no scale
of their own. Using :func:`solprop.fourier.oversample` and :func:`solprop.fourier.crop`
here instead would count both factors twice.

**Convention.** Every complex array is in the FFT convention of
``docs/dev/conventions.md`` section 2; nothing here conjugates. The Raman
kernel spectrum is an internal-convention array whose imaginary part is what
distinguishes gain from loss, so it is never passed through
:func:`solprop.fourier.to_optics` (``docs/guide/media/raman-numerics.md``, "The sign
that means gain").

**The one physical constant.** :data:`solprop.constants.EPS0` is read here, by the Raman
pass alone. Every other constant reaches a backend through a compiled coefficient --
``eps0 chi3`` for the Kerr term, ``e^2/m_e`` for the plasma -- but the Raman response
writes ``eps0`` *outside* the convolution integral, so folding it into a row would
change the stated units of ``raman_kernel_undamped`` and cost the bitwise agreement with
:meth:`solprop.nonlinear.raman.RamanResponse.evaluate` that pins this pass.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int, Num

from solprop import boundaries, fourier
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.raman import raman_drive
from solprop.nonlinear.responses import dispersive_cubic
from solprop.stepper.controller import PIController
from solprop.translate.ionisation import RateTable
from solprop.translate.raman import kernel_spectrum
from solprop.translate.responses import (
    CompiledResponses,
    DispersiveKerrTerms,
    KerrCoefficients,
    PlasmaChannel,
    PlasmaCoefficients,
    RamanCoefficients,
    ResponseStage,
)
from solprop.translate.stage import StageScalars

__all__ = [
    "NonlinearSources",
    "absorber_term",
    "accept_step",
    "accumulate",
    "apply_hard_mask",
    "apply_temporal_rate",
    "dispersive_kerr_polarisation",
    "error_norm",
    "held_plasma_spectrum",
    "ionisation_rate",
    "kerr_polarisation",
    "mixture_raman_spectrum",
    "nonlinear_polarisation",
    "nonlinear_sources",
    "plasma_channel_fractions",
    "plasma_channels_polarisation",
    "plasma_current",
    "plasma_integration_diagnostics",
    "plasma_integration_statistics",
    "plasma_scan",
    "project",
    "raman_polarisation",
    "raman_spectrum",
    "stage_input",
    "stage_output",
    "synthesise",
    "tensor_raman_spectrum",
    "to_frequency",
    "to_time",
]


def stage_input(
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    stage_sum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    dz: float,
    propagator: Complex[np.ndarray, "n_batch_p n_modes_p n_pol_p n_omega_p"],
    noise: Complex[np.ndarray, "n_batch_n n_modes_n n_pol_n n_omega_n"] | None,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Combine the stages, return to the lab frame and add the noise.

    ``u = exp(dPhi_i) (y + h sum_j a_ij k_j) + N(z_n + c_i h)``, the first pass of
    ``architecture.md`` section 8. The interaction picture is anchored at the start of
    the step, so ``exp(dPhi_i)`` carries the stage input forward to where the nonlinear
    term is evaluated, and the shot-noise field is added there, in the frequency domain,
    where it costs one multiply-add and no transform.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the anchor, complex128, shape
        ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts.
    stage_sum : numpy.ndarray
        ``sum_j a_ij k_j`` in the interaction variable, same shape and dtype, in the
        square root of watts per metre.
    dz : float
        Length of the step, m, positive.
    propagator : numpy.ndarray
        ``exp(dPhi_i)``, dimensionless, broadcastable against the state; the batch axis
        is 1 when the members share a fibre.
    noise : numpy.ndarray or None
        The shot-noise field already advanced to this position,
        ``N0 exp(i Im Phi(omega, z))``, in the square root of watts, or ``None`` when no
        noise model is active.

    Returns
    -------
    numpy.ndarray
        The lab-frame field the nonlinear term is evaluated at, same shape and dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import stage_input
    >>> state = np.ones((1, 1, 1, 2), dtype=complex)
    >>> zero = np.zeros_like(state)
    >>> stage_input(state, zero, 0.1, 2.0 * np.ones_like(state), None).ravel()
    array([2.+0.j, 2.+0.j])
    """
    combined = propagator * (state + dz * stage_sum)
    return combined if noise is None else combined + noise


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
    >>> from solprop.backends.numpy_ref.passes import to_time
    >>> spectrum = np.array([[[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]])
    >>> index = np.arange(3, dtype=np.intp)
    >>> to_time(spectrum, index, 8, kind="field")
    array([[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]])
    """
    n_modes, n_pol = spectrum.shape[:2]
    n_omega_fine = fourier.n_omega_for(n_to, kind)
    # The spectrum's own precision: complex128, or complex64 on SEP 33's
    # single-precision path.
    fine = np.zeros(
        (n_modes * n_pol, n_omega_fine), dtype=np.result_type(spectrum, np.complex64)
    )
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
    >>> from solprop.backends.numpy_ref.passes import synthesise
    >>> modal = np.ones((1, 1, 4))
    >>> synthesise(modal, np.array([[2.0], [3.0]]), 1.0, 1).shape
    (2, 1, 4)
    """
    scaled = synthesis * taper
    nodes = np.tensordot(scaled, modal_t, axes=([1], [0]))
    n_rows, n_pol_state, n_to = nodes.shape
    return nodes.reshape(n_rows // n_pol, n_pol * n_pol_state, n_to)


def kerr_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    kerr: KerrCoefficients,
    density: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate the instantaneous Kerr polarisation at every node of a chunk.

    The ``kerr`` stage of the fused pass. The formulas are the two cubic forms of
    :class:`solprop.translate.responses.KerrCoefficients`, which were read off
    :meth:`solprop.nonlinear.responses.KerrResponse.evaluate` rather than re-derived:

    ``real``
        ``P_p = c rho (sum_q E_q^2) E_p`` -- the full cubic of a real field, which
        already carries the third harmonic and the counter-rotating terms.
    ``analytic``
        ``P_p = c rho [s (sum_q |a_q|^2) a_p + k (sum_q a_q^2) conj(a_p)]``, plus
        ``c rho m exp(2 i omega_0 t) (sum_q a_q^2) a_p`` when the envelope grid radiates
        the harmonic, with ``s``, ``k`` and ``m`` the self, conjugate and harmonic
        factors. For one component the first collapses to ``(3/4)|a|^2 a``.

    The sums run over the **polarisation** axis of one node, which is what the
    vector-isotropic coupling means: two components of the field at one point of the
    cross-section, never two different points.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid.
    kerr : solprop.translate.responses.KerrCoefficients
        The compiled coefficients, whose ``coefficient`` is ``eps0 chi3`` per unit of
        the density path.
    density : float
        The density path at this position, ``responses.density(z)``: a number density in
        m^-3 for a gas, 1.0 for a medium whose ``chi3`` is already absolute.
        Non-negative.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, shape ``(n_to,)``. Required when
        ``kerr.harmonic_factor`` is not zero; the harmonic term is added whenever a
        carrier is given, and ``None`` selects the fundamental cubic alone.

    Returns
    -------
    numpy.ndarray
        The polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import kerr_polarisation
    >>> from solprop.translate.responses import KerrCoefficients
    >>> coefficients = KerrCoefficients(1.0, "analytic", 0.5, 0.25, 0.0, False, False)
    >>> field = np.array([[[2.0 + 0.0j]]])
    >>> polarisation = kerr_polarisation(field, coefficients, 1.0)
    >>> bool(np.allclose(polarisation, 0.75 * abs(2.0) ** 2 * 2.0))
    True
    """
    values = fourier.analytic_signal(field_t) if kerr.analytic_signal else field_t
    factor = kerr.coefficient * density
    if kerr.cubic_form == "real":
        polarisation = factor * np.sum(values**2, axis=-2, keepdims=True) * values
    else:
        squares = np.sum(values**2, axis=-2, keepdims=True)
        moduli = np.sum(np.abs(values) ** 2, axis=-2, keepdims=True)
        polarisation = factor * (
            kerr.self_factor * moduli * values
            + kerr.conjugate_factor * squares * np.conj(values)
        )
        if carrier is not None:
            polarisation = polarisation + (
                factor * kerr.harmonic_factor * carrier * squares * values
            )
    return polarisation.real if kerr.take_real else polarisation


def dispersive_kerr_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    terms: DispersiveKerrTerms,
    densities: Sequence[float],
    *,
    kind: GridKind,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    window: Float[np.ndarray, " n_to_window"] | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate SEP 6's dispersive electronic polarisation at every node of a chunk.

    The ``dispersive_kerr`` stage of the fused pass: per channel, the transforms of
    the field the evaluator's forms need, the products of the filtered fields in the
    leaf's sector, the temporal collar on each product, the output multipliers and one
    inverse transform, by :func:`solprop.nonlinear.responses.dispersive_cubic`, which
    is the arithmetic of the reference
    :meth:`~solprop.nonlinear.responses.DispersiveKerrResponse.evaluate` on the leaf's
    sampled multipliers. The polynomial's channels add, ``P = sum_s c_s rho_s
    cubic_s(E)``; the symmetric evaluator's mixture takes one filter built from its
    species' densities at this stage, ``P = F G_mix[w (G_mix E)^3]``
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.evaluations`). It reads the
    bare field and touches no other response's source.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid.
    terms : solprop.translate.responses.DispersiveKerrTerms
        The compiled leaf.
    densities : sequence of float
        Each channel's number density at this position, m^-3, in the leaf's order.
    kind : {'field', 'envelope'}
        The spectral layout.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, shape ``(n_to,)``; required when
        the leaf's ``carrier`` is set.
    window : numpy.ndarray or None, optional
        The temporal collar, shape ``(n_to,)``, applied to each product before the
        output multipliers, ``G[w (G E)^3]``; ``None`` is a unit collar.

    Returns
    -------
    numpy.ndarray
        The polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid, the collar already inside it.

    Raises
    ------
    BackendError
        If the densities do not match the channels, or the carrier is missing.
    """
    if len(densities) != terms.n_channels:
        msg = (
            f"the dispersive Kerr leaf has {terms.n_channels} channels and the stage "
            f"carries {len(densities)} densities"
        )
        raise BackendError(msg)
    if terms.carrier and carrier is None:
        raise BackendError(
            "the dispersive Kerr leaf keeps the full cubic on an envelope grid and no "
            "harmonic carrier was given"
        )
    total = np.zeros_like(field_t)
    for _, factor, weights in terms.evaluations(densities):
        total = total + factor * dispersive_cubic(
            field_t,
            weights,
            kind=kind,
            sector=terms.sector,
            carrier=carrier if terms.carrier else None,
            window=window,
        )
    return total


def raman_spectrum(
    undamped: Float[np.ndarray, "n_terms n_to_doubled"],
    rates: Sequence[float] | np.ndarray,
    fraction: float,
    dt: float,
) -> Complex[np.ndarray, " n_omega_doubled"]:
    r"""Damp each undamped Raman row at its own rate, sum, scale and transform.

    The half of the kernel that moves along z. ``raman_kernel_undamped`` holds one
    causal, tapered row per independently damped term and ``raman_decay_rates`` one
    scalar path per row (``kernel-contract.md`` section 5); this forms
    ``f_R sum_k u_k(t) exp(-gamma_k t)`` on the doubled grid and hands it to
    :func:`solprop.translate.raman.kernel_spectrum`, which folds in the sample spacing
    the discrete sum needs to become an integral. There is one statement of that
    normalisation and this is not it.

    Damping each term separately is the whole reason the contract carries rows rather
    than one array: a molecular gas's rotational and vibrational coherences decay 3.5
    times (H2) to 17 times (N2) apart at one bar, so damping their sum once would be
    wrong for every such gas (``docs/guide/media/raman-numerics.md``).

    The terms are summed in the order of the rows and the fraction multiplies the sum,
    which is the arithmetic of :func:`solprop.translate.raman.sample_kernel` operation
    for operation, so the two spectra agree to the last bit. The padding half of each
    row is zero, so damping it costs nothing and changes nothing.

    Parameters
    ----------
    undamped : numpy.ndarray
        ``responses.raman_kernel_undamped``, float64, shape ``(n_terms, 2 n_to)``, in
        m^5 V^-2 s^-1.
    rates : array_like
        The decay rate of each term at this position, s^-1, non-negative, one per row:
        ``[table(z) for table in responses.raman_decay_rates]``.
    fraction : float
        ``responses.raman_fraction``, in ``(0, 1]``; 1 for an absolute molecular
        response.
    dt : float
        Sample spacing of the fine time grid, s, positive.

    Returns
    -------
    numpy.ndarray
        The damped kernel spectrum with the ``dt`` scaling folded in, complex128, shape
        ``(n_to + 1,)``, in the **internal** convention.

    Raises
    ------
    BackendError
        If there is not one rate per row.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import raman_spectrum
    >>> rows = np.zeros((1, 8))
    >>> rows[0, 1] = 1.0
    >>> spectrum = raman_spectrum(rows, [0.0], 1.0, 2.0)
    >>> spectrum.shape, bool(np.isclose(spectrum[0], 2.0))
    ((5,), True)
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size != undamped.shape[0]:
        msg = (
            f"the Raman kernel has {undamped.shape[0]} row(s) and {values.size} decay "
            "rate(s) were given; a backend damps row k at rate k"
        )
        raise BackendError(msg)
    delays = np.arange(undamped.shape[1]) * dt
    total = np.zeros(undamped.shape[1])
    for row, rate in zip(undamped, values, strict=True):
        total += row * np.exp(-rate * delays)
    return kernel_spectrum(fraction * total, dt)


def raman_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    spectrum: Complex[np.ndarray, "... n_omega_doubled"],
    *,
    kind: GridKind,
    thg: bool,
    density: float,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate the delayed Raman polarisation at every node of a chunk.

    The scalar reduction preserves the formula of
    :meth:`solprop.nonlinear.raman.RamanResponse.evaluate` in its original order of
    operations:

    .. math::

        P(t) = \varepsilon_0\,\rho\,E(t) \int h(t - t')\,D(t')\,\mathrm{d}t' ,

    with ``D`` the real drive :func:`solprop.nonlinear.raman.raman_drive` chooses for
    the grid -- ``E^2`` on a field grid that keeps the harmonic, ``|E_a|^2/2`` on one
    that does not, ``|A|^2/2`` on an envelope grid.

    The convolution runs on a **doubled, zero-padded** grid, because a transform product
    is a circular convolution and a rotational coherence outlives the window by orders
    of magnitude: hydrogen's lives about 2.9 ns at one bar against a window of tens of
    picoseconds, so the wrapped tail would be almost the whole response. The drive and
    the kernel occupy the first half of an array of twice the length, which makes the
    product the linear convolution exactly (``docs/guide/media/raman-numerics.md``, "The
    doubled grid").

    For two Cartesian components, convolve the trace, difference and twice the
    off-diagonal entry of the real quadratic tensor, then apply the resulting
    symmetric matrix to the field. Isotropic-only declarations need just the trace
    history. Each history is real on both grids and uses one real transform pair.
    ``numpy.fft`` preserves the scalar oracle's transform arithmetic.

    Parameters
    ----------
    field_t : numpy.ndarray
        Field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid. One or two physical
        Cartesian components.
    spectrum : numpy.ndarray
        Damped complex128 kernels with ``dt`` included. Scalar fields use shape
        ``(n_to + 1,)`` from :func:`raman_spectrum`; vector fields use shape
        ``(n_raman_kernels, n_to + 1)`` from :func:`tensor_raman_spectrum`, with one
        trace kernel or the trace/traceless pair.
    kind : {'field', 'envelope'}
        The spectral layout the samples belong to.
    thg : bool
        Whether a field grid's drive keeps the ``2 omega`` beat, and so radiates a third
        harmonic through the later multiplication by the field. Ignored on an envelope
        grid, which cannot carry it in a quadratic drive.
    density : float
        The density path at this position, ``responses.density(z)``, in m^-3 for a gas
        and 1.0 for a medium whose kernel is already absolute. Non-negative.

    Returns
    -------
    numpy.ndarray
        Raman polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid.

    Raises
    ------
    BackendError
        If the component count or effective kernel shape is incompatible with the
        doubled temporal grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import raman_polarisation
    >>> field = np.ones((1, 1, 4))
    >>> flat = np.zeros(5, dtype=complex)
    >>> polarisation = raman_polarisation(
    ...     field, flat, kind="field", thg=True, density=1.0
    ... )
    >>> bool(np.all(polarisation == 0.0))
    True
    """
    if field_t.shape[-2] == 2:
        return _vector_raman_polarisation(
            field_t, spectrum, kind=kind, thg=thg, density=density
        )
    if field_t.shape[-2] != 1:
        raise BackendError("Raman needs one or two physical components")
    n_to = field_t.shape[-1]
    if spectrum.shape != (n_to + 1,):
        msg = (
            f"the Raman kernel spectrum has shape {spectrum.shape}, expected "
            f"{(n_to + 1,)}: the one-sided spectrum of the doubled grid"
        )
        raise BackendError(msg)
    drive = np.zeros((field_t.shape[0], 2 * n_to))
    # One column at a time, as the oracle takes them: the drive of a field grid without
    # the harmonic is an analytic signal, and forming it per column is what makes this
    # pass reproduce the oracle rather than merely agree with it.
    for node in range(field_t.shape[0]):
        drive[node, :n_to] = raman_drive(field_t[node], kind=kind, thg=thg)
    convolved = np.fft.irfft(
        spectrum * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    return (EPS0 * density) * field_t * convolved[:, np.newaxis, :]


def _vector_raman_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk 2 n_to"],
    spectrum: Complex[np.ndarray, "... n_omega_doubled"],
    *,
    kind: GridKind,
    thg: bool,
    density: float,
) -> Num[np.ndarray, "n_nodes_chunk 2 n_to"]:
    """Convolve trace, difference and coherence drives with compiled kernels.

    Parameters
    ----------
    field_t : numpy.ndarray
        Cartesian field in V/m, shape ``(n_nodes_chunk, 2, n_to)``.
    spectrum : numpy.ndarray
        One or two effective kernels, shape ``(n_raman_kernels, n_to + 1)``.
    kind : {'field', 'envelope'}
        Temporal representation.
    thg : bool
        Keep the real-field harmonic drive; ignored for envelopes.
    density : float
        Outside amplitude density in m^-3, or one for a density-weighted mixture.

    Returns
    -------
    numpy.ndarray
        Cartesian polarisation in C/m^2 with the field's shape.
    """
    n_to = field_t.shape[-1]
    if spectrum.ndim != 2 or spectrum.shape not in ((1, n_to + 1), (2, n_to + 1)):
        raise BackendError("vector Raman needs one or two tensor kernel spectra")
    values = field_t
    if kind == "field" and not thg:
        values = np.array(
            [
                [fourier.analytic_signal(np.real(row)) for row in node]
                for node in field_t
            ]
        )
    x, y = values[:, 0], values[:, 1]
    half = 0.5 if kind == "envelope" or not thg else 1.0
    xx = half * (np.real(x) ** 2 + np.imag(x) ** 2)
    yy = half * (np.real(y) ** 2 + np.imag(y) ** 2)
    n_drives = 1 if spectrum.shape[0] == 1 else 3
    drive = np.zeros((field_t.shape[0], n_drives, 2 * n_to))
    drive[:, 0, :n_to] = xx + yy
    if n_drives == 3:
        drive[:, 1, :n_to] = xx - yy
        drive[:, 2, :n_to] = (
            2 * half * (np.real(x) * np.real(y) + np.imag(x) * np.imag(y))
        )
    kernels = spectrum if n_drives == 1 else spectrum[[0, 1, 1]]
    convolved = np.fft.irfft(
        kernels * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    if n_drives == 1:
        return (EPS0 * density) * field_t * convolved
    u, v, w = convolved[:, 0], convolved[:, 1], convolved[:, 2]
    result = np.empty_like(field_t)
    result[:, 0] = (u + v) * field_t[:, 0] + w * field_t[:, 1]
    result[:, 1] = w * field_t[:, 0] + (u - v) * field_t[:, 1]
    return (EPS0 * density) * result


def ionisation_rate(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    table: RateTable,
) -> Float[np.ndarray, "n_nodes_chunk n_to"]:
    r"""Look the ionisation rate up on the magnitude of the field.

    The ``intensity`` and ``ionisation_rate`` stages of the fused pass, together: the
    magnitude is what the intensity stage forms and the only thing the rate needs. For
    one component that is ``|E|``; for two it is ``hypot(E_x, E_y)`` rather than
    ``sqrt(E_x^2 + E_y^2)``, because the tunnelling exponent is tens to hundreds and
    turns one unit in the last place of the magnitude into 1e-13 of the rate
    (:meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate` takes the same route).

    The table is exactly zero below its ``field_min`` and holds its last value above
    ``field_max`` rather than extrapolating, which is what keeps the loss current of
    :func:`plasma_current` finite and a diverged step from becoming an infinity.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    table : solprop.translate.ionisation.RateTable
        The compiled rate, ``responses.plasma_coefficients.rate``.

    Returns
    -------
    numpy.ndarray
        The rate per atom in s^-1, shape ``(n_nodes_chunk, n_to)``, finite and
        non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import ionisation_rate
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.translate.ionisation import build_rate_table
    >>> table = build_rate_table(OptmatRate("He", 800e-9), (2e9, 1e11), rtol=1e-4)
    >>> field = np.array([[[5e10, 1e6]]])
    >>> np.round(np.log10(ionisation_rate(field, table)[0, 0]), 3).item()
    11.207
    """
    return table(_magnitude(field_t))


def plasma_scan(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    coefficients: PlasmaCoefficients,
    *,
    preionisation: float,
    dt: float,
    integral: RunningIntegral | None = None,
) -> tuple[
    Float[np.ndarray, "n_nodes_chunk n_to"],
    Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
]:
    r"""Integrate the ionised fraction and the free-electron current along t.

    The ``plasma_scan`` stage: two of the three cumulative integrals the response
    declares, which is why its domain is ``memory`` -- the value at one sample depends
    on every earlier sample and on none of the later ones.

    ``f = 1 - (1 - f_0) exp(-int W dt)``, with the pre-ionised fraction **inside** the
    exponential. Writing ``f_0 + 1 - exp(-int W dt)`` instead would let ``f`` exceed
    one, make ``1 - f`` negative and stop ``df/dt`` being ``W (1 - f)``, so the loss
    term would no longer pay exactly one ionisation potential per electron. Then
    ``J_free = (e^2/m_e) int f E dt``, the current of electrons already freed and still
    being accelerated: what lowers the refractive index and blue-shifts the pulse.

    Both integrals are the problem's running-integral rule (SEP 18), applied in the
    argument order of :meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate`, each
    a vectorised cumulative sum (``scipy.integrate.cumulative_trapezoid``); this pass
    and the oracle are the same computation, and other backends' scans are held to the
    scan's rounding bound (SEP 25), not to its order. Under the band
    rule the exposure is floored at zero, so the fraction stays in
    ``[preionisation, 1)`` and is non-decreasing to within the filter's undershoot. Here
    the multipass form **is** the reference, and fusing the three sums into one
    sequential sweep belongs to the compiled backends
    (``docs/guide/media/nonlinear.md``).

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    rate : numpy.ndarray
        The ionisation rate from :func:`ionisation_rate`, s^-1, shape
        ``(n_nodes_chunk, n_to)``.
    coefficients : solprop.translate.responses.PlasmaCoefficients
        The compiled plasma group; only ``charge_mass_ratio`` is read here.
    preionisation : float
        The pre-ionised fraction at this position,
        ``coefficients.preionisation(z)``, in ``[0, 1]``.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule, ``responses.plasma_integral``; ``None`` is the
        trapezoid.

    Returns
    -------
    fraction : numpy.ndarray
        The fraction ionised, shape ``(n_nodes_chunk, n_to)``, in
        ``[preionisation, 1]`` and non-decreasing in time (under the band rule, to
        within the filter's undershoot).
    free_current : numpy.ndarray
        ``J_free`` per unit density, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        C/(m^2 s) per m^-3.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import plasma_scan
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> compiled = plasma_coefficients(
    ...     PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)), (1e9, 1e11), z_max=1.0
    ... )
    >>> field = np.zeros((1, 1, 4))
    >>> fraction, current = plasma_scan(
    ...     field, np.zeros((1, 4)), compiled, preionisation=0.25, dt=1e-16
    ... )
    >>> fraction.tolist(), bool(np.all(current == 0.0))
    ([[0.25, 0.25, 0.25, 0.25]], True)
    """
    rule = _rule(integral)
    exposure = rule.floor(rule.integrate(rate, dt))
    fraction = 1.0 - (1.0 - preionisation) * np.exp(-exposure)
    integrand = fraction[..., np.newaxis, :] * coefficients.charge_mass_ratio * field_t
    free_current = rule.integrate(integrand, dt)
    return fraction, free_current


def _rule(integral: RunningIntegral | None) -> RunningIntegral:
    """Return the running-integral rule a plasma pass uses.

    Parameters
    ----------
    integral : solprop.nonlinear.ionisation.RunningIntegral or None
        The problem's rule, or ``None`` for the trapezoid.

    Returns
    -------
    solprop.nonlinear.ionisation.RunningIntegral
        `integral`, or the trapezoid.
    """
    return RunningIntegral.trapezoid() if integral is None else integral


def plasma_current(  # noqa: PLR0913 - the oracle's own inputs, three of them per node
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    fraction: Float[np.ndarray, "n_nodes_chunk n_to"],
    free_current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    coefficients: PlasmaCoefficients,
    *,
    density: float,
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Add the ionisation loss to the free current and integrate to a polarisation.

    The ``plasma_current`` stage, and the third cumulative integral.
    ``J_loss = I_p W (1 - f) E / |E|^2`` is added pointwise; its form is fixed by
    ``E . J_loss = I_p W (1 - f) = I_p df/dt``, so the field pays exactly one ionisation
    potential for every electron it frees. For one component it is ``I_p W (1-f)/E``,
    sign and all; for two it is that magnitude **directed along the field**, which is
    what makes a linearly polarised pulse give the scalar answer rotated. The
    polarisation is ``P = rho int (J_free + J_loss) dt``.

    The division is written with a ``where`` mask rather than a branch, as the oracle
    writes it, so it never happens at a sample where the rate or the field vanishes: the
    term is finite only because the rate is **exactly** zero below threshold, which is
    what :class:`solprop.translate.ionisation.RateTable` guarantees below its
    ``field_min``; a clipped but non-zero rate would divide by a tiny ``|E|^2``.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    rate : numpy.ndarray
        The ionisation rate from :func:`ionisation_rate`, s^-1, shape
        ``(n_nodes_chunk, n_to)``.
    fraction : numpy.ndarray
        The fraction ionised from :func:`plasma_scan`, shape
        ``(n_nodes_chunk, n_to)``.
    free_current : numpy.ndarray
        ``J_free`` from :func:`plasma_scan`, shape ``(n_nodes_chunk, n_pol, n_to)``.
    coefficients : solprop.translate.responses.PlasmaCoefficients
        The compiled plasma group; only ``ionisation_potential`` is read here.
    density : float
        The density path at this position, ``responses.density(z)``, in m^-3,
        non-negative. The currents are per atom; this is what turns them into a
        polarisation.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule, ``responses.plasma_integral``; ``None`` is the
        trapezoid.

    Returns
    -------
    numpy.ndarray
        The plasma polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import plasma_current
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> compiled = plasma_coefficients(
    ...     PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)), (1e9, 1e11), z_max=1.0
    ... )
    >>> zeros = np.zeros((1, 1, 4))
    >>> flat = np.zeros((1, 4))
    >>> plasma_current(
    ...     zeros, flat, flat, zeros, compiled, density=1e25, dt=1e-16
    ... ).tolist()
    [[[0.0, 0.0, 0.0, 0.0]]]
    """
    current = _plasma_total_current(field_t, rate, fraction, free_current, coefficients)
    return density * _rule(integral).integrate(current, dt)


def _plasma_total_current(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    fraction: Float[np.ndarray, "n_nodes_chunk n_to"],
    free_current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    coefficients: PlasmaCoefficients,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    """Add the guarded loss current to the free current, per unit density.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real field in V/m, shape ``(n_nodes_chunk, n_pol, n_to)``.
    rate : numpy.ndarray
        Ionisation rate in s⁻¹, shape ``(n_nodes_chunk, n_to)``.
    fraction : numpy.ndarray
        Dimensionless ionised population, with the same shape as the rate.
    free_current : numpy.ndarray
        Free current per unit number density, with the same shape as the field.
    coefficients : PlasmaCoefficients
        Ionisation potential in J and the compiled single-channel model.

    Returns
    -------
    numpy.ndarray
        Total current per unit number density in A m, with the field's shape.
    """
    # The squared magnitude rather than the field itself: it is the divisor the guard
    # has to test, and it underflows to zero below about 1e-162 V/m.
    squared = _magnitude(field_t) ** 2
    scale = np.zeros_like(squared)
    np.divide(
        coefficients.ionisation_potential * rate * (1.0 - fraction),
        squared,
        out=scale,
        where=(rate > 0.0) & (squared > 0.0),
    )
    return free_current + scale[..., np.newaxis, :] * field_t


def plasma_channel_fractions(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> tuple[
    Float[np.ndarray, "n_channels n_nodes_chunk n_to"],
    Float[np.ndarray, "n_channels n_nodes_chunk n_to"],
]:
    """Evaluate independent exposure scans in channel order.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape (n_nodes_chunk, n_pol, n_to).
    channels : tuple of PlasmaChannel
        The independent rate tables and population metadata.
    preionisation : tuple of float
        Initial fractions, in channel order, in [0, 1].
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule; ``None`` is the trapezoid.

    Returns
    -------
    rates : numpy.ndarray
        Instantaneous rates in s^-1, shape (n_channels, n_nodes_chunk, n_to).
    fractions : numpy.ndarray
        Ionised fractions with the same shape, dimensionless.

    Raises
    ------
    BackendError
        If the channel scalars do not match their declared count.
    """
    if not channels or len(channels) != len(preionisation):
        raise BackendError("plasma channels require one initial fraction each")
    magnitude = _magnitude(field_t)
    rates = np.stack([channel.rate(magnitude) for channel in channels])
    fractions = np.empty_like(rates)
    rule = _rule(integral)
    for index, initial in enumerate(preionisation):
        exposure = rule.floor(rule.integrate(rates[index], dt))
        fractions[index] = 1.0 - (1.0 - initial) * np.exp(-exposure)
    return rates, fractions


def plasma_integration_diagnostics(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: Sequence[PlasmaCoefficients],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, " 6"]:
    """Return the parts of the column's integration diagnostics (SEP 18).

    For every channel and node: the exposure ``S`` as the rule forms it, before the
    floor, and its increments; the fraction; the work the loss current pays per
    ionisation potential, ``int W (1 - f) dt`` by the trapezoid of the raw rate; the
    electrons that freed, ``f_end - f_0``; and the free-current integrand ``|f E|``.
    Every part is a maximum over nodes and channels, so chunks combine by taking the
    largest of each, and each ratio is normalised by the most-ionised node rather than
    node by node: a node the field barely ionises, whose rate is one sampled spike,
    would otherwise report the filter's full undershoot on a rate at the level of
    rounding (:func:`plasma_integration_statistics` forms the ratios).

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape ``(n_nodes_chunk, n_pol, n_to)``.
    channels : sequence of PlasmaCoefficients
        The channels' compiled coefficients, each with its rate table.
    preionisation : tuple of float
        Each channel's initial fraction at this plane.
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        The largest exposure dip ``-min dS`` and rise ``max dS``; the largest
        ``|paid - freed|`` and ``freed``; the largest ``|f E|`` in the first ``2K``
        samples and anywhere. Dimensionless but for the last two, in V/m.
    """
    rule = _rule(integral)
    reach = 2 * rule.half_width
    magnitude = _magnitude(field_t)
    parts = np.zeros(6)
    for channel, initial in zip(channels, preionisation, strict=True):
        rate = channel.rate(magnitude)
        exposure = rule.integrate(rate, dt)
        steps = np.diff(exposure, axis=-1)
        fraction = 1.0 - (1.0 - initial) * np.exp(-rule.floor(exposure))
        paid = np.trapezoid(rate * (1.0 - fraction), dx=dt, axis=-1)
        freed = fraction[..., -1] - fraction[..., 0]
        integrand = np.abs(fraction[..., np.newaxis, :] * field_t)
        early = float(np.max(integrand[..., :reach])) if reach else 0.0
        parts = np.maximum(
            parts,
            [
                max(-float(np.min(steps)), 0.0),
                float(np.max(steps)),
                float(np.max(np.abs(paid - freed))),
                float(np.max(freed)),
                early,
                float(np.max(integrand)),
            ],
        )
    return parts


def plasma_integration_statistics(
    parts: Float[np.ndarray, " 6"],
) -> tuple[float, float, float]:
    """Form the three diagnostics from :func:`plasma_integration_diagnostics`'s parts.

    Parameters
    ----------
    parts : numpy.ndarray
        The six maxima, combined over every chunk of nodes.

    Returns
    -------
    tuple of float
        The undershoot ``max(-dS) / max(dS)``, the work defect
        ``max |paid - freed| / max freed`` and the boundary level, each zero where
        its denominator is.
    """
    dip, rise, defect, freed, early, peak = (float(part) for part in parts)
    return (
        dip / rise if rise > 0.0 else 0.0,
        defect / freed if freed > 0.0 else 0.0,
        early / peak if peak > 0.0 else 0.0,
    )


def plasma_channels_polarisation(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    densities: tuple[float, ...],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Sum channel weights before the two shared current integrals.

    Each fraction has its own exposure scan. The free-current weight is
    ``sum rho_c f_c e_c²/m_c``; the loss numerator is
    ``sum rho_c I_p,c W_c (1-f_c)``. Their time integrals are shared.
    The original one-channel pass retains its density outside the integrals.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape (n_nodes_chunk, n_pol, n_to).
    channels : tuple of PlasmaChannel
        Ordered independent channel coefficients.
    densities : tuple of float
        Number density of each population in m^-3.
    preionisation : tuple of float
        Initial fraction of each population, dimensionless in [0, 1].
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule shared by every channel; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        Plasma polarisation in C/m², with the field shape.

    Raises
    ------
    BackendError
        If there is not one density per channel.
    """
    return _rule(integral).integrate(
        _plasma_channels_current(
            field_t, channels, densities, preionisation, dt, integral
        ),
        dt,
    )


def _plasma_channels_current(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    densities: tuple[float, ...],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Sum every channel into the free and loss current densities.

    Each fraction has its own exposure scan. The free-current weight is
    ``sum rho_c f_c e_c²/m_c``; the loss numerator is
    ``sum rho_c I_p,c W_c (1-f_c)``. The free-current integral is shared; the loss
    current is pointwise.
    The original one-channel pass retains its density outside the integrals.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape (n_nodes_chunk, n_pol, n_to).
    channels : tuple of PlasmaChannel
        Ordered independent channel coefficients.
    densities : tuple of float
        Number density of each population in m^-3.
    preionisation : tuple of float
        Initial fraction of each population, dimensionless in [0, 1].
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule shared by every channel; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        Total plasma current density in A/m², with the field shape.

    Raises
    ------
    BackendError
        If there is not one density per channel.
    """
    if len(densities) != len(channels):
        raise BackendError("plasma channels require one density each")
    rule = _rule(integral)
    rates, fractions = plasma_channel_fractions(
        field_t, channels, preionisation, dt, rule
    )
    weight = np.zeros_like(rates[0])
    work = np.zeros_like(weight)
    for channel, density, rate, fraction in zip(
        channels, densities, rates, fractions, strict=True
    ):
        weight += (
            density * fraction * channel.charge_mass_ratio * channel.electrons_per_event
        )
        work += density * channel.ionisation_potential * rate * (1.0 - fraction)
    free = rule.integrate(weight[:, None, :] * field_t, dt)
    squared = _magnitude(field_t) ** 2
    guard = np.zeros_like(work)
    np.divide(work, squared, out=guard, where=squared > 0.0)
    loss = guard[:, None, :] * field_t
    return free + loss


def mixture_raman_spectrum(
    components: tuple[RamanCoefficients, ...],
    rates: tuple[float, ...],
    densities: tuple[float, ...],
    dt: float,
) -> Complex[np.ndarray, " n_omega_doubled"]:
    """Form one amplitude-weighted kernel spectrum for all Raman components.

    Parameters
    ----------
    components : tuple of RamanCoefficients
        Raman-active populations in mixture order.
    rates : tuple of float
        Decay rates in s^-1, flattened by component then kernel row.
    densities : tuple of float
        Component amplitude densities in m^-3.
    dt : float
        Fine-grid sample spacing in s.

    Returns
    -------
    numpy.ndarray
        One damped kernel spectrum including densities and dt.

    Raises
    ------
    BackendError
        If the amplitude or decay-rate counts disagree with the components.
    """
    if (
        not components
        or len(densities) != len(components)
        or len(rates) != sum(c.n_terms for c in components)
    ):
        raise BackendError("Raman component amplitudes and rates must match their rows")
    delays = np.arange(components[0].undamped.shape[1]) * dt
    total = np.zeros_like(delays)
    offset = 0
    for component, density in zip(components, densities, strict=True):
        amplitude = np.zeros_like(total)
        for row in component.undamped:
            amplitude += row * np.exp(-rates[offset] * delays)
            offset += 1
        total += density * component.fraction * amplitude
    return kernel_spectrum(total, dt)


def tensor_raman_spectrum(
    components: tuple[RamanCoefficients, ...],
    rates: tuple[float, ...],
    densities: tuple[float, ...],
    dt: float,
) -> Complex[np.ndarray, "n_raman_kernels n_omega_doubled"]:
    """Form effective trace and traceless spectra from independently damped terms.

    Parameters
    ----------
    components : tuple of RamanCoefficients
        Tensor-capable components in mixture order, with per-row numeric weights.
    rates : tuple of float
        Decay rates in s^-1, flattened by component then kernel row.
    densities : tuple of float
        Amplitude densities in m^-3. Supply ``(1.0,)`` when a single component's
        density is multiplied outside the convolution.
    dt : float
        Fine-grid sample spacing in s.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_raman_kernels, n_to + 1)`` with dt included. One trace kernel
        for isotropic-only declarations; otherwise trace and traceless rows.

    Raises
    ------
    BackendError
        If counts disagree or any term lacks a vector contraction.
    """
    if (
        not components
        or len(densities) != len(components)
        or len(rates) != sum(c.n_terms for c in components)
        or not all(c.tensor_capable for c in components)
    ):
        raise BackendError("vector Raman requires matching tensor-capable components")
    n_kernels = 2 if any(np.any(c.tensor_weights[:, 1]) for c in components) else 1
    delays = np.arange(components[0].undamped.shape[1]) * dt
    total = np.zeros((n_kernels, delays.size))
    offset = 0
    for component, density in zip(components, densities, strict=True):
        amplitude = np.zeros_like(total)
        for row, weights in zip(
            component.undamped, component.tensor_weights, strict=True
        ):
            damped = row * np.exp(-rates[offset] * delays)
            for channel in range(n_kernels):
                amplitude[channel] += weights[channel] * damped
            offset += 1
        total += density * component.fraction * amplitude
    return np.fft.rfft(total, axis=-1) * dt


def nonlinear_polarisation(  # noqa: PLR0913 - the fused pass takes what the responses need
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    responses: CompiledResponses,
    t_window_fine: Float[np.ndarray, " n_to_window"],
    *,
    kind: GridKind,
    density: float,
    preionisation: float,
    dt: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    raman_spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None = None,
    scalars: StageScalars | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Run the fused response pass over a chunk of nodes and window it.

    Pass 4 of ``architecture.md`` section 8: the traversal ``responses.response_order``
    names, term by term, accumulating ``P(t)`` and applying the temporal collar at the
    end -- but for SEP 6's dispersive response, whose collar acts on its products
    before its output multipliers (:func:`dispersive_kerr_polarisation`) and which is
    added after. Each stage is one of the named functions of this module, so the order
    the translation layer compiled is the order that runs, and a term this backend
    cannot execute is refused at construction rather than dropped here.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes from :func:`synthesise`, shape
        ``(n_nodes_chunk, n_pol, n_to)``, in V/m.
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the kernel contract: the order, the coefficients and the tables.
    t_window_fine : numpy.ndarray
        The temporal collar profile in ``[0, 1]``, float64, shape ``(n_to,)``.
    kind : {'field', 'envelope'}
        The spectral layout, which the Raman drive depends on.
    density : float
        The density path at this position, ``responses.density(z)``, non-negative.
    preionisation : float
        The pre-ionised fraction at this position, in ``[0, 1]``. Ignored when no
        plasma is compiled.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, for an envelope-grid harmonic
        term.
    raman_spectrum : numpy.ndarray or None, optional
        The damped Raman kernel spectrum at this position from :func:`raman_spectrum`,
        shape ``(n_to + 1,)``. Required when the order names ``raman_drive``; the caller
        forms it once per stage rather than once per chunk, because it does not depend
        on the nodes.

    scalars : StageScalars or None, optional
        Labelled component densities and preionisations at the stage. Required for
        multiple plasma channels; None retains the standalone single-response call.

    Returns
    -------
    numpy.ndarray
        The total polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on
        a field grid and complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import nonlinear_polarisation
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.nonlinear.responses import KerrResponse
    >>> from solprop.profiles import Constant
    >>> from solprop.translate.responses import compile_responses
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0], omega_ref=grid.omega0, aeff=1e-10
    ... )
    >>> compiled = compile_responses(
    ...     [KerrResponse(chi3=1.0)],
    ...     grid,
    ...     mode,
    ...     density=Constant(1.0),
    ...     z_max=1.0,
    ...     prefactor_index=1.0,
    ... )
    >>> field = np.full((1, 1, 4), 2.0)
    >>> out = nonlinear_polarisation(
    ...     field, compiled, np.ones(4), kind="field", density=1.0, preionisation=0.0,
    ...     dt=grid.dt_fine,
    ... )
    >>> bool(np.allclose(out, compiled.kerr_coefficients.coefficient * 8.0))
    True
    """
    sources = nonlinear_sources(
        field_t,
        responses,
        t_window_fine,
        kind=kind,
        density=density,
        preionisation=preionisation,
        dt=dt,
        carrier=carrier,
        raman_spectrum=raman_spectrum,
        scalars=scalars,
    )
    if sources.current is None:
        return sources.polarisation
    # Physical-polarisation oracle and private historical comparison only. Production
    # assembles the current in frequency through held_plasma_spectrum (SEP 27).
    return sources.polarisation + t_window_fine * responses.plasma_integral.integrate(
        sources.current, dt
    )


@dataclass(frozen=True)
class NonlinearSources:
    """Carry the separate local polarisation and field-plasma current.

    Attributes
    ----------
    polarisation : numpy.ndarray
        Collared Kerr and Raman polarisation, including the separately collared
        dispersive Kerr, in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``.
    current : numpy.ndarray or None
        Total ungated plasma current density in A/m^2 with the same shape; ``None``
        when no field plasma is compiled. Every channel and both currents enter once.
    """

    polarisation: Num[np.ndarray, "n_nodes_chunk n_pol n_to"]
    current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"] | None


def nonlinear_sources(  # noqa: PLR0913 - the fused pass takes what the responses need
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    responses: CompiledResponses,
    t_window_fine: Float[np.ndarray, " n_to_window"],
    *,
    kind: GridKind,
    density: float,
    preionisation: float,
    dt: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    raman_spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None = None,
    scalars: StageScalars | None = None,
) -> NonlinearSources:
    r"""Separate the collared local polarisation and the ungated plasma current.

    Pass 4 of ``architecture.md`` section 8: the traversal ``responses.response_order``
    names, term by term, accumulating ``P(t)`` and applying the temporal collar at the
    end -- but for SEP 6's dispersive response, whose collar acts on its products
    before its output multipliers (:func:`dispersive_kerr_polarisation`) and which is
    added after. Each stage is one of the named functions of this module, so the order
    the translation layer compiled is the order that runs, and a term this backend
    cannot execute is refused at construction rather than dropped here.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes from :func:`synthesise`, shape
        ``(n_nodes_chunk, n_pol, n_to)``, in V/m.
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the kernel contract: the order, the coefficients and the tables.
    t_window_fine : numpy.ndarray
        The temporal collar profile in ``[0, 1]``, float64, shape ``(n_to,)``.
    kind : {'field', 'envelope'}
        The spectral layout, which the Raman drive depends on.
    density : float
        The density path at this position, ``responses.density(z)``, non-negative.
    preionisation : float
        The pre-ionised fraction at this position, in ``[0, 1]``. Ignored when no
        plasma is compiled.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, for an envelope-grid harmonic
        term.
    raman_spectrum : numpy.ndarray or None, optional
        The damped Raman kernel spectrum at this position from :func:`raman_spectrum`,
        shape ``(n_to + 1,)``. Required when the order names ``raman_drive``; the caller
        forms it once per stage rather than once per chunk, because it does not depend
        on the nodes.

    scalars : StageScalars or None, optional
        Labelled component densities and preionisations at the stage. Required for
        multiple plasma channels; None retains the standalone single-response call.

    Returns
    -------
    NonlinearSources
        The collared non-plasma polarisation and total ungated current density.
    """
    total = np.zeros_like(field_t)
    current = None
    dispersive = None
    multiple = len(responses.plasma_channels) > 1
    if multiple and scalars is None:
        raise BackendError("multiple plasma channels require labelled stage scalars")
    plasma_density = (
        density
        if scalars is None or not scalars.plasma_densities
        else scalars.plasma_densities[0]
    )
    kerr_density = density if scalars is None else scalars.kerr_density
    raman_density = (
        1.0
        if len(responses.raman_terms) > 1
        else density
        if scalars is None or not scalars.raman_densities
        else scalars.raman_densities[0]
    )
    rate = None
    scan = None
    for stage in responses.response_order:
        if stage is ResponseStage.IONISATION_RATE and not multiple:
            rate = ionisation_rate(field_t, _plasma_rate(responses))
        elif stage is ResponseStage.PLASMA_SCAN and not multiple:
            scan = plasma_scan(
                field_t,
                _checked(rate),
                _plasma(responses),
                preionisation=preionisation,
                dt=dt,
                integral=responses.plasma_integral,
            )
        elif stage is ResponseStage.PLASMA_CURRENT and multiple:
            assert scalars is not None
            current = _plasma_channels_current(
                field_t,
                responses.plasma_channels,
                scalars.plasma_densities,
                scalars.plasma_preionisation,
                dt,
                responses.plasma_integral,
            )
        elif stage is ResponseStage.PLASMA_CURRENT:
            fraction, free_current = _checked(scan)
            current = plasma_density * _plasma_total_current(
                field_t,
                _checked(rate),
                fraction,
                free_current,
                _plasma(responses),
            )
        elif stage is ResponseStage.KERR:
            total = total + kerr_polarisation(
                field_t, _kerr(responses), kerr_density, carrier
            )
        elif stage is ResponseStage.DISPERSIVE_KERR:
            if scalars is None:
                raise BackendError(
                    "the dispersive Kerr stage reads each channel's density from the "
                    "stage scalars"
                )
            # The collar is inside: it acts on the products, before G.
            dispersive = dispersive_kerr_polarisation(
                field_t,
                _checked(responses.dispersive_kerr),
                scalars.dispersive_kerr_densities,
                kind=kind,
                carrier=carrier,
                window=t_window_fine,
            )
        elif stage is ResponseStage.RAMAN_DRIVE:
            total = total + raman_polarisation(
                field_t,
                _raman(raman_spectrum),
                kind=kind,
                thg=_raman_thg(responses),
                density=raman_density,
            )
    windowed = total * t_window_fine
    return NonlinearSources(
        windowed if dispersive is None else windowed + dispersive, current
    )


def held_plasma_spectrum(
    gated: Float[np.ndarray, "... n_to"],
    edge: Float[np.ndarray, "... n_to"],
    omega: Float[np.ndarray, " n_omega"],
    crop_index: Int[np.ndarray, " n_omega"],
    active: Bool[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "... n_omega"]:
    r"""Transform a current held at the temporal collar (SEP 27 section 4.4).

    Form ``DFT[g J]/(i omega) - DFT[g' J]/(i omega)^2`` only on active bins.
    Projection can precede or follow this linear map; crop and amplitude factors
    belong to the route. No running integral or numerical derivative enters it.

    Parameters
    ----------
    gated, edge : numpy.ndarray
        Sampled ``g J`` and ``g' J``, in A/m^2 and A/(m^2 s), with time last.
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape ``(n_omega,)``.
    crop_index : numpy.ndarray
        Coarse positions in the fine half-spectrum, shape ``(n_omega,)``.
    active : numpy.ndarray
        Active frequency mask, shape ``(n_omega,)``; all selected frequencies positive.

    Returns
    -------
    numpy.ndarray
        Polarisation-equivalent spectrum in raw DFT units, with inactive bins zero.
    """
    current = np.fft.rfft(gated, axis=-1)[..., crop_index]
    derivative = np.fft.rfft(edge, axis=-1)[..., crop_index]
    result = np.zeros_like(current)
    q = 1j * omega[active]
    result[..., active] = current[..., active] / q - derivative[..., active] / q**2
    return result


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
    >>> from solprop.backends.numpy_ref.passes import project
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
    :func:`stage_output` applies next.

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
    >>> from solprop.backends.numpy_ref.passes import to_frequency
    >>> samples = np.ones((1, 1, 8))
    >>> index = np.arange(3, dtype=np.intp)
    >>> to_frequency(samples, index, kind="field").real
    array([[[8., 0., 0.]]])
    """
    n_modes, n_pol = polarisation_t.shape[:2]
    flat = polarisation_t.reshape(n_modes * n_pol, -1)
    spectrum = fourier.forward(flat, kind=kind)[..., crop_index]
    return spectrum.reshape(n_modes, n_pol, -1)


def _magnitude(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
) -> Float[np.ndarray, "n_nodes_chunk n_to"]:
    """Return the field magnitude the ionisation rate and the loss term take.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``.

    Returns
    -------
    numpy.ndarray
        ``|E|`` in V/m, shape ``(n_nodes_chunk, n_to)``.
    """
    if field_t.shape[-2] == 1:
        return np.abs(field_t[..., 0, :])
    return np.hypot(field_t[..., 0, :], field_t[..., 1, :])


def _kerr(responses: CompiledResponses) -> KerrCoefficients:
    """Return the compiled Kerr coefficients a declared Kerr stage promised.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses.

    Returns
    -------
    solprop.translate.responses.KerrCoefficients
        The coefficients.

    Raises
    ------
    BackendError
        If the order names the stage and the coefficients are absent.
    """
    if responses.kerr_coefficients is None:
        msg = "response_order names the Kerr stage but no coefficients were compiled"
        raise BackendError(msg)
    return responses.kerr_coefficients


def _plasma(responses: CompiledResponses) -> PlasmaCoefficients:
    """Return the compiled plasma group a declared plasma stage promised.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses.

    Returns
    -------
    solprop.translate.responses.PlasmaCoefficients
        The group.

    Raises
    ------
    BackendError
        If the order names a plasma stage and the group is absent.
    """
    if responses.plasma_coefficients is None:
        msg = "response_order names a plasma stage but no coefficients were compiled"
        raise BackendError(msg)
    return responses.plasma_coefficients


def _raman_thg(responses: CompiledResponses) -> bool:
    """Return whether a field grid's Raman drive keeps the ``2 omega`` beat.

    The drive form is a property of the problem and not of each response, and until
    contract version 3 it had no row: the translation layer checked that the declared
    Kerr and Raman responses agree about it on a field grid, refused a field-grid Raman
    with no Kerr beside it, and a backend read the answer off the Kerr term's own flag.
    Both refusals stand -- they are what makes the agreement true -- but the answer is
    now carried, so it is read where it belongs. An envelope grid's drive is ``|A|^2/2``
    whatever it says, which is why the value is not checked there.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses.

    Returns
    -------
    bool
        True when the drive is ``E^2`` rather than ``|E_a|^2/2``.

    Raises
    ------
    BackendError
        If no Raman leaf was compiled, so there is no drive to describe.
    """
    compiled = responses.raman_terms
    if not compiled:
        msg = (
            "the fused pass reached the raman_drive stage with no "
            "responses.raman_coefficients; the stage is in response_order exactly when "
            "the leaf is present"
        )
        raise BackendError(msg)
    return compiled[0].thg


def _raman(
    spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None,
) -> Complex[np.ndarray, "... n_omega_doubled"]:
    """Return the Raman kernel spectrum a declared Raman stage promised.

    Parameters
    ----------
    spectrum : numpy.ndarray or None
        What the caller formed for this position.

    Returns
    -------
    numpy.ndarray
        The same spectrum.

    Raises
    ------
    BackendError
        If the order names the stage and no spectrum was formed.
    """
    if spectrum is None:
        msg = (
            "response_order names the Raman stage but no kernel spectrum was given; "
            "form it once per stage with raman_spectrum and pass it in"
        )
        raise BackendError(msg)
    return spectrum


def _plasma_rate(responses: CompiledResponses) -> RateTable:
    """Return the rate table the ionisation stage reads.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses.

    Returns
    -------
    solprop.translate.ionisation.RateTable
        The table.

    Raises
    ------
    BackendError
        If the order names the stage and no table was compiled.
    """
    if len(responses.ionisation_table) != 1:
        msg = "response_order names the ionisation stage but no rate table was compiled"
        raise BackendError(msg)
    return responses.ionisation_table[0]


def _checked[T](value: T | None) -> T:
    """Return a stage result the compiled order guarantees is present.

    Parameters
    ----------
    value : object or None
        What the earlier stage produced.

    Returns
    -------
    object
        The same value.

    Raises
    ------
    BackendError
        If a stage ran before the one that feeds it.
    """
    if value is None:
        msg = (
            "a plasma stage ran before the one that feeds it; response_order is out of "
            "sequence"
        )
        raise BackendError(msg)
    return value


def absorber_term(
    modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
    temporal_rate_fine: Float[np.ndarray, " n_to"],
    crop_index: Int[np.ndarray, " n_crop"],
    active_mask: Bool[np.ndarray, " n_active"],
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Return the temporal absorber as a contribution to the stage derivative.

    The ``"nonlinear"`` absorber scheme: ``-alpha(t) E(t) / 2`` on the modal time-domain
    field, transformed and cropped so that it can be added to the stage in the same
    place the normalised polarisation is. ``alpha`` is a **power** coefficient, so the
    field carries the half (:func:`solprop.boundaries.nonlinear_absorber_term`).

    It cannot ride the polarisation's own transform, and this is the whole reason the
    two schemes exist: the polarisation is multiplied by the UPPE prefactor *after* the
    forward transform, and the absorber is already a rate on the field rather than a
    polarisation, so it would have to be divided by a prefactor that is exactly zero
    outside the active band. One extra modal forward transform per stage is its price,
    and it is what buys FSAL, since nothing then changes the state after the last stage
    of a step.

    The **fine** grid is used rather than the coarse one because ``alpha(t) E(t)`` has
    more bandwidth than ``E(t)`` and the coarse grid would alias the product back into
    the band; the contract carries both axes for exactly this reason
    (``kernel-contract.md`` section 5).

    The result is masked to the active band, as the polarisation beside it already is
    through a prefactor that is exactly zero there. ``alpha(t) E(t)`` has content at
    every frequency, and the active mask is the statement that the equation is not
    solved outside that band: leaving it in would put content into the state that the
    accept pass's hard mask then removes, which is both meaningless and enough to break
    the first-same-as-last identity (measured at 7.4e-7 before the mask and 1.8e-16
    after it).

    Parameters
    ----------
    modal_t : numpy.ndarray
        The lab-frame modal field on the fine time grid, shape
        ``(n_modes, n_pol, n_to)``, as :func:`to_time` produced it.
    temporal_rate_fine : numpy.ndarray
        The collar rate on the fine time axis, 1/m, non-negative, shape ``(n_to,)``.
    crop_index : numpy.ndarray
        Where each coarse bin sits in the fine spectrum, ``numpy.intp``, shape
        ``(n_omega,)``.
    active_mask : numpy.ndarray
        The band the equation is solved on, Boolean, shape ``(n_omega,)``.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        The absorber's contribution on the coarse grid, complex128, shape
        ``(n_modes, n_pol, n_omega)``, exactly zero outside the active band, in the
        same units as the stage derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import absorber_term
    >>> field = np.ones((1, 1, 8))
    >>> rate = np.full(8, 2.0)
    >>> index = np.arange(2, dtype=np.intp)
    >>> absorber_term(field, rate, index, np.ones(2, bool), kind="field").real
    array([[[-8.,  0.]]])
    """
    n_modes, n_pol = modal_t.shape[:2]
    flat = modal_t.reshape(n_modes * n_pol, -1)
    damped = boundaries.nonlinear_absorber_term(flat, temporal_rate_fine)
    spectrum = fourier.forward(damped, kind=kind)[..., crop_index] * active_mask
    return spectrum.reshape(n_modes, n_pol, -1)


def stage_output(
    polarisation: Complex[np.ndarray, "n_modes n_pol n_omega"],
    prefactor: Complex[np.ndarray, "n_modes_p n_pol_p n_omega_p"],
    omega_window: Float[np.ndarray, " n_window"],
    inverse_propagator: Complex[np.ndarray, "n_batch_i n_modes_i n_pol_i n_omega_i"],
    absorber: Complex[np.ndarray, "n_modes_a n_pol_a n_omega_a"] | None = None,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Normalise the polarisation and return it to the interaction picture.

    ``k_i = exp(-dPhi_i) (p(omega, z) W(omega) crop(Ptilde) + A_i)``, pass 7 of
    ``architecture.md`` section 8, with ``A_i`` the temporal absorber's own term when
    the ``"nonlinear"`` absorber scheme is active and nothing otherwise. The absorber
    sits beside the normalised polarisation rather than inside it because it is a rate
    on the field and not a polarisation: the prefactor does not apply to it. The
    prefactor is the UPPE factor ``-i omega / (2 eps0 c Re n_eff)`` (or its envelope
    form) with the crop scale folded in, and it is exactly zero outside the active band,
    so no unnormalised polarisation survives there for the stages and the error estimate
    to see.

    Parameters
    ----------
    polarisation : numpy.ndarray
        The modal polarisation on the coarse grid, complex128, shape
        ``(n_modes, n_pol, n_omega)``, in C/m^2.
    prefactor : numpy.ndarray
        ``responses.prefactor_at(responses, z)``, complex128, in m/(V s), broadcastable
        against the polarisation.
    omega_window : numpy.ndarray
        The spectral collar profile in ``[0, 1]``, float64, shape ``(n_omega,)``.
    inverse_propagator : numpy.ndarray
        ``exp(-dPhi_i)``, dimensionless, broadcastable against the state; the negation
        of the exponent :func:`stage_input` used, never an independent evaluation.
    absorber : numpy.ndarray or None, optional
        The temporal absorber's contribution from :func:`absorber_term`, or ``None``
        under the ``"split_step"`` scheme, where the rate is applied to the state
        once per accepted step instead.

    Returns
    -------
    numpy.ndarray
        The stage derivative in the interaction variable, complex128, shape
        ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts per metre.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import stage_output
    >>> polarisation = np.ones((1, 1, 2), dtype=complex)
    >>> prefactor = np.array([[[-2.0j, 0.0j]]])
    >>> unit = np.ones((1, 1, 1, 2), dtype=complex)
    >>> stage_output(polarisation, prefactor, np.ones(2), unit).ravel().tolist()
    [-2j, 0j]
    """
    normalised = prefactor * omega_window * polarisation
    if absorber is not None:
        normalised = normalised + absorber
    return inverse_propagator * normalised


def accumulate(
    total: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    stage: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    weight: float,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Add one weighted stage derivative to a running accumulator.

    ``total + w k_i``. The solution and the embedded error are running accumulators
    rather than materialised stage combinations, so nothing beyond the stages the next
    one needs is kept.

    Parameters
    ----------
    total : numpy.ndarray
        The accumulator so far, complex128, shape
        ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts per metre.
    stage : numpy.ndarray
        The stage derivative ``k_i``, same shape and units.
    weight : float
        The tableau weight: ``b_i`` for the solution, ``b_i - bhat_i`` for the error.

    Returns
    -------
    numpy.ndarray
        The updated accumulator, same shape and units.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import accumulate
    >>> total = np.zeros((1, 1, 1, 2), dtype=complex)
    >>> accumulate(total, np.ones_like(total), 0.5).ravel()
    array([0.5+0.j, 0.5+0.j])
    """
    return total + weight * stage


def error_norm(
    error: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    state_new: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    norm_mask: Bool[np.ndarray, " n_norm"],
    controller: PIController,
) -> float:
    r"""Reduce the embedded error estimate to the scaled norm the controller consumes.

    ``||e||_2 / (rtol max(||y||_2, ||y_new||_2) + atol)`` over the **in-band** mask, one
    global scale -- the weak norm of ``docs/dev/stepper.md`` section 3. When the
    controller carries weights the scale is per bin instead and the division happens
    before the reduction; the magnitude stays global either way, because a per-element
    magnitude collapses the step to zero on a spectrum whose in-band dynamic range
    reaches 1e17 (section 10, trap 1).

    Parameters
    ----------
    error : numpy.ndarray
        The embedded difference of the propagated state, complex128, shape
        ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts.
    state, state_new : numpy.ndarray
        The state before and after the trial step, same shape and units.
    norm_mask : numpy.ndarray
        The bins the norm scores (``grids.norm_mask``, the controller's ``region``),
        Boolean, shape ``(n_omega,)``.
    controller : solprop.stepper.controller.PIController
        The controller, which carries the tolerances and the choice of norm.

    Returns
    -------
    float
        The scaled norm, dimensionless. One means the step sat exactly at tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import PIController
    >>> from solprop.backends.numpy_ref.passes import error_norm
    >>> state = np.ones((1, 1, 1, 2), dtype=complex)
    >>> mask = np.ones(2, dtype=bool)
    >>> control = PIController(rtol=1e-2, atol=0.0)
    >>> round(error_norm(1e-2 * state, state, state, mask, control), 12)
    1.0
    """
    scale = controller.scale(
        float(np.linalg.norm(state[..., norm_mask])),
        float(np.linalg.norm(state_new[..., norm_mask])),
    )
    residual = error[..., norm_mask]
    if isinstance(scale, np.ndarray):
        residual = residual / scale[norm_mask]
    else:
        residual = residual / scale
    return float(np.linalg.norm(residual))


def accept_step(
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    propagator: Complex[np.ndarray, "n_batch_p n_modes_p n_pol_p n_omega_p"],
    dz: float,
    accumulated: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Advance the state over a completed step.

    ``y <- exp(dPhi) (y + h sum_i b_i k_i)``. The interaction picture is anchored at the
    start of the step, so the whole interaction-variable solution -- the state and the
    accumulated stages together -- is carried to the end of the step by one exponential.
    The same factor applied to ``h sum_i (b_i - bhat_i) k_i`` gives the local error of
    the *propagated* state, which is what :func:`error_norm` scales.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the anchor, complex128, shape
        ``(n_batch, n_modes, n_pol, n_omega)``, in the square root of watts.
    propagator : numpy.ndarray
        ``exp(dPhi)`` over the whole step, dimensionless, broadcastable against the
        state.
    dz : float
        Length of the step, m, positive.
    accumulated : numpy.ndarray
        ``sum_i b_i k_i``, same shape as the state, in the square root of watts per
        metre.

    Returns
    -------
    numpy.ndarray
        The state at the end of the step, same shape and units.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import accept_step
    >>> state = np.ones((1, 1, 1, 2), dtype=complex)
    >>> accept_step(state, 2.0 * np.ones_like(state), 0.5, state).ravel()
    array([3.+0.j, 3.+0.j])
    """
    return propagator * (state + dz * accumulated)


def apply_hard_mask(
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    active_mask: Bool[np.ndarray, " n_active"],
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Zero the state outside the active band.

    ``y <- y * active_mask``, once per accepted step. The hard band limit is the one
    boundary that stays a 0/1 mask rather than becoming a rate, because it is
    idempotent: the operator and the prefactor are already exactly zero outside the
    active band, so nothing regrows there and the total absorption does not depend on
    the step count. The **active** mask, not the narrower in-band one, which would erase
    the spectral collar the rate acts on.

    Parameters
    ----------
    state : numpy.ndarray
        The state, complex128, shape ``(n_batch, n_modes, n_pol, n_omega)``, in the
        square root of watts.
    active_mask : numpy.ndarray
        The active band, ``omega_min/2 < omega < 1.1 omega_max``, Boolean, shape
        ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The masked state, same shape and units.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import apply_hard_mask
    >>> state = np.ones((1, 1, 1, 3), dtype=complex)
    >>> apply_hard_mask(state, np.array([True, True, False])).ravel()
    array([1.+0.j, 1.+0.j, 0.+0.j])
    """
    return state * active_mask


def apply_temporal_rate(
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    rate: Float[np.ndarray, " n_rate"],
    dz: float,
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    r"""Apply the temporal absorber as an exact rate over one accepted step.

    ``y <- F[F^-1[y] exp(-alpha(t) h / 2)]``, the split-step absorber scheme. The
    absorber cannot be folded into ``Phi`` because it is not diagonal in omega, and a
    per-step *window* would give total absorption ``W^n`` and no converged answer; as a
    rate the factors telescope, so the total is ``exp(-alpha(t) L / 2)`` however the
    steps fall. ``alpha`` is a power coefficient, hence the half on the field.

    Parameters
    ----------
    state : numpy.ndarray
        The state, complex128, shape ``(n_batch, n_modes, n_pol, n_omega)``, in the
        square root of watts.
    rate : numpy.ndarray
        ``alpha(t)`` on the coarse time grid, float64, shape ``(n_t,)``, in 1/m,
        non-negative.
    dz : float
        Length of the accepted step, m, positive.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        The absorbed state, same shape and units.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes import apply_temporal_rate
    >>> state = np.array([[[[4.0 + 0.0j, 0.0j, 0.0j]]]])
    >>> absorbed = apply_temporal_rate(state, np.zeros(4), 1.0, kind="field")
    >>> bool(np.allclose(absorbed, state))
    True
    """
    time = fourier.inverse(state, kind=kind)
    return fourier.forward(time * np.exp(-0.5 * rate * dz), kind=kind)

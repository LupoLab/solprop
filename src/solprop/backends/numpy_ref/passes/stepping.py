# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The step arithmetic around a stage, and the once-per-step boundary treatments.

The interaction-picture Runge-Kutta trial's own passes: :func:`stage_input` (pass 1:
combine the stages, return to the lab frame, add the noise), :func:`accumulate` (the
running solution and error), :func:`accept_step` (carry the step to its end) and
:func:`error_norm` (the controller's scaled norm). :func:`stage_output` (pass 7:
normalise the polarisation and return to the step's frame) serves every march: the
Adams, exponential and low-storage engines call it with a unit frame factor.
:func:`absorber_term` is the in-stage absorber beside it (ADR-025's ``"nonlinear"``
scheme); :func:`apply_hard_mask` and :func:`apply_temporal_rate` are applied once per
accepted step.
"""

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int

from solprop import boundaries, fourier
from solprop.backends.numpy_ref.passes.transforms import to_frequency
from solprop.fourier import GridKind
from solprop.stepper.controller import PIController

__all__ = [
    "absorber_term",
    "accept_step",
    "accumulate",
    "apply_hard_mask",
    "apply_temporal_rate",
    "error_norm",
    "stage_input",
    "stage_output",
]


def stage_input(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    stage_sum: Complex[np.ndarray, "n_modes n_pol n_omega"],
    dz: float,
    propagator: Complex[np.ndarray, "n_modes_p n_pol_p n_omega_p"],
    noise: Complex[np.ndarray, "n_modes_n n_pol_n n_omega_n"] | None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
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
        ``(n_modes, n_pol, n_omega)``, in the square root of watts.
    stage_sum : numpy.ndarray
        ``sum_j a_ij k_j`` in the interaction variable, same shape and dtype, in the
        square root of watts per metre.
    dz : float
        Length of the step, m, positive.
    propagator : numpy.ndarray
        ``exp(dPhi_i)``, dimensionless, broadcastable against the state.
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
    >>> from solprop.backends.numpy_ref.passes.stepping import stage_input
    >>> state = np.ones((1, 1, 2), dtype=complex)
    >>> zero = np.zeros_like(state)
    >>> stage_input(state, zero, 0.1, 2.0 * np.ones_like(state), None).ravel()
    array([2.+0.j, 2.+0.j])
    """
    combined = propagator * (state + dz * stage_sum)
    return combined if noise is None else combined + noise


def stage_output(
    polarisation: Complex[np.ndarray, "n_modes n_pol n_omega"],
    prefactor: Complex[np.ndarray, "n_modes_p n_pol_p n_omega_p"],
    omega_window: Float[np.ndarray, " n_window"],
    inverse_propagator: Complex[np.ndarray, "n_modes_i n_pol_i n_omega_i"],
    absorber: Complex[np.ndarray, "n_modes_a n_pol_a n_omega_a"] | None = None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
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
        ``(n_modes, n_pol, n_omega)``, in the square root of watts per metre.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.stepping import stage_output
    >>> polarisation = np.ones((1, 1, 2), dtype=complex)
    >>> prefactor = np.array([[[-2.0j, 0.0j]]])
    >>> unit = np.ones((1, 1, 2), dtype=complex)
    >>> stage_output(polarisation, prefactor, np.ones(2), unit).ravel().tolist()
    [-2j, 0j]
    """
    normalised = prefactor * omega_window * polarisation
    if absorber is not None:
        normalised = normalised + absorber
    return inverse_propagator * normalised


def accumulate(
    total: Complex[np.ndarray, "n_modes n_pol n_omega"],
    stage: Complex[np.ndarray, "n_modes n_pol n_omega"],
    weight: float,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Add one weighted stage derivative to a running accumulator.

    ``total + w k_i``. The solution and the embedded error are running accumulators
    rather than materialised stage combinations, so nothing beyond the stages the next
    one needs is kept.

    Parameters
    ----------
    total : numpy.ndarray
        The accumulator so far, complex128, shape
        ``(n_modes, n_pol, n_omega)``, in the square root of watts per metre.
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
    >>> from solprop.backends.numpy_ref.passes.stepping import accumulate
    >>> total = np.zeros((1, 1, 2), dtype=complex)
    >>> accumulate(total, np.ones_like(total), 0.5).ravel()
    array([0.5+0.j, 0.5+0.j])
    """
    return total + weight * stage


def accept_step(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    propagator: Complex[np.ndarray, "n_modes_p n_pol_p n_omega_p"],
    dz: float,
    accumulated: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
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
        ``(n_modes, n_pol, n_omega)``, in the square root of watts.
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
    >>> from solprop.backends.numpy_ref.passes.stepping import accept_step
    >>> state = np.ones((1, 1, 2), dtype=complex)
    >>> accept_step(state, 2.0 * np.ones_like(state), 0.5, state).ravel()
    array([3.+0.j, 3.+0.j])
    """
    return propagator * (state + dz * accumulated)


def error_norm(
    error: Complex[np.ndarray, "n_modes n_pol n_omega"],
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    state_new: Complex[np.ndarray, "n_modes n_pol n_omega"],
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
        ``(n_modes, n_pol, n_omega)``, in the square root of watts.
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
    >>> from solprop.backends.numpy_ref.passes.stepping import error_norm
    >>> state = np.ones((1, 1, 2), dtype=complex)
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


def absorber_term(
    modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
    temporal_rate_fine: Float[np.ndarray, " n_to"] | None,
    radial: Float[np.ndarray, "n_modes_r n_modes_r"] | None,
    crop_index: Int[np.ndarray, " n_crop"],
    active_mask: Bool[np.ndarray, " n_active"],
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Return the in-stage absorbers as a contribution to the stage derivative.

    The ``"nonlinear"`` absorber scheme: ``-alpha(t) E(t) / 2`` on the modal time-domain
    field, transformed and cropped so that it can be added to the stage in the same
    place the normalised polarisation is. ``alpha`` is a **power** coefficient, so the
    field carries the half (:func:`solprop.boundaries.nonlinear_absorber_term`). The
    radial route's transverse collar is a matrix on the mode axis
    (:func:`solprop.backends.spatial.radial_absorber_matrix`) and adds to the same
    time-domain term, so both share one transform.

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
        ``(n_modes, n_pol, n_to)``, as
        :func:`~solprop.backends.numpy_ref.passes.transforms.to_time` produced it.
    temporal_rate_fine : numpy.ndarray or None
        The collar rate on the fine time axis, 1/m, non-negative, shape ``(n_to,)``;
        ``None`` when the temporal collar is not applied in-stage.
    radial : numpy.ndarray or None
        The radial collar's matrix on the mode axis, 1/m, shape
        ``(n_modes, n_modes)``; ``None`` off the radial route or without its collar.
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
    >>> from solprop.backends.numpy_ref.passes.stepping import absorber_term
    >>> field = np.ones((1, 1, 8))
    >>> rate = np.full(8, 2.0)
    >>> index = np.arange(2, dtype=np.intp)
    >>> absorber_term(field, rate, None, index, np.ones(2, bool), kind="field").real
    array([[[-8.,  0.]]])
    """
    term = np.zeros_like(modal_t)
    if temporal_rate_fine is not None:
        term = boundaries.nonlinear_absorber_term(modal_t, temporal_rate_fine)
    if radial is not None:
        n_modes = modal_t.shape[0]
        term = term + (radial @ modal_t.reshape(n_modes, -1)).reshape(modal_t.shape)
    return to_frequency(term, crop_index, kind=kind) * active_mask


def apply_hard_mask(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    active_mask: Bool[np.ndarray, " n_active"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
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
        The state, complex128, shape ``(n_modes, n_pol, n_omega)``, in the
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
    >>> from solprop.backends.numpy_ref.passes.stepping import apply_hard_mask
    >>> state = np.ones((1, 1, 3), dtype=complex)
    >>> apply_hard_mask(state, np.array([True, True, False])).ravel()
    array([1.+0.j, 1.+0.j, 0.+0.j])
    """
    return state * active_mask


def apply_temporal_rate(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    rate: Float[np.ndarray, " n_rate"],
    dz: float,
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Apply the temporal absorber as an exact rate over one accepted step.

    ``y <- F[F^-1[y] exp(-alpha(t) h / 2)]``, the split-step absorber scheme. The
    absorber cannot be folded into ``Phi`` because it is not diagonal in omega, and a
    per-step *window* would give total absorption ``W^n`` and no converged answer; as a
    rate the factors telescope, so the total is ``exp(-alpha(t) L / 2)`` however the
    steps fall. ``alpha`` is a power coefficient, hence the half on the field. On a
    field grid the real inverse transform drops the imaginary parts of the DC bin, and
    of the Nyquist bin of an even ``n_t``; the backends' finishes mask both bins before
    the collar.

    Parameters
    ----------
    state : numpy.ndarray
        The state, complex128, shape
        ``(n_modes, n_pol, n_omega)``, in the square root of watts. On a
        field grid its DC and Nyquist bins are real (above).
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
    >>> from solprop.backends.numpy_ref.passes.stepping import apply_temporal_rate
    >>> state = np.array([[[4.0 + 0.0j, 0.0j, 0.0j]]])
    >>> absorbed = apply_temporal_rate(state, np.zeros(4), 1.0, kind="field")
    >>> bool(np.allclose(absorbed, state))
    True
    """
    time = fourier.inverse(state, kind=kind)
    return fourier.forward(time * np.exp(-0.5 * rate * dz), kind=kind)

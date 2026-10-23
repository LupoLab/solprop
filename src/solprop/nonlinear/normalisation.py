# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prefactors that turn a nonlinear polarisation into a rate of change of the field.

The nonlinear term of the propagation equation is the polarisation, transformed to the
frequency domain, multiplied by

``-i omega / (2 eps0 c n_eff(omega))``

(``docs/dev/architecture.md`` section 2). That one expression covers both grids: on
a field grid it multiplies the transform of the real polarisation, on an envelope grid
the transform of its envelope, and in both ``omega`` is the absolute frequency the grid
stores. Its ``omega`` dependence **is** the optical shock term, so turning the shock off
is not a different equation but the same prefactor with ``omega`` frozen at the
carrier.

Mode-averaged propagation adds one more step: the propagated amplitude is in the square
root of watts, and a response needs volts per metre. :func:`sqrtw_to_vm` is that
conversion and :func:`mode_average_prefactor` is the two composed.

What is *not* here is the transform normalisation: the crop and pad scale factors belong
to the grid and are folded in by the translation layer (``docs/dev/kernel-contract.md``
section 5).
"""

import logging
import math

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike

from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode

__all__ = [
    "modal_kerr_coefficient",
    "mode_average_prefactor",
    "sqrtw_to_vm",
    "uppe_prefactor",
]

logger = logging.getLogger(__name__)

_HERMITIAN_ATOL = 1e-12
"""Relative departure from Hermitian a declared flux metric may carry.

A sampled Gram is built as ``(E x conj(H) + conj(E) x H)/4`` and is Hermitian by
construction up to the rounding of that reduction, which is of order the node count
times machine epsilon. A matrix further from Hermitian than this is not a sampled
flux metric and is refused rather than symmetrised.
"""


def sqrtw_to_vm(aeff: float) -> float:
    """Return the factor converting a mode-averaged amplitude to a field strength.

    ``E[V/m] = a[sqrt(W)] * sqrtw_to_vm(A_eff)`` with
    ``sqrtw_to_vm = 1 / sqrt(eps0 c A_eff / 2)``, applied to the internal
    vacuum-normalised amplitude ``a``, so that ``eps0 c |E_a|^2 / 2`` is the
    vacuum-admittance intensity ``|a|^2 / A_eff`` -- the true power ``P = Re(n_eff)
    |a|^2`` divided by ``Re(n_eff) A_eff`` (``docs/dev/conventions.md`` section 4 and
    ``docs/guide/results/power-and-intensity.md``).

    Parameters
    ----------
    aeff : float
        Effective area in m^2, positive and finite.

    Returns
    -------
    float
        The conversion factor in V/(m sqrt(W)).

    Raises
    ------
    ModeError
        If ``aeff`` is not positive and finite.

    Examples
    --------
    >>> from solprop.nonlinear.normalisation import sqrtw_to_vm
    >>> f"{sqrtw_to_vm(1e-10):.6e}"
    '2.744924e+06'
    """
    if not (math.isfinite(aeff) and aeff > 0.0):
        msg = f"aeff must be positive and finite, got {aeff!r}"
        raise ModeError(msg)
    return 1.0 / math.sqrt(0.5 * EPS0 * C * aeff)


def uppe_prefactor(
    mode: Mode,
    grid: TimeGrid,
    z: float = 0.0,
    *,
    shock: bool = True,
    index: float | None = None,
) -> Complex[np.ndarray, " n_omega"]:
    """Return the UPPE prefactor on the coarse frequency grid.

    ``-i omega / (2 eps0 c n_eff(omega, z))``, evaluated on the grid's active band and
    zero elsewhere. Zero rather than one out of band: the state there is removed by the
    hard band mask, and leaving raw, unnormalised polarisation in those bins would be a
    defect that every Runge-Kutta stage and error estimate sees.

    With ``shock=False`` the ``omega`` of the numerator is frozen at the grid's carrier,
    which removes self-steepening; the ``n_eff(omega)`` of the denominator is part of
    the projection, not of the shock, and stays frequency dependent.

    Only the **real** part of the effective index appears. The loss belongs in the
    linear operator; carrying it here as well would give the nonlinear term a small
    real prefactor, that is a spurious gain, and the prefactor would no longer be
    purely imaginary.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The mode whose effective index appears in the denominator. Unused when `index`
        is given.
    grid : solprop.grid.TimeGrid
        The grid; supplies the frequency axis, the active band and the carrier.
    z : float, optional
        Position along the waveguide, m.
    shock : bool, optional
        Keep the ``omega`` dependence of the numerator. Default ``True``.
    index : float or None, optional
        Replace the effective index by this constant. The GNLSE passes 1.0:
        its ``beta`` is a fitted dispersion rather than a physical effective index, so
        ``c beta / omega`` need not be anywhere near one. ``None``, the default, takes
        the index from the mode.

    Returns
    -------
    numpy.ndarray
        The prefactor in m/(V s) per unit polarisation, purely imaginary, shape
        ``(n_omega,)``.

    Raises
    ------
    ModeError
        If the real part of the effective index is not finite, or vanishes, anywhere in
        the active band; it is not lower-bounded to keep the run going.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import C, EPS0
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.nonlinear.normalisation import uppe_prefactor
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor([1.0], omega_ref=grid.omega0)
    >>> prefactor = uppe_prefactor(mode, grid, index=1.0)
    >>> centre = int(np.argmin(abs(grid.omega - grid.omega0)))
    >>> expected = -1j * grid.omega0 / (2 * EPS0 * C)
    >>> bool(np.isclose(prefactor[centre], expected))
    True
    """
    active = grid.active_mask
    omega = grid.omega[active]
    if index is None:
        effective = np.real(np.asarray(mode.neff(omega, z)))
    else:
        effective = np.full(omega.shape, float(index))
    if not np.all(np.isfinite(effective)) or np.any(effective == 0.0):
        msg = (
            f"the real effective index is zero or not finite at z = {z!r} m in "
            f"{int(np.sum(~np.isfinite(effective) | (effective == 0.0)))} of "
            f"{omega.size} active bins, so the prefactor has no value there"
        )
        raise ModeError(msg)

    numerator = omega if shock else np.full(omega.shape, grid.omega0)
    prefactor = np.zeros(grid.n_omega, dtype=complex)
    prefactor[active] = -1j * numerator / (2.0 * EPS0 * C * effective)
    return prefactor


def mode_average_prefactor(
    mode: Mode,
    grid: TimeGrid,
    z: float = 0.0,
    *,
    shock: bool = True,
    index: float | None = None,
) -> Complex[np.ndarray, " n_omega"]:
    """Return the prefactor a mode-averaged run applies, amplitude to amplitude.

    The composition of :func:`uppe_prefactor` with :func:`sqrtw_to_vm`: the response is
    evaluated on a field in V/m obtained by multiplying the propagated amplitude by
    ``sqrtw_to_vm``, and its polarisation is brought back by dividing by the same
    factor, so

    ``mode_average_prefactor = uppe_prefactor / sqrtw_to_vm(A_eff)``.

    Written out this is ``-i omega sqrt(eps0 c A_eff / 2) / (2 eps0 c n_eff)``.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The mode; supplies the effective index and the effective area.
    grid : solprop.grid.TimeGrid
        The grid.
    z : float, optional
        Position along the waveguide, m.
    shock : bool, optional
        Keep the ``omega`` dependence of the numerator. Default ``True``.
    index : float or None, optional
        Replace the effective index by this constant; the GNLSE passes 1.0.

    Returns
    -------
    numpy.ndarray
        The prefactor, complex, shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If the effective index or the effective area is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.nonlinear.normalisation import (
    ...     mode_average_prefactor,
    ...     sqrtw_to_vm,
    ...     uppe_prefactor,
    ... )
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor([1.0], omega_ref=grid.omega0)
    >>> composed = mode_average_prefactor(mode, grid, index=1.0)
    >>> plain = uppe_prefactor(mode, grid, index=1.0) / sqrtw_to_vm(mode.aeff())
    >>> bool(np.array_equal(composed, plain))
    True
    """
    return uppe_prefactor(mode, grid, z, shock=shock, index=index) / sqrtw_to_vm(
        mode.aeff(z)
    )


def modal_kerr_coefficient(
    electric: Complex[np.ndarray, "n_nodes_electric n_components_electric"],
    weights: Float[np.ndarray, " n_nodes_weights"],
    chi3: ArrayLike | float,
    omega: float,
    *,
    metric: ArrayLike | float | None = None,
    channel: int | None = None,
) -> float:
    r"""Return the self-phase-modulation coefficient of one vector channel.

    .. math::
       \gamma = \frac{\omega\varepsilon_0}{16 W}\int\chi^{(3)}
       \left(2|\mathbf E|^4 + |\mathbf E\cdot\mathbf E|^2\right)\,dA

    with the three-component field of a channel normalised to unit power
    (``docs/guide/waveguides/vector-fields.md``) and ``W`` its sampled longitudinal
    flux, one by construction and by default. The instantaneous isotropic Kerr
    polarisation of
    an analytic signal is ``P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4``, and the
    modal source functional of a bound reciprocal basis is
    ``da/dz = -i omega W^-1 integral(E* . P) dA / 4``
    (``docs/guide/waveguides/general-modal.md``).
    Putting the first into the second at unit amplitude leaves ``-i gamma``, which is
    this expression: both the longitudinal field and the ``|E.E|^2`` term of a vector
    channel are in it, and neither is in the scalar ``n_2 omega / (c A_eff)``.

    **The metric is the caller's to supply, and the identity is the default.**
    Nothing here measures it: the field is taken to be a channel normalised to unit
    longitudinal Poynting flux over its whole physical domain, and the integral is
    evaluated on the caller's weights. A rule that truncates or under-resolves the
    channel carries less than unit flux **where it is sampled**, and a general
    modal run divides by that sampled flux -- the ``W^-1`` inside the route's
    projection map -- rather than by one. Passing `metric` makes this coefficient
    the same statement as that run's, to rounding, instead of one that differs by
    the basis's departure from orthonormal. A
    table built by :func:`solprop.translate.modal_tables.sample_modal_table` gates
    that departure at ``DEFAULT_GRAM_ATOL`` and reports it, and a caller assembling
    a rule by hand checks it itself.

    **It is the metric the route's own contraction shares, not a scalar flux.**
    The route projects with ``W^-1``, so the coefficient of one channel ``m``
    carries ``[W^-1]_mm`` -- which is ``1/W_mm`` only when ``W`` is diagonal. Give
    the whole Hermitian metric of a degenerate block, with `channel` saying which
    row this field is, and the inverse is taken once here; give a positive float
    and it is read as that channel's own diagonal entry ``W_mm``, which is the
    diagonal statement.
    Only the channel's own entry enters either way, because only one channel's
    field is supplied: one channel's polarisation also drives its partners, no
    single coefficient describes that, and the route's own projection map is then
    the only complete statement.

    Parameters
    ----------
    electric : numpy.ndarray
        One channel's Cartesian E in V/m per sqrt(W), shape (n_nodes, 3), in the
        internal convention and normalised over the whole physical domain. Its two
        axes carry independent names, so that the agreement with `weights` and the
        three components are checked here, in production, and not only under the
        test session's runtime hook.
    weights : numpy.ndarray
        Area weights in m², shape (n_nodes,), of the rule the field was sampled on.
    chi3 : array_like or float
        Third-order susceptibility in m²/V² at each node, shape (n_nodes,), or one
        value for every node. It is zero where a region does not respond, which is
        how a linear cladding is excluded; the caller owns that assignment.
    omega : float
        Positive angular frequency in rad/s.
    metric : array_like, float or None, optional
        The sampled Hermitian flux metric of this channel's degenerate block
        (:func:`solprop.modes.vector.flux_product`), dimensionless: a square
        ``(n_modes, n_modes)`` array, of which ``[W^-1]`` at `channel` is used, or
        a positive float read as this channel's own diagonal entry ``W_mm``.
        ``None``, the default, assumes the identity the normalisation declares.
    channel : int or None, optional
        Which row of a square `metric` this field is, required when the metric has
        more than one row and refused when `metric` is not a matrix.

    Returns
    -------
    float
        ``gamma`` in 1/(W m), non-negative for a non-negative susceptibility.

    Raises
    ------
    ModeError
        If the shapes disagree, the frequency is not positive, the metric is not a
        positive scalar or a Hermitian positive-definite matrix, `channel` is
        missing, out of range or given without a matrix, or the field, the weights
        or the susceptibility is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.normalisation import modal_kerr_coefficient
    >>> field = np.array([[1.0e6 + 0j, 0j, 0j]])
    >>> f"{modal_kerr_coefficient(field, np.array([1e-12]), 1e-22, 2.0e15):.6e}"
    '3.320320e-07'
    >>> half = modal_kerr_coefficient(
    ...     field, np.array([1e-12]), 1e-22, 2.0e15, metric=2.0
    ... )
    >>> f"{half:.6e}"
    '1.660160e-07'
    >>> dense = modal_kerr_coefficient(
    ...     field, np.array([1e-12]), 1e-22, 2.0e15,
    ...     metric=np.array([[2.0, 0.0], [0.0, 1.0]]), channel=0,
    ... )
    >>> f"{dense:.6e}"
    '1.660160e-07'
    """
    inverse_entry = _metric_entry(metric, channel)
    values = np.asarray(electric, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or area.shape != values.shape[:1]:
        raise ModeError(
            "a modal Kerr coefficient needs one three-component field per weight, "
            f"got {values.shape} against {area.shape}"
        )
    susceptibility = np.broadcast_to(np.asarray(chi3, dtype=np.float64), area.shape)
    if not (
        math.isfinite(omega)
        and omega > 0.0
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(area))
        and np.all(np.isfinite(susceptibility))
    ):
        raise ModeError(
            "the frequency must be positive and the field, weights and "
            "susceptibility finite"
        )
    intensity = np.sum(abs(values) ** 2, axis=1)
    invariant = 2.0 * intensity**2 + abs(np.sum(values**2, axis=1)) ** 2
    coefficient = float(omega * EPS0 * np.sum(area * susceptibility * invariant) / 16.0)
    return coefficient * inverse_entry


def _metric_entry(metric: ArrayLike | float | None, channel: int | None) -> float:
    r"""Return ``[W^-1]_mm`` for one channel of a declared flux metric.

    Parameters
    ----------
    metric : array_like, float or None
        The square Hermitian metric, one channel's diagonal entry ``W_mm``, or
        ``None`` for the identity.
    channel : int or None
        The row a square metric is read at.

    Returns
    -------
    float
        The diagonal entry of ``W^-1`` this channel's coefficient is scaled by:
        one for the identity, ``1/W_mm`` for a scalar.

    Raises
    ------
    ModeError
        If the metric is not a positive scalar or a Hermitian positive-definite
        matrix, or `channel` is missing, out of range or given without a matrix.
    """
    if metric is None:
        if channel not in (None, 0):
            msg = (
                f"channel={channel!r} names a row of a flux metric and none was "
                "given; the identity metric has one channel"
            )
            raise ModeError(msg)
        return 1.0
    values = np.asarray(metric, dtype=np.complex128)
    if values.ndim == 0:
        scalar = float(np.real(values))
        if channel not in (None, 0):
            msg = (
                f"channel={channel!r} names a row of a square flux metric and a "
                "scalar was given, which is already this channel's own W_mm"
            )
            raise ModeError(msg)
        if not (math.isfinite(scalar) and scalar > 0.0 and np.imag(values) == 0.0):
            msg = (
                f"a scalar flux metric is this channel's positive W_mm, got {metric!r}"
            )
            raise ModeError(msg)
        return 1.0 / scalar
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.shape[0]:
        msg = (
            "a flux metric is a square (n_modes, n_modes) Hermitian matrix or one "
            f"channel's positive W_mm, got shape {values.shape}"
        )
        raise ModeError(msg)
    rows = int(values.shape[0])
    index = 0 if channel is None and rows == 1 else channel
    if index is None or not 0 <= index < rows:
        msg = (
            f"channel= says which of the metric's {rows} rows this field is, and "
            f"got {channel!r}"
        )
        raise ModeError(msg)
    scale = float(np.max(abs(values)))
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, values.conj().T, rtol=0.0, atol=_HERMITIAN_ATOL * scale
    ):
        msg = (
            "a flux metric is the Hermitian pairing of a basis with itself and "
            "this one is not Hermitian to "
            f"{_HERMITIAN_ATOL:.1e} of its largest entry"
        )
        raise ModeError(msg)
    eigenvalues = np.linalg.eigvalsh(values)
    if float(eigenvalues[0]) <= 0.0:
        msg = (
            f"a flux metric is positive definite and this one's smallest "
            f"eigenvalue is {float(eigenvalues[0]):.6e}: a basis with a null or "
            "negative-power direction carries no Kerr coefficient"
        )
        raise ModeError(msg)
    return float(np.real(np.linalg.inv(values)[index, index]))

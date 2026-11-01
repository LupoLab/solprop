# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assembly of the linear operator ``D(omega, z)``.

The linear operator carries the dispersion and the loss of one mode in a co-moving
frame, plus any spectral absorber expressed as a rate:

``D = -i (beta - beta_0 - beta_1 (omega - omega_ref)) - alpha/2 - alpha_abs/2``

with ``beta = (omega/c) Re n_eff`` and ``alpha = 2 (omega/c) Im n_eff``, in the internal
FFT convention of ``docs/dev/conventions.md`` section 2. It is diagonal in ``omega``,
which is the property the whole solver rests on; this module only evaluates it at one
``z``, and the translation layer turns those samples into the cumulative operator
``Phi(omega, z)``, piecewise Chebyshev in ``z`` (see ``docs/dev/decisions.md``).

The frame's ``beta_1`` comes from the mode's analytic ``dneff_domega`` rather than
from a finite-difference stencil, and the loss limit is a stated policy derived from the
fibre length rather than empirical constants. The narrative, including which
reference each term uses on each grid, is in ``docs/guide/numerics/linear-operator.md``.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float

from solprop.constants import C
from solprop.exceptions import GridError, ModeError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode

__all__ = [
    "DEFAULT_LOSS_POLICY",
    "MAX_ATTENUATION_EXPONENT",
    "Frame",
    "LossPolicy",
    "dispersion_operator",
    "frame_origin",
    "frame_quantities",
    "log_power_weight",
    "power_weight",
]

logger = logging.getLogger(__name__)


def power_weight(
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    z: float = 0.0,
    *,
    index: float | None = None,
) -> Float[np.ndarray, " n_omega"]:
    """Return the modal power weight ``w = Re n_eff``.

    The one place the weight is evaluated and validated: the interface amplitude is
    ``b = sqrt(w) a``, so a weight that is zero, negative or non-finite would make the
    conversion meaningless rather than merely inaccurate, and the refusal therefore
    belongs to the weight itself rather than to each of its callers.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        Physical mode; unused when the index override is supplied.
    omega : numpy.ndarray
        Positive angular frequencies in rad/s, shape ``(n_omega,)``. The caller
        selects the active band, as for the nonlinear prefactor.
    z : float, optional
        Position along the waveguide in m.
    index : float or None, optional
        Positive finite index override. The GNLSE mapping supplies 1.0.

    Returns
    -------
    numpy.ndarray
        Dimensionless ``Re n_eff`` of shape ``(n_omega,)``, positive and finite.

    Raises
    ------
    ModeError
        If the power weight is not positive and finite. A mode specified by its
        dispersion alone has no physical effective index -- a Taylor ``beta`` gives
        ``Re n_eff = beta c / omega``, which vanishes at the reference when
        ``beta(omega_ref) = 0`` -- so the message names the override.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.linear import power_weight
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.constants import C
    >>> mode = SpecifiedMode.from_taylor(
    ...     [1.45 * 1.8e15 / C], omega_ref=1.8e15
    ... )
    >>> float(power_weight(mode, np.array([1.8e15]), index=1.0)[0])
    1.0
    """
    weight = (
        np.real(np.asarray(mode.neff(omega, z)))
        if index is None
        else np.full(omega.shape, index)
    )
    if np.any(~np.isfinite(weight) | (weight <= 0.0)):
        msg = (
            f"the modal power weight must be positive and finite at z = {z!r} m; "
            "a mode without a physical effective index needs prefactor_index "
            "(the GNLSE interface passes 1.0)"
        )
        raise ModeError(msg)
    return weight


def log_power_weight(
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    z: float = 0.0,
    *,
    index: float | None = None,
) -> Float[np.ndarray, " n_omega"]:
    """Return the logarithm of the modal power weight.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        Physical mode; unused when the index override is supplied.
    omega : numpy.ndarray
        Positive angular frequencies in rad/s, shape ``(n_omega,)``. The caller
        selects the active band, as for the nonlinear prefactor.
    z : float, optional
        Position along the waveguide in m.
    index : float or None, optional
        Positive finite index override. The GNLSE mapping supplies 1.0.

    Returns
    -------
    numpy.ndarray
        Dimensionless ``ln(Re n_eff)`` of shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If the power weight is not positive and finite.
    """
    return np.log(power_weight(mode, omega, z, index=index))


MAX_ATTENUATION_EXPONENT = 708.0
"""Largest ``alpha z / 2`` worth representing: ``exp(-708)`` is still a normal double.

Beyond it the transmitted amplitude has underflowed, so no clamp placed at or above
this exponent can change any answer (see :class:`LossPolicy`).
"""


@dataclass(frozen=True)
class LossPolicy:
    """What to do with a loss too large to represent.

    An effective index can produce an arbitrarily large ``alpha`` — a capillary bin
    below cutoff, a table evaluated where it should not be — and two things then go
    wrong: ``exp(-alpha z / 2)`` underflows, and the adaptive Chebyshev builder spends
    intervals resolving a bin whose field is zero. The answer here is a limit derived
    from the fibre itself rather than empirical constants.

    The default limit is ``2 * MAX_ATTENUATION_EXPONENT / length``: the loss at which
    the field has already underflowed over the *whole* fibre, so clamping there cannot
    change the field at the fibre's end. A clamped bin is less attenuated than its true
    loss at a plane near the entrance. There is no lower limit, so a negative ``alpha`` — gain — passes
    through rather than being clamped to zero.

    Whatever the limit, it is applied to ``D`` before ``Phi`` is integrated, never to
    an accumulated exponent afterwards (``docs/guide/numerics/linear-operator.md``).

    Attributes
    ----------
    max_alpha : float or None, optional
        Largest power attenuation in 1/m, positive and finite. ``None``, the default,
        derives it from the fibre length. Pass, say, 3000.0 for a fixed limit
        independent of the length.

    Examples
    --------
    >>> from solprop.linear import LossPolicy
    >>> f"{LossPolicy().limit(1.5):.3f}"
    '944.000'
    >>> LossPolicy(max_alpha=3000.0).limit(1.5)
    3000.0
    """

    max_alpha: float | None = None

    def __post_init__(self) -> None:
        """Check the limit.

        Raises
        ------
        GridError
            If ``max_alpha`` is given but is not positive and finite.
        """
        if self.max_alpha is None:
            return
        if not (math.isfinite(self.max_alpha) and self.max_alpha > 0.0):
            msg = f"max_alpha must be positive and finite, got {self.max_alpha!r}"
            raise GridError(msg)

    def limit(self, length: float) -> float:
        """Return the largest attenuation this policy allows over a fibre.

        Parameters
        ----------
        length : float
            Propagation distance the operator will be integrated over, m, positive.

        Returns
        -------
        float
            The limit in 1/m.

        Raises
        ------
        GridError
            If ``length`` is not positive and finite.

        Examples
        --------
        >>> from solprop.linear import LossPolicy
        >>> f"{LossPolicy().limit(0.1):.1f}"
        '14160.0'
        """
        if not (math.isfinite(length) and length > 0.0):
            msg = f"length must be positive and finite, got {length!r}"
            raise GridError(msg)
        if self.max_alpha is not None:
            return self.max_alpha
        return 2.0 * MAX_ATTENUATION_EXPONENT / length


DEFAULT_LOSS_POLICY = LossPolicy()
"""The length-derived loss limit, used when a caller states no policy."""


@dataclass(frozen=True)
class Frame:
    """The reference the linear operator is written in.

    The frame removes from ``beta(omega)`` a constant ``beta_0`` and a linear
    ``beta_1 (omega - omega_ref)``. The linear term is a change of time origin — it
    keeps the pulse in the window on a gradient fibre — and the constant term
    is a change of carrier phase.

    **Which terms apply depends on the grid**, because the two grids store different
    things:

    ===================  =========================================================
    grid                 operator
    ===================  =========================================================
    field                ``-i (beta - beta_1 omega) - alpha/2``
    envelope, no THG     ``-i (beta - beta_0 - beta_1 (omega - omega_ref)) - alpha/2``
    envelope, with THG   ``-i (beta - beta_1 omega) - alpha/2``
    ===================  =========================================================

    A field grid stores the real field, so subtracting ``beta_0`` would rotate its
    carrier-envelope phase, which is physical rather than a global phase; and its frame
    term is measured from zero frequency, because a pure delay is a phase linear in
    ``omega`` through the origin. An envelope grid has the carrier removed already, so
    ``beta_0`` may be taken out — except when a third-harmonic term is present, whose
    ``exp(2 i omega_ref t)`` factor is tied to the true carrier phase. Such a frame
    keeps the carrier phase whole, so its delay is also measured from zero frequency:
    ``beta_1 (omega - omega_ref)`` would still remove a constant ``-beta_1 omega_ref``,
    a carrier phase of ``beta_1 omega_ref z`` that the cubic term does not share (see
    :func:`frame_origin`). ``conventions.md`` section 2 writes the envelope
    form for both grids. :meth:`comoving` chooses it for you.

    Attributes
    ----------
    omega0 : float
        Reference angular frequency, rad/s, positive: where ``beta_0`` and ``beta_1``
        are evaluated and, on an envelope grid that removes ``beta_0``, the origin of
        the frame term.
    subtract_beta0 : bool
        Whether the constant phase is removed. ``False`` keeps the whole carrier
        phase, so the frame term is then measured from zero frequency on either grid.
    beta0 : float or None, optional
        Fixed ``beta_0`` in 1/m. ``None``, the default, recomputes it from the mode at
        each ``z``, which is what makes the frame co-moving on a gradient fibre.
    beta1 : float or None, optional
        Fixed ``beta_1`` in s/m. ``None``, the default, recomputes it from the mode.

    Notes
    -----
    A frame belongs to a *problem*, not to a mode: a multimode operator is written in
    one frame for the whole set, because ``beta_1 omega`` is a change of the time
    variable and modes given their own would no longer be simultaneous. Handed a set,
    :func:`solprop.translate.linop.build_phi` therefore fixes both quantities on the
    first mode at each position and gives that frame to every column.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid, FieldGrid
    >>> from solprop.linear import Frame
    >>> limits = (400e-9, 2e-6)
    >>> field = FieldGrid.from_limits(1.0, 835e-9, limits, 20e-12)
    >>> Frame.comoving(field).subtract_beta0
    False
    >>> envelope = EnvelopeGrid.from_limits(1.0, 835e-9, limits, 20e-12)
    >>> Frame.comoving(envelope).subtract_beta0
    True
    >>> Frame.comoving(envelope, thg=True).subtract_beta0
    False
    """

    omega0: float
    subtract_beta0: bool
    beta0: float | None = None
    beta1: float | None = None

    def __post_init__(self) -> None:
        """Check the reference frequency.

        Raises
        ------
        GridError
            If ``omega0`` is not positive and finite.
        """
        if not (math.isfinite(self.omega0) and self.omega0 > 0.0):
            msg = f"omega0 must be positive and finite, got {self.omega0!r}"
            raise GridError(msg)

    @classmethod
    def comoving(
        cls,
        grid: TimeGrid,
        *,
        thg: bool = False,
        # Quoted because beartype resolves the annotation while the class body is
        # still executing, when the bare class name is not yet bound.
    ) -> "Frame":  # noqa: UP037
        """Build the frame a grid asks for, with both quantities taken from the mode.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            The grid the operator will be evaluated on; its ``omega0`` becomes the
            reference and its ``kind`` decides whether ``beta_0`` is removed.
        thg : bool, optional
            Whether a third-harmonic term is present. On an envelope grid this keeps
            ``beta_0``; on a field grid it changes nothing. Default ``False``.

        Returns
        -------
        Frame
            A co-moving frame: both ``beta_0`` and ``beta_1`` follow the mode along z.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.linear import Frame
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 20e-12)
        >>> frame = Frame.comoving(grid)
        >>> frame.omega0 == grid.omega0, frame.beta1 is None
        (True, True)
        """
        return cls(
            omega0=grid.omega0, subtract_beta0=grid.kind == "envelope" and not thg
        )


def frame_origin(frame: Frame, kind: str) -> float:
    """Return where the frame's delay term ``beta_1 (omega - origin)`` starts.

    A frame either removes the carrier phase or keeps it whole. One that removes it —
    an envelope grid without a third harmonic — measures the delay from its carrier,
    ``omega0``, alongside ``beta_0``. One that keeps it — a field grid, or an envelope
    grid with ``subtract_beta0=False`` — measures the delay from zero frequency, so the
    frame is a pure change of time origin. Measured from ``omega0`` instead, the delay
    term would also remove the constant ``-beta_1 omega0``: the envelope would turn at
    ``beta_1 omega0`` per metre, about 1e7 rad/m in a gas at 1 um, while
    ``A**3 exp(2 i omega0 t)`` turns at three times that, so the third harmonic would be
    driven with a spurious mismatch of ``2 beta_1 omega0``.

    Parameters
    ----------
    frame : Frame
        The frame.
    kind : {'field', 'envelope'}
        The spectral layout of the grid the operator is evaluated on.

    Returns
    -------
    float
        ``frame.omega0`` on an envelope grid whose frame removes ``beta_0``, else 0.0.

    Examples
    --------
    >>> from solprop.linear import Frame, frame_origin
    >>> frame_origin(Frame(1.8e15, subtract_beta0=True), "envelope")
    1800000000000000.0
    >>> frame_origin(Frame(1.8e15, subtract_beta0=False), "envelope")
    0.0
    """
    return frame.omega0 if kind == "envelope" and frame.subtract_beta0 else 0.0


def frame_quantities(mode: Mode, frame: Frame, z: float = 0.0) -> tuple[float, float]:
    """Return the ``(beta_0, beta_1)`` the frame removes from the operator.

    ``beta_0 = omega_ref Re n_eff / c`` and, differentiating ``beta = omega n_eff / c``,
    ``beta_1 = (Re n_eff + omega_ref Re dn_eff/domega) / c``. The derivative is the
    mode's analytic one: a differenced ``beta_1`` has a roughness floor near 5e-8 that
    would cap the whole cumulative operator at 1e-8, and costs about seven ``neff``
    evaluations per stage.

    Only the real part of the index enters, so the loss never affects the frame.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The mode whose dispersion the frame follows.
    frame : Frame
        The frame; its fixed values, where given, override the mode's.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    tuple of float
        ``(beta_0, beta_1)`` in 1/m and s/m, as they enter ``D``. ``beta_0`` is zero
        when the frame does not subtract it, so that these are the quantities the
        kernel contract records as already being inside ``Phi``.

    Examples
    --------
    >>> from solprop.constants import C
    >>> from solprop.linear import Frame, frame_quantities
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> mode = SpecifiedMode.from_taylor(
    ...     [1.45 * 1.8e15 / C, 1.47 / C], omega_ref=1.8e15
    ... )
    >>> beta0, beta1 = frame_quantities(mode, Frame(1.8e15, subtract_beta0=True))
    >>> f"{beta0:.6e}", f"{beta1 * C:.6f}"
    ('8.706023e+06', '1.470000')
    """
    reference = np.array([frame.omega0])
    beta0 = frame.beta0
    beta1 = frame.beta1
    if beta0 is None or beta1 is None:
        index = np.asarray(mode.neff(reference, z))
        if beta0 is None:
            beta0 = float(frame.omega0 * np.real(index)[0] / C)
        if beta1 is None:
            derivative = np.asarray(mode.dneff_domega(reference, z))
            beta1 = float(
                (np.real(index)[0] + frame.omega0 * np.real(derivative)[0]) / C
            )
    return (beta0 if frame.subtract_beta0 else 0.0), beta1


def _check_absorber(
    absorber_rate: np.ndarray, grid: TimeGrid
) -> Float[np.ndarray, " n_omega"]:
    """Validate a spectral absorber rate against the grid it belongs to.

    Parameters
    ----------
    absorber_rate : numpy.ndarray
        The rate the caller supplied.
    grid : solprop.grid.TimeGrid
        The grid it must match.

    Returns
    -------
    numpy.ndarray
        The rate as float64, shape ``(n_omega,)``.

    Raises
    ------
    GridError
        If it is the wrong length, not finite, or negative anywhere. An absorber
        removes energy; a negative rate is gain wearing a boundary's clothes.
    """
    values = np.asarray(absorber_rate, dtype=float)
    if values.shape != (grid.n_omega,):
        msg = f"absorber_rate has shape {values.shape}, expected {(grid.n_omega,)}"
        raise GridError(msg)
    if not np.all(np.isfinite(values)):
        msg = "absorber_rate must be finite everywhere"
        raise GridError(msg)
    if np.any(values < 0.0):
        msg = f"absorber_rate must be non-negative; smallest is {values.min():.6e}"
        raise GridError(msg)
    return values


def dispersion_operator(
    mode: Mode,
    grid: TimeGrid,
    frame: Frame,
    z: float = 0.0,
    *,
    loss: LossPolicy = DEFAULT_LOSS_POLICY,
    absorber_rate: Float[np.ndarray, " n_omega"] | None = None,
) -> Complex[np.ndarray, " n_omega"]:
    """Return the linear operator on the coarse frequency grid at one position.

    The mode is evaluated on the grid's **active band** only
    (``omega_min/2 < omega < 1.1 omega_max``). Outside it the operator
    is zero, because an effective index is not defined at zero frequency and the state
    there is removed by the hard band mask anyway (see ``docs/dev/decisions.md``). The
    absorber rate, in contrast, is added over the **whole** axis: with the collar
    expressed as a rate, out-of-band bins must carry it rather than propagate
    loss-free.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The mode. Only ``neff`` and ``dneff_domega`` are used.
    grid : solprop.grid.TimeGrid
        The grid; supplies ``omega``, the active band, the layout and the length the
        loss policy is derived from.
    frame : Frame
        The co-moving frame, usually from :meth:`Frame.comoving`.
    z : float, optional
        Position along the waveguide, m.
    loss : LossPolicy, optional
        What to do with an attenuation too large to represent. Defaults to the
        length-derived limit.
    absorber_rate : numpy.ndarray or None, optional
        Spectral absorber rate ``alpha_abs(omega)`` in 1/m, shape ``(n_omega,)``,
        non-negative. :mod:`solprop.boundaries` produces one from the grid's
        collar; ``None`` means no absorber.

    Returns
    -------
    numpy.ndarray
        ``D(omega, z)`` in 1/m, complex, shape ``(n_omega,)``.

    Raises
    ------
    GridError
        If the frame's reference lies outside the grid, or the absorber rate is
        invalid.
    ModeError
        If the mode returns a non-finite effective index anywhere in the active band,
        rather than letting a NaN through.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import C
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.linear import Frame, dispersion_operator
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [1.45 * grid.omega0 / C, 1.47 / C, 2e-26], omega_ref=grid.omega0
    ... )
    >>> operator = dispersion_operator(mode, grid, Frame.comoving(grid))
    >>> operator.shape == (grid.n_omega,)
    True
    >>> bool(np.allclose(operator[np.argmin(abs(grid.omega - grid.omega0))], 0.0))
    True
    """
    if not grid.omega.min() <= frame.omega0 <= grid.omega.max():
        msg = (
            f"the frame's omega0 = {frame.omega0:.6e} rad/s lies outside the grid, "
            f"which runs from {grid.omega.min():.6e} to {grid.omega.max():.6e} rad/s"
        )
        raise GridError(msg)

    active = grid.active_mask
    omega = grid.omega[active]
    index = np.asarray(mode.neff(omega, z))
    if not np.all(np.isfinite(index)):
        msg = (
            f"the mode returned a non-finite effective index at z = {z!r} m for "
            f"{int(np.sum(~np.isfinite(index)))} of {omega.size} active bins"
        )
        raise ModeError(msg)

    beta = omega * np.real(index) / C
    attenuation = 2.0 * omega * np.imag(index) / C
    limit = loss.limit(grid.length)
    clamped = np.minimum(attenuation, limit)

    beta0, beta1 = frame_quantities(mode, frame, z)
    # A frame that keeps the carrier phase is a pure change of time origin: a phase
    # linear in omega through zero. One that removes it measures the delay from the
    # carrier. See `frame_origin`.
    origin = frame_origin(frame, grid.kind)

    operator = np.zeros(grid.n_omega, dtype=complex)
    operator[active] = -1j * (beta - beta0 - beta1 * (omega - origin)) - 0.5 * clamped
    if absorber_rate is not None:
        operator = operator - 0.5 * _check_absorber(absorber_rate, grid)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "D at z = %.6g m: %d of %d bins active, beta0 = %.6e 1/m, "
            "beta1 = %.6e s/m, alpha up to %.6e 1/m (limit %.6e), %d clamped",
            z,
            int(np.sum(active)),
            grid.n_omega,
            beta0,
            beta1,
            float(attenuation.max(initial=0.0)),
            limit,
            int(np.sum(attenuation > limit)),
        )
    return operator

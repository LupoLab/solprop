# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Absorbing boundaries as absorption rates per unit distance.

Both axes of the grid are finite and periodic, so energy that reaches the edge of the
frequency or the time window does not leave the simulation -- it wraps around, and the
answer is wrong. The collars of :mod:`solprop.grid` mark where the propagation is
absorbed instead of modelled; this module turns them into rates.

Multiplying the solution by the collar profile ``W`` once per accepted step would make
the total absorption ``W^n`` for however many steps the adaptive controller happened to
take. The answer would then depend on ``rtol``: refining the tolerance sixteen-fold
erodes the spectral wings fifty-fold, and the limit it approaches is not the taper the
profile describes but a brick wall at the edge of the flat region. As a rate it is
exact instead, because traversing a distance attenuates by a factor that depends only on
the distance. The spectral collar becomes

.. math::

    \alpha(\omega) = -\frac{2 \log W(\omega)}{\ell},

the rate that reproduces one multiplication by the window over the reference length,
and the temporal collar becomes the bounded profile

.. math::

    \alpha(t) = \frac{A\,(1 - W(t))}{\ell}, \qquad A = 12.6,

whose integral over the collar is what the capped logarithm gave and whose largest
value is a fifth of the cap (:data:`TEMPORAL_ALPHA_LENGTH`).

``alpha`` is a **power** coefficient, as everywhere else in solprop: power falls as
``exp(-alpha z)``, the field as ``exp(-alpha z / 2)``, and the linear operator carries
``-alpha/2`` (``docs/dev/conventions.md`` section 2). The factor of two in the spectral
definition is what makes ``exp(-alpha ell / 2) = W`` -- one reference length applies the
historical profile to the *field* exactly once, which is how the profile was used. The
reference length is ``ell = length / boundary_N`` by default: apply the window
``boundary_N`` times over the whole propagation.

The two rates are applied by different mechanisms, each chosen because it is exact
(see ``docs/dev/decisions.md``):

* ``alpha(omega)`` is diagonal in omega, so it is folded into the linear operator before
  the cumulative operator is integrated. The propagator then applies it exactly over
  whatever sub-interval the stepper takes, at no cost.
* ``alpha(t)`` is diagonal in t and cannot ride the propagator. It is applied as the
  term ``-alpha(t) E(t) / 2`` inside every stage evaluation
  (:func:`nonlinear_absorber_term`), the default because the alternative -- the
  split-step factor ``exp(-alpha(t) h / 2)`` after each accepted step
  (:func:`split_step_factor`), exact for the absorber alone because the factors
  telescope -- was measured to be a first-order splitting of the whole. Both are here.
* The hard band limit outside the active mask stays a 0/1 mask
  (:func:`apply_band_mask`), which is idempotent and never had the problem.

Two lengths set the collars' strengths (SEP 22 section 4.6). ``reference_length``
sets the spectral rate and the ceiling every spectral rate is validated against;
``collar_length``, ``reference_length`` unless given, sets the in-stage collars, the
temporal ``12.6 (1 - W(t)) / ell_c`` and the spatial ``12.6 (1 - W(x, y)) / ell_c``, and
their ceiling. A custom spectral rate (:func:`custom_absorber`) can then be as strong as
a measurement needs without moving the explicit in-stage collars or their stable step.

The manual page is ``docs/guide/numerics/boundaries.md``.
"""

import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from jaxtyping import Bool, Float, Inexact

from solprop.exceptions import GridError, SolpropWarning
from solprop.grid import Collar, FreeGrid, RadialGrid, TimeGrid

__all__ = [
    "DEFAULT_BOUNDARY_N",
    "MAX_ALPHA_LENGTH",
    "MIN_COLLAR_ATTENUATION",
    "MIN_COLLAR_SAMPLES",
    "STABILITY_RADIUS",
    "TEMPORAL_ALPHA_LENGTH",
    "Absorber",
    "TransverseCollar",
    "absorber",
    "apply_band_mask",
    "apply_split_step",
    "collar_rate",
    "custom_absorber",
    "no_absorber",
    "nonlinear_absorber_term",
    "split_step_factor",
    "stable_step",
    "temporal_collar_rate",
    "transverse_collar_rate",
    "warn_walkoff",
]

logger = logging.getLogger(__name__)

DEFAULT_BOUNDARY_N: Final = 20
"""Default absorber strength: applications of the collar profile over the propagation.

It sets the reference length ``ell = length / boundary_N``, so the strength is a
physical distance rather than a number of steps.
"""

MAX_ALPHA_LENGTH: Final = 60.0
"""Cap on the spectral ``alpha * ell``, and the ceiling every rate is validated against.

The spectral collar reaches exactly zero at the band edge and ``-2 log 0`` is an
infinity, which would make the operator ``inf * 0`` somewhere and hence a NaN. At 60
the field is attenuated by ``exp(-30)``, about 1e-13, over one reference length -- far
below anything a propagation can represent, so the cap costs nothing and keeps every
rate finite. The temporal rate is bounded by construction at
:data:`TEMPORAL_ALPHA_LENGTH`, well inside this ceiling: were it the same capped
logarithm, then under the in-stage absorber term the cap would set the stability limit
and the step count.
"""

TEMPORAL_ALPHA_LENGTH: Final = 12.6
"""Depth of the temporal absorber at the edge of the time window, as ``alpha * ell``.

The temporal rate is ``TEMPORAL_ALPHA_LENGTH * (1 - W(t)) / ell``, the complement of
the collar window (:func:`temporal_collar_rate`). At 12.6 its integral over the collar
is what ``-2 log W`` capped at :data:`MAX_ALPHA_LENGTH` gave -- 6.3 per collar width
per reference length, both edges, to a few per cent on the grids solprop builds -- so
the walk-off protection and the meaning of ``boundary_N`` are unchanged, while the
largest rate is a fifth of the capped logarithm's: the stability limit of the in-stage
absorber term (:func:`stable_step`) is 4.8 times longer, and the error estimate does
not see a rate of ``60 / ell`` on the collar's last samples, which would cost the
reduced M2 case 40 % of its steps. The capped logarithm puts 73 % of each edge's
attenuation in its outer 14 %; this profile spreads it over the edge, so a case whose
field reaches the collar is absorbed earlier and answers differently. At the very edge
the field falls by ``exp(-6.3)``, 1.8e-3, per reference length.
"""

MIN_COLLAR_SAMPLES: Final = 4
"""Grid samples a collar edge must span before it counts as a taper.

With fewer the discrete profile steps from one to zero rather than tapering, the rate
jumps to its cap in a single sample, and the boundary is the brick wall the rate form
exists to avoid. A temporal collar built naively is exactly this degenerate for a
``trange`` that lands on a power of two; solprop's grids guarantee a minimum width, and
this is the guard that says so.
"""

MIN_COLLAR_ATTENUATION: Final = math.log(1e3)
"""Field attenuation, in nepers, a walking packet must accumulate crossing the collar.

``log(1e3)``: the field falls by a thousand and the power by sixty decibels. Below that
:func:`warn_walkoff` says the absorber is too weak for the walk-off the run can produce.
"""

STABILITY_RADIUS: Final = 3.3
"""How far DOPRI5's stability region reaches along the negative real axis.

The textbook value for both order-5(4) pairs this package carries, and what
:func:`stable_step` turns into a step limit for the in-stage absorber. Measured
on the 1 mm capillary of the driver tests, whose collar reached 1.2e6 m^-1 under the
capped logarithm temporal profile: a fixed schedule diverges at ``h alpha / 2 = 3.75``
and completes at 3.13, which brackets it.
"""

_MIN_SPACING_RATIO: Final = 1e-9
"""Smallest collar edge, as a fraction of the axis span, that is not degenerate."""


@dataclass(frozen=True)
class Absorber:
    """The absorbing boundaries of one grid, as rates per unit distance.

    Attributes
    ----------
    spectral_rate : numpy.ndarray
        Power absorption coefficient on the coarse frequency axis, 1/m, shape
        ``(n_omega,)``. Exactly zero in the flat region and outside the active mask,
        where the linear operator is zero anyway. Folded into the linear operator as
        ``-alpha/2``.
    temporal_rate : numpy.ndarray
        Power absorption coefficient on the coarse time axis, 1/m, shape ``(n_t,)``.
    temporal_rate_fine : numpy.ndarray
        The same on the fine time axis, 1/m, shape ``(n_to,)``, for a response that is
        evaluated there.
    band_mask : numpy.ndarray
        The hard band limit, shape ``(n_mask,)`` matching ``spectral_rate``: true where
        the physics lives. Applied as a 0/1 multiply, which is idempotent.
    reference_length : float
        The length ``ell`` in metres over which the collar profile is applied once to
        the field. It sets the spectral rate's ceiling, ``60 / ell``.
    collar_length : float or None
        The length ``ell_c`` in metres that sets the in-stage collars' strength: the
        temporal collar's largest rate is ``12.6 / ell_c``, and the spatial collar of
        a radial, Cartesian or planar run is ``12.6 (1 - W) / ell_c``; it also sets
        the temporal rates' ceiling, ``60 / ell_c``. ``None``, the default, means
        `reference_length`, as for every absorber before SEP 22; a value equal to
        `reference_length` is stored as ``None``, so that the record carries it only
        where the two differ. Left out of the run record at ``None``.
    temporal : bool
        Whether the collar has a temporal part. ``False`` for a spectral-only collar
        (:func:`absorber` with ``temporal=False``), whose temporal rates are zero
        arrays; it is recorded only when ``False``. Any collar whose temporal rates
        are all zero, however it was built, is translated to no temporal absorber
        (:func:`solprop.translate.masks.temporal_rates`).

    Raises
    ------
    GridError
        If a length is not positive and finite, the mask and the spectral rate
        disagree in shape, a rate is negative, not finite, or above its ceiling, or
        a collar declared spectral-only carries a temporal rate.

    Examples
    --------
    >>> from solprop.boundaries import absorber
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> boundary = absorber(grid)
    >>> boundary.reference_length
    0.075
    >>> float(boundary.spectral_rate[grid.inband_mask].max())
    0.0
    """

    spectral_rate: Float[np.ndarray, " n_omega"]
    temporal_rate: Float[np.ndarray, " n_t"]
    temporal_rate_fine: Float[np.ndarray, " n_to"]
    band_mask: Bool[np.ndarray, " n_mask"]
    reference_length: float
    collar_length: float | None = None
    temporal: bool = True

    def __post_init__(self) -> None:
        """Validate what the shape annotations cannot express.

        Raises
        ------
        GridError
            Naming the offending field.
        """
        _check_length("reference_length", self.reference_length)
        if self.collar_length is not None:
            _check_length("collar_length", self.collar_length)
            if self.collar_length == self.reference_length:
                # The record carries the collar length only where the two differ.
                object.__setattr__(self, "collar_length", None)
        if self.band_mask.shape != self.spectral_rate.shape:
            msg = (
                f"band_mask has shape {self.band_mask.shape} but spectral_rate has "
                f"{self.spectral_rate.shape}; both live on the coarse frequency axis"
            )
            raise GridError(msg)
        for name, length in (
            ("spectral_rate", self.reference_length),
            ("temporal_rate", self.in_stage_length),
            ("temporal_rate_fine", self.in_stage_length),
        ):
            ceiling = MAX_ALPHA_LENGTH / length
            rate: Float[np.ndarray, " n"] = getattr(self, name)
            if not np.all(np.isfinite(rate)):
                msg = f"{name} must be finite; the collar reached zero without the cap"
                raise GridError(msg)
            if np.any(rate < 0.0) or np.any(rate > ceiling * (1.0 + 1e-12)):
                msg = (
                    f"{name} must lie in [0, {ceiling:.6g}] 1/m; an absorber outside "
                    "that either amplifies or is not finite over its length"
                )
                raise GridError(msg)
        if not self.temporal and (
            np.any(self.temporal_rate) or np.any(self.temporal_rate_fine)
        ):
            msg = (
                "a spectral-only collar (temporal=False) carries no temporal rate; "
                "set the temporal rates to zero, or temporal=True"
            )
            raise GridError(msg)

    @property
    def in_stage_length(self) -> float:
        """Return the length that sets the in-stage collars' strength.

        Returns
        -------
        float
            `collar_length`, or `reference_length` when it is ``None``, m.

        Examples
        --------
        >>> from solprop.boundaries import absorber
        >>> from solprop.grid import FieldGrid
        >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
        >>> absorber(grid).in_stage_length
        0.075
        """
        return (
            self.reference_length if self.collar_length is None else self.collar_length
        )


def _check_length(name: str, length: float) -> None:
    """Refuse a collar length that is not positive and finite.

    Parameters
    ----------
    name : str
        The field's name, for the message.
    length : float
        The length, m.

    Raises
    ------
    GridError
        If the length is not positive and finite.
    """
    if not (math.isfinite(length) and length > 0.0):
        msg = (
            f"{name} must be positive and finite, got {length!r}; a zero length fills "
            "the grid with NaN and a negative one turns the absorber into a gain"
        )
        raise GridError(msg)


def collar_rate(
    window: Float[np.ndarray, " n"], reference_length: float
) -> Float[np.ndarray, " n"]:
    """Convert a collar profile into a power absorption coefficient by its logarithm.

    The spectral collar's rate; the temporal collar takes the bounded profile of
    :func:`temporal_collar_rate` instead.
    ``alpha = -2 log(W) / ell``, so that propagating through ``ell`` attenuates the
    field by exactly ``W``: the collar profiles were multiplied into the field, and this
    is the rate that reproduces one such multiplication over the reference length.
    Exactly zero where ``W == 1``, and capped at ``MAX_ALPHA_LENGTH / ell`` where the
    profile reaches zero.

    Parameters
    ----------
    window : numpy.ndarray
        Collar profile in ``[0, 1]``, one in the interior and falling to zero at the
        boundary, shape ``(n,)``.
    reference_length : float
        The length in metres over which the profile is applied once; positive, finite.

    Returns
    -------
    numpy.ndarray
        Power absorption coefficient in 1/m, shape ``(n,)``, non-negative and finite.

    Raises
    ------
    GridError
        If the reference length is not positive and finite, or the profile leaves
        ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import collar_rate
    >>> alpha = collar_rate(np.array([1.0, 0.5, 0.0]), 0.1)
    >>> np.round(alpha, 6).tolist()
    [0.0, 13.862944, 600.0]
    >>> float(np.exp(-alpha[1] * 0.1 / 2))
    0.5
    """
    if not (math.isfinite(reference_length) and reference_length > 0.0):
        msg = f"reference_length must be positive and finite, got {reference_length!r}"
        raise GridError(msg)
    profile = np.asarray(window, dtype=np.float64)
    if np.any(profile < 0.0) or np.any(profile > 1.0):
        msg = "a collar profile must lie in [0, 1]"
        raise GridError(msg)
    ceiling = MAX_ALPHA_LENGTH / reference_length
    rate = np.full(profile.shape, ceiling)
    # Evaluated only where the profile is positive: it is exactly zero outside the
    # window, and log(0) is an error in this package's test session, not a warning.
    positive = profile > 0.0
    # The trailing zero turns the negative zero that -2 log(1) produces into a positive
    # one, so that a flat collar reads as 0.0 rather than -0.0 wherever it is printed or
    # compared against a literal.
    rate[positive] = (
        np.clip(-2.0 * np.log(profile[positive]) / reference_length, 0.0, ceiling) + 0.0
    )
    return rate


def temporal_collar_rate(
    axis: Float[np.ndarray, " n"], collar: Collar, reference_length: float
) -> Float[np.ndarray, " n"]:
    """Convert the temporal collar into a bounded power absorption coefficient.

    ``alpha(t) = TEMPORAL_ALPHA_LENGTH * (1 - W(t)) / ell``: exactly zero over the flat
    region, rising with every derivative continuous as the Planck window falls, and
    reaching ``TEMPORAL_ALPHA_LENGTH / ell`` at the edge of the window. The logarithm of
    :func:`collar_rate` is the right rate for the spectral collar, which rides the
    linear operator for free; for the temporal collar it would be a stiff term in every
    stage of the in-stage absorber, its cap would set the stability limit and the step
    count, and it puts nearly three quarters of the attenuation in the outer seventh of
    each edge. This profile carries the same integrated attenuation with a largest rate
    a fifth of the cap.

    Parameters
    ----------
    axis : numpy.ndarray
        Time axis to evaluate the rate on, s, shape ``(n,)``.
    collar : solprop.grid.Collar
        The temporal collar geometry.
    reference_length : float
        The length ``ell`` in metres; positive, finite.

    Returns
    -------
    numpy.ndarray
        Power absorption coefficient in 1/m, shape ``(n,)``, in
        ``[0, TEMPORAL_ALPHA_LENGTH / ell]``.

    Raises
    ------
    GridError
        If the reference length is not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, temporal_collar_rate
    >>> from solprop.grid import Collar
    >>> collar = Collar(-2.0, -1.0, 1.0, 2.0)
    >>> rate = temporal_collar_rate(np.array([-2.0, 0.0, 2.0]), collar, 0.1)
    >>> rate.tolist() == [TEMPORAL_ALPHA_LENGTH / 0.1, 0.0, TEMPORAL_ALPHA_LENGTH / 0.1]
    True
    """
    if not (math.isfinite(reference_length) and reference_length > 0.0):
        msg = f"reference_length must be positive and finite, got {reference_length!r}"
        raise GridError(msg)
    window = collar.window(np.asarray(axis, dtype=np.float64))
    return TEMPORAL_ALPHA_LENGTH * (1.0 - window) / reference_length


def _reference_length(
    grid: TimeGrid, boundary_n: float, boundary_length: float | None
) -> float:
    """Return the absorber reference length in metres.

    Parameters
    ----------
    grid : TimeGrid
        The grid, for its propagation length.
    boundary_n : float
        Applications of the collar profile over the whole propagation.
    boundary_length : float or None
        An explicit reference length in metres, which wins if given.

    Returns
    -------
    float
        The reference length in metres.

    Raises
    ------
    GridError
        If the result would not be positive and finite. This is checked here so that
        every caller is covered: a zero length makes the rate ``0/0`` in the interior
        and fills the grid with NaN, and a negative one inverts the clamp so that the
        absorber amplifies everywhere rather than absorbing in the collar.
    """
    if boundary_length is not None:
        if not (math.isfinite(boundary_length) and boundary_length > 0.0):
            msg = (
                f"boundary_length must be positive and finite, got {boundary_length!r}"
            )
            raise GridError(msg)
        return float(boundary_length)
    if not (math.isfinite(boundary_n) and boundary_n > 0.0):
        msg = f"boundary_N must be positive and finite, got {boundary_n!r}"
        raise GridError(msg)
    return float(grid.length) / float(boundary_n)


def _check_collar(
    name: str, collar: Collar, axis: Float[np.ndarray, " n"], units: str
) -> None:
    """Reject a collar whose edges are too narrow to be a taper on this axis.

    Parameters
    ----------
    name : str
        The collar's name, for the message.
    collar : Collar
        The geometry to check.
    axis : numpy.ndarray
        The axis the collar is evaluated on, shape ``(n,)``.
    units : str
        The axis units, for the message.

    Raises
    ------
    GridError
        If either edge has non-positive width, or is resolved by fewer than
        ``MIN_COLLAR_SAMPLES`` samples.
    """
    span = float(np.max(axis) - np.min(axis))
    for edge, width, low, high in (
        ("rising", collar.width_low, collar.zero_low, collar.one_low),
        ("falling", collar.width_high, collar.one_high, collar.zero_high),
    ):
        if width <= _MIN_SPACING_RATIO * span:
            msg = (
                f"the {name} collar's {edge} edge has zero width in {units}: the "
                "profile steps rather than tapers, so the boundary is a brick wall "
                "rather than an absorber"
            )
            raise GridError(msg)
        samples = int(np.count_nonzero((axis > low) & (axis < high)))
        if samples < MIN_COLLAR_SAMPLES:
            msg = (
                f"the {name} collar's {edge} edge spans {samples} grid samples, fewer "
                f"than the {MIN_COLLAR_SAMPLES} a taper needs; widen the collar or the "
                "grid"
            )
            raise GridError(msg)


def absorber(
    grid: TimeGrid,
    *,
    boundary_N: float = DEFAULT_BOUNDARY_N,  # noqa: N803 - the established parameter name
    boundary_length: float | None = None,
    temporal: bool = True,
) -> Absorber:
    """Build the absorbing boundaries of a grid.

    Parameters
    ----------
    grid : TimeGrid
        The grid, for its collars, its masks and its propagation length.
    boundary_N : float, optional
        Absorber strength, as the number of times the collar profile is applied to the
        field over the whole propagation; the reference length is ``length /
        boundary_N``.
    boundary_length : float or None, optional
        Reference length in metres, overriding ``boundary_N``.
    temporal : bool, optional
        ``False`` builds a **spectral-only** collar (SEP 22): its temporal rates are
        zero arrays on both time axes, the band mask and the spectral rate are as
        before, and the record says ``temporal=False``. It is translated to no
        temporal absorber, so the kernels compile no temporal pass under either
        absorber scheme. The grid is unchanged.

    Returns
    -------
    Absorber
        The spectral and temporal rates, the hard band mask and the reference length.

    Raises
    ------
    GridError
        If the reference length is not positive and finite, or either collar is too
        narrow on its axis to be a taper.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import absorber
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> boundary = absorber(grid, boundary_N=20)
    >>> boundary.reference_length
    0.075
    >>> field_left = np.exp(-boundary.spectral_rate * boundary.reference_length / 2)
    >>> active = grid.active_mask
    >>> bool(np.allclose(field_left[active], grid.omega_window[active]))
    True
    >>> spectral_only = absorber(grid, temporal=False)
    >>> float(spectral_only.temporal_rate.max()), spectral_only.temporal
    (0.0, False)
    """
    length = _reference_length(grid, boundary_N, boundary_length)
    _check_collar("spectral", grid.omega_collar, grid.omega, "rad/s")
    _check_collar("temporal", grid.t_collar, grid.t, "s")
    spectral = collar_rate(grid.omega_window, length)
    # Outside the active mask the linear operator is zero and the nonlinear polarisation
    # is masked, so there is nothing left for a rate to absorb; leaving the capped value
    # there would only amplify the band edge in the interaction picture.
    band_mask = grid.active_mask
    spectral[~band_mask] = 0.0
    boundary = Absorber(
        spectral_rate=spectral,
        temporal_rate=temporal_collar_rate(grid.t, grid.t_collar, length)
        if temporal
        else np.zeros(grid.n_t),
        temporal_rate_fine=temporal_collar_rate(grid.t_fine, grid.t_collar, length)
        if temporal
        else np.zeros(grid.n_to),
        band_mask=band_mask,
        reference_length=length,
        temporal=temporal,
    )
    _log_summary(grid, boundary)
    return boundary


def no_absorber(
    grid: TimeGrid,
    *,
    boundary_N: float = DEFAULT_BOUNDARY_N,  # noqa: N803 - the established parameter name
    boundary_length: float | None = None,
) -> Absorber:
    """Build boundaries with both soft collars off and the hard band mask kept.

    What ``absorber=False`` on the keyword interfaces means (SEP 21 section 5): the
    temporal and spectral rates are zero, so no light is absorbed at either edge of
    the window, and the hard band mask still keeps the field inside the representable
    band. The reference length is kept, so that a radial, Cartesian or planar run's
    real-space collar, which reads it, keeps its strength.

    Parameters
    ----------
    grid : TimeGrid
        The grid, for its masks and its propagation length.
    boundary_N : float, optional
        As for :func:`absorber`; it sets only the reference length here.
    boundary_length : float or None, optional
        Reference length in metres, overriding ``boundary_N``.

    Returns
    -------
    Absorber
        Zero spectral and temporal rates, the band mask and the reference length,
        recorded as ``temporal=False``.

    Examples
    --------
    >>> from solprop.boundaries import no_absorber
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> boundary = no_absorber(grid)
    >>> float(boundary.spectral_rate.max()), float(boundary.temporal_rate.max())
    (0.0, 0.0)
    >>> bool(boundary.band_mask.any())
    True
    """
    base = absorber(
        grid, boundary_N=boundary_N, boundary_length=boundary_length, temporal=False
    )
    return replace(base, spectral_rate=np.zeros_like(base.spectral_rate))


def custom_absorber(
    grid: TimeGrid,
    spectral_rate: float
    | Float[np.ndarray, " n_omega"]
    | Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]],
    *,
    collar: Absorber | None = None,
) -> Absorber:
    """Add a user's spectral power rate to a collar, keeping its in-stage collars.

    The spectral rate is diagonal in frequency and folded into the linear operator,
    where it is exact and free whatever its strength (ADR-007); the in-stage collars
    are explicit and their strength bounds a replayed step (SEP 22 section 4.6). The
    user's rate is therefore added to the collar's spectral rate, and the collar's
    temporal rates, band mask and in-stage strength are kept as they are: its
    ``collar_length`` is the collar's own ``in_stage_length``. Its
    ``reference_length`` stays the collar's when the summed rate fits under the
    collar's ceiling ``60 / reference_length``, and otherwise becomes ``60`` over the
    sum's largest value, the longest length that admits it. The collar length is
    recorded only where it differs from the reference length, so a zero user rate
    returns a collar equal to the one given, field for field.

    Parameters
    ----------
    grid : TimeGrid
        The grid the collar is for, for its frequency axis.
    spectral_rate : float, numpy.ndarray or callable
        The user's power absorption coefficient, 1/m: a scalar, an array of shape
        ``(n_omega,)`` on the grid's coarse frequency axis, or a callable of the
        angular frequency ``omega`` in rad/s (the grid's own axis, an envelope grid's
        in FFT order) returning that array. Finite and non-negative; it is zeroed
        outside the collar's band mask, where the operator is zero.
    collar : Absorber or None, optional
        The collar it is added to, taken by value; ``None`` is :func:`absorber` of the
        grid with its defaults. Any collar is accepted as it is, a spectral-only one
        (``absorber(grid, temporal=False)``) among them.

    Returns
    -------
    Absorber
        The collar with the summed spectral rate.

    Raises
    ------
    GridError
        If the rate is not finite and non-negative, or not of the frequency axis's
        shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import absorber, custom_absorber
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> collar = absorber(grid)
    >>> same = custom_absorber(grid, 0.0, collar=collar)
    >>> same.reference_length == collar.reference_length, same.collar_length
    (True, None)
    >>> strong = custom_absorber(grid, 1e5, collar=collar)
    >>> strong.reference_length < collar.reference_length
    True
    >>> strong.in_stage_length == collar.in_stage_length
    True
    >>> bool(np.array_equal(strong.temporal_rate, collar.temporal_rate))
    True
    """
    base = absorber(grid) if collar is None else collar
    omega = np.asarray(grid.omega, dtype=np.float64)
    values = np.asarray(
        spectral_rate(omega) if callable(spectral_rate) else spectral_rate,
        dtype=np.float64,
    )
    if values.shape not in ((), omega.shape):
        msg = (
            f"the spectral rate has shape {values.shape}; give a scalar or an array "
            f"of the frequency axis's shape {omega.shape}"
        )
        raise GridError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "a custom spectral rate must be finite and non-negative, in 1/m"
        raise GridError(msg)
    added = np.where(base.band_mask, values, 0.0)
    # A zero rate leaves the collar's own array, signed zeros and all, so the record
    # is the collar's field for field.
    total = base.spectral_rate + added if np.any(added) else base.spectral_rate
    largest = float(np.max(total, initial=0.0))
    reference = base.reference_length
    if largest > MAX_ALPHA_LENGTH / reference:
        # The longest length whose ceiling admits the sum; the in-stage collars keep
        # their own length, so nothing explicit moves (SEP 22 section 5.4).
        reference = MAX_ALPHA_LENGTH / largest
    return replace(
        base,
        spectral_rate=total,
        reference_length=reference,
        collar_length=base.in_stage_length,
    )


@dataclass(frozen=True)
class TransverseCollar:
    r"""A transverse-wavevector (k-perp) safety ramp on a periodic Cartesian grid.

    SEP 22 section 4.3. With :math:`\rho = \max(|k_x|/k_{x,\max}, |k_y|/k_{y,\max})`
    the power rate is :math:`\alpha_\perp = S\,P(\rho)`, where :math:`P` is zero for
    :math:`\rho \le 1 - f` and rises as a raised cosine to one at the grid's largest
    wavenumber. It enters the separable operator as a diagonal rate, which is exact
    under the cumulative flow and costs no transform (``docs/dev/kernel-contract.md``
    section 3). It damps content that reaches the outer band and keeps rounding from
    accumulating at the Nyquist edge; it does **not** remove an alias that in-band
    content generates, which grid sizing by the one-half rule prevents
    (``docs/guide/free-space/cartesian.md``).

    Attributes
    ----------
    strength : float
        The rate :math:`S` at the grid's edge, a power coefficient in m^-1, finite and
        nonnegative. ``S L / 2 >= log(1e3)`` removes 60 dB of field at the edge over
        an interaction length ``L``.
    fraction : float
        The ramp's width :math:`f` as a fraction of each axis's largest wavenumber,
        in (0, 1). Default 0.1, the width of the spatial grids' default collar.

    Raises
    ------
    GridError
        If the strength is negative or not finite, or the fraction is outside (0, 1).

    Examples
    --------
    >>> from solprop.boundaries import TransverseCollar
    >>> TransverseCollar(strength=2e4)
    TransverseCollar(strength=20000.0, fraction=0.1)
    """

    strength: float
    fraction: float = 0.1

    def __post_init__(self) -> None:
        """Check the strength and the fraction."""
        if not math.isfinite(self.strength) or self.strength < 0:
            msg = (
                "TransverseCollar.strength must be a finite nonnegative rate in 1/m, "
                f"got {self.strength!r}"
            )
            raise GridError(msg)
        if not math.isfinite(self.fraction) or not 0 < self.fraction < 1:
            msg = f"TransverseCollar.fraction must lie in (0, 1), got {self.fraction!r}"
            raise GridError(msg)
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "fraction", float(self.fraction))


def transverse_collar_rate(
    space: FreeGrid | RadialGrid, collar: TransverseCollar
) -> Float[np.ndarray, " n_columns"]:
    """Evaluate a transverse collar's power rate on a Cartesian grid's columns.

    Parameters
    ----------
    space : solprop.grid.FreeGrid
        The periodic Cartesian grid, or a :class:`solprop.grid.LineGrid`, on which
        the collar acts on ``k_x`` alone. A radial grid is refused.
    collar : TransverseCollar
        The strength and the ramp's width.

    Returns
    -------
    numpy.ndarray
        The nonnegative power rate in m^-1, shape ``(n_x*n_y,)``, in column order
        ``ix + n_x*iy``.

    Raises
    ------
    GridError
        If `space` is radial, which is not periodic. The hybrid planar kind's
        ``k_y``-only collar is not offered either (SEP 22, Q5); translation refuses it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import TransverseCollar, transverse_collar_rate
    >>> from solprop.grid import FreeGrid
    >>> space = FreeGrid(radius_x=1e-4, n_x=20, radius_y=1e-4, n_y=20)
    >>> rate = transverse_collar_rate(space, TransverseCollar(strength=1e4))
    >>> rate.shape, float(rate.max()), float(rate[0])
    ((400,), 10000.0, 0.0)
    """
    if not isinstance(space, FreeGrid):
        msg = (
            "a TransverseCollar acts on a periodic Cartesian grid (FreeGrid or "
            f"LineGrid), not on {type(space).__name__}: the radial kind is not "
            "periodic, and the hybrid planar kind's k_y-only collar is not offered"
        )
        raise GridError(msg)
    profile = [
        _ramp(_normalised(axis), collar.fraction) for axis in (space.k_x, space.k_y)
    ]
    # The max-norm (Q5): the larger of the two axes' ramps, since P is monotone.
    ramp = np.maximum(profile[0][None, :], profile[1][:, None]).reshape(-1)
    return collar.strength * ramp


def _normalised(axis: Float[np.ndarray, " n_axis"]) -> Float[np.ndarray, " n_axis"]:
    """Return ``|k| / max |k|`` on one axis, zeros on a one-sample axis.

    Parameters
    ----------
    axis : numpy.ndarray
        Angular frequencies in rad/m.

    Returns
    -------
    numpy.ndarray
        Values in [0, 1].
    """
    largest = float(np.max(np.abs(axis)))
    if largest == 0.0:
        return np.zeros(axis.size)
    return np.abs(axis) / largest


def _ramp(
    rho: Float[np.ndarray, " n_axis"], fraction: float
) -> Float[np.ndarray, " n_axis"]:
    """Return the raised-cosine ramp, zero up to ``1 - fraction`` and one at one.

    Parameters
    ----------
    rho : numpy.ndarray
        Normalised wavenumbers in [0, 1].
    fraction : float
        The ramp's width, in (0, 1).

    Returns
    -------
    numpy.ndarray
        The profile, in [0, 1].
    """
    inside = np.clip((rho - 1 + fraction) / fraction, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * inside))


def stable_step(temporal_rate: Float[np.ndarray, " n_t"]) -> float:
    r"""Return the longest step an explicit method can take with an in-stage collar.

    The ``"nonlinear"`` absorber scheme adds ``-alpha(t) E(t) / 2`` to the derivative,
    so the collar's largest rate becomes an eigenvalue of the linearised problem and an
    explicit Runge-Kutta method must keep ``h alpha / 2`` inside its stability region.
    The split factor of :func:`split_step_factor` integrates the same term *exactly* and
    is unconditionally stable, which is the one thing it is better at.

    DOPRI5's and Tsit5's stability region reaches :data:`STABILITY_RADIUS` along the
    negative real axis, so the limit is ``2 R / max(alpha)``. Measured on the 1 mm
    capillary of the driver tests, whose collar reached 1.2e6 m^-1 under the capped
    logarithm then in force: a fixed schedule diverges at ``h alpha / 2 = 3.75`` and
    completes at 3.13, which brackets the textbook 3.3. With the bounded profile the
    largest rate is ``TEMPORAL_ALPHA_LENGTH / ell``, so the limit is ``0.52 ell``.

    The rate reaches its ceiling only at the very edge of the window, where the field
    is by design negligible, so an *adaptive* run is not usually held at this limit --
    the error norm is scaled by the state and an enormous derivative on an empty bin
    scores nothing. What the limit binds is a replayed schedule, and a floor `dz_min`
    set above it.

    Parameters
    ----------
    temporal_rate : numpy.ndarray
        The collar's power absorption coefficient on the time axis, 1/m, shape
        ``(n_t,)``.

    Returns
    -------
    float
        The longest stable step in m, or ``inf`` when the collar is identically zero.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import stable_step
    >>> round(stable_step(np.array([0.0, 6.6, 3.3])), 6)
    1.0
    >>> stable_step(np.zeros(4))
    inf
    """
    largest = float(np.max(temporal_rate))
    if largest <= 0.0:
        return math.inf
    return 2.0 * STABILITY_RADIUS / largest


def split_step_factor(
    temporal_rate: Float[np.ndarray, " n_t"], step: float
) -> Float[np.ndarray, " n_t"]:
    """Return the factor the temporal absorber multiplies the field by over one step.

    ``exp(-alpha(t) h / 2)``: the field, not the power, because that is what the state
    holds, and ``alpha`` is a power coefficient. Successive factors telescope exactly to
    ``exp(-alpha(t) L / 2)`` however the solver subdivides the propagation, which is the
    whole point of rate semantics -- and the property the tests assert.

    The factor of one half is required: ``exp(-alpha(t) h)`` could not be right
    alongside the ``-alpha(t) E(t) / 2`` of the alternative scheme, or alongside
    ``alpha = -2 log W / ell``: with it, one reference length would apply the collar
    profile to the field twice, and the two schemes would differ at first order in the
    step rather than agreeing there.

    Parameters
    ----------
    temporal_rate : numpy.ndarray
        Power absorption coefficient on the time axis, 1/m, shape ``(n_t,)``.
    step : float
        Distance travelled in metres, non-negative.

    Returns
    -------
    numpy.ndarray
        Multiplicative factor in ``(0, 1]``, shape ``(n_t,)``.

    Raises
    ------
    GridError
        If the step is negative or not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import collar_rate, split_step_factor
    >>> alpha = collar_rate(np.array([1.0, 0.5]), 0.1)
    >>> np.round(split_step_factor(alpha, 0.1), 6).tolist()
    [1.0, 0.5]
    """
    if not (math.isfinite(step) and step >= 0.0):
        msg = f"step must be non-negative and finite, got {step!r}"
        raise GridError(msg)
    return np.exp(-0.5 * temporal_rate * step)


def apply_split_step(
    field: Inexact[np.ndarray, "*batch n_t"],
    temporal_rate: Float[np.ndarray, " n_t"],
    step: float,
) -> Inexact[np.ndarray, "*batch n_t"]:
    """Apply the temporal absorber to a time-domain field over one accepted step.

    The reference implementation of the split-step absorber scheme, the alternative to
    the default in-stage term of :func:`nonlinear_absorber_term`, and what that term is
    measured against.

    Parameters
    ----------
    field : numpy.ndarray
        Time-domain field, shape ``(..., n_t)``; real on a field grid, complex on an
        envelope grid.
    temporal_rate : numpy.ndarray
        Power absorption coefficient on the time axis, 1/m, shape ``(n_t,)``.
    step : float
        Distance travelled in metres.

    Returns
    -------
    numpy.ndarray
        The attenuated field, with the shape of ``field``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import apply_split_step, collar_rate
    >>> alpha = collar_rate(np.array([1.0, 0.5]), 0.1)
    >>> np.round(apply_split_step(np.ones((1, 2)), alpha, 0.1), 6).tolist()
    [[1.0, 0.5]]
    """
    return field * split_step_factor(temporal_rate, step)


def nonlinear_absorber_term(
    field: Inexact[np.ndarray, "*batch n_t"],
    temporal_rate: Float[np.ndarray, " n_t"],
) -> Inexact[np.ndarray, "*batch n_t"]:
    """Return the temporal absorber as a term to add inside the nonlinear evaluation.

    ``-alpha(t) E(t) / 2``: the derivative form of the same rate, and the reference
    implementation of the default in-stage absorber scheme. Integrating it exactly over
    a step would give :func:`apply_split_step`; a Runge-Kutta stage integrates it to the
    order of the method, so the two agree to first order in the step and differ at
    second, which is what the comparison of the two schemes measures.

    Parameters
    ----------
    field : numpy.ndarray
        Time-domain field, shape ``(..., n_t)``.
    temporal_rate : numpy.ndarray
        Power absorption coefficient on the time axis, 1/m, shape ``(n_t,)``.

    Returns
    -------
    numpy.ndarray
        The contribution to ``dE/dz`` in the time domain, with the shape of ``field``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import collar_rate, nonlinear_absorber_term
    >>> alpha = collar_rate(np.array([0.8, 0.5]), 0.1)
    >>> np.round(nonlinear_absorber_term(np.ones(2), alpha), 6).tolist()
    [-2.231436, -6.931472]
    """
    return -0.5 * temporal_rate * field


def apply_band_mask(
    spectrum: Inexact[np.ndarray, "*batch n_omega"],
    band_mask: Bool[np.ndarray, " n_omega"],
) -> Inexact[np.ndarray, "*batch n_omega"]:
    """Zero a spectrum outside the active band.

    The hard part of the boundary, kept as a 0/1 multiply rather than made a rate: it is
    idempotent, so applying it once per accepted step never accumulates, and it is the
    definition of the band the equation is solved on rather than an absorber.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectral amplitudes, shape ``(..., n_omega)``.
    band_mask : numpy.ndarray
        True where the physics lives, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The masked spectrum, with the shape of ``spectrum``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.boundaries import apply_band_mask
    >>> mask = np.array([True, False, True])
    >>> apply_band_mask(np.array([1.0, 2.0, 3.0]), mask).tolist()
    [1.0, 0.0, 3.0]
    """
    return np.where(band_mask, spectrum, 0.0)


def warn_walkoff(
    grid: TimeGrid,
    group_delay_span: float,
    *,
    boundary_N: float = DEFAULT_BOUNDARY_N,  # noqa: N803 - the established parameter name
    boundary_length: float | None = None,
) -> None:
    """Warn if the absorber is too weak for the walk-off the run can produce.

    A packet whose group delay drifts by ``group_delay_span`` over the whole propagation
    crosses the temporal collar at ``group_delay_span / length`` seconds per metre, so
    it spends ``dt / v`` metres at each sample and accumulates a field attenuation of
    ``(length / group_delay_span) * integral(alpha) / 2`` nepers on the way through.
    Below :data:`MIN_COLLAR_ATTENUATION` the packet reaches the edge of the window still
    alive, wraps around, and aliases onto the rest of the field.

    The span is the caller's, deliberately. A prediction from the group delay across the
    whole simulation band would, on a grid running to four microns, be dominated by band
    edges carrying nothing but shot noise, and so would fire on almost every run. A
    caller who passes the span of the band it cares about does not have that problem
    (``docs/dev/stepper.md`` section 6).

    Parameters
    ----------
    grid : TimeGrid
        The grid, for its temporal collar and its propagation length.
    group_delay_span : float
        Spread of group delay accumulated over the propagation, in seconds. Zero or
        negative means nothing walks anywhere and nothing is checked.
    boundary_N : float, optional
        Absorber strength, as in :func:`absorber`.
    boundary_length : float or None, optional
        Reference length in metres, overriding ``boundary_N``.

    Raises
    ------
    GridError
        If the span is not finite, or the reference length is invalid.

    Warns
    -----
    SolpropWarning
        If a packet drifting by that span would survive the collar.

    Examples
    --------
    >>> import warnings
    >>> from solprop.boundaries import warn_walkoff
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     warn_walkoff(grid, 50e-15)
    """
    if not math.isfinite(group_delay_span):
        msg = f"group_delay_span must be finite, got {group_delay_span!r}"
        raise GridError(msg)
    length = _reference_length(grid, boundary_N, boundary_length)
    if group_delay_span <= 0.0:
        return
    rate = temporal_collar_rate(grid.t, grid.t_collar, length)
    crossing = (
        0.5 * float(np.trapezoid(rate, dx=grid.dt)) * grid.length / group_delay_span
    )
    if crossing >= MIN_COLLAR_ATTENUATION:
        return
    warnings.warn(
        f"the temporal absorber attenuates a packet walking off by "
        f"{group_delay_span * 1e15:.3g} fs over {grid.length:.3g} m by only "
        f"{crossing:.2g} nepers, below the {MIN_COLLAR_ATTENUATION:.2g} that stops it "
        f"wrapping around the time window; widen trange, or raise boundary_N above "
        f"{boundary_N * MIN_COLLAR_ATTENUATION / max(crossing, 1e-30):.3g}",
        SolpropWarning,
        stacklevel=2,
    )


def _log_summary(grid: TimeGrid, boundary: Absorber) -> None:
    """Record the absorber's strength and geometry at debug level.

    Parameters
    ----------
    grid : TimeGrid
        The grid the absorber was built from.
    boundary : Absorber
        The finished absorber.
    """
    if logger.isEnabledFor(logging.DEBUG):
        collar = grid.t_collar
        window = float(np.max(grid.t) - np.min(grid.t))
        applications = grid.length / boundary.reference_length
        logger.debug(
            "absorbing boundaries: reference length %.3g m (%.3g applications over "
            "%.3g m; the half point of a taper attenuates the field by %.1e over the "
            "propagation), temporal collar %.3g fs, %.1f%% of the time window",
            boundary.reference_length,
            applications,
            grid.length,
            0.5**applications,
            collar.width_low * 1e15,
            100.0 * collar.width_low / window,
        )

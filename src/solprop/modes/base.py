# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Derived quantities of any mode: beta, alpha, group delay, dispersion, ZDW.

Everything here is a function of a :class:`~solprop.modes.protocol.Mode`, not a method
on one, so a new mode family gets them for nothing: the propagation constant and
power loss (:func:`beta`, :func:`alpha`, :func:`loss_length`, :func:`db_per_m`,
:func:`transmission`), the dispersion orders (:func:`beta1`, :func:`beta2`,
:func:`beta3`), the zero-dispersion wavelength (:func:`zdw`, a sampled search of
``beta_2``'s sign) and the effective area (:func:`aeff`). The conversions are the
package's (``docs/dev/conventions.md`` section 2):
:math:`\beta = (\omega/c)\,\mathrm{Re}\,n_{\rm eff}` and
:math:`\alpha = (2\omega/c)\,\mathrm{Im}\,n_{\rm eff}`, so a positive imaginary
index is loss and ``alpha`` is a power loss coefficient per metre.

:func:`beta1` is **analytic**, straight from the mode's ``dneff_domega``, because the
co-moving frame rides on it (``docs/dev/stepper.md`` section 5); :func:`beta2` and
:func:`beta3` are Richardson differences of that analytic ``beta1``, with steps
:data:`BETA2_RELATIVE_STEP` and :data:`BETA3_RELATIVE_STEP`. Their accuracy is in
``docs/guide/waveguides/capillary-model.md``, section "The frequency derivative".
"""

import logging
import math
import warnings
from typing import Final

import numpy as np
from jaxtyping import Float
from scipy.integrate import quad_vec

from solprop.constants import C
from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialRangeWarning,
    ModeError,
    NoZeroDispersionError,
)
from solprop.modes.protocol import Mode

__all__ = [
    "BETA2_RELATIVE_STEP",
    "BETA3_RELATIVE_STEP",
    "aeff",
    "alpha",
    "beta",
    "beta1",
    "beta2",
    "beta3",
    "db_per_m",
    "loss_length",
    "transmission",
    "zdw",
]

logger = logging.getLogger(__name__)

BETA2_RELATIVE_STEP: Final = 3e-3
"""Relative frequency step of the Richardson difference behind :func:`beta2`.

Larger than a finite difference usually wants, and deliberately so. ``beta_1`` is of
order ``1/c`` while its variation over the step is ``beta_2 h``, which for a capillary
is nine orders smaller, so the rounding error of the difference is ``eps beta_1 / h``
-- it *falls* as the step grows. Truncation, ``(h^2/6) beta_4``, rises. Measured
against the closed-form second derivative of an evacuated capillary
(``tests/modes/test_base.py``), the two cross near a relative step of 3e-3, where the
result is good to 5e-9; at 1e-4 it is 1.6e-7 and at 1e-5 it is 2.3e-6.

That is three orders better than differencing ``beta`` itself, and it is
enough: nothing in the solver consumes ``beta_2``. The co-moving frame uses ``beta_1``,
which is analytic. Making this analytic too needs the second derivative of the gas
susceptibility, which is not part of the material interface.
"""

BETA3_RELATIVE_STEP: Final = 4e-3
"""Relative frequency step of the Richardson second difference behind :func:`beta3`.

A second difference divides the rounding of ``beta_1`` by the step squared, while the
truncation grows with the structure the fill puts into ``beta_1``, and the two pull
the step opposite ways. Against the closed form of an evacuated capillary
(``tests/modes/test_base.py``), which has little structure, the error is 3.8e-6 at a
relative step of 3e-3, 1.5e-6 at 4e-3 and 4.8e-7 at 8e-3. A helium-filled capillary
(125 um, one bar, 400 nm to 2 um) has more: there, steps of 3e-3 and 5e-3 agree with
4e-3 to 1.5e-5 of the largest value, while a step of 2e-2 is off by 4e-4.
"""

_DECIBEL_PER_NEPER: Final = 10.0 / math.log(10.0)
"""Conversion from a power loss coefficient in 1/m to dB/m."""

_ZDW_SAMPLES: Final = 512
"""Samples, evenly spaced in frequency, on which :func:`zdw` reads the sign of
``beta_2``: 3.6e13 rad/s apart across its default window."""

_ZDW_REFINE_ROUNDS: Final = 4
"""Rounds of array samples that refine a sign change of the dispersion. Four take a
bracket of 3.6e13 rad/s to 3.4e7 rad/s, where interpolating linearly reaches the noise
of the differenced ``beta2``."""

_ZDW_REFINE_SAMPLES: Final = 33
"""Samples of each refinement round: 32 intervals, so each round narrows 32-fold."""

_ZDW_STENCIL_RATIO: Final = (2.0, 8.0)
"""Bounds on the rise of ``beta1`` from a candidate zero at a full step of the
``beta2`` stencil over its rise at a half step, on each side. A resolved extremum
gives 4; see :func:`_is_zero`."""


def _as_omega(omega: Float[np.ndarray, " n_omega"]) -> Float[np.ndarray, " n_omega"]:
    """Validate an angular-frequency argument for a derived quantity.

    - ``omega``: angular frequencies in rad/s, shape ``(n_omega,)``.
    """
    values = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ModeError("angular frequency must be finite")
    return values


def beta(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the propagation constant of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta = omega Re(n_eff) / c`` in 1/m, shape ``(n_omega,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta(mode, np.array([2.356e15]))[0]:.6f}"
    '7858997.479689'
    """
    values = _as_omega(omega)
    return np.asarray(values * np.real(mode.neff(values, z)) / C)


def alpha(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the power loss coefficient of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``alpha = 2 omega Im(n_eff) / c`` in 1/m, shape ``(n_omega,)``. Positive is
        loss, because ``Im n_eff > 0`` is.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import alpha
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{alpha(mode, np.array([2.356e15]))[0]:.6f}"
    '0.141459'
    """
    values = _as_omega(omega)
    return np.asarray(2.0 * values * np.imag(mode.neff(values, z)) / C)


def beta1(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the group delay per unit length, analytically.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta_1 = d(beta)/d(omega) = (Re n_eff + omega d(Re n_eff)/d(omega)) / c`` in
        s/m, shape ``(n_omega,)``, from the mode's analytic derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta1
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta1(mode, np.array([2.356e15]))[0]:.10e}"
    '3.3357582351e-09'
    """
    values = _as_omega(omega)
    index = np.real(mode.neff(values, z))
    derivative = np.real(mode.dneff_domega(values, z))
    return np.asarray((index + values * derivative) / C)


def beta2(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the group-velocity dispersion, by differencing the analytic `beta1`.

    Departs from the analytic derivatives elsewhere in this module, and says so: the
    second derivative of a gas susceptibility is not part of the material interface,
    so this is a Richardson-extrapolated central difference of :func:`beta1` at a
    relative step of :data:`BETA2_RELATIVE_STEP`, whose documentation gives the
    measured accuracy. Only the co-moving frame needs better -- which uses
    ``beta1``, not this.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``, > 0.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta_2 = d^2(beta)/d(omega)^2`` in s^2/m, shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If any frequency is not finite or not positive, since the step is relative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta2
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta2(mode, np.array([2.356e15]))[0]:.6e}"
    '-7.491419e-30'
    """
    values = _as_omega(omega)
    if np.any(values <= 0.0):
        raise ModeError(
            "beta2 differences beta1 with a relative step, so the frequency must be "
            "positive"
        )
    step = BETA2_RELATIVE_STEP * values
    wide = (beta1(mode, values + step, z) - beta1(mode, values - step, z)) / (
        2.0 * step
    )
    half = step / 2.0
    narrow = (beta1(mode, values + half, z) - beta1(mode, values - half, z)) / (
        2.0 * half
    )
    return np.asarray((4.0 * narrow - wide) / 3.0)


def beta3(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the third-order dispersion, by differencing the analytic `beta1` twice.

    A Richardson-extrapolated second central difference of :func:`beta1` at a relative
    step of :data:`BETA3_RELATIVE_STEP`, for the reason :func:`beta2` gives: the
    material interface has no higher derivative of a gas susceptibility.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``, > 0.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta_3 = d^3(beta)/d(omega)^3`` in s^3/m, shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If any frequency is not finite or not positive, since the step is relative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta3
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta3(mode, np.array([2.356e15]))[0]:.3e}"
    '1.124e-44'
    """
    values = _as_omega(omega)
    if np.any(values <= 0.0):
        raise ModeError(
            "beta3 differences beta1 with a relative step, so the frequency must be "
            "positive"
        )
    centre = beta1(mode, values, z)
    step = BETA3_RELATIVE_STEP * values
    wide = _second_difference(mode, values, centre, step, z)
    narrow = _second_difference(mode, values, centre, step / 2.0, z)
    return np.asarray((4.0 * narrow - wide) / 3.0)


def _second_difference(
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    centre: Float[np.ndarray, " n_omega"],
    step: Float[np.ndarray, " n_omega"],
    z: float,
) -> Float[np.ndarray, " n_omega"]:
    """Return the central second difference of `beta1` at one step.

    - ``omega``: angular frequencies in rad/s.
    - ``centre``: ``beta1`` at `omega`, s/m.
    - ``step``: the step at each frequency, rad/s.
    - ``z``: position along the waveguide, m.
    - Returns: the difference, s^3/m.
    """
    above = beta1(mode, omega + step, z)
    below = beta1(mode, omega - step, z)
    return np.asarray((above - 2.0 * centre + below) / step**2)


def _refine_zero(mode: Mode, low: float, high: float, z: float) -> float:
    """Return a sign change of the dispersion inside a bracket, by array samples.

    A mode's index models evaluate an array at about the cost of one point, so each
    round narrows the bracket 32-fold for the cost of one Brent step, where a scalar
    search costs a glass model evaluation per iterate. The rounds stop short of a pole
    of the differenced ``beta2``, which a scalar search converges onto until the model
    returns NaN.

    - ``low``, ``high``: the bracket, rad/s, whose ends have ``beta2`` of opposite
      signs.
    - ``z``: position along the waveguide, m.
    - Returns: the first sign change upwards in frequency, rad/s, linearly interpolated
      in the last bracket; or NaN when samples that are not finite separate the signs.
    """
    first = last = math.nan
    for _ in range(_ZDW_REFINE_ROUNDS):
        grid = np.linspace(low, high, _ZDW_REFINE_SAMPLES)
        sampled = beta2(mode, grid, z)
        crossing = np.flatnonzero(np.sign(sampled[:-1]) * np.sign(sampled[1:]) <= 0.0)
        if crossing.size == 0:
            return math.nan
        i = int(crossing[0])
        low, high = float(grid[i]), float(grid[i + 1])
        # The first change has a nonzero sample below it, so the two cannot be equal.
        first, last = float(sampled[i]), float(sampled[i + 1])
    return low - first * (high - low) / (last - first)


def _is_zero(mode: Mode, omega: float, z: float) -> bool:
    """Whether a refined sign change of the differenced `beta2` is a zero of it.

    The test reads the analytic ``beta1`` at the five frequencies that span the
    difference behind :func:`beta2`, ``omega (1 + s h)`` for ``s`` in -1, -1/2, 0,
    1/2, 1 and ``h`` :data:`BETA2_RELATIVE_STEP`. At a zero of ``beta_2`` that the
    difference resolves, ``beta1`` is strictly extremal there and its Taylor series
    converges across the stencil, so its rise from the centre is quadratic: four times
    as large at a full step as at a half step, on each side. A factor of two either
    way (:data:`_ZDW_STENCIL_RATIO`) admits a cubic term up to two thirds of the
    quadratic one at the full step. Each artefact of a resonance fails it: a pole at
    the change (``beta1`` diverges, its rise does not grow with the step), a stencil
    point on or across a line (the rise is lopsided, or one-signed through the centre
    so ``beta1`` is monotone), and the evanescent floor (``beta1`` constant, no rise).
    A non-finite ``beta1`` fails every comparison.
    """
    offsets = BETA2_RELATIVE_STEP * np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rise = beta1(mode, omega * (1.0 + offsets), z)
    rise = rise - rise[2]
    half, full = rise[[1, 3]], rise[[0, 4]]
    if not half[0] * half[1] > 0.0:
        return False
    ratio = full / half
    low, high = _ZDW_STENCIL_RATIO
    return bool(np.all((ratio >= low) & (ratio <= high)))


def _zero_frequencies(
    mode: Mode, z: float, lower: float, upper: float, near: float | None
) -> tuple[list[float], int]:
    """Return the zeros of `beta2` in a frequency window, and its sign changes there.

    ``beta2`` is sampled at :data:`_ZDW_SAMPLES` frequencies evenly spaced across the
    window; each sign change is refined (:func:`_refine_zero`) and kept only if it is
    a zero (:func:`_is_zero`). With a reference frequency ``near``, the changes are
    taken outwards from it and only the nearest zero is returned. A change that is
    not a zero marks a resonance, and the changes beyond it on that side are not
    taken: light at the reference cannot cross the line to reach them. A change
    further than the best zero so far cannot beat it, which spares the dozens an
    antiresonant fibre's resonances give.

    - ``lower``, ``upper``: the window, rad/s.
    - ``near``: reference frequency, rad/s, or ``None`` for every zero.
    - Returns: the zeros in rad/s, increasing, and the number of sign changes sampled.
    """
    grid = np.linspace(lower, upper, _ZDW_SAMPLES)
    zeros: list[float] = []
    # The nearest resonances found below and above the reference, rad/s.
    below, above = -math.inf, math.inf
    # The window is probed, not answered: a model's range and dilute-scaling warnings
    # are left to the caller's evaluation at the answer. The refinement converges onto
    # a pole as readily as onto a zero, and a model may divide by zero exactly there.
    with (
        warnings.catch_warnings(),
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", DiluteScalingWarning)
        sampled = beta2(mode, grid, z)
        changes = np.flatnonzero(np.sign(sampled[:-1]) * np.sign(sampled[1:]) < 0.0)
        if near is not None:
            # How near each bracket can bring a zero: 0 for one holding the reference.
            reach = np.maximum.reduce(
                [grid[changes] - near, near - grid[changes + 1], np.zeros(changes.size)]
            )
            changes = changes[np.argsort(reach, kind="stable")]
        for i in changes:
            low, high = float(grid[i]), float(grid[i + 1])
            if near is not None:
                if zeros and max(low - near, near - high) >= abs(zeros[0] - near):
                    break
                if high <= below or low >= above:
                    continue
            root = _refine_zero(mode, low, high, z)
            if math.isnan(root) or not _is_zero(mode, root, z):
                if near is not None and low >= near:
                    above = low
                elif near is not None and high <= near:
                    below = high
                continue
            if near is None:
                zeros.append(root)
            elif not zeros or abs(root - near) < abs(zeros[0] - near):
                zeros = [root]
    return sorted(zeros), int(changes.size)


def zdw(
    mode: Mode,
    z: float = 0.0,
    *,
    wavelength_limits: tuple[float, float] = (100e-9, 3e-6),
    near: float | None = None,
) -> float:
    """Return the zero-dispersion wavelength of a mode.

    A sampled search: ``beta_2`` is read at 512 frequencies evenly spaced across the
    window, each sign change is refined, and a change is a zero only where the
    analytic group delay ``beta_1`` has a resolved extremum. That drops the sign
    changes a gas resonance gives the differenced ``beta_2`` without a zero: through
    the resonance's pole, where the difference's stencil reaches a line, and on the
    evanescent floor above one (``docs/guide/waveguides/mode-sets.md``). Two limits
    remain: a zero and a pole inside one sample interval restore the sign and are
    both missed, and a sample that is not finite brackets nothing.

    Parameters
    ----------
    mode : Mode
        The mode.
    z : float, optional
        Position along the waveguide, m.
    wavelength_limits : tuple of float, optional
        Shortest and longest wavelength in m to search. Defaults to 100 nm to 3 um.
        The window is probed silently; the zero found is evaluated again, so one that
        lies outside the validity range of a material model behind the mode warns.
    near : float, optional
        Reference wavelength in m, > 0, typically the pump's. The zero nearest it in
        frequency is returned, among those no resonance separates from it: light at
        the reference cannot cross a line to reach a zero beyond. ``None`` (the
        default) takes every zero in the window, and refuses several.

    Returns
    -------
    float
        The wavelength in m where ``beta_2`` vanishes.

    Raises
    ------
    ModeError
        If the range is not ordered and positive, if `near` is not a positive finite
        wavelength, or if the window holds several zeros and `near` is ``None``: the
        message lists them.
    NoZeroDispersionError
        If the search finds no zero in the window (on the side of the resonances
        that holds `near`, when it is given): a mode with no zero-dispersion point
        there is a fact the caller has to see, not a missing value to propagate.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import zdw
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> f"{zdw(mode, wavelength_limits=(150e-9, 900e-9)) * 1e9:.1f}"
    '376.7'
    """
    shortest, longest = wavelength_limits
    if not 0.0 < shortest < longest:
        raise ModeError(
            f"wavelength_limits must be positive and ordered, got {wavelength_limits}"
        )
    if near is not None and not (math.isfinite(near) and near > 0.0):
        raise ModeError(f"near must be a positive, finite wavelength in m, got {near}")
    window = f"{shortest * 1e9:.4g}-{longest * 1e9:.4g} nm"
    zeros, changes = _zero_frequencies(
        mode,
        z,
        2.0 * math.pi * C / longest,
        2.0 * math.pi * C / shortest,
        None if near is None else 2.0 * math.pi * C / near,
    )
    if not zeros:
        sampled = f"{_ZDW_SAMPLES} samples over {window}"
        if changes == 0:
            found = f"beta2 does not change sign between the {sampled}"
        elif near is None:
            found = (
                f"beta2 changes sign {changes} times between the {sampled}, but none "
                "is a zero: each is a pole or a resonance artefact of the differenced "
                "beta2"
            )
        else:
            found = (
                f"beta2 changes sign {changes} times between the {sampled}, but none "
                f"is a zero between the resonances nearest {near * 1e9:.4g} nm"
            )
        raise NoZeroDispersionError(
            f"{found}, so the search found no zero-dispersion wavelength there"
        )
    if len(zeros) > 1:
        listed = ", ".join(
            f"{2.0 * math.pi * C / root * 1e9:.1f}" for root in reversed(zeros)
        )
        raise ModeError(
            f"beta2 has {len(zeros)} zeros over {window}, at {listed} nm; pass near= "
            "a reference wavelength to choose the nearest, or narrow wavelength_limits"
        )
    # Outside the probing, so a zero beyond a material model's range warns.
    beta(mode, np.array([zeros[0]]), z)
    return 2.0 * math.pi * C / zeros[0]


def loss_length(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the 1/e power loss length of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``1 / alpha`` in m, shape ``(n_omega,)``; infinite where the mode is lossless.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import loss_length
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{loss_length(mode, np.array([2.356e15]))[0]:.4f}"
    '7.0692'
    """
    with np.errstate(divide="ignore"):
        return np.asarray(1.0 / alpha(mode, omega, z))


def db_per_m(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the loss of a mode in decibels per metre.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``10 alpha / ln(10)`` in dB/m, shape ``(n_omega,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import db_per_m
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{db_per_m(mode, np.array([2.356e15]))[0]:.6f}"
    '0.614348'
    """
    return np.asarray(_DECIBEL_PER_NEPER * alpha(mode, omega, z))


def transmission(
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    length: float,
    start: float = 0.0,
) -> Float[np.ndarray, " n_omega"]:
    """Return the power transmission of a length of waveguide.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    length : float
        Length propagated in m, >= 0.
    start : float, optional
        Position the length starts at, m.

    Returns
    -------
    numpy.ndarray
        ``exp(-int alpha dz)`` over ``[start, start + length]``, shape ``(n_omega,)``.
        The integral is taken numerically because a taper or a gradient makes
        ``alpha`` a function of position; for a uniform waveguide it is exactly
        ``exp(-alpha L)``.

    Raises
    ------
    ModeError
        If `length` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import transmission
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{transmission(mode, np.array([2.356e15]), 1.0)[0]:.6f}"
    '0.868091'
    """
    values = _as_omega(omega)
    if length < 0.0:
        raise ModeError(f"a propagation length must not be negative, got {length}")
    if length == 0.0:
        return np.ones_like(values)
    integral = quad_vec(
        lambda position: alpha(mode, values, position), start, start + length
    )[0]
    return np.asarray(np.exp(-integral))


def aeff(mode: Mode, z: float = 0.0) -> float:
    """Return the effective area of a mode.

    A uniform accessor beside :func:`beta` and :func:`alpha`, so that code working over
    a mode set never has to mix method calls with function calls.

    Parameters
    ----------
    mode : Mode
        The mode.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    float
        ``A_eff`` in m^2.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import aeff
    >>> f"{aeff(MarcatiliMode(75e-6, 'He', 1.0)):.6e}"
    '8.421575e-09'
    """
    return mode.aeff(z)

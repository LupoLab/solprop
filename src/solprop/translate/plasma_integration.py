# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plasma column's running-integral rule, designed once per grid (SEP 18).

The production field-grid plasma source integrates the exposure and free current
along t. The physical polarisation returned by ``PlasmaResponse.evaluate`` adds a
third integral; production uses the held-current spectral conversion (SEP 27).
The plain cumulative trapezoid integrates a Fourier
component ``exp(i omega t)`` with the multiplier ``rho(theta) = (theta/2) cot(theta/2)``
times the exact ``-i/omega``, ``theta = omega dt``, which on the default grid leaves the
top of the active band 11 % short per integral. The band rule filters each integrand
with symmetric taps ``c_0 ... c_K`` before the trapezoid, chosen so that the composite
multiplier ``H = C rho``, with ``C(theta) = c_0 + 2 sum_k c_k cos(k theta)``, is one to
within :data:`DESIGN_TOLERANCE` on ``[0, theta_b]``, ``theta_b = omega_max_window dt``.

The design is hand-rolled numerics -- a constrained least-squares fit and a certified
extremum search -- so it lives here in translation, and the physics layer only carries
the result (:class:`solprop.nonlinear.ionisation.RunningIntegral`). The rules, the
certified ladder of designs and the evidence for each choice are in SEP 18, section 4.3;
the manual page is ``docs/guide/media/field-plasma.md``.
"""

import functools
import logging
import math
from collections.abc import Callable
from typing import Final

import numpy as np
from jaxtyping import Float
from scipy.optimize import minimize_scalar

from solprop.exceptions import TranslationError
from solprop.grid import TimeGrid
from solprop.nonlinear.ionisation import IntegrationPolicy, RunningIntegral

__all__ = [
    "ABOVE_BAND_BOUND",
    "CERTIFICATION_SAMPLES",
    "CERTIFICATION_XATOL",
    "CONDITION_LIMIT",
    "DESIGN_TOLERANCE",
    "FIT_NODES",
    "MAX_HALF_WIDTH",
    "design_running_integral",
    "integration_record",
    "trapezoid_multiplier",
]

logger = logging.getLogger(__name__)

DESIGN_TOLERANCE: Final = 1e-5
"""The bound on ``|H - 1|`` in band; the smallest K meeting it is chosen."""

MAX_HALF_WIDTH: Final = 8
"""The largest ``K`` tried; a grid no design up to it can serve is refused."""

ABOVE_BAND_BOUND: Final = 1.05
"""The bound on ``|H|`` from the band edge to Nyquist: at most 5 % amplification of
content above the band relative to the exact integral."""

FIT_NODES: Final = 4097
"""Uniform least-squares nodes on ``[0, theta_b]``, zero included, unit weights."""

CERTIFICATION_SAMPLES: Final = 8193
"""Independent samples of the certified maxima, refined by a bounded scalar search."""

CERTIFICATION_XATOL: Final = 1e-12
"""Absolute tolerance in ``theta`` of the refinement of each sampled local maximum."""

CONDITION_LIMIT: Final = 1e10
"""The largest scaled condition number of an admitted fit; no admitted grid nears it."""

_SERIES_BELOW: Final = 1e-4
"""Below this ``theta`` the multiplier is its series, equal there to round-off."""


def trapezoid_multiplier(
    theta: Float[np.ndarray, "*shape"],
) -> Float[np.ndarray, "*shape"]:
    """Return the trapezoid's multiplier relative to the exact integral.

    ``(theta/2) cot(theta/2)``, with its series ``1 - theta^2/12 - theta^4/720`` below
    ``1e-4``, where the closed form loses digits.

    Parameters
    ----------
    theta : numpy.ndarray
        ``omega dt``, in ``[0, pi]``.

    Returns
    -------
    numpy.ndarray
        The multiplier, the shape of `theta`; one at zero and zero at Nyquist.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.plasma_integration import trapezoid_multiplier
    >>> f"{trapezoid_multiplier(np.array(np.pi / 3)):.6f}"
    '0.906900'
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(theta)
    small = theta < _SERIES_BELOW
    out[small] = 1.0 - theta[small] ** 2 / 12.0 - theta[small] ** 4 / 720.0
    half = theta[~small] / 2.0
    out[~small] = half / np.tan(half)
    return out


def design_running_integral(
    grid: TimeGrid, policy: IntegrationPolicy
) -> RunningIntegral:
    """Return the running-integral rule a plasma column on `grid` uses.

    The only producer of a band rule. Its taps depend on the fine sample spacing and the
    upper edge of the apodised window alone, so the design is cached on those two
    numbers and every problem on one grid shares one rule.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The field grid; its ``dt_fine`` and ``omega_max_window`` set ``theta_b``.
    policy : {'band', 'trapezoid'}
        The rule requested by the plasma responses.

    Returns
    -------
    RunningIntegral
        The trapezoid, or the certified band design with the smallest ``K``.

    Raises
    ------
    TranslationError
        If no design up to :data:`MAX_HALF_WIDTH` meets :data:`DESIGN_TOLERANCE` and
        :data:`ABOVE_BAND_BOUND`, which happens below an oversampling factor of about
        3.55; the message names the achieved error and the two ways out.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> from solprop.translate.plasma_integration import design_running_integral
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> rule = design_running_integral(grid, "band")
    >>> rule.half_width, rule.certified_error < 1e-5
    (4, True)
    """
    if policy == "trapezoid":
        return RunningIntegral.trapezoid()
    # The annotation refuses another rule first wherever the annotations are checked
    # at run time, as in the test suite.
    if policy != "band":  # pragma: no cover
        msg = f"integration must be 'band' or 'trapezoid', got {policy!r}"
        raise TranslationError(msg)
    return _design(float(grid.dt_fine), float(grid.omega_max_window))


def integration_record(rule: RunningIntegral) -> dict[str, object]:
    """Return the record of a running-integral rule a saved run carries (SEP 18, Q9).

    Every field of the rule at full precision -- the taps as Python floats, which
    survive the JSON round trip bit for bit -- with the derived ``theta_b`` and ``K``,
    so that a saved file says how its
    plasma was integrated without being translated again, and a resume that would
    integrate differently is refused by the configuration hash.

    Parameters
    ----------
    rule : solprop.nonlinear.ionisation.RunningIntegral
        The problem's rule.

    Returns
    -------
    dict
        The record, plain Python values only.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import RunningIntegral
    >>> from solprop.translate.plasma_integration import integration_record
    >>> integration_record(RunningIntegral.trapezoid())["policy"]
    'trapezoid'
    """
    theta_b = (
        None if rule.dt is None or rule.band_edge is None else rule.dt * rule.band_edge
    )
    return {
        "policy": rule.policy,
        "half_width": rule.half_width,
        "taps": tuple(float(tap) for tap in rule.taps),
        "dt": rule.dt,
        "band_edge": rule.band_edge,
        "theta_b": theta_b,
        "certified_error": rule.certified_error,
        "design_tolerance": DESIGN_TOLERANCE,
        "boundary": rule.boundary,
        "exposure": rule.exposure,
    }


@functools.lru_cache(maxsize=32)
def _design(dt: float, band_edge: float) -> RunningIntegral:
    """Design and certify the band rule for one sample spacing and band edge.

    Parameters
    ----------
    dt : float
        Fine sample spacing, s.
    band_edge : float
        Upper edge of the apodised window, rad/s.

    Returns
    -------
    RunningIntegral
        The band rule with the smallest admitted ``K``.

    Raises
    ------
    TranslationError
        If no admitted design exists.
    """
    theta_b = band_edge * dt
    if not 0.0 < theta_b < math.pi:
        msg = (
            f"the band edge {band_edge:.6g} rad/s lies at or above the fine grid's "
            f"Nyquist frequency (theta_b = {theta_b:.4g}); the band rule needs a finer "
            "dt, or use integration='trapezoid'"
        )
        raise TranslationError(msg)
    best = math.inf
    for half in range(MAX_HALF_WIDTH + 1):
        taps, condition = _fit(half, theta_b)
        if condition > CONDITION_LIMIT:
            msg = (
                f"the band-rule fit with K = {half} is ill-conditioned "
                f"({condition:.3g}) at theta_b = {theta_b:.4g}; use "
                "integration='trapezoid'"
            )
            raise TranslationError(msg)
        in_band = _certified_max(
            lambda theta, c=taps: _composite(c, theta) - 1.0, 0.0, theta_b
        )
        above = _certified_max(
            lambda theta, c=taps: _composite(c, theta), theta_b, math.pi
        )
        best = min(best, in_band)
        if in_band <= DESIGN_TOLERANCE and above <= ABOVE_BAND_BOUND:
            logger.debug(
                "band running integral: theta_b %.6g, K %d, max|H-1| %.3g, "
                "max|H| above %.6g, condition %.3g",
                theta_b,
                half,
                in_band,
                above,
                condition,
            )
            return RunningIntegral(
                "band", taps, dt=dt, band_edge=band_edge, certified_error=in_band
            )
    msg = (
        f"no band-rule design with K <= {MAX_HALF_WIDTH} reaches |H - 1| <= "
        f"{DESIGN_TOLERANCE:g} at theta_b = {theta_b:.4g} (best {best:.3g}); use a "
        "finer dt (a larger oversampling factor) or integration='trapezoid'"
    )
    raise TranslationError(msg)


def _fit(half: int, theta_b: float) -> tuple[Float[np.ndarray, " n_taps"], float]:
    """Fit ``C`` to ``1/rho`` on ``[0, theta_b]`` under ``C(0) = 1``.

    ``C = 1 + 2 sum_k c_k (cos k theta - 1)`` holds the constraint for any ``c_k``,
    which leaves a ``K``-unknown linear least-squares problem. Its columns are scaled to
    unit norm before the SVD-based solve.

    Parameters
    ----------
    half : int
        ``K``, the number of side taps.
    theta_b : float
        The band edge in ``omega dt``.

    Returns
    -------
    taps : numpy.ndarray
        ``(c_0, ..., c_K)``, with ``c_0 = 1 - 2 sum c_k``.
    condition : float
        The scaled design matrix's condition number; one when ``K <= 1``.
    """
    if half == 0:
        return np.ones(1), 1.0
    theta = np.linspace(0.0, theta_b, FIT_NODES)
    k = np.arange(1, half + 1)
    design = 2.0 * (np.cos(np.outer(theta, k)) - 1.0)
    scale = np.linalg.norm(design, axis=0)
    scaled = design / scale
    target = 1.0 / trapezoid_multiplier(theta) - 1.0
    side, *_ = np.linalg.lstsq(scaled, target, rcond=None)
    side = side / scale
    condition = float(np.linalg.cond(scaled))
    return np.concatenate([[1.0 - 2.0 * side.sum()], side]), condition


def _composite(
    taps: Float[np.ndarray, " n_taps"], theta: Float[np.ndarray, " n"]
) -> Float[np.ndarray, " n"]:
    """Return ``H = C rho``, the filtered trapezoid's multiplier.

    Parameters
    ----------
    taps : numpy.ndarray
        ``(c_0, ..., c_K)``.
    theta : numpy.ndarray
        ``omega dt``.

    Returns
    -------
    numpy.ndarray
        The composite multiplier relative to the exact integral.
    """
    theta = np.asarray(theta, dtype=np.float64)
    response = np.full_like(theta, taps[0])
    for k in range(1, taps.size):
        response = response + 2.0 * taps[k] * np.cos(k * theta)
    return response * trapezoid_multiplier(theta)


def _certified_max(
    function: Callable[[np.ndarray], np.ndarray], lower: float, upper: float
) -> float:
    """Return the maximum of ``|function|`` on ``[lower, upper]``, refined.

    Dense samples at :data:`CERTIFICATION_SAMPLES` points independent of the fitting
    nodes, then a bounded scalar search about every sampled local maximum that could
    exceed half the sampled maximum: between samples this fine an analytic function
    rises above its sampled peak by orders of magnitude less than that, so the smaller
    peaks cannot decide the result and refining them would spend the search on rounding
    noise where ``H`` is flat.

    Parameters
    ----------
    function : callable
        Vectorised, from ``theta`` to a real array.
    lower, upper : float
        The interval.

    Returns
    -------
    float
        The certified maximum of the absolute value.
    """
    theta = np.linspace(lower, upper, CERTIFICATION_SAMPLES)
    values = np.abs(function(theta))
    best = float(values.max())
    inner = values[1:-1]
    peaks = np.flatnonzero(
        (inner >= values[:-2]) & (inner >= values[2:]) & (inner >= 0.5 * best)
    )
    for index in peaks + 1:
        result = minimize_scalar(
            lambda x: -abs(float(function(np.array([x]))[0])),
            bounds=(theta[index - 1], theta[index + 1]),
            method="bounded",
            options={"xatol": CERTIFICATION_XATOL},
        )
        best = max(best, -float(result.fun))
    return best

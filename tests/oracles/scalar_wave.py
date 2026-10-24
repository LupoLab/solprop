# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solve the one-dimensional scalar wave equation through a real index profile.

``E'' + q(z)^2 E = 0`` with q real and positive on ``[start, end]`` and constant
outside it. The field is split exactly into directional amplitudes,

    E = q^(-1/2) (a e^(i theta) + b e^(-i theta)),
    E' = i q^(1/2) (a e^(i theta) - b e^(-i theta)),    theta' = q,

which turns the equation into ``a' = c b e^(-2 i theta)``,
``b' = c a e^(2 i theta)``, ``c = q'/(2q)``. No term is dropped: this is the
full scalar problem, not its forward approximation. A declared jump of q is
crossed by the exact matching of E and E' (a hyperbolic transfer), and smooth
pieces are integrated by an explicit eighth-order Runge-Kutta method at a
tight tolerance, backwards from the exit where nothing comes in
(``a = 1, b = 0``). Each piece is integrated in variables relative to its
entrance amplitude, so a weak reflection or a small forward correction is
resolved relative to itself rather than to the unit amplitude.

The reflectance is ``|b/a|^2`` at the entrance and the forward ratio is
``t_exact / t_WKB = a(end) / a(start)``: the adiabatic (forward-only) model
keeps ``a`` constant. Each result carries a refinement estimate, the change
when the tolerance is tightened a hundredfold.

This module does not import the production solver; it is a test witness for
the adiabatic certificate of :mod:`solprop.adiabatic`.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Profile:
    """A real positive wave number on ``[start, end]``, constant outside it.

    Attributes
    ----------
    q : callable
        ``q(z)`` in rad/m for a float z, positive. At a jump its value is either
        side's; the witness evaluates each side a float away from the jump.
    log_slope : callable
        ``q'(z) / q(z)`` in 1/m on the smooth pieces; zero for piecewise-constant q.
    start, end : float
        The path in m.
    jumps : tuple of float
        Positions in m strictly inside the path where q is discontinuous.
    kinks : tuple of float
        Positions where ``log_slope`` is discontinuous; integration splits there.
    """

    q: Callable[[float], float]
    log_slope: Callable[[float], float]
    start: float
    end: float
    jumps: tuple[float, ...] = ()
    kinks: tuple[float, ...] = ()


@dataclass(frozen=True)
class Scattering:
    """The exact linear scattering of a unit transmitted flux.

    Attributes
    ----------
    reflectance : float
        Reflected over incident power, ``|b/a|^2`` at the entrance.
    reflectance_error : float
        Refinement estimate of the absolute error of `reflectance`.
    forward_ratio : complex
        ``t_exact / t_WKB``, the transmitted amplitude over the forward model's.
    forward_ratio_error : float
        Refinement estimate of the absolute error of `forward_ratio`.
    flux_residual : float
        ``abs(|a|^2 - |b|^2 - 1)`` at the entrance, which the exact problem
        conserves.
    steps : int
        Accepted integration steps of the finer solve.
    """

    reflectance: float
    reflectance_error: float
    forward_ratio: complex
    forward_ratio_error: float
    flux_residual: float
    steps: int = field(default=0)

    @property
    def forward_error(self) -> float:
        """``|t_exact / t_WKB - 1|``, the forward model's relative field error."""
        return abs(self.forward_ratio - 1.0)


def _cross_jump(a, b, theta, q_before, q_after):
    """Carry (a, b) from just after a jump to just before it."""
    r = math.sqrt(q_after / q_before)
    plus, minus = 0.5 * (1.0 / r + r), 0.5 * (1.0 / r - r)
    phase = complex(math.cos(2.0 * theta), math.sin(2.0 * theta))
    return plus * a + minus * b / phase, minus * a * phase + plus * b


def _segment(profile, lower, upper, a, b, theta, rtol):  # noqa: PLR0913, PLR0917 - piece and state
    """Integrate one smooth piece backwards from ``upper`` to ``lower``."""
    # Relative variables: a = a_s (1 + alpha), b = a_s beta, theta = theta_s + phi.
    beta0 = b / a
    probe = np.linspace(lower, upper, 257)
    coupling = max(abs(profile.log_slope(z)) / 2.0 for z in probe)
    q_low = min(profile.q(z) for z in probe)
    scale_beta = max(abs(beta0), coupling / q_low, 1e-300)
    scale_alpha = max(scale_beta * coupling * (upper - lower), 1e-300)
    phase_scale = max(profile.q(z) for z in probe) * (upper - lower)

    def rhs(z, y):
        c = 0.5 * profile.log_slope(z)
        if c == 0.0:
            return [0.0, 0.0, 0.0, 0.0, profile.q(z)]
        angle = 2.0 * (theta + y[4])
        e = complex(math.cos(angle), math.sin(angle))
        alpha = complex(y[0], y[1])
        beta = complex(y[2], y[3])
        da = c * beta / e
        db = c * (1.0 + alpha) * e
        return [da.real, da.imag, db.real, db.imag, profile.q(z)]

    atol = (
        rtol
        * 1e-3
        * np.array(
            [scale_alpha, scale_alpha, scale_beta, scale_beta, max(phase_scale, 1.0)]
        )
    )
    solution = solve_ivp(
        rhs,
        (upper, lower),
        [0.0, 0.0, beta0.real, beta0.imag, 0.0],
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"scalar-wave integration failed: {solution.message}")
    y = solution.y[:, -1]
    alpha = complex(y[0], y[1])
    beta = complex(y[2], y[3])
    return a * (1.0 + alpha), a * beta, theta + y[4], solution.t.size


def _solve_once(profile, rtol):
    """Integrate the whole path once at one tolerance."""
    edges = sorted({profile.start, profile.end, *profile.jumps, *profile.kinks})
    jumps = set(profile.jumps)
    a, b, theta, steps = 1.0 + 0.0j, 0.0j, 0.0, 0
    for lower, upper in reversed(list(itertools.pairwise(edges))):
        a, b, theta, taken = _segment(profile, lower, upper, a, b, theta, rtol)
        steps += taken
        if lower in jumps:
            q_before = profile.q(math.nextafter(lower, -math.inf))
            q_after = profile.q(lower)
            a, b = _cross_jump(a, b, theta, q_before, q_after)
    return a, b, steps


def scatter(profile: Profile, rtol: float = 1e-11) -> Scattering:
    """Solve the scattering problem and estimate its error by refinement.

    Parameters
    ----------
    profile : Profile
        The path.
    rtol : float, optional
        Relative tolerance of the coarser solve; the finer one uses ``rtol/100``.

    Returns
    -------
    Scattering
        The finer solve's values, with the coarse-fine differences as errors.
    """
    coarse_a, coarse_b, _ = _solve_once(profile, rtol)
    a, b, steps = _solve_once(profile, rtol / 100.0)
    rho, coarse_rho = b / a, coarse_b / coarse_a
    delta = abs(rho - coarse_rho)
    ratio, coarse_ratio = 1.0 / a, 1.0 / coarse_a
    return Scattering(
        reflectance=abs(rho) ** 2,
        reflectance_error=2.0 * abs(rho) * delta + delta**2,
        forward_ratio=ratio,
        forward_ratio_error=abs(ratio - coarse_ratio),
        flux_residual=abs(abs(a) ** 2 - abs(b) ** 2 - 1.0),
        steps=steps,
    )


def fresnel_step(q_before: float, q_after: float) -> tuple[float, float]:
    """Return the exact reflectance and forward error of a single abrupt step.

    Parameters
    ----------
    q_before, q_after : float
        The wave numbers on either side, positive.

    Returns
    -------
    tuple of float
        ``((q1 - q0)/(q1 + q0))^2`` and ``1 - sech(ln(q1/q0)/2)``: with unit
        transmitted flux the incident amplitude is ``cosh`` of the step's half
        log ratio.
    """
    reflectance = ((q_after - q_before) / (q_after + q_before)) ** 2
    half = 0.5 * math.log(q_after / q_before)
    return reflectance, 1.0 - 1.0 / math.cosh(half)


def epstein_reflectance(q_before: float, q_after: float, scale: float) -> float:
    """Return the exact reflectance of an Epstein transition of ``q^2``.

    Parameters
    ----------
    q_before, q_after : float
        The asymptotic wave numbers.
    scale : float
        ``s`` in ``q^2 = q0^2 + (q1^2 - q0^2) / (1 + exp(-z/s))``, in m.

    Returns
    -------
    float
        ``[sinh(pi s (q1 - q0)) / sinh(pi s (q1 + q0))]^2`` (Epstein 1930; the
        transmission-line form of Budden, *The Propagation of Radio Waves*, 1985).
    """
    x = math.pi * scale * abs(q_after - q_before)
    y = math.pi * scale * (q_after + q_before)
    # sinh(x)/sinh(y) = exp(x - y) (1 - exp(-2x)) / (1 - exp(-2y)), without overflow.
    return (math.exp(x - y) * -math.expm1(-2.0 * x) / -math.expm1(-2.0 * y)) ** 2

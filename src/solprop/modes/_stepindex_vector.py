# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recover bound Maxwell fields analytically from the existing step-index roots."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from jaxtyping import Complex, Float
from scipy.integrate import quad
from scipy.special import jv, jvp, kve

from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.stepindex import PROFILE_MIN_W, StepIndexMode, _material
from solprop.physdata.glasses import GlassIndex

__all__: list[str] = []


def _radial(
    order: int, argument: Float[np.ndarray, " n_points"], edge: float, inside: bool
) -> tuple[
    Float[np.ndarray, " n_points"],
    Float[np.ndarray, " n_points"],
    Float[np.ndarray, " n_points"],
]:
    """Evaluate the scaled radial function, slope and axis limit.

    Parameters
    ----------
    order : int
        Non-negative angular Bessel order.
    argument : numpy.ndarray
        Dimensionless Bessel arguments, shape (n_points,).
    edge : float
        Dimensionless Bessel argument at the interface.
    inside : bool
        Whether to use the regular core solution.

    Returns
    -------
    tuple of numpy.ndarray
        Dimensionless radial function, argument derivative and divided value.
    """
    if inside:
        value = jv(order, argument) / jv(order, edge)
        slope = jvp(order, argument) / jv(order, edge)
        # J_n(x)/x has a finite nonzero axis limit only for n=1;
        # the angular derivative multiplying it vanishes for n=0.
        divided = np.divide(
            value, argument, out=np.zeros_like(value), where=argument != 0
        )
        divided = np.where(
            argument == 0, 0.5 / jv(order, edge) if order == 1 else 0.0, divided
        )
    else:
        factor = np.exp(edge - argument) / kve(order, edge)
        value = kve(order, argument) * factor
        slope = -(kve(abs(order - 1), argument) + kve(order + 1, argument)) * factor / 2
        divided = value / argument
    return np.asarray(value), np.asarray(slope), np.asarray(divided)


@dataclass(frozen=True)
class _FieldParameters:
    """Hold one lossless vector solution at a fixed frequency and position.

    Attributes
    ----------
    n : int
        Non-negative azimuthal order.
    parity : str
        Angular parity, ``even`` or ``odd``.
    radius : float
        Core radius in m.
    k, beta : float
        Vacuum and modal wavenumbers in rad/m.
    u, w : float
        Dimensionless core and cladding transverse parameters.
    core, clad : float
        Real core and cladding indices.
    electric_amplitude, magnetic_amplitude : float
        Longitudinal amplitudes in the existing unnormalised Maxwell solution.
    """

    n: int
    parity: str
    radius: float
    k: float
    beta: float
    u: float
    w: np.float64
    core: float
    clad: float
    electric_amplitude: float
    magnetic_amplitude: float | np.float64


def _field_parameters(mode: StepIndexMode, omega: float, z: float) -> _FieldParameters:
    """Solve the position-independent parameters of one transverse field.

    Parameters
    ----------
    mode : StepIndexMode
        Existing bound lossless mode.
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Axial position in m within the declared path.

    Returns
    -------
    _FieldParameters
        Immutable scalar parameters for this frequency and axial position.
    """
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(z):
        raise ModeError("omega must be positive and z finite")
    frequency = np.array([omega], dtype=float)
    if np.any(mode._loss_values(frequency)[0] != 0):
        raise ModeError(
            "leaky or attenuating E/H modes are unqualified: bound lossless fields only"
        )
    if any(
        isinstance(material, GlassIndex)
        and np.any(np.imag(material.index(frequency)) != 0)
        for material in (mode._core, mode._cladding)
    ):
        raise ModeError("absorbing material E/H modes are unqualified")
    index, transverse = mode._eigenpairs(frequency, z)
    radius = mode._radius(z)
    core = float(_material(mode._core, frequency)[0][0])
    clad = float(_material(mode._cladding, frequency)[0][0])
    k, beta, u = omega / C, float(index[0]) * omega / C, float(transverse[0])
    w = np.float64(np.sqrt((k * radius) ** 2 * (core - clad) * (core + clad) - u**2))
    if w < PROFILE_MIN_W:
        raise ModeError(
            "full-vector profile is too close to cutoff: "
            "cladding decay below PROFILE_MIN_W"
        )
    a = jvp(mode.n, u) / (u * jv(mode.n, u))
    b = -(kve(abs(mode.n - 1), w) + kve(mode.n + 1, w)) / (2 * w * kve(mode.n, w))
    electric_amplitude = 0.0 if mode.kind == "TE" else 1.0
    magnetic_amplitude = 1.0 if mode.kind == "TE" else 0.0
    if mode.n:
        magnetic_amplitude = np.float64(
            -beta / k * mode.n * (1 / u**2 + 1 / w**2) / (a + b)
        )
    return _FieldParameters(
        mode.n,
        mode.parity,
        radius,
        k,
        beta,
        u,
        w,
        core,
        clad,
        electric_amplitude,
        magnetic_amplitude,
    )


def _fields(
    mode: StepIndexMode,
    points: Float[np.ndarray, "n_points 2"],
    omega: float,
    z: float,
    *,
    exterior: bool | None = None,
) -> tuple[Complex[np.ndarray, "n_points 3"], Complex[np.ndarray, "n_points 3"]]:
    """Recover full Maxwell fields from analytic longitudinal amplitudes.

    Parameters
    ----------
    mode : StepIndexMode
        Existing bound lossless mode.
    points : numpy.ndarray
        Explicit radial fractions or Cartesian points as required by this helper.
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Position in m within the declared path.
    exterior : bool or None, optional
        Force one side for an exact interface trace; None follows each point.

    Returns
    -------
    tuple of numpy.ndarray
        Unnormalised E and H, each shape (n_points, 3).
    """
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ModeError("points must be finite Cartesian (x, y) pairs")
    return _prepared_fields(
        _field_parameters(mode, omega, z), points, exterior=exterior
    )


def _prepared_fields(
    parameters: _FieldParameters,
    points: Float[np.ndarray, "n_points 2"],
    *,
    exterior: bool | None = None,
) -> tuple[Complex[np.ndarray, "n_points 3"], Complex[np.ndarray, "n_points 3"]]:
    """Evaluate one solved mode at Cartesian positions without solving it again.

    Parameters
    ----------
    parameters : _FieldParameters
        Immutable solution at the requested frequency and axial position.
    points : numpy.ndarray
        Cartesian positions in m, shape (n_points, 2).
    exterior : bool or None, optional
        Force one side for an exact interface trace; None follows each point.

    Returns
    -------
    tuple of numpy.ndarray
        Unnormalised E and H, each shape (n_points, 3).
    """
    radius, k, beta = parameters.radius, parameters.k, parameters.beta
    u, w, core, clad = parameters.u, parameters.w, parameters.core, parameters.clad
    electric_amplitude = parameters.electric_amplitude
    magnetic_amplitude = parameters.magnetic_amplitude
    r, theta = (
        np.hypot(points[:, 0], points[:, 1]),
        np.arctan2(points[:, 1], points[:, 0]),
    )
    if parameters.n == 0:
        angular_e, angular_h = np.ones_like(r), np.ones_like(r)
        derivative_e, derivative_h = np.zeros_like(r), np.zeros_like(r)
    elif parameters.parity == "even":
        angular_e, angular_h = (
            np.cos(parameters.n * theta),
            np.sin(parameters.n * theta),
        )
        derivative_e, derivative_h = -parameters.n * angular_h, parameters.n * angular_e
    else:
        angular_e, angular_h = (
            np.sin(parameters.n * theta),
            -np.cos(parameters.n * theta),
        )
        derivative_e, derivative_h = -parameters.n * angular_h, parameters.n * angular_e
    electric, magnetic = np.zeros((len(r), 3), complex), np.zeros((len(r), 3), complex)
    for inside in (True, False):
        mask = (
            (r <= radius) if exterior is None else np.full(r.shape, not exterior)
        ) == inside
        rr, tt = r[mask], theta[mask]
        scale = (u if inside else w) / radius
        q2 = scale**2 if inside else -(scale**2)
        eps = core**2 if inside else clad**2
        value, slope, divided = _radial(
            parameters.n, scale * rr, u if inside else w, inside
        )
        er = (
            beta * electric_amplitude * scale * slope * angular_e[mask]
            + k * magnetic_amplitude * scale * divided * derivative_h[mask]
        ) / q2
        et = (
            beta * electric_amplitude * scale * divided * derivative_e[mask]
            - k * magnetic_amplitude * scale * slope * angular_h[mask]
        ) / q2
        hr = (
            beta * magnetic_amplitude * scale * slope * angular_h[mask]
            - k * eps * electric_amplitude * scale * divided * derivative_e[mask]
        ) / q2
        ht = (
            beta * magnetic_amplitude * scale * divided * derivative_h[mask]
            + k * eps * electric_amplitude * scale * slope * angular_e[mask]
        ) / q2
        electric[mask] = np.column_stack(
            (
                er * np.cos(tt) - et * np.sin(tt),
                er * np.sin(tt) + et * np.cos(tt),
                1j * electric_amplitude * value * angular_e[mask],
            )
        )
        magnetic[mask] = (
            EPS0
            * C
            * np.column_stack(
                (
                    hr * np.cos(tt) - ht * np.sin(tt),
                    hr * np.sin(tt) + ht * np.cos(tt),
                    1j * magnetic_amplitude * value * angular_h[mask],
                )
            )
        )
    return electric, magnetic


_TAIL_EPSABS_FRACTION = 1e-15
"""Absolute quadrature floor for the exterior power piece, as a fraction of the core.

A bound mode's exterior power falls exponentially -- 1e-176 of the total for a
25 um fibre integrated from two core radii -- and asking for 2e-11 relative
accuracy on such a number is asking the adaptive rule to subdivide forever.
1e-15 of the core piece is three orders below the 1e-12 discarded-power
certificate the provider's node rule asserts, so the floor can never be what
makes that certificate pass, and it binds only where the tail is already
negligible: measured, it changes no returned power and leaves the largest
relative error of the total at 3e-12.
"""

_POWER_CACHE_SIZE = 4096
"""Distinct (mode, omega, z, start) power integrals kept.

The key holds the mode, so an unbounded cache pins every mode ever normalised
for the life of the process, along with its material objects. A dense modal
table is the heaviest caller: 17 frequency knots times 5 positions times a
handful of channels, with one full-domain and one tail integral each, is a few
hundred entries, so 4096 keeps every repeat within one table and several tables
besides.
"""


@lru_cache(maxsize=_POWER_CACHE_SIZE)
def _power(
    mode: StepIndexMode, omega: float, z: float, start: float = 0.0
) -> tuple[float, float]:
    """Integrate signed full-domain or exterior-tail Poynting power.

    Parameters
    ----------
    mode : StepIndexMode
        Existing bound lossless mode.
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Position in m within the declared path.
    start : float, optional
        Lower integration radius divided by the core radius.

    Returns
    -------
    tuple of float
        Power in W and the adaptive absolute integration error in W.
    """
    # The quadrature varies radius alone. Resolve the eigenpair and material
    # values once for this integral, without caching a mutable profile's inputs.
    parameters = _field_parameters(mode, omega, z)
    radius = parameters.radius
    angles = np.arange(2 * mode.n + 3) * 2 * np.pi / (2 * mode.n + 3)

    def integrand(x: float) -> float:
        """Evaluate the angularly integrated radial flux.

        Parameters
        ----------
        x : float
            Radius divided by the core radius.

        Returns
        -------
        float
            Dimensionless-radius integrand before the squared-radius factor.
        """
        points = radius * x * np.column_stack((np.cos(angles), np.sin(angles)))
        e, h = _prepared_fields(parameters, points)
        return float(
            np.mean(np.real(e[:, 0] * h[:, 1].conj() - e[:, 1] * h[:, 0].conj()))
            * np.pi
            * x
        )

    # Dimensionless radius avoids an absolute SI-area tolerance hiding missing flux,
    # but the integrand still carries the field normalisation, so a fixed epsabs is
    # not scale free: 1e-12 was 4.8e-11 of the 250 nm strand's power and 1.3e-14 of
    # the 25 um fibre's, the first looser than the epsrel beside it. The core piece
    # therefore asks for relative accuracy alone; only the exponentially small
    # exterior piece gets an absolute floor, and that floor is a fixed fraction of
    # the core piece, so it means the same thing at every scale. The floor sits
    # _TAIL_EPSABS_FRACTION below the power, three orders under the 1e-12 tail
    # certificate of the provider's node rule, which reads the error returned here.
    # A tail integrated on its own (start >= 1) is that certificate's own integral
    # and gets no floor: it must be resolved relative to its own tiny value.
    if start < 1:
        core, core_error = quad(
            integrand, start, 1.0, epsabs=0.0, epsrel=2e-11, limit=150
        )
        tail, tail_error = quad(
            integrand,
            1.0,
            np.inf,
            epsabs=_TAIL_EPSABS_FRACTION * abs(core),
            epsrel=2e-11,
            limit=150,
        )
        return radius**2 * (core + tail), radius**2 * (core_error + tail_error)
    value, error = quad(integrand, start, np.inf, epsabs=0.0, epsrel=2e-11, limit=150)
    return radius**2 * value, radius**2 * error

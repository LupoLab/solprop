# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The transverse boundary of a free-space run: its specification and its design.

SEP 40. A free-space run (radial, Cartesian, line or planar) has a wall or a wrap at
its grid's outer size. By default it carries no real-space collar: a
transverse-wavenumber *safety ramp* removes what reaches the grid's outer wavenumbers,
and the boundary monitor (:mod:`solprop.monitor`) watches the edge. A
:class:`SpatialAbsorber` asks for a real-space collar outside the requested aperture as
well, for fast content the run declares; translation resolves it into a
:class:`SpatialAbsorberDesign`.

The design is a complex absorbing potential [Kosloff and Kosloff 1986; Muga et al.
2004]. For one frequency, with :math:`k_z(\kappa) = \sqrt{k^2 - \kappa^2}` and a collar
of power rate :math:`\alpha(x) = \alpha_0 P(x/w)` over a width :math:`w`, a component of
transverse wavenumber :math:`\kappa` crossing the collar and coming back is attenuated
in field by the local WKB integral with the exact symbol (SEP 40 section 4.2),

.. math::

    \Gamma(\kappa; \alpha_0) = 2 w \int_0^1 \left|\operatorname{Im} q\right| ds,
    \qquad q^2 = \kappa^2 + i k_z \alpha_0 P(s) + \alpha_0^2 P(s)^2 / 4,

which the phase-0 calibration on the discrete radial, Fourier and slab operators
matched to between -1.3 % and +0.04 %. The strength :math:`\alpha_0` is its root at the
protected band's edge :math:`\kappa_p`, :math:`\Gamma = \Gamma_\min`, over every
frequency of the requested band; the run length does not enter it. Below the reflection
threshold :math:`\kappa_\mathrm{lo}` the collar reflects rather than absorbs (section
4.3), with the measured law
:math:`\kappa_\mathrm{lo} w = 7.36\,(k_\max \alpha_0 w^2)^{0.182}` for the Planck
profile :math:`P = 1 - W`.

``alpha`` is a power coefficient and attenuations are field nepers, as in
:mod:`solprop.boundaries` (``docs/dev/conventions.md`` section 2).

The manual page is ``docs/guide/free-space/model.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from jaxtyping import Float
from scipy.optimize import brentq

from solprop.exceptions import GridError
from solprop.grid import planck_taper

__all__ = [
    "CROSSING_ADAPTIVE",
    "CROSSING_FIXED",
    "DAMPING_BOUND",
    "DEFAULT_ATTENUATION",
    "LAUNCH_BAND_FACTOR",
    "PROFILE_FLOOR",
    "PROTECTED_CUTOFF",
    "RAMP_K_FLOOR",
    "RAMP_MARGIN",
    "THRESHOLD_LAW",
    "SpatialAbsorber",
    "SpatialAbsorberDesign",
    "collar_attenuation",
    "collar_profile",
    "collar_strength",
    "crossing_limit",
    "reflection_threshold",
    "rms_wavenumber",
    "safety_ramp_strength",
]

DEFAULT_ATTENUATION: Final = math.log(5e3)
"""The collar's returned attenuation at the protected band's edge, field nepers.

``log(5e3)``: a returned field below ``2e-4`` of the incident, one of the four equal
shares of the ``1e-3`` aperture budget (SEP 40 sections 4.8 and 7.1)."""

THRESHOLD_LAW: Final = (7.36, 0.182)
"""``(A, gamma)`` of the threshold ``kappa_lo w = A (k_max alpha0 w^2)^gamma``.

Measured on the discrete operators for the Planck profile, for the wavenumber above
which ``|r| <= 2e-4`` (SEP 40 section 4.3, Part II). The exponent is the Born onset
law's, not the simple estimate's one third."""

PROFILE_FLOOR: Final = 22.3
"""The smallest ``kappa_p w`` at which the Planck collar's reflection at its band edge
falls below ``2e-4``; a narrower design is refused (SEP 40 section 4.3)."""

LAUNCH_BAND_FACTOR: Final = 3.0
"""``c_b``: the launch-derived band's edge in units of the launch's rms wavenumber
(SEP 40 Q5, measured in phase 0)."""

PROTECTED_CUTOFF: Final = 0.9
"""A protected band reaching beyond this fraction of ``k`` at any requested frequency is
refused: near cutoff the ray picture fails and the collar's flux gain is unbounded."""

RAMP_MARGIN: Final = 2.0
"""Nepers added to the safety ramp's attenuation, measured: inside the ramp the collar's
own wavenumber broadening defeats the additive estimate by up to 1.7 Np."""

RAMP_K_FLOOR: Final = 0.44
"""The floor on ``k_s`` as a fraction of ``k``: ``k_z`` at ``0.9 k``, the protected
band's limit, so that the ramp's strength stays finite as its band nears cutoff."""

CROSSING_ADAPTIVE: Final = 0.5
"""``c_x`` of the adaptive ``dz_max`` clamp: a step crosses at most half the collar."""

CROSSING_FIXED: Final = 0.15
"""``c_x`` a fixed in-stage schedule is checked at, measured under Runge-Kutta."""

DAMPING_BOUND: Final = 1.3
"""The largest ``alpha0 h / 2`` at which a fixed in-stage schedule damps accurately."""

_PROFILE_SAMPLES: Final = 4001
"""Samples of the profile in the WKB integral, as calibrated."""


def collar_profile(s: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
    """Return the collar's profile ``P = 1 - W`` on its normalised depth.

    The complement of the grids' Planck window (:func:`solprop.grid.planck_taper`),
    zero at and before the collar's onset ``s = 0`` and one at and beyond its far edge
    ``s = 1``, with every derivative continuous (SEP 40 section 4.5).

    Parameters
    ----------
    s : numpy.ndarray
        Depth into the collar as a fraction of its width.

    Returns
    -------
    numpy.ndarray
        The profile, in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import collar_profile
    >>> collar_profile(np.array([-1.0, 0.0, 0.5, 1.0, 2.0])).tolist()
    [0.0, 0.0, 0.5, 1.0, 1.0]
    """
    s = np.asarray(s, dtype=np.float64)
    out = np.where(s >= 1.0, 1.0, 0.0)
    inside = (s > 0.0) & (s < 1.0)
    out[inside] = 1.0 - planck_taper(s[inside], -2.0, -1.0, 0.0, 1.0)
    return out


_DEPTH = np.linspace(0.0, 1.0, _PROFILE_SAMPLES)
_PROFILE = collar_profile(_DEPTH)


def collar_attenuation(
    kappa: Float[np.ndarray, "*shape"] | float,
    k: Float[np.ndarray, "*shape"] | float,
    alpha0: float,
    width: float,
) -> Float[np.ndarray, "*shape"]:
    """Return the field attenuation of a collar's round trip, by local WKB.

    ``2 w int_0^1 |Im q| ds`` with the exact symbol
    ``q^2 = kappa^2 + i k_z alpha0 P + (alpha0 P)^2 / 4`` and ``k_z`` taken at `kappa`
    (SEP 40 section 4.2). It counts the way out and the way back, through a radial
    collar to its wall or a periodic one to its image.

    Parameters
    ----------
    kappa : numpy.ndarray or float
        Transverse wavenumbers normal to the collar, rad/m.
    k : numpy.ndarray or float
        The medium's wavenumbers ``n omega / c`` at the same points, rad/m.
    alpha0 : float
        The collar's peak power rate, 1/m.
    width : float
        Its width, m.

    Returns
    -------
    numpy.ndarray
        Field nepers, broadcast to the shape of `kappa` and `k`; zero where
        ``kappa >= k``, which the design refuses before it gets here.

    Examples
    --------
    >>> from solprop.spatial_absorber import collar_attenuation
    >>> gamma = collar_attenuation(1e5, 7.85e6, 1000.0, 300e-6)
    >>> round(float(gamma), 2)
    11.24
    """
    kappa, k = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    k_z = np.sqrt(np.maximum(k * k - kappa * kappa, 0.0))
    rate = alpha0 * _PROFILE
    q = np.asarray(
        np.sqrt(
            (kappa * kappa)[..., None]
            + 1j * k_z[..., None] * rate
            + (rate * rate / 4.0)
        ),
        dtype=np.complex128,
    )
    return np.asarray(2.0 * width * np.trapezoid(np.abs(q.imag), _DEPTH, axis=-1))


def collar_strength(
    kappa_p: Float[np.ndarray, " n"] | float,
    k: Float[np.ndarray, " n"],
    width: float,
    attenuation: float = DEFAULT_ATTENUATION,
) -> float:
    """Return the collar's peak rate that attenuates the protected band's edge enough.

    The root of ``min_omega Gamma(kappa_p(omega), omega; alpha0) = attenuation``, a
    monotone scalar root (SEP 40 section 4.2). For a declared wavenumber `kappa_p` is
    one number and the smallest ``k_z`` decides; for a declared angle it varies with
    frequency, and the root is solved over the band rather than pairing the bluest
    wavenumber with the reddest ``k_z``.

    Parameters
    ----------
    kappa_p : numpy.ndarray or float
        The protected band's edge at each frequency, rad/m, or one value for all.
    k : numpy.ndarray
        The medium's wavenumbers over the requested band, rad/m.
    width : float
        The collar's width, m.
    attenuation : float, optional
        The returned attenuation required, field nepers.

    Returns
    -------
    float
        ``alpha0``, 1/m.

    Raises
    ------
    GridError
        If a frequency has ``kappa_p >= k``, or the root cannot be bracketed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import collar_attenuation, collar_strength
    >>> k = np.array([7.85e6, 1.0e7])
    >>> alpha0 = collar_strength(1e5, k, 300e-6)
    >>> round(float(collar_attenuation(1e5, k, alpha0, 300e-6).min()), 6)
    8.517193
    """
    kappa_p, k = np.broadcast_arrays(
        np.asarray(kappa_p, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    if np.any(kappa_p >= k) or np.any(kappa_p <= 0.0):
        msg = "the protected band must lie strictly between zero and k"
        raise GridError(msg)
    if not width > 0.0 or not attenuation > 0.0:
        msg = "a collar needs a positive width and a positive attenuation"
        raise GridError(msg)
    k_z = np.sqrt(k * k - kappa_p * kappa_p)
    # The weak-loss formula, Gamma ~ k_z alpha0 w <P> / kappa with <P> = 1/2, read at
    # its worst frequency, brackets the root from below (it overstates Gamma).
    estimate = float(np.max(attenuation * kappa_p / (0.5 * k_z * width)))

    def short(alpha0: float) -> float:
        """Return how far the band's weakest attenuation exceeds the requirement.

        Parameters
        ----------
        alpha0 : float
            A trial peak rate, 1/m.

        Returns
        -------
        float
            Field nepers; negative while the collar is too weak.
        """
        reached = float(np.min(collar_attenuation(kappa_p, k, alpha0, width)))
        return reached - attenuation

    low, high = 0.5 * estimate, 2.0 * estimate
    for _ in range(60):
        if short(high) > 0.0:
            break
        high *= 2.0
    else:
        msg = "the collar's strength could not be bracketed"
        raise GridError(msg)
    while short(low) > 0.0:
        low *= 0.5
    return float(brentq(short, low, high, xtol=1e-12 * high, rtol=1e-12))


def reflection_threshold(
    alpha0: float, width: float, k: Float[np.ndarray, " n"]
) -> float:
    """Return the wavenumber above which the collar reflects less than ``2e-4``.

    ``kappa_lo w = A (k_max alpha0 w^2)^gamma`` (:data:`THRESHOLD_LAW`), with ``k_max``
    the largest ``k_z`` at ``kappa_lo`` over the requested band, found by iterating
    from the largest ``k`` (SEP 40 section 4.3).

    Parameters
    ----------
    alpha0 : float
        The collar's peak power rate, 1/m.
    width : float
        Its width, m.
    k : numpy.ndarray
        The medium's wavenumbers over the requested band, rad/m.

    Returns
    -------
    float
        ``kappa_lo``, rad/m.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import reflection_threshold
    >>> round(reflection_threshold(698.0, 300e-6, np.array([1.25e7])) * 300e-6, 2)
    24.76
    """
    a, gamma = THRESHOLD_LAW
    k = np.asarray(k, dtype=np.float64)
    k_max = float(np.max(k))
    kappa_lo = a * (k_max * alpha0 * width * width) ** gamma / width
    for _ in range(3):
        k_max = float(np.max(np.sqrt(np.maximum(k * k - kappa_lo * kappa_lo, 0.0))))
        kappa_lo = a * (k_max * alpha0 * width * width) ** gamma / width
    return kappa_lo


def safety_ramp_strength(
    kappa_half: float,
    k: Float[np.ndarray, " n"],
    length: float,
    attenuation: float = DEFAULT_ATTENUATION,
) -> float:
    """Return the safety ramp's peak rate on one axis.

    ``alpha_s = 2 (Gamma_min + 2) kappa_half / (k_s ell)``: content in the ramp's upper
    half, where the raised cosine is at least half its peak, is attenuated by
    ``Gamma_min + 2`` nepers over the distance ``ell`` in which it would cross and come
    back, with ``k_s`` the smallest over the band of
    ``max(k_z(kappa_half), 0.44 k)`` (SEP 40 section 4.4).

    Parameters
    ----------
    kappa_half : float
        The ramp's midpoint wavenumber, rad/m.
    k : numpy.ndarray
        The medium's wavenumbers over the requested band, rad/m.
    length : float
        ``ell``: the collar's width when there is one, else the requested size, m.
    attenuation : float, optional
        ``Gamma_min``, field nepers.

    Returns
    -------
    float
        The ramp's peak power rate, 1/m.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import safety_ramp_strength
    >>> round(safety_ramp_strength(6e5, np.array([7.85e6]), 300e-6))
    5375
    """
    k = np.asarray(k, dtype=np.float64)
    k_z = np.sqrt(np.maximum(k * k - kappa_half * kappa_half, 0.0))
    k_s = float(np.min(np.maximum(k_z, RAMP_K_FLOOR * k)))
    return 2.0 * (attenuation + RAMP_MARGIN) * kappa_half / (k_s * length)


def crossing_limit(
    kappa_p: Float[np.ndarray, " n"] | float,
    k: Float[np.ndarray, " n"],
    width: float,
    factor: float = CROSSING_ADAPTIVE,
) -> float:
    """Return the longest step that cannot carry the protected band across the collar.

    ``h <= c_x w min_omega k_z(kappa_p) / kappa_p`` (SEP 40 section 4.7): a component
    moves ``h kappa / k_z`` across over a step.

    Parameters
    ----------
    kappa_p : numpy.ndarray or float
        The protected band's edge per frequency, or one value, rad/m.
    k : numpy.ndarray
        The medium's wavenumbers over the requested band, rad/m.
    width : float
        The collar's width, m.
    factor : float, optional
        ``c_x``: :data:`CROSSING_ADAPTIVE` for the adaptive clamp,
        :data:`CROSSING_FIXED` for a fixed schedule.

    Returns
    -------
    float
        The step, m.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import crossing_limit
    >>> round(crossing_limit(1e5, np.array([7.85e6]), 300e-6) * 1e3, 3)
    11.774
    """
    kappa_p, k = np.broadcast_arrays(
        np.asarray(kappa_p, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    k_z = np.sqrt(np.maximum(k * k - kappa_p * kappa_p, 0.0))
    return float(factor * width * np.min(k_z / kappa_p))


def rms_wavenumber(
    power: Float[np.ndarray, " n_columns"], wavenumber: Float[np.ndarray, " n_columns"]
) -> float:
    """Return an energy-weighted rms transverse wavenumber.

    ``sqrt(sum P kappa^2 / sum P)`` over the columns, with `power` each column's
    energy (quadrature weight times ``|E|^2``, summed over the requested band).

    Parameters
    ----------
    power : numpy.ndarray
        Each column's energy, any consistent unit.
    wavenumber : numpy.ndarray
        Each column's wavenumber along the axis, or ``|kappa_perp|`` radially, rad/m.

    Returns
    -------
    float
        The wavenumber, rad/m; zero for no energy.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial_absorber import rms_wavenumber
    >>> rms_wavenumber(np.array([1.0, 1.0]), np.array([1.0, 3.0]))
    2.23606797749979
    """
    total = float(np.sum(power))
    if not total > 0.0:
        return 0.0
    return math.sqrt(float(np.sum(power * wavenumber * wavenumber)) / total)


@dataclass(frozen=True)
class SpatialAbsorber:
    """The transverse boundary a free-space run asks for (SEP 40 section 5.2).

    A specification, resolved at translation against the grid, the medium and the
    launch into a :class:`SpatialAbsorberDesign`. By default (``real_space=True``) it
    asks for a real-space collar in the room the grid has outside its requested size
    (``collar_fraction``), as well as the safety ramp; ``real_space=False`` asks for
    the ramp alone, which is how a run with no ``spatial_absorber`` is resolved.

    Attributes
    ----------
    kappa : float or None
        The protected band's edge, ``|kappa| <= kappa``, rad/m.
    angle : float or None
        The protected band as an angle in vacuum behind a planar exit face,
        ``sin(theta) = c kappa / omega`` (SEP 23's convention), rad, in (0, pi/2):
        a band that scales with frequency. Neither `kappa` nor `angle` takes the band
        from the launch, ``c_b sqrt(2) kappa_rms`` per axis
        (:data:`LAUNCH_BAND_FACTOR`).
    discard_above : float or None
        A wavenumber above which the user declares no physical light, rad/m: the
        safety ramp then starts there. Never lowered by default.
    attenuation : float
        ``Gamma_min``, the collar's returned attenuation at the band's edge, field
        nepers. Default ``log(5e3)``.
    safety_fraction : float
        The ramp's width as a fraction of each axis's largest wavenumber, in (0, 1).
        Default 0.1.
    real_space : bool
        Whether a real-space collar is asked for. Default ``True``.

    Raises
    ------
    GridError
        If both `kappa` and `angle` are given, a value is out of range, or a band is
        declared with no collar to protect it.

    Examples
    --------
    >>> from solprop.spatial_absorber import SpatialAbsorber
    >>> SpatialAbsorber(angle=25e-3).angle
    0.025
    >>> SpatialAbsorber(real_space=False, safety_fraction=0.2).real_space
    False
    """

    kappa: float | None = None
    angle: float | None = None
    discard_above: float | None = None
    attenuation: float = DEFAULT_ATTENUATION
    safety_fraction: float = 0.1
    real_space: bool = True

    def __post_init__(self) -> None:
        """Check the band, the attenuation and the ramp's width."""
        if self.kappa is not None and self.angle is not None:
            msg = "SpatialAbsorber takes kappa or angle, not both"
            raise GridError(msg)
        for name in ("kappa", "discard_above"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0.0):
                msg = f"SpatialAbsorber.{name} must be finite and positive, rad/m"
                raise GridError(msg)
        if self.angle is not None and not (
            math.isfinite(self.angle) and 0.0 < self.angle < math.pi / 2
        ):
            msg = "SpatialAbsorber.angle must lie in (0, pi/2), rad"
            raise GridError(msg)
        if not (math.isfinite(self.attenuation) and self.attenuation > 0.0):
            msg = "SpatialAbsorber.attenuation must be finite and positive, nepers"
            raise GridError(msg)
        if not (
            math.isfinite(self.safety_fraction) and 0.0 < self.safety_fraction < 1.0
        ):
            msg = "SpatialAbsorber.safety_fraction must lie in (0, 1)"
            raise GridError(msg)
        if not isinstance(self.real_space, bool):
            msg = "SpatialAbsorber.real_space must be a bool"
            raise GridError(msg)
        if not self.real_space and (self.kappa is not None or self.angle is not None):
            msg = (
                "a protected band (kappa or angle) belongs to a real-space collar; "
                "real_space=False asks for the safety ramp alone"
            )
            raise GridError(msg)
        if (
            self.discard_above is not None
            and self.kappa is not None
            and self.discard_above < self.kappa
        ):
            msg = "discard_above lies below the protected band's edge kappa"
            raise GridError(msg)
        for name in ("kappa", "angle", "discard_above"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "attenuation", float(self.attenuation))
        object.__setattr__(self, "safety_fraction", float(self.safety_fraction))

    @property
    def band_source(self) -> Literal["kappa", "angle", "launch"]:
        """Return where the protected band comes from, in order of precedence.

        Returns
        -------
        str
            ``"kappa"``, ``"angle"`` or ``"launch"``.
        """
        if self.kappa is not None:
            return "kappa"
        if self.angle is not None:
            return "angle"
        return "launch"


@dataclass(frozen=True, eq=False)
class SpatialAbsorberDesign:
    """A transverse boundary resolved at translation (SEP 40 section 5.2).

    Immutable, and part of the translated problem: derived runs share it
    (:meth:`solprop.driver.Simulation.with_input`), and the design check that belongs
    to a launch is rerun for a new one. Every derived value is recorded in the run's
    output beside the specification, not hashed with it.

    Attributes
    ----------
    specification : SpatialAbsorber
        What the run asked for.
    ramp_rate : numpy.ndarray
        The safety ramp's power rate on the spectral columns, 1/m.
    ramp_strength : float
        Its peak, 1/m, the larger of the axes'.
    kappa_s : tuple of float
        Where the ramp starts, per axis (one radially), rad/m.
    collar_rate : numpy.ndarray or None
        The real-space collar's power rate on the nodes, 1/m; ``None`` without one.
    alpha0 : float
        The collar's peak rate, 1/m; zero without one.
    width : tuple of float
        The collar's width per face's axis, m; empty without one.
    kappa_p : tuple of float
        The protected band's edge per axis at the reddest requested frequency,
        rad/m; empty without a collar.
    kappa_lo : float
        The reflection threshold of the strongest face, rad/m; zero without a collar.
    crossing_step : float
        The adaptive crossing limit, m; infinite without a collar.
    unguarded_above : float
        The wavenumber above which content could in principle cross within a clamped
        step, ``(0.75 / c_x) kappa_p``, rad/m; infinite without a collar.
    check : dict
        The design check's record (SEP 40 section 4.4): its status, the launch's
        return, the bare wall's at the same size, the frequency nodes and its wall
        time; empty until it has run, and without a collar.
    notes : dict
        Further derived values for the record: the intermediate band's report, the
        reflected QDHT columns, the launch's share in the ramp's band.
    """

    specification: SpatialAbsorber
    ramp_rate: Float[np.ndarray, " n_columns"]
    ramp_strength: float
    kappa_s: tuple[float, ...]
    collar_rate: Float[np.ndarray, " n_nodes"] | None = None
    alpha0: float = 0.0
    width: tuple[float, ...] = ()
    kappa_p: tuple[float, ...] = ()
    kappa_lo: float = 0.0
    crossing_step: float = math.inf
    unguarded_above: float = math.inf
    check: dict[str, object] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        for name in ("ramp_rate", "collar_rate"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.array(values, dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                msg = f"SpatialAbsorberDesign.{name} must be finite and nonnegative"
                raise GridError(msg)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def record(self) -> dict[str, object]:
        """Return the design's derived values for the run's record.

        Returns
        -------
        dict
            JSON-ready scalars and tuples: the ramp's start per axis and strength, and
            with a collar its peak rate, widths, band edges, threshold, crossing step
            and unguarded band, then the design check's record and the notes. The
            rates on the nodes and columns are in the operator's own record.
        """
        record: dict[str, object] = {
            "ramp_start": list(self.kappa_s),
            "ramp_strength": self.ramp_strength,
            "collar": self.has_collar,
        }
        if self.has_collar:
            record |= {
                "alpha0": self.alpha0,
                "width": list(self.width),
                "kappa_p": list(self.kappa_p),
                "kappa_lo": self.kappa_lo,
                "crossing_step": self.crossing_step,
                "unguarded_above": self.unguarded_above,
            }
        record["check"] = dict(self.check)
        record["notes"] = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self.notes.items()
        }
        return record

    @property
    def has_collar(self) -> bool:
        """Return whether the design carries a real-space collar.

        Returns
        -------
        bool
            True when :attr:`collar_rate` is set.
        """
        return self.collar_rate is not None

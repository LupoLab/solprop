# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Certify adiabatic forward transport through a real index that varies along z.

A radial or Cartesian spectral column propagates along z with the real
longitudinal wave number :math:`q(z) = \operatorname{Re}k_z =
\sqrt{k(\omega, z)^2 - k_\perp^2}`. The forward-only model keeps the exact forward
directional amplitude and drops its coupling to the backward one. For a positive
real q of bounded variation over the run path, constant outside it, the exact
scalar problem bounds what that drops (SEP 34, section 4.5): with

.. math:: \Lambda = \tfrac12\operatorname{TV}_{[z_0, z_1]}(\ln q),

which counts kinks and declared jumps alike, the reflected power is at most
:math:`R_* = \tanh^2\Lambda` and the relative complex error of the retained
forward field at most :math:`\delta_* = \operatorname{expm1}(\Lambda\tanh\Lambda)`.
A monotone path has :math:`\Lambda = \tfrac12|\ln(q_1/q_0)|`, and the abrupt step
attains :math:`R_*`. The bound is for the linear background problem only: it
says nothing about the omitted backward field's local interference, nonlinear
feedback or ionisation.

**Admission.** :class:`AdiabaticAccuracy` holds the two budgets (both
:math:`10^{-4}` by default). Every propagating column of the active band must
satisfy :math:`\delta_* \le` ``field_error`` and :math:`R_* \le`
``reflected_energy`` over the whole path; a column that is propagating at one
point and not at another (it crosses :math:`q = 0`) refuses, as does a varying
imaginary index and a custom path without declared variation.

**The enclosure is not sampled.** Every material this module reads has a
wave-number square affine in its number densities,

.. math:: s(z) = q^2 = (\omega/c)^2\Bigl(\varepsilon(\omega)
          + \sum_i \rho_i(z)\,\chi_i(\omega)\Bigr) - k_\perp^2,

with :math:`\varepsilon = 1` for a gas or mixture (whose index is
:math:`\sqrt{1 + \sum_i\rho_i\chi_i}`) and :math:`\varepsilon = n^2` for a glass or
constant index, which carries no path. Each density path supplies a
:class:`solprop.profiles.PathVariation`: pieces between knots on which its values
and variation are enclosed, exactly where it is certified monotone. On the union of
those pieces:

* where every density is monotone and every term :math:`\chi_i\,\Delta\rho_i`
  moves one way, s is monotone, so :math:`\ln q` is too and its variation is the
  endpoint difference :math:`|\tfrac12\ln(s_1/s_0)|`, evaluated as
  ``log1p`` of the exactly formed difference over s: exact, and attained;
* otherwise :math:`\operatorname{TV}\ln q \le \sum_i|b_i|\operatorname{TV}\rho_i
  / (2 s_{\min})` with :math:`b_i = (\omega/c)^2\operatorname{Re}\chi_i` and
  :math:`s_{\min}` the minimum of the affine s over the box of density ranges,
  reached at a corner;
* a declared jump adds its own one-sided :math:`|\Delta\ln q|`.

Positivity follows from the same quantities: s is positive on a monotone piece
when it is at both ends, and on a box when it is at the lowest corner. The
arithmetic is bounded by a relative margin on each term: a few unit roundoffs for
forming the difference and the ratio, plus the conditioning
:math:`4u\,S/s_{\min}` of a square whose terms of magnitude S cancel towards
cutoff, amplified by the ``log1p`` condition number. The certificate is of the
represented path, the function the profile evaluates; how well that represents a
physical medium (an equation-of-state table, a measured profile) is a separate
envelope.
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from jaxtyping import Bool, Float, Num

from solprop.constants import C
from solprop.exceptions import MaterialError, TranslationError
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.mixtures import Mixture
from solprop.profiles import PathVariation, ZProfile, profile_identity
from solprop.spatial import SpatialDispersion

__all__ = [
    "AdiabaticAccuracy",
    "AdiabaticAdmission",
    "certify_radial_path",
    "certify_wave_path",
    "field_error_bound",
    "reflection_bound",
]

_UNIT: Final = 2.0**-53
"""The unit roundoff of binary64."""


@dataclass(frozen=True)
class AdiabaticAccuracy:
    """The budgets a z-varying path must meet to use adiabatic forward transport.

    These are model budgets chosen by the maintainer, not physical constants or
    integrator tolerances. Callers may tighten them.

    Attributes
    ----------
    field_error : float
        Largest certified relative error of the retained forward field,
        ``expm1(Lambda tanh Lambda)``, finite and positive. Default 1e-4.
    reflected_energy : float
        Largest certified reflected fraction of the power, ``tanh(Lambda)**2``,
        finite and positive. Default 1e-4.

    Raises
    ------
    MaterialError
        If a budget is not finite and positive.

    Examples
    --------
    >>> from solprop.adiabatic import AdiabaticAccuracy
    >>> AdiabaticAccuracy().field_error
    0.0001
    """

    field_error: float = 1e-4
    reflected_energy: float = 1e-4

    def __post_init__(self) -> None:
        """Refuse a budget that is not finite and positive."""
        for name in ("field_error", "reflected_energy"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise MaterialError(
                    f"AdiabaticAccuracy.{name} must be finite and positive, "
                    f"got {value!r}"
                )
            object.__setattr__(self, name, float(value))


def reflection_bound(variation: float | np.ndarray) -> float | np.ndarray:
    """Return the reflected-power bound ``tanh(Lambda)**2``.

    Parameters
    ----------
    variation : float or numpy.ndarray
        ``Lambda``, half the total variation of ``ln q``, nonnegative.

    Returns
    -------
    float or numpy.ndarray
        The bound; ``tanh`` is accurate to a few ulps for tiny arguments.

    Examples
    --------
    >>> from solprop.adiabatic import reflection_bound
    >>> float(reflection_bound(1e-9))
    1e-18
    """
    return np.tanh(variation) ** 2


def field_error_bound(variation: float | np.ndarray) -> float | np.ndarray:
    """Return the forward-field error bound ``expm1(Lambda tanh Lambda)``.

    Parameters
    ----------
    variation : float or numpy.ndarray
        ``Lambda``, half the total variation of ``ln q``, nonnegative.

    Returns
    -------
    float or numpy.ndarray
        The relative complex error bound of the retained forward field.

    Examples
    --------
    >>> from solprop.adiabatic import field_error_bound
    >>> float(field_error_bound(1e-9))
    1e-18
    """
    return np.expm1(variation * np.tanh(variation))


@dataclass(frozen=True, eq=False)
class AdiabaticAdmission:
    """The record that a path met its adiabatic budgets.

    Attributes
    ----------
    identity : collections.abc.Mapping
        The represented material path: the material and each density path's
        structural identity (``None`` for a custom path, which has none).
    z_span : tuple of float
        The certified run path ``(z0, z1)`` in m.
    accuracy : AdiabaticAccuracy
        The budgets met.
    variation : float
        The largest enclosed ``Lambda`` over the propagating columns.
    reflection : float
        ``tanh(variation)**2``, the largest reflected-power bound.
    field_error : float
        ``expm1(variation tanh variation)``, the largest forward-field bound.
    worst_kperp_sq : float or None
        Transverse wave-number square in m^-2 of the column attaining `variation`;
        ``None`` when no column varies.
    worst_omega : float or None
        Its angular frequency in rad/s.
    omega : numpy.ndarray
        The certified angular frequencies in rad/s, shape ``(n_omega,)``.
    spectral_variation : numpy.ndarray
        The largest ``Lambda`` over the propagating columns at each frequency,
        zero where none propagates, shape ``(n_omega,)``.
    support : numpy.ndarray
        Which columns propagate over the whole path, shape ``(n_modes, n_omega)``;
        the others are nonpropagating everywhere.
    jumps : numpy.ndarray
        Declared jumps of the material inside the path, in m.
    exact : bool
        Whether every piece was certified monotone, so that ``variation`` is the
        attained value up to rounding rather than an upper bound.
    interpolation_envelope : float
        Relative error of the run's tabulated operator against the represented
        path, a separate envelope from the certificate's bounds (SEP 34 section
        4.5); zero when no table stands between them.
    """

    identity: Mapping[str, object]
    z_span: tuple[float, float]
    accuracy: AdiabaticAccuracy
    variation: float
    reflection: float
    field_error: float
    worst_kperp_sq: float | None
    worst_omega: float | None
    omega: Float[np.ndarray, " n_omega"] = field(repr=False)
    spectral_variation: Float[np.ndarray, " n_omega"] = field(repr=False)
    support: Bool[np.ndarray, "n_modes n_omega"] = field(repr=False)
    jumps: Float[np.ndarray, " n_jumps"] = field(repr=False)
    exact: bool = True
    interpolation_envelope: float = 0.0

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        for name in ("omega", "spectral_variation", "support", "jumps"):
            values = np.array(getattr(self, name))
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class _Pieces:
    """The density paths' enclosures on the union of their pieces.

    Attributes
    ----------
    knots : numpy.ndarray
        Union of the piece edges in m.
    first, last, variation, low, high : numpy.ndarray
        Per path and piece, shape ``(n_paths, n_pieces)``, as in
        :class:`solprop.profiles.PathVariation`.
    monotone : numpy.ndarray
        Per path and piece, whether the enclosure is certified monotone.
    """

    knots: Float[np.ndarray, " n_knots"]
    first: Float[np.ndarray, "n_paths n_pieces"]
    last: Float[np.ndarray, "n_paths n_pieces"]
    variation: Float[np.ndarray, "n_paths n_pieces"]
    low: Float[np.ndarray, "n_paths n_pieces"]
    high: Float[np.ndarray, "n_paths n_pieces"]
    monotone: Bool[np.ndarray, "n_paths n_pieces"]


@dataclass(frozen=True)
class _AffineSquare:
    """The wave-number square ``s = base + sum_i b_i rho_i`` of every column.

    Attributes
    ----------
    base : numpy.ndarray
        ``(omega/c)^2 Re(epsilon) - k_perp^2`` in m^-2, shape ``(n_modes, n_omega)``.
    slope : numpy.ndarray
        ``b_i = (omega/c)^2 Re(chi_i)``, shape ``(n_paths, n_omega)``.
    pieces : _Pieces
        The density paths' aligned enclosures.
    kperp_sq : numpy.ndarray
        Transverse squares in m^-2, for the rounding scale.
    """

    base: Float[np.ndarray, "n_modes n_omega"]
    slope: Float[np.ndarray, "n_paths n_omega"]
    pieces: _Pieces
    kperp_sq: Float[np.ndarray, " n_modes"]

    @property
    def magnitude(self) -> Float[np.ndarray, "n_modes n_omega"]:
        """Return the scale S of the terms that form s, for its rounding error.

        Returns
        -------
        numpy.ndarray
            ``|base + k_perp^2| + sum_i |b_i| max|rho_i| + k_perp^2`` in m^-2.
        """
        pieces = self.pieces
        scale = np.max(
            np.abs(np.concatenate([pieces.low, pieces.high], axis=1)),
            axis=1,
            initial=0.0,
        )
        return (
            np.abs(self.base + self.kperp_sq[:, None])
            + np.sum(np.abs(self.slope) * scale[:, None], axis=0)[None, :]
            + self.kperp_sq[:, None]
        )

    def at(self, values: Float[np.ndarray, " n_paths"]) -> np.ndarray:
        """Return s for one density per path.

        Parameters
        ----------
        values : numpy.ndarray
            One density per path.

        Returns
        -------
        numpy.ndarray
            The square s in m^-2, shape ``(n_modes, n_omega)``.
        """
        return self.base + np.sum(self.slope * values[:, None], axis=0)[None, :]

    def _term(
        self, change: np.ndarray, reference: np.ndarray, magnitude: np.ndarray
    ) -> np.ndarray:
        """Return ``|Delta ln q|`` from ``Delta s / s``, enlarged by its rounding.

        Parameters
        ----------
        change : numpy.ndarray
            The change of s, formed from the density changes, not by subtraction.
        reference : numpy.ndarray
            The square s at the start of the change.
        magnitude : numpy.ndarray
            The scale S of the terms forming s.

        Returns
        -------
        numpy.ndarray
            ``|log1p(change/reference)| / 2`` times one plus its relative error
            bound; zero where the reference is not positive (a crossing).
        """
        terms = self.slope.shape[0] + 3
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(reference > 0.0, change / reference, 0.0)
            value = np.abs(0.5 * np.log1p(np.maximum(ratio, -1.0 + 2 * _UNIT)))
            # Forming the difference and the ratio costs a few unit roundoffs;
            # s itself carries 4u S of cancellation error towards cutoff; log1p
            # amplifies a relative error of its argument by its condition number.
            relative = 2.0 * (
                _log_condition(ratio)
                * (terms * _UNIT + 4.0 * _UNIT * magnitude / np.abs(reference))
                + 2.0 * _UNIT
            )
        return np.where(reference > 0.0, value * (1.0 + relative), 0.0)

    def _box(self, j: int, low: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """Return the bound ``sum_i |b_i| TV(rho_i) / (2 s_min)`` on piece j.

        Parameters
        ----------
        j : int
            The piece.
        low : numpy.ndarray
            ``s_min`` over the piece's box of densities.
        magnitude : numpy.ndarray
            The scale S of the terms forming s.

        Returns
        -------
        numpy.ndarray
            The bound of ``TV(ln q)`` on the piece, enlarged by its rounding, and
            infinite where ``s_min`` is not positive.
        """
        terms = self.slope.shape[0] + 4
        change = np.sum(
            np.abs(self.slope) * self.pieces.variation[:, j : j + 1], axis=0
        )[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = 2.0 * (terms * _UNIT + 4.0 * _UNIT * magnitude / np.abs(low))
            return np.where(low > 0.0, change / (2.0 * low) * (1.0 + relative), np.inf)

    def enclose(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Enclose ``Lambda`` and the range of s for every column.

        Returns
        -------
        tuple
            ``Lambda`` (meaningful where s stays positive), the lowest and highest
            enclosed s, each of shape ``(n_modes, n_omega)``, and whether every
            varying piece was monotone, so that ``Lambda`` is attained.
        """
        pieces, slope = self.pieces, self.slope
        magnitude = self.magnitude
        shape = self.base.shape
        lowest = np.full(shape, np.inf)
        highest = np.full(shape, -np.inf)
        total = np.zeros(shape)
        exact = True
        n_pieces = pieces.knots.size - 1
        for j in range(n_pieces):
            first, last = pieces.first[:, j], pieces.last[:, j]
            s_first, s_last = self.at(first), self.at(last)
            moves = slope * (last - first)[:, None]
            same_way = np.all(moves >= 0.0, axis=0) | np.all(moves <= 0.0, axis=0)
            monotone = (bool(np.all(pieces.monotone[:, j])) & same_way)[None, :]
            ends = slope * pieces.low[:, j : j + 1], slope * pieces.high[:, j : j + 1]
            box_low = self.base + np.sum(np.minimum(*ends), axis=0)[None, :]
            box_high = self.base + np.sum(np.maximum(*ends), axis=0)[None, :]
            low = np.where(monotone, np.minimum(s_first, s_last), box_low)
            lowest = np.minimum(lowest, low)
            highest = np.maximum(
                highest, np.where(monotone, np.maximum(s_first, s_last), box_high)
            )
            if np.any(pieces.variation[:, j] > 0.0):
                smooth = self._term(np.sum(moves, axis=0)[None, :], s_first, magnitude)
                total = total + np.where(monotone, smooth, self._box(j, low, magnitude))
                exact = exact and bool(np.all(monotone))
            if j > 0:
                before = pieces.last[:, j - 1]
                change = np.sum(slope * (first - before)[:, None], axis=0)[None, :]
                if np.any(change != 0.0):
                    total = total + self._term(change, self.at(before), magnitude)
        # Lambda is half TV(ln q); the sum rounds once per term.
        return 0.5 * total * (1.0 + (2 * n_pieces + 1) * _UNIT), lowest, highest, exact


def _variation_of(path: ZProfile, start: float, end: float) -> PathVariation:
    """Return a path's enclosure, refusing one that cannot give it.

    Parameters
    ----------
    path : ZProfile
        A density path.
    start, end : float
        The interval in m.

    Returns
    -------
    PathVariation
        The path's record.

    Raises
    ------
    MaterialError
        If the path has no enclosure, such as a callable without declared variation.
    """
    try:
        return path.variation(start, end)
    except TranslationError as error:
        raise MaterialError(
            f"the adiabatic certificate needs a bounded variation of {path!r}: {error}"
        ) from error


def _pieces(paths: Sequence[ZProfile], start: float, end: float) -> _Pieces:
    """Collect every path's enclosure on the union of their pieces.

    Parameters
    ----------
    paths : sequence of ZProfile
        The density paths, at least one.
    start, end : float
        The run path in m.

    Returns
    -------
    _Pieces
        The aligned records.
    """
    knots = np.unique(
        np.concatenate([_variation_of(p, start, end).knots for p in paths])
    )
    rows = [
        [_variation_of(p, a, b).collapse() for a, b in itertools.pairwise(knots)]
        for p in paths
    ]

    def table(name: str) -> np.ndarray:
        """Return one field of every collapsed piece, paths by pieces.

        Parameters
        ----------
        name : str
            A :class:`~solprop.profiles.PathVariation` field.

        Returns
        -------
        numpy.ndarray
            Its first value per path and piece.
        """
        return np.array([[float(getattr(c, name)[0]) for c in row] for row in rows])

    return _Pieces(
        knots,
        table("first"),
        table("last"),
        table("variation"),
        table("low"),
        table("high"),
        np.array([[bool(c.monotone[0]) for c in row] for row in rows]),
    )


def _log_condition(ratio: np.ndarray) -> np.ndarray:
    """Return the relative condition number of ``log1p`` at each argument.

    Parameters
    ----------
    ratio : numpy.ndarray
        Arguments greater than -1.

    Returns
    -------
    numpy.ndarray
        ``|x / ((1 + x) log1p(x))|``, which tends to one at zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.abs(ratio / ((1.0 + ratio) * np.log1p(ratio)))
    return np.where(np.abs(ratio) < 1e-8, 1.0, value)


def _wavelength_text(omega: float) -> str:
    """Return a vacuum wavelength for a message.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/s.

    Returns
    -------
    str
        For example ``'800 nm'``.
    """
    return f"{2 * math.pi * C / omega * 1e9:.6g} nm"


def certify_wave_path(  # noqa: PLR0913, PLR0917 - the affine model's arrays and the run path
    omega: Float[np.ndarray, " n_omega"],
    permittivity: Num[np.ndarray, " n_omega"],
    susceptibilities: Num[np.ndarray, "n_paths n_omega"],
    paths: Sequence[ZProfile],
    kperp_sq: Float[np.ndarray, " n_modes"],
    z_span: tuple[float, float],
    accuracy: AdiabaticAccuracy | None = None,
    *,
    identity: Mapping[str, object] | None = None,
) -> AdiabaticAdmission:
    r"""Certify the columns of a material whose wave-number square is affine in z.

    The longitudinal square of every column is
    :math:`s = (\omega/c)^2(\varepsilon + \sum_i\rho_i(z)\chi_i) - k_\perp^2`
    (the module docstring gives the enclosure).

    Parameters
    ----------
    omega : numpy.ndarray
        Finite positive angular frequencies in rad/s, shape ``(n_omega,)``.
    permittivity : numpy.ndarray
        The constant part ``epsilon`` of the relative permittivity, shape
        ``(n_omega,)``.
    susceptibilities : numpy.ndarray
        Per-density susceptibility ``chi_i``, in the inverse unit of the paths,
        shape ``(n_paths, n_omega)``.
    paths : sequence of ZProfile
        The densities ``rho_i(z)``; each must enclose its variation.
    kperp_sq : numpy.ndarray
        Transverse wave-number squares in m^-2, finite and nonnegative,
        shape ``(n_modes,)``.
    z_span : tuple of float
        The run path ``(z0, z1)`` in m, finite with ``z0 < z1``.
    accuracy : AdiabaticAccuracy or None, optional
        The budgets; ``None`` takes the defaults.
    identity : collections.abc.Mapping or None, optional
        The represented path's identity, recorded in the admission.

    Returns
    -------
    AdiabaticAdmission
        The record of a path within budget.

    Raises
    ------
    MaterialError
        If a budget is exceeded (naming the worst column, its frequency and the
        bound), a column crosses ``q = 0`` over the path, the imaginary index
        varies, a path cannot enclose its variation, or an input is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.adiabatic import certify_wave_path
    >>> from solprop.profiles import Linear
    >>> omega = np.array([2.35e15])
    >>> admission = certify_wave_path(
    ...     omega, np.ones(1), np.array([[1e-3]]), [Linear(0.0, 1.0, 1e-3)],
    ...     np.zeros(1), (0.0, 1e-3))
    >>> f"{admission.variation:.6e}"
    '2.498751e-04'
    """
    accuracy = AdiabaticAccuracy() if accuracy is None else accuracy
    start, end = (float(v) for v in z_span)
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise MaterialError(f"the certified path needs finite z0 < z1, got {z_span}")
    omega = np.asarray(omega, dtype=float)
    kperp_sq = np.asarray(kperp_sq, dtype=float)
    permittivity = np.asarray(permittivity, dtype=complex)
    chi = np.asarray(susceptibilities, dtype=complex).reshape(len(paths), omega.size)
    if (
        omega.ndim != 1
        or not np.all(np.isfinite(omega))
        or np.any(omega <= 0.0)
        or permittivity.shape != omega.shape
        or not np.all(np.isfinite(permittivity))
        or not np.all(np.isfinite(chi))
    ):
        raise MaterialError(
            "the certificate needs finite positive frequencies and finite material data"
        )
    if kperp_sq.ndim != 1 or not np.all(np.isfinite(kperp_sq)) or np.any(kperp_sq < 0):
        raise MaterialError("transverse wave-number squares must be finite and >= 0")

    wave_sq = (omega / C) ** 2
    base = wave_sq[None, :] * np.real(permittivity)[None, :] - kperp_sq[:, None]
    if paths:
        pieces = _pieces(paths, start, end)
    else:
        none = np.empty((0, 1))
        pieces = _Pieces(
            np.array([start, end]), none, none, none, none, none, none.astype(bool)
        )
    jump_sizes = np.abs(pieces.first[:, 1:] - pieces.last[:, :-1])
    varies = np.sum(pieces.variation, axis=1) + np.sum(jump_sizes, axis=1) > 0.0
    if np.any(varies):
        # Any loss makes Im k_z = Im(k**2) / (2 Re k_z) vary with Re k_z along a
        # varying path, even when the absorbing component's own density is fixed.
        lossy = np.any(np.imag(chi) != 0.0, axis=0) | (np.imag(permittivity) != 0.0)
        if np.any(lossy):
            where = int(np.argmax(lossy))
            raise MaterialError(
                "the adiabatic certificate needs a real index along a varying path, "
                f"and the imaginary index varies at {_wavelength_text(omega[where])}: "
                "a nonconstant absorbing path is not supported (uniform loss is)"
            )

    square = _AffineSquare(base, wave_sq[None, :] * np.real(chi), pieces, kperp_sq)
    total, lowest, highest, exact = square.enclose()
    shape = total.shape

    propagating = lowest > 0.0
    nonpropagating = highest <= 0.0
    crossing = ~(propagating | nonpropagating)
    if np.any(varies) and np.any(crossing):
        mode, bin_ = np.unravel_index(int(np.argmax(crossing)), shape)
        raise MaterialError(
            "a column crosses q = 0 along the path (propagating at one point and not "
            f"at another, or not shown otherwise): k_perp^2 = {kperp_sq[mode]:.6g} "
            f"m^-2 at {_wavelength_text(omega[bin_])}, with q^2 between "
            f"{lowest[mode, bin_]:.6g} and {highest[mode, bin_]:.6g} m^-2; "
            f"{int(np.count_nonzero(crossing))} of {crossing.size} columns cross"
        )
    variation = np.where(propagating, total, 0.0)
    if not np.any(varies):
        variation = np.zeros(shape)
    worst = int(np.argmax(variation)) if variation.size else 0
    mode, bin_ = np.unravel_index(worst, shape)
    largest = float(variation[mode, bin_]) if variation.size else 0.0
    reflection = float(reflection_bound(largest))
    field_error = float(field_error_bound(largest))
    if field_error > accuracy.field_error or reflection > accuracy.reflected_energy:
        failing = int(
            np.count_nonzero(
                (field_error_bound(variation) > accuracy.field_error)
                | (reflection_bound(variation) > accuracy.reflected_energy)
            )
        )
        over = [
            f"field error {field_error:.3g} > {accuracy.field_error:.3g}"
            if field_error > accuracy.field_error
            else "",
            f"reflected energy {reflection:.3g} > {accuracy.reflected_energy:.3g}"
            if reflection > accuracy.reflected_energy
            else "",
        ]
        raise MaterialError(
            "the z-varying medium exceeds its adiabatic budget: "
            + " and ".join(o for o in over if o)
            + f" at the worst column, k_perp^2 = {kperp_sq[mode]:.6g} m^-2 at "
            f"{_wavelength_text(omega[bin_])} (Lambda = {largest:.6g}, half the total "
            f"variation of ln q over z = {start:g} to {end:g} m); {failing} of "
            f"{int(np.count_nonzero(propagating))} propagating columns exceed a "
            "budget. Smooth the gradient, move the band away from cutoff, or pass a "
            "looser AdiabaticAccuracy"
        )
    return AdiabaticAdmission(
        identity=dict(identity or {}),
        z_span=(start, end),
        accuracy=accuracy,
        variation=largest,
        reflection=reflection,
        field_error=field_error,
        worst_kperp_sq=float(kperp_sq[mode]) if largest > 0.0 else None,
        worst_omega=float(omega[bin_]) if largest > 0.0 else None,
        omega=omega,
        spectral_variation=np.max(variation, axis=0, initial=0.0),
        support=propagating,
        jumps=np.unique(
            np.concatenate([np.asarray(p.jumps, dtype=float) for p in paths])
            if paths
            else np.empty(0)
        ),
        exact=exact,
    )


def _material_terms(
    dispersion: SpatialDispersion, omega: Float[np.ndarray, " n_omega"]
) -> tuple[np.ndarray, np.ndarray, list[ZProfile], dict[str, object]]:
    """Split a spatial medium into its affine permittivity and density paths.

    Parameters
    ----------
    dispersion : solprop.spatial.SpatialDispersion
        The medium and grid.
    omega : numpy.ndarray
        Active angular frequencies in rad/s.

    Returns
    -------
    tuple
        Constant permittivity, per-path susceptibilities, the density paths and
        the material identity.
    """
    medium = dispersion.medium
    material = medium.material
    if isinstance(material, Mixture):
        paths = [c.density for c in material.components]
        chi = np.array(
            [
                np.asarray(c.species.susceptibility.susceptibility(omega))
                for c in material.components
            ]
        )
        name = "mixture of " + ", ".join(material.labels)
        permittivity = np.ones(omega.size, dtype=complex)
    elif medium.susceptibility is not None and not isinstance(material, GlassIndex):
        paths = [medium.density]
        chi = np.asarray(medium.susceptibility.susceptibility(omega))[None, :]
        name = getattr(material, "name", None) or str(material)
        permittivity = np.ones(omega.size, dtype=complex)
    else:
        paths = []
        chi = np.empty((0, omega.size), dtype=complex)
        name = repr(material)
        permittivity = medium.index(omega) ** 2
    identities: list[object] = []
    for path in paths:
        try:
            identities.append(profile_identity(path))
        except TranslationError:
            identities.append(None)
    return permittivity, chi, paths, {"material": name, "paths": identities}


def certify_radial_path(
    dispersion: SpatialDispersion,
    kperp_sq: Float[np.ndarray, " n_modes"],
    z_span: tuple[float, float],
    accuracy: AdiabaticAccuracy | None = None,
) -> AdiabaticAdmission:
    """Certify a spatial medium's z-varying path for every column of a radial grid.

    Parameters
    ----------
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid, whose active band is certified.
    kperp_sq : numpy.ndarray
        Transverse wave-number squares in m^-2, for a radial grid ``k_r**2``.
    z_span : tuple of float
        The run path ``(z0, z1)`` in m.
    accuracy : AdiabaticAccuracy or None, optional
        The budgets; ``None`` takes the defaults.

    Returns
    -------
    AdiabaticAdmission
        The record of a path within budget; a uniform medium is admitted with
        zero variation.

    Raises
    ------
    MaterialError
        As :func:`certify_wave_path`.
    """
    grid = dispersion.grid
    omega = np.asarray(grid.omega[grid.active_mask], dtype=float)
    permittivity, chi, paths, identity = _material_terms(dispersion, omega)
    return certify_wave_path(
        omega,
        permittivity,
        chi,
        paths,
        np.asarray(kperp_sq, dtype=float),
        z_span,
        accuracy,
        identity=identity,
    )

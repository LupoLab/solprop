# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Refractive-index models for glasses, crystals and metals, with exact derivatives.

Fused silica, which is what a hollow capillary is made of, the window crystals and glass
of the offline optics of :mod:`solprop.optics`, and the metals of a waveguide wall. A
glass is described by its complex refractive index ``n(omega)`` directly, not by a
per-particle susceptibility: unlike a gas fill it has no density to scale, and the
quantity a capillary mode needs is the cladding permittivity
:math:`\varepsilon_{\rm cl} = n^2` and its frequency derivative.

The sign convention is the package's (``docs/dev/conventions.md`` section 5):
``n = n' + i k`` with ``k >= 0`` for loss.

Every built-in solid is optmat's (SEP 3): :class:`OptmatGlass` adapts optmat's
absolute index, relative to vacuum, with its exact frequency derivative and its
qualified intervals, and the factories (:func:`fused_silica`, :func:`calcium_fluoride`,
:func:`magnesium_fluoride`, :func:`bk7`, :func:`sapphire`, :func:`barium_fluoride`,
:func:`glass`) return optmat's recommended model or the one ``model=`` names.
:class:`SellmeierGlass`, :class:`TabulatedGlass` and
:class:`CompositeGlass` remain for a caller's own data.

Evaluating outside a model's validity range emits
:class:`~solprop.exceptions.MaterialRangeWarning`, as the gas models do.
"""

import logging
import math
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import _optmat
from solprop.physdata.selection import (
    RAMAN_POLICIES,
    MaterialLog,
    MaterialRecord,
    ModelChoice,
    OptmatModel,
)

__all__ = [
    "TRANSPARENT_ABSORPTION_BOUND",
    "CompositeGlass",
    "GlassIndex",
    "OptmatGlass",
    "SellmeierGlass",
    "SilicaBundle",
    "TabulatedGlass",
    "barium_fluoride",
    "bk7",
    "calcium_fluoride",
    "fused_silica",
    "glass",
    "glass_names",
    "lossless_index",
    "magnesium_fluoride",
    "sapphire",
]

logger = logging.getLogger(__name__)

_MICRON: Final[float] = 1e-6
"""One micrometre in metres. Every published glass Sellmeier expression uses microns."""

_MICRON_OMEGA: Final[float] = 2.0 * math.pi * C / _MICRON
"""Factor `k` with ``lambda[um] = k / omega``."""


def _as_omega(omega: ArrayLike | float | int) -> Float[np.ndarray, "*shape"]:
    """Validate an angular-frequency argument and return it as a float array.

    Parameters
    ----------
    omega : array_like
        Angular frequency in rad/s, >= 0 and finite. A scalar, a sequence or an array.

    Returns
    -------
    ndarray
        The same values as a float array, 0-d for scalar input.

    Raises
    ------
    MaterialError
        If the values are not real, or any of them is negative or not finite.
    """
    # numpy casts a complex argument to float with a warning rather than an error, so
    # it is rejected here before the conversion.
    if np.iscomplexobj(omega):
        raise MaterialError("angular frequency must be real, got a complex value")
    try:
        values = np.asarray(omega, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MaterialError(f"angular frequency must be real: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise MaterialError("angular frequency must be finite")
    if np.any(values < 0.0):
        raise MaterialError(
            "angular frequency must not be negative; a refractive index is not defined "
            "there. Evaluate on the physical band of the grid."
        )
    return values


@dataclass(frozen=True, kw_only=True, eq=False)
class GlassIndex(ABC):
    """A complex refractive-index model for one glass from one source.

    Subclasses supply the index and its analytic frequency derivative; this base class
    adds the permittivity forms the capillary cladding term needs and checks the
    validity range. Instances compare by identity rather than by value, because a
    tabulated model holds arrays.

    Attributes
    ----------
    name : str
        The material, e.g. ``"SiO2"``.
    source : str
        The name used to select this model, e.g. ``"malitson"``.
    wavelength_range : tuple of float
        Shortest and longest wavelength in m over which the source is valid.
    citation : str
        Where the model comes from, including its stated validity range.
    """

    name: str
    source: str
    wavelength_range: tuple[float, float]
    citation: str

    @abstractmethod
    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return the complex index and its frequency derivative.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, >= 0, already validated.

        Returns
        -------
        tuple of ndarray
            ``n(omega)`` and ``dn/domega`` in s/rad.
        """

    def _check_range(self, omega: Float[np.ndarray, "*shape"]) -> None:
        """Warn if any frequency lies outside the validity range of the source.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, >= 0.

        Warns
        -----
        MaterialRangeWarning
            If any requested wavelength lies outside ``wavelength_range``.
        """
        shortest_valid, longest_valid = self.wavelength_range
        highest, lowest = float(np.max(omega)), float(np.min(omega))
        shortest = _MICRON_OMEGA * _MICRON / highest if highest > 0.0 else math.inf
        longest = _MICRON_OMEGA * _MICRON / lowest if lowest > 0.0 else math.inf
        if shortest >= shortest_valid and longest <= longest_valid:
            return
        warnings.warn(
            f"{self.name} '{self.source}' is valid over "
            f"{shortest_valid * 1e9:.4g}-{longest_valid * 1e9:.4g} nm; the requested "
            f"spectrum spans {shortest * 1e9:.4g}-{longest * 1e9:.4g} nm. "
            f"Source: {self.citation}",
            MaterialRangeWarning,
            stacklevel=3,
        )

    def index(self, omega: ArrayLike | float | int) -> Complex[np.ndarray, "*shape"]:
        """Return the complex refractive index.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``n = n' + i k`` with ``k >= 0`` for loss. 0-d for scalar input, otherwise
            the shape of `omega`.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        >>> from solprop.physdata import fused_silica
        >>> silica = fused_silica(model="malitson_1965/sellmeier")
        >>> f"{complex(silica.index(2.356e15)):.6f}"
        '1.453722+0.000000j'
        """
        values = _as_omega(omega)
        self._check_range(values)
        return self._index_and_derivative(values)[0]

    def dindex_domega(
        self, omega: ArrayLike | float | int
    ) -> Complex[np.ndarray, "*shape"]:
        """Return the analytic frequency derivative of the complex refractive index.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``dn/domega`` in s/rad. For a tabulated model this is the exact derivative
            of the interpolant, whose own accuracy is that of the tabulated data.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        >>> from solprop.physdata import fused_silica
        >>> silica = fused_silica(model="malitson_1965/sellmeier")
        >>> f"{complex(silica.dindex_domega(2.356e15)):.4e}"
        '5.8781e-18+0.0000e+00j'
        """
        values = _as_omega(omega)
        self._check_range(values)
        return self._index_and_derivative(values)[1]

    def permittivity(
        self, omega: ArrayLike | float | int
    ) -> Complex[np.ndarray, "*shape"]:
        """Return the complex relative permittivity ``n^2``.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``epsilon = n^2``, dimensionless.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        >>> from solprop.physdata import fused_silica
        >>> silica = fused_silica(model="malitson_1965/sellmeier")
        >>> f"{complex(silica.permittivity(2.356e15)):.6f}"
        '2.113309+0.000000j'
        """
        values = _as_omega(omega)
        self._check_range(values)
        index, _ = self._index_and_derivative(values)
        return np.asarray(index**2)

    def dpermittivity_domega(
        self, omega: ArrayLike | float | int
    ) -> Complex[np.ndarray, "*shape"]:
        """Return the analytic frequency derivative of the relative permittivity.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``depsilon/domega = 2 n dn/domega`` in s/rad.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        >>> from solprop.physdata import fused_silica
        >>> silica = fused_silica(model="malitson_1965/sellmeier")
        >>> f"{complex(silica.dpermittivity_domega(2.356e15)):.4e}"
        '1.7090e-17+0.0000e+00j'
        """
        values = _as_omega(omega)
        self._check_range(values)
        index, derivative = self._index_and_derivative(values)
        return np.asarray(2.0 * index * derivative)


@dataclass(frozen=True, kw_only=True, eq=False)
class SellmeierGlass(GlassIndex):
    r"""A real refractive index from a sum-of-poles Sellmeier expansion.

    The expansion is the standard form

    .. math:: n^2 = 1 + \sum_j \frac{B_j\lambda^2}{\lambda^2 - \lambda_j^2}

    with the wavelength in microns. Written in ``u = 1/lambda^2`` it is rational,
    ``n^2 = 1 + sum_j B_j / (1 - lambda_j^2 u)``, so the frequency derivative follows in
    closed form and no finite difference is needed.

    The expansion has **no constant term**: every entry carried so far has a leading
    coefficient of zero in its source. A fit with one (Li's 1980 fluorides have 0.2762
    and 0.33973) cannot be represented here and must not be transcribed by dropping
    it.

    Below the shortest pole ``n^2`` turns negative; the square root is taken in the
    complex plane there, so the deep ultraviolet returns an evanescent
    index rather than a NaN.

    Attributes
    ----------
    poles : tuple of tuple of float
        One ``(B_j, lambda_j)`` pair per pole, with ``lambda_j`` in microns.
    """

    poles: tuple[tuple[float, float], ...]

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return the index and its frequency derivative from the pole sum.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, >= 0.

        Returns
        -------
        tuple of ndarray
            ``n(omega)`` and ``dn/domega``.
        """
        # u = 1/lambda[um]^2 = (omega/_MICRON_OMEGA)^2, so du/domega = 2 u / omega.
        u = np.asarray((omega / _MICRON_OMEGA) ** 2)
        squared = np.ones_like(u)
        dsquared_du = np.zeros_like(u)
        for strength, pole in self.poles:
            denominator = 1.0 - pole**2 * u
            squared = squared + strength / denominator
            dsquared_du = dsquared_du + strength * pole**2 / denominator**2
        index = np.asarray(np.sqrt(squared.astype(complex)))
        # d(u)/d(omega) = 2u/omega, and omega = 0 gives u = 0 and a vanishing
        # derivative, so the ratio is written as 2 omega / _MICRON_OMEGA^2 to stay
        # finite at the static limit.
        du_domega = 2.0 * omega / _MICRON_OMEGA**2
        return index, np.asarray(dsquared_du * du_domega / (2.0 * index))


@dataclass(frozen=True, kw_only=True, eq=False)
class TabulatedGlass(GlassIndex):
    """A complex refractive index interpolated from a caller's data table.

    The real part is splined against ``log lambda``, on which the data are close to
    uniformly sampled; the extinction coefficient is splined as ``log k`` on a floored
    copy of the data, so that exponentiating the interpolant can never return a negative
    ``k``. Both derivatives follow from the spline derivative and
    ``d(log lambda)/d(omega) = -1/omega``, so they are exact for the interpolant.

    Beyond the ends of the table the index is held constant, with a zero derivative, and
    :class:`~solprop.exceptions.MaterialRangeWarning` is raised: a cubic extrapolated
    over nine decades is meaningless, so it is not done silently.

    Attributes
    ----------
    provenance : collections.abc.Mapping
        The table's provenance as the caller records it: its source record,
        citation, licence and checksum. A built-in table is optmat's
        (``TabulatedModel``, adapted by :class:`OptmatGlass`).
    """

    provenance: Mapping[str, str]
    _real: CubicSpline
    _log_extinction: CubicSpline
    _log_wavelength_limits: tuple[float, float]

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return the index and its frequency derivative from the splines.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, >= 0.

        Returns
        -------
        tuple of ndarray
            ``n(omega)`` and ``dn/domega``.
        """
        lowest, highest = self._log_wavelength_limits
        with np.errstate(divide="ignore"):
            # omega = 0 is the infinite-wavelength limit; the clip below sends it to the
            # long-wavelength end of the table, where the warning has already fired.
            log_wavelength = np.log(_MICRON_OMEGA) - np.log(np.asarray(omega))
        clipped = np.clip(log_wavelength, lowest, highest)
        inside = (log_wavelength >= lowest) & (log_wavelength <= highest)
        real = np.asarray(self._real(clipped))
        extinction = np.exp(np.asarray(self._log_extinction(clipped)))
        # d/domega = d/d(log lambda) * (-1/omega), and zero where the value is
        # held constant beyond the ends of the table.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(inside, -1.0 / np.asarray(omega), 0.0)
        scale = np.where(np.isfinite(scale), scale, 0.0)
        dreal = np.asarray(self._real(clipped, 1)) * scale
        dextinction = extinction * np.asarray(self._log_extinction(clipped, 1)) * scale
        return (
            np.asarray(real + 1j * extinction),
            np.asarray(dreal + 1j * dextinction),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CompositeGlass(GlassIndex):
    """A glass index taking its real part and its absorption from different sources.

    The two parts are evaluated independently and added, so the validity range is the
    intersection of theirs and the citation names both.

    Attributes
    ----------
    real_part : GlassIndex
        Supplies ``Re n`` and ``d(Re n)/d(omega)``.
    absorption : GlassIndex
        Supplies ``Im n`` and ``d(Im n)/d(omega)``.
    """

    real_part: GlassIndex
    absorption: GlassIndex

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return the combined index and its frequency derivative.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, >= 0.

        Returns
        -------
        tuple of ndarray
            ``n(omega)`` and ``dn/domega``.
        """
        real, dreal = self.real_part._index_and_derivative(omega)
        imaginary, dimaginary = self.absorption._index_and_derivative(omega)
        return (
            np.asarray(np.real(real) + 1j * np.imag(imaginary)),
            np.asarray(np.real(dreal) + 1j * np.imag(dimaginary)),
        )


TRANSPARENT_ABSORPTION_BOUND: Final = 100.0
"""Largest intensity absorption coefficient, 1/m, a transparent index drops quietly.

A transparent optmat index (:meth:`OptmatGlass.as_transparent`) keeps the real part
and drops ``k``; beyond this bound on ``alpha = 4 pi k / lambda`` it warns once. The
bound is chosen from fused silica (``fit/SiO2``, optmat 0.1.0): across its transparent
window, 200 nm to 2.5 um, ``alpha`` stays below 60 /m (1.9 /m at 200 nm, 60 /m in the
2.2 um band, 25 /m at 2.5 um); its ultraviolet edge passes 100 /m near 176 nm (510 /m
at 170 nm, 1e8 /m at 120 nm) and its OH band at 2.73 um reaches 2800 /m. 100 /m, an
e-folding of the intensity over a centimetre, therefore passes the window and fails
the edges.
"""


@dataclass
class _DroppedLoss:
    """What a transparent index has dropped so far: a mutable record of one adapter.

    Attributes
    ----------
    extinction : float
        The largest ``k`` dropped, dimensionless.
    absorption : float
        The largest ``4 pi k / lambda`` dropped, 1/m.
    wavelength : float
        The vacuum wavelength in m of that absorption, or NaN.
    warned : bool
        Whether the bound has been reported.
    """

    extinction: float = 0.0
    absorption: float = 0.0
    wavelength: float = math.nan
    warned: bool = False


@dataclass(frozen=True, kw_only=True, eq=False)
class OptmatGlass(GlassIndex):
    """Take a solid's complex refractive index from an optmat model.

    The index is optmat's absolute ``n + i k`` (relative to vacuum, ``k >= 0`` for
    absorption, never conjugated: optmat's sign is solprop's), with its exact
    frequency derivative: analytic for the Sellmeier and fitted models, the
    interpolant's own for a tabulated metal. A band outside the model's qualified
    intervals is evaluated all the same and warns once per evaluation context
    (:class:`~solprop.exceptions.MaterialRangeWarning`); a metal table holds its end
    values there.

    :meth:`as_transparent` takes the real part of the index, as a recorded
    approximation within a declared domain. A route that solves only a lossless
    problem (the step-index modes, a general modal guide's material laws) takes
    that approximation itself, through :func:`lossless_index`, and never refuses an
    absorbing index (SEP 3, Q3). The transparent adapter records the largest
    extinction and absorption coefficient it has dropped (:attr:`transparency`),
    and warns when the absorption coefficient exceeds its bound,
    :data:`TRANSPARENT_ABSORPTION_BOUND` by default: once per run inside a run, once
    per adapter outside one.

    The adapter keeps its last few evaluations by frequency array (the bridge's
    ``EvaluationCache``): a frequency array equal to an earlier one takes the stored
    index, bitwise the same, and raises and records what the first evaluation did.

    Two adapters are equal, and hash alike, when they compute the same index: the
    same model by its identity digest, under the same transparent approximation and
    bound (SEP 3, Q26). Two default walls or cores built separately are therefore one
    material, as two modes of one guide require; what each adapter has evaluated and
    dropped is its own measurement and does not enter the comparison.

    Attributes
    ----------
    model : OptmatModel
        The optmat solid model: a registry model, or one from ``with_parameters``,
        ``FormulaModel.sellmeier`` or ``TabulatedModel.from_arrays``.
    transparent : bool
        Whether the index is the model's real part only, a declared approximation.
    absorption_bound : float
        For a transparent index, the largest intensity absorption coefficient
        ``4 pi k / lambda`` in 1/m it drops without a warning; positive.

    Examples
    --------
    >>> from solprop.physdata import fused_silica
    >>> silica = fused_silica(model="recommended")
    >>> silica.source
    'fit/SiO2'
    >>> f"{float(silica.index(2.356e15).real):.6f}"
    '1.453719'
    """

    model: OptmatModel
    transparent: bool = False
    absorption_bound: float = TRANSPARENT_ABSORPTION_BOUND
    _dropped: _DroppedLoss = field(init=False, repr=False, default_factory=_DroppedLoss)
    _evaluations: _optmat.EvaluationCache = field(
        init=False, repr=False, default_factory=_optmat.EvaluationCache
    )

    def __post_init__(self) -> None:
        """Validate the absorption bound.

        Raises
        ------
        MaterialError
            If the bound is not positive.
        """
        if not self.absorption_bound > 0:
            msg = (
                "a transparent index's absorption bound must be positive, got "
                f"{self.absorption_bound} 1/m"
            )
            raise MaterialError(msg)

    def _identity_key(self) -> tuple[str, bool, float]:
        """Return what the index depends on: the digest and the approximation.

        Returns
        -------
        tuple
            The model's identity digest, whether it is transparent, and the bound.
        """
        return (
            _optmat.record(self.model).digest,
            self.transparent,
            self.absorption_bound,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two adapters by the index they compute, not as objects.

        Parameters
        ----------
        other : object
            Another object.

        Returns
        -------
        bool
            Whether `other` is an adapter of the same class, model digest,
            transparency and bound.
        """
        if type(other) is not type(self):
            return NotImplemented
        return self._identity_key() == cast("OptmatGlass", other)._identity_key()

    def __hash__(self) -> int:
        """Hash the adapter by the index it computes, as :meth:`__eq__` compares it.

        Returns
        -------
        int
            The hash of the digest, the transparency and the bound.
        """
        return hash(self._identity_key())

    def __repr__(self) -> str:
        """Return the material, the optmat model and its digest.

        Returns
        -------
        str
            For example ``OptmatGlass(SiO2, fit/SiO2, digest=..., transparent=False)``;
            the digest identifies the numbers.
        """
        entry = _optmat.record(self.model)
        bound = (
            f", absorption_bound={self.absorption_bound:g}" if self.transparent else ""
        )
        return (
            f"OptmatGlass({self.name}, {entry.name}, digest={entry.digest}, "
            f"transparent={self.transparent}{bound})"
        )

    @classmethod
    def from_model(
        cls,
        model: object,
        *,
        transparent: bool = False,
        absorption_bound: float = TRANSPARENT_ABSORPTION_BOUND,
    ) -> "OptmatGlass":  # noqa: UP037 - the hook resolves class names eagerly
        """Adapt any optmat solid model, taken as it is.

        Parameters
        ----------
        model : OptmatModel
            The optmat solid model.
        transparent : bool, optional
            Take the real part of the index only, as a recorded approximation.
        absorption_bound : float, optional
            For a transparent index, the absorption coefficient in 1/m beyond which
            dropping ``k`` warns.

        Returns
        -------
        OptmatGlass
            The adapted model, named by optmat's material and model name.

        Warns
        -----
        MaterialStatusWarning
            If the model is ``provisional`` or ``untested``.
        """
        resolved = _optmat.untyped(model)
        entry = _optmat.check_status(resolved, str(resolved.material))
        low, high = resolved.wavelength_range
        return cls(
            name=str(resolved.material),
            source=entry.name,
            wavelength_range=(float(low), float(high)),
            citation=(
                f"{resolved.citation or entry.name} (optmat {entry.optmat_version}, "
                f"{entry.name})"
            ),
            model=resolved,
            transparent=transparent,
            absorption_bound=absorption_bound,
        )

    def as_transparent(
        self, absorption_bound: float = TRANSPARENT_ABSORPTION_BOUND
    ) -> "OptmatGlass":  # noqa: UP037 - see from_model
        """Return the same model with its index taken as real, a recorded approximation.

        Parameters
        ----------
        absorption_bound : float, optional
            The intensity absorption coefficient ``4 pi k / lambda`` in 1/m up to
            which ``k`` is dropped quietly; beyond it the adapter warns once.

        Returns
        -------
        OptmatGlass
            A new adapter, ``transparent=True``, whose dropped loss is tracked afresh.

        Raises
        ------
        MaterialError
            If the bound is not positive.
        """
        return OptmatGlass.from_model(
            self.model, transparent=True, absorption_bound=absorption_bound
        )

    @property
    def axis(self) -> str:
        """Return the crystal axis the model describes.

        Returns
        -------
        str
            ``"o"`` or ``"e"`` for a uniaxial crystal, empty otherwise.
        """
        return str(getattr(self.model, "axis", ""))

    @property
    def largest_dropped_extinction(self) -> float:
        """Return the largest extinction coefficient a transparent index has dropped.

        Returns
        -------
        float
            ``max k`` over every band evaluated so far, dimensionless; 0 for a model
            that is not transparent or has not been evaluated.
        """
        return self._dropped.extinction

    @property
    def largest_dropped_absorption(self) -> float:
        """Return the largest absorption coefficient a transparent index has dropped.

        Returns
        -------
        float
            ``max 4 pi k / lambda`` over every band evaluated so far, in 1/m.
        """
        return self._dropped.absorption

    @property
    def transparency(self) -> Mapping[str, float | bool | None]:
        """Return the record of the transparent approximation, for a run's manifest.

        Returns
        -------
        collections.abc.Mapping
            ``transparent``, the ``absorption_bound`` in 1/m, the largest dropped
            extinction and absorption coefficient (1/m) with its wavelength in m
            (None before any is dropped), and whether the bound was exceeded.
        """
        dropped = self._dropped
        return {
            "transparent": self.transparent,
            "absorption_bound": self.absorption_bound,
            "largest_dropped_extinction": dropped.extinction,
            "largest_dropped_absorption": dropped.absorption,
            "wavelength": None
            if math.isnan(dropped.wavelength)
            else dropped.wavelength,
            "bound_exceeded": dropped.absorption > self.absorption_bound,
        }

    def run_transparency(
        self, log: MaterialLog | None
    ) -> Mapping[str, float | bool | None]:
        """Return the record of the transparent approximation within one run.

        Parameters
        ----------
        log : MaterialLog or None
            The run's evaluation log; ``None`` gives :attr:`transparency`, what the
            adapter has dropped over its life.

        Returns
        -------
        collections.abc.Mapping
            As :attr:`transparency`, with the largest dropped extinction and
            absorption those of the run's own evaluations (SEP 3, Q28).

        Examples
        --------
        >>> from solprop.physdata import fused_silica
        >>> from solprop.physdata.selection import MaterialLog
        >>> window = fused_silica(model="recommended").as_transparent()
        >>> log = MaterialLog()
        >>> window.run_transparency(log)["largest_dropped_absorption"]
        0.0
        """
        if log is None:
            return self.transparency
        extinction, absorption, wavelength = log.dropped.get(
            (_optmat.record(self.model).digest, self.absorption_bound),
            (0.0, 0.0, math.nan),
        )
        return {
            "transparent": self.transparent,
            "absorption_bound": self.absorption_bound,
            "largest_dropped_extinction": extinction,
            "largest_dropped_absorption": absorption,
            "wavelength": None if math.isnan(wavelength) else wavelength,
            "bound_exceeded": absorption > self.absorption_bound,
        }

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the record of the optmat model, for a run's manifest.

        Returns
        -------
        tuple of MaterialRecord
            The solid model's identity.
        """
        return (_optmat.record(self.model),)

    def _check_range(self, omega: Float[np.ndarray, "*shape"]) -> None:
        """Leave the range check to optmat's qualified intervals.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s; the bridge reports any part outside the
            model's intervals when the index is evaluated.
        """

    def _record_dropped(
        self,
        extinction: Float[np.ndarray, "*shape"],
        wavelength: Float[np.ndarray, "*shape"],
    ) -> None:
        """Record the loss a transparent index drops, and warn once beyond the bound.

        Parameters
        ----------
        extinction : numpy.ndarray
            The dropped ``k``, dimensionless.
        wavelength : numpy.ndarray
            The vacuum wavelengths in m it was dropped at.

        Warns
        -----
        MaterialRangeWarning
            When ``4 pi k / lambda`` exceeds the bound: once per run inside a run's
            evaluation context, and once per adapter outside every context.
        """
        kappa = np.ravel(np.asarray(extinction, dtype=float))
        if not kappa.size:
            return
        lam = np.ravel(np.asarray(wavelength, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            absorption = np.where(np.isfinite(lam), 4 * math.pi * kappa / lam, 0.0)
        worst = int(np.argmax(absorption))
        largest, where = float(absorption[worst]), float(lam[worst])
        dropped = self._dropped
        dropped.extinction = max(dropped.extinction, float(np.max(kappa)))
        if largest > dropped.absorption:
            dropped.absorption, dropped.wavelength = largest, where
        # A run records what it dropped itself, whatever the adapter dropped in
        # other runs before it (SEP 3, Q28).
        _optmat.record_dropped(
            self.model, self.absorption_bound, float(np.max(kappa)), largest, where
        )
        inside = _optmat.open_log() is not None
        if (largest if inside else dropped.absorption) <= self.absorption_bound:
            return
        if not inside:
            largest, where = dropped.absorption, dropped.wavelength
        message = (
            f"{self.name} ({self.source}) is taken as transparent, and drops an "
            f"absorption coefficient of {largest:.3g} /m (k = "
            f"{float(kappa[worst]):.3g}) at {where * 1e9:.4g} nm, "
            f"beyond the transparent approximation's bound of "
            f"{self.absorption_bound:g} /m: the band reaches an absorbing region "
            "the lossless solve does not represent"
        )
        # One adapter can serve several runs (a mode reused, or the transparent
        # counterpart every lossless route of one source shares): inside a run the
        # warning is the run's, once; outside every run it is the adapter's, once.
        if inside:
            digest = _optmat.record(self.model).digest
            _optmat.warn_once(
                ("transparent", digest, repr(self.absorption_bound)),
                message,
                MaterialRangeWarning,
                stacklevel=5,
            )
        elif not dropped.warned:
            warnings.warn(message, MaterialRangeWarning, stacklevel=4)
        dropped.warned = True

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return optmat's complex index and its exact frequency derivative.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s, > 0.

        Returns
        -------
        tuple of ndarray
            ``n(omega)`` and ``dn/domega`` in s/rad; their real parts only for a
            transparent index.
        """
        wavelength = _optmat.omega_to_wavelength(omega)
        model = _optmat.untyped(self.model)

        def evaluate() -> tuple[np.ndarray, np.ndarray]:
            """Evaluate optmat's index and its derivative at the band.

            Returns
            -------
            tuple of numpy.ndarray
                ``n(omega)`` and ``dn/domega``, complex.
            """
            return (
                np.asarray(model.refractive_index(wavelength), dtype=complex),
                np.asarray(model.dindex_domega(wavelength), dtype=complex),
            )

        # The index does not depend on z, yet a translation reads it at every sample
        # of a z table, and the permittivity and its derivative read it twice more:
        # each frequency array is evaluated once (SEP 3, Q22).
        index, slope = self._evaluations.evaluate(
            "refractive index",
            omega,
            self.model,
            material=self.name,
            wavelength=wavelength,
            compute=evaluate,
        )
        if not self.transparent:
            return index, slope
        self._record_dropped(np.imag(index), wavelength)
        return (
            np.asarray(np.real(index), dtype=complex),
            np.asarray(np.real(slope), dtype=complex),
        )


_LOSSLESS_LOCK: Final = threading.Lock()
"""Serialises the creation of an optmat index's transparent counterpart."""


def lossless_index[Material: GlassIndex | float | complex](
    material: Material,
) -> Material:
    """Return the index a solve that admits no loss takes of a material.

    An absorbing optmat index on a lossless-only route -- the step-index modes, a
    general modal guide's material laws -- is taken through its transparent
    approximation, never refused (SEP 3, Q3): the real part of the index and its
    real derivative, with the dropped ``k`` recorded and a
    :class:`~solprop.exceptions.MaterialRangeWarning` once per run where the
    dropped absorption ``4 pi k / lambda`` exceeds
    :data:`TRANSPARENT_ABSORPTION_BOUND`. Every lossless route of one adapter takes
    one counterpart, so two modes of one guide still share their material and its
    record is one. A model with ``k = 0`` everywhere drops nothing. Any other
    material, a transparent optmat index included, is returned as it is: a
    complex index of solprop's own keeps each route's existing handling.

    Parameters
    ----------
    material : GlassIndex, float or complex
        The material a lossless route was given.

    Returns
    -------
    GlassIndex, float or complex
        The transparent counterpart of an absorbing-capable optmat index, otherwise
        `material` itself.

    Examples
    --------
    >>> from solprop.physdata import fused_silica
    >>> from solprop.physdata.glasses import lossless_index
    >>> silica = fused_silica(model="recommended")
    >>> lossless = lossless_index(silica)
    >>> lossless.transparent, lossless_index(silica) is lossless
    (True, True)
    >>> lossless_index(1.45)
    1.45
    """
    if not isinstance(material, OptmatGlass) or material.transparent:
        return material
    with _LOSSLESS_LOCK:
        # Held outside the dataclass fields, so that neither a copy by `replace` nor
        # a run's manifest walk (which enters fields only) sees it: the counterpart
        # is recorded where a route holds it, and only there.
        counterpart = material.__dict__.get("_lossless")
        if counterpart is None:
            counterpart = material.as_transparent()
            object.__setattr__(material, "_lossless", counterpart)
    return cast("Material", counterpart)


_UNIAXIAL: Final = frozenset({"MgF2", "Al2O3"})
"""The built-in solids that are uniaxial crystals, whose ray must be named."""


def _material_id(name: str) -> str:
    """Return optmat's material id of a name, or the name for a caller's own.

    Parameters
    ----------
    name : str
        A material id or alias (``"sapphire"``), or a caller's own name.

    Returns
    -------
    str
        The canonical id, such as ``"Al2O3"``.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.materials import as_material  # noqa: PLC0415 - as above

    try:
        return str(as_material(name))
    except OptmatError:
        return name


def _check_axis(material: str, axis: str | None) -> None:
    """Refuse an axis that is not a ray of the material.

    Parameters
    ----------
    material : str
        The material's optmat id.
    axis : str or None
        The ray asked for.

    Raises
    ------
    MaterialError
        If a uniaxial crystal is given anything but ``"o"`` or ``"e"``, or an
        isotropic material is given an axis.
    """
    if material in _UNIAXIAL:
        if axis not in (None, "o", "e"):
            msg = f"{material} is uniaxial: axis is 'o' or 'e', not {axis!r}"
            raise MaterialError(msg)
    elif axis is not None:
        msg = f"{material} is isotropic and takes no axis, got {axis!r}"
        raise MaterialError(msg)


def _optmat_solid(
    material: str, model: ModelChoice, axis: str | None = None
) -> OptmatGlass:
    """Resolve a built-in solid's optmat model from a name or an object.

    Parameters
    ----------
    material : str
        The material's optmat id: ``"SiO2"``, ``"CaF2"``, ``"MgF2"``, ``"BaF2"``,
        ``"Al2O3"``, ``"N-BK7"``, ``"Ag"`` or ``"Al"``.
    model : str or OptmatModel
        ``"recommended"``, an optmat model name, or a model object, taken as it is.
    axis : str or None, optional
        ``"o"`` or ``"e"`` for a uniaxial crystal; required with ``"recommended"``.

    Returns
    -------
    OptmatGlass
        The adapted model.

    Raises
    ------
    MaterialError
        If a uniaxial crystal's recommended model is asked for without an axis,
        optmat has no such model, or the model describes another material or ray.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.solid.models import registry  # noqa: PLC0415 - as above

    _check_axis(material, axis)
    if model == "recommended" and material in _UNIAXIAL and axis is None:
        msg = (
            f"{material} is uniaxial and optmat recommends a model per ray: give "
            "axis='o' (ordinary) or axis='e' (extraordinary)"
        )
        raise MaterialError(msg)
    try:
        if model == "recommended":
            resolved = _optmat.untyped(registry.default_model(material, axis=axis))
        elif isinstance(model, str):
            resolved = _optmat.untyped(registry.model(model))
        else:
            resolved = _optmat.untyped(model)
    except OptmatError as error:
        msg = f"{material}: optmat has no solid model {model!r}: {error}"
        raise MaterialError(msg) from error
    described = _material_id(str(resolved.material))
    if described != material:
        msg = (
            f"the solid model {resolved.identity.name!r} describes {described}, not "
            f"{material}: a model is never matched to another material"
        )
        raise MaterialError(msg)
    ray = str(getattr(resolved, "axis", ""))
    if axis is not None and ray and ray != axis:
        msg = (
            f"the solid model {resolved.identity.name!r} describes {material}'s "
            f"{ray} ray, and axis={axis!r} was asked for"
        )
        raise MaterialError(msg)
    return OptmatGlass.from_model(resolved)


def _resolve(
    material: str,
    model: ModelChoice | None,
    axis: str | None = None,
) -> OptmatGlass:
    """Resolve a factory's ``model=`` to an optmat solid.

    Parameters
    ----------
    material : str
        The material's optmat id.
    model : str or OptmatModel or None
        ``None`` or ``"recommended"``, an optmat model name, or a model object.
    axis : str or None, optional
        The ray of a uniaxial crystal.

    Returns
    -------
    OptmatGlass
        The adapted model.
    """
    return _optmat_solid(material, "recommended" if model is None else model, axis)


def fused_silica(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return a refractive-index model for fused silica.

    The default is optmat's recommended silica, ``fit/SiO2``: Franta's universal
    dispersion model refitted to absolute refractometry, causal and absorbing from
    24.8 nm to 125 um, with its exact frequency derivative (:class:`OptmatGlass`). It
    is what a capillary cladding wants: one coherent law for the real index and the
    ultraviolet and mid-infrared absorption that the Marcatili loss term needs at the
    edges of a broad spectrum. A lossless-only route takes its transparent
    approximation itself (:func:`lossless_index`).

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's recommended silica; an optmat
        model name or object selects another (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        The model, exposing ``index``, ``dindex_domega``, ``permittivity`` and
        ``dpermittivity_domega``.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not silica.

    Examples
    --------
    >>> from solprop.physdata import fused_silica
    >>> silica = fused_silica()
    >>> silica.source
    'fit/SiO2'
    >>> f"{float(silica.index(2.356e15).real):.6f}"
    '1.453719'
    >>> fused_silica(model="malitson_1965/sellmeier").source
    'malitson_1965/sellmeier'
    """
    return _resolve("SiO2", model)


def calcium_fluoride(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return a refractive-index model for calcium fluoride.

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``daimon_2002/sellmeier_20C``,
        made absolute from its nitrogen reference; an optmat model name or object
        selects another (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        A lossless Sellmeier model with an analytic frequency derivative.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not CaF2.

    Examples
    --------
    >>> from solprop.physdata import calcium_fluoride
    >>> f"{float(calcium_fluoride().index(2.356e15).real):.5f}"
    '1.43096'
    >>> calcium_fluoride(model="malitson_1963/sellmeier").source
    'malitson_1963/sellmeier'
    """
    return _resolve("CaF2", model)


def magnesium_fluoride(
    *,
    axis: str | None = None,
    model: ModelChoice | None = None,
) -> GlassIndex:
    """Return the index of magnesium fluoride, a uniaxial crystal.

    Parameters
    ----------
    axis : {'o', 'e'} or None, optional
        The ray: ``"o"`` (ordinary), which a c-cut window presents at normal
        incidence, or ``"e"`` (extraordinary). optmat's models are per ray, so the
        default needs it; a named model names its own.
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``dodge_1984/sellmeier_o`` or
        ``_e``, made absolute; an optmat model name or object selects another
        (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        A lossless Sellmeier model, 200 nm to 7 um.

    Raises
    ------
    MaterialError
        If the axis is not a ray, or is missing for optmat's recommended model, or
        the optmat model is unknown, not MgF2 or of the other ray.

    Examples
    --------
    >>> from solprop.physdata import magnesium_fluoride
    >>> extraordinary = magnesium_fluoride(axis="e")
    >>> f"{float(extraordinary.index(2.356e15).real):.5f}"
    '1.38713'
    """
    return _resolve("MgF2", model, axis)


def bk7(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return the index of SCHOTT N-BK7.

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``schott_2023/sellmeier``,
        lossless; an optmat model name or object selects another
        (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        A lossless Sellmeier model, 300 nm to 2.5 um.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not N-BK7.

    Examples
    --------
    >>> from solprop.physdata import bk7
    >>> bk7().source
    'schott_2023/sellmeier'
    """
    return _resolve("N-BK7", model)


def sapphire(
    *,
    axis: str | None = None,
    model: ModelChoice | None = None,
) -> GlassIndex:
    """Return the index of synthetic sapphire, a uniaxial crystal.

    Parameters
    ----------
    axis : {'o', 'e'} or None, optional
        The ray, ``"o"`` or ``"e"``; optmat's models are per ray, so the default
        needs it. A named model names its own.
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``weber_2003/sellmeier_o`` or
        ``_e``, made absolute; an optmat model name or object selects another
        (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        A lossless Sellmeier model, 200 nm to 5 um.

    Raises
    ------
    MaterialError
        As :func:`magnesium_fluoride`, for sapphire.

    Examples
    --------
    >>> from solprop.physdata import sapphire
    >>> sapphire(axis="o").source
    'weber_2003/sellmeier_o'
    """
    return _resolve("Al2O3", model, axis)


def barium_fluoride(*, model: ModelChoice | None = None) -> GlassIndex:
    """Return the index of barium fluoride.

    Parameters
    ----------
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's ``malitson_1964/sellmeier``, made
        absolute; an optmat model name or object selects another
        (:class:`OptmatGlass`).

    Returns
    -------
    GlassIndex
        A lossless Sellmeier model, 265 nm to 10.3 um.

    Raises
    ------
    MaterialError
        If the optmat model is unknown or not BaF2.

    Examples
    --------
    >>> from solprop.physdata import barium_fluoride
    >>> barium_fluoride().source
    'malitson_1964/sellmeier'
    """
    return _resolve("BaF2", model)


_GLASS_ALIASES: Final[dict[str, str]] = {
    "sio2": "sio2",
    "silica": "sio2",
    "fused silica": "sio2",
    "caf2": "caf2",
    "mgf2": "mgf2",
    "bk7": "bk7",
    "n-bk7": "bk7",
    "sapphire": "sapphire",
    "al2o3": "sapphire",
    "baf2": "baf2",
}
"""Names :func:`glass` accepts, lower case, mapped to a canonical key."""


_OPTMAT_IDS: Final[dict[str, str]] = {
    "sio2": "SiO2",
    "caf2": "CaF2",
    "mgf2": "MgF2",
    "bk7": "N-BK7",
    "sapphire": "Al2O3",
    "baf2": "BaF2",
}
"""optmat's material id of each canonical glass key."""


def glass_names() -> tuple[str, ...]:
    """Return the names :func:`glass` accepts, lower case.

    Returns
    -------
    tuple of str
        Every accepted spelling, aliases included.

    Examples
    --------
    >>> from solprop.physdata import glass_names
    >>> "caf2" in glass_names() and "silica" in glass_names()
    True
    """
    return tuple(_GLASS_ALIASES)


def glass(
    name: str,
    *,
    model: ModelChoice | None = None,
    axis: str | None = None,
) -> GlassIndex:
    """Return a glass by name.

    Parameters
    ----------
    name : str
        One of :func:`glass_names`, in any case: ``"SiO2"``, ``"silica"``, ``"CaF2"``,
        ``"MgF2"``, ``"BK7"``, ``"sapphire"``, ``"Al2O3"``, ``"BaF2"``.
    model : str or OptmatModel or None, optional
        ``None`` or ``"recommended"`` is optmat's recommended model of the glass;
        an optmat model name or an optmat model object selects another
        (:class:`OptmatGlass`).
    axis : {'o', 'e'} or None, optional
        The ray of a uniaxial crystal (MgF2, sapphire); required with optmat's
        recommended model, refused for an isotropic glass.

    Returns
    -------
    GlassIndex
        The model.

    Raises
    ------
    MaterialError
        If the name, the axis or the optmat model is unknown or does not fit the
        glass, or a uniaxial crystal's recommended model is asked for without its ray.

    Examples
    --------
    >>> from solprop.physdata import glass
    >>> glass("CaF2").source
    'daimon_2002/sellmeier_20C'
    >>> glass("sapphire", axis="o").source
    'weber_2003/sellmeier_o'
    >>> glass("CaF2", model="malitson_1963/sellmeier").source
    'malitson_1963/sellmeier'
    """
    key = _GLASS_ALIASES.get(name.strip().lower())
    if key is None:
        known = ", ".join(_GLASS_ALIASES)
        raise MaterialError(f"unknown glass {name!r}; solprop knows {known}")
    return _resolve(_OPTMAT_IDS[key], model, axis)


@dataclass(frozen=True, eq=False)
class SilicaBundle:
    r"""Take fused silica's nonlinear response from optmat: total n2, f_R and h_R.

    optmat's silica bundle (``optmat.solid.bundle``) is the triple a propagation code
    in the :math:`n_0 = 1` convention of the GNLSE takes, so that its parts cannot be
    mismatched: the Kerr fit's **total** long-pulse :math:`n_2(\lambda)`
    (``kerr/SiO2``), a Raman shape :math:`h_R` of unit integral, and that shape's own
    Raman fraction :math:`f_R`. The electronic part is :math:`(1 - f_R)\,n_2` and the
    delayed one :math:`f_R\,n_2` convolved with :math:`h_R`, so a scalar and a vector
    run that hold the same total agree for long pulses (SEP 3, section 4.5).

    - ``"default"``: Hollenbeck and Cantrell's shape with the Kerr fit's
      :math:`f_R = 0.2188`, *paired* with it; copolarised only.
    - ``"tensor"``: Lin and Agrawal's shape with their own :math:`f_R = 0.245`,
      which has Hellwarth's isotropic part :math:`a` and anisotropic part :math:`b`
      (:math:`a + b = h_R`).

    optmat's ``raman_coefficient`` and ``electronic_susceptibility`` include its
    silica :math:`n_0^2`; solprop's GNLSE calibration and vacuum-normalised
    amplitudes take :math:`n_0 = 1`, so only :math:`n_2`, :math:`f_R` and the
    normalised :math:`h_R` are taken from here, never those two.

    Attributes
    ----------
    model : OptmatModel
        The optmat bundle (``SilicaNonlinearBundle``); kept out of a run's
        configuration record, which names it by `digest`.
    digest : str
        Its identity digest, filled in from the model.

    Raises
    ------
    MaterialError
        If the model is not an optmat silica bundle, or `digest` is not its own.

    Examples
    --------
    >>> from solprop.physdata.glasses import SilicaBundle
    >>> bundle = SilicaBundle.resolve()
    >>> bundle.name, f"{bundle.raman_fraction:.4f}", bundle.has_tensor
    ('silica_bundle/default', '0.2188', False)
    >>> SilicaBundle.resolve(tensor=True).raman_fraction
    0.245
    """

    model: OptmatModel = field(repr=False, metadata={"record": False})
    digest: str = ""

    def __eq__(self, other: object) -> bool:
        """Compare two bundles by their identity digest, not as objects (SEP 3, Q26).

        Parameters
        ----------
        other : object
            Another object.

        Returns
        -------
        bool
            Whether `other` is a bundle of the same digest.
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.digest == cast("SilicaBundle", other).digest

    def __hash__(self) -> int:
        """Hash the bundle by its identity digest.

        Returns
        -------
        int
            The digest's hash.
        """
        return hash(self.digest)

    def __post_init__(self) -> None:
        """Check the model and fill in its digest.

        Raises
        ------
        MaterialError
            If the model is not a silica bundle or the digest is another model's.
        """
        entry = _optmat.record(self.model)
        if entry.kind != "silica_bundle":
            msg = (
                f"{entry.name} is optmat's {entry.kind} model, not a silica bundle: a "
                "bundle holds n2, f_R and h_R together (optmat.solid.bundle)"
            )
            raise MaterialError(msg)
        if self.digest and self.digest != entry.digest:
            msg = (
                f"digest {self.digest[:12]} is not {entry.name}'s, {entry.digest[:12]}"
            )
            raise MaterialError(msg)
        object.__setattr__(self, "digest", entry.digest)

    def __repr__(self) -> str:
        """Return the bundle's name, its shape and its fraction.

        Returns
        -------
        str
            For example ``SilicaBundle(silica_bundle/default, hollenbeck_2002,
            f_R=0.2188)``.
        """
        return f"SilicaBundle({self.name}, {self.shape}, f_R={self.raman_fraction:.4g})"

    @classmethod
    def resolve(
        cls, choice: ModelChoice = "recommended", *, tensor: bool = False
    ) -> "SilicaBundle":  # noqa: UP037 - the hook resolves class names eagerly
        """Resolve a silica bundle by selection name, optmat name or object.

        Parameters
        ----------
        choice : str or OptmatModel, optional
            ``"recommended"``, the curated default
            (:data:`~solprop.physdata.selection.RAMAN_POLICIES`): optmat's
            ``"default"`` bundle, or its ``"tensor"`` bundle when `tensor` is set.
            ``"provisional"`` is a gas policy and is refused by name. Any other name
            is an optmat bundle name (``"default"``, ``"tensor"``, ``"lin_2006"``,
            ``"blow_1989"``, ...); an object is taken as it is.
        tensor : bool, optional
            Whether the run needs the isotropic and anisotropic parts: a vector run.

        Returns
        -------
        SilicaBundle
            The adapter.

        Raises
        ------
        MaterialError
            If optmat has no bundle of that name, the name is the gas policy
            ``"provisional"``, or the object is not a bundle.

        Warns
        -----
        MaterialStatusWarning
            If the bundle is ``provisional`` or ``untested``.
        """
        from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
        from optmat.solid.bundle import silica_bundle  # noqa: PLC0415 - as above

        if isinstance(choice, str):
            name = choice
            if choice == "provisional":
                msg = (
                    "MaterialSelection.raman='provisional' admits a gas's provisional "
                    "Raman bands and means nothing for fused silica; a silica run "
                    "takes 'recommended', an optmat bundle name or a bundle object"
                )
                raise MaterialError(msg)
            if choice in RAMAN_POLICIES:
                name = "tensor" if tensor else "default"
            try:
                model: object = silica_bundle(name)
            except OptmatError as error:
                msg = f"optmat has no silica bundle {name!r}: {error}"
                raise MaterialError(msg) from error
        else:
            model = choice
        bundle = cls(cast("OptmatModel", model))
        _optmat.check_status(bundle.model, "SiO2")
        return bundle

    @property
    def _bundle(self) -> _optmat.Untyped:
        """Return the optmat bundle for attribute access.

        Returns
        -------
        optmat.solid.bundle.SilicaNonlinearBundle
            The model.
        """
        return _optmat.untyped(self.model)

    @property
    def name(self) -> str:
        """Return the bundle's optmat name.

        Returns
        -------
        str
            For example ``"silica_bundle/default"``.
        """
        return _optmat.record(self.model).name

    @property
    def shape(self) -> str:
        """Return the name of the bundle's Raman shape.

        Returns
        -------
        str
            For example ``"hollenbeck_2002"``.
        """
        return str(self._bundle.response.name)

    @property
    def raman_fraction(self) -> float:
        """Return the shape's own Raman fraction, the delayed share of the total n2.

        Returns
        -------
        float
            ``f_R``, in ``(0, 1)``.
        """
        return float(self._bundle.raman_fraction)

    @property
    def paired(self) -> bool:
        """Return whether ``f_R`` is the one the Kerr fit found with this shape.

        Returns
        -------
        bool
            True for the default bundle.
        """
        return bool(self._bundle.paired)

    @property
    def has_tensor(self) -> bool:
        """Return whether the shape has Hellwarth's isotropic and anisotropic parts.

        Returns
        -------
        bool
            True for Lin and Agrawal's shape; False for a copolarised-only one.
        """
        return bool(self._bundle.response.has_tensor)

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the records of the bundle, its Kerr fit and its Raman shape.

        Returns
        -------
        tuple of MaterialRecord
            The bundle's first.
        """
        bundle = self._bundle
        return (
            _optmat.record(bundle),
            _optmat.record(bundle.kerr),
            _optmat.record(bundle.response),
        )

    def n2(self, wavelength: float) -> float:
        """Return the total long-pulse nonlinear index at a carrier wavelength.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in m, positive and finite.

        Returns
        -------
        float
            The Kerr fit's total ``n2`` in m^2/W.

        Raises
        ------
        MaterialError
            If the wavelength is not positive and finite.

        Warns
        -----
        MaterialRangeWarning
            Once per evaluation context, outside the Kerr fit's 350 nm to 3.5 um.
        """
        value = float(wavelength)
        if not (math.isfinite(value) and value > 0.0):
            msg = (
                f"SiO2: n2 needs a positive, finite wavelength in m, got {wavelength!r}"
            )
            raise MaterialError(msg)
        bundle = self._bundle
        with _optmat.translated(
            bundle.kerr, "n2", material="SiO2", wavelength=np.array([value])
        ):
            return float(bundle.n2(value))

    def h_r(
        self, t: Float[np.ndarray, " n_to"], component: str = "parallel"
    ) -> Float[np.ndarray, " n_to"]:
        """Return the normalised Raman response or one of its tensor parts.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``; zero and negative delays give zero.
        component : {'parallel', 'a', 'b'}, optional
            ``"parallel"``, the copolarised :math:`h_R = a + b` of unit integral, or
            Hellwarth's isotropic ``"a"`` and anisotropic ``"b"``.

        Returns
        -------
        numpy.ndarray
            The response in 1/s, shape ``(n_to,)``.

        Raises
        ------
        MaterialError
            If a tensor part is asked of a copolarised-only shape, naming it.
        """
        from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy

        bundle = self._bundle
        try:
            return np.asarray(bundle.h_r(np.asarray(t, dtype=float), component))
        except OptmatError as error:
            msg = (
                f"{self.name} ({self.shape}) has no Raman part {component!r}: "
                f"{error}. A vector run needs the tensor bundle, "
                "SilicaBundle.resolve(tensor=True)"
            )
            raise MaterialError(msg) from error

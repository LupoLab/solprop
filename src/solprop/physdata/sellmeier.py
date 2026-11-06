# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Refractive-index models for gases, with analytic frequency derivatives.

A gas is described by its **per-particle susceptibility** ``gamma(omega)``, in m^3, so
that the linear susceptibility at a number density ``rho`` is simply

.. math:: \\chi^{(1)}(\\omega) = \\gamma(\\omega)\\,\\rho

and the refractive index is ``n = sqrt(1 + chi)``. Keeping the density outside the model
is what lets ``n - 1`` be formed directly rather than as ``(1 + x) - 1``
(``docs/dev/conventions.md`` section 5), and it makes a pressure gradient a scalar
multiplier on one fixed spectral profile.

Every built-in gas's law is optmat's polarizability (:class:`OptmatSusceptibility`,
SEP 3), with its exact frequency derivative and its qualified wavelength intervals;
:func:`gamma` returns it and the accessors below evaluate it at a state, with optmat's
equation of state. A model variant is chosen with a
:class:`~solprop.physdata.selection.MaterialSelection`, never a separate pseudo-gas.

The formula types (:class:`PoleSumSusceptibility`, :class:`InversePoleIndex`,
:class:`WavenumberPoleIndex`) are for a caller's own published expression: each is a
rational function of ``1 / lambda^2``, so its derivative with respect to angular
frequency is available in closed form. That is a requirement, not a convenience: the
co-moving frame needs ``beta_1`` to about 1e-10 and a finite-difference derivative of a
Sellmeier expression has a roughness floor near 5e-8 (see ``docs/dev/decisions.md``).
Each carries the wavelength range over which its source is valid, the state at which
it reports its refractivity (:class:`ReferenceState`) and a citation. Evaluating outside
that range emits :class:`~solprop.exceptions.MaterialRangeWarning` once per call.
"""

import functools
import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np
from jaxtyping import Float, Inexact
from numpy.typing import ArrayLike

from solprop.constants import EPS0, K_B, ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import _optmat
from solprop.physdata.equation_of_state import density
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.selection import MaterialRecord, ModelChoice, OptmatModel

__all__ = [
    "GasSusceptibility",
    "InversePoleIndex",
    "OptmatSusceptibility",
    "PoleSumSusceptibility",
    "ReferenceState",
    "RefractivitySusceptibility",
    "WavenumberPoleIndex",
    "chi1",
    "dchi1_domega",
    "drefractive_index_domega",
    "gamma",
    "refractive_index",
    "refractive_index_minus_one",
]

logger = logging.getLogger(__name__)

_MICRON: Final[float] = 1e-6
"""One micrometre in metres. Every published gas Sellmeier expression uses microns."""

_INVERSE_MICRON_PER_OMEGA: Final[float] = _MICRON / (2.0 * math.pi * C)
"""Factor `k` with ``1 / lambda[um] = k * omega``, so that ``u = (k omega)^2``."""


@dataclass(frozen=True)
class ReferenceState:
    """The thermodynamic state at which a measured refractivity was reported.

    A published Sellmeier expression gives the refractivity of a gas at one temperature
    and pressure. Dividing by the number density there turns it into the per-particle
    quantity this module works with, so getting the state right matters: 1 bar against
    1 atm is a 1.3 % error in every susceptibility.

    Attributes
    ----------
    pressure_bar : float
        Absolute pressure of the reported measurement, in bar.
    temperature : float
        Temperature of the reported measurement, in K, or room temperature where the
        source declares none.
    """

    pressure_bar: float
    temperature: float


@functools.cache
def _reference_density(gas: Gas, state: ReferenceState) -> float:
    """Return the number density at a model's reference state (cached).

    Parameters
    ----------
    gas : Gas
        The species.
    state : ReferenceState
        The state the source reports its refractivity at.

    Returns
    -------
    float
        Number density in m^-3, from optmat's equation of state.
    """
    return density(gas, state.pressure_bar, state.temperature)


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


def _wavelength_extremes(omega: Float[np.ndarray, "*shape"]) -> tuple[float, float]:
    """Return the shortest and longest wavelength of a set of angular frequencies.

    Parameters
    ----------
    omega : ndarray
        Angular frequencies in rad/s, >= 0.

    Returns
    -------
    tuple of float
        Shortest and longest wavelength in m; the longest is infinite if any
        frequency is zero.
    """
    highest, lowest = float(np.max(omega)), float(np.min(omega))
    shortest = 2.0 * math.pi * C / highest if highest > 0.0 else math.inf
    longest = 2.0 * math.pi * C / lowest if lowest > 0.0 else math.inf
    return shortest, longest


@dataclass(frozen=True, kw_only=True)
class GasSusceptibility(ABC):
    """A per-particle linear susceptibility model for one gas from one source.

    Subclasses supply the published expression and its derivative with respect to
    ``u = 1 / lambda^2`` (lambda in microns) at the source's reference state; this base
    class turns that into the per-particle quantity ``gamma(omega)`` in m^3 and its
    analytic derivative, and checks the validity range.

    Attributes
    ----------
    gas : Gas
        The species the coefficients describe.
    source : str
        The name used to select this model in :func:`gamma`.
    reference_state : ReferenceState
        The state at which the source reports its refractivity.
    wavelength_range : tuple of float
        Shortest and longest wavelength in m over which the source is valid.
    citation : str
        Where the coefficients come from, including the validity range and any
        deviation from the source.
    """

    gas: Gas
    source: str
    reference_state: ReferenceState
    wavelength_range: tuple[float, float]
    citation: str

    @abstractmethod
    def _chi_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return the susceptibility at the reference state and its derivative in `u`.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            ``chi(u)`` at the reference state (dimensionless) and ``dchi/du``.
        """

    @property
    def reference_density(self) -> float:
        """Return the number density at the reference state of the source.

        Returns
        -------
        float
            Number density in m^-3.
        """
        return _reference_density(self.gas, self.reference_state)

    @property
    def absorbing(self) -> bool:
        """Return whether the model has an imaginary part anywhere.

        Returns
        -------
        bool
            False: a refractivity fit is real by construction.
        """
        return False

    def _check_range(self, omega: Float[np.ndarray, "*shape"]) -> None:
        """Warn if any frequency lies outside the validity range of the source.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies in rad/s.

        Warns
        -----
        MaterialRangeWarning
            If any requested wavelength lies outside ``wavelength_range``.
        """
        shortest_valid, longest_valid = self.wavelength_range
        shortest, longest = _wavelength_extremes(omega)
        if shortest >= shortest_valid and longest <= longest_valid:
            return
        sides = []
        if shortest < shortest_valid:
            sides.append(
                "below its short-wavelength end, where extrapolation approaches a "
                "pole of the expression and is not reliable"
            )
        if longest > longest_valid:
            sides.append(
                "beyond its long-wavelength end, where the expression is smooth but "
                "unverified against data"
            )
        warnings.warn(
            f"{self.gas.value} '{self.source}' is fitted over "
            f"{shortest_valid * 1e9:.4g}-{longest_valid * 1e9:.4g} nm; the requested "
            f"spectrum spans {shortest * 1e9:.4g}-{longest * 1e9:.4g} nm, "
            f"{' and '.join(sides)}. Source: {self.citation}",
            MaterialRangeWarning,
            stacklevel=3,
        )

    def susceptibility(
        self, omega: ArrayLike | float | int
    ) -> Float[np.ndarray, "*shape"]:
        """Return the per-particle susceptibility.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0. Zero gives the static limit.

        Returns
        -------
        ndarray
            ``gamma(omega)`` in m^3, such that ``chi1 = gamma * rho``. 0-d for scalar
            input, otherwise the shape of `omega`.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        >>> from solprop.physdata import Gas, gamma
        >>> model = gamma(Gas.Ar)
        >>> f"{float(model.susceptibility(2.356e15)):.6e}"
        '2.079791e-29'
        """
        values = _as_omega(omega)
        self._check_range(values)
        u = np.asarray((_INVERSE_MICRON_PER_OMEGA * values) ** 2)
        chi, _ = self._chi_and_derivative(u)
        return np.asarray(chi / self.reference_density)

    def dsusceptibility_domega(
        self, omega: ArrayLike | float | int
    ) -> Float[np.ndarray, "*shape"]:
        """Return the analytic frequency derivative of the per-particle susceptibility.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``dgamma/domega`` in m^3 s/rad, exact to rounding rather than to a
            finite-difference floor.

        Warns
        -----
        MaterialRangeWarning
            If any wavelength lies outside the validity range of the source.

        Examples
        --------
        The derivative is positive: these gases are normally dispersive.

        >>> from solprop.physdata import Gas, gamma
        >>> float(gamma(Gas.Ar).dsusceptibility_domega(2.356e15)) > 0.0
        True
        """
        values = _as_omega(omega)
        self._check_range(values)
        scale = _INVERSE_MICRON_PER_OMEGA
        _, dchi_du = self._chi_and_derivative(np.asarray((scale * values) ** 2))
        return np.asarray(dchi_du * 2.0 * scale**2 * values / self.reference_density)


@dataclass(frozen=True, kw_only=True)
class PoleSumSusceptibility(GasSusceptibility):
    """A susceptibility written directly as a sum of poles in ``lambda^2``.

    ``chi(lambda) = sum_i b_i lambda^2 / (lambda^2 - c_i)`` with lambda in microns, the
    form used by Börzsönyi's two-pole fits and by the three-pole fits extended into the
    vacuum ultraviolet. In terms of ``u = 1 / lambda^2`` this is
    ``sum_i b_i / (1 - c_i u)``, which stays finite at zero frequency.

    Attributes
    ----------
    b : tuple of float
        Pole strengths, dimensionless, being the susceptibility at the reference state.
    c : tuple of float
        Pole positions in um^2, one per entry of `b`. A negative entry is a fitting
        parameter with no resonance, not a physical pole.
    """

    b: tuple[float, ...]
    c: tuple[float, ...]

    def _chi_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return ``sum b_i / (1 - c_i u)`` and its derivative in `u`.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            The susceptibility at the reference state and its derivative.
        """
        chi = np.zeros_like(u)
        dchi_du = np.zeros_like(u)
        for b, c in zip(self.b, self.c, strict=True):
            denominator = 1.0 - c * u
            chi = chi + b / denominator
            dchi_du = dchi_du + b * c / denominator**2
        return np.asarray(chi), np.asarray(dchi_du)


@dataclass(frozen=True, kw_only=True)
class RefractivitySusceptibility(GasSusceptibility):
    """A model whose source publishes the refractivity ``n - 1`` rather than ``chi``.

    The susceptibility follows as ``chi = (1 + (n - 1))^2 - 1``. Keeping that step in
    one place is deliberate: omitting either the subtraction or the squaring when
    implementing the He et al. expression is an easy error, and omitting both is wrong
    by a factor of about 1200.
    """

    @abstractmethod
    def _refractivity_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return ``n - 1`` at the reference state and its derivative in `u`.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            The refractivity and its derivative with respect to `u`.
        """

    def _chi_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return the susceptibility and its derivative from the refractivity.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            ``(1 + dn)^2 - 1`` and its derivative with respect to `u`.
        """
        refractivity, drefractivity_du = self._refractivity_and_derivative(u)
        # (1 + dn)^2 - 1 written as dn(2 + dn): no cancellation for dn ~ 1e-4.
        return (
            np.asarray(refractivity * (2.0 + refractivity)),
            np.asarray(2.0 * (1.0 + refractivity) * drefractivity_du),
        )


@dataclass(frozen=True, kw_only=True)
class InversePoleIndex(RefractivitySusceptibility):
    """A refractivity written as a sum of poles in ``1 / lambda^2``.

    ``n - 1 = offset + sum_i b_i / (c_i - 1 / lambda^2)`` with lambda in microns, the
    form used by Peck and Huang for hydrogen, by Zhang for oxygen (with a non-zero
    offset) and by Bideau-Mehu for the noble gases in the vacuum ultraviolet.

    Attributes
    ----------
    offset : float
        Constant term of the refractivity at the reference state, dimensionless.
    b : tuple of float
        Pole strengths in um^-2.
    c : tuple of float
        Pole positions in um^-2, one per entry of `b`.
    """

    offset: float
    b: tuple[float, ...]
    c: tuple[float, ...]

    def _refractivity_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return ``offset + sum b_i / (c_i - u)`` and its derivative in `u`.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            The refractivity and its derivative with respect to `u`.
        """
        refractivity = np.full_like(u, self.offset)
        derivative = np.zeros_like(u)
        for b, c in zip(self.b, self.c, strict=True):
            denominator = c - u
            refractivity = refractivity + b / denominator
            derivative = derivative + b / denominator**2
        return np.asarray(refractivity), np.asarray(derivative)


@dataclass(frozen=True, kw_only=True)
class WavenumberPoleIndex(RefractivitySusceptibility):
    """A refractivity written as one pole in the squared wavenumber.

    ``(n - 1) * 1e8 = a + b / (c - nu^2)`` with the wavenumber ``nu = 1e4 / lambda[um]``
    in cm^-1, the form used by He et al. (2021) for methane, nitrous oxide and sulfur
    hexafluoride.

    Attributes
    ----------
    a : float
        Constant term of ``(n - 1) * 1e8`` at the reference state.
    b : float
        Pole strength in cm^-2.
    c : float
        Pole position in cm^-2.
    """

    a: float
    b: float
    c: float

    def _refractivity_and_derivative(
        self, u: Float[np.ndarray, "*shape"]
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Return ``n - 1`` and its derivative in `u` for the wavenumber form.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            The refractivity and its derivative with respect to `u`.
        """
        # nu[cm^-1] = 1e4 / lambda[um], so nu^2 = 1e8 u; the 1e-8 of the published
        # form and the 1e8 of the chain rule cancel in the derivative.
        denominator = self.c - 1e8 * u
        return (
            np.asarray(1e-8 * (self.a + self.b / denominator)),
            np.asarray(self.b / denominator**2),
        )


def _model_absorbs(model: object) -> bool:
    """Return whether an optmat gas model, or an air model's constituents, absorb.

    Parameters
    ----------
    model : OptmatModel
        A gas dispersion model (with ``absorbing``) or an air model (with
        ``models``, its constituents').

    Returns
    -------
    bool
        True if any part of the model carries an imaginary polarizability.
    """
    constituents = getattr(model, "models", None)
    if constituents is not None:
        return any(_model_absorbs(part) for part in constituents)
    return bool(_optmat.untyped(model).absorbing)


def _optmat_gas_model(gas: Gas, choice: ModelChoice) -> _optmat.Untyped:
    """Resolve an optmat dispersion model of a gas from a name or an object.

    Parameters
    ----------
    gas : Gas
        The species the model must describe.
    choice : str or OptmatModel
        ``"recommended"``, an optmat model name (for air, a composition name), or a
        model object.

    Returns
    -------
    OptmatModel
        The model, checked to describe `gas`.

    Raises
    ------
    MaterialError
        If optmat has no such model, or the model describes another gas.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.air import AIR_COMPOSITIONS, DEFAULT_AIR, air_model  # noqa: PLC0415
    from optmat.gas.models.registry import default_model, model  # noqa: PLC0415

    try:
        if gas is Gas.Air and isinstance(choice, str):
            name = DEFAULT_AIR if choice == "recommended" else choice
            if name not in AIR_COMPOSITIONS:
                msg = (
                    f"Gas.Air takes optmat's air compositions {AIR_COMPOSITIONS}, "
                    f"not {name!r}"
                )
                raise MaterialError(msg)
            return air_model(name)
        if choice == "recommended":
            return default_model(gas.value)
        resolved = _optmat.untyped(model(choice) if isinstance(choice, str) else choice)
    except OptmatError as error:
        msg = f"{gas.value}: optmat has no linear model {choice!r}: {error}"
        raise MaterialError(msg) from error
    air = hasattr(resolved, "constituents")
    if air != (gas is Gas.Air) or (not air and str(resolved.gas) != gas.value):
        msg = (
            f"the linear model {resolved.identity.name!r} describes "
            f"{resolved.identity.material}, not {gas.value}: a model is never matched "
            "to another gas"
        )
        raise MaterialError(msg)
    return resolved


@dataclass(frozen=True, kw_only=True, eq=False)
class OptmatSusceptibility(GasSusceptibility):
    r"""Take a gas's per-particle susceptibility from optmat's polarizability.

    The susceptibility per particle is :math:`\gamma = \alpha / \varepsilon_0`, with
    optmat's SI refractive polarizability :math:`\alpha(\omega)` (C m^2/V, magnetic
    contribution included), so that ``chi = gamma * rho`` is linear density scaling,
    the dilute limit of optmat's own Lorentz--Lorenz law
    (:meth:`dilute_scaling_error` measures the difference). The derivative is
    optmat's exact :math:`\partial_\omega\alpha`, taken at fixed state.

    The model holds over its qualified ``intervals`` of vacuum wavelength. A band
    outside them is evaluated all the same and warns once per evaluation context
    (:class:`~solprop.exceptions.MaterialRangeWarning`), naming the parts outside
    and the reasons of any gaps. A transparent model returns a real
    susceptibility; an absorbing one a complex one with a non-negative imaginary
    part. Build one with :meth:`resolve`. The adapter keeps its last few
    evaluations by frequency array, as :class:`~solprop.physdata.glasses.OptmatGlass`
    does, and two adapters of one model compare equal by its identity digest
    (SEP 3, Q26).

    Attributes
    ----------
    model : OptmatModel
        The optmat dispersion model: a registry model, an ``AirModel``, or a model
        from ``with_parameters``.

    Examples
    --------
    >>> from solprop.physdata.sellmeier import OptmatSusceptibility
    >>> argon = OptmatSusceptibility.resolve("Ar")
    >>> argon.source
    'fit/Ar'
    >>> f"{float(argon.susceptibility(2.356e15)):.4e}"
    '2.0798e-29'
    """

    model: OptmatModel
    _evaluations: _optmat.EvaluationCache = field(
        init=False, repr=False, default_factory=_optmat.EvaluationCache
    )

    def __eq__(self, other: object) -> bool:
        """Compare two adapters by their model's identity digest, not as objects.

        The base class's field comparison would miss an override, which keeps the
        model's name.

        Parameters
        ----------
        other : object
            Another object.

        Returns
        -------
        bool
            Whether `other` is an adapter of the same gas and model digest.
        """
        if type(other) is not type(self):
            return NotImplemented
        same = cast("OptmatSusceptibility", other)
        return (self.gas, _optmat.record(self.model).digest) == (
            same.gas,
            _optmat.record(same.model).digest,
        )

    def __hash__(self) -> int:
        """Hash the adapter by its gas and its model's identity digest.

        Returns
        -------
        int
            The hash.
        """
        return hash((self.gas, _optmat.record(self.model).digest))

    @classmethod
    def resolve(
        cls, gas: Gas | str, model: ModelChoice = "recommended"
    ) -> "OptmatSusceptibility":  # noqa: UP037 - the hook resolves class names eagerly
        """Resolve a gas's optmat dispersion model and adapt it.

        Parameters
        ----------
        gas : Gas or str
            The species. ``Gas.Air`` is optmat's named composition's summed
            polarizabilities, ``"dry air (N2, O2, Ar)"`` by default.
        model : str or OptmatModel, optional
            ``"recommended"``, an optmat model name (for air, a composition name), or
            an optmat model object for `gas`, taken as it is.

        Returns
        -------
        OptmatSusceptibility
            The adapted model.

        Raises
        ------
        MaterialError
            If the gas is unknown, optmat has no such model, or the model describes
            another gas.

        Warns
        -----
        MaterialStatusWarning
            If the model is ``provisional`` or ``untested``.
        """
        species = as_gas(gas)
        resolved = _optmat.untyped(_optmat_gas_model(species, model))
        entry = _optmat.check_status(resolved, species.value)
        citation = getattr(resolved, "citation", "") or (
            f"optmat {entry.name}; sources {', '.join(resolved.identity.sources)}"
        )
        reference = resolved.identity.reference_temperature
        low, high = resolved.wavelength_range
        return cls(
            gas=species,
            source=entry.name,
            # The polarizability is a zero-density, per-particle quantity: there is no
            # finite reference state to divide out. The temperature is the one its
            # source refers it to, where it declares one.
            reference_state=ReferenceState(
                0.0, ROOM_TEMPERATURE if reference is None else float(reference)
            ),
            wavelength_range=(float(low), float(high)),
            citation=f"{citation} (optmat {entry.optmat_version}, {entry.name})",
            model=resolved,
        )

    def _chi_and_derivative(
        self,
        u: Float[np.ndarray, "*shape"],  # noqa: ARG002 - the abstract signature
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Refuse a reference-state susceptibility, which a polarizability has not.

        Parameters
        ----------
        u : ndarray
            Inverse squared wavelength in um^-2.

        Returns
        -------
        tuple of ndarray
            Never returns.

        Raises
        ------
        MaterialError
            Always: use :meth:`susceptibility`, which divides by no density.
        """
        msg = (
            f"{self.gas.value} {self.source}: an optmat polarizability is per particle "
            "and has no reference-state susceptibility"
        )
        raise MaterialError(msg)

    @property
    def reference_density(self) -> float:
        """Return the density of the reference state, the dilute limit.

        Returns
        -------
        float
            0.0 m^-3.
        """
        return 0.0

    @property
    def absorbing(self) -> bool:
        """Return whether the optmat model carries an imaginary polarizability.

        Returns
        -------
        bool
            The model's own ``absorbing``, or any constituent's for an air model.
        """
        return _model_absorbs(self.model)

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the record of the optmat model, for a run's manifest.

        Returns
        -------
        tuple of MaterialRecord
            The dispersion model's identity.
        """
        return (_optmat.record(self.model),)

    def _evaluate(
        self, omega: ArrayLike | float | int, method: str, quantity: str
    ) -> Inexact[np.ndarray, "*shape"]:
        """Evaluate one optmat method at angular frequencies, divided by eps0.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0; zero is the static limit.
        method : str
            ``"complex_polarizability"`` or ``"dpolarizability_domega"``.
        quantity : str
            What is evaluated, for messages.

        Returns
        -------
        ndarray
            The method's value over eps0, real for a transparent model.
        """
        values = _as_omega(omega)
        wavelength = _optmat.omega_to_wavelength(values)
        evaluator = getattr(self.model, method)
        # Per particle, the polarizability does not depend on z, yet a translation
        # reads it at every sample of a z table: each frequency array is evaluated
        # once (SEP 3, Q22).
        (sampled,) = self._evaluations.evaluate(
            quantity,
            values,
            self.model,
            material=self.gas.value,
            wavelength=wavelength,
            compute=lambda: (np.asarray(evaluator(wavelength)) / EPS0,),
        )
        # A transparent model's imaginary part is identically zero, not small: taking
        # the real part drops nothing and keeps the legacy real dtype downstream.
        return np.asarray(sampled if self.absorbing else np.real(sampled))

    def susceptibility(
        self, omega: ArrayLike | float | int
    ) -> Inexact[np.ndarray, "*shape"]:
        """Return the per-particle susceptibility ``alpha / eps0``.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0. Zero gives the static limit.

        Returns
        -------
        ndarray
            ``gamma(omega)`` in m^3, such that ``chi1 = gamma * rho``; complex, with
            ``Im gamma >= 0``, for an absorbing model.

        Warns
        -----
        MaterialRangeWarning
            If the band leaves the model's qualified intervals.
        """
        return self._evaluate(omega, "complex_polarizability", "polarizability")

    def dsusceptibility_domega(
        self, omega: ArrayLike | float | int
    ) -> Inexact[np.ndarray, "*shape"]:
        """Return optmat's exact frequency derivative of the susceptibility.

        Parameters
        ----------
        omega : array_like
            Angular frequency in rad/s, >= 0.

        Returns
        -------
        ndarray
            ``dgamma/domega`` in m^3 s/rad.

        Warns
        -----
        MaterialRangeWarning
            If the band leaves the model's qualified intervals.
        """
        return self._evaluate(
            omega, "dpolarizability_domega", "polarizability derivative"
        )

    def dilute_scaling_error(
        self,
        density: float,
        temperature: float,
        omega: Float[np.ndarray, " n_omega"],
    ) -> tuple[float, float]:
        r"""Compare linear density scaling with optmat's finite-density index law.

        ``(n - 1)`` from :math:`\chi = N\alpha/\varepsilon_0` and
        :math:`n - 1 = \chi/(\sqrt{1+\chi}+1)` is compared with optmat's
        ``refractivity`` (Lorentz--Lorenz with the refractivity virial) at the
        same number density. The comparison is made at the density itself: the
        state handed to optmat is the ideal-gas one, ``p = N k_B T`` with
        ``ideal=True``, whose density is `density` exactly, so no equation of
        state or its inverse is consulted. Wavelengths outside the model's
        qualified intervals are left out, unless none is inside.

        Parameters
        ----------
        density : float
            Number density in m^-3, > 0.
        temperature : float
            Temperature in K, > 0.
        omega : numpy.ndarray
            Angular frequencies in rad/s of the band, > 0.

        Returns
        -------
        tuple of float
            The relative error ``(n-1)_dilute / (n-1)_optmat - 1`` of largest
            magnitude over the band, and the vacuum wavelength in m where it is.

        Raises
        ------
        MaterialError
            If optmat refuses the evaluation, or the band is empty.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.physdata.sellmeier import OptmatSusceptibility
        >>> argon = OptmatSusceptibility.resolve("Ar")
        >>> error, _ = argon.dilute_scaling_error(2.47e26, 293.15, np.array([2.356e15]))
        >>> f"{error:.2e}"
        '-1.88e-03'
        """
        from optmat.conditions import State  # noqa: PLC0415 - optmat is lazy

        values = np.asarray(omega, dtype=float).ravel()
        wavelength = _optmat.omega_to_wavelength(values[values > 0])
        if not wavelength.size:
            msg = f"{self.gas.value}: the dilute-scaling check needs a positive band"
            raise MaterialError(msg)
        model = _optmat.untyped(self.model)
        inside = np.zeros(wavelength.shape, dtype=bool)
        for low, high in model.intervals:
            inside |= (wavelength >= low) & (wavelength <= high)
        if np.any(inside):
            wavelength = wavelength[inside]
        state = f"{density:.6g} m^-3 and {temperature:g} K"
        # Beside a pole of the law 1 + chi can fall below zero, in optmat's refractivity
        # and in the comparison below; the error there is NaN, as before, without a
        # RuntimeWarning at every sample of a pressure search.
        with (
            np.errstate(invalid="ignore"),
            _optmat.translated(
                self.model,
                "dilute-scaling check",
                material=self.gas.value,
                wavelength=wavelength,
                state=state,
            ),
        ):
            # optmat's ideal-gas state of this density: its law then takes N = p /
            # (k_B T), which is `density` to one rounding, and the refractivity
            # virial still enters. Its real-gas inverse is not needed (and before
            # optmat 0.1.1 it failed for Ar, N2 and CH4 below about 255-284 K).
            ideal = State(density * K_B * temperature, temperature)
            exact = np.asarray(
                model.refractivity(wavelength, ideal, ideal=True, warn=False)
            )
            alpha = np.asarray(model.polarizability(wavelength, warn=False))
            chi = density * alpha / EPS0
            error = (chi / (np.sqrt(1.0 + chi) + 1.0)) / exact - 1.0
        worst = int(np.argmax(np.abs(error)))
        return float(error[worst]), float(wavelength[worst])


@functools.cache
def _default_law(gas: Gas) -> "OptmatSusceptibility":  # noqa: UP037 - defined above
    """Return optmat's recommended law of a gas, one adapter per gas.

    Parameters
    ----------
    gas : Gas
        The species.

    Returns
    -------
    OptmatSusceptibility
        The adapter, which keeps its own evaluations.
    """
    return OptmatSusceptibility.resolve(gas)


def gamma(gas: Gas | str, *, model: ModelChoice = "recommended") -> GasSusceptibility:
    """Return the per-particle susceptibility model for a gas: optmat's.

    The returned object exposes ``susceptibility(omega)`` in m^3, its exact
    derivative ``dsusceptibility_domega(omega)``, the ``wavelength_range`` its
    qualified intervals span, and the ``citation``.

    Parameters
    ----------
    gas : Gas or str
        The species.
    model : str or OptmatModel, optional
        ``"recommended"`` (the default), an optmat model name of this gas, such as
        ``"borzsonyi_2008/sellmeier_Ar"``, or an optmat model object.

    Returns
    -------
    GasSusceptibility
        The :class:`OptmatSusceptibility`.

    Raises
    ------
    MaterialError
        If the gas is unknown, optmat has no such model, or it describes another gas.

    Examples
    --------
    >>> from solprop.physdata import Gas, gamma
    >>> gamma(Gas.Kr).source
    'fit/Kr'
    >>> gamma(Gas.Kr, model="borzsonyi_2008/sellmeier_Kr").source
    'borzsonyi_2008/sellmeier_Kr'
    """
    return OptmatSusceptibility.resolve(gas, model)


def chi1(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return the linear susceptibility of a gas.

    ``chi1 = gamma(omega) * rho``, with the number density from the equation of state.

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s, >= 0.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    ndarray
        Dimensionless susceptibility, 0-d for scalar input.

    Warns
    -----
    MaterialRangeWarning
        If any wavelength lies outside the validity range of the source.

    Examples
    --------
    >>> from solprop.physdata import Gas, chi1
    >>> f"{float(chi1(Gas.Ar, 2.356e15, pressure=1.0)):.6e}"
    '5.142114e-04'
    """
    model = _default_law(as_gas(gas))
    rho = density(gas, pressure, temperature)
    return np.asarray(model.susceptibility(omega) * rho)


def dchi1_domega(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return the analytic frequency derivative of the linear susceptibility.

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s, >= 0.
    pressure : float
        Absolute pressure in **bar**, >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    ndarray
        ``dchi1/domega`` in s/rad, 0-d for scalar input.

    Warns
    -----
    MaterialRangeWarning
        If any wavelength lies outside the validity range of the source.

    Examples
    --------
    >>> from solprop.physdata import Gas, dchi1_domega
    >>> f"{float(dchi1_domega(Gas.Ar, 2.356e15, pressure=1.0)):.6e}"
    '3.630134e-21'
    """
    model = _default_law(as_gas(gas))
    rho = density(gas, pressure, temperature)
    return np.asarray(model.dsusceptibility_domega(omega) * rho)


def _one_plus_chi(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    pressure: float,
    temperature: float,
) -> Float[np.ndarray, "*shape"]:
    """Return ``1 + chi1``, checked to be positive so that the square root is real.

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s.
    pressure : float
        Absolute pressure in bar.
    temperature : float
        Temperature in K.

    Returns
    -------
    ndarray
        ``1 + chi1``.

    Raises
    ------
    MaterialError
        If the susceptibility is at or below -1, which happens only when a model is
        extrapolated onto a pole.
    """
    values = np.asarray(
        1.0 + chi1(gas, omega, pressure=pressure, temperature=temperature)
    )
    if np.any(values <= 0.0):
        raise MaterialError(
            f"the index model for {as_gas(gas).value} gives a non-positive dielectric "
            "constant here; it has been extrapolated onto a pole of the expansion"
        )
    return values


def refractive_index(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return the refractive index of a gas.

    ``n = sqrt(1 + chi1)``, real: these models carry no absorption. Where the quantity
    of interest is ``n - 1``, use :func:`refractive_index_minus_one`, which avoids the
    cancellation.

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s, >= 0.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    ndarray
        Refractive index, 0-d for scalar input.

    Warns
    -----
    MaterialRangeWarning
        If any wavelength lies outside the validity range of the source.

    Examples
    --------
    Argon at 800 nm and 1 bar.

    >>> from solprop.physdata import Gas, refractive_index
    >>> f"{float(refractive_index(Gas.Ar, 2.356e15, pressure=1.0)):.9f}"
    '1.000257073'
    """
    return np.asarray(np.sqrt(_one_plus_chi(gas, omega, pressure, temperature)))


def refractive_index_minus_one(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return ``n - 1`` for a gas, without cancellation.

    Computed as ``chi1 / (1 + sqrt(1 + chi1))``, which is exact for the small
    susceptibilities of a gas; forming ``sqrt(1 + chi1) - 1`` instead would lose about
    four significant figures at 1 bar (``docs/dev/conventions.md`` section 5).

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s, >= 0.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    ndarray
        ``n - 1``, 0-d for scalar input.

    Warns
    -----
    MaterialRangeWarning
        If any wavelength lies outside the validity range of the source.

    Examples
    --------
    >>> from solprop.physdata import Gas, refractive_index_minus_one
    >>> f"{float(refractive_index_minus_one(Gas.Ar, 2.356e15, pressure=1.0)):.6e}"
    '2.570726e-04'
    """
    values = _one_plus_chi(gas, omega, pressure, temperature)
    return np.asarray((values - 1.0) / (1.0 + np.sqrt(values)))


def drefractive_index_domega(
    gas: Gas | str,
    omega: ArrayLike | float | int,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return the analytic frequency derivative of the refractive index.

    ``dn/domega = (dchi1/domega) / (2 n)``. Together with ``n`` this gives the group
    index and the group delay per unit length, ``beta_1 = (n + omega dn/domega) / c``,
    which the co-moving frame needs to about 1e-10.

    Parameters
    ----------
    gas : Gas or str
        The species.
    omega : array_like
        Angular frequency in rad/s, >= 0.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    ndarray
        ``dn/domega`` in s/rad, 0-d for scalar input.

    Warns
    -----
    MaterialRangeWarning
        If any wavelength lies outside the validity range of the source.

    Examples
    --------
    >>> from solprop.physdata import Gas, drefractive_index_domega
    >>> f"{float(drefractive_index_domega(Gas.Ar, 2.356e15, pressure=1.0)):.6e}"
    '1.814601e-21'
    """
    one_plus_chi = _one_plus_chi(gas, omega, pressure, temperature)
    derivative = dchi1_domega(gas, omega, pressure=pressure, temperature=temperature)
    return np.asarray(derivative / (2.0 * np.sqrt(one_plus_chi)))

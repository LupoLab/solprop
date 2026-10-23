# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strong-field ionisation: the rate protocol, its conventions and diagnostics.

The rate ``W(|E|)`` is the probability per unit time that one atom loses its outermost
electron in a field of amplitude ``|E|``. It is the input to the plasma response: the
ionised fraction is ``1 - exp(-int W dt)`` and the energy the field pays for each
electron is the energy per event.

solprop keeps no ionisation-rate formula of its own (SEP 3, D1). Every rate of a
built-in gas, and of a custom species from explicit parameters, is one of optmat's
rate models, adapted by :class:`solprop.nonlinear.optmat_rates.OptmatRate`; a caller may
also supply any object satisfying :class:`IonisationRate` (or the narrower
:class:`EvaluableRate`), or a tabulated :class:`solprop.translate.ionisation.RateTable`.
This module holds the contract they meet, their optical-cycle conventions, the energy
binding of the plasma loss (:func:`bind_rate_energy`), the rate ceiling
:data:`MAX_RATE`, and the diagnostics :func:`keldysh`,
:func:`barrier_suppression_field` and :func:`ionisation_fraction`.

The manual page is ``docs/guide/media/ionisation.md``. Rates are per atom or molecule,
in
s^-1, for a field amplitude in V/m, and are exactly zero below the rate's threshold
field and clipped at its ceiling so that a diverged step cannot put a NaN into the
plasma column (``docs/dev/stepper.md`` section 10, trap 9).
"""

import dataclasses
import logging
import math
from typing import Final, Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike
from scipy import integrate

from solprop.constants import AU_EFIELD, AU_ENERGY, AU_TIME, C
from solprop.exceptions import GridError, MaterialError
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.species import (
    IonisationData,
    Species,
    ionisation_potential,
    species,
)

__all__ = [
    "DT_RTOL",
    "MAX_RATE",
    "EvaluableRate",
    "IntegrationPolicy",
    "IonisationRate",
    "RateConvention",
    "RunningIntegral",
    "barrier_suppression_field",
    "bind_rate_energy",
    "ionisation_fraction",
    "keldysh",
    "rate_convention",
    "rate_energy",
    "rate_species",
]

logger = logging.getLogger(__name__)

RateConvention = Literal["instantaneous-isotropic", "cycle-averaged", "unspecified"]
"""Optical-cycle convention of a magnitude rate, distinct from orientation averaging."""

MAX_RATE: Final[float] = 1.0 / AU_TIME
"""Default ceiling on an ionisation rate, s^-1.

One ionisation per atomic unit of time (4.13e16 s^-1). No quasistatic rate means
anything at or beyond this -- the electron would leave in less than an orbit -- so
clipping here cannot touch a physical result, and it is what stops a diverged step from
propagating an overflow into the plasma column.
"""


@runtime_checkable
class EvaluableRate(Protocol):
    """Anything that turns a field amplitude into an ionisation rate.

    The narrow contract a consumer of a rate actually needs: call it with an amplitude
    in V/m and get a rate per atom in s^-1. Both an :class:`IonisationRate` (optmat's
    :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or a caller's) and the
    tabulated
    :class:`solprop.translate.ionisation.RateTable` satisfy it, structurally and without
    inheriting from it.

    It exists because the plasma response
    (:class:`solprop.nonlinear.responses.PlasmaResponse`) accepts either, and a response
    is in the physics layer while a table is in the translation layer: imports run
    P to T and never back (``docs/dev/architecture.md`` section 3), so the response
    cannot name ``RateTable`` even in an annotation, because the test session's
    ``beartype`` hook checks annotations at run time.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import EvaluableRate
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> isinstance(OptmatRate("Ar", 800e-9), EvaluableRate)
    True
    """

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the rate for a field amplitude.

        Parameters
        ----------
        field_amplitude : array_like or float
            Field amplitude in V/m; the sign is ignored.

        Returns
        -------
        numpy.ndarray
            Rate per atom in s^-1, with the shape of ``field_amplitude``.
        """
        ...


@runtime_checkable
class IonisationRate(Protocol):
    """A field-dependent single-ionisation rate per atom or molecule, in s^-1.

    Implementations are vectorised in the field amplitude, even in its sign, exactly
    zero below :attr:`threshold` and never above :attr:`maximum`, so the translation
    layer can tabulate them and a kernel can evaluate the table without a divergence
    branch.
    """

    @property
    def threshold(self) -> float:
        """Field amplitude, V/m, below which the rate is exactly zero."""
        ...

    @property
    def maximum(self) -> float:
        """Ceiling on the rate, s^-1; the value is clipped there."""
        ...

    @property
    def citation(self) -> str:
        """One-line reference for the model and the options in force."""
        ...

    @property
    def parameters(self) -> tuple[tuple[str, float | int | bool | str], ...]:
        """Every setting that changes the rate, name-sorted and hashable.

        The translation layer builds its table cache key from this, so two rates that
        differ in any way that matters give different keys and no ordering of keyword
        arguments can give two keys for one rate.
        """
        ...

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the rate for a field amplitude.

        Parameters
        ----------
        field_amplitude : array_like or float
            Field amplitude in V/m; the sign is ignored.

        Returns
        -------
        numpy.ndarray
            Rate per atom or molecule in s^-1, with the shape of ``field_amplitude``.
        """
        ...


def rate_convention(rate: EvaluableRate) -> RateConvention:
    """Read an explicit optical-cycle convention without guessing for legacy rates.

    Parameters
    ----------
    rate : EvaluableRate
        Magnitude rate, optionally declaring a ``convention`` attribute.

    Returns
    -------
    str
        ``'instantaneous-isotropic'``, ``'cycle-averaged'`` or ``'unspecified'``.
        Missing or unrecognised declarations are unspecified. Field-grid consumers
        retain their existing acceptance rules; envelope plasma requires the first.
    """
    declared = getattr(rate, "convention", None)
    if declared == "instantaneous-isotropic":
        return "instantaneous-isotropic"
    if declared == "cycle-averaged":
        return "cycle-averaged"
    return "unspecified"


def rate_species(rate: EvaluableRate) -> tuple[str | None, float | None]:
    """Read a rate's declared species name and binding energy.

    Parameters
    ----------
    rate : EvaluableRate
        Analytic rate with ``gas``, or a table retaining ``species_identity`` and
        ``ionisation_potential``. An anonymous callable has neither.

    Returns
    -------
    tuple of (str or None, float or None)
        Chemical species name and first ionisation potential in J. Anonymous rates
        return two ``None`` values; their response supplies the material binding.
    """
    gas = getattr(rate, "gas", None)
    if isinstance(gas, Species):
        potential = None if gas.ionisation is None else gas.ionisation.potential
        return gas.name, potential
    if isinstance(gas, (Gas, str)):
        return as_gas(gas).value, float(ionisation_potential(gas))
    return (
        getattr(rate, "species_identity", None),
        getattr(rate, "ionisation_potential", None),
    )


def rate_energy(rate: EvaluableRate) -> tuple[str, float] | None:
    """Read the energy per ionisation event a rate declares for the plasma loss.

    A rate that carries its own energy per event (SEP 3, D8: the loss uses the energy
    the rate's exponent uses, :class:`solprop.nonlinear.optmat_rates.OptmatRate`)
    declares it as ``plasma_energy`` beside its ``species_identity``. Every other rate
    declares none, and its response takes the species' ionisation potential, as
    before.

    Parameters
    ----------
    rate : EvaluableRate
        Any rate.

    Returns
    -------
    tuple of (str, float) or None
        The species name and the energy per event in J, or ``None`` for a rate that
        declares no energy.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import rate_energy
    >>> def rate(field):
    ...     return 0.0 * field
    >>> rate_energy(rate) is None
    True
    """
    energy = getattr(rate, "plasma_energy", None)
    if energy is None:
        return None
    return str(getattr(rate, "species_identity", "")), float(energy)


def bind_rate_energy(
    gas: Gas | str | Species, rate: EvaluableRate
) -> Gas | str | Species:
    """Return the species a plasma response takes, its energy the rate's own.

    For a rate that declares its energy per event (:func:`rate_energy`), the species
    with its ionisation potential replaced by that energy: the plasma loss then pays
    the energy the rate's exponent uses, and a response's potential, its compiled
    channel and the envelope's consistency check all agree with the rate. A rate that
    declares none leaves the species exactly as given.

    Parameters
    ----------
    gas : Gas, str or Species
        The response's species.
    rate : EvaluableRate
        The response's rate.

    Returns
    -------
    Gas, str or Species
        The species as given, or a :class:`~solprop.physdata.species.Species` whose
        ionisation potential is the rate's energy in J.

    Raises
    ------
    MaterialError
        If the rate declares an energy for another species.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import bind_rate_energy
    >>> def rate(field):
    ...     return 0.0 * field
    >>> bind_rate_energy("Ar", rate)
    'Ar'
    """
    declared = rate_energy(rate)
    if declared is None:
        return gas
    name, energy = declared
    base = gas if isinstance(gas, Species) else species(gas)
    if base.name != name:
        msg = (
            f"the plasma rate is {name}'s and the response's species is {base.name}; "
            "a rate that carries its own energy per event must name the response's "
            "species"
        )
        raise MaterialError(msg)
    citation = f"energy per event of the rate {getattr(rate, 'citation', '')}"
    data = base.ionisation
    bound = (
        IonisationData(energy, citation=citation)
        if data is None
        else dataclasses.replace(data, potential=energy, citation=citation)
    )
    return dataclasses.replace(base, ionisation=bound)


def barrier_suppression_field(ionisation_potential_j: float, charge: float) -> float:
    """Return the field at which the Coulomb barrier is suppressed below the level.

    Classically the electron is free once the field has pulled the barrier down to the
    bound-state energy, which happens at ``E_b = Ip^2 / (4 Z)`` in atomic units. It is
    the natural scale for the barrier-suppression corrections and for the upper end of a
    rate table.

    Parameters
    ----------
    ionisation_potential_j : float
        Ionisation potential in joules.
    charge : float
        Charge seen by the departing electron, in units of the elementary charge.

    Returns
    -------
    float
        Barrier-suppression field amplitude in V/m.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import barrier_suppression_field
    >>> from solprop.physdata import Gas, ionisation_potential
    >>> f"{barrier_suppression_field(ionisation_potential(Gas.He), 1.0):.3e}"
    '1.050e+11'
    """
    ip_au = ionisation_potential_j / AU_ENERGY
    return ip_au**2 / (4.0 * charge) * AU_EFIELD


def keldysh(
    gas: Gas | str, wavelength: float, field_amplitude: ArrayLike | float
) -> Float[np.ndarray, " *shape"]:
    """Return the Keldysh parameter of a gas at a wavelength and field amplitude.

    ``gamma = omega sqrt(2 Ip) / E`` in atomic units: the ratio of the laser period to
    the time the electron needs to tunnel. Ionisation is multiphoton for ``gamma >> 1``
    and quasistatic tunnelling for ``gamma << 1``.

    Parameters
    ----------
    gas : Gas or str
        The species; its first ionisation energy is optmat's
        (:func:`~solprop.physdata.species.ionisation_potential`).
    wavelength : float
        Driving wavelength in metres.
    field_amplitude : array_like or float
        Field amplitude in V/m; the sign is ignored.

    Returns
    -------
    numpy.ndarray
        Keldysh parameter, dimensionless, with the shape of ``field_amplitude``.

    Raises
    ------
    MaterialError
        If the gas is unknown, has no ionisation potential, or the wavelength or field
        is not positive and finite.

    Examples
    --------
    >>> from solprop.nonlinear.ionisation import keldysh
    >>> f"{keldysh('Ar', 800e-9, 5e10):.3f}"
    '0.630'
    """
    _validate_positive("wavelength", wavelength)
    ip_au = ionisation_potential(gas) / AU_ENERGY
    omega0_au = 2.0 * math.pi * C / wavelength * AU_TIME
    field = _finite_field(field_amplitude)
    return np.asarray(omega0_au * math.sqrt(2.0 * ip_au) / (field / AU_EFIELD))


IntegrationPolicy = Literal["band", "trapezoid"]
"""How the plasma column integrates along t (SEP 18).

``"band"`` filters each integrand with the grid's designed symmetric taps before the
trapezoid, so that the running integral is exact to a certified tolerance across the
active band; ``"trapezoid"`` is the plain cumulative trapezoid, kept permanently for the
frozen witness comparisons.
"""

DT_RTOL: Final[float] = 1e-12
"""Relative tolerance on the spacing a band :class:`RunningIntegral` is used at."""


@dataclasses.dataclass(frozen=True, eq=False)
class RunningIntegral:
    r"""The resolved running-integral rule for field-grid plasma (SEP 18).

    Each running integral ``int y dt`` is the cumulative trapezoid of a filtered
    integrand,

    .. math::

        \tilde y_n = c_0 y_n + \sum_{k=1}^{K} c_k (y_{n-k} + y_{n+k}),

    with symmetric taps ``c`` whose response, times the trapezoid's own multiplier
    ``(theta/2) cot(theta/2)``, is one to within ``certified_error`` across the active
    band. The taps are a property of the grid: the translation layer designs them
    (:func:`solprop.translate.plasma_integration.design_running_integral`), and this
    value type only carries and applies them, so the physics layer never fits anything
    itself. ``K`` samples beyond each end of the window are supplied by point
    reflection, ``y_{-m} = 2 y_0 - y_m``, which is exact for constant and linear
    integrands. The exposure is floored at zero rather than clamped, so the fraction
    stays in ``[f_0, 1)`` and is non-decreasing to within the filter's undershoot.

    :meth:`trapezoid` is the legacy rule: one identity tap, no floor, and the
    arithmetic of :func:`scipy.integrate.cumulative_trapezoid` unchanged.

    Attributes
    ----------
    policy : {'band', 'trapezoid'}
        The rule.
    taps : numpy.ndarray
        ``(c_0, c_1, ..., c_K)``, centre first, read-only float64, shape ``(K + 1,)``;
        ``(1.0,)`` for the trapezoid. The filter has ``2K + 1`` points.
    dt : float or None
        The sample spacing the taps were designed for, s; ``None`` for the trapezoid,
        which is the same rule at every spacing.
    band_edge : float or None
        The upper edge of the active band the taps were fitted to, rad/s; ``None`` for
        the trapezoid.
    certified_error : float
        The certified ``max |H - 1|`` across the band, where ``H`` is the composite
        multiplier relative to the exact integral; zero for the trapezoid, whose error
        is not designed.
    boundary : str
        How the filter reads past the window's ends: ``'point-reflection'``, or
        ``'none'`` for the trapezoid.
    exposure : str
        The positivity policy of the exposure: ``'floor-at-zero'``, or ``'none'`` for
        the trapezoid.

    Raises
    ------
    GridError
        If the taps are not a non-empty one-dimensional finite array, the trapezoid
        carries anything but the identity tap, or a band rule has no positive finite
        ``dt`` and ``band_edge``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.ionisation import RunningIntegral
    >>> legacy = RunningIntegral.trapezoid()
    >>> legacy.integrate(np.ones(4), 0.5).tolist()
    [0.0, 0.5, 1.0, 1.5]
    """

    policy: IntegrationPolicy
    taps: Float[np.ndarray, " n_taps"]
    dt: float | None = None
    band_edge: float | None = None
    certified_error: float = 0.0
    boundary: str = "point-reflection"
    exposure: str = "floor-at-zero"

    def __post_init__(self) -> None:
        """Freeze the taps and check the rule is complete.

        Raises
        ------
        GridError
            If the rule is incomplete or inconsistent.
        """
        taps = np.array(self.taps, dtype=np.float64, order="C")
        taps.setflags(write=False)
        object.__setattr__(self, "taps", taps)
        if taps.ndim != 1 or taps.size == 0 or not np.all(np.isfinite(taps)):
            msg = f"taps must be a non-empty finite 1-D array, got shape {taps.shape}"
            raise GridError(msg)
        if self.policy == "trapezoid":
            if (
                taps.tolist() != [1.0]
                or self.boundary != "none"
                or self.exposure != ("none")
            ):
                msg = "the trapezoid rule is one identity tap with no boundary or floor"
                raise GridError(msg)
            return
        # The annotation refuses another rule first wherever the annotations are
        # checked at run time, as in the test suite.
        if self.policy != "band":  # pragma: no cover
            msg = f"integration must be 'band' or 'trapezoid', got {self.policy!r}"
            raise GridError(msg)
        for name in ("dt", "band_edge"):
            value = getattr(self, name)
            if value is None or not (math.isfinite(value) and value > 0.0):
                msg = f"a band rule needs a positive finite {name}, got {value!r}"
                raise GridError(msg)
        if (self.boundary, self.exposure) != ("point-reflection", "floor-at-zero"):
            msg = (
                "a band rule closes its ends by point reflection and floors the "
                f"exposure at zero, got {self.boundary!r} and {self.exposure!r}"
            )
            raise GridError(msg)

    @classmethod
    def trapezoid(cls) -> "RunningIntegral":  # noqa: UP037 - the runtime type hook resolves this while the class body runs
        """Return the legacy rule: the plain cumulative trapezoid.

        Returns
        -------
        RunningIntegral
            One identity tap, no boundary rule and no exposure floor.

        Examples
        --------
        >>> from solprop.nonlinear.ionisation import RunningIntegral
        >>> RunningIntegral.trapezoid().taps.tolist()
        [1.0]
        """
        return cls("trapezoid", np.ones(1), boundary="none", exposure="none")

    @property
    def half_width(self) -> int:
        """Return ``K``, the number of samples the filter reads on each side.

        Returns
        -------
        int
            ``len(taps) - 1``; zero for the trapezoid.
        """
        return self.taps.size - 1

    def check(self, dt: float) -> None:
        """Refuse a band rule used at a sample spacing it was not designed for.

        Parameters
        ----------
        dt : float
            The spacing of the samples about to be integrated, s.

        Raises
        ------
        GridError
            If the rule is ``band`` and ``dt`` differs from its own by more than
            :data:`DT_RTOL` relative.
        """
        if self.dt is not None and not abs(dt - self.dt) <= DT_RTOL * self.dt:
            msg = (
                f"this running integral was designed for dt = {self.dt!r} s and cannot "
                f"integrate samples spaced {dt!r} s; design it for this grid"
            )
            raise GridError(msg)

    def integrate(
        self, values: Float[np.ndarray, "... n_to"], dt: float
    ) -> Float[np.ndarray, "... n_to"]:
        """Return the running integral of `values` along their last axis.

        The filter's sum has a fixed order, which the compiled backends reproduce: the
        centre term first, then the pairs ``k = 1 ... K``, each pair added before it is
        weighted. The virtual samples are formed as ``2 y_0 - y_m``. The integral starts
        at zero, as the trapezoid's does.

        Parameters
        ----------
        values : numpy.ndarray
            The integrand, sampled uniformly along the last axis, at least ``K + 1``
            samples long.
        dt : float
            The sample spacing, s.

        Returns
        -------
        numpy.ndarray
            The running integral, the shape of `values`.

        Raises
        ------
        GridError
            If `dt` is not the band rule's own, or the axis is shorter than ``K + 1``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.ionisation import RunningIntegral
        >>> taps = np.array([1.1, -0.05])
        >>> rule = RunningIntegral("band", taps, dt=0.5, band_edge=1.0)
        >>> rule.integrate(np.arange(4.0), 0.5).tolist()
        [0.0, 0.25, 1.0, 2.25]
        """
        self.check(dt)
        half = self.half_width
        if half == 0:
            return integrate.cumulative_trapezoid(values, dx=dt, initial=0, axis=-1)
        return integrate.cumulative_trapezoid(
            self.filter(values), dx=dt, initial=0, axis=-1
        )

    def filter(
        self, values: Float[np.ndarray, "... n_to"]
    ) -> Float[np.ndarray, "... n_to"]:
        """Return the filtered integrand, with its ends closed by point reflection.

        Parameters
        ----------
        values : numpy.ndarray
            The integrand along the last axis, at least ``K + 1`` samples long.

        Returns
        -------
        numpy.ndarray
            ``c_0 y_n + sum_k c_k (y_{n-k} + y_{n+k})``, the shape of `values`.

        Raises
        ------
        GridError
            If the axis is shorter than ``K + 1``, so the reflection has nothing to
            reflect.
        """
        half = self.half_width
        values = np.asarray(values, dtype=np.float64)
        n_to = values.shape[-1]
        if n_to <= half:
            msg = (
                f"a filter reading {half} samples each side needs at least {half + 1} "
                f"samples, got {n_to}"
            )
            raise GridError(msg)
        reach = np.arange(half, 0, -1)
        head = 2.0 * values[..., :1] - values[..., reach]
        tail = 2.0 * values[..., -1:] - values[..., n_to - 1 - reach[::-1]]
        extended = np.concatenate([head, values, tail], axis=-1)
        taps = self.taps
        filtered = taps[0] * values
        for k in range(1, half + 1):
            filtered = filtered + taps[k] * (
                extended[..., half - k : half - k + n_to]
                + extended[..., half + k : half + k + n_to]
            )
        return filtered

    def floor(
        self, exposure: Float[np.ndarray, "*shape"]
    ) -> Float[np.ndarray, "*shape"]:
        """Apply the exposure policy: the floor at zero, or nothing for the trapezoid.

        The floor is pointwise, not a running maximum, so it accumulates no bias: the
        final exposure is the unfloored one wherever that is positive.

        Parameters
        ----------
        exposure : numpy.ndarray
            The running integral of the rate, dimensionless.

        Returns
        -------
        numpy.ndarray
            ``max(exposure, 0)`` under the band rule, `exposure` itself otherwise.
        """
        if self.policy == "trapezoid":
            return exposure
        return np.maximum(exposure, 0.0)


def ionisation_fraction(
    rate: EvaluableRate,
    field: Float[np.ndarray, " n_to"],
    dt: float,
    *,
    preionisation: float = 0.0,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, " n_to"]:
    """Return the ionised fraction along a time axis for a given field.

    Each atom or molecule survives with probability ``exp(-int W dt)``. The fraction is
    ``f = 1 - (1 - f0) exp(-int W dt)`` where ``f0`` is the fraction already ionised
    before the pulse. Carrying ``f0`` in the survival factor keeps ``f`` at or below
    one.

    The integral is the plasma column's own: pass the problem's
    :class:`RunningIntegral` to get the fraction a propagation forms under the band rule
    (SEP 18). Without one this is the plain cumulative trapezoid.

    Parameters
    ----------
    rate : EvaluableRate
        The rate to integrate, s^-1 as a function of field amplitude in V/m: an
        :class:`IonisationRate` or a tabulated
        :class:`solprop.translate.ionisation.RateTable`.
    field : numpy.ndarray
        Field amplitude in V/m, shape ``(n_to,)``, sampled uniformly in time.
    dt : float
        Sample spacing in seconds.
    preionisation : float, optional
        Fraction already ionised before the pulse, in ``[0, 1]``, at the one position
        this field is sampled at. A response whose preionisation is a path along z is
        resolved to its value at a plane before this is called.
    integral : RunningIntegral or None, optional
        The running-integral rule; ``None`` is :meth:`RunningIntegral.trapezoid`.

    Returns
    -------
    numpy.ndarray
        Ionised fraction in ``[preionisation, 1]``, shape ``(n_to,)``, non-decreasing in
        time (under the band rule, to within the filter's undershoot).

    Raises
    ------
    MaterialError
        If ``dt`` is not positive and finite or ``preionisation`` is outside ``[0, 1]``.
    GridError
        If `integral` was designed for another ``dt``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.ionisation import ionisation_fraction
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> t = np.linspace(-2e-14, 2e-14, 129)
    >>> field = 4e10 * np.exp(-((t / 5e-15) ** 2))
    >>> rate = OptmatRate("Ar", 800e-9)
    >>> fraction = ionisation_fraction(rate, field, float(t[1] - t[0]))
    >>> f"{fraction[-1]:.4f}"
    '0.2148'
    """
    _validate_positive("dt", dt)
    if not 0.0 <= preionisation <= 1.0:
        msg = f"preionisation must be in [0, 1], got {preionisation!r}"
        raise MaterialError(msg)
    rule = RunningIntegral.trapezoid() if integral is None else integral
    exposure = rule.floor(rule.integrate(rate(field), dt))
    return 1.0 - (1.0 - preionisation) * np.exp(-exposure)


def _validate_positive(name: str, value: float) -> None:
    """Raise unless ``value`` is a positive, finite number.

    Parameters
    ----------
    name : str
        The name to quote in the message.
    value : float
        The number to check.

    Raises
    ------
    MaterialError
        Naming the offending argument.
    """
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be positive and finite, got {value!r}"
        raise MaterialError(msg)


def _finite_field(field_amplitude: ArrayLike | float) -> Float[np.ndarray, " *shape"]:
    """Return ``|field_amplitude|`` as a float array, rejecting anything not finite.

    Parameters
    ----------
    field_amplitude : array_like or float
        Field amplitude in V/m.

    Returns
    -------
    numpy.ndarray
        The absolute value as float64, with the shape of the argument.

    Raises
    ------
    MaterialError
        If any element is a NaN or an infinity.
    """
    # np.abs on a zero-dimensional array returns a numpy scalar rather than an array,
    # so the result is wrapped again: every public rate returns an ndarray, including
    # for a scalar argument, as the rest of the package does.
    field = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
    if not np.all(np.isfinite(field)):
        msg = "field amplitude must be finite; a NaN or infinity reached the rate"
        raise MaterialError(msg)
    return field

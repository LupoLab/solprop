# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cycle-averaged plasma sources for scalar and Cartesian vector envelopes.

The physical envelope is in V/m. This independent oracle integrates the real
Cartesian field around an optical cycle with adaptive quadrature. It does not
use the ellipse rules or tables compiled for a backend. The two returned current
sources have different frequency factors; neither is a time-domain polarisation.
See ``docs/guide/media/ionisation.md`` for the derivation and the frozen-cycle regime.
"""

import itertools
import math
from dataclasses import KW_ONLY, dataclass
from typing import Final

import numpy as np
from jaxtyping import Complex, Float
from scipy import integrate, optimize

from solprop.constants import CHARGE_MASS_RATIO
from solprop.exceptions import ConvergenceError, GridError, MaterialError
from solprop.nonlinear.ionisation import (
    EvaluableRate,
    bind_rate_energy,
    rate_convention,
    rate_species,
)
from solprop.nonlinear.responses import ResponseTraits
from solprop.physdata import Gas, as_gas, ionisation_potential
from solprop.physdata.mixtures import check_preionisation
from solprop.physdata.species import Species
from solprop.profiles import ZProfile

__all__ = [
    "EnvelopePlasmaResponse",
    "EnvelopePlasmaSources",
    "linear_cycle_average",
    "linear_cycle_averages",
]


@dataclass(frozen=True, eq=False)
class EnvelopePlasmaSources:
    """Own the population and the two distinct envelope plasma source arrays.

    Attributes
    ----------
    ionisation_rate : numpy.ndarray
        Cycle-averaged rate in s^-1, shape ``(n_to,)``, finite and non-negative.
    fraction : numpy.ndarray
        Ionised fraction in ``[0, 1]``, shape ``(n_to,)``.
    drude_drive : numpy.ndarray
        ``(e²/m_e) density fraction field_t``, in A m^-2 s^-1, complex,
        shape ``(n_pol, n_to)`` with one or two Cartesian components.
    loss_current : numpy.ndarray
        Ionisation-work current in A m^-2, complex, shape ``(n_pol, n_to)``.

    Notes
    -----
    Construction copies the arrays. Calls never share source or population storage.
    """

    ionisation_rate: Float[np.ndarray, " n_rate_t"]
    fraction: Float[np.ndarray, " n_fraction_t"]
    drude_drive: Complex[np.ndarray, "n_drive_pol n_drive_t"]
    loss_current: Complex[np.ndarray, "n_loss_pol n_loss_t"]

    def __post_init__(self) -> None:
        """Copy and validate the source arrays and their common shape.

        Raises
        ------
        GridError
            If arrays have incompatible shapes, non-finite data, invalid rates or
            fractions, or a component count other than one or two.
        """
        for name in ("ionisation_rate", "fraction", "drude_drive", "loss_current"):
            value = np.array(getattr(self, name), copy=True)
            if not np.all(np.isfinite(value)):
                raise GridError(f"envelope plasma {name} must be finite")
            object.__setattr__(self, name, value)
        rate, fraction = self.ionisation_rate, self.fraction
        if rate.ndim != 1 or rate.size == 0 or fraction.shape != rate.shape:
            raise GridError(
                "envelope plasma rate and fraction need the same nonempty time axis"
            )
        drive, loss = self.drude_drive, self.loss_current
        if (
            drive.ndim != 2
            or drive.shape[0] not in (1, 2)
            or drive.shape[1] != rate.size
            or loss.shape != drive.shape
        ):
            raise GridError("envelope plasma currents need shape (1 or 2, n_to)")
        if np.any(rate < 0) or np.any(fraction < 0) or np.any(fraction > 1):
            raise GridError(
                "envelope plasma needs non-negative rates and fractions in [0, 1]"
            )


def _checked_rate(rate: EvaluableRate, amplitude: float) -> float:
    """Reject a malformed rate before it enters an angular integral.

    Parameters
    ----------
    rate : EvaluableRate
        Instantaneous magnitude model.
    amplitude : float
        Non-negative field magnitude in V/m.

    Returns
    -------
    float
        Finite non-negative rate in s^-1.
    """
    value = np.asarray(rate(amplitude))
    if value.shape != () or not np.isrealobj(value):
        raise GridError(
            "the envelope plasma rate must return a real scalar for a scalar field"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise GridError("the envelope plasma rate must be finite and non-negative")
    return result


def _cycle_points(
    envelope: Complex[np.ndarray, " n_pol"], scale: float, rate: EvaluableRate
) -> Float[np.ndarray, " n_points"]:
    """Seed Cartesian extrema and declared cutoff crossings for adaptive integration.

    Parameters
    ----------
    envelope : numpy.ndarray
        Dimensionless envelope scaled to a largest real component of one.
    scale : float
        Physical amplitude scale in V/m.
    rate : EvaluableRate
        Magnitude model, optionally declaring threshold or table bounds.

    Returns
    -------
    numpy.ndarray
        Sorted full-cycle phase breakpoints in rad.
    """
    real, imaginary = np.real(envelope), np.imag(envelope)
    phase = 0.5 * math.atan2(
        2 * float(real @ imaginary), float(real @ real - imaginary @ imaginary)
    )
    extrema = np.sort(
        np.r_[0.0, (phase + np.arange(4) * (np.pi / 2)) % (2 * np.pi), 2 * np.pi]
    )
    points = list(np.r_[np.linspace(0, 2 * np.pi, 33), extrema])
    # A table holds its value above field_max, and an optmat rate above its
    # maximum_field: both are corners the adaptive rule must be told about.
    bounds = (
        getattr(rate, "field_min", getattr(rate, "threshold", None)),
        getattr(rate, "field_max", None),
        getattr(rate, "maximum_field", None),
    )

    def crossing(theta: float, boundary: float) -> float:
        """Locate a rate boundary directly in the Cartesian real field.

        Parameters
        ----------
        theta : float
            Optical phase in rad.
        boundary : float
            Rate threshold divided by the physical field scale, dimensionless.

        Returns
        -------
        float
            Signed difference between the scaled magnitude and the boundary.
        """
        return float(np.linalg.norm((envelope * np.exp(-1j * theta)).real)) - boundary

    for bound in bounds:
        if bound is None:
            continue
        boundary = float(bound) / scale
        for left, right in itertools.pairwise(extrema):
            if crossing(float(left), boundary) * crossing(float(right), boundary) < 0:
                # A 1e-14 rad location error is below the oracle's angular
                # budget, including the thin nearly linear cutoff intervals.
                points.append(
                    optimize.brentq(
                        crossing,
                        float(left),
                        float(right),
                        args=(boundary,),
                        xtol=1e-14,
                    )
                )
    return np.unique(points)


def _cycle_sources(
    envelope: Complex[np.ndarray, " n_pol"], rate: EvaluableRate
) -> tuple[float, Complex[np.ndarray, " n_pol"]]:
    """Integrate the Cartesian real field and its first current harmonic.

    Parameters
    ----------
    envelope : numpy.ndarray
        One physical V/m envelope sample, shape ``(n_pol,)``.
    rate : EvaluableRate
        Instantaneous isotropic magnitude model.

    Returns
    -------
    tuple
        Averaged rate in s^-1 and current divided by potential, density and
        survival, in m/(V s).
    """
    scale = float(
        max(np.max(np.abs(np.real(envelope))), np.max(np.abs(np.imag(envelope))))
    )
    if scale == 0:
        return 0.0, np.zeros_like(envelope)
    # Complex division may form an overflowing reciprocal for subnormal scale
    # even when each real quotient is bounded by one.
    normalised = np.real(envelope) / scale + 1j * (np.imag(envelope) / scale)
    real, imaginary = np.real(normalised), np.imag(normalised)
    square = float(real @ real)
    if square == float(imaginary @ imaginary) and float(real @ imaginary) == 0:
        # The circular magnitude is constant. Evaluating cos²+sin² at a hard
        # table cutoff would otherwise switch the rate on/off through roundoff.
        circular_rate = _checked_rate(rate, scale * math.sqrt(square))
        return circular_rate, (circular_rate / scale) * normalised / square
    # Scaling keeps the adaptive error criterion independent of SI magnitudes.
    # Seed the entire circle so a sharply localised tunnelling peak is not missed.
    points = _cycle_points(normalised, scale, rate)
    component_scale = np.maximum(np.abs(normalised), np.finfo(float).tiny)
    amplitudes = [
        float(np.linalg.norm((normalised * np.exp(-1j * theta)).real)) * scale
        for theta in points
    ]
    if not all(math.isfinite(amplitude) for amplitude in amplitudes):
        raise GridError("the physical field magnitude is not representable")
    rate_scale = max(
        1e-300, *(_checked_rate(rate, amplitude) for amplitude in amplitudes)
    )

    def integrand(theta: float) -> Complex[np.ndarray, " n_integrand"]:
        """Form scaled Cartesian rate and first harmonic at one cycle phase.

        Parameters
        ----------
        theta : float
            Optical phase in radians.

        Returns
        -------
        numpy.ndarray
            Rate followed by the complex current components, dimensionless.
        """
        carrier = np.exp(1j * theta)
        field = (normalised * carrier.conjugate()).real
        square = float(field @ field)
        weight = _checked_rate(rate, math.sqrt(square) * scale) / rate_scale
        current = np.zeros(envelope.shape, dtype=np.complex128)
        if square > 0:
            current = 2 * weight * field * carrier / square / component_scale
        return np.concatenate((np.array([weight]), current))

    result, _, info = integrate.quad_vec(
        integrand,
        0.0,
        2 * np.pi,
        # The accepted 1e-30/s comparison floor prevents subnormal cutoff noise
        # from demanding a relative answer when the entire rate is negligible.
        epsabs=max(1e-13, 2e-10 * (1e-30 / rate_scale)),
        epsrel=2e-10,
        limit=2000,
        points=points,
        full_output=True,
    )
    if not info.success or not np.all(np.isfinite(result)):
        raise ConvergenceError(
            "the Cartesian envelope plasma cycle integral did not converge"
        )
    return (
        float(result[0].real * (rate_scale / (2 * np.pi))),
        np.asarray(
            result[1:] * component_scale * (rate_scale / scale / (2 * np.pi)),
            dtype=np.complex128,
        ),
    )


_CYCLE_ORDER: Final = 32
"""Gauss--Legendre order on each smooth piece of the quarter cycle.

:func:`linear_cycle_averages` evaluates the rate once, at this rule's nodes and the
embedded :data:`_CYCLE_CHECK_ORDER` rule's, on each piece between the field bounds
the rate declares, where its integrand is analytic: its error then falls
geometrically with the order. Measured against the adaptive average, the
:data:`_CYCLE_CHECK_ORDER` rule alone is within 5e-13 relative for optmat's argon,
helium, xenon and nitrogen rates at 800 nm from their thresholds to three times their
maximum fields, the hold corner included (SEP 3, Q46), so the check passes for every
rate optmat ships and this rule, one order higher, is used.
"""

_CYCLE_CHECK_ORDER: Final = 24
"""Order of the embedded lower rule whose difference bounds the average's error."""

_CYCLE_TOLERANCE: Final = 2e-10
"""Relative tolerance of the cycle average, the adaptive oracle's own ``epsrel``."""

_CYCLE_FLOOR: Final = 1e-30
"""Absolute comparison floor in s^-1, the adaptive oracle's, for negligible rates."""


def _linear_cycle_rules(
    rate: EvaluableRate, amplitudes: Float[np.ndarray, " n_amplitude"]
) -> tuple[Float[np.ndarray, " n_amplitude"], Float[np.ndarray, " n_amplitude"]]:
    """Return the fixed-order cycle averages of a rate and their embedded check.

    For linear polarisation ``|a cos theta|`` falls monotonically over the quarter
    cycle ``[0, pi/2]``, which carries the whole average by symmetry, so each bound
    the rate declares (its threshold or ``field_min``, ``field_max`` and
    ``maximum_field``) is crossed once, at ``arccos(bound / a)``. The quarter is split
    there; below the lower bound the rate is exactly zero and that piece is skipped.

    Parameters
    ----------
    rate : EvaluableRate
        Instantaneous magnitude rate in s^-1 of a field in V/m, vectorised.
    amplitudes : numpy.ndarray
        Envelope amplitudes in V/m, finite and positive, shape ``(n_amplitude,)``.

    Returns
    -------
    tuple of numpy.ndarray
        The averages in s^-1 by the :data:`_CYCLE_ORDER` rule and by the
        :data:`_CYCLE_CHECK_ORDER` rule, from one evaluation of the rate.

    Raises
    ------
    GridError
        If the rate does not return a finite, non-negative real array of the
        field's shape.
    """
    lower = getattr(rate, "field_min", getattr(rate, "threshold", None))
    bounds = sorted(
        {
            float(bound)
            for bound in (
                lower,
                getattr(rate, "field_max", None),
                getattr(rate, "maximum_field", None),
            )
            if bound is not None and math.isfinite(float(bound)) and float(bound) > 0
        }
    )
    ratio = np.array(bounds)[None, :] / amplitudes[:, None]
    crossings = np.where(ratio < 1.0, np.arccos(np.minimum(ratio, 1.0)), 0.0)
    edges = np.sort(
        np.concatenate(
            (
                np.zeros((amplitudes.size, 1)),
                crossings,
                np.full((amplitudes.size, 1), np.pi / 2),
            ),
            axis=1,
        ),
        axis=1,
    )
    left, right = edges[:, :-1], edges[:, 1:]
    # A piece whose fields are all below the lower bound has an exactly zero rate.
    active = right > left
    if lower is not None:
        active &= amplitudes[:, None] * np.cos(left) > float(lower)
    middle, half = 0.5 * (left + right)[active], 0.5 * (right - left)[active]
    rules = [
        np.polynomial.legendre.leggauss(order)
        for order in (_CYCLE_ORDER, _CYCLE_CHECK_ORDER)
    ]
    nodes = np.concatenate([rule[0] for rule in rules])
    theta = middle[:, None] + half[:, None] * nodes[None, :]
    field = np.repeat(amplitudes[:, None], active.shape[1], axis=1)[active]
    values = np.asarray(rate(np.abs(field[:, None] * np.cos(theta))))
    if values.shape != theta.shape or not np.isrealobj(values):
        raise GridError(
            "the envelope plasma rate must return a real array of the field's shape"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise GridError("the envelope plasma rate must be finite and non-negative")
    owner = np.nonzero(active)[0]
    averages = []
    offset = 0
    for points, weights in rules:
        # A row-wise sum, not `@`: BLAS gemv orders its sums by the row count, so one
        # amplitude's average would differ by an ulp with the batch it was taken in.
        pieces = (values[:, offset : offset + points.size] * weights).sum(axis=1) * half
        offset += points.size
        averages.append(
            np.bincount(owner, weights=pieces, minlength=amplitudes.size) * (2 / np.pi)
        )
    return averages[0], averages[1]


def linear_cycle_averages(
    rate: EvaluableRate, amplitudes: Float[np.ndarray, " n_amplitude"]
) -> Float[np.ndarray, " n_amplitude"]:
    """Return a rate averaged around one linearly polarised cycle, at many amplitudes.

    The envelope plasma's own average, ``(1/2 pi) int W(|a cos theta|) d theta`` for
    each envelope amplitude ``a``, to the relative tolerance of its adaptive
    quadrature, 2e-10. It evaluates the rate once for all the amplitudes, on a fixed
    Gauss--Legendre rule of order 32 over each piece of the quarter cycle between the
    field bounds the rate declares, with an embedded rule of order 24: where the two
    agree to the tolerance, the higher one's error is below it; elsewhere (a corner
    the rate does not declare, such as a ceiling on its value) that amplitude takes
    the adaptive quadrature of the envelope's sources instead.

    Parameters
    ----------
    rate : EvaluableRate
        Instantaneous magnitude rate in s^-1 of a field in V/m, vectorised.
    amplitudes : numpy.ndarray
        Envelope amplitudes in V/m, non-negative and finite, shape
        ``(n_amplitude,)``.

    Returns
    -------
    numpy.ndarray
        The cycle-averaged rates in s^-1, shape ``(n_amplitude,)``.

    Raises
    ------
    GridError
        If an amplitude is not finite and non-negative, or the rate returns values
        that are not finite, non-negative and real.
    ConvergenceError
        If an amplitude the fixed rule could not certify exhausts the adaptive
        integral's budget.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.envelope_plasma import linear_cycle_averages
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> linear_cycle_averages(OptmatRate("Ar", 800e-9), np.array([0.0, 1e9])).shape
    (2,)
    """
    values = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise GridError(
            "the cycle amplitudes must be a one-dimensional array of finite, "
            "non-negative values"
        )
    result = np.zeros_like(values)
    positive = values > 0
    if np.any(positive):
        fixed, check = _linear_cycle_rules(rate, values[positive])
        certified = np.abs(fixed - check) <= _CYCLE_TOLERANCE * fixed + _CYCLE_FLOOR
        averages = np.where(certified, fixed, 0.0)
        for index in np.nonzero(~certified)[0]:
            averages[index] = _cycle_sources(
                np.array([complex(values[positive][index])]), rate
            )[0]
        result[positive] = averages
    return result


def linear_cycle_average(rate: EvaluableRate, amplitude: float) -> float:
    """Return a rate averaged around one linearly polarised optical cycle.

    The envelope plasma's own average, ``(1/2 pi) int W(|a cos theta|) d theta`` for
    an envelope amplitude ``a``, to its adaptive quadrature's 2e-10, from one
    evaluation of the rate (:func:`linear_cycle_averages`, which takes many).

    Parameters
    ----------
    rate : EvaluableRate
        Instantaneous magnitude rate in s^-1 of a field in V/m, vectorised.
    amplitude : float
        Envelope amplitude in V/m, non-negative and finite.

    Returns
    -------
    float
        The cycle-averaged rate in s^-1.

    Raises
    ------
    GridError
        If the amplitude is not finite and non-negative.
    ConvergenceError
        If the fixed rule could not certify the amplitude and the adaptive integral
        exhausts its budget.

    Examples
    --------
    >>> from solprop.nonlinear.envelope_plasma import linear_cycle_average
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> linear_cycle_average(OptmatRate("Ar", 800e-9), 0.0)
    0.0
    """
    if not (math.isfinite(amplitude) and amplitude >= 0):
        raise GridError("the cycle amplitude must be finite and non-negative")
    return float(linear_cycle_averages(rate, np.array([float(amplitude)]))[0])


def _validate_field(field_t: Complex[np.ndarray, "n_pol n_to"], dt: float) -> None:
    """Check physical envelope shape, finite data and time spacing.

    Parameters
    ----------
    field_t : numpy.ndarray
        Physical V/m envelope, shape ``(n_pol, n_to)``.
    dt : float
        Fine time spacing in s.
    """
    if (
        field_t.ndim != 2
        or field_t.shape[0] not in (1, 2)
        or field_t.shape[1] == 0
        or not np.iscomplexobj(field_t)
    ):
        raise GridError(
            "envelope plasma needs a complex field with shape (1 or 2, n_to)"
        )
    if not np.all(np.isfinite(field_t)):
        raise GridError("envelope plasma field must be finite")
    if not math.isfinite(dt) or dt <= 0:
        raise GridError("envelope plasma dt must be positive and finite")


def _population(
    rate: Float[np.ndarray, " n_to"], dt: float, preionisation: float
) -> tuple[Float[np.ndarray, " n_to"], Float[np.ndarray, " n_to"]]:
    """Keep survival directly, including after the fraction rounds to one.

    Parameters
    ----------
    rate : numpy.ndarray
        Cycle-averaged rates in s^-1, shape ``(n_to,)``.
    dt : float
        Positive time spacing in s.
    preionisation : float
        Initial fraction in ``[0, 1]``.

    Returns
    -------
    tuple
        Fresh ionised fraction and surviving neutral fraction arrays.
    """
    # Positive overflow represents saturated exposure, not a non-finite source.
    # Multiply the half-rates before adding to avoid premature overflow of W+W.
    with np.errstate(over="ignore"):
        intervals = (0.5 * rate[:-1] + 0.5 * rate[1:]) * dt
        exposure = np.concatenate((np.zeros(1), np.cumsum(intervals)))
    survival = (1 - preionisation) * np.exp(-exposure)
    fraction = preionisation + (1 - preionisation) * (-np.expm1(-exposure))
    return fraction, survival


@dataclass(frozen=True, eq=False)
class EnvelopePlasmaResponse:
    """Average an instantaneous isotropic rate over a physical envelope's cycle.

    Attributes
    ----------
    gas : Gas or str or Species
        Chemical species supplying the first ionisation potential. Beside a rate
        that carries its own energy per event (an
        :class:`~solprop.nonlinear.optmat_rates.OptmatRate`), the species is stored
        with that energy as its potential
        (:func:`~solprop.nonlinear.ionisation.bind_rate_energy`, SEP 3, D8).
    rate : EvaluableRate
        Magnitude rate in s^-1 for V/m, declaring
        ``convention = 'instantaneous-isotropic'``, as optmat's rates do
        (:class:`~solprop.nonlinear.optmat_rates.OptmatRate`); a cycle-averaged rate
        is refused. Molecular orientation averaging is compatible.
        The rate must vanish at zero field and give an integrable loss current.
    preionisation : float or ZProfile
        Initial ionised fraction in ``[0, 1]``, default zero: a number, or a path along
        z in m giving the fraction ionised before the pulse arrives at each position.
        The reference methods see one plane, so a path is resolved by the translation
        layer and the oracle for a plane is :func:`dataclasses.replace` with its value.
    species : str or None
        Optional nonempty component label, distinct from the chemical identity.

    Notes
    -----
    This is the frozen-cycle approximation: the envelope and population vary
    slowly compared with one optical cycle. It neither models carrier-scale
    depletion nor changes the physical parameters of the supplied rate model.
    The loss current retains both directional moments of a vector ellipse.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> rate = OptmatRate("Ar", 800e-9)
    >>> response = EnvelopePlasmaResponse("Ar", rate, preionisation=0.25)
    >>> source = response.sources(np.zeros((2, 3), complex), 1e25, dt=1e-16)
    >>> source.fraction.tolist()
    [0.25, 0.25, 0.25]
    """

    gas: Gas | str | Species
    rate: EvaluableRate
    _: KW_ONLY
    preionisation: float | ZProfile = 0.0
    species: str | None = None

    def __post_init__(self) -> None:
        """Validate the material, convention and initial population.

        Raises
        ------
        MaterialError
            If the material or rate binding is inconsistent, the convention is
            not instantaneous-isotropic, the initial fraction is invalid, or the
            rate does not vanish at zero field.
        """
        if not isinstance(self.gas, Species):
            object.__setattr__(self, "gas", as_gas(self.gas))
        # A rate that carries its own energy per event sets the energy the loss pays
        # (SEP 3, D8); the consistency check below then holds by construction.
        object.__setattr__(self, "gas", bind_rate_energy(self.gas, self.rate))
        object.__setattr__(
            self, "preionisation", check_preionisation(self.preionisation)
        )
        if self.species is not None and not self.species.strip():
            raise MaterialError("the envelope plasma component label must not be empty")
        if rate_convention(self.rate) != "instantaneous-isotropic":
            raise MaterialError(
                "envelope plasma requires an explicit "
                "instantaneous-isotropic rate convention"
            )
        gas_name = (
            self.gas.name if isinstance(self.gas, Species) else as_gas(self.gas).value
        )
        name, potential = rate_species(self.rate)
        if (name is not None and name != gas_name) or (
            potential is not None and potential != self.potential
        ):
            raise MaterialError(
                "the envelope plasma rate and response must use consistent species "
                "and ionisation potential"
            )
        _ = self.potential
        if (
            np.asarray(self.rate(0.0)).shape != ()
            or float(np.asarray(self.rate(0.0))) != 0
        ):
            raise MaterialError(
                "the envelope plasma rate must be exactly zero at zero field"
            )

    @property
    def potential(self) -> float:
        """Return the species' first ionisation potential.

        Returns
        -------
        float
            Binding energy in J.

        Raises
        ------
        MaterialError
            If the material supplies no ionisation data.
        """
        if isinstance(self.gas, Species):
            if self.gas.ionisation is None:
                raise MaterialError(f"{self.gas.name} has no ionisation data")
            return self.gas.ionisation.potential
        return float(ionisation_potential(self.gas))

    @property
    def traits(self) -> ResponseTraits:
        """Declare temporal memory, envelope support and isotropic vector behaviour.

        Returns
        -------
        ResponseTraits
            Density and pre-ionisation profiles, optionally labelled per component.
            The compiled response also carries a distinct spectral assembly.
        """
        suffix = "" if self.species is None else f":{self.species}"
        return ResponseTraits(
            "memory",
            ("envelope",),
            "vector_isotropic",
            (f"density{suffix}", f"preionisation{suffix}"),
        )

    def _plane_preionisation(self) -> float:
        """Return the initial fraction at the one plane the reference methods see.

        Returns
        -------
        float
            The fraction in ``[0, 1]``.

        Raises
        ------
        MaterialError
            If the fraction is a path along z, which only the translation layer
            resolves.
        """
        if isinstance(self.preionisation, ZProfile):
            raise MaterialError(
                "this envelope plasma response carries its preionisation as a path "
                "along z, and the reference evaluation sees one plane; evaluate "
                "dataclasses.replace(response, preionisation=<the value at that plane>)"
            )
        return float(self.preionisation)

    def _moments(
        self, field_t: Complex[np.ndarray, "n_pol n_to"], dt: float
    ) -> tuple[Float[np.ndarray, " n_to"], Complex[np.ndarray, "n_pol n_to"]]:
        """Recompute every cycle from this call's field, without population state.

        Parameters
        ----------
        field_t : numpy.ndarray
            Physical V/m envelope, shape ``(n_pol, n_to)``.
        dt : float
            Positive time spacing in s.

        Returns
        -------
        tuple
            Averaged rates and first current harmonic per potential, density and
            surviving neutral fraction.
        """
        _validate_field(field_t, dt)
        rates = np.empty(field_t.shape[1], dtype=np.float64)
        currents = np.empty(field_t.shape, dtype=np.complex128)
        for index in range(field_t.shape[1]):
            rates[index], currents[:, index] = _cycle_sources(
                field_t[:, index], self.rate
            )
        return rates, currents

    def sources(
        self, field_t: Complex[np.ndarray, "n_pol n_to"], density: float, *, dt: float
    ) -> EnvelopePlasmaSources:
        """Return fresh cycle-averaged population, Drude drive and loss current.

        Parameters
        ----------
        field_t : numpy.ndarray
            Physical complex envelope in V/m, shape ``(n_pol, n_to)``; one or two
            Cartesian components and at least one sample. Modal √W amplitudes must
            be converted to physical fields before this call.
        density : float
            Initial particle density in m^-3, finite and non-negative.
        dt : float
            Time spacing in s, finite and positive.

        Returns
        -------
        EnvelopePlasmaSources
            New arrays, with the first fraction exactly ``preionisation``. The
            population is integrated by the trapezoidal prefix convention.

        Raises
        ------
        GridError
            If inputs or evaluated rates are invalid, or sources are not finite.
        ConvergenceError
            If the independent adaptive cycle integral exhausts its budget.
        MaterialError
            If the preionisation is a path along z rather than one plane's value.
        """
        if not math.isfinite(density) or density < 0:
            raise GridError("envelope plasma density must be finite and non-negative")
        rates, current = self._moments(field_t, dt)
        fraction, survival = _population(rates, dt, self._plane_preionisation())
        with np.errstate(over="ignore", invalid="ignore"):
            drive = (CHARGE_MASS_RATIO * density * fraction)[None, :] * field_t
            loss = (self.potential * density * survival)[None, :] * current
        return EnvelopePlasmaSources(rates, fraction, drive, loss)

    def ionised_fraction(
        self, field_t: Complex[np.ndarray, "n_pol n_to"], *, dt: float
    ) -> Float[np.ndarray, " n_to"]:
        """Return a fresh population history from cycle-averaged exposure.

        Parameters
        ----------
        field_t : numpy.ndarray
            Physical complex V/m envelope, shape ``(n_pol, n_to)``.
        dt : float
            Positive finite time spacing in s.

        Returns
        -------
        numpy.ndarray
            Fraction in ``[preionisation, 1]``, shape ``(n_to,)``. The first sample
            is exactly the initial fraction, including for a one-sample input.

        Raises
        ------
        GridError
            If the field, spacing or evaluated rates are invalid.
        ConvergenceError
            If adaptive angular integration fails.
        MaterialError
            If the preionisation is a path along z rather than one plane's value.
        """
        rates, _ = self._moments(field_t, dt)
        return _population(rates, dt, self._plane_preionisation())[0]

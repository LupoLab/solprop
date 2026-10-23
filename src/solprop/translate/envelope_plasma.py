# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compile validated cycle integrals and the envelope plasma spectral assembly.

Only numeric, read-only data cross into a backend. Density and population histories
are not part of the cycle cache; they are evaluated afresh for every RHS input.
"""

import hashlib
import itertools
import math
import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float
from scipy import special

from solprop.constants import CHARGE_MASS_RATIO
from solprop.exceptions import ConvergenceError, SolpropWarning, TranslationError
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import IonisationRate
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.ionisation import RateTable, build_rate_table

__all__ = [
    "AngularRule",
    "CycleIntegralData",
    "CycleValidation",
    "EnvelopePlasmaAssembly",
    "EnvelopePlasmaChannel",
    "EnvelopeSourceFactors",
    "assembly_record",
    "build_cycle_integrals",
    "cycle_moments",
    "source_factors",
]

_ORDERS = (64, 128, 256, 512)
_MOMENT_RTOL = 2e-6
_NEGLIGIBLE_RATE = 1e-30
_FORMAT_VERSION = 1
_CYCLES: dict[str, CycleIntegralData] = {}


def _readonly(values: np.ndarray) -> np.ndarray:
    """Copy an array into owned, contiguous, read-only storage.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric array of any shape.

    Returns
    -------
    numpy.ndarray
        Owned C-contiguous copy.
    """
    result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class AngularRule:
    """Hold one Gauss-Legendre rule on the dimensionless interval [-1, 1].

    Attributes
    ----------
    nodes, weights : numpy.ndarray
        Finite, read-only vectors of equal length, with positive weights.
    """

    nodes: Float[np.ndarray, " n_nodes"]
    weights: Float[np.ndarray, " n_weights"]

    def __post_init__(self) -> None:
        """Own and validate an angular quadrature rule.

        Raises
        ------
        TranslationError
            If the rule is empty, non-finite, unordered or has invalid weights.
        """
        if (
            self.nodes.ndim != 1
            or self.nodes.size < 2
            or self.weights.shape != self.nodes.shape
            or not np.all(np.isfinite(self.nodes))
            or not np.all(np.isfinite(self.weights))
            or np.any(np.abs(self.nodes) >= 1)
            or np.any(np.diff(self.nodes) <= 0)
            or np.any(self.weights <= 0)
            or abs(float(np.sum(self.weights)) - 2) > 32 * np.finfo(float).eps
        ):
            raise TranslationError("invalid envelope plasma angular rule")
        object.__setattr__(self, "nodes", _readonly(self.nodes))
        object.__setattr__(self, "weights", _readonly(self.weights))


@dataclass(frozen=True, eq=False)
class CycleValidation:
    """Record independent off-knot validation of a cycle representation.

    Attributes
    ----------
    probes : numpy.ndarray
        Major amplitude in V/m and minor/major ratio, shape ``(n_probe, 2)``.
    residuals : numpy.ndarray
        Relative errors in the mean rate and both directional moments, shape
        ``(n_probe, 3)``. The denominator floor is ``negligible_rate``.
    cutoff_rate : float
        Source rate in s^-1 at the table's lower cutoff, recorded separately
        from interpolation error. For an explicitly supplied table, zero denotes
        that table's own exactly-zero discarded domain.
    format_version : int
        Probe and representation version, used by the cache identity.
    """

    probes: Float[np.ndarray, "n_probe n_coordinates"]
    residuals: Float[np.ndarray, "n_residual n_quantities"]
    cutoff_rate: float
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        """Own finite validation evidence and check its shape.

        Raises
        ------
        TranslationError
            If the evidence is empty, malformed or not finite.
        """
        if (
            self.probes.ndim != 2
            or self.probes.shape[1] != 2
            or not len(self.probes)
            or self.residuals.shape != (len(self.probes), 3)
            or not np.all(np.isfinite(self.probes))
            or not np.all(np.isfinite(self.residuals))
            or np.any(self.residuals < 0)
            or not math.isfinite(self.cutoff_rate)
            or self.cutoff_rate < 0
            or self.format_version != _FORMAT_VERSION
        ):
            raise TranslationError("invalid envelope plasma cycle validation evidence")
        object.__setattr__(self, "probes", _readonly(self.probes))
        object.__setattr__(self, "residuals", _readonly(self.residuals))


@dataclass(frozen=True, eq=False)
class CycleIntegralData:
    """Hold an instantaneous rate table and its validated angular rules.

    Attributes
    ----------
    rate : RateTable
        Read-only instantaneous-isotropic magnitude rate.
    rules : tuple of AngularRule
        Rules of orders 64, 128, 256 and 512, in refinement order.
    validation : CycleValidation
        Independent Cartesian oracle evidence on off-knot probes.
    identity : str
        Digest including the rate, conventions, rules, budgets and probe version.
    moment_rtol : float
        Relative budget for each positive directional moment, fixed at 2e-6.
    negligible_rate : float
        Comparison denominator floor in s^-1, fixed at 1e-30.
    """

    rate: RateTable
    rules: tuple[AngularRule, ...]
    validation: CycleValidation
    identity: str
    moment_rtol: float = _MOMENT_RTOL
    negligible_rate: float = _NEGLIGIBLE_RATE

    def __post_init__(self) -> None:
        """Check the accepted representation and independent accuracy evidence.

        Raises
        ------
        TranslationError
            If conventions, rule orders, tolerances or validation are inconsistent.
        """
        if (
            self.rate.convention != "instantaneous-isotropic"
            or tuple(len(rule.nodes) for rule in self.rules) != _ORDERS
            or self.moment_rtol != _MOMENT_RTOL
            or self.negligible_rate != _NEGLIGIBLE_RATE
            or not self.identity
            or np.max(self.validation.residuals) > self.moment_rtol
        ):
            raise TranslationError("unvalidated envelope plasma cycle representation")
        # Reuse a previously compiled owning, read-only table during material
        # updates. Borrowed views and mutable user tables still need owned copies.
        if not all(
            array.flags.owndata
            and array.flags.c_contiguous
            and not array.flags.writeable
            for array in (self.rate.log_rate, self.rate.coefficients)
        ):
            object.__setattr__(
                self,
                "rate",
                replace(
                    self.rate,
                    log_rate=_readonly(self.rate.log_rate),
                    coefficients=_readonly(self.rate.coefficients),
                ),
            )


@dataclass(frozen=True, eq=False)
class EnvelopePlasmaChannel:
    """Bind one independent envelope population to compiled material data.

    Attributes
    ----------
    label : str
        Unique component label.
    species_identity : str
        Chemical species identity for provenance.
    ionisation_potential : float
        First ionisation potential in J, positive.
    charge_mass_ratio : float
        Compiled e²/m_e in C²/kg, positive.
    density, preionisation : PiecewiseChebyshev
        Particle density in m^-3 and initial ionised fraction along z.
    cycle : CycleIntegralData
        Validated cycle-integral representation.
    electrons_per_event : int
        Exactly one; sequential charge states are outside this model.
    """

    label: str
    species_identity: str
    ionisation_potential: float
    density: PiecewiseChebyshev
    preionisation: PiecewiseChebyshev
    cycle: CycleIntegralData
    charge_mass_ratio: float = CHARGE_MASS_RATIO
    electrons_per_event: int = 1

    def __post_init__(self) -> None:
        """Validate compiled material constants.

        Raises
        ------
        TranslationError
            If constants or identities are invalid or more than one electron is freed.
        """
        if (
            not self.label.strip()
            or not self.species_identity.strip()
            or not math.isfinite(self.ionisation_potential)
            or self.ionisation_potential <= 0
            or not math.isfinite(self.charge_mass_ratio)
            or self.charge_mass_ratio <= 0
            or self.electrons_per_event != 1
        ):
            raise TranslationError("invalid envelope plasma channel constants")

    @property
    def rate(self) -> RateTable:
        """Return the instantaneous magnitude table carried by this channel.

        Returns
        -------
        RateTable
            The table used inside the cycle integral, not an averaged rate.
        """
        return self.cycle.rate


@dataclass(frozen=True, eq=False)
class EnvelopeSourceFactors:
    """Own the coarse-axis internal-FFT plasma multipliers.

    Attributes
    ----------
    active : numpy.ndarray
        Supported positive absolute-frequency mask, shape ``(n_omega,)``.
    drude : numpy.ndarray
        -1/omega², in s², zero outside the mask.
    loss : numpy.ndarray
        -i/omega, in s, zero outside the mask.
    carrier_policy : str
        ``'absolute-frequency'`` or the warned ``'frozen-carrier'`` approximation.
    omega0 : float
        Positive carrier angular frequency in rad/s.
    """

    active: Bool[np.ndarray, " n_active"]
    drude: Float[np.ndarray, " n_drude"]
    loss: Complex[np.ndarray, " n_loss"]
    carrier_policy: Literal["absolute-frequency", "frozen-carrier"]
    omega0: float

    def __post_init__(self) -> None:
        """Own finite, consistently masked source factors.

        Raises
        ------
        TranslationError
            If shapes, signs, masks or carrier frequency are invalid.
        """
        if (
            self.active.ndim != 1
            or self.active.dtype != np.bool_
            or self.drude.shape != self.active.shape
            or self.loss.shape != self.active.shape
            or not math.isfinite(self.omega0)
            or self.omega0 <= 0
            or not np.all(np.isfinite(self.drude))
            or not np.all(np.isfinite(self.loss))
            or np.any(self.drude[~self.active] != 0)
            or np.any(self.loss[~self.active] != 0)
            or np.any(self.drude[self.active] >= 0)
            or np.any(np.real(self.loss) != 0)
            or np.any(np.imag(self.loss)[self.active] >= 0)
            or self.carrier_policy not in ("absolute-frequency", "frozen-carrier")
        ):
            raise TranslationError("invalid envelope plasma source factors")
        for name in ("active", "drude", "loss"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class EnvelopePlasmaAssembly:
    """Declare the independent currents and their final spectral assembly.

    Attributes
    ----------
    channels : tuple of EnvelopePlasmaChannel
        Unique labelled populations in material order.
    factors : EnvelopeSourceFactors
        Multipliers applied after projection, temporal FFT and coarse cropping.
        Ordinary polarisation is summed only at this spectral assembly point.
    """

    channels: tuple[EnvelopePlasmaChannel, ...]
    factors: EnvelopeSourceFactors

    def __post_init__(self) -> None:
        """Require at least one uniquely labelled channel.

        Raises
        ------
        TranslationError
            If a channel is absent or its label is duplicated.
        """
        if not self.channels or len(
            {channel.label for channel in self.channels}
        ) != len(self.channels):
            raise TranslationError(
                "envelope plasma assembly needs unique labelled channels"
            )


def assembly_record(
    assembly: EnvelopePlasmaAssembly, omega: Float[np.ndarray, " n_omega"]
) -> dict[str, object]:
    """Describe the physical approximation and numerical source representation.

    Parameters
    ----------
    assembly : EnvelopePlasmaAssembly
        Compiled channels and masked frequency factors.
    omega : numpy.ndarray
        Absolute coarse frequencies in rad/s.

    Returns
    -------
    dict
        Additive configuration/provenance data, independent of output layouts.
        No time-dependent population state is serialised here.
    """
    factors = assembly.factors
    active_omega = omega[factors.active]
    return {
        "model": "frozen-cycle-instantaneous-isotropic-v1",
        "carrier_policy": factors.carrier_policy,
        "omega0": factors.omega0,
        "minimum_active_omega": None
        if not active_omega.size
        else float(np.min(active_omega)),
        "maximum_drude_multiplier": float(np.max(np.abs(factors.drude), initial=0)),
        "maximum_loss_multiplier": float(np.max(np.abs(factors.loss), initial=0)),
        "rate_convention": "cycle-averaged-from-instantaneous-isotropic",
        "channels": tuple(
            {
                "label": channel.label,
                "species": channel.species_identity,
                "ionisation_potential_j": channel.ionisation_potential,
                "rate_table_identity": channel.rate.key,
                "cycle_identity": channel.cycle.identity,
                "angular_orders": tuple(
                    len(rule.nodes) for rule in channel.cycle.rules
                ),
                "moment_rtol": channel.cycle.moment_rtol,
                "negligible_rate_s_inv": channel.cycle.negligible_rate,
                "validation_residual": float(
                    np.max(channel.cycle.validation.residuals)
                ),
                "model_cutoff_rate_s_inv": channel.cycle.validation.cutoff_rate,
            }
            for channel in assembly.channels
        ),
    }


def source_factors(grid: EnvelopeGrid, *, shock: bool) -> EnvelopeSourceFactors:
    """Compile inverse-frequency factors only on the positive supported band.

    Parameters
    ----------
    grid : EnvelopeGrid
        Coarse spectral axis, containing absolute angular frequencies in rad/s.
    shock : bool
        Use the frequency-dependent factors; false freezes them at the carrier.
        Translation also freezes the explicit UPPE numerator under this policy.

    Returns
    -------
    EnvelopeSourceFactors
        Finite read-only internal-FFT factors and their active mask.

    Warns
    -----
    SolpropWarning
        Once per call when ``shock=False`` selects the carrier approximation.
    """
    active = grid.active_mask & (grid.omega > 0)
    omega = (
        grid.omega[active] if shock else np.full(np.count_nonzero(active), grid.omega0)
    )
    drude = np.zeros(grid.omega.shape, dtype=np.float64)
    loss = np.zeros(grid.omega.shape, dtype=np.complex128)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse = 1 / omega
        drude[active] = -(inverse * inverse)
        loss[active] = -1j * inverse
    if not shock:
        warnings.warn(
            "envelope plasma with shock=False freezes the plasma frequency "
            "factors and explicit UPPE numerator at omega0",
            SolpropWarning,
            stacklevel=2,
        )
    return EnvelopeSourceFactors(
        active,
        drude,
        loss,
        "absolute-frequency" if shock else "frozen-carrier",
        grid.omega0,
    )


def _integrate_rule(
    major: float, ratio: float, rate: RateTable, rule: AngularRule
) -> Float[np.ndarray, " n_moments"]:
    """Split at the table cutoff and clipping crossings before quadrature.

    Parameters
    ----------
    major : float
        Major-axis physical field in V/m, positive.
    ratio : float
        Minor/major amplitude ratio in ``[0, 1]``.
    rate : RateTable
        Instantaneous magnitude table.
    rule : AngularRule
        Rule on [-1, 1].

    Returns
    -------
    numpy.ndarray
        Positive cosine and sine directional moments in s^-1.
    """
    edges = [0.0, math.pi / 2]
    # _refine handles the exactly circular case before reaching this rule.
    for field in (rate.field_min, rate.field_max):
        relative = field / major
        if ratio < relative < 1:
            edges.append(
                math.acos(math.sqrt((relative**2 - ratio**2) / (1 - ratio**2)))
            )
    edges.sort()
    total = np.zeros(2)
    for lower, upper in itertools.pairwise(edges):
        theta = lower + (rule.nodes + 1) * ((upper - lower) / 2)
        cosine, sine = np.cos(theta) ** 2, np.sin(theta) ** 2
        denominator = cosine + ratio**2 * sine
        rates = rate(major * np.sqrt(denominator)) / denominator
        total += (
            (upper - lower)
            / np.pi
            * np.array(
                [
                    np.dot(rule.weights, rates * cosine),
                    np.dot(rule.weights, rates * sine),
                ]
            )
        )
    return total


def cycle_moments(
    major: float, ratio: float, cycle: CycleIntegralData
) -> Float[np.ndarray, " n_moments"]:
    """Evaluate both accepted directional moments with explicit refinement failure.

    Parameters
    ----------
    major : float
        Finite non-negative major-axis field amplitude in V/m.
    ratio : float
        Finite minor/major amplitude ratio in ``[0, 1]``.
    cycle : CycleIntegralData
        Validated numeric cycle representation.

    Returns
    -------
    numpy.ndarray
        Cosine and sine moments in s^-1. Their combination Mc + ratio² Ms
        is the cycle-averaged rate.

    Raises
    ------
    TranslationError
        If the amplitude or ratio is invalid.
    ConvergenceError
        If either moment fails to converge by the 512-point rule.
    """
    if (
        not math.isfinite(major)
        or major < 0
        or not math.isfinite(ratio)
        or not 0 <= ratio <= 1
    ):
        raise TranslationError("invalid ellipse amplitude or ratio")
    return _refine(major, ratio, cycle.rate, cycle.rules)


def _refine(
    major: float, ratio: float, rate: RateTable, rules: tuple[AngularRule, ...]
) -> Float[np.ndarray, " n_moments"]:
    """Refine the numeric rule independently of its validation record.

    Parameters
    ----------
    major : float
        Non-negative major field in V/m.
    ratio : float
        Minor/major ratio in [0, 1].
    rate : RateTable
        Instantaneous magnitude rate.
    rules : tuple of AngularRule
        Increasing-order quadrature rules.

    Returns
    -------
    numpy.ndarray
        Converged cosine and sine moments in s^-1.
    """
    if major < rate.field_min:
        return np.zeros(2)
    if ratio == 1:
        return np.full(2, float(rate(major)) / 2)
    previous = _integrate_rule(major, ratio, rate, rules[0])
    for rule in rules[1:]:
        current = _integrate_rule(major, ratio, rate, rule)
        if np.all(np.isfinite(current)) and np.all(
            np.abs(current - previous)
            <= _MOMENT_RTOL * np.maximum(current, _NEGLIGIBLE_RATE)
        ):
            return current
        previous = current
    raise ConvergenceError(
        "envelope plasma directional moments did not converge by order 512"
    )


def build_cycle_integrals(
    response: EnvelopePlasmaResponse, field_range: tuple[float, float] | None
) -> CycleIntegralData:
    """Build and independently validate the accepted cycle representation.

    Parameters
    ----------
    response : EnvelopePlasmaResponse
        Species-bound explicitly instantaneous magnitude rate.
    field_range : tuple of float or None
        Intended amplitude domain in V/m. The table begins at the model's own
        threshold, even when the requested domain begins higher. A supplied
        RateTable states its own domain and requires None.

    Returns
    -------
    CycleIntegralData
        Cached immutable rate and rule data, with off-knot Cartesian evidence.

    Raises
    ------
    TranslationError
        If the model cannot be tabulated or its domain is not declared.
    ConvergenceError
        If angular refinement or independent validation fails.
    """
    original = response.rate
    if isinstance(original, RateTable):
        if field_range is not None:
            raise TranslationError(
                "an envelope RateTable already states its own field range"
            )
        table, cutoff = original, 0.0
    elif isinstance(original, IonisationRate):
        if field_range is None:
            raise TranslationError("envelope plasma needs ionisation_field_range")
        if not 0 < field_range[0] < field_range[1] or not all(
            map(math.isfinite, field_range)
        ):
            raise TranslationError("invalid envelope ionisation_field_range")
        table = build_rate_table(
            original, (original.threshold, field_range[1]), rtol=1e-7
        )
        cutoff = float(original(table.field_min))
    else:
        raise TranslationError(
            "envelope plasma rate must declare threshold, maximum and parameters "
            "for tabulation"
        )
    key = repr(
        (
            table.key,
            hashlib.sha256(table.coefficients.tobytes()).hexdigest(),
            table.log_field_min,
            table.d_log_field,
            table.field_min,
            table.field_max,
            table.rate_max,
            cutoff,
            response.potential,
            table.convention,
            table.species_identity,
            table.ionisation_potential,
            _ORDERS,
            _MOMENT_RTOL,
            _NEGLIGIBLE_RATE,
            _FORMAT_VERSION,
        )
    )
    identity = hashlib.sha256(key.encode()).hexdigest()
    if identity in _CYCLES:
        cached = _CYCLES[identity]
        # A material update borrows its existing immutable table. Equivalent
        # cached cycle evidence must not replace that explicit table's ownership.
        return (
            replace(cached, rate=table)
            if isinstance(original, RateTable) and cached.rate is not table
            else cached
        )
    rules = tuple(AngularRule(*special.roots_legendre(order)) for order in _ORDERS)
    oracle = replace(response, rate=table, preionisation=0.0)
    amplitudes = np.concatenate(
        (
            np.geomspace(table.field_min, table.field_max, 6),
            table.field_min * np.array([0.99, 1.01]),
            table.field_max * np.array([0.99, 1.01, 1.5]),
        )
    )
    probes = np.array(
        [
            (float(amplitude), ratio)
            for amplitude in amplitudes
            for ratio in (0.0, 1e-6, 0.35, 1 - 1e-6, 1.0)
        ]
    )
    residuals = []
    for major, ratio in probes:
        moments = _refine(float(major), float(ratio), table, rules)
        field = np.array([major, 1j * major * ratio])[:, None]
        source = oracle.sources(field, 1.0, dt=1.0)
        current = source.loss_current[:, 0] / response.potential
        # At the linear boundary the null-axis moment does no work. Validate
        # that axis in the interior probes and compare the active component here.
        reference = np.array(
            [
                current[0].real * major / 2,
                current[1].imag * major / (2 * ratio) if ratio else moments[1],
            ]
        )
        errors = np.abs(moments - reference) / np.maximum(
            np.abs(reference), _NEGLIGIBLE_RATE
        )
        average = moments[0] + ratio**2 * moments[1]
        rate_error = abs(average - source.ionisation_rate[0]) / max(
            source.ionisation_rate[0], _NEGLIGIBLE_RATE
        )
        residuals.append([rate_error, *errors])
    validation = CycleValidation(probes, np.array(residuals), cutoff)
    if np.max(validation.residuals) > _MOMENT_RTOL:
        raise ConvergenceError(
            "independent envelope plasma cycle validation failed: "
            f"{np.max(validation.residuals):.3e}"
        )
    result = CycleIntegralData(table, rules, validation, identity)
    _CYCLES[identity] = result
    return result

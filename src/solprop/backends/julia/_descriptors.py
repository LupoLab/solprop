# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own frequency-first numerical descriptors for Julia modal preparation."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from weakref import WeakKeyDictionary

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int

from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.envelope_plasma import CycleIntegralData
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.ionisation import RateTable
from solprop.translate.problem import PropagationProblem
from solprop.translate.responses import KerrCoefficients, RamanCoefficients
from solprop.translate.routes import ModalRoute, tdse_descriptor

__all__ = [
    "CoefficientTransfer",
    "CycleTransfer",
    "KerrTransfer",
    "ModalTransfer",
    "PlasmaChannelTransfer",
    "PlasmaTransfer",
    "RamanTransfer",
    "RateTransfer",
    "TableTransfer",
    "lower_modal",
    "lower_profile",
    "lower_table",
    "owned",
    "transfer_memory",
]


@dataclass(frozen=True, eq=False)
class TableTransfer:
    """Own one represented table without evaluating or refitting it.

    Attributes
    ----------
    edges : numpy.ndarray
        Increasing interval edges, metres.
    lower, upper : float
        Admitted span including the source table's own edge slack, metres.
    coefficients : numpy.ndarray
        Real or complex coefficients, shape ``(n_values, n_deg, n_intervals)``.
        Values retain their translated units and C-order value-axis enumeration.
    """

    edges: Float[np.ndarray, " n_edges"]
    lower: float
    upper: float
    coefficients: Inexact[np.ndarray, "n_values n_deg n_intervals"]


@dataclass(frozen=True, eq=False)
class CoefficientTransfer:
    """Own one constant, represented or expanded spectral coefficient.

    Attributes
    ----------
    values : numpy.ndarray or None
        Constant ``(n_omega,n_columns)`` values or expanded fixed terms with one
        additional final term axis. Units follow the translated quantity.
    table : TableTransfer or None
        Full represented values, present only for the table representation.
    weights : TableTransfer or None
        Scalar expansion weights, present only with fixed terms. Operator weights
        are cumulative; prefactor and power weights are direct. Lowering preserves
        this distinction without evaluating either kind.
    """

    values: Inexact[np.ndarray, "n_omega n_columns *terms"] | None
    table: TableTransfer | None
    weights: TableTransfer | None


@dataclass(frozen=True, eq=False)
class KerrTransfer:
    """Hold translated cubic factors and their independent stage profile.

    Attributes
    ----------
    coefficient : float
        Translated material coefficient, including epsilon_0 exactly once.
    form : str
        The declared ``real`` or ``analytic`` cubic.
    self_factor, conjugate_factor, harmonic_factor : float
        Dimensionless cubic factors, unchanged by lowering.
    take_real : bool
        Whether an analytic field response returns its real part.
    density : float or TableTransfer
        Independent stage weight; a composite Kerr weight may be signed.
    """

    coefficient: float
    form: str
    self_factor: float
    conjugate_factor: float
    harmonic_factor: float
    take_real: bool
    density: float | TableTransfer


@dataclass(frozen=True, eq=False)
class RamanTransfer:
    """Own the independent kernel and profiles of one scalar Raman response.

    Attributes
    ----------
    undamped : numpy.ndarray
        Undamped response rows, shape ``(2*n_to,n_terms)``, in translated units.
    rates : tuple of float or TableTransfer
        One independent decay profile per kernel row, in inverse seconds.
    density : float or TableTransfer
        This component's own nonnegative material stage weight.
    fraction : float
        Dimensionless delayed fraction.
    material_factor : float
        Translated epsilon_0 factor in SI units, applied once after convolution.
    thg : bool
        The declared full-field or analytic-signal drive policy.
    """

    undamped: Float[np.ndarray, "n_to_doubled n_terms"]
    rates: tuple[float | TableTransfer, ...]
    density: float | TableTransfer
    fraction: float
    material_factor: float
    thg: bool


@dataclass(frozen=True, eq=False)
class RateTransfer:
    """Own the numeric portion of an ionisation spline.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Highest-degree-first cubic coefficients, shape ``(4,n_intervals)``.
    field_min, field_max : float
        Clipping bounds, V/m.
    rate_max : float
        Clipped rate, inverse seconds.
    log_field_min, d_log_field : float
        Logarithmic knot origin and spacing.
    """

    coefficients: Float[np.ndarray, "n_deg n_intervals"]
    field_min: float
    field_max: float
    rate_max: float
    log_field_min: float
    d_log_field: float


@dataclass(frozen=True, eq=False)
class CycleTransfer:
    """Own already qualified angular rules without choosing a new quadrature.

    Attributes
    ----------
    nodes, weights : numpy.ndarray
        Concatenated dimensionless angular rules in refinement order.
    ends : tuple of int
        Cumulative end offsets, also Julia's one-based final indices.
    moment_rtol, negligible_rate : float
        Translated relative comparison budget and inverse-second denominator floor.
    """

    nodes: Float[np.ndarray, " n_nodes"]
    weights: Float[np.ndarray, " n_nodes"]
    ends: tuple[int, ...]
    moment_rtol: float
    negligible_rate: float


@dataclass(frozen=True, eq=False)
class PlasmaChannelTransfer:
    """Keep each independent population's numeric material data together.

    Attributes
    ----------
    rate : RateTransfer
        Instantaneous magnitude-rate spline.
    charge_mass_ratio, ionisation_potential : float
        Translated e squared over electron mass in C^2/kg and ionisation energy in J.
    density, preionisation : float or TableTransfer
        Independent number-density and initial-fraction profiles.
    cycle : CycleTransfer or None
        Angular rules for an envelope, absent for a field response.
    """

    rate: RateTransfer
    charge_mass_ratio: float
    ionisation_potential: float
    density: float | TableTransfer
    preionisation: float | TableTransfer
    cycle: CycleTransfer | None


@dataclass(frozen=True, eq=False)
class PlasmaTransfer:
    """Own field-current or envelope spectral assembly data.

    Attributes
    ----------
    channels : tuple of PlasmaChannelTransfer
        Independent electron populations in material order.
    taps : numpy.ndarray or None
        Centre-first running-integral FIR, present only for field plasma.
    floor_exposure : bool
        The translated exposure-floor policy for that integral.
    drude, loss : numpy.ndarray or None
        Envelope spectral factors, already zero outside their active band.
        Their units and any frozen-carrier policy remain those of translation.
    """

    channels: tuple[PlasmaChannelTransfer, ...]
    taps: Float[np.ndarray, " n_taps"] | None
    floor_exposure: bool
    drude: Float[np.ndarray, " n_omega"] | None
    loss: Complex[np.ndarray, " n_omega"] | None


@dataclass(frozen=True, eq=False)
class ModalTransfer:
    """Own scalar modal field/envelope descriptors without a runtime or workspace.

    Attributes
    ----------
    kind : str
        ``field`` or ``envelope``.
    n_time, dt : int, float
        Fine-grid sample count and spacing in seconds.
    synthesis, projection, coarse_projection : numpy.ndarray or None
        Physical maps with shapes ``(n_nodes,n_modes)`` and ``(n_modes,n_nodes)``.
        Synthesis includes the existing padding scale; projection includes quadrature.
        The coarse projection is absent when the shared transform has no embedded pair.
    taper : float or TableTransfer
        Independent positive dimensionless taper scale.
    prefactor, operator, power_weight : CoefficientTransfer
        Source coefficient, cumulative linear representation or constant generator,
        and dimensionless export weights. No quantity is sampled at preparation.
    crop_index : numpy.ndarray
        Zero-based coarse-to-fine frequency map.
    omega, active, omega_window : numpy.ndarray
        Coarse angular frequencies in rad/s, active mask and spectral source window.
    source_window, source_window_derivative : numpy.ndarray
        Fine temporal source window and its inverse-second derivative.
    temporal_rate : numpy.ndarray or None
        Fine in-stage temporal absorption rate in inverse metres.
    carrier : numpy.ndarray or None
        Translated harmonic carrier on the fine time axis.
    kerr, raman, plasma : KerrTransfer or None, tuple, PlasmaTransfer or None
        Independent material responses, retaining each response's own profiles.
    """

    kind: str
    n_time: int
    dt: float
    synthesis: Float[np.ndarray, "n_nodes n_modes"]
    projection: Float[np.ndarray, "n_modes n_nodes"]
    coarse_projection: Float[np.ndarray, "n_modes n_nodes"] | None
    taper: float | TableTransfer
    prefactor: CoefficientTransfer
    operator: CoefficientTransfer
    power_weight: CoefficientTransfer
    crop_index: Int[np.ndarray, " n_omega"]
    omega: Float[np.ndarray, " n_omega"]
    active: Bool[np.ndarray, " n_omega"]
    omega_window: Float[np.ndarray, " n_omega"]
    source_window: Float[np.ndarray, " n_to"]
    source_window_derivative: Float[np.ndarray, " n_to"]
    temporal_rate: Float[np.ndarray, " n_to"] | None
    carrier: Complex[np.ndarray, " n_to"] | None
    kerr: KerrTransfer | None
    raman: tuple[RamanTransfer, ...]
    plasma: PlasmaTransfer | None


_MODAL: WeakKeyDictionary[PropagationProblem, ModalTransfer] = WeakKeyDictionary()


def owned(values: np.ndarray) -> np.ndarray:
    """Copy values into read-only owned Fortran-contiguous storage.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric values in their translated units and requested shape.

    Returns
    -------
    numpy.ndarray
        An independently owned copy with unchanged dtype and values.
    """
    result = np.array(values, copy=True, order="F")
    result.setflags(write=False)
    return result


def _broadcast(values: np.ndarray, leading: int, shape: tuple[int, ...]) -> np.ndarray:
    # Padding belongs after interval/degree/term axes, before the value axes.
    """Expand only value axes, preserving leading interval or term axes.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric coefficients in their original units.
    leading : int
        Number of leading axes retained without broadcasting.
    shape : tuple of int
        Target value shape.

    Returns
    -------
    numpy.ndarray
        Broadcast view used only while forming an owned transfer.
    """
    missing = len(shape) - (values.ndim - leading)
    expanded = values.reshape(
        (*values.shape[:leading], *((1,) * missing), *values.shape[leading:])
    )
    return np.broadcast_to(expanded, (*values.shape[:leading], *shape))


def lower_table(
    table: PiecewiseChebyshev, *, shape: tuple[int, ...] | None = None
) -> TableTransfer:
    """Copy the exact represented coefficients and the table's admitted edge slack.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev
        Validated represented values in their original units.
    shape : tuple of int or None, optional
        Expanded value shape; broadcast singleton state axes before flattening.

    Returns
    -------
    TableTransfer
        Owned column-major coefficients with original value-axis enumeration.
    """
    edges, lower, upper, packed = table.packed()
    if shape is not None:
        expanded = _broadcast(table.coefficients, 2, shape)
        packed = expanded.reshape(*expanded.shape[:2], -1)
    return TableTransfer(owned(edges), lower, upper, owned(packed.transpose(2, 1, 0)))


def lower_profile(table: PiecewiseChebyshev | None) -> float | TableTransfer:
    """Copy a scalar profile or preserve its exact represented constant.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev or None
        Scalar profile in its own units; None denotes a unit taper.

    Returns
    -------
    float or TableTransfer
        The exact constant or the complete represented profile.

    Raises
    ------
    BackendError
        If the translated profile is not scalar.
    """
    if table is None:
        return 1.0
    coefficients = table.coefficients
    if coefficients.ndim != 2:
        raise BackendError("julia requires scalar material and taper profiles")
    if np.all(coefficients[:, 1:] == 0) and np.all(
        coefficients[:, 0] == coefficients[0, 0]
    ):
        return float(coefficients[0, 0])
    return lower_table(table)


def _coefficient(
    constant: np.ndarray | None,
    table: PiecewiseChebyshev | None,
    expansion: ParameterExpansion | None,
    shape: tuple[int, ...],
) -> CoefficientTransfer:
    """Lower one spectral representation without changing its meaning.

    Parameters
    ----------
    constant : numpy.ndarray or None
        Constant values, when the constant representation is declared.
    table : PiecewiseChebyshev or None
        Full represented values.
    expansion : ParameterExpansion or None
        Fixed terms with their direct or cumulative scalar weights.
    shape : tuple of int
        Target state-shaped value axes, retaining original units.

    Returns
    -------
    CoefficientTransfer
        Owned frequency-first numerical data.
    """
    if constant is not None:
        matrix = np.broadcast_to(constant, shape).reshape(-1, shape[-1]).T
        return CoefficientTransfer(owned(matrix), None, None)
    if expansion is not None:
        terms = _broadcast(expansion.terms, 1, shape)
        matrix = terms.reshape(terms.shape[0], -1, shape[-1]).transpose(2, 1, 0)
        return CoefficientTransfer(owned(matrix), None, lower_table(expansion.weights))
    assert table is not None
    return CoefficientTransfer(None, lower_table(table, shape=shape), None)


def _kerr(
    coefficients: KerrCoefficients | None, density: PiecewiseChebyshev
) -> KerrTransfer | None:
    """Lower cubic coefficients and their independent stage weight.

    Parameters
    ----------
    coefficients : KerrCoefficients or None
        Resolved cubic data, absent without Kerr.
    density : PiecewiseChebyshev
        Legacy fallback weight when the cubic carries no independent profile.

    Returns
    -------
    KerrTransfer or None
        Owned numerical cubic data.
    """
    if coefficients is None:
        return None
    return KerrTransfer(
        coefficients.coefficient,
        coefficients.cubic_form,
        coefficients.self_factor,
        coefficients.conjugate_factor,
        coefficients.harmonic_factor,
        coefficients.take_real,
        lower_profile(coefficients.density or density),
    )


def _raman(component: RamanCoefficients, density: PiecewiseChebyshev) -> RamanTransfer:
    """Lower one independently damped Raman component.

    Parameters
    ----------
    component : RamanCoefficients
        Translated undamped rows, decay profiles and delayed fraction.
    density : PiecewiseChebyshev
        Legacy fallback weight when the component has no own profile.

    Returns
    -------
    RamanTransfer
        Owned response rows and independent profiles.
    """
    return RamanTransfer(
        owned(component.undamped.T),
        tuple(lower_profile(rate) for rate in component.decay_rates),
        lower_profile(component.density or density),
        component.fraction,
        EPS0,
        component.thg,
    )


def _rate(table: RateTable) -> RateTransfer:
    """Copy a rate spline's evaluation data and clipping bounds.

    Parameters
    ----------
    table : RateTable
        Validated instantaneous rate representation.

    Returns
    -------
    RateTransfer
        Owned numeric spline data with unchanged units.
    """
    return RateTransfer(
        owned(table.coefficients),
        table.field_min,
        table.field_max,
        table.rate_max,
        table.log_field_min,
        table.d_log_field,
    )


def _cycle(cycle: CycleIntegralData) -> CycleTransfer:
    """Concatenate the existing angular refinement rules.

    Parameters
    ----------
    cycle : CycleIntegralData
        Qualified rate-specific quadrature and comparison budgets.

    Returns
    -------
    CycleTransfer
        Owned rules with unchanged budgets and refinement order.
    """
    ends = np.cumsum([len(rule.nodes) for rule in cycle.rules])
    return CycleTransfer(
        owned(np.concatenate([rule.nodes for rule in cycle.rules])),
        owned(np.concatenate([rule.weights for rule in cycle.rules])),
        tuple(int(end) for end in ends),
        cycle.moment_rtol,
        cycle.negligible_rate,
    )


def _plasma(problem: PropagationProblem) -> PlasmaTransfer | None:
    """Lower independent populations and their declared source assembly.

    Parameters
    ----------
    problem : PropagationProblem
        Admitted scalar field or envelope descriptor.

    Returns
    -------
    PlasmaTransfer or None
        Owned channel data, absent without plasma.
    """
    responses = problem.responses
    envelope = responses.envelope_plasma
    if envelope is not None:
        return PlasmaTransfer(
            tuple(
                PlasmaChannelTransfer(
                    _rate(channel.cycle.rate),
                    channel.charge_mass_ratio,
                    channel.ionisation_potential,
                    lower_profile(channel.density),
                    lower_profile(channel.preionisation),
                    _cycle(channel.cycle),
                )
                for channel in envelope.channels
            ),
            None,
            False,
            owned(envelope.factors.drude),
            owned(envelope.factors.loss),
        )
    if not responses.plasma_channels:
        return None
    rule = responses.plasma_integral
    rule.check(problem.grids.dt_fine)
    return PlasmaTransfer(
        tuple(
            PlasmaChannelTransfer(
                _rate(channel.rate),
                channel.charge_mass_ratio,
                channel.ionisation_potential,
                lower_profile(channel.density),
                lower_profile(channel.preionisation),
                None,
            )
            for channel in responses.plasma_channels
        ),
        owned(rule.taps),
        rule.exposure == "floor-at-zero",
        None,
        None,
    )


def lower_modal(problem: PropagationProblem) -> ModalTransfer:
    """Cache owned scalar field/envelope numerical data without native or Julia entry.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated binary64 descriptor with a scalar static modal transform.

    Returns
    -------
    ModalTransfer
        Immutable owned transfer; its weak cache does not retain the problem.

    Raises
    ------
    BackendError
        Naming a presently unsupported route or response before runtime allocation.
    """
    if problem in _MODAL:
        return _MODAL[problem]
    responses, transform, linear, grid = (
        problem.responses,
        problem.transform,
        problem.linear,
        problem.grids,
    )
    checks = (
        (problem.precision.single, "binary32"),
        (problem.shape[1] != 1, "vector polarisation"),
        (not isinstance(problem.route, ModalRoute), "non-modal routes"),
        (transform.modal_tables is not None, "general modal transforms"),
        (tdse_descriptor(problem.route) is not None, "TDSE responses"),
        (linear.coupled_flow is not None, "coupled flows"),
        (linear.modal_metric is not None, "general modal power metrics"),
        (responses.dispersive_kerr is not None, "dispersive Kerr"),
        (responses.regional_responses is not None, "regional responses"),
        (problem.flags.has_noise, "input noise"),
        (
            problem.flags.has_temporal_absorber
            and responses.absorber_scheme != "nonlinear",
            "split temporal absorbers",
        ),
    )
    for refused, capability in checks:
        if refused:
            raise BackendError(
                f"this julia modal composition does not yet run {capability}"
            )
    result = ModalTransfer(
        problem.grid_kind,
        grid.n_to,
        grid.dt_fine,
        owned(transform.synthesis),
        owned(transform.projection),
        None
        if np.array_equal(transform.projection, transform.projection_coarse)
        else owned(transform.projection_coarse),
        lower_profile(transform.taper_scale),
        _coefficient(
            responses.prefactor_constant,
            responses.prefactor,
            responses.prefactor_expansion,
            problem.shape,
        ),
        _coefficient(
            linear.phi_constant,
            linear.polynomial if linear.phi_expansion is None else None,
            linear.phi_expansion,
            (1, *problem.shape),
        ),
        _coefficient(
            linear.power_weight_constant,
            linear.power_weight,
            linear.power_weight_expansion,
            problem.shape,
        ),
        owned(grid.crop_index),
        owned(grid.omega),
        owned(grid.active_mask),
        owned(grid.omega_window),
        owned(grid.t_window_fine),
        owned(grid.t_window_fine_derivative),
        None
        if responses.temporal_absorber_fine is None
        else owned(responses.temporal_absorber_fine),
        None if grid.carrier_2w0 is None else owned(grid.carrier_2w0),
        _kerr(responses.kerr_coefficients, responses.density),
        tuple(
            _raman(component, responses.density) for component in responses.raman_terms
        ),
        _plasma(problem),
    )
    _MODAL[problem] = result
    return result


def transfer_memory(transfer: object, *, exclude: object = None) -> dict[str, int]:
    """Enumerate owned arrays in nested numerical transfer records once.

    Parameters
    ----------
    transfer : object
        An immutable transfer dataclass, tuple, scalar or array.
    exclude : object, optional
        Arrays or records whose array owners are already counted by another owner.

    Returns
    -------
    dict
        Dotted paths mapped to retained array bytes, excluding repeated references.
    """
    ledger: dict[str, int] = {}
    seen: set[int] = set()

    def visit(value: object, path: str) -> None:
        """Record each owned array once while descending numerical records.

        Parameters
        ----------
        value : object
            Current nested record, tuple, array or scalar.
        path : str
            Dotted retained-buffer name.
        """
        if isinstance(value, np.ndarray):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if id(owner) not in seen:
                ledger[path] = owner.nbytes
                seen.add(id(owner))
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}".lstrip("."))
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        elif isinstance(value, Mapping):
            for name, item in value.items():
                visit(item, f"{path}.{name}")

    if exclude is not None:
        visit(exclude, "")
        ledger.clear()
    visit(transfer, "")
    return ledger

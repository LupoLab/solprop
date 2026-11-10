# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower translated numerical descriptors without importing an execution backend."""

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop.exceptions import BackendError
from solprop.stepper.controller import START_GROWTH, PIController
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.problem import PropagationProblem, validate_run
from solprop.translate.routes import ModalRoute, tdse_descriptor
from solprop.translate.run_request import ErrorControlSpec, MethodKind, RunRequest

__all__ = ["EnvelopeTransfer", "RunTransfer", "lower_envelope", "lower_runge_kutta"]


@dataclass(frozen=True, eq=False)
class EnvelopeTransfer:
    """Own constant scalar-envelope arrays in Julia's frequency-first storage.

    Attributes
    ----------
    synthesis, projection : numpy.ndarray
        Real maps of shape ``(n_nodes, n_modes)`` and ``(n_modes, n_nodes)``.
        Include the translated pad scale and constant taper and its inverse.
    prefactor, operator : numpy.ndarray
        Complex arrays of shape ``(n_omega, n_modes)``. The prefactor includes
        the crop scale; the linear generator has units of inverse metres.
    power_weight : numpy.ndarray
        Positive dimensionless export weights, shape ``(n_omega, n_modes)``.
    crop_index : numpy.ndarray
        Zero-based coarse-to-fine indices, shape ``(n_omega,)``.
    omega_window, source_window : numpy.ndarray
        Dimensionless coarse spectral and fine temporal windows.
    kerr : tuple or None
        Coefficient ``epsilon_0 * chi3`` per unit density weight, self factor,
        conjugate factor and harmonic factor; ``None`` without Kerr.
    density : float
        Kerr's own constant density weight, never another response's profile.
    carrier : numpy.ndarray or None
        The translated harmonic carrier on the fine time axis.
    """

    synthesis: Float[np.ndarray, "n_nodes n_modes"]
    projection: Float[np.ndarray, "n_modes n_nodes"]
    prefactor: Complex[np.ndarray, "n_omega n_modes"]
    operator: Complex[np.ndarray, "n_omega n_modes"]
    power_weight: Float[np.ndarray, "n_omega n_modes"]
    crop_index: Int[np.ndarray, " n_omega"]
    omega_window: Float[np.ndarray, " n_omega"]
    source_window: Float[np.ndarray, " n_to"]
    kerr: tuple[float, float, float, float] | None
    density: float
    carrier: Complex[np.ndarray, " n_to"] | None


@dataclass(frozen=True, eq=False)
class RunTransfer:
    """Hold one resolved Runge-Kutta request for one-time Julia conversion.

    Attributes
    ----------
    control : solprop.translate.run_request.ErrorControlSpec
        Immutable scalar control policy, already resolved by shared translation.
    nodes, matrix, solution, error : numpy.ndarray
        Tableau abscissae, stage matrix, solution and embedded-difference weights.
    active, observed : numpy.ndarray
        Boolean active and error-norm masks, shape ``(n_omega,)``.
    weights : numpy.ndarray
        Positive per-bin relative tolerance weights, shape ``(n_omega,)``.
    planes, breakpoints, hard_kinks : numpy.ndarray
        Observation and landing positions, metres. Breakpoints already reflect
        whether the request caps steps to containing intervals.
    steps, edges : numpy.ndarray
        Fixed replay lengths and their authoritative accumulated edges, metres;
        empty for an adaptive request.
    step_planes : int
        Accepted-step observation cadence; zero disables these observations.
    start_growth : float
        Translated dimensionless maximum startup growth factor.
    """

    control: ErrorControlSpec
    nodes: Float[np.ndarray, " n_stages"]
    matrix: Float[np.ndarray, "n_stages n_stages"]
    solution: Float[np.ndarray, " n_stages"]
    error: Float[np.ndarray, " n_stages"]
    active: Bool[np.ndarray, " n_omega"]
    observed: Bool[np.ndarray, " n_omega"]
    weights: Float[np.ndarray, " n_omega"]
    planes: Float[np.ndarray, " n_z"]
    breakpoints: Float[np.ndarray, " n_intervals"]
    hard_kinks: Float[np.ndarray, " n_kinks"]
    steps: Float[np.ndarray, " n_steps"]
    edges: Float[np.ndarray, " n_edges"]
    step_planes: int
    start_growth: float


_ENVELOPES: WeakKeyDictionary[PropagationProblem, EnvelopeTransfer] = (
    WeakKeyDictionary()
)


def _owned(values: np.ndarray) -> np.ndarray:
    """Copy an array into read-only frequency-first storage.

    Parameters
    ----------
    values : numpy.ndarray
        Translated numerical values in their original units.

    Returns
    -------
    numpy.ndarray
        An owned Fortran-contiguous copy with the same shape and dtype.
    """
    result = np.array(values, copy=True, order="F")
    result.setflags(write=False)
    return result


def _columns(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Broadcast a state coefficient and flatten its modal component columns.

    Parameters
    ----------
    values : numpy.ndarray
        Values broadcastable to the translated state, in their original units.
    shape : tuple of int
        State shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        Owned read-only array of shape ``(n_omega, n_modes * n_pol)``.
    """
    return _owned(np.broadcast_to(values, shape).reshape(-1, shape[-1]).T)


def _constant(table: PiecewiseChebyshev | None, name: str, default: float) -> float:
    """Extract one represented constant without sampling away any variation.

    Parameters
    ----------
    table : PiecewiseChebyshev or None
        Scalar profile, or None to use the default.
    name : str
        Capability named in a varying-profile refusal.
    default : float
        Absent-profile value in that profile's units.

    Returns
    -------
    float
        Constant profile value in its original units.

    Raises
    ------
    BackendError
        If the profile is nonscalar or has any represented variation.
    """
    if table is None:
        return default
    coefficients = table.coefficients
    if (
        coefficients.ndim != 2
        or np.any(coefficients[:, 1:] != 0)
        or np.any(coefficients[:, 0] != coefficients[0, 0])
    ):
        raise BackendError(
            f"julia's constant-envelope composition refuses varying {name}"
        )
    return float(coefficients[0, 0])


def lower_envelope(problem: PropagationProblem) -> EnvelopeTransfer:
    """Cache an owned constant-envelope descriptor without retaining its weak key.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated scientific descriptor in the first binary64 modal composition.

    Returns
    -------
    EnvelopeTransfer
        Read-only numerical arrays and scalars, independent of the source descriptor.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming a capability not yet implemented by this composition. No runtime,
        native package or numerical workspace is allocated before admission.
    """
    if problem in _ENVELOPES:
        return _ENVELOPES[problem]
    checks = (
        (problem.precision.single, "binary32"),
        (problem.grid_kind != "envelope", "field grids"),
        (problem.shape[1] != 1, "vector polarisation"),
        (not isinstance(problem.route, ModalRoute), "non-modal routes"),
        (tdse_descriptor(problem.route) is not None, "TDSE responses"),
        (problem.linear.coupled_flow is not None, "coupled flows"),
        (problem.linear.phi_constant is None, "varying linear operators"),
        (problem.responses.prefactor_constant is None, "varying prefactors"),
        (problem.responses.raman_terms != (), "Raman responses"),
        (problem.responses.plasma_channels != (), "plasma responses"),
        (problem.responses.dispersive_kerr is not None, "dispersive Kerr"),
        (problem.responses.regional_responses is not None, "regional responses"),
        (problem.responses.spectral_responses != (), "spectral responses"),
        (problem.flags.has_temporal_absorber, "temporal absorbers"),
        (problem.flags.has_noise, "input noise"),
        (problem.linear.power_weight_constant is None, "varying power weights"),
    )
    for refused, capability in checks:
        if refused:
            raise BackendError(
                f"julia's constant-envelope composition does not yet run {capability}"
            )
    transform, responses = problem.transform, problem.responses
    taper = _constant(transform.taper_scale, "taper", 1.0)
    coefficients = responses.kerr_coefficients
    if coefficients is not None and (
        coefficients.cubic_form != "analytic"
        or coefficients.take_real
        or coefficients.analytic_signal
    ):
        raise BackendError("julia's envelope composition requires the analytic cubic")
    density_table = (
        coefficients.density
        if coefficients is not None and coefficients.density is not None
        else responses.density
    )
    density = _constant(density_table, "Kerr density", 1.0)
    kerr = (
        None
        if coefficients is None
        else (
            coefficients.coefficient,
            coefficients.self_factor,
            coefficients.conjugate_factor,
            coefficients.harmonic_factor,
        )
    )
    operator = problem.linear.phi_constant
    prefactor = responses.prefactor_constant
    weight = problem.linear.power_weight_constant
    assert operator is not None and prefactor is not None and weight is not None
    carrier = problem.grids.carrier_2w0
    result = EnvelopeTransfer(
        _owned(transform.synthesis * taper),
        _owned(transform.projection / taper),
        _columns(prefactor, problem.shape),
        _columns(operator[0], problem.shape),
        _columns(weight, problem.shape),
        _owned(problem.grids.crop_index),
        _owned(problem.grids.omega_window),
        _owned(problem.grids.t_window_fine),
        kerr,
        density,
        None if carrier is None else _owned(carrier),
    )
    _ENVELOPES[problem] = result
    return result


def lower_runge_kutta(problem: PropagationProblem, request: RunRequest) -> RunTransfer:
    """Lower a resolved landed-plane Runge-Kutta request without numerical work.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The scientific descriptor; its numerical arrays are separately cached.
    request : solprop.translate.run_request.RunRequest
        Resolved immutable method, control and observation policy.

    Returns
    -------
    RunTransfer
        Owned method arrays and numerical policy, ready for one-time conversion.

    Raises
    ------
    solprop.exceptions.BackendError
        For a method or service not yet implemented in this first composition.
    """
    validate_run(problem, request)
    if request.integration.kind is not MethodKind.RUNGE_KUTTA:
        raise BackendError("this julia march composition requires Runge-Kutta")
    if type(request.error_control.controller) is not PIController:
        raise BackendError("julia refuses custom controllers; use numpy_ref")
    if request.observation.dense:
        raise BackendError("julia refuses interpolated planes; use native or numpy_ref")
    if request.observation.stats.names or request.observation.stats.periodic:
        raise BackendError(
            "this julia march composition does not yet evaluate statistics"
        )
    if request.boundary.monitor is not None:
        raise BackendError(
            "this julia march composition does not yet evaluate monitors"
        )
    tableau, control = request.integration.tableau, request.error_control
    replay = request.replay
    weights = control.controller.weights
    return RunTransfer(
        control,
        _owned(tableau.nodes),
        _owned(tableau.matrix),
        _owned(tableau.solution_weights),
        _owned(tableau.error_weights),
        _owned(problem.grids.active_mask),
        _owned(problem.grids.norm_mask),
        _owned(np.ones(problem.shape[-1]) if weights is None else weights),
        _owned(request.observation.save_planes),
        _owned(
            problem.linear.phi_breakpoints
            if control.interval_cap
            else problem.linear.phi_breakpoints[[0, -1]]
        ),
        _owned(problem.linear.hard_kinks),
        _owned(np.empty(0) if replay is None else replay.schedule.steps),
        _owned(np.empty(0) if replay is None else replay.schedule.edges),
        request.observation.step_planes,
        START_GROWTH,
    )

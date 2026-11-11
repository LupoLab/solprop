# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower resolved run policy without importing an execution backend."""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float

from solprop.backends.julia._descriptors import owned
from solprop.backends.julia._precision import ArithmeticTransfer, lower_arithmetic
from solprop.exceptions import BackendError
from solprop.stepper.controller import START_GROWTH, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import MarchPolicy, Multistep, march_policy
from solprop.stepper.tableau import continuous_extension
from solprop.translate.problem import PropagationProblem, validate_run
from solprop.translate.run_request import ErrorControlSpec, MethodKind, RunRequest

__all__ = ["RunTransfer", "lower_run"]


@dataclass(frozen=True, eq=False)
class RunTransfer:
    """Hold one resolved numerical session request for one-time Julia conversion.

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
    adams : solprop.stepper.multistep.MarchPolicy or None
        The shared resolved exponential Adams policy, absent for Runge-Kutta.
    source_absent : bool
        Whether admitted descriptors prove the complete source structurally zero.
        No sampled zero or constant profile establishes this property.
    low_storage : bool
        Select the fixed RK4IP register scheme instead of an embedded tableau.
    method : str
        Method representation retained as portable history provenance.
    history_key : str or None
        Shared Adams history identity, independent of execution choices and controls.
    dense_weights : numpy.ndarray or None
        The shared tableau's continuous-extension weights for interpolated RK
        observations, absent for landed requests and other method families.
    tabulate_flow : bool
        Retain exact-interval low-storage flow tables. Like the native owner, other
        march families recompute. Automatic low-storage selection tabulates only a
        constant operator; an explicit low-storage request can tabulate varying data.
    arithmetic : ArithmeticTransfer
        Scientific precision and original-launch scales, shared with source lowering.
        Portable history rows retain this state precision without physical scaling.
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
    adams: MarchPolicy | None = None
    source_absent: bool = False
    low_storage: bool = False
    method: str = "Runge-Kutta"
    history_key: str | None = None
    dense_weights: Float[np.ndarray, " n_stages"] | None = None
    tabulate_flow: bool = False
    arithmetic: ArithmeticTransfer = field(default_factory=ArithmeticTransfer)


def lower_run(problem: PropagationProblem, request: RunRequest) -> RunTransfer:
    """Lower a resolved method and numerical observation policy once.

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
    low_storage = request.integration.kind is MethodKind.LOW_STORAGE
    if low_storage and request.replay is None:
        raise BackendError("julia low-storage RK4IP requires a fixed replay schedule")
    if low_storage and type(request.integration.method) is not LowStorageRungeKutta:
        raise BackendError("julia refuses custom low-storage methods; use numpy_ref")
    validate_run(problem, request)
    if type(request.error_control.controller) is not PIController:
        raise BackendError("julia refuses custom controllers; use numpy_ref")
    tableau, control = request.integration.tableau, request.error_control
    dense_weights = None
    if request.observation.dense and request.integration.kind is MethodKind.RUNGE_KUTTA:
        dense_weights = continuous_extension(tableau)
        if dense_weights is None:
            raise BackendError(
                f"julia has no continuous extension for tableau {tableau.name!r}"
            )
    replay = request.replay
    weights = control.controller.weights
    responses = problem.responses
    in_stage = (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    )
    geometry = problem.transform.geometry
    spatial_absorber = geometry is not None and bool(np.any(geometry.absorber_rate))
    return RunTransfer(
        control,
        owned(tableau.nodes),
        owned(tableau.matrix),
        owned(tableau.solution_weights),
        owned(tableau.error_weights),
        owned(problem.grids.active_mask),
        owned(problem.grids.norm_mask),
        owned(np.ones(problem.shape[-1]) if weights is None else weights),
        owned(request.observation.save_planes),
        owned(
            problem.linear.phi_breakpoints
            if control.interval_cap
            else problem.linear.phi_breakpoints[[0, -1]]
        ),
        owned(problem.linear.hard_kinks),
        owned(np.empty(0) if replay is None else replay.schedule.steps),
        owned(np.empty(0) if replay is None else replay.schedule.edges),
        request.observation.step_planes,
        START_GROWTH,
        _adams_policy(problem, request),
        # A physical-space collar remains a source even without material responses.
        len(responses.response_order) == 1 and not in_stage and not spatial_absorber,
        low_storage,
        repr(request.integration.method),
        request.integration.method.history_key
        if isinstance(request.integration.method, Multistep)
        else None,
        None if dense_weights is None else owned(dense_weights),
        low_storage
        and (
            request.integration.propagator_mode == "tabulate"
            or (
                request.integration.propagator_mode == "auto"
                and problem.flags.constant_linop
            )
        ),
        lower_arithmetic(problem),
    )


def _adams_policy(
    problem: PropagationProblem, request: RunRequest
) -> MarchPolicy | None:
    """Resolve the admitted Adams policy through the shared method definition.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Scientific descriptor containing represented operator boundaries and collars.
    request : solprop.translate.run_request.RunRequest
        Resolved run policy with an optional exponential Adams method.

    Returns
    -------
    solprop.stepper.multistep.MarchPolicy or None
        Immutable shared scalar policy, absent for a different method family.

    Raises
    ------
    BackendError
        For a comparison method, interaction formulation or carrier frame.
    """
    if request.integration.kind is not MethodKind.ADAMS:
        return None
    method = request.integration.method
    if type(method) is not Multistep:
        raise BackendError("julia refuses comparison Adams methods; use numpy_ref")
    if method.formulation != "exponential":
        raise BackendError("julia refuses interaction Adams; use native or numpy_ref")
    if method.carrier_frame != "off":
        raise BackendError(
            "julia refuses the Adams carrier frame; use native or numpy_ref"
        )
    control, observation = request.error_control, request.observation
    if method.interval_cap != control.interval_cap:
        raise BackendError("the Adams method and resolved interval-cap policy disagree")
    responses, linear = problem.responses, problem.linear
    collar = responses.temporal_absorber
    collar_rate = (
        0.5 * float(np.max(collar))
        if collar is not None and responses.absorber_scheme == "nonlinear"
        else 0.0
    )
    monitor = request.boundary.monitor
    return march_policy(
        method,
        dz_min=control.dz_min,
        dz_max=control.dz_max,
        stretch=control.stretch,
        z_end=float(observation.save_planes[-1]),
        save_planes=observation.save_planes.tolist(),
        hard_kinks=linear.hard_kinks.tolist(),
        breakpoints=linear.phi_breakpoints.tolist(),
        interpolate=observation.dense,
        collar_rate=collar_rate,
        carrier=False,
        quiet_interval=None if monitor is None else monitor.interval,
    )

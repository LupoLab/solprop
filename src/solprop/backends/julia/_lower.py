# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower resolved run policy without importing an execution backend."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float

from solprop.backends.julia._descriptors import owned
from solprop.exceptions import BackendError
from solprop.stepper.controller import START_GROWTH, PIController
from solprop.translate.problem import PropagationProblem, validate_run
from solprop.translate.run_request import ErrorControlSpec, MethodKind, RunRequest

__all__ = ["RunTransfer", "lower_runge_kutta"]


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
    )

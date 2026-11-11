# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Runge-Kutta march's controller as functions of its policy and state (SEP 45).

The decisions of ``HostDriver._march_to`` -- the
equivalent norm and acceptance of an attempt, the start-up rule, the next proposal
after an accepted step and the shrink after a rejected one -- each as a function of the
run's :class:`RungeKuttaPolicy` and the values it decides on, with no backend or field
in reach. The march calls them, and SEP 45 section 4 specifies them; the native march
executes the same algorithm and is held to these functions by the decision traces of
``tests/data/rk_traces.json.gz``.

Every function performs the operations of the code it was extracted from, in the same
order. The controller's state is its own proposal: a cap or a landing shortens the
executed step, never the proposal these functions take and return (SEP 45 section
4.3). A function that calls :func:`~solprop.stepper.controller.limit_step` raises its
:class:`~solprop.exceptions.ConvergenceError` before returning anything, so a caller
that assigns only what is returned keeps the state the floor found (the three floor
sites of section 4.3).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from solprop.stepper.controller import START_GROWTH, PIController, limit_step
from solprop.translate.problem import StepperSettings

__all__ = [
    "RungeKuttaPolicy",
    "decide",
    "propose",
    "reject",
    "rk_policy",
    "start_up",
]


@dataclass(frozen=True)
class RungeKuttaPolicy:
    """Every number the Runge-Kutta march's controller reads during a run (SEP 45).

    Attributes
    ----------
    controller : solprop.stepper.controller.PIController
        The controller, whose gains, factors, lag, estimator order and acceptance rule
        the decisions read.
    dz_min, dz_max : float
        The floor and the cap on a proposal, m.
    auto_init : bool
        Whether the first accepted step takes the start-up rule.
    """

    controller: PIController
    dz_min: float
    dz_max: float
    auto_init: bool


def rk_policy(stepper: StepperSettings) -> RungeKuttaPolicy:
    """Return the run's controller policy from its stepper settings.

    Parameters
    ----------
    stepper : solprop.translate.problem.StepperSettings
        The translated stepper settings.

    Returns
    -------
    RungeKuttaPolicy
        Its controller, floor, cap and start-up flag.
    """
    return RungeKuttaPolicy(
        controller=stepper.controller,
        dz_min=stepper.dz_min,
        dz_max=stepper.dz_max,
        auto_init=stepper.auto_init,
    )


def decide(
    policy: RungeKuttaPolicy, norm: float, dz: float, proposal: float
) -> tuple[float, bool]:
    """Return an attempt's equivalent norm and whether it is accepted.

    What the controller learns from, and what a rejection is scaled by, is the norm the
    proposal would have scored; what decides acceptance is the error the executed step
    committed.

    Parameters
    ----------
    policy : RungeKuttaPolicy
        The run's policy.
    norm : float
        The executed step's scaled error norm.
    dz : float
        The executed step, m.
    proposal : float
        The controller's proposal the step was landed from, m.

    Returns
    -------
    tuple of float and bool
        The equivalent norm, and whether the attempt is accepted.
    """
    controller = policy.controller
    equivalent = controller.equivalent_norm(norm, dz_taken=dz, dz_proposed=proposal)
    return equivalent, controller.accepts(norm)


def start_up(
    policy: RungeKuttaPolicy, proposal: float, norm: float
) -> tuple[float, float]:
    """Return the proposal after the first accepted step, and the norm it records.

    The first proposal is a guess; its scaled norm says how far it is from the
    tolerance. The next proposal is the deadbeat step that norm implies, at most
    :data:`~solprop.stepper.controller.START_GROWTH` times the first, and the history
    is told the norm the first step would have scored at that length, or zero when the
    cap binds, a norm so small it is round-off (SEP 14 section 4.2).

    Parameters
    ----------
    policy : RungeKuttaPolicy
        The run's policy.
    proposal : float
        The first accepted step's proposal, m.
    norm : float
        Its equivalent norm, the history's last value.

    Returns
    -------
    tuple of float
        The next proposal, m, and the value that replaces the history's last.

    Raises
    ------
    ConvergenceError
        If the next proposal falls below the floor; the history is then left as it
        was.
    """
    controller = policy.controller
    factor = controller.start_factor(norm)
    following = limit_step(proposal, factor, policy.dz_min, policy.dz_max)
    recorded = (
        0.0
        if factor >= START_GROWTH
        else controller.equivalent_norm(norm, dz_taken=proposal, dz_proposed=following)
    )
    return following, recorded


def propose(policy: RungeKuttaPolicy, proposal: float, norms: Sequence[float]) -> float:
    """Return the proposal after an accepted step, from the history it completed.

    Parameters
    ----------
    policy : RungeKuttaPolicy
        The run's policy.
    proposal : float
        The controller's proposal, m.
    norms : collections.abc.Sequence of float
        The accepted steps' equivalent norms, oldest first, ending with this step's.

    Returns
    -------
    float
        The next proposal, m.

    Raises
    ------
    ConvergenceError
        If it falls below the floor.
    """
    return limit_step(
        proposal, policy.controller.propose(norms), policy.dz_min, policy.dz_max
    )


def reject(policy: RungeKuttaPolicy, proposal: float, equivalent: float) -> float:
    """Return the proposal after a rejected step: the proposal itself shrunk.

    A rejected step is retried at the step the rejection factor leaves; the factor is
    applied once, here, and not again on acceptance.

    Parameters
    ----------
    policy : RungeKuttaPolicy
        The run's policy.
    proposal : float
        The controller's proposal, m.
    equivalent : float
        The rejected attempt's equivalent norm.

    Returns
    -------
    float
        The next proposal, m.

    Raises
    ------
    ConvergenceError
        If it falls below the floor.
    """
    return limit_step(
        proposal,
        policy.controller.rejection_factor(equivalent),
        policy.dz_min,
        policy.dz_max,
    )

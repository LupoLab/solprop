# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march's controller as functions of its policy and its state (SEP 39).

The decisions of :class:`~solprop.backends.multistep.AdamsMarch` -- the stencil and
scored orders, the step-size factor, a rejection and its restart, the order choice, the
proposal's caps and the approach to a landing -- each as a function of the run's
:class:`~solprop.stepper.multistep.MarchPolicy` and the values it decides on, with no
backend, engine or field in reach. The march calls them, and SEP 39 section 4 specifies
them; the native march executes the same algorithm and is held to these functions by
the decision traces of ``tests/data/adams_traces.json.gz``.

Every function performs the operations of the code it was extracted from, in the same
order, including how a comparison treats a NaN: ``min(a, b)`` is ``b`` only if
``b < a``, ``max`` over a list keeps the first element unless a later one compares
strictly greater.
"""

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Int

from solprop.stepper.multistep import MarchPolicy
from solprop.stepper.schedule import Landing, LandingTarget, StepTargets

__all__ = [
    "Rejection",
    "Stencil",
    "approach",
    "carrier_capped",
    "choose_order",
    "collar_reentry",
    "factor",
    "next_landing",
    "order_weight",
    "quiet_capped",
    "rejection",
    "replay_kink",
    "stable_factor",
    "stencil",
    "targets",
]


@dataclass(frozen=True)
class Stencil:
    """A trial's history and the orders it scores (SEP 39 section 4.3, step 5).

    Attributes
    ----------
    nodes : list of float
        The history's normalised positions ``(z_j - z) / h``, newest first.
    rows : numpy.ndarray
        Their engine rows, in the same order.
    order : int
        The step's own order, ``min(q, n)``.
    orders : list of int
        The orders scored: the step's own first, then under variable order its lower
        and its higher neighbour where the history can form them.
    """

    nodes: list[float]
    rows: Int[np.ndarray, " n_held"]
    order: int
    orders: list[int]


@dataclass(frozen=True)
class Rejection:
    """What a rejected trial leaves (SEP 39 section 4.4).

    Attributes
    ----------
    factor : float
        The factor on the rejected step that gives the next proposal.
    rejections : int
        The consecutive rejections after this one, reset to zero by a restart.
    restart : bool
        Whether the history is emptied: the restart after
        :attr:`~solprop.stepper.multistep.MarchPolicy.restart_after` in a row.
    """

    factor: float
    rejections: int
    restart: bool


def targets(policy: MarchPolicy, target: float) -> StepTargets:
    """Return the caps and landings of one march to `target`.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    target : float
        The end of this march, m.

    Returns
    -------
    solprop.stepper.schedule.StepTargets
        The planes (none when they are interpolated), the kinks and the interval
        edges; the run's end when planes are interpolated, otherwise `target`.
    """
    return StepTargets(
        # Dense observation steps across the save planes (SEP 13 section 6.4); hard
        # kinks and the end of the run are still landed on.
        save_planes=np.asarray(policy.save_planes, dtype=np.float64),
        hard_kinks=np.asarray(policy.hard_kinks, dtype=np.float64),
        breakpoints=np.asarray(policy.breakpoints, dtype=np.float64),
        dz_max=policy.dz_max,
        z_end=policy.z_end if policy.interpolate else target,
        stretch=policy.stretch,
    )


def factor(policy: MarchPolicy, norm: float, order: int, *, growth: float) -> float:
    """Return the elementary step-size factor of an order-`order` local defect.

    The defect scales as ``h^{order+1}`` (SEP 13 section 4.4), so the exponent is
    ``1/(order + 1)``, not the Runge-Kutta pair's fixed one.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy, for the safety and smallest factors.
    norm : float
        The scaled defect.
    order : int
        The order the defect belongs to.
    growth : float
        The largest factor allowed.

    Returns
    -------
    float
        The factor on the step.
    """
    if norm <= 0.0:
        return growth
    raw = policy.safety * norm ** (-1.0 / (order + 1))
    return min(growth, max(policy.min_factor, raw))


def stencil(  # noqa: PLR0913 - the march's state, by name
    policy: MarchPolicy,
    z: float,
    dz: float,
    positions: Sequence[float],
    rows: Sequence[int],
    stored_order: int,
    *,
    last_rejected: bool,
) -> Stencil:
    """Return a trial's nodes, rows, order and scored orders.

    Each order's defect is its ``k + 1``-node corrector through the new source less its
    ``k``-node predictor; a neighbour is scored when the history can form it, and the
    order never rises straight after a rejection.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    z : float
        The anchor, m.
    dz : float
        The step, m.
    positions, rows : sequence
        The history's positions, m, and engine rows, oldest first.
    stored_order : int
        The order the last acceptance stored, ``q``.
    last_rejected : bool
        Whether the last attempt was rejected.

    Returns
    -------
    Stencil
        The stencil.
    """
    recent = list(positions)[::-1]
    nodes = [(position - z) / dz for position in recent]
    slots = np.array(list(rows)[::-1], dtype=np.int64)
    held = len(recent)
    order = min(stored_order, held)
    rising = not last_rejected and held > order
    orders = [order] + (
        [
            other
            for other in (order - 1, order + 1)
            if 1 <= other <= policy.order
            and other <= held
            and (other < order or rising)
        ]
        if policy.variable_order
        else []
    )
    return Stencil(nodes, slots, order, orders)


def stable_factor(policy: MarchPolicy, order: int, dz: float, *, quiet: bool) -> float:
    """Return the largest factor on `dz` an in-stage collar leaves an order stable.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    order : int
        The predictor order.
    dz : float
        The step, m.
    quiet : bool
        Whether the stages leave a quiet collar out.

    Returns
    -------
    float
        ``c P_order / (dz rho)``, infinite without an in-stage collar or while it rests.
    """
    if policy.collar_rate <= 0.0 or quiet:
        # No in-stage collar, or one the stages leave out while it is quiet: no stiff
        # term to weigh the orders by.
        return math.inf
    interval = policy.collar_stability * policy.real_intervals[order - 1]
    return interval / (policy.collar_rate * dz)


def order_weight(
    policy: MarchPolicy, norm: float, order: int, dz: float, *, quiet: bool
) -> float:
    """Return an order's weight: its defect's step factor, capped for stability.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    norm : float
        The order's scaled defect.
    order : int
        The order.
    dz : float
        The accepted step, m.
    quiet : bool
        Whether the stages leave a quiet collar out.

    Returns
    -------
    float
        ``safety norm ** (-1 / (order + 1))``, infinite for a zero defect, at most
        :func:`stable_factor`.
    """
    rate = math.inf if norm <= 0.0 else norm ** (-1.0 / (order + 1))
    allowed = policy.safety * rate
    return min(allowed, stable_factor(policy, order, dz, quiet=quiet))


def choose_order(
    policy: MarchPolicy,
    current: int,
    scores: Mapping[int, float],
    dz: float,
    *,
    quiet: bool,
) -> tuple[int, int, float]:
    """Return the order to store, and the order and norm that set the next step.

    Under a fixed order the stored order is the method's while the step's own defect
    sets the next step (SEP 39 section 4.5, R1). Under variable order the scored order
    with the largest weight is adopted when it beats the step's own by the hysteresis;
    each order is weighed by the step its defect allows, capped at the step an in-stage
    collar leaves it stable at. The cap weighs the choice only; the step is the chosen
    order's defect's.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    current : int
        The step's own order, the first key of `scores`.
    scores : mapping
        The scaled defect of every scored order, in the stencil's order.
    dz : float
        The accepted step, m.
    quiet : bool
        Whether the stages leave a quiet collar out.

    Returns
    -------
    tuple
        The stored order ``q_s``, the control order ``q_c`` and its norm.
    """
    if not policy.variable_order:
        return policy.order, current, scores[current]
    weights = {
        order: order_weight(policy, norm, order, dz, quiet=quiet)
        for order, norm in scores.items()
    }
    best = max(scores, key=weights.__getitem__)
    if best != current and weights[best] > policy.hysteresis * weights[current]:
        return best, best, scores[best]
    return current, current, scores[current]


def rejection(
    policy: MarchPolicy, norm: float, order: int, *, rejections: int, held: int
) -> Rejection:
    """Return what a rejected trial of order `order` leaves.

    A rejection leaves the accepted history untouched. Repeated rejections mean the
    history itself is the problem -- a parasitic mode of the multistep recurrence that
    a smaller step cannot remove, because every stencil still interpolates it -- so
    after :attr:`~solprop.stepper.multistep.MarchPolicy.restart_after` in a row the
    method restarts from the accepted state, paying one startup call.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    norm : float
        The rejected trial's scaled defect, possibly not finite.
    order : int
        Its order.
    rejections : int
        The consecutive rejections before this one.
    held : int
        The history entries held.

    Returns
    -------
    Rejection
        The factor, the new count and whether the history restarts.
    """
    count = rejections + 1
    restart = count >= policy.restart_after and held > 1
    scale = (
        policy.min_factor
        if not math.isfinite(norm)
        else factor(policy, norm, order, growth=policy.rejection_growth)
    )
    return Rejection(scale, 0 if restart else count, restart)


def carrier_capped(
    policy: MarchPolicy,
    z: float,
    proposal: float,
    chord_phase: Callable[[float, float], float],
) -> float:
    """Return the proposal shortened to the carrier frame's phase cap.

    Under a carrier frame no step covers more than the policy's carrier cap of carrier
    phase (SEP 35 section 6.4), measured by one chord of the reference element's
    operator over the proposal, the largest over the batch. Applied before the landing
    rule, so a shortened step is not a landing; a fixed schedule's replay is not capped.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    z : float
        The current position, m.
    proposal : float
        The controller's proposal, m.
    chord_phase : callable
        ``(z1, z2) -> phase``, the chord's largest phase over the batch, rad.

    Returns
    -------
    float
        The proposal, or the shorter step the cap allows, m.
    """
    if not policy.carrier:
        return proposal
    # Measured over the part of the proposal inside the run, where the operator's
    # representation is defined; the landing rule clips the rest.
    end = min(z + proposal, policy.z_end)
    phase = chord_phase(z, end)
    if phase <= policy.carrier_cap:
        return proposal
    return (end - z) * (policy.carrier_cap / phase)


def quiet_capped(proposal: float, *, quiet: bool, interval: float | None) -> float:
    """Return the proposal, held to the monitor's interval while the collar rests.

    The rule of :meth:`solprop.backends.driver.StepDriver._quiet_capped`, which the
    Python march calls through its backend, with the interval passed in.

    Parameters
    ----------
    proposal : float
        The controller's proposal, m.
    quiet : bool
        Whether the stages leave a quiet collar out.
    interval : float or None
        The monitor's interval, m, or ``None`` without a monitor.

    Returns
    -------
    float
        The proposal, or the interval while the collar is left out, m.
    """
    if not quiet or interval is None:
        return proposal
    return min(proposal, interval)


def collar_reentry(
    policy: MarchPolicy,
    proposal: float,
    order: int,
    *,
    was_quiet: bool,
    quiet: bool,
) -> float:
    """Return the proposal held to the collar's stable step as its term returns.

    While a quiet collar is left out of the stages the order choice ignores its
    stability limit (:func:`stable_factor`), so the march may hold a high order and a
    long step; on the first attempt after the term returns that step would be unstable
    for it. That attempt is held to the stable step of the stored order, and the order
    choice weighs the limit again from there. The caller records `quiet` as the flag of
    the attempt, whatever this returns.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    proposal : float
        The proposal, m.
    order : int
        The stored order, ``q``.
    was_quiet : bool
        Whether the collar was quiet at the last attempt, ``w_Q``.
    quiet : bool
        Whether it is quiet now, ``Q``.

    Returns
    -------
    float
        The proposal, or the collar's stable step when its term has just returned, m.
    """
    returning = was_quiet and not quiet
    if not returning or policy.collar_rate <= 0.0:
        return proposal
    interval = policy.collar_stability * policy.real_intervals[order - 1]
    return min(proposal, interval / policy.collar_rate)


def next_landing(
    policy: MarchPolicy, z: float, step_targets: StepTargets, target: float
) -> float | None:
    """Return the next forced landing after `z`: save plane, hard kink or end.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy, for its landing tolerance.
    z : float
        The current position, m.
    step_targets : solprop.stepper.schedule.StepTargets
        The planes and kinks.
    target : float
        The end of the current march, m.

    Returns
    -------
    float or None
        The nearest landing strictly beyond ``z`` plus the tolerance, or ``None`` when
        none lies ahead.
    """
    tolerance = policy.landing_ulps * math.ulp(max(abs(z), abs(target)))
    candidates = [target]
    for positions in (step_targets.planes, step_targets.kinks):
        # The first strictly beyond `z + tolerance`; the positions increase strictly.
        index = bisect.bisect_right(positions, z + tolerance)
        if index < len(positions):
            candidates.append(positions[index])
    nearest = min(candidates)
    return nearest if nearest > z + tolerance else None


def approach(
    policy: MarchPolicy,
    z: float,
    stop: Landing,
    step_targets: StepTargets,
    target: float,
) -> tuple[float, float, bool]:
    """Return the step to take, where it ends and whether it lands on a kink.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy.
    z : float
        The current position, m.
    stop : solprop.stepper.schedule.Landing
        The landing the shared rule chose for the proposal.
    step_targets : solprop.stepper.schedule.StepTargets
        The caps and landings.
    target : float
        The end of this march, m.

    Returns
    -------
    tuple
        The step, m; its end position, m; and whether that end is a hard kink.
    """
    landed = stop.z_target is not None
    if not landed and policy.equal_approach:
        # A full step followed by a sliver to a forced landing is a large step ratio,
        # which variable-step Adams tolerates poorly (SEP 13 section 4.4): between one
        # and two proposals away, take two equal steps instead.
        ahead = next_landing(policy, z, step_targets, target)
        if ahead is not None and stop.dz < ahead - z < 2.0 * stop.dz:
            half = 0.5 * (ahead - z)
            return half, z + half, False
    end = stop.z_target if landed else z + stop.dz
    assert end is not None
    # A kink that is also a save plane may be reported as the plane; it still restarts
    # the history, which must not interpolate across a jump in D. The run's own end is
    # not one: nothing follows it to restart.
    tolerance = policy.landing_ulps * math.ulp(max(abs(end), 1.0))
    interior = end < policy.z_end - tolerance
    on_kink = interior and any(
        abs(kink - end) <= tolerance for kink in step_targets.kinks
    )
    return stop.dz, end, stop.target is LandingTarget.HARD_KINK or on_kink


def replay_kink(policy: MarchPolicy, end: float, slack: float) -> bool:
    """Return whether a replayed step's end is a hard kink: one within `slack` of it.

    Parameters
    ----------
    policy : solprop.stepper.multistep.MarchPolicy
        The run's policy, for its kinks.
    end : float
        The schedule's edge the step ends on, m.
    slack : float
        The replay's tolerance, m (:func:`solprop.backends.driver.replay_tolerance`).

    Returns
    -------
    bool
        Whether the history restarts there.
    """
    return any(abs(kink - end) <= slack for kink in policy.hard_kinks)

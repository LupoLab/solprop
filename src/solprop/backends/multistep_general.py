# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general Adams path: the comparison variants SEP 13 did not select.

PECE, explicit Adams-Bashforth and the matched predictor-corrector pairs controlled
by Milne's estimate (:class:`~solprop.stepper.multistep.ComparisonMultistep`) combine
a list of host arrays one product at a time rather than handing a store of history
rows to an engine. :class:`~solprop.backends.multistep.AdamsMarch` hands their trials to
:class:`GeneralPath`, which keeps their arithmetic and shares the march's controller,
history, ledger and lifecycle (SEP 39 phase 0, Q7). The selected method and the
one-order-higher PEC variants never reach it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from solprop._helpers import Helpers
from solprop._interrupts import committing
from solprop.backends import adams_core
from solprop.backends.driver import StepDriver
from solprop.backends.multistep import (
    AdamsMarch,
    _combine,
    _DenseStep,
    _Derivative,
    _State,
)
from solprop.stepper.multistep import integrated_weights, milne_factor

__all__ = ["GeneralPath"]


@dataclass
class _Trial:
    """One trial step: its stencil, its propagator and its predictor."""

    z: float
    dz: float
    order: int
    nodes: np.ndarray
    values: list[_State]
    state: _State
    propagator: _State
    predictor: _State
    scale: Callable[[_State], float]

    @classmethod
    def start(
        cls,
        history: list[_Derivative],
        order: int,
        z: float,
        dz: float,
        state: _State,
        backend: StepDriver,
    ) -> "_Trial":  # noqa: UP037 - a runtime hook evaluates it
        """Form the predictor of a step from the history, newest derivative first.

        Parameters
        ----------
        history : list
            The accepted derivatives, oldest first, in the anchor's frame.
        order : int
            The order asked for; fewer derivatives during startup lower it.
        z, dz : float
            The anchor and the step, m.
        state : numpy.ndarray
            The accepted state at the anchor.
        backend : solprop.backends.driver.StepDriver
            The backend, for the step's propagator and norm.

        Returns
        -------
        _Trial
            The trial, with its predictor formed.
        """
        recent = history[::-1]
        nodes = np.array([(entry.z - z) / dz for entry in recent])
        values = [cast("_State", entry.value) for entry in recent]
        used = min(order, len(history))
        return cls(
            z=z,
            dz=dz,
            order=used,
            nodes=nodes,
            values=values,
            state=state,
            propagator=backend._adams_propagator(z, z + dz),
            predictor=_combine(
                integrated_weights(nodes[:used], check=False, helpers=backend.helpers),
                values[:used],
                dz,
            ),
            scale=_norm_function(backend, state),
        )

    def lab(self, increment: _State) -> _State:
        """Return the lab-frame field at the end of the step for an increment.

        Parameters
        ----------
        increment : numpy.ndarray
            ``v - a_n`` in the anchor's frame.

        Returns
        -------
        numpy.ndarray
            ``U(z + dz, z) (a_n + increment)``.
        """
        return self.propagator * (self.state + increment)

    def norm(self, defect: _State) -> float:
        """Return the weak norm of a defect, carried to the end of the step.

        Parameters
        ----------
        defect : numpy.ndarray
            A local-defect estimate in the anchor's frame.

        Returns
        -------
        float
            The scaled norm; one is the tolerance.
        """
        return self.scale(self.propagator * defect)


class GeneralPath:
    """The trials of the general Adams path, on the march they belong to.

    Parameters
    ----------
    march : solprop.backends.multistep.AdamsMarch
        The march whose history, order, ledger and controller the trials use.
    """

    def __init__(self, march: AdamsMarch) -> None:
        """Hold the march."""
        self._march = march

    def attempt(
        self,
        dz: float,
        *,
        fixed: bool = False,
        landing: tuple[float, bool] | None = None,
    ) -> tuple[bool, float]:
        """Try one step of length `dz` on the general path; commit it if accepted.

        Parameters
        ----------
        dz : float
            The step, m.
        fixed : bool, optional
            Whether the step is a prescribed schedule's, accepted whatever its norm.
        landing : tuple or None, optional
            Where an accepted step ends, m, and whether that is a hard kink, which
            restarts the history; ``None`` ends it at ``z + dz`` and restarts nothing.

        Returns
        -------
        tuple
            Whether the step was accepted, and its scaled error norm.
        """
        march = self._march
        backend = march._backend
        state = backend._adams_state()
        z = backend._z
        backend._hoist(z)
        trial = _Trial.start(march._history, march._order, z, dz, state, backend)
        if march._method.corrector == "none" and len(trial.values) > trial.order:
            # Free rejection on the history alone: the next Adams-Bashforth formula's
            # difference from this one is this one's defect (SEP 13 section 4.3).
            wider = integrated_weights(
                trial.nodes[: trial.order + 1], check=False, helpers=backend.helpers
            )
            early = _combine(wider, trial.values[: trial.order + 1], dz)
            norm = trial.norm(early - trial.predictor)
            if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
                march._reject(dz, norm, trial.order)
                return False, norm

        march.ledger.predictor += 1
        source, _ = march._source(
            trial.lab(trial.predictor), z, z + dz, purpose="predictor"
        )
        increment, defect, formula = self.correct(trial, source)
        norm = trial.norm(defect)
        if not norm <= 1.0 and not (fixed and math.isfinite(norm)):
            march.ledger.rejected += 1
            march._reject(dz, norm, trial.order)
            return False, norm
        self.commit(trial, trial.lab(increment), source, norm, formula, landing)
        return True, norm

    def correct(
        self, trial: _Trial, source: _State
    ) -> tuple[_State, _State, tuple[np.ndarray, list[_State]]]:
        """Return the accepted increment and its local-defect estimate.

        Parameters
        ----------
        trial : _Trial
            The step being tried.
        source : numpy.ndarray
            The source at the predicted field, in the anchor's frame.

        Returns
        -------
        tuple
            The increment ``v - a_n`` the step would accept and the estimate of its
            local defect, both in the anchor's frame, and the nodes and derivatives of
            the formula that produced the increment, for dense observation.
        """
        march = self._march
        order, nodes, values, dz = trial.order, trial.nodes, trial.values, trial.dz
        if march._method.corrector == "none":
            # Explicit Adams-Bashforth keeps its predictor; the stencil that ends at the
            # new derivative rechecks it after the call (SEP 13 section 4.3).
            check = np.concatenate(([1.0], nodes[:order]))
            corrected = _combine(
                integrated_weights(check, check=False, helpers=march._backend.helpers),
                [source, *values[:order]],
                dz,
            )
            predictor = (nodes[:order], values[:order])
            return trial.predictor, corrected - trial.predictor, predictor
        count = march._method.corrector_nodes(order)
        corrector_nodes = np.concatenate(([1.0], nodes[: count - 1]))
        corrected = _combine(
            integrated_weights(
                corrector_nodes, check=False, helpers=march._backend.helpers
            ),
            [source, *values[: count - 1]],
            dz,
        )
        difference = corrected - trial.predictor
        formula = (corrector_nodes, [source, *values[: count - 1]])
        if count == order:
            factor = milne_factor(nodes[:order], corrector_nodes)
            return corrected, factor * difference, formula
        # A corrector one order higher: the difference is the predictor's defect, a
        # conservative estimate of the accepted corrector's.
        return corrected, difference, formula

    def commit(
        self,
        trial: _Trial,
        advanced: _State,
        source: _State,
        norm: float,
        formula: tuple[np.ndarray, list[_State]],
        landing: tuple[float, bool] | None = None,
    ) -> None:
        """Adopt an accepted step: the state, the re-anchored history and the proposal.

        Parameters
        ----------
        trial : _Trial
            The accepted step.
        advanced : numpy.ndarray
            The state at its end, before the boundary treatments.
        source : numpy.ndarray
            The source at the predicted field, in the step's anchor frame.
        norm : float
            The step's scaled error norm.
        formula : tuple
            The nodes and anchor-frame derivatives of the accepted formula.
        landing : tuple or None, optional
            Where the step ends and whether it is a hard kink (:meth:`attempt`).
        """
        march = self._march
        backend = march._backend
        method = march._method
        z, dz, propagator = trial.z, trial.dz, trial.propagator
        # The comparison variants' corrected evaluation is inside the commit too.
        with committing():
            finished = backend._adams_adopt(advanced, dz)
            if method.corrector == "pece":
                march.ledger.corrected += 1
                try:
                    stored, _ = march._source(finished, z, z + dz, purpose="corrected")
                except BaseException:
                    # Adoption installs the candidate before its corrected response. A
                    # failed response must leave the checkpoint at the previous state.
                    backend._adams_restore(z, trial.state)
                    raise
                identity = "accepted"
            else:
                stored = source
                identity = "predicted" if method.corrector == "pec" else "accepted"
            for entry in march._history:
                entry.value = propagator * cast("_State", entry.value)
            march._history.append(_Derivative(z + dz, propagator * stored, identity))
            march._history = march._history[-(method.order + 1) :]
            if march._interpolate:
                march._engine.dense = _DenseStep(
                    z=z,
                    dz=dz,
                    start=trial.state,
                    end=finished / propagator,
                    start_slope=trial.values[0],
                    end_slope=stored,
                    nodes=formula[0],
                    slopes=formula[1],
                )
            march.ledger.accepted_steps += 1
            march._rejections = 0
            march.ledger.orders.append(trial.order)
            order, norm = self.next_order(trial, source, norm)
            march._last_rejected = False
            march._land(z, dz, landing)
            policy = march._policy
            march._propose(
                dz, adams_core.factor(policy, norm, order, growth=policy.max_growth)
            )

    def next_order(
        self, trial: _Trial, source: _State, norm: float
    ) -> tuple[int, float]:
        """Choose the next step's order from neighbouring estimates; no extra call.

        For the one-order-higher PEC/PECE family the defect of order ``k`` is the
        difference between the ``k + 1``-derivative corrector through the new source
        and the ``k``-derivative predictor, so orders ``q - 1`` and ``q + 1`` are
        estimated from the same history and source as the step itself (SEP 13 section
        4.4). The order maximising the predicted step wins if it beats the current one
        by :data:`~solprop.stepper.multistep.ORDER_HYSTERESIS`; the order never rises
        straight after a rejection.

        Parameters
        ----------
        trial : _Trial
            The accepted step.
        source : numpy.ndarray
            Its predicted-state source, in the step's anchor frame.
        norm : float
            Its scaled defect at its own order.

        Returns
        -------
        tuple
            The next order and the norm it is controlled with.
        """
        march = self._march
        method = march._method
        current = trial.order
        if not method.variable_order:
            march._order = method.order
            return current, norm
        # A neighbour is scored when the history can form it; the order never rises
        # straight after a rejection, nor beyond the derivatives the history holds.
        rising = not march._last_rejected and len(trial.values) > current
        scores = {current: norm} | {
            order: _defect_at(trial, source, order, march._backend.helpers)
            for order in (current - 1, current + 1)
            if 1 <= order <= method.order
            and order <= len(trial.values)
            and (order < current or rising)
        }
        return march._choose_order(current, scores, trial.dz)


def _defect_at(trial: _Trial, source: _State, order: int, helpers: Helpers) -> float:
    """Return the scaled defect an accepted step would have had at another order.

    The order's predictor and its one-order-higher corrector are formed from the same
    history and the same predicted-state source, so scoring a neighbour costs no call.

    Parameters
    ----------
    trial : _Trial
        The accepted step.
    source : numpy.ndarray
        Its predicted-state source, in the step's anchor frame.
    order : int
        The neighbouring order, no more than the derivatives the trial holds.
    helpers : solprop._helpers.Helpers
        The owning backend's implementation of the Adams basis.

    Returns
    -------
    float
        The scaled predictor-corrector difference at that order.
    """
    nodes, values = trial.nodes[:order], trial.values[:order]
    predicted = _combine(
        integrated_weights(nodes, check=False, helpers=helpers), values, trial.dz
    )
    corrected = _combine(
        integrated_weights(
            np.concatenate(([1.0], nodes)), check=False, helpers=helpers
        ),
        [source, *values],
        trial.dz,
    )
    return trial.norm(corrected - predicted)


def _norm_function(
    backend: StepDriver,
    state: _State,
) -> Callable[[_State], float]:
    """Return the weak norm of a defect, scaled by the accepted state before the step.

    The Runge-Kutta march scales by the larger of the states before and after the step.
    An Adams step outside its stability region can multiply the state many times over
    while its predictor and corrector still agree to a fraction of the *new* state, so
    that scale lets a divergence pass as accurate: SEP 13 phase 2 measured PEC4 growing
    from 1e7 to 1e236 in five accepted steps with every scaled norm below 0.6. The state
    the step starts from is the scale that cannot grow with the step it judges.

    Parameters
    ----------
    backend : solprop.backends.driver.StepDriver
        The backend, for the controller and the in-band mask.
    state : numpy.ndarray
        The accepted state before the step.

    Returns
    -------
    callable
        ``defect -> norm`` for a defect carried to the end of the step.
    """
    problem = backend._problem
    mask, controller = problem.grids.norm_mask, problem.stepper.controller
    # The scale is the state's, the same for every defect of the step, so it is
    # reduced once: :func:`passes.error_norm`'s arithmetic, with the two state norms
    # it would take per call hoisted out.
    magnitude = float(np.linalg.norm(state[..., mask]))
    scale = controller.scale(magnitude, magnitude)
    divisor = scale[mask] if isinstance(scale, np.ndarray) else scale

    def scaled(defect: _State) -> float:
        """Return the scaled norm of a defect at the end of the step.

        Parameters
        ----------
        defect : numpy.ndarray
            The defect, carried to the end of the step.

        Returns
        -------
        float
            The scaled norm.
        """
        return float(np.linalg.norm(defect[..., mask] / divisor))

    return scaled

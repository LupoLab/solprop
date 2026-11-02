# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step-size control: the error-norm scaling and the PI controller.

The backend reduces the embedded pair's error estimate to a scalar; everything here is
scalar logic on that number (``docs/dev/stepper.md`` section 3). Two pieces make up the
control loop:

* a **norm scale**, the denominator the backend divides the error by, so that a scaled
  norm of 1 means the step sat exactly at the requested tolerance;
* the **controller**, which turns a history of scaled norms into a step-size factor.

The controller is the proportional-integral rule of Soederlind and Wang [1]_ rather
than the elementary rule of [2]_, because the integral term damps the response to
fluctuations in the error estimate. It is a frozen dataclass and
:meth:`PIController.propose` is a pure function of the norm history, so replaying a
run's norms reproduces its step sizes bitwise. That is what makes a fixed schedule
a faithful replay rather than an approximation.

References
----------
.. [1] G. Soederlind and L. Wang, "Adaptive time-stepping and computational stability",
       J. Comput. Appl. Math. 185, 225 (2006). doi:10.1016/j.cam.2005.03.008
.. [2] E. Hairer, S. P. Noersett and G. Wanner, *Solving Ordinary Differential
       Equations I*, 2nd ed., Springer (1993), section II.4.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.exceptions import ConvergenceError, StepperError

__all__ = [
    "DZ_MIN_RELATIVE",
    "START_GROWTH",
    "ControlRecord",
    "PIController",
    "default_dz_min",
    "limit_step",
    "weak_norm_scale",
    "weighted_norm_scale",
]


START_GROWTH = 100.0
"""Largest factor the start-up rule may take the first step's successor by."""


def weak_norm_scale(
    norm_y: float, norm_y_new: float, rtol: float, atol: float
) -> float:
    """Return the weak error-norm scale, one global number for the whole state.

    The scaled norm the controller consumes is ``‖e‖ / scale`` with

    ``scale = rtol * max(‖y‖, ‖y_new‖) + atol``,

    reduced over the in-band mask. A single global scale is deliberate. Scaling per
    element instead is catastrophic here: the in-band dynamic range of a hollow-fibre
    spectrum reaches 1e17 and roughly a third of in-band bins carry no light, so a
    5e-15 error in an empty bin scores several thousand and the step collapses to zero
    before a single step is accepted (``docs/dev/stepper.md`` section 10, trap 1).

    Note that ``atol`` is **additive**, not a floor on the state norm: a floor would
    never engage at realistic field magnitudes.

    Parameters
    ----------
    norm_y, norm_y_new : float
        Euclidean norms of the state before and after the trial step, in state units
        (sqrt(W) for modal amplitudes), non-negative.
    rtol : float
        Relative tolerance, dimensionless and positive.
    atol : float
        Absolute tolerance, in state units, non-negative.

    Returns
    -------
    float
        The scale to divide the error norm by, in state units.

    Examples
    --------
    >>> from solprop.stepper import weak_norm_scale
    >>> f"{weak_norm_scale(2.0, 3.0, rtol=1e-8, atol=0.0):.3e}"
    '3.000e-08'
    >>> weak_norm_scale(0.0, 0.0, rtol=1e-8, atol=1e-30)
    1e-30
    """
    return rtol * max(norm_y, norm_y_new) + atol


def weighted_norm_scale(
    norm_y: float,
    norm_y_new: float,
    rtol: float,
    atol: float,
    weights: Float[np.ndarray, " n_omega"],
) -> Float[np.ndarray, " n_omega"]:
    """Return a per-bin error-norm scale that tightens chosen parts of the spectrum.

    ``scale_i = rtol * w_i * max(‖y‖, ‖y_new‖) + atol``. The magnitude stays **global**,
    so this cannot reproduce the collapse that per-bin magnitudes cause (see
    :func:`weak_norm_scale`); the weights only redistribute the tolerance. A weight
    below 1 tightens that bin, and all-ones reproduces the weak scale exactly. This is
    the option for observables five to eight orders below the pump, such as a
    dispersive wave, which a whole-field norm leaves with no error control at all.

    .. note::
       Provisional. Which weighting actually serves weak observables best -- fixed
       per-band weights, per-mode weights or a rolling per-bin scale -- is the first
       experiment of ``docs/dev/stepper.md`` section 9 and is not yet settled.

    Parameters
    ----------
    norm_y, norm_y_new : float
        Euclidean norms of the state before and after the trial step, in state units,
        non-negative.
    rtol : float
        Relative tolerance, dimensionless and positive.
    atol : float
        Absolute tolerance, in state units, non-negative.
    weights : numpy.ndarray
        Positive, dimensionless weights of shape ``(n_omega,)``, one per spectral bin.

    Returns
    -------
    numpy.ndarray
        Per-bin scales of shape ``(n_omega,)``, in state units.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import weak_norm_scale, weighted_norm_scale
    >>> weights = np.array([1.0, 0.1])
    >>> weighted_norm_scale(2.0, 3.0, 1e-8, 0.0, weights)
    array([3.e-08, 3.e-09])
    >>> ones = np.ones(2)
    >>> bool(
    ...     np.array_equal(
    ...         weighted_norm_scale(2.0, 3.0, 1e-8, 0.0, ones),
    ...         np.full(2, weak_norm_scale(2.0, 3.0, 1e-8, 0.0)),
    ...     )
    ... )
    True
    """
    return rtol * weights * max(norm_y, norm_y_new) + atol


def limit_step(dz: float, factor: float, dz_min: float, dz_max: float) -> float:
    """Apply a step-size factor and clamp the result to the allowed range.

    Falling below ``dz_min`` is a failure, not something to clamp away. Clamping to
    the minimum step *and* accepting it would let a run that cannot meet its tolerance
    silently abandon it with no warning, no counter and no record in the output, so
    this raises instead.

    Parameters
    ----------
    dz : float
        The step just taken, m, positive.
    factor : float
        Step-size factor from the controller, dimensionless and positive.
    dz_min : float
        Smallest step the run may take, m, non-negative.
    dz_max : float
        Largest step the run may take, m, at least ``dz_min``.

    Returns
    -------
    float
        The next step, m, in ``[dz_min, dz_max]``.

    Raises
    ------
    StepperError
        If ``dz_max`` is smaller than ``dz_min``.
    ConvergenceError
        If the requested step falls below ``dz_min``.

    Examples
    --------
    >>> from solprop.stepper import limit_step
    >>> limit_step(1e-3, 1.1, dz_min=1e-9, dz_max=1e-2)
    0.0011
    >>> limit_step(1e-3, 1000.0, dz_min=1e-9, dz_max=1e-2)
    0.01
    """
    if dz_max < dz_min:
        raise StepperError(f"dz_max {dz_max:.3e} is below dz_min {dz_min:.3e}")
    proposed = min(dz * factor, dz_max)
    if proposed < dz_min:
        raise ConvergenceError(
            f"step control asked for {proposed:.3e} m, below the floor dz_min ="
            f" {dz_min:.3e} m; the tolerance cannot be met here"
        )
    return proposed


DZ_MIN_RELATIVE = 1e-9
"""Default step floor as a fraction of the propagation length.

The floor exists to fail fast, not to protect accuracy: a run whose controller wants a
step below a billionth of the length would need over a billion steps and never finish,
so raising at once turns hours of silence into an immediate ``ConvergenceError``. The
finest legitimate scale on a real gradient fibre sits about four orders below the median
step (``docs/dev/stepper.md`` section 10, trap 2), five orders above this floor, so it
never binds a healthy run.
"""


def default_dz_min(length: float) -> float:
    """Return the default step floor for a run of the given length.

    ``DZ_MIN_RELATIVE`` times the length. The floor applies to the *controller's
    proposal*, never to a step that landing clipped: a short step onto a save plane or
    a kink is a landing, not a failure to meet the tolerance, and
    :func:`solprop.stepper.schedule.landing` does not consult the floor.

    Parameters
    ----------
    length : float
        Propagation length, m, finite and positive.

    Returns
    -------
    float
        The floor, m.

    Raises
    ------
    StepperError
        If ``length`` is not finite and positive.

    Examples
    --------
    >>> from solprop.stepper import default_dz_min
    >>> default_dz_min(2.0)
    2e-09
    """
    if not math.isfinite(length) or length <= 0.0:
        raise StepperError(f"length must be finite and positive, got {length}")
    return DZ_MIN_RELATIVE * length


# PR02 is suppressed because numpydoc's static hook cannot see the __init__ that
# @dataclass generates, so it reads every documented field as an unknown parameter.
@dataclass(frozen=True, eq=False)
class PIController:  # numpydoc ignore=PR02
    """Proportional-integral step-size controller with lag and a reject-free mode.

    The factor applied after a step whose scaled norms are ``e_n`` and ``e_{n-1}`` is

    ``factor = clip(safety * (1/e_n)**beta1 * (1/e_{n-1})**beta2,
    min_factor, max_factor)``,

    the Lund (PI.4.2) rule of [1]_. A pure proportional controller reacts to every
    fluctuation of the estimate; the integral term damps that, which is what keeps the
    step size smooth enough for the step history to be worth replaying.

    Equality is by identity (``eq=False``) because ``weights`` is an array.

    Parameters
    ----------
    rtol : float
        Relative tolerance, dimensionless and positive.
    atol : float
        Absolute tolerance, in state units, non-negative.
    safety : float
        Factor applied to every proposal, dimensionless, in ``(0, 1]``. The default
        0.8 is the value measured with the reject-free mode below.
    max_factor : float
        Largest growth per step, dimensionless and above 1. The default 1.1 is a tight
        cap by the usual standards; it is what makes the reject-free mode safe, and
        SEP 14 measured a looser one to cost accuracy in weak spectral bands, which the
        norm does not see, for little saving once the start-up rule
        (:meth:`start_factor`) has removed the slow growth from `dz_init`.
    min_factor : float
        Smallest shrink per step, dimensionless, in ``(0, 1)``.
    beta1 : float
        Proportional gain, dimensionless and positive.
    beta2 : float
        Integral gain, dimensionless. The defaults 3/25 and -1/25 are Soederlind's
        ``(0.6, -0.2)`` already divided by ``k = estimator_order + 1 = 5``. The
        exponent follows the *estimator's* order, not the scheme's: an order-5 pair
        with an order-4 embedded companion has ``k = 5``. Use
        :meth:`for_estimator_order` for any other estimator.
    lag : int
        Number of steps by which the update is deferred, so the factor computed from
        step ``n`` is applied to step ``n + 1 + lag``. This lets the error norm come
        back from a device asynchronously. Lag 1 was measured to cost nothing (error
        x1.00-1.05, cost x0.99-1.00); lag 2 drifted to x2.45 on one configuration.
    reject_free : bool
        Never reject a step whose error norm is finite, letting the lagged update
        absorb the overshoot. Rejections happen 0.0-0.4 % of the time in practice, so
        this abolishes something that was barely happening, and it removes the
        per-step device synchronisation. A **non-finite** norm is still rejected: no
        lagged update can absorb a poisoned state.
    weights : numpy.ndarray or None
        Per-bin weights of shape ``(n_omega,)`` selecting
        :func:`weighted_norm_scale`; ``None`` selects :func:`weak_norm_scale`.
    estimator_order : int
        Order ``q`` of the error estimate, which sets the exponent of the shrink
        applied to a rejected step. 4 for an embedded order-5(4) pair.
    region : str
        The spectral bins the norm scores. ``"active"`` (the default) scores every bin
        the propagation evolves, ``omega_min / 2 < omega < 1.1 omega_max``: the
        requested band and the margin around it, where generated light propagates and
        the spectral collar absorbs it. ``"band"`` scores only the requested
        wavelength limits, the norm before this choice: light in the margin is then
        integrated without control, and can be wrong by as much as it holds while the
        band's error stays within tolerance (``docs/dev/stepper.md``).

    Raises
    ------
    StepperError
        If a tolerance, gain or factor is outside its valid range, or the region is
        not ``"active"`` or ``"band"``.

    References
    ----------
    .. [1] G. Soederlind and L. Wang, "Adaptive time-stepping and computational
           stability", J. Comput. Appl. Math. 185, 225 (2006).
           doi:10.1016/j.cam.2005.03.008

    Examples
    --------
    >>> from solprop.stepper import PIController
    >>> control = PIController(rtol=1e-8, atol=1e-30)
    >>> control.accepts(0.5), control.accepts(2.0)
    (True, False)
    >>> f"{control.propose([0.5, 0.5]):.4f}"
    '0.8694'
    """

    rtol: float
    atol: float
    safety: float = 0.8
    max_factor: float = 1.1
    min_factor: float = 0.2
    beta1: float = 3 / 25
    beta2: float = -1 / 25
    lag: int = 1
    reject_free: bool = False
    weights: Float[np.ndarray, " n_omega"] | None = None
    estimator_order: int = 4
    region: Literal["active", "band"] = "active"

    def __post_init__(self) -> None:
        """Validate the settings and adopt a read-only copy of the weights."""
        if self.rtol <= 0.0:
            raise StepperError(
                f"rtol must be positive, got {self.rtol}", parameter="rtol"
            )
        if self.atol < 0.0:
            raise StepperError(
                f"atol must be non-negative, got {self.atol}", parameter="atol"
            )
        if not 0.0 < self.safety <= 1.0:
            raise StepperError(f"safety must lie in (0, 1], got {self.safety}")
        if not 0.0 < self.min_factor < 1.0 < self.max_factor:
            raise StepperError(
                "need 0 < min_factor < 1 < max_factor, got"
                f" {self.min_factor} and {self.max_factor}"
            )
        if self.beta1 <= 0.0:
            raise StepperError(f"beta1 must be positive, got {self.beta1}")
        if self.lag < 0:
            raise StepperError(f"lag must be non-negative, got {self.lag}")
        if self.region not in ("active", "band"):
            raise StepperError(
                f"region must be 'active' or 'band', got {self.region!r}"
            )
        if self.estimator_order < 1:
            raise StepperError(
                f"estimator_order must be at least 1, got {self.estimator_order}"
            )
        if self.weights is not None:
            if not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
                raise StepperError("norm weights must be finite and positive")
            frozen = np.array(self.weights, dtype=np.float64)
            frozen.setflags(write=False)
            object.__setattr__(self, "weights", frozen)

    @classmethod
    def for_estimator_order(
        cls, estimator_order: int, rtol: float, atol: float
    ) -> "PIController":  # noqa: UP037 - bound after the class body
        """Build a controller whose PI gains match an error estimate of order ``q``.

        The Lund gains are ``beta1 = 0.6 / k`` and ``beta2 = -0.2 / k`` with
        ``k = q + 1``. Using the scheme's order in place of the estimator's is a
        standing mistake: it makes the controller oscillate rather than converge.

        Parameters
        ----------
        estimator_order : int
            Order ``q`` for which the error estimate behaves as ``C h**(q+1)``,
            at least 1.
        rtol : float
            Relative tolerance, dimensionless and positive.
        atol : float
            Absolute tolerance, in state units, non-negative.

        Returns
        -------
        PIController
            A controller with every other setting at its default.

        Examples
        --------
        >>> from solprop.stepper import PIController
        >>> control = PIController.for_estimator_order(4, rtol=1e-8, atol=0.0)
        >>> control.beta1 == 3 / 25 and control.beta2 == -1 / 25
        True
        """
        gain = 5 * (estimator_order + 1)
        return cls(
            rtol=rtol,
            atol=atol,
            beta1=3 / gain,
            beta2=-1 / gain,
            estimator_order=estimator_order,
        )

    def scale(
        self, norm_y: float, norm_y_new: float
    ) -> float | Float[np.ndarray, " n_omega"]:
        """Return the scale the backend divides the error estimate by.

        Parameters
        ----------
        norm_y, norm_y_new : float
            Euclidean norms of the state before and after the trial step, in state
            units, non-negative.

        Returns
        -------
        float or numpy.ndarray
            A single scale for the weak norm, or one per spectral bin when weights
            were given.
        """
        if self.weights is None:
            return weak_norm_scale(norm_y, norm_y_new, self.rtol, self.atol)
        return weighted_norm_scale(
            norm_y, norm_y_new, self.rtol, self.atol, self.weights
        )

    def propose(self, norm_history: Sequence[float]) -> float:
        """Return the step-size factor for the next step.

        A pure function of the history, so the same norms always give the same
        factors, bit for bit. That is what lets a discovered schedule be replayed.

        Parameters
        ----------
        norm_history : collections.abc.Sequence
            Scaled error norms of the accepted steps so far, oldest first,
            dimensionless. A value of 1 means the step sat exactly at tolerance.

        Returns
        -------
        float
            Factor to multiply the step size by, dimensionless, in
            ``[min_factor, max_factor]``.

        Examples
        --------
        >>> from solprop.stepper import PIController
        >>> control = PIController(rtol=1e-8, atol=0.0, lag=1)
        >>> control.propose([])          # nothing to go on yet
        1.0
        >>> control.propose([0.1])       # the lag has not filled yet either
        1.0
        >>> f"{control.propose([0.1, 0.2]):.4f}"
        '1.0546'
        >>> control.propose([1e-6, 0.2])   # growth is capped
        1.1
        """
        # With a lag of L the factor is computed from the norm L steps back, so the
        # update reaches the step size one step late and the reduction that produced
        # the norm has a whole step in which to come back from a device. Until the
        # history is that long the step is left alone, which primes the lag queue with
        # unit factors.
        index = len(norm_history) - 1 - self.lag
        if index < 0:
            return 1.0
        current = norm_history[index]
        if not math.isfinite(current):
            return self.min_factor
        if current <= 0.0:
            # An exactly zero estimate means the nonlinearity did nothing over this
            # step; grow as fast as the cap allows. It is not evidence of convergence.
            return self.max_factor
        previous = norm_history[index - 1] if index >= 1 else 1.0
        if not math.isfinite(previous) or previous <= 0.0:
            previous = 1.0
        factor = (
            self.safety * (1.0 / current) ** self.beta1 * (1.0 / previous) ** self.beta2
        )
        return min(max(factor, self.min_factor), self.max_factor)

    def start_factor(self, norm: float) -> float:
        """Return the factor the start-up rule takes the first accepted step by.

        The first step of a run is a guess, ``dz_init``. Its own scaled norm says how
        far it is from the tolerance, so the step after it is the elementary deadbeat
        one, ``safety * norm**(-1/(q+1))`` with ``q`` the estimator's order, at most
        :data:`START_GROWTH` times the first and never below `min_factor`. A run then
        starts near its natural step instead of growing to it at the cap of 1.1,
        which from the usual ``dz_init`` takes tens of steps (SEP 14 section 4.2).

        Parameters
        ----------
        norm : float
            Scaled error norm of the first accepted step at its proposed length,
            dimensionless.

        Returns
        -------
        float
            Factor in ``[min_factor, START_GROWTH]``, dimensionless.

        Examples
        --------
        >>> from solprop.stepper import PIController
        >>> control = PIController(rtol=1e-6, atol=0.0)
        >>> f"{control.start_factor(1e-5):.2f}"
        '8.00'
        >>> control.start_factor(0.0), control.start_factor(1e-30)
        (100.0, 100.0)
        """
        if not math.isfinite(norm) or norm <= 0.0:
            # No estimate to go on: a source that did nothing allows any step.
            return START_GROWTH if norm == 0.0 else self.min_factor
        factor = self.safety * (1.0 / norm) ** (1.0 / (self.estimator_order + 1))
        return min(max(factor, self.min_factor), START_GROWTH)

    def accepts(self, norm: float) -> bool:
        """Return whether a step with this scaled error norm is accepted.

        Parameters
        ----------
        norm : float
            Scaled error norm of the trial step, dimensionless.

        Returns
        -------
        bool
            True to keep the step. ``reject_free`` keeps every step with a finite
            norm; a NaN or infinite norm is rejected whatever the mode, because the
            state is already poisoned and no later update can undo that.

        Examples
        --------
        >>> import math
        >>> from solprop.stepper import PIController
        >>> free = PIController(rtol=1e-8, atol=0.0, reject_free=True)
        >>> free.accepts(1e6), free.accepts(math.nan)
        (True, False)
        """
        if not math.isfinite(norm):
            return False
        return self.reject_free or norm <= 1.0

    def rejection_factor(self, norm: float) -> float:
        """Return the factor to shrink by after a rejected step.

        Applied at once, without the lag: a rejected step has to be retried now. The
        exponent is the deadbeat ``1/(q+1)`` of the estimator's order, and the result
        can never exceed 1, so a rejection cannot grow the step.

        Parameters
        ----------
        norm : float
            Scaled error norm of the rejected step, dimensionless.

        Returns
        -------
        float
            Factor in ``[min_factor, 1]``, dimensionless.

        Examples
        --------
        >>> from solprop.stepper import PIController
        >>> control = PIController(rtol=1e-8, atol=0.0)
        >>> f"{control.rejection_factor(4.0):.4f}"
        '0.6063'
        """
        if not math.isfinite(norm):
            # A diverged trial step: shrink as hard as the settings allow and retry.
            return self.min_factor
        if norm <= 0.0:
            # Unreachable from a real rejection, which needs norm > 1; keep the step.
            return 1.0
        factor = self.safety * (1.0 / norm) ** (1.0 / (self.estimator_order + 1))
        return min(max(factor, self.min_factor), 1.0)

    def equivalent_norm(
        self, norm: float, dz_taken: float, dz_proposed: float
    ) -> float:
        """Rescale the norm of a clipped or stretched step to the step proposed.

        Landing shortens a step onto a save plane or a kink, or stretches it a little
        to reach one; the controller must not learn from that step as if it had chosen
        it. Its state is the *proposal*: the driver bases the next step on the proposal,
        not on the executed step, and feeds the history the norm the proposal would
        have scored under the asymptotic error model, ``norm * (dz_proposed /
        dz_taken) ** (q + 1)`` with ``q`` the estimator's order. Without this a step
        onto a plane restarts the controller from the sliver and it regrows at
        ``max_factor`` per step; a thousandfold recovery at 1.1 takes over seventy
        steps, and over two hundred planes that is more steps than the run needs.

        The same rescaling serves a rejection: a clipped step that failed means the
        longer proposal would have failed harder, and ``rejection_factor`` applied to
        the equivalent norm shrinks the proposal accordingly.

        Parameters
        ----------
        norm : float
            Scaled error norm of the step actually taken, dimensionless. A non-finite
            norm is returned as it is: it must still be rejected.
        dz_taken : float
            Length of the step taken, m, positive.
        dz_proposed : float
            Length the controller proposed before landing, m, positive.

        Returns
        -------
        float
            The norm to enter into the history, dimensionless.

        Raises
        ------
        StepperError
            If either length is not finite and positive.

        Examples
        --------
        >>> from solprop.stepper import PIController
        >>> control = PIController(rtol=1e-8, atol=0.0)
        >>> control.equivalent_norm(0.3, dz_taken=1e-3, dz_proposed=1e-3)
        0.3
        >>> f"{control.equivalent_norm(1e-6, dz_taken=1e-4, dz_proposed=1e-3):.2f}"
        '0.10'
        """
        for name, value in (("dz_taken", dz_taken), ("dz_proposed", dz_proposed)):
            if not math.isfinite(value) or value <= 0.0:
                raise StepperError(f"{name} must be finite and positive, got {value}")
        if not math.isfinite(norm):
            return norm
        return norm * (dz_proposed / dz_taken) ** (self.estimator_order + 1)


@dataclass(frozen=True)
class ControlRecord:
    """What a completed run achieved against what it asked for.

    The scaled error norm *is* achieved-over-requested for the local error: 1 means the
    step sat exactly at the requested tolerance. The map from that local tolerance to
    the global error at the output is a different matter entirely -- it is problem
    dependent and misses by one to three orders in either direction, which is why
    every run records what it achieved instead of assuming
    (``docs/dev/stepper.md`` section 3).
    The global ratio comes from ``solprop.stepper.schedule.Calibration``.

    Attributes
    ----------
    requested_rtol : float
        Relative tolerance the run asked for, dimensionless.
    requested_atol : float
        Absolute tolerance the run asked for, in state units.
    n_accepted : int
        Number of accepted steps.
    n_rejected : int
        Number of rejected attempts.
    max_norm : float
        Largest scaled error norm over the accepted steps, dimensionless; 0 for a run
        with no accepted steps.
    mean_norm : float
        Mean scaled error norm over the accepted steps, dimensionless; 0 for a run with
        no accepted steps.
    worst_step : int
        Index, among the accepted steps, of the one with the largest norm; -1 for a
        run with no accepted steps. The step history says where it was.
    """

    requested_rtol: float
    requested_atol: float
    n_accepted: int
    n_rejected: int
    max_norm: float
    mean_norm: float
    worst_step: int

    @classmethod
    def from_norms(
        cls,
        controller: PIController,
        accepted_norms: Sequence[float],
        n_rejected: int = 0,
    ) -> "ControlRecord":  # noqa: UP037 - bound after the class body
        """Summarise a run from the scaled norms of its accepted steps.

        Parameters
        ----------
        controller : PIController
            The controller the run used, which carries the requested tolerances.
        accepted_norms : collections.abc.Sequence
            Scaled error norm of every accepted step, dimensionless.
        n_rejected : int, optional
            Number of rejected attempts.

        Returns
        -------
        ControlRecord
            The summary.

        Examples
        --------
        >>> from solprop.stepper import ControlRecord, PIController
        >>> control = PIController(rtol=1e-8, atol=0.0)
        >>> record = ControlRecord.from_norms(control, [0.2, 0.8, 0.5], n_rejected=1)
        >>> record.tolerance_ratio, f"{record.rejection_rate:.2f}"
        (0.8, '0.25')
        """
        norms = list(accepted_norms)
        return cls(
            requested_rtol=controller.rtol,
            requested_atol=controller.atol,
            n_accepted=len(norms),
            n_rejected=n_rejected,
            max_norm=max(norms) if norms else 0.0,
            mean_norm=math.fsum(norms) / len(norms) if norms else 0.0,
            worst_step=max(range(len(norms)), key=norms.__getitem__) if norms else -1,
        )

    @property
    def met_tolerance(self) -> bool:
        """Return whether every accepted step sat within the requested tolerance.

        A fixed schedule is replayed without control, so this is the check that a
        fixed run was not too coarse: the embedded estimate is computed regardless, and
        a largest norm above 1 means a step exceeded the tolerance the norm is scaled
        by. An adaptive run in the rejecting mode meets it by construction; a
        reject-free run may not, by design.

        Returns
        -------
        bool
            True if the largest scaled norm is at most 1.
        """
        return self.max_norm <= 1.0

    @property
    def rejection_rate(self) -> float:
        """Return the fraction of attempts that were rejected.

        Returns
        -------
        float
            Between 0 and 1; 0 for a run with no attempts.
        """
        attempts = self.n_accepted + self.n_rejected
        return self.n_rejected / attempts if attempts else 0.0

    @property
    def tolerance_ratio(self) -> float:
        """Return the achieved local error as a multiple of the requested tolerance.

        Returns
        -------
        float
            The largest scaled norm over the accepted steps. Below 1 the run stayed
            inside its local tolerance everywhere.
        """
        return self.max_norm

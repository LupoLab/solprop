# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixed step schedules: discover, calibrate, replay, and step landing.

Adaptive stepping does not remove the convergence study, because ``rtol`` is a *local*
tolerance and the global error it buys is problem dependent. Fixed stepping is not less
accurate, and it wins outright for ensembles (identical steps make the numerical error a
common systematic that cancels rather than a per-member variance that adds), for
reproducibility, and on a device. So both are first class:

1. **discover** -- one adaptive run records ``h(z)`` as a :class:`FixedSchedule`;
2. **calibrate** -- :func:`calibrate` replays it at ``h``, ``h/r`` and ``h/r**2`` and
   measures the observed order, with a gate that refuses an order far above the
   scheme's;
3. **replay** -- :func:`replay` rescales the schedule to the step size that reaches a
   target error, and production runs, ensemble members and scan points execute that.

Separately, :func:`landing` is the per-step rule that clips a trial step so it ends
exactly on the next save plane or hard kink, which is what replaces dense output
(see ``docs/dev/decisions.md``). See ``docs/dev/stepper.md`` sections 3 and 8.
"""

import bisect
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from jaxtyping import Float

from solprop.exceptions import ConvergenceError, StepperError

__all__ = [
    "CONCLUSIVE_FACTOR",
    "ORDER_MARGIN",
    "Calibration",
    "FixedSchedule",
    "Landing",
    "LandingTarget",
    "StepTargets",
    "calibrate",
    "discover",
    "landing",
    "observed_order",
    "replay",
]

ORDER_MARGIN = 3
"""How far the observed order may exceed the scheme's before it is refused.

Three, because the two failure modes have to be told apart. A legitimate excursion --
a strongly nonlinear stretch where the asymptotic constant is still moving -- reads
about half an order high. A schedule that is off the convergence curve altogether reads
nine orders high: refining removed something that was not truncation error at all. A
two-level extrapolant cannot distinguish them, which is why the gate needs three levels
(``docs/dev/stepper.md`` section 10, trap 4).
"""

CONCLUSIVE_FACTOR = 1e3
"""How far above the round-off floor the coarsest difference must sit to be believed.

Below the floor the differences measure accumulated rounding rather than truncation and
the ratio means nothing; the same protocol has returned an observed order of -0.83 there
and would have been read as healthy.
"""

# A landing target within this many ulp of the current position counts as already
# passed. Eight covers the rounding of a snap onto the target plus the addition that
# put the position there; without it, landing exactly on a save plane would make the
# next step zero-length and the march would stall.
_LANDING_ULPS = 8

# Relative guard on the step count of a uniform segment (see FixedSchedule.uniform).
_UNIFORM_GUARD = 1e-12


@dataclass(frozen=True, eq=False)
class FixedSchedule:  # numpydoc ignore=PR02
    """A recorded sequence of step lengths covering one propagation.

    Equality is by identity (``eq=False``) because ``steps`` is an array.

    Parameters
    ----------
    steps : numpy.ndarray
        Step lengths of shape ``(n_steps,)``, m, strictly positive and finite.
    z_start : float, optional
        Position the first step starts from, m.

    Raises
    ------
    StepperError
        If the schedule is empty or any step is not finite and positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import FixedSchedule
    >>> schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
    >>> schedule.n_steps, schedule.z_end
    (3, 1.0)
    >>> schedule.refined(2).n_steps
    6
    """

    steps: Float[np.ndarray, " n_steps"]
    z_start: float = 0.0

    def __post_init__(self) -> None:
        """Validate the steps and adopt a read-only copy."""
        steps = np.array(self.steps, dtype=np.float64)
        if steps.size == 0:
            raise StepperError("a schedule needs at least one step")
        if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
            raise StepperError("every step must be finite and positive")
        if not math.isfinite(self.z_start):
            raise StepperError(f"z_start must be finite, got {self.z_start}")
        steps.setflags(write=False)
        object.__setattr__(self, "steps", steps)

    @classmethod
    def uniform(
        cls,
        z_start: float,
        z_end: float,
        dz: float,
        *,
        save_planes: Sequence[float] | np.ndarray = (),
        hard_kinks: Sequence[float] | np.ndarray = (),
    ) -> "FixedSchedule":  # noqa: UP037 - bound after the class body
        """Build a plain fixed-step schedule that lands on every plane and kink.

        A fixed-step run with no adaptivity at all. The run is cut into segments at
        every save plane and hard kink, and each segment is covered by the fewest
        equal steps of at most ``dz``, so no step exceeds ``dz`` and every target is
        an edge of the schedule (to rounding in the accumulated edges, which the
        driver snaps away as it does for a landing). There is no dense output, so a
        fixed schedule *must* contain the planes; this is the constructor that
        guarantees it.

        Parameters
        ----------
        z_start : float
            Where the run starts, m.
        z_end : float
            Where it ends, m, after ``z_start``.
        dz : float
            Largest step, m, positive. Steps are shortened, never lengthened, to fit
            the segments.
        save_planes : collections.abc.Sequence or numpy.ndarray, optional
            Positions the driver will save at, m, within ``[z_start, z_end]``.
            Positions at either end are dropped.
        hard_kinks : collections.abc.Sequence or numpy.ndarray, optional
            Positions where the linear operator jumps or kinks, m, within
            ``[z_start, z_end]``. Genuine kinks only, as for
            :class:`StepTargets`.

        Returns
        -------
        FixedSchedule
            The schedule.

        Raises
        ------
        StepperError
            If ``z_end`` is not after ``z_start``, ``dz`` is not finite and positive,
            or a position is non-finite or outside the run.

        Examples
        --------
        >>> from solprop.stepper import FixedSchedule
        >>> FixedSchedule.uniform(0.0, 1.0, 0.3, save_planes=[0.5]).steps
        array([0.25, 0.25, 0.25, 0.25])
        """
        if not (math.isfinite(z_start) and math.isfinite(z_end)) or z_end <= z_start:
            raise StepperError(f"need z_end > z_start, got {z_start} and {z_end}")
        if not math.isfinite(dz) or dz <= 0.0:
            raise StepperError(f"dz must be finite and positive, got {dz}")
        positions = np.concatenate(
            [
                np.asarray(save_planes, dtype=np.float64).ravel(),
                np.asarray(hard_kinks, dtype=np.float64).ravel(),
            ]
        )
        if not np.all(np.isfinite(positions)):
            raise StepperError("save_planes and hard_kinks must be finite")
        if np.any(positions < z_start) or np.any(positions > z_end):
            raise StepperError(
                "every save plane and hard kink must lie within [z_start, z_end]"
            )
        anchors = np.unique(np.concatenate([[z_start, z_end], positions]))
        pieces = []
        for start, end in itertools.pairwise(anchors):
            # A segment that is a whole number of steps to rounding must not gain a
            # sliver: 0.9 / 0.3 is 3.0000000000000004, and a bare ceil would make it 4.
            n = max(1, math.ceil((end - start) / dz * (1.0 - _UNIFORM_GUARD)))
            pieces.append(np.diff(np.linspace(start, end, n + 1)))
        return cls(np.concatenate(pieces), z_start)

    @classmethod
    def from_history(
        cls,
        z: Float[np.ndarray, " n_attempts"] | Sequence[float],
        h: Float[np.ndarray, " n_attempts"] | Sequence[float],
        accepted: Sequence[bool] | np.ndarray | None = None,
    ) -> "FixedSchedule":  # noqa: UP037 - bound after the class body
        """Return the accepted steps of a run's step history, to replay them exactly.

        A run's history records every attempt, rejected ones too: its start ``z``, its
        length ``h`` and whether it was accepted (``output["step_history/z"]`` and the
        like, the whole run concatenated). The accepted attempts are the steps the
        run took, so replaying them (``schedule=`` on the settings, with the same save
        planes) repeats the run's discretisation without its controller. Replayed in
        another precision or on another backend, that isolates rounding from
        truncation: the two runs make the same truncation errors, so their difference
        is the arithmetic's alone.

        Parameters
        ----------
        z : numpy.ndarray or collections.abc.Sequence
            Start of each attempt, m.
        h : numpy.ndarray or collections.abc.Sequence
            Length of each attempt, m.
        accepted : numpy.ndarray or collections.abc.Sequence or None, optional
            Whether each attempt was accepted; every attempt by default.

        Returns
        -------
        FixedSchedule
            The accepted steps, starting at the first accepted attempt's ``z``.

        Raises
        ------
        StepperError
            If the arrays differ in length, nothing was accepted, or the accepted
            steps do not follow each other (a step starting other than where the
            last ended, as in a history from two different runs).

        Examples
        --------
        >>> from solprop.stepper import FixedSchedule
        >>> z, h = [0.0, 0.0, 0.25, 0.75], [0.5, 0.25, 0.5, 0.25]
        >>> schedule = FixedSchedule.from_history(z, h, [False, True, True, True])
        >>> schedule.steps.tolist(), schedule.z_end
        ([0.25, 0.5, 0.25], 1.0)
        """
        starts = np.asarray(z, dtype=np.float64).ravel()
        lengths = np.asarray(h, dtype=np.float64).ravel()
        kept = (
            np.ones(starts.shape, dtype=np.bool_)
            if accepted is None
            else np.asarray(accepted, dtype=np.bool_).ravel()
        )
        if not starts.shape == lengths.shape == kept.shape:
            raise StepperError(
                "z, h and accepted must have one entry per attempt, got "
                f"{starts.size}, {lengths.size} and {kept.size}"
            )
        if not np.any(kept):
            raise StepperError("the history holds no accepted step")
        starts, lengths = starts[kept], lengths[kept]
        # A landing step ends exactly on its target rather than at z + h, so the next
        # start may differ from the sum by the rounding of that snap: a few ulp of z.
        ends = starts[:-1] + lengths[:-1]
        slack = 64 * np.finfo(np.float64).eps * np.maximum(np.abs(ends), lengths[:-1])
        gaps = np.abs(starts[1:] - ends)
        if np.any(gaps > slack):
            index = int(np.argmax(gaps > slack))
            raise StepperError(
                f"accepted step {index + 2} starts at {starts[index + 1]:.9g} m, not "
                f"where step {index + 1} ended ({ends[index]:.9g} m): the history is "
                "not one run's"
            )
        return cls(lengths, float(starts[0]))

    @property
    def n_steps(self) -> int:
        """Return the number of steps.

        Returns
        -------
        int
            Number of steps.
        """
        return int(self.steps.size)

    @property
    def edges(self) -> Float[np.ndarray, " n_edges"]:
        """Return the positions the steps start and end at.

        Returns
        -------
        numpy.ndarray
            Positions of shape ``(n_steps + 1,)``, m, accumulated in step order so
            that they match what a driver marching the schedule would reach.
        """
        return self.z_start + np.concatenate(([0.0], np.cumsum(self.steps)))

    @property
    def z_end(self) -> float:
        """Return the position the last step ends at.

        Returns
        -------
        float
            Final position, m.
        """
        return float(self.edges[-1])

    @property
    def total_length(self) -> float:
        """Return the propagation length the schedule covers.

        Returns
        -------
        float
            Length, m.
        """
        return self.z_end - self.z_start

    @property
    def max_step(self) -> float:
        """Return the longest step.

        Returns
        -------
        float
            Length, m.
        """
        return float(self.steps.max())

    @property
    def min_step(self) -> float:
        """Return the shortest step.

        Returns
        -------
        float
            Length, m.
        """
        return float(self.steps.min())

    def refined(self, factor: int) -> "FixedSchedule":  # noqa: UP037 - bound after the class body
        """Return the schedule with every step split into ``factor`` equal pieces.

        This is the refinement the three-level ladder uses. It preserves the shape of
        the schedule, including wherever it landed on a kink or a save plane, so the
        levels are nested and the ratio of their differences measures truncation and
        nothing else.

        Parameters
        ----------
        factor : int
            Number of pieces per step, at least 1. A power of two divides a double
            exactly, so each piece is exactly ``h / factor``; the accumulated endpoint
            can still move by an ulp or two, because the cumulative sum now has more
            terms in it.

        Returns
        -------
        FixedSchedule
            The refined schedule, covering the same span.

        Raises
        ------
        StepperError
            If ``factor`` is below 1.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.stepper import FixedSchedule
        >>> FixedSchedule(np.array([1.0, 0.5])).refined(2).steps.tolist()
        [0.5, 0.5, 0.25, 0.25]
        """
        if factor < 1:
            raise StepperError(f"refinement factor must be at least 1, got {factor}")
        if factor == 1:
            return self
        return FixedSchedule(np.repeat(self.steps / factor, factor), self.z_start)

    def rescaled(self, factor: float) -> "FixedSchedule":  # noqa: UP037 - bound after the class body
        """Return a schedule of the same shape and span with steps scaled.

        The step-size *profile* transfers while the endpoint stays put: the walk takes
        ``factor`` times the local step of the original schedule at each position, so a
        schedule discovered at one tolerance can be replayed at another without losing
        where it chose to go slowly. When the final piece comes out shorter than half
        its predecessor, the last two are evened up instead of leaving a sliver.

        Parameters
        ----------
        factor : float
            Scale applied to the local step, dimensionless and positive. Below 1
            refines, above 1 coarsens.

        Returns
        -------
        FixedSchedule
            The rescaled schedule, spanning the same interval to within the rounding
            of the accumulated sum.

        Raises
        ------
        StepperError
            If ``factor`` is not finite and positive.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.stepper import FixedSchedule
        >>> schedule = FixedSchedule(np.array([0.4, 0.4, 0.2]))
        >>> coarse = schedule.rescaled(2.0)
        >>> coarse.steps.tolist(), coarse.z_end
        ([0.5, 0.5], 1.0)
        """
        if not math.isfinite(factor) or factor <= 0.0:
            raise StepperError(
                f"scale factor must be finite and positive, got {factor}"
            )
        if factor == 1.0:
            return self
        edges = self.edges
        z_end = float(edges[-1])
        position = self.z_start
        lengths: list[float] = []
        # Terminates because every step is at least min_step * factor > 0 and the
        # walk only continues while it stays strictly short of the end.
        while True:
            local = float(self.steps[_interval_index(edges, position)])
            step = local * factor
            if position + step >= z_end:
                lengths.append(z_end - position)
                break
            lengths.append(step)
            position += step
        if len(lengths) > 1 and lengths[-1] < 0.5 * lengths[-2]:
            # Split the last two evenly rather than absorbing the sliver into its
            # predecessor, which would leave one step longer than the scale asked
            # for right where the run is about to stop.
            half = 0.5 * (lengths[-1] + lengths[-2])
            lengths[-2] = lengths[-1] = half
        return FixedSchedule(np.array(lengths), self.z_start)


def _interval_index(edges: Float[np.ndarray, " n_edges"], position: float) -> int:
    """Return the index of the interval of ``edges`` containing ``position``.

    Parameters
    ----------
    edges : numpy.ndarray
        Strictly increasing interval edges of shape ``(n + 1,)``, m.
    position : float
        Position to locate, m.

    Returns
    -------
    int
        Interval index, clamped to ``[0, n - 1]`` so that a position on either end of
        the covered range belongs to the interval that touches it.
    """
    index = int(np.searchsorted(edges, position, side="right")) - 1
    return min(max(index, 0), int(edges.size) - 2)


def discover(
    steps: Sequence[float] | np.ndarray, z_start: float = 0.0
) -> FixedSchedule:
    """Record the step sizes an adaptive run chose, as a schedule that can be replayed.

    This is the first step, discovery. The driver collects the length of every accepted
    step; this turns that record into a schedule, which :func:`calibrate` then measures
    and :func:`replay` rescales.

    Parameters
    ----------
    steps : collections.abc.Sequence
        Length of every accepted step, m, in the order they were taken.
    z_start : float, optional
        Position the run started from, m.

    Returns
    -------
    FixedSchedule
        The recorded schedule.

    Raises
    ------
    StepperError
        If the record is empty or holds a step that is not finite and positive.

    Examples
    --------
    >>> from solprop.stepper import discover
    >>> discover([0.1, 0.2, 0.15]).n_steps
    3
    """
    return FixedSchedule(np.asarray(steps, dtype=np.float64), z_start)


def observed_order(coarse: float, fine: float, refinement: int = 2) -> float:
    """Return the convergence order implied by two successive differences.

    Parameters
    ----------
    coarse : float
        Difference between the two coarsest levels, in the units of the observable.
    fine : float
        Difference between the two finest levels, same units.
    refinement : int, optional
        Ratio between successive step sizes.

    Returns
    -------
    float
        ``log(coarse / fine) / log(refinement)``, or NaN when either difference is not
        positive, in which case no order can be read.

    Examples
    --------
    >>> from solprop.stepper import observed_order
    >>> observed_order(1.6e-3, 1e-4)
    4.0
    """
    if coarse <= 0.0 or fine <= 0.0:
        return math.nan
    return math.log(coarse / fine) / math.log(refinement)


@dataclass(frozen=True)
class Calibration:
    """What a three-level replay measured, and what it licenses.

    Attributes
    ----------
    observed_order : float
        Order implied by the two differences, dimensionless; NaN when none could be
        read.
    declared_order : int
        Order the scheme claims, dimensionless.
    refinement : int
        Ratio between successive step sizes of the ladder.
    differences : tuple
        ``(|Q0 - Q1|, |Q1 - Q2|)`` for the observable ``Q`` at the three levels, in
        the units of the observable.
    achieved_error : float
        Richardson estimate of the error of the *base* schedule, same units. This is
        the global error the run actually delivered.
    requested : float
        Local tolerance the discovering run asked for, dimensionless.
    conclusive : bool
        Whether the coarsest difference sits far enough above the round-off floor for
        the order to mean anything.
    message : str
        One line saying what was measured, for logs and for the output metadata.
    """

    observed_order: float
    declared_order: int
    refinement: int
    differences: tuple[float, float]
    achieved_error: float
    requested: float
    conclusive: bool
    message: str

    @property
    def ratio(self) -> float:
        """Return the achieved global error as a multiple of the requested tolerance.

        Returns
        -------
        float
            ``achieved_error / requested``. It is not 1 and it is not constant: the map
            from a local tolerance to the global error is problem dependent and misses
            by one to three orders in either direction, which is the whole reason this
            is measured rather than assumed.
        """
        return self.achieved_error / self.requested

    def scale_for(self, target_error: float) -> float:
        """Return the step-size scale that would reach ``target_error``.

        Parameters
        ----------
        target_error : float
            Global error wanted, in the units of the observable, positive.

        Returns
        -------
        float
            Factor to pass to :meth:`FixedSchedule.rescaled`, dimensionless.

        Raises
        ------
        StepperError
            If ``target_error`` is not positive.
        ConvergenceError
            If the calibration was inconclusive or read no order, so there is no
            measured exponent to extrapolate along. Assuming one is exactly the
            mistake calibration exists to prevent.

        Examples
        --------
        >>> from solprop.stepper import Calibration
        >>> calibration = Calibration(
        ...     observed_order=5.0,
        ...     declared_order=5,
        ...     refinement=2,
        ...     differences=(1e-6, 3.1e-8),
        ...     achieved_error=1e-6,
        ...     requested=1e-8,
        ...     conclusive=True,
        ...     message="",
        ... )
        >>> calibration.ratio
        100.0
        >>> f"{calibration.scale_for(1e-8):.4f}"
        '0.3981'
        """
        if target_error <= 0.0:
            raise StepperError(f"target_error must be positive, got {target_error}")
        usable = (
            self.conclusive
            and math.isfinite(self.observed_order)
            and self.observed_order > 0.0
        )
        if not usable:
            raise ConvergenceError(
                "cannot scale a schedule from an inconclusive calibration:"
                f" {self.message}"
            )
        return (target_error / self.achieved_error) ** (1.0 / self.observed_order)


def calibrate(
    schedule: FixedSchedule,
    evaluate: Callable[[FixedSchedule], float],
    order: int,
    requested: float,
    refinement: int = 2,
    floor: float = 0.0,
) -> Calibration:
    """Replay a schedule at three levels and measure the order it converges at.

    The second step, calibration. ``evaluate`` is asked for a scalar observable at
    ``h``, ``h/r`` and ``h/r**2``; the two differences give the observed order and, by
    Richardson extrapolation, the error the base schedule actually carried.

    The observable is a scalar by design: this layer never touches an array of the
    field shape (``docs/dev/architecture.md`` section 3). The driver supplies whatever
    functional it wants calibrated -- an energy, a peak power, a band's content.

    Parameters
    ----------
    schedule : FixedSchedule
        The base schedule, usually one that :func:`discover` recorded.
    evaluate : collections.abc.Callable
        Runs a schedule and returns a scalar observable.
    order : int
        Order the scheme claims, at least 1.
    requested : float
        Local tolerance the discovering run asked for, dimensionless and positive.
    refinement : int, optional
        Ratio between successive step sizes, at least 2.
    floor : float, optional
        Round-off floor of the observable, same units, non-negative. When the coarsest
        difference is within ``CONCLUSIVE_FACTOR`` of it, the result is marked
        inconclusive and no order is pronounced. Left at 0 every ladder is believed,
        which is only safe when the caller knows the differences are above the floor.

    Returns
    -------
    Calibration
        The measurement.

    Raises
    ------
    StepperError
        If ``order``, ``requested`` or ``refinement`` is out of range.
    ConvergenceError
        If the observed order exceeds ``order + ORDER_MARGIN``. Refining removed
        something that is not truncation error, so the base schedule is off the
        convergence curve rather than merely large, and extrapolating from it would be
        meaningless.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import FixedSchedule, calibrate
    >>> base = FixedSchedule(np.full(8, 0.125))
    >>> # A synthetic order-5 observable: Q(h) = 1 + h**5.
    >>> quantity = lambda s: 1.0 + s.max_step**5
    >>> calibration = calibrate(base, quantity, order=5, requested=1e-8)
    >>> round(calibration.observed_order, 6)
    5.0
    """
    if order < 1:
        raise StepperError(f"order must be at least 1, got {order}")
    if requested <= 0.0:
        raise StepperError(f"requested must be positive, got {requested}")
    if refinement < 2:
        raise StepperError(f"refinement must be at least 2, got {refinement}")
    if floor < 0.0:
        raise StepperError(f"floor must be non-negative, got {floor}")

    coarse = evaluate(schedule)
    medium = evaluate(schedule.refined(refinement))
    fine = evaluate(schedule.refined(refinement * refinement))
    differences = (abs(coarse - medium), abs(medium - fine))
    conclusive = differences[0] > CONCLUSIVE_FACTOR * floor
    measured = observed_order(differences[0], differences[1], refinement)

    if conclusive and math.isfinite(measured) and measured > order + ORDER_MARGIN:
        raise ConvergenceError(
            f"observed order {measured:.2f} far exceeds the scheme's {order}:"
            " refining removed something that is not truncation error, so this"
            " schedule is off its convergence curve, not merely coarse"
        )

    # The exponent used for the extrapolation is the measured one where it can be
    # trusted; falling back on the declared order keeps the estimate defined for an
    # inconclusive ladder, whose achieved_error is not to be acted on anyway.
    usable = conclusive and math.isfinite(measured) and measured > 0.0
    exponent = measured if usable else float(order)
    achieved = differences[0] / (1.0 - refinement ** (-exponent))

    if not conclusive:
        message = (
            f"inconclusive: the coarsest difference {differences[0]:.2e} is within"
            f" {CONCLUSIVE_FACTOR:.0f}x of the round-off floor {floor:.2e}. This does"
            " NOT mean the step size is adequate; it means the ladder cannot see"
            " truncation error from here."
        )
    elif not math.isfinite(measured):
        message = (
            "inconclusive: one of the two differences vanished, so no order can be"
            " read from this ladder."
        )
    else:
        message = (
            f"observed order {measured:.2f} against a stated order of {order};"
            f" achieved error {achieved:.2e} for a requested tolerance of"
            f" {requested:.2e}"
        )
    return Calibration(
        observed_order=measured,
        declared_order=order,
        refinement=refinement,
        differences=differences,
        achieved_error=achieved,
        requested=requested,
        conclusive=conclusive and math.isfinite(measured),
        message=message,
    )


def replay(
    schedule: FixedSchedule, calibration: Calibration, target_error: float
) -> FixedSchedule:
    """Rescale a discovered schedule to the step size that reaches a target error.

    The third step, replay. Production runs, ensemble members and scan points execute
    the schedule this returns, so their numerical error is a common systematic rather
    than a per-run variance.

    Parameters
    ----------
    schedule : FixedSchedule
        The schedule that was calibrated.
    calibration : Calibration
        The measurement from :func:`calibrate`.
    target_error : float
        Global error wanted, in the units of the calibrated observable, positive.

    Returns
    -------
    FixedSchedule
        The schedule to run.

    Raises
    ------
    StepperError
        If ``target_error`` is not positive.
    ConvergenceError
        If the calibration was inconclusive, so no exponent was measured.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import FixedSchedule, calibrate, replay
    >>> base = FixedSchedule(np.full(8, 0.125))
    >>> quantity = lambda s: 1.0 + s.max_step**5
    >>> calibration = calibrate(base, quantity, order=5, requested=1e-8)
    >>> replay(base, calibration, calibration.achieved_error / 32).n_steps
    16
    """
    return schedule.rescaled(calibration.scale_for(target_error))


class LandingTarget(StrEnum):
    """What bound the length of a step.

    ``NONE`` means the controller's own proposal stood. ``MAX_STEP`` and
    ``INTERVAL_WIDTH`` are *caps*: they shorten the step but the step ends nowhere in
    particular. ``HARD_KINK``, ``SAVE_PLANE`` and ``END`` are *landings*: the step ends
    exactly on that position. When several coincide the later member of this list wins,
    so a save plane that sits on a kink is reported as the save plane.
    """

    NONE = "none"
    MAX_STEP = "max_step"
    INTERVAL_WIDTH = "interval_width"
    HARD_KINK = "hard_kink"
    SAVE_PLANE = "save_plane"
    END = "end"


_LANDING_PRIORITY = {target: rank for rank, target in enumerate(LandingTarget)}


@dataclass(frozen=True, eq=False)
class StepTargets:  # numpydoc ignore=PR02
    """Everything that can shorten a trial step.

    Equality is by identity (``eq=False``) because the fields are arrays.

    Parameters
    ----------
    save_planes : numpy.ndarray
        Positions of shape ``(n_z,)`` the driver will save at, m, strictly increasing.
        Steps land on them exactly, which is what replaces dense output.
    hard_kinks : numpy.ndarray
        Positions of shape ``(n_kinks,)`` where the linear operator jumps or kinks --
        pressure-stage boundaries, taper corners -- m, strictly increasing. Only
        *genuine* kinks belong here, never the edges the adaptive Chebyshev split
        introduces to resolve smooth variation: a typical fibre has thirty intervals
        and two kinks, so stopping at every edge would force fifteen times more
        boundaries than the physics needs.
    breakpoints : numpy.ndarray
        Chebyshev interval edges of shape ``(n_breakpoints,)``, m, strictly
        increasing, at least two.
    dz_max : float
        Largest step allowed anywhere, m, positive.
    z_end : float
        Where the propagation stops, m. The last step lands on it exactly rather than
        stepping past it.
    stretch : float, optional
        How far a step may be lengthened, as a factor on the capped trial step, to
        land on a target just beyond it instead of taking the step and then a sliver.
        The default, 1.1, is the controller's growth cap: a stretch the controller
        already permits every step, raising the local error by at most ``1.1**5``.
        Pass 1 to switch stretching off. A stretch never exceeds ``dz_max``.

    Attributes
    ----------
    kinks, planes, edges : tuple of float
        ``hard_kinks``, ``save_planes`` and ``breakpoints`` as floats, which the
        landing rule searches every step.

    Raises
    ------
    StepperError
        If an array is not strictly increasing, ``breakpoints`` has fewer than two
        entries, ``dz_max`` is not positive, or ``stretch`` is below 1.
    """

    save_planes: Float[np.ndarray, " n_z"]
    hard_kinks: Float[np.ndarray, " n_kinks"]
    breakpoints: Float[np.ndarray, " n_breakpoints"]
    dz_max: float
    z_end: float
    stretch: float = 1.1
    kinks: tuple[float, ...] = field(init=False, repr=False)
    planes: tuple[float, ...] = field(init=False, repr=False)
    edges: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the target sets and adopt read-only copies."""
        for name in ("save_planes", "hard_kinks", "breakpoints"):
            values = np.array(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise StepperError(f"{name} has a non-finite entry")
            if np.any(np.diff(values) <= 0.0):
                raise StepperError(f"{name} must be strictly increasing")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.breakpoints.size < 2:
            raise StepperError("breakpoints must describe at least one interval")
        if not self.dz_max > 0.0 or not math.isfinite(self.dz_max):
            raise StepperError(f"dz_max must be finite and positive, got {self.dz_max}")
        if not math.isfinite(self.z_end):
            raise StepperError(f"z_end must be finite, got {self.z_end}")
        if not math.isfinite(self.stretch) or self.stretch < 1.0:
            raise StepperError(
                f"stretch must be finite and at least 1, got {self.stretch}"
            )
        # The same positions as floats, for the searches a march makes every step:
        # `bisect` on a short tuple finds the index `numpy.searchsorted` finds on
        # strictly increasing values, at a fraction of the call's cost.
        for name, values in (
            ("kinks", self.hard_kinks),
            ("planes", self.save_planes),
            ("edges", self.breakpoints),
        ):
            object.__setattr__(self, name, tuple(values.tolist()))


@dataclass(frozen=True)
class Landing:
    """The step a landing rule allows, and what bound it.

    Attributes
    ----------
    dz : float
        Step to take, m, positive and no longer than the trial step.
    target : LandingTarget
        What bound it.
    z_target : float or None
        The position the step ends on for a landing, m; ``None`` for a cap, which
        shortens the step without aiming it anywhere. A driver snaps its position to
        this value after the step rather than accumulating ``z + dz``, so that a save
        plane is hit exactly however many steps have gone before.
    stretched : bool
        True if the step was lengthened beyond the caps to reach ``z_target``.
    """

    dz: float
    target: LandingTarget
    z_target: float | None
    stretched: bool = False


def _bound_order(
    bound: tuple[float, LandingTarget] | tuple[float, LandingTarget, float],
) -> tuple[float, int]:
    """Order a cap or a landing: shortest first, then the highest priority.

    Parameters
    ----------
    bound : tuple
        The step it allows, m, and what it is, then its position for a landing.

    Returns
    -------
    tuple
        The step and the negated priority.
    """
    return bound[0], -_LANDING_PRIORITY[bound[1]]


def landing(z: float, dz: float, targets: StepTargets) -> Landing:
    """Clip a trial step to the caps and the next position it must land on.

    Two kinds of bound apply (``docs/dev/stepper.md`` section 3):

    * **caps** -- ``dz_max``, and the width of the Chebyshev interval *containing*
      ``z``. The containing interval, never the narrowest one anywhere: on a real
      fibre the finest local scale is four orders below the median, and capping
      globally by it would turn a metre that is smooth almost everywhere into over a
      million steps (section 10, trap 2).
    * **landings** -- the next hard kink, the next save plane and the end of the run,
      each of which the step ends exactly on.

    The nearest landing wins when it lies within the capped step. When it lies just
    beyond, within ``targets.stretch`` times the capped step and no further than
    ``dz_max``, the step is *stretched* to reach it rather than taking the step and
    then a sliver; Hairer's DOPRI5 stretches its final step by one percent and
    MATLAB's ode45 by ten. The floor ``dz_min`` is never consulted here: a landing may
    be as short as the geometry demands
    (:func:`solprop.stepper.controller.default_dz_min`).

    Parameters
    ----------
    z : float
        Current position, m, inside the range the breakpoints cover and before
        ``z_end``.
    dz : float
        Step the controller proposed, m, positive.
    targets : StepTargets
        The caps and landing positions.

    Returns
    -------
    Landing
        The allowed step and what bound it.

    Raises
    ------
    StepperError
        If ``dz`` is not positive, or ``z`` is outside the breakpoints or at or past
        ``z_end``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stepper import StepTargets, landing
    >>> targets = StepTargets(
    ...     save_planes=np.array([0.25, 0.5, 0.75, 1.0]),
    ...     hard_kinks=np.array([0.4]),
    ...     breakpoints=np.array([0.0, 1.0]),
    ...     dz_max=0.3,
    ...     z_end=1.0,
    ... )
    >>> stop = landing(0.0, 1.0, targets)
    >>> stop.dz, stop.target.value, stop.z_target
    (0.25, 'save_plane', 0.25)
    >>> landing(0.25, 1.0, targets).z_target      # the kink, before the next plane
    0.4
    >>> landing(0.9, 0.05, targets).target        # nothing binds
    <LandingTarget.NONE: 'none'>
    >>> stop = landing(0.0, 0.24, targets)        # 0.25 lies within 1.1 x 0.24
    >>> stop.z_target, stop.stretched
    (0.25, True)
    """
    if not dz > 0.0 or not math.isfinite(dz):
        raise StepperError(f"dz must be finite and positive, got {dz}")
    # The searches run on the targets' float tuples (`StepTargets.edges` and the
    # rest): the rule runs once a step, so its small costs are a step's.
    edges = targets.edges
    if not edges[0] <= z <= edges[-1]:
        raise StepperError(
            f"z = {z:.6g} m lies outside the operator table"
            f" [{float(edges[0]):.6g}, {float(edges[-1]):.6g}] m"
        )
    # `math.ulp` is `np.spacing` of a non-negative float.
    tolerance = _LANDING_ULPS * math.ulp(max(abs(z), abs(targets.z_end)))
    if z >= targets.z_end - tolerance:
        raise StepperError(f"z = {z:.6g} m is already at or past z_end")

    # `_interval_index` on the tuple.
    index = min(max(bisect.bisect_right(edges, z) - 1, 0), len(edges) - 2)
    width = edges[index + 1] - edges[index]
    caps: list[tuple[float, LandingTarget]] = [
        (dz, LandingTarget.NONE),
        (targets.dz_max, LandingTarget.MAX_STEP),
        (width, LandingTarget.INTERVAL_WIDTH),
    ]
    landings: list[tuple[float, LandingTarget, float]] = []
    # The first position of each set at or beyond `z + tolerance`; the end is a set
    # of one.
    for kind, positions in (
        (LandingTarget.HARD_KINK, targets.kinks),
        (LandingTarget.SAVE_PLANE, targets.planes),
        (LandingTarget.END, (float(targets.z_end),)),
    ):
        found = bisect.bisect_left(positions, z + tolerance)
        if found < len(positions):
            landings.append((positions[found] - z, kind, positions[found]))
    # Shortest wins; among equals the highest-priority target, so a save plane sitting
    # on a kink is reported as the save plane and the caller snaps to it either way.
    cap_dz, cap_kind = min(caps, key=_bound_order)
    gap, kind, ahead = min(landings, key=_bound_order)
    if gap <= cap_dz:
        return Landing(dz=gap, target=kind, z_target=ahead)
    if gap <= min(targets.stretch * cap_dz, targets.dz_max):
        return Landing(dz=gap, target=kind, z_target=ahead, stretched=True)
    return Landing(dz=cap_dz, target=cap_kind, z_target=None)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The stepper section of a run: method, controller, step limits, schedule and planes.

:class:`StepperSettings` is section 7 of the kernel contract
(``docs/dev/kernel-contract.md``): the record that
:attr:`RunRequest.stepper <solprop.translate.run_request.RunRequest.stepper>` gathers a
request's method, error control, itinerary and planes back into, and that
:func:`~solprop.translate.run_request.describe_run` builds a request from. It is
run-specification data (``docs/dev/architecture.md`` section 3), so it lives beside the
tableaux, controllers and schedules it holds: below translation, which resolves it, and
below the backends and the driver, which read it. :data:`PropagatorMode` names the
stage-propagator choice it carries.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop._ownership import frozen_copy
from solprop.exceptions import TranslationError
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import Tableau, continuous_extension

__all__ = ["PropagatorMode", "StepperSettings"]

PropagatorMode = Literal["auto", "tabulate", "recompute"]
"""Whether the backend tabulates the stage propagators or recomputes them.

``"auto"``, the default, is the backend's own choice, resolved at construction and
recorded in the run's metadata: ``tabulate`` for ``numpy_ref``, whose two complex
exponentials per stage the table is worth memoising, and ``recompute`` for
``native``, where both signs come from one real ``exp`` and a ``sincos`` in the
stage that formed them, so the table buys residency and nothing else
(``docs/dev/backends.md`` section 3.6). Either can still be asked for by name; the two
agree bitwise, so the choice is memory and not numbers.
"""


_ADAMS_CONTROL = frozenset({"rtol", "atol", "weights", "region"})
"""The controller fields the Adams march reads. It chooses its steps by its own
``safety``, ``min_factor`` and ``max_growth``
(:class:`~solprop.stepper.multistep.Multistep`)."""


def _refuse_unread_control(controller: PIController) -> None:
    """Refuse a controller setting the Adams march would ignore, by name."""
    default = PIController(rtol=controller.rtol, atol=controller.atol)
    unread = [
        one.name
        for one in fields(PIController)
        if one.name not in _ADAMS_CONTROL
        and getattr(controller, one.name) != getattr(default, one.name)
    ]
    if unread:
        msg = (
            "Multistep() reads only the controller's rtol, atol, weights and region, "
            "and chooses its steps by its own safety, min_factor and max_growth: "
            f"PIController's {', '.join(unread)} would be ignored. Set them on "
            "Multistep, or leave them at their defaults"
        )
        raise TranslationError(msg)


@dataclass(frozen=True, eq=False)
class StepperSettings:
    """The tableau, the controller and the schedule of a run.

    Section 7 of the kernel contract. The tolerances and the norm choice live in the
    controller and nowhere else.

    Attributes
    ----------
    tableau : solprop.stepper.tableau.Tableau
        Nodes, matrix, solution weights, error weights ``b - b_hat``, orders and the
        first-same-as-last flag.
    controller : solprop.stepper.controller.PIController
        Tolerances, safety, growth caps, the PI exponents, the lag, the reject-free
        switch, the estimator order and the norm weights.
    dz_min : float
        Floor on the controller's *proposal*, m. A proposal below it raises rather than
        being clamped silently; it never applies to a landing step.
    dz_max : float
        Cap on the proposal, m.
    dz_init : float
        First step, m.
    stretch : float
        Landing stretch factor: a target within this multiple of the capped proposal is
        landed on directly. 1 switches it off.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        A fixed schedule to replay, whose edges must contain every save plane and hard
        kink. ``None`` for an adaptive run.
    save_planes : numpy.ndarray
        Positions in z the driver will ask for, m, shape ``(n_z,)``, increasing. Steps
        land on them exactly; there is no dense output.
    propagator_mode : {'auto', 'tabulate', 'recompute'}
        Whether the backend tabulates the stage propagator factors or recomputes them.
        ``"auto"`` leaves the choice to the backend, which records what it resolved.
    integrator : solprop.stepper.multistep.Multistep or LowStorageRungeKutta or None
        A low-call Adams method to step with instead of the tableau (SEP 13), the
        low-storage RK4 march (SEP 24), or ``None`` for the Runge-Kutta march. With a
        `schedule` an Adams method replays the schedule's steps (SEP 16 section 5);
        the low-storage march replays one and nothing else.
    interval_cap : bool
        Whether a step is capped at the width of the operator-table interval it starts
        in. The cumulative table makes the flow exact across intervals, so the cap is
        off on the diagonal-flow routes and on for a coupled flow, whose flow error the
        embedded estimate does not see (SEP 14 section 4.1).
    auto_init : bool
        Whether the start-up rule takes the step after the first from the first step's
        own error estimate (``PIController.start_factor``); ``False`` grows from
        `dz_init` at the growth cap.
    interpolate_planes : bool
        Step across the save planes and observe each inside its accepted step, through
        the tableau's continuous extension or the Adams march's own polynomial (SEP 14
        phase 3); ``False`` lands on them. The end of the run and every hard kink are
        landed on either way.
    step_planes : int
        A positive ``k`` also saves the end of every ``k``-th accepted step: the
        march returns to its session there, short of its target, and marches on
        (SEP 14 phase 3). ``0`` saves the requested planes only.

    Raises
    ------
    TranslationError
        If the step caps are not ordered, the stretch is below one, the save
        planes are not increasing and positive, or the low-storage march is asked for
        without a schedule, with interpolated planes or with a schedule step longer
        than `dz_max`.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5, PIController
    >>> from solprop.stepper.settings import StepperSettings
    >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
    ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "tabulate")
    >>> settings.save_planes.tolist()
    [0.5, 1.0]
    """

    tableau: Tableau
    controller: PIController
    dz_min: float
    dz_max: float
    dz_init: float
    stretch: float
    schedule: FixedSchedule | None
    save_planes: Float[np.ndarray, " n_z"] | Sequence[float]
    propagator_mode: PropagatorMode
    integrator: Multistep | LowStorageRungeKutta | None = None
    interval_cap: bool = True
    auto_init: bool = False
    interpolate_planes: bool = False
    step_planes: int = 0

    def __post_init__(self) -> None:
        """Adopt a read-only plane array and check the caps.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        planes = frozen_copy(np.reshape(self.save_planes, -1), np.float64)
        object.__setattr__(self, "save_planes", planes)
        if planes.size == 0 or not np.all(np.diff(planes) > 0.0):
            msg = f"save_planes must be non-empty and increasing, got {planes}"
            raise TranslationError(msg)
        if planes[0] <= 0.0:
            msg = f"save_planes must be positive, got {planes[0]!r} first"
            raise TranslationError(msg)
        if not 0.0 < self.dz_min <= self.dz_init <= self.dz_max:
            msg = (
                f"the step caps must satisfy 0 < dz_min <= dz_init <= dz_max, got "
                f"{self.dz_min!r}, {self.dz_init!r}, {self.dz_max!r}"
            )
            raise TranslationError(msg)
        if not self.stretch >= 1.0:
            msg = f"stretch must be at least 1, got {self.stretch!r}"
            raise TranslationError(msg)
        if self.step_planes < 0:
            msg = f"step_planes must be 0 (off) or a positive count: {self.step_planes}"
            raise TranslationError(msg)
        if isinstance(self.integrator, LowStorageRungeKutta):
            self._check_low_storage(self.integrator)
        if isinstance(self.integrator, Multistep):
            _refuse_unread_control(self.controller)
        if self.interpolate_planes and self.schedule is not None:
            msg = "interpolated planes lie inside adaptive steps; a schedule lands"
            raise TranslationError(msg)
        if (
            self.interpolate_planes
            and self.integrator is None
            and continuous_extension(self.tableau) is None
        ):
            msg = f"{self.tableau.name} has no continuous extension; use DOPRI5"
            raise TranslationError(msg)

    def _check_low_storage(self, method: LowStorageRungeKutta) -> None:
        """Refuse what the low-storage march cannot replay, before any allocation.

        The march accepts every step in place and has no estimate, so it replays a
        prescribed schedule and nothing else (SEP 24 section 5); `dz_max` stays a true
        statement about the run it records, so a longer schedule step is refused.
        """
        if self.schedule is None:
            msg = (
                f"the {method.name} march replays a prescribed schedule; pass "
                "schedule=FixedSchedule.uniform(...), or step adaptively with "
                "RungeKutta()"
            )
            raise TranslationError(msg)
        if self.interpolate_planes:
            msg = (
                f"the {method.name} march lands on its schedule's edges and has no "
                "dense output; drop interpolate_planes"
            )
            raise TranslationError(msg)
        # A schedule's steps are differences of rounded positions, so one may exceed
        # the step it was cut to by a few ulps of those positions.
        span = max(abs(self.schedule.z_start), abs(self.schedule.z_end), self.dz_max)
        longest = self.schedule.max_step
        if longest > self.dz_max + 4.0 * float(np.spacing(span)):
            msg = (
                f"the schedule's longest step is {longest:.6g} m and dz_max is "
                f"{self.dz_max:.6g} m; the {method.name} march records dz_max as a "
                "limit of the run, so raise dz_max to the schedule's longest step"
            )
            raise TranslationError(msg)

    @property
    def z_end(self) -> float:  # numpydoc ignore=RT01
        """Last save plane, where the run stops, m.

        Examples
        --------
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.stepper.settings import StepperSettings
        >>> StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14), 1e-9, 0.1,
        ...                 1e-3, 1.1, None, [0.5, 1.0], "tabulate").z_end
        1.0
        """
        return float(np.max(self.save_planes))

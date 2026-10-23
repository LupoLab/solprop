# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage interaction-picture Runge-Kutta march as a run's integrator (SEP 24).

A third integrator beside :class:`solprop.stepper.multistep.Multistep` and
:class:`solprop.stepper.runge_kutta.RungeKutta`: the classical fourth-order Runge-Kutta
method in the interaction picture anchored at the step's midpoint, Lawson's
transformation [1]_ in Hult's arrangement [2]_ (RK4IP). It replays a prescribed
:class:`~solprop.stepper.schedule.FixedSchedule` and holds three field-sized registers
beside the source pipeline's scratch: the state, overwritten in place by its
interaction-picture image and then by the accepted step, one accumulator and one stage
register. It has no embedded error estimate; its accuracy is calibrated from outside,
on the measured observable, by :func:`~solprop.stepper.schedule.calibrate`
(``docs/guide/numerics/fixed-schedules.md``).

This module holds the specification a run carries. The march and its register passes
are in :mod:`solprop.backends.low_storage`.

References
----------
.. [1] J. D. Lawson, "Generalized Runge-Kutta processes for stable systems with large
       Lipschitz constants", SIAM Journal on Numerical Analysis 4, 372 (1967).
       doi:10.1137/0704033
.. [2] J. Hult, "A fourth-order Runge-Kutta in the interaction picture method for
       simulating supercontinuum generation in optical fibers", J. Lightwave Technol.
       25, 3770 (2007). doi:10.1109/JLT.2007.909373
"""

from dataclasses import dataclass
from typing import Literal

from solprop.exceptions import StepperError

__all__ = [
    "LOW_STORAGE_REAL_INTERVAL",
    "LowStorageRungeKutta",
    "LowStorageScheme",
]

LowStorageScheme = Literal["rk4ip"]
"""The register schemes :class:`LowStorageRungeKutta` steps with.

``"rk4ip"`` is classical RK4 in the midpoint interaction picture, four source
evaluations a step and no estimate (SEP 24 section 4.2).
"""

LOW_STORAGE_REAL_INTERVAL = 2.785
"""Negative-real-axis stability interval of classical RK4, rounded down.

RK4's stability function is ``R(x) = 1 + x + x**2/2 + x**3/6 + x**4/24``; on the
negative real axis ``|R(x)| <= 1`` for ``x >= -2.7853`` (the root of ``|R| = 1``), so a
damping rate integrated explicitly -- the in-stage collars, plasma loss -- must keep
``h alpha_max / 2`` at most 2.785, against 3.3 for DOPRI5
(:data:`solprop.boundaries.STABILITY_RADIUS`). Checked against the stability function in
``tests/stepper/test_low_storage.py``.
"""


@dataclass(frozen=True)
class LowStorageRungeKutta:
    """RK4 in the midpoint interaction picture, replaying a schedule in three registers.

    The integrator for a fixed-step run whose state is too large for the Runge-Kutta
    march's stage rows or the Adams history: a Cartesian free-space grid of gigabytes,
    an ensemble or a scan on one schedule. Each step makes four source evaluations and
    accepts its result in place, so the march needs a schedule
    (``FixedSchedule.uniform`` for plain fixed steps) and records no error estimate:
    the step history's norm column is 0.0, meaning "not estimated". Calibrate the
    schedule on the observable that matters with
    :func:`~solprop.stepper.schedule.calibrate` at ``order=4``
    (``docs/guide/numerics/choosing-a-stepper.md``).

    Attributes
    ----------
    scheme : {'rk4ip'}
        The register scheme; ``"rk4ip"``, the only one, is Hult's RK4IP.

    Raises
    ------
    StepperError
        If the scheme is not one of :data:`LowStorageScheme`.

    Examples
    --------
    >>> from solprop.stepper.low_storage import LowStorageRungeKutta
    >>> LowStorageRungeKutta()
    LowStorageRungeKutta(scheme='rk4ip')
    >>> LowStorageRungeKutta().name, LowStorageRungeKutta().order
    ('low-storage-rk4ip', 4)
    """

    scheme: LowStorageScheme = "rk4ip"

    def __post_init__(self) -> None:
        """Check the scheme.

        Raises
        ------
        StepperError
            Naming the scheme asked for.
        """
        # The Literal annotation refuses an unknown scheme first wherever the
        # annotations are checked at run time, as in the test suite; this check holds
        # where they are not.
        if self.scheme != "rk4ip":
            msg = f"scheme must be 'rk4ip', got {self.scheme!r}"
            raise StepperError(msg)

    @property
    def name(self) -> str:
        """Return the method's record name.

        Returns
        -------
        str
            ``low-storage-<scheme>``.
        """
        return f"low-storage-{self.scheme}"

    @property
    def order(self) -> int:
        """Return the scheme's classical order, the ``order`` to calibrate at.

        Returns
        -------
        int
            Four.
        """
        return 4

    @property
    def evaluations(self) -> int:
        """Return the source evaluations a step makes.

        Returns
        -------
        int
            Four: no slope carries over, because the fourth is taken at a predicted
            state, not at the accepted one.
        """
        return 4

    @property
    def real_stability_interval(self) -> float:
        """Return the negative-real stability interval for a damping term.

        Returns
        -------
        float
            :data:`LOW_STORAGE_REAL_INTERVAL`: the largest ``h |lambda|`` for a damping
            rate ``lambda`` integrated explicitly.
        """
        return LOW_STORAGE_REAL_INTERVAL

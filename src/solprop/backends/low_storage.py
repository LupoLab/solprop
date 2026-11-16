# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage interaction-picture RK4 march: replay-only, accepted in place.

SEP 24 (``docs/dev/seps/archive/``,
``sep-0024-low-storage-interaction-picture-runge-kutta.md``).
Classical fourth-order Runge-Kutta in the interaction picture anchored at the step's
midpoint, Lawson's transformation [1]_ in Hult's arrangement [2]_. For a step
``[z_n, z_n + h]`` with ``z_m = z_n + h/2``, ``P = U(z_m, z_n)``, ``Q = U(z_n + h,
z_m)`` and ``U(z_2, z_1) = exp(Phi(z_2) - Phi(z_1))``, the step is

``g1 = P F(z_n, A_n)``, ``F2 = F(z_m, P A_n + h/2 g1)``,
``F3 = F(z_m, P A_n + h/2 F2)``, ``F4 = F(z_n + h, Q (P A_n + h F3))``,
``A_{n+1} = Q (P A_n + h/6 g1 + h/3 F2 + h/3 F3) + h/6 F4``,

with ``F`` the complete source (the response, the in-stage absorbers and the
``modified`` noise draw at the evaluation's position). Only forward flows are applied,
four per element, and on a constant operator ``P = Q = exp(h D / 2)``.

**The register form** (SEP 24 section 4.3) holds three field-sized registers beside the
source pipeline's scratch ``S``: the state ``A``, overwritten in place by ``P A`` and
then by the accepted step, the accumulator ``B`` and the stage register ``C``. A step
is four passes, each forming an input, evaluating ``F`` into ``S`` and folding it into
the registers:

====  ==============  ======================================================
pass  evaluation      finish
====  ==============  ======================================================
1     ``F1`` at z_n   ``A <- P A``; ``g <- P S``; ``B <- A + h/6 g``;
                      ``C <- A + h/2 g``
2     ``F2`` at z_m   ``B <- B + h/3 S``; ``C <- A + h/2 S``
3     ``F3`` at z_m   ``B <- B + h/3 S``; ``C <- Q (A + h S)``
4     ``F4`` at end   ``A <- Q B + h/6 S``
====  ==============  ======================================================

**The march** (the reference's
:class:`solprop.backends.host.low_storage.HostLowStorageMarch`; the native session's
replay) replays a prescribed schedule and never rejects, so it accepts each step in
place: from the start of the step's trial to its accept the engine is *in flight*
and every entry that reads or advances the state refuses it, until a load replaces the
state (SEP 24 section 6.2, ADR-050). It records no error estimate: the step history's
norm is 0.0, "not estimated" (section 6.3). Divergence is detected on the accepted
state's norm instead.

**The engine** holds the registers and makes the passes: it is a
:class:`~solprop.backends.engines.RungeKuttaEngine` with the documented destructive
trial (SEP 52 section 5.2). The reference's is
:class:`~solprop.backends.numpy_ref.engines.NumpyLowStorage`, over the workspace's
``lab_source`` and ``propagator``; native's is resident. The table keys
(:func:`table_steps`, :func:`schedule_key`) are stateless numerical code the two
backends share (SEP 47 Part III decision 5).

References
----------
.. [1] J. D. Lawson, "Generalized Runge-Kutta processes for stable systems with large
       Lipschitz constants", SIAM Journal on Numerical Analysis 4, 372 (1967).
       doi:10.1137/0704033
.. [2] J. Hult, "A fourth-order Runge-Kutta in the interaction picture method for
       simulating supercontinuum generation in optical fibers", J. Lightwave Technol.
       25, 3770 (2007). doi:10.1109/JLT.2007.909373
"""

import numpy as np
from jaxtyping import Float

from solprop.stepper.schedule import LANDING_ULPS, FixedSchedule

__all__ = ["schedule_key", "table_steps"]


def table_steps(schedule: FixedSchedule) -> Float[np.ndarray, " n_steps"]:
    """Return the step each of a schedule's steps builds and keys the flow table on.

    SEP 24 Q10, decided in phase 2: the mean step of the step's **run**, the longest
    stretch of consecutive steps within :data:`~solprop.stepper.schedule.LANDING_ULPS`
    ulps of the run's first, the ulps being of the run's farthest position (or of the
    step, near the origin). :meth:`~solprop.stepper.schedule.FixedSchedule.uniform`
    takes a segment's steps from differences of rounded positions, which differ in their
    last bits by up to an ulp of the *position*: far from the origin that is many ulps
    of the step, and a table keyed on one step's own value within ulps of the step was
    rebuilt nearly every step. Keyed on the run's mean ``h_bar`` instead, a uniform
    segment builds one table and reuses it; each flow's exponent is then off by ``|D| |h
    - h_bar| / 2``, a few ulps of the position times ``|D|``, and the error does not
    drift: the steps of a run sum to ``n h_bar`` to rounding, so the phase the flows
    apply over a run is the run's own. A schedule whose steps all differ (a discovered
    one) has runs of one step and keys each step on itself.

    Parameters
    ----------
    schedule : solprop.stepper.schedule.FixedSchedule
        The schedule replayed.

    Returns
    -------
    numpy.ndarray
        One key per step, m, shape ``(n_steps,)``: constant over a run.

    Examples
    --------
    >>> from solprop.backends.low_storage import table_steps
    >>> from solprop.stepper import FixedSchedule
    >>> schedule = FixedSchedule.uniform(0.0, 1.0, 0.1, save_planes=[0.25, 1.0])
    >>> keys = table_steps(schedule)
    >>> len(set(keys.tolist())), round(float(keys[0]), 12), round(float(keys[-1]), 12)
    (2, 0.083333333333, 0.09375)
    """
    steps, edges = schedule.steps, schedule.edges
    keys = np.empty_like(steps)
    start = 0
    while start < steps.size:
        end = start + 1
        while end < steps.size:
            reach = max(abs(float(edges[start])), abs(float(edges[end + 1])))
            slack = LANDING_ULPS * float(
                max(np.spacing(reach), np.spacing(steps[start]))
            )
            if abs(float(steps[end] - steps[start])) > slack:
                break
            end += 1
        # A run of one keys on its own step, unrounded by the edges' difference.
        keys[start:end] = (
            steps[start]
            if end - start == 1
            else (edges[end] - edges[start]) / (end - start)
        )
        start = end
    return keys


def schedule_key(
    keys: Float[np.ndarray, " n_steps"],
    edges: Float[np.ndarray, " n_edges"],
    z: float,
    dz: float,
) -> float:
    """Return the table key of the schedule step from `z`, from :func:`table_steps`.

    Found by position rather than by a step counter, so that it needs no count of the
    steps taken before.

    Parameters
    ----------
    keys : numpy.ndarray
        :func:`table_steps` of the schedule.
    edges : numpy.ndarray
        The schedule's edges, m.
    z : float
        Where the step starts, m.
    dz : float
        The step, m.

    Returns
    -------
    float
        The key, m.
    """
    index = int(np.searchsorted(edges, z + 0.5 * dz)) - 1
    return float(keys[min(max(index, 0), keys.size - 1)])

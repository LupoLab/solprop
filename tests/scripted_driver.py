# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A step driver whose numerical outcomes are scripted rather than computed.

:class:`solprop.backends.driver.StepDriver` is the marching half of a backend with the
kernels left abstract. :class:`ScriptedDriver` fills them with a stage evaluation that
multiplies by a constant and an error norm taken from a list, so step control, the
session lifecycle and the service loop can be tested without any physics being
evaluated correctly. ``tests/backends/test_driver.py`` tests the step loop with it and
``tests/sessions`` the execution-session lifecycle (SEP 47 section 7.1: "Script the
same numerical outcomes to test this exact lifecycle property on every executor").
"""

import numpy as np

from solprop.backends.driver import StepDriver

SCALE = 0.02 / 8
"""The step length, m, whose default scripted norm is one: an eighth of a 2 cm run."""


class ScriptedDriver(StepDriver):
    """A driver whose stage evaluation multiplies by a constant and scores from a list.

    The state advances by a factor per step so that a rejected step can be seen not to
    have been kept, and the error norms come from a script so that acceptance,
    rejection and the controller's response are exercised without needing a problem
    whose physics happens to produce them.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem whose grid, planes and controller the march uses.
    norms : sequence of float, optional
        The error norms of the first trials, in order; later trials score
        ``(dz / scale) ** 5``.
    factor : float, optional
        What each trial multiplies the state by.
    scale : float, optional
        The step length, m, whose default norm is one.
    **kwargs
        Passed to :class:`~solprop.backends.driver.StepDriver`.
    """

    name = "scripted"

    def __init__(self, problem, *, norms=(), factor=1.0, scale=SCALE, **kwargs):
        super().__init__(problem, **kwargs)
        self._norms = list(norms)
        self._factor = factor
        self._scale = scale
        self.calls: list[tuple[float, float]] = []
        self.proposals: list[float] = []
        self.accepted_hooks = 0
        self.planes_begun = 0

    def _trial_step(self, z, dz):
        self.calls.append((z, dz))
        self.proposals.append(self._proposal)
        # A fifth-order embedded estimate: the norm has to fall like dz^5 or the
        # controller's equivalent_norm reads every short landing step as a failure and
        # the proposal collapses. A constant would test the loop against a stand-in the
        # loop was never written for.
        norm = self._norms.pop(0) if self._norms else (dz / self._scale) ** 5
        state = self._require_state() * self._factor
        return norm, self._apply_mask(state)

    def _apply_mask(self, state):
        return state * self._problem.grids.active_mask

    def _on_accepted_step(self):
        self.accepted_hooks += 1

    def stats(self, spec) -> dict[str, np.ndarray]:
        """Return one scriptable number per name: the position the row was taken at."""
        state = self._require_state()
        return {name: np.full((state.shape[0],), self._z) for name in spec.names}

    def _begin_plane(self):
        self.planes_begun += 1

    def close(self):
        """Release the state, as the seam requires."""
        self._state = None
        self._closed = True

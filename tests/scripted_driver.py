# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A reference backend whose numerical outcomes are scripted rather than computed.

:class:`HostScriptedDriver` is the reference backend with its Runge-Kutta engine
replaced by a scripted engine under the real march and the real session: a trial
multiplies the state by a constant and scores an error norm taken from a list, so step
control, the session lifecycle and the service loop can be tested without any physics
being evaluated correctly. ``tests/sessions`` tests the execution-session lifecycle
with it (SEP 47 section 7.1: "Script the same numerical outcomes to test this exact
lifecycle property on every executor"). Its statistics are scripted too: every name
reads the position it was taken at.
"""

from types import MethodType
from typing import Any

import numpy as np

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.numpy_ref.workspace import NumpyWorkspace

SCALE = 0.02 / 8
"""The step length, m, whose default scripted norm is one: an eighth of a 2 cm run."""


def _scripted_stats(workspace: NumpyWorkspace, spec: Any, z: float) -> dict:
    """Return one scriptable number per name: the position the row was taken at."""
    state = workspace.committed()
    return {name: np.full((state.shape[0],), z) for name in spec.names}


class HostScriptedDriver(NumpyRefBackend):
    """The reference backend over a scripted Runge-Kutta engine.

    The state advances by a factor per step so that a rejected step can be seen not to
    have been kept, and the error norms come from a script so that acceptance,
    rejection and the controller's response are exercised without needing a problem
    whose physics happens to produce them.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem whose grid the march uses.
    run_request : solprop.translate.run_request.RunRequest
        Its run's request, whose planes and controller the march uses.
    norms : sequence of float, optional
        The error norms of the first trials, in order; later trials score
        ``(dz / scale) ** 5``.
    factor : float, optional
        What each trial multiplies the state by.
    scale : float, optional
        The step length, m, whose default norm is one.
    **kwargs
        Passed to the reference backend.
    """

    name = "scripted"

    def __init__(
        self, problem, run_request, *, norms=(), factor=1.0, scale=SCALE, **kwargs
    ):
        self._norms = list(norms)
        self._factor = factor
        self._scale = scale
        self.calls: list[tuple[float, float]] = []
        super().__init__(problem, run_request, **kwargs)
        self._workspace.stats = MethodType(_scripted_stats, self._workspace)  # pyright: ignore[reportAttributeAccessIssue]

    def _trial_step(self, z: float, dz: float) -> tuple[float, np.ndarray]:
        """Return a trial's scripted norm and end state, the state times a factor."""
        self.calls.append((z, dz))
        # A fifth-order embedded estimate: the norm has to fall like dz^5 or the
        # controller's equivalent_norm reads every short landing step as a failure and
        # the proposal collapses. A constant would test the loop against a stand-in the
        # loop was never written for.
        norm = self._norms.pop(0) if self._norms else (dz / self._scale) ** 5
        state = self._workspace.committed() * self._factor
        return norm, state * self._problem.grids.active_mask

    def _runge_kutta(self, workspace: NumpyWorkspace) -> NumpyRungeKutta:
        """Return the scripted engine in place of the reference's."""
        return _ScriptedEngine(workspace, self)


class _ScriptedEngine(NumpyRungeKutta):
    """A Runge-Kutta engine whose trial is the backend's script (``_trial_step``).

    It is in flight from a trial the script left partway until the next engine is
    built, as an in-place step abandoned there is (the commit contract).
    """

    def __init__(self, workspace: NumpyWorkspace, backend: Any) -> None:
        super().__init__(workspace, workspace.stepper.tableau)
        self._backend = backend
        self.left = False

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether the script left a trial in flight."""
        return self.left

    def trial(self, z: float, dz: float) -> float:
        norm, state = self._backend._trial_step(z, dz)
        self._trial = (state, [])
        return norm

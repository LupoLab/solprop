# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A step driver whose numerical outcomes are scripted rather than computed.

:class:`HostScriptedDriver` is the reference backend with its Runge-Kutta engine
replaced by a script: a trial multiplies the state by a constant and scores an error
norm taken from a list, so step control, the session lifecycle and the service loop
can be tested without any physics being evaluated correctly. ``tests/sessions`` tests
the execution-session lifecycle with it (SEP 47 section 7.1: "Script the same
numerical outcomes to test this exact lifecycle property on every executor").
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.numpy_ref.workspace import NumpyWorkspace

SCALE = 0.02 / 8
"""The step length, m, whose default scripted norm is one: an eighth of a 2 cm run."""


class _Scripted:
    """A driver whose stage evaluation multiplies by a constant and scores from a list.

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
        Passed to the step driver.
    """

    name = "scripted"

    if TYPE_CHECKING:
        # What the mixin reads of the driver it is mixed into.
        _problem: Any

        @property
        def _z(self) -> float: ...

        @property
        def _proposal(self) -> float: ...

        _closed: bool
        _workspace: Any

        def _require_state(self) -> Any: ...

    def __init__(
        self, problem, run_request, *, norms=(), factor=1.0, scale=SCALE, **kwargs
    ):
        super().__init__(problem, run_request, **kwargs)  # pyright: ignore[reportCallIssue]
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

    def _runge_kutta(self, workspace: NumpyWorkspace) -> NumpyRungeKutta:
        """Return the scripted engine in place of the reference's."""
        return _ScriptedEngine(workspace, self)

    def close(self):
        """Release the state, as the seam requires."""
        self._workspace.close()
        self._closed = True


class _ScriptedEngine(NumpyRungeKutta):
    """A Runge-Kutta engine whose trial is the driver's script (``_trial_step``)."""

    def __init__(self, workspace: NumpyWorkspace, driver: Any) -> None:
        super().__init__(workspace, workspace.stepper.tableau)
        self._driver = driver

    def trial(self, z: float, dz: float) -> float:
        norm, state = self._driver._trial_step(z, dz)
        self._trial = (state, [])
        return norm


class HostScriptedDriver(_Scripted, NumpyRefBackend):
    """The script on the reference backend's engine."""

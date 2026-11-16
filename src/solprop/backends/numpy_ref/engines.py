# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's Runge-Kutta engine over its workspace (SEP 52 section 5.2).

:class:`NumpyRungeKutta` is the reference's
:class:`~solprop.backends.engines.RungeKuttaEngine`, the counterpart of native's
``engines::ResidentRungeKutta``: it forms an interaction-picture embedded Runge-Kutta
trial from the workspace's committed state and keeps the trial's stages and end state
until :meth:`~NumpyRungeKutta.accept` finishes and commits it. A rejected trial leaves
the committed state as it was. The reference's other engines are the low-storage step
(:class:`solprop.backends.low_storage.HostLowStorage`) and the Adams history
(:class:`solprop.backends.numpy_ref.multistep.HostHistory`).
"""

import numpy as np
from jaxtyping import Complex

from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.workspace import NumpyWorkspace
from solprop.exceptions import BackendError
from solprop.stepper.tableau import Tableau, extension_coefficients

__all__ = ["NumpyRungeKutta"]

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]
_Extension = tuple[float, float, list[np.ndarray]]
"""A continuous extension: its step's start and length, m, and its coefficients."""


class NumpyRungeKutta:
    """The interaction-picture embedded Runge-Kutta trial over a workspace.

    Every stage is evaluated. The tableau's first-same-as-last property would let the
    last stage of one step serve as the first of the next, after rebasing it by
    ``exp(dPhi)`` because the interaction picture is re-anchored every step; the
    reference does not take that saving, so that no stage it uses depends on the
    bookkeeping of the step before (``docs/dev/stepper.md`` section 10, trap 10).

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state is stepped.
    tableau : solprop.stepper.tableau.Tableau
        The embedded pair.

    Attributes
    ----------
    evaluations : int
        Physical source evaluations of the trials since the engine was built, for one
        run, rejected trials included (SEP 30 records work rather than estimating it
        from steps).
    """

    def __init__(self, workspace: NumpyWorkspace, tableau: Tableau) -> None:
        """Hold the workspace; no trial is held yet."""
        self._workspace = workspace
        self._tableau = tableau
        self._trial: tuple[_State, list[_State]] | None = None
        self._extension: tuple[float, float, list[np.ndarray]] | None = None
        self.evaluations = 0

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether a trial overwrote the committed state; never, on this engine."""
        return False

    @property
    def trial_state(self) -> _State:  # numpydoc ignore=RT01
        """The held trial's raw end state, before the accepted step's finish."""
        if self._trial is None:
            raise BackendError("take a trial before reading its state")
        return self._trial[0]

    @property
    def stages(self) -> list[_State]:  # numpydoc ignore=RT01
        """The held trial's stage derivatives, in its anchor's frame."""
        if self._trial is None:
            raise BackendError("take a trial before reading its stages")
        return list(self._trial[1])

    def trial(self, z: float, dz: float) -> float:
        """Form a trial from the committed state at `z` over `dz`; return its norm.

        A problem carrying ``linear.coupled_flow`` multiplies each stage's scalar
        propagator by the section's Schur matrix-function action; the two factors
        commute, so their product is the flow of the sum. A ``"frozen_residual"`` flow
        adds the explicit residual to each stage's lab-frame source, before the
        interaction picture's own factors carry it back, so it is integrated by the
        embedded pair exactly as the material response is.

        Parameters
        ----------
        z : float
            Where the trial starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The scaled error norm of the step's embedded difference.
        """
        workspace = self._workspace
        state = workspace.committed()
        workspace.hoist(z)
        tableau = self._tableau
        # The whole step and every stage, read together: each table once.
        workspace.stage_bundles(
            z, [z + dz, *(z + float(node) * dz for node in tableau.nodes)]
        )
        exponents = workspace.exponents(z, dz)
        step = workspace.stage_scalars(z, z + dz)
        actions = workspace.flow_actions(step.flow_section, dz)
        stages: list[_State] = []
        solution = np.zeros_like(state)
        error = np.zeros_like(state)
        for index, node in enumerate(tableau.nodes):
            fraction = float(node)
            combination = np.zeros_like(state)
            for previous in range(index):
                combination = passes.accumulate(
                    combination,
                    stages[previous],
                    float(tableau.matrix[index, previous]),
                )
            bundle = workspace.stage_scalars(z, z + fraction * dz)
            exponent = exponents(fraction, bundle)
            forward, inverse = actions(fraction)
            lab = passes.stage_input(
                workspace.channels(state, forward),
                workspace.channels(combination, forward),
                dz,
                np.exp(exponent),
                workspace.noise_at(bundle),
            )
            back = np.exp(-exponent)
            self.evaluations += 1
            drive = workspace.nonlinear(lab, bundle, back)
            residual = workspace.residual(lab, step.flow_section, bundle)
            if residual is not None:
                drive = drive + back * residual
            stage = workspace.channels(drive, inverse)
            stages.append(stage)
            solution = passes.accumulate(
                solution, stage, float(tableau.solution_weights[index])
            )
            error = passes.accumulate(error, stage, float(tableau.error_weights[index]))
        propagator = np.exp(exponents(1.0, step))
        whole = actions(1.0)[0]
        advanced = passes.accept_step(
            workspace.channels(state, whole),
            propagator,
            dz,
            workspace.channels(solution, whole),
        )
        # The embedded difference is carried to the end of the step by the same
        # exponential as the solution, so the norm scales the local error of the
        # propagated state rather than of the interaction variable.
        norm = workspace.error_norm(
            propagator * (dz * workspace.channels(error, whole)), state, advanced
        )
        # Kept for the accept, and for the continuous extension of an accepted step
        # (SEP 14 phase 3).
        self._trial = (advanced, stages)
        return norm

    def accept(self, dz: float) -> AcceptRecord:
        """Finish the held trial and commit it as the workspace's state.

        Parameters
        ----------
        dz : float
            The accepted step, m.

        Returns
        -------
        solprop.backends.engines.AcceptRecord
            The finish's split collar's sums.
        """
        finished, record = self._workspace.finish(self.trial_state, dz)
        self._workspace.install(finished)
        return record

    def extend(self, z: float, dz: float) -> None:
        """Hold the held trial's continuous extension, before it is accepted.

        With ``v0`` the state at the step's anchor, ``k_i`` its stage derivatives in
        the anchor's frame and ``v1 = v0 + h sum_i b_i k_i``, the coefficients are
        DOPRI5's (Hairer, Noersett and Wanner, II.6): ``v0``, ``v1 - v0``, ``h k1 - (v1
        - v0)``, ``(v1 - v0) - h k7 - (h k1 - (v1 - v0))`` and ``h sum_i d_i k_i``.

        Parameters
        ----------
        z : float
            Where the trial started, m.
        dz : float
            The step, m.
        """
        workspace = self._workspace
        start = workspace.committed().reshape(workspace.problem.shape)
        coefficients = extension_coefficients(self._tableau, start, self.stages, dz)
        self._extension = (z, dz, coefficients)

    @property
    def extension(self) -> "_Extension | None":  # numpydoc ignore=RT01
        """The held extension: its step's start and length, m, and its coefficients."""
        return self._extension

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The workspace's reading.
        """
        return self._workspace.reading()

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluate reference Runge-Kutta stage arrays across a leading member axis."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from solprop._helpers import PLAIN
from solprop.backends.batch import LockstepRunner, _refuse_invalid_norm
from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.exceptions import BackendError

__all__ = ["NumpyBatchBackend"]


class NumpyBatchBackend(LockstepRunner):
    """Evaluate batch stage arithmetic with independently parameterised responses.

    Stage sums, input construction and the final update operate on arrays with a
    leading member axis. Response evaluation retains each member's own transform,
    material and statistics context. Norms are reduced within each member in the
    same order as the singleton reference.

    Parameters
    ----------
    members : sequence of solprop.backends.numpy_ref.NumpyRefBackend
        Loaded reference workspaces with matching shapes, fixed schedules and
        Runge-Kutta tableaus. Ownership transfers on successful construction.

    Raises
    ------
    BackendError
        If the fixed-schedule contract, backend or tableau differs between members.
    """

    helpers = PLAIN

    def __init__(self, members: Sequence[NumpyRefBackend]) -> None:
        self._reference_members = tuple(members)
        for member in self._reference_members:
            if not isinstance(member, NumpyRefBackend):
                raise BackendError("NumpyBatchBackend requires numpy_ref members")
            # Before the lockstep contract, because this is a statement about the
            # problem rather than about the group: this class reimplements the
            # Runge-Kutta loop over a member axis and does **not** apply the flow
            # action or the explicit residual, so a member carrying
            # `linear.coupled_flow` would run, converge and propagate the diagonal
            # half of an equation whose off-diagonal it silently dropped. A
            # lockstep group is a group of *singleton* members, each with
            # `n_batch == 1`, so the singleton backend's own batch-axis refusal
            # never fires for it.
            if member._problem.linear.coupled_flow is not None:
                msg = (
                    "the numpy_ref lockstep batch does not execute "
                    "linear.coupled_flow: it evaluates its own stage arithmetic "
                    "over the member axis and applies neither the flow action nor "
                    "its explicit residual, so a coupled member would propagate "
                    "only the diagonal half of its own equation. Run each member "
                    "on its own"
                )
                raise BackendError(msg)
        super().__init__([cast(StepDriver[object], member) for member in members])

    def _trial_step(self, z: float, dz: float) -> tuple[float, tuple[object, ...]]:
        """Evaluate vector stage arithmetic with independent member responses.

        Parameters
        ----------
        z : float
            Start position, m.
        dz : float
            Positive fixed step length, m.

        Returns
        -------
        tuple
            Maximum diagnostic norm and member trial states.
        """
        members = self._reference_members
        state = np.concatenate([member._require_state() for member in members])
        tableau = self._problem.stepper.tableau
        positions = [z + dz, *(z + float(node) * dz for node in tableau.nodes)]
        for member in members:
            member._hoist(z)
            member._stage_bundles(z, positions)
        exponents = [member._exponents(z, dz) for member in members]
        stages = []
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
            bundles = [
                member._stage_scalars(z, z + fraction * dz) for member in members
            ]
            exponent = np.concatenate(
                [
                    function(fraction, bundle)
                    for function, bundle in zip(exponents, bundles, strict=True)
                ]
            )
            noises = [
                member._noise_at(bundle)
                for member, bundle in zip(members, bundles, strict=True)
            ]
            if any(noise is None for noise in noises) and not all(
                noise is None for noise in noises
            ):
                raise BackendError(
                    "reference batch members require compatible noise models"
                )
            noise = (
                None
                if noises[0] is None
                else np.concatenate([one for one in noises if one is not None])
            )
            lab = passes.stage_input(state, combination, dz, np.exp(exponent), noise)
            inverse = np.exp(-exponent)
            stage = np.concatenate(
                [
                    member._nonlinear(lab[i : i + 1], bundles[i], inverse[i : i + 1])
                    for i, member in enumerate(members)
                ]
            )
            stages.append(stage)
            solution = passes.accumulate(
                solution, stage, float(tableau.solution_weights[index])
            )
            error = passes.accumulate(error, stage, float(tableau.error_weights[index]))
        propagator = np.exp(
            np.concatenate(
                [
                    function(1.0, member._stage_scalars(z, z + dz))
                    for function, member in zip(exponents, members, strict=True)
                ]
            )
        )
        advanced = passes.accept_step(state, propagator, dz, solution)
        residual = propagator * (dz * error)
        self._member_norms = tuple(
            passes.error_norm(
                residual[i : i + 1],
                state[i : i + 1],
                advanced[i : i + 1],
                member._problem.grids.norm_mask,
                member._problem.stepper.controller,
            )
            for i, member in enumerate(members)
        )
        _refuse_invalid_norm(self._member_norms, "reference")
        return max(self._member_norms), tuple(
            advanced[i : i + 1] for i in range(len(members))
        )

    def metadata(self) -> dict[str, object]:
        """Record vector stage arithmetic and independent response workspaces.

        Returns
        -------
        dict
            Batch execution settings; no device-kernel fusion is claimed.
        """
        return super().metadata() | {"batch_strategy": "numpy_stage_arrays"}

    def memory_ledger(self) -> dict[str, int]:
        """Include transient array packing alongside the reference step peak.

        Returns
        -------
        dict of str to int
            Conservative peak workspace bytes. Existing member ledgers account
            for stage stacks and accumulators; these entries cover concatenation
            inputs, outputs and expression temporaries that coexist with them.
        """
        if self._closed:
            return {}
        size = self.max_batch * self._problem.initial.field0.nbytes
        return super().memory_ledger() | {
            f"batch/{name}": size
            for name in (
                "packed_state",
                "combination",
                "exponent",
                "lab",
                "inverse",
                "packed_noise",
                "advanced",
                "residual",
                "concatenation_input",
                "accumulator_temporary",
                "exponential_temporary",
                "update_temporary",
            )
        }

    def close(self) -> None:
        """Release the reference member workspaces and common march state."""
        try:
            super().close()
        finally:
            self._reference_members = ()

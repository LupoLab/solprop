# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute CPU member stages with one compiled batch update and segmented norms."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from solprop._helpers import NUMBA
from solprop.backends.batch import LockstepRunner, _refuse_invalid_norm
from solprop.backends.driver import StepDriver
from solprop.backends.numba_cpu import kernels
from solprop.backends.numba_cpu.backend import NumbaCpuBackend
from solprop.exceptions import BackendError

__all__ = ["NumbaCpuBatchBackend"]


class NumbaCpuBatchBackend(LockstepRunner):
    """Run fixed CPU batches with contiguous member update buffers.

    Each member uses its own compiled response, FFT and BLAS policies. Stage
    evaluation is interleaved; the accepted update traverses every member's columns
    in one existing Numba kernel. Error partials are segmented before the unchanged
    singleton reduction, preserving operation order and member error histories.

    Parameters
    ----------
    members : sequence of solprop.backends.numba_cpu.NumbaCpuBackend
        Loaded fibre workspaces with the same field layout, frequency mask and
        tableau. Material parameters and initial fields may differ.

    Raises
    ------
    BackendError
        If a member is not a compatible Numba CPU fibre problem.
    """

    helpers = NUMBA

    def __init__(self, members: Sequence[NumbaCpuBackend]) -> None:
        self._cpu_members = tuple(members)
        # Refused exactly as the reference's lockstep batch refuses it, and for this
        # class's own reason: the accept pass below is one fused traversal of every
        # member's columns through the diagonal propagator, which applies no flow
        # action, so a coupled member would advance the diagonal half of its own
        # equation alone.
        for member in self._cpu_members:
            if (
                isinstance(member, NumbaCpuBackend)
                and member._problem.linear.coupled_flow is not None
            ):
                msg = (
                    "the numba_cpu lockstep batch does not execute "
                    "linear.coupled_flow: its accept pass is one traversal of every "
                    "member through the diagonal propagator and applies no flow "
                    "action, so a coupled member would propagate only the diagonal "
                    "half of its own equation. Run each member on its own"
                )
                raise BackendError(msg)
        super().__init__([cast(StepDriver[object], member) for member in members])
        if any(
            not isinstance(member, NumbaCpuBackend)
            or member._problem.transform.geometry is not None
            for member in self._cpu_members
        ):
            raise BackendError("NumbaCpuBatchBackend requires Numba CPU fibre members")
        columns = self._cpu_members[0]._n_columns
        omega = self._problem.shape[-1]
        shape = (len(members) * columns, omega)
        self._batch_current = np.empty(shape, dtype=np.complex128)
        self._batch_propagator = np.empty(shape, dtype=np.complex128)
        self._batch_solution = np.empty(shape, dtype=np.complex128)
        self._batch_error = np.empty(shape, dtype=np.complex128)
        self._batch_advanced = np.empty(shape, dtype=np.complex128)
        self._batch_partials = np.empty((3, shape[0]), dtype=np.float64)
        # Every member's columns in one traversal, threaded when there are enough.
        self._accept, _ = kernels.accept_kernels(
            self._cpu_members[0]._pools.numba, *shape
        )
        self._member_columns = tuple(
            slice(index * columns, (index + 1) * columns)
            for index in range(len(members))
        )
        for member, rows in zip(self._cpu_members, self._member_columns, strict=True):
            member._solution = self._batch_solution[rows]
            member._error = self._batch_error[rows]
            member._advanced = self._batch_advanced[rows]
            member._partials = self._batch_partials[:, rows]

    def _trial_step(self, z: float, dz: float) -> tuple[float, tuple[object, ...]]:
        """Evaluate member stages and fuse their accepted-update traversal.

        Parameters
        ----------
        z : float
            Start position, m.
        dz : float
            Positive fixed step, m.

        Returns
        -------
        tuple
            Maximum diagnostic norm and owning member states before boundary handling.
        """
        members = self._cpu_members
        prepared = tuple(member._start_trial(z, dz) for member in members)
        tableau = self._problem.stepper.tableau
        for index, node in enumerate(tableau.nodes):
            for member, (state, _, first) in zip(members, prepared, strict=True):
                if index >= first:
                    member._stage(
                        state, index, dz, member._stage_scalars(z, z + float(node) * dz)
                    )
        for member, rows, (state, whole, _) in zip(
            members, self._member_columns, prepared, strict=True
        ):
            np.copyto(self._batch_current[rows], state)
            np.copyto(self._batch_propagator[rows], member._propagator(1.0, whole))
        self._accept(
            self._batch_current,
            self._batch_propagator,
            dz,
            self._batch_solution,
            self._batch_error,
            self._problem.grids.norm_mask,
            self._batch_advanced,
            self._batch_partials,
        )
        self._member_norms = tuple(
            member._score_trial(whole, dz)
            for member, (_, whole, _) in zip(members, prepared, strict=True)
        )
        _refuse_invalid_norm(self._member_norms, "Numba CPU")
        advanced = self._batch_advanced.copy()
        return max(self._member_norms), tuple(
            advanced[rows].reshape(member._problem.shape)
            for member, rows in zip(members, self._member_columns, strict=True)
        )

    def metadata(self) -> dict[str, object]:
        """Record compiled update fusion while retaining each member's policies.

        Returns
        -------
        dict
            Batch strategy and the specific compiled operation shared across members.
        """
        return super().metadata() | {
            "batch_strategy": "numba_stage_workspaces",
            "batch_fused_operations": ("accept_and_partials",),
        }

    def memory_ledger(self) -> dict[str, int]:
        """Count member slices once and include the two additional packing buffers.

        Returns
        -------
        dict of str to int
            Workspace bytes. Member update and partial views partition their backing
            arrays, so summing their ledgers already counts those allocations once.
        """
        if self._closed:
            return {}
        return super().memory_ledger() | {
            "batch/current": self._batch_current.nbytes,
            "batch/propagator": self._batch_propagator.nbytes,
        }

    def close(self) -> None:
        """Release member owners and all packed CPU workspace arrays."""
        try:
            super().close()
        finally:
            self._cpu_members = ()
            empty = np.empty((0, 0), dtype=np.complex128)
            self._batch_current = empty
            self._batch_propagator = empty
            self._batch_solution = empty
            self._batch_error = empty
            self._batch_advanced = empty
            self._batch_partials = np.empty((0, 0), dtype=np.float64)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expose Kokkos member ownership through solprop's versioned batch seam."""

import warnings
from collections.abc import Sequence
from typing import cast

import numpy as np
from solprop.backends.batch import LockstepRunner
from solprop.backends.driver import StepDriver
from solprop.exceptions import BackendError, HostMarchWarning

from . import _core, check_build
from ._helpers import NATIVE
from .backend import (
    NativeBackend,
    _NativeAdamsMarch,  # pyright: ignore[reportPrivateUsage] - the members' march
)

__all__ = ["NativeBatchBackend"]


class NativeBatchBackend(LockstepRunner):
    """Advance compatible resident Kokkos members under one fixed march.

    The current implementation retains independent native workspaces. It preserves
    per-member parameters, noise and statistics and fuses the accepted update into
    one native launch. Response and FFT/BLAS work is still issued per member.

    Parameters
    ----------
    members : sequence of NativeBackend
        Freshly loaded singleton owners with compatible fixed schedules and layouts.
        Successful construction transfers their lifetime to the batch.

    Raises
    ------
    BackendError
        If the installed seam or one member is incompatible.
    """

    helpers = NATIVE

    def __init__(self, members: Sequence[NativeBackend]) -> None:
        check_build()
        if any(
            not isinstance(cast(object, member), NativeBackend) for member in members
        ):
            raise BackendError("NativeBatchBackend requires Kokkos member owners")
        if any(
            member._problem.linear.coupled_flow is not None  # pyright: ignore[reportPrivateUsage] - the member's own validated problem
            for member in members
        ):
            # As solprop's own lockstep batches: the fused accepted update carries
            # no flow action, so a batch of flows would propagate the diagonal alone.
            raise BackendError(
                "NativeBatchBackend does not execute linear.coupled_flow: the fused "
                "accepted update applies no flow action, so each coupled-flow member "
                "runs on its own NativeBackend"
            )
        super().__init__([cast(StepDriver[object], member) for member in members])
        # The batch replays its members in lockstep through each march's own
        # `replay_step`, solprop's march (SEP 39 Q14): the members say so, and the run
        # is warned once.
        marches = [
            march
            for member in members
            if isinstance(march := member._adams, _NativeAdamsMarch) and march.native  # pyright: ignore[reportPrivateUsage] - the member's own march
        ]
        for march in marches:
            march.refusal = "a lockstep batch's replay"
        if marches:
            warnings.warn(
                "solprop-native replays this batch's Adams members in lockstep with "
                "solprop's Python march: the native march (SEP 39) steps single runs",
                HostMarchWarning,
                stacklevel=2,
            )

        self._native_members: tuple[NativeBackend, ...] = tuple(members)
        self._batch_native = _core.BatchState(
            [member._native_state for member in members]  # pyright: ignore[reportPrivateUsage] - batch owns these implementation states
        )

    def _trial_step(self, z: float, dz: float) -> tuple[float, tuple[object, ...]]:
        """Evaluate member stages and one native update without merging their norms."""
        for member in self._native_members:
            member._prepare_trial(z, dz, defer_update=True)  # pyright: ignore[reportPrivateUsage] - split internal trial around the fused update
        partials = self._batch_native.accept(dz)
        self._member_norms = tuple(
            member._score_trial(partials[3 * index : 3 * (index + 1)])  # pyright: ignore[reportPrivateUsage] - reuse the singleton operation order
            for index, member in enumerate(self._native_members)
        )
        # `StepHistory` and `StepSummary` refuse a non-finite or negative norm on
        # every route, so the batch owes the reader the member's index rather than
        # an anonymous history error one save plane later.
        for index, norm in enumerate(self._member_norms):
            if not (np.isfinite(norm) and norm >= 0.0):
                raise BackendError(
                    f"Kokkos batch member {index} returned an invalid error norm "
                    f"{norm!r}, which its own step record also refuses"
                )
        return max(self._member_norms), tuple(
            member._native_state  # pyright: ignore[reportPrivateUsage] - return owned states to the shared march
            for member in self._native_members
        )

    def metadata(self) -> dict[str, object]:
        """Record native update fusion separately from member response launches.

        Returns
        -------
        dict
            Shared execution settings and the fused operation name.
        """
        return super().metadata() | {
            "batch_strategy": "native_member_descriptors",
            "batch_fused_operations": ("accept_error_partials",),
            "batch_execution_sizing": self._batch_native.execution_sizing,
        }

    def memory_ledger(self) -> dict[str, int]:
        """Count native member allocations once and add the small descriptor arrays.

        Returns
        -------
        dict
            Owned native allocations in bytes; empty after close.
        """
        return super().memory_ledger() | {
            f"batch/{name}": size
            for name, size in self._batch_native.memory_ledger().items()
        }

    def close(self) -> None:
        """Drop borrowed descriptors before releasing member states and FFT plans."""
        try:
            self._batch_native.close()
        finally:
            super().close()
            self._native_members = ()

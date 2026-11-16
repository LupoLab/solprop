# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single trials of a loaded native run, committing nothing (solprop SEP 52 5.5).

``Simulation.estimate`` and the SEP 46 calibration time one step's arithmetic from a
run's start (:meth:`solprop.backends.protocol.Backend.timing_probe`). The probe here
wraps the workspace's own (``_core.Workspace.probe``): a Runge-Kutta trial on the
stage tables, which leaves the loaded state, position and history alone, and a
readback of its end state for the stage-2 refinement's plan comparison; or, for the
low-storage march (SEP 24), one step of the resident registers, which accepts in place,
whose field is refused and whose workspace closes with the probe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from solprop.exceptions import BackendError, ConvergenceError

from . import _core
from ._load import Loaded

if TYPE_CHECKING:
    from .backend import NativeBackend

__all__ = ["NativeProbe"]


class NativeProbe:
    """The native backend's :class:`~solprop.backends.protocol.TimingProbe`."""

    def __init__(
        self, backend: NativeBackend, probe: _core.Probe, loaded: Loaded
    ) -> None:
        """Hold the workspace's probe over the loaded start; no trial is taken yet."""
        self._backend = backend
        self._probe = probe
        self._z = float(loaded.z)
        self._cursor = int(loaded.cursor)

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        - ``z``: where the trial starts, m; the loaded position by default.
        - ``dz``: its length, m, positive; the problem's ``dz_init`` by default, or the
          scheduled step at the loaded position for the low-storage march.
        """
        backend = self._backend
        stepper = backend._stepper  # pyright: ignore[reportPrivateUsage] - the owner's policy
        start = self._z if z is None else float(z)
        if self._probe.low_storage and dz is None:
            schedule = stepper.schedule
            assert schedule is not None  # the translation refuses the march without one
            length = float(schedule.steps[min(self._cursor, schedule.n_steps - 1)])
        else:
            length = float(stepper.dz_init) if dz is None else float(dz)
        try:
            self._probe.trial(start, length)
        except _core.EnvelopeConvergenceError as error:
            raise ConvergenceError(str(error)) from error
        except _core.BackendRefusal as error:
            raise BackendError(str(error)) from error
        if self._probe.low_storage:
            backend._low_storage_evaluations = int(self._probe.evaluations)  # pyright: ignore[reportPrivateUsage]

    def field(self) -> npt.NDArray[np.complex128]:
        """Return the last trial's end state on the host, as a kept step would be.

        The trial lives in the workspace, which only finishing the step there (the
        band mask, and the split-step absorber where it runs) can read back: the
        loaded state is kept first and loaded again after, exactly, so that the probe
        still commits nothing.

        - Returns: an owning copy, internal units and the FFT convention, shape
          ``(n_modes, n_pol, n_omega)``.
        """
        try:
            field = self._probe.field()
        except _core.BackendRefusal as error:
            raise BackendError(str(error)) from error
        return field.reshape(self._backend._problem.shape)  # pyright: ignore[reportPrivateUsage]

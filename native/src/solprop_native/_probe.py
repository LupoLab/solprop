# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single trials of a loaded native run, committing nothing (solprop SEP 47 6.6).

``Simulation.estimate`` and the SEP 46 calibration time one step's arithmetic from a
run's start (:meth:`solprop.backends.protocol.Backend.timing_probe`). The probe here is
the native backend's: a Runge-Kutta trial on the staged rows (``rk_trial``, solprop
SEP 45), which leaves the loaded state, position and history alone, and a readback of
its end state for the stage-2 refinement's plan comparison; or, for the low-storage
march (SEP 24), one step of the resident registers, which accepts in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from solprop.exceptions import BackendError
from solprop.noise import NoiseRealisation
from solprop.stepper.low_storage import LowStorageRungeKutta

from ._load import Loaded

if TYPE_CHECKING:
    from .backend import NativeBackend

__all__ = ["NativeProbe"]


class NativeProbe:
    """The native backend's :class:`~solprop.backends.protocol.TimingProbe`."""

    def __init__(
        self,
        backend: NativeBackend,
        loaded: Loaded,
        noise: NoiseRealisation | None,
    ) -> None:
        """Hold the loaded backend; no trial is taken yet."""
        self._backend = backend
        self._loaded = loaded
        self._noise = noise
        self._length: float | None = None

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        - ``z``: where the trial starts, m; the loaded position by default.
        - ``dz``: its length, m, positive; the problem's ``dz_init`` by default, or the
          scheduled step at the loaded position for the low-storage march.
        """
        backend = self._backend
        start = self._loaded.z if z is None else float(z)
        stepper = backend._stepper  # pyright: ignore[reportPrivateUsage] - the owner's policy
        engine = backend._resident_low_storage  # pyright: ignore[reportPrivateUsage]
        if engine is not None:
            schedule = stepper.schedule
            assert schedule is not None  # the translation refuses the march without one
            index = min(self._loaded.cursor, schedule.n_steps - 1)
            length = float(schedule.steps[index]) if dz is None else float(dz)
            engine.step(start, length)
            method = stepper.integrator
            assert isinstance(method, LowStorageRungeKutta)
            backend._low_storage_evaluations += method.evaluations  # pyright: ignore[reportPrivateUsage]
            return
        length = float(stepper.dz_init) if dz is None else float(dz)
        backend._trial_step(start, length)  # pyright: ignore[reportPrivateUsage]
        self._length = length

    def field(self) -> npt.NDArray[np.complex128]:
        """Return the last trial's end state on the host, as a kept step would be.

        The trial lives in the native state, which only finishing the step there (the
        band mask, and the split-step absorber where it runs) can read back: the loaded
        state is kept first and loaded again after, exactly, so that the probe still
        commits nothing.

        - Returns: an owning copy, internal units and the FFT convention, shape
          ``(n_modes, n_pol, n_omega)``.
        """
        backend = self._backend
        if backend._resident_low_storage is not None:  # pyright: ignore[reportPrivateUsage]
            msg = (
                "the low-storage march accepts its step in place and has no trial "
                "field to read back"
            )
            raise BackendError(msg)
        if self._length is None:
            raise BackendError("take a trial before reading its field")
        native = backend._native_state  # pyright: ignore[reportPrivateUsage]
        kept = np.array(backend._observed_field(native))  # pyright: ignore[reportPrivateUsage]
        finished = backend._finish(native, self._length)  # pyright: ignore[reportPrivateUsage]
        field = np.array(backend._observed_field(finished)[0])  # pyright: ignore[reportPrivateUsage]
        # The kept state already carries any `input` draw: a resume adds none.
        backend._load(kept[0], self._noise, z=self._loaded.z, resume=True)  # pyright: ignore[reportPrivateUsage]
        self._length = None
        return field

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own discarded Julia trials without constructing a propagation session."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Complex

from solprop.backends.julia import _calls
from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._fft_cache import publish_fft_cache, refresh_fft_cache
from solprop.backends.julia._fft_runtime import read_planning
from solprop.backends.julia._noise import lower_input_noise
from solprop.backends.julia._ownership import JuliaHandle, require_main_thread
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.session import SessionError, Start
from solprop.exceptions import BackendError

if TYPE_CHECKING:
    from solprop.backends.julia.backend import JuliaPrepared

__all__ = ["JuliaTimingProbe"]


class JuliaTimingProbe:
    """Retain a loaded Julia trial engine and its immutable preparation.

    Parameters
    ----------
    prepared : JuliaPrepared
        Admitted numerical preparation, retained until this probe is closed.
    start : solprop.backends.session.Start
        Initial complex128 field and position, consumed only after construction.
        Monitor, controller and march continuations are deliberately unused.

    Raises
    ------
    BackendError
        For an unsupported or invalid numerical start.
    SessionError
        If the start was already consumed or the execution owner is closed.
    """

    def __init__(self, prepared: JuliaPrepared, start: Start) -> None:
        """Copy the initial state into an independently owned trial workspace."""
        prepared._owner._check()
        field = start.field
        if field is None:
            raise SessionError("this start was already used")
        if (
            field.shape != prepared._shape
            or field.dtype != np.complex128
            or not np.all(np.isfinite(field))
            or not math.isfinite(start.z)
        ):
            raise BackendError("julia requires a finite complex128 starting field")
        self._prepared: JuliaPrepared | None = prepared
        self._runtime = prepared._runtime
        self._shape = prepared._shape
        self._engine: JuliaHandle | None = None
        self._create(start)

    def _create(self, start: Start) -> None:
        """Finish numerical construction before adopting the start.

        Parameters
        ----------
        start : Start
            Validated and unconsumed initial state.
        """
        prepared = self._prepared
        assert prepared is not None
        cache = prepared._owner._fft_cache
        if cache is not None:
            cache = refresh_fft_cache(cache)
        planning: JuliaHandle | None = None
        field = start.field
        try:
            if cache is not None:
                planning = self._runtime.own_call("prepare_fft_planning", cache)
            self._engine = self._runtime.own_call(
                "start_trial_probe",
                prepared._preparation,
                prepared._transfer,
                start.field,
                float(start.z),
                lower_input_noise(prepared._problem, noise=start.noise),
                bool(start.resume),
                planning,
                owners=(prepared,),
            )
            record = (
                None if planning is None else read_planning(self._runtime, planning)
            )
            if planning is not None:
                planning.close()
                planning = None
            start.take_field()
            if cache is not None and record is not None:
                publish_fft_cache(cache, dict(record.measured))
        except BaseException as failure:
            start.field = field
            try:
                self.close()
            except BaseException as cleanup:
                failure.add_note(f"Julia timing-probe cleanup also failed: {cleanup}")
            if planning is not None:
                try:
                    planning.close()
                except BaseException as cleanup:
                    failure.add_note(f"Julia planning cleanup also failed: {cleanup}")
                finally:
                    planning = None
            raise
        finally:
            if planning is not None:
                planning.close()

    def _check(self) -> JuliaHandle:
        """Require a live probe on the runtime's main thread.

        Returns
        -------
        JuliaHandle
            The retained numerical owner.

        Raises
        ------
        SessionError
            If this probe has been closed.
        """
        require_main_thread()
        if self._engine is None:
            raise SessionError("the Julia timing probe is closed")
        return self._engine

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one discarded trial in Julia, without advancing a session.

        Parameters
        ----------
        z : float or None, optional
            Trial position in metres; the loaded position by default.
        dz : float or None, optional
            Positive trial length in metres; the embedded initial step or loaded
            low-storage schedule step by default.
        """
        engine = self._check()
        _calls.call_void(
            self._runtime,
            "trial_probe!",
            engine,
            None if z is None else float(z),
            None if dz is None else float(dz),
        )

    def field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Copy the last unread candidate after its accepted-step treatments.

        Returns
        -------
        numpy.ndarray
            Owning complex128 array, shape ``(n_modes, n_pol, n_omega)``, in internal
            units and the FFT convention. Later trials cannot change this copy.

        Raises
        ------
        BackendError
            Without an unread trial, after numerical failure, or for low storage,
            whose discarded register updates provide no readable trial field.
        SessionError
            If this probe has been closed.
        """
        engine = self._check()
        output = np.empty(self._shape, dtype=np.complex128)
        _calls.call_void(self._runtime, "copy_trial_probe_field!", output, engine)
        return output

    def memory_ledger(self) -> dict[str, int]:
        """Return every owned numerical buffer, excluding shared preparation.

        Returns
        -------
        dict
            Detached buffer names and byte counts; empty after closure.
        """
        require_main_thread()
        if self._engine is None:
            return {}
        self._runtime.check()
        return invoke(
            lambda: _memory(self._runtime, self._check()),
            "the julia timing-probe memory read failed",
        )

    def close(self) -> None:
        """Close the numerical owner on main before releasing its preparation."""
        require_main_thread()
        engine = self._engine
        if engine is None:
            self._prepared = None
            return
        try:
            _calls.call_void(self._runtime, "close_trial_probe!", engine)
        finally:
            engine.close()
            self._engine = None
            self._prepared = None


def _memory(runtime: JuliaRuntime, handle: JuliaHandle) -> dict[str, int]:
    """Detach a numerical ledger without retaining Julia wrappers.

    Parameters
    ----------
    runtime : JuliaRuntime
        Runtime inside the guarded main-thread bridge.
    handle : JuliaHandle
        Retained timing probe.

    Returns
    -------
    dict
        Owned buffer names and byte counts.
    """
    result = runtime.engine.trial_probe_memory(handle.borrow())
    names = runtime.namespace.seval("propertynames")(result)
    return {str(name): int(getattr(result, str(name))) for name in names}

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connect resident Julia integrators to the existing Python march policies."""

import math
from collections.abc import Mapping

import numpy as np

from solprop._helpers import PLAIN
from solprop.backends.julia._backend import JuliaBackend
from solprop.backends.julia._runtime import call_julia
from solprop.backends.low_storage import schedule_key, table_steps
from solprop.backends.multistep import compensation_tables, exponential_tables
from solprop.exceptions import BackendError
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.translate.linop import rate_weights


class JuliaRK4:
    """Own the small RK4IP policy while Julia holds all field registers."""

    atomic = True

    def __init__(self, backend: JuliaBackend, method: LowStorageRungeKutta) -> None:
        self.backend = backend
        self.method = method
        self.stiffness = math.nan
        schedule = backend._problem.stepper.schedule
        assert schedule is not None
        self.keys = table_steps(schedule)
        self.edges = schedule.edges

    def step(self, z: float, dz: float) -> float:
        """Execute the four resident source passes and return the state norm."""
        backend = self.backend
        key = schedule_key(self.keys, self.edges, z, dz)
        squared, self.stiffness = call_julia(
            backend._jl.rk4_b, backend._engine, dz, key, nogil=True
        )
        return float(squared)

    def reset(self) -> None:
        """Reset diagnostics after loading an initial state."""
        self.stiffness = math.nan

    def close(self) -> None:
        """Release no arrays: the backend owns the resident registers."""
        self.reset()


class JuliaAdams:
    """Keep exponential PEC samples resident and exchange only scalar trial tables."""

    dense = None

    def __init__(self, backend: JuliaBackend, method: Multistep) -> None:
        self.backend = backend
        self.method = method
        assert backend._engine is not None
        self.compensated = bool(backend._engine.work.adams.compensated)
        self._dz = 0.0

    @property
    def n_slots(self) -> int:
        """Return the shared march's required history capacity."""
        return self.method.order + 2

    def reset(self) -> None:
        """Forget the pending trial; startup replaces the first resident row."""
        self._dz = 0.0

    def start(self, z: float, *, stored: bool) -> None:
        """Evaluate the accepted initial state's first source sample."""
        del stored
        backend = self.backend
        if backend._varying:
            backend._prepare_stages(z, 0.0)
        if self.compensated:
            call_julia(
                backend._jl.configure_adams_rate_b,
                backend._engine,
                rate_weights(backend._problem.linear, z, helpers=PLAIN),
            )
        call_julia(backend._jl.adams_start_b, backend._engine, nogil=True)

    def value(self, slot: int) -> None:
        """Keep history rows opaque to the march."""
        del slot

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        """Copy scalar tables, then run the resident trial without the GIL."""
        backend = self.backend
        if backend._varying:
            backend._prepare_stages(z, dz)
        predictor, corrector, defects = exponential_tables(
            nodes, order, orders, helpers=PLAIN
        )
        spans = carried = endpoint = corrections = rate = np.empty(0)
        if self.compensated:
            spans, carried, endpoint, corrections = compensation_tables(
                nodes, order, orders, helpers=PLAIN
            )
            rate = rate_weights(
                backend._problem.linear, z + dz, before=True, helpers=PLAIN
            )
        call_julia(
            backend._jl.configure_adams_b,
            backend._engine,
            predictor,
            corrector,
            defects,
            slots[: corrector.shape[0] - 1],
            spans,
            carried,
            endpoint,
            corrections,
            rate,
        )
        result = call_julia(backend._jl.adams_trial_b, backend._engine, dz, nogil=True)
        self._dz = dz
        return np.asarray(result, dtype=float).copy()

    def exponential_accept(
        self, keep: np.ndarray, free: int, *, interpolate: bool
    ) -> None:
        """Save the endpoint sample and commit the accepted candidate."""
        del keep
        if interpolate:
            raise BackendError("julia does not support interpolated Adams planes")
        backend = self.backend
        call_julia(backend._jl.adams_accept_b, backend._engine, free, nogil=True)
        backend._state = backend._finish(backend._engine, self._dz)

    def snapshot_rows(self) -> dict[str, np.ndarray]:
        """Copy owning history rows into the existing resume layout."""
        backend = self.backend
        names = ("store", "states", "rates") if self.compensated else ("store",)
        shape = (self.n_slots, *backend._problem.shape)
        return {
            name: np.asarray(call_julia(backend._jl.adams_rows, backend._engine, name))
            .transpose(2, 1, 0)
            .copy()
            .reshape(shape)
            for name in names
        }

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Validate all snapshot rows before replacing the resident history."""
        names = {"store", "states", "rates"} if self.compensated else {"store"}
        expected = (self.n_slots, *shape)
        if set(rows) != names or any(
            row.shape != expected or row.dtype != np.complex128 for row in rows.values()
        ):
            raise BackendError("julia Adams history does not match the snapshot")
        backend = self.backend
        for name, row in rows.items():
            call_julia(
                backend._jl.adams_load_rows_b,
                backend._engine,
                name,
                row[:, 0, :, 0, :].transpose(2, 1, 0).copy(),
            )
        self.reset()

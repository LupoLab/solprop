# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor: guard and collar occupancy, and the energy the collar removes.

SEP 21 decisions D5 and D7. At the launch and at every committed step the monitor
reads the field's time-domain energy density, ``P(t)``, summed over modes,
polarisations and transverse columns (with their quadrature weights), and forms:

- the **guard occupancy** of each edge, the share of ``P`` within the outer
  :attr:`BoundaryCheck.guard_fraction` of the realised window: what has reached the
  window's edge and is about to wrap or be absorbed;
- the **collar occupancy**, the share of ``P`` weighted by ``1 - W`` of the temporal
  collar;
- the **removal rate** ``int alpha(t) P(t) dt`` of the temporal collar, integrated
  along z by the trapezoid over committed steps into the removed fraction of the
  launch energy.

``P`` is the energy-density proxy of SEP 21 section 4.6: exact for modal true-power
amplitudes, and on spatial routes the quadrature-weighted field norm rather than the
admittance-weighted flux. The quantities are relative, so the proxy's scale cancels.

Policy (D7): the square root of a guard occupancy above
:attr:`BoundaryCheck.budget` warns, and above ``stop_factor`` times it stops the run;
at the launch an occupancy above :attr:`BoundaryCheck.launch_threshold` warns. At the
end of a run the removed fraction is reported, and warns when its square root exceeds
the budget (D5).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.fourier import GridKind
from solprop.grid import TimeGrid

__all__ = ["BoundaryCheck", "BoundaryMonitor"]

_FLOOR = 1e-12
"""A state holding less than this share of the launch energy is not judged: its
occupancies are of rounding residue, not of light (a zero launch, or one absorbed)."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundaryCheck:
    """How a run watches its time window's edges (SEP 21, D5 and D7).

    Attributes
    ----------
    guard_fraction : float
        Width of each edge's guard, as a fraction of the realised window, in (0, 0.5).
        Default 0.05.
    budget : float
        The run's field-error budget: the square root of a guard occupancy, or of the
        removed fraction, above it warns. Default 1e-3.
    stop_factor : float
        The run stops when the square root of a guard occupancy exceeds this many
        budgets. Default 10.
    launch_threshold : float
        The launch warns when its guard or collar occupancy exceeds this. Default
        1e-4.

    Examples
    --------
    >>> from solprop.monitor import BoundaryCheck
    >>> check = BoundaryCheck()
    >>> check.guard_fraction, check.budget, check.stop_factor
    (0.05, 0.001, 10.0)
    """

    guard_fraction: float = 0.05
    budget: float = 1e-3
    stop_factor: float = 10.0
    launch_threshold: float = 1e-4

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises
        ------
        BoundaryError
            Naming the offending field.
        """
        if not 0.0 < self.guard_fraction < 0.5:
            msg = f"guard_fraction must lie in (0, 0.5), got {self.guard_fraction!r}"
            raise BoundaryError(msg)
        for name in ("budget", "stop_factor", "launch_threshold"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                msg = f"{name} must be finite and positive, got {value!r}"
                raise BoundaryError(msg)
        if self.stop_factor < 1.0:
            msg = f"stop_factor must be at least 1, got {self.stop_factor!r}"
            raise BoundaryError(msg)


@dataclass
class BoundaryMonitor:
    """The monitor of one run: its weights, its running totals and its record.

    Attributes
    ----------
    check : BoundaryCheck
        The thresholds.
    weights : numpy.ndarray
        Time weights on the coarse axis, shape ``(4, n_t)``: the low guard, the high
        guard, the collar's ``1 - W`` and the temporal collar rate in 1/m.
    kind : {'field', 'envelope'}
        The grid's spectral layout.
    columns : numpy.ndarray or None
        Quadrature weights of the transverse columns, or ``None`` for modal routes.
    """

    check: BoundaryCheck
    weights: Float[np.ndarray, "4 n_t"]
    kind: GridKind
    columns: Float[np.ndarray, " n_columns"] | None = None
    _launch: float = field(default=math.nan, init=False)
    _last: tuple[float, float] | None = field(default=None, init=False)
    _removed: float = field(default=0.0, init=False)
    _guard_max: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _guard_at: list[float] = field(
        default_factory=lambda: [math.nan, math.nan], init=False
    )
    _collar_max: float = field(default=0.0, init=False)
    _launch_occupancy: tuple[float, float, float] = field(
        default=(math.nan, math.nan, math.nan), init=False
    )
    _warned: bool = field(default=False, init=False)
    _steps: int = field(default=0, init=False)

    @classmethod
    def build(
        cls,
        grid: TimeGrid,
        temporal_rate: Float[np.ndarray, " n_t"] | None,
        check: BoundaryCheck,
        *,
        columns: Float[np.ndarray, " n_columns"] | None = None,
    ) -> BoundaryMonitor:
        """Return a monitor for a grid and its temporal collar.

        Parameters
        ----------
        grid : TimeGrid
            The run's grid.
        temporal_rate : numpy.ndarray or None
            The collar's power rate on the coarse axis, 1/m, or ``None`` without one.
        check : BoundaryCheck
            The thresholds.
        columns : numpy.ndarray or None, optional
            Quadrature weights of a spatial run's transverse columns.

        Returns
        -------
        BoundaryMonitor
            A monitor with nothing recorded.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.monitor import BoundaryCheck, BoundaryMonitor
        >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
        >>> BoundaryMonitor.build(grid, None, BoundaryCheck()).weights.shape[0]
        4
        """
        t = np.asarray(grid.t, dtype=np.float64)
        span = grid.n_t * grid.dt
        edge = 0.5 * span - check.guard_fraction * span
        low = (t < -edge).astype(np.float64)
        high = (t > edge).astype(np.float64)
        collar = 1.0 - np.asarray(grid.t_collar.window(t), dtype=np.float64)
        rate = (
            np.zeros_like(t)
            if temporal_rate is None
            else np.asarray(temporal_rate, dtype=np.float64)
        )
        weights = np.stack([low, high, collar, rate])
        weights.setflags(write=False)
        return cls(check, weights, grid.kind, columns)

    def _power(
        self, field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Float[np.ndarray, " n_t"]:
        """Return the time-domain energy density summed over everything but time."""
        samples = fourier.inverse(field_spectrum, kind=self.kind)
        density = np.abs(samples) ** 2
        if self.columns is not None:
            density = density * self.columns[None, :, None, None]
        return density.reshape(-1, density.shape[-1]).sum(axis=0)

    def _shares(
        self, field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> tuple[float, Float[np.ndarray, " 4"]]:
        """Return the total and the four weighted shares of one state."""
        power = self._power(field_spectrum)
        total = float(power.sum())
        if total <= 0.0 or not math.isfinite(total):
            return 0.0, np.zeros(4)
        return total, (self.weights @ power) / total

    def launch(
        self,
        field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        z: float,
    ) -> None:
        """Record the launch, and warn if it already fills a guard or the collar.

        Parameters
        ----------
        field_spectrum : numpy.ndarray
            The launched state, canonical layout, FFT convention.
        z : float
            Its position, m.
        """
        total, shares = self._shares(field_spectrum)
        self._launch = total
        self._last = (z, float(shares[3]))
        self._launch_occupancy = (float(shares[0]), float(shares[1]), float(shares[2]))
        worst = max(self._launch_occupancy)
        if total > 0.0 and worst > self.check.launch_threshold:
            self._warned = True
            warnings.warn(
                f"the launched field already puts {worst:.2e} of its energy in the "
                "time window's guards or collar, above the "
                f"{self.check.launch_threshold:.0e} a clean launch leaves: the window "
                "clips the input, which can seed spurious radiation; widen trange or "
                "move the launch",
                BoundaryWarning,
                stacklevel=2,
            )

    def step(
        self,
        field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        z: float,
    ) -> None:
        """Account one committed step, and warn or stop on a guard occupancy.

        Parameters
        ----------
        field_spectrum : numpy.ndarray
            The committed state, canonical layout, FFT convention.
        z : float
            Its position, m.

        Raises
        ------
        BoundaryError
            When the square root of a guard occupancy exceeds ``stop_factor`` budgets.
        """
        total, shares = self._shares(field_spectrum)
        self._steps += 1
        if not total > _FLOOR * self._launch:
            # Nothing was launched, or nothing is left: no light to judge.
            self._last = (z, 0.0)
            return
        # The removal rate as a fraction of the launch energy per metre: the share is
        # relative to the state's own energy, which the launch's rescales.
        rate = float(shares[3]) * total / self._launch if self._launch > 0.0 else 0.0
        if self._last is not None:
            previous_z, previous_rate = self._last
            self._removed += 0.5 * (z - previous_z) * (previous_rate + rate)
        self._last = (z, rate)
        self._collar_max = max(self._collar_max, float(shares[2]))
        for edge in (0, 1):
            if shares[edge] > self._guard_max[edge]:
                self._guard_max[edge] = float(shares[edge])
                self._guard_at[edge] = z
        worst = math.sqrt(max(float(shares[0]), float(shares[1]), 0.0))
        if worst > self.check.stop_factor * self.check.budget:
            msg = (
                f"at z = {z:.6e} m the field puts {worst**2:.2e} of its energy in the "
                f"time window's {self.check.guard_fraction:g} guards, whose square "
                f"root {worst:.2e} is above {self.check.stop_factor:g} times the "
                f"{self.check.budget:.0e} budget: light is reaching the window's edge "
                "and will wrap around or be absorbed; widen trange, move the launch, "
                "or relax the check (settings.boundary_check)"
            )
            raise BoundaryError(msg)
        if worst > self.check.budget and not self._warned:
            self._warned = True
            warnings.warn(
                f"at z = {z:.6e} m the square root of the time window's guard "
                f"occupancy is {worst:.2e}, above the {self.check.budget:.0e} budget: "
                "light is reaching the window's edge; widen trange or move the launch",
                BoundaryWarning,
                stacklevel=2,
            )

    def summary(self) -> dict[str, float | None]:
        """Return the record of the run so far, for its execution record.

        Returns
        -------
        dict
            Launch occupancies, the largest guard and collar occupancies and where the
            guards peaked, the removed fraction of the launch energy and the number of
            committed steps accounted; ``None`` for what was never measured (a
            launch the run resumed past, a guard nothing reached).
        """

        def finite(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "launch_guard_low": finite(self._launch_occupancy[0]),
            "launch_guard_high": finite(self._launch_occupancy[1]),
            "launch_collar": finite(self._launch_occupancy[2]),
            "guard_low_max": self._guard_max[0],
            "guard_high_max": self._guard_max[1],
            "guard_low_at": finite(self._guard_at[0]),
            "guard_high_at": finite(self._guard_at[1]),
            "collar_max": self._collar_max,
            "removed_fraction": self._removed,
            "steps": self._steps,
        }

    def finish(self) -> None:
        """Report the removed fraction at the end of a run (D5).

        Logged always; warns when its square root exceeds the budget.
        """
        removed = self._removed
        logger.info(
            "the temporal collar removed %.3e of the launch energy; the guards peaked "
            "at %.2e (low) and %.2e (high)",
            removed,
            self._guard_max[0],
            self._guard_max[1],
        )
        if removed > 0.0 and math.sqrt(removed) > self.check.budget:
            warnings.warn(
                f"the temporal collar removed {removed:.2e} of the launch energy, "
                f"whose square root is above the {self.check.budget:.0e} budget: what "
                "left the window may have mattered; widen trange to keep it",
                BoundaryWarning,
                stacklevel=2,
            )

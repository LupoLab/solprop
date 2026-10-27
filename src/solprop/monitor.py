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

**The shot-noise floor (Q24).** An input-noise launch carries one photon per mode
across the whole window, guards and collar included. Light below that floor cannot be
told from the vacuum, so the monitor judges what lies above it: from each region's
energy it subtracts :data:`FLOOR_MARGIN` times the floor's expected share before
taking that energy's share of the whole field (the floor is stationary, so its expected
density is uniform in time; the removal rate subtracts the expectation alone). Without
noise nothing is subtracted; with a weak launch the field is mostly floor and what lies
above it is a small share. The raw values, of the whole field, are recorded beside the
judged ones.

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

FLOOR_MARGIN = 3.0
"""How many times its expected energy the floor is allowed in a region before light is
counted there. A region of ``n`` time samples holds the floor to a relative spread of
about ``1/sqrt(n)``; three times the expectation keeps the narrowest guards in use (a
few samples) from reading the floor's fluctuation as light."""

CADENCE = 8
"""Committed steps between checks while the edges are quiet: every one of them while a
guard or the collar holds more than :data:`ACTIVE` budgets squared, and the state of
every saved plane, are checked regardless."""

CHECKS = 256
"""The fewest checks over the guide's length: a quiet window is checked at least every
``length / CHECKS`` however long the steps, so light must walk about twenty windows
over the run to cross the collar and guards between two checks."""

ACTIVE = 0.1
"""The fraction of the budget, as a field amplitude, above which a guard or the collar
counts as active and every step is checked."""

QUIET = 1e-12
"""The share of the field's energy, above the shot-noise floor, below which the collar
counts as quiet: its in-stage term may then be left out of the stages (SEP 21). As a
field amplitude, a thousandth of the default budget's thousandth."""

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
    interval : float
        The longest distance, m, a quiet window goes unchecked (:data:`CHECKS`).
    split : bool
        Whether the temporal collar is applied once per accepted step (the
        ``"split_step"`` scheme), whose removal is then read exactly from each
        committed state rather than by the trapezoid (D5).
    """

    check: BoundaryCheck
    weights: Float[np.ndarray, "4 n_t"]
    kind: GridKind
    columns: Float[np.ndarray, " n_columns"] | None = None
    split: bool = False
    interval: float = math.inf
    _launch: float = field(default=math.nan, init=False)
    _floor: float = field(default=0.0, init=False)
    _floor_regions: Float[np.ndarray, " 4"] = field(
        default_factory=lambda: np.zeros(4), init=False
    )
    _raw_guard_max: float = field(default=0.0, init=False)
    _raw_removed: float = field(default=0.0, init=False)
    _raw_last: tuple[float, float] | None = field(default=None, init=False)
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
    _collar_now: float = field(default=0.0, init=False)
    _since: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _planes: dict[str, list[float]] = field(
        default_factory=lambda: {
            name: []
            for name in (
                "z",
                "collar_occupancy",
                "guard_low",
                "guard_high",
                "removed_fraction",
            )
        },
        init=False,
    )
    _steps: int = field(default=0, init=False)
    _skipped: int = field(default=0, init=False)
    _since_check: int = field(default=0, init=False)
    _busy: bool = field(default=True, init=False)
    _quiet: bool = field(default=False, init=False)
    _stale: bool = field(default=False, init=False)

    @classmethod
    def build(
        cls,
        grid: TimeGrid,
        temporal_rate: Float[np.ndarray, " n_t"] | None,
        check: BoundaryCheck,
        *,
        columns: Float[np.ndarray, " n_columns"] | None = None,
        split: bool = False,
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
        split : bool, optional
            Whether the collar is applied by the ``"split_step"`` scheme.

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
        return cls(check, weights, grid.kind, columns, split, grid.length / CHECKS)

    def _power(
        self, field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Float[np.ndarray, " n_t"]:
        """Return the time-domain energy density summed over everything but time."""
        samples = fourier.inverse(field_spectrum, kind=self.kind)
        density = np.abs(samples) ** 2
        if self.columns is not None:
            density = density * self.columns[None, :, None, None]
        return density.reshape(-1, density.shape[-1]).sum(axis=0)

    def _regions(
        self, field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> tuple[float, Float[np.ndarray, " 4"], Float[np.ndarray, " n_t"]]:
        """Return the total energy of one state, its four weighted energies and ``P``.

        The first three are the energies in the low guard, the high guard and the
        collar; the fourth is the collar's removal rate, energy per metre.
        """
        power = self._power(field_spectrum)
        total = float(power.sum())
        if total <= 0.0 or not math.isfinite(total):
            return 0.0, np.zeros(4), power
        return total, self.weights @ power, power

    def _above(self, regions: Float[np.ndarray, " 4"]) -> Float[np.ndarray, " 4"]:
        """Return each region's energy above the shot-noise floor.

        The occupancies must clear the floor's fluctuation and are clipped at zero.
        The removal rate subtracts the floor's expectation and keeps its sign: its
        integral along z averages the fluctuation out, which clipping would rectify
        into a spurious removal.
        """
        above = regions - self._floor_regions * np.array(
            [FLOOR_MARGIN, FLOOR_MARGIN, FLOOR_MARGIN, 1.0]
        )
        above[:3] = np.maximum(above[:3], 0.0)
        return above

    def launch(
        self,
        field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        z: float,
        *,
        floor: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None = None,
    ) -> None:
        """Record the launch, and warn if it already fills a guard or the collar.

        Parameters
        ----------
        field_spectrum : numpy.ndarray
            The launched state, canonical layout, FFT convention.
        z : float
            Its position, m.
        floor : numpy.ndarray or None, optional
            The input-noise draw the launch carries, in the same layout and
            convention; ``None`` without one. Its energy is the vacuum floor the
            monitor judges above (Q24).
        """
        if floor is not None:
            self._floor = float(self._power(floor).sum())
            # The floor is stationary: its expected density is uniform in time.
            self._floor_regions = self._floor * self.weights.mean(axis=1)
        total, regions, _ = self._regions(field_spectrum)
        self._launch = total
        if total <= 0.0:
            self._last = (z, 0.0)
            self._raw_last = (z, 0.0)
            self._settle(None)
            return
        shares = self._above(regions) / total
        self._settle(shares)
        self._last = (z, float(shares[3]))
        self._raw_last = (z, float(regions[3]) / total)
        self._launch_occupancy = (float(shares[0]), float(shares[1]), float(shares[2]))
        self._collar_now = float(shares[2])
        self._since = [float(shares[0]), float(shares[1])]
        self._plane_row(z)
        worst = max(self._launch_occupancy)
        if worst > self.check.launch_threshold:
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
        total, regions, power = self._regions(field_spectrum)
        self._steps += 1
        # The split's exact removal reads one step; across unchecked ones the
        # trapezoid integrates the rate instead.
        exact = self.split and self._since_check == 0
        self._since_check = 0
        self._stale = False
        if not total > _FLOOR * self._launch:
            # Nothing was launched, or nothing is left: no light to judge.
            self._last = (z, 0.0)
            self._raw_last = (z, 0.0)
            self._collar_now = 0.0
            self._settle(None)
            return
        # The whole field's record, floor included, kept beside the judged one.
        raw_rate = float(regions[3]) / self._launch
        if exact and self._raw_last is not None:
            self._split_removal(power, z - self._raw_last[0])
        elif self._raw_last is not None:
            previous_z, previous_rate = self._raw_last
            self._raw_removed += 0.5 * (z - previous_z) * (previous_rate + raw_rate)
        self._raw_last = (z, raw_rate)
        self._raw_guard_max = max(
            self._raw_guard_max, float(max(regions[0], regions[1])) / total
        )
        above = self._above(regions)
        shares = above / total
        self._settle(shares)
        # The removal rate as a fraction of the launch energy per metre.
        rate = float(above[3]) / self._launch
        if self._last is not None and not exact:
            previous_z, previous_rate = self._last
            self._removed += 0.5 * (z - previous_z) * (previous_rate + rate)
        self._last = (z, rate)
        self._collar_max = max(self._collar_max, float(shares[2]))
        self._collar_now = float(shares[2])
        for edge in (0, 1):
            self._since[edge] = max(self._since[edge], float(shares[edge]))
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

    def _settle(self, shares: Float[np.ndarray, " 4"] | None) -> None:
        """Set the check cadence and the collar's quiet flag from a checked state.

        ``None`` is a state with no light to judge: quiet, and checked lazily.
        """
        if shares is None:
            self._busy, self._quiet = False, True
            return
        active = (ACTIVE * self.check.budget) ** 2
        self._busy = bool(max(shares[0], shares[1], shares[2]) > active)
        self._quiet = bool(shares[2] < QUIET)

    def due(self, z: float) -> bool:
        """Return whether the step just committed at `z` is to be checked.

        Every step is while a guard or the collar is active (:data:`ACTIVE`), and
        otherwise every :data:`CADENCE`-th or after :attr:`interval`, whichever comes
        first; the caller passes the others to :meth:`skip` without reading the state.

        Parameters
        ----------
        z : float
            The committed position, m.

        Returns
        -------
        bool
            True to call :meth:`step` with the committed state.
        """
        if self._busy or self._since_check + 1 >= CADENCE:
            return True
        return self._last is None or z - self._last[0] >= self.interval

    def skip(self) -> None:
        """Count a committed step that is not checked."""
        self._skipped += 1
        self._since_check += 1
        self._stale = True

    @property
    def stale(self) -> bool:
        """Whether the last committed step went unchecked."""
        return self._stale

    @property
    def collar_quiet(self) -> bool:
        """Whether the collar held less than :data:`QUIET` above the floor.

        As of the last check; while it holds, the collar's in-stage term may be left
        out of the next stages.
        """
        return self._quiet

    def _split_removal(self, power: Float[np.ndarray, " n_t"], step: float) -> None:
        """Add one split step's exact removal, read back from its committed state.

        The split multiplies the field by ``exp(-alpha(t) h / 2)`` once per accepted
        step, so the power it took is ``P(t) (exp(alpha(t) h) - 1)`` of the power
        ``P`` it left; the floor's share is that of a uniform density. The band mask
        that follows the split is not counted.
        """
        if not step > 0.0:
            return
        gain = np.expm1(self.weights[3] * step)
        taken = float(power @ gain)
        floor = self._floor / power.size * float(gain.sum())
        self._raw_removed += taken / self._launch
        self._removed += (taken - floor) / self._launch

    def _plane_row(self, z: float) -> None:
        """Append one saved plane's row and start the next plane's guard maxima."""
        row = self._planes
        if row["z"] and row["z"][-1] == z:
            return
        row["z"].append(z)
        row["collar_occupancy"].append(self._collar_now)
        row["guard_low"].append(self._since[0])
        row["guard_high"].append(self._since[1])
        row["removed_fraction"].append(max(self._removed, 0.0))
        self._since = [0.0, 0.0]

    def plane(self, z: float) -> None:
        """Record the row of the saved plane the run has just reached.

        The row holds the collar occupancy at the plane, each guard's largest
        occupancy over the committed steps since the previous plane, and the removed
        fraction so far, all above the shot-noise floor (SEP 21 section 6, D5).

        Parameters
        ----------
        z : float
            The plane's position, m.
        """
        self._plane_row(z)

    def summary(self) -> dict[str, float | dict[str, list[float]] | None]:
        """Return the record of the run so far, for its execution record.

        Returns
        -------
        dict
            Launch occupancies, the largest guard and collar occupancies and where the
            guards peaked, the removed fraction of the launched signal and the number
            of committed steps accounted, all judged above the shot-noise floor;
            ``None`` for what was never measured (a launch the run resumed past, a
            guard nothing reached). Beside them, the floor's share of the launch
            energy (``noise_floor``, zero without input noise), and the largest guard
            share and the removed fraction of the whole field, floor included
            (``raw_guard_max``, ``raw_removed_fraction``). Under ``planes``, one row
            per saved plane from the launch on: ``z``, the ``collar_occupancy`` there,
            each guard's largest occupancy since the previous plane (``guard_low``,
            ``guard_high``) and the ``removed_fraction`` so far.
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
            "removed_fraction": max(self._removed, 0.0),
            "steps": self._steps,
            "skipped": self._skipped,
            "noise_floor": self._floor / self._launch if self._launch > 0.0 else 0.0,
            "raw_guard_max": self._raw_guard_max,
            "raw_removed_fraction": self._raw_removed,
            "planes": {name: list(values) for name, values in self._planes.items()},
        }

    def finish(self) -> None:
        """Report the removed fraction at the end of a run (D5).

        Logged always; warns when its square root exceeds the budget.
        """
        removed = max(self._removed, 0.0)
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

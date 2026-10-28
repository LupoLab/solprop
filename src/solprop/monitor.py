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
  launch energy. Under the ``"split_step"`` scheme a checked step is read just before
  its split instead, which gives the power the split took exactly
  (:meth:`BoundaryMonitor.before_split`).

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

Policy (D7, one rule from the launch on): the square root of a guard occupancy above
:attr:`BoundaryCheck.budget` warns, and above ``stop_factor`` times it stops the run, or
refuses it at the launch before a step is spent. At the end of a run the removed
fraction is reported, and warns when its square root exceeds the budget (D5).

**The quiet collar is opt-in** (:attr:`BoundaryCheck.quiet_collar`, SEP 21 D15 as
amended by ADR-057). When asked for, a collar that held less than :data:`QUIET` of the
field's energy, the shot-noise floor included, at :data:`QUIET_CHECKS` consecutive
checks may be left out of the in-stage stages until light returns. It is judged on the
whole field because light below the floor is still light the collar would have
absorbed: judged above the floor, the threshold became a few times the floor itself
and dropped a term that moved the field by up to a percent. By default the collar
stays in every stage.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.fourier import GridKind
from solprop.grid import TimeGrid

__all__ = [
    "ACTIVE",
    "CADENCE",
    "CHECKS",
    "DEFAULT_BOUNDARY_CHECK",
    "FLOOR_MARGIN",
    "QUIET",
    "QUIET_CHECKS",
    "BoundaryCheck",
    "BoundaryMonitor",
]

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
``length / CHECKS`` (or at every committed step, when the steps are longer), so light
must walk about twenty windows over the run to cross the collar and guards between two
checks. While an opted-in quiet collar is left out of the stages no step is longer than
that interval, so the premise holds where an unabsorbed crossing would matter."""

ACTIVE = 0.1
"""The fraction of the budget, as a field amplitude, above which a guard or the collar
counts as active and every step is checked."""

QUIET = 1e-12
"""The share of the field's energy, floor included, below which the collar counts as
quiet: when the run opts in (:attr:`BoundaryCheck.quiet_collar`), its in-stage term may
then be left out of the stages (SEP 21). As a field amplitude, a thousandth of the
default budget's thousandth."""

QUIET_CHECKS = 2
"""Consecutive quiet checks before the collar's term is left out. Light returning puts
it back at the first check that sees it; asking for two quiet ones before leaving it
out again keeps a collar hovering at the threshold from switching at every check, each
switch costing the march a reused stage or an Adams rejection."""

logger = logging.getLogger(__name__)


def _number(
    record: Mapping[str, object], name: str, default: float = math.nan
) -> float:
    """Return a record's number, or `default` when it is absent or not a number.

    Parameters
    ----------
    record : collections.abc.Mapping
        A monitor record read back from a run's file.
    name : str
        The field.
    default : float, optional
        What an absent or unmeasured field stands for.

    Returns
    -------
    float
        The value.
    """
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


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
        budgets, and a launch above it is refused. Default 10.
    quiet_collar : bool
        Leave the in-stage temporal collar out of the stages while it is quiet
        (:data:`QUIET`, :data:`QUIET_CHECKS`), a speed-up for runs whose light stays
        clear of the edges. Off by default: the collar then stays in every stage. It
        applies on binary64 modal routes with the ``"nonlinear"`` absorber scheme
        and a temporal collar alone; elsewhere the collar always stays.

    Examples
    --------
    >>> from solprop.monitor import BoundaryCheck
    >>> check = BoundaryCheck()
    >>> check.guard_fraction, check.budget, check.stop_factor, check.quiet_collar
    (0.05, 0.001, 10.0, False)
    """

    guard_fraction: float = 0.05
    budget: float = 1e-3
    stop_factor: float = 10.0
    quiet_collar: bool = False

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
        for name in ("budget", "stop_factor"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                msg = f"{name} must be finite and positive, got {value!r}"
                raise BoundaryError(msg)
        if self.stop_factor < 1.0:
            msg = f"stop_factor must be at least 1, got {self.stop_factor!r}"
            raise BoundaryError(msg)


DEFAULT_BOUNDARY_CHECK = BoundaryCheck()
"""The check a run takes unless its settings or an interface keyword say otherwise."""


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
        ``"split_step"`` scheme), whose removal is then read exactly from the state
        before the split (:meth:`before_split`) rather than by the trapezoid (D5).
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
    _quiet_run: int = field(default=0, init=False)
    _left_out: bool = field(default=False, init=False)
    _left_out_checks: int = field(default=0, init=False)
    _stale: bool = field(default=False, init=False)
    _pending: tuple[float, Float[np.ndarray, " n_t"]] | None = field(
        default=None, init=False
    )
    _resuming: bool = field(default=False, init=False)
    _resume_plane: bool = field(default=False, init=False)
    _resumed_at: list[float] = field(default_factory=list, init=False)

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
        """Return the time-domain energy density summed over everything but time.

        Parameters
        ----------
        field_spectrum : numpy.ndarray
            A state, canonical layout, FFT convention.

        Returns
        -------
        numpy.ndarray
            ``P(t)`` on the coarse time axis, weighted by the columns' quadrature.
        """
        samples = fourier.inverse(field_spectrum, kind=self.kind)
        density = np.abs(samples) ** 2
        if self.columns is not None:
            density = density * self.columns[None, :, None, None]
        return density.reshape(-1, density.shape[-1]).sum(axis=0)

    def _regions(
        self, power: Float[np.ndarray, " n_t"]
    ) -> tuple[float, Float[np.ndarray, " 4"]]:
        """Return the total energy of one state and its four weighted energies.

        Parameters
        ----------
        power : numpy.ndarray
            The state's ``P(t)`` (:meth:`_power`).

        Returns
        -------
        total : float
            The state's energy in the proxy's units; zero for a state with none.
        regions : numpy.ndarray
            The energies in the low guard, the high guard and the collar, and the
            collar's removal rate, energy per metre.
        """
        total = float(power.sum())
        if total <= 0.0 or not math.isfinite(total):
            return 0.0, np.zeros(4)
        return total, self.weights @ power

    def _above(self, regions: Float[np.ndarray, " 4"]) -> Float[np.ndarray, " 4"]:
        """Return each region's energy above the shot-noise floor.

        The occupancies must clear the floor's fluctuation and are clipped at zero.
        The removal rate subtracts the floor's expectation and keeps its sign: its
        integral along z averages the fluctuation out, which clipping would rectify
        into a spurious removal.

        Parameters
        ----------
        regions : numpy.ndarray
            A state's four weighted energies (:meth:`_regions`).

        Returns
        -------
        numpy.ndarray
            The same four, above the floor.
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
        """Record the launch, and warn or refuse if it already fills a guard.

        The rule of every later check (:meth:`step`): the square root of a guard
        occupancy above the budget warns, and above ``stop_factor`` budgets refuses
        the run before its first step. After :meth:`resume` the state is the one the
        run resumes from, recorded and not judged.

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

        Raises
        ------
        BoundaryError
            When the square root of a guard occupancy exceeds ``stop_factor`` budgets.
        """
        if floor is not None:
            self._floor = float(self._power(floor).sum())
            # The floor is stationary: its expected density is uniform in time.
            self._floor_regions = self._floor * self.weights.mean(axis=1)
        total, regions = self._regions(self._power(field_spectrum))
        resuming, self._resuming = self._resuming, False
        if resuming:
            self._resumed_at.append(z)
        if not (resuming and self._launch > 0.0):
            # A resume restored its original launch energy (`resume`), unless the
            # record it continues had none.
            self._launch = total
        if total <= 0.0:
            self._last = (z, 0.0)
            self._raw_last = (z, 0.0)
            self._settle(None)
            return
        above = self._above(regions)
        shares = above / total
        self._settle(shares, float(regions[2]) / total)
        self._last = (z, float(above[3]) / self._launch)
        self._raw_last = (z, float(regions[3]) / self._launch)
        self._collar_now = float(shares[2])
        self._since = [float(shares[0]), float(shares[1])]
        if resuming:
            if self._resume_plane:
                self._plane_row(z)
            return
        self._launch_occupancy = (float(shares[0]), float(shares[1]), float(shares[2]))
        self._plane_row(z)
        self._judge(shares, None)

    def resume(self, record: Mapping[str, object] | None, *, plane: bool) -> None:
        """Continue the record of the executions a resumed run follows (SEP 21).

        Called before :meth:`launch`, which then records the resumed state without
        judging it: the first execution judged the launch, and a run stopped at an
        edge may be resumed under a relaxed check. When `record` carries the
        monitor's totals, ``launch_energy`` among them, they are restored, so the
        removed fraction, the guard maxima and the rows continue against the
        original launch; otherwise the record starts again from the resumed state,
        its launch fields unmeasured.

        Parameters
        ----------
        record : collections.abc.Mapping or None
            The ``boundary_monitor`` record of the latest earlier execution, or
            ``None`` without one. Rows are restored from a closing record; a running
            one, written at a plane, carries none
            (:func:`solprop.driver.run.run_problem`).
        plane : bool
            Whether the run resumes at a saved plane, whose row is then recorded
            unless the restored rows hold it.
        """
        self._resuming = True
        self._resume_plane = plane
        if record is None:
            return
        launch = _number(record, "launch_energy")
        if not launch > 0.0:
            return
        self._launch = launch
        low, high, collar = (
            _number(record, name)
            for name in ("launch_guard_low", "launch_guard_high", "launch_collar")
        )
        self._launch_occupancy = (low, high, collar)
        self._guard_max = [
            _number(record, "guard_low_max", 0.0),
            _number(record, "guard_high_max", 0.0),
        ]
        self._guard_at = [
            _number(record, "guard_low_at"),
            _number(record, "guard_high_at"),
        ]
        self._collar_max = _number(record, "collar_max", 0.0)
        self._removed = _number(record, "removed_fraction", 0.0)
        self._raw_removed = _number(record, "raw_removed_fraction", 0.0)
        self._raw_guard_max = _number(record, "raw_guard_max", 0.0)
        self._steps = int(_number(record, "steps", 0.0))
        self._skipped = int(_number(record, "skipped", 0.0))
        self._left_out_checks = int(_number(record, "collar_left_out_checks", 0.0))
        resumed = record.get("resumed_at")
        if isinstance(resumed, list):
            self._resumed_at = [float(one) for one in cast("list[float]", resumed)]
        rows = record.get("planes")
        if isinstance(rows, Mapping) and set(cast("Mapping[str, object]", rows)) == set(
            self._planes
        ):
            self._planes = {
                name: [float(one) for one in cast("list[float]", values)]
                for name, values in cast("Mapping[str, list[float]]", rows).items()
            }

    def step(
        self,
        field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None,
        z: float,
    ) -> None:
        """Account one committed step, and warn or stop on a guard occupancy.

        Parameters
        ----------
        field_spectrum : numpy.ndarray or None
            The committed state, canonical layout, FFT convention; ``None`` for the
            state :meth:`before_split` read before the step's split.
        z : float
            Its position, m.

        Raises
        ------
        BoundaryError
            When the square root of a guard occupancy exceeds ``stop_factor`` budgets,
            or when no state was given and none was read before the split.
        """
        pending, self._pending = self._pending, None
        if field_spectrum is not None:
            power, before = self._power(field_spectrum), None
        elif pending is not None:
            # The split leaves exp(-alpha h) of the power it found; the band mask
            # after it is not counted.
            before = pending
            power = before[1] * np.exp(-self.weights[3] * before[0])
        else:
            msg = "the boundary monitor was given no state and read none before a split"
            raise BoundaryError(msg)
        total, regions = self._regions(power)
        self._steps += 1
        self._left_out_checks += int(self._left_out)
        # The split's exact removal reads one step from the state before it; across
        # unchecked ones, or from a committed state, the trapezoid integrates the rate.
        exact = before if self._since_check == 0 else None
        self._since_check = 0
        self._stale = False
        if not total > _FLOOR * self._launch:
            # Nothing was launched, or nothing is left: no light to judge.
            self._last = (z, 0.0)
            self._raw_last = (z, 0.0)
            self._collar_now = 0.0
            self._settle(None)
            return
        # The whole field's record, floor included, kept beside the judged one. While
        # the stages left a quiet collar out (`note_collar`) it absorbed nothing, so
        # the steps since the last check add no removal.
        raw_rate = float(regions[3]) / self._launch
        previous_z, previous_rate = self._raw_last or (z, raw_rate)
        if exact is not None:
            self._split_removal(*exact)
        elif not self._left_out:
            self._raw_removed += 0.5 * (z - previous_z) * (previous_rate + raw_rate)
        self._raw_last = (z, raw_rate)
        self._raw_guard_max = max(
            self._raw_guard_max, float(max(regions[0], regions[1])) / total
        )
        above = self._above(regions)
        shares = above / total
        self._settle(shares, float(regions[2]) / total)
        # The removal rate as a fraction of the launch energy per metre.
        rate = float(above[3]) / self._launch
        if self._last is not None and exact is None and not self._left_out:
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
        self._judge(shares, z)

    def _judge(self, shares: Float[np.ndarray, " 4"], z: float | None) -> None:
        """Warn once, or stop, on a state's guard occupancy (D7).

        Parameters
        ----------
        shares : numpy.ndarray
            The state's shares above the floor.
        z : float or None
            Its position, m, or ``None`` for the launch.

        Raises
        ------
        BoundaryError
            When the square root of a guard occupancy exceeds ``stop_factor`` budgets.
        """
        check = self.check
        worst = math.sqrt(max(float(shares[0]), float(shares[1]), 0.0))
        if z is None:
            where = "at the launch"
            cause = "the window clips the input, which can seed spurious radiation"
            fate = ""
        else:
            where = f"at z = {z:.6e} m"
            cause = "light is reaching the window's edge"
            fate = " and will wrap around or be absorbed"
        if worst > check.stop_factor * check.budget:
            msg = (
                f"{where} the field puts {worst**2:.2e} of its energy in the time "
                f"window's {check.guard_fraction:g} guards, whose square root "
                f"{worst:.2e} is above {check.stop_factor:g} times the "
                f"{check.budget:.0e} budget: {cause}{fate}; widen trange, move the "
                "launch, or relax the check (settings.boundary_check)"
            )
            raise BoundaryError(msg)
        if worst > check.budget and not self._warned:
            self._warned = True
            warnings.warn(
                f"{where} the square root of the time window's guard occupancy is "
                f"{worst:.2e}, above the {check.budget:.0e} budget: {cause}; widen "
                "trange or move the launch",
                BoundaryWarning,
                stacklevel=3,
            )

    def _settle(
        self, shares: Float[np.ndarray, " 4"] | None, raw_collar: float = 0.0
    ) -> None:
        """Set the check cadence and the collar's quiet flag from a checked state.

        The cadence follows the shares above the floor; the quiet flag follows the
        collar's share of the whole field, floor included (:data:`QUIET`), and is
        raised only for a run that opted in, after :data:`QUIET_CHECKS` quiet checks
        in a row.

        Parameters
        ----------
        shares : numpy.ndarray or None
            The state's shares above the floor, or ``None`` for a state with no light
            to judge, which is quiet and checked lazily.
        raw_collar : float, optional
            The collar's share of the whole field, floor included.
        """
        if shares is None:
            self._busy, calm = False, True
        else:
            active = (ACTIVE * self.check.budget) ** 2
            self._busy = bool(max(shares[0], shares[1], shares[2]) > active)
            calm = raw_collar < QUIET
        self._quiet_run = self._quiet_run + 1 if calm else 0
        self._quiet = self.check.quiet_collar and self._quiet_run >= QUIET_CHECKS

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
        self._pending = None

    @property
    def stale(self) -> bool:
        """Whether the last committed step went unchecked.

        Returns
        -------
        bool
            True when :meth:`skip` was the last call.
        """
        return self._stale

    @property
    def collar_quiet(self) -> bool:
        """Whether the collar's in-stage term may be left out of the next stages.

        True only when the run opted in (:attr:`BoundaryCheck.quiet_collar`) and the
        collar held less than :data:`QUIET` of the field's energy, floor included, at
        the last :data:`QUIET_CHECKS` checks.

        Returns
        -------
        bool
            The flag.
        """
        return self._quiet

    def note_collar(self, left_out: bool) -> None:
        """Record whether the stages leave the collar's term out from here on.

        The run's driver decides, from this monitor's :attr:`collar_quiet` and from
        whether its route can leave the term out at all; while it is out the collar
        absorbs nothing, and :meth:`step` adds no removal for those steps.

        Parameters
        ----------
        left_out : bool
            True while the in-stage term is left out.
        """
        self._left_out = left_out

    def before_split(
        self,
        field_spectrum: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        step: float,
    ) -> None:
        """Read an accepted step's state just before its split collar is applied.

        Under the ``"split_step"`` scheme the run's driver calls this, on a step it is
        about to adopt and :meth:`due` to check, and then :meth:`step` with ``None``.
        The split multiplies the field by ``exp(-alpha(t) h / 2)``, so the power it
        takes, ``P(t) (1 - exp(-alpha(t) h))`` of the power ``P`` it finds, is read
        exactly from this state. Read back from the committed state the same removal
        would be ``P(t) (exp(alpha(t) h) - 1)`` of the power the split left, which
        multiplies whatever the band mask after the split leaves deep in the collar by
        ``exp(alpha h)``: a removal made from nothing.

        Parameters
        ----------
        field_spectrum : numpy.ndarray
            The accepted state before the split, canonical layout, FFT convention.
        step : float
            The step's length, m.
        """
        self._pending = (step, self._power(field_spectrum))

    @property
    def read_ahead(self) -> bool:
        """Whether :meth:`before_split` read the step :meth:`step` is to account.

        Returns
        -------
        bool
            True until the next :meth:`step` takes the reading.
        """
        return self._pending is not None

    def _split_removal(self, step: float, before: Float[np.ndarray, " n_t"]) -> None:
        """Add one split step's exact removal, read from the state before its split.

        The floor's share is that of a uniform density.

        Parameters
        ----------
        step : float
            The step's length, m.
        before : numpy.ndarray
            ``P(t)`` of the accepted state before the split (:meth:`before_split`).
        """
        if not step > 0.0:
            return
        taken_share = -np.expm1(-self.weights[3] * step)
        taken = float(before @ taken_share)
        floor = self._floor / before.size * float(taken_share.sum())
        self._raw_removed += taken / self._launch
        self._removed += (taken - floor) / self._launch

    def _plane_row(self, z: float) -> None:
        """Append one saved plane's row and start the next plane's guard maxima.

        Parameters
        ----------
        z : float
            The plane's position, m.
        """
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

    def summary(
        self,
    ) -> dict[
        str, float | list[float] | dict[str, list[float]] | dict[str, object] | None
    ]:
        """Return the record of the run so far, for its execution record.

        Returns
        -------
        dict
            The thresholds of this execution (``check``); the launch's energy in the
            monitor's own units, the denominator of every fraction
            (``launch_energy``), and where the run was resumed (``resumed_at``,
            empty for an uninterrupted one). Launch occupancies, the largest guard
            and collar occupancies and where the guards peaked, the removed fraction
            of the launched signal and the number of committed steps accounted, all
            judged above the shot-noise floor; ``None`` for what was never measured
            (a launch a resume could not recover, a guard nothing reached). Beside
            them, the floor's share of the launch energy (``noise_floor``, zero
            without input noise), and the largest guard share and the removed
            fraction of the whole field, floor included (``raw_guard_max``,
            ``raw_removed_fraction``), and how many checks found the collar's term
            left out since the previous one (``collar_left_out_checks``, zero unless
            the run opted in to :attr:`BoundaryCheck.quiet_collar`). Under
            ``planes``, one row per saved plane from the launch on: ``z``, the
            ``collar_occupancy`` there, each guard's largest occupancy since the
            previous plane (``guard_low``, ``guard_high``) and the
            ``removed_fraction`` so far. A resumed run continues all of these
            (:meth:`resume`).
        """

        def finite(value: float) -> float | None:
            """Return `value`, or ``None`` when it was never measured (not finite).

            Parameters
            ----------
            value : float
                A recorded value.

            Returns
            -------
            float or None
                The value, or ``None``.
            """
            return value if math.isfinite(value) else None

        return {
            "check": dataclasses.asdict(self.check),
            "launch_energy": finite(self._launch),
            "resumed_at": list(self._resumed_at),
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
            "collar_left_out_checks": self._left_out_checks,
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

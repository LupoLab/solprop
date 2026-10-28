# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The seam: what a backend is, and what comes back from one.

Everything that touches the propagated state during a run is reached through this one
protocol, and only the backends know how the state is laid out, transformed and stepped
(``docs/dev/backends.md`` section 1). The protocol is drawn at the level of *operations
on the state* rather than at the array-namespace level, because an ``xp``-style
abstraction would force every backend to keep the array compiler's structure —
out-of-place transforms, separate scaling passes, materialised stages, one reduction per
statistic — and that structure is precisely what a fast backend exists to change.

A backend is constructed from a validated
:class:`solprop.translate.problem.KernelProblem`, takes host arrays and returns host
arrays; its buffers, layout and plans are private, and the state is never a parameter of
:meth:`Backend.propagate_to` or :meth:`Backend.observe`. Selection by name and the
availability rules are in :mod:`solprop.backends.registry`.

**Sign convention.** Every complex array crossing this seam, in either direction, is in
the **FFT convention** of ``docs/dev/conventions.md`` section 2, with the phase origin
at the first time sample: ``spectral_convention = "fft"``. The output layer conjugates
on the way to disk and the processing accessors on the way to a user. Backends
never conjugate.
"""

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop.exceptions import BackendError, BoundaryError
from solprop.log import ProgressReporter
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.stats import StatsSpec

__all__ = [
    "Backend",
    "BatchBackend",
    "Observation",
    "StepHistory",
    "StepStats",
    "StepSummary",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class StepHistory:
    """What the stepper did between two observations.

    Recorded whether the run was adaptive or replayed a fixed schedule, because the
    embedded error estimate costs nothing extra and is computed either way
    (``docs/dev/stepper.md`` section 8): a replayed schedule too coarse for the
    tolerance it is scaled by is then reported rather than silent.

    Attributes
    ----------
    z : numpy.ndarray
        Position at the *end* of each step, m, float64, shape ``(n_steps,)``.
    dz : numpy.ndarray
        Length of each step, m, float64, shape ``(n_steps,)``, positive.
    error_norm : numpy.ndarray
        The scaled embedded error norm of each step, float64, shape ``(n_steps,)``.
        For a clipped or stretched step this is the equivalent norm of the controller's
        proposal, which is what the controller's state is
        (:meth:`solprop.stepper.controller.PIController.equivalent_norm`). A rejected
        trial that diverged, whose norm overflowed or was NaN, is recorded as
        infinity; every other entry is finite.
    accepted : numpy.ndarray
        Whether each step was accepted, Boolean, shape ``(n_steps,)``.
    wall_time : numpy.ndarray
        Seconds elapsed at the end of each step, measured from the previous observation,
        float64, shape ``(n_steps,)``, non-decreasing.

    Raises
    ------
    BackendError
        If the arrays are not all the same length, a step length is not positive, or an
        entry is not finite, other than a rejected step's infinite norm.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import StepHistory
    >>> history = StepHistory(np.array([0.1, 0.2]), np.array([0.1, 0.1]),
    ...                       np.array([0.4, 0.5]), np.array([True, True]),
    ...                       np.array([0.01, 0.02]))
    >>> history.n_steps, history.n_rejected
    (2, 0)
    """

    z: Float[np.ndarray, " n_steps"]
    dz: Float[np.ndarray, " n_dz"]
    error_norm: Float[np.ndarray, " n_norm"]
    accepted: Bool[np.ndarray, " n_accepted"]
    wall_time: Float[np.ndarray, " n_wall"]

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the record.

        Raises
        ------
        BackendError
            Naming the offending field.
        """
        for name, dtype in (
            ("z", np.float64),
            ("dz", np.float64),
            ("error_norm", np.float64),
            ("accepted", np.bool_),
            ("wall_time", np.float64),
        ):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            # The rank is fixed by the annotations, which the test session enforces;
            # what a caller can still get wrong is the number of entries.
            if values.shape != self.z.shape:
                msg = (
                    f"{name} has {values.size} entries, expected {self.z.size}: "
                    "one entry per step, as z has"
                )
                raise BackendError(msg)
            if (
                dtype is np.float64
                and name != "error_norm"
                and not np.all(np.isfinite(values))
            ):
                msg = f"{name} has entries that are not finite"
                raise BackendError(msg)
        # A diverged trial is a rejection like any other: the controller shrinks by its
        # smallest factor and retries, so its infinite norm is a record, not a failure.
        # An accepted step is always finite (``PIController.accepts``).
        diverged = np.isposinf(self.error_norm) & ~self.accepted
        if not np.all(np.isfinite(self.error_norm) | diverged):
            msg = (
                "error_norm has entries that are not finite, other than a rejected "
                "step's infinity"
            )
            raise BackendError(msg)
        if self.dz.size and not np.all(self.dz > 0.0):
            msg = f"dz must be positive at every step, got a minimum of {self.dz.min()}"
            raise BackendError(msg)

    @property
    def n_steps(self) -> int:
        """Return the number of steps recorded, accepted and rejected together.

        Returns
        -------
        int
            The length of ``z``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import StepHistory
        >>> StepHistory(*(np.zeros(0) for _ in range(3)),
        ...             np.zeros(0, bool), np.zeros(0)).n_steps
        0
        """
        return int(self.z.size)

    @property
    def n_rejected(self) -> int:
        """Return the number of rejected steps.

        Returns
        -------
        int
            How many entries of ``accepted`` are false.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import StepHistory
        >>> StepHistory(np.array([0.1]), np.array([0.1]), np.array([2.0]),
        ...             np.array([False]), np.array([0.5])).n_rejected
        1
        """
        return int(np.count_nonzero(~self.accepted))


@dataclass(frozen=True, eq=False)
class StepStats:
    """The statistics a backend evaluated *between* two observations.

    The channel of ``docs/dev/kernel-contract.md`` section 9. A run whose
    :class:`solprop.stats.StatsSpec` names a ``periodic`` subset gets one row every
    ``period`` accepted steps, evaluated inside the backend's own step loop from the
    state that step just committed, and drained here at each save plane. Evaluating
    every statistic on every accepted step costs about a fifth of a step; a subset and
    a period are the two knobs that make the same information affordable.

    Empty when the run asked for no periodic subset, which is the default.

    Attributes
    ----------
    z : numpy.ndarray
        Position of each row, m, float64, shape ``(n_rows,)``, increasing.
    step : numpy.ndarray
        Index of the accepted step each row was taken after, into the *same
        observation's* :class:`StepHistory`, int64, shape ``(n_rows,)``, increasing.
        The history counts rejected attempts too, so this is not ``row * period``.
    values : collections.abc.Mapping
        The statistics, keyed by catalogue name, each with a leading row axis and then
        the leading batch axis every statistic crossing this seam carries:
        ``(n_rows, n_batch, *shape)``.

    Raises
    ------
    BackendError
        If the two index arrays disagree in length, are not increasing, or a value's
        leading axis is not the row count.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import StepStats
    >>> rows = StepStats(np.array([0.1, 0.2]), np.array([0, 1]),
    ...                  {"energy": np.ones((2, 1, 1, 1))})
    >>> rows.n_rows, rows.names
    (2, ('energy',))
    >>> StepStats.empty().n_rows
    0
    """

    z: Float[np.ndarray, " n_rows"]
    step: Int[np.ndarray, " n_rows_step"]
    values: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the rows.

        Raises
        ------
        BackendError
            Naming the offending column or statistic.
        """
        positions = np.ascontiguousarray(self.z, dtype=np.float64)
        indices = np.ascontiguousarray(self.step, dtype=np.int64)
        for name, values in (("z", positions), ("step", indices)):
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if indices.shape != positions.shape:
            msg = (
                f"step has {indices.size} entries and z has {positions.size}: "
                "one entry per row, as z has"
            )
            raise BackendError(msg)
        if not np.all(np.isfinite(positions)):
            msg = "z has entries that are not finite"
            raise BackendError(msg)
        # Increasing rather than merely distinct: the rows are taken in the order the
        # run took its steps, and a reader plots them against z without sorting.
        for name, values in (("z", positions), ("step", indices)):
            if values.size > 1 and not np.all(np.diff(values) > 0):
                msg = f"{name} must increase from row to row, and does not"
                raise BackendError(msg)
        adopted = {}
        for name, values in self.values.items():
            # `asarray` before `ascontiguousarray`, because the latter promotes a
            # zero-dimensional array to one dimension and a scalar would then pass
            # as a single row.
            row = np.asarray(values)
            if row.ndim == 0 or row.shape[0] != positions.size:
                msg = (
                    f"the {name!r} rows have leading axis {row.shape}, expected "
                    f"{positions.size} rows as z has"
                )
                raise BackendError(msg)
            row = np.ascontiguousarray(row)
            row.setflags(write=False)
            adopted[name] = row
        object.__setattr__(self, "values", adopted)

    @classmethod
    def empty(cls) -> "StepStats":  # noqa: UP037 - bound after the class body
        """Return the channel a run without a periodic subset carries.

        Returns
        -------
        StepStats
            No rows and no statistics.
        """
        return cls(np.empty(0), np.empty(0, dtype=np.int64), {})

    @property
    def n_rows(self) -> int:
        """Return how many rows were taken since the previous observation.

        Returns
        -------
        int
            The row count.
        """
        return int(self.z.size)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the statistics the channel carries, sorted.

        Returns
        -------
        tuple of str
            The catalogue names.
        """
        return tuple(sorted(self.values))


@dataclass(frozen=True, eq=False)
class Observation:
    """The state at a save plane, on the host.

    Section 9 of ``docs/dev/kernel-contract.md``. Host arrays only: device arrays never
    cross the seam. The field is in the **lab frame** and the FFT convention
    (``spectral_convention = "fft"``), so the output layer conjugates it on the way to
    disk and the processing accessors on the way to a user.

    Attributes
    ----------
    z : float
        Position of the plane, m.
    field : numpy.ndarray or None
        The state, complex128, shape ``(n_batch, n_modes, n_pol, n_omega)``, in the
        square root of watts; ``None`` for an observation that asked for no field
        (``observe(field=False)``, SEP 22 section 5.1).
    noise_phase : numpy.ndarray or None
        ``Im Phi(omega, z)`` in radians, same shape, when a noise model is active and
        the field is observed, so that processing can restore the shot-noise
        background the propagating field never carried. ``None`` otherwise.
    step_history : StepHistory
        What the stepper did since the previous observation.
    n_steps : int
        Steps taken since the previous observation, accepted and rejected together.
    n_rejected : int
        Rejected steps among them.
    step_stats : StepStats
        The rows a periodic statistics subset produced between the previous observation
        and this one, empty unless the run asked for one. Separate from
        ``step_history`` because the two are sampled differently: the history is every
        attempt, these are every ``period``-th accepted step.

    Raises
    ------
    BackendError
        If a field is not four-dimensional and finite, a noise phase is given without
        a field or does not match it, the step counts disagree with the history, or a
        statistics row points past the history it indexes.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import Observation, StepHistory
    >>> history = StepHistory(np.array([1.0]), np.array([1.0]), np.array([0.3]),
    ...                       np.array([True]), np.array([0.02]))
    >>> plane = Observation(1.0, np.zeros((1, 1, 1, 8), complex), None, history, 1, 0)
    >>> plane.z, plane.field.flags.writeable
    (1.0, False)
    """

    z: float
    field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None
    noise_phase: Float[np.ndarray, "n_batch_p n_modes_p n_pol_p n_omega_p"] | None
    step_history: StepHistory
    n_steps: int
    n_rejected: int
    # `dataclasses.field` qualified, because this class has an attribute called
    # `field` and a bare name would read as that one.
    step_stats: StepStats = dataclasses.field(default_factory=StepStats.empty)

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the plane.

        Raises
        ------
        BackendError
            Naming the offending field.
        """
        if self.field is None:
            if self.noise_phase is not None:
                msg = "a noise_phase is given without a field"
                raise BackendError(msg)
            self._check_counts()
            return
        state = np.ascontiguousarray(self.field, dtype=np.complex128)
        state.setflags(write=False)
        object.__setattr__(self, "field", state)
        if not np.all(np.isfinite(state)):
            bad = int(np.sum(~np.isfinite(state)))
            msg = (
                f"field has {bad} of {state.size} entries that are not finite at "
                f"z = {self.z!r} m; a diverged run must raise, not observe"
            )
            raise BackendError(msg)
        if self.noise_phase is not None:
            phase = np.ascontiguousarray(self.noise_phase, dtype=np.float64)
            phase.setflags(write=False)
            object.__setattr__(self, "noise_phase", phase)
            if phase.shape != state.shape:
                msg = (
                    f"noise_phase has shape {phase.shape}, expected {state.shape} to "
                    "match the field"
                )
                raise BackendError(msg)
        self._check_counts()

    def require_field(self) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the observed field, refusing an observation that asked for none.

        Returns
        -------
        numpy.ndarray
            The read-only field, shape ``(n_batch, n_modes, n_pol, n_omega)``.

        Raises
        ------
        BackendError
            If the observation carries no field (``observe(field=False)``).

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import Observation, StepHistory
        >>> empty = StepHistory(np.empty(0), np.empty(0), np.empty(0),
        ...                     np.empty(0, bool), np.empty(0))
        >>> Observation(0.0, np.ones((1, 1, 1, 2), complex), None, empty, 0, 0
        ...             ).require_field().shape
        (1, 1, 1, 2)
        """
        if self.field is None:
            msg = (
                f"the observation at z = {self.z!r} m carries no field: it was made "
                "with observe(field=False)"
            )
            raise BackendError(msg)
        return self.field

    def _check_counts(self) -> None:
        """Check the step counts against the history and the rows against both.

        Raises
        ------
        BackendError
            Naming the disagreement.
        """
        if self.n_steps != self.step_history.n_steps:
            msg = (
                f"n_steps is {self.n_steps} but the history records "
                f"{self.step_history.n_steps}"
            )
            raise BackendError(msg)
        if self.n_rejected != self.step_history.n_rejected:
            msg = (
                f"n_rejected is {self.n_rejected} but the history records "
                f"{self.step_history.n_rejected}"
            )
            raise BackendError(msg)
        rows = self.step_stats
        if rows.n_rows and int(rows.step[-1]) >= self.step_history.n_steps:
            msg = (
                f"a statistics row indexes step {int(rows.step[-1])} of a history with "
                f"{self.step_history.n_steps} attempts; the two are flushed together "
                "and must describe the same interval"
            )
            raise BackendError(msg)


@dataclass(frozen=True, eq=False)
class StepSummary:
    """What one call to :meth:`Backend.propagate_to` cost and achieved.

    A handful of host scalars, so that a driver can report progress and warn about a
    tolerance that was not met without pulling the whole step history across the seam
    every plane.

    Attributes
    ----------
    z : float
        Position reached, m, which is the plane that was asked for.
    n_steps : int
        Steps taken, accepted and rejected together.
    n_rejected : int
        Rejected steps among them.
    max_norm : float
        Largest scaled error norm over the accepted steps; 0 when none were taken.
    mean_norm : float
        Mean scaled error norm over the accepted steps.
    wall_time : float
        Seconds the call took.
    met_tolerance : bool
        Whether every accepted step scored at or below one. False on a replayed schedule
        too coarse for its tolerance, which the driver logs (``stepper.md`` section 8).

    Raises
    ------
    BackendError
        If a count is negative, more steps were rejected than taken, or a norm or time
        is negative or not finite.

    Examples
    --------
    >>> from solprop.backends.protocol import StepSummary
    >>> StepSummary(1.0, 42, 0, 0.7, 0.4, 0.31, True).met_tolerance
    True
    """

    z: float
    n_steps: int
    n_rejected: int
    max_norm: float
    mean_norm: float
    wall_time: float
    met_tolerance: bool

    def __post_init__(self) -> None:
        """Check the counters and the norms.

        Raises
        ------
        BackendError
            Naming the offending field.
        """
        if self.n_steps < 0 or not 0 <= self.n_rejected <= self.n_steps:
            msg = (
                f"the step counts must satisfy 0 <= n_rejected <= n_steps, got "
                f"{self.n_rejected!r} of {self.n_steps!r}"
            )
            raise BackendError(msg)
        for name in ("max_norm", "mean_norm", "wall_time"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0.0):
                msg = f"{name} must be non-negative and finite, got {value!r}"
                raise BackendError(msg)


@runtime_checkable
class Backend(Protocol):
    """What every execution backend provides, and nothing more.

    ``docs/dev/backends.md`` section 1. A backend is constructed from a validated
    :class:`solprop.translate.problem.KernelProblem`; the numbers are the contract, and
    a backend is correct when it reproduces the ``numpy_ref`` oracle to the stated
    bound.

    Attributes
    ----------
    name : str
        The registry name, for example ``"numpy_ref"``.
    max_batch : int
        Largest batch axis this backend steps at once; 1 for a backend that steps one
        problem.

    Examples
    --------
    >>> from solprop.backends.protocol import Backend
    >>> [name for name in sorted(dir(Backend)) if not name.startswith("_")][:4]
    ['available_device_memory', 'close', 'memory_ledger', 'metadata']
    >>> [name for name in sorted(dir(Backend)) if not name.startswith("_")][4:7]
    ['observe', 'propagate_to', 'restore_call_ledgers']
    >>> [name for name in sorted(dir(Backend)) if not name.startswith("_")][7:]
    ['set_initial', 'set_progress', 'stats']
    """

    name: str
    max_batch: int

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state and the noise realisation at a position, host to device, once.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout, FFT convention.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry along the run, or ``None``.
        z : float, optional
            The position the state belongs to, m, inside the operator table. Zero is
            the entrance; a positive value is a **resume** from a plane the run reached
            before, so the backend takes
            its position -- and its place in a replayed schedule -- from here. A state
            past the entrance already carries an ``input``-model noise realisation, and
            a backend must not add one to it a second time. The step controller starts
            afresh either way, so an adaptive run resumed agrees with an uninterrupted
            one to the controller's tolerance and not bitwise, while a replayed
            schedule agrees bitwise on a backend that recycles no stage and to
            round-off on one that does (``docs/dev/stepper.md`` section 3).
        """
        ...

    def set_progress(self, reporter: ProgressReporter) -> None:
        """Attach the reporter the step loop calls as it goes.

        The driver attaches one before the first plane and never changes it mid-run. A
        backend that steps on the host calls
        :meth:`~solprop.log.ProgressReporter.step` once per attempted step, including
        the rejected ones; a backend whose loop is on a device may call it once per
        plane instead (``docs/dev/backends.md`` section 1). Nothing a reporter does may
        change the propagation: a run with one is bitwise the run without.

        Parameters
        ----------
        reporter : solprop.log.ProgressReporter
            Where the step events go. :class:`solprop.log.NullReporter` is the way to
            turn them off.
        """
        ...

    def propagate_to(self, z: float) -> StepSummary:
        """Step the state to a position through the shared step loop.

        Parameters
        ----------
        z : float
            The save plane to reach, m. Steps land on it exactly.

        Returns
        -------
        StepSummary
            What the call cost and achieved.
        """
        ...

    def observe(self, *, field: bool = True) -> Observation:
        """Return the state at the current position, as host arrays.

        Parameters
        ----------
        field : bool, optional
            Whether the field and the noise phase cross to the host; ``False``
            observes the step history alone (SEP 22 section 5.1).

        Returns
        -------
        Observation
            The field, the noise phase and the step history since the last call.
        """
        ...

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Compute the requested statistics in one fused pass.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name, each with a leading batch axis.
        """
        ...

    def metadata(self) -> dict[str, object]:
        """Return what this execution is, for the run's record.

        A saved run has to say what produced it, not only what was computed: the
        thread configuration alone moved one step by a factor of nine
        (``docs/dev/backends.md`` section 3.4), so a result whose execution is not
        recorded is not reproducible. The driver writes this into
        :attr:`solprop.output.schema.RunMetadata.backend`, which is outside the resume
        key -- a run continued on the other backend is the same run.

        Returns
        -------
        dict
            JSON-serialisable scalars keyed by name. ``backend`` and
            ``propagator_mode`` are always present; pools, plan flags and wisdom
            digests are a backend's own.
        """
        ...

    def restore_call_ledgers(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore cumulative budgets after loading a saved state.

        Parameters
        ----------
        logical : mapping
            Persisted logical response counters, or empty for ordinary RK runs.
        atomic : mapping or None, optional
            Persisted provider attempts. Numerical history always restarts fresh.
        """
        ...

    def available_device_memory(self) -> int | None:
        """Return currently free device bytes, or None for a host backend.

        Returns
        -------
        int or None
            Free device memory excluding live allocations and allocator reserves.
            Host memory is admitted independently by the controller.
        """
        ...

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the backend holds.

        Returns
        -------
        dict
            Buffer name to bytes, including transform work areas. Every buffer is named
            even when it is empty.
        """
        ...

    def close(self) -> None:
        """Release the buffers and plans. Idempotent."""
        ...


@runtime_checkable
class BatchBackend(Backend, Protocol):
    """Advance compatible loaded members and return their separate observations.

    Construct through :func:`solprop.backends.registry.select_batch` after loading
    each member's initial state and noise through the ordinary backend seam.
    Successful construction transfers ownership of those backends to the batch.
    Their parameter tables may differ, but grids, fixed schedules, save planes and
    statistics layouts must match. ``max_batch`` records the constructed count.

    ``observe_members`` replaces ``observe`` for batch output: each member keeps
    its own error history and periodic rows. No device array crosses this interface.
    The common maximum error norm is diagnostic and never changes a fixed schedule.
    """

    def observe_members(self) -> tuple[Observation, ...]:
        """Drain independently owned host observations in member order.

        Returns
        -------
        tuple of Observation
            Singleton field shapes and each member's own diagnostic history.
        """
        ...

    def member_metadata(self) -> tuple[dict[str, object], ...]:
        """Return resolved execution policies for every batch member.

        Returns
        -------
        tuple of dict
            Complete per-member backend provenance, including the batch strategy.
        """
        ...

    def attach_monitors(self, monitors: Sequence[BoundaryMonitor | None]) -> None:
        """Give each member its own boundary monitor, or none (SEP 21).

        Parameters
        ----------
        monitors : collections.abc.Sequence
            One monitor with nothing recorded, or ``None``, per member.
        """
        ...

    @property
    def member_stops(self) -> tuple[BoundaryError | None, ...]:
        """Return what stopped each member's watch, ``None`` where nothing did.

        Returns
        -------
        tuple
            One entry per member, in member order.
        """
        ...

    def finish_monitors(self) -> None:
        """Report each still-watched member's removed fraction at the run's end."""
        ...

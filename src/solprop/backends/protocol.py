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
:class:`solprop.translate.problem.PropagationProblem`, takes host arrays and returns
host arrays; its buffers, layout and plans are private, and the state is never a
parameter of anything a run's session is asked. Selection by name and the availability
rules are in :mod:`solprop.backends.registry`.

**Sign convention.** Every complex array crossing this seam, in either direction, is in
the **FFT convention** of ``docs/dev/conventions.md`` section 2, with the phase origin
at the first time sample: ``spectral_convention = "fft"``. The output layer conjugates
on the way to disk and the processing accessors on the way to a user. Backends
never conjugate.
"""

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop._ownership import frozen_copy, hold
from solprop._process import parallel_rows
from solprop.exceptions import BackendError

if TYPE_CHECKING:
    # The session module builds on this one's records, so its start is a forward name.
    from solprop.backends.session import PreparedExecution, Start
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest

__all__ = [
    "Backend",
    "BackendHistory",
    "Observation",
    "StepStats",
    "StepSummary",
    "TimingProbe",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class BackendHistory:
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
        The scaled embedded error norm of each step as executed, float64, shape
        ``(n_steps,)``: a clipped or stretched step records the norm it actually
        scored, the same norm that decides its acceptance. The controller's own
        history holds the proposal's equivalent norm instead
        (:meth:`solprop.stepper.controller.PIController.equivalent_norm`;
        ``docs/dev/stepper.md`` section 3), which is never recorded here. A rejected
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
    >>> from solprop.backends.protocol import BackendHistory
    >>> history = BackendHistory(np.array([0.1, 0.2]), np.array([0.1, 0.1]),
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
            # At least one-dimensional, as numpy.ascontiguousarray made it: a scalar
            # stands for one entry.
            values = frozen_copy(np.atleast_1d(getattr(self, name)), dtype)
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
    def n_steps(self) -> int:  # numpydoc ignore=RT01
        """Number of steps recorded, accepted and rejected together.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import BackendHistory
        >>> BackendHistory(*(np.zeros(0) for _ in range(3)),
        ...             np.zeros(0, bool), np.zeros(0)).n_steps
        0
        """
        return int(self.z.size)

    @property
    def n_rejected(self) -> int:  # numpydoc ignore=RT01
        """Number of rejected steps.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import BackendHistory
        >>> BackendHistory(np.array([0.1]), np.array([0.1]), np.array([2.0]),
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
        observation's* :class:`BackendHistory`, int64, shape ``(n_rows,)``, increasing.
        The history counts rejected attempts too, so this is not ``row * period``.
    values : collections.abc.Mapping
        The statistics, keyed by catalogue name, each with a leading row axis and then
        the statistic's own shape: ``(n_rows, *shape)``.

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
    ...                  {"energy": np.ones((2, 1, 1))})
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
        # At least one-dimensional, as numpy.ascontiguousarray made them: a scalar
        # position is one row.
        positions = frozen_copy(np.atleast_1d(self.z), np.float64)
        indices = frozen_copy(np.atleast_1d(self.step), np.int64)
        object.__setattr__(self, "z", positions)
        object.__setattr__(self, "step", indices)
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
            # A copy, unless solprop already holds the rows; it keeps a scalar
            # zero-dimensional, so a scalar cannot pass as a single row.
            row = frozen_copy(values)
            if row.ndim == 0 or row.shape[0] != positions.size:
                msg = (
                    f"the {name!r} rows have leading axis {row.shape}, expected "
                    f"{positions.size} rows as z has"
                )
                raise BackendError(msg)
            adopted[name] = row
        object.__setattr__(self, "values", adopted)

    @classmethod
    def empty(cls) -> "StepStats":
        """Return the channel a run without a periodic subset carries.

        Returns
        -------
        StepStats
            No rows and no statistics.
        """
        return cls(hold(np.empty(0)), hold(np.empty(0, dtype=np.int64)), {})

    @property
    def n_rows(self) -> int:  # numpydoc ignore=RT01
        """How many rows were taken since the previous observation."""
        return int(self.z.size)

    @property
    def names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics the channel carries, sorted."""
        return tuple(sorted(self.values))


def _all_finite(state: Complex[np.ndarray, "*shape n_omega"]) -> bool:
    """Return whether every entry of a complex state is finite, allocating no mask.

    A sum holding a NaN or an infinity isn't finite, so a finite sum settles it in one
    pass over the rows. Only a sum that overflows takes the second look, which allows
    for finite entries too large to add.

    - ``state``: a contiguous complex state.
    - Returns: True when no entry is a NaN or an infinity.
    """
    if state.size == 0:
        return True
    rows = state.reshape(-1, state.shape[-1])

    def finite(selected: slice) -> bool:
        """Return whether the selected rows are finite.

        - Returns: True when none holds a NaN or an infinity.
        """
        part = rows[selected]
        # Per thread: a sum that overflows is what the second look is for.
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.sum(part)
        return bool(np.isfinite(total)) or bool(np.all(np.isfinite(part)))

    return all(parallel_rows(rows.shape[0], state.nbytes, finite))


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
        The state, complex128, shape ``(n_modes, n_pol, n_omega)``, in the square root
        of watts; ``None`` for a field-free observation, one that drained the step
        attempts without reading the state (SEP 22 section 5.1).
    noise_phase : numpy.ndarray or None
        ``Im Phi(omega, z)`` in radians, same shape, when a noise model is active and
        the field is observed, so that processing can restore the shot-noise
        background the propagating field never carried. ``None`` otherwise.
    step_history : BackendHistory
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
        If a field is not finite, a noise phase is given without
        a field or does not match it, the step counts disagree with the history, or a
        statistics row points past the history it indexes.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import Observation, BackendHistory
    >>> history = BackendHistory(np.array([1.0]), np.array([1.0]), np.array([0.3]),
    ...                       np.array([True]), np.array([0.02]))
    >>> plane = Observation(1.0, np.zeros((1, 1, 8), complex), None, history, 1, 0)
    >>> plane.z, plane.field.flags.writeable
    (1.0, False)
    """

    z: float
    field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None
    noise_phase: Float[np.ndarray, "n_modes_p n_pol_p n_omega_p"] | None
    step_history: BackendHistory
    n_steps: int
    n_rejected: int
    # `dataclasses.field` qualified, because this class has an attribute called
    # `field` and a bare name would read as that one.
    step_stats: StepStats = dataclasses.field(default_factory=StepStats.empty)

    def __post_init__(self) -> None:
        """Freeze the arrays, copying any solprop does not already hold, and check.

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
        # A backend holds what it hands over (solprop._ownership.hold), which is then
        # taken as it stands; anything else, as from a caller, is copied.
        state = frozen_copy(self.field, np.complex128)
        object.__setattr__(self, "field", state)
        if not _all_finite(state):
            bad = int(np.sum(~np.isfinite(state)))
            msg = (
                f"field has {bad} of {state.size} entries that are not finite at "
                f"z = {self.z!r} m; a diverged run must raise, not observe"
            )
            raise BackendError(msg)
        if self.noise_phase is not None:
            phase = frozen_copy(self.noise_phase, np.float64)
            object.__setattr__(self, "noise_phase", phase)
            if phase.shape != state.shape:
                msg = (
                    f"noise_phase has shape {phase.shape}, expected {state.shape} to "
                    "match the field"
                )
                raise BackendError(msg)
        self._check_counts()

    def require_field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the observed field, refusing an observation that asked for none.

        Returns
        -------
        numpy.ndarray
            The read-only field, shape ``(n_modes, n_pol, n_omega)``.

        Raises
        ------
        BackendError
            If the observation is field-free.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.backends.protocol import Observation, BackendHistory
        >>> empty = BackendHistory(np.empty(0), np.empty(0), np.empty(0),
        ...                     np.empty(0, bool), np.empty(0))
        >>> Observation(0.0, np.ones((1, 1, 2), complex), None, empty, 0, 0
        ...             ).require_field().shape
        (1, 1, 2)
        """
        if self.field is None:
            msg = (
                f"the observation at z = {self.z!r} m carries no field: it drained the "
                "step attempts alone, without reading the state"
            )
            raise BackendError(msg)
        return self.field

    def _check_counts(self) -> None:
        """Check the step counts against the history and the rows against both."""
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
    """What one march of a session towards a plane cost and achieved.

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
class TimingProbe(Protocol):
    """Single trials of a loaded run, for an estimate or a calibration to time.

    Returned by :meth:`Backend.timing_probe` (SEP 47 section 6.6). A trial is one
    step's arithmetic from the loaded state, and commits nothing: the backend's
    state, position and step history stay those of the load, though a work counter
    may count the trial's evaluations. It is a supported internal operation for
    :meth:`solprop.driver.simulation.Simulation.estimate` and the SEP 46 calibration,
    not a propagation executor: a run is driven through a session
    (:mod:`solprop.backends.session`).

    The low-storage march (SEP 24) has no separate trial: its step accepts in place, so
    a probe's trial advances the discarded registers the load put there, and it has no
    trial field to read back. A backend probed so is closed afterwards.
    """

    def trial(self, z: float | None = None, dz: float | None = None) -> None:
        """Take one trial step and keep its end state for :meth:`field`.

        Parameters
        ----------
        z : float or None, optional
            Where the trial starts, m; the loaded position by default.
        dz : float or None, optional
            Its length, m, positive; the problem's ``dz_init`` by default, or the
            scheduled step at `z` for the low-storage march.
        """
        ...

    def field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the last trial's end state on the host, as a kept step would be.

        The state after the accepted-step boundary treatments, read once per trial. A
        backend that can read it back only by finishing the step in its own state
        loads the loaded state again afterwards, so the probe still commits nothing.

        Returns
        -------
        numpy.ndarray
            An owning copy in internal units and the FFT convention.

        Raises
        ------
        solprop.exceptions.BackendError
            With no trial since the last readback, or for the low-storage march, which
            has no trial state.
        """
        ...


@runtime_checkable
class Backend(Protocol):
    """What every execution backend provides, and nothing more.

    ``docs/dev/backends.md`` section 1. A backend is constructed from a validated
    :class:`solprop.translate.problem.PropagationProblem` and the
    :class:`solprop.translate.run_request.RunRequest` it is run with (SEP 47 section
    5.2); the numbers are the contract, and a backend is correct when it reproduces the
    ``numpy_ref`` oracle to the stated bound.

    Attributes
    ----------
    name : str
        The registry name, for example ``"numpy_ref"``.

    Examples
    --------
    >>> from solprop.backends.protocol import Backend
    >>> names = [name for name in sorted(dir(Backend)) if not name.startswith("_")]
    >>> names[:4]
    ['close', 'closed', 'edge_memory', 'memory_ledger']
    >>> names[4:]
    ['metadata', 'prepare', 'request', 'timing_probe']
    """

    name: str

    @property
    def request(self) -> "RunRequest":  # numpydoc ignore=RT01
        """Run request the backend was built for.

        Its method, error control, itinerary, planes and statistics; a run adapts it
        with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        ...

    def prepare(
        self,
        problem: "PropagationProblem",
        request: "RunRequest",
    ) -> "PreparedExecution":
        """Prepare a run for one problem and request (SEP 47 section 5.2).

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The problem the backend was built for, or one derived from it with
            another input.
        request : solprop.translate.run_request.RunRequest
            The run's request: the backend's own, or one run's adaptation of it.

        Returns
        -------
        solprop.backends.session.PreparedExecution
            The prepared execution, whose ``start`` opens the backend's session.
        """
        ...

    def metadata(self) -> dict[str, object]:
        """Return what this execution is, for the run's record.

        A saved run has to say what produced it, not only what was computed: the
        thread configuration alone moved one step by a factor of nine
        (``docs/dev/backends.md`` section 3.4), so a result whose execution is not
        recorded is not reproducible. The driver writes this into
        :attr:`solprop.output.metadata.RunMetadata.backend`, which is outside the resume
        key -- a run continued on the other backend is the same run.

        Returns
        -------
        dict
            JSON-serialisable scalars keyed by name. ``backend`` and
            ``propagator_mode`` are always present; pools, plan flags and wisdom
            digests are a backend's own.
        """
        ...

    def edge_memory(self, space: object) -> int:
        """Return what reading the transverse edge holds in the state's memory space.

        Parameters
        ----------
        space : object
            The run's transverse grid.

        Returns
        -------
        int
            The bytes, from the grid alone; zero for a read on the host.
        """
        ...

    def timing_probe(
        self,
        start: "Start",
    ) -> TimingProbe:
        """Load a start and return single trials of this run's march from it.

        The start's field, noise, position and resume flag are loaded as a session
        loads them; its method history, ledgers and monitor continuation are not read.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the trials start; its field is taken.

        Returns
        -------
        TimingProbe
            The probe (SEP 47 section 6.6).
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

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the backend has released its buffers, as a failure may."""
        ...

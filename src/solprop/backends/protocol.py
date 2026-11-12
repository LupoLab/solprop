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
parameter of :meth:`HostStepping.propagate_to` or :meth:`HostStepping.observe`.
Selection by name and the availability rules are in :mod:`solprop.backends.registry`.

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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop._ownership import frozen_copy
from solprop._process import parallel_rows
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.log import ProgressReporter, StopFlag
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.schema import MarchSnapshot
from solprop.stats import StatsSpec

if TYPE_CHECKING:
    # The session module builds on this one's records, so its start is a forward name.
    from solprop.backends.session import PreparedExecution, PreparedGroup, Start
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest

__all__ = [
    "Backend",
    "BackendHistory",
    "BatchBackend",
    "HostStepping",
    "Observation",
    "StepStats",
    "StepSummary",
    "TimingProbe",
    "stepping",
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
    def n_steps(self) -> int:
        """Return the number of steps recorded, accepted and rejected together.

        Returns
        -------
        int
            The length of ``z``.

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
    def n_rejected(self) -> int:
        """Return the number of rejected steps.

        Returns
        -------
        int
            How many entries of ``accepted`` are false.

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
        the statistic's own shape, with no batch axis: ``(n_rows, *shape)``.

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


def _all_finite(state: Complex[np.ndarray, "*shape n_omega"]) -> bool:
    """Return whether every entry of a complex state is finite, allocating no mask.

    A sum holding a NaN or an infinity isn't finite, so a finite sum settles it in one
    pass over the rows. Only a sum that overflows takes the second look, which allows
    for finite entries too large to add.

    Parameters
    ----------
    state : numpy.ndarray
        A contiguous complex state.

    Returns
    -------
    bool
        True when no entry is a NaN or an infinity.
    """
    if state.size == 0:
        return True
    rows = state.reshape(-1, state.shape[-1])

    def finite(selected: slice) -> bool:
        """Return whether the selected rows are finite.

        Parameters
        ----------
        selected : slice
            The rows.

        Returns
        -------
        bool
            True when none holds a NaN or an infinity.
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
        of watts; ``None`` for an observation that asked for no field
        (``observe(field=False)``, SEP 22 section 5.1).
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
    field_owned : bool
        Whether the backend made `field` for this observation alone and keeps no
        reference to it: a copy from a device or a widened single-precision state,
        made fresh at each call. ``False`` where the field may be the backend's own
        state array, frozen and left behind by the next step, and always ``False``
        without a field. The session that observed it, its only holder, converts an
        owned field to output units where it is, rather than into a new array
        (:meth:`solprop.backends.host.session.HostSession.export`); for any other
        field that conversion is what detaches it from the backend (SEP 47
        section 5.4).

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
    field_owned: bool = False

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
            object.__setattr__(self, "field_owned", False)
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
            If the observation carries no field (``observe(field=False)``).

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
    """What one call to :meth:`HostStepping.propagate_to` cost and achieved.

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
    max_batch : int
        Trajectories this backend steps at once: 1 for a backend that steps one
        problem, the member count for a group runner.
    reads_transverse_edge : bool
        Whether it reads a free-space run's transverse edge (SEP 40 section 5.3), which
        a run with a requested real-space collar needs.

    Examples
    --------
    >>> from solprop.backends.protocol import Backend
    >>> names = [name for name in sorted(dir(Backend)) if not name.startswith("_")]
    >>> names[:4]
    ['available_device_memory', 'close', 'closed', 'edge_memory']
    >>> names[4:8]
    ['march_export_refusal', 'memory_ledger', 'metadata', 'prepare']
    >>> names[8:]
    ['prepare_group', 'request', 'timing_probe']
    """

    name: str
    max_batch: int
    reads_transverse_edge: bool

    @property
    def request(self) -> "RunRequest":  # noqa: UP037 - the request module imports this one's
        """Return the run request the backend was built for.

        Its method, error control, itinerary, planes and statistics; a run adapts it
        with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        ...

    def prepare(
        self,
        problem: "PropagationProblem",  # noqa: UP037 - the translation imports this one
        request: "RunRequest",  # noqa: UP037
    ) -> "PreparedExecution":  # noqa: UP037 - the session module imports this one
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

    def prepare_group(
        self,
        members: "Sequence[Backend]",  # noqa: UP037 - bound after the class body
        problems: "Sequence[PropagationProblem]",  # noqa: UP037
        requests: "Sequence[RunRequest]",  # noqa: UP037
    ) -> "PreparedGroup":  # noqa: UP037 - the session module imports this one
        """Prepare a group of compatible members of this backend (SEP 47 section 6.7).

        Called on the group's first member, ``members[0]``, so that the backend, not
        the run layer, chooses the group it runs
        (:func:`solprop.backends.sessions.start_group`). A backend that runs no group
        raises :class:`~solprop.exceptions.BackendError` saying so.

        Parameters
        ----------
        members : collections.abc.Sequence of Backend
            One constructed backend per member, this one first, all of this backend,
            holding no state yet.
        problems : sequence of solprop.translate.problem.PropagationProblem
            Each member's independently translated problem.
        requests : sequence of solprop.translate.run_request.RunRequest
            Each member's run request, with its own fresh boundary monitor.

        Returns
        -------
        solprop.backends.session.PreparedGroup
            The prepared group, whose ``start`` opens the backend's group session.
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

    def march_export_refusal(self) -> str | None:
        """Return why this backend's Adams march cannot carry its history, or ``None``.

        Returns
        -------
        str or None
            The reason, or ``None`` when it can or when the march keeps none.
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
        start: "Start",  # noqa: UP037 - the session module imports this one
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

    @property
    def closed(self) -> bool:
        """Return whether the backend has released its buffers, as a failure may.

        Returns
        -------
        bool
            ``True`` when it can run no other input.
        """
        ...


@runtime_checkable
class HostStepping(Backend, Protocol):
    """The host stepping seam a host-owned session drives (SEP 47 section 6.6).

    The reference (:class:`solprop.backends.host.driver.HostDriver`) loads, steps,
    observes and restores its state through these members, which its own session
    (:class:`solprop.backends.host.session.HostSession`) and its lockstep group call.
    They are no part of :class:`Backend`: the native backend runs only in its native
    session and has none of them (SEP 47 section 6.8). A caller holding a
    :class:`Backend` it knows steps on the host narrows it with :func:`stepping`.

    Examples
    --------
    >>> from solprop.backends.protocol import Backend, HostStepping
    >>> added = set(dir(HostStepping)) - set(dir(Backend))
    >>> sorted(name for name in added if not name.startswith("_"))[:4]
    ['attach_monitor', 'in_flight', 'march_snapshot', 'march_snapshot_nbytes']
    """

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
        resume: bool = False,
    ) -> None:
        """Load a state and the noise realisation at a position, host to device, once.

        Parameters
        ----------
        field : numpy.ndarray
            The state, shape ``(n_modes, n_pol, n_omega)``, FFT convention.
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
        resume : bool, optional
            Whether `field` came back from a saved plane of this run, and so already
            carries the ``input`` realisation even at the launch position, where a file
            holding only that plane resumes. ``False`` by default.
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

    def set_stop(self, flag: StopFlag | None) -> None:
        """Attach the flag the march polls, or detach it with ``None``.

        Parameters
        ----------
        flag : solprop.log.StopFlag or None
            Any object with an ``is_set()`` method, such as a
            :class:`threading.Event`.
        """
        ...

    def attach_monitor(self, monitor: BoundaryMonitor) -> None:
        """Watch the time window's edges from here on, starting with the current state.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            The monitor, fed every committed step.
        """
        ...

    def propagate_to(self, z: float) -> StepSummary:
        """Step the state to a position through the shared driver's marches.

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

        Nothing writes to the returned arrays again, the backend included: a run hands
        them to its output without a copy (:func:`solprop.driver.run.run_problem`), and
        a handler may keep them. A backend either returns arrays it no longer owns or
        leaves the ones it returns behind, as a step that replaces its state does.

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

    @property
    def in_flight(self) -> tuple[float, float] | None:
        """Return the low-storage step a failure abandoned in flight, or ``None``.

        Returns
        -------
        tuple of float or None
            The step's start and length, m, while the state is neither of its ends.
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
            Small host arrays keyed by statistic name, without a batch axis.
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

    def march_snapshot(self) -> MarchSnapshot | None:
        """Return the Adams march's state at the accepted position, for a resume.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            The snapshot, or ``None`` wherever the march has nothing to continue.
        """
        ...

    def march_snapshot_nbytes(self, z: float) -> int | None:
        """Return the bytes :meth:`march_snapshot` would copy at `z`, copying nothing.

        Parameters
        ----------
        z : float
            The position the snapshot is wanted at, m.

        Returns
        -------
        int or None
            The size, or ``None`` where a snapshot would be ``None`` or stand elsewhere.
        """
        ...

    def restore_march(self, snapshot: MarchSnapshot) -> str | None:
        """Continue the Adams march from a snapshot at the loaded position.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
            The snapshot a file holds.

        Returns
        -------
        str or None
            ``None`` when the history was continued, otherwise why it starts afresh.
        """
        ...


@runtime_checkable
class BatchBackend(HostStepping, Protocol):
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

    def observe_members(self, *, field: bool = True) -> tuple[Observation, ...]:
        """Drain independently owned host observations in member order.

        Parameters
        ----------
        field : bool, optional
            Whether each member's field and noise phase cross to the host, as for
            :meth:`solprop.backends.protocol.HostStepping.observe`. Default ``True``.

        Returns
        -------
        tuple of Observation
            Singleton field shapes, or no field, and each member's own diagnostic
            history.
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
    def member_stops(self) -> tuple[BoundaryError | BoundaryWarning | None, ...]:
        """Return what stopped each member's watch, ``None`` where nothing did.

        A stop, or a warning raised as an error under an ``"error"`` filter.

        Returns
        -------
        tuple
            One entry per member, in member order.
        """
        ...

    @property
    def member_records(self) -> tuple[Mapping[str, object] | None, ...]:
        """Return each member's monitor record as it closed, ``None`` if unwatched.

        Returns
        -------
        tuple
            One entry per member, in member order.
        """
        ...

    def finish_monitors(self) -> None:
        """Report each still-watched member's removed fraction at the run's end.

        A warning raised as an error fails that member alone (:attr:`member_stops`).
        """
        ...


def stepping(backend: Backend) -> HostStepping:
    """Return a backend as the host stepping seam it implements.

    Parameters
    ----------
    backend : Backend
        A backend that steps on the host: the reference, or a test's scripted one.

    Returns
    -------
    HostStepping
        The same object.

    Raises
    ------
    solprop.exceptions.BackendError
        If it does not implement the seam: the native backend runs only in its
        session (SEP 47 section 6.8).
    """
    if not isinstance(backend, HostStepping):
        msg = (
            f"{backend.name} runs in its own session, not through the host stepping "
            "seam: start a run with backend.prepare(problem, request).start(start)"
        )
        raise BackendError(msg)
    return backend

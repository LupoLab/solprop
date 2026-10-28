# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Establish fixed-schedule lockstep with independent backend workspaces.

This runner is the correctness baseline for fused batch kernels. It executes the
selected backend's existing kernels for each member at each shared step. It does
not amortise GPU kernel launches and is not a device-batching performance claim.
"""

import copy
import functools
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import cast

import numpy as np
from jaxtyping import Complex

from solprop.backends.driver import StepDriver, replay_tolerance
from solprop.backends.protocol import Backend, Observation
from solprop.exceptions import BackendError, BoundaryError
from solprop.monitor import BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import KernelProblem

__all__ = ["LockstepRunner"]


def _refuse_invalid_norm(norms: Sequence[float], backend: str) -> None:
    """Name the member whose diagnostic no step record on any route can carry.

    A batch only replays, so every step it records is accepted, and a non-finite or
    negative error norm on an accepted step is refused by ``StepHistory`` and
    ``StepSummary`` at the next save plane whether the member ran in a batch or
    alone, so this is not a batch-only rule; what the batch owes the caller is the
    member's index, rather than an anonymous history error one plane later that
    the reader cannot attribute to a point.

    Parameters
    ----------
    norms : collections.abc.Sequence
        Each member's own scaled error norm for the step just evaluated,
        dimensionless, in member order.
    backend : str
        Name of the batch runner, for the message.

    Raises
    ------
    BackendError
        Naming the first member with an invalid norm and the value it returned.
    """
    for index, norm in enumerate(norms):
        if not (np.isfinite(norm) and norm >= 0.0):
            raise BackendError(
                f"{backend} batch member {index} returned an invalid error norm "
                f"{norm!r}, which its own step record also refuses"
            )


class LockstepRunner(StepDriver[tuple[object, ...]]):
    """Advance loaded member workspaces through one shared fixed-schedule march.

    Each workspace retains its own parameter tables, noise, error estimate and
    statistics storage hooks. The maximum norm is a batch diagnostic only: fixed
    replay does not use it to change any member's schedule. Member observations
    carry their own error histories, never that maximum.

    The caller validates layout compatibility and admits memory before constructing
    this runner. Ownership of the loaded workspaces transfers to the runner after
    successful construction; ``close`` releases every member.

    Each member watches its own time window (SEP 21). A member whose monitor stops it
    stops watching while the others go on, and :attr:`member_stops` says which, so
    that a batched scan fails that point alone, as a run of it alone would fail.

    Parameters
    ----------
    members : sequence of solprop.backends.driver.StepDriver
        Nonempty sequence of loaded singleton backends with the same name and fixed
        schedule. All must be at the entrance with no previously attempted steps.

    Raises
    ------
    BackendError
        If members mix backends, lack a common fixed schedule, are already advanced,
        or have different save planes or periodic-statistics requests.
    """

    def __init__(self, members: Sequence[StepDriver[object]]) -> None:
        self._members: tuple[StepDriver[object], ...] = tuple(members)
        _validate_members(self._members)
        first = self._members[0]
        super().__init__(first._problem)
        self.name = first.name
        self.max_batch = len(self._members)
        self._state = tuple(member._require_state() for member in self._members)
        self._member_norms: tuple[float, ...] = ()
        self._stops: list[BoundaryError | None] = [None] * len(self._members)

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> tuple[object, ...]:
        """Refuse loading a shared field in place of independently loaded members.

        Parameters
        ----------
        state : numpy.ndarray
            Unused shared initial field.

        Raises
        ------
        BackendError
            Always; member owners must be initialised before construction.
        """
        del state
        raise BackendError("LockstepRunner requires independently loaded members")

    def _trial_step(self, z: float, dz: float) -> tuple[float, tuple[object, ...]]:
        """Evaluate independent trial states at one common fixed step.

        Parameters
        ----------
        z : float
            Start position, m.
        dz : float
            Positive step length, m.

        Returns
        -------
        tuple
            Maximum diagnostic norm and independently owned trial states.
        """
        trials = tuple(member._trial_step(z, dz) for member in self._members)
        self._member_norms = tuple(norm for norm, _ in trials)
        _refuse_invalid_norm(self._member_norms, "lockstep")
        return max(self._member_norms), tuple(state for _, state in trials)

    def _replay_to(self, schedule: FixedSchedule, target: float) -> None:
        """Replay the shared schedule; members stepped by their own march each own one.

        Runge-Kutta members share the base replay through :meth:`_trial_step`. An
        Adams member's history, order and ledger are its own, and a low-storage
        member's registers and flow table (SEP 24), so each takes the common step
        through its own march
        (:meth:`solprop.backends.multistep.AdamsMarch.replay_step`,
        :meth:`solprop.backends.low_storage.LowStorageMarch.replay_step`) and the
        shared itinerary is recorded once, with each member's own norm. A member left
        in flight by an abandoned step refuses the next one.

        Parameters
        ----------
        schedule : solprop.stepper.schedule.FixedSchedule
            The common schedule.
        target : float
            Position to reach, m.

        Raises
        ------
        BackendError
            If the schedule runs out before the target is reached.
        """
        marches = [
            member._adams if member._adams is not None else member._low_storage
            for member in self._members
        ]
        if marches[0] is None:
            super()._replay_to(schedule, target)
            return
        edges = schedule.edges
        slack = replay_tolerance(schedule, target)
        # Before any member steps: a member left in flight refuses, and the members
        # ahead of it in the batch must not be advanced again past the itinerary.
        for member in self._members:
            member._refuse_in_flight()
        while self._schedule_index < schedule.n_steps and self._z < target - slack:
            dz = float(schedule.steps[self._schedule_index])
            end = float(edges[self._schedule_index + 1])
            self._member_norms = tuple(
                march.replay_step(dz, end, slack)
                for march in marches
                if march is not None
            )
            _refuse_invalid_norm(self._member_norms, "lockstep")
            self._state = tuple(member._require_state() for member in self._members)
            self._schedule_index += 1
            self._z = end
            self._record(end, dz, max(self._member_norms), accepted=True)
            self._accepted_norms.append(max(self._member_norms))
            self._take_step_stats()
            if self._natural_stop(target):
                return
        if self._z < target - slack:
            msg = (
                f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                f"cannot reach z = {target:.6g} m"
            )
            raise BackendError(msg)

    def _finish(self, state: tuple[object, ...], dz: float) -> tuple[object, ...]:
        """Apply each member's accepted-step boundary handling.

        Parameters
        ----------
        state : tuple
            Trial states in member order.
        dz : float
            Accepted step length, m.

        Returns
        -------
        tuple
            Independently finished states, also installed on their owners.
        """
        finished = tuple(
            member._finish(value, dz)
            for member, value in zip(self._members, state, strict=True)
        )
        for member, value in zip(self._members, finished, strict=True):
            member._state = value
        return finished

    def _record(self, z: float, dz: float, norm: float, accepted: bool) -> None:
        """Record the common itinerary and each member's own error estimate.

        Parameters
        ----------
        z : float
            End position, m.
        dz : float
            Step length, m.
        norm : float
            Maximum member diagnostic norm.
        accepted : bool
            Whether the shared step was accepted.
        """
        super()._record(z, dz, norm, accepted)
        for member, own_norm in zip(self._members, self._member_norms, strict=True):
            member._z = z
            member._schedule_index = self._schedule_index
            member._record(z, dz, own_norm, accepted)
            member._accepted_norms.append(own_norm)

    def attach_monitor(self, monitor: BoundaryMonitor) -> None:
        """Give every member its own copy of the boundary monitor (SEP 21).

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor
            A monitor with nothing recorded, copied per member.
        """
        self.attach_monitors([copy.deepcopy(monitor) for _ in self._members])

    def attach_monitors(self, monitors: Sequence[BoundaryMonitor | None]) -> None:
        """Give each member its own boundary monitor, or none (SEP 21).

        A launch the monitor refuses stops that member's watch alone
        (:attr:`member_stops`).

        Parameters
        ----------
        monitors : collections.abc.Sequence
            One monitor with nothing recorded, or ``None``, per member, in member
            order.

        Raises
        ------
        BackendError
            If the count differs from the members'.
        """
        if len(monitors) != len(self._members):
            raise BackendError(
                f"{len(monitors)} boundary monitors for {len(self._members)} members"
            )
        for index, (member, monitor) in enumerate(
            zip(self._members, monitors, strict=True)
        ):
            if monitor is not None:
                self._watched(index, functools.partial(member.attach_monitor, monitor))

    @property
    def member_stops(self) -> tuple[BoundaryError | None, ...]:
        """Return what stopped each member's watch, in member order.

        Returns
        -------
        tuple
            The :class:`~solprop.exceptions.BoundaryError` that stopped a member, or
            ``None`` for one still watched or never watched.
        """
        return tuple(self._stops)

    def finish_monitors(self) -> None:
        """Report each still-watched member's removed fraction at the run's end."""
        for member in self._members:
            if member._monitor is not None:
                member._monitor.finish()

    def _watched(self, index: int, call: Callable[[], None]) -> None:
        """Make one member's monitor call; a stop ends that member's watch alone.

        The member's march goes on with the others, unwatched and with its collar in
        every stage; its result is the caller's to discard.

        Parameters
        ----------
        index : int
            The member's position.
        call : collections.abc.Callable
            The call, which may raise the member's stop.
        """
        member = self._members[index]
        try:
            call()
        except BoundaryError as error:
            self._stops[index] = error
            member._monitor = None
            # The member's backend stops reading for it, a split's removal included.
            member._bind_monitor(None)
            member._set_collar_quiet(False)

    def _monitor_plane(self) -> None:
        """Close each member's boundary-monitor row for the plane just reached."""
        for index, member in enumerate(self._members):
            self._watched(index, member._monitor_plane)

    def _take_step_stats(self) -> None:
        """Collect periodic rows through each backend's storage hooks."""
        for index, member in enumerate(self._members):
            self._watched(index, member._take_step_stats)

    def _begin_plane(self) -> None:
        """Reset each member's observation interval without releasing buffers."""
        for member in self._members:
            member._reset_step_stats(release=False)
            member._begin_plane()

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect member statistics with a leading batch axis.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Names and windows evaluated independently by each member.

        Returns
        -------
        dict of str to numpy.ndarray
            Host statistics concatenated on their existing batch axes.
        """
        self._require_state()
        values = [member.stats(spec) for member in self._members]
        if any(value.keys() != values[0].keys() for value in values):
            raise BackendError("lockstep members returned different statistic names")
        return {
            name: np.concatenate([value[name] for value in values], axis=0)
            for name in values[0]
        }

    def observe_members(self) -> tuple[Observation, ...]:
        """Drain each member's host observation and its own diagnostic history.

        Returns
        -------
        tuple of solprop.backends.protocol.Observation
            Singleton observations in batch order, with unchanged field shapes and
            independently retained periodic rows.
        """
        self._require_state()
        observations = tuple(member.observe() for member in self._members)
        self._history = []
        return observations

    def observe(self, *, field: bool = True) -> Observation:  # noqa: ARG002 - refused
        """Refuse an observation that would substitute a shared member history.

        Parameters
        ----------
        field : bool, optional
            Unused: the observation is refused either way.

        Raises
        ------
        BackendError
            Always; use ``observe_members`` to retain each member's error norms.
        """
        raise BackendError("use observe_members to preserve individual error histories")

    def available_device_memory(self) -> int | None:
        """Read the selected device's free bytes through its first member.

        Returns
        -------
        int or None
            Free device bytes, or None for a host batch.
        """
        self._check_open()
        return self._members[0].available_device_memory()

    def metadata(self) -> dict[str, object]:
        """Describe the lockstep execution strategy without implying fused kernels.

        Returns
        -------
        dict
            Selected backend, member count and workspace strategy.
        """
        return {
            "backend": self.name,
            "batch_size": self.max_batch,
            "batch_strategy": "member_workspaces",
            "fused_batch_kernels": False,
        }

    def member_metadata(self) -> tuple[dict[str, object], ...]:
        """Return each member's resolved policies with the shared batch strategy.

        Returns
        -------
        tuple of dict
            Complete backend provenance in member order.
        """
        return tuple(member.metadata() | self.metadata() for member in self._members)

    def memory_ledger(self) -> dict[str, int]:
        """Name each member's workspace allocations separately.

        Returns
        -------
        dict of str to int
            Backend allocation bytes under member-prefixed names. Shared translation
            products are accounted separately by the common driver's host budget.
        """
        return {
            f"member/{index}/{name}": size
            for index, member in enumerate(self._members)
            for name, size in cast(Backend, member).memory_ledger().items()
        }

    def close(self) -> None:
        """Release every member workspace and the common march state."""
        try:
            with ExitStack() as cleanup:
                for member in self._members:
                    cleanup.callback(cast(Backend, member).close)
        finally:
            self._state = None
            self._closed = True
            self._history = []
            self._rows.clear()
            self._members = ()


def _validate_members(members: tuple[StepDriver[object], ...]) -> None:
    """Require one backend and a shared entrance-to-end replay contract.

    Parameters
    ----------
    members : tuple of solprop.backends.driver.StepDriver
        Loaded workspaces whose ownership has not yet transferred.

    Raises
    ------
    BackendError
        If any workspace lies outside the lockstep contract.
    """
    if not members:
        raise BackendError("lockstep execution requires at least one member")
    first = members[0]
    settings = first._problem.stepper
    schedule = settings.schedule
    if schedule is None:
        raise BackendError("lockstep execution requires an explicit fixed schedule")
    for member in members:
        if not isinstance(member, Backend):
            raise BackendError("lockstep members must implement the backend protocol")
        member._require_state()
        if member.name != first.name:
            raise BackendError("lockstep execution refuses mixed backends")
        if member._z != 0.0 or member._history or member._schedule_index != 0:
            raise BackendError(
                "lockstep members must be freshly loaded at the entrance"
            )
        problem = member._problem
        tableau, reference_tableau = problem.stepper.tableau, settings.tableau
        if any(
            not np.array_equal(getattr(tableau, name), getattr(reference_tableau, name))
            for name in ("matrix", "nodes", "solution_weights", "error_weights")
        ):
            raise BackendError("lockstep members require the same tableau")
        if not np.array_equal(problem.grids.norm_mask, first._problem.grids.norm_mask):
            raise BackendError("lockstep members require the same frequency mask")
        other = problem.stepper.schedule
        if problem.stepper.integrator != settings.integrator:
            raise BackendError("lockstep members require the same integrator")
        if problem.shape[0] != 1 or problem.shape[1:] != first._problem.shape[1:]:
            raise BackendError(
                "lockstep members require matching singleton field shapes"
            )
        if (
            other is None
            or other.z_start != schedule.z_start
            or not np.array_equal(other.steps, schedule.steps)
            or not np.array_equal(problem.stepper.save_planes, settings.save_planes)
        ):
            raise BackendError(
                "lockstep members require the same schedule and save planes"
            )
        spec, reference = problem.stats_spec, first._problem.stats_spec
        if (spec.names, spec.periodic, spec.period) != (
            reference.names,
            reference.periodic,
            reference.period,
        ) or not (
            np.array_equal(spec.energy_windows, reference.energy_windows)
            and spec.same_tables(reference)
        ):
            raise BackendError("lockstep members require the same statistics request")
        _refuse_mixed_guides(problem, first._problem)


def _general_leaves(problem: KernelProblem) -> object:
    """Return what general modal guide a problem carries, comparably.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A member's problem.

    Returns
    -------
    object
        A scalar-and-string description, or ``None`` where the problem carries no
        general modal route at all.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return None
    metric = problem.linear.modal_metric
    responses = problem.responses.regional_responses
    # The flow is in the description because a declared basis of a split block puts
    # one on this route, and two members whose blocks differ are two different
    # guides however identical their tables read.
    flow = problem.linear.coupled_flow
    return (
        tuple(sorted(dict(tables.identity).items(), key=str)),
        None if metric is None else (metric.kind, metric.gauge_id),
        None if responses is None else (responses.model_id, responses.policy_id),
        None
        if flow is None
        else (flow.gauge_id, flow.certificate_id, flow.section_methods),
    )


def _refuse_mixed_guides(problem: KernelProblem, reference: KernelProblem) -> None:
    """Refuse a lockstep group whose members are not one general modal guide.

    The narrow admission rule for a batch of general modal guides, applied at the
    backend: a general-guide batch shares the physical guide, the ordered channels,
    the materials, the gauge and the table identity, and its members differ in
    their launch and their noise seed alone. The grouping signature separates two
    guides already (:func:`solprop.driver.batch.refuse_mixed_guides`); this is the
    check that does not depend on a caller having grouped carefully, written
    against the leaves a backend actually sees rather than against the driver's
    descriptor.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The member being admitted.
    reference : solprop.translate.problem.KernelProblem
        Member zero's problem, whose guide the group is.

    Raises
    ------
    BackendError
        If the two carry different general modal guides, or only one carries one.
    """
    if _general_leaves(problem) == _general_leaves(reference):
        return
    msg = (
        "general-guide batch requires identical guide/table/gauge identity: this "
        "member's transform.modal_tables, linear.modal_metric or "
        "responses.regional_responses is not member zero's. Members may differ in "
        "their launch and their noise seed and in nothing else, because one "
        "immutable reference is shared across the group"
    )
    raise BackendError(msg)

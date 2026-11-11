# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare independently owned Julia sessions from translated numerical descriptors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from weakref import WeakSet

from solprop import _interrupts
from solprop.backends.driver_session import foreign_request
from solprop.backends.julia import _calls
from solprop.backends.julia._blas import BlasHold, acquire_blas
from solprop.backends.julia._descriptors import lower_modal, transfer_memory
from solprop.backends.julia._fft_cache import FFTCache, refresh_fft_cache
from solprop.backends.julia._fft_runtime import load_planning_cache
from solprop.backends.julia._group import JuliaGroupSession
from solprop.backends.julia._lower import RunTransfer, lower_run
from solprop.backends.julia._node_plan import JuliaNodePlan, plan_julia_nodes
from solprop.backends.julia._noise import lower_input_noise
from solprop.backends.julia._observations import ObservationTransfer, lower_observations
from solprop.backends.julia._ownership import JuliaHandle, require_main_thread
from solprop.backends.julia._probe import JuliaTimingProbe
from solprop.backends.julia._runtime import JuliaRuntime, get_runtime
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia._spatial import lower_spatial
from solprop.backends.julia._transverse import TransverseFormat, transverse_format
from solprop.backends.protocol import ExecutionBackend, TimingProbe
from solprop.backends.session import (
    ExecutionOptions,
    GroupSession,
    SessionError,
    Start,
    check_group,
)
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.log import StopFlag
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryCheck
from solprop.profiles import ZProfile
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import SpatialRoute
from solprop.translate.run_request import RunRequest

__all__ = ["JuliaBackend", "JuliaPrepared"]


class JuliaBackend:
    """Own immutable Julia preparation and create complete numerical sessions.

    Construction performs pure descriptor admission. The first :meth:`prepare`
    selects a verified setup and initializes Julia on Python's main thread. Every
    started session owns its mutable march, transforms, history and event buffers.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Translated scientific descriptor.
    request : solprop.translate.run_request.RunRequest
        Resolved immutable run policy.
    grid : solprop.grid.TimeGrid or None, optional
        Host grid context for preparation of statistical tables.
    modes : sequence of Mode or None, optional
        Propagated mode context, in state order.
    profiles : mapping or None, optional
        Host profile context for profile statistics.
    options : solprop.backends.session.ExecutionOptions or None, optional
        Owned execution choices; Julia accepts ``wide`` and ``setup_id``.

    Raises
    ------
    BackendError
        Naming unsupported capabilities before numerical storage is allocated.
    """

    name: str = "julia"

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        options: ExecutionOptions | None = None,
    ) -> None:
        """Admit descriptors and options without importing JuliaCall."""
        require_main_thread()
        selected = ExecutionOptions() if options is None else options
        selected.check_backend_options(self.name, ("wide", "setup_id"))
        self._options = dict(selected.backend_options)
        self._execution_options = selected
        wide = self._options.get("wide", "binary64")
        setup_id = self._options.get("setup_id")
        if not isinstance(wide, str) or wide not in ("binary64", "df32"):
            raise BackendError(
                "julia backend_options['wide'] must be 'binary64' or 'df32'"
            )
        if setup_id is not None and (not isinstance(setup_id, str) or not setup_id):
            raise BackendError(
                "julia backend_options['setup_id'] must be a nonempty string"
            )
        if wide != "binary64":
            raise BackendError(
                "this Julia binary64 composition requires wide='binary64'"
            )
        if request.integration.propagator_mode == "tabulate":
            raise BackendError(
                "this Julia composition does not yet tabulate propagators"
            )
        self._problem = problem
        self._request = request
        self._spatial = isinstance(problem.route, SpatialRoute)
        self._lowered = (
            lower_spatial(problem) if self._spatial else lower_modal(problem)
        )
        self._node_plan = plan_julia_nodes(problem, request, selected)
        self._node_chunk = self._node_plan.plan.chunk
        lower_run(problem, request)
        self._observation = (
            lower_observations(
                problem,
                request,
                grid=grid,
                modes=modes,
                profiles=profiles,
                trace_chunk=self._node_chunk,
            )
            if request.observation.stats.names
            or request.boundary.monitor is not None
            or (
                problem.responses.absorber_scheme == "split_step"
                and problem.responses.temporal_absorber is not None
            )
            else None
        )
        self._setup_id = setup_id
        self._wide = wide
        self._grid, self._modes, self._profiles = grid, modes, profiles
        self._runtime: JuliaRuntime | None = None
        self._preparation: JuliaHandle | None = None
        self._blas: BlasHold | None = None
        self._fft_cache: FFTCache | None = None
        self._sessions: WeakSet[JuliaSession] = WeakSet()
        self._probes: WeakSet[JuliaTimingProbe] = WeakSet()
        self._groups: WeakSet[JuliaGroupSession] = WeakSet()
        self._prepared: WeakSet[JuliaPrepared] = WeakSet()
        self._closed = False

    @property
    def request(self) -> RunRequest:
        """Return the immutable request this execution owner admitted.

        Returns
        -------
        solprop.translate.run_request.RunRequest
            The construction request, before per-run retention/start adaptation.
        """
        return self._request

    @property
    def closed(self) -> bool:
        """Return whether this owner has released its resources.

        Returns
        -------
        bool
            True after closure.
        """
        return self._closed

    def _check(self) -> None:
        """Require an open execution owner on Python's main thread.

        Raises
        ------
        BackendError
            If called from another Python thread.
        SessionError
            If the owner is closed.
        """
        require_main_thread()
        if self._closed:
            raise SessionError("the Julia execution owner is closed")

    def prepare(
        self, problem: PropagationProblem, request: RunRequest
    ) -> JuliaPrepared:
        """Bind one admitted request to verified immutable Julia numerical preparation.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The admitted descriptor, or one sharing its science with a different input.
        request : solprop.translate.run_request.RunRequest
            This owner's request or a permitted per-run adaptation.

        Returns
        -------
        JuliaPrepared
            A reusable factory for independently owned sessions.

        Raises
        ------
        SessionError
            For a closed owner or a foreign numerical request.
        BackendError
            For an unsupported composition or a missing/incompatible runtime setup.
        """
        self._check()
        if reason := foreign_request(self._request, request):
            raise SessionError(f"julia cannot prepare this run: {reason}")
        if problem is not self._problem:
            from solprop.driver.simulation import check_derived_problem  # noqa: PLC0415

            check_derived_problem(self._problem, problem, self._request, request)
        transfer = lower_run(problem, request)
        node_plan = (
            self._node_plan
            if request is self._request
            else plan_julia_nodes(problem, request, self._execution_options)
        )
        observation = (
            self._observation
            if request is self._request
            else lower_observations(
                problem,
                request,
                grid=self._grid,
                modes=self._modes,
                profiles=self._profiles,
                trace_chunk=node_plan.trace_chunk,
            )
            if request.observation.stats.names
            or request.boundary.monitor is not None
            or (
                problem.responses.absorber_scheme == "split_step"
                and problem.responses.temporal_absorber is not None
            )
            else None
        )
        if self._preparation is None:
            self._prepare_numerical()
        assert self._runtime is not None and self._preparation is not None
        if observation is not None:
            prepared = self._runtime.own_call(
                "prepare_observations",
                self._preparation,
                observation,
                node_plan.plan.chunk,
            )
        elif node_plan.plan.chunk != self._node_chunk:
            prepared = self._runtime.own_call(
                "with_node_chunk", self._preparation, node_plan.plan.chunk
            )
        else:
            prepared = self._preparation
        return JuliaPrepared(
            self,
            self._runtime,
            prepared,
            transfer,
            problem.shape,
            problem=problem,
            observation=observation,
            node_plan=node_plan,
            monitor=(
                None
                if request.boundary.monitor is None
                else request.boundary.monitor.check
            ),
            transverse=transverse_format(
                None
                if request.boundary.monitor is None
                else request.boundary.monitor.transverse
            ),
        )

    def _prepare_numerical(self) -> None:
        """Publish numerical preparation and its BLAS lifetime as one transaction."""
        runtime = get_runtime(setup_id=self._setup_id, device="cpu")
        fft_cache = load_planning_cache(runtime)
        blas: BlasHold | None = None
        handle: JuliaHandle | None = None
        try:
            with _interrupts.committing():
                blas = acquire_blas(provider=runtime.cpu_blas)
                handle = runtime.own_call(
                    "prepare_spatial" if self._spatial else "prepare_modal",
                    self._lowered,
                    runtime.threads,
                    self._node_chunk,
                    owners=(blas,),
                    on_release=blas.close,
                )
                self._runtime, self._preparation, self._blas = runtime, handle, blas
                self._fft_cache = fft_cache
        except BaseException as failure:
            published = (
                handle is not None
                and self._runtime is runtime
                and self._preparation is handle
                and self._blas is blas
            )
            if not published:
                self._runtime, self._preparation, self._blas = None, None, None
                resource = handle if handle is not None else blas
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as cleanup:
                        # The Python-only hold retains failed foreign restoration.
                        # Keep the construction error primary and report the retry.
                        failure.add_note(
                            f"BLAS cleanup failed; retained for retry: {cleanup}"
                        )
            raise

    def metadata(self) -> dict[str, object]:
        """Return actual execution choices, outside scientific and resume identities.

        Returns
        -------
        dict
            JSON-compatible backend, precision, device, wide arithmetic and setup facts.
        """
        runtime = self._runtime
        record: dict[str, object] = {
            "backend": self.name,
            "device": "cpu",
            "precision": "double",
            "wide": self._wide,
            "propagator_mode": "recompute",
            "backend_options": dict(self._options),
            **_node_metadata(self._node_plan),
        }
        if runtime is not None:
            record.update(
                setup_id=runtime.setup.setup_id,
                julia_version=runtime.setup.julia_version,
                julia_threads=runtime.threads,
                python_entry="main_thread",
            )
        if self._blas is not None:
            record.update(self._blas.metadata())
        return record

    def march_export_refusal(self) -> str | None:
        """Report whether this composition can export portable march history.

        Returns
        -------
        str or None
            None: Adams exports portable history; Runge-Kutta uses an explicit reset.
        """
        return None

    def edge_memory(self, space: object) -> int:
        """Return no device edge storage for this CPU modal composition.

        Parameters
        ----------
        space : object
            The spatial geometry, unused by the admitted modal composition.

        Returns
        -------
        int
            Zero bytes.
        """
        del space
        return 0

    def timing_probe(self, start: Start) -> TimingProbe:
        """Load an isolated Julia-owned trial engine for the shared estimate.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Initial state, consumed only after successful numerical construction.
            Its field is copied; monitor and march continuation are unused.

        Returns
        -------
        solprop.backends.protocol.TimingProbe
            An owned probe that commits no session state or observation.

        Raises
        ------
        BackendError
            Naming an invalid start or unsupported numerical capability.
        """
        self._check()
        prepared = self.prepare(self._problem, self._request)
        probe = JuliaTimingProbe(prepared, start)
        self._probes.add(probe)
        return probe

    def available_device_memory(self) -> int | None:
        """Return no device-memory limit for CPU execution.

        Returns
        -------
        int or None
            None; host-memory admission belongs to the shared controller.
        """
        return None

    def memory_ledger(self) -> dict[str, int]:
        """Return owned transfer buffers and every live session's retained storage.

        Returns
        -------
        dict
            Named byte counts, with each session in its own namespace.
        """
        self._check()
        ledger = {
            f"transfer.{name}": size
            for name, size in transfer_memory(self._lowered).items()
        }
        observation_transfers = (
            self._observation,
            *(prepared._observation for prepared in self._prepared),
        )
        ledger.update(
            (f"observation.transfer{name}", size)
            for name, size in transfer_memory(observation_transfers).items()
        )
        if self._runtime is not None and self._preparation is not None:
            ledger.update(
                (f"preparation.{name}", size)
                for name, size in _calls.memory(
                    self._runtime, self._preparation, preparation=True
                ).items()
            )
        for index, prepared in enumerate(self._prepared):
            ledger.update(
                (f"run.{index}.{name}", size)
                for name, size in transfer_memory(prepared._transfer).items()
                # Control is the shared immutable request; the run's owned arrays
                # are separate fields, including its copied tolerance weights.
                if not name.startswith("control.")
            )
            if prepared._observation is not None:
                ledger.update(
                    (f"observation.{index}.preparation.{name}", size)
                    for name, size in _calls.memory(
                        prepared._runtime, prepared._preparation, preparation=True
                    ).items()
                )
        for index, session in enumerate(self._sessions):
            ledger.update(
                (f"session.{index}.{name}", size)
                for name, size in session.memory_ledger().items()
            )
        for index, probe in enumerate(self._probes):
            ledger.update(
                (f"timing.{index}.{name}", size)
                for name, size in probe.memory_ledger().items()
            )
        for index, group in enumerate(self._groups):
            if group._owners[0] is self:
                ledger.update(
                    (f"group.{index}.{name}", size)
                    for name, size in group._memory().items()
                )
        return ledger

    def start_group(
        self,
        members: Sequence[ExecutionBackend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
        starts: Sequence[Start],
        *,
        stop: StopFlag | None = None,
    ) -> GroupSession:
        """Start independent members on one Julia-owned common fixed schedule.

        Parameters
        ----------
        members : sequence of ExecutionBackend
            Distinct Julia owners, adopted only after successful construction.
        problems : sequence of PropagationProblem
            Their descriptors.
        requests : sequence of RunRequest
            Their resolved policies.
        starts : sequence of Start
            Their initial states, retained unchanged on refusal.
        stop : solprop.log.StopFlag or None, optional
            Group cancellation flag.

        Returns
        -------
        solprop.backends.session.GroupSession
            Common-window group, with Julia selecting every numerical member turn.

        Raises
        ------
        BackendError
            Naming incompatible owners, schedules, methods or execution choices.
        """
        self._check()
        check_group(len(members), problems, requests, starts)
        if len({id(member) for member in members}) != len(members):
            raise BackendError("julia groups require distinct member owners")
        if members[0] is not self or any(
            not isinstance(member, JuliaBackend) for member in members
        ):
            raise BackendError("julia groups require their own member owners")
        admitted = tuple(
            member for member in members if isinstance(member, JuliaBackend)
        )
        return JuliaGroupSession(admitted, problems, requests, starts, stop=stop)

    def register(self, session: JuliaSession) -> None:
        """Retain a weak lifetime record for an independently owned session.

        Parameters
        ----------
        session : JuliaSession
            A successfully constructed session.
        """
        self._check()
        self._sessions.add(session)

    def release(self, session: JuliaSession, *, owns: bool) -> None:
        """Remove a closed session and close its owner when ownership was transferred.

        Parameters
        ----------
        session : JuliaSession
            The session whose engine readers have finished.
        owns : bool
            Whether closing this session also closes the execution owner.
        """
        self._sessions.discard(session)
        if owns:
            self.close()

    def close(self) -> None:
        """Close live sessions before immutable preparation, idempotently on main."""
        require_main_thread()
        if self._closed:
            return
        groups = tuple(self._groups)
        for group in groups:
            group.check_close()
        sessions = tuple(self._sessions)
        for session in sessions:
            session.check_close()
        self._closed = True
        failure: BaseException | None = None
        for resource in (*groups, *sessions, *self._probes):
            try:
                resource.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        for prepared in self._prepared:
            if prepared._preparation is not self._preparation:
                prepared._preparation.close()
        if self._preparation is not None:
            self._preparation.close()
            self._preparation = None
        self._blas = None
        if failure is not None:
            raise failure


class JuliaPrepared:
    """Bind immutable Julia preparation to one admitted run policy.

    Parameters
    ----------
    owner : JuliaBackend
        Execution owner retained by every session.
    runtime : JuliaRuntime
        Verified main-thread runtime.
    preparation : JuliaHandle
        Immutable numerical preparation.
    transfer : RunTransfer
        Per-run policy, copied into each independently owned engine session.
    shape : tuple of int
        Host field shape ``(n_modes, n_pol, n_omega)``.
    problem : PropagationProblem or None, optional
        Actual scientific input descriptor for per-start noise selection; the owner's
        descriptor by default. A derived input keeps its own draw without a draw cache.
    observation : ObservationTransfer or None, optional
        Request-specific immutable observation layout and represented inputs.
    node_plan : JuliaNodePlan or None, optional
        This request's retained node-array budget; the owner's initial plan by default.
    monitor : solprop.monitor.BoundaryCheck or None, optional
        Immutable temporal monitor policy for formatting Julia's records and notices.
    transverse : TransverseFormat or None, optional
        Host-only spatial message context, without a live edge monitor.
    """

    def __init__(  # noqa: PLR0913 - numerical preparation and independent service context
        self,
        owner: JuliaBackend,
        runtime: JuliaRuntime,
        preparation: JuliaHandle,
        transfer: RunTransfer,
        shape: tuple[int, int, int],
        *,
        problem: PropagationProblem | None = None,
        observation: ObservationTransfer | None = None,
        node_plan: JuliaNodePlan | None = None,
        monitor: BoundaryCheck | None = None,
        transverse: TransverseFormat | None = None,
    ) -> None:
        """Retain numerical preparation without creating mutable march state."""
        self._owner = owner
        self._runtime = runtime
        self._preparation = preparation
        self._transfer = transfer
        self._shape = shape
        self._problem = owner._problem if problem is None else problem
        self._observation = observation
        self._node_plan = owner._node_plan if node_plan is None else node_plan
        self._monitor = monitor
        self._transverse = transverse
        self._owner._prepared.add(self)

    @property
    def backend(self) -> str:
        """Return the registered backend name.

        Returns
        -------
        str
            ``julia``.
        """
        return self._owner.name

    def start(
        self,
        start: Start,
        *,
        listener: Callable[[float, float, bool], None] | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> JuliaSession:
        """Construct a complete Julia session before consuming its starting field.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Initial state or supported restart.
        listener : callable or None, optional
            Attempt observer, called only at a closed Julia event.
        started : callable or None, optional
            Propagation announcement, after an owed launch observation.
        stop : solprop.log.StopFlag or None, optional
            Cooperative cancellation flag.
        owns : bool, optional
            Whether closing the session closes the execution owner.

        Returns
        -------
        JuliaSession
            Independently owned numerical state and bounded delivery buffers.

        Raises
        ------
        SessionError
            If its execution owner is already closed.
        """
        self._owner._check()
        noise = lower_input_noise(self._problem, noise=start.noise)
        session = JuliaSession(
            self._runtime,
            self._preparation,
            self._transfer,
            start,
            self._shape,
            record=self.metadata(),
            released=partial(self._release, owns=owns),
            stop=stop,
            listener=listener,
            started=started,
            observation=self._observation,
            monitor=self._monitor,
            transverse=self._transverse,
            noise=noise,
            fft_cache=(
                None
                if self._owner._fft_cache is None
                else refresh_fft_cache(self._owner._fft_cache)
            ),
        )
        self._owner.register(session)
        return session

    def _release(self, session: JuliaSession, *, owns: bool) -> None:
        """Retain this prepared factory until its session releases numerical resources.

        Parameters
        ----------
        session : JuliaSession
            Closed independent session.
        owns : bool
            Whether the session also owns the execution owner.
        """
        self._owner.release(session, owns=owns)

    def metadata(self) -> Mapping[str, object]:
        """Return execution choices from the immutable owner.

        Returns
        -------
        collections.abc.Mapping
            Detached execution metadata.
        """
        record = self._owner.metadata()
        record.update(_node_metadata(self._node_plan))
        if self._observation is not None:
            record["statistics_tables"] = self._observation.profiles.record()
        return record

    def memory_ledger(self) -> Mapping[str, int]:
        """Return the owner's retained buffers and live sessions.

        Returns
        -------
        collections.abc.Mapping
            Buffer names mapped to byte counts.
        """
        return self._owner.memory_ledger()

    def close(self) -> None:
        """Close this execution owner and its live sessions, idempotently."""
        self._owner.close()


def _node_metadata(plan: JuliaNodePlan) -> dict[str, object]:
    """Describe the request's node-array bound separately from total session memory."""
    return {
        **plan.plan.record(),
        "node_chunk_array_bytes": plan.chunk_bytes,
        "node_fixed_workspace_bytes": plan.fixed_workspace_bytes,
        "node_full_volume_workspace_bytes": plan.full_volume_workspace_bytes,
        "node_opaque_fft_plans": list(plan.opaque_fft_plans),
        "node_budget_exclusions": list(plan.exclusions),
        "node_array_inventory": [
            {
                "name": entry.name,
                "per_node_bytes": entry.per_node_bytes,
                "fixed_bytes": entry.fixed_bytes,
                "full_volume_bytes": entry.full_volume_bytes,
            }
            for entry in plan.inventory
        ],
    }

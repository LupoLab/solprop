# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare independently owned Julia sessions from translated numerical descriptors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from weakref import WeakSet

from solprop.backends.driver_session import foreign_request
from solprop.backends.julia import _calls
from solprop.backends.julia._descriptors import lower_modal, transfer_memory
from solprop.backends.julia._lower import RunTransfer, lower_runge_kutta
from solprop.backends.julia._ownership import JuliaHandle, require_main_thread
from solprop.backends.julia._runtime import JuliaRuntime, get_runtime
from solprop.backends.julia._session import JuliaSession
from solprop.backends.protocol import ExecutionBackend, TimingProbe
from solprop.backends.session import (
    ExecutionOptions,
    GroupSession,
    SessionError,
    Start,
)
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.log import StopFlag
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.translate.problem import PropagationProblem
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
        if selected.budget is not None:
            raise BackendError(
                "this Julia composition does not yet implement node memory budgets"
            )
        if request.integration.propagator_mode == "tabulate":
            raise BackendError(
                "this Julia composition does not yet tabulate propagators"
            )
        self._problem = problem
        self._request = request
        self._lowered = lower_modal(problem)
        self._node_chunk = min(
            problem.transform.n_nodes,
            problem.transform.n_nodes if selected.chunk is None else selected.chunk,
        )
        lower_runge_kutta(problem, request)
        self._setup_id = setup_id
        self._wide = wide
        self._grid, self._modes, self._profiles = grid, modes, profiles
        self._runtime: JuliaRuntime | None = None
        self._preparation: JuliaHandle | None = None
        self._sessions: WeakSet[JuliaSession] = WeakSet()
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
        transfer = lower_runge_kutta(problem, request)
        if self._preparation is None:
            runtime = get_runtime(setup_id=self._setup_id, device="cpu")
            handle = runtime.own_call(
                "prepare_modal", self._lowered, runtime.threads, self._node_chunk
            )
            self._runtime, self._preparation = runtime, handle
        assert self._runtime is not None
        return JuliaPrepared(
            self, self._runtime, self._preparation, transfer, problem.shape
        )

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
            "node_chunk": self._node_chunk,
        }
        if runtime is not None:
            record.update(
                setup_id=runtime.setup.setup_id,
                julia_version=runtime.setup.julia_version,
                julia_threads=runtime.threads,
                python_entry="main_thread",
            )
        return record

    def march_export_refusal(self) -> str | None:
        """Return no history-export refusal for the admitted Runge-Kutta composition.

        Returns
        -------
        str or None
            None; Runge-Kutta restarts from its field with an explicit reset record.
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
        """Refuse an estimate until this composition has a standalone trial probe.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Start retained unchanged on refusal.

        Returns
        -------
        solprop.backends.protocol.TimingProbe
            No probe is currently constructed.

        Raises
        ------
        BackendError
            Naming the pending timing-probe capability.
        """
        self._check()
        del start
        raise BackendError(
            "this Julia composition does not yet implement timing probes"
        )

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
        if self._runtime is not None and self._preparation is not None:
            ledger.update(
                (f"preparation.{name}", size)
                for name, size in _calls.memory(
                    self._runtime, self._preparation, preparation=True
                ).items()
            )
        for index, session in enumerate(self._sessions):
            ledger.update(
                (f"session.{index}.{name}", size)
                for name, size in session.memory_ledger().items()
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
        """Refuse groups until Julia's common-schedule composition is available.

        Parameters
        ----------
        members : sequence of ExecutionBackend
            Member execution owners, retained unchanged on refusal.
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
            No group is currently constructed.

        Raises
        ------
        BackendError
            Naming the pending common-schedule group capability.
        """
        self._check()
        del members, problems, requests, starts, stop
        raise BackendError(
            "this Julia composition does not yet implement member groups"
        )

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
        sessions = tuple(self._sessions)
        for session in sessions:
            session.check_close()
        self._closed = True
        failure: BaseException | None = None
        for session in sessions:
            try:
                session.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if self._preparation is not None:
            self._preparation.close()
            self._preparation = None
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
    """

    def __init__(
        self,
        owner: JuliaBackend,
        runtime: JuliaRuntime,
        preparation: JuliaHandle,
        transfer: RunTransfer,
        shape: tuple[int, int, int],
    ) -> None:
        """Retain numerical preparation without creating mutable march state."""
        self._owner = owner
        self._runtime = runtime
        self._preparation = preparation
        self._transfer = transfer
        self._shape = shape

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
        if self._owner.closed:
            raise SessionError("the Julia prepared execution is closed")
        session = JuliaSession(
            self._runtime,
            self._preparation,
            self._transfer,
            start,
            self._shape,
            record=self._owner.metadata(),
            released=lambda one: self._owner.release(one, owns=owns),
            stop=stop,
            listener=listener,
            started=started,
        )
        self._owner.register(session)
        return session

    def metadata(self) -> Mapping[str, object]:
        """Return execution choices from the immutable owner.

        Returns
        -------
        collections.abc.Mapping
            Detached execution metadata.
        """
        return self._owner.metadata()

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

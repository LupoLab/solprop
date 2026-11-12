# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The executors the lifecycle conformance suite runs on, and how it drives them.

SEP 47 section 7.1 asks for one lifecycle conformance suite that every session
implementation passes, with numerical outcomes scripted so the lifecycle is tested
apart from the physics, and section 6.6 forbids making the native session the
lifecycle oracle. An :class:`Executor` is therefore only a factory: given a problem,
its run request and a start, it returns a started :class:`~solprop.backends.session.
RunSession` with whatever probes the implementation can offer. Every executor answers
the same tests.

- ``reference`` is the NumPy reference, in its own session
  (:class:`~solprop.backends.host.session.HostSession`, SEP 47 phase 2).
- ``native-session`` is the native execution session
  (:class:`solprop_native.session.NativeSession`), the only native executor since SEP
  47 phase 5: the native backend owns the run, its boundary monitor included, whose
  decisions and record are native (:attr:`Executor.native_monitor`); it evaluates
  every periodic-capable statistic natively. It is skipped where ``solprop_native``
  is not installed.
- ``native-session-low-storage`` is the native execution session replaying the same
  problem's planes on a uniform schedule of :data:`LOW_STORAGE_STEPS` steps with the
  low-storage march (SEP 47 section 6.3), whose loop the native session owns.
- ``scripted-host`` is :class:`HostSessionScript`, the reference's own driver in the
  reference's session, whose norms, rejections, failures, in-flight failures and
  warnings are chosen by the test.

The problem is the fundamental GNLSE soliton of ``tests/backends/conftest.py`` on its
768-bin window, over 2 cm in four planes: a few dozen attempts on any executor. A
free-space run's transverse edge (SEP 40 section 5.3) is the re-review's tilted packet
on its periodic line (:func:`edged`), which an executor starts with that run's grid
(``context``).
"""

import dataclasses
import functools
import os
import re
import warnings
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from scripted_driver import SCALE, HostScriptedDriver
from solprop.backends.events import (
    EventBatch,
    ObservationKind,
    ObservationReady,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.host.session import HostSession
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.registry import select
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    RunSession,
    SessionStatus,
    Start,
)
from solprop.grid import EnvelopeGrid
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.nonlinear.gnlse import GnlseProblem, problem_from
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import PropagationProblem, Translation, assemble
from solprop.translate.run_request import (
    IntegrationSpec,
    MethodKind,
    ReplaySpec,
    RunRequest,
)

BETA2 = -11.83e-27  # s^2/m, as tests/backends/conftest.py
GAMMA = 0.11  # 1/(W m)
TAU0 = 100e-15  # s, the natural sech width
WAVELENGTH = 835e-9  # m

LENGTH = 0.02
"""The run's length, m."""

PLANES = (LENGTH / 4, LENGTH / 2, 3 * LENGTH / 4, LENGTH)
"""The save planes, m."""

LOW_STORAGE_STEPS = 40
"""The steps of the low-storage executor's uniform schedule over :data:`LENGTH`."""

FACTOR = 2.0
"""What a scripted trial multiplies the state by: a power of two, so the field after
``k`` accepted steps is the launch times ``2**k`` exactly."""

REJECTED = 5.0
"""The norm a scripted rejection scores, far above the controller's threshold of one."""

EMPTY = BackendHistory(
    np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=np.bool_), np.empty(0)
)
"""An empty attempt record."""

IDENTITY = ExportRecipe(power=False, optics=False)
CONSUMER = ExportRecipe(power=True, optics=False)
"""The field a saved plane, a plane observer and a checkpoint share (service loop)."""
STORED = ExportRecipe(power=True, optics=True)
"""The field a plane store writes."""
_NO_FIELD = np.empty(0, dtype=np.complex128)
"""A plane's field in a run driven without an export recipe."""

RECIPES = (IDENTITY, CONSUMER, ExportRecipe(power=False, optics=True), STORED)


@functools.cache
def window() -> EnvelopeGrid:
    """Return the 768-bin envelope grid of ``tests/backends/conftest.py``.

    Returns
    -------
    solprop.grid.EnvelopeGrid
        The grid.
    """
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)


@functools.cache
def fibre() -> GnlseProblem:
    """Return the dispersion-specified fibre of the soliton problems.

    Returns
    -------
    solprop.nonlinear.gnlse.GnlseProblem
        Its mode and Kerr response.
    """
    return problem_from(
        [0.0, 0.0, BETA2], omega_ref=window().omega0, gamma=GAMMA, shock=False
    )


def soliton(
    *,
    step_planes: int = 0,
    period: int = 2,
    interpolate: bool = False,
    integrator: Multistep | None = None,
    planes: Collection[float] = PLANES,
) -> Translation:
    """Return the fundamental soliton over :data:`PLANES`, and its run's request.

    Parameters
    ----------
    step_planes : int, optional
        Observe the end of every ``step_planes``-th accepted step; ``0`` for none.
    period : int, optional
        The period of the periodic ``energy`` rows, accepted steps.
    interpolate : bool, optional
        Interpolate the save planes inside the steps that cross them (dense output).
    integrator : solprop.stepper.multistep.Multistep or None, optional
        The Adams method; the default Runge-Kutta tableau by default.
    planes : collection of float, optional
        The save planes, m, increasing and ending at :data:`LENGTH`.

    Returns
    -------
    solprop.translate.problem.Translation
        The translated problem and its request.
    """
    grid = window()
    peak = abs(BETA2) / (GAMMA * TAU0**2)
    return assemble(
        grid,
        fibre().mode,
        [fibre().kerr],
        SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid),
        save_planes=list(planes),
        controller=PIController(rtol=1e-4, atol=1e-16),
        dz_max=LENGTH,
        dz_init=1e-3,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        step_planes=step_planes,
        interpolate_planes=interpolate,
        integrator=integrator,
        stats_spec=StatsSpec(
            names=("energy", "peak_power"), periodic=("energy",), period=period
        ),
    )


def entrance(problem: PropagationProblem) -> Start:
    """Return the fresh start of a problem, its launch plane owed.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.

    Returns
    -------
    solprop.backends.session.Start
        The start at the entrance.
    """
    return Start(field=problem.initial.field0, z=float(problem.initial.z))


class ScriptedError(RuntimeError):
    """What a scripted trial raises."""


class ScriptedWarning(UserWarning):
    """What a scripted committed endpoint warns with."""


class _Outcomes:
    """A scripted driver with the outcomes the lifecycle tests choose.

    Trials and accepted steps are counted from 1 since the load.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    run_request : solprop.translate.run_request.RunRequest
        Its run's request.
    reject_at : collection of int, optional
        Trials that score :data:`REJECTED`.
    fail_at : int or None, optional
        The trial that raises :class:`ScriptedError` before it is recorded.
    in_flight : bool, optional
        Whether that failure leaves the step in flight, as an in-place low-storage step
        abandoned partway does: the state is then neither end of the step.
    signal : bool, optional
        Whether that trial raises ``KeyboardInterrupt`` instead, as a signal handler
        does that runs between the driver's steps: a failure outside the session,
        with nothing in flight.
    warn_at : int or None, optional
        The accepted step whose committed endpoint warns with
        :class:`ScriptedWarning`, after the step's statistics row.
    **kwargs
        Passed to :class:`scripted_driver.HostScriptedDriver`.
    """

    def __init__(
        self,
        problem: PropagationProblem,
        run_request: RunRequest,
        *,
        reject_at: Collection[int] = (),
        fail_at: int | None = None,
        in_flight: bool = False,
        signal: bool = False,
        warn_at: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, run_request, factor=FACTOR, scale=SCALE, **kwargs)  # pyright: ignore[reportCallIssue]
        self._reject_at = frozenset(reject_at)
        self._fail_at = fail_at
        self._leave_in_flight = in_flight
        self._signal = signal
        self._warn_at = warn_at
        self.trials = 0
        self.accepted = 0

    def _trial_step(self, z, dz):
        self.trials += 1
        if self.trials == self._fail_at:
            if self._leave_in_flight:
                self._in_flight = (z, dz)
            if self._signal:
                raise KeyboardInterrupt
            msg = f"scripted failure at trial {self.trials}"
            raise ScriptedError(msg)
        norm, state = super()._trial_step(z, dz)  # pyright: ignore[reportAttributeAccessIssue]
        return (REJECTED if self.trials in self._reject_at else norm), state

    def _take_step_stats(self):
        super()._take_step_stats()  # pyright: ignore[reportAttributeAccessIssue]
        self.accepted += 1
        if self.accepted == self._warn_at:
            warnings.warn(
                f"scripted warning at accepted step {self.accepted}",
                ScriptedWarning,
                stacklevel=2,
            )


class HostSessionScript(_Outcomes, HostScriptedDriver):
    """The scripted outcomes on the reference's own driver (SEP 47 phase 2)."""


@dataclass(frozen=True, eq=False)
class Started:
    """A started session and the probes its executor offers.

    Attributes
    ----------
    session : solprop.backends.session.RunSession
        The session.
    field_reads : callable or None
        Returns how many times a field crossed out of the live state so far, or ``None``
        when the executor cannot count it.
    live_state : callable or None
        Returns the live propagation state as a host array, for the aliasing checks of
        SEP 47 section 5.4, or ``None`` when the state is not a host array.
    """

    session: RunSession
    field_reads: Callable[[], int] | None = None
    live_state: Callable[[], np.ndarray | None] | None = None


class Factory(Protocol):
    """Start a session of one executor."""

    def __call__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        start: Start,
        *,
        stop: object | None = None,
        listener: Callable[[float, float, bool], None] | None = None,
        context: Mapping[str, Any] | None = None,
        **script: Any,
    ) -> Started:
        """Return a started session; `script` is for the scripted executor alone, and
        `context` (the grid and modes) for a problem other than the soliton."""
        ...


def _over_driver(
    make: Callable[..., Any], *, host_state: bool, session: type = HostSession
) -> Factory:
    """Return a factory of sessions over a step driver.

    Parameters
    ----------
    make : callable
        ``(problem, request, **script) -> driver``.
    host_state : bool
        Whether the driver's state is a host array the aliasing checks can read.
    session : type, optional
        The session class: the reference's by default.

    Returns
    -------
    Factory
        The factory.
    """

    def start_session(
        problem, request, start, *, stop=None, listener=None, context=None, **script
    ):
        driver = make(problem, request, context=context, **script)
        reads = [0]
        observe = driver.observe

        def counted(*, field: bool = True):
            reads[0] += bool(field)
            return observe(field=field)

        driver.observe = counted
        started = session(driver, problem, request, start, listener=listener, stop=stop)
        return Started(
            started,
            field_reads=lambda: reads[0],
            live_state=(lambda: driver._state) if host_state else None,
        )

    return start_session


def _selected(name: str) -> Callable[..., Any]:
    """Return ``(problem) -> driver`` for a registered backend.

    Parameters
    ----------
    name : str
        The registry name.

    Returns
    -------
    callable
        The constructor; it takes no script.
    """

    def make(
        problem: PropagationProblem,
        request: RunRequest,
        *,
        context: Mapping[str, Any] | None,
    ) -> Any:
        return select(
            name,
            problem,
            request,
            **(_soliton_context() if context is None else context),
        )

    return make


def _soliton_context() -> dict[str, Any]:
    """Return the soliton's grid and mode, as a backend takes them.

    Returns
    -------
    dict
        ``grid`` and ``modes``.
    """
    return {"grid": window(), "modes": [fibre().mode]}


def _host_scripted(
    problem: PropagationProblem,
    request: RunRequest,
    *,
    context: Mapping[str, Any] | None,
    **script: Any,
) -> HostSessionScript:
    """Return a scripted driver on the reference's own driver for a problem.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request.
    context : collections.abc.Mapping or None
        Refused: a script runs the soliton alone.
    **script
        The outcomes, as :class:`HostSessionScript` takes them.

    Returns
    -------
    HostSessionScript
        The driver.
    """
    assert context is None, "a script runs the soliton alone"
    return HostSessionScript(problem, request, grid=window(), **script)


def _native_session(
    problem, request, start, *, stop=None, listener=None, context=None, **script
):
    """Start a native execution session (SEP 47's prototype) for a problem.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start
        Where the run starts.
    stop : object or None, optional
        The stop token.
    listener : callable or None, optional
        The attempt listener.
    context : collections.abc.Mapping or None, optional
        The grid and modes of a problem other than the soliton.
    **script
        Refused: the native session's outcomes are not scripted.

    Returns
    -------
    Started
        The session, counting the fields it exports natively.
    """
    assert not script, "the native session takes no script"
    from solprop_native.session import (  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
        NativeSession,
    )

    session = NativeSession(
        problem,
        request,
        start,
        stop=stop,
        listener=listener,
        **(_soliton_context() if context is None else context),
    )
    return Started(session, field_reads=lambda: session.native.status().exports)


def low_storage(
    problem: PropagationProblem, request: RunRequest
) -> tuple[PropagationProblem, RunRequest]:
    """Return a problem and its request replayed by the low-storage march.

    The same problem and policies, its planes replayed on a uniform schedule of
    :data:`LOW_STORAGE_STEPS` steps over :data:`LENGTH` with
    :class:`~solprop.stepper.low_storage.LowStorageRungeKutta`.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        Its run request.

    Returns
    -------
    tuple
        The replayed problem and its request.
    """
    stepper = request.stepper
    schedule = FixedSchedule.uniform(
        float(problem.initial.z),
        stepper.z_end,
        LENGTH / LOW_STORAGE_STEPS,
        save_planes=list(stepper.save_planes),
    )
    method = LowStorageRungeKutta()
    integration = IntegrationSpec(
        kind=MethodKind.LOW_STORAGE,
        tableau=request.integration.tableau,
        method=method,
        propagator_mode=request.integration.propagator_mode,
    )
    return problem, dataclasses.replace(
        request, integration=integration, replay=ReplaySpec(schedule)
    )


def _native_low_storage(problem, request, start, *, stop=None, listener=None, **script):
    """Start a native execution session replaying a problem by the low-storage march.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem, whose planes the replay lands on (:func:`low_storage`).
    request : solprop.translate.run_request.RunRequest
        Its run request.
    start : solprop.backends.session.Start
        Where the run starts.
    stop : object or None, optional
        The stop token.
    listener : callable or None, optional
        The attempt listener.
    **script
        Refused: the native session's outcomes are not scripted.

    Returns
    -------
    Started
        The session, counting the fields it exports natively.
    """
    replayed, policies = low_storage(problem, request)
    return _native_session(
        replayed, policies, start, stop=stop, listener=listener, **script
    )


MONITOR = "monitor"
"""The capability of watching the boundary with a monitor (SEP 21)."""

STATISTICS = "statistics"
"""The capability of evaluating the catalogue's statistics of the soliton's route,
every periodic-capable one as a periodic row (SEP 47 section 6.4); a scripted driver
reports positions in their place."""

TRANSVERSE = "transverse"
"""The capability of watching a free-space run's transverse edge (SEP 40 section 5.3):
its reads on their spacing, held on the steps, and its judgement (:func:`edged`)."""

DENSE = "dense output"
"""The capability of interpolating save planes inside the steps that cross them (SEP 14
phase 3); a scripted driver's trials hold no stages to interpolate, and a replay lands
on its planes."""

COUPLED = "coupled flow"
"""The capability of stepping a coupled flow on the exponential Adams march, wider than
16 channels or with bins off its eigenbasis route (SEP 47 section 6.3, phase 4:
:func:`coupled_flow`); a scripted driver has no flow, and the low-storage replay
none beside one."""

CAPABILITIES = frozenset({MONITOR, STATISTICS, TRANSVERSE, DENSE, COUPLED})
"""Every capability a test may need of an executor."""


COUPLED_KINDS = ("wide", "off route")
"""The coupled flows :func:`coupled_flow` builds."""


def coupled_flow(
    kind: str,
) -> tuple[PropagationProblem, RunRequest, dict[str, Any]]:
    """Return a coupled flow's ten fixed Adams steps, with its context.

    ``"wide"`` is the 17-channel lossless taper of
    ``tests/driver/test_native_flow_session.py``; ``"off route"`` the coherent-wall
    taper of ``tests/flow_parity.py`` with every bin off the eigenbasis route, which
    the native backend reads from its decomposition, formed here under an eigenbasis
    limit of zero and kept with the problem (``solprop_native.backend._HOST_FLOWS``);
    the reference takes the augmented exponential on every bin either way.

    Parameters
    ----------
    kind : str
        One of :data:`COUPLED_KINDS`.

    Returns
    -------
    tuple
        The problem, its request without statistics, and the grid, modes and
        profiles an executor starts it with.
    """
    from unittest import mock  # noqa: PLC0415

    from driver.test_native_flow_session import wide_simulation  # noqa: PLC0415
    from flow_parity import QUIET, build  # noqa: PLC0415

    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        simulation = wide_simulation(17) if kind == "wide" else build("taper")
        problem = simulation.translate()
    if kind != "wide":
        try:
            from solprop_native import (  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
                backend,
            )
        except ImportError:
            pass
        else:
            with mock.patch(
                "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
            ):
                backend._HOST_FLOWS[problem] = backend._decomposed_flow(problem)
    request = simulation._run_request().for_run(stats=StatsSpec())
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    return problem, request, context


def edged(
    check: BoundaryCheck, *, noise: Literal["input"] | None = None
) -> tuple[PropagationProblem, RunRequest, dict[str, Any]]:
    """Return the re-review's tilted packet, edged, with its context.

    The packet of ``tests/test_transverse_reads.py`` on its 256-sample periodic line,
    marched by Runge-Kutta over 1 mm in two planes; under the default budget the edge
    stops it, and a relaxed one reads it to the end.

    Parameters
    ----------
    check : solprop.monitor.BoundaryCheck
        The run's thresholds.
    noise : {"input", None}, optional
        The input noise launched beside the packet, seeded, whose allowance the edge
        judges above; none by default.

    Returns
    -------
    tuple
        The problem, without statistics, its monitored request, and the grid and modes
        an executor starts it with.
    """
    from test_transverse_reads import REREVIEW, packet_run  # noqa: PLC0415

    simulation = packet_run(
        REREVIEW,
        check=check,
        planes=[REREVIEW.length / 2, REREVIEW.length],
        backend="numpy_ref",
        noise=noise,
    )
    problem = simulation.translate()
    request = simulation._run_request().for_run(
        stats=StatsSpec(), monitor=simulation._boundary_monitor()
    )
    return problem, request, {"grid": simulation.grid, "modes": simulation.modes}


def _one_thread() -> bool:
    """Return whether the native backend runs one thread, as the default lane sets it.

    Returns
    -------
    bool
        True when ``SOLPROP_NATIVE_THREADS`` or ``SOLPROP_THREADS`` is ``"1"``.
    """
    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


@functools.cache
def device_build() -> bool:
    """Return whether the installed native backend is a device (CUDA or HIP) build.

    Returns
    -------
    bool
        False without ``solprop_native``.
    """
    try:
        import solprop_native  # noqa: PLC0415 - optional  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return solprop_native.build_info().device_backend != "none"


def skip_device_refused(problem: PropagationProblem) -> None:
    """Skip the calling test where a device build refuses its problem.

    The native backend refuses by name a z table that is not device-ready on a device
    build, before any session is asked. Nothing on a CPU build.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    """
    if not device_build() or problem.device_ready:
        return
    import pytest  # noqa: PLC0415 - only a test calls it

    pytest.skip("device builds refuse a z table by name")


@dataclass(frozen=True)
class Executor:
    """One session implementation under the conformance suite.

    Attributes
    ----------
    name : str
        Its name in test identifiers.
    start : Factory
        Starts a session.
    requires : str or None
        A module it needs, skipped without it.
    scripted : bool
        Whether its numerical outcomes are scripted (:class:`HostSessionScript`).
    capabilities : frozenset of str
        What it can be asked to run beyond the soliton (:data:`CAPABILITIES`); a test
        that needs another skips it explicitly (:meth:`require`).
    native_monitor : bool
        Whether the session owns its boundary monitor's state natively, so that the
        request's solprop monitor stays fresh and the session's
        :meth:`~solprop.backends.session.RunSession.monitor_record` alone shows what
        it did (SEP 47 section 6.4).
    """

    name: str
    start: Factory
    requires: str | None = None
    scripted: bool = False
    capabilities: frozenset[str] = CAPABILITIES
    native_monitor: bool = False

    def require(self, capability: str) -> None:
        """Skip the calling test when this executor lacks a capability.

        Parameters
        ----------
        capability : str
            The capability, one of :data:`CAPABILITIES`.
        """
        import pytest  # noqa: PLC0415 - only a test calls it

        if capability not in self.capabilities:
            pytest.skip(f"the {self.name} executor does not run a {capability}")

    @property
    def deterministic(self) -> bool:
        """Return whether repeated runs are bitwise the same in this process.

        SEP 47 section 7.1 asks for bit-identical yields "in deterministic CPU cases";
        a native run on several threads may vary its reduction order.

        Returns
        -------
        bool
            True for the reference and the script, and for native on one thread.
        """
        return not self.name.startswith("native") or _one_thread()


EXECUTORS: dict[str, Executor] = {
    "reference": Executor(
        "reference",
        _over_driver(_selected("numpy_ref"), host_state=True),
    ),
    "native-session": Executor(
        "native-session",
        _native_session,
        requires="solprop_native",
        native_monitor=True,
    ),
    "native-session-low-storage": Executor(
        "native-session-low-storage",
        _native_low_storage,
        requires="solprop_native",
        capabilities=CAPABILITIES - {DENSE, COUPLED},
        native_monitor=True,
    ),
    "scripted-host": Executor(
        "scripted-host",
        _over_driver(_host_scripted, host_state=True),
        scripted=True,
        capabilities=frozenset({MONITOR, STATISTICS}),
    ),
}
"""Every executor of the suite, by name."""


@dataclass(frozen=True, eq=False)
class Plane:
    """One observation of a run, with what it delivered since the previous one.

    Attributes
    ----------
    event : solprop.backends.events.ObservationReady
        The observation.
    field : numpy.ndarray
        Its field as the run exported it; empty for a run driven without an export
        recipe.
    history : solprop.backends.protocol.BackendHistory
        Every attempt since the previous observation (:meth:`Run.assembled`).
    rows : solprop.backends.protocol.StepStats
        The periodic rows since the previous observation.
    """

    event: ObservationReady
    field: np.ndarray
    history: BackendHistory
    rows: StepStats

    @property
    def z(self) -> float:
        """Return the observation's position, m.

        Returns
        -------
        float
            The position.
        """
        return float(self.event.z)

    @property
    def kind(self) -> ObservationKind:
        """Return why the plane was observed.

        Returns
        -------
        solprop.backends.events.ObservationKind
            The kind.
        """
        return self.event.kind

    @property
    def plane_index(self) -> int:
        """Return the save plane the observation answers, ``-1`` for the launch.

        Returns
        -------
        int
            The index.
        """
        return self.event.plane_index

    @property
    def stats(self) -> dict[str, np.ndarray]:
        """Return the observation's statistics by catalogue name.

        Returns
        -------
        dict
            The statistics.
        """
        return {name: np.asarray(value) for name, value in self.event.stats.items()}

    @property
    def summary(self) -> StepSummary:
        """Return the cost and tolerance of the propagation that reached the plane.

        Returns
        -------
        solprop.backends.protocol.StepSummary
            The summary.
        """
        return self.event.summary


@dataclass
class Run:
    """What driving a session to its end produced.

    Attributes
    ----------
    batches : list
        Every batch, in order.
    fields : list
        For each observation, its field exported with the run's recipe, copied, or
        nothing without a recipe.
    status : solprop.backends.session.SessionStatus
        The status after the terminal batch was acknowledged.
    """

    batches: list[EventBatch]
    fields: list[np.ndarray]
    status: SessionStatus

    @property
    def events(self) -> list[object]:
        """Return every event in committed order.

        Returns
        -------
        list
            The events.
        """
        return [event for batch in self.batches for event in batch.events]

    @property
    def observations(self) -> list[ObservationReady]:
        """Return the observations in committed order.

        Returns
        -------
        list
            The observations.
        """
        return [event for event in self.events if isinstance(event, ObservationReady)]

    @property
    def terminal(self) -> Terminal:
        """Return the terminal event.

        Returns
        -------
        solprop.backends.events.Terminal
            The last event of the last batch.
        """
        terminal = self.batches[-1].terminal
        assert terminal is not None
        return terminal

    @property
    def chunks(self) -> list[tuple[BackendHistory, StepStats]]:
        """Return every event's chunk of attempts and rows, in delivery order.

        Each attempt and row is delivered once, by the first event after it, of any
        kind (SEP 47 section 5.3); an event that carries none gives an empty chunk.

        Returns
        -------
        list
            ``(history, rows)`` per event.
        """
        found = []
        for event in self.events:
            history = getattr(event, "history", None)
            rows = getattr(event, "step_stats", None)
            found.append(
                (
                    EMPTY if history is None else history,
                    StepStats.empty() if rows is None else rows,
                )
            )
        return found

    @property
    def histories(self) -> list[BackendHistory]:
        """Return each attempt record exactly once: every event's chunk.

        Returns
        -------
        list
            The histories, in committed order.
        """
        return [history for history, _ in self.chunks]

    def assembled(self) -> list[tuple[object, BackendHistory, StepStats]]:
        """Assemble the chunks as the service loop does, per observation and terminal.

        An observation takes every chunk since the previous observation, its own last;
        the terminal event likewise, as a terminal checkpoint does. Snapshot and
        progress chunks are kept for the next of them.

        Returns
        -------
        list
            ``(event, history, rows)`` per observation and for the terminal event.
        """
        found: list[tuple[object, BackendHistory, StepStats]] = []
        held: list[tuple[BackendHistory, StepStats]] = []
        for event, chunk in zip(self.events, self.chunks, strict=True):
            held.append(chunk)
            if isinstance(event, ObservationReady | Terminal):
                found.append((event, *join_attempts(held)))
                held = []
        return found

    @property
    def attempts(self) -> int:
        """Return the attempts the histories report.

        Returns
        -------
        int
            Accepted and rejected.
        """
        return sum(history.n_steps for history in self.histories)

    @property
    def joined(self) -> tuple[BackendHistory, StepStats]:
        """Return every attempt and every periodic row of the run, joined in order.

        Returns
        -------
        tuple
            The attempts and the rows, without a batch axis.
        """
        return join_attempts(self.chunks)

    def all_planes(self) -> list[Plane]:
        """Return every observation, the launch plane's included, in order.

        Returns
        -------
        list of Plane
            The observations, each with its exported field (when the run exported
            one) and the attempts and rows assembled since the previous observation.
        """
        fields = (
            list(self.fields) if self.fields else [_NO_FIELD] * len(self.observations)
        )
        found: list[Plane] = []
        for event, history, rows in self.assembled():
            if isinstance(event, ObservationReady):
                found.append(Plane(event, fields[len(found)], history, rows))
        return found

    @property
    def planes(self) -> list[Plane]:
        """Return every observation after the launch plane, in order.

        Returns
        -------
        list of Plane
            The observations of save, step and dense planes.
        """
        return [
            plane
            for plane in self.all_planes()
            if plane.kind is not ObservationKind.LAUNCH
        ]

    @property
    def launch(self) -> Plane | None:
        """Return the launch plane's observation, when the start owed it.

        Returns
        -------
        Plane or None
            The launch plane.
        """
        for plane in self.all_planes():
            if plane.kind is ObservationKind.LAUNCH:
                return plane
        return None

    @property
    def final(self) -> Plane:
        """Return the last observation after the launch plane.

        Returns
        -------
        Plane
            The observation.
        """
        return self.planes[-1]

    @property
    def met_tolerance(self) -> bool:
        """Return whether every propagation between observations met its tolerance.

        Returns
        -------
        bool
            True when every plane after the launch reports it met.
        """
        return all(plane.summary.met_tolerance for plane in self.planes)

    def accepted_ends(self) -> np.ndarray:
        """Return the end of every accepted attempt, in order, m.

        Returns
        -------
        numpy.ndarray
            The positions.
        """
        if not self.histories:
            return np.empty(0)
        return np.concatenate(
            [history.z[history.accepted] for history in self.histories]
        )


def drive(
    started: Started,
    *,
    export: ExportRecipe | None = None,
    work_limit: int = DEFAULT_WORK_LIMIT,
    deadline: float = DEFAULT_DEADLINE,
    between: Callable[[RunSession, EventBatch], None] | None = None,
    observations: int | None = None,
    until: float | None = None,
    raises: bool = True,
) -> Run:
    """Advance a session to its end, acknowledging each batch.

    Parameters
    ----------
    started : Started
        The started session.
    export : solprop.backends.export.ExportRecipe or None, optional
        Export every observation's field with this recipe, before acknowledging it.
    work_limit : int, optional
        The attempt bound of every advance.
    deadline : float, optional
        The yield deadline of every advance, s.
    between : callable or None, optional
        Called with the session and each batch, after any export and before the
        acknowledgment.
    observations : int or None, optional
        Stop once this many observations were reported, the launch plane's included,
        instead of driving the session to its end.
    until : float or None, optional
        Stop after the batch of the first observation at this position, m.
    raises : bool, optional
        Raise a failed run's terminal error, as a service loop does; ``False`` returns
        the run with its terminal failure.

    Returns
    -------
    Run
        What the session reported; the caller closes the session.
    """
    session = started.session
    batches: list[EventBatch] = []
    fields: list[np.ndarray] = []
    seen = 0
    reached = False
    while not session.finished:
        batch = session.advance(work_limit=work_limit, deadline=deadline)
        batches.append(batch)
        for event in batch.events:
            if isinstance(event, ServiceRequired) and event.kind is ServiceKind.WARNING:
                # A warning barrier: emitted as a service loop emits it (SEP 47 5.5).
                emit_warning(event)
            if isinstance(event, ObservationReady):
                seen += 1
                reached = reached or (until is not None and event.z == until)
        if export is not None:
            fields.extend(
                np.array(session.export(event.ticket, export))
                for event in batch.events
                if isinstance(event, ObservationReady)
            )
        if between is not None:
            between(session, batch)
        session.acknowledge(batch.receipt)
        if reached or (observations is not None and seen >= observations):
            break
    run = Run(batches, fields, session.status())
    if raises and batches and batches[-1].terminal is not None:
        error = run.terminal.error
        if error is not None:
            raise error
    return run


def _history(history: BackendHistory | None) -> tuple[object, ...]:
    """Return a history's control columns as bytes: every column but the wall time.

    Parameters
    ----------
    history : solprop.backends.protocol.BackendHistory or None
        The history.

    Returns
    -------
    tuple
        The columns' bytes, or ``("none",)``.
    """
    if history is None or history.n_steps == 0:
        return ("empty",)
    return tuple(
        np.ascontiguousarray(column).tobytes()
        for column in (history.z, history.dz, history.error_norm, history.accepted)
    )


def _rows(rows: StepStats | None) -> tuple[object, ...]:
    """Return periodic rows as bytes.

    Parameters
    ----------
    rows : solprop.backends.protocol.StepStats or None
        The rows.

    Returns
    -------
    tuple
        Their positions, attempt indices and values, as bytes.
    """
    if rows is None or rows.n_rows == 0:
        return ("empty",)
    return (
        rows.z.tobytes(),
        rows.step.tobytes(),
        *(
            (name, np.ascontiguousarray(rows.values[name]).tobytes())
            for name in sorted(rows.values)
        ),
    )


def trace(run: Run) -> list[tuple[object, ...]]:
    """Return a run's numerical and control trace, exactly comparable.

    Every observation's kind, plane, position, assembled attempts (without wall times),
    periodic rows and statistics, and the terminal outcome with its assembled attempts
    and rows (:meth:`Run.assembled`). Snapshot and progress events are left out as
    events: their chunks are assembled into the next observation's.

    Parameters
    ----------
    run : Run
        The run.

    Returns
    -------
    list
        One tuple per observation and one for the terminal event.
    """
    found: list[tuple[object, ...]] = []
    for event, history, rows in run.assembled():
        if isinstance(event, ObservationReady):
            found.append(
                (
                    "observation",
                    event.kind,
                    event.plane_index,
                    event.z,
                    _history(history),
                    _rows(rows),
                    tuple(
                        (name, np.ascontiguousarray(event.stats[name]).tobytes())
                        for name in sorted(event.stats)
                    ),
                )
            )
        elif isinstance(event, Terminal):
            found.append(
                (
                    "terminal",
                    event.outcome,
                    event.z,
                    event.eligibility,
                    _history(history),
                    _rows(rows),
                )
            )
    return found


def assert_same_run(run: Run, baseline: Run) -> None:
    """Assert two runs have bit-identical traces and exported fields.

    Parameters
    ----------
    run : Run
        The run under test.
    baseline : Run
        The run it must reproduce.
    """
    got, expected = trace(run), trace(baseline)
    assert len(got) == len(expected), (len(got), len(expected))
    for index, (one, other) in enumerate(zip(got, expected, strict=True)):
        assert one == other, f"the traces differ at event {index}: {one[:4]}"
    assert len(run.fields) == len(baseline.fields)
    for index, (one, other) in enumerate(zip(run.fields, baseline.fields, strict=True)):
        assert one.tobytes() == other.tobytes(), f"field {index} differs"


def assert_close_trace(
    run: Run, baseline: Run, *, rtol: float, terminal_rtol: float | None = None
) -> None:
    """Assert two runs of one problem on different backends observe the same planes.

    The structure is exact -- each observation's kind, plane index and statistics
    names, and the terminal outcome and eligibility -- and the numbers agree to
    `rtol`: each observation's position and the terminal position, and each
    statistic relative to its largest magnitude. The attempts are not compared:
    where a trial's error norm is at the level of rounding, two backends' rounding
    takes different steps (SEP 25). This is the comparison against the reference of
    a native run whose bitwise baseline, the transitional bridge, was deleted in SEP
    47 phase 5.

    Parameters
    ----------
    run : Run
        The run under test.
    baseline : Run
        The reference run.
    rtol : float
        The relative bound.
    terminal_rtol : float or None, optional
        The bound on the terminal position, where a run stops at a check that falls
        on its own steps; `rtol` by default.
    """
    got, expected = run.observations, baseline.observations
    assert [(one.kind, one.plane_index) for one in got] == [
        (one.kind, one.plane_index) for one in expected
    ]
    for one, other in zip(got, expected, strict=True):
        assert one.z == pytest_approx(other.z, rtol)
        assert sorted(one.stats) == sorted(other.stats)
        for name in one.stats:
            mine = np.asarray(one.stats[name], dtype=np.float64)
            theirs = np.asarray(other.stats[name], dtype=np.float64)
            assert mine.shape == theirs.shape, name
            scale = float(np.max(np.abs(theirs), initial=0.0))
            np.testing.assert_allclose(
                mine, theirs, rtol=0.0, atol=rtol * scale, err_msg=name
            )
    terminal, reference = run.terminal, baseline.terminal
    assert (terminal.outcome, terminal.eligibility) == (
        reference.outcome,
        reference.eligibility,
    )
    assert terminal.z == pytest_approx(
        reference.z, rtol if terminal_rtol is None else terminal_rtol
    )


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?")


def masked(text: str) -> str:
    """Return a message with its numbers masked: what it says, not where.

    A native run checked against the reference takes other steps (SEP 25), so its
    warnings and failures name the same decisions at other positions and readings.

    Parameters
    ----------
    text : str
        The message.

    Returns
    -------
    str
        The message, each number replaced by ``#``.
    """
    return _NUMBER.sub("#", text)


def record_layout(record: Any) -> Any:
    """Return a monitor record's keys, flags and texts, its numbers left out.

    Parameters
    ----------
    record : object
        A monitor record, or one of its values.

    Returns
    -------
    object
        The same nesting, each number ``"number"`` and each sequence ``"list"``.
    """
    if isinstance(record, dict):
        return {key: record_layout(value) for key, value in record.items()}
    if isinstance(record, bool) or record is None or isinstance(record, str):
        return record
    if isinstance(record, (list, tuple, np.ndarray)):
        return "list"
    return "number"


RECORD_RTOL = 1e-6
"""How closely a native monitor record's real numbers follow the reference's, relative
to each: its energies and energy fractions, its positions and its gaps. The two
backends take the same decisions on steps whose lengths differ by rounding, at most
2e-7 relative, and whose readings round differently (SEP 25), a small reading
relatively more: measured 4.7e-8 at most, a resumed walker's guard reading of 4e-12 of
the launch's energy (2e-19 absolute), and 9e-9 elsewhere, the transverse edge's ramp
sums, each backend's own reading (``transverse.hpp``)."""

RECORD_ATOL = 1e-24
"""The absolute floor below :data:`RECORD_RTOL`, as a fraction of the launch's energy
(which nearly every number of a record is): an empty guard or collar reads the squared
modulus of a transform's rounding, of order ``n eps^2`` of the launch's energy, so a
reading of ``1e-33`` or ``1e-29`` is rounding on both sides; measured 2e-26 at most."""


def assert_record_close(
    record: Any,
    reference: Any,
    *,
    rtol: float = RECORD_RTOL,
    atol: float = RECORD_ATOL,
    what: str = "record",
) -> None:
    """Assert a monitor record is the reference's, its real numbers to a bound.

    The keys, their order, the flags, the texts and every integer -- the checked and
    skipped steps, the quiet checks, the reads -- are equal exactly, and each real
    number within ``atol + rtol |reference|`` (:data:`RECORD_RTOL`,
    :data:`RECORD_ATOL`); sequences element by element, at their own lengths.

    Parameters
    ----------
    record : object
        The record under test, or one of its values.
    reference : object
        The reference's.
    rtol, atol : float, optional
        The relative and absolute bounds on a real number.
    what : str, optional
        Where in the record, for the message.
    """
    if isinstance(reference, dict):
        assert isinstance(record, dict), what
        assert list(record) == list(reference), what
        for key, value in reference.items():
            assert_record_close(
                record[key], value, rtol=rtol, atol=atol, what=f"{what}.{key}"
            )
        return
    if isinstance(reference, (list, tuple, np.ndarray)):
        assert isinstance(record, (list, tuple, np.ndarray)), what
        assert len(record) == len(reference), what
        for index, (one, other) in enumerate(zip(record, reference, strict=True)):
            assert_record_close(
                one, other, rtol=rtol, atol=atol, what=f"{what}[{index}]"
            )
        return
    if reference is None or isinstance(
        reference, (bool, str, np.bool_, int, np.integer)
    ):
        assert record == reference, (what, record, reference)
        return
    error = abs(float(record) - float(reference))
    assert error <= atol + rtol * abs(float(reference)), (what, record, reference)


def committed_steps(run: Run) -> int:
    """Return the accepted attempts a run committed, assembled across its events.

    Parameters
    ----------
    run : Run
        The run.

    Returns
    -------
    int
        Its accepted attempts: the steps its monitor checked or skipped.
    """
    return sum(
        int(np.count_nonzero(history.accepted)) for _, history, _ in run.assembled()
    )


def one_thread() -> bool:
    """Return whether the native backend runs one thread, as the default lane does.

    Returns
    -------
    bool
        True when ``SOLPROP_NATIVE_THREADS`` or ``SOLPROP_THREADS`` is ``"1"``.
    """
    return _one_thread()


def pytest_approx(value: float, rtol: float) -> object:
    """Return ``pytest.approx(value, rel=rtol)``, imported where a test calls it.

    Parameters
    ----------
    value : float
        The expected value.
    rtol : float
        The relative bound.

    Returns
    -------
    object
        The approximate value.
    """
    import pytest  # noqa: PLC0415 - only a test calls it

    return pytest.approx(value, rel=rtol)


@dataclass
class SpyMonitor(BoundaryMonitor):
    """A boundary monitor that writes its launch and its steps into a shared log.

    Attributes
    ----------
    log : list
        Where ``("launch", z)`` and ``("step", z)`` entries go; a test writes its own
        markers into the same list.
    """

    log: list[tuple[str, float]] = field(default_factory=list)

    def launch(self, energies, z, **kwargs):
        """Log the launch, then record it."""
        self.log.append(("launch", float(z)))
        super().launch(energies, z, **kwargs)

    def step(self, energies, z, **kwargs):
        """Log a checked step, then account it."""
        self.log.append(("step", float(z)))
        super().step(energies, z, **kwargs)

    def skip(self, z):
        """Log an unchecked step, then count it."""
        self.log.append(("step", float(z)))
        super().skip(z)


def spy_monitor(log: list[tuple[str, float]]) -> SpyMonitor:
    """Return a fresh spy monitor for :func:`window`, writing into `log`.

    Parameters
    ----------
    log : list
        The shared log.

    Returns
    -------
    SpyMonitor
        The monitor, with nothing recorded.
    """
    from solprop.monitor import BoundaryCheck  # noqa: PLC0415 - beside its one use

    built = BoundaryMonitor.build(window(), None, BoundaryCheck())
    return SpyMonitor(
        built.check,
        built.weights,
        built.kind,
        built.columns,
        built.split,
        built.interval,
        log=log,
    )

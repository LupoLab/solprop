# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference backend: its lowering, its marches and the refusals (SEP 52).

One :class:`NumpyRefBackend` steps one problem in plain NumPy. It builds the
workspace that holds the state and does every state-sized operation
(:class:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace`), the engines a march
asks of it (:class:`~solprop.backends.numpy_ref.engines.NumpyRungeKutta`,
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`,
:class:`~solprop.backends.low_storage.HostLowStorage`) and the run's march, and keeps
``prepare``, the metadata, the memory ledger, the timing probe and ``close``. The step
control follows ``docs/dev/stepper.md`` section 3 exactly: the controller's
*proposal* is its state, the executed step's own norm decides acceptance and is what
the history records, and the equivalent norm of the proposal is what the controller
and a rejection see.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from jaxtyping import Complex

from solprop.backends.exponential import (
    CarrierReference,
    refuse_carrier_frame,
    refuse_formulation,
)
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.runge_kutta import RungeKuttaMarch
from solprop.backends.low_storage import HostLowStorage
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.numpy_ref.workspace import (
    BlockResidual,
    NumpyWorkspace,
    SourceTables,
)
from solprop.backends.records import adams_policy, runge_kutta_policy
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import (
    MAX_N_POL,
    PropagationProblem,
    StepperSettings,
    validate_run,
)
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    SpatialRoute,
    route_name,
    tdse_descriptor,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = ["BlockResidual", "NumpyRefBackend", "SourceTables"]

logger = logging.getLogger(__name__)


class NumpyRefBackend(HostDriver[Complex[np.ndarray, "n_modes n_pol n_omega"]]):
    """Propagate one Kerr and plasma problem with interaction-picture DOPRI5.

    The oracle of ``docs/dev/backends.md`` section 2. It implements
    :class:`solprop.backends.protocol.Backend` and steps a single problem on either
    grid: mode-averaged or multimode on the fixed transverse quadrature, scalar or
    vector-isotropic, with the Kerr term, the delayed Raman term and the plasma.
    Mode-averaged propagation is the one-node case of the same pipeline and not a
    second code path.

    **Refusals.** A problem is refused by name, with
    :class:`solprop.exceptions.BackendError`, when its flags declare a spectral
    response, when a Raman kernel meets two polarisation components, or when the
    transform and the state would both carry the transverse components. A term
    silently dropped would run, converge and be wrong.

    **Convention.** Every array held or returned is in the FFT convention of
    ``docs/dev/conventions.md`` section 2; the backend never conjugates. The
    output layer converts on the way to disk and the processing accessors on the way to
    a user.

    **The grid.** :meth:`stats` delegates to :func:`solprop.stats.collect`, which needs
    a :class:`solprop.grid.TimeGrid`; the kernel contract carries the array group
    :class:`solprop.translate.masks.Grids` and no grid object, so the grid is supplied
    here instead. This is a known gap in the contract.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated kernel problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies: method, error control, itinerary, planes and statistics.
        :func:`~solprop.translate.problem.validate_run` checks the pair again here,
        before anything is allocated: validation is the backend's defence, not the
        caller's courtesy.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, needed only by :meth:`stats`.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, needed only by the profile statistics.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned persistent atomic plans, required exactly when the problem
        declares a TDSE response. The driver closes this binding after the backend.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options, from which the node chunk is planned; ``None``
        for the automatic choice.

    Raises
    ------
    BackendError
        Naming what the backend refuses.

    Examples
    --------
    >>> from solprop.backends.numpy_ref import NumpyRefBackend
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.assembly import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> backend = NumpyRefBackend(problem, request, grid=grid)
    >>> backend.set_initial(problem.initial.field0, None)
    >>> summary = backend.propagate_to(0.01)
    >>> summary.z, summary.n_rejected, summary.met_tolerance
    (0.01, 0, True)
    >>> backend.observe().field.shape
    (1, 1, 768)
    >>> backend.close()
    """

    reads_transverse_edge = True

    name = "numpy_ref"
    # The oracle evaluates its stage tables and Adams bases with the uncompiled forms,
    # never with the compiled code it exists to check (SEP 29 section 6.6).
    helpers = NumpyWorkspace.helpers
    # The reference forms `exp(+dPhi)` and `exp(-dPhi)` afresh at every stage, so
    # keeping the six distinct exponents of a step is worth its memory here where it is
    # not on a backend that takes both signs from one transcendental.
    propagator_default = "tabulate"

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
        options: ExecutionOptions | None = None,
    ) -> None:
        """Adopt a validated problem and request; refuse what is not executed here."""
        _refuse_unsupported(problem, request.stepper)
        validate_run(problem, request)
        if (tdse_descriptor(problem.route) is None) != (tdse_binding is None):
            raise BackendError(
                "TDSE problems require an explicit host response binding"
            )
        super().__init__(
            problem, request, grid=grid, modes=modes, profiles=profiles, options=options
        )
        self._workspace = NumpyWorkspace(
            problem,
            request,
            chunk_plan=self._chunk_plan,
            propagator_mode=self._propagator_mode,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
            name=self.name,
        )
        stepper = self._stepper
        self._runge_kutta_engine = self._runge_kutta(self._workspace)
        integrator = stepper.integrator
        if isinstance(integrator, Multistep):
            self._use_adams(integrator)
        elif isinstance(integrator, LowStorageRungeKutta):
            # The host engine is the oracle (SEP 24 section 6.5).
            schedule = stepper.schedule
            assert schedule is not None  # the translation refused it without one
            engine = HostLowStorage(
                self._workspace, integrator, schedule, self._propagator_mode
            )
            self._low_storage = HostLowStorageMarch(
                engine,
                integrator,
                self._hooks,
                schedule=schedule,
                proposal=stepper.dz_init,
                lag=stepper.controller.lag,
                name=self.name,
            )
            self._march = self._low_storage
        else:
            self._march = RungeKuttaMarch(
                self._runge_kutta_engine,
                runge_kutta_policy(problem, stepper),
                self._hooks,
                schedule=stepper.schedule,
                proposal=stepper.dz_init,
                name=self.name,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "numpy_ref: %s grid, shape %s, %d Phi intervals, %s propagators, "
                "%s schedule",
                problem.grid_kind,
                problem.shape,
                problem.linear.n_intervals,
                self._propagator_mode,
                "replayed" if stepper.schedule is not None else "adaptive",
            )

    def _runge_kutta(self, workspace: NumpyWorkspace) -> NumpyRungeKutta:
        """Return the Runge-Kutta engine over the workspace; a test may script one."""
        return NumpyRungeKutta(workspace, self._stepper.tableau)

    def _use_adams(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"] = "transported",
    ) -> None:
        """Build the Adams engine and march, and step the run with them.

        - ``compensation``: how a varying operator's linear compensation is integrated
          (SEP 16 Q8): ``"transported"``, the delivered treatment, or
          ``"untransported"``, kept as a test witness only.
        """
        # The engine reads this package's passes, so it is imported when a run asks
        # for it rather than with the package.
        from solprop.backends.host.adams import HostAdamsMarch  # noqa: PLC0415
        from solprop.backends.numpy_ref.multistep import (  # noqa: PLC0415
            AdamsKernels,
            ExponentialKernels,
            HostHistory,
        )

        stepper = self._stepper
        interpolate = bool(stepper.interpolate_planes)
        engine = HostHistory(
            self._workspace,
            method,
            AdamsKernels(helpers=self.helpers),
            # SEP 16's exponential formulation keeps untransported samples in the same
            # store and shares everything else: the lifecycle, the controller, the
            # ledger.
            ExponentialKernels(helpers=self.helpers)
            if method.formulation == "exponential"
            else None,
            compensation,
            interpolate=interpolate,
        )
        carrier = engine.carrier
        # Every number the controller and the landings read, from one record (SEP 39
        # section 4.2). An in-stage temporal collar's largest rate, the one stiff
        # term the exponential history integrates, bounds each order's stable step.
        policy = adams_policy(
            self._problem,
            stepper,
            method,
            carrier=isinstance(carrier, CarrierReference) and carrier.active,
        )
        self._adams = HostAdamsMarch(
            engine,
            method,
            self._hooks,
            policy=policy,
            field=self._workspace.committed,
            schedule=stepper.schedule,
            interpolate=interpolate,
            shape=self._problem.shape,
            proposal=stepper.dz_init,
            lag=stepper.controller.lag,
            name=self.name,
        )
        self._march = self._adams

    @property
    def workspace(self) -> NumpyWorkspace:  # numpydoc ignore=RT01
        """The workspace: the state and every state-sized operation on it."""
        return self._workspace

    def metadata(self) -> dict[str, object]:
        """Return execution facts and cumulative logical and atomic call ledgers.

        Returns
        -------
        dict
            Provider-free, serializable facts. A PEC predictor is explicitly
            labelled and is never reported as an accepted-field diagnostic.
        """
        record = super().metadata()
        record.update(self._workspace.metadata())
        return record

    def restore_call_ledgers(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore cumulative counters while leaving the Adams history empty.

        Parameters
        ----------
        logical : mapping
            Persisted logical call counters and accepted-order record.
        atomic : mapping or None, optional
            Persisted atomic attempts; unused reservations are released.

        Raises
        ------
        BackendError
            If a ledger is incompatible with this run.
        """
        super().restore_call_ledgers(logical)
        if atomic:
            self._workspace.restore_atomic_ledger(atomic)

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the backend works in, in bytes.

        The workspace's (:meth:`NumpyWorkspace.memory_ledger`): what one step has
        resident at its peak, every buffer named even when it is empty.

        Returns
        -------
        dict
            Buffer name to bytes.
        """
        return self._workspace.memory_ledger()

    def close(self) -> None:
        """Release the buffers. Idempotent."""
        self._workspace.close()
        if self._low_storage is not None:
            engine = self._low_storage.engine
            assert isinstance(engine, HostLowStorage)
            engine.close()
        self._closed = True


def _refuse_adams(problem: PropagationProblem, stepper: StepperSettings) -> None:
    """Refuse what the Adams march does not execute (SEP 13 sections 2 and 6.2).

    Its history is of source samples along a diagonal flow, so a coupled flow is
    refused by name rather than stepped with a stale history, and the exponential
    formulation refuses a varying operator until SEP 16's phase 4
    (:func:`solprop.backends.exponential.refuse_formulation`). The ``modified`` noise is
    admitted for both formulations (SEP 16 section 4.4): its draw is fixed for the run
    and added to each sample's field at that sample's own position, never to the
    propagating state, so conditional on the draw the source is a deterministic,
    nonautonomous function of position that the history interpolates. The split-step
    temporal collar is admitted: it changes
    the state only inside the collar, where its first-order splitting error already
    lies (``docs/dev/stepper.md`` section 6), and it removes the collar's stability
    limit, which binds the Adams methods long before Runge-Kutta's (SEP 13 phase 2).
    """
    integrator = stepper.integrator
    if not isinstance(integrator, Multistep):
        return
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        # SEP 16 Q3: only the exponential formulation steps a coupled flow; the
        # interaction one would transport its history through inverse actions.
        msg = (
            f"the {integrator.name} integrator does not step a problem with a "
            "coupled linear flow; use Multistep(formulation='exponential') or the "
            "Runge-Kutta march"
        )
        raise BackendError(msg)
    refuse_formulation(problem, stepper, "numpy_ref")
    # SEP 35: the scalar shift, on the guided routes.
    refuse_carrier_frame(problem, stepper, "numpy_ref", delivered=frozenset({"scalar"}))


def _refuse_unsupported(problem: PropagationProblem, stepper: StepperSettings) -> None:
    """Refuse, by name, every problem this backend does not execute."""
    # The route first, by its variant (SEP 47 section 6.2): the reference executes
    # every route of the union and the modal route's TDSE extension.
    match problem.route:
        case (
            ModalRoute()
            | GeneralRoute()
            | SpatialRoute(kind="radial" | "cartesian" | "planar" | "line")
        ):
            pass
        case route:
            msg = f"numpy_ref does not execute the {route_name(route)} route"
            raise BackendError(msg)
    _refuse_adams(problem, stepper)
    if (
        problem.flags.has_raman
        and problem.transform.n_pol * problem.shape[1] > 1
        and (
            not problem.responses.raman_terms
            or not all(term.tensor_capable for term in problem.responses.raman_terms)
        )
    ):
        raise BackendError(
            "vector Raman requires tensor-capable terms; found scalar_only"
        )
    _, n_pol, _ = problem.shape
    if problem.transform.n_pol * n_pol > MAX_N_POL:
        msg = (
            f"the transform resolves {problem.transform.n_pol} transverse components "
            f"and the state carries {n_pol}; the components belong to one of the two "
            "and not to both, so this problem would count them twice"
        )
        raise BackendError(msg)

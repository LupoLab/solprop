# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solprop-dependent native backend adapter."""

from __future__ import annotations

import dataclasses
import logging
import math
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from solprop._process import acquire_host_blas, thread_budget
from solprop.backends import tdse_source
from solprop.backends.exponential import carrier_reference, refuse_adams
from solprop.backends.ledger import source_absent
from solprop.backends.node_chunk import (
    NodeChunkPlan,
    general_planned_chunk,
    plan_node_chunk,
)
from solprop.backends.protocol import TimingProbe
from solprop.backends.records import (
    RunRecords,
    low_storage_settings,
    run_records,
)
from solprop.backends.session import (
    ExecutionOptions,
    PreparedExecution,
    Start,
)
from solprop.backends.spatial import FarFieldTables
from solprop.boundaries import SPLIT_RATE_BUCKETS
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.profiles import ZProfile
from solprop.spatial import FAR_FIELD_STATISTICS
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.tableau import continuous_extension
from solprop.translate.coupled import (
    CoupledFlowFactors,
    CoupledFlowTables,
    coupled_flow_factors,
    schur_eigenbasis,
)
from solprop.translate.linop import (
    delta_phi,
    unitary_phase,
)
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import (
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
    validate_run,
)
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    SpatialRoute,
    tdse_descriptor,
    transverse_of,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.stage import (
    StageScalars,
    combine,
    require_device_ready,
    terms,
)
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding
from solprop.transverse_monitor import (
    TransverseEdge,
    _transverse_edge_bytes,  # pyright: ignore[reportPrivateUsage] - released together
)

# The profile's state is read from the package at each use: a first-use calibration in
# this process refreshes it (solprop SEP 46 section 5).
import solprop_native as _package

from . import (
    PROC_BIND,
    THREAD_BUDGET,
    WAIT_POLICY,
    _core,
    _counters,
    _host_blas,
    _wisdom,
    build_info,
    check_build,
)
from ._device import _build_facts, _on_device, _resident_flows, native_node_chunk
from ._execution import ExecutionRecord
from ._helpers import NATIVE
from ._host_source import TDSEHostSource
from ._load import Loaded, StageCache, load, propagator_mode
from ._lowering import (
    _chord_tables,
    _edge_lines,
    _flux_tables,
    _lower,
    _LowerPlan,
    _section_tables,
    _ZTables,
)
from ._march import _envelope_errors, _NativeAdamsMarch
from ._probe import NativeProbe
from ._statistics import (
    StatisticsLowering,
    metric_refusal,
)

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


ADAMS_MAX_CHANNELS = 16
"""The most channels of a coupled block whose channel vectors a CPU build's resident
Adams engine holds on the stack.

The native ``adams_max_channels``. Either build steps any width (solprop SEP 47 phase
4): a CPU build's per-bin passes hold a wider block's channel vectors in a row per bin,
and a device build's passes take a thread per (bin, channel) and hold none.
"""


class NativeBackend:
    """Propagate modal, radial or Cartesian fields with persistent Kokkos state.

    Parameters
    ----------
    problem
        Validated kernel problem. Accepts modal or spatial scalar or two-component
        propagation (radial, Cartesian, or the hybrid planar kind of slab modes in x
        times an FFT in y), instantaneous
        Kerr, plasma, scalar time-domain Raman, weighted control, both absorber
        schemes, and device-owned statistics; and solprop's general modal route, on a
        constant guide or a moving one whose maps are combined per stage. Statistics
        are evaluated and finished natively (solprop SEP 47 section 6.4).
    grid, modes, profiles
        Host context used only for plane statistics.
    cpu_execution
        A CPU execution record to build the workspace with (a SEP 46 refinement's);
        the process's own by default.
    fft_planning
        An FFT planning to apply (``fft_planning``); the plan policy by default.
    far_field
        The far-field tables already formed for this problem (a native session's),
        formed here by default.
    options
        The run's execution options, from which the node chunk is planned.

    The backend is a resource holder: it lowers the problem and request into the
    native state once, and every run is a native session it prepares
    (:meth:`prepare`, solprop SEP 47 section 5.2) or a timing probe's trials
    (:meth:`timing_probe`). It keeps no march state between them.

    Raises
    ------
    solprop.exceptions.BackendError
        If the build or problem lies outside the implemented seam contract.
    """

    name = "native"
    helpers = NATIVE
    far_field_statistics: ClassVar[frozenset[str]] = frozenset(FAR_FIELD_STATISTICS)
    """The far-field statistics of solprop SEP 23 this build forms natively.

    solprop's registry admits them on this backend by this attribute and refuses
    them, by name, on a build without it.
    """

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        cpu_execution: ExecutionRecord | None = None,
        fft_planning: Mapping[str, object] | None = None,
        far_field: FarFieldTables | None = None,
        options: ExecutionOptions | None = None,
        tdse_binding: TDSEBinding | None = None,
    ) -> None:
        check_build()
        stepper = request.stepper
        _refuse_unsupported(problem, stepper)
        _refuse_tdse(problem, tdse_binding)
        validate_run(problem, request)
        stats = request.observation.stats
        refused = metric_refusal(problem, (*stats.names, *stats.periodic))
        if refused is not None:
            # Before anything is allocated (SEP 47 phase 4's decision).
            raise BackendError(f"native refuses {refused}")
        # solprop SEP 29 G1: a z table is evaluated natively on a CPU build; a device
        # build refuses it, naming the table's size and the CPU backends.
        if not problem.device_ready and _on_device():
            require_device_ready(problem, "native")
        # solprop SEP 47 section 6.3: the native stage tables read every
        # representation the translation forms; one they do not read is refused here,
        # before anything is allocated, not stepped with rows the host forms.
        refused = host_rows_refusal(problem)
        if refused is not None:
            raise BackendError(refused)
        self._problem = problem
        self._request = request
        # The stepping policy as one record (`RunRequest.stepper`).
        self._stepper = stepper
        # SEP 47 section 6.5: the node chunk is an execution choice, planned from the
        # problem and the options, and lowered for this build below.
        self._options = ExecutionOptions() if options is None else options
        self._chunk_plan: NodeChunkPlan = plan_node_chunk(problem, self._options)
        # The fixed arrays of the stage seam, held for the backend's life; the weights
        # that multiply them are a bundle of host floats per stage (`_stages`).
        self._terms = terms(problem)
        self._stages = StageCache(problem, self.helpers)
        self._grid = grid
        self._modes = modes
        self._profiles = profiles
        self._closed = False
        # The start the native state was last loaded with (`_load`).
        self._loaded: Loaded | None = None
        # solprop SEP 47 section 6.5: this backend's lowering of the host plan, and
        # the general route's own chunk at the plan's budget.
        self._node_chunk = native_node_chunk(
            problem, self.chunk_plan, device=_on_device()
        )
        self._general_chunk = (
            None
            if problem.transform.modal_tables is None
            else general_planned_chunk(problem, self.chunk_plan)
        )
        self._z_tables = None if problem.device_ready else _ZTables(problem, stepper)
        self._flow = _host_flow(problem)
        tables = problem.transform.modal_tables
        self._moving_maps = (
            tables is not None
            and tables.synthesis is not None
            and tables.synthesis.shape[0] * tables.synthesis.shape[1] > 1
        )
        # solprop SEP 23: the far-field tables are formed once, which refuses a
        # geometry the far-field statistics do not admit before anything is built.
        self._far_field = (
            FarFieldTables(problem, request.observation.stats)
            if far_field is None
            else far_field
        )
        # solprop SEP 47: the native statistics plans lowered for the specs asked.
        self._statistics_plans: dict[
            tuple[int, tuple[str, ...]], tuple[StatsSpec, StatisticsLowering]
        ] = {}
        self._statistics_tables: dict[str, dict[str, object]] = {}
        # solprop ADR-076: the plans runs of this machine and build measured before
        # are offered to the workspace, and those it measures are kept after it.
        wisdom = _wisdom.wisdom_file() if fft_planning is None else None
        stored, self._wisdom_loaded = (
            _wisdom.plans(wisdom) if fft_planning is None else ({}, False)
        )
        # solprop SEP 46: the workspace's CPU execution record, the process's (the
        # calibration's where it holds, else the built-in table) unless one is given,
        # as a stage-2 refinement gives one; read back from the workspace it built.
        self._execution_source = (
            _package.EXECUTION_SOURCE if cpu_execution is None else "given"
        )
        integrator = stepper.integrator
        # The run's carrier frame (solprop SEP 35), whose chord the stage tables read.
        self._carrier = (
            carrier_reference(problem, integrator)
            if isinstance(integrator, Multistep)
            else None
        )
        plan = _LowerPlan(
            terms=self._terms,
            flow=self._flow,
            tables=self._z_tables,
            node_chunk=self._node_chunk,
            general_chunk=self._general_chunk,
            cpu_execution=cpu_execution,
            fft_planning=_planning(fft_planning, stored),
            device=_on_device(),
            setup=self._setup(),
        )
        # One workspace for the backend's life (solprop SEP 52 section 5.3): the
        # problem and every set-once table, guard and geometry cross once, here.
        self._workspace = _core.Workspace(_lower(problem, request, plan))
        diagnostics = self._workspace.diagnostics()
        self._applied_execution = ExecutionRecord.from_native(
            diagnostics["cpu_execution"]
        )
        if cpu_execution is None:
            # A stage-2 refinement of this very execution (solprop SEP 46 section
            # 6.6): the workspace is built again with its record.
            refined = _refined_record(problem, stepper, diagnostics)
            if refined is not None:
                stored |= _keep_plans(diagnostics, wisdom)
                self._workspace.close()
                self._workspace = _core.Workspace(
                    _lower(
                        problem,
                        request,
                        dataclasses.replace(
                            plan,
                            cpu_execution=refined[0],
                            fft_planning=_planning(fft_planning, stored),
                        ),
                    )
                )
                diagnostics = self._workspace.diagnostics()
                self._execution_source = refined[1]
                self._applied_execution = ExecutionRecord.from_native(
                    diagnostics["cpu_execution"]
                )
        # The workspace's facts that its build fixes, for every metadata record; its
        # counters are read live (`Workspace.counters`).
        self._facts: dict[str, Any] = dict(diagnostics)
        # solprop SEP 49: the TDSE response is the march's host source, handed to each
        # run and probe; the driver owns the binding and closes it.
        self._tdse_binding = tdse_binding
        self._host_source: TDSEHostSource | None = None
        if tdse_binding is not None:
            self._host_source = TDSEHostSource(tdse_binding, problem, self._terms)
        if fft_planning is None:
            _keep_plans(diagnostics, wisdom)
        self._wisdom_digest = (
            _wisdom.digest(
                _applied_plans(
                    diagnostics,
                    stored,
                    fft_planning if fft_planning is not None else _package.PLANNING,
                )
            )
            if not _on_device()
            else None
        )
        # The native Adams march's record (SEP 39): its policy, ledger and history,
        # over the resident history the workspace keeps.
        self._adams: _NativeAdamsMarch | None = (
            _NativeAdamsMarch(self, integrator, diagnostics["history_rows"])
            if isinstance(integrator, Multistep)
            else None
        )
        # The low-storage march (SEP 24) is the native session's; its resident
        # registers' step is a timing probe's trial (`timing_probe`), and the
        # evaluations the probes make are counted for the metadata.
        self._low_storage = isinstance(integrator, LowStorageRungeKutta)
        self._low_storage_evaluations = 0
        # The open timing probe, which leases the workspace until the next run or probe.
        self._probe: _core.Probe | None = None
        # NumPy's own BLAS pool would spin beside the native one while this backend
        # lives; it is held to one thread until the last native backend closes.
        self._host_blas = _host_blas.hold(
            build_info().blas_library,
            THREAD_BUDGET.threads,
            acquire=acquire_host_blas,
        )

    @property
    def request(self) -> RunRequest:  # numpydoc ignore=RT01
        """Run request this backend was built for (solprop SEP 47 5.2).

        Its method, error control, itinerary, planes and statistics; a run adapts
        it with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        return self._request

    @property
    def chunk_plan(self) -> NodeChunkPlan:  # numpydoc ignore=RT01
        """Host rule's node-chunk plan, made at construction (SEP 47 6.5)."""
        return self._chunk_plan

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the backend has released its buffers and can run no other input."""
        return self._closed

    def _load(
        self,
        field: ComplexArray,
        noise: NoiseRealisation | None,
        *,
        z: float,
        resume: bool,
    ) -> Loaded:
        """Form a start's load as a fresh backend's (SEP 22 4.7).

        The field and the noise realisation (:func:`solprop_native._load.load`) and the
        Adams record's history and accounting started afresh; the workspace takes the
        load with the run's start (`_run_start`), which puts the stages' collar back
        in, drops the recycled stage and binds no monitor before a session or a timing
        probe goes on from it.

        - ``field``: the state, ``complex128`` of shape ``(n_modes, n_pol, n_omega)``,
          of any strides.
        - ``noise``: the realisation, or ``None`` for the problem's own.
        - ``z``: where the state belongs, m.
        - ``resume``: the state came back from a saved plane, taking no ``input`` draw.
        """
        if self._closed:
            raise BackendError("this backend is closed; its buffers were released")
        _require_initial_array(field, self._problem.shape)
        loaded = load(self._problem, self._stepper, field, noise, z=z, resume=resume)
        if (counters := _counters.active) is not None:
            counters.load("start", loaded.field.nbytes)
        if self._adams is not None:
            self._adams.reset()
            self._adams.begin_run()
        self._low_storage_evaluations = 0
        return loaded

    def _restore(self, start: Start, loaded: Loaded) -> _Restored:
        """Restore a resumed start's ledgers and method history after its load.

        - Returns: why the snapshot's history was not installed (``None`` when it was or
          none was given), the next step's proposal, m (the snapshot's when its
          history continues, else ``dz_init``), and the history's own state and rows
          the run then starts from.
        """
        adams = self._adams
        proposal = float(self._stepper.dz_init)
        if start.ledgers is not None:
            atomic = start.ledgers.get("atomic", {})
            if atomic:
                if self._tdse_binding is None:
                    raise BackendError(
                        f"{self.name} does not restore atomic response call ledgers "
                        "without a TDSE binding"
                    )
                self._tdse_binding.restore_ledger(
                    AtomicCallLedger.from_record(cast("Mapping[str, object]", atomic))
                )
            if adams is not None:
                adams.restore_ledger(
                    cast("Mapping[str, object]", start.ledgers.get("logical", {}))
                )
        snapshot = start.march
        if snapshot is None:
            return _Restored(None, proposal)
        if adams is None:
            march = "low-storage" if self._low_storage else "Runge-Kutta"
            return _Restored(
                f"the run steps with the {march} march, which keeps no history",
                proposal,
            )
        reason, rows = adams.resume(snapshot, loaded.z)
        if reason is not None:
            return _Restored(reason, proposal)
        if (counters := _counters.active) is not None:
            counters.load("restore", snapshot.state.nbytes)
        return _Restored(
            None,
            float(snapshot.proposal),
            np.ascontiguousarray(snapshot.state, dtype=np.complex128).reshape(-1),
            rows,
        )

    def _setup(self) -> dict[str, object]:
        """Return the workspace's set-once data (solprop SEP 52 section 5.3).

        The stage tables, the z tables and their slots, a coupled flow's guard, the
        Runge-Kutta trial's nodes and error control, the resident Adams history, the
        low-storage settings, the split collar's occupancy buckets and the crossings'
        threads, each absent where the problem has none.
        """
        problem, stepper = self._problem, self._stepper
        integrator = stepper.integrator
        controller = stepper.controller
        weights = (
            np.empty(0)
            if controller.weights is None
            else np.ascontiguousarray(controller.weights, dtype=np.float64)
        )
        adams = isinstance(integrator, Multistep)
        dense = (
            continuous_extension(stepper.tableau)
            if stepper.interpolate_planes and not adams
            else None
        )
        tables = self._z_tables
        setup: dict[str, object] = {
            "stage_tables": self._stage_tables(),
            "z_slots": None if tables is None else tables.slots_record(),
            "z_tables": ([] if tables is None else tables.records())
            + _flux_tables(problem),
            "flow_guard": _flow_guard(self._flow),
            "runge_kutta": {
                "nodes": np.ascontiguousarray(stepper.tableau.nodes, dtype=np.float64),
                "rtol": float(controller.rtol),
                "atol": float(controller.atol),
                "weights": weights,
                "dense_weights": np.empty(0)
                if dense is None
                else np.ascontiguousarray(dense, dtype=np.float64),
            },
            "adams": None,
            "low_storage": None,
            "occupancy": None,
            "crossing_workers": thread_budget().threads,
        }
        if isinstance(integrator, Multistep):
            setup["adams"] = {
                "slots": integrator.order + 2,
                "compensated": self._compensated(integrator, "transported"),
                "interpolate": bool(stepper.interpolate_planes),
                "fused": integrator.formulation != "exponential",
                "rtol": float(controller.rtol),
                "atol": float(controller.atol),
                "weights": weights,
            }
        settings = low_storage_settings(problem, stepper)
        if settings is not None:
            setup["low_storage"] = dataclasses.replace(
                settings,
                flows=_resident_flows(settings.flows, stepper, device=_on_device()),
            )
        crossing = problem.split_crossing
        if crossing is not None:
            # The split collar's walk-off bucket of each bin (solprop #440), bound by
            # the first run whose monitor reads its crossing limit.
            setup["occupancy"] = {
                "buckets": np.ascontiguousarray(
                    crossing.buckets, dtype=np.uint8
                ).reshape(-1),
                "count": SPLIT_RATE_BUCKETS,
            }
        return setup

    def _compensated(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"],
    ) -> bool:
        """Return whether the resident Adams history carries a compensation (SEP 16).

        Either formulation keeps its history, its step arithmetic and its norms in the
        workspace (solprop ADR-043), beside a coupled flow too (SEP 16 phase 6), whose
        interaction formulation the admission refuses (`refuse_adams`). A varying
        operator's exponential formulation carries the transported compensation
        natively, as does a frozen-residual flow's residual, whatever the diagonal's
        representation; the untransported test witness is the reference's alone.
        """
        flow = self._flow
        linear = self._problem.linear
        residual = flow is not None and flow.residual_terms is not None
        compensated = method.formulation == "exponential" and (
            residual
            or (
                linear.phi_representation in ("expansion", "table")
                and not source_absent(self._problem)
            )
        )
        if compensated and compensation != "transported":
            msg = (
                "solprop-native steps the transported compensation only: the "
                "untransported one is a test witness of backend='numpy_ref'"
            )
            raise BackendError(msg)
        return compensated

    def _stage_tables(self) -> dict[str, object] | None:
        """Return the tables the native marches read each stage's scalars from.

        solprop SEP 39 phases 1 and 4: every scalar table ``translate.stage.scalars``
        reads at a stage -- the weights of a constant or expanded ``Phi`` and
        prefactor, the density, the taper scale, the Raman decay rates and densities,
        the Kerr density, the plasma channels' densities and fractions, SEP 6's
        channel densities -- and an expansion's rate table, each packed as its batch
        pass reads it (``PiecewiseChebyshev.packed``). A separable operator's rows are
        a constant's, the distance and the position with a unit prefactor, and read no
        span. A z table's rows are its slots, which the workspace loads through the
        cache the z tables share. A coupled flow's section and residual weights and a
        moving guide's map weights are sectioned Chebyshev rows, formed natively from
        their edges; the carrier frame's chord is read beside them
        (:func:`_chord_tables`). ``None`` where the rows are not tables, which the
        admission refuses (:func:`host_rows_refusal`).
        """
        problem = self._problem
        if not stage_tables_available(problem):
            return None
        linear, responses = problem.linear, problem.responses
        phi_weights = phi_rate = prefactor_weights = None
        separable = self._terms.separable is not None
        tables = self._z_tables
        if tables is not None and tables.phi:
            phi = "table"
        elif separable or linear.phi_constant is not None:
            phi = "constant"
        elif linear.phi_expansion is not None:
            phi = "expansion"
            phi_weights = linear.phi_expansion.weights.packed()
            phi_rate = linear.rate.packed()
        else:
            return None
        low = high = 0.0
        if tables is not None and tables.prefactor:
            prefactor = "table"
        elif separable:
            prefactor = "constant"
            low, high = -math.inf, math.inf
        elif responses.prefactor_constant is not None:
            prefactor = "constant"
            low, high = responses.prefactor_span
        elif responses.prefactor_expansion is not None:
            prefactor = "expansion"
            prefactor_weights = responses.prefactor_expansion.weights.packed()
        else:
            return None
        kerr = responses.kerr_coefficients
        taper = problem.transform.taper_scale
        return (
            {
                "n_stages": self._stepper.tableau.n_stages,
                "phi": phi,
                "phi_weights": phi_weights,
                "phi_rate": phi_rate,
                "prefactor": prefactor,
                "prefactor_weights": prefactor_weights,
                "prefactor_low": float(low),
                "prefactor_high": float(high),
                "density": responses.density.packed(),
                "taper": None if taper is None else taper.packed(),
                "raman_rates": [
                    table.packed()
                    for term in responses.raman_terms
                    for table in term.decay_rates
                ],
                "kerr": "none"
                if kerr is None
                else "density"
                if kerr.density is None
                else "table",
                "kerr_density": None
                if kerr is None or kerr.density is None
                else kerr.density.packed(),
                "plasma_densities": [
                    channel.density.packed()
                    for channel in responses.population_channels
                ],
                "plasma_preionisation": [
                    channel.preionisation.packed()
                    for channel in responses.population_channels
                ],
                "raman_densities": [
                    None if term.density is None else term.density.packed()
                    for term in responses.raman_terms
                ],
                "dispersive_densities": []
                if responses.dispersive_kerr is None
                else [table.packed() for table in responses.dispersive_kerr.densities],
            }
            | _section_tables(
                problem,
                moving=self._moving_maps,
                residual=self._flow is not None
                and self._flow.residual_terms is not None,
            )
            | _chord_tables(self._carrier)
        )

    @property
    def _propagator_mode(self) -> PropagatorMode:
        """Return the flow treatment this run uses, ``"auto"`` resolved.

        Only the low-storage march holds flows, so every other march recomputes whatever
        was asked (#464). That march resolves as solprop's driver does, except that a
        device build forms its flows per element under ``"auto"``: there an extra field
        read costs more than the exponential (SEP 24 section 4.3).
        """
        return propagator_mode(self._problem, self._stepper, device=_on_device())

    def _unitary_phase(self, bundle: StageScalars) -> RealArray:
        """Return ``Im Phi`` at the stage's position, formed on the host.

        An expansion of ``Phi`` is cumulative and vanishes at the start of the range,
        so its value at a position is the bundle's ``phi_at`` against the same terms. A
        z table's value is the workspace's (``Session.noise_phase``).
        """
        if bundle.phi_at is None or self._terms.phi is None:
            return unitary_phase(self._problem.linear, bundle.z1)
        return np.imag(combine(bundle.phi_at, self._terms.phi))

    def _delta_phi(self, bundle: StageScalars) -> ComplexArray:
        """Return the stage's exponent formed on the host from the native terms.

        The bundle's weights over the fixed terms, or the operator's own increment where
        it has none (a z table's, which the workspace holds in its slots).
        """
        if bundle.phi_delta is None or self._terms.phi is None:
            return delta_phi(
                self._problem.linear, bundle.z0, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.phi_delta, self._terms.phi)

    def _stage_scalars(self, z0: float, z1: float) -> StageScalars:
        """Return one stage's scalars (`StageCache.scalars`)."""
        return self._stages.scalars(z0, z1)

    def _diagnostics(self) -> dict[str, Any]:
        """Return the workspace's facts and counters (``Workspace.diagnostics``)."""
        return self._workspace.diagnostics()

    def _monitor_binding(self, monitor: BoundaryMonitor) -> dict[str, object]:
        """Return a run's boundary monitor weights where the state lives (SEP 21).

        From the run's start on, the engine's finish adds a split-step collar's removal
        at every accepted step, on every march; a split monitor reads the problem's
        walk-off buckets beside its five numbers (solprop #440).
        """
        return {
            "weights": np.ascontiguousarray(monitor.weights, dtype=np.float64).reshape(
                -1
            ),
            "column_weights": np.empty(0, dtype=np.float64)
            if monitor.columns is None
            else np.ascontiguousarray(monitor.columns, dtype=np.float64),
            "split": bool(monitor.split) and self._problem.split_crossing is not None,
        }

    def _edge_binding(self, edge: TransverseEdge) -> dict[str, object]:
        """Return a transverse edge's tables where the state lives (solprop SEP 44).

        Weights and no faces: what it reads comes back to the edge's judging. A
        requested collar's lines go as arrays.
        """
        tables = edge.tables
        synthesis = edge.reduction.synthesis()
        return {
            "faces": np.ascontiguousarray(tables.faces, dtype=np.float64).reshape(-1),
            "clip": np.ascontiguousarray(tables.clip, dtype=np.float64).reshape(-1),
            "totals": np.ascontiguousarray(tables.totals, dtype=np.float64).reshape(-1),
            "synthesis": np.empty(0, dtype=np.float64)
            if synthesis is None
            else np.ascontiguousarray(synthesis, dtype=np.float64).reshape(-1),
            "first_row": int(np.argmax(tables.rows)),
            "lines": _edge_lines(edge),
        }

    def edge_memory(self, space: object) -> int:
        """Return what reading the edge holds in the state's memory space (SEP 44).

        Parameters
        ----------
        space : object
            The run's transverse grid.

        Returns
        -------
        int
            The bytes of a `transverse_edge` ledger row, from the grid alone.
        """
        _, n_pol, n_omega = self._problem.shape
        return _transverse_edge_bytes(
            cast("Any", space), transverse_of(self._problem.route), n_pol, n_omega
        )

    def _launch_clean_columns(
        self, edge: TransverseEdge, loaded: Loaded
    ) -> RealArray | None:
        """Return each column's launched energy without the input draw, or ``None``.

        What the transverse edge's safety ramp is judged on (SEP 40), so a run with no
        ramp needs none. Formed where the load held both parts (`_load`); a launch that
        resumed without its edge's record has none, and is formed from the loaded state.

        - Returns: the unweighted energies, or ``None`` with no draw, no ramp or on a
          resumed launch.
        """
        floor = loaded.floor
        if floor is None or edge.resumed or edge.design is None:
            return None
        if loaded.clean_columns is not None:
            return loaded.clean_columns
        state = np.asarray(loaded.field).reshape(self._problem.shape)
        return edge.reduction.column_power(state - np.asarray(floor))

    def _take_workspace(self) -> _core.Workspace:
        """Return the workspace for a run or a probe, closing an open probe first."""
        if self._closed:
            raise BackendError("this backend is closed; its buffers were released")
        if self._probe is not None:
            self._probe.close()
            self._probe = None
            if self._workspace.closed:
                raise BackendError(
                    "this backend's workspace closed with its low-storage timing "
                    "probe, whose trials advanced the registers in place (solprop SEP "
                    "52 section 5.5); build another backend"
                )
        return self._workspace

    def timing_probe(self, start: Start) -> TimingProbe:
        """Load a start and return single trials of this run's march from it.

        The start's field, noise, position and resume flag are loaded as a session
        loads them (`_load`); its method history, ledgers and monitor continuation are
        not read. A Runge-Kutta trial leaves the loaded state alone; the low-storage
        march's step advances the discarded registers in place, so a backend probed so
        is closed afterwards (solprop SEP 52 section 5.5). The probe holds the
        workspace until the backend's next run or probe.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the trials start; its field is taken.

        Returns
        -------
        solprop.backends.protocol.TimingProbe
            The probe (:class:`solprop_native._probe.NativeProbe`).
        """
        loaded = self._load(
            np.asarray(start.take_field()),
            start.noise,
            z=float(start.z),
            resume=start.resume,
        )
        records = run_records(self._problem, self._request, start)
        workspace = self._take_workspace()
        with _envelope_errors():
            probe = workspace.probe(
                _probe_start(records, loaded), host_source=self._host_source
            )
        self._probe = probe
        return NativeProbe(self, probe, loaded)

    def metadata(self) -> dict[str, object]:
        """Return the shared execution metadata plus the native build record.

        The build record and the installed solprop version cannot change within a
        process (thread settings are read once, at import), so they are read once
        (:func:`_build_facts`); the native state's counters are read at every call.
        """
        facts = _build_facts()
        # The workspace's fixed facts, read once it was built, and its live counters.
        native = self._facts | self._workspace.counters()
        work: dict[str, object] = {
            "backend": self.name,
            "propagator_mode": self._propagator_mode,
            # SEP 47 section 4.3: execution metadata, outside the resume key.
            **self._chunk_plan.record(),
            "node_chunk": self.node_chunk,
        }
        if self._problem.responses.plasma_channels:
            work["plasma_source"] = plasma_source_record(self._problem.grids)
        # The evaluations this backend made: the Adams ledger's logical calls, the
        # low-storage march's physical ones (SEP 24 section 5), or the native
        # Runge-Kutta trials' dispatches (SEP 30); a session states its own run's.
        if self._adams is not None:
            work["response_calls"] = self._adams.ledger.to_record()
        elif self._low_storage:
            work["source_evaluations"] = self._low_storage_evaluations
        else:
            work["source_evaluations"] = native["rk_source_evaluations"]
        # Which march steps the run (solprop SEP 39 and 45 section 5): the native one,
        # the only one since SEP 47 phase 5.
        if self._adams is not None:
            work["adams_march"] = "native"
        elif not self._low_storage:
            work["rk_march"] = "native"
        if self._statistics_tables:
            # The profile statistics' tables and their measured errors (SEP 47 6.4).
            work["statistics_tables"] = dict(self._statistics_tables)
        if self._general_chunk is not None:
            work["general_node_chunk"] = self._general_chunk
        work["node_chunk_rule"] = (
            "device" if _on_device() and self.chunk_plan.automatic else "host"
        )
        if self._tdse_binding is not None and self._host_source is not None:
            # As the reference records them (solprop SEP 49 section 5.1).
            work["atomic_calls"] = dataclasses.asdict(self._tdse_binding.ledger)
            work["tdse_execution"] = dict(self._tdse_binding.metadata())
            work["tdse_last_evaluation"] = dict(self._host_source.last)
            work["tdse_source"] = tdse_source.source_record(self._problem)
        return {
            **work,
            **facts.info,
            "solprop_version": facts.solprop_version,
            "thread_budget": THREAD_BUDGET.threads,
            "thread_budget_source": THREAD_BUDGET.source,
            "openmp_wait_policy": WAIT_POLICY,
            "openmp_proc_bind": PROC_BIND,
            "calibration": _package.PROFILE.record(
                _package.PROCESS,
                self._applied_execution,
                _package.BUILTIN_EXECUTION,
                self._execution_source,
            )
            | _planning_record(native),
            "host_blas": self._host_blas,
            "planning_seconds": native["planning_seconds"],
            "wisdom_loaded": self._wisdom_loaded,
            "wisdom_digest": self._wisdom_digest,
            "raman_fft_provider": native["raman_fft_provider"],
            "raman_fft_tasks": native["raman_fft_tasks"],
            "execution_sizing": native["execution_sizing"],
            "raman_planning_seconds": native["raman_planning_seconds"],
            "raman_enabled": self._problem.flags.has_raman,
            "raman_terms": sum(
                term.n_terms for term in self._problem.responses.raman_terms
            ),
            "plasma_channels": len(self._problem.responses.population_channels),
            "envelope_plasma_enabled": self._problem.responses.envelope_plasma
            is not None,
            "envelope_refinement_boundary": "scalar status read before transforms",
            "raman_components": len(self._problem.responses.raman_terms),
            "raman_drive": (
                "none"
                if not self._problem.responses.raman_terms
                else "field_squared"
                if self._problem.responses.raman_terms[0].thg
                and self._problem.grid_kind == "field"
                else "cycle_averaged"
            ),
            "raman_kernel_transforms": native["raman_kernel_transforms"],
            "fsal_ready": native["fsal_ready"],
            "dispersive_path": dict(native["dispersive_path"]) or None,
            "synthesis_provider": native["synthesis_provider"],
            "projection_provider": native["projection_provider"],
            "gemm_shape_policy": native["gemm_shape_policy"],
            "device_team_size": native["response_team_size"],
            "device_team_size_limit": facts.device_team_size,
            "device_team_size_policy": (
                "256 below n_to=8192; configured limit otherwise"
                if facts.info["device_backend"] == "cuda"
                else "fixed configured team width"
            ),
            "device_reduction_size": facts.device_reduction_size,
        }

    @property
    def node_chunk(self) -> int:  # numpydoc ignore=RT01
        """The nodes a native response pass holds at once: the host plan, lowered."""
        return self._node_chunk

    def available_device_memory(self) -> int | None:
        """Return free accelerator bytes, or None for a host build.

        Returns
        -------
        int or None
            Free bytes from the selected accelerator runtime.
        """
        return _core.available_device_memory()

    def memory_ledger(self) -> dict[str, int]:
        """Return every persistent native allocation's byte count by role.

        A Runge-Kutta run's dense-output storage is listed beside them before a
        session allocates it.
        """
        ledger = {
            str(name): int(value)
            for name, value in self._diagnostics()["memory_ledger"].items()
        }
        stepper = self._stepper
        if (
            stepper.interpolate_planes
            and not isinstance(stepper.integrator, Multistep)
            and not ledger.get("rk_dense")
        ):
            # A Runge-Kutta run's dense output, before a native session allocates it
            # (SEP 47): the extension's five coefficients and the committed state held
            # across an observation, six fields, and a byte a bin marking the stiff
            # ones, so that an estimate made before the run counts them.
            field = int(ledger.get("state", 0))
            ledger["rk_dense"] = 6 * field + field // 16
        if self._tdse_binding is not None:
            # The host source's field and source rows (SEP 49), before a run binds
            # them, so that an estimate made before the run counts them.
            ledger.setdefault("host_source", 2 * 16 * int(self._problem.shape[-1]))
            ledger |= {
                f"tdse_{name}": int(size)
                for name, size in self._tdse_binding.memory_ledger().items()
            }
        return ledger

    def prepare(
        self, problem: PropagationProblem, request: RunRequest
    ) -> PreparedExecution:
        """Prepare a run of this backend for one problem and request (SEP 47 5.2).

        The backend lowered the request where it was built; each start of the result
        opens the native session (:class:`solprop_native.session.NativePrepared`),
        which refuses by name what it does not run.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The translated problem: this backend's, or one derived from it with
            another input.
        request : solprop.translate.run_request.RunRequest
            The run's request.

        Returns
        -------
        solprop.backends.session.PreparedExecution
            The prepared execution.
        """
        from .session import NativePrepared

        return NativePrepared(self, problem, request)

    def close(self) -> None:
        """Release native plans and views; repeated calls have no effect."""
        if not self._closed:
            # Closing the workspace closes a session or probe that leases it first and
            # drops a host source with the engine that borrowed it (SEP 49), whose
            # reference the garbage collector cannot see. A session that could not
            # close still uses the workspace, and keeps it.
            self._probe = None
            self._workspace.close()
            _host_blas.release()
        self._closed = True


def _require_initial_array(field: object, shape: tuple[int, ...]) -> None:
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a NumPy ndarray")
    array = cast(npt.NDArray[np.generic], field)
    if array.dtype != np.dtype(np.complex128):
        raise TypeError(f"field must have dtype complex128, got {array.dtype}")
    if array.ndim != 3:
        raise ValueError(f"field must have rank 3, got rank {array.ndim}")
    if array.shape != shape:
        raise BackendError(
            f"the initial state has shape {array.shape}, expected {shape}"
        )
    # Any strides: a problem's initial state may be a broadcast view (kernel schema
    # 42), and the driver's load copies the field into a contiguous state first.


@dataclasses.dataclass(frozen=True)
class _Restored:
    """What a resumed start restores beside its load (`NativeBackend._restore`).

    `reason` says why the snapshot's history was not installed (``None`` when it was or
    none was given); `proposal` is the next step's, m; `field`, when the history
    continues, is its own state, flat, which the run starts from in the loaded one's
    place, and `rows` its persistent rows as the workspace takes them (``HistoryRow``).
    """

    reason: str | None
    proposal: float
    field: ComplexArray | None = None
    rows: tuple[dict[str, object], ...] = ()


def _probe_start(records: RunRecords, loaded: Loaded) -> dict[str, object]:
    """Return a timing probe's start (solprop SEP 52 section 5.5).

    The run's records (the low-storage replay's landings and schedule), the loaded
    field and noise and its position; no march state, history or monitor.
    """
    return {
        "records": records,
        "field": np.ascontiguousarray(loaded.field).reshape(-1),
        "noise": np.ascontiguousarray(loaded.native_noise()),
        "adams": None,
        "runge_kutta": None,
        "rows": [],
        "plane_statistics": None,
        "row_statistics": None,
        "floor": None,
        "edge_noise": None,
        "monitor": None,
        "edge": None,
    }


def stage_tables_available(problem: PropagationProblem) -> bool:
    """Return whether the native state reads a problem's stage rows from tables.

    ``NativeBackend._stage_tables`` uploads them exactly when this holds: the ``Phi``
    and the prefactor are each a z table (on a CPU build), a separable operator's, a
    constant or an expansion (solprop SEP 39 phases 1 and 4). Decided from the problem
    alone, so that the native session can refuse a host-formed trial before anything
    is allocated (solprop SEP 47 section 5.2).
    """
    linear, responses = problem.linear, problem.responses
    tables = not problem.device_ready
    separable = linear.separable is not None
    phi = (
        (tables and linear.phi_representation == "table")
        or separable
        or linear.phi_constant is not None
        or linear.phi_expansion is not None
    )
    prefactor = (
        (tables and responses.prefactor_representation == "table")
        or separable
        or responses.prefactor_constant is not None
        or responses.prefactor_expansion is not None
    )
    return phi and prefactor


def host_rows_refusal(problem: PropagationProblem) -> str | None:
    """Return why the native stage tables cannot read a problem's rows, or ``None``.

    Every representation the translation forms -- a constant, an expansion, a
    separable operator's, and on a CPU build a z table -- is one the native state reads
    from its stage tables (:func:`stage_tables_available`); a device build refuses a z
    table before this is asked. What is left is a problem built by hand with a
    representation and no leaves for it, which no built-in configuration reaches: the
    backend refuses it by name at admission (solprop SEP 47 section 6.3) rather than
    stepping it with rows the host forms.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.

    Returns
    -------
    str or None
        The refusal, naming the representation, or ``None``.
    """
    if stage_tables_available(problem):
        return None
    linear, responses = problem.linear, problem.responses
    parts: list[str] = []
    if not (
        (not problem.device_ready and linear.phi_representation == "table")
        or linear.separable is not None
        or linear.phi_constant is not None
        or linear.phi_expansion is not None
    ):
        parts.append(f"the operator's {linear.phi_representation!r} representation")
    if not parts:
        parts.append(
            f"the prefactor's {responses.prefactor_representation!r} representation"
        )
    return (
        f"solprop-native does not read {parts[0]} in its stage tables: it holds no "
        "leaves the native state evaluates (solprop SEP 47 section 6.3); use "
        "backend='numpy_ref'"
    )


def _refuse_tdse(problem: PropagationProblem, binding: TDSEBinding | None) -> None:
    """Refuse a TDSE problem without its binding, or a binding without one."""
    if (tdse_descriptor(problem.route) is None) != (binding is None):
        raise BackendError("TDSE problems require an explicit host response binding")


def _refuse_unknown_method(stepper: StepperSettings) -> None:
    """Refuse a method object the native marches do not step, by name."""
    integrator = stepper.integrator
    # A request built by hand may hold a method its annotation does not name.
    unnamed = not isinstance(integrator, Multistep | LowStorageRungeKutta)  # pyright: ignore[reportUnnecessaryIsInstance]
    if integrator is not None and unnamed:
        raise BackendError(
            f"solprop-native does not step the method {type(integrator).__name__}; "
            "use Multistep(), LowStorageRungeKutta() or a Runge-Kutta tableau"
        )




def _refuse_unsupported(problem: PropagationProblem, stepper: StepperSettings) -> None:
    _refuse_unknown_method(stepper)
    # The route by its variant (solprop SEP 47 section 6.2), refused by name before
    # anything is allocated: every route of the union but the modal route's TDSE
    # extension, which is the reference's alone.
    # Read as any object: a variant this build does not know is refused by name too.
    route = cast("object", problem.route)
    match route:
        case ModalRoute(tdse=tdse) if tdse is not None and _on_device():
            space = str(_build_facts().info["device_backend"]).upper()
            raise BackendError(
                "the TDSE response runs on the native CPU build; this build is "
                f"{space} (solprop SEP 49)"
            )
        case (
            ModalRoute()
            | GeneralRoute()
            | SpatialRoute(kind="radial" | "cartesian" | "planar" | "line")
        ):
            pass
        case _:
            raise BackendError(
                f"solprop-native does not execute the {type(route).__name__} route"
            )
    # solprop SEP 35 phase 3: the scalar carrier shift on CPU builds; device builds
    # wait for their qualification.
    device = _on_device()
    refuse_adams(
        problem,
        stepper,
        "native (device build)" if device else "native",
        delivered=frozenset() if device else frozenset({"scalar"}),
    )


@dataclasses.dataclass(frozen=True)
class _HostFlow(CoupledFlowFactors):
    """A coupled flow's Schur factors per section on its active bins, host-side.

    solprop's shared factors (``solprop.translate.coupled.CoupledFlowFactors``), which
    the reference executes too: the frozen block of each section read out of the leaf,
    decomposed per active bin, the declared conditioning row, and a frozen-residual
    flow's terms; with each
    section's eigenbasis of its triangular factor (solprop's
    ``schur_eigenbasis``), whose route marks the bins formed from it, and the largest
    row-sum norm of the factor, which the squaring refusal reads. The eigenbasis
    crosses to the native engine once, with the Schur factors of the bins it does not
    cover; the engine forms the actions per section and step length. ``scored``
    marks, per active bin, the requested band (``grids.inband_mask``): the
    explicit residual is applied there only, as solprop's reference applies it
    (``residual_bins``).
    """

    scored: npt.NDArray[np.bool_]
    left: ComplexArray
    right: ComplexArray
    eigenvalues: ComplexArray
    route: npt.NDArray[np.bool_]
    row_sum: RealArray


_HOST_FLOWS: weakref.WeakKeyDictionary[PropagationProblem, _HostFlow | None] = (
    weakref.WeakKeyDictionary()
)
"""Each problem's decomposed coupled flow while it lives: admission reads its route,
and a backend built for the problem its factors, which the run only reads."""


def _host_flow(problem: PropagationProblem) -> _HostFlow | None:
    """Decompose an admitted coupled flow once per problem, or return ``None``.

    The Schur decompositions (one per active bin and section) are the expensive part
    of admitting a coupled flow, so the decomposition is kept with the problem.

    - Returns: the decomposition, or ``None`` without a coupled flow.
    """
    if problem not in _HOST_FLOWS:
        _HOST_FLOWS[problem] = _decomposed_flow(problem)
    return _HOST_FLOWS[problem]


def _decomposed_flow(problem: PropagationProblem) -> _HostFlow | None:
    """Decompose an admitted coupled flow, or return ``None`` without one."""
    flow: CoupledFlowTables | None = problem.linear.coupled_flow
    if flow is None:
        return None
    factors = coupled_flow_factors(flow)
    active = factors.active
    bases = [
        schur_eigenbasis(vectors, triangular)
        for vectors, triangular in zip(factors.vectors, factors.triangular, strict=True)
    ]
    return _HostFlow(
        **vars(factors),
        scored=np.ascontiguousarray(problem.grids.inband_mask[active], dtype=np.bool_),
        left=np.ascontiguousarray(np.stack([basis[0] for basis in bases])),
        right=np.ascontiguousarray(np.stack([basis[1] for basis in bases])),
        eigenvalues=np.ascontiguousarray(np.stack([basis[2] for basis in bases])),
        route=np.stack([basis[3] for basis in bases]),
        row_sum=np.asarray(
            [
                float(np.max(np.sum(abs(triangular), axis=-1), initial=0.0))
                for triangular in factors.triangular
            ],
            dtype=np.float64,
        ),
    )


_TAYLOR_SCALE = 0.5
"""Row-sum norm solprop's reference scales an exponent below before its series."""

_MAX_FLOW_SQUARINGS = 30
"""Largest number of squarings a flow action may need before it is refused."""


def _flow_allowance(modes: int) -> float:
    """Return the rounding allowance on two spectral norms of `modes` channels."""
    return 1.0 + 64.0 * modes * float(np.finfo(float).eps)


def _flow_guard(flow: _HostFlow | None) -> dict[str, object] | None:
    """Return a coupled flow's certificate and the guard's constants, or ``None``.

    Each section's row-sum norm and conditioning bound per active bin, with the scale
    solprop's reference reduces an exponent below, the most squarings and the
    rounding allowance on two spectral norms (solprop SEP 45 phase 4).
    """
    if flow is None:
        return None
    row_sum = np.ascontiguousarray(flow.row_sum, dtype=np.float64)
    return {
        "row_sum": row_sum,
        "condition_bound": np.ascontiguousarray(
            flow.condition_bound, dtype=np.float64
        ).reshape(row_sum.size, -1),
        "taylor_scale": _TAYLOR_SCALE,
        "max_squarings": _MAX_FLOW_SQUARINGS,
        "allowance": _flow_allowance(int(flow.vectors.shape[-1])),
    }


def _refined_record(
    problem: PropagationProblem,
    stepper: StepperSettings,
    diagnostics: Mapping[str, Any],
) -> tuple[ExecutionRecord, str] | None:
    """Return a stored stage-2 refinement of this execution, or ``None``.

    Its record applies only while the machine record it refined is still the
    process's: a refinement adds to the machine profile and never outlives it.
    """
    if _package.PROFILE.fingerprint is None:
        return None
    from ._profile import descriptor
    from ._store import refinement, stores

    key = descriptor(
        # The store's key keeps the four-axis shape it has always been written with,
        # so that refinements recorded before the batch axis left the problem (solprop
        # SEP 47 section 6.2) are found again.
        shape=(1, *problem.shape),
        grid_kind=problem.grid_kind,
        integrator=type(stepper.integrator).__name__,
        sizing=diagnostics["execution_sizing"],
        process=_package.PROCESS,
        fingerprint=_package.PROFILE.fingerprint.digest,
    )
    found = refinement(_package.PROFILE.fingerprint.digest, key, stores())
    if found is None or found.get("base") != _package.EXECUTION.native():
        return None
    changes = found.get("workspace")
    if not isinstance(changes, dict):
        return None
    try:
        return _package.EXECUTION.updated(
            cast(dict[str, object], changes)
        ), f"refinement {key}"
    except ValueError:
        return None


def _planning(
    given: Mapping[str, object] | None, stored: Mapping[str, str]
) -> dict[str, object] | None:
    """Return the problem's FFT planning: the one given, or the run's own.

    A run's own is the profile's, or the plan policy's, offered the plans runs measured
    before (`stored`) and keeping those it measures (solprop ADR-076); one given, a
    calibration's, plans on its own. None on a device, which plans with its vendor's
    FFT.
    """
    if _on_device():
        return None
    if given is not None:
        return dict(given)
    chosen = _package.PLANNING or {"mode": "legacy"}
    return dict(chosen) | {"stored": dict(stored), "keep": True}


def _keep_plans(diagnostics: Mapping[str, Any], path: Path | None) -> dict[str, str]:
    """Keep the plans a workspace measured, in this process and in `path`; return them.

    A key the profile holds is never kept: its winner takes precedence.
    """
    if _on_device():
        return {}
    measured = cast(dict[str, str], diagnostics["fft_planning"]["measured"])
    profiled = cast(Mapping[str, str], (_package.PLANNING or {}).get("wisdom", {}))
    kept = {key: plan for key, plan in measured.items() if key not in profiled}
    if kept:
        _wisdom.remember(kept)
        _wisdom.store(path, kept)
    return kept


def _applied_plans(
    diagnostics: Mapping[str, Any],
    stored: Mapping[str, str],
    planning: Mapping[str, object] | None,
) -> dict[str, str]:
    """Return the wisdom of each plan a workspace built from wisdom or measured.

    `planning` is the one it was given, or the profile's; `stored` the plans offered.
    """
    found = diagnostics["fft_planning"]
    profiled = cast(Mapping[str, str], (planning or {}).get("wisdom", {}))
    missing = set(cast(list[str], found["missing"]))
    applied = {
        key: profiled[key]
        for key in cast(dict[str, str], found["groups"])
        if key in profiled and key not in missing
    }
    applied |= {key: stored[key] for key in cast(list[str], found["reused"])}
    return applied | cast(dict[str, str], found["measured"])


def _planning_record(diagnostics: Mapping[str, Any]) -> dict[str, object]:
    """Return what a workspace's FFT planning applied, for its run's metadata.

    The plans keyed in its profile that no longer build from their wisdom are listed:
    they ran at estimate (solprop SEP 46 section 6.7).
    """
    if _on_device():
        return {}
    found = diagnostics["fft_planning"]
    groups = cast(dict[str, str], found["groups"])
    if not groups:
        return {"fft_planning": "policy"}
    return {
        "fft_planning": {
            "plans": len(groups),
            "missing": list(cast(list[str], found["missing"])),
        }
    }

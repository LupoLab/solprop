# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solprop-dependent native backend adapter."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import types
import weakref
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from solprop import transverse_monitor
from solprop._helpers import Helpers
from solprop._process import acquire_host_blas, thread_budget
from solprop.backends.exponential import (
    CarrierReference,
    carrier_reference,
    refuse_carrier_frame,
    refuse_formulation,
)
from solprop.backends.low_storage import schedule_key, table_steps
from solprop.backends.memory import (
    LOCAL_FLUX_STATISTICS,
    low_storage_flows,
    low_storage_stage_register,
    z_table_slots,
)
from solprop.backends.multistep import (
    CallLedger,
    _Derivative,  # pyright: ignore[reportPrivateUsage] - the march's history entries
    source_absent,
)
from solprop.backends.node_chunk import NodeChunkPlan, plan_node_chunk
from solprop.backends.numpy_ref.admission import refuse_general_route
from solprop.backends.numpy_ref.general_modal import (
    DELAYED_TERMS,
    general_planned_chunk,
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.backends.protocol import Backend, TimingProbe
from solprop.backends.session import (
    ExecutionOptions,
    PreparedExecution,
    PreparedGroup,
    Start,
)
from solprop.backends.spatial import (
    FarFieldTables,
    far_field_partials,
    radial_absorber_matrix,
)
from solprop.boundaries import SPLIT_RATE_BUCKETS
from solprop.constants import EPS0, C
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
)
from solprop.grid import RadialGrid, TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.schema import MarchSnapshot
from solprop.profiles import ZProfile
from solprop.spatial import (
    ABSORBED_CHANNELS,
    FAR_FIELD_STATISTICS,
    ApertureTables,
)
from solprop.stats import (
    StatsSpec,
)
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import (
    MarchPolicy,
    Multistep,
    integrated_weights,
    lagrange_coefficients,
    march_policy,
)
from solprop.translate.chebyshev import (
    _EDGE_SLACK_FRACTION,  # pyright: ignore[reportPrivateUsage] - the tables' own admitted span
    PiecewiseChebyshev,
)
from solprop.translate.coupled import (
    CoupledFlowTables,
    flow_edge_slack,
    frozen_generator,
    schur_blocks,
    schur_eigenbasis,
)
from solprop.translate.linop import (
    delta_phi,
    power_weight_at,
    rate_weights,
    unitary_phase,
)
from solprop.translate.plasma_source import (
    _BENCHMARK_GATED,  # pyright: ignore[reportPrivateUsage] - refuse the private NumPy benchmark hook
    plasma_source_record,
)
from solprop.translate.problem import (
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
    validate_run,
)
from solprop.translate.responses import CompiledResponses, ResponseStage
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    SpatialRoute,
    flux_of,
    transverse_of,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.stage import (
    StageScalars,
    StageTerms,
    combine,
    require_device_ready,
    terms,
)
from solprop.transverse_monitor import (
    TransverseEdge,
    _EdgeReading,  # pyright: ignore[reportPrivateUsage] - released together
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
from ._execution import ExecutionRecord
from ._helpers import NATIVE
from ._load import Loaded, StageCache, collar_rests, load, propagator_mode
from ._probe import NativeProbe
from ._statistics import (
    StatisticsLowering,
    metric_refusal,
)

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class _envelope_errors:
    """Translate refinement failures without importing solprop in the extension.

    A class rather than a generator: it wraps every native step call.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        """Enter the native call."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Raise a refinement failure leaving the call as solprop's error."""
        del kind, traceback
        if isinstance(error, _core.EnvelopeConvergenceError):
            raise ConvergenceError(str(error)) from error
        if isinstance(error, _core.BackendRefusal):
            raise BackendError(str(error)) from error


ADAMS_MAX_CHANNELS = 16
"""The most channels of a coupled block whose channel vectors a CPU build's resident
Adams engine holds on the stack.

The native ``adams_max_channels``. Either build steps any width (solprop SEP 47 phase
4): a CPU build's per-bin passes hold a wider block's channel vectors in a row per bin,
and a device build's passes take a thread per (bin, channel) and hold none.
"""

_SPATIAL_STAT_FLAGS = {
    "energy": 1,
    "on_axis_intensity": 2,
    "peak_intensity": 4,
    "on_axis_spectrum": 8,
    "beam_radius": 16,
    "fwhm_r": 16,
}

_FAR_FIELD_FLAGS = {
    "beam_spectrum": 1 << 5,
    "absorbed_power": 1 << 7,
    "region_spectrum": 1 << 8,
    "aperture_spectrum": 1 << 9,
    "aperture_on_axis_spectrum": 1 << 10,
    "on_axis_spectrum": 1 << 11,
}
"""The native far-field pass's flags (solprop SEP 23 section 6), from 32.

``on_axis_spectrum`` is the pass's ``W = 1`` slot on a Cartesian or line grid; on the
other spatial kinds it stays ``spatial_statistics``' flag 8.
"""

_FAR_FIELD_ORDER = (
    "beam_spectrum",
    "absorbed_power",
    "region_spectrum",
    "aperture_spectrum",
    "aperture_on_axis_spectrum",
    "on_axis_spectrum",
)
"""The order in which the native pass packs its statistics."""

_STAT_FLAGS = {
    "omega0": 1 << 0,
    "energy": 1 << 1,
    "energy_window": 1 << 2,
    "peak_power": 1 << 3,
    "peak_power_window": 1 << 4,
    "peak_power_total": 1 << 5,
    "fwhm_t": 1 << 6,
    "fwhm_t_outer": 1 << 6,
    "electron_density": 1 << 7,
    "peak_ionisation_rate": 1 << 8,
    "peak_intensity": 1 << 9,
    "quadrature_error": 1 << 10,
}

_STATISTICS_PLANS = 16
"""The most native statistics plans a backend keeps, one per spec and names."""


class NativeBackend:
    """Propagate modal, radial or Cartesian fields with persistent Kokkos state.

    Parameters
    ----------
    problem
        Validated kernel problem. Accepts one batch member,
        modal or spatial scalar or two-component propagation (radial, Cartesian, or
        the hybrid planar kind of slab modes in x times an FFT in y), instantaneous
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
    max_batch = 1
    helpers = NATIVE
    far_field_statistics: ClassVar[frozenset[str]] = frozenset(FAR_FIELD_STATISTICS)
    """The far-field statistics of solprop SEP 23 this build forms natively.

    solprop's registry admits them on this backend by this attribute and refuses
    them, by name, on a build without it.
    """
    observes_fresh_field: ClassVar[bool] = True
    """Each observation is a new host array (``field_to_host``) that nothing here keeps.

    So solprop converts an observed field to output units where it is (solprop SEP 47
    section 5.4).
    """
    reads_transverse_edge = True
    """The transverse edge is read from the native state where it lives (solprop SEP 40
    section 5.3, SEP 44), by the native session at each read."""

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
    ) -> None:
        if _BENCHMARK_GATED.get():
            raise BackendError("the gated plasma benchmark requires numpy_ref")
        check_build()
        stepper = request.stepper
        _refuse_unsupported(problem, stepper)
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
        # solprop SEP 45 phase 2: the staged Runge-Kutta rows, decided on first use.
        self._rk_staged = False
        self._rk_nodes: list[float] = []
        self._rk_weighted = False
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
        self._bound_buckets: np.ndarray | None = None
        # solprop SEP 46: the workspace's CPU execution record, the process's (the
        # calibration's where it holds, else the built-in table) unless one is given,
        # as a stage-2 refinement gives one; read back from the state it built.
        self._execution_source = (
            _package.EXECUTION_SOURCE if cpu_execution is None else "given"
        )
        self._native_state = _core.State(
            _native_problem(
                problem,
                request,
                self._terms,
                self._flow,
                self._z_tables,
                node_chunk=self._node_chunk,
                general_chunk=self._general_chunk,
            )
            | _low_storage_payload(problem, stepper, device=_on_device())
            | {"cpu_execution": (cpu_execution or _package.EXECUTION).native()}
            | _planning_payload(fft_planning, stored)
        )
        self._applied_execution = ExecutionRecord.from_native(
            self._native_state.cpu_execution
        )
        if cpu_execution is None:
            # A stage-2 refinement of this very execution (solprop SEP 46 section
            # 6.6): the workspace is built again with its record.
            refined = _refined_record(problem, stepper, self._native_state)
            if refined is not None:
                stored |= _keep_plans(self._native_state, wisdom)
                self._native_state.close()
                self._native_state = _core.State(
                    _native_problem(
                        problem,
                        request,
                        self._terms,
                        self._flow,
                        self._z_tables,
                        node_chunk=self._node_chunk,
                        general_chunk=self._general_chunk,
                    )
                    | _low_storage_payload(problem, stepper, device=_on_device())
                    | {"cpu_execution": refined[0].native()}
                    | _planning_payload(fft_planning, stored)
                )
                self._execution_source = refined[1]
                self._applied_execution = ExecutionRecord.from_native(
                    self._native_state.cpu_execution
                )
        self._native_state.set_crossing_workers(thread_budget().threads)
        if fft_planning is None:
            _keep_plans(self._native_state, wisdom)
        self._wisdom_digest = (
            _wisdom.digest(
                _applied_plans(
                    self._native_state,
                    stored,
                    fft_planning if fft_planning is not None else _package.PLANNING,
                )
            )
            if not _on_device()
            else None
        )
        if self._z_tables is not None:
            self._z_tables.upload(self._native_state)
        _upload_flux_table(self._native_state, problem)
        # The native Adams march's record (SEP 39): its policy, ledger and history,
        # over the resident history the native state keeps (`_adams_engine`).
        integrator = stepper.integrator
        self._resident: _ResidentAdams | None = None
        self._adams: _NativeAdamsMarch | None = (
            _NativeAdamsMarch(self, integrator)
            if isinstance(integrator, Multistep)
            else None
        )
        # The low-storage march (SEP 24) is the native session's; its resident
        # registers' step is a timing probe's trial (`timing_probe`), and the
        # evaluations the probe makes are counted for the metadata.
        self._resident_low_storage = (
            _ResidentLowStorage(self, integrator)
            if isinstance(integrator, LowStorageRungeKutta)
            else None
        )
        self._low_storage_evaluations = 0
        # NumPy's own BLAS pool would spin beside the native one while this backend
        # lives; it is held to one thread until the last native backend closes.
        self._host_blas = _host_blas.hold(
            build_info().blas_library,
            THREAD_BUDGET.threads,
            acquire=acquire_host_blas,
        )

    @property
    def request(self) -> RunRequest:
        """Return the run request this backend was built for (solprop SEP 47 5.2).

        Returns
        -------
        solprop.translate.run_request.RunRequest
            Its method, error control, itinerary, planes and statistics; a run adapts
            it with :meth:`~solprop.translate.run_request.RunRequest.for_run`.
        """
        return self._request

    @property
    def chunk_plan(self) -> NodeChunkPlan:
        """Return the host rule's node-chunk plan, made at construction (SEP 47 6.5).

        Returns
        -------
        solprop.backends.node_chunk.NodeChunkPlan
            The plan, which :attr:`node_chunk` lowers for this build.
        """
        return self._chunk_plan

    @property
    def closed(self) -> bool:
        """Return whether the backend has released its buffers.

        Returns
        -------
        bool
            ``True`` once :meth:`close` ran: it can run no other input.
        """
        return self._closed

    @property
    def _layout(self) -> tuple[int, int, int, int]:
        """Return the build's state layout: the shape behind a unit axis.

        The C++ kernels keep their four-axis layouts; a single trajectory crosses the
        Python seam without the axis (solprop SEP 47 section 6.2).
        """
        n_modes, n_pol, n_omega = self._problem.shape
        return (1, n_modes, n_pol, n_omega)

    def _load(
        self,
        field: ComplexArray,
        noise: NoiseRealisation | None,
        *,
        z: float,
        resume: bool,
    ) -> Loaded:
        """Load a start into the native state, as a fresh backend's (SEP 22 4.7).

        The field and the noise realisation (:func:`solprop_native._load.load`), the
        stages' collar back in and the recycled stage dropped, no monitor bound, and
        the Adams record's history and accounting started afresh; a session or a
        timing probe goes on from here.

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
        self._native_state.unbind_monitor()
        self._native_state.set_initial(loaded.field, loaded.native_noise())
        if (counters := _counters.active) is not None:
            counters.load("start", loaded.field.nbytes)
        # A collar an earlier run rested goes back into the stages (SEP 21).
        self._native_state.set_absorber_quiet(False)
        self._native_state.clear_fsal()
        if self._adams is not None:
            self._adams.reset()
            self._adams.begin_run()
        if self._resident_low_storage is not None:
            self._resident_low_storage.reset()
        self._low_storage_evaluations = 0
        # The state holds the field now: the record keeps no host copy past the load.
        kept = dataclasses.replace(loaded, field=_EMPTY)
        self._loaded = kept
        return kept

    def _install(self, field: ComplexArray) -> None:
        """Load a resumed history's own state, with the loaded noise realisation.

        - ``field``: the state as the interrupted run held it, contiguous.
        """
        loaded = self._loaded
        self._native_state.set_initial(
            np.ascontiguousarray(field.reshape(-1)),
            _EMPTY if loaded is None else loaded.native_noise(),
        )
        if (counters := _counters.active) is not None:
            counters.load("restore", field.nbytes)

    def _restore(self, start: Start, loaded: Loaded) -> tuple[str | None, float]:
        """Restore a resumed start's ledgers and method history after its load.

        - Returns: why the snapshot's history was not installed (``None`` when it was or
          none was given), and the next step's proposal, m: the snapshot's when its
          history continues, else ``dz_init``.
        """
        adams = self._adams
        proposal = float(self._stepper.dz_init)
        if start.ledgers is not None:
            atomic = start.ledgers.get("atomic", {})
            if atomic:
                raise BackendError(
                    f"{self.name} does not restore atomic response call ledgers"
                )
            if adams is not None:
                adams.restore_ledger(
                    cast("Mapping[str, object]", start.ledgers.get("logical", {}))
                )
        snapshot = start.march
        if snapshot is None:
            return None, proposal
        if adams is None:
            march = (
                "low-storage"
                if self._resident_low_storage is not None
                else "Runge-Kutta"
            )
            return f"the run steps with the {march} march, which keeps no history", (
                proposal
            )
        reason = adams.resume(snapshot, loaded.z)
        if reason is not None:
            return reason, proposal
        self._install(np.array(snapshot.state, copy=True)[None])
        return None, float(snapshot.proposal)

    def _adams_engine(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"],
    ) -> object:
        """Return the resident Adams history (solprop SEP 16).

        Either formulation keeps its history, its step arithmetic and its norms in the
        native state (solprop ADR-043), beside a coupled flow too (SEP 16 phase 6),
        whose interaction formulation the admission refuses (`_refuse_adams`).
        """
        # A varying operator's exponential formulation carries the transported
        # compensation natively, as does a frozen-residual flow's residual, whatever
        # the diagonal's representation; the untransported test witness is the
        # reference's alone.
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
        self._resident = _ResidentAdams(
            native=self._native_state,
            method=method,
            controller=self._stepper.controller,
            n_omega=self._problem.grids.n_omega,
            arrays=self._resident_arrays,
            hoist=self._hoist,
            adopted=_adopt_nothing,
            rate=self._rate_weights,
            shape=self._layout,
            helpers=self.helpers,
            compensated=compensated,
            interpolate=self._stepper.interpolate_planes,
            carrier=carrier_reference(self._problem, method),
        )
        # solprop SEP 39 phases 1 and 3: the native state reads the scalar tables
        # itself, uploaded once; the admission refused a problem whose rows they do not
        # read (`host_rows_refusal`).
        tables = self._stage_tables()
        assert tables is not None
        self._native_state.adams_stage_tables(
            tables | _chord_tables(self._resident.carrier)
        )
        return self._resident

    def _stage_tables(self) -> dict[str, object] | None:
        """Return the tables a resident Adams trial reads natively, or ``None``.

        solprop SEP 39 phases 1 and 4: every scalar table ``translate.stage.scalars``
        reads at a stage -- the weights of a constant or expanded ``Phi`` and
        prefactor, the density, the taper scale, the Raman decay rates and densities,
        the Kerr density, the plasma channels' densities and fractions, SEP 6's
        channel densities -- and an expansion's rate table, each packed as its batch
        pass reads it (``PiecewiseChebyshev.packed``). A separable operator's rows are
        a constant's, the distance and the position with a unit prefactor
        (`_trial_arrays`), and read no span. A z table's rows are its slots, which the
        native state loads through the cache `_ZTables` shares. A coupled flow's
        section and residual weights and a moving guide's map weights are sectioned
        Chebyshev rows (`flow_weights`, `transform_weights`), formed natively from
        their edges.
        """
        problem = self._problem
        if not stage_tables_available(problem):
            return None
        linear, responses = problem.linear, problem.responses
        phi_weights = phi_rate = prefactor_weights = None
        separable = self._terms.separable is not None
        tables = self._z_tables
        if tables is not None and tables.phi:
            phi = _STAGE_TABLE
        elif separable or linear.phi_constant is not None:
            phi = _STAGE_CONSTANT
        elif linear.phi_expansion is not None:
            phi = _STAGE_EXPANSION
            phi_weights = linear.phi_expansion.weights.packed()
            phi_rate = linear.rate.packed()
        else:
            return None
        low = high = 0.0
        if tables is not None and tables.prefactor:
            prefactor = _STAGE_TABLE
        elif separable:
            prefactor = _STAGE_CONSTANT
            low, high = -math.inf, math.inf
        elif responses.prefactor_constant is not None:
            prefactor = _STAGE_CONSTANT
            low, high = responses.prefactor_span
        elif responses.prefactor_expansion is not None:
            prefactor = _STAGE_EXPANSION
            prefactor_weights = responses.prefactor_expansion.weights.packed()
        else:
            return None
        kerr = responses.kerr_coefficients
        taper = problem.transform.taper_scale
        return {
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
            "kerr": 0 if kerr is None else 1 if kerr.density is None else 2,
            "kerr_density": None
            if kerr is None or kerr.density is None
            else kerr.density.packed(),
            "plasma_densities": [
                channel.density.packed() for channel in responses.population_channels
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
        } | _section_tables(
            problem,
            moving=self._moving_maps,
            residual=self._flow is not None and self._flow.residual_terms is not None,
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

    def _resident_arrays(self, anchor: float, position: float) -> _TrialArrays:
        """Return a resident trial's arrays, every stage row at `position`."""
        bundle = self._stage_scalars(anchor, position)
        self._last_bundle = bundle
        stages = self._stepper.tableau.n_stages
        arrays, material = self._repeated_arrays(bundle)
        flow = self._flow
        if flow is not None:
            # Beside a coupled flow the trial's section and a frozen-residual flow's
            # weights at `position`; the resident passes need no Runge-Kutta action.
            material |= {
                "flow_section": bundle.flow_section,
                "flow_weights": np.empty(0, dtype=np.float64)
                if flow.residual_terms is None
                else np.ascontiguousarray(
                    np.tile(_required(bundle.flow_weights, "flow_weights"), stages),
                    dtype=np.float64,
                ),
            }
        return arrays, material

    def _repeated_arrays(self, bundle: StageScalars) -> _TrialArrays:
        """Return the trial arrays of a step whose every stage row is `bundle`.

        `_trial_arrays` of the bundle repeated, value for value. With the weights a
        constant's or an expansion's own and no further material rows (SEP 6's
        channels, a moving guide's maps), each array is one row repeated in one call
        rather than packed from a list of the same bundle, which costs a resident
        Adams trial more than its coefficient tables. The other problems take
        `_trial_arrays`.
        """
        stages = self._stepper.tableau.n_stages
        if (
            self._z_tables is not None
            or self._terms.separable is not None
            or self._problem.responses.dispersive_kerr is not None
            or self._moving_maps
        ):
            return self._trial_arrays([bundle] * stages, bundle)
        phi_delta = _required(bundle.phi_delta, "phi_delta")[np.newaxis]
        phi_at = _required(bundle.phi_at, "phi_at")[np.newaxis]
        prefactor = _required(bundle.prefactor, "prefactor")[np.newaxis]
        return (
            np.repeat(phi_delta, stages + 1, axis=0),
            np.repeat(phi_at, stages, axis=0),
            np.repeat(prefactor, stages, axis=0),
            np.full(stages, bundle.density, dtype=np.float64),
            np.full(stages, bundle.taper_scale, dtype=np.float64),
            np.full(stages, bundle.preionisation, dtype=np.float64),
            _repeated_row(bundle.raman_rates, stages),
        ), {
            "kerr_density": np.full(stages, bundle.kerr_density, dtype=np.float64),
            "plasma_densities": _repeated_row(bundle.plasma_densities, stages),
            "plasma_preionisation": _repeated_row(bundle.plasma_preionisation, stages),
            "raman_densities": _repeated_row(bundle.raman_densities, stages),
        }

    def _trial_step(self, z: float, dz: float) -> tuple[float, _core.State]:
        partials = self._prepare_trial(z, dz)
        return self._score_trial(partials), self._native_state

    def _staged_rows(self) -> None:
        """Upload the stage tables a Runge-Kutta trial's rows are formed from, once.

        solprop SEP 45 phase 2: every row a trial reads is one of the uploaded stage
        tables', and the state evaluates each stage's scalars at ``z + c_i dz``
        natively. Phase 4 adds the z table, whose positions the state loads into the
        one slot cache stage by stage. Beside a coupled flow the state also forms the
        step's flow actions behind its guards, transcribed with their constants. The
        admission refuses a problem whose rows the tables do not read
        (:func:`host_rows_refusal`), so the tables are always formed.
        """
        if self._rk_staged:
            return
        tables = self._stage_tables()
        assert tables is not None  # `host_rows_refusal` admitted the problem
        self._native_state.adams_stage_tables(tables | _chord_tables(None))
        self._rk_nodes = [float(node) for node in self._stepper.tableau.nodes]
        flow = self._flow
        if flow is not None:
            modes = int(flow.vectors.shape[-1])
            self._native_state.flow_guard_configure(
                [float(value) for value in flow.row_sum],
                [float(value) for value in np.ravel(flow.condition_bound)],
                _TAYLOR_SCALE,
                _MAX_FLOW_SQUARINGS,
                _flow_allowance(modes),
            )
        self._rk_staged = True

    def _prepare_trial(self, z: float, dz: float) -> RealArray:
        """Prepare a Runge-Kutta trial's stages on the staged rows and score it."""
        self._staged_rows()
        with _envelope_errors():
            return self._native_state.rk_trial(
                z, dz, self._rk_nodes, prepare_only=False
            )

    def _trial_arrays(
        self, bundles: Sequence[StageScalars], whole: StageScalars
    ) -> tuple[
        tuple[
            RealArray, RealArray, RealArray, RealArray, RealArray, RealArray, RealArray
        ],
        dict[str, npt.NDArray[np.float64] | int],
    ]:
        """Pack the stages' host scalars into the engine's trial arrays."""
        if self._terms.separable is not None:
            phi_delta = np.array([[one.z1 - one.z0] for one in [*bundles, whole]])
            phi_at = np.array([[one.z1] for one in bundles])
            prefactor = np.ones((len(bundles), 1))
        else:
            phi_delta, phi_at, prefactor = self._weight_rows(bundles, whole)
        density = np.ascontiguousarray(
            np.array([bundle.density for bundle in bundles], dtype=np.float64)
        )
        taper = np.ascontiguousarray(
            np.array([bundle.taper_scale for bundle in bundles], dtype=np.float64)
        )
        preionisation = np.ascontiguousarray(
            np.array([bundle.preionisation for bundle in bundles], dtype=np.float64)
        )
        raman_rates = np.ascontiguousarray(
            np.array([bundle.raman_rates for bundle in bundles], dtype=np.float64)
        )
        material: dict[str, npt.NDArray[np.float64] | int] = {
            "kerr_density": np.asarray([one.kerr_density for one in bundles]),
            "plasma_densities": np.asarray([one.plasma_densities for one in bundles]),
            "plasma_preionisation": np.asarray(
                [one.plasma_preionisation for one in bundles]
            ),
            "raman_densities": np.asarray([one.raman_densities for one in bundles]),
        }
        if self._problem.responses.dispersive_kerr is not None:
            # solprop SEP 6's channel densities, each stage's row; the native response
            # weighs its channels and rebuilds a mixture's filter from them.
            material["dispersive_densities"] = np.ascontiguousarray(
                [one.dispersive_kerr_densities for one in bundles], dtype=np.float64
            )
        if self._moving_maps:
            # Schema 14's row: each stage's weights of a moving guide's maps, which the
            # engine combines natively per stage.
            material["transform_weights"] = np.ascontiguousarray(
                np.stack(
                    [
                        _required(one.transform_weights, "transform_weights")
                        for one in bundles
                    ]
                ),
                dtype=np.float64,
            )
        arrays = (
            phi_delta,
            phi_at,
            prefactor,
            density,
            taper,
            preionisation,
            raman_rates,
        )
        return arrays, material

    def _weight_rows(
        self, bundles: Sequence[StageScalars], whole: StageScalars
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Return the stages' weights of ``Phi`` and of the prefactor over the terms.

        Each stage's increment, the whole step's and each stage's value of ``Phi``, and
        each stage's prefactor: the bundles' own weights of a constant or an expansion,
        or, for a z table, the +-1 weights of its slots (solprop SEP 29 G1), whose
        positions are evaluated natively first where no slot holds them.
        """
        tables = self._z_tables
        slot_of: dict[float, int] = {}
        if tables is not None:
            stages = [*bundles, whole]
            slot_of = tables.load(
                [one.z0 for one in stages if tables.phi] + [one.z1 for one in stages]
            )
        if tables is not None and tables.phi:
            phi_delta = np.stack(
                [tables.phi_row(slot_of, one.z0, one.z1) for one in [*bundles, whole]]
            )
            phi_at = np.stack(
                [tables.at_row(slot_of, one.z1, tables.n_phi) for one in bundles]
            )
        else:
            phi_delta = np.stack(
                [
                    *(_required(bundle.phi_delta, "phi_delta") for bundle in bundles),
                    _required(whole.phi_delta, "whole phi_delta"),
                ]
            )
            phi_at = np.stack(
                [_required(bundle.phi_at, "phi_at") for bundle in bundles]
            )
        if tables is not None and tables.prefactor:
            prefactor = np.stack(
                [tables.at_row(slot_of, one.z1, tables.slots) for one in bundles]
            )
        else:
            prefactor = np.stack(
                [_required(bundle.prefactor, "prefactor") for bundle in bundles]
            )
        return (
            np.ascontiguousarray(phi_delta),
            np.ascontiguousarray(phi_at),
            np.ascontiguousarray(prefactor),
        )

    def _phi_increment(self, z0: float, z1: float) -> RealArray:
        """Return the weights of ``Phi(z1) - Phi(z0)`` over the native terms.

        The bundle's own for a constant or an expansion; a z table's two slots,
        evaluated where no slot holds them, for a z table (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            return tables.phi_row(tables.load([z0, z1]), z0, z1)
        return _required(self._stage_scalars(z0, z1).phi_delta, "phi_delta")

    def _rate_weights(self, z: float, *, before: bool = False) -> RealArray:
        """Return ``D(z)``'s weights over the native ``Phi`` terms (solprop SEP 16).

        solprop's :func:`~solprop.translate.linop.rate_weights` for a constant or an
        expansion; for a z table, the rate table evaluated natively into its slot
        (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            return tables.rate(z, before=before)
        return rate_weights(
            self._problem.linear, z, before=before, helpers=self.helpers
        )

    def _delta_phi(self, bundle: StageScalars) -> ComplexArray:
        """Return the stage's exponent on the host, a z table's from its native slots.

        A host engine's propagator and the dense observation's stiffness test read it;
        a z table's two values are copied from the native slots and subtracted once, as
        the stage kernels subtract them (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            here = tables.values("phi", bundle.z1)
            anchored = tables.values("phi", bundle.z0)
            return (here - anchored).reshape(self._layout)
        if bundle.phi_delta is None or self._terms.phi is None:
            return delta_phi(
                self._problem.linear, bundle.z0, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.phi_delta, self._terms.phi)

    def _unitary_phase(self, bundle: StageScalars) -> RealArray:
        """Return ``Im Phi`` at the stage's position, a z table's from its slot.

        An expansion of ``Phi`` is cumulative and vanishes at the start of the range,
        so its value at a position is the bundle's ``phi_at`` against the same terms.
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            return np.imag(tables.values("phi", bundle.z1)).reshape(self._layout)
        if bundle.phi_at is None or self._terms.phi is None:
            return unitary_phase(self._problem.linear, bundle.z1)
        return np.imag(combine(bundle.phi_at, self._terms.phi))

    def _hoist(self, z0: float) -> None:
        """Hoist a step's anchor in the stage cache (`_load.StageCache.hoist`)."""
        self._stages.hoist(z0)

    def _stage_bundles(
        self, z0: float, positions: Sequence[float]
    ) -> tuple[StageScalars, ...]:
        """Return several stages' scalars taken together (`StageCache.bundles`)."""
        return self._stages.bundles(z0, positions)

    def _stage_scalars(self, z0: float, z1: float) -> StageScalars:
        """Return one stage's scalars (`StageCache.scalars`)."""
        return self._stages.scalars(z0, z1)

    def _score_trial(self, partials: RealArray) -> float:
        """Reduce one member's partials to the trial's scaled error norm.

        solprop's own weak scale is formed natively, the partials summed in order
        (solprop SEP 45 phase 2, Q1), and a weighted one likewise (phase 4). A
        controller with a scale of its own never reaches here: the backend refuses it
        where it is built.
        """
        controller = self._stepper.controller
        if controller.weights is None:
            return float(_core.rk_score(partials, controller.rtol, controller.atol))
        # The weighted scale and its reduction, in order (SEP 45 phase 4).
        self._rk_weights_ready()
        return float(
            self._native_state.rk_weighted_score(
                np.ascontiguousarray(partials, dtype=np.float64),
                controller.rtol,
                controller.atol,
            )
        )

    def _rk_weights_ready(self) -> None:
        """Hand the native state a weighted controller's weights, once (SEP 45)."""
        if not self._rk_weighted:
            weights = self._stepper.controller.weights
            assert weights is not None
            self._native_state.rk_weights_configure(
                [float(value) for value in np.asarray(weights, dtype=np.float64)]
            )
            self._rk_weighted = True

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Hold the boundary monitor's weights in the native state (solprop SEP 21).

        From here on the engine's finish adds a split-step collar's removal at every
        accepted step, on every march, and :meth:`_boundary_energies` reads it.
        """
        if monitor is None:
            self._native_state.unbind_monitor()
            return
        columns = (
            np.empty(0, dtype=np.float64)
            if monitor.columns is None
            else np.ascontiguousarray(monitor.columns, dtype=np.float64)
        )
        self._native_state.bind_monitor(
            np.ascontiguousarray(monitor.weights, dtype=np.float64).reshape(-1), columns
        )
        crossing = monitor.crossing
        # A runner's points share the table: uploading it again would move a sixteenth
        # of a state per point.
        if crossing is not None and crossing.buckets is not self._bound_buckets:
            # The split collar's crossing limit, read from the state's occupancy of
            # its walk-off buckets at every check (solprop #440).
            self._native_state.bind_occupancy(
                np.ascontiguousarray(crossing.buckets, dtype=np.uint8).reshape(-1),
                SPLIT_RATE_BUCKETS,
            )
            self._bound_buckets = crossing.buckets
        edge = getattr(monitor, "transverse", None)
        if edge is not None:
            # Weights and no faces: what it reads comes back to the edge's judging.
            tables = edge.tables
            synthesis = edge.reduction.synthesis()
            self._native_state.bind_edge(
                np.ascontiguousarray(tables.faces, dtype=np.float64).reshape(-1),
                np.ascontiguousarray(tables.clip, dtype=np.float64).reshape(-1),
                np.ascontiguousarray(tables.totals, dtype=np.float64).reshape(-1),
                np.empty(0, dtype=np.float64)
                if synthesis is None
                else np.ascontiguousarray(synthesis, dtype=np.float64).reshape(-1),
                int(np.argmax(tables.rows)),
            )
            self._bind_edge_lines(edge)

    def _bind_edge_lines(self, edge: TransverseEdge) -> None:
        """Hand native a requested collar's lines, as arrays (solprop SEP 44)."""
        lines = edge.reduction.lines
        axes = {"r": 0, "x": 1, "y": 2}
        none = 2**64 - 1
        strip = [np.asarray(line.strip, dtype=np.uint64) for line in lines]
        window = [np.asarray(line.window, dtype=np.float64) for line in lines]
        onsets = [
            line.onset if line.onset is not None else (none, 0, 0) for line in lines
        ]
        space = edge.space
        self._native_state.bind_edge_lines(
            np.array([axes[line.axis] for line in lines], dtype=np.uint64),
            np.array([array.size for array in strip], dtype=np.uint64),
            np.concatenate(strip) if strip else np.empty(0, dtype=np.uint64),
            np.concatenate(window) if window else np.empty(0, dtype=np.float64),
            np.array([line.inward for line in lines], dtype=np.uint64).reshape(-1),
            np.array(
                [
                    line.spacing / (2 * array.size) * line.measure
                    if array.size
                    else 0.0
                    for line, array in zip(lines, strip, strict=True)
                ],
                dtype=np.float64,
            ),
            np.array([line.measure for line in lines], dtype=np.float64),
            np.array(onsets, dtype=np.uint64).reshape(-1),
            np.array([line.onset_step for line in lines], dtype=np.float64),
            float(edge.k_min),
            np.sqrt(np.asarray(space.r, dtype=np.float64))
            if lines and isinstance(space, RadialGrid)
            else np.empty(0, dtype=np.float64),
            transverse_monitor._STRIP_BATCH_BYTES,  # pyright: ignore[reportPrivateUsage]
        )

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

    def _edge_reading(
        self,
        state: _core.State,
        edge: TransverseEdge,
        *,
        faces: bool = True,
        current: bool = False,
        full: bool = False,
    ) -> _EdgeReading:
        """Read the state for the transverse edge where it lives (solprop SEP 44).

        - Returns: the reading, of which only numbers crossed. A requested collar still
          reduces a host copy of the state.
        """
        if state is not self._native_state:
            raise BackendError("the read state does not belong to this backend")
        if (counters := _counters.active) is not None:
            counters.transverse_reads += 1
        collar = edge.every_step
        sums, columns, strip, rate = state.edge_reading(
            faces, full, collar, faces and collar, current
        )
        rate_value = float(rate) if current else None
        if not faces:
            return _EdgeReading(current=rate_value)
        values = np.asarray(sums).reshape(-1)
        if full:
            clip, totals = values[4:8], values[8:12]
        else:
            clip, totals = None, values[4:8] if collar else None
        return _EdgeReading(
            faces=values[:4],
            clip=clip,
            totals=totals,
            columns=np.asarray(columns).reshape(-1),
            strip=np.asarray(strip).reshape(-1, 2) if collar else None,
            current=rate_value,
        )

    def _finish(self, state: _core.State, dz: float) -> _core.State:
        if state is not self._native_state:
            raise BackendError("the advanced state does not belong to this backend")
        state.finish(dz)
        return state

    def _collar_rests(self) -> bool:
        """Return whether this run's stages may leave a quiet collar out (SEP 21)."""
        return collar_rests(self._problem, self._stepper)

    def _launch_clean_columns(self, edge: TransverseEdge) -> RealArray | None:
        """Return each column's launched energy without the input draw, or ``None``.

        What the transverse edge's safety ramp is judged on (SEP 40), so a run with no
        ramp needs none. Formed where the load held both parts (`_load`), which needs
        no state on the host; only a launch that resumed without its edge's record has
        none, and then the loaded state crosses once.

        - Returns: the unweighted energies, or ``None`` with no draw, no ramp or on a
          resumed launch.
        """
        loaded = self._loaded
        floor = None if loaded is None else loaded.floor
        if loaded is None or floor is None or edge.resumed or edge.design is None:
            return None
        if loaded.clean_columns is not None:
            return loaded.clean_columns
        state = np.asarray(self._observed_field(self._native_state)[0])
        return edge.reduction.column_power(state - np.asarray(floor[0]))

    def timing_probe(self, start: Start) -> TimingProbe:
        """Load a start and return single trials of this run's march from it.

        The start's field, noise, position and resume flag are loaded as a session
        loads them (`_load`); its method history, ledgers and monitor continuation are
        not read. A Runge-Kutta trial (``rk_trial``) leaves the loaded state alone; the
        low-storage march's step advances the discarded registers in place, so a
        backend probed so is closed afterwards (solprop SEP 47 section 6.6).

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
        return NativeProbe(self, loaded, start.noise)

    def _observed_field(self, state: _core.State) -> ComplexArray:
        if state is not self._native_state:
            raise BackendError("the observed state does not belong to this backend")
        field = state.field_to_host().reshape(self._layout)
        if (counters := _counters.active) is not None:
            counters.crossing("plane", field.nbytes)
        return field

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> None:
        """Prepare the native Raman kernel for private memoization diagnostics."""
        if not self._problem.responses.raman_terms:
            return None
        self._native_state.prepare_raman(
            np.ascontiguousarray(np.asarray(rates, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(densities, dtype=np.float64)),
        )

    @property
    def _raman_transforms(self) -> int:
        """Return the number of native Raman-kernel transforms."""
        return int(self._native_state.raman_kernel_transforms)

    def _far_field_statistics(
        self, names: set[str], spec: StatsSpec | None
    ) -> dict[str, np.ndarray]:
        """Form the far-field statistics in the native banded pass (SEP 23 section 6).

        The tables are those solprop's :class:`~solprop.backends.spatial.FarFieldTables`
        forms for the call's spec, packed as :func:`_aperture_payload` says, and
        ``(n_regions, 4)`` boxes; the pass returns each statistic finished, in
        :data:`_FAR_FIELD_ORDER`.
        """
        far = names & set(FAR_FIELD_STATISTICS)
        tables = (
            None
            if not far
            else self._far_field.for_spec(
                self._request.observation.stats if spec is None else spec
            )
        )
        if far and tables is None:
            raise BackendError(
                f"the far-field statistics {sorted(far)} need the tables the backend "
                "forms at construction"
            )
        n_omega = self._problem.shape[-1]
        apertures, regions, exponent = _aperture_payload(tables, n_omega)
        flags = 0
        for name in names:
            flags |= _FAR_FIELD_FLAGS[name]
        packed = np.asarray(
            self._native_state.aperture_statistics(
                flags, apertures, exponent, regions, 0.5 * EPS0 * C
            )
        ).ravel()
        n_apertures = apertures.shape[0]
        rows = {
            "beam_spectrum": None,
            "absorbed_power": 0,
            "region_spectrum": regions.shape[0],
            "aperture_spectrum": n_apertures,
            "aperture_on_axis_spectrum": n_apertures,
            "on_axis_spectrum": None,
        }
        values: dict[str, np.ndarray] = {}
        offset = 0
        for name in _FAR_FIELD_ORDER:
            if name not in names:
                continue
            count = rows[name]
            if name == "absorbed_power":
                size, shape = len(ABSORBED_CHANNELS), (1, len(ABSORBED_CHANNELS))
            elif count is None:
                size, shape = n_omega, (1, n_omega)
            else:
                size, shape = count * n_omega, (1, count, n_omega)
            values[name] = packed[offset : offset + size].reshape(shape)
            offset += size
        if offset != packed.size:
            raise BackendError(
                f"native far-field statistics returned {packed.size} values; "
                f"consumed {offset}"
            )
        return values

    def march_export_refusal(self) -> str | None:
        """Return ``None``: every march the backend admits carries its history.

        Returns
        -------
        None
            The resident Adams engine exports and installs its rows; a Runge-Kutta
            or low-storage march keeps none.
        """
        return None

    def metadata(self) -> dict[str, object]:
        """Return the shared execution metadata plus the native build record.

        The build record and the installed solprop version cannot change within a
        process (thread settings are read once, at import), so they are read once
        (:func:`_build_facts`); the native state's counters are read at every call.
        """
        facts = _build_facts()
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
        elif self._resident_low_storage is not None:
            work["source_evaluations"] = self._low_storage_evaluations
        else:
            work["source_evaluations"] = self._native_state.rk_source_evaluations
        # Which march steps the run (solprop SEP 39 and 45 section 5): the native one,
        # the only one since SEP 47 phase 5.
        if self._adams is not None:
            work["adams_march"] = "native"
        elif self._resident_low_storage is None:
            work["rk_march"] = "native"
        if self._statistics_tables:
            # The profile statistics' tables and their measured errors (SEP 47 6.4).
            work["statistics_tables"] = dict(self._statistics_tables)
        if self._general_chunk is not None:
            work["general_node_chunk"] = self._general_chunk
        work["node_chunk_rule"] = (
            "device" if _on_device() and self.chunk_plan.automatic else "host"
        )
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
            | _planning_record(self._native_state),
            "host_blas": self._host_blas,
            "planning_seconds": self._native_state.planning_seconds,
            "wisdom_loaded": self._wisdom_loaded,
            "wisdom_digest": self._wisdom_digest,
            "raman_fft_provider": self._native_state.raman_fft_provider,
            "raman_fft_tasks": self._native_state.raman_fft_tasks,
            "execution_sizing": self._native_state.execution_sizing,
            "raman_planning_seconds": self._native_state.raman_planning_seconds,
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
            "raman_kernel_transforms": self._native_state.raman_kernel_transforms,
            "fsal_ready": self._native_state.fsal_ready,
            "dispersive_path": dict(self._native_state.dispersive_path) or None,
            "synthesis_provider": self._native_state.synthesis_provider,
            "projection_provider": self._native_state.projection_provider,
            "gemm_shape_policy": self._native_state.gemm_shape_policy,
            "device_team_size": self._native_state.response_team_size,
            "device_team_size_limit": facts.device_team_size,
            "device_team_size_policy": (
                "256 below n_to=8192; configured limit otherwise"
                if facts.info["device_backend"] == "cuda"
                else "fixed configured team width"
            ),
            "device_reduction_size": facts.device_reduction_size,
        }

    @property
    def node_chunk(self) -> int:
        """Return the nodes a native response pass holds at once.

        Returns
        -------
        int
            The host plan lowered by :func:`native_node_chunk`.
        """
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
            for name, value in self._native_state.memory_ledger().items()
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

    def prepare_group(
        self,
        members: Sequence[Backend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
    ) -> PreparedGroup:
        """Prepare a group of compatible native members (SEP 47 section 6.7).

        Its start opens the native group session
        (:class:`solprop_native.group.NativePreparedGroup`), which refuses by name a
        group it does not run.

        Parameters
        ----------
        members : sequence of solprop.backends.protocol.Backend
            One constructed native backend per member, this one first.
        problems : sequence of solprop.translate.problem.PropagationProblem
            Each member's independently translated problem.
        requests : sequence of solprop.translate.run_request.RunRequest
            Each member's run request.

        Returns
        -------
        solprop.backends.session.PreparedGroup
            The prepared group.
        """
        from .group import NativePreparedGroup

        return NativePreparedGroup(members, problems, requests)

    def close(self) -> None:
        """Release native plans and views; repeated calls have no effect."""
        if not self._closed:
            self._native_state.close()
            _host_blas.release()
        self._closed = True


def _adopt_nothing(z: float | None) -> None:
    """Adopt nothing: the backend keeps no march state beside the native state.

    - ``z``: where the engine's state moved, m; unused.
    """
    del z


@dataclasses.dataclass(frozen=True)
class _BuildFacts:
    """The parts of a backend's metadata that are fixed for the process."""

    info: Mapping[str, object]
    solprop_version: str
    device_team_size: int
    device_reduction_size: int


@functools.cache
def _build_facts() -> _BuildFacts:
    """Return the build record and the installed solprop version, read once.

    Reading them costs about a millisecond (the package metadata lookup and the native
    record), which a per-plane metadata record paid at every saved plane.
    """
    native_record = {
        key: value for key, value in _core.build_record().items() if value is not None
    }
    return _BuildFacts(
        info=types.MappingProxyType(dataclasses.asdict(build_info())),
        solprop_version=version("solprop"),
        device_team_size=int(native_record["device_team_size"]),
        device_reduction_size=int(native_record["device_reduction_size"]),
    )


def _repeated_row(values: tuple[float, ...], count: int) -> RealArray:
    """Return a stage row of host scalars repeated `count` times, ``(count, n)``."""
    return np.repeat(np.array([values], dtype=np.float64), count, axis=0)


def _required(values: RealArray | None, name: str) -> RealArray:
    if values is None:
        raise BackendError(f"device-ready problem has no {name}")
    return values


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


type _TrialArrays = tuple[
    tuple[RealArray, RealArray, RealArray, RealArray, RealArray, RealArray, RealArray],
    dict[str, npt.NDArray[np.float64] | int],
]


@dataclasses.dataclass
class _ResidentAdams:
    """solprop's Adams history kept in the native state (solprop SEP 16 phase 3b).

    The native Adams march's engine (``solprop.backends.multistep.AdamsEngine``):
    the history's rows, a trial's predictor, its source, the corrected field and the
    defect norms, the adoption and transport of an accepted step and dense observation
    run in the engine, with only the trial's scalars, coefficient tables and norms
    crossing. solprop's host engine (``HostHistory``) is its reference. The backend
    hands it the native state and three of its own operations.
    """

    native: _core.State
    method: Multistep
    controller: PIController
    n_omega: int
    arrays: Callable[[float, float], _TrialArrays]
    """``(anchor, position) -> trial arrays`` with every stage row at `position`."""
    hoist: Callable[[float], None]
    adopted: Callable[[float | None], None]
    """Make the native state the backend's, at a new position when one is given."""
    rate: Callable[..., RealArray]
    """``(z, *, before=False) -> D(z)``'s weights over the native ``Phi`` terms."""
    shape: tuple[int, ...]
    """The state layout, ``(1, n_modes, n_pol, n_omega)``."""
    helpers: Helpers
    """The backend's Chebyshev and Adams-basis helpers (``NativeBackend.helpers``)."""
    interpolate: bool
    compensated: bool = False
    """Whether trials carry a varying operator's transported compensation."""
    carrier: CarrierReference | None = None
    """The run's carrier frame (solprop SEP 35): its shift is formed on the host from
    the reference element's chord, one scalar a trial, and nothing is read back."""
    dense: None = None
    _trial: tuple[float, float, list[float], npt.NDArray[np.int64], int] | None = None
    _step: tuple[float, float, npt.NDArray[np.float64]] | None = None
    _exponential: bool = False
    _coefficients: npt.NDArray[np.float64] | None = None
    _dense_nodes: npt.NDArray[np.float64] | None = None
    _trial_carrier: RealArray = dataclasses.field(default_factory=lambda: _NO_CARRIER)
    _dense_carrier: RealArray = dataclasses.field(default_factory=lambda: _NO_CARRIER)

    def __post_init__(self) -> None:
        """Allocate the history's rows in the native state."""
        self.native.adams_allocate(self.n_slots, self.compensated, self.interpolate)
        self._exponential = self.method.formulation == "exponential"
        # The controller's scale is formed on the device each trial, so that the
        # state's magnitude never crosses: it is solprop's own, since PIController
        # is final (ADR-077).
        controller = self.controller
        weights = controller.weights
        self.native.adams_scale(
            float(controller.rtol),
            float(controller.atol),
            np.empty(0, dtype=np.float64)
            if weights is None
            else np.ascontiguousarray(weights, dtype=np.float64),
        )

    @property
    def n_slots(self) -> int:
        """The history's rows: the method's order and two more."""
        return self.method.order + 2

    def march_stepped(
        self,
        step: tuple[float, float, list[float]] | None,
        *,
        fused: bool,
        carrier: Sequence[float],
    ) -> None:
        """Adopt what a native march call leaves (solprop SEP 39 phase 2).

        No trial is held, and the last accepted step's anchor, step and nodes ``(1,
        theta...)``, if the call accepted one, set the dense observation's
        interpolant: the fused path's nodes, or the exponential corrector's
        coefficients on them, with the step's carrier rotations.
        """
        self._trial = None
        if step is not None:
            z, dz, nodes = step
            interpolant = (
                np.array(nodes)
                if fused
                else lagrange_coefficients(nodes, check=False, helpers=self.helpers)
            )
            self._step = (z, dz, interpolant) if self.interpolate else None
            self._dense_nodes = np.array(nodes)
            self._dense_carrier = (
                np.asarray(carrier, dtype=np.float64) if carrier else _NO_CARRIER
            )
        self.adopted(None)

    def reset(self) -> None:
        """Forget the last accepted step and any trial in flight."""
        self._trial = None
        self._step = None
        self._dense_carrier = _NO_CARRIER

    def _carrier(self, z: float, dz: float, nodes: list[float], rows: int) -> RealArray:
        """Return a trial's packed carrier frame, empty without one.

        ``(Im sigma, Re f_0, Im f_0, ...)`` over the `rows` history rows the trial
        reads, ``f_t = exp(sigma (1 - theta_t))``; ``sigma`` is the reference element's
        chord over the step (solprop SEP 35 section 6.1), formed from the operator's
        own representation, so a device build reads nothing back.
        """
        carrier = self.carrier
        if carrier is None or not carrier.active:
            return _NO_CARRIER
        assert carrier.chord is not None
        # One batch member per native state.
        shift = float(np.imag(carrier.chord.exponent(z, z + dz))[0])
        packed = np.empty(1 + 2 * rows, dtype=np.float64)
        packed[0] = shift
        # exp(i shift s) as NumPy's complex exponential forms it: exp(0) (cos + i sin).
        spans = 1.0 - np.asarray(nodes[:rows], dtype=np.float64)
        packed[1::2] = np.cos(shift * spans)
        packed[2::2] = np.sin(shift * spans)
        return packed

    def hold(self) -> object:
        """Keep the accepted state aside in the engine across a dense observation."""
        self.native.adams_hold()
        return None

    def restore(self, z: float, held: object) -> None:
        """Make the held accepted state current again at `z`."""
        del held
        self.native.adams_restore()
        self.adopted(z)

    def start(self, z: float) -> None:
        """Evaluate the history's first sample at the accepted state into row zero."""
        self.hoist(z)
        arrays, material = self.arrays(z, z)
        with _envelope_errors():
            self.native.adams_start(0, *arrays, material)
        if self.compensated:
            # The startup sample's state and rate, read on the far side of a kink.
            self.native.adams_compensation_start(0, self.rate(z))

    def value(self, slot: int) -> None:
        """The rows are resident: there is no host row to name."""
        del slot

    # --- persistence across a resume (solprop SEP 29 G2) ------------------------------

    def snapshot_nbytes(self) -> int:
        """Return the bytes :meth:`snapshot_rows` would copy, copying nothing."""
        rows = len(self.native.adams_row_names())
        return rows * self.n_slots * math.prod(self.shape) * 2 * 8

    def snapshot_rows(self) -> dict[str, ComplexArray]:
        """Return host copies of the rows a resume needs to continue the history.

        The rows ``HostHistory`` also writes keep its names and layout,
        ``(n_slots, n_batch, n_modes, n_pol, n_omega)``: ``"store"`` and, compensated,
        ``"states"`` and ``"rates"``. A frozen-residual flow's history holds each
        sample's generator applied to its state, ``"applied"``, where the host engine
        holds the sample's weights (``"weights_<row>"``); the two are not
        interchangeable, and each engine refuses the other's. The copies are lossless.
        Only the trial's scratch stays behind; the step's bookkeeping is the march's.
        """
        expected = (self.n_slots, *self.shape)
        return {
            name: self.native.adams_rows_to_host(name).reshape(expected)
            for name in self.native.adams_row_names()
        }

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows `snapshot_rows` returned, for a state of `shape`.

        Everything is checked before any row is copied, so a refused snapshot leaves
        the history as it was. The trial in flight and the last step's interpolant
        are forgotten, as ``HostHistory.load_rows`` forgets them: the march's next step
        is a new trial.
        """
        names = self.native.adams_row_names()
        weights = sorted(name for name in rows if name.startswith("weights_"))
        if weights:
            listed = ", ".join(weights)
            raise BackendError(
                "the native backend cannot continue this Adams history: the snapshot "
                f"holds the host engine's frozen-residual weights ({listed}), and the "
                "native engine keeps each sample's applied generator ('applied') "
                "instead"
            )
        foreign = sorted(set(rows) - set(names))
        missing = [name for name in names if name not in rows]
        if foreign or missing:
            raise BackendError(
                "the native backend cannot continue this Adams history: its engine "
                f"keeps the rows {names}, and the snapshot "
                + (f"lacks {missing}" if missing else f"also holds {foreign}")
            )
        expected = (self.n_slots, *shape)
        for name in names:
            row = rows[name]
            if row.shape != expected or not np.iscomplexobj(row):
                raise BackendError(
                    f"the snapshot's {name!r} rows ({row.dtype}, shape {row.shape}) do "
                    f"not fit the native backend's complex store of shape {expected}"
                )
        self.reset()
        for name in names:
            flat = np.ascontiguousarray(rows[name], dtype=np.complex128)
            self.native.adams_rows_from_host(
                name, flat.reshape(self.n_slots, -1, shape[-1])
            )

    def fused_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: npt.NDArray[np.int64],
        order: int,
        orders: list[int],
    ) -> RealArray:
        """Form an interaction-formulation trial natively; return its scored norms.

        The native state forms the tables and the stage scalars from the nodes alone
        (solprop SEP 39 phase 3), by the same operations as the host's.
        """
        with _envelope_errors():
            norms = self.native.adams_fused_trial_at(
                z,
                dz,
                np.asarray(nodes, dtype=np.float64),
                np.ascontiguousarray(slots[: max(max(orders), order)], dtype=np.uint64),
                order,
                list(orders),
            )
        self._trial = (z, dz, nodes, slots, order)
        return np.asarray(norms, dtype=np.float64).reshape(-1)

    def fused_accept(
        self, keep: npt.NDArray[np.int64], free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held trial in the engine; keep its interpolant there."""
        assert self._trial is not None
        z, dz, nodes, slots, order = self._trial
        self._trial = None
        self.native.adams_fused_accept(
            dz,
            np.ascontiguousarray(keep, dtype=np.uint64),
            free,
            np.ascontiguousarray(slots[:order], dtype=np.uint64),
            interpolate,
        )
        self.adopted(None)
        self._step = (z, dz, np.array([1.0, *nodes[:order]])) if interpolate else None

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: npt.NDArray[np.int64],
        order: int,
        orders: list[int],
    ) -> RealArray:
        """Form an exponential trial natively; return its scored norms.

        The step's exponent, flow and moments come from the trial's own arrays; the
        predicted field's source is evaluated in the laboratory frame, at a trial
        whose anchor is the step's end. The native state forms the tables, the stage
        scalars and the rate from the nodes alone (solprop SEP 39 phase 1), by the
        same operations as the host's: the coefficient tables
        (``lagrange::trial_tables``), the stage scalars and the rate from the tables
        uploaded once, and the divisor; the carrier frame's shift is the host's. A run
        that interpolates its planes keeps the corrector's coefficients on the host for
        its observations, from the same basis.
        """
        width = 1 + max(orders)
        carrier = self._carrier(z, dz, nodes, width - 1)
        with _envelope_errors():
            norms = self.native.adams_exponential_trial_at(
                z,
                dz,
                np.asarray(nodes, dtype=np.float64),
                np.ascontiguousarray(slots[: max(width - 1, order)], dtype=np.uint64),
                order,
                list(orders),
                self.compensated,
                carrier,
            )
        self._trial = (z, dz, nodes, slots, order)
        self._trial_carrier = carrier
        self._coefficients = (
            lagrange_coefficients(
                [1.0, *nodes[:order]], check=False, helpers=self.helpers
            )
            if self.interpolate
            else _NO_COEFFICIENTS
        )
        self._dense_nodes = np.array([1.0, *nodes[:order]])
        return np.asarray(norms, dtype=np.float64).reshape(-1)

    def exponential_accept(
        self, keep: npt.NDArray[np.int64], free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held trial in the engine; the new sample takes row `free`."""
        del keep  # untransported samples are never re-anchored
        assert self._trial is not None
        z, dz, _, slots, order = self._trial
        self._trial = None
        self.native.adams_exponential_accept(
            dz,
            free,
            np.ascontiguousarray(slots[:order], dtype=np.uint64),
            interpolate,
        )
        self.adopted(None)
        coefficients = self._coefficients
        assert coefficients is not None
        self._step = (z, dz, coefficients) if interpolate else None
        # The accepted step's shift and the rotations of the `order` history rows its
        # interpolant reads, held across later trials and rejections.
        trial = self._trial_carrier
        self._dense_carrier = trial[: 1 + 2 * order] if trial.size else _NO_CARRIER

    def observe(self, position: float) -> object:
        """Interpolate the last accepted step natively into the engine's state."""
        assert self._step is not None
        z, dz, nodes = self._step
        if self._exponential:
            # `nodes` holds the accepted corrector's coefficients here; a varying
            # operator's compensation is interpolated over the step's own samples.
            xi = (position - z) / dz
            weights = thetas = np.empty(0, dtype=np.float64)
            if self.compensated:
                assert self._dense_nodes is not None
                thetas = self._dense_nodes
                weights = integrated_weights(
                    thetas, xi, check=False, helpers=self.helpers
                )
            self.native.adams_exponential_observe(
                xi,
                dz,
                np.ascontiguousarray(nodes, dtype=np.float64).reshape(-1),
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(thetas, dtype=np.float64),
                np.ascontiguousarray(self._dense_carrier),
            )
            return self.native
        # The step's own derivative polynomial.
        theta = (position - z) / dz
        weights = dz * integrated_weights(
            nodes, theta, check=False, helpers=self.helpers
        )
        arrays, material = self.arrays(z, position)
        self.native.adams_observe(
            np.ascontiguousarray(weights, dtype=np.float64), *arrays, material
        )
        return self.native


_EMPTY = np.empty(0, dtype=np.complex128)
"""No modified-noise realisation: the native state adds none."""

_NO_SCHEDULE = np.empty(0, dtype=np.float64)
"""The empty schedule of an adaptive march's configuration."""

_NO_CARRIER = np.empty(0, dtype=np.float64)
"""The empty carrier frame: the native trial takes the unshifted arithmetic."""

_NO_COEFFICIENTS = np.empty((0, 0), dtype=np.float64)
"""The corrector's coefficients a staged trial keeps when no plane is interpolated."""

_STAGE_CONSTANT = 1
_STAGE_EXPANSION = 2
_STAGE_TABLE = 3
"""How the stage tables name a constant, an expanded or a z table's ``Phi`` and
prefactor (``stage::Terms``)."""


def _section_tables(
    problem: PropagationProblem, *, moving: bool, residual: bool
) -> dict[str, object]:
    """Return a coupled flow's and a moving guide's sections as the stage tables take
    them: their edges, degrees and the edges' slack (`flow_edge_slack`,
    `transform_section`), and whether the flow carries a frozen residual."""
    flow = problem.linear.coupled_flow
    maps = problem.transform.modal_tables if moving else None
    out: dict[str, object] = {
        "flow_edges": [],
        "flow_degrees": 0,
        "flow_slack": 0.0,
        "flow_residual": False,
        "transform_edges": [],
        "transform_degrees": 0,
        "transform_slack": 0.0,
    }
    if flow is not None:
        out |= {
            "flow_edges": np.asarray(flow.z_edges, dtype=np.float64).tolist(),
            "flow_degrees": int(flow.n_deg),
            "flow_slack": float(flow_edge_slack(flow)),
            "flow_residual": residual,
        }
    if maps is not None and maps.synthesis is not None:
        edges = np.asarray(maps.z_edges, dtype=np.float64)
        out |= {
            "transform_edges": edges.tolist(),
            "transform_degrees": int(maps.synthesis.shape[1]),
            "transform_slack": _EDGE_SLACK_FRACTION * float(edges[-1] - edges[0]),
        }
    return out


def _chord_tables(carrier: CarrierReference | None) -> dict[str, object]:
    """Return the carrier frame's chord as the native stage tables take it.

    The imaginary part of the reference element's operator for the one batch member
    (`CarrierChord`): a constant's rate, an expansion's terms over the ``Phi`` weights
    the tables already hold, or a z table's reference-element column as a scalar table
    of its own (solprop issue #255), packed as its batch pass reads it. None without an
    active frame.
    """
    chord = None if carrier is None or not carrier.active else carrier.chord
    if chord is not None and chord.constant is not None:
        return {
            "chord": 1,
            "chord_rate": float(np.imag(chord.constant)[0]),
            "chord_terms": [],
            "chord_table": None,
        }
    if chord is not None and chord.expansion is not None:
        terms = np.imag(np.asarray(chord.expansion.terms))[:, 0]
        return {
            "chord": 2,
            "chord_rate": 0.0,
            "chord_terms": terms.tolist(),
            "chord_table": None,
        }
    if chord is not None and chord.table is not None:
        edges, lower, upper, coefficients = chord.table.packed()
        # The one batch member's column: the imaginary part a complex Clenshaw pass
        # forms at a real position, by the same operations (`stage::chord`).
        column = np.ascontiguousarray(np.imag(coefficients[:, :, :1]))
        return {
            "chord": 3,
            "chord_rate": 0.0,
            "chord_terms": [],
            "chord_table": (edges, lower, upper, column),
        }
    return {"chord": 0, "chord_rate": 0.0, "chord_terms": [], "chord_table": None}


def _on_device() -> bool:
    """Return whether the installed build executes on an accelerator."""
    return _build_facts().info["device_backend"] != "none"


DEVICE_BUDGET_SHARE = 4
"""The share of the accelerator's free bytes an automatic plan budgets: one quarter.

The budget is spent on the host plan's conservative inventory (twenty column buffers a
Cartesian column, about three times the live buffers, solprop issue #349), so the live
space-domain buffers take near a twelfth of the free bytes, leaving the rest for the
state, the march's registers and a scan's other members."""


def device_node_chunk(
    problem: PropagationProblem, planned: int, free: int | None
) -> int:
    """Return the node chunk a device build takes when no budget or chunk is given.

    solprop issue #361: on a device a node chunk is a kernel launch, and fewer, larger
    launches are faster: one chunk took 0.74-0.92 of the default's time on AMD Instinct
    MI300X spatial routes (SEP 47 Part II), at the cost of space-domain buffers for
    every node at once. So an automatic plan budgets :data:`DEVICE_BUDGET_SHARE`'s share
    of the free device bytes, through the host plan's own rule and response admissions
    (:func:`solprop.backends.node_chunk.plan_node_chunk`): one chunk wherever that fits,
    otherwise the largest that does, and never fewer nodes than the host plan's chunk
    at the route's default budget. It is consulted only for an automatic plan: an
    explicit budget or chunk always wins. It runs at backend construction, after the
    translation, so no device runtime starts before a scan forks.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    planned : int
        The host plan's chunk at the default budget
        (:func:`solprop.backends.node_chunk.plan_node_chunk`).
    free : int or None
        The accelerator's free bytes, or ``None`` where none is known.

    Returns
    -------
    int
        The device's chunk, between 1 and the node count.
    """
    if free is None:
        return planned
    budget = max(1, free // DEVICE_BUDGET_SHARE)
    try:
        roomy = plan_node_chunk(problem, ExecutionOptions(node_budget=budget))
    except BackendError:
        # A quarter of the free bytes cannot hold one node: the default plan stands,
        # and its own allocation reports what does not fit.
        return planned
    return max(planned, roomy.chunk)


def native_node_chunk(
    problem: PropagationProblem, plan: NodeChunkPlan, *, device: bool
) -> int:
    """Lower the host node-chunk plan to the chunk this backend executes.

    A device build's automatic plan goes through :func:`device_node_chunk`. The chunk
    is then bounded by the node count, and on the hybrid planar kind made whole y rows,
    as many as it holds and at least one, which is solprop's reference's own blocking.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    plan : solprop.backends.node_chunk.NodeChunkPlan
        The host plan.
    device : bool
        Whether this build executes on an accelerator.

    Returns
    -------
    int
        The chunk the native problem is built with.
    """
    chunk = plan.chunk
    if device and plan.automatic:
        chunk = device_node_chunk(problem, chunk, _core.available_device_memory())
    chunk = min(chunk, problem.transform.n_nodes)
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "slab":
        n_x, n_y = geometry.shape
        chunk = min(n_y, max(1, chunk // n_x)) * n_x
    return chunk


def _low_storage_payload(
    problem: PropagationProblem, stepper: StepperSettings, *, device: bool
) -> dict[str, object]:
    """Return the native problem's low-storage rows (solprop kernel schema 32).

    The march's own allocation: its registers in place of the Runge-Kutta views, the
    stage register elided where solprop's estimate elides it, and the flows solprop's
    estimate counts -- except that a device build recomputes under ``"auto"``, where
    the estimate's CPU count over-states it by the table. A device's input crosses the
    state's layout through tiles (``docs/dev/benchmarks.md``); a CPU's has its own.
    """
    if not isinstance(stepper.integrator, LowStorageRungeKutta):
        return {}
    flows = low_storage_flows(problem, stepper, "native")
    if device and stepper.propagator_mode == "auto":
        flows = 0
    return {
        "low_storage": True,
        "low_storage_stage_register": low_storage_stage_register(problem, "native"),
        "low_storage_flows": flows,
        "low_storage_tiled_input": device,
    }


_MARCH_COUNTERS = (
    "startup",
    "predictor",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
    "dense_observations",
)
"""The call ledger's counters the native march keeps."""

_REASONS = (
    "reached",
    "host_step",
    "monitor",
    "bound",
    "budget",
    "floor",
    "error",
    "diverged",
    "watch",
    "capacity",
    "dense",
    "stopped",
    "halted",
)
"""Why a native march call returned (``march::Reason``), by name, in its order; the
benchmark counters (:mod:`solprop_native._counters`) report these names."""
(
    _REACHED,
    _HOST_STEP,
    _MONITOR,
    _BOUND,
    _BUDGET,
    _FLOOR,
    _ERROR,
    _DIVERGED,
    _WATCH,
    _CAPACITY,
    _DENSE,
    _STOPPED,
    _HALTED,
) = range(len(_REASONS))

_CONVERGENCE, _BACKEND, _VALUE, _REFUSED = 1, 2, 3, 4
"""Which error a native failure maps to (``march::Category``)."""


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


class _NativeAdamsMarch:
    """The native Adams march's record: its policy, ledger and history (SEP 39, 47).

    The native state steps solprop's exponential Adams march (solprop SEP 39 phase 2):
    the controller, the history, the landings and the call ledger's counts advance
    natively, from one policy record (:func:`~solprop.stepper.multistep.march_policy`).
    This record is what the run starts from: the policy, the history's entries
    (position, identity and engine row), the order, the rejection flags and the
    :class:`~solprop.backends.multistep.CallLedger`, each fresh at a load
    (:meth:`reset`, :meth:`begin_run`) or restored from a resume's snapshot and ledger
    (:meth:`resume`, :meth:`restore_ledger`). A native session hands it to the native
    state once (:meth:`start_state`, :meth:`_configure`) and holds the march's state
    there from then on (solprop SEP 47 section 5.2).
    """

    def __init__(
        self, backend: Any, method: Multistep, *, engine: object | None = None
    ) -> None:
        """Build the record over the backend's resident history, with no history."""
        self._backend = backend
        self._configured = False
        self._method = method
        stepper = backend._stepper
        problem = backend._problem
        # Whether the run interpolates its planes (ADR-036, ADR-039).
        self._interpolate = bool(stepper.interpolate_planes)
        exponential_form = method.formulation == "exponential"
        # The interaction formulation takes the fused path.
        self._fused = not exponential_form
        resident = engine is None
        self._engine: Any = (
            backend._adams_engine(method, "transported") if engine is None else engine
        )
        self._history: list[_Derivative] = []
        self._rejections = 0
        self._last_rejected = False
        self._order = 1 if method.variable_order else method.order
        self.ledger = CallLedger()
        # An in-stage temporal collar's largest rate bounds each order's stable step
        # in the exponential formulation (`adams_core`'s order choice).
        responses = problem.responses
        collar = responses.temporal_absorber
        collar_rate = (
            0.5 * float(np.max(collar))
            if collar is not None
            and responses.absorber_scheme == "nonlinear"
            and exponential_form
            else 0.0
        )
        linear = problem.linear
        carrier = getattr(self._engine, "carrier", None)
        # Every number the native controller and landings read (SEP 39 section 4.2).
        self._policy: MarchPolicy = march_policy(
            method,
            dz_min=stepper.dz_min,
            dz_max=stepper.dz_max,
            stretch=stepper.stretch,
            z_end=stepper.z_end,
            save_planes=np.asarray(stepper.save_planes, dtype=np.float64).tolist(),
            hard_kinks=np.asarray(linear.hard_kinks, dtype=np.float64).tolist(),
            breakpoints=np.asarray(linear.phi_breakpoints, dtype=np.float64).tolist(),
            interpolate=self._interpolate,
            collar_rate=collar_rate,
            carrier=isinstance(carrier, CarrierReference) and carrier.active,
        )
        # Whether the stages left a quiet collar out at the last attempt.
        self._collar_was_quiet = False
        # The admission refuses a carrier frame on an operator with no chord
        # (``refuse_carrier_frame``); a scripted harness keeps its own engine.
        assert not resident or (
            isinstance(self._engine, _ResidentAdams)
            and not (
                self._policy.carrier
                and self._engine.carrier is not None
                and self._engine.carrier.chord is None
            )
        )

    # -- the record ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the history: the state was loaded outside the march."""
        self._history = []
        self._rejections = 0
        self._engine.reset()

    def begin_run(self) -> None:
        """Start a run's accounting afresh: the ledger and the last attempt's flags.

        A backend loaded for another run counts that run's calls alone and chooses its
        first order and step as a fresh backend would (SEP 22 section 4.7, SEP 39 R9).
        A resume restores its persisted ledger and flags afterwards.
        """
        self.ledger = CallLedger()
        self._last_rejected = False
        self._collar_was_quiet = False

    def restore_ledger(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore a resumed run's logical call counts, not its history.

        - ``logical``: the persisted logical counts; empty installs nothing.
        - ``atomic``: persisted atomic counts, which only the TDSE reference keeps.
        """
        if atomic:
            raise BackendError("native does not restore atomic response call ledgers")
        if logical:
            self.ledger = CallLedger.from_record(logical)

    def resume(self, snapshot: MarchSnapshot, z: float) -> str | None:
        """Install a snapshot's history taken where the run resumes, at `z`.

        The history's rows go to the resident engine and its entries, order and flags
        to this record; the snapshot's state and proposal are the caller's to load. A
        snapshot that cannot continue the march changes nothing.

        - ``z``: the loaded position, m.
        - Returns: ``None`` when the history was installed, otherwise why the march
          starts afresh, as a resume did before snapshots were kept.
        """
        reason = self._refusal(snapshot, z)
        if reason is not None:
            return reason
        # The rows keep the engine's layout, a leading axis of one beside their own
        # (the snapshot's disk format, SEP 47 section 6.7).
        try:
            self._engine.load_rows(snapshot.rows, (1, *self._backend._problem.shape))
        except BackendError as error:
            return str(error)
        self._history = [
            _Derivative(float(z), self._engine.value(int(slot)), identity, int(slot))
            for z, identity, slot in zip(
                snapshot.positions, snapshot.identities, snapshot.slots, strict=True
            )
        ]
        self._order = snapshot.order
        self._rejections = snapshot.rejections
        self._last_rejected = snapshot.last_rejected
        self._collar_was_quiet = snapshot.collar_was_quiet
        return None

    def _refusal(self, snapshot: MarchSnapshot, z: float) -> str | None:
        """Return why a snapshot cannot continue this march at `z`, or ``None``."""
        shape = tuple(self._backend._problem.shape)
        checks = (
            (snapshot.history_key is None, "the snapshot names no history key"),
            (
                snapshot.history_key != self._method.history_key,
                f"the snapshot's history was kept by {snapshot.history_key}, and "
                f"this run steps with {self._method.history_key}",
            ),
            (
                snapshot.z != z,
                f"the snapshot is at z = {snapshot.z!r} m and the run resumes at "
                f"z = {z!r} m",
            ),
            (
                snapshot.state.shape != shape,
                f"the snapshot's state has shape {snapshot.state.shape}",
            ),
            (
                # A landing snaps to its plane, a rounding from the step's z + dz.
                not math.isclose(
                    float(snapshot.positions[-1]), snapshot.z, rel_tol=1e-12
                ),
                "the snapshot's newest sample is not at its position",
            ),
            (
                bool(
                    np.any(
                        (snapshot.slots < 0) | (snapshot.slots >= self._engine.n_slots)
                    )
                ),
                "the snapshot's history names no row of this engine",
            ),
        )
        return next((reason for failed, reason in checks if failed), None)

    def start_state(self, z: float, proposal: float, cursor: int) -> dict[str, object]:
        """Return the march's state at a start, as the native call takes it.

        - ``z``: the loaded position, m.
        - ``proposal``: the next step's proposal, m: ``dz_init``, or a resumed
          snapshot's.
        - Returns: the position, the proposal, the order, the rejection flags, the
          cursor, the history's entries and the ledger's counts.
        """
        ledger = self.ledger
        return {
            "z": float(z),
            "proposal": float(proposal),
            "order": self._order,
            "rejections": self._rejections,
            "last_rejected": self._last_rejected,
            "collar_was_quiet": self._collar_was_quiet,
            "cursor": int(cursor),
            "history": [
                (entry.z, entry.identity == "predicted", entry.slot)
                for entry in self._history
            ],
            "ledger": {name: getattr(ledger, name) for name in _MARCH_COUNTERS},
        }

    def _configure(self) -> None:
        """Hand the native state the run's policy and a replay's schedule, once.

        The policy carries no quiet interval: a session's native monitor holds its
        own (SEP 47 section 6.4).
        """
        if self._configured:
            return
        policy = dataclasses.replace(self._policy, quiet_interval=None)
        backend = self._backend
        engine = cast("_ResidentAdams", self._engine)
        schedule = backend._stepper.schedule
        steps = edges = _NO_SCHEDULE
        if schedule is not None:
            steps = np.ascontiguousarray(schedule.steps, dtype=np.float64)
            edges = np.ascontiguousarray(schedule.edges, dtype=np.float64)
        backend._native_state.adams_march_configure(
            dataclasses.asdict(policy),
            self._fused,
            engine.compensated,
            engine.interpolate,
            steps,
            edges,
        )
        self._configured = True

    def _raise(self, reason: int, result: dict[str, Any]) -> None:
        """Raise the error a native call returned with, if any.

        A session forms the call budget's own error (``NativeSession._budget_error``)
        before it asks here.
        """
        if reason == _FLOOR:
            proposed = float(result["floor_proposal"])
            msg = (
                f"step control asked for {proposed:.3e} m, below the floor dz_min ="
                f" {self._policy.dz_min:.3e} m; the tolerance cannot be met here"
            )
            raise ConvergenceError(msg)
        if reason == _DIVERGED:
            dz, end = float(result["replay_dz"]), float(result["replay_end"])
            norm = float(result["replay_norm"])
            msg = (
                f"{self._backend.name}: a replayed {self._method.name} step of "
                f"{dz:.4g} m ending at z = {end:.6g} m has a non-finite error norm "
                f"({norm!r}): the field has diverged. A replayed schedule takes "
                "its steps as given and cannot reject one, so a step beyond an "
                "explicit stability limit -- the in-stage temporal collar's or a "
                "coupled flow's advisory step -- grows until it overflows; the "
                "translation refuses a declared step past the collar's limit and "
                "warns on one past the flow's. Take shorter steps, or let the "
                "controller choose them"
            )
            raise ConvergenceError(msg)
        if reason == _ERROR:
            _raise_native(result)


def _raise_native(result: Mapping[str, Any]) -> None:
    """Raise a native march's error as the exception class its category names."""
    message = str(result["message"])
    category = int(result["category"])
    if category == _CONVERGENCE:
        raise ConvergenceError(message)
    if category == _VALUE:
        raise ValueError(message)
    if category == _REFUSED:
        raise BackendError(message)
    raise RuntimeError(message)


class _ResidentLowStorage:
    """solprop's low-storage RK4IP engine on the native state (SEP 24 section 6.7).

    A ``solprop.backends.low_storage.LowStorageEngine``: the native state is the
    register ``A`` and holds ``B``, ``C`` (unless elided, solprop's Q4) and the flows;
    a step is one native call making the four register passes in the route's own
    scratch (the transverse scratch, or the modal fine spectrum) through the stage core
    the Runge-Kutta march shares, accepted in place. Only the step's host scalars
    cross: one bundle per position (the start, the midpoint, the end) and the flows'
    intervals.

    **The flows.** A constant operator tabulating holds one volume, ``exp(h_t D / 2)``
    for the key ``h_t`` of solprop's ``table_steps`` (the step's run's mean, SEP 24
    Q10), built from the operator's origin as solprop's engines build it and rebuilt
    only when the key changes; a varying operator tabulating holds the step's ``P``
    and ``Q``; recomputing holds none and forms each flow per element in the finish
    that applies it -- on a modal route from the phase terms, so a varying operator's
    flows are never transients outside the ledger.

    **The stiffness indicator** (SEP 24 Q11) is not formed:
    ``stiffness`` is NaN. Recovering ``g1`` and ``F2`` needs ``C``, which is elided on
    the Cartesian and line routes the march is for or split across the exchange's two
    evaluations, and two more reductions a step.

    Attributes
    ----------
    stiffness : float
        Always NaN: not formed on this backend.
    atomic : bool
        Always ``True``: the step is one native call, during which no signal handler
        runs, so solprop's march commits the step, the finish and the move together.
    """

    def __init__(self, backend: NativeBackend, method: LowStorageRungeKutta) -> None:
        """Hold the backend and the schedule's table keys."""
        self._backend = backend
        self._method = method
        self.stiffness = math.nan
        self.atomic = True
        problem = backend._problem  # pyright: ignore[reportPrivateUsage] - the owner's problem
        stepper = backend._stepper  # pyright: ignore[reportPrivateUsage] - the owner's policy
        payload = _low_storage_payload(problem, stepper, device=_on_device())
        self._flows = int(cast(int, payload["low_storage_flows"]))
        schedule = stepper.schedule
        assert schedule is not None  # the translation refused the march without one
        self._keys, self._edges = table_steps(schedule), schedule.edges
        self._table_step = math.nan
        self._separable = backend._terms.separable is not None  # pyright: ignore[reportPrivateUsage] - the owner's fixed terms

    def reset(self) -> None:
        """Drop the table's key: a resume builds it again from its own step."""
        self._table_step = math.nan

    def close(self) -> None:
        """Forget the table; the native state releases the views when it closes."""
        self.reset()

    def step(self, z: float, dz: float) -> float:
        """Take one step in four native register passes, accepted in place.

        - ``z``: where the step starts, m.
        - ``dz``: the step, m, positive.
        """
        backend = self._backend
        if (counters := _counters.active) is not None:
            counters.low_storage_steps += 1
        midpoint, end = z + 0.5 * dz, z + dz
        # The engine reads its owner's tables, as solprop's engines read theirs.
        hoist = backend._hoist  # pyright: ignore[reportPrivateUsage]
        bundles_at = backend._stage_bundles  # pyright: ignore[reportPrivateUsage]
        # A z table's increments are its slots' (solprop SEP 29 G1).
        increment = backend._phi_increment  # pyright: ignore[reportPrivateUsage]
        problem = backend._problem  # pyright: ignore[reportPrivateUsage]
        hoist(z)
        # Each evaluation reads its position as its own anchor (solprop's engines);
        # a modal flow formed for the step reads its interval.
        modal = not self._separable and self._flows != 1
        bundles = [
            bundles_at(z, [z, midpoint] if modal else [z])[0],
            bundles_at(midpoint, [midpoint, end] if modal else [midpoint])[0],
            bundles_at(end, [end])[0],
        ]
        stages = backend._stepper.tableau.n_stages  # pyright: ignore[reportPrivateUsage]
        padded = bundles + [bundles[-1]] * (stages - len(bundles))
        arrays, material = backend._trial_arrays(padded, bundles[-1])  # pyright: ignore[reportPrivateUsage]
        phi_delta = np.zeros_like(arrays[0])
        rebuild = False
        key = math.nan
        if self._flows == 1:
            key = schedule_key(self._keys, self._edges, z, dz)
            rebuild = key != self._table_step
        if self._separable:
            # A constant operator's increment is the distance itself.
            phi_delta[0:2, 0] = 0.5 * dz
            if rebuild:
                phi_delta[2, 0] = 0.5 * key
        else:
            if modal:
                phi_delta[0] = increment(z, midpoint)
                phi_delta[1] = increment(midpoint, end)
            if rebuild:
                # From the operator's origin, not fl(z + h/2) - z (solprop's engines).
                origin = float(problem.linear.z_span[0])
                half = origin + 0.5 * key
                bundles_at(origin, [half])
                phi_delta[2] = increment(origin, half)
        with _envelope_errors():
            squared = backend._native_state.low_storage_step(  # pyright: ignore[reportPrivateUsage]
                dz,
                np.ascontiguousarray(phi_delta),
                *arrays[1:],
                material,
                rebuild,
            )
        if rebuild:
            self._table_step = key
        return float(squared)


_TABLE_POINTS = 16
"""The most positions one native table evaluation takes (``z_table::max_points``)."""


class _ZTables:
    """The general z tables of one native state (solprop SEP 29 section 6.6, G1).

    solprop falls back to a Chebyshev table of the state's size for ``Phi``
    (``linear.phi_coefficients``) or the response prefactor (``responses.prefactor``)
    when neither is a weighted sum of fixed arrays. The coefficients cross once, at
    construction (``table_upload``), and stay resident; a trial's positions are then
    evaluated natively into the native state's term slots, ``slots`` of them, and the
    stage weights select them: ``Phi(z1) - Phi(z0)`` is the row ``e_1 - e_0`` and
    ``Phi(z1)`` the row ``e_1``, so every native consumer of the terms (the Runge-Kutta
    stages, the low-storage flows, the Adams phase and the noise) reads the table
    unchanged, as one rounded subtraction of two values, as solprop's reference forms
    it. The prefactor's slots are its values at the same positions, one-hot weighted.

    The slots are kept as a least-recently-used set of positions, in the native state
    (``stage::ZSlots``, solprop SEP 39 phase 4), where the native Adams march reads
    them too: each call's positions are evaluated only where no slot holds them yet,
    so an accepted step's end, which is the next step's anchor, is never evaluated
    twice. A call's positions
    number at most ``slots`` (the tableau's distinct stage fractions, five for the
    low-storage march), so none of one call's positions is evicted by another of its
    own. A compensated Adams march reads ``D = dPhi/dz`` from the derivative table
    (``linear.rate``) into one more ``Phi`` slot, ``rate_slot``.

    Attributes
    ----------
    phi, prefactor : bool
        Which quantities are z tables.
    slots : int
        The position slots, each one value of each table.
    rate_slot : int or None
        The ``Phi`` term that holds the rate, or ``None`` when no march reads it.
    """

    def __init__(self, problem: PropagationProblem, stepper: StepperSettings) -> None:
        """Size the slots from the problem and its method; nothing crosses yet."""
        self._problem = problem
        self.phi = problem.linear.phi_representation == "table"
        self.prefactor = problem.responses.prefactor_representation == "table"
        integrator = stepper.integrator
        self.slots = z_table_slots(stepper)
        if self.slots > _TABLE_POINTS:
            raise BackendError(
                f"native evaluates a z table at {_TABLE_POINTS} positions a call, and "
                f"this tableau's stages need {self.slots}"
            )
        compensated = (
            self.phi
            and isinstance(integrator, Multistep)
            and integrator.formulation == "exponential"
            and not source_absent(problem)
        )
        self.rate_slot = self.slots if compensated else None
        self._native: _core.State | None = None

    @property
    def evaluations(self) -> int:
        """Positions evaluated so far, a count for the tests and the benchmarks."""
        return 0 if self._native is None else self._native.table_evaluations()

    @property
    def n_phi(self) -> int:
        """The native ``Phi`` term count: the slots, and the rate's when it is read."""
        return self.slots + (self.rate_slot is not None)

    @property
    def nbytes(self) -> int:
        """The bytes of the coefficients that cross to the native state."""
        coefficients = 0
        if self.phi:
            phi = self._problem.linear.phi_coefficients
            assert phi is not None
            coefficients += phi.nbytes
            if self.rate_slot is not None:
                coefficients += self._problem.linear.rate.coefficients.nbytes
        if self.prefactor:
            table = self._problem.responses.prefactor
            assert table is not None
            coefficients += table.coefficients.nbytes
        return int(coefficients)

    def upload(self, native: _core.State) -> None:
        """Make the tables resident in `native`, once, before its first trial."""
        self._native = native
        native.table_slots(
            self.phi, self.prefactor, self.slots, self.rate_slot is not None
        )
        if self.phi:
            self._upload("phi", self._problem.linear.polynomial)
            if self.rate_slot is not None:
                self._upload("rate", self._problem.linear.rate)
        if self.prefactor:
            table = self._problem.responses.prefactor
            assert table is not None
            self._upload("prefactor", table)

    def _upload(self, kind: str, table: PiecewiseChebyshev) -> None:
        """Send one table's breakpoints and coefficients, and its admitted span."""
        assert self._native is not None
        low, high = table.z_span
        slack = _EDGE_SLACK_FRACTION * (high - low)
        n_intervals, n_terms = table.coefficients.shape[:2]
        self._native.table_upload(
            kind,
            np.ascontiguousarray(table.breakpoints, dtype=np.float64),
            low - slack,
            high + slack,
            np.ascontiguousarray(table.coefficients, dtype=np.complex128).reshape(
                n_intervals, n_terms, -1
            ),
        )

    def load(self, positions: Sequence[float]) -> dict[float, int]:
        """Return each position's slot, evaluating those no slot holds, in one call.

        - ``positions``: the positions one native call reads, m, at most ``slots``
          distinct.
        """
        assert self._native is not None
        wanted = [float(z) for z in positions]
        try:
            slots = self._native.table_load(wanted)
        except ValueError as error:
            raise BackendError(str(error)) from error
        return dict(zip(wanted, slots, strict=True))

    def phi_row(self, slot_of: Mapping[float, int], z0: float, z1: float) -> RealArray:
        """Return the weights of ``Phi(z1) - Phi(z0)`` over the native terms."""
        row = np.zeros(self.n_phi)
        row[slot_of[z1]] += 1.0
        row[slot_of[z0]] -= 1.0
        return row

    def at_row(self, slot_of: Mapping[float, int], z: float, count: int) -> RealArray:
        """Return the one-hot weights of a table's value at `z` over `count` terms."""
        row = np.zeros(count)
        row[slot_of[z]] = 1.0
        return row

    def rate(self, z: float, *, before: bool) -> RealArray:
        """Evaluate ``D(z)`` into the rate slot; return its weights over the terms.

        - ``z``: position, m.
        - ``before``: read an interior breakpoint from the interval ending there.
        - Returns: shape ``(n_phi,)``, one at the rate slot.
        """
        assert self._native is not None
        assert self.rate_slot is not None
        self._native.table_rate_at(float(z), before)
        row = np.zeros(self.n_phi)
        row[self.rate_slot] = 1.0
        return row

    def values(self, kind: Literal["phi", "prefactor"], z: float) -> ComplexArray:
        """Return a table's value at `z` on the host, from its native slot."""
        assert self._native is not None
        slot = self.load([z])[float(z)]
        return self._native.table_terms_to_host(kind, slot)


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


def _refuse_adams(problem: PropagationProblem, stepper: StepperSettings) -> None:
    integrator = stepper.integrator
    # The low-storage march (solprop SEP 24 phase 3) runs on every route the
    # translation admits it on: no coupled flow and no TDSE response.
    if integrator is None or isinstance(integrator, LowStorageRungeKutta):
        return
    # solprop SEP 16 phase 3: every route's stage source serves both formulations,
    # with the modified noise. A coupled flow is stepped by the exponential
    # formulation only (phase 5), as solprop's reference does.
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        raise BackendError(
            "solprop-native does not step a problem with a coupled linear flow with "
            f"the {integrator.name} integrator; use Multistep(formulation="
            "'exponential') or the Runge-Kutta march"
        )
    refuse_formulation(problem, stepper, "native")
    # SEP 35 phase 3: the scalar shift on CPU builds; device builds wait for their
    # qualification.
    device = _on_device()
    refuse_carrier_frame(
        problem,
        stepper,
        "native (device build)" if device else "native",
        delivered=frozenset() if device else frozenset({"scalar"}),
    )


def _refuse_omitted_stages(responses: CompiledResponses) -> None:
    """Refuse a ``response_order`` that leaves out a response the problem holds.

    Native takes each response from its own leaves and doesn't read the order, so an
    omitted stage would still run. The translation never produces this; a hand-built
    problem can, and ``numpy_ref``, which reads the order, would run it without.
    """
    envelope = responses.envelope_plasma is not None
    held = {
        ResponseStage.KERR: responses.kerr_coefficients is not None,
        ResponseStage.DISPERSIVE_KERR: responses.dispersive_kerr is not None,
        ResponseStage.RAMAN_DRIVE: bool(responses.raman_terms),
        ResponseStage.IONISATION_RATE: bool(responses.ionisation_table)
        and not envelope,
        ResponseStage.PLASMA_SCAN: bool(responses.plasma_channels) and not envelope,
        ResponseStage.PLASMA_CURRENT: bool(responses.plasma_channels) and not envelope,
        ResponseStage.ENVELOPE_POPULATION: envelope,
    }
    omitted = [
        stage.value
        for stage, present in held.items()
        if present and stage not in responses.response_order
    ]
    if omitted:
        raise BackendError(
            f"response_order omits {', '.join(omitted)}, whose responses the problem "
            "holds; solprop-native takes each response from its own leaves and would "
            "run it anyway. Remove the response too, or use backend='numpy_ref', which "
            "follows the order"
        )


def _refuse_unsupported(problem: PropagationProblem, stepper: StepperSettings) -> None:
    _refuse_unknown_method(stepper)
    _refuse_omitted_stages(problem.responses)
    if problem.precision.single:
        raise BackendError(
            "solprop-native computes in binary64 only and refuses precision='single' "
            "(solprop SEP 33); use backend='numpy_ref', the single-precision oracle"
        )
    # The route by its variant (solprop SEP 47 section 6.2), refused by name before
    # anything is allocated: every route of the union but the modal route's TDSE
    # extension, which is the reference's alone.
    # Read as any object: a variant this build does not know is refused by name too.
    route = cast("object", problem.route)
    match route:
        case ModalRoute(tdse=tdse) if tdse is not None:
            raise BackendError(
                "solprop-native does not execute TDSEGasResponse (solprop SEP 12); "
                "use backend='numpy_ref' with the independent solprop-tdse atomic "
                "engine"
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
    # Schema 16's low-call Adams integrator (solprop SEP 13) runs here on the modal
    # fibre route (SEP 14 phase 2), its source evaluated by the general engine's stage
    # pipeline; elsewhere it is refused by name rather than stepped by this backend's
    # Runge-Kutta march, which would silently be a different method.
    _refuse_adams(problem, stepper)
    # One list for every backend: the general modal route and the coupled flow, each
    # refusal naming this backend.
    # A moving guide's synthesis and projection, combined per stage over schema 14's
    # `StageScalars.transform_weights`, run here beside the frozen-residual flow that
    # carries their connection.
    refuse_general_route(problem, backend="native", moving_maps=True)
    _refuse_native_route(problem)
    flags = problem.flags
    if problem.transform.n_pol * problem.shape[1] > 2:
        raise BackendError(
            "the polarisation components must belong to one of the two supported "
            "representations; Kokkos supports at most two effective node "
            "components, but the transform and state request "
            f"{problem.transform.n_pol * problem.shape[1]}"
        )
    if (
        flags.has_raman
        and problem.transform.n_pol * problem.shape[1] != 1
        and (
            not problem.responses.raman_terms
            or not all(term.tensor_capable for term in problem.responses.raman_terms)
        )
    ):
        raise BackendError(
            "vector Raman requires tensor-capable terms; found scalar_only"
        )


def _refuse_native_route(problem: PropagationProblem) -> None:
    """Refuse what solprop's shared list admits and this backend has not built.

    The general modal route and the coupled flow are admitted by solprop's one list,
    :func:`solprop.backends.numpy_ref.admission.refuse_general_route`, run under this
    backend's name before anything is allocated. What remains
    here is this backend's own: the native delayed response contracts the six pair
    histories of three Cartesian components, which the reference's arithmetic does
    not require of a record.
    """
    tables = problem.transform.modal_tables
    regional = problem.responses.regional_responses
    if tables is None or regional is None or not regional.n_terms:
        return
    components = int(tables.electric.shape[2])
    if regional.n_terms != DELAYED_TERMS or components != 3:
        raise BackendError(
            f"native executes the {DELAYED_TERMS} delayed channels of the silica "
            "tensor Raman model on three components, and "
            f"responses.regional_responses carries {regional.n_terms} term(s) on "
            f"{components}"
        )


@dataclasses.dataclass(frozen=True)
class _HostFlow:
    """A coupled flow's Schur factors per section on its active bins, host-side.

    Formed as solprop's reference backend forms its own (``_coupled_flow_maps``): the
    frozen block of each section read out of the leaf, decomposed per active bin, the
    declared conditioning row, and a frozen-residual flow's terms; with each
    section's eigenbasis of its triangular factor (solprop's
    ``schur_eigenbasis``), whose route marks the bins formed from it, and the largest
    row-sum norm of the factor, which the squaring refusal reads. The eigenbasis
    crosses to the native engine once, with the Schur factors of the bins it does not
    cover; the engine forms the actions per section and step length. ``scored``
    marks, per active bin, the requested band (``grids.inband_mask``): the
    explicit residual is applied there only, as solprop's reference applies it
    (``residual_bins``).
    """

    active: npt.NDArray[np.bool_]
    scored: npt.NDArray[np.bool_]
    vectors: ComplexArray
    triangular: ComplexArray
    frozen: ComplexArray
    condition_bound: RealArray
    residual_terms: ComplexArray | None
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
    generator = flow.generator
    assert generator is not None
    sections = len(flow.section_methods)
    active = np.asarray(np.any(generator != 0.0, axis=(0, 1, 2, 3)), dtype=np.bool_)
    frozen = np.stack(
        [
            np.moveaxis(frozen_generator(flow, index)[:, :, active], -1, 0)
            for index in range(sections)
        ]
    )
    factors = [schur_blocks(np.moveaxis(block, 0, -1)) for block in frozen]
    bases = [schur_eigenbasis(*pair) for pair in factors]
    residual: ComplexArray | None = None
    if flow.section_methods != ("constant_schur",):
        terms = generator[:, :, :, :, active]
        if flow.connection is not None:
            terms = terms - flow.connection[:, :, :, :, active]
        residual = np.ascontiguousarray(
            np.moveaxis(terms, (2, 3), (-2, -1)).reshape(
                sections * flow.n_deg, -1, flow.n_modes, flow.n_modes
            ),
            dtype=np.complex128,
        )
    return _HostFlow(
        active=active,
        scored=np.ascontiguousarray(problem.grids.inband_mask[active], dtype=np.bool_),
        vectors=np.stack([pair[0] for pair in factors]),
        triangular=np.stack([pair[1] for pair in factors]),
        frozen=np.ascontiguousarray(frozen, dtype=np.complex128),
        condition_bound=np.ascontiguousarray(
            flow.inverse_condition_bound[:, active], dtype=np.float64
        ),
        residual_terms=residual,
        left=np.ascontiguousarray(np.stack([basis[0] for basis in bases])),
        right=np.ascontiguousarray(np.stack([basis[1] for basis in bases])),
        eigenvalues=np.ascontiguousarray(np.stack([basis[2] for basis in bases])),
        route=np.stack([basis[3] for basis in bases]),
        row_sum=np.asarray(
            [
                float(np.max(np.sum(abs(pair[1]), axis=-1), initial=0.0))
                for pair in factors
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


def _flow_payload(flow: _HostFlow | None) -> dict[str, object]:
    """Pack a coupled flow's bins and residual data; its actions follow per step."""
    if flow is None:
        return {"coupled_flow": False}
    slot = np.zeros(flow.active.size, dtype=np.uint64)
    slot[flow.active] = np.arange(1, int(np.count_nonzero(flow.active)) + 1)
    residual = flow.residual_terms
    empty = np.empty(0, dtype=np.complex128)
    # The bins formed by scaling and squaring are numbered 1, 2, ... in row-major
    # (section, bin) order and carry their Schur factors; the rest take the
    # eigenbasis, whose `W` and `Y` are zero where it is not used.
    fallback = ~flow.route
    route = np.zeros(flow.route.shape, dtype=np.uint64)
    route[fallback] = np.arange(1, int(np.count_nonzero(fallback)) + 1)
    # The residual's data are zero on the bins the error norm does not score, so the
    # engine's residual there is an exact zero and the frozen flow alone carries them,
    # as in solprop's reference. Every term is finite, so each
    # product with a zero is a signed zero and the running sums stay +0: the source is
    # what adding nothing leaves, but for the sign of a zero.
    scored = flow.scored[None, :, None, None]
    return {
        "coupled_flow": True,
        "flow_active": int(np.count_nonzero(flow.active)),
        "flow_sections": int(flow.frozen.shape[0]),
        "flow_terms": 0 if residual is None else int(residual.shape[0]),
        "flow_slot": slot,
        "flow_frozen": empty
        if residual is None
        else np.ascontiguousarray(np.where(scored, flow.frozen, 0.0)),
        "flow_residual_terms": empty
        if residual is None
        else np.ascontiguousarray(np.where(scored, residual, 0.0)),
        "flow_left": flow.left,
        "flow_right": flow.right,
        "flow_eigenvalues": flow.eigenvalues,
        "flow_route": route.reshape(-1),
        "flow_fallback_vectors": np.ascontiguousarray(flow.vectors[fallback]),
        "flow_fallback_triangular": np.ascontiguousarray(flow.triangular[fallback]),
    }


_APERTURE_CONSTANTS = 11
"""The constants before an aperture row's scale row (``far_field.hpp``)."""


def _aperture_payload(
    tables: ApertureTables | None, n_omega: int
) -> tuple[RealArray, RealArray, int]:
    """Pack far-field tables as the native pass reads them (solprop SEP 23 section 6).

    Returns the ``(n_apertures, 11 + n_omega)`` aperture rows (centre x and y, the
    half-width and half-height, the two stretches, the reciprocal edge width, the edge
    or order, the profile and shape codes, the distance, then the scale row
    ``z c / omega`` that every backend reads to the bit), the ``(n_regions, 4)`` boxes
    and the exponent; empty tables without any.
    """
    if tables is None:
        return np.empty((0, _APERTURE_CONSTANTS + n_omega)), np.empty((0, 4)), 2
    assert tables.half_heights is not None
    assert tables.stretches is not None
    constants = np.column_stack(
        (
            tables.centres,
            tables.half_widths,
            tables.half_heights,
            tables.stretches,
            tables.inverse_widths,
            tables.edges,
            tables.profile_codes,
            tables.shape_codes,
            tables.distances,
        )
    )
    apertures = np.ascontiguousarray(np.hstack((constants, tables.scale)))
    return (
        apertures,
        np.ascontiguousarray(tables.regions, dtype=np.float64),
        tables.exponent,
    )


def _refined_record(
    problem: PropagationProblem, stepper: StepperSettings, state: _core.State
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
        sizing=state.execution_sizing,
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


def _planning_payload(
    given: Mapping[str, object] | None, stored: Mapping[str, str]
) -> dict[str, object]:
    """Return the problem's ``fft_planning``: the one given, or the run's own.

    A run's own is the profile's, or the plan policy's, offered the plans runs measured
    before (`stored`) and keeping those it measures (solprop ADR-076); one given, a
    calibration's, plans on its own.
    """
    if _on_device():
        return {}
    if given is not None:
        return {"fft_planning": dict(given)}
    chosen = _package.PLANNING or {"mode": "legacy"}
    return {"fft_planning": dict(chosen) | {"stored": dict(stored), "keep": True}}


def _keep_plans(state: _core.State, path: Path | None) -> dict[str, str]:
    """Keep the plans a workspace measured, in this process and in `path`; return them.

    A key the profile holds is never kept: its winner takes precedence.
    """
    if _on_device():
        return {}
    measured = cast(dict[str, str], state.fft_planning["measured"])
    profiled = cast(Mapping[str, str], (_package.PLANNING or {}).get("wisdom", {}))
    kept = {key: plan for key, plan in measured.items() if key not in profiled}
    if kept:
        _wisdom.remember(kept)
        _wisdom.store(path, kept)
    return kept


def _applied_plans(
    state: _core.State,
    stored: Mapping[str, str],
    planning: Mapping[str, object] | None,
) -> dict[str, str]:
    """Return the wisdom of each plan a workspace built from wisdom or measured.

    `planning` is the one it was given, or the profile's; `stored` the plans offered.
    """
    found = state.fft_planning
    profiled = cast(Mapping[str, str], (planning or {}).get("wisdom", {}))
    missing = set(cast(list[str], found["missing"]))
    applied = {
        key: profiled[key]
        for key in cast(dict[str, str], found["groups"])
        if key in profiled and key not in missing
    }
    applied |= {key: stored[key] for key in cast(list[str], found["reused"])}
    return applied | cast(dict[str, str], found["measured"])


def _planning_record(state: _core.State) -> dict[str, object]:
    """Return what a workspace's FFT planning applied, for its run's metadata.

    The plans keyed in its profile that no longer build from their wisdom are listed:
    they ran at estimate (solprop SEP 46 section 6.7).
    """
    if _on_device():
        return {}
    found = state.fft_planning
    groups = cast(dict[str, str], found["groups"])
    if not groups:
        return {"fft_planning": "policy"}
    return {
        "fft_planning": {
            "plans": len(groups),
            "missing": list(cast(list[str], found["missing"])),
        }
    }


def _native_problem(
    problem: PropagationProblem,
    request: RunRequest,
    fixed: StageTerms,
    flow: _HostFlow | None = None,
    tables: _ZTables | None = None,
    *,
    node_chunk: int | None = None,
    general_chunk: int | None = None,
) -> dict[str, object]:
    stepper = request.stepper
    # The constructor hands its lowered chunks; a direct caller takes the automatic
    # plan's (solprop SEP 47 section 6.5).
    if node_chunk is None:
        node_chunk = native_node_chunk(
            problem, plan_node_chunk(problem), device=_on_device()
        )
    if general_chunk is None and problem.transform.modal_tables is not None:
        general_chunk = general_planned_chunk(problem, plan_node_chunk(problem))
    kerr = problem.responses.kerr_coefficients
    dispersive = problem.responses.dispersive_kerr
    # solprop SEP 6's number-conserving sector is formed on analytic signals, of the
    # filtered fields and, for the polynomial, of the bare one; a Raman drive beside it
    # is the cycle-averaged one, which reads the analytic signal too.
    # The general modal route synthesises its own analytic signal on the real family.
    family = (
        "envelope"
        if problem.grid_kind == "envelope"
        else "field_analytic"
        if (kerr is not None and kerr.analytic_signal)
        or (
            dispersive is not None
            and dispersive.sector == "conserving"
            and problem.transform.modal_tables is None
        )
        else "field_real"
    )
    absorber = problem.responses.temporal_absorber
    absorber_fine = problem.responses.temporal_absorber_fine
    spatial_absorber = radial_absorber_matrix(
        problem.transform, problem.grids.pad_scale
    )
    temporal_in_stage = (
        absorber_fine is not None and problem.responses.absorber_scheme == "nonlinear"
    )
    carrier = problem.grids.carrier_2w0
    n_omega = problem.shape[-1]
    phi_terms = fixed.phi
    prefactor_terms = fixed.prefactor
    spatial = fixed.separable
    if spatial is not None:
        phi = np.empty((1, 0, n_omega), dtype=np.complex128)
        prefactor = np.empty((1, 0, n_omega), dtype=np.complex128)
    else:
        # A z table's terms are slots the native evaluation fills (solprop SEP 29 G1):
        # an empty array whose leading extent is their count crosses in their place.
        if tables is not None and tables.phi:
            phi = np.empty((tables.n_phi, 0), dtype=np.complex128)
        elif phi_terms is None:
            raise BackendError("Kokkos requires a device-ready fixed-term problem")
        else:
            phi = np.ascontiguousarray(
                phi_terms.reshape(phi_terms.shape[0], -1, n_omega)
            )
        if tables is not None and tables.prefactor:
            prefactor = np.empty((tables.slots, 0), dtype=np.complex128)
        elif prefactor_terms is None:
            raise BackendError("Kokkos requires a device-ready fixed-term problem")
        else:
            prefactor = np.ascontiguousarray(
                prefactor_terms.reshape(prefactor_terms.shape[0], -1, n_omega)
            )
    channels = problem.responses.population_channels
    raman_terms = problem.responses.raman_terms
    plasma = channels[0] if channels else None
    raman = raman_terms[0] if raman_terms else None
    rate = None if plasma is None else plasma.rate
    coarse = (
        np.empty(0, dtype=np.float64)
        if problem.transform.n_nodes == 1
        or np.array_equal(
            problem.transform.projection_coarse, problem.transform.projection
        )
        else problem.transform.projection_coarse
    )
    geometry = problem.transform.geometry
    # Free space and schema 15's hybrid planar kind share the separable operator and
    # the transverse FFT engine; the planar kind adds its slab rule in x.
    free = geometry is not None and geometry.kind in ("free", "slab")
    spatial_absorption = bool(spatial_absorber.size) or (
        free and geometry is not None and bool(np.any(geometry.absorber_rate))
    )
    coarse_matrix = coarse
    spatial_data: dict[str, object] = {}
    if free and geometry is not None and spatial is not None:
        spatial_data = {
            "n_x": geometry.shape[0],
            "n_y": geometry.shape[1],
            "spatial_k_squared": spatial.k_squared,
            "spatial_k_squared_low": spatial.k_squared_low,
            "spatial_kperp_sq": spatial.kperp_sq,
            "spatial_kperp_sq_low": spatial.kperp_sq_low,
            "spatial_frame": spatial.frame,
            "spatial_spectral_rate": spatial.spectral_rate,
            "spatial_user_rate": spatial.user_rate,
            "spatial_numerator": spatial.prefactor_numerator,
            "spatial_clamped": spatial.prefactor_clamped,
            "spatial_evanescent_rate": spatial.evanescent_rate,
            "spatial_absorber_rate": geometry.absorber_rate,
            # A transverse collar's rate per column (solprop kernel schema 31).
            "spatial_kperp_rate": spatial.kperp_rate,
        }
        if geometry.kind == "free":
            # solprop SEP 23: the stored axes and the slots of the problem's own
            # far-field request, which size the native band partials.
            k_x, k_y = geometry.spectral_axes
            slots, _ = far_field_partials(problem, request.observation.stats)
            spatial_data |= {
                "spatial_k_x": np.ascontiguousarray(k_x, dtype=np.float64),
                "spatial_k_y": np.ascontiguousarray(k_y, dtype=np.float64),
                "spatial_far_field_slots": slots,
            }
        if geometry.kind == "slab":
            spatial_data |= _slab_payload(problem)
            # The reference scores no quadrature alarm on this kind, so no embedded
            # Kronrod projection is packed.
            coarse_matrix = np.empty(0, dtype=np.float64)
    # The accepted step's last stage leaves the plasma statistics behind only when
    # that stage's input is the committed field: a first-same-as-last
    # tableau, no split-step absorber (applied to the field after the last stage)
    # and no coupled flow; and not under the modified noise model, where that input
    # carries the noise draw while solprop defines the statistics on the committed
    # field its reference scans. Otherwise they are scanned afresh.
    stash_plasma = (
        plasma is not None
        and problem.responses.envelope_plasma is None
        and problem.initial.noise_model != "modified"
        and stepper.tableau.first_same_as_last
        and not (
            absorber is not None and problem.responses.absorber_scheme == "split_step"
        )
        and problem.linear.coupled_flow is None
        and bool(
            {
                "electron_density",
                "peak_ionisation_rate",
                "species_ionisation",
                "species_electron_density",
            }
            & set(request.observation.stats.periodic)
        )
    )
    return {
        # solprop SEP 46: how the workspace's CPU operations thread, the process's
        # record unless the backend was given one.
        "cpu_execution": _package.EXECUTION.native(),
        **spatial_data,
        **_channel_payload(problem),
        **_envelope_payload(problem),
        **_general_payload(problem, general_chunk),
        **_flow_payload(flow),
        **_dispersive_payload(problem),
        **_flux_payload(problem, request.observation.stats),
        "n_omega": n_omega,
        "n_t": problem.grids.n_t,
        "n_to": problem.grids.n_to,
        "n_stages": stepper.tableau.n_stages,
        "n_modes": problem.shape[0],
        "n_pol_state": problem.shape[1],
        "n_pol_transform": problem.transform.n_pol,
        "n_nodes": problem.transform.n_nodes,
        # solprop SEP 47 section 6.5: the backend's execution choice, lowered by
        # `native_node_chunk`.
        "node_chunk": node_chunk,
        "family": family,
        "nonlinear": kerr is not None,
        "modified_noise": problem.initial.noise_model == "modified",
        "in_stage_absorber": temporal_in_stage or spatial_absorption,
        "split_step_absorber": absorber is not None
        and problem.responses.absorber_scheme == "split_step",
        "first_same_as_last": stepper.tableau.first_same_as_last,
        "crop_index": np.ascontiguousarray(problem.grids.crop_index, dtype=np.uint64),
        "active_mask": np.ascontiguousarray(problem.grids.active_mask),
        "norm_mask": np.ascontiguousarray(problem.grids.norm_mask),
        "omega_window": np.ascontiguousarray(problem.grids.omega_window),
        "omega": np.ascontiguousarray(problem.grids.omega),
        "time_window": np.ascontiguousarray(problem.grids.t_window_fine),
        "time_window_derivative": np.ascontiguousarray(
            problem.grids.t_window_fine_derivative
            if problem.grid_kind == "field" and problem.responses.plasma_channels
            else np.empty(0)
        ),
        "absorber_fine": np.ascontiguousarray(
            absorber_fine
            if temporal_in_stage
            else np.zeros(problem.grids.n_to)
            if spatial_absorption
            else np.empty(0),
            dtype=np.float64,
        ),
        "spatial_absorber_matrix": spatial_absorber,
        "spatial_spectral_weights": np.empty(0)
        if geometry is None
        else geometry.spectral_weights,
        "spatial_on_axis": np.empty(0, dtype=np.complex128)
        if geometry is None
        else geometry.on_axis,
        "absorber_coarse": np.ascontiguousarray(
            np.empty(0) if absorber is None else absorber, dtype=np.float64
        ),
        "carrier": np.ascontiguousarray(
            np.ones(problem.grids.n_to, dtype=np.complex128)
            if carrier is None
            else carrier
        ),
        "synthesis_matrix": np.ascontiguousarray(problem.transform.synthesis),
        "projection_matrix": np.ascontiguousarray(problem.transform.projection),
        "projection_coarse_matrix": np.ascontiguousarray(coarse_matrix),
        "phi_terms": phi,
        "prefactor_terms": prefactor,
        "phi_table": tables is not None and tables.phi,
        "prefactor_table": tables is not None and tables.prefactor,
        "power_weight": np.ascontiguousarray(
            np.empty(0) if geometry is not None else power_weight_at(problem, 0.0)
        ),
        "tableau_matrix": np.ascontiguousarray(stepper.tableau.matrix),
        "tableau_nodes": np.ascontiguousarray(stepper.tableau.nodes),
        "solution_weights": np.ascontiguousarray(stepper.tableau.solution_weights),
        "error_weights": np.ascontiguousarray(stepper.tableau.error_weights),
        "synthesis": 0.0 if free else float(problem.transform.synthesis[0, 0]),
        "projection": 0.0 if free else float(problem.transform.projection[0, 0]),
        "kerr_coefficient": 0.0 if kerr is None else kerr.coefficient,
        "self_factor": 0.0 if kerr is None else kerr.self_factor,
        "conjugate_factor": 0.0 if kerr is None else kerr.conjugate_factor,
        "harmonic_factor": 0.0 if kerr is None else kerr.harmonic_factor,
        "take_real": kerr is not None and kerr.take_real,
        "plasma": plasma is not None,
        "raman": raman is not None,
        "raman_thg": raman is not None
        and bool(raman.thg)
        and problem.grid_kind == "field",
        "raman_tensor_capable": bool(raman_terms)
        and all(term.tensor_capable for term in raman_terms),
        "raman_tensor_weights": np.ascontiguousarray(
            np.concatenate([term.tensor_weights for term in raman_terms], axis=0)
            if raman_terms
            else np.empty((0, 2), dtype=np.float64)
        ),
        "stash_plasma": stash_plasma,
        "weighted_controller": stepper.controller.weights is not None,
        "dt_fine": problem.grids.dt_fine,
        "dt": problem.grids.dt,
        "raman_fraction": 0.0 if raman is None else float(raman.fraction),
        "raman_epsilon0": EPS0 if raman is not None else 0.0,
        "raman_undamped": np.ascontiguousarray(
            np.empty((0, 2 * problem.grids.n_to), dtype=np.float64)
            if raman is None
            else np.concatenate([term.undamped for term in raman_terms], axis=0)
        ),
        "charge_mass_ratio": 0.0 if plasma is None else plasma.charge_mass_ratio,
        "ionisation_potential": 0.0 if plasma is None else plasma.ionisation_potential,
        "log_field_min": 0.0 if rate is None else rate.log_field_min,
        "d_log_field": 0.0 if rate is None else rate.d_log_field,
        "field_min": 0.0 if rate is None else rate.field_min,
        "field_max": 0.0 if rate is None else rate.field_max,
        "rate_max": 0.0 if rate is None else rate.rate_max,
        "rate_coefficients": np.ascontiguousarray(
            np.empty((4, 0), dtype=np.float64) if rate is None else rate.coefficients
        ),
        # solprop SEP 18: the running-integral rule, one per problem.
        "plasma_band": problem.responses.plasma_integral.policy == "band",
        "plasma_taps": problem.responses.plasma_integral.taps,
    }


def _dispersive_payload(problem: PropagationProblem) -> dict[str, object]:
    """Lay out solprop SEP 6's dispersive response for the native engine.

    The structure the native response takes on every route: the channels grouped as the
    leaf groups them (``DispersiveKerrTerms.groups``) -- the symmetric evaluator's one
    mixture per node region, the polynomial's species of equal leg weight and region --
    with one filtered field and one product per symmetric group, and for the
    polynomial one ``E^3`` per distinct region followed by two fields and two products
    per group; each output term's multiplier, product and factor, every multiplier
    carrying the sector's ``Pi_+`` and factor (one on a field grid's full cubic,
    ``Pi_+ / 8`` for its conserving sector, ``Pi_+ / 4`` on an envelope grid), and the
    input multipliers ``Pi_+`` on a field grid's analytic family. A mixture's rows are
    rebuilt natively at every stage whose densities change, from its channels' ``g^4``
    (``mixture_filter``'s arithmetic); the first species' stand until then. On the
    general modal route each group's key is its region and every node's region is
    packed beside them; the route's grid is a field grid.
    """
    leaf = problem.responses.dispersive_kerr
    if leaf is None:
        return {"dispersive": False}
    tables = problem.transform.modal_tables
    general = tables is not None
    kind = "field" if general else problem.grid_kind
    form = (
        ("real" if leaf.sector == "full" else "conserving")
        if kind == "field"
        else ("harmonic" if leaf.sector == "full" else "conserving")
    )
    analytic = form == "conserving" and kind == "field"
    symmetric = leaf.evaluator == "symmetric"
    n_channels = leaf.n_channels
    positive = leaf.positive.astype(np.float64)
    if kind == "envelope":
        scale = positive / 4.0
    elif analytic:
        scale = positive / 8.0
    else:
        scale = np.ones(positive.shape)
    input_scale = positive if analytic else np.ones(positive.shape)
    groups = leaf.groups()
    # A group's key is its general modal region; every other route has one, so one
    # cube is shared by every polynomial group there.
    regions = leaf.regions if general else None
    keys = [0 if regions is None else int(regions[group[0]]) for group in groups]
    distinct = list(dict.fromkeys(keys))
    cubes = [distinct.index(key) for key in keys]
    n_cubes = 0 if symmetric else len(distinct)
    mixtures = tuple(
        index for index, group in enumerate(groups) if symmetric and len(group) > 1
    )
    fields: list[RealArray] = []
    rows: list[int] = []
    channels: list[int] = []
    multipliers: list[RealArray] = []
    for index, group in enumerate(groups):
        weight = leaf.filters[group[0]]
        if symmetric:
            fields.append(weight)
            rows.append(index)
            channels.append(
                n_channels + 1 + mixtures.index(index)
                if index in mixtures
                else group[0]
            )
            multipliers.append(weight * scale)
            continue
        assert leaf.outputs is not None
        fields.extend([weight, weight * weight])
        first = n_cubes + 2 * index
        for channel in group:
            output = leaf.outputs[channel]
            rows.extend([cubes[index], first])
            channels.extend([channel, channel])
            multipliers.extend([output[0] * scale, output[1] * scale])
        rows.append(first + 1)
        channels.append(n_channels)
        multipliers.append(scale.copy())
    stacked = np.array(fields, dtype=np.float64)
    if analytic:
        stacked = stacked * leaf.positive
    constants = (
        np.zeros((n_channels, 2))
        if leaf.products is None
        else np.asarray(leaf.products, dtype=np.float64)
    )
    if mixtures:
        squares = leaf.filters * leaf.filters
        quartics = np.ascontiguousarray(squares * squares, dtype=np.float64)
    else:
        quartics = np.empty(0, dtype=np.float64)
    offsets = np.cumsum([0, *(len(group) for group in groups)])
    route: dict[str, object] = {}
    if tables is not None:
        route = {
            "dispersive_group_keys": np.asarray(keys, dtype=np.uint64),
            "dispersive_node_region": np.ascontiguousarray(
                tables.region_index, dtype=np.uint64
            ),
        }
    return {
        **route,
        "dispersive": True,
        "dispersive_symmetric": symmetric,
        "dispersive_form": {"real": 0, "conserving": 1, "harmonic": 2}[form],
        "dispersive_inputs": np.ascontiguousarray(stacked),
        "dispersive_term_multipliers": np.ascontiguousarray(
            np.array(multipliers, dtype=np.float64)
        ),
        "dispersive_term_rows": np.asarray(rows, dtype=np.uint64),
        "dispersive_term_channels": np.asarray(channels, dtype=np.uint64),
        "dispersive_coefficients": np.asarray(leaf.coefficients, dtype=np.float64),
        "dispersive_pair_constants": np.ascontiguousarray(constants),
        "dispersive_group_offsets": np.asarray(offsets, dtype=np.uint64),
        "dispersive_group_members": np.asarray(
            [channel for group in groups for channel in group], dtype=np.uint64
        ),
        "dispersive_quartics": quartics,
        "dispersive_input_scale": np.ascontiguousarray(input_scale),
        "dispersive_output_scale": np.ascontiguousarray(scale),
    }


def _flux_payload(problem: PropagationProblem, stats: StatsSpec) -> dict[str, object]:
    """Pack the plane measure of solprop SEP 34 (kernel schema 35).

    A separable run (Cartesian, line, planar) is measured by its own operator's root,
    which the native state already holds: nothing more is packed, and a plan that is
    not that operator is refused. A radial run packs the entrance's material and
    transverse squares and whether a material z table follows
    (:func:`_upload_flux_table`); the admission record is not read. ``flux_closure``
    sizes the magnetic closure's buffers for the run's own local statistics,
    `stats`.
    """
    flux = flux_of(problem.route)
    if flux is None:
        return {}
    closure = bool(set(stats.names) & LOCAL_FLUX_STATISTICS)
    if problem.linear.separable is not None:
        if flux.base is not problem.linear.separable or flux.material is not None:
            raise BackendError(
                "solprop-native measures a separable run by its own operator's root, "
                "and this plane measure is another"
            )
        return {"flux_closure": closure}
    base = flux.base
    return {
        "flux_k_squared": np.ascontiguousarray(base.k_squared, dtype=np.complex128),
        "flux_k_squared_low": np.ascontiguousarray(
            base.k_squared_low, dtype=np.complex128
        ),
        "flux_kperp_sq": np.ascontiguousarray(base.kperp_sq, dtype=np.float64),
        "flux_kperp_sq_low": np.ascontiguousarray(base.kperp_sq_low, dtype=np.float64),
        "flux_table": flux.material is not None,
        "flux_closure": closure,
    }


def _upload_flux_table(native: _core.State, problem: PropagationProblem) -> None:
    """Make a radial plane measure's material z table resident, once (SEP 34)."""
    flux = flux_of(problem.route)
    if flux is None or flux.material is None:
        return
    table = flux.material
    low, high = table.z_span
    slack = _EDGE_SLACK_FRACTION * (high - low)
    n_intervals, n_terms = table.coefficients.shape[:2]
    native.table_upload(
        "flux",
        np.ascontiguousarray(table.breakpoints, dtype=np.float64),
        low - slack,
        high + slack,
        np.ascontiguousarray(table.coefficients, dtype=np.complex128).reshape(
            n_intervals, n_terms, -1
        ),
    )


def _slab_payload(problem: PropagationProblem) -> dict[str, object]:
    """Pack the hybrid planar kind's channels, wall terms and per-column collar rate.

    Schema 15's ``SeparableOperator.kappa_sq`` is ``(n_channels, n_omega)``, column
    ``c`` belonging to channel ``c % n_channels``; the geometry's collar rate depends on
    y alone (solprop checks it), so it is packed per column ``p + n_p * iy``, the index
    of the temporal row it acts on.
    """
    geometry = problem.transform.geometry
    separable = problem.linear.separable
    assert geometry is not None
    assert separable is not None
    assert separable.kappa_sq is not None
    n_x, n_y = geometry.shape
    n_p = separable.n_channels
    return {
        "n_p": n_p,
        "spatial_kappa_sq": np.ascontiguousarray(separable.kappa_sq),
        "spatial_absorber_rate": np.ascontiguousarray(
            np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)
        ),
    }


def _general_payload(
    problem: PropagationProblem, chunk: int | None
) -> dict[str, object]:
    """Pack the general modal route's maps, derived as ``numpy_ref`` derives them.

    The per-node tensor, the form and the delayed maps come from solprop's own
    public derivations (``solprop.backends.numpy_ref.general_modal``), so this
    backend executes exactly the arrays the reference executes; the chunk is the
    reference's own memory strategy at the same budget
    (:func:`solprop.backends.numpy_ref.general_modal.general_planned_chunk`), given as
    `chunk`. Nothing is packed for a problem without ``transform.modal_tables``.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return {"general_modal": False}
    assert tables.synthesis is not None
    assert tables.projection is not None
    regional = problem.responses.regional_responses
    components = int(tables.electric.shape[2])
    nodes = int(tables.region_index.size)
    assert chunk is not None
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    terms = 0 if delayed_tensor is None else int(delayed_tensor.shape[1])
    intervals, n_deg = (int(one) for one in tables.synthesis.shape[:2])
    return {
        "general_modal": True,
        "general_real_form": node_kerr_form(regional) == "real",
        # A moving guide's maps are every Chebyshev coefficient along z, flattened
        # over (n_intervals, n_deg) in the order of the stage bundle's
        # transform_weights; a constant guide's are the one coefficient.
        "general_map_intervals": intervals,
        "general_map_n_deg": n_deg,
        "general_components": components,
        "general_nodes": nodes,
        "general_node_chunk": chunk,
        "general_responses": 0 if delayed_tensor is None else delayed_tensor.shape[0],
        "general_terms": terms,
        "general_pad_scale": float(problem.grids.pad_scale),
        "general_epsilon0": EPS0,
        "general_synthesis": np.ascontiguousarray(tables.synthesis),
        "general_projection": np.ascontiguousarray(tables.projection),
        "general_tensor": np.ascontiguousarray(node_kerr_tensor(tables, regional)),
        "general_frequency_factor": np.ascontiguousarray(
            tables.source_frequency_factor, dtype=np.complex128
        ),
        "general_delayed_tensor": np.ascontiguousarray(
            np.empty(0) if delayed_tensor is None else delayed_tensor
        ),
        "general_delayed_spectrum": np.ascontiguousarray(
            np.empty(0, dtype=np.complex128)
            if delayed_spectrum is None
            else delayed_spectrum
        ),
        "general_node_response": np.ascontiguousarray(
            np.empty(0, dtype=np.uint64) if not terms else owner.astype(np.int64) + 1,
            dtype=np.uint64,
        ),
    }


def _envelope_payload(problem: PropagationProblem) -> dict[str, object]:
    """Pack the accepted cycle rules and already masked spectral factors."""
    assembly = problem.responses.envelope_plasma
    if assembly is None:
        return {"envelope_plasma": False}
    nodes = np.zeros((len(assembly.channels), 4, 512))
    weights = np.zeros_like(nodes)
    for channel_index, channel in enumerate(assembly.channels):
        for level, rule in enumerate(channel.cycle.rules):
            nodes[channel_index, level, : len(rule.nodes)] = rule.nodes
            weights[channel_index, level, : len(rule.weights)] = rule.weights
    return {
        "envelope_plasma": True,
        "envelope_rule_nodes": nodes,
        "envelope_rule_weights": weights,
        "envelope_drude_factor": assembly.factors.drude,
        "envelope_loss_factor": assembly.factors.loss,
    }


def _channel_payload(problem: PropagationProblem) -> dict[str, object]:
    """Pack independent rate tables without resampling their coefficients."""
    parameters: list[list[float]] = []
    coefficients: list[np.ndarray] = []
    offset = 0
    for channel in problem.responses.population_channels:
        rate = channel.rate
        parameters.append(
            [
                rate.log_field_min,
                rate.d_log_field,
                rate.field_min,
                rate.field_max,
                rate.rate_max,
                rate.coefficients.shape[1],
                offset,
                channel.charge_mass_ratio,
                channel.ionisation_potential,
            ]
        )
        coefficients.append(rate.coefficients.ravel())
        offset += rate.coefficients.size
    terms = problem.responses.raman_terms
    return {
        "n_channels": len(parameters),
        "channel_parameters": np.asarray(parameters, dtype=np.float64).reshape(-1, 9),
        "channel_coefficients": np.concatenate(coefficients)
        if coefficients
        else np.empty(0),
        "n_raman_components": len(terms),
        "raman_row_component": np.asarray(
            [component for component, term in enumerate(terms) for _ in term.undamped],
            dtype=np.float64,
        ),
        "raman_component_fraction": np.asarray([term.fraction for term in terms]),
    }

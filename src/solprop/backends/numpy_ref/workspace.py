# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's workspace: its state and every state-sized operation (SEP 52).

:class:`NumpyWorkspace` holds one run's state and the problem's tables, and does every
operation a march, an engine or the session asks of the state: loading it, the stage
scalars and the source, the accepted step's finish, the band mask, the statistics, the
boundary monitor's and the transverse edge's readings, the quadrature alarm and the
host source's reservation. It holds no march state -- no position, proposal, history
or ledger -- and is reused across a scan's points, a load replacing its state. The
engines (:mod:`solprop.backends.numpy_ref.engines`,
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`,
:class:`~solprop.backends.low_storage.HostLowStorage`) are what the marches ask of it.

The contract is consumed only through its evaluation functions --
:func:`solprop.translate.linop.delta_phi`,
:func:`solprop.translate.linop.unitary_phase`,
:func:`solprop.translate.responses.prefactor_at`, and the scalar tables
``responses.density`` and ``transform.taper_scale`` -- never through the coefficient
arrays, so the representation of the z tables can change without touching a backend.
"""

import collections
import dataclasses
import functools
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Num

from solprop._helpers import PLAIN, Helpers
from solprop.backends import tdse_source
from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.backends.memory import adams_memory, reference_base_memory
from solprop.backends.node_chunk import (
    NodeChunkPlan,
    general_planned_chunk,
    plan_node_chunk,
)
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.envelope_plasma import (
    envelope_populations,
    envelope_sources,
)
from solprop.backends.numpy_ref.general_modal import (
    GeneralMaps,
    general_memory,
    general_source,
)
from solprop.backends.populations import axis_refusal, population_statistics
from solprop.backends.spatial import (
    FarFieldTables,
    collect_spatial,
    radial_absorber_matrix,
    slab_columns,
    slab_rows,
    slab_synthesis,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.records import SavedPlane
from solprop.profiles import ZProfile
from solprop.stats import (
    _POYNTING_STATISTICS,
    CATALOGUE,
    METRIC_WEIGHTED_STATISTICS,
    ON_AXIS_PLASMA_STATISTICS,
    PLASMA_INTEGRATION_STATISTICS,
    WEIGHT_DEPENDENT_STATISTICS,
    StatsSpec,
    _poynting_statistics,
    analytic_field,
    collect,
    weight_refusal_for,
)
from solprop.translate.coupled import (
    CoupledFlowFactors,
    coupled_flow_factors,
    metric_at,
)
from solprop.translate.general import (
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.translate.linop import delta_phi, power_weight_at, unitary_phase
from solprop.translate.masks import noise_spectrum
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.problem import (
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
)
from solprop.translate.responses import prefactor_at
from solprop.translate.routes import (
    flux_of,
    tdse_of,
    transverse_of,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.spatial import separable_values
from solprop.translate.stage import (
    StageScalars,
    anchor,
    combine,
    scalars,
    scalars_at,
    terms,
)
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding, TDSEPurpose
from solprop.translate.tdse_source import (
    TDSE_UNAVAILABLE_STATISTICS,
)
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

__all__ = ["BlockResidual", "NumpyWorkspace", "SourceTables"]

logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]


class NumpyWorkspace:
    """The reference's state and every state-sized operation on it (SEP 52 section 5.1).

    Built once per backend from the validated problem and request, and reused for every
    run the backend serves: :meth:`load` replaces the state and everything that
    describes it. It keeps no march state. The engines read and write the state only
    through these operations, and the session (today the backend's step driver) reads
    the statistics and the boundary readings from them.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request: the stepper's tableau and controller, which the trials read,
        and the statistics, whose far-field tables are formed here.
    chunk_plan : solprop.backends.node_chunk.NodeChunkPlan
        The node chunk the response passes hold at once.
    propagator_mode : {'tabulate', 'recompute'}
        The resolved propagator mode: whether a step's exponents are kept for the step.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, which the statistics need.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    tdse_binding : solprop.translate.tdse.TDSEBinding, optional
        The host response binding a TDSE problem declares; ``None`` otherwise.
    name : str, optional
        The backend's name, for messages.

    Attributes
    ----------
    collar_quiet : bool
        Whether the modal stage leaves a quiet in-stage temporal collar out (SEP 21):
        set by the session from the boundary monitor, read by every stage.
    axis_scans : int
        The on-axis plasma scans actually run (SEP 19), so that a test can see that a
        rule with a node on the axis supplies that row instead of a fresh scan.
    helpers : solprop._helpers.Helpers
        The uncompiled stage-table and Adams-basis evaluations, ``PLAIN`` (SEP 29
        section 6.6).
    """

    helpers: Helpers = PLAIN

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        chunk_plan: NodeChunkPlan,
        propagator_mode: PropagatorMode,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
        name: str = "numpy_ref",
    ) -> None:
        """Gather the problem's tables; no state is loaded yet."""
        self.name = name
        self._problem = problem
        self._stepper = request.stepper
        self._chunk_plan = chunk_plan
        self._propagator_mode = propagator_mode
        self._grid = grid
        self._modes = modes
        self._profiles = profiles
        self._state: _State | None = None
        self._noise: _State | None = None
        # The input draw rides in the state, a resumed one's too; the boundary monitor
        # reads it as the vacuum floor beneath the light it judges (SEP 21, Q24).
        self._input_floor: _State | None = None
        self._clean_columns: Float[np.ndarray, " n_columns"] | None = None
        self._monitor: BoundaryMonitor | None = None
        self.collar_quiet = False
        # The fixed arrays of the seam's device form, held for the whole run; the
        # weights that multiply them are a bundle of host floats per stage.
        self._terms = terms(problem)
        # Hoisted per step by `hoist`; the entrance's until the first step asks.
        self._anchor = anchor(problem, 0.0, helpers=self.helpers)
        # A step's bundles taken together by `stage_bundles`, by anchor and position;
        # `hoist` drops them.
        self._prepared: dict[tuple[float, float], StageScalars] = {}
        self._on_axis: Float[np.ndarray, "n_rows n_modes"] | None = None
        self.axis_scans = 0
        self._axis_node: int | None = None
        self._axis_node_known = False
        self._tdse_binding = tdse_binding
        self._tdse_last_purpose: TDSEPurpose | None = None
        self._tdse_last_z: float | None = None
        self._tdse_last_projection: dict[str, object] | None = None
        self._gather(problem, request)

    def _gather(self, problem: PropagationProblem, request: RunRequest) -> None:
        """Gather the tables and caches the stages and the statistics read."""
        self._statistics_power_weight = np.empty(
            problem.shape if problem.transform.geometry is None else (0,)
        )
        # Formed once, so a geometry the far-field statistics refuse is refused here,
        # before any step (SEP 23 section 6).
        self._far_field = FarFieldTables(problem, request.observation.stats)
        self._gated_plasma = _BENCHMARK_GATED.get()
        self._last_plasma: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._kind: GridKind = problem.grid_kind
        self._tables = SourceTables.of(problem)
        self._carrier = _harmonic_carrier(problem)
        self._embedded = _embedded_projection(problem)
        self._general = _general_maps(problem, self._chunk_plan)
        self._general_responds = self._general is not None and (
            bool(np.any(self._general.tensor)) or self._general.dispersive is not None
        )
        self._flow = _coupled_flow_maps(problem)
        self._block_residual: BlockResidual | None = None
        self._flow_cache: (
            tuple[
                tuple[int, float],
                dict[float, tuple[np.ndarray | None, np.ndarray | None]],
            ]
            | None
        ) = None
        self._last_projected: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_coarse: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_bundle: StageScalars | None = None
        self._last_envelope: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._raman_rates: tuple[float, ...] | None = None
        self._raman_densities: tuple[float, ...] | None = None
        self._raman_kernel: Complex[np.ndarray, "... n_omega_doubled"] | None = None
        self._raman_transforms = 0
        self._alarm = math.nan
        self._alarm_reported = False
        # ADR-025's in-stage scheme is a second term on every stage rather than a
        # factor on the accepted state, so a problem that has no response at all still
        # has a derivative when it is active.
        self._spatial_absorber = radial_absorber_matrix(
            problem.transform, problem.grids.pad_scale
        )
        self._in_stage_absorber = bool(self._spatial_absorber.size) or (
            problem.responses.temporal_absorber_fine is not None
            and problem.responses.absorber_scheme == "nonlinear"
        )
        geometry = problem.transform.geometry
        if geometry is not None and geometry.kind in ("free", "slab"):
            self._in_stage_absorber |= bool(np.any(geometry.absorber_rate))
        self._held: (
            tuple[float, float, StageScalars, Complex[np.ndarray, ...]] | None
        ) = None
        # The lab-frame source's unit "propagator": one element broadcast to the state's
        # shape, so it holds no volume beside the state.
        self._lab_frame = np.broadcast_to(
            np.ones((), dtype=np.complex128), problem.shape
        )

    # --- the state ---------------------------------------------------------------

    @property
    def problem(self) -> PropagationProblem:  # numpydoc ignore=RT01
        """The problem the workspace was built for."""
        return self._problem

    @property
    def stepper(self) -> StepperSettings:  # numpydoc ignore=RT01
        """The run's stepper settings: the tableau, controller and schedule."""
        return self._stepper

    @property
    def node_chunk(self) -> int:  # numpydoc ignore=RT01
        """The nodes a response pass holds at once."""
        return self._chunk_plan.chunk

    @property
    def state(self) -> _State | None:  # numpydoc ignore=RT01
        """The committed state, internal units and FFT convention; ``None`` unloaded."""
        return self._state

    @property
    def noise(self) -> _State | None:  # numpydoc ignore=RT01
        """The ``modified`` noise model's realisation, or ``None`` with any other."""
        return self._noise

    @property
    def input_floor(self) -> _State | None:  # numpydoc ignore=RT01
        """The ``input`` noise draw the state carries (the monitor's floor), or None."""
        return self._input_floor

    @property
    def quadrature(self) -> float:  # numpydoc ignore=RT01
        """The quadrature alarm of the plane's last accepted step; NaN without one."""
        return self._alarm

    def load(
        self,
        field: _State,
        noise: NoiseRealisation | None,
        *,
        z: float,
        resume: bool,
    ) -> None:
        """Load a state and its noise realisation, masked to the active band.

        The ``input`` noise model's draw is added at the run's launch, never on a
        resume; the ``modified`` model keeps it apart and adds it at every stage. The
        quadrature alarm of a previous run is forgotten (SEP 22 section 4.7).

        Parameters
        ----------
        field : numpy.ndarray
            The state, ``complex128``, shape ``(n_modes, n_pol, n_omega)``, which the
            caller has checked against the problem's.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry, or ``None`` for the problem's own.
        z : float
            The position the state belongs to, m, inside the operator table.
        resume : bool
            Whether `field` came back from a saved plane, and so never takes the
            ``input`` draw.
        """
        values = field
        drawn = self._noise_field(noise)
        model = self._problem.initial.noise_model
        self._clean_columns = None
        if (
            model == "input"
            and drawn is not None
            and not resume
            and z == self._problem.initial.z
        ):
            if transverse_of(self._problem.route) is not None:
                self._clean_columns = _clean_launch_columns(
                    values, drawn, self._problem.grids.active_mask
                )
            values = values + drawn
        self._state = self.mask(values)
        self._noise = drawn if model == "modified" else None
        self._input_floor = drawn if model == "input" else None
        self._alarm = math.nan

    def install(self, state: _State) -> None:
        """Make an array the committed state, as it is.

        A dense observation's interpolated field and the accepted state it restores
        afterwards, and a resumed snapshot's state, go in here; nothing is masked.

        Parameters
        ----------
        state : numpy.ndarray
            The state, shape ``(n_modes, n_pol, n_omega)``.
        """
        self._state = state

    def bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Read a boundary monitor's energies from now on, or stop with ``None``.

        A monitor of the ``"split_step"`` scheme also has each accepted step's
        :meth:`finish` take the split collar's sums.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor or None
            The run's monitor.
        """
        self._monitor = monitor

    def reading(self) -> BoundaryReading:
        """Return the boundary monitor's reading of the committed state (SEP 21).

        The state's energy, its four weighted energies and, under a split collar's
        crossing limit, its occupancy; the split's own sums travel in each step's
        :class:`~solprop.backends.engines.AcceptRecord`.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The reading.

        Raises
        ------
        BackendError
            If no monitor is bound or no state is loaded.
        """
        monitor = self._monitor
        if monitor is None:
            raise BackendError(f"{self.name}: no boundary monitor is attached")
        field = self.committed()
        energies = monitor.energies(field)
        crossing = monitor.crossing
        return BoundaryReading(
            total=energies.total,
            regions=energies.regions,
            occupancy=None
            if crossing is None
            else crossing.occupancy(field, monitor.columns),
        )

    def edge_reading(
        self,
        edge: TransverseEdge,
        *,
        faces: bool = True,
        current: bool = False,
        full: bool = False,
    ) -> _EdgeReading:
        """Return what the transverse edge asks of the committed state (SEP 44).

        Parameters
        ----------
        edge : solprop.transverse_monitor.TransverseEdge
            The run's edge.
        faces : bool, optional
            Whether the faces are read.
        current : bool, optional
            Whether a collar's inward current is wanted.
        full : bool, optional
            Whether every node is wanted, not only the rows a read uses.

        Returns
        -------
        object
            The edge's reading of the host state.
        """
        return edge.reduction.reading(
            np.asarray(self.committed()),
            faces=faces,
            current=current,
            full=full,
        )

    def clean_columns(
        self, edge: TransverseEdge
    ) -> Float[np.ndarray, " n_columns"] | None:
        """Return each column's launched energy without the input noise, or ``None``.

        What the safety ramp is judged on, so a run with no ramp needs none. It is
        formed where the load has both parts; only a resume that found no record of
        the monitor has none, and then it is formed from the state.

        Parameters
        ----------
        edge : solprop.transverse_monitor.TransverseEdge
            The run's edge.

        Returns
        -------
        numpy.ndarray or None
            The unweighted energies, or ``None`` with no noise, no ramp or on a resumed
            launch.
        """
        floor = self._input_floor
        if floor is None or edge.resumed or edge.design is None:
            return None
        if self._clean_columns is not None:
            return self._clean_columns
        state = np.asarray(self.committed())
        return edge.reduction.column_power(state - np.asarray(floor))

    def mask(self, state: _State) -> _State:
        """Zero a state outside the active band.

        Parameters
        ----------
        state : numpy.ndarray
            A state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        numpy.ndarray
            A new masked array.
        """
        return passes.apply_hard_mask(state, self._problem.grids.active_mask)

    def committed(self) -> _State:
        """Return the committed state, refusing an unloaded workspace.

        Returns
        -------
        numpy.ndarray
            The state itself, not a copy.

        Raises
        ------
        BackendError
            If no state is loaded.
        """
        if self._state is None:
            msg = "no initial state: call set_initial before propagating or observing"
            raise BackendError(msg)
        return self._state

    # --- the host source (SEP 49) ---------------------------------------------------

    def reserve_calls(self, count: int) -> None:
        """Reserve a whole trial's host-source calls before its first evaluation.

        Parameters
        ----------
        count : int
            The trial's logical response calls.

        Raises
        ------
        solprop.exceptions.ResponseBudgetError
            If the bound source's budget cannot pay for them.
        """
        if self._tdse_binding is not None:
            self._tdse_binding.reserve(count)

    def release_calls(self) -> None:
        """Release the unspent part of a trial's reservation."""
        if self._tdse_binding is not None:
            self._tdse_binding.release_reservation()

    def record_trial(self, *, accepted: bool) -> None:
        """Tell the bound source how the last reserved trial ended (SEP 20 5.8).

        Parameters
        ----------
        accepted : bool
            Whether it was accepted.
        """
        if self._tdse_binding is not None:
            self._tdse_binding.record_trial(accepted=accepted)

    def restore_atomic_ledger(self, atomic: Mapping[str, object]) -> None:
        """Restore a resumed run's atomic call counts into the bound source.

        Parameters
        ----------
        atomic : collections.abc.Mapping
            The persisted counts.

        Raises
        ------
        BackendError
            If no host source is bound.
        """
        if self._tdse_binding is None:
            raise BackendError("atomic call ledger requires a TDSE binding")
        self._tdse_binding.restore_ledger(AtomicCallLedger.from_record(atomic))

    def metadata(self) -> dict[str, object]:
        """Return the workspace's execution facts for the run's record.

        Returns
        -------
        dict
            The benchmark-gated plasma source's form and, with a host source, its
            ledger, execution and last evaluation; JSON-serialisable.
        """
        record: dict[str, object] = {}
        if self._gated_plasma:
            record["_benchmark_plasma_source"] = "gated_nonresumable"
            record["plasma_source"] = {"form": "gated_benchmark", "resumable": False}
        if self._tdse_binding is not None:
            record["atomic_calls"] = dataclasses.asdict(self._tdse_binding.ledger)
            record["tdse_execution"] = dict(self._tdse_binding.metadata())
            record["tdse_last_evaluation"] = {
                "purpose": self._tdse_last_purpose,
                "z": self._tdse_last_z,
                "projection": self._tdse_last_projection,
            }
            record["tdse_source"] = tdse_source.source_record(self._problem)
        return record

    # --- the quadrature alarm --------------------------------------------------------

    def reset_quadrature(self) -> None:
        """Forget the alarm, so that the next plane reports its own."""
        self._alarm = math.nan

    def hoist(self, z0: float) -> None:
        """Evaluate once per step what every stage of it subtracts.

        ``Phi`` is cumulative, so each stage's increment is against the same anchor.
        On the RDW gradient the Clenshaw pass over the scalar tables is the largest
        part of what the host does per step, and this is a sixth of it. The bundles
        :meth:`stage_bundles` kept are dropped.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        """
        self._anchor = anchor(self._problem, z0, helpers=self.helpers)
        self._prepared.clear()

    def stage_bundles(
        self, z0: float, positions: Sequence[float]
    ) -> tuple[StageScalars, ...]:
        """Return the host scalars of several stages of one step, taken together.

        A trial knows all its stage positions before its first stage, and reading each
        scalar table once over all of them (:func:`solprop.translate.stage.scalars_at`)
        costs a Python call and a Clenshaw dispatch per table rather than per table and
        position. The bundles equal :meth:`stage_scalars`'s bit for bit, and are kept
        until the next :meth:`hoist`: a later :meth:`stage_scalars` at the same anchor
        and position returns the kept bundle, so a step loop written one stage at a
        time takes them without change. Bundles of several anchors are kept side by
        side (kernel schema 29): the low-storage engine reads each of its three
        positions as its own anchor. A caller passes only positions the trial will
        evaluate.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        positions : collections.abc.Sequence of float
            The stages' positions, m.

        Returns
        -------
        tuple of solprop.translate.stage.StageScalars
            One bundle per position, in order.
        """
        prepared = self._prepared
        missing = [z1 for z1 in dict.fromkeys(positions) if (z0, z1) not in prepared]
        if missing:
            held = self._anchor if self._anchor.z == z0 else None
            for bundle in scalars_at(
                self._problem, z0, missing, at=held, helpers=self.helpers
            ):
                prepared[z0, bundle.z1] = bundle
        return tuple(prepared[z0, z1] for z1 in positions)

    def stage_scalars(self, z0: float, z1: float) -> StageScalars:
        """Return the host scalars of one stage, from the anchor to the node.

        The anchor's own weights come from :meth:`hoist` when the caller has already
        taken them for this step, and are evaluated here otherwise. A bundle
        :meth:`stage_bundles` took for this anchor and position is returned as kept.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        z1 : float
            The stage's position, m.

        Returns
        -------
        solprop.translate.stage.StageScalars
            The bundle.
        """
        kept = self._prepared.get((z0, z1))
        if kept is not None:
            return kept
        held = self._anchor if self._anchor.z == z0 else None
        return scalars(self._problem, z0, z1, at=held, helpers=self.helpers)

    def delta_phi(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the stage's propagator exponent from the bundle.

        One linear combination of the terms the workspace holds over the bundle's
        floats. A problem that took the z table has no such form and falls back to
        :func:`solprop.translate.linop.delta_phi`.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.

        Returns
        -------
        numpy.ndarray
            ``Phi(z1) - Phi(z0)`` in the state's layout.
        """
        if bundle.phi_delta is None or self._terms.phi is None:
            return delta_phi(
                self._problem.linear, bundle.z0, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.phi_delta, self._terms.phi)

    def prefactor(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the response prefactor at the stage's position, from the bundle.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.

        Returns
        -------
        numpy.ndarray
            The prefactor times the crop scale, in m/(V s) per unit polarisation.
        """
        if bundle.prefactor is None or self._terms.prefactor is None:
            return prefactor_at(
                self._problem.responses, bundle.z1, helpers=self.helpers
            )
        return combine(bundle.prefactor, self._terms.prefactor)

    def unitary_phase(
        self, bundle: StageScalars
    ) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
        """Return ``Im Phi`` at the stage's position, from the same terms.

        An expansion of ``Phi`` is cumulative and vanishes at the start of the range,
        so its value at a position is the bundle's ``phi_at`` against the same arrays
        and the noise costs no second representation.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.

        Returns
        -------
        numpy.ndarray
            The phase, rad, in the state's layout.
        """
        if bundle.phi_at is None or self._terms.phi is None:
            return unitary_phase(self._problem.linear, bundle.z1)
        return np.imag(combine(bundle.phi_at, self._terms.phi))

    def _noise_field(
        self, noise: NoiseRealisation | None
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the noise realisation to carry, in the canonical layout.

        - Returns: the noise spectrum, or ``None`` when no noise model is active.
        """
        if self._problem.initial.noise_model == "none":
            return None
        if noise is not None:
            return noise_spectrum(noise, self._problem.shape)
        return self._problem.initial.noise0

    def noise_at(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the shot-noise field advanced to a stage's position.

        ``N(z) = N0 exp(i Im Phi(omega, z))``: dispersion yes, loss and gain no, from
        the same terms as the linear flow, so the noise costs one multiply-add per
        element per stage and no transform.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.

        Returns
        -------
        numpy.ndarray or None
            The advanced noise, or ``None`` when the modified model is not in use.
        """
        if self._noise is None:
            return None
        return self._noise * np.exp(1j * self.unitary_phase(bundle))

    def noise_phase(
        self, z: float
    ) -> Float[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return ``Im Phi`` at a position, or ``None`` without a noise model.

        Parameters
        ----------
        z : float
            The position, m.

        Returns
        -------
        numpy.ndarray or None
            The phase in radians, in the state's layout.
        """
        if self._problem.initial.noise_model == "none":
            return None
        phase = self.unitary_phase(self.stage_scalars(z, z))
        return np.ascontiguousarray(np.broadcast_to(phase, self._problem.shape))

    def source(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[
        Complex[np.ndarray, "n_modes n_pol n_omega"],
        Complex[np.ndarray, "n_modes n_pol n_omega"],
    ]:
        """Evaluate the source once, in the interaction frame of a step's anchor.

        The one expensive evaluation of an interaction-formulation Adams step: the
        nonlinear source at a lab-frame field, with the run's noise and in-stage
        absorber. A propagator :meth:`propagator` held for the same pair is reused.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`, in the state's layout.
        anchor, position : float
            The step's anchor and the evaluation position, m.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}, optional
            The host source's accounting label; ignored when none is bound.

        Returns
        -------
        numpy.ndarray
            ``U(anchor, position) N(lab, position)``.
        numpy.ndarray
            ``U(position, anchor)``, with ``U(z_2, z_1) = exp(Phi(z_2) - Phi(z_1))``.
        """
        held = self._held
        if held is not None and held[0] == anchor and held[1] == position:
            bundle, propagator = held[2], held[3]
        else:
            bundle = self.stage_scalars(anchor, position)
            propagator = np.exp(self.delta_phi(bundle))
        self._held = None
        noise = self.noise_at(bundle)
        field_in = lab if noise is None else lab + noise
        if self._tdse_binding is not None:
            return (
                self.nonlinear(field_in, bundle, 1.0 / propagator, purpose=purpose),
                propagator,
            )
        return self.nonlinear(field_in, bundle, 1.0 / propagator), propagator

    def coupled_block(self, z: float, dz: float) -> object:
        """Return the frozen block of the section a step from `z` is anchored in.

        The exponential formulation steps a coupled flow's constant-Schur and
        frozen-residual sections against the reference ``(Delta Phi / h) I + G_s``
        (SEP 16 section 4.5), with ``G_s`` the frozen block of the step's section.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m.

        Returns
        -------
        object
            The step's :class:`~solprop.backends.numpy_ref.multistep.CoupledBlock`,
            or ``None`` without a coupled flow.
        """
        from solprop.backends.numpy_ref.multistep import (  # noqa: PLC0415 - cycle
            CoupledBlock,
        )

        flow = self._flow
        if flow is None:
            return None
        bundle = self.stage_scalars(z, z + dz)
        section = bundle.flow_section
        assert section is not None
        residual = None
        if flow.residual_terms is not None:
            if self._block_residual is None:
                self._block_residual = BlockResidual(flow)
            residual = functools.partial(self._block_residual, section)
        return CoupledBlock(
            active=flow.active,
            vectors=flow.vectors[section],
            triangular=flow.triangular[section],
            section=section,
            end_weights=bundle.flow_weights,
            residual=residual,
        )

    def lab_source(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate the source once, untransported, at a lab-frame field.

        The same passes as :meth:`source`, with the stage's inverse propagator
        replaced by ones: a multiplication by one is exact, so the sample is the
        lab-frame source bit for bit, and every route's evaluation stays the one the
        Runge-Kutta trial uses. The sample the exponential formulation and the
        low-storage step integrate.

        `lab` is taken as given: it is not masked to the active band, as
        :meth:`load` masks the state it loads. On a masked field the sample equals the
        hoisted first stage at the same position bit for bit; a field with content
        outside the band (an unmasked launch) is another input, whose source differs by
        what that content makes (4.3e-13 of the peak on the optmat argon capillary).
        The stage scalars at ``(position, position)`` do not depend on the anchor last
        hoisted: a held anchor is reused only at its own position.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`, in the state's layout.
        position : float
            The evaluation position, m.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}, optional
            The host source's accounting label; ignored when none is bound.

        Returns
        -------
        numpy.ndarray
            The complete source ``F`` in the lab frame.
        """
        bundle = self.stage_scalars(position, position)
        noise = self.noise_at(bundle)
        field_in = lab if noise is None else lab + noise
        if self._tdse_binding is not None:
            return self.nonlinear(field_in, bundle, self._lab_frame, purpose=purpose)
        return self.nonlinear(field_in, bundle, self._lab_frame)

    def propagator(
        self, anchor: float, position: float, *, hold: bool = False
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the linear propagator ``U(position, anchor)``.

        Only the Adams engine holds it for the step's source call (:meth:`source`): the
        low-storage engine applies its flows at once, and a held one would be a volume
        its ledger does not name.

        Parameters
        ----------
        anchor, position : float
            The step's anchor and its end, m.
        hold : bool, optional
            Whether to keep it for the next :meth:`source` at the same pair.

        Returns
        -------
        numpy.ndarray
            ``exp(Phi(position) - Phi(anchor))`` in the state's layout, an array the
            caller owns.
        """
        bundle = self.stage_scalars(anchor, position)
        propagator = np.exp(self.delta_phi(bundle))
        if hold:
            self._held = (anchor, position, bundle, propagator)
        return propagator

    def channels(
        self,
        values: Complex[np.ndarray, "n_modes n_pol n_omega"],
        action: Complex[np.ndarray, "n_active_c n_modes_c_out n_modes_c_in"] | None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Mix the channel axis by one coupled-flow action, or return the values.

        Parameters
        ----------
        values : numpy.ndarray
            Any array of the state's layout.
        action : numpy.ndarray or None
            The action on the active bins, or ``None`` for a problem with no coupled
            flow, which is every route but the coherent wall.

        Returns
        -------
        numpy.ndarray
            The mixed values, or `values` itself.
        """
        if self._flow is None or action is None:
            return values
        return _apply_flow(values, action, self._flow.active)

    def residual(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        section: int,
        bundle: StageScalars,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the explicit residual's lab-frame source term, or nothing.

        ``None`` for every problem but a ``"frozen_residual"`` coupled flow: the
        constant section has an identically zero residual and is not multiplied by
        one, so its arithmetic is bit for bit what it was.

        Parameters
        ----------
        lab : numpy.ndarray
            The stage's lab-frame field.
        section : int
            The step's flow section.
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars, with the section's weights.

        Returns
        -------
        numpy.ndarray or None
            ``R_s(z_1) u`` in the state's layout, or ``None``.
        """
        flow = self._flow
        if flow is None or flow.residual_terms is None:
            return None
        assert bundle.flow_weights is not None
        return _residual_drive(
            lab,
            _residual_action(flow, section, bundle.flow_weights),
            flow.active,
            flow.residual_bins,
        )

    def flow_actions(
        self, section: int, dz: float
    ) -> Callable[
        [float],
        tuple[
            Complex[np.ndarray, "n_active_a n_modes_a_out n_modes_a_in"] | None,
            Complex[np.ndarray, "n_active_b n_modes_b_out n_modes_b_in"] | None,
        ],
    ]:
        """Return the memoised forward and inverse flow actions of one step.

        The generator is **frozen** across the step -- at the midpoint of the section
        the step's anchor lies in -- so an action depends on the stage fraction only
        through the length ``c_i h`` it covers, and the six distinct DOPRI5 fractions
        need at most six pairs; the tableau uses the last one twice and the zero
        fraction is the identity. The whole step's pair is checked against that
        section's declared conditioning row, which bounds every shorter one because
        ``exp(c dz (mu_max - mu_min))`` increases with ``c``.

        The pairs also survive **between** steps, for the one section and trial length
        last seen: a fixed schedule and an accepted adaptive step both repeat their
        ``dz``, and an action that depends on nothing but ``dz`` and a frozen generator
        is then the same matrix. Exactly one pair of them is held, so what a step has
        resident at its peak -- which is what :meth:`memory_ledger` reports -- is
        unchanged, and a step that crosses into the next section rebuilds, which is a
        cache invalidation at the section edge and is what makes the edge a kink of
        the flow (``docs/dev/stepper.md`` section 2). The conditioning of a held
        length was measured when it was built.

        Parameters
        ----------
        section : int
            The section the step's anchor lies in.
        dz : float
            The length of the trial step, m.

        Returns
        -------
        callable
            ``fraction -> (U, U^-1)``, each ``None`` when no coupled flow is carried.
        """
        flow = self._flow
        if flow is None:
            return lambda fraction: (None, None)  # noqa: ARG005
        held = self._flow_cache
        key = (section, dz)
        cache: dict[float, tuple[np.ndarray | None, np.ndarray | None]] = (
            held[1] if held is not None and held[0] == key else {}
        )
        self._flow_cache = (key, cache)

        def _action(
            fraction: float,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            """Return the pair at a stage fraction, computing it at most once.

            - ``fraction``: the tableau node ``c_i``, dimensionless, in ``[0, 1]``.
            - Returns: the forward and inverse actions, or ``(None, None)`` at fraction
              zero, where both are the identity.
            """
            if fraction == 0.0:
                return None, None
            if fraction not in cache:
                pair = _flow_action(flow, section, fraction * dz)
                if fraction == 1.0:
                    _check_flow_conditioning(flow, section, pair, dz)
                cache[fraction] = pair
            return cache[fraction]

        return _action

    def finish(
        self,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        dz: float,
    ) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], AcceptRecord]:
        """Apply the once-per-accepted-step boundary treatments to a kept step.

        The hard band mask and, when the ``"split_step"`` absorber scheme is active,
        the temporal collar, followed by the mask again: its product in time spreads
        content into every bin, so the band is imposed last. Applied only to a step
        that is kept, never to one about to be thrown away. While a monitor of that
        scheme is bound, the split collar's sums are taken here from the masked field
        **before** the split, the field the removal and the occupancies describe
        (SEP 52 section 4): what the split removes, ``P (1 - exp(-alpha h))``, and
        the energies of ``P exp(-alpha h / 2)``.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the end of an accepted step, shape ``(n_modes, n_pol,
            n_omega)``; not modified.
        dz : float
            The length of that step, m, which the collar's rate needs.

        Returns
        -------
        numpy.ndarray
            The finished state, a new array.
        solprop.backends.engines.AcceptRecord
            The split collar's sums; the defaults (nothing removed, nothing found)
            without a split collar or a bound monitor of its scheme.
        """
        masked = self.mask(state)
        responses = self._problem.responses
        rate = responses.temporal_absorber
        if rate is None or responses.absorber_scheme != "split_step":
            # Either there is no collar, or ADR-025's other scheme applied it inside
            # every stage, where it is already in the accepted state.
            return masked, AcceptRecord()
        record = AcceptRecord()
        monitor = self._monitor
        if monitor is not None and monitor.split:
            # The time factor acts alike on every column of any transverse basis, so
            # the stored state's power is what the split takes of, wherever it acts.
            # What it finds is what the occupancies are judged on.
            sums = monitor.split_sums(masked, dz)
            record = AcceptRecord(removed=float(sums[0]), found=sums[1:])
        return self._split(masked, rate, dz), record

    def _split(
        self,
        masked: Complex[np.ndarray, "n_modes n_pol n_omega"],
        rate: np.ndarray,
        dz: float,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Apply the split temporal collar to a masked state, and the mask after it."""
        geometry = self._problem.transform.geometry
        if geometry is not None and geometry.kind == "slab":
            # A (p, y) row is a modal amplitude whose time signal is a real field's,
            # and the rate is the same at every node, so it is applied row by row.
            transform = self._problem.transform
            physical = slab_rows(transform, masked)
            chunk = self.node_chunk
            for first in range(0, len(physical), chunk):
                rows = slice(first, first + chunk)
                physical[rows] = passes.apply_temporal_rate(
                    physical[rows], rate, dz, kind=self._kind
                )
            return self.mask(slab_columns(transform, physical))
        if geometry is not None and geometry.kind == "free":
            nx, ny = geometry.shape
            shape = (ny, nx, *masked.shape[1:])
            physical = np.fft.ifft2(masked.reshape(shape), axes=(0, 1)).reshape(
                masked.shape
            )
            chunk = self.node_chunk
            for first in range(0, len(physical), chunk):
                rows = slice(first, first + chunk)
                physical[rows] = passes.apply_temporal_rate(
                    physical[rows], rate, dz, kind=self._kind
                )
            result = np.fft.fft2(physical.reshape(shape), axes=(0, 1)).reshape(
                masked.shape
            )
            return self.mask(result)
        # The product in time spreads content across every bin, the DC among them, so
        # the band is imposed again after it, as the in-stage term is masked.
        return self.mask(passes.apply_temporal_rate(masked, rate, dz, kind=self._kind))

    def score_quadrature(self) -> tuple[float, float]:
        """Score the quadrature from the last stage of an accepted step.

        The relative difference between the projections of the same node polarisation
        with the Kronrod rule and with its embedded Gauss subset, at the cost of one
        extra forward transform per accepted step rather than a whole second evaluation
        of the transform.
        It is a **statistic and a warning, never a controller input**: a
        quadrature too coarse for the integrand is a modelling error the step size
        cannot fix, and feeding it to the controller would only shrink the step until
        the run stopped.

        The value is ``nan`` when the rule has no embedded pair, because the difference
        is then identically zero and reading that as convergence would be wrong; and
        ``nan`` where the polarisation itself vanishes, at a true vacuum end, because a
        relative measure of nothing says nothing. The session reports a value above
        :data:`~solprop.translate.modal.QUADRATURE_ALARM` once per plane, and the
        ``quadrature_error`` statistic reads it (:attr:`quadrature`).

        Returns
        -------
        float
            The alarm, dimensionless, or NaN.
        float
            Where the last stage was evaluated, m, or NaN without one.
        """
        if (
            self._last_coarse is None
            or self._last_projected is None
            or self._last_bundle is None
        ):
            self._alarm = math.nan
            return self._alarm, math.nan
        grids = self._problem.grids
        weight = self.prefactor(self._last_bundle) * self._tables.omega_window
        fine = weight * passes.to_frequency(
            self._last_projected, grids.crop_index, kind=self._kind
        )
        coarse = weight * passes.to_frequency(
            self._last_coarse, grids.crop_index, kind=self._kind
        )
        if self._last_envelope is not None:
            assembly = self._problem.responses.envelope_plasma
            assert assembly is not None
            drive, loss, coarse_drive, coarse_loss = self._last_envelope
            assert coarse_drive is not None and coarse_loss is not None
            for factor, full, reduced in (
                (assembly.factors.drude, drive, coarse_drive),
                (assembly.factors.loss, loss, coarse_loss),
            ):
                fine += (
                    weight
                    * factor
                    * passes.to_frequency(full, grids.crop_index, kind=self._kind)
                )
                coarse += (
                    weight
                    * factor
                    * passes.to_frequency(reduced, grids.crop_index, kind=self._kind)
                )
        if self._last_plasma is not None:
            gated, edge, coarse_gated, coarse_edge = self._last_plasma
            assert coarse_gated is not None and coarse_edge is not None
            fine += weight * self._held_streams(gated, edge)
            coarse += weight * self._held_streams(coarse_gated, coarse_edge)
        scale = float(np.linalg.norm(fine))
        self._alarm = (
            math.nan if scale == 0.0 else float(np.linalg.norm(coarse - fine) / scale)
        )
        return self._alarm, self._last_bundle.z1

    def _envelope_sources(
        self, field_t: Complex[np.ndarray, "n_nodes n_pol n_to"], bundle: StageScalars
    ) -> tuple[
        Complex[np.ndarray, "n_nodes n_pol n_to"],
        Complex[np.ndarray, "n_nodes n_pol n_to"],
    ]:
        """Return the envelope-grid plasma's windowed drive and loss at some nodes.

        - ``field_t``: the field at the nodes, V/m.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        return envelope_sources(
            field_t,
            envelope,
            bundle,
            self._problem.grids.dt_fine,
            self._tables.t_window,
        )

    def _envelope_populations(
        self,
        nodes: Complex[np.ndarray, "n_nodes n_pol n_to"],
        initials: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each envelope channel's final fraction and peak rate at the nodes.

        - ``nodes``: the physical envelope at the nodes, V/m.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        return envelope_populations(
            nodes, envelope, initials, self._problem.grids.dt_fine
        )

    def error_norm(
        self,
        error: Complex[np.ndarray, "n_modes n_pol n_omega"],
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        advanced: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> float:
        """Return the controller's scaled norm of a trial step's embedded difference.

        Parameters
        ----------
        error : numpy.ndarray
            The embedded difference, carried to the step's end.
        state : numpy.ndarray
            The state before the step.
        advanced : numpy.ndarray
            The state after it.

        Returns
        -------
        float
            The scaled norm.
        """
        return passes.error_norm(
            error,
            state,
            advanced,
            self._problem.grids.norm_mask,
            self._stepper.controller,
        )

    def stats(self, spec: StatsSpec, z: float) -> dict[str, np.ndarray]:
        """Collect the requested statistics of the committed state at a position.

        The statistics of the saved field go to :func:`solprop.stats.collect`; there is
        exactly one definition of each in the package and that is it. The statistics
        :class:`solprop.stats.StatDefinition` marks ``backend`` are of quantities a
        saved plane does not carry -- the electron density, the peak ionisation rate,
        the on-axis intensity, the quadrature error -- and are formed here from the
        workspace's own buffers.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.
        z : float
            The committed state's position, m.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name.
        """
        state = self.committed()
        grid = self._require_grid()
        if self._tdse_binding is not None:
            unavailable = TDSE_UNAVAILABLE_STATISTICS.intersection(spec.names)
            if unavailable:
                raise BackendError(
                    f"TDSE does not define legacy statistics {sorted(unavailable)}"
                )
        # Before the spatial branch: a spatial run asked for a
        # general name would otherwise reach `collect_spatial` and be answered from a
        # geometry that carries no certified metric at all.
        _refuse_inadmissible(self._problem, spec)
        if self._problem.transform.geometry is not None:
            response_names = (
                {"electron_density", "peak_ionisation_rate", "quadrature_error"}
                | set(PLASMA_INTEGRATION_STATISTICS)
                | {name for name in CATALOGUE if name.startswith("species_")}
            )
            spatial_spec = dataclasses.replace(
                spec,
                names=tuple(name for name in spec.names if name not in response_names),
                periodic=(),
            )
            tables = self._far_field.for_spec(spatial_spec)
            spatial = collect_spatial(
                spatial_spec,
                grid,
                self._problem.transform,
                state,
                z=z,
                flux=flux_of(self._problem.route),
                profiles=self._profiles,
                tables=tables,
                operator=self._problem.linear.separable,
                node_chunk=self.node_chunk,
            )
            return {
                name: np.array(spatial[name]) for name in spatial_spec.names
            } | self._backend_statistics(
                tuple(name for name in spec.names if name in response_names), state, z
            )
        # The weight is a Chebyshev evaluation per column, which is the whole of what
        # the Poynting weighting adds to a statistics call on a CPU (0.053 ms of
        # 0.056 ms on the reduced-M2 case), so a spec that asks for none of the
        # Poynting statistics -- a periodic plasma subset, say -- never
        # evaluates it.
        needed = bool(set(spec.names) & WEIGHT_DEPENDENT_STATISTICS)
        if needed:
            np.copyto(self._statistics_power_weight, power_weight_at(self._problem, z))
        weight_refusal = weight_refusal_for(spec.names, self._statistics_power_weight)
        if weight_refusal is not None:
            raise BackendError(weight_refusal)
        weighted = needed and not np.all(self._statistics_power_weight == 1.0)
        power_names = tuple(
            name for name in spec.names if weighted and name in _POYNTING_STATISTICS
        )
        axis = None
        if "peak_intensity" in power_names:
            if self._modes is None:
                msg = "the peak on-axis intensity needs the mode set"
                raise BackendError(msg)
            axis = self._on_axis_synthesis(self._modes, z)
        power_values = (
            _poynting_statistics(spec, grid, state, self._statistics_power_weight, axis)
            if weighted
            else {}
        )
        owned = tuple(
            name
            for name in spec.names
            if CATALOGUE[name].backend and name not in power_names
        )
        delegated = dataclasses.replace(
            spec,
            names=tuple(
                name
                for name in spec.names
                if name not in owned and name not in power_names
            ),
            periodic=tuple(
                name
                for name in spec.periodic
                if name not in owned and name not in power_names
            ),
        )
        # The basis metric at this plane: the
        # statistics of `METRIC_WEIGHTED_STATISTICS` are functionals of the physical
        # same-frequency flux `a.H W a`, and on a declared change of basis W is not
        # the identity. `metric_at` returns None for an identity metric, which is
        # the statistics' own bitwise-unchanged path, so an ordinary run pays
        # nothing.
        metric = (
            metric_at(self._problem.linear.modal_metric, z)
            if set(delegated.names) & METRIC_WEIGHTED_STATISTICS
            else None
        )
        collected = collect(
            delegated,
            grid,
            SavedPlane(z=z, field=state),
            modes=self._modes,
            profiles=self._profiles,
            metric=metric,
        )
        values = {name: np.array(collected[name]) for name in delegated.names}
        values.update({name: np.array(power_values[name]) for name in power_names})
        return values | self._backend_statistics(owned, state, z)

    def _backend_statistics(
        self,
        names: tuple[str, ...],
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> dict[str, np.ndarray]:
        """Share each fresh envelope population scan among the requested statistics.

        - ``state``: accepted state at this plane.
        - Returns: requested values, with no retained field cache.
        """
        population_names = (
            tuple(
                name
                for name in names
                if name.startswith("species_")
                or name in ("electron_density", "peak_ionisation_rate")
            )
            if self._problem.responses.envelope_plasma is not None
            else ()
        )
        integration_names = tuple(
            name for name in names if name in PLASMA_INTEGRATION_STATISTICS
        )
        axis_names = tuple(name for name in names if name in ON_AXIS_PLASMA_STATISTICS)
        if axis_names and not population_names:
            # The species statistics are the node scan of `_species_statistics` on
            # every route, so sharing one scan with the on-axis statistics changes
            # none of their values and lets a node on the axis supply its row.
            population_names = tuple(
                name for name in names if name.startswith("species_")
            )
        # Node scans of this observation only: a rule with a node on the axis hands
        # its row to the on-axis statistics, and nothing outlives the call, so no
        # stale stage or rejected step can be reused.
        scans: list[tuple[np.ndarray, np.ndarray]] = []
        result = {
            name: self._backend_stat(name, state, z)
            for name in names
            if name not in population_names
            and name not in integration_names
            and name not in axis_names
        }
        if integration_names:
            diagnostics = self._integration_diagnostics(state, z)
            result.update(
                {name: np.array(diagnostics[name]) for name in integration_names}
            )
        if population_names:
            species = self._species_statistics(state, z, scans)
            result.update({name: np.array(species[name]) for name in population_names})
        if axis_names:
            axis = self._axis_statistics(state, scans[0] if scans else None, z)
            result.update({name: np.array(axis[name]) for name in axis_names})
        return result

    def _backend_stat(
        self,
        name: str,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> np.ndarray:
        """Compute one backend-owned statistic."""
        if name == "quadrature_error":
            return np.full((), self._alarm)
        if name == "peak_intensity":
            return np.array(self._peak_intensity(state, z))
        if (
            name.startswith("species_")
            or self._problem.flags.n_channels > 1
            or self._problem.responses.envelope_plasma is not None
        ):
            return np.array(self._species_statistics(state, z)[name])
        return np.array(
            self._ionisation(state, z)[0 if name == "electron_density" else 1]
        )

    def _species_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        scans: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate independent populations on the actual nonlinear nodes.

        - ``scans``: where to leave this observation's channel fractions and peak rates
          at every node, for the on-axis statistics to reuse.
        - Returns: the five species statistics and the two total plasma statistics.
        """
        responses = self._problem.responses
        transform, grids = self._problem.transform, self._problem.grids
        channels = responses.population_channels
        if channels:
            if transform.geometry is not None and transform.geometry.kind == "free":
                nx, ny = transform.geometry.shape
                physical = np.fft.ifft2(
                    field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
                ).reshape(field.shape)
                nodes = passes.to_time(
                    physical * grids.pad_scale,
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
            elif transform.geometry is not None and transform.geometry.kind == "slab":
                # The slab synthesis carries the pad scale already.
                nodes = passes.to_time(
                    slab_synthesis(transform, field),
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
            else:
                nodes = self._nodes_at(field, z)
            if responses.envelope_plasma is not None:
                fractions, peaks = self._envelope_populations(
                    nodes, tuple(float(c.preionisation(z)) for c in channels)
                )
                if scans is not None:
                    scans.append((fractions, peaks))
                return population_statistics(self._problem, z, fractions, peaks)
            rates, population = passes.plasma_channel_fractions(
                nodes,
                responses.plasma_channels,
                tuple(float(c.preionisation(z)) for c in channels),
                grids.dt_fine,
                responses.plasma_integral,
            )
            if scans is not None:
                scans.append((population[:, :, -1], rates.max(axis=-1)))
            return population_statistics(
                self._problem, z, population[:, :, -1], rates.max(axis=-1)
            )
        empty = np.zeros((0, transform.n_nodes))
        return population_statistics(self._problem, z, empty, empty)

    def _integration_diagnostics(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"], z: float
    ) -> dict[str, np.ndarray]:
        """Return the plasma column's integration diagnostics at this plane (SEP 18).

        The undershoot of the exposure increments, the relative work defect of the
        loss current and the boundary level of the free-current integrand, over every
        node and field plasma channel
        (:data:`solprop.stats.PLASMA_INTEGRATION_STATISTICS`).

        - Returns: the three statistics, each zero-dimensional and dimensionless.
        """
        responses = self._problem.responses
        channels = responses.plasma_channels
        if not channels:
            msg = (
                "the plasma integration diagnostics read the field-grid plasma "
                "column, and this problem carries none"
            )
            raise BackendError(msg)
        initial = tuple(float(channel.preionisation(z)) for channel in channels)
        parts = np.zeros(6)
        for nodes in self._plasma_nodes(field, z):
            parts = np.maximum(
                parts,
                passes.plasma_integration_diagnostics(
                    nodes,
                    channels,
                    initial,
                    self._problem.grids.dt_fine,
                    responses.plasma_integral,
                ),
            )
        undershoot, defect, boundary = passes.plasma_integration_statistics(parts)
        return {
            "plasma_undershoot": np.asarray(undershoot),
            "plasma_work_defect": np.asarray(defect),
            "plasma_boundary_level": np.asarray(boundary),
        }

    def _plasma_nodes(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"], z: float
    ) -> Iterator[Float[np.ndarray, "n_nodes_chunk n_pol n_to"]]:
        """Yield the real node fields the plasma column is evaluated on, by chunk.

        The same synthesis as :meth:`_ionisation`: the physical nodes of a separable
        geometry, chunk by chunk, or the modal quadrature nodes at once.

        - Yields: node fields on the fine time grid, V/m.
        """
        grids = self._problem.grids
        if self._problem.linear.separable is None:
            yield self._nodes_at(field, z)
            return
        geometry = self._problem.transform.geometry
        assert geometry is not None
        nx, ny = geometry.shape
        physical = (
            slab_synthesis(self._problem.transform, field) / grids.pad_scale
            if geometry.kind == "slab"
            else np.fft.ifft2(
                field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
            ).reshape(field.shape)
        )
        chunk = self.node_chunk
        for first in range(0, len(physical), chunk):
            yield passes.to_time(
                physical[first : first + chunk] * grids.pad_scale,
                grids.crop_index,
                grids.n_to,
                kind=self._kind,
            )

    def _ionisation(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the electron density and the peak ionisation rate at this plane.

        Both are read at the **node of largest ionised fraction**, which is where the
        plasma the run carries is densest and the only node either number can honestly
        describe; for a mode-averaged run that is the one node. The axis is not a
        quadrature node at all and is not where the polarisation was evaluated, so the
        field is not re-synthesised there.

        The electron density is ``density(z)`` times the fraction at the **end** of the
        window: what the pulse leaves behind, not what it reaches in passing.

        - Returns: the electron density in m^-3 and the peak rate in s^-1, both
          zero-dimensional.
        """
        responses = self._problem.responses
        plasma = responses.plasma_coefficients
        if plasma is None:
            msg = (
                "the electron density and the peak ionisation rate need a compiled "
                "plasma response, and this problem declares none"
            )
            raise BackendError(msg)
        if self._problem.linear.separable is not None:
            geometry = self._problem.transform.geometry
            assert geometry is not None
            nx, ny = geometry.shape
            physical = (
                # The slab synthesis carries the pad scale, multiplied back below.
                slab_synthesis(self._problem.transform, field)
                / self._problem.grids.pad_scale
                if geometry.kind == "slab"
                else np.fft.ifft2(
                    field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
                ).reshape(field.shape)
            )
            largest, peak = -1.0, 0.0
            grids = self._problem.grids
            chunk = self.node_chunk
            for first in range(0, len(physical), chunk):
                nodes = passes.to_time(
                    physical[first : first + chunk] * grids.pad_scale,
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
                rate = passes.ionisation_rate(nodes, plasma.rate)
                fraction, _ = passes.plasma_scan(
                    nodes,
                    rate,
                    plasma,
                    preionisation=float(plasma.preionisation(z)),
                    dt=grids.dt_fine,
                    integral=responses.plasma_integral,
                )
                densest = int(np.argmax(fraction[:, -1]))
                if fraction[densest, -1] > largest:
                    largest = float(fraction[densest, -1])
                    peak = float(np.max(rate[densest]))
            return np.asarray(float(plasma.density(z)) * largest), np.asarray(peak)
        nodes = self._nodes_at(field, z)
        rate = passes.ionisation_rate(nodes, plasma.rate)
        fraction, _ = passes.plasma_scan(
            nodes,
            rate,
            plasma,
            preionisation=float(plasma.preionisation(z)),
            dt=self._problem.grids.dt_fine,
            integral=responses.plasma_integral,
        )
        densest = int(np.argmax(fraction[:, -1]))
        density = float(plasma.density(z)) * float(fraction[densest, -1])
        return np.asarray(density), np.asarray(float(np.max(rate[densest])))

    def _peak_intensity(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> np.ndarray:
        """Return the largest intensity on the axis of the waveguide, in W/m^2.

        ``eps0 c/2 max_t sum_c |E_{a,c}(0, t)|^2`` with the analytic field synthesised
        at ``r = 0`` from the modes' own profiles, both Cartesian components. The axis
        is not a quadrature node, so the mode fields are evaluated there rather than
        interpolated, and both components are taken so that the answer does not depend
        on which one the transform resolves or the state carries. One definition for
        every run: a mode-averaged run synthesises its one mode the same way, which is
        ``P / A_eff`` for a top-hat profile and the true on-axis value for a real one
        -- 1.77 times the average for HE_11, so ``P / A_eff`` would under-report it by
        that factor. The on-axis matrix is built once and rescaled with the taper
        (:meth:`_on_axis_synthesis`), which is what makes this statistic cheap enough
        for the between-planes channel.
        """
        if self._modes is None:
            msg = (
                "the peak on-axis intensity needs the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                "NumpyRefBackend(problem, request, modes=modes)"
            )
            raise BackendError(msg)
        analytic = analytic_field(self._require_grid(), field)
        columns = np.tensordot(
            self._on_axis_synthesis(self._modes, z), analytic, axes=([1], [0])
        )
        power = np.sum(np.abs(columns) ** 2, axis=tuple(range(columns.ndim - 1)))
        return np.asarray(0.5 * EPS0 * C * float(np.max(power)))

    def _nodes_at(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> Num[np.ndarray, "n_nodes n_components n_to"]:
        """Return the physical field at every quadrature node on the fine time grid.

        - Returns: the field in V/m, shape ``(n_nodes, n_components, n_to)``. Here
          ``n_components = transform.n_pol * n_pol`` counts physical components,
          independently of the state's polarisation axis.
        """
        grids, transform = self._problem.grids, self._problem.transform
        taper = (
            1.0 if transform.taper_scale is None else float(transform.taper_scale(z))
        )
        modal_t = passes.to_time(field, grids.crop_index, grids.n_to, kind=self._kind)
        return passes.synthesise(modal_t, transform.synthesis, taper, transform.n_pol)

    def _axis_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        scan: tuple[np.ndarray, np.ndarray] | None,
        z: float,
    ) -> dict[str, np.ndarray]:
        """Return the four on-axis plasma statistics of the state (SEP 19).

        A rule with a node on the axis hands over that node's row from this
        observation's node scan; otherwise the axis row is scanned once
        (:func:`solprop.backends.numpy_ref.populations.axis_populations`).

        - ``field``: the state at this plane, shape (n_modes, n_pol, n_omega), or
          ``None`` from a backend whose :meth:`_axis_row_populations` reads its own
          state.
        - ``scan``: this observation's channel fractions and peak rates at every node,
          when the node statistics already scanned them.
        - Returns: the four ``on_axis_*`` statistics.
        """
        if self._modes is None:
            msg = (
                "the on-axis plasma statistics need the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                f"select({self.name!r}, problem, request, modes=modes)"
            )
            raise BackendError(msg)
        from solprop.backends.populations import (  # noqa: PLC0415 - a cycle
            axis_node,
            axis_statistics,
        )

        if not self._axis_node_known:
            self._axis_node_known = True
            self._axis_node = axis_node(
                self._problem,
                mode_matrix(
                    self._modes,
                    on_axis_rule(self._modes[0].dimlimits(0.0)[0]),
                    0.0,
                    components=("x", "y"),
                ).synthesis,
            )
        if scan is not None and self._axis_node is not None:
            fractions, peaks = scan
            node = self._axis_node
            return axis_statistics(self._problem, z, fractions[:, node], peaks[:, node])
        self.axis_scans += 1
        fractions, peaks = self._axis_row_populations(
            field, self._on_axis_synthesis(self._modes, z), z
        )
        return axis_statistics(self._problem, z, fractions, peaks)

    def _axis_row_populations(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        axis: Float[np.ndarray, "n_rows n_modes"],
        z: float,
    ) -> tuple[Float[np.ndarray, " n_channels"], Float[np.ndarray, " n_channels"]]:
        """Scan this plane's axis row: one row, every channel.

        - ``axis``: the mode set's synthesis on the axis at this position
          (:meth:`_on_axis_synthesis`).
        - Returns: the final ionised fraction and the largest rate on the axis, per
          channel.
        """
        if field is None:
            raise BackendError("the on-axis statistics scan needs the field")
        from solprop.backends.numpy_ref.populations import (  # noqa: PLC0415 - a cycle
            axis_populations,
        )

        return axis_populations(self._problem, field, z, axis)

    def _on_axis_synthesis(
        self, modes: Sequence[Mode], z: float
    ) -> Float[np.ndarray, "n_rows n_modes"]:
        """Return the mode set's synthesis matrix on the axis, at the current position.

        Built once at ``z = 0`` and rescaled by the transform's own taper factor: the
        axis is the single node ``r = 0`` of
        :func:`solprop.modes.quadrature.on_axis_rule`, so the law that rescales the
        transform's matrices -- a scale-invariant profile goes as ``1/a`` and its
        normalisation as ``a^2`` -- applies to it exactly, and a set that is neither
        constant nor scale invariant never reaches a backend (``translate.modal``).
        Rebuilding the matrix at every call was what kept ``peak_intensity`` out of the
        between-planes channel.

        - Returns: shape ``(n_rows, n_modes)``, one row per Cartesian component on the
          axis.
        """
        if self._on_axis is None:
            rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
            matrices = mode_matrix(modes, rule, 0.0, components=("x", "y"))
            self._on_axis = np.ascontiguousarray(matrices.synthesis)
        taper = self._problem.transform.taper_scale
        if taper is None:
            return self._on_axis
        return self._on_axis * float(taper(z))

    def _require_grid(self) -> TimeGrid:
        """Return the grid the state lives on, or say why statistics cannot be taken."""
        if self._grid is None:
            msg = (
                "statistics need the grid the state lives on, which the kernel "
                f"contract does not carry; construct the backend with select("
                f"{self.name!r}, problem, request, grid=grid)"
            )
            raise BackendError(msg)
        return self._grid

    def nonlinear(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through the transform passes.

        The pipeline of ``architecture.md`` section 8 for any node count: one batched
        inverse transform of every modal column, then, per chunk of
        :attr:`node_chunk` nodes, the synthesis GEMM, the fused response pass and
        the projection GEMM accumulated over the chunks, then one batched forward
        transform, the crop and the prefactor. Mode-averaged propagation is the
        one-node, one-column case of exactly this code.

        Parameters
        ----------
        lab : numpy.ndarray
            The stage's lab-frame field, in the state's layout.
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.
        inverse_propagator : numpy.ndarray
            The factor that carries the derivative back to the step's anchor.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}, optional
            The host source's accounting label; ignored when none is bound.

        Returns
        -------
        numpy.ndarray
            The stage derivative in the anchor's frame.
        """
        responses = self._problem.responses
        grids = self._problem.grids
        if self._tdse_binding is not None:
            return self._nonlinear_tdse(lab, bundle, inverse_propagator, purpose)
        if self._problem.linear.separable is not None:
            geometry = self._problem.transform.geometry
            evaluate = (
                self._nonlinear_slab
                if geometry is not None and geometry.kind == "slab"
                else self._nonlinear_free
            )
            return evaluate(lab, bundle, inverse_propagator)
        if self._general is not None:
            return self._nonlinear_general(lab, bundle, inverse_propagator)
        if len(responses.response_order) == 1:
            # Nothing but the accumulation was compiled, so the polarisation is
            # identically zero and the step is the exact linear flow. Running the
            # transforms to multiply by nothing would only add round-off -- unless
            # ADR-025's in-stage absorber is active, which is a term of its own and
            # would otherwise be dropped from a problem that has no response at all.
            if not self._in_stage_absorber:
                return np.zeros_like(lab)
            alone = passes.to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
            return passes.stage_output(
                np.zeros_like(lab),
                np.zeros_like(lab),
                self._tables.omega_window,
                inverse_propagator,
                self._absorber_term(alone, may_quiet=True),
            )
        modal_t = passes.to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
        geometry = self._problem.transform.geometry
        if (
            geometry is not None
            and geometry.kind == "radial"
            and bool(responses.plasma_channels)
            and not self._gated_plasma
        ):
            spectrum = self._radial_plasma_spectrum(modal_t, bundle)
        else:
            projected, coarse = self._over_nodes(modal_t, bundle)
            self._last_projected = projected
            self._last_coarse = coarse
            self._last_bundle = bundle
            spectrum = passes.to_frequency(projected, grids.crop_index, kind=self._kind)
            if self._last_plasma is not None:
                gated, edge, _, _ = self._last_plasma
                spectrum += self._held_streams(gated, edge)
            if responses.envelope_plasma is not None:
                assert self._last_envelope is not None
                drive, loss, _, _ = self._last_envelope
                factors = responses.envelope_plasma.factors
                spectrum += factors.drude * passes.to_frequency(
                    drive, grids.crop_index, kind=self._kind
                )
                spectrum += factors.loss * passes.to_frequency(
                    loss, grids.crop_index, kind=self._kind
                )
        return passes.stage_output(
            spectrum,
            self.prefactor(bundle),
            self._tables.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t, may_quiet=True),
        )

    def _nonlinear_tdse(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
        purpose: TDSEPurpose,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate the declared atomic binding and combine independent P/J sources.

        Steps 1 to 4 are :mod:`solprop.backends.tdse_source`'s, shared with native;
        the stage output is this backend's own.
        """
        problem = self._problem
        grids = problem.grids
        binding = self._tdse_binding
        assert binding is not None and tdse_of(problem.route) is not None
        modal_t = passes.to_time(lab, grids.crop_index, grids.n_to, kind="field")
        fields = tdse_source.node_fields(problem, lab, bundle)
        result = binding.evaluate(
            fields, z=bundle.z1, purpose=purpose, density=bundle.density
        )
        polarisation, current = tdse_source.atomic_sources(
            fields, result, problem, bundle
        )
        source, projection = tdse_source.project_atomic_sources(
            polarisation,
            current,
            result,
            problem,
            bundle,
            self.prefactor(bundle),
            edge=tdse_source.atomic_edge_source(result, problem, bundle),
        )
        self._tdse_last_purpose, self._tdse_last_z = purpose, bundle.z1
        self._tdse_last_projection = projection
        return passes.stage_output(
            source,
            np.ones_like(source),
            grids.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t),
        )

    def _nonlinear_general(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through a general modal guide's maps.

        The route's own execution path, in
        :mod:`solprop.backends.numpy_ref.general_modal`, beside the fused pass and
        not an extension of it. The synthesis is a complex
        frequency-dependent map of three Cartesian components, so it happens in the
        frequency domain and the field a response sees is the analytic signal. What
        is shared with the legacy path is everything after the projection -- the
        prefactor, which is the crop scale alone here, the spectral collar, the
        in-stage absorber and the return to the interaction picture.

        The quadrature alarm is not scored on this route: the general node rule has
        no embedded coarse pair, and a difference that is identically zero read as
        convergence would be wrong, as for any unpaired rule.

        A guide with no responding region is short-circuited, as the legacy path
        short-circuits a problem whose `response_order` is the accumulation alone.
        """
        assert self._general is not None
        grids = self._problem.grids
        # A guide whose every material region is linear has an identically zero
        # tensor, so the source is zero for every state and every position: the
        # synthesis, the 3 x n_nodes transforms, the contraction and the projection
        # would spend a tenth of a second per step producing it (measured 108 ms on
        # the 288-node strand). ADR-025's in-stage absorber is a term of its own and
        # is not dropped with it.
        spectrum = (
            general_source(
                lab,
                self._general,
                grids.crop_index,
                n_to=grids.n_to,
                pad_scale=grids.pad_scale,
                transform_weights=bundle.transform_weights,
                dispersive_densities=bundle.dispersive_kerr_densities,
            )
            if self._general_responds
            else np.zeros_like(lab)
        )
        absorber = None
        if self._in_stage_absorber:
            absorber = self._absorber_term(
                passes.to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
            )
        return passes.stage_output(
            spectrum,
            self.prefactor(bundle),
            grids.omega_window,
            inverse_propagator,
            absorber,
        )

    def _column_sources(
        self,
        field_t: np.ndarray,
        bundle: StageScalars,
        raman: np.ndarray | None,
    ) -> passes.NonlinearSources:
        """Evaluate one column; the historical integral is benchmark-only.

        - ``field_t``: physical node fields in V/m, with time last.
        - ``raman``: damped Raman convolution spectrum, when required by the response.
        - Returns: the collared local polarisation in C/m² and the ungated current in
          A/m².
        """
        responses, grids = self._problem.responses, self._problem.grids
        sources = passes.nonlinear_sources(
            field_t,
            responses,
            self._tables.t_window,
            kind=self._kind,
            density=bundle.density,
            preionisation=bundle.preionisation,
            dt=grids.dt_fine,
            carrier=self._carrier,
            raman_spectrum=raman,
            scalars=bundle,
        )
        if not self._gated_plasma or sources.current is None:
            return sources
        return passes.NonlinearSources(
            sources.polarisation
            + self._tables.t_window
            * responses.plasma_integral.integrate(sources.current, grids.dt_fine),
            None,
        )

    def _held_streams(self, gated: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Transform and combine the current products in polarisation units.

        - ``gated``, ``edge``: fine-time ``gJ`` and ``g'J`` rows, optionally projected
          before this call.
        - Returns: cropped raw-DFT polarisation-equivalent source, zero on inactive
          bins.
        """
        grids = self._problem.grids
        return passes.held_plasma_spectrum(
            gated, edge, grids.omega, grids.crop_index, grids.active_mask
        )

    def _held_current(self, current: np.ndarray) -> np.ndarray:
        """Form and transform the two collared products of a nodal current.

        - ``current``: the ungated nodal current in A/m², fine time last.
        - Returns: cropped raw-DFT polarisation-equivalent source at the nodes.
        """
        return self._held_streams(
            self._tables.t_window * current, self._tables.t_window_derivative * current
        )

    def _radial_plasma_spectrum(
        self,
        modal_t: np.ndarray,
        bundle: StageScalars,
    ) -> np.ndarray:
        """Transform three nodal streams, combine, then project once in frequency."""
        problem = self._problem
        transform, grids = problem.transform, problem.grids
        spectrum = np.zeros(problem.shape, dtype=np.complex128)
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        for first in range(0, transform.n_nodes, self.node_chunk):
            rows = slice(
                first * transform.n_pol,
                (first + self.node_chunk) * transform.n_pol,
            )
            field = passes.synthesise(
                modal_t,
                self._tables.synthesis[rows],
                bundle.taper_scale,
                transform.n_pol,
            )
            sources = self._column_sources(field, bundle, raman)
            assert sources.current is not None
            nodal = passes.to_frequency(
                sources.polarisation, grids.crop_index, kind="field"
            )
            nodal += self._held_current(sources.current)
            spectrum += passes.project(
                nodal,
                self._tables.projection[:, rows],
                bundle.taper_scale,
                transform.n_pol,
            )
        return spectrum

    def _nonlinear_free(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Apply transverse FFTs before bounded temporal response transforms.

        - ``lab``: Lab-frame stage spectrum.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        if geometry is None:
            msg = "Cartesian nonlinear evaluation requires spatial geometry"
            raise BackendError(msg)
        shape = (geometry.shape[1], geometry.shape[0], *lab.shape[1:])
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        scratch = np.empty_like(lab)
        if len(responses.response_order) > 1:
            scratch.reshape(shape)[...] = np.fft.ifft2(lab.reshape(shape), axes=(0, 1))
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            for first in range(0, transform.n_nodes, self.node_chunk):
                rows = slice(first, first + self.node_chunk)
                field_t = grids.pad_scale * passes.to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                sources = self._column_sources(field_t, bundle, raman)
                polarisation_t = sources.polarisation
                scratch[rows] = passes.to_frequency(
                    polarisation_t, grids.crop_index, kind=self._kind
                )
                if sources.current is not None:
                    scratch[rows] += self._held_current(sources.current)
                if responses.envelope_plasma is not None:
                    drive, loss = self._envelope_sources(field_t, bundle)
                    factors = responses.envelope_plasma.factors
                    scratch[rows] += factors.drude * passes.to_frequency(
                        drive, grids.crop_index, kind=self._kind
                    )
                    scratch[rows] += factors.loss * passes.to_frequency(
                        loss, grids.crop_index, kind=self._kind
                    )
            view = scratch.reshape(shape)
            view[...] = np.fft.fft2(view, axes=(0, 1))
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_nodes, self.node_chunk):
                rows = slice(first, first + self.node_chunk)
                _, prefactor = separable_values(data, rows)
                result[rows] = prefactor * self._tables.omega_window * scratch[rows]
        if self._in_stage_absorber:
            # The rate is additive to the normalised material term. Reusing the
            # transverse scratch avoids a second retained physical spectral volume.
            scratch.reshape(shape)[...] = np.fft.ifft2(lab.reshape(shape), axes=(0, 1))
            temporal = self._tables.temporal_rate
            timed = temporal is not None and responses.absorber_scheme == "nonlinear"
            for first in range(0, transform.n_nodes, self.node_chunk):
                rows = slice(first, first + self.node_chunk)
                rate = geometry.absorber_rate[rows, None, None]
                if not timed:
                    # A rate of the node alone commutes with the temporal transforms:
                    # a spatial-only pass needs no temporal pair (SEP 40 section 4.7).
                    scratch[rows] = -0.5 * rate * scratch[rows]
                    continue
                field_t = passes.to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                scratch[rows] = passes.to_frequency(
                    -0.5 * (rate + temporal) * field_t,
                    grids.crop_index,
                    kind=self._kind,
                )
            view = scratch.reshape(shape)
            view[...] = np.fft.fft2(view, axes=(0, 1))
            result += grids.active_mask * scratch
        return result * inverse_propagator

    def _nonlinear_slab(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one hybrid planar stage: slab synthesis in x, an FFT in y.

        The ``(p, k_y)`` columns go to ``(p, y)`` rows by the inverse DFT in y, and
        the rows to time -- a row is a slab amplitude, whose time signal is a real
        field's on a field grid -- so that, as on the modal route, only ``n_p * n_y``
        temporal transforms are taken each way. In time, the slab synthesis (which
        carries the pad scale) gives the field at the ``(x, y)`` nodes, the responses
        are the Cartesian ones node by node, and the slab rule's dual basis projects
        the polarisation back onto the rows; the rows return to frequency and to
        ``(p, k_y)``, and each column takes its own prefactor from the separable leaf
        (kernel schema 15). The y collar's in-stage rate depends on y alone, so it
        acts on the rows directly.

        - ``lab``: Lab-frame stage spectrum.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rows_f = slab_rows(transform, lab)
        temporal = responses.temporal_absorber_fine
        timed = temporal is not None and responses.absorber_scheme == "nonlinear"
        rows_t = (
            passes.to_time(rows_f, grids.crop_index, grids.n_to, kind=self._kind)
            if len(responses.response_order) > 1 or timed
            else None
        )
        if len(responses.response_order) > 1:
            assert rows_t is not None
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            envelope = responses.envelope_plasma is not None
            plasma = bool(responses.plasma_channels) and not self._gated_plasma
            projected = np.zeros_like(rows_t)
            # The envelope plasma's Drude drive and loss current, projected as the
            # polarisation is; empty unless the response is present.
            drives = [
                np.zeros_like(rows_t) for _ in range(2 if envelope or plasma else 0)
            ]
            # Whole y rows at a time, as many as the node chunk holds.
            block = max(1, self.node_chunk // n_x)
            for first in range(0, n_y, block):
                rows = slice(first * n_p, min(first + block, n_y) * n_p)
                count = (rows.stop - rows.start) // n_p
                field_t = np.matmul(
                    transform.synthesis, rows_t[rows].reshape(count, n_p, -1)
                ).reshape(count * n_x, *rows_t.shape[1:])
                sources = self._column_sources(field_t, bundle, raman)
                current = sources.current
                source_rows = [sources.polarisation]
                if envelope:
                    assert responses.envelope_plasma is not None
                    source_rows.extend(
                        envelope_sources(
                            field_t,
                            responses.envelope_plasma,
                            bundle,
                            grids.dt_fine,
                            grids.t_window_fine,
                        )
                    )
                if current is not None:
                    source_rows.extend(
                        (
                            grids.t_window_fine * current,
                            grids.t_window_fine_derivative * current,
                        )
                    )
                for target, source in zip(
                    (projected, *drives), source_rows, strict=True
                ):
                    target[rows] = np.matmul(
                        transform.projection, source.reshape(count, n_x, -1)
                    ).reshape(count * n_p, *rows_t.shape[1:])
            spectrum = passes.to_frequency(projected, grids.crop_index, kind=self._kind)
            if envelope:
                assert responses.envelope_plasma is not None
                factors = responses.envelope_plasma.factors
                spectrum += factors.drude * passes.to_frequency(
                    drives[0], grids.crop_index, kind=self._kind
                )
                spectrum += factors.loss * passes.to_frequency(
                    drives[1], grids.crop_index, kind=self._kind
                )
            if plasma:
                spectrum += self._held_streams(drives[0], drives[1])
            columns = slab_columns(transform, spectrum)
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_modes, self.node_chunk):
                rows = slice(first, first + self.node_chunk)
                _, prefactor = separable_values(data, rows)
                result[rows] = prefactor * grids.omega_window * columns[rows]
        if self._in_stage_absorber:
            result += self._slab_absorber(rows_f, rows_t)
        return result * inverse_propagator

    def _slab_absorber(
        self,
        rows_f: Complex[np.ndarray, "n_modes n_pol n_omega"],
        rows_t: Inexact[np.ndarray, "n_modes n_pol n_to"] | None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the planar in-stage collar's term on the ``(p, k_y)`` columns.

        The collar's rate depends on y alone, so it acts on the ``(p, y)`` rows. With
        no temporal rate in-stage it commutes with the temporal transforms, and the
        pass needs no temporal pair (SEP 40 section 4.7).

        - ``rows_t``: the same rows in time; required when a temporal rate applies
          in-stage.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rate = np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)[
            :, None, None
        ]
        temporal = responses.temporal_absorber_fine
        if temporal is not None and responses.absorber_scheme == "nonlinear":
            assert rows_t is not None
            term = passes.to_frequency(
                -0.5 * (rate + temporal) * rows_t, grids.crop_index, kind=self._kind
            )
        else:
            term = -0.5 * rate * rows_f
        return grids.active_mask * slab_columns(transform, term)

    def _absorber_term(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        *,
        may_quiet: bool = False,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the in-stage absorber's contribution, or ``None`` in the other scheme.

        On the modal stage (`may_quiet`) a temporal collar alone is left out while the
        boundary monitor finds it quiet (SEP 21): the term is then below
        :data:`solprop.monitor.QUIET`, and its transform is the stage's largest cost
        after the response's.

        - Returns: the absorber's term on the coarse grid, or ``None`` when the
          ``"split_step"`` scheme is active and the rate is applied to the state after
          the step instead.
        """
        rate = self._tables.temporal_rate
        if not self._in_stage_absorber:
            return None
        if may_quiet and self.collar_quiet and not self._spatial_absorber.size:
            return None
        grids = self._problem.grids
        term = np.zeros_like(modal_t)
        if rate is not None and self._problem.responses.absorber_scheme == "nonlinear":
            term = -0.5 * rate * modal_t
        if self._spatial_absorber.size:
            term = term + (
                self._spatial_absorber @ modal_t.reshape(modal_t.shape[0], -1)
            ).reshape(modal_t.shape)
        return (
            passes.to_frequency(term, grids.crop_index, kind=self._kind)
            * grids.active_mask
        )

    def _over_nodes(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        bundle: StageScalars,
    ) -> tuple[
        Num[np.ndarray, "n_modes n_pol n_to"],
        Num[np.ndarray, "n_modes_c n_pol_c n_to_c"] | None,
    ]:
        """Run the response pass over the quadrature in chunks and project it back.

        The chunk loop of ``architecture.md`` section 13: the space-domain buffers are
        sized by :attr:`node_chunk` and not by the node count, so a chunk of one
        and a chunk of every node differ only in round-off. The embedded coarse rule is
        projected in the same loop, from the same node polarisation, because that is
        what makes the quadrature alarm free: the two rules share every node the coarse
        one has.

        - Returns: the modal polarisation on the fine time grid, and the same under the
          embedded coarse rule, or ``None`` when the quadrature has no embedded pair
          and the estimate would be identically zero.
        """
        responses, grids = self._problem.responses, self._problem.grids
        transform = self._problem.transform
        taper = bundle.taper_scale
        # Once per stage rather than once per chunk: the kernel spectrum does not depend
        # on the nodes, and along a gradient it is one real transform of 2 n_to against
        # n_modes n_pol transforms of n_to that the stage costs anyway.
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        n_pol = transform.n_pol
        projected = np.zeros(
            (transform.n_modes, self._problem.shape[1], grids.n_to),
            dtype=modal_t.dtype,
        )
        coarse = None if self._embedded is None else np.zeros_like(projected)
        envelope = self._problem.responses.envelope_plasma
        envelope_projected = (
            None
            if envelope is None
            else (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
        )
        plasma_projected = (
            (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
            if bool(responses.plasma_channels) and not self._gated_plasma
            else None
        )
        for first in range(0, transform.n_nodes, self.node_chunk):
            rows = slice(first * n_pol, (first + self.node_chunk) * n_pol)
            field_t = passes.synthesise(
                modal_t, self._tables.synthesis[rows], taper, n_pol
            )
            sources = self._column_sources(field_t, bundle, raman)
            polarisation_t = sources.polarisation
            projected = projected + passes.project(
                polarisation_t, self._tables.projection[:, rows], taper, n_pol
            )
            if coarse is not None and self._embedded is not None:
                coarse = coarse + passes.project(
                    polarisation_t, self._embedded[:, rows], taper, n_pol
                )
            if sources.current is not None and plasma_projected is not None:
                gated, edge, coarse_gated, coarse_edge = plasma_projected
                for target, reduced, window in (
                    (gated, coarse_gated, self._tables.t_window),
                    (edge, coarse_edge, self._tables.t_window_derivative),
                ):
                    current = window * sources.current
                    target[...] += passes.project(
                        current, self._tables.projection[:, rows], taper, n_pol
                    )
                    if reduced is not None and self._embedded is not None:
                        reduced[...] += passes.project(
                            current, self._embedded[:, rows], taper, n_pol
                        )
            if envelope is not None and envelope_projected is not None:
                drive, loss = self._envelope_sources(field_t, bundle)
                full_drive, full_loss, coarse_drive, coarse_loss = envelope_projected
                full_drive += passes.project(
                    drive, self._tables.projection[:, rows], taper, n_pol
                )
                full_loss += passes.project(
                    loss, self._tables.projection[:, rows], taper, n_pol
                )
                if (
                    self._embedded is not None
                    and coarse_drive is not None
                    and coarse_loss is not None
                ):
                    coarse_drive += passes.project(
                        drive, self._embedded[:, rows], taper, n_pol
                    )
                    coarse_loss += passes.project(
                        loss, self._embedded[:, rows], taper, n_pol
                    )
        self._last_plasma = plasma_projected
        self._last_envelope = envelope_projected
        return projected, coarse

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> Complex[np.ndarray, "... n_omega_doubled"] | None:
        """Return the damped Raman kernel spectrum for a set of rates, anew.

        Memoised on the **decay rates** rather than on the density, because the rates
        are what the spectrum depends on: a glass's are zero at every density, so its
        kernel is transformed once for the whole run, and a molecular gas on a gradient
        presents new rates at every stage and is transformed once per stage. Only the
        most recent set is kept, which is the right choice here -- a growing cache would
        hold thousands of spectra and hit none.

        - ``rates``: the decay rate of each term at the stage's position, s^-1, off the
          bundle.
        - ``densities``: component amplitude densities in m^-3, in Raman-component
          order. Multiple components include every amplitude in the cache key; one keeps
          density outside the convolution and retains the original arithmetic.
        - Returns: the spectrum of shape ``(n_to + 1,)``, or ``None`` when no Raman
          kernel was compiled.
        """
        components = self._problem.responses.raman_terms
        if not components:
            return None
        amplitudes = densities if len(components) > 1 else ()
        if (
            self._raman_kernel is None
            or self._raman_rates != rates
            or self._raman_densities != amplitudes
        ):
            if self._problem.transform.n_pol * self._problem.shape[1] == 2:
                self._raman_kernel = passes.tensor_raman_spectrum(
                    components,
                    rates,
                    densities if len(components) > 1 else (1.0,),
                    self._problem.grids.dt_fine,
                )
            elif len(components) == 1:
                compiled = components[0]
                self._raman_kernel = passes.raman_spectrum(
                    compiled.undamped,
                    rates,
                    compiled.fraction,
                    self._problem.grids.dt_fine,
                )
            else:
                self._raman_kernel = passes.mixture_raman_spectrum(
                    components, rates, densities, self._problem.grids.dt_fine
                )
            self._raman_rates = rates
            self._raman_densities = amplitudes
            self._raman_transforms += 1
        return self._raman_kernel

    def exponents(
        self, z: float, dz: float
    ) -> Callable[[float, StageScalars], Complex[np.ndarray, "n_modes n_pol n_omega"]]:
        """Return the memoised ``dPhi`` of a step, keyed by the stage fraction.

        The anchor is hoisted once per step and each exponent is evaluated once and
        negated for the inverse, never evaluated forward and backward independently
        (``docs/dev/stepper.md`` section 4). The propagator mode decides whether the
        exponents are kept for the whole step or evaluated per stage; the exponent
        itself is one linear combination of the terms over the bundle's weights,
        whatever the representation (:mod:`solprop.translate.stage`).

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the step, m.

        Returns
        -------
        callable
            ``(fraction, bundle) -> dPhi``, of the state's broadcast shape.
        """
        del z, dz
        if self._propagator_mode == "recompute":
            return lambda fraction, bundle: self.delta_phi(bundle)  # noqa: ARG005
        cache: dict[float, Complex[np.ndarray, "n_modes n_pol n_omega"]] = {}

        def _exponent(
            fraction: float, bundle: StageScalars
        ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
            """Return ``dPhi`` at a stage fraction, computing it at most once.

            - ``fraction``: the tableau node ``c_i``, dimensionless, in ``[0, 1]``.
            """
            if fraction not in cache:
                cache[fraction] = self.delta_phi(bundle)
            return cache[fraction]

        return _exponent

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the workspace works in, in bytes.

        The reference backend allocates its stage and transform buffers per pass instead
        of keeping a preallocated pool, so this reports what one step has resident at
        its peak -- which is what the optimised backends hold for the whole run, and
        what the budget of ``docs/dev/architecture.md`` section 8 is measured against.
        Every buffer is named even when it is empty.

        The space-domain entries are sized by
        :attr:`node_chunk` and not by the
        node count, which is the whole point of the chunk loop (``architecture.md``
        section 13): holding ``(n_to, n_pol n_nodes)`` for every node at once would be
        210 MB against 45 MB on the comb-to-continuum target.

        Returns
        -------
        dict
            Buffer name to bytes.
        """
        problem = self._problem
        distinct = int(np.unique(np.append(self._stepper.tableau.nodes, 1.0)).size)
        general: dict[str, int] = dict.fromkeys(_GENERAL_LEDGER_ROWS, 0)
        if self._general is not None:
            # The route exists exactly when the tables do (`_general_maps`).
            tables = problem.transform.modal_tables
            assert tables is not None
            general = general_memory(self._general, tables, problem.grids.n_to)
        ledger = adams_memory(problem, self._stepper) | {
            # The rows' one definition, in `solprop.backends.memory`.
            **reference_base_memory(
                problem,
                self._stepper,
                tabulating=self._propagator_mode == "tabulate",
                embedded=self._embedded is not None,
                noise=self._noise is not None,
                statistics_weight=self._statistics_power_weight.nbytes,
                spatial_absorber=self._spatial_absorber.nbytes,
                node_chunk=self.node_chunk,
            ),
            # The coupled flow's own resident arrays: the frozen block of every
            # section with its Schur factors, the residual's terms, and the one
            # forward/inverse action pair held for the section and length last seen.
            # All of them are on the active bins alone.
            "coupled_flow_factors": 0
            if self._flow is None
            else self._flow.vectors.nbytes
            + self._flow.triangular.nbytes
            + self._flow.frozen.nbytes,
            "coupled_flow_residual_terms": 0
            if self._flow is None or self._flow.residual_terms is None
            else self._flow.residual_terms.nbytes,
            "coupled_flow_actions": 0
            if self._flow is None
            else 2 * distinct * self._flow.frozen[0].nbytes,
            # The general modal route's buffers are its own: the two execution maps,
            # the per-node tensor and the record's resident E/H reference are
            # whole-problem allocations no chunk shrinks, and its chunk buffers are
            # three components wide and complex on a grid where the legacy ones are
            # one or two and real. Every row is zero on every
            # other route.
            **general,
            # A low-storage march's state at its last field-free plane (SEP 22, Q14),
            # held from the first such plane; the admission charges it beforehand.
            **(
                {}
                if self._tdse_binding is None
                else {
                    f"tdse_{name}": size
                    for name, size in self._tdse_binding.memory_ledger().items()
                }
            ),
        }
        if self._gated_plasma:
            for name in ledger:
                if (
                    name.startswith("held_current_")
                    and name != "held_current_derivative"
                ):
                    ledger[name] = 0
        return ledger

    def close(self) -> None:
        """Release the buffers. Idempotent."""
        self._state = None
        self._noise = None
        self._raman_kernel = None
        self._last_plasma = None
        self._spatial_absorber = np.empty((0, 0))
        self._raman_rates = None
        self._raman_densities = None
        self._flow_cache = None
        self._statistics_power_weight = np.empty(0)
        self._held = None


@dataclass(frozen=True)
class SourceTables:
    """The problem's tables a source evaluation multiplies by.

    The arrays the passes read, gathered once from the problem.

    Attributes
    ----------
    synthesis : numpy.ndarray
        ``transform.synthesis``.
    projection : numpy.ndarray
        ``transform.projection``.
    t_window : numpy.ndarray
        ``grids.t_window_fine``.
    t_window_derivative : numpy.ndarray
        ``grids.t_window_fine_derivative``.
    omega_window : numpy.ndarray
        ``grids.omega_window``.
    temporal_rate : numpy.ndarray or None
        ``responses.temporal_absorber_fine``.
    """

    synthesis: Float[np.ndarray, "n_rows n_modes_s"]
    projection: Float[np.ndarray, "n_modes n_columns"]
    t_window: Float[np.ndarray, " n_to"]
    t_window_derivative: Float[np.ndarray, " n_to"]
    omega_window: Float[np.ndarray, " n_omega"]
    temporal_rate: Float[np.ndarray, " n_to"] | None

    @classmethod
    def of(cls, problem: PropagationProblem) -> "SourceTables":
        """Return the problem's own tables.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The problem.

        Returns
        -------
        SourceTables
            References to its arrays, uncopied.
        """
        grids = problem.grids
        return cls(
            problem.transform.synthesis,
            problem.transform.projection,
            grids.t_window_fine,
            grids.t_window_fine_derivative,
            grids.omega_window,
            problem.responses.temporal_absorber_fine,
        )


_GENERAL_LEDGER_ROWS = (
    "general_synthesis",
    "general_projection",
    "general_stage_maps",
    "general_tensor",
    "general_electric",
    "general_magnetic",
    "general_backward_electric",
    "general_backward_magnetic",
    "general_reciprocal_gram",
    "general_coordinate_dual",
    "general_scaled_state",
    "general_node_spectrum",
    "general_node_field",
    "general_node_pair",
    "general_node_weighted",
    "general_node_polarisation",
    "general_node_source",
    "general_node_drive",
    "general_node_histories",
    "general_node_drive_spectrum",
    "general_node_delayed",
    "general_delayed_tensor",
    "general_delayed_spectrum",
)
"""The ledger rows of the general modal route, named even when they are empty.

``memory_ledger`` names every buffer on every route, so a problem that is not a
general modal guide reports these as zero rather than omitting them
(:func:`solprop.backends.numpy_ref.general_modal.general_memory` fills them).
"""


def _refuse_inadmissible(problem: PropagationProblem, spec: StatsSpec) -> None:
    """Refuse a statistic this problem's basis does not define, by name.

    The second door of the accessor whitelist (:mod:`solprop.output.capability`),
    after the run loop's own admission: a caller may build a backend and ask it for
    statistics directly, and a general guide's static synthesis matrix is a checked
    **zero** placeholder, so an on-axis intensity taken from one would be zero
    rather than obviously wrong.
    """
    refusal = refusal_for(
        declared_capability(
            general_modal=problem.transform.modal_tables is not None,
            spatial=problem.transform.geometry is not None,
        ),
        spec.names,
        withheld=withheld_accessors(
            general_modal=problem.transform.modal_tables is not None,
            coupled_flow=problem.linear.coupled_flow is not None,
        ),
    )
    if refusal is not None:
        raise BackendError(refusal)
    refusal = axis_refusal(problem, spec.names)
    if refusal is not None:
        raise BackendError(refusal)


def _general_maps(
    problem: PropagationProblem, plan: NodeChunkPlan | None = None
) -> GeneralMaps | None:
    """Derive the general modal route's maps, or return ``None`` for another route.

    The maps are read out of the leaves once, here, because the route propagates
    constant geometry: one interval of one Chebyshev degree along z, so there is
    nothing to evaluate per stage. The chunk is the backend's own memory strategy
    within the plan's budget, or the plan's exact chunk when one was forced.

    - ``plan``: the backend's node-chunk plan; ``None`` plans the automatic one.
    - Returns: the derived maps, or ``None`` when the problem carries no modal tables.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return None
    assert tables.synthesis is not None and tables.projection is not None
    chunk = general_planned_chunk(
        problem, plan_node_chunk(problem) if plan is None else plan
    )
    regional = problem.responses.regional_responses
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    moving = tables.synthesis.shape[0] * tables.synthesis.shape[1] > 1
    maps = GeneralMaps(
        tables.synthesis[0, 0],
        tables.projection[0, 0],
        node_kerr_tensor(tables, regional),
        tables.source_frequency_factor,
        chunk,
        node_kerr_form(regional),
        owner,
        delayed_tensor,
        delayed_spectrum,
        # A moving guide's coefficients, flattened over (n_intervals, n_deg) in the
        # order the stage bundle's transform_weights are.
        tables.synthesis.reshape(-1, *tables.synthesis.shape[2:]) if moving else None,
        tables.projection.reshape(-1, *tables.projection.shape[2:]) if moving else None,
        problem.responses.dispersive_kerr,
        None if problem.responses.dispersive_kerr is None else tables.region_index,
    )
    logger.debug(
        "numpy_ref general modal route: %d channels on %d nodes, chunk %d, "
        "%d fine bins, gauge %s, %s cubic, %d delayed term(s)",
        maps.n_modes,
        maps.n_nodes,
        maps.node_chunk,
        int(tables.omega_fine.size),
        tables.gauge_id,
        maps.form,
        maps.n_terms,
    )
    return maps


@dataclass(frozen=True)
class _CoupledFlow(CoupledFlowFactors):
    """What ``numpy_ref`` holds to execute one coherent block, frozen per section.

    The shared factors (:class:`~solprop.translate.coupled.CoupledFlowFactors`, which
    native reads too), with the section axis leading and only the bins the generator
    is non-zero at held: the rest of the band has an identity action and the state is
    masked to zero there anyway.

    A ``"constant_schur"`` leaf is the one-section case with no residual at all, and
    goes through exactly the constant-section arithmetic. A ``"frozen_residual"`` leaf
    adds `residual_terms`: the coefficients of ``G(z) - C(z)`` along z, flattened over
    ``(n_intervals, n_deg)`` in that order, so that the stage bundle's
    ``flow_weights`` combine them into ``G(z_1) - C(z_1)`` exactly as ``phi_delta``
    combines the terms of ``dPhi`` (``docs/dev/kernel-contract.md`` section 1). The
    explicit residual is that combination minus the frozen block of the step's own
    section. It is applied only on `residual_bins`, the active bins of the requested
    band; in the spectral collar outside them the frozen flow alone carries the state
    (``docs/guide/waveguides/general-modal.md``).

    Attributes
    ----------
    residual_bins : numpy.ndarray
        Which bins the explicit residual is applied at, bool, shape ``(n_omega,)``:
        `active` and the problem's ``grids.inband_mask``, the requested band: in the
        collar beyond it the wall model is outside its validity, and the residual is
        left out whether or not the error norm scores there.
    n_modes : int
        Channels the block couples.
    """

    residual_bins: Bool[np.ndarray, " n_omega_scored"]
    n_modes: int


_TAYLOR_ORDER = 16
"""Degree of the Taylor series a scaled triangular exponential is summed to.

Scaling brings the row-sum norm below :data:`_TAYLOR_SCALE`, so the truncated tail is
at most ``_TAYLOR_SCALE**(n+1)/(n+1)!``, which is 2.1e-20 at this degree -- below the
rounding of the squarings that follow it and of the two unitary products around it.
The alternative, a Pade approximant, buys a lower degree at the cost of a solve per
frequency; the oracle takes the series (``docs/dev/architecture.md`` section 3:
physics and reference code carry no performance tricks).
"""

_TAYLOR_SCALE = 0.5
"""Row-sum norm a block is scaled below before the series is summed."""

_MAX_FLOW_SQUARINGS = 30
"""Largest number of squarings a flow action may need before it is refused.

``2**30`` times the scale is a step whose own generator spans 21 orders of magnitude
in the exponent; a problem that needs it has not been certified by anything the
translation layer measured.
"""


def _flow_squarings(largest: float, *, backend: str = "numpy_ref") -> int:
    """Return how many squarings an exponent of this row-sum norm needs, or refuse.

    The count brings the norm below :data:`_TAYLOR_SCALE`; past
    :data:`_MAX_FLOW_SQUARINGS` the step is refused, as the compiled backends refuse it
    whichever route forms their actions.
    """
    squarings = 0
    if largest > _TAYLOR_SCALE:
        squarings = math.ceil(math.log2(largest / _TAYLOR_SCALE))
    if squarings > _MAX_FLOW_SQUARINGS:
        msg = (
            f"{backend} cannot take a coupled flow action whose exponent has row-sum "
            f"norm {largest:.4g}: it would need {squarings} squarings, above the "
            f"{_MAX_FLOW_SQUARINGS} this backend admits. Cap dz_max"
        )
        raise BackendError(msg)
    return squarings


def _triangular_exponential(
    blocks: Complex[np.ndarray, "n_active_e n_modes_e_out n_modes_e_in"],
    *,
    backend: str = "numpy_ref",
) -> Complex[np.ndarray, "n_active_x n_modes_x_out n_modes_x_in"]:
    """Return ``exp(A)`` for a stack of blocks, by scaling, a series and squaring.

    - Returns: ``exp(A)`` in the same layout.
    """
    largest = float(np.max(np.sum(abs(blocks), axis=-1), initial=0.0))
    squarings = _flow_squarings(largest, backend=backend)
    scaled = blocks / float(2**squarings)
    identity = np.broadcast_to(
        np.eye(blocks.shape[1], dtype=np.complex128), blocks.shape
    )
    series = identity.copy()
    for term in range(_TAYLOR_ORDER, 0, -1):
        series = identity + scaled @ series / term
    for _ in range(squarings):
        series = series @ series
    return series


def _flow_action(
    flow: _CoupledFlow, section: int, length: float, *, backend: str = "numpy_ref"
) -> tuple[
    Complex[np.ndarray, "n_active_f n_modes_f_out n_modes_f_in"],
    Complex[np.ndarray, "n_active_i n_modes_i_out n_modes_i_in"],
]:
    """Return the forward and inverse flow actions of one section over one length.

    ``U = Z exp(l T) Z^H`` and ``U^-1 = Z exp(-l T) Z^H``: the inverse is the matrix
    function of the negated exponent, never a numerical inverse of ``U``, so the
    interaction picture's backward action is as accurate as its forward one and does
    not inherit the conditioning of a solve (``docs/dev/stepper.md`` section 4: no
    unbounded inverse propagation).

    - ``length``: distance in m, positive for the forward action.
    """
    vectors = flow.vectors[section]
    exponent = float(length) * flow.triangular[section]
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    forward = vectors @ _triangular_exponential(exponent, backend=backend) @ adjoint
    inverse = vectors @ _triangular_exponential(-exponent, backend=backend) @ adjoint
    return forward, inverse


def _residual_action(
    flow: _CoupledFlow, section: int, weights: Float[np.ndarray, " n_flow_terms"]
) -> Complex[np.ndarray, "n_active_r n_modes_r_out n_modes_r_in"]:
    """Return the explicit residual of one stage, ``R_s(z_1)``, on the active bins.

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``, evaluated as one linear
    combination of the held terms over the bundle's weights minus the frozen block of
    the **step's** section. The weights are zero outside the section containing the
    stage's own position, so no section's expansion is extrapolated; the frozen block
    is the step's, because the exponential the interaction picture is built on is, and
    freezing is a rearrangement of the same equation whichever block it names.

    - Returns: the residual in 1/m, shape ``(n_active, n_modes, n_modes)``.
    """
    assert flow.residual_terms is not None
    return combine(weights, flow.residual_terms) - flow.frozen[section]


class BlockResidual:
    """``R_s(z) state`` for a held flow's positions (SEP 16 phases 5 and 6).

    A history sample's residual is formed against the section of every trial it serves,
    about six; its action depends only on the section and the sample's flow weights, so
    each pair's action is formed once and kept, the least recently used dropped first:
    each trial's endpoint, at a new position, is used once.

    Parameters
    ----------
    flow : _CoupledFlow
        The held flow.
    """

    _KEPT = 64
    """Most actions kept: well above a history's samples times the sections it spans."""

    def __init__(self, flow: _CoupledFlow) -> None:
        """Hold the flow, with no action formed yet."""
        self._flow = flow
        self._actions: collections.OrderedDict[tuple[int, bytes], np.ndarray] = (
            collections.OrderedDict()
        )

    def __call__(
        self,
        section: int,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        weights: Float[np.ndarray, " n_flow_terms"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return ``R_s(z) state`` for the flow weights of a position.

        Parameters
        ----------
        section : int
            The step's section, whose frozen block the residual is relative to.
        state : numpy.ndarray
            The state the residual acts on.
        weights : numpy.ndarray
            The flow's term weights at the position.

        Returns
        -------
        numpy.ndarray
            The residual's term, zero off the scored active bins.
        """
        key = (section, np.ascontiguousarray(weights).tobytes())
        action = self._actions.get(key)
        if action is None:
            if len(self._actions) >= self._KEPT:
                self._actions.popitem(last=False)
            action = _residual_action(self._flow, section, weights)
            self._actions[key] = action
        else:
            self._actions.move_to_end(key)
        flow = self._flow
        return _residual_drive(state, action, flow.active, flow.residual_bins)


def _residual_drive(
    lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_d n_modes_d_out n_modes_d_in"],
    active: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega_scored"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the explicit residual's contribution to one stage's lab-frame source.

    Unlike :func:`_apply_flow`, which *transforms* a state and therefore leaves the
    bins it does not cover alone, this **adds a term** and is exactly zero off them:
    the residual is zero at every bin the wall formula was never evaluated at, and a
    copy of the state there would enter the source as a derivative of its own size.

    It is also zero on the active bins the error norm does not score, the spectral
    collar. There the residual can reach a spectral radius the
    embedded pair is unstable at, at steps the in-band error allows, and nothing
    would see the growth; the frozen flow, exact at any step, carries those bins
    alone. The block is diagonal in frequency, so no in-band bin reads them through
    it; only the nonlinear source couples them in, at the size of the collar's
    content.

    - ``action``: ``R_s(z_1)`` on the active bins, 1/m.
    - Returns: ``R_s(z_1) u`` in the state's layout, zero off the scored active bins.
    """
    drive = np.zeros_like(lab)
    drive[:, :, scored] = np.einsum(
        "wij,jpw->ipw", action[scored[active]], lab[:, :, scored]
    )
    return drive


def _apply_flow(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_a n_modes_a_out n_modes_a_in"],
    active: Bool[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Apply a per-frequency channel action to the canonical state.

    The identity case is the caller's: a problem with no coupled flow, or the zero
    stage fraction, never reaches here.

    - ``state``: the state, or any array of its layout.
    - Returns: the state with its channel axis mixed on the active bins and untouched
      off them, where the block is zero and the action is the identity.
    """
    carried = state.copy()
    carried[:, :, active] = np.einsum("wij,jpw->ipw", action, state[:, :, active])
    return carried


def _check_flow_conditioning(
    flow: _CoupledFlow,
    section: int,
    pair: tuple[
        Complex[np.ndarray, "n_active_p n_modes_p_out n_modes_p_in"],
        Complex[np.ndarray, "n_active_q n_modes_q_out n_modes_q_in"],
    ],
    dz: float,
    *,
    backend: str = "numpy_ref",
) -> None:
    """Refuse a step whose flow action exceeds the leaf's declared conditioning.

    The certificate row the translation layer declared is
    ``exp(dz_max (mu_max - mu_min))``, a bound on ``norm(U)_2 norm(U^-1)_2``; here the
    two spectral norms are **measured**, from the singular values of the actual
    action, and compared with it. That is the point of a certificate: the row is a
    claim the executor checks rather than a number it trusts, and a step longer than
    the ``dz_max`` the row was declared over is exactly what this catches.

    The inverse is what the interaction picture carries the nonlinear source back by,
    so an unbounded one is an unbounded amplification of that source (the
    ``norm(g(b) - V^-1 N(V b))`` inequality). Local re-anchoring is what keeps it
    bounded at all: the picture is re-anchored at every accepted step, so no inverse
    action is ever taken over more than one of them.

    - ``pair``: the forward and inverse actions over the whole step.
    - ``dz``: length of the step, m, used in the refusal.
    """
    forward, inverse = pair
    if not forward.size:
        return
    measured = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
        inverse, ord=2, axis=(-2, -1)
    )
    _refuse_flow_conditioning(flow, section, measured, dz, backend=backend)


def _refuse_flow_conditioning(
    flow: _CoupledFlow,
    section: int,
    measured: Float[np.ndarray, " n_active_m"],
    dz: float,
    *,
    backend: str = "numpy_ref",
) -> None:
    """Refuse a step whose measured ``norm(U) norm(U^-1)`` exceeds the certificate.

    The decision of :func:`_check_flow_conditioning`, on a measurement made
    elsewhere: a compiled backend measures the two spectral norms in its own
    kernels and hands the per-bin products here, so every backend
    compares them with the declared row by the same allowance and refuses in the same
    words.

    - ``dz``: length of the step, m, used in the refusal.
    """
    declared = flow.condition_bound[section]
    # A bin whose block is normal has a bound of exactly one and a measured product
    # that is one to within the rounding of two spectral norms; the allowance is that
    # rounding and nothing more, because the bound is a true bound on the exact
    # action and any excess above it is arithmetic.
    allowance = 1.0 + 64.0 * flow.n_modes * float(np.finfo(float).eps)
    excess = measured / (declared * allowance)
    worst = int(np.argmax(excess))
    if excess[worst] > 1.0:
        msg = (
            # Enough digits, and the ratio itself: the two quantities differ by as
            # little as 1e-13 at the step that first exceeds the bound, and `.4g`
            # printed both as the same number with no excess to read.
            f"{backend} measured a coupled flow conditioning of "
            f"{float(measured[worst]):.12g} over a step of {dz:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at "
            f"active bin {worst} of section {section} -- exceeding it by a factor "
            f"{float(excess[worst]):.12g} of the rounding allowance: the inverse "
            "action carries the nonlinear source back by the measured factor. Lower "
            "dz_max, which is what the bound was declared over"
        )
        raise BackendError(msg)


def _coupled_flow_maps(problem: PropagationProblem) -> _CoupledFlow | None:
    """Decompose the coupled flow this backend will execute, once.

    - Returns: the Schur factors and the certificate, or ``None`` for a problem carrying
      no coupled flow.
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return None
    assert flow.generator is not None
    factors = coupled_flow_factors(flow)
    active, residual = factors.active, factors.residual_terms
    sections = len(flow.section_methods)
    held = _CoupledFlow(
        **vars(factors),
        residual_bins=active & problem.grids.inband_mask,
        n_modes=flow.n_modes,
    )
    logger.debug(
        "coupled flow: %d channel(s) on %d of %d bins over %d section(s), %s, largest "
        "declared conditioning %.3e, largest declared residual %.3e 1/m, largest "
        "declared representation error %.3e",
        held.n_modes,
        int(np.count_nonzero(active)),
        len(active),
        sections,
        "no explicit residual" if residual is None else f"{residual.shape[0]} terms",
        float(np.max(held.condition_bound, initial=1.0)),
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.generator_error)),
    )
    return held


def _harmonic_carrier(
    problem: PropagationProblem,
) -> Complex[np.ndarray, " n_to"] | None:
    """Return ``exp(2 i omega_0 t)`` on the fine grid when the harmonic is radiated.

    The envelope-grid third-harmonic term needs the carrier the coarse grid does not
    resolve. Until contract version 3 the contract carried neither it nor a fine time
    axis and a backend built it from ``omega0``, ``n_to`` and ``dt_fine`` -- one piece
    of physics inside a backend, against ``docs/dev/architecture.md`` section 3, which
    is why it now arrives as ``grids.carrier_2w0``. The sign is the FFT convention's
    (``docs/guide/media/nonlinear.md``). ``validate`` refuses a harmonic term without
    the carrier, so only a hand-built problem reaches the refusal here.

    - Returns: the carrier of shape ``(n_to,)``, or ``None`` when no harmonic term is
      present.
    """
    if not problem.responses.harmonic_carrier:
        return None
    if problem.grids.carrier_2w0 is None:
        msg = (
            "a third-harmonic Kerr term is compiled and grids.carrier_2w0 is absent; "
            "the envelope's harmonic sits an offset 2 omega_0 from its carrier and the "
            "response pass cannot place it without the factor"
        )
        raise BackendError(msg)
    return problem.grids.carrier_2w0


def _embedded_projection(
    problem: PropagationProblem,
) -> Float[np.ndarray, "n_modes n_columns"] | None:
    """Return the embedded coarse-rule projection, when the quadrature has one.

    ``transform.projection_coarse`` equals ``transform.projection`` when the rule has no
    embedded pair, in which case the difference of the two projections is identically
    zero and reading it as convergence would be wrong
    (:class:`solprop.modes.quadrature.ModeMatrices`). The alarm reports ``nan`` there
    instead, and the second projection is not computed at all.

    - Returns: the coarse projection, or ``None`` when it is the fine one.
    """
    transform = problem.transform
    if np.array_equal(transform.projection, transform.projection_coarse):
        return None
    return transform.projection_coarse


def _clean_launch_columns(
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    noise: Complex[np.ndarray, "n_modes n_pol n_omega"],
    active: np.ndarray,
) -> Float[np.ndarray, " n_modes"]:
    """Return each column's energy in a masked launch, less the input noise it carries.

    That is ``mask(field + noise) - noise``: the launch on the active band, and the
    noise alone off it, as a supplied realisation needn't be masked. Formed from the
    parts, so that no state has to come back from where it lives to subtract the noise.

    - ``active``: the active band's mask, shape ``(n_omega,)``.
    """
    inside = np.abs(field[..., active]) ** 2
    outside = np.abs(noise[..., ~active]) ** 2
    return np.sum(inside, axis=(1, 2)) + np.sum(outside, axis=(1, 2))

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's workspace: its state and every state-sized operation (SEP 52).

:class:`NumpyWorkspace` holds one run's state and does every operation a march, an
engine or the session asks of it: loading it with its noise and band mask; the stage
scalars, propagators and exponents of a step; the source, which its
:class:`~solprop.backends.numpy_ref.source.StageSource` evaluates; the coupled flow's
actions and residual (:mod:`~solprop.backends.numpy_ref.coupled_flow`); the accepted
step's finish; the boundary monitor's and the transverse edge's readings; the
statistics, which its
:class:`~solprop.backends.numpy_ref.statistics.ReferenceStatistics` forms; the host
source's reservation; and the memory ledger. It holds no march state -- no position,
proposal, history or ledger -- and is reused across a scan's points, a load replacing
its state. The engines (:mod:`~solprop.backends.numpy_ref.engines`,
:mod:`~solprop.backends.numpy_ref.multistep`) are what the marches ask of it.

The contract is consumed only through its evaluation functions --
:func:`solprop.translate.linop.delta_phi`,
:func:`solprop.translate.linop.unitary_phase`,
:func:`solprop.translate.responses.prefactor_at`, and the scalar tables
``responses.density`` and ``transform.taper_scale`` -- never through the coefficient
arrays, so the representation of the z tables can change without touching a backend.
"""

import dataclasses
import functools
import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from jaxtyping import Complex, Float

from solprop._helpers import PLAIN
from solprop.backends import tdse_source
from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.backends.memory import adams_memory, reference_base_memory
from solprop.backends.node_chunk import (
    NodeChunkPlan,
)
from solprop.backends.numpy_ref.adams_kernels import CoupledBlock
from solprop.backends.numpy_ref.coupled_flow import (
    BlockResidual,
    apply_flow,
    check_flow_conditioning,
    flow_action,
    held_flow,
    residual_action,
    residual_drive,
)
from solprop.backends.numpy_ref.general_modal import (
    general_memory,
)
from solprop.backends.numpy_ref.passes.stepping import (
    apply_hard_mask,
    apply_temporal_rate,
    error_norm,
)
from solprop.backends.numpy_ref.source import StageSource
from solprop.backends.numpy_ref.statistics import ReferenceStatistics
from solprop.backends.spatial import (
    slab_columns,
    slab_rows,
)
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.profiles import ZProfile
from solprop.stats import (
    StatsSpec,
)
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.masks import noise_spectrum
from solprop.translate.problem import (
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
)
from solprop.translate.responses import prefactor_at
from solprop.translate.routes import (
    transverse_of,
)
from solprop.translate.run_request import RunRequest
from solprop.translate.stage import (
    StageScalars,
    anchor,
    combine,
    scalars,
    scalars_at,
    terms,
)
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding, TDSEPurpose
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

__all__ = ["NumpyWorkspace"]

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
    statistics : solprop.backends.numpy_ref.statistics.ReferenceStatistics
        The statistics the reference forms itself, which :meth:`stats` asks.
    stage_source : solprop.backends.numpy_ref.source.StageSource
        The stage derivative's evaluation, which :meth:`nonlinear` asks.
    """

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
        self._state: _State | None = None
        self._noise: _State | None = None
        # The input draw rides in the state, a resumed one's too; the boundary monitor
        # reads it as the vacuum floor beneath the light it judges (SEP 21, Q24).
        self._input_floor: _State | None = None
        self._clean_columns: Float[np.ndarray, " n_columns"] | None = None
        self._monitor: BoundaryMonitor | None = None
        # The fixed arrays of the seam's device form, held for the whole run; the
        # weights that multiply them are a bundle of host floats per stage.
        self._terms = terms(problem)
        # Hoisted per step by `hoist`; the entrance's until the first step asks.
        self._anchor = anchor(problem, 0.0, helpers=PLAIN)
        # A step's bundles taken together by `stage_bundles`, by anchor and position;
        # `hoist` drops them.
        self._prepared: dict[tuple[float, float], StageScalars] = {}
        self._tdse_binding = tdse_binding
        self._kind: GridKind = problem.grid_kind
        self.statistics = ReferenceStatistics(
            problem,
            request,
            node_chunk=chunk_plan.chunk,
            grid=grid,
            modes=modes,
            profiles=profiles,
            name=name,
        )
        self.stage_source = StageSource(
            problem,
            chunk_plan=chunk_plan,
            prefactor=self.prefactor,
            tdse_binding=tdse_binding,
        )
        # The coupled flow's factors, the residual's kept actions, and the one step's
        # forward and inverse actions held for the section and length last seen.
        self._flow = held_flow(problem)
        self._block_residual: BlockResidual | None = None
        self._flow_cache: (
            tuple[
                tuple[int, float],
                dict[float, tuple[np.ndarray | None, np.ndarray | None]],
            ]
            | None
        ) = None
        # A propagator `propagator` holds for the next `source` at the same pair.
        self._held: (
            tuple[float, float, StageScalars, Complex[np.ndarray, ...]] | None
        ) = None
        # The lab-frame source's unit "propagator": one element broadcast to the state's
        # shape, so it holds no volume beside the state.
        self._lab_frame = np.broadcast_to(
            np.ones((), dtype=np.complex128), problem.shape
        )

    @property
    def collar_quiet(self) -> bool:  # numpydoc ignore=RT01
        """Whether the modal stage leaves a quiet in-stage temporal collar out (SEP 21).

        Set by the session from the boundary monitor, read by every stage.
        """
        return self.stage_source.collar_quiet

    @collar_quiet.setter
    def collar_quiet(self, quiet: bool) -> None:  # numpydoc ignore=PR01
        """Set whether the in-stage temporal collar is left out while quiet."""
        self.stage_source.collar_quiet = quiet

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
        self.stage_source.reset_alarm()

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
        return apply_hard_mask(state, self._problem.grids.active_mask)

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
            msg = "no initial state: start a session before propagating or observing"
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
        if self.stage_source.gated_plasma:
            record["_benchmark_plasma_source"] = "gated_nonresumable"
            record["plasma_source"] = {"form": "gated_benchmark", "resumable": False}
        if self._tdse_binding is not None:
            record["atomic_calls"] = dataclasses.asdict(self._tdse_binding.ledger)
            record["tdse_execution"] = dict(self._tdse_binding.metadata())
            record["tdse_last_evaluation"] = self.stage_source.tdse_last_evaluation
            record["tdse_source"] = tdse_source.source_record(self._problem)
        return record

    # --- the quadrature alarm --------------------------------------------------------

    def reset_quadrature(self) -> None:
        """Forget the alarm, so that the next plane reports its own."""
        self.stage_source.reset_alarm()

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
        self._anchor = anchor(self._problem, z0, helpers=PLAIN)
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
                self._problem, z0, missing, at=held, helpers=PLAIN
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
        return scalars(self._problem, z0, z1, at=held, helpers=PLAIN)

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
            return delta_phi(self._problem.linear, bundle.z0, bundle.z1, helpers=PLAIN)
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
            return prefactor_at(self._problem.responses, bundle.z1, helpers=PLAIN)
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
        return (
            self.nonlinear(field_in, bundle, 1.0 / propagator, purpose=purpose),
            propagator,
        )

    def coupled_block(self, z: float, dz: float) -> CoupledBlock | None:
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
        solprop.backends.numpy_ref.adams_kernels.CoupledBlock or None
            The step's frozen block, or ``None`` without a coupled flow.
        """
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
        return self.nonlinear(field_in, bundle, self._lab_frame, purpose=purpose)

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
        return apply_flow(values, action, self._flow.active)

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
        return residual_drive(
            lab,
            residual_action(flow, section, bundle.flow_weights),
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
                pair = flow_action(flow, section, fraction * dz)
                if fraction == 1.0:
                    check_flow_conditioning(flow, section, pair, dz)
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
                physical[rows] = apply_temporal_rate(
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
                physical[rows] = apply_temporal_rate(
                    physical[rows], rate, dz, kind=self._kind
                )
            result = np.fft.fft2(physical.reshape(shape), axes=(0, 1)).reshape(
                masked.shape
            )
            return self.mask(result)
        # The product in time spreads content across every bin, the DC among them, so
        # the band is imposed again after it, as the in-stage term is masked.
        return self.mask(apply_temporal_rate(masked, rate, dz, kind=self._kind))

    def score_quadrature(self) -> tuple[float, float]:
        """Score the quadrature from the last stage of an accepted step.

        :meth:`StageSource.score_quadrature
        <solprop.backends.numpy_ref.source.StageSource.score_quadrature>`: a statistic
        and a warning, never a controller input.

        Returns
        -------
        float
            The alarm, dimensionless, or NaN.
        float
            Where the last stage was evaluated, m, or NaN without one.
        """
        return self.stage_source.score_quadrature()

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
        return error_norm(
            error,
            state,
            advanced,
            self._problem.grids.norm_mask,
            self._stepper.controller,
        )

    def stats(self, spec: StatsSpec, z: float) -> dict[str, np.ndarray]:
        """Collect the requested statistics of the committed state at a position.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.
        z : float
            The committed state's position, m.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name
            (:meth:`~solprop.backends.numpy_ref.statistics.ReferenceStatistics.collect`).
        """
        return self.statistics.collect(
            spec, self.committed(), z, alarm=self.stage_source.alarm
        )

    def nonlinear(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through the transform passes.

        :meth:`StageSource.evaluate
        <solprop.backends.numpy_ref.source.StageSource.evaluate>`, on every route.

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
        return self.stage_source.evaluate(
            lab, bundle, inverse_propagator, purpose=purpose
        )

    def exponents(
        self,
    ) -> Callable[[float, StageScalars], Complex[np.ndarray, "n_modes n_pol n_omega"]]:
        """Return the memoised ``dPhi`` of a step, keyed by the stage fraction.

        The anchor is hoisted once per step and each exponent is evaluated once and
        negated for the inverse, never evaluated forward and backward independently
        (``docs/dev/stepper.md`` section 4). The propagator mode decides whether the
        exponents are kept for the whole step or evaluated per stage; the exponent
        itself is one linear combination of the terms over the bundle's weights,
        whatever the representation (:mod:`solprop.translate.stage`).

        Returns
        -------
        callable
            ``(fraction, bundle) -> dPhi``, of the state's broadcast shape.
        """
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
        maps = self.stage_source.general
        if maps is not None:
            # The route exists exactly when the tables do (`_general_maps`).
            tables = problem.transform.modal_tables
            assert tables is not None
            general = general_memory(maps, tables, problem.grids.n_to)
        ledger = adams_memory(problem, self._stepper) | {
            # The rows' one definition, in `solprop.backends.memory`.
            **reference_base_memory(
                problem,
                self._stepper,
                tabulating=self._propagator_mode == "tabulate",
                embedded=self.stage_source.embedded,
                noise=self._noise is not None,
                statistics_weight=self.statistics.power_weight_bytes,
                spatial_absorber=self.stage_source.spatial_absorber_bytes,
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
        if self.stage_source.gated_plasma:
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
        self.stage_source.close()
        self._flow_cache = None
        self.statistics.close()
        self._held = None


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

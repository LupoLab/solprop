# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The lowering of solprop's problem and request into the native problem.

``_lower`` fills ``_core.Problem``'s structs once, where a backend is built; the table
packers form the set-once tables the workspace makes resident (z tables, sections, the
carrier frame's chord, a collar's lines), and ``_ZTables`` sizes the general z tables'
slots (solprop SEP 29 section 6.6).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from solprop import transverse_monitor
from solprop.backends.exponential import CarrierReference
from solprop.backends.ledger import source_absent
from solprop.backends.memory import (
    LOCAL_FLUX_STATISTICS,
    low_storage_stage_register,
    z_table_slots,
)
from solprop.backends.node_chunk import general_planned_chunk, plan_node_chunk
from solprop.backends.spatial import far_field_partials, radial_absorber_matrix
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.grid import RadialGrid
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.translate.chebyshev import (
    _EDGE_SLACK_FRACTION,  # pyright: ignore[reportPrivateUsage] - the tables' own admitted span
    PiecewiseChebyshev,
)
from solprop.translate.coupled import flow_edge_slack
from solprop.translate.general import (
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.translate.linop import power_weight_at
from solprop.translate.plasma_source import (
    _BENCHMARK_GATED,  # pyright: ignore[reportPrivateUsage] - the private gated benchmark
)
from solprop.translate.problem import PropagationProblem, StepperSettings
from solprop.translate.routes import flux_of
from solprop.translate.run_request import RunRequest
from solprop.translate.stage import StageTerms
from solprop.transverse_monitor import TransverseEdge

# The CPU execution record is read from the package at each use: a first-use
# calibration in this process refreshes it (solprop SEP 46 section 5).
import solprop_native as _package

from . import _core
from ._device import _low_storage_flows, _on_device, native_node_chunk
from ._execution import ExecutionRecord

if TYPE_CHECKING:
    from .backend import (
        _HostFlow,  # pyright: ignore[reportPrivateUsage] - the flow's host factors
    )

__all__ = [
    "_LowerPlan",
    "_ZTables",
    "_chord_tables",
    "_edge_lines",
    "_flux_tables",
    "_lower",
    "_section_tables",
]

type RealArray = npt.NDArray[np.float64]


def _edge_lines(edge: TransverseEdge) -> dict[str, object]:
    """Return a requested collar's lines as the workspace takes them (SEP 44)."""
    lines = edge.reduction.lines
    axes = {"r": 0, "x": 1, "y": 2}
    none = 2**64 - 1
    strip = [np.asarray(line.strip, dtype=np.uint64) for line in lines]
    window = [np.asarray(line.window, dtype=np.float64) for line in lines]
    onsets = [line.onset if line.onset is not None else (none, 0, 0) for line in lines]
    space = edge.space
    return {
        "axes": np.array([axes[line.axis] for line in lines], dtype=np.uint64),
        "counts": np.array([array.size for array in strip], dtype=np.uint64),
        "positions": np.concatenate(strip) if strip else np.empty(0, dtype=np.uint64),
        "windows": np.concatenate(window) if window else np.empty(0, dtype=np.float64),
        "inward": np.array([line.inward for line in lines], dtype=np.uint64).reshape(
            -1
        ),
        "factors": np.array(
            [
                line.spacing / (2 * array.size) * line.measure if array.size else 0.0
                for line, array in zip(lines, strip, strict=True)
            ],
            dtype=np.float64,
        ),
        "measures": np.array([line.measure for line in lines], dtype=np.float64),
        "onsets": np.array(onsets, dtype=np.uint64).reshape(-1),
        "steps": np.array([line.onset_step for line in lines], dtype=np.float64),
        "k_min": float(edge.k_min),
        "scale": np.sqrt(np.asarray(space.r, dtype=np.float64))
        if lines and isinstance(space, RadialGrid)
        else np.empty(0, dtype=np.float64),
        "batch_bytes": transverse_monitor._STRIP_BATCH_BYTES,  # pyright: ignore[reportPrivateUsage]
    }


def _z_table(kind: str, table: PiecewiseChebyshev) -> dict[str, object]:
    """Return one z table as the workspace makes it resident: its breakpoints, the
    admitted span (``_EDGE_SLACK_FRACTION`` past each end) and its coefficients."""
    low, high = table.z_span
    slack = _EDGE_SLACK_FRACTION * (high - low)
    n_intervals, n_terms = table.coefficients.shape[:2]
    return {
        "kind": kind,
        "edges": np.ascontiguousarray(table.breakpoints, dtype=np.float64),
        "lower": low - slack,
        "upper": high + slack,
        "coefficients": np.ascontiguousarray(
            table.coefficients, dtype=np.complex128
        ).reshape(n_intervals, n_terms, -1),
    }


def _flux_tables(problem: PropagationProblem) -> list[dict[str, object]]:
    """Return a radial plane measure's material z table, or none (SEP 34)."""
    flux = flux_of(problem.route)
    if flux is None or flux.material is None:
        return []
    return [_z_table("flux", flux.material)]


_EMPTY = np.empty(0, dtype=np.complex128)
"""No modified-noise realisation: the native state adds none."""


def _section_tables(
    problem: PropagationProblem, *, moving: bool, residual: bool
) -> dict[str, object]:
    """Return a coupled flow's and a moving guide's sections as the stage tables take
    them: their edges, degrees and the edges' slack (`flow_edge_slack`,
    `transform_section`), and whether the flow carries a frozen residual."""
    flow = problem.linear.coupled_flow
    maps = problem.transform.modal_tables if moving else None
    none: dict[str, object] = {"edges": [], "degrees": 0, "slack": 0.0}
    out: dict[str, object] = {
        "flow": none,
        "flow_residual": False,
        "transform": none,
    }
    if flow is not None:
        out |= {
            "flow": {
                "edges": np.asarray(flow.z_edges, dtype=np.float64).tolist(),
                "degrees": int(flow.n_deg),
                "slack": float(flow_edge_slack(flow)),
            },
            "flow_residual": residual,
        }
    if maps is not None and maps.synthesis is not None:
        edges = np.asarray(maps.z_edges, dtype=np.float64)
        out["transform"] = {
            "edges": edges.tolist(),
            "degrees": int(maps.synthesis.shape[1]),
            "slack": _EDGE_SLACK_FRACTION * float(edges[-1] - edges[0]),
        }
    return out


def _chord_tables(carrier: CarrierReference | None) -> dict[str, object]:
    """Return the carrier frame's chord as the native stage tables take it.

    The imaginary part of the reference element's operator (`CarrierChord`): a
    constant's rate, an expansion's terms over the ``Phi`` weights
    the tables already hold, or a z table's reference-element column as a scalar table
    of its own (solprop issue #255), packed as its batch pass reads it. None without an
    active frame.
    """
    chord = None if carrier is None or not carrier.active else carrier.chord
    if chord is not None and chord.constant is not None:
        return {
            "chord": "constant",
            "chord_rate": float(np.imag(chord.constant)),
            "chord_terms": [],
            "chord_table": None,
        }
    if chord is not None and chord.expansion is not None:
        terms = np.imag(np.asarray(chord.expansion.terms))
        return {
            "chord": "expansion",
            "chord_rate": 0.0,
            "chord_terms": terms.tolist(),
            "chord_table": None,
        }
    if chord is not None and chord.table is not None:
        edges, lower, upper, coefficients = chord.table.packed()
        # The imaginary part a complex Clenshaw pass forms at a real position, by the
        # same operations (`stage::chord`).
        column = np.ascontiguousarray(np.imag(coefficients))
        return {
            "chord": "table",
            "chord_rate": 0.0,
            "chord_terms": [],
            "chord_table": (edges, lower, upper, column),
        }
    return {"chord": "none", "chord_rate": 0.0, "chord_terms": [], "chord_table": None}


_TABLE_POINTS = 16
"""The most positions one native table evaluation takes (``z_table::max_points``)."""


class _ZTables:
    """The general z tables of one workspace (solprop SEP 29 section 6.6, G1).

    solprop falls back to a Chebyshev table of the state's size for ``Phi``
    (``linear.phi_coefficients``) or the response prefactor (``responses.prefactor``)
    when neither is a weighted sum of fixed arrays. The coefficients cross once, with
    the workspace's setup (:meth:`records`), and stay resident; a trial's positions are
    then evaluated natively into the state's term slots, ``slots`` of them, and the
    stage weights select them: ``Phi(z1) - Phi(z0)`` is the row ``e_1 - e_0`` and
    ``Phi(z1)`` the row ``e_1``, so every native consumer of the terms (the Runge-Kutta
    stages, the low-storage flows, the Adams phase and the noise) reads the table
    unchanged, as one rounded subtraction of two values, as solprop's reference forms
    it. The prefactor's slots are its values at the same positions, one-hot weighted.

    The slots are kept as a least-recently-used set of positions in the workspace
    (``stage::ZSlots``, solprop SEP 39 phase 4): each call's positions are evaluated
    only where no slot holds them yet, so an accepted step's end, which is the next
    step's anchor, is never evaluated twice. A call's positions number at most
    ``slots`` (the tableau's distinct stage fractions, five for the low-storage march),
    so none of one call's positions is evicted by another of its own. A compensated
    Adams march reads ``D = dPhi/dz`` from the derivative table (``linear.rate``) into
    one more ``Phi`` slot, ``rate_slot``.

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

    @property
    def n_phi(self) -> int:
        """The native ``Phi`` term count: the slots, and the rate's when it is read."""
        return self.slots + (self.rate_slot is not None)

    @property
    def nbytes(self) -> int:
        """The bytes of the coefficients that cross to the workspace."""
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

    def slots_record(self) -> dict[str, object]:
        """Return the slots as the workspace configures them (``ZSlots``)."""
        return {
            "phi": self.phi,
            "prefactor": self.prefactor,
            "slots": self.slots,
            "rate": self.rate_slot is not None,
        }

    def records(self) -> list[dict[str, object]]:
        """Return the tables the workspace makes resident (``ZTable``)."""
        out: list[dict[str, object]] = []
        if self.phi:
            out.append(_z_table("phi", self._problem.linear.polynomial))
            if self.rate_slot is not None:
                out.append(_z_table("rate", self._problem.linear.rate))
        if self.prefactor:
            table = self._problem.responses.prefactor
            assert table is not None
            out.append(_z_table("prefactor", table))
        return out


@dataclasses.dataclass(frozen=True)
class _LowerPlan:
    """The backend's choices a lowering takes beside the problem and its request.

    `terms` are the fixed stage terms, `flow` a coupled flow's host factors and
    `tables` the z tables; the chunks are the backend's lowered execution choices
    (solprop SEP 47 section 6.5), the automatic plan's when absent. `cpu_execution`
    is the workspace's CPU record (the process's when absent) and `fft_planning` its
    FFT planning record (none on a device); `device` says the state is a device's, and
    `setup` holds the workspace's set-once data by part (`NativeBackend._setup`).
    """

    terms: StageTerms
    flow: _HostFlow | None = None
    tables: _ZTables | None = None
    node_chunk: int | None = None
    general_chunk: int | None = None
    cpu_execution: ExecutionRecord | None = None
    fft_planning: Mapping[str, object] | None = None
    device: bool = False
    setup: Mapping[str, object] | None = None


def _lower(
    problem: PropagationProblem, request: RunRequest, plan: _LowerPlan
) -> _core.Problem:
    """Fill the native problem's structs from solprop's problem, request and `plan`."""
    stepper = request.stepper
    node_chunk = plan.node_chunk
    if node_chunk is None:
        node_chunk = native_node_chunk(
            problem, plan_node_chunk(problem), device=_on_device()
        )
    general_chunk = plan.general_chunk
    if general_chunk is None and problem.transform.modal_tables is not None:
        general_chunk = general_planned_chunk(problem, plan_node_chunk(problem))
    native = _core.Problem()
    native.cpu_execution = (plan.cpu_execution or _package.EXECUTION).native()
    if plan.fft_planning is not None:
        native.fft_planning = dict(plan.fft_planning)
    kerr = problem.responses.kerr_coefficients
    dispersive = problem.responses.dispersive_kerr
    absorber = problem.responses.temporal_absorber
    absorber_fine = problem.responses.temporal_absorber_fine
    spatial_absorber = radial_absorber_matrix(
        problem.transform, problem.grids.pad_scale
    )
    temporal_in_stage = (
        absorber_fine is not None and problem.responses.absorber_scheme == "nonlinear"
    )
    geometry = problem.transform.geometry
    # Free space and schema 15's hybrid planar kind share the separable operator and
    # the transverse FFT engine; the planar kind adds its slab rule in x.
    free = geometry is not None and geometry.kind in ("free", "slab")
    spatial_absorption = bool(spatial_absorber.size) or (
        free and geometry is not None and bool(np.any(geometry.absorber_rate))
    )

    dims = native.dims
    dims.n_omega = problem.shape[-1]
    dims.n_t = problem.grids.n_t
    dims.n_to = problem.grids.n_to
    dims.n_modes = problem.shape[0]
    dims.n_pol_state = problem.shape[1]
    dims.n_pol_transform = problem.transform.n_pol
    dims.n_nodes = problem.transform.n_nodes
    # solprop SEP 47 section 6.5: the backend's execution choice, lowered by
    # `native_node_chunk`.
    dims.node_chunk = node_chunk
    # solprop SEP 6's number-conserving sector is formed on analytic signals, of the
    # filtered fields and, for the polynomial, of the bare one; a Raman drive beside
    # it is the cycle-averaged one, which reads the analytic signal too. The general
    # modal route synthesises its own analytic signal on the real family.
    dims.family = (
        _core.FieldFamily.envelope
        if problem.grid_kind == "envelope"
        else _core.FieldFamily.field_analytic
        if (kerr is not None and kerr.analytic_signal)
        or (
            dispersive is not None
            and dispersive.sector == "conserving"
            and problem.transform.modal_tables is None
        )
        else _core.FieldFamily.field_real
    )

    grids = problem.grids
    carrier = grids.carrier_2w0
    grid = native.grid
    grid.crop_index = np.ascontiguousarray(grids.crop_index, dtype=np.uint64)
    grid.active_mask = np.ascontiguousarray(grids.active_mask)
    grid.norm_mask = np.ascontiguousarray(grids.norm_mask)
    grid.omega_window = np.ascontiguousarray(grids.omega_window)
    grid.omega = np.ascontiguousarray(grids.omega)
    grid.time_window = np.ascontiguousarray(grids.t_window_fine)
    if problem.grid_kind == "field" and problem.responses.plasma_channels:
        grid.time_window_derivative = np.ascontiguousarray(
            grids.t_window_fine_derivative
        )
    if geometry is None:
        grid.power_weight = np.ascontiguousarray(power_weight_at(problem, 0.0))
    grid.carrier = np.ascontiguousarray(
        np.ones(grids.n_to, dtype=np.complex128) if carrier is None else carrier
    )
    grid.dt_fine = grids.dt_fine
    grid.dt = grids.dt

    _lower_linear(native, problem, plan)
    coarse_matrix = _lower_spatial(native, problem, request)
    _lower_stepper(native, problem, stepper, device=plan.device)

    native.absorber.in_stage = temporal_in_stage or spatial_absorption
    native.absorber.split_step = (
        absorber is not None and problem.responses.absorber_scheme == "split_step"
    )
    if temporal_in_stage:
        native.absorber.fine = np.ascontiguousarray(absorber_fine, dtype=np.float64)
    elif spatial_absorption:
        native.absorber.fine = np.zeros(grids.n_to)
    if absorber is not None:
        native.absorber.coarse = np.ascontiguousarray(absorber, dtype=np.float64)

    transform = native.transform
    transform.synthesis_matrix = np.ascontiguousarray(problem.transform.synthesis)
    transform.projection_matrix = np.ascontiguousarray(problem.transform.projection)
    transform.projection_coarse_matrix = np.ascontiguousarray(coarse_matrix)
    transform.synthesis = 0.0 if free else float(problem.transform.synthesis[0, 0])
    transform.projection = 0.0 if free else float(problem.transform.projection[0, 0])

    if kerr is not None:
        native.kerr.nonlinear = True
        native.kerr.coefficient = kerr.coefficient
        native.kerr.self_factor = kerr.self_factor
        native.kerr.conjugate_factor = kerr.conjugate_factor
        native.kerr.harmonic_factor = kerr.harmonic_factor
        native.kerr.take_real = kerr.take_real

    _lower_plasma(native, problem, request)
    _lower_raman(native, problem)
    _lower_general(native, problem, general_chunk)
    _lower_flow(native, plan.flow)
    _lower_dispersive(native, problem)
    _lower_flux(native, problem, request.observation.stats)
    if plan.setup is not None:
        # The set-once data, each part read through its record's table (SEP 52).
        held: Any = native
        for name, value in plan.setup.items():
            setattr(held, name, value)
    return native


def _lower_linear(
    native: _core.Problem, problem: PropagationProblem, plan: _LowerPlan
) -> None:
    """Fill the fixed terms, (n_terms, rows, n_omega): one empty term on a separable
    route, a z table's slots."""
    del problem
    tables = plan.tables
    linear = native.linear
    n_omega = native.dims.n_omega
    linear.phi_table = tables is not None and tables.phi
    linear.prefactor_table = tables is not None and tables.prefactor
    if plan.terms.separable is not None:
        linear.n_phi_terms = 1
        linear.n_prefactor_terms = 1
        return
    # A z table's terms are slots the native evaluation fills (solprop SEP 29 G1):
    # their count crosses in their place.
    if tables is not None and tables.phi:
        linear.n_phi_terms = tables.n_phi
    elif plan.terms.phi is None:
        raise BackendError("Kokkos requires a device-ready fixed-term problem")
    else:
        linear.n_phi_terms = plan.terms.phi.shape[0]
        linear.phi_terms = np.ascontiguousarray(
            plan.terms.phi.reshape(plan.terms.phi.shape[0], -1, n_omega)
        )
    if tables is not None and tables.prefactor:
        linear.n_prefactor_terms = tables.slots
    elif plan.terms.prefactor is None:
        raise BackendError("Kokkos requires a device-ready fixed-term problem")
    else:
        linear.n_prefactor_terms = plan.terms.prefactor.shape[0]
        linear.prefactor_terms = np.ascontiguousarray(
            plan.terms.prefactor.reshape(plan.terms.prefactor.shape[0], -1, n_omega)
        )


def _lower_spatial(
    native: _core.Problem, problem: PropagationProblem, request: RunRequest
) -> np.ndarray:
    """Fill the separable operator and a radial run's rows; return the coarse matrix.

    The quadrature alarm's embedded Kronrod projection is the transform's own unless
    it is the plain projection or the route has one node, and the hybrid planar kind
    packs none: the reference scores no alarm on it.
    """
    transform = problem.transform
    coarse = (
        np.empty(0, dtype=np.float64)
        if transform.n_nodes == 1
        or np.array_equal(transform.projection_coarse, transform.projection)
        else transform.projection_coarse
    )
    geometry = transform.geometry
    spatial = native.spatial
    spatial.absorber_matrix = radial_absorber_matrix(transform, problem.grids.pad_scale)
    if geometry is not None:
        spatial.spectral_weights = geometry.spectral_weights
        spatial.on_axis = geometry.on_axis
    separable = problem.linear.separable
    if geometry is None or geometry.kind not in ("free", "slab") or separable is None:
        return coarse
    spatial.n_x, spatial.n_y = geometry.shape
    spatial.k_squared = separable.k_squared
    spatial.k_squared_low = separable.k_squared_low
    spatial.kperp_sq = separable.kperp_sq
    spatial.kperp_sq_low = separable.kperp_sq_low
    spatial.frame = separable.frame
    spatial.spectral_rate = separable.spectral_rate
    assert separable.user_rate is not None
    assert separable.kperp_rate is not None
    spatial.user_rate = separable.user_rate
    spatial.numerator = separable.prefactor_numerator
    spatial.clamped = separable.prefactor_clamped
    spatial.evanescent_rate = separable.evanescent_rate
    spatial.absorber_rate = geometry.absorber_rate
    # A transverse collar's rate per column (solprop kernel schema 31).
    spatial.kperp_rate = separable.kperp_rate
    if geometry.kind == "free":
        # solprop SEP 23: the stored axes and the slots of the problem's own
        # far-field request, which size the native band partials.
        k_x, k_y = geometry.spectral_axes
        slots, _ = far_field_partials(problem, request.observation.stats)
        spatial.k_x = np.ascontiguousarray(k_x, dtype=np.float64)
        spatial.k_y = np.ascontiguousarray(k_y, dtype=np.float64)
        spatial.far_field_slots = slots
        return coarse
    # Schema 15's `SeparableOperator.kappa_sq` is (n_channels, n_omega), column `c`
    # belonging to channel `c % n_channels`; the geometry's collar rate depends on y
    # alone (solprop checks it), so it is packed per column `p + n_p * iy`, the index
    # of the temporal row it acts on.
    assert separable.kappa_sq is not None
    n_x, n_y = geometry.shape
    n_p = separable.n_channels
    spatial.n_p = n_p
    spatial.kappa_sq = np.ascontiguousarray(separable.kappa_sq)
    spatial.absorber_rate = np.ascontiguousarray(
        np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)
    )
    return np.empty(0, dtype=np.float64)


def _lower_stepper(
    native: _core.Problem,
    problem: PropagationProblem,
    stepper: StepperSettings,
    *,
    device: bool,
) -> None:
    """Fill the tableau and the low-storage march's rows (solprop kernel schema 32)."""
    tableau = stepper.tableau
    native_stepper = native.stepper
    native_stepper.n_stages = tableau.n_stages
    native_stepper.tableau_matrix = np.ascontiguousarray(tableau.matrix)
    native_stepper.tableau_nodes = np.ascontiguousarray(tableau.nodes)
    native_stepper.solution_weights = np.ascontiguousarray(tableau.solution_weights)
    native_stepper.error_weights = np.ascontiguousarray(tableau.error_weights)
    native_stepper.first_same_as_last = tableau.first_same_as_last
    native_stepper.weighted_controller = stepper.controller.weights is not None
    native_stepper.modified_noise = problem.initial.noise_model == "modified"
    flows = _low_storage_flows(problem, stepper, device=device)
    if flows is None:
        return
    native_stepper.low_storage = True
    native_stepper.low_storage_stage_register = low_storage_stage_register(
        problem, "native"
    )
    native_stepper.low_storage_flows = flows
    native_stepper.low_storage_tiled_input = device


def _lower_plasma(
    native: _core.Problem, problem: PropagationProblem, request: RunRequest
) -> None:
    """Fill the plasma columns: the first channel's rate, every channel's rows."""
    channels = problem.responses.population_channels
    channel = channels[0] if channels else None
    plasma = native.plasma
    # solprop SEP 18: the running-integral rule, one per problem.
    plasma.band = problem.responses.plasma_integral.policy == "band"
    plasma.taps = problem.responses.plasma_integral.taps
    if channel is not None:
        rate = channel.rate
        plasma.enabled = True
        # The private gated benchmark (solprop's tests/gated_source.py), read where
        # the backend is built, as the reference reads it.
        plasma.gated = _BENCHMARK_GATED.get()
        plasma.charge_mass_ratio = channel.charge_mass_ratio
        plasma.ionisation_potential = channel.ionisation_potential
        plasma.log_field_min = rate.log_field_min
        plasma.d_log_field = rate.d_log_field
        plasma.field_min = rate.field_min
        plasma.field_max = rate.field_max
        plasma.rate_max = rate.rate_max
        plasma.rate_intervals = rate.coefficients.shape[1]
        plasma.rate_coefficients = np.ascontiguousarray(rate.coefficients)
    plasma.stash = _stash_plasma(problem, request)
    _lower_channels(native, problem)
    _lower_envelope(native, problem)


def _stash_plasma(problem: PropagationProblem, request: RunRequest) -> bool:
    """Whether the accepted step's last stage leaves the plasma statistics behind.

    Only when that stage's input is the committed field: a first-same-as-last tableau,
    no split-step absorber (applied to the field after the last stage) and no coupled
    flow; and not under the modified noise model, where that input carries the noise
    draw while solprop defines the statistics on the committed field its reference
    scans. Otherwise they are scanned afresh.
    """
    absorber = problem.responses.temporal_absorber
    return (
        bool(problem.responses.population_channels)
        and problem.responses.envelope_plasma is None
        and problem.initial.noise_model != "modified"
        and request.stepper.tableau.first_same_as_last
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


def _lower_raman(native: _core.Problem, problem: PropagationProblem) -> None:
    """Fill the Raman terms, row-major (n_terms, 2 n_to) and (n_terms, 2)."""
    terms = problem.responses.raman_terms
    raman = native.raman
    raman.n_components = len(terms)
    raman.row_component = np.asarray(
        [component for component, term in enumerate(terms) for _ in term.undamped],
        dtype=np.float64,
    )
    raman.component_fraction = np.asarray([term.fraction for term in terms])
    if not terms:
        return
    first = terms[0]
    undamped = np.concatenate([term.undamped for term in terms], axis=0)
    raman.enabled = True
    raman.thg = bool(first.thg) and problem.grid_kind == "field"
    raman.tensor_capable = all(term.tensor_capable for term in terms)
    raman.n_terms = undamped.shape[0]
    raman.fraction = float(first.fraction)
    raman.epsilon0 = EPS0
    raman.undamped = np.ascontiguousarray(undamped)
    raman.tensor_weights = np.ascontiguousarray(
        np.concatenate([term.tensor_weights for term in terms], axis=0)
    )


def _lower_dispersive(native: _core.Problem, problem: PropagationProblem) -> None:
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
        return
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
    lowered = native.dispersive
    if tables is not None:
        lowered.group_keys = np.asarray(keys, dtype=np.uint64)
        lowered.node_region = np.ascontiguousarray(tables.region_index, dtype=np.uint64)
    lowered.enabled = True
    lowered.symmetric = symmetric
    lowered.form = {"real": 0, "conserving": 1, "harmonic": 2}[form]
    lowered.inputs = np.ascontiguousarray(stacked)
    lowered.term_multipliers = np.ascontiguousarray(
        np.array(multipliers, dtype=np.float64)
    )
    lowered.term_rows = np.asarray(rows, dtype=np.uint64)
    lowered.term_channels = np.asarray(channels, dtype=np.uint64)
    lowered.coefficients = np.asarray(leaf.coefficients, dtype=np.float64)
    lowered.pair_constants = np.ascontiguousarray(constants)
    lowered.group_offsets = np.asarray(offsets, dtype=np.uint64)
    lowered.group_members = np.asarray(
        [channel for group in groups for channel in group], dtype=np.uint64
    )
    lowered.quartics = quartics
    lowered.input_scale = np.ascontiguousarray(input_scale)
    lowered.output_scale = np.ascontiguousarray(scale)


def _lower_flux(
    native: _core.Problem, problem: PropagationProblem, stats: StatsSpec
) -> None:
    """Fill the plane measure of solprop SEP 34 (kernel schema 35).

    A separable run (Cartesian, line, planar) is measured by its own operator's root,
    which the native state already holds: nothing more is packed, and a plan that is
    not that operator is refused. A radial run packs the entrance's material and
    transverse squares and whether a material z table follows
    (:func:`_flux_tables`); the admission record is not read. ``closure``
    sizes the magnetic closure's buffers for the run's own local statistics,
    `stats`.
    """
    flux = flux_of(problem.route)
    if flux is None:
        return
    native.flux.closure = bool(set(stats.names) & LOCAL_FLUX_STATISTICS)
    if problem.linear.separable is not None:
        if flux.base is not problem.linear.separable or flux.material is not None:
            raise BackendError(
                "solprop-native measures a separable run by its own operator's root, "
                "and this plane measure is another"
            )
        return
    base = flux.base
    native.flux.k_squared = np.ascontiguousarray(base.k_squared, dtype=np.complex128)
    native.flux.k_squared_low = np.ascontiguousarray(
        base.k_squared_low, dtype=np.complex128
    )
    native.flux.kperp_sq = np.ascontiguousarray(base.kperp_sq, dtype=np.float64)
    native.flux.kperp_sq_low = np.ascontiguousarray(base.kperp_sq_low, dtype=np.float64)
    native.flux.table = flux.material is not None


def _lower_general(
    native: _core.Problem, problem: PropagationProblem, chunk: int | None
) -> None:
    """Fill the general modal route's maps, derived as the reference derives them.

    The per-node tensor, the form and the delayed maps come from solprop's own
    derivations (``solprop.translate.general``), so both backends execute the same
    arrays; the chunk is the route's own at the same budget
    (:func:`solprop.backends.node_chunk.general_planned_chunk`), given as `chunk`.
    Nothing is filled for a problem without ``transform.modal_tables``.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return
    assert tables.synthesis is not None
    assert tables.projection is not None
    assert chunk is not None
    regional = problem.responses.regional_responses
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    terms = 0 if delayed_tensor is None else int(delayed_tensor.shape[1])
    general = native.general
    general.enabled = True
    general.real_form = node_kerr_form(regional) == "real"
    # A moving guide's maps are every Chebyshev coefficient along z, flattened over
    # (n_intervals, n_deg) in the order of the stage bundle's transform_weights; a
    # constant guide's are the one coefficient.
    general.map_intervals, general.map_n_deg = (
        int(one) for one in tables.synthesis.shape[:2]
    )
    general.components = int(tables.electric.shape[2])
    general.nodes = int(tables.region_index.size)
    general.node_chunk = chunk
    general.responses = 0 if delayed_tensor is None else delayed_tensor.shape[0]
    general.terms = terms
    general.pad_scale = float(problem.grids.pad_scale)
    general.epsilon0 = EPS0
    general.synthesis = np.ascontiguousarray(tables.synthesis)
    general.projection = np.ascontiguousarray(tables.projection)
    general.tensor = np.ascontiguousarray(node_kerr_tensor(tables, regional))
    general.frequency_factor = np.ascontiguousarray(
        tables.source_frequency_factor, dtype=np.complex128
    )
    if delayed_tensor is not None:
        general.delayed_tensor = np.ascontiguousarray(delayed_tensor)
    if delayed_spectrum is not None:
        general.delayed_spectrum = np.ascontiguousarray(delayed_spectrum)
    if terms:
        general.node_response = np.ascontiguousarray(
            owner.astype(np.int64) + 1, dtype=np.uint64
        )


def _lower_envelope(native: _core.Problem, problem: PropagationProblem) -> None:
    """Fill the accepted cycle rules, (n_channels, 4, 512), and masked factors."""
    assembly = problem.responses.envelope_plasma
    if assembly is None:
        return
    nodes = np.zeros((len(assembly.channels), 4, 512))
    weights = np.zeros_like(nodes)
    for channel_index, channel in enumerate(assembly.channels):
        for level, rule in enumerate(channel.cycle.rules):
            nodes[channel_index, level, : len(rule.nodes)] = rule.nodes
            weights[channel_index, level, : len(rule.weights)] = rule.weights
    native.plasma.envelope = True
    native.plasma.envelope_rule_nodes = nodes
    native.plasma.envelope_rule_weights = weights
    native.plasma.envelope_drude_factor = assembly.factors.drude
    native.plasma.envelope_loss_factor = assembly.factors.loss


def _lower_channels(native: _core.Problem, problem: PropagationProblem) -> None:
    """Fill independent rate tables without resampling their coefficients."""
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
    native.plasma.n_channels = len(parameters)
    native.plasma.channel_parameters = np.asarray(parameters, dtype=np.float64).reshape(
        -1, 9
    )
    if coefficients:
        native.plasma.channel_coefficients = np.concatenate(coefficients)


def _lower_flow(native: _core.Problem, flow: _HostFlow | None) -> None:
    """Fill a coupled flow's bins and residual data; its actions follow per step."""
    if flow is None:
        return
    slot = np.zeros(flow.active.size, dtype=np.uint64)
    slot[flow.active] = np.arange(1, int(np.count_nonzero(flow.active)) + 1)
    residual = flow.residual_terms
    # The bins formed by scaling and squaring are numbered 1, 2, ... in row-major
    # (section, bin) order and carry their Schur factors; the rest take the
    # eigenbasis, whose `W` and `Y` are zero where it is not used.
    fallback = ~flow.route
    route = np.zeros(flow.route.shape, dtype=np.uint64)
    route[fallback] = np.arange(1, int(np.count_nonzero(fallback)) + 1)
    lowered = native.flow
    lowered.enabled = True
    lowered.active = int(np.count_nonzero(flow.active))
    lowered.sections = int(flow.frozen.shape[0])
    lowered.terms = 0 if residual is None else int(residual.shape[0])
    lowered.slot = slot
    if residual is not None:
        # The residual's data are zero on the bins the error norm does not score, so
        # the engine's residual there is an exact zero and the frozen flow alone
        # carries them, as in solprop's reference. Every term is finite, so each
        # product with a zero is a signed zero and the running sums stay +0: the
        # source is what adding nothing leaves, but for the sign of a zero.
        scored = flow.scored[None, :, None, None]
        lowered.frozen = np.ascontiguousarray(np.where(scored, flow.frozen, 0.0))
        lowered.residual_terms = np.ascontiguousarray(np.where(scored, residual, 0.0))
    lowered.left = flow.left
    lowered.right = flow.right
    lowered.eigenvalues = flow.eigenvalues
    lowered.route = route
    lowered.fallback_vectors = np.ascontiguousarray(flow.vectors[fallback])
    lowered.fallback_triangular = np.ascontiguousarray(flow.triangular[fallback])

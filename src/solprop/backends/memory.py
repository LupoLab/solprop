# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Estimate a run's persistent memory before a backend allocates it.

A backend names every buffer it holds in
:meth:`~solprop.backends.protocol.Backend.memory_ledger`, but only once it exists, and
by then it has allocated them. :func:`estimate_memory` returns the same bytes from the
translated problem alone, grouped as :func:`ledger_groups` groups a constructed
backend's ledger, so that a run can be admitted against a budget **before** it
allocates (the admission itself is the driver's, :mod:`solprop.driver.memory`).

The route groups are the general modal route's memory budget, per backend, checked
against every backend's ledger to the byte:

``general_maps``
    The general modal route's two execution maps and per-node tensor,
    ``2 S + 648 Q`` with ``S = 16 Q C M n_omega`` one dense map; a moving guide's
    maps are ``s d`` coefficients each over ``s`` sections of ``d`` terms, and a
    stage evaluates one of each, ``2 s d S + 2 S + 648 Q``, and ``native`` holds a
    trial's weights too, ``8 n_stages s d``.
``general_reference``
    The record's resident E/H reference, its backward partners, the reciprocal Gram
    and the coordinate dual, ``6 n_z S + 16 n_z M^2 n_omega``. **Already resident**
    when the estimate is formed: the translated problem holds it.
``general_chunk``
    The node-chunk buffers and the per-channel vectors, sized by the node chunk
    ``K`` of :func:`~solprop.backends.numpy_ref.general_modal.general_node_chunk`.
``general_delayed``
    What a delayed (tensor Raman) response adds per chunk node, and its kernels.
``coupled_flow``
    The coupled flow's blocks, ``B = 16 A N^2`` each over the ``A`` active bins, with
    the eigenbasis and formation scratch. On Kokkos the Schur factors of the
    bins the eigenbasis does not take are held too; how many is decided per bin, so
    the estimate decides it as the adapter does (the one quantity it computes rather
    than reads off a shape, with temporaries of the flow's own block size).

``adams_history``
    The Adams march's history (SEP 16 phase 3b), ``S`` one state's bytes and ``q``
    the method's order: on the host engine (``numpy_ref``, ``numba_cpu``) the store
    of ``q + 2`` rows, with a varying operator's state and rate rows beside it in the
    exponential formulation, and ``numba_cpu``'s moment scratch of
    ``MAX_ADAMS_ORDER + 3`` rows; ``native``'s resident engine holds ``2 (q + 2) + 4``
    rows and the norm's per-bin scale, and with a varying operator ``3 (q + 2) + 3``
    more rows and the rate's term weights. Interpolating save planes adds ``q + 6``
    rows for dense observation, and another ``q + 2`` with a varying operator.

``base`` is every other row: the state, the stages, the propagators and the transform
buffers of the legacy mode-set, free-space and spatial routes. It is the reference's
own rows (:func:`reference_base_memory`, which ``numpy_ref``'s ledger calls), exact on
``numpy_ref``; on ``numba_cpu`` and Kokkos it is **the reference's count of those
buffers, not the backend's own**, plus each compiled backend's far-field band partials
(:func:`far_field_memory`, to the byte) and SEP 6's dispersive response's
(:func:`dispersive_memory`): on the fused node path its filtered fields, node slots,
products and their spectra, each compiled backend's ledger rows to the byte; on the
Cartesian, line and planar routes the buffers per node the node budget admits for it,
times the node chunk, as the general route's chunk counts its own. On twelve measured
cases (without the response) ``numba_cpu``'s own base is 0.975 to 1.046 times the
reference's count, and Kokkos's 1.05 to 2.48 times (up to 5.4 MB more on the varying
wall), because the native base also counts the backend's host copies of the problem's
tables, its statistics buffers and a device copy of each z table.
For SEP 27's held field-current source, the estimate replaces the reference's
allowance with the selected compiled backend's current buffers, projection rows,
spectra and tables. FFT-provider allocator growth remains a measured ledger row.

What is **not counted** is named on every estimate (:attr:`MemoryEstimate.not_counted`).

**Before translation.** The general route's reference and maps are the largest
allocations of a large guide, and the route builds them when the simulation is
constructed, so an estimate of the translated problem comes after them.
:func:`estimate_guide_memory` returns the ``general_reference``, ``general_maps`` and
``general_chunk`` groups from the guide's declared shape and the grid alone -- its
channels, its node rule, the grid's bins and time samples, and a moving guide's sections
and degree -- equal to the byte to what :func:`estimate_memory` returns for those groups
once the problem exists, so that the driver admits a guide before its reference is
built.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from solprop.backends.registry import resolve_name
from solprop.backends.spatial import (
    AXIS_FOLDED,
    AXIS_SPECTRUM_ONLY,
    far_field_partials,
)
from solprop.exceptions import BackendError
from solprop.fourier import n_omega_for
from solprop.grid import TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.tableau import DOPRI5
from solprop.translate.coupled import (
    CoupledFlowTables,
    frozen_generator,
    schur_blocks,
    schur_eigenbasis,
)
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.problem import KernelProblem, dispersive_node_buffers
from solprop.translate.responses import ResponseStage

__all__ = [
    "ESTIMATED_BACKENDS",
    "LOCAL_FLUX_STATISTICS",
    "MEMORY_GROUPS",
    "MLX_GRAPH_BUDGET",
    "RESIDENT_CHANNELS",
    "MemoryEstimate",
    "adams_memory",
    "dispersive_memory",
    "estimate_guide_memory",
    "estimate_memory",
    "far_field_memory",
    "flux_closure_memory",
    "held_current_memory",
    "ledger_groups",
    "low_storage_flow_scratch",
    "low_storage_flows",
    "low_storage_memory",
    "low_storage_stage_register",
    "mixed_precision_rows",
    "mlx_bounded",
    "mlx_graph_bytes",
    "mlx_rows",
    "mlx_stage_bytes",
    "reference_base_memory",
    "single_precision_rows",
    "z_table_slots",
]

ESTIMATED_BACKENDS = ("numpy_ref", "numba_cpu", "native", "mlx")
"""The backends :func:`estimate_memory` has formulas for."""

MEMORY_GROUPS = (
    "base",
    "general_maps",
    "general_reference",
    "general_chunk",
    "general_delayed",
    "coupled_flow",
    "adams_history",
)
"""The groups of an estimate and of a grouped ledger, in this order."""

_PAIR_HISTORIES = 6

_MAP_ROWS = frozenset(
    {
        "general_synthesis",
        "general_projection",
        "general_stage_maps",
        "general_map_weights",
        "general_tensor",
    }
)
_REFERENCE_ROWS = frozenset(
    {
        "general_electric",
        "general_magnetic",
        "general_backward_electric",
        "general_backward_magnetic",
        "general_reciprocal_gram",
        "general_coordinate_dual",
    }
)
_DELAYED_ROWS = frozenset(
    {
        "general_node_drive",
        "general_node_histories",
        "general_node_drive_spectrum",
        "general_node_delayed",
        "general_node_product",
        "general_node_response",
        "general_delayed_tensor",
        "general_delayed_spectrum",
    }
)

_GENERAL_ROWS = (
    ("general_maps", _MAP_ROWS),
    ("general_reference", _REFERENCE_ROWS),
    ("general_delayed", _DELAYED_ROWS),
)

_NOT_COUNTED = {
    "numpy_ref": (
        "SEP 6's dispersive response's per-call transients",
        "per-stage transients freed inside a pass (6.2 % of the general route's rows "
        "at their peak on a measured general-route case)",
        "a noise field loaded after construction (one state's bytes)",
        "the translated problem's own tables other than the general reference",
        "the Adams general path's history (PECE and the comparison variants) and a "
        "trial's transient rows",
    ),
    "numba_cpu": (
        "SEP 6's dispersive response's multipliers, scratch and synthesis copy on the "
        "fused node path, and on the Cartesian, line and planar routes the difference "
        "between the buffers the node budget admits and those it holds",
        "the heap FFTW planning grows at construction (fft_plan_heap_growth and "
        "general_fft_plan_heap_growth; 34.2 and 66.5 MiB on the delayed route at "
        "n_to 256 and 512, more than every buffer it plans for), measured, not "
        "predictable",
        "the backend's own base buffers beyond the reference's count of them",
        "a noise field loaded after construction (one state's bytes)",
        "the translated problem's own tables other than the general reference",
        "the Adams general path's history (PECE and the comparison variants) and a "
        "trial's transient rows",
    ),
    "native": (
        "SEP 6's dispersive response's multipliers and packed projections",
        "the heap FFT planning grows at construction (fft_plan_heap_growth), measured",
        "the backend's own base buffers beyond the reference's count of them (its "
        "host copies of the problem's tables are counted natively)",
        "the Schur factors of a coupled flow, formed on the host",
        "a noise field loaded after construction (one state's bytes)",
        "the translated problem's own tables other than the general reference",
        "the Adams general path's history (PECE and the comparison variants)",
    ),
    "mlx": (
        "MLX's compiled kernels and its allocator's cache beyond one step's graph",
        "a noise field loaded after construction (refused under single precision)",
        "the translated problem's own tables other than the general reference",
    ),
}


@dataclass(frozen=True)
class MemoryEstimate:
    """A run's persistent bytes on one backend, by group, before it allocates them.

    Attributes
    ----------
    backend : str
        The registry name the estimate is for.
    groups : collections.abc.Mapping
        Bytes by group, every one of :data:`MEMORY_GROUPS` present, non-negative.
    not_counted : tuple of str
        What the estimate leaves out, each in words (the backend's measured planning
        heap, per-stage transients, a noise field loaded later).
    resident : int
        Bytes of the estimate already allocated when it was formed: the general
        route's reference, which the translated problem holds. Non-negative.

    Examples
    --------
    >>> from solprop.backends.memory import MemoryEstimate
    >>> estimate = MemoryEstimate("numpy_ref", {"base": 2048, "coupled_flow": 512})
    >>> estimate.total, estimate.largest(1)
    (2560, (('base', 2048),))
    """

    backend: str
    groups: Mapping[str, int]
    not_counted: tuple[str, ...] = ()
    resident: int = 0

    def __post_init__(self) -> None:
        """Adopt a plain copy of the groups."""
        object.__setattr__(self, "groups", dict(self.groups))

    @property
    def total(self) -> int:
        """Return the bytes of every group together.

        Returns
        -------
        int
            The sum of :attr:`groups`, bytes.
        """
        return sum(self.groups.values())

    def largest(self, count: int = 3) -> tuple[tuple[str, int], ...]:
        """Return the largest non-empty groups, largest first.

        Parameters
        ----------
        count : int, optional
            How many to return at most, positive.

        Returns
        -------
        tuple of (str, int)
            Group names and bytes; ties keep :data:`MEMORY_GROUPS`' order.
        """
        present = [(name, size) for name, size in self.groups.items() if size]
        return tuple(sorted(present, key=lambda item: -item[1])[:count])


def estimate_memory(problem: KernelProblem, backend: str) -> MemoryEstimate:
    """Return the persistent bytes a backend will allocate for a problem.

    Construction-free: nothing the backend would allocate is allocated here. The route
    groups are the formulas of the module docstring for the named backend; the
    ``base`` group is the reference's count on every backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated, validated problem.
    backend : str
        A registry name, one of :data:`ESTIMATED_BACKENDS`, or ``"auto"``, resolved
        as :func:`solprop.backends.registry.resolve_name` resolves it.

    Returns
    -------
    MemoryEstimate
        Bytes by group, what is not counted, and what is already resident.

    Raises
    ------
    BackendError
        If the backend has no formulas here.

    Examples
    --------
    >>> from solprop.backends.memory import estimate_memory
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> estimate = estimate_memory(problem, "numpy_ref")
    >>> estimate.total == estimate.groups["base"] > 0
    True
    """
    name = _estimated(backend)
    groups = dict.fromkeys(MEMORY_GROUPS, 0)
    base = _reference_base(problem)
    if problem.precision.single:
        base = single_precision_rows(problem, base)
    elif problem.precision.mixed:
        base = mixed_precision_rows(problem, base)
    if name == "mlx":
        base |= mlx_rows(problem, base)
    if name in {"numba_cpu", "native"} and problem.responses.plasma_channels:
        base = {
            key: value
            for key, value in base.items()
            if not key.startswith("held_current_")
        }
        base.update(
            held_current_memory(problem)
            if name == "numba_cpu"
            else _native_held_current_memory(problem)
        )
    groups["base"] = (
        sum(base.values())
        + sum(dispersive_memory(problem, name).values())
        + sum(far_field_memory(problem, name).values())
        + sum(flux_closure_memory(problem, name).values())
    )
    if isinstance(
        problem.stepper.integrator, LowStorageRungeKutta
    ) and not low_storage_stage_register(problem, name):
        # The reference's count holds the stage register; this backend elides it.
        groups["base"] -= 16 * math.prod(problem.shape)
    if isinstance(
        problem.stepper.integrator, LowStorageRungeKutta
    ) and low_storage_flow_scratch(problem, name):
        groups["base"] += 16 * math.prod(problem.shape)
    groups |= _general_groups(problem, name)
    groups["coupled_flow"] = _flow_bytes(problem, name)
    groups["adams_history"] = sum(adams_memory(problem, name).values())
    return MemoryEstimate(
        name,
        groups,
        not_counted=_NOT_COUNTED[name],
        resident=groups["general_reference"],
    )


# The reference's rows that stay binary64 on the single-precision path: the exponents
# a step tabulates, the host evaluations (rates, envelope moments, statistics) and the
# scalars (SEP 33 section 6.1).
_BINARY64_ROWS = frozenset(
    {
        "propagators",
        "statistics_power_weight",
        "node_ionisation_rate",
        "plasma_channel_scalars",
        "envelope_source_streams",
        "envelope_histories",
        "envelope_current_scratch",
    }
)


def single_precision_rows(
    problem: KernelProblem, rows: dict[str, int]
) -> dict[str, int]:
    """Return the reference's base rows for a single-precision problem.

    SEP 33 section 6.1: the element sizes follow the precision. Every buffer the
    single-precision path holds in binary32 is half the reference's; the rows of
    ``_BINARY64_ROWS`` stay as they are; the narrowed copies of the source tables
    and the binary64 state the statistics widen are added.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A single-precision problem.
    rows : dict
        The reference's rows, bytes.

    Returns
    -------
    dict
        The rows, bytes, with ``single_tables`` and ``single_widened_state`` added.
    """
    single = {
        name: size if name in _BINARY64_ROWS else size // 2
        for name, size in rows.items()
    }
    single["single_tables"] = _narrowed_table_bytes(problem)
    single["single_widened_state"] = 16 * math.prod(problem.shape)
    return single


# The rows that touch the propagated state, binary64 under mixed precision: the state,
# every row stored in the interaction variable, the propagators and the noise.
_STATE_ROWS = frozenset(
    {
        "state",
        "noise",
        "stages",
        "accumulator_solution",
        "accumulator_error",
        "register_accumulator",
        "register_stage",
    }
)


def mixed_precision_rows(
    problem: KernelProblem, rows: dict[str, int]
) -> dict[str, int]:
    """Return the reference's base rows for a mixed-precision problem.

    SEP 36 section 6.7: every row stored beside the state is binary64, as are the rows
    of ``_BINARY64_ROWS``; the source's workspaces are binary32 and half the
    reference's. The narrowed copies of the source tables and the narrowed stage input
    are added.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A mixed-precision problem.
    rows : dict
        The reference's rows, bytes.

    Returns
    -------
    dict
        The rows, bytes, with ``single_tables`` and ``mixed_narrowed_input`` added,
        and on free space with a split-step collar ``mixed_split_transverse``: the
        collar transforms the binary64 state in its own binary64 transverse volume
        beside the source's binary32 one (section 4.2).
    """
    kept = _BINARY64_ROWS | _STATE_ROWS
    mixed = {name: size if name in kept else size // 2 for name, size in rows.items()}
    mixed["single_tables"] = _narrowed_table_bytes(problem)
    mixed["mixed_narrowed_input"] = 8 * math.prod(problem.shape)
    geometry = problem.transform.geometry
    responses = problem.responses
    if (
        geometry is not None
        and geometry.kind == "free"
        and responses.temporal_absorber is not None
        and responses.absorber_scheme == "split_step"
    ):
        mixed["mixed_split_transverse"] = 16 * math.prod(problem.shape[1:])
    return mixed


def _narrowed_table_bytes(problem: KernelProblem) -> int:
    """Return the bytes of the source tables a reduced-precision path narrows.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Returns
    -------
    int
        Four bytes per narrowed element.
    """
    transform, grids = problem.transform, problem.grids
    tables = transform.synthesis.size + transform.projection.size
    if not np.array_equal(transform.projection, transform.projection_coarse):
        tables += transform.projection_coarse.size
    tables += 2 * grids.n_to + grids.omega_window.size
    if problem.responses.temporal_absorber_fine is not None:
        tables += grids.n_to
    return 4 * tables


# The single-precision rows a step's lazy graph holds once per stage on the device.
_PER_STAGE_ROWS = frozenset(
    {
        "fine_spectrum",
        "modal_time",
        "projected",
        "projected_coarse",
        "node_field",
        "node_polarisation",
        "node_ionisation_rate",
        "node_fraction",
        "node_free_current",
        "node_raman_drive",
        "node_raman_spectrum",
        "node_raman_convolved",
        "held_current_streams",
        "held_current_projected",
        "held_current_embedded",
        "held_current_spectra",
        "transverse_scratch",
        "spatial_absorber_time",
    }
)


def mlx_rows(problem: KernelProblem, rows: dict[str, int]) -> dict[str, int]:
    """Return the ``mlx`` backend's device rows beside the oracle's host rows.

    SEP 33 section 6.3: the device copies of the narrowed tables, the operator's and
    the prefactor's terms as double-float pairs, and the live graph. Where a step's
    graph would exceed :data:`MLX_GRAPH_BUDGET` the backend evaluates each stage as it
    is formed and each running integral as it is scanned, so at most one stage's
    working rows (``_PER_STAGE_ROWS``, twice, for the graph's
    temporaries), a scan's ``2 ceil(log2 n_to) + 4`` arrays over the node chunk, and the
    stages and their propagators are live at once; below it, a whole step's seven
    stages of rows are, the scans' levels released as they go. Measured against MLX's
    own peak in
    ``tests/backends/test_mlx.py``.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A single-precision problem on a fibre, radial or Cartesian route.
    rows : dict
        Its single-precision reference rows.

    Returns
    -------
    dict
        ``mlx_tables``, ``mlx_terms`` and ``mlx_step_graph``, bytes.
    """
    transform, grids = problem.transform, problem.grids
    tables = transform.synthesis.size + 2 * transform.projection.size
    tables += 3 * grids.n_to + 3 * grids.omega_window.size
    state = math.prod(problem.shape)
    linear = problem.linear
    phi_terms = 1 if linear.phi_expansion is None else linear.phi_expansion.n_terms
    per_stage = _mlx_per_stage(problem, rows)
    scan = _mlx_scan_bytes(problem)
    stages = problem.stepper.tableau.n_stages
    stage = 2 * per_stage + scan
    whole = stages * (2 * per_stage + scan // 2)
    graph = stage if mlx_bounded(stages, stage) else whole
    return {
        "mlx_tables": 4 * tables,
        "mlx_terms": 16 * state * (phi_terms + 1),
        "mlx_step_graph": graph + (2 * stages + 6) * 8 * state,
    }


MLX_GRAPH_BUDGET = 1 << 30
"""Bytes a step's lazy graph may hold before ``mlx`` evaluates stages as it forms them.

Below it the whole step is one graph, evaluated with its norm; above it each stage and
each running integral is evaluated as it is formed, which bounds the live graph to one
stage (SEP 33 section 6.3).
"""


def mlx_bounded(stages: int, stage: int) -> bool:
    """Return whether ``mlx`` evaluates each stage as it is formed.

    Parameters
    ----------
    stages : int
        The tableau's stage count.
    stage : int
        One stage's live-graph bytes.

    Returns
    -------
    bool
        True when a whole step's graph would exceed :data:`MLX_GRAPH_BUDGET`.
    """
    return stages * stage > MLX_GRAPH_BUDGET


def mlx_stage_bytes(problem: KernelProblem) -> int:
    """Return one stage's live-graph bytes on the ``mlx`` backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A single-precision problem on a fibre, radial or Cartesian route.

    Returns
    -------
    int
        Twice the per-stage working rows, with a scan's levels.
    """
    rows = single_precision_rows(problem, _reference_base(problem))
    return 2 * _mlx_per_stage(problem, rows) + _mlx_scan_bytes(problem)


# Fine-grid column arrays a stage's graph holds beyond the reference's rows on the
# spatial routes: MLX's transverse transforms copy their axes to the last one and
# back, and the reference's in-place passes are separate arrays on the device.
# Measured against MLX's peak on radial and Cartesian beams (tests/backends/test_mlx).
_MLX_TRANSIENT_COLUMNS = {"free": 18, "radial": 6}


def _mlx_per_stage(problem: KernelProblem, rows: dict[str, int]) -> int:
    """Return one stage's working bytes on ``mlx``, before the graph's doubling.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A single-precision problem.
    rows : dict
        Its single-precision reference rows.

    Returns
    -------
    int
        The ``_PER_STAGE_ROWS`` and, on a spatial route, its transform transients.
    """
    per_stage = sum(size for name, size in rows.items() if name in _PER_STAGE_ROWS)
    transform = problem.transform
    geometry = transform.geometry
    if geometry is not None:
        columns = min(transform.node_chunk, transform.n_nodes) * transform.n_pol
        fine = columns * problem.shape[2] * problem.grids.n_to * 8
        per_stage += _MLX_TRANSIENT_COLUMNS.get(geometry.kind, 0) * fine
    if problem.responses.envelope_plasma is not None:
        # The device's envelope plasma per node sample and channel: the drive, the
        # loss and the loss current, complex64, and the rate, exposure, fraction,
        # survival, failure flag and scan scratch, four bytes each.
        nodes = min(transform.node_chunk, transform.n_nodes)
        samples = nodes * problem.grids.n_to
        channels = len(problem.responses.envelope_plasma.channels)
        per_stage += samples * (3 * 8 * transform.n_pol + channels * 6 * 4)
    return per_stage


def _mlx_scan_bytes(problem: KernelProblem) -> int:
    """Return the bytes of one running integral's evaluation on ``mlx``.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Returns
    -------
    int
        ``2 ceil(log2 n_to) + 4`` arrays over the node chunk, or 0 without plasma.
    """
    if not problem.flags.has_plasma:
        return 0
    transform, grids = problem.transform, problem.grids
    chunk = min(transform.node_chunk, transform.n_nodes) * problem.shape[2]
    levels = 2 * math.ceil(math.log2(grids.n_to)) + 4
    return levels * chunk * transform.n_pol * grids.n_to * 4


def mlx_graph_bytes(problem: KernelProblem) -> int:
    """Return the ``mlx`` backend's live-graph row for a problem, bytes.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A single-precision problem on a fibre, radial or Cartesian route.

    Returns
    -------
    int
        ``mlx_step_graph`` of :func:`mlx_rows`.
    """
    rows = single_precision_rows(problem, _reference_base(problem))
    return mlx_rows(problem, rows)["mlx_step_graph"]


def estimate_guide_memory(
    guide: ModalGuide,
    grid: TimeGrid,
    *,
    z_max: float,
    backend: str,
    n_stages: int = DOPRI5.n_stages,
) -> MemoryEstimate:
    """Return a general modal guide's route groups from its declared shape alone.

    Nothing is solved or allocated: the sizes are the guide's channels, its node
    rule's nodes, the grid's coarse bins, fine bins and time samples, and, for a
    moving guide, the sections its path keeps up to `z_max` and their degree. The
    groups are the three the route fixes before translation -- ``general_reference``
    (the six maps of the dense reference: E and H, their backward partners, the
    reciprocal Gram and the coordinate dual), ``general_maps`` and ``general_chunk``
    -- each equal to the byte to :func:`estimate_memory`'s on the translated problem;
    the others are zero here and named in :attr:`MemoryEstimate.not_counted`.

    **The backward partners are counted.** Whether a basis carries them is a property
    of the provider's solved sample, which this does not form. Both providers this
    package ships return them; a provider that returns none is estimated high by
    those rows, two thirds of the fields again, and may be refused here where its
    translated estimate would be admitted.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        A guide built by :meth:`solprop.modes.guide.ModalGuide.from_provider`.
    grid : solprop.grid.TimeGrid
        The run's field grid.
    z_max : float
        Where the run ends, m, positive: the snapshot's reference is stored at the
        entrance and at the tables' far end, and a moving guide keeps the sections
        its path needs to reach it.
    backend : str
        A registry name, one of :data:`ESTIMATED_BACKENDS`, or ``"auto"``.
    n_stages : int, optional
        The tableau's stage count, which sizes Kokkos's per-stage weights on a moving
        guide; the default tableau's by default.

    Returns
    -------
    MemoryEstimate
        The three route groups, nothing resident.

    Raises
    ------
    BackendError
        If the backend has no formulas here, or the guide declares no sampling.

    Examples
    --------
    The reference a real guide's route holds, estimated before it is built:

    >>> from solprop.backends.memory import estimate_guide_memory
    >>> from solprop.grid import FieldGrid
    >>> from solprop.processing._examples import example_general_run
    >>> run, tables, guide = example_general_run()
    >>> grid = FieldGrid.from_limits(1e-5, 1.03e-6, (1.0e-6, 1.06e-6), 100e-15)
    >>> estimate = estimate_guide_memory(guide, grid, z_max=1e-5, backend="numpy_ref")
    >>> held = sum(
    ...     array.nbytes
    ...     for array in (
    ...         tables.electric, tables.magnetic, tables.backward_electric,
    ...         tables.backward_magnetic, tables.reciprocal_gram,
    ...         tables.coordinate_dual,
    ...     )
    ... )
    >>> estimate.groups["general_reference"] == held
    True
    """
    name = _estimated(backend)
    sampling = guide.sampling
    if sampling is None:
        msg = (
            "a guide without a sampling has no reference to estimate: build it with "
            "ModalGuide.from_provider, which declares the node rule and the knots"
        )
        raise BackendError(msg)
    nodes = int(sampling.nodes.weights.size)
    n_modes = len(guide.mode_collection.modes)
    n_omega = int(grid.omega.size)
    path = sampling.path
    if path is None:
        # The snapshot's reference is stored at the entrance and at the tables' far
        # end, and its maps are one coefficient of one interval.
        positions, terms = 2, 1
    else:
        kept = max(int(np.searchsorted(path.z_edges, float(z_max), side="left")), 1)
        positions, terms = kept + 1, kept * (path.degree + 1)
    shape = _RouteShape(
        n_modes=n_modes,
        nodes=nodes,
        region_nodes=int(sampling.nodes.region_index.size),
        components=3,
        n_omega=n_omega,
        n_to=grid.n_to,
        n_omega_fine=int(grid.omega_fine.size),
        terms=terms,
        n_stages=n_stages,
    )
    maps, chunk_bytes, _ = _route_bytes(shape, name)
    # E, H and their backward partners, 3 components each, the coordinate dual, 6
    # rows per node, and the M x M reciprocal Gram, at every stored position and
    # coarse bin.
    reference = 16 * positions * n_omega * n_modes * (nodes * (4 * 3 + 6) + n_modes)
    groups: dict[str, int] = dict.fromkeys(MEMORY_GROUPS, 0)
    groups |= {
        "general_maps": maps,
        "general_reference": reference,
        "general_chunk": chunk_bytes,
    }
    return MemoryEstimate(
        name,
        groups,
        not_counted=(
            "the base, delayed-response and coupled-flow groups, which translation "
            "fixes",
            "the knot table the reference is interpolated from, held while it is "
            "built (one provider sample per knot)",
            *_NOT_COUNTED[name],
        ),
    )


def ledger_groups(ledger: Mapping[str, int]) -> dict[str, int]:
    """Group a constructed backend's ledger as :func:`estimate_memory` groups it.

    Parameters
    ----------
    ledger : collections.abc.Mapping
        What :meth:`solprop.backends.protocol.Backend.memory_ledger` returned, on any
        of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Bytes for every one of :data:`MEMORY_GROUPS`, and ``measured``: the planning
        heap rows (``*_heap_growth``), measured at construction and not estimated.

    Examples
    --------
    >>> from solprop.backends.memory import ledger_groups
    >>> ledger_groups({"state": 16, "general_electric": 32, "fft_plan_heap_growth": 8})
    {'base': 16, 'general_maps': 0, 'general_reference': 32, 'general_chunk': 0, \
'general_delayed': 0, 'coupled_flow': 0, 'adams_history': 0, 'measured': 8}
    """
    groups: dict[str, int] = dict.fromkeys((*MEMORY_GROUPS, "measured"), 0)
    for row, size in ledger.items():
        groups[_group_of(row)] += int(size)
    return groups


def reference_base_memory(
    problem: KernelProblem,
    *,
    tabulating: bool,
    embedded: bool,
    noise: bool,
    statistics_weight: int,
    spatial_absorber: int,
) -> dict[str, int]:
    """Return the reference backend's rows outside the general route and the flow.

    One definition of the ``base`` group: ``numpy_ref``'s ``memory_ledger`` returns
    these rows and :func:`estimate_memory` sums them. What one step has resident at its
    peak (the reference allocates its buffers per pass); the space-domain rows are sized
    by ``transform.node_chunk``, not by the node count.

    **Method-aware** (SEP 24 section 6.5). The Runge-Kutta march holds the tableau's
    stage rows, two accumulators and the propagator pair or the six fractions' pairs.
    The low-storage march holds none of them: two registers beside the state
    (``register_accumulator`` and ``register_stage``) and, tabulating, the flow table
    in ``propagators`` -- one volume of ``exp(h D / 2)`` on a constant operator, the
    step's ``P`` and ``Q`` on a varying one, nothing when recomputing.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    tabulating : bool
        Whether the Runge-Kutta march's propagators are tabulated (six distinct
        exponent pairs held) or recomputed (one pair); the low-storage march counts its
        flows by :func:`low_storage_flows`.
    embedded : bool
        Whether the quadrature has an embedded coarse projection of its own.
    noise : bool
        Whether a noise field is loaded.
    statistics_weight : int
        Bytes of the statistics power weight, non-negative.
    spatial_absorber : int
        Bytes of the composed radial absorber matrix, non-negative.

    Returns
    -------
    dict
        Buffer name to bytes, every row named even when it is zero.
    """
    n_batch, n_modes, n_pol, n_omega = problem.shape
    field_bytes = n_batch * n_modes * n_pol * n_omega * 16
    stepping = _stepping_rows(problem, field_bytes, tabulating=tabulating)
    transform = problem.transform
    n_to = problem.grids.n_to
    chunk_columns = transform.node_chunk * transform.n_pol * n_pol
    n_omega_fine = n_omega_for(n_to, problem.grid_kind)
    element = 8 if problem.grid_kind == "field" else 16
    free = problem.linear.separable is not None
    temporal_columns = chunk_columns if free else n_modes * n_pol
    modal_time = temporal_columns * n_to * element
    envelope = problem.responses.envelope_plasma is not None
    tensor = transform.n_pol * n_pol == 2
    raman_kernels = (
        2
        if tensor
        and any(np.any(c.tensor_weights[:, 1]) for c in problem.responses.raman_terms)
        else 1
    )
    raman_columns = transform.node_chunk * (3 if raman_kernels == 2 else 1)
    flags = problem.flags
    plasma_nodes = flags.n_channels * transform.node_chunk * n_to * 8
    held = bool(problem.responses.plasma_channels)
    radial = transform.geometry is not None and transform.geometry.kind == "radial"
    held_columns = chunk_columns if free or radial else n_modes * n_pol
    return {
        # TDSE keeps the complete radial input while stable native chunks are
        # evaluated. Include subtraction operands and embedded P/J projection
        # transients in the same admission used by the constructed reference.
        "tdse_nodal_sources": 0
        if problem.tdse is None
        else 8 * transform.n_nodes * n_to * 8,
        "tdse_projected_sources": 0
        if problem.tdse is None
        else 6 * modal_time + 8 * field_bytes,
        # A gated source's edge term (SEP 26): the scaled g'S, its two absolute
        # rounding operands and the raw operand sum on the nodes; the projected edge
        # array and its two absolute projections, for the fine and embedded maps; and
        # the edge spectrum with 1/(i omega) and the diagnostic's band products.
        "tdse_edge_sources": 0
        if problem.tdse is None or not problem.tdse.gated
        else 4 * transform.n_nodes * n_to * 8 + 6 * modal_time + 8 * field_bytes,
        # SEP 27: total J and its two products coexist with the non-plasma
        # polarisation. Radial/free routes combine spectra on nodes; modal routes
        # first project both products, including the embedded quadrature.
        "held_current_streams": 3 * chunk_columns * n_to * 8 if held else 0,
        "held_current_projected": 2 * modal_time
        if held and not (free or radial)
        else 0,
        "held_current_embedded": 2 * modal_time if held and embedded else 0,
        "held_current_spectra": 2 * held_columns * n_omega_fine * 16 if held else 0,
        "held_current_derivative": n_to * 8 if held else 0,
        "envelope_source_streams": 2 * chunk_columns * n_to * 16 if envelope else 0,
        "envelope_histories": 5 * transform.node_chunk * n_to * 8 if envelope else 0,
        "envelope_current_scratch": chunk_columns * n_to * 16 if envelope else 0,
        "envelope_projected": 2 * modal_time if envelope else 0,
        "envelope_embedded_projected": 2 * modal_time if envelope and embedded else 0,
        "envelope_spectra": 2 * temporal_columns * n_omega_fine * 16 if envelope else 0,
        # A slab's physical nodes (n_x * n_y) are not its columns (n_p * n_y): the
        # reference holds the whole node volume of the synthesised field.
        "transverse_scratch": (
            n_batch * transform.n_nodes * n_pol * n_omega * 16
            if transform.geometry is not None and transform.geometry.kind == "slab"
            else field_bytes
            if free
            else 0
        ),
        "state": field_bytes,
        "statistics_power_weight": statistics_weight,
        "spatial_absorber_matrix": spatial_absorber,
        # The matrix product, summed time term and fine/coarse FFT outputs coexist
        # transiently in the reference's spatial absorber pass.
        "spatial_absorber_time": 3 * modal_time if spatial_absorber else 0,
        "spatial_absorber_spectrum": (
            n_modes * n_pol * n_omega_fine * 16 + 2 * field_bytes
            if spatial_absorber
            else 0
        ),
        **stepping,
        "noise": field_bytes if noise else 0,
        "fine_spectrum": temporal_columns * n_omega_fine * 16,
        "modal_time": modal_time,
        "projected": modal_time,
        "projected_coarse": modal_time if embedded else 0,
        "node_field": chunk_columns * n_to * element,
        "node_polarisation": chunk_columns * n_to * element,
        "node_ionisation_rate": plasma_nodes if flags.has_plasma else 0,
        "node_fraction": plasma_nodes if flags.has_plasma else 0,
        "plasma_channel_scalars": 2 * flags.n_channels * 8,
        "plasma_weighted_sums": 3 * transform.node_chunk * n_to * 8
        if flags.n_channels > 1
        else 0,
        "node_free_current": chunk_columns * n_to * 8 if flags.has_plasma else 0,
        # The Raman buffers live on the doubled grid: the real drive of every column in
        # the chunk, its one-sided spectrum, and the one damped kernel spectrum the
        # stage shares between the chunks.
        "node_raman_drive": raman_columns * 2 * n_to * 8 if flags.has_raman else 0,
        "node_raman_spectrum": raman_columns * (n_to + 1) * 16
        if flags.has_raman
        else 0,
        "node_raman_convolved": raman_columns * 2 * n_to * 8 if flags.has_raman else 0,
        "raman_kernel_spectrum": raman_kernels * (n_to + 1) * 16
        if flags.has_raman
        else 0,
    }


def held_current_memory(problem: KernelProblem) -> dict[str, int]:
    """Count Numba's held-current buffers before allocating its workspace.

    Parameters
    ----------
    problem : KernelProblem
        Translated problem. Without field plasma every row is zero.

    Returns
    -------
    dict
        Array bytes, excluding provider-owned FFT plan storage measured on
        construction. Includes the shared derivative table once.
    """
    transform, grids = problem.transform, problem.grids
    geometry = transform.geometry
    _, modes, components, frequencies = problem.shape
    nodes = min(transform.node_chunk, transform.n_nodes)
    columns = modes * components
    radial = geometry is not None and geometry.kind == "radial"
    if geometry is not None and geometry.kind == "slab":
        nx, ny = geometry.shape
        block = min(ny, max(1, transform.node_chunk // nx))
        nodes = block * nx
        columns = block * transform.synthesis.shape[1] * components
    elif problem.linear.separable is not None:
        columns = nodes * transform.n_pol * components
    chunk = nodes * transform.n_pol * components
    columns = chunk if radial else columns
    streams = 3 if radial else 2
    time = columns * grids.n_to * 8
    spectral = columns * (grids.n_to // 2 + 1) * 16
    embedded = (geometry is None or geometry.kind != "slab") and not np.array_equal(
        transform.projection, transform.projection_coarse
    )
    ledger = {
        "held_current_current": chunk * grids.n_to * 8,
        "held_current_sources": 2 * chunk * grids.n_to * 8,
        "held_current_projected": streams * time,
        "held_current_spectrum": streams * spectral,
        "held_current_projection_scratch": 0 if radial else time,
        "held_current_coarse_projected": streams * time if embedded else 0,
        "held_current_coarse_spectrum": streams * spectral if embedded else 0,
        "held_current_nodal_spectrum": chunk * frequencies * 16 if radial else 0,
        "held_current_radial_projection": modes * components * frequencies * 16
        if radial
        else 0,
        "held_current_inverse": frequencies * 8,
        "held_current_derivative": grids.n_to * 8,
    }
    return ledger if problem.responses.plasma_channels else dict.fromkeys(ledger, 0)


def _native_held_current_memory(problem: KernelProblem) -> dict[str, int]:
    """Count native current streams, embedded projections and radial packing.

    The shared shapes match Numba's; Kokkos holds one stacked embedded projection
    per stream and a packed radial projection matrix instead of Numba's shared
    projection scratch. Its owning host collar copy is counted separately.

    Parameters
    ----------
    problem : KernelProblem
        Translated problem with complete current-stream extents.

    Returns
    -------
    dict
        Native owned bytes per ledger entry; zero without field plasma.
    """
    shared = held_current_memory(problem)
    transform = problem.transform
    geometry = transform.geometry
    radial = geometry is not None and geometry.kind == "radial"
    streams = 3 if radial else 2
    per_stream = (
        sum(
            shared[key]
            for key in (
                "held_current_projected",
                "held_current_spectrum",
                "held_current_coarse_spectrum",
            )
        )
        // streams
    )
    # Fine and coarse outputs plus two stacked rows for each embedded stream.
    per_stream += 3 * shared["held_current_coarse_projected"] // streams
    source = shared["held_current_sources"] // 2
    return {
        "held_current": shared["held_current_current"],
        "held_current_gj": source + per_stream,
        "held_current_edge": source + per_stream,
        "held_current_nonplasma": per_stream if radial else 0,
        "held_current_projection": (
            transform.n_modes
            * min(transform.node_chunk, transform.n_nodes)
            * transform.n_pol
            * 8
            if radial and problem.responses.plasma_channels
            else 0
        ),
        "held_current_radial_spectra": shared["held_current_nodal_spectrum"]
        + shared["held_current_radial_projection"],
        "held_current_tables": shared["held_current_inverse"]
        + shared["held_current_derivative"],
        "held_current_host_derivative": shared["held_current_derivative"],
    }


def _stepping_rows(
    problem: KernelProblem, field_bytes: int, *, tabulating: bool
) -> dict[str, int]:
    """Return the march's own field-sized rows: stages, accumulators, registers, flows.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, whose integrator decides the rows.
    field_bytes : int
        One state's bytes.
    tabulating : bool
        Whether the propagators are tabulated.

    Returns
    -------
    dict
        Buffer name to bytes, every row named even when it is zero.
    """
    if isinstance(problem.stepper.integrator, LowStorageRungeKutta):
        return _low_storage_rows(problem, field_bytes)
    distinct = _distinct_fractions(problem)
    return {
        "stages": problem.stepper.tableau.n_stages * field_bytes,
        "accumulator_solution": field_bytes,
        "accumulator_error": field_bytes,
        "register_accumulator": 0,
        "register_stage": 0,
        "propagators": 2 * distinct * field_bytes if tabulating else 2 * field_bytes,
    }


def _low_storage_rows(problem: KernelProblem, field_bytes: int) -> dict[str, int]:
    """Return the low-storage march's field-sized rows: two registers and the flows.

    Every backend that runs the march holds these, the compiled engines to the byte
    (SEP 24 section 6.6), the flows counted by :func:`low_storage_flows`: one volume of
    ``exp(h D / 2)`` on a constant operator, a varying one's ``P`` and ``Q`` formed
    once per step, nothing when recomputing.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    field_bytes : int
        One state's bytes.

    Returns
    -------
    dict
        Buffer name to bytes, every row of :func:`_stepping_rows` named.
    """
    return {
        "stages": 0,
        "accumulator_solution": 0,
        "accumulator_error": 0,
        "register_accumulator": field_bytes,
        "register_stage": field_bytes,
        "propagators": low_storage_flows(problem, "numpy_ref") * field_bytes,
    }


def low_storage_flows(problem: KernelProblem, backend: str) -> int:
    """Return how many flow volumes a compiled backend's low-storage march holds.

    The one predicate the engines and the estimate share (SEP 24 section 4.3): with
    ``propagator_mode="auto"`` resolved for this march as
    ``StepDriver._propagator_mode`` resolves it on a CPU, a tabulated constant operator
    -- every separable one included, whose ``phi_representation`` is ``"separable"`` --
    holds one volume, ``exp(h_t D / 2)``; a tabulated varying operator holds the
    step's ``P`` and ``Q``; recomputing holds none. A native device build resolves
    ``"auto"`` to recomputing and holds none, which this CPU count over-states by the
    table: the estimate is then conservative.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        A registry name; every backend that runs the march counts the same.

    Returns
    -------
    int
        0, 1 or 2 volumes of one state's size.

    Examples
    --------
    >>> from solprop.backends.memory import low_storage_flows
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> low_storage_flows(problem, "numba_cpu")  # a constant operator's table
    1
    """
    del backend
    asked = problem.stepper.propagator_mode
    constant = problem.flags.constant_linop
    if asked == "recompute" or (asked == "auto" and not constant):
        return 0
    return 1 if constant else 2


def low_storage_flow_scratch(problem: KernelProblem, backend: str) -> bool:
    """Return whether a backend's low-storage march holds one flow scratch volume.

    ``numba_cpu``'s modal and general routes, recomputing, form the step's flows
    ``P`` (read by pass 1) and ``Q`` (passes 3 and 4) one after the other in one
    reusable volume, rather than as transients outside the ledger. The separable
    kinds form them per element, and ``native`` forms a modal route's from the phase
    terms per element, so neither holds it; ``numpy_ref`` allocates its flows per call,
    as it allocates every transient.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        A registry name.

    Returns
    -------
    bool
        Whether the scratch is held, one state's bytes, in the ``propagators`` row.

    Examples
    --------
    >>> import dataclasses
    >>> from solprop.backends.memory import low_storage_flow_scratch
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> recomputing = dataclasses.replace(
    ...     problem,
    ...     stepper=dataclasses.replace(problem.stepper, propagator_mode="recompute"),
    ... )
    >>> low_storage_flow_scratch(recomputing, "numba_cpu")
    True
    >>> low_storage_flow_scratch(problem, "numba_cpu")  # the table is held instead
    False
    """
    geometry = problem.transform.geometry
    modal = problem.linear.separable is None and (
        geometry is None or geometry.kind != "slab"
    )
    return backend == "numba_cpu" and modal and low_storage_flows(problem, backend) == 0


def low_storage_stage_register(problem: KernelProblem, backend: str) -> bool:
    """Return whether a backend's low-storage march holds the stage register ``C``.

    SEP 24 Q4, decided in phase 2: ``numba_cpu`` (and Kokkos, from phase 3) elides it
    on the Cartesian and line routes where a pass reads its input once -- a material
    response and no in-stage absorber -- since each finish can write the next pass's
    input straight into the transverse scratch. Under an in-stage absorber the
    exchange re-forms the input from ``C`` for the absorber's evaluation, so it is
    held; the modal and planar routes and ``numpy_ref`` hold it.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        A registry name.

    Returns
    -------
    bool
        Whether ``C`` is held, one state's bytes.

    Examples
    --------
    >>> from solprop.backends.memory import low_storage_stage_register
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> low_storage_stage_register(problem, "numba_cpu")  # a modal route holds C
    True
    """
    geometry = problem.transform.geometry
    responses = problem.responses
    free = (
        problem.linear.separable is not None
        and geometry is not None
        and geometry.kind != "slab"
    )
    if backend not in {"numba_cpu", "native"} or not free or geometry is None:
        return True
    in_stage = bool(np.any(geometry.absorber_rate)) or (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    )
    return in_stage or len(responses.response_order) <= 1


def low_storage_memory(problem: KernelProblem, backend: str) -> int:
    """Return the persistent bytes a run would take under the low-storage march.

    The same run's :func:`estimate_memory` with its Adams history dropped and its
    march's stepping rows -- stage rows, accumulators and propagators -- replaced by
    the low-storage march's two registers and flow table (SEP 24 section 6.6), the
    table taken as ``propagator_mode="auto"`` takes it for that march: the figure the
    admission names beside its refusal of an Adams history that does not fit.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, under any integrator.
    backend : str
        A registry name, one of :data:`ESTIMATED_BACKENDS`, or ``"auto"``.

    Returns
    -------
    int
        Bytes.

    Examples
    --------
    >>> from solprop.backends.memory import estimate_memory, low_storage_memory
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> low_storage_memory(problem, "numba_cpu") < estimate_memory(
    ...     problem, "numba_cpu").total
    True
    """
    estimate = estimate_memory(problem, backend)
    field_bytes = 16 * math.prod(problem.shape)
    asked = problem.stepper.propagator_mode
    current = _stepping_rows(
        problem, field_bytes, tabulating=_tabulating(problem, asked)
    )
    march = _low_storage_rows(problem, field_bytes)
    scratch = field_bytes if low_storage_flow_scratch(problem, estimate.backend) else 0
    if not low_storage_stage_register(problem, estimate.backend):
        march["register_stage"] = 0
    march["propagators"] += scratch
    if isinstance(problem.stepper.integrator, LowStorageRungeKutta):
        # The estimate already took the elided register out of its base, and the
        # flow scratch into it.
        if not low_storage_stage_register(problem, estimate.backend):
            current["register_stage"] = 0
        current["propagators"] += scratch
    return (
        estimate.total
        - int(estimate.groups["adams_history"])
        - sum(current.values())
        + sum(march.values())
    )


def _tabulating(problem: KernelProblem, asked: str) -> bool:
    """Return whether a problem's propagators are tabulated, ``"auto"`` resolved.

    ``StepDriver._propagator_mode`` with the reference's default of ``"tabulate"``;
    the low-storage march tabulates a constant operator and recomputes a varying one.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    asked : str
        The contract's propagator mode.

    Returns
    -------
    bool
        Whether the propagators are tabulated.
    """
    return asked == "tabulate" or (
        asked == "auto"
        and (
            problem.flags.constant_linop
            if isinstance(problem.stepper.integrator, LowStorageRungeKutta)
            else problem.linear.separable is None
        )
    )


def _estimated(backend: str) -> str:
    """Return the registry name of a backend this module has formulas for.

    Parameters
    ----------
    backend : str
        A registry name or ``"auto"``.

    Returns
    -------
    str
        The resolved name.

    Raises
    ------
    BackendError
        If it is not one of :data:`ESTIMATED_BACKENDS`.
    """
    name = resolve_name(backend)
    if name not in ESTIMATED_BACKENDS:
        msg = (
            f"no memory estimate for the backend {name!r}; the estimate has formulas "
            f"for {', '.join(ESTIMATED_BACKENDS)}"
        )
        raise BackendError(msg)
    return name


def _group_of(row: str) -> str:
    """Return the group a ledger row belongs to.

    Parameters
    ----------
    row : str
        A ledger row's name.

    Returns
    -------
    str
        One of :data:`MEMORY_GROUPS`, or ``"measured"`` for a planning heap row.
    """
    if row.endswith("heap_growth"):
        return "measured"
    if row.startswith(("coupled_flow_", "flow_")):
        return "coupled_flow"
    if row.startswith("adams_"):
        return "adams_history"
    if not row.startswith("general_"):
        return "base"
    return next(
        (group for group, rows in _GENERAL_ROWS if row in rows), "general_chunk"
    )


def adams_memory(problem: KernelProblem, backend: str) -> dict[str, int]:
    """Return the Adams march's history rows a backend holds for a problem.

    The rows of the ``adams_history`` group, which the backends' ledgers report and
    :func:`estimate_memory` sums (module docstring). Empty without an Adams
    integrator, and on the general path, whose history is a list of arrays.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        One of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Row name to bytes.

    Examples
    --------
    >>> from solprop.backends.memory import adams_memory
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.stepper.multistep import Multistep
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0,
    ...                    integrator=Multistep(order=4))
    >>> adams_memory(problem, "numpy_ref") == {"adams_store": 6 * 16 * grid.n_omega}
    True
    """
    from solprop.backends.multistep import source_absent  # noqa: PLC0415 - import cycle
    from solprop.stepper.multistep import (  # noqa: PLC0415
        MAX_ADAMS_ORDER,
        Multistep,
    )

    method = problem.stepper.integrator
    if not isinstance(method, Multistep):
        return {}
    exponential = method.formulation == "exponential"
    fused = exponential or (
        method.corrector == "pec"
        and all(
            method.corrector_nodes(order) == order + 1
            for order in range(1, method.order + 1)
        )
    )
    if not fused:
        return {}
    state = 16 * math.prod(problem.shape)
    slots = method.order + 2
    flow = problem.linear.coupled_flow
    # A frozen-residual flow's residual joins the compensation whatever the diagonal's
    # representation (SEP 16 phase 5).
    residual = flow is not None and "frozen_residual" in flow.section_methods
    compensated = exponential and (
        residual
        or (
            problem.linear.phi_representation in ("expansion", "table")
            and not source_absent(problem)
        )
    )
    if backend == "native":
        if flow is not None and not _resident_coupled(problem, flow):
            # The host engine holds the history in host arrays, outside the ledger.
            return {}
        # The resident engine (Kokkos's _adams_engine), its rows in one native entry.
        dense = problem.stepper.interpolate_planes
        rows = (2 * slots + 4) * state + 8 * problem.grids.n_omega
        if dense:
            rows += (slots + 4) * state
        controller = problem.stepper.controller
        if (
            isinstance(controller, PIController)
            and type(controller).scale is PIController.scale
        ):
            # solprop's own controller's scale is formed on the device: the magnitude's
            # sum and, for a weighted norm, the weights.
            weights = 0 if controller.weights is None else problem.grids.n_omega
            rows += 8 * (1 + weights)
        if compensated:
            # The rate's weights over the terms of ``Phi``: an expansion's terms (a
            # constant ``Phi`` is compensated only beside a frozen-residual flow, which
            # carries it as an expansion), or a z table's position slots and the rate's
            # own slot on the CPU build (SEP 29 section 6.6).
            expansion = problem.linear.phi_expansion
            terms = (
                int(expansion.terms.shape[0])
                if expansion is not None
                else z_table_slots(problem) + 1
                if problem.linear.phi_representation == "table"
                else 0
            )
            rows += (3 * slots + 3) * state + 8 * terms
            if dense:
                rows += slots * state
        if flow is not None:
            # The block's flow and moments beside the history; a residual flow's
            # generator applied at each sample and at the trial's end, and the end's
            # weights (SEP 16 phase 6).
            generator = flow.generator
            assert generator is not None
            active = int(np.count_nonzero(np.any(generator != 0.0, axis=(0, 1, 2, 3))))
            rows += 16 * (slots + 1) * active * flow.n_modes * flow.n_modes
            if compensated and flow.section_methods != ("constant_schur",):
                terms = len(flow.section_methods) * flow.n_deg
                rows += (slots + 1) * state + 8 * terms
        return {"adams_history": rows}
    ledger = {"adams_store": slots * state}
    if compensated:
        ledger |= {"adams_states": slots * state, "adams_rates": slots * state}
    if backend == "numba_cpu" and exponential:
        ledger |= {"adams_moments": (MAX_ADAMS_ORDER + 2) * state, "adams_flow": state}
    return ledger


def far_field_memory(problem: KernelProblem, backend: str) -> dict[str, int]:
    """Return the far-field pass's band partials on a backend, by row (SEP 23).

    A row of the ``base`` group, to the byte: ``numba_cpu``'s ``statistics_partials``,
    the partials of :func:`solprop.backends.spatial.far_field_partials` and their
    band sum; Kokkos's ``far_field_partials``, the partials alone (its native pass
    adds the bands as it finishes them), without the ``W = 1`` slot for
    ``on_axis_intensity``, which Kokkos forms in its own pass; empty on ``numpy_ref``,
    whose pass allocates per call.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        One of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Row name to bytes.
    """
    if backend == "numpy_ref":
        return {}
    native = backend == "native"
    n_slots, n_bands = far_field_partials(
        problem, axis_folded=AXIS_SPECTRUM_ONLY if native else AXIS_FOLDED
    )
    if n_slots == 0:
        return {}
    if native:
        return {"far_field_partials": 8 * n_bands * n_slots * problem.shape[3]}
    return {"statistics_partials": 8 * (n_bands + 1) * n_slots * problem.shape[3]}


LOCAL_FLUX_STATISTICS = frozenset(
    {"on_axis_intensity", "peak_intensity", "beam_radius", "fwhm_r"}
)
"""The spatial statistics that synthesise the magnetic closure in time (SEP 34)."""


def flux_closure_memory(problem: KernelProblem, backend: str) -> dict[str, int]:
    """Return the magnetic closure's transverse volume a backend holds (SEP 34).

    A row of the ``base`` group, to the byte: on Kokkos a Cartesian, line or planar
    run whose own request has a local flux statistic holds the closure's
    transverse volume beside the transform's scratch, one state's bytes. Its time
    rows and per-node magnitudes are chunk-sized statistics buffers, which the
    reference's count already stands for; ``numpy_ref`` and ``numba_cpu`` form
    the closure per call.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        One of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Row name to bytes.
    """
    geometry = problem.transform.geometry
    if (
        backend != "native"
        or geometry is None
        or geometry.kind not in ("free", "slab")
        or not set(problem.stats_spec.names) & LOCAL_FLUX_STATISTICS
    ):
        return {}
    return {"flux_closure_transverse": 16 * math.prod(problem.shape[1:])}


def dispersive_memory(problem: KernelProblem, backend: str) -> dict[str, int]:
    """Return the buffers SEP 6's dispersive response holds on a backend, by row.

    Rows of the ``base`` group (decided by John Travers, 2026-10-08, SEP 6 Phase 2c,
    open question 3), empty without the response and on ``numpy_ref`` (whose pass
    allocates per call).

    - **The fused node path** (mode-averaged, multimode, radial): the compiled
      backends' own rows, to the byte -- one structure on both (SEP 6, Phases 2a and
      3a). Each group's filtered fields over the modal columns, unless a lone symmetric
      response filters the stage's own spectrum in place (no in-stage absorber reads
      the bare field); the node slots of a chunk, one beside other responses or two
      for the polynomial, none alone; the products and, with an embedded rule, its
      products, with their spectra, unless the lone product is the stage's own
      polarisation (all but a field grid's analytic family).
    - **The Cartesian, line and planar routes**: on ``native`` its rows, to the byte
      (SEP 6 Phase 3b): a chunk's products and their spectra over its temporal rows,
      unless the lone product is the chunk's own polarisation; a lone symmetric response
      filters in place, the native in-stage absorber being a pass of its own; free
      space's filtered fields are blocks of its products, the planar route's are its
      own, with one node slot (symmetric) or two (polynomial) of a block. On
      ``numba_cpu``, the complex-equivalent buffers per node and component the node
      budget admits for the response
      (:func:`~solprop.translate.problem.dispersive_node_buffers`) times the chunk, an
      upper bound of its holdings.
    - **The general modal route**: on ``native`` its rows, to the byte: per chunk of the
      route's own (which counts the response,
      :func:`~solprop.backends.numpy_ref.general_modal.general_dispersive_buffers`),
      the filtered fields and their spectra, the products and theirs. On
      ``numba_cpu`` the route's chunk group counts them instead (``_route_bytes``).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        One of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Row name to bytes.
    """
    leaf = problem.responses.dispersive_kerr
    transform = problem.transform
    if leaf is None or backend == "numpy_ref":
        return {}
    if transform.modal_tables is not None:
        return _general_dispersive_memory(problem) if backend == "native" else {}
    responses = problem.responses
    _, n_modes, n_pol, _ = problem.shape
    n_to = problem.grids.n_to
    components = transform.n_pol * n_pol
    chunk = min(transform.node_chunk, transform.n_nodes)
    geometry = transform.geometry
    if geometry is not None and geometry.kind in ("free", "slab"):
        if backend == "native":
            return _separable_dispersive_memory(problem)
        buffers = dispersive_node_buffers(responses, spatial=True)
        return {"dispersive_node_buffers": 16 * chunk * components * buffers * n_to}
    symmetric = leaf.evaluator == "symmetric"
    field_grid = problem.grid_kind == "field"
    complex_time = not (field_grid and leaf.sector == "full")
    analytic = field_grid and leaf.sector == "conserving"
    element = 16 if complex_time else 8
    groups = len(leaf.groups())
    fields = groups if symmetric else 2 * groups
    products = groups if symmetric else 1 + 2 * groups
    alone = (
        set(responses.response_order) - {ResponseStage.ACCUMULATE}
        == {ResponseStage.DISPERSIVE_KERR}
        and responses.envelope_plasma is None
    )
    lone = alone and symmetric and products == 1
    in_stage = (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    ) or _radial_absorber_bytes(problem) > 0
    in_place = lone and not in_stage
    borrowed = lone and not analytic
    embedded = transform.n_nodes > 1 and not np.array_equal(
        transform.projection, transform.projection_coarse
    )
    column = n_modes * n_pol * n_to * element
    bins = n_to if complex_time else n_to // 2 + 1
    return {
        "dispersive_filtered_time": 0 if in_place else fields * column,
        "dispersive_node_fields": 0
        if lone
        else (1 if symmetric else 2) * chunk * components * n_to * element,
        "dispersive_products": 0 if borrowed else products * column,
        "dispersive_coarse_products": products * column
        if embedded and not borrowed
        else 0,
        "dispersive_product_spectra": 0
        if borrowed
        else products * n_modes * n_pol * bins * 16,
    }


def _dispersive_alone(problem: KernelProblem) -> bool:
    """Return whether SEP 6's dispersive response is the stage's only source.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    bool
        Whether no constant Kerr term, Raman, plasma or envelope plasma runs beside it.
    """
    responses = problem.responses
    return (
        set(responses.response_order) - {ResponseStage.ACCUMULATE}
        == {ResponseStage.DISPERSIVE_KERR}
        and responses.envelope_plasma is None
    )


def _separable_dispersive_memory(problem: KernelProblem) -> dict[str, int]:
    """Return Kokkos's dispersive rows on the Cartesian, line and planar routes.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem on a free-space or planar geometry carrying the response.

    Returns
    -------
    dict
        Row name to bytes, as Kokkos's ``SeparableDispersive`` holds them.
    """
    leaf = problem.responses.dispersive_kerr
    transform = problem.transform
    geometry = transform.geometry
    assert leaf is not None and geometry is not None
    n_to = problem.grids.n_to
    symmetric = leaf.evaluator == "symmetric"
    field_grid = problem.grid_kind == "field"
    complex_time = not (field_grid and leaf.sector == "full")
    analytic = field_grid and leaf.sector == "conserving"
    element = 16 if complex_time else 8
    bins = n_to if complex_time else n_to // 2 + 1
    groups = len(leaf.groups())
    fields = groups if symmetric else 2 * groups
    products = groups if symmetric else 1 + 2 * groups
    lone = _dispersive_alone(problem) and symmetric and products == 1
    borrowed = lone and not analytic
    slab = geometry.kind == "slab"
    if slab:
        # Whole y rows per chunk (the native adapter's blocking): the block's (p, y)
        # rows go to time, its (x, y) nodes take the products.
        n_x, n_y = geometry.shape
        units = min(n_y, max(1, transform.node_chunk // n_x))
        rows = units * (problem.shape[1] // n_y)
        node_rows = units * n_x
    else:
        rows = min(transform.node_chunk, transform.n_nodes) * problem.shape[2]
        node_rows = rows
    return {
        "dispersive_filtered_time": fields * rows * n_to * element
        if slab and not lone
        else 0,
        "dispersive_node_fields": (1 if symmetric else 2) * node_rows * n_to * element
        if slab and not lone
        else 0,
        "dispersive_products": 0 if borrowed else products * rows * n_to * element,
        "dispersive_product_spectra": 0 if borrowed else products * rows * bins * 16,
    }


def _general_dispersive_memory(problem: KernelProblem) -> dict[str, int]:
    """Return Kokkos's dispersive rows on the general modal route.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem on the general modal route carrying the response.

    Returns
    -------
    dict
        Row name to bytes, as Kokkos's ``GeneralDispersive`` holds them.
    """
    # Imported here: the reference backend imports this module for its base rows.
    from solprop.backends.numpy_ref.general_modal import (  # noqa: PLC0415 - see above
        general_dispersive_buffers,
        general_node_chunk,
    )

    leaf = problem.responses.dispersive_kerr
    tables = problem.transform.modal_tables
    assert leaf is not None and tables is not None
    n_to = problem.grids.n_to
    components = int(tables.electric.shape[2])
    chunk = min(
        general_node_chunk(
            components,
            n_to,
            int(tables.omega_fine.size),
            DEFAULT_NODE_BUDGET,
            dispersive=general_dispersive_buffers(leaf),
        ),
        int(tables.region_index.size),
    )
    rows = chunk * components
    real = leaf.sector == "full"
    groups = len(leaf.groups())
    if leaf.evaluator == "symmetric":
        fields = products = groups
    else:
        assert leaf.regions is not None
        fields = 2 * groups
        products = len(set(leaf.regions.tolist())) + 2 * groups
    bins = n_to // 2 + 1 if real else n_to
    return {
        "dispersive_filtered_time": 16 * fields * rows * n_to,
        "dispersive_products": (8 if real else 16) * products * rows * n_to,
        "dispersive_product_spectra": 16
        * (products * rows * bins + fields * rows * n_to),
    }


RESIDENT_CHANNELS = 16
"""The most channels of a coupled block ``native``'s resident Adams engine steps.

``solprop_native.backend.ADAMS_MAX_CHANNELS``; a larger block takes the host engine.
"""


def _resident_coupled(problem: KernelProblem, flow: CoupledFlowTables) -> bool:
    """Return whether ``native``'s resident Adams engine steps a coupled problem.

    It does when every section's every bin is on the flow actions' eigenbasis route,
    the block has at most :data:`RESIDENT_CHANNELS` channels and the state one batch
    row (``NativeBackend._adams_engine``); otherwise the host engine does.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    flow : solprop.translate.coupled.CoupledFlowTables
        Its coupled flow.

    Returns
    -------
    bool
        Whether the history is resident.
    """
    return (
        problem.shape[0] == 1
        and flow.n_modes <= RESIDENT_CHANNELS
        and _fallback_bins(flow) == 0
    )


def z_table_slots(problem: KernelProblem) -> int:
    """Return the position slots the native CPU build holds for a z table.

    SEP 29 section 6.6: the native backend evaluates a z table at each native call's
    distinct positions into term slots of the state's size, which the stage weights
    then select. A Runge-Kutta trial reads the step's start, its stage nodes and its
    end, so the tableau's distinct fractions with 0 and 1; the low-storage march
    reads the start, the midpoint and the end, and a constant operator's table
    interval, five at most; an Adams call reads two.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    int
        The slot count, at least two; ``Phi``'s and the prefactor's tables share them.

    Examples
    --------
    >>> from solprop.backends.memory import z_table_slots
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> z_table_slots(problem)  # DOPRI5: 0, 1/5, 3/10, 4/5, 8/9 and 1
    6
    """
    fractions = np.concatenate(([0.0, 1.0], problem.stepper.tableau.nodes))
    low_storage = (
        5 if isinstance(problem.stepper.integrator, LowStorageRungeKutta) else 0
    )
    return int(max(np.unique(fractions).size, low_storage, 2))


def _distinct_fractions(problem: KernelProblem) -> int:
    """Return the tableau's distinct stage fractions, the step's end included.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    int
        The count, positive.
    """
    return int(np.unique(np.append(problem.stepper.tableau.nodes, 1.0)).size)


def _reference_base(problem: KernelProblem) -> dict[str, int]:
    """Return the reference's base rows as its construction would fix them.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    dict
        Buffer name to bytes, :func:`reference_base_memory`'s rows.
    """
    transform = problem.transform
    return reference_base_memory(
        problem,
        tabulating=_tabulating(problem, problem.stepper.propagator_mode),
        embedded=not np.array_equal(transform.projection, transform.projection_coarse),
        noise=False,
        statistics_weight=8 * math.prod(problem.shape[1:])
        if transform.geometry is None
        else 0,
        spatial_absorber=_radial_absorber_bytes(problem),
    )


def _radial_absorber_bytes(problem: KernelProblem) -> int:
    """Return the bytes of the radial absorber matrix the reference composes.

    :func:`solprop.backends.spatial.radial_absorber_matrix`'s shape and type, without
    forming the product.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    int
        Bytes, zero where the problem has no radial absorber.
    """
    transform = problem.transform
    geometry = transform.geometry
    if (
        geometry is None
        or geometry.kind != "radial"
        or not np.any(geometry.absorber_rate)
    ):
        return 0
    item = np.result_type(
        transform.projection.dtype, transform.synthesis.dtype, np.float64
    ).itemsize
    return int(transform.projection.shape[0] * transform.synthesis.shape[-1] * item)


def _general_groups(problem: KernelProblem, backend: str) -> dict[str, int]:
    """Return the general modal route's four groups.

    ``S = 16 Q C M n_omega`` is one dense map; ``K`` the node chunk; the delayed
    response's kernels are its scaled tensors and its doubled-axis spectra.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        A name of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    dict
        Bytes of the four ``general_*`` groups, or empty on another route.
    """
    # Imported here: the reference backend imports this module for its base rows.
    from solprop.backends.numpy_ref.general_modal import (  # noqa: PLC0415 - see above
        general_dispersive_buffers,
    )

    tables = problem.transform.modal_tables
    if tables is None:
        return {}

    _, n_modes, _, n_omega = problem.shape
    n_to = problem.grids.n_to
    nodes = int(tables.electric.shape[1])
    components = int(tables.electric.shape[2])
    # A moving guide holds every coefficient of both maps,
    # sections times degree of them, and evaluates one of each per stage.
    assert tables.synthesis is not None
    terms = int(tables.synthesis.shape[0] * tables.synthesis.shape[1])
    shape = _RouteShape(
        n_modes=n_modes,
        nodes=nodes,
        region_nodes=int(tables.region_index.size),
        components=components,
        n_omega=n_omega,
        n_to=n_to,
        n_omega_fine=int(tables.omega_fine.size),
        terms=terms,
        n_stages=problem.stepper.tableau.n_stages,
        dispersive=general_dispersive_buffers(problem.responses.dispersive_kerr),
    )
    maps, chunk_bytes, chunk = _route_bytes(shape, backend)
    reference = sum(
        int(array.nbytes)
        for array in (
            tables.electric,
            tables.magnetic,
            tables.backward_electric,
            tables.backward_magnetic,
            tables.reciprocal_gram,
            tables.coordinate_dual,
        )
        if array is not None
    )
    if backend == "numpy_ref":
        delayed = 8 * chunk * n_to * (
            components**2 + _PAIR_HISTORIES + components
        ) + 16 * _PAIR_HISTORIES * chunk * (n_to + 1)
    else:
        # The compiled backends plan the doubled delayed axis at full length.
        delayed = (
            2 * 8 * chunk * _PAIR_HISTORIES * 2 * n_to
            + 2 * 16 * _PAIR_HISTORIES * chunk * (n_to + 1)
            + 8 * chunk * components * n_to
            + (8 * nodes if backend == "native" else 0)
        )
    regional = problem.responses.regional_responses
    if regional is None or not regional.n_terms:
        delayed = 0
    else:
        kernels = math.prod(regional.delayed_kernel.shape[2:-1])
        delayed += 8 * kernels * components**4 + 16 * kernels * (n_to + 1)
    return {
        "general_maps": maps,
        "general_reference": reference,
        "general_chunk": chunk_bytes,
        "general_delayed": delayed,
    }


@dataclass(frozen=True)
class _RouteShape:
    """The sizes the general route's maps and node chunk are fixed by.

    ``terms`` is the maps' coefficients along z, sections times degree plus one, one
    for a snapshot; ``region_nodes`` the nodes the chunk may cover; ``dispersive`` the
    buffers per node and component SEP 6's dispersive response adds to a chunk.
    """

    n_modes: int
    nodes: int
    region_nodes: int
    components: int
    n_omega: int
    n_to: int
    n_omega_fine: int
    terms: int
    n_stages: int
    dispersive: int = 0


def _route_bytes(shape: _RouteShape, backend: str) -> tuple[int, int, int]:
    """Return the general route's map bytes, chunk bytes and node chunk.

    ``S = 16 Q C M n_omega`` is one dense map. A moving guide's maps are ``terms``
    coefficients each and a stage evaluates one of each; Kokkos also holds a trial's
    weights, one row per stage.

    Parameters
    ----------
    shape : _RouteShape
        The sizes.
    backend : str
        A name of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    tuple of int
        ``general_maps`` bytes, ``general_chunk`` bytes and the chunk in nodes.
    """
    # Imported here: the reference backend imports this module for its base rows.
    from solprop.backends.numpy_ref.general_modal import (  # noqa: PLC0415 - see above
        general_node_chunk,
    )

    components, n_to, n_omega = shape.components, shape.n_to, shape.n_omega
    chunk = min(
        general_node_chunk(
            components,
            n_to,
            shape.n_omega_fine,
            DEFAULT_NODE_BUDGET,
            dispersive=shape.dispersive,
        ),
        shape.region_nodes,
    )
    one_map = 16 * shape.nodes * components * shape.n_modes * n_omega
    channel = 16 * shape.n_modes * n_omega
    if backend == "numpy_ref":
        chunk_bytes = (
            16 * chunk * components * (2 * n_omega + (2 + 2 * components) * n_to)
            + channel
        )
    else:
        # The compiled backends keep a real polarisation and no per-node pair buffers.
        chunk_bytes = chunk * components * (40 * n_to + 16 * (n_to // 2 + 1))
        chunk_bytes += (
            8 * n_omega + 5 * channel
            if backend == "numba_cpu"
            else 16 * n_omega + 2 * channel
        )
        # SEP 6's dispersive response: Kokkos's rows are `dispersive_memory`'s, in
        # `base`, as its ledger names them.
        if backend == "numba_cpu":
            chunk_bytes += 16 * chunk * components * shape.dispersive * n_to
    stage = 2 * one_map if shape.terms > 1 else 0
    if shape.terms > 1 and backend == "native":
        stage += 8 * shape.n_stages * shape.terms
    maps = 2 * shape.terms * one_map + stage + 8 * components**4 * shape.nodes
    return maps, chunk_bytes, chunk


def _flow_bytes(problem: KernelProblem, backend: str) -> int:
    """Return the coupled flow's bytes.

    ``B = 16 A N^2`` is one dense block over the active bins; ``s`` sections of ``d``
    Chebyshev terms; ``c`` distinct stage fractions, the step's end included.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : str
        A name of :data:`ESTIMATED_BACKENDS`.

    Returns
    -------
    int
        Bytes, zero without a coupled flow.
    """
    flow = problem.linear.coupled_flow
    if flow is None or flow.generator is None:
        return 0
    n_omega = problem.shape[-1]
    sections, degree = int(flow.generator.shape[0]), int(flow.generator.shape[1])
    n_modes = int(flow.n_modes)
    active = int(np.count_nonzero(np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))))
    varying = flow.section_methods != ("constant_schur",)
    block = 16 * active * n_modes * n_modes
    column = 16 * n_omega * n_modes
    residual = sections * degree * block if varying else 0
    # The eigenbasis of every section's bins: `W` and `Y` (N x N) and the
    # eigenvalues, complex, with the route that says which bins take it.
    eigenbasis = 16 * sections * active * (2 * n_modes * n_modes + n_modes)
    if backend != "native":
        held = 3 * sections * block + 2 * _distinct_fractions(problem) * block
        held += residual
        if backend == "numba_cpu":
            # The two bin-to-block indices, the actions' and the residual's, one intp
            # per bin each.
            held += 3 * column + 16 * n_omega + (block if varying else 0)
            # The route as a boolean per bin, a row-sum norm per section, and the
            # formation kernels' scratch: two exponent rows, three work blocks and a
            # Gram block of one bin, and a measured conditioning per active bin.
            held += eigenbasis + sections * active + 8 * sections
            held += 16 * (2 * n_modes + 4 * n_modes * n_modes) + 8 * active
        return held
    # A forward and an inverse action per stage and for the whole step, the carried
    # vectors, and on a varying guide the frozen blocks, the residual's terms, the
    # lab-frame input, the stage weights and the listing of each
    # stage's non-zero weights with its counts; the Schur factors stay on the host.
    n_stages = problem.stepper.tableau.n_stages
    held = 2 * (n_stages + 1) * block + 5 * column + 8 * n_omega
    if varying:
        terms = sections * degree
        held += sections * block + residual + column
        held += 8 * n_stages * terms + 8 * n_stages * (terms + 1)
    # The eigenbasis with an index route per bin; the Schur factors and
    # three work blocks of every bin the eigenbasis does not take; and per active bin
    # two exponent rows, a Gram block and a measured conditioning, with the slots'
    # fractions.
    held += eigenbasis + 8 * sections * active
    held += 16 * 5 * n_modes * n_modes * _fallback_bins(flow)
    held += 16 * active * (2 * n_modes + n_modes * n_modes) + 8 * (n_stages + 1)
    held += 8 * active
    return held


def _fallback_bins(flow: CoupledFlowTables) -> int:
    """Return how many section bins take the Schur route rather than the eigenbasis.

    The one size no shape fixes: which bins' eigenbasis is well enough conditioned
    is decided per bin by :func:`solprop.translate.coupled.schur_eigenbasis`, so it is
    decided here as the native adapter decides it, from the same Schur factors. The
    decomposition's temporaries are of the flow's own block size and freed on return.

    Parameters
    ----------
    flow : solprop.translate.coupled.CoupledFlowTables
        The problem's coupled flow, with a generator.

    Returns
    -------
    int
        The count of (section, active bin) pairs formed by scaling and squaring.
    """
    generator = flow.generator
    assert generator is not None
    active = np.any(generator != 0.0, axis=(0, 1, 2, 3))
    fallback = 0
    for index in range(len(flow.section_methods)):
        frozen = np.moveaxis(frozen_generator(flow, index)[:, :, active], -1, 0)
        vectors, triangular = schur_blocks(np.moveaxis(frozen, 0, -1))
        fallback += int(np.count_nonzero(~schur_eigenbasis(vectors, triangular)[3]))
    return fallback

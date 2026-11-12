# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How many nodes a response pass holds at once: the backend's node-chunk plan.

A response pass synthesises the field at a chunk of the transform's nodes, evaluates
the responses there and projects back; the chunk bounds the space-domain buffers. It is
an execution choice, not part of the scientific problem (SEP 47 sections 4.3 and 6.5):
two runs that differ only in their chunks compute the same fields to rounding, which
the ``chunked`` rows of ``tests/data/convergence/spatial_accuracy.json`` assert. So the
translation records none, and each backend plans its chunk when it is prepared, from the
problem and its :class:`~solprop.backends.session.ExecutionOptions`
(:func:`plan_node_chunk`).

The rule is the one the translation applied before SEP 47 moved it here, so that every
default and every explicit budget chooses the chunk it chose then:

- **The budget.** An explicit ``node_budget`` is authoritative; without one a
  free-space, line or planar problem takes :data:`DEFAULT_SPATIAL_NODE_BUDGET` and every
  other :data:`DEFAULT_NODE_BUDGET`.
- **The base chunk.** A modal or radial node holds :data:`BUFFERS_PER_NODE` complex
  temporal columns per component; a Cartesian or planar column holds
  :data:`FREE_COLUMN_BUFFERS`. A mode-averaged run has one node, and a general modal
  guide's static matrices are a placeholder of one (its route sizes its own chunk from
  :func:`solprop.backends.numpy_ref.general_modal.general_node_chunk` at the same
  budget).
- **The response admissions** then shrink the chunk until each node's envelope plasma,
  Raman, dispersive and held-current buffers fit the same budget, refusing a budget that
  cannot hold one node (or one planar row), and the last step spreads the nodes evenly
  over the passes the admitted chunk needs.

The counts are conservative inventories, not the live buffers of each route: SEP 28's
phase 0 found the twenty Cartesian column-equivalents about three times the live group
buffers (solprop issue #349). They stay, so that a user's budget keeps buying the chunk
it bought before; counting a route's live buffers changes the chunk of every explicit
budget, which is the maintainer's decision under #349 and #89, not this move's.

A device backend may choose differently when no budget is given (solprop issue #361);
the native backend's rule is ``solprop_native.backend.device_node_chunk``, and an
explicit budget or chunk is never overridden by it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.translate.responses import ResponseStage

if TYPE_CHECKING:
    from solprop.translate.modal import Transform
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.responses import CompiledResponses

__all__ = [
    "BUFFERS_PER_NODE",
    "DEFAULT_NODE_BUDGET",
    "DEFAULT_SPATIAL_NODE_BUDGET",
    "FREE_COLUMN_BUFFERS",
    "ChunkSource",
    "NodeChunkPlan",
    "default_node_budget",
    "dispersive_node_buffers",
    "plan_node_chunk",
]

DEFAULT_NODE_BUDGET = 64 * 2**20
"""Bytes of node-chunk buffers a problem may hold, before the chunk size is reduced.

Architecture section 13 chooses the whole node set when it fits a per-problem budget and
the largest chunk that fits otherwise. Sixty-four mebibytes reproduces the chunk of
about sixteen nodes that section 8 tabulates for the comb-to-continuum target (32768
time samples, six modes, Raman) while leaving a small problem un-chunked. It is an
estimate until the benchmark suite measures the crossover.

The free-space, line and planar grids default to :data:`DEFAULT_SPATIAL_NODE_BUDGET`.
"""

DEFAULT_SPATIAL_NODE_BUDGET = 2**30
"""Node-chunk bytes a free-space, line or planar problem may hold by default (SEP 28).

A spatial evaluation pays a fixed cost per node chunk: its launches, its temporal
transforms' thread wake-ups, and its gather and scatter. A field volume is gigabytes, so
a larger group is cheap in memory. With the per-chunk copies and clears removed, one
gibibyte was the fastest of 64, 256 and 1024 MiB for the 384- and 768-square Kerr beams
on Zen 5 and on Apple silicon, 2.3 to 2.4 times faster than 64 MiB on the native
backend's march;
on an H200, IP-DOPRI5 was 1.3 times faster at one gibibyte than at 64 MiB.
"""

BUFFERS_PER_NODE = 3
"""Node-chunk buffers the fused pass holds: ``E``, ``P`` and the Raman drive."""

FREE_COLUMN_BUFFERS = 20
"""Complex-equivalent buffers per component a Cartesian or planar column holds.

Eight response and FFT buffers, three doubled Raman buffers, 3.5 device plasma buffers,
two split-step buffers and 2.5 statistic buffers: 19, and one more column for the real
FFT's endpoint bins. The transverse scratch and the opaque FFT plans are named
separately in the memory ledgers. A conservative inventory, about three times the live
group buffers (solprop issue #349); the dispersive admission adds its own to it.
"""

ChunkSource = Literal["default", "explicit", "forced"]
"""Where a plan's chunk came from: the default budget, an explicit one, or a chunk."""


@dataclass(frozen=True)
class NodeChunkPlan:
    """A backend's node chunk and what chose it, recorded as execution metadata.

    Attributes
    ----------
    chunk : int
        Nodes a response pass holds at once, between 1 and the transform's node count;
        one on a general modal guide, whose static matrices are a placeholder.
    budget : int
        The budget the chunk was planned within, bytes: the explicit one, or the
        problem's default (:func:`default_node_budget`).
    source : {'default', 'explicit', 'forced'}
        ``"default"`` without an explicit budget (the route's default budget),
        ``"explicit"`` with one, and ``"forced"`` for an exact chunk
        (:attr:`solprop.backends.session.ExecutionOptions.node_chunk`).
    requested : int or None
        The exact chunk asked for, before it was bounded by the node count; ``None``
        unless `source` is ``"forced"``. A general modal guide's route takes it in
        place of its own budgeted chunk.

    Examples
    --------
    >>> from solprop.backends.node_chunk import NodeChunkPlan
    >>> NodeChunkPlan(16, 2**26, "default").record()
    {'node_chunk': 16, 'node_budget': 67108864, 'node_chunk_source': 'default'}
    """

    chunk: int
    budget: int
    source: ChunkSource
    requested: int | None = None

    @property
    def automatic(self) -> bool:
        """Return whether no budget or chunk was given, so a machine rule may apply.

        Returns
        -------
        bool
            ``True`` when `source` is ``"default"``.
        """
        return self.source == "default"

    def record(self) -> dict[str, object]:
        """Return the plan as execution metadata, outside every identity.

        Returns
        -------
        dict
            ``node_chunk``, ``node_budget`` and ``node_chunk_source``.
        """
        return {
            "node_chunk": self.chunk,
            "node_budget": self.budget,
            "node_chunk_source": self.source,
        }


def default_node_budget(transform: Transform) -> int:
    """Return the node budget a problem takes when none is given.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        The problem's transform.

    Returns
    -------
    int
        :data:`DEFAULT_SPATIAL_NODE_BUDGET` on a Cartesian (free-space or line) or
        planar transform, :data:`DEFAULT_NODE_BUDGET` otherwise.
    """
    return DEFAULT_SPATIAL_NODE_BUDGET if _cartesian(transform) else DEFAULT_NODE_BUDGET


def plan_node_chunk(
    problem: PropagationProblem, options: ExecutionOptions | None = None
) -> NodeChunkPlan:
    """Plan a problem's node chunk on the host rule, before anything is allocated.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options; ``None`` for the defaults.

    Returns
    -------
    NodeChunkPlan
        The chunk, its budget and its source.

    Raises
    ------
    solprop.exceptions.BackendError
        If the budget cannot hold one node's buffers (one complete temporal column,
        or one planar transverse row, with every response's own).

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> from solprop.backends.node_chunk import plan_node_chunk
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> plan_node_chunk(problem).chunk
    1
    """
    options = ExecutionOptions() if options is None else options
    transform = problem.transform
    explicit = options.budget
    budget = default_node_budget(transform) if explicit is None else explicit
    requested = options.chunk
    if requested is not None:
        chunk = (
            1
            if transform.modal_tables is not None
            else min(requested, transform.n_nodes)
        )
        return NodeChunkPlan(chunk, budget, "forced", requested)
    source: ChunkSource = "default" if explicit is None else "explicit"
    return NodeChunkPlan(_host_chunk(problem, budget), budget, source)


def _host_chunk(problem: PropagationProblem, budget: int) -> int:
    """Return the host rule's chunk within `budget`: the base chunk, then admissions.

    - ``budget``: node workspace budget, bytes, positive.
    """
    transform = problem.transform
    if transform.modal_tables is not None:
        # A general modal guide's static matrices are a checked placeholder of one
        # node; its route sizes its own chunk from the same budget.
        return 1
    n_pol = problem.shape[1]
    n_to = problem.grids.n_to
    responses = problem.responses
    chunk = _base_chunk(transform, n_pol, n_to, budget)
    if responses.envelope_plasma is not None:
        chunk = _envelope_chunk(chunk, transform, n_pol, n_to, budget)
    chunk = _raman_chunk(chunk, transform, responses, n_pol, n_to, budget)
    chunk = _dispersive_chunk(chunk, transform, responses, n_pol, n_to, budget)
    chunk = _held_current_chunk(chunk, transform, responses, n_pol, n_to, budget)
    return _balanced_chunk(chunk, transform)


def _base_chunk(transform: Transform, n_pol: int, n_to: int, budget: int) -> int:
    """Return the chunk a node's temporal columns admit, before the responses.

    - ``budget``: node workspace budget, bytes.
    """
    geometry = transform.geometry
    if transform.mode_averaged:
        return 1
    if geometry is not None and _cartesian(transform):
        column_bytes = FREE_COLUMN_BUFFERS * 16 * n_pol * n_to
        if budget < column_bytes:
            name = "Cartesian" if geometry.kind == "free" else "planar"
            msg = f"{name} node_budget cannot hold one complete temporal column"
            raise BackendError(msg)
        return min(transform.n_nodes, budget // column_bytes)
    # A modal node's components are the transform's; a radial node's, the state's.
    components = n_pol if geometry is not None else transform.n_pol
    per_node = BUFFERS_PER_NODE * components * n_to * np.dtype(np.complex128).itemsize
    return int(min(transform.n_nodes, max(1, budget // per_node)))


def _cartesian(transform: Transform) -> bool:
    """Return whether the transform's nodes are Cartesian or planar columns."""
    return transform.geometry is not None and transform.geometry.kind in (
        "free",
        "slab",
    )


def _row_nodes(transform: Transform) -> int:
    """Return the physical nodes of one planar transverse row, one elsewhere.

    - Returns: ``n_x`` on the planar kind, else one.
    """
    geometry = transform.geometry
    return geometry.shape[0] if geometry is not None and geometry.kind == "slab" else 1


def _even_chunk(count: int, limit: int) -> int:
    """Return the smallest chunk that covers `count` items in as few passes as `limit`.

    - ``count``: items to cover (positive).
    - ``limit``: largest admitted chunk (positive).
    - Returns: ``ceil(count / ceil(count / limit))``, at most `limit` and `count`.
    """
    passes = -(-count // limit)
    return -(-count // passes)


def _balanced_chunk(chunk: int, transform: Transform) -> int:
    """Spread the nodes evenly over the passes the admitted chunk already needs.

    The budget admissions each take ``budget // per_node``, which can leave a
    last pass of a node or two: 33 RDW nodes under a limit of 32 ran a whole
    32-node pass and then a one-node tail on one thread, about 1.6 times the cost
    of one 33-node pass. The same number of passes of near-equal size (17 and 16)
    never exceeds the admitted chunk, so every budget still holds. A planar chunk
    of whole transverse rows stays whole rows; one shorter than a row is kept.
    """
    row_nodes = _row_nodes(transform)
    rows = transform.n_nodes // row_nodes
    rows_per_chunk, partial = divmod(chunk, row_nodes)
    if partial or rows_per_chunk >= rows:
        return chunk
    return _even_chunk(rows, rows_per_chunk) * row_nodes


def _envelope_chunk(
    chunk: int, transform: Transform, n_pol: int, n_to: int, budget: int
) -> int:
    """Admit envelope plasma's current streams and histories into the chunk.

    Modal projected arrays are accounted separately by the backends' ledgers, because
    they do not scale with the chunk.

    - ``budget``: node workspace budget in bytes.
    """
    components = transform.n_pol * n_pol
    free = _cartesian(transform)
    baseline = (FREE_COLUMN_BUFFERS if free else BUFFERS_PER_NODE) * 16 * components
    # Bound both host and device storage: two complex sources, up to six float
    # histories, per-sample angular status, and fixed256 scan totals/carries.
    # Cartesian chunks also own projected spectra and projection scratch.
    extra = 32 * components + 56 + (80 * components if free else 0)
    per_node = n_to * (baseline + extra) + 16 * ((n_to + 255) // 256) + 8
    if budget < per_node:
        raise BackendError("node_budget cannot hold one envelope plasma source history")
    return min(chunk, budget // per_node)


def _held_current_chunk(  # noqa: PLR0917 - called positionally
    chunk: int,
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> int:
    """Admit complete held-current histories and their nodal transforms.

    Modal projected streams do not scale with the node chunk and belong to
    total-memory admission. The allowance here includes the largest nodal
    transform layout (radial), the band scan, Raman and dispersive buffers
    together, so a mixed response cannot consume the same budget twice.

    - ``budget``: node workspace budget, bytes.
    """
    if not responses.plasma_channels:
        return chunk
    spatial = transform.geometry is not None
    free = _cartesian(transform)
    components = transform.n_pol * n_pol
    baseline = FREE_COLUMN_BUFFERS if free else BUFFERS_PER_NODE
    planar = transform.geometry is not None and transform.geometry.kind == "slab"
    # J and its two products; three real FFT inputs and spectra; cropped nodal
    # output. Each complex FFT row is bounded by n_to complex values, including
    # the DC/Nyquist pair. The device band's three scalar and three component
    # histories bound the host layouts too; tiled prefixes need one total per
    # component per minimum-size tile. Channels reuse those histories.
    source_bytes = (
        components * n_to * (3 * 8 + (3 * 8 + 4 * 16 if spatial and not planar else 0))
    )
    histories = (
        (3 + 3 * components) * n_to * 8
        + components * math.ceil(n_to / 256) * 8
        + len(responses.plasma_channels) * 4 * 8
    )
    radial_projection = (
        transform.n_modes * transform.n_pol * 8 if spatial and not free else 0
    )
    raman = (3 * (48 * n_to + 16)) if responses.raman_terms else 0
    dispersive = dispersive_node_buffers(responses, spatial=free)
    per_node = (
        (baseline + dispersive) * components * n_to * 16
        + source_bytes
        + histories
        + radial_projection
        + raman
    )
    row_nodes = _row_nodes(transform)
    # A planar row contracts n_x physical nodes to n_p modal columns before its
    # temporal FFT. Charge the two projected sources and spectra, plus a shared
    # projection scratch, per modal column rather than per physical node.
    # This matters when a full row is the smallest possible chunk.
    projected_row = (
        transform.synthesis.shape[1]
        * components
        * (3 * n_to * 8 + 2 * (n_to // 2 + 1) * 16)
        if planar
        else 0
    )
    per_row = per_node * row_nodes + projected_row
    if budget < per_row:
        raise BackendError(
            "node_budget cannot hold one held plasma current history"
            " (one full transverse row for a planar grid)"
        )
    # Planar projection contracts complete x rows. Admit the rounded row here,
    # before a backend would otherwise have to increase a sub-row allocation.
    return min(max(1, chunk // row_nodes), budget // per_row) * row_nodes


def _raman_chunk(  # noqa: PLR0917 - called positionally
    chunk: int,
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> int:
    """Admit complete Raman histories, including combined envelope plasma storage.

    - ``budget``: node workspace budget in bytes.
    """
    if not responses.raman_terms:
        return chunk
    components = transform.n_pol * n_pol
    drives = (
        3
        if components == 2
        and any(np.any(term.tensor_weights[:, 1]) for term in responses.raman_terms)
        else 1
    )
    free = _cartesian(transform)
    # Cartesian admission already reserves three complex-equivalent Raman rows
    # per component. Replace that allowance by the selected real drive layout.
    baseline = (17 if free else 3) * 16 * components * n_to
    extra = (48 * n_to + 16) * drives
    if responses.envelope_plasma is not None:
        extra += n_to * (32 * components + 56 + (80 * components if free else 0))
        extra += 16 * ((n_to + 255) // 256) + 8
    per_node = baseline + extra
    if budget < per_node:
        raise BackendError("node_budget cannot hold one complete Raman node history")
    # Full and actual-width tail plans share these arrays. Opaque provider plan
    # storage and node-independent kernels appear separately in backend ledgers.
    return min(chunk, budget // per_node)


def dispersive_node_buffers(responses: CompiledResponses, *, spatial: bool) -> int:
    """Return the buffers per node and component SEP 6's dispersive response adds.

    On the fused node path a backend forms the response's products at the nodes:
    one node buffer beside other responses for the symmetric evaluator, which a lone
    one shares with the bare field, and two for the polynomial, whose cube and
    derivative products take them in turn. A Cartesian or planar column also holds
    its filtered fields and products in time and their spectra: two buffers per
    product, the symmetric evaluator's one (a mixture's under its mixture filter) and
    the polynomial's ``1 + 2 G`` for ``G`` groups of equal leg bound
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.groups`), and the planar
    nodes' one or two besides.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses, with or without the dispersive leaf.
    spatial : bool
        Whether the route's nodes are Cartesian or planar columns.

    Returns
    -------
    int
        Complex-equivalent buffers per node and component; zero without the leaf.
    """
    leaf = responses.dispersive_kerr
    if leaf is None:
        return 0
    symmetric = leaf.evaluator == "symmetric"
    alone = (
        set(responses.response_order) - {ResponseStage.ACCUMULATE}
        == {ResponseStage.DISPERSIVE_KERR}
        and responses.envelope_plasma is None
    )
    groups = len(leaf.groups())
    nodes = (0 if alone and groups == 1 else 1) if symmetric else 2
    if not spatial:
        return nodes
    products = groups if symmetric else 1 + 2 * groups
    return nodes + 2 * products


def _dispersive_chunk(  # noqa: PLR0917 - called positionally
    chunk: int,
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> int:
    """Admit the dispersive response's node buffers into the chunk's budget.

    - ``budget``: node workspace budget in bytes.
    """
    if responses.dispersive_kerr is None:
        return chunk
    spatial = _cartesian(transform)
    components = transform.n_pol * n_pol
    baseline = FREE_COLUMN_BUFFERS if spatial else BUFFERS_PER_NODE
    extra = dispersive_node_buffers(responses, spatial=spatial)
    per_node = (baseline + extra) * components * n_to * 16
    if budget < per_node:
        raise BackendError(
            "node_budget cannot hold one node of the dispersive electronic response"
        )
    return min(chunk, budget // per_node)

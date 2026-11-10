# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Byte counts and sizes the backends share for their memory ledgers.

A backend names every buffer it holds in
:meth:`~solprop.backends.protocol.Backend.memory_ledger`. ``numpy_ref``'s ledger takes
its rows from :func:`reference_base_memory` and :func:`adams_memory`. The native
adapter decides its low-storage registers and z-table slots here too.
"""

import math

import numpy as np

from solprop.fourier import n_omega_for
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.translate.problem import KernelProblem

__all__ = [
    "LOCAL_FLUX_STATISTICS",
    "adams_memory",
    "low_storage_flows",
    "low_storage_stage_register",
    "reference_base_memory",
    "z_table_slots",
]


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

    ``numpy_ref``'s ``memory_ledger`` returns these rows. What one step has resident at
    its peak (the reference allocates its buffers per pass); the space-domain rows are
    sized by ``transform.node_chunk``, not by the node count.

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
    n_modes, n_pol, n_omega = problem.shape
    field_bytes = n_modes * n_pol * n_omega * 16
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
            transform.n_nodes * n_pol * n_omega * 16
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

    Every backend that runs the march holds these, the native engine to the byte
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

    The one predicate the engines share (SEP 24 section 4.3): with
    ``propagator_mode="auto"`` resolved for this march as
    ``StepDriver._propagator_mode`` resolves it on a CPU, a tabulated constant operator
    -- every separable one included, whose ``phi_representation`` is ``"separable"`` --
    holds one volume, ``exp(h_t D / 2)``; a tabulated varying operator holds the
    step's ``P`` and ``Q``; recomputing holds none. A native device build resolves
    ``"auto"`` to recomputing and holds none, which this CPU count over-states by the
    table.

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
    >>> low_storage_flows(problem, "native")  # a constant operator's table
    1
    """
    del backend
    asked = problem.stepper.propagator_mode
    constant = problem.flags.constant_linop
    if asked == "recompute" or (asked == "auto" and not constant):
        return 0
    return 1 if constant else 2


def low_storage_stage_register(problem: KernelProblem, backend: str) -> bool:
    """Return whether a backend's low-storage march holds the stage register ``C``.

    SEP 24 Q4: Kokkos, from phase 3, elides it
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
    >>> low_storage_stage_register(problem, "native")  # a modal route holds C
    True
    """
    geometry = problem.transform.geometry
    responses = problem.responses
    free = (
        problem.linear.separable is not None
        and geometry is not None
        and geometry.kind != "slab"
    )
    if backend != "native" or not free or geometry is None:
        return True
    in_stage = bool(np.any(geometry.absorber_rate)) or (
        responses.temporal_absorber_fine is not None
        and responses.absorber_scheme == "nonlinear"
    )
    return in_stage or len(responses.response_order) <= 1


def adams_memory(problem: KernelProblem) -> dict[str, int]:
    """Return the Adams march's history rows ``numpy_ref`` holds for a problem.

    Empty without an Adams integrator, and on the general path, whose history is a list
    of arrays.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

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
    >>> adams_memory(problem) == {"adams_store": 6 * 16 * grid.n_omega}
    True
    """
    from solprop.backends.multistep import source_absent  # noqa: PLC0415 - import cycle
    from solprop.stepper.multistep import Multistep  # noqa: PLC0415

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
    ledger = {"adams_store": slots * state}
    if compensated:
        ledger |= {"adams_states": slots * state, "adams_rates": slots * state}
    return ledger


LOCAL_FLUX_STATISTICS = frozenset(
    {"on_axis_intensity", "peak_intensity", "beam_radius", "fwhm_r"}
)
"""The spatial statistics that synthesise the magnetic closure in time (SEP 34)."""


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
